import pytest
from pathlib import Path
from canopee.core import (
    ConfigBase,
    Patch,
    Diff,
    wrap,
    evolve,
    diff,
    to_flat,
    select,
    patched,
    schema_tree,
)

# --- Sample Configs for Testing ---


class OptimizerConfig(ConfigBase):
    lr: float = 1e-3
    beta: float = 0.9


class TrainingConfig(ConfigBase):
    epochs: int = 10
    batch_size: int = 32
    optimizer: OptimizerConfig = OptimizerConfig()
    tags: list[str] = ["baseline"]
    save: str = "checkpoint.pt"


# --- Tests for Patch ---


def test_patch_init():
    p = Patch({"a": 1, "b.c": 2})
    assert p["a"] == 1
    assert p["b.c"] == 2
    assert "a" in p
    assert "d" not in p
    assert bool(p) is True
    assert len(p) == 2
    assert set(iter(p)) == {"a", "b.c"}
    assert dict(p.items()) == {"a": 1, "b.c": 2}
    assert p.as_dict() == {"a": 1, "b.c": 2}


def test_patch_empty():
    p = Patch.identity()
    assert bool(p) is False
    assert len(p) == 0


def test_patch_from_kwargs():
    p = Patch.from_kwargs(a=1, b__c=2)
    assert p.as_dict() == {"a": 1, "b.c": 2}


def test_patch_from_flat():
    p = Patch.from_flat({"a.b": 1})
    assert p["a.b"] == 1


def test_patch_composition():
    p1 = Patch({"a": 1, "b": 2})
    p2 = Patch({"b": 3, "c": 4})
    p3 = p1 & p2
    assert p3.as_dict() == {"a": 1, "b": 3, "c": 4}


def test_patch_scoped():
    p = Patch({"lr": 1e-3, "beta": 0.9})
    ps = p.scoped("opt")
    assert ps.as_dict() == {"opt.lr": 1e-3, "opt.beta": 0.9}


def test_patch_filtered():
    p = Patch({"opt.lr": 1e-3, "opt.beta": 0.9, "epochs": 10})
    pf = p.filtered("opt")
    assert pf.as_dict() == {"opt.lr": 1e-3, "opt.beta": 0.9}

    pf2 = p.filtered("epochs")
    assert pf2.as_dict() == {"epochs": 10}


def test_patch_repr():
    p = Patch({"a": 1})
    assert repr(p) == "Patch({'a': 1})"


def test_patch_eq():
    assert Patch({"a": 1}) == Patch({"a": 1})
    assert Patch({"a": 1}) != Patch({"a": 2})
    assert Patch({"a": 1}) != {"a": 1}


# --- Tests for Diff ---


def test_diff_init():
    d = Diff({"a": (1, 2)})
    assert d["a"] == (1, 2)
    assert "a" in d
    assert bool(d) is True
    assert len(d) == 1
    assert list(iter(d)) == ["a"]


def test_diff_empty():
    d = Diff({})
    assert bool(d) is False
    assert len(d) == 0
    assert repr(d) == "Diff(∅)"


def test_diff_values():
    d = Diff({"a": (1, 2), "b.c": (3, 4)})
    assert d.old_values() == {"a": 1, "b.c": 3}
    assert d.new_values() == {"a": 2, "b.c": 4}
    assert d.to_patch() == Patch({"a": 2, "b.c": 4})


def test_diff_invert():
    d = Diff({"a": (1, 2)})
    di = d.invert()
    assert di["a"] == (2, 1)


def test_diff_repr():
    d = Diff({"a": (1, 2)})
    assert "a" in repr(d)
    assert "1 → 2" in repr(d)


def test_diff_eq():
    assert Diff({"a": (1, 2)}) == Diff({"a": (1, 2)})
    assert Diff({"a": (1, 2)}) != Diff({"a": (1, 3)})
    assert Diff({"a": (1, 2)}) != {"a": (1, 2)}


# --- Tests for ConfigBase ---


def test_config_base_merge():
    cfg = TrainingConfig()
    p = Patch({"epochs": 20})
    cfg2 = cfg | p
    assert cfg2.epochs == 20
    assert cfg2.batch_size == 32  # unchanged

    cfg3 = cfg | {"epochs": 30}
    assert cfg3.epochs == 30

    cfg4 = cfg2 | TrainingConfig(epochs=40)
    assert cfg4.epochs == 40


def test_config_base_invalid_merge():
    cfg = TrainingConfig()
    with pytest.raises(TypeError):
        cfg | 123


def test_config_base_fingerprint():
    cfg1 = TrainingConfig(epochs=10)
    cfg2 = TrainingConfig(epochs=10)
    cfg3 = TrainingConfig(epochs=20)

    assert cfg1.fingerprint == cfg2.fingerprint
    assert cfg1.fingerprint != cfg3.fingerprint
    assert len(cfg1.fingerprint) == 16


def test_config_base_hash():
    cfg1 = TrainingConfig(epochs=10)
    cfg2 = TrainingConfig(epochs=10)
    assert hash(cfg1) == hash(cfg2)
    s = {cfg1, cfg2}
    assert len(s) == 1


def test_config_base_repr():
    cfg = TrainingConfig(epochs=10)
    r = repr(cfg)
    assert "TrainingConfig" in r
    assert "epochs=10" in r


# --- Tests for Free Functions ---


def test_evolve():
    cfg = TrainingConfig()
    cfg2 = evolve(cfg, epochs=20, optimizer__lr=1e-4)
    assert cfg2.epochs == 20
    assert cfg2.optimizer.lr == 1e-4


def test_diff_func():
    cfg1 = TrainingConfig(epochs=10)
    cfg2 = TrainingConfig(epochs=20)
    d = diff(cfg1, cfg2)
    assert d["epochs"] == (10, 20)


def test_to_flat():
    cfg = TrainingConfig(epochs=10, optimizer=OptimizerConfig(lr=1e-3))
    flat = to_flat(cfg)
    assert flat["epochs"] == 10
    assert flat["optimizer.lr"] == 1e-3


def test_select():
    cfg = TrainingConfig(tags=["a", "b"])
    assert select(cfg, "epochs") == 10
    assert select(cfg, "optimizer.lr") == 1e-3
    assert select(cfg, "tags.0") == "a"

    with pytest.raises(KeyError):
        select(cfg, "nonexistent")
    with pytest.raises(KeyError):
        select(cfg, "optimizer.nonexistent")
    with pytest.raises(KeyError):
        select(cfg, "tags.10")
    with pytest.raises(KeyError):
        select(cfg, "tags.invalid")


def test_patched():
    cfg = TrainingConfig(epochs=10)
    with patched(cfg, epochs=20) as c:
        assert c.epochs == 20
        assert cfg.epochs == 10
    assert cfg.epochs == 10


def test_schema_tree():
    tree = schema_tree(TrainingConfig)
    assert tree["epochs"] == int
    assert tree["optimizer"]["lr"] == float

    cfg = TrainingConfig()
    assert schema_tree(cfg) == tree


# --- Tests for ConfigProxy ---


def test_proxy_basic():
    cfg = TrainingConfig(epochs=10)
    p = wrap(cfg)
    assert p.unwrap() == cfg
    assert p.evolve(epochs=20).unwrap().epochs == 20
    assert (p | {"epochs": 30}).unwrap().epochs == 30
    assert p.apply(Patch({"epochs": 40})).unwrap().epochs == 40

    other = TrainingConfig(epochs=20)
    d = p.diff(other)
    assert d["epochs"] == (10, 20)

    assert p.diff(wrap(other))["epochs"] == (10, 20)

    assert p.to_flat()["epochs"] == 10
    assert p.select("epochs") == 10
    assert p.schema_tree()["epochs"] == int


def test_proxy_patched():
    cfg = TrainingConfig(epochs=10)
    with wrap(cfg).patched(epochs=20) as pw:
        assert pw.unwrap().epochs == 20
    assert cfg.epochs == 10


def test_proxy_repr_eq_hash():
    cfg = TrainingConfig(epochs=10)
    p = wrap(cfg)
    assert "wrap" in repr(p)
    assert p == wrap(cfg)
    assert p == cfg
    assert hash(p) == hash(cfg)


# --- Tests for internal helpers (Edge cases) ---

from canopee.core import _set_path, _apply_dotpath


def test_set_path_list_extension():
    # Test extending a list via _set_path
    data = {"layers": []}
    result = _set_path(data, ["layers", "0", "units"], 64)
    assert result["layers"][0]["units"] == 64

    result2 = _set_path(data, ["layers", "2", "units"], 128)
    assert len(result2["layers"]) == 3
    assert result2["layers"][2]["units"] == 128
    assert result2["layers"][1] is None  # Filled with None


def test_set_path_errors():
    with pytest.raises(KeyError, match="List index must be an integer"):
        _set_path([], ["not_an_int"], 1)

    with pytest.raises(TypeError, match="Cannot index into"):
        _set_path(123, ["a"], 1)


def test_apply_dotpath_complex():
    target = {"a": {"b": 1}, "c": [10, 20]}
    overrides = {"a.b": 2, "c.1": 30, "d": 40}
    result = _apply_dotpath(target, overrides)
    assert result == {"a": {"b": 2}, "c": [10, 30], "d": 40}
