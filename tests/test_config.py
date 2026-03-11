import json

import pytest
from pydantic import Field, ValidationError, computed_field

from canopee import (
    ConfigBase,
    ConfigStore,
    diff,
    dumps,
    evolve,
    global_store,
    load,
    loads,
    save,
    to_flat,
)
from canopee.sweep import Sweep, choice, uniform


# ---------------------------------------------------------------------------
# Fixtures & Dummy Configs
# ---------------------------------------------------------------------------


class OptimizerConfig(ConfigBase):
    name: str = "adam"
    lr: float = 1e-3


class ModelConfig(ConfigBase):
    hidden: int = 256
    layers: int = 4

    @computed_field
    @property
    def params(self) -> int:
        return self.hidden * self.layers * 100


class TrainConfig(ConfigBase):
    epochs: int = 10
    opt: OptimizerConfig = Field(default_factory=OptimizerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    @computed_field
    @property
    def total_steps(self) -> int:
        return self.epochs * 1000


@pytest.fixture
def base_config():
    return TrainConfig()


# ---------------------------------------------------------------------------
# ConfigBase Core Tests
# ---------------------------------------------------------------------------


def test_instantiation_and_defaults(base_config):
    assert base_config.epochs == 10
    assert base_config.opt.name == "adam"
    assert base_config.total_steps == 10000


def test_immutability(base_config):
    with pytest.raises(ValidationError):
        # frozen=True disables assignment
        base_config.epochs = 20


def test_evolve(base_config):
    v2 = evolve(base_config, epochs=5)
    assert v2 is not base_config
    assert v2.epochs == 5
    assert v2.total_steps == 5000  # computed field re-evaluated
    assert base_config.epochs == 10  # original unchanged


def test_evolve_nested_double_underscore(base_config):
    v2 = evolve(base_config, opt__lr=0.5)
    assert v2.opt.lr == 0.5
    assert base_config.opt.lr == 1e-3


def test_or_operator_flat_and_nested(base_config):
    # Dict on the right
    v2 = base_config | {"epochs": 3, "opt.lr": 1e-2}
    assert v2.epochs == 3
    assert v2.opt.lr == 1e-2
    assert v2.opt.name == "adam"


def test_ror_operator(base_config):
    # Dict on the left
    v2 = {"epochs": 3, "opt.lr": 1e-2} | base_config
    assert v2.epochs == 3
    assert v2.opt.lr == 1e-2


def test_serialization_excludes_computed(base_config):
    d = json.loads(dumps(base_config, "json"))
    assert "total_steps" not in d
    assert "params" not in d["model"]

    d_all = json.loads(dumps(base_config, "json", include_computed=True))
    assert "total_steps" in d_all
    assert "params" in d_all["model"]


def test_fingerprint(base_config):
    v2 = evolve(base_config, epochs=10)  # unchanged logically
    assert base_config.fingerprint == v2.fingerprint

    v3 = evolve(base_config, epochs=11)
    assert base_config.fingerprint != v3.fingerprint


def test_diff(base_config):
    v2 = base_config | {"epochs": 3, "opt.lr": 1e-2}
    result = diff(base_config, v2)
    # opt.lr changed, and fingerprint changed
    assert "opt.lr" in result
    assert "fingerprint" in result


def test_to_flat(base_config):
    flat = to_flat(base_config)
    assert flat["epochs"] == 10
    assert flat["opt.name"] == "adam"
    assert flat["model.hidden"] == 256


# ---------------------------------------------------------------------------
# Serialization Tests
# ---------------------------------------------------------------------------


def test_string_serialization(base_config):
    # JSON
    js = dumps(base_config, "json")
    v_js = loads(TrainConfig, "json", js)
    assert v_js == base_config

    # TOML
    tm = dumps(base_config, "toml")
    v_tm = loads(TrainConfig, "toml", tm)
    assert v_tm == base_config

    # YAML
    ym = dumps(base_config, "yaml")
    v_ym = loads(TrainConfig, "yaml", ym)
    assert v_ym == base_config


def test_file_serialization(base_config, tmp_path):
    path = tmp_path / "config.toml"
    save(base_config, path)
    assert path.exists()

    v_load = load(TrainConfig, path)
    assert v_load == base_config

    path_json = tmp_path / "config.json"
    save(base_config, path_json, indent=4)
    v_load_json = load(TrainConfig, path_json)
    assert v_load_json == base_config


# ---------------------------------------------------------------------------
# ConfigStore Tests
# ---------------------------------------------------------------------------


def test_store_dict_api():
    global_store.clear()
    cfg1 = TrainConfig(epochs=1)

    global_store["v1"] = cfg1
    assert "v1" in global_store
    assert len(global_store) == 1
    assert list(global_store) == ["v1"]

    retrieved = global_store["v1"]
    assert retrieved == cfg1

    # typed retrieval
    retrieved_typed = global_store.get("v1", TrainConfig)
    assert retrieved_typed == cfg1

    global_store.clear()


def test_store_decorator():
    global_store.clear()

    @global_store.entry("decorated", epochs=99)
    class CustomConfig(ConfigBase):
        epochs: int = 1

    cfg = global_store["decorated"]
    assert isinstance(cfg, CustomConfig)
    assert cfg.epochs == 99


def test_store_inheritance():
    global_store.clear()
    global_store.register("base", TrainConfig(epochs=2))
    global_store.register("fast", TrainConfig(epochs=1), parent="base")

    cfg = global_store["fast"]
    assert cfg.epochs == 1

    lineage = global_store.lineage("fast")
    assert lineage == ["base", "fast"]


def test_isolated_store():
    global_store.clear()

    # Prove that instantiable stores are fully isolated
    my_store = ConfigStore(name="my_store")
    my_store["isolated"] = TrainConfig(epochs=5)

    assert "isolated" in my_store
    assert "isolated" not in global_store


# ---------------------------------------------------------------------------
# Sweep Tests
# ---------------------------------------------------------------------------


def test_sweep_run(base_config):
    sweep = (
        Sweep(base_config)
        .vary("epochs", choice(1, 2, 3))
        .vary("opt.lr", uniform(1e-4, 1e-2))
        .strategy("grid")  # 3 epochs * 5 default points = 15 variants
    )

    def dummy_train(cfg: TrainConfig) -> float:
        # Dummy metric: lower is better. epochs=3 is best.
        # (This is just an arbitrary function)
        return float(10 - cfg.epochs)

    result = sweep.run(dummy_train)

    best = result.best(minimize=True)
    assert best is not None
    assert best.epochs == 3

    history = result.results()
    assert len(history) == 15

    df = result.dataframe()
    assert len(df) == 15
    assert "metric" in df[0]
