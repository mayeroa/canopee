import json
import pytest
from pydantic import Field, ValidationError, computed_field
from canopee.core import ConfigBase, evolve, diff, to_flat, _apply_dotpath

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

def test_evolve_empty(base_config):
    assert evolve(base_config) is base_config

def test_or_operator_flat_and_nested(base_config):
    # Dict on the right
    v2 = base_config | {"epochs": 3, "opt.lr": 1e-2}
    assert v2.epochs == 3
    assert v2.opt.lr == 1e-2
    assert v2.opt.name == "adam"

def test_ror_operator(base_config):
    # Dict on the left
    # overrides | cfg  ->  cfg.__ror__(overrides) -> cfg.__or__(overrides) -> cfg | overrides
    v2 = {"epochs": 3, "opt.lr": 1e-2} | base_config
    assert v2.epochs == 3
    assert v2.opt.lr == 1e-2

def test_config_base_or_config_base():
    cfg1 = TrainConfig(epochs=1)
    cfg2 = TrainConfig(epochs=2)
    cfg3 = cfg1 | cfg2
    assert cfg3.epochs == 2

def test_repr(base_config):
    r = repr(base_config)
    assert "TrainConfig" in r
    assert "epochs=10" in r

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

def test_diff_complex():
    cfg1 = TrainConfig(epochs=10, opt=OptimizerConfig(lr=1e-3))
    cfg2 = TrainConfig(epochs=20, opt=OptimizerConfig(lr=1e-2))
    
    d = diff(cfg1, cfg2)
    assert d["epochs"] == (10, 20)
    assert d["opt.lr"] == (0.001, 0.01)

def test_to_flat(base_config):
    flat = to_flat(base_config)
    assert flat["epochs"] == 10
    assert flat["opt.name"] == "adam"
    assert flat["model.hidden"] == 256

def test_serialization_delegation(tmp_path, base_config):
    # dumps / loads
    s = base_config.dumps("json")
    loaded = TrainConfig.loads("json", s)
    assert loaded.epochs == 10
    
    # save / load
    path = tmp_path / "cfg.json"
    base_config.save(path)
    loaded_file = TrainConfig.load(path)
    assert loaded_file.epochs == 10

def test_apply_dotpath_overwrite_scalar_with_dict():
    target = {"a": 1}
    _apply_dotpath(target, {"a.b": 2})
    assert target["a"] == {"b": 2}

def test_dump_for_validation_excludes_computed(base_config):
    d = base_config._dump_for_validation()
    assert "total_steps" not in d
    assert "params" not in d["model"]
