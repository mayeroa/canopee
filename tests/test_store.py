import pytest
from canopee.core import ConfigBase, Patch
from canopee.store import ConfigStore, global_store

# --- Mock Models for Testing ---
class OptConfig(ConfigBase):
    lr: float = 1e-3
    type: str = "adam"

class DataConfig(ConfigBase):
    name: str = "mnist"
    batch_size: int = 32

class AppConfig(ConfigBase):
    optimizer: OptConfig = OptConfig()
    dataset: DataConfig = DataConfig()
    epochs: int = 10

# --- Tests ---

def test_store_init():
    s = ConfigStore("test")
    assert s.name == "test"
    assert len(s._entries) == 0

def test_store_add_get():
    s = ConfigStore()
    opt = OptConfig()
    s.add("base", opt, group="optimizer")
    
    # Successful get
    res = s.get("base", group="optimizer")
    assert res == opt
    
    # Get with type checking
    res_typed = s.get("base", group="optimizer", as_type=OptConfig)
    assert res_typed == opt
    
    # Get default group
    s.add("main", AppConfig())
    res_main = s.get("main")
    assert isinstance(res_main, AppConfig)

def test_store_get_errors():
    s = ConfigStore()
    s.add("base", OptConfig(), group="optimizer")
    
    # Missing node
    with pytest.raises(KeyError, match="Config node 'missing' not found in group 'optimizer'"):
        s.get("missing", group="optimizer")
        
    # Wrong type
    with pytest.raises(TypeError, match="Config node 'base' is a OptConfig, expected AppConfig"):
        s.get("base", group="optimizer", as_type=AppConfig)

def test_store_variant_kwargs():
    s = ConfigStore()
    s.add("base", OptConfig(lr=1e-3), group="optimizer")
    
    # Create variant via kwargs
    s.variant("fast", base="base", group="optimizer", lr=1e-2)
    
    res = s.get("fast", group="optimizer")
    assert res.lr == 1e-2

def test_store_variant_patch():
    s = ConfigStore()
    s.add("base", OptConfig(lr=1e-3), group="optimizer")
    
    # Create variant via Patch
    s.variant("fast_patch", base="base", group="optimizer", patch=Patch({"lr": 1e-1}))
    
    res = s.get("fast_patch", group="optimizer")
    assert res.lr == 1e-1

def test_store_compose():
    s = ConfigStore()
    # Populate groups
    s.add("adam_fast", OptConfig(lr=1e-2), group="optimizer")
    s.add("cifar", DataConfig(name="cifar10", batch_size=64), group="dataset")
    
    # Populate base
    s.add("baseline", AppConfig())
    
    # Compose
    composed = s.compose(
        base="baseline",
        overrides={
            "optimizer": "adam_fast",
            "dataset": "cifar"
        }
    )
    
    assert isinstance(composed, AppConfig)
    assert composed.optimizer.lr == 1e-2
    assert composed.dataset.name == "cifar10"
    assert composed.dataset.batch_size == 64
    assert composed.epochs == 10  # unchanged base value

def test_store_list_methods():
    s = ConfigStore()
    s.add("adam", OptConfig(), group="optimizer")
    s.add("sgd", OptConfig(), group="optimizer")
    s.add("mnist", DataConfig(), group="dataset")
    s.add("baseline", AppConfig()) # group=None
    
    groups = s.list_groups()
    assert set(groups) == {"optimizer", "dataset", None}
    
    opt_nodes = s.list_nodes(group="optimizer")
    assert set(opt_nodes) == {"adam", "sgd"}
    
    none_nodes = s.list_nodes(group=None)
    assert none_nodes == ["baseline"]

def test_store_clear_and_repr():
    s = ConfigStore("test")
    s.add("adam", OptConfig(), group="opt")
    
    rep = repr(s)
    assert "name='test'" in rep
    assert "groups=1" in rep
    assert "nodes=1" in rep
    
    s.clear()
    assert len(s.list_groups()) == 0

def test_module_level_store():
    # test the global 'store' instance
    store.clear()
    store.add("global_test", AppConfig())
    assert isinstance(store.get("global_test"), AppConfig)
    store.clear()
