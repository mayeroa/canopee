import pytest
from canopee.core import ConfigBase
from canopee.store import ConfigStore, global_store


class MyConfig(ConfigBase):
    val: int = 1
    name: str = "base"


class OtherConfig(ConfigBase):
    x: float = 0.0


def test_store_basic():
    store = ConfigStore("test")
    cfg = MyConfig(val=10)

    # dict access
    store["base"] = cfg
    assert store["base"] == cfg
    assert "base" in store
    assert len(store) == 1
    assert list(store) == ["base"]

    # typed get
    assert store.get("base", MyConfig) == cfg

    with pytest.raises(TypeError):
        store.get("base", OtherConfig)

    with pytest.raises(KeyError):
        store.get("missing")


def test_store_registration():
    store = ConfigStore()
    cfg = MyConfig(val=1)

    store.register("v1", cfg)
    with pytest.raises(KeyError, match="already registered"):
        store.register("v1", cfg)

    store.register("v1", MyConfig(val=2), overwrite=True)
    assert store["v1"].val == 2


def test_store_inheritance():
    store = ConfigStore()
    store.register("base", MyConfig(val=1, name="base"))

    # inherit and override
    store.register("child", MyConfig(val=2), parent="base")
    assert store["child"].val == 2
    assert store["child"].name == "base"

    with pytest.raises(KeyError, match="Parent config 'missing' not found"):
        store.register("orphan", MyConfig(), parent="missing")


def test_store_lineage():
    store = ConfigStore()
    store.register("a", MyConfig(name="a"))
    store.register("b", MyConfig(name="b"), parent="a")
    store.register("c", MyConfig(name="c"), parent="b")

    assert store.lineage("c") == ["a", "b", "c"]
    assert store.lineage("a") == ["a"]


def test_store_lineage_cycle():
    store = ConfigStore()
    # Manually create a cycle since register() doesn't currently prevent it (it doesn't check ancestry)
    store.register("a", MyConfig())
    store.register("b", MyConfig(), parent="a")
    # Poke into internals to force a cycle for testing the detection logic
    store._entries["a"].parent_name = "b"

    with pytest.raises(RuntimeError, match="Cycle detected"):
        store.lineage("a")


def test_store_decorator():
    store = ConfigStore()

    @store.entry("decorated", val=42)
    class DecoratedConfig(ConfigBase):
        val: int = 0

    assert store["decorated"].val == 42
    assert isinstance(store["decorated"], DecoratedConfig)


def test_store_utils():
    store = ConfigStore()
    store["a"] = MyConfig()
    store["b"] = MyConfig()

    assert set(store.list()) == {"a", "b"}
    assert "ConfigStore" in repr(store)

    store.clear()
    assert len(store) == 0


def test_global_store():
    # Verify the singleton is indeed an instance
    assert isinstance(global_store, ConfigStore)
    assert global_store.name == "global"
