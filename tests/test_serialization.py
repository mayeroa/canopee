import pytest
from pathlib import Path
from canopee.core import ConfigBase
from canopee.serialization import (
    save,
    load,
    dumps,
    loads,
    to_json,
    from_json,
    dumps_json,
    loads_json,
    to_yaml,
    from_yaml,
    dumps_yaml,
    loads_yaml,
    to_toml,
    from_toml,
    dumps_toml,
    loads_toml,
    _to_path,
)
from pydantic import computed_field


class SubConfig(ConfigBase):
    val: int = 1


class MyConfig(ConfigBase):
    name: str = "test"
    sub: SubConfig = SubConfig()

    @computed_field
    @property
    def upper_name(self) -> str:
        return self.name.upper()


def test_to_path():
    p = Path("test.txt")
    assert _to_path(p) == p
    assert _to_path("test.txt") == Path("test.txt")


def test_json_serialization(tmp_path):
    cfg = MyConfig(name="json")
    path = tmp_path / "test.json"

    # File API
    to_json(cfg, path)
    loaded = from_json(MyConfig, path)
    assert loaded.name == "json"
    assert loaded.upper_name == "JSON"

    # String API
    s = dumps_json(cfg)
    loaded_s = loads_json(MyConfig, s)
    assert loaded_s.name == "json"

    # With computed
    s_comp = dumps_json(cfg, include_computed=True)
    assert "upper_name" in s_comp


def test_yaml_serialization(tmp_path):
    cfg = MyConfig(name="yaml")
    path = tmp_path / "test.yaml"

    # File API
    to_yaml(cfg, path)
    loaded = from_yaml(MyConfig, path)
    assert loaded.name == "yaml"

    # String API
    s = dumps_yaml(cfg)
    assert "fingerprint:" in s
    loaded_s = loads_yaml(MyConfig, s)
    assert loaded_s.name == "yaml"


def test_toml_serialization(tmp_path):
    cfg = MyConfig(name="toml")
    path = tmp_path / "test.toml"

    # File API
    to_toml(cfg, path)
    loaded = from_toml(MyConfig, path)
    assert loaded.name == "toml"

    # String API
    s = dumps_toml(cfg)
    loaded_s = loads_toml(MyConfig, s)
    assert loaded_s.name == "toml"


def test_toml_sanitize():
    class TomlConfig(ConfigBase):
        opt: int | None = None
        list_val: list[int | None] = [1, None, 2]

    cfg = TomlConfig()
    s = dumps_toml(cfg)
    # None should be omitted in TOML
    assert "opt" not in s
    # Check that 1 and 2 are there, but None is not
    assert "1" in s


def test_yaml_import_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ImportError, match="requires PyYAML"):
        dumps_yaml(MyConfig())


def test_toml_writer_import_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "tomli_w", None)
    with pytest.raises(ImportError, match="requires tomli-w"):
        dumps_toml(MyConfig())


def test_dispatcher_save_load(tmp_path):
    cfg = MyConfig(name="dispatch")

    for ext in [".json", ".toml", ".yaml", ".yml"]:
        path = tmp_path / f"test{ext}"
        save(cfg, path)
        loaded = load(MyConfig, path)
        assert loaded.name == "dispatch"

    with pytest.raises(ValueError, match="Unsupported extension"):
        save(cfg, tmp_path / "test.txt")

    with pytest.raises(ValueError, match="Unsupported extension"):
        load(MyConfig, tmp_path / "test.txt")


def test_dispatcher_dumps_loads():
    cfg = MyConfig(name="dispatch")

    for fmt in ["json", "toml", "yaml"]:
        s = dumps(cfg, fmt)
        loaded = loads(MyConfig, fmt, s)
        assert loaded.name == "dispatch"

    with pytest.raises(ValueError, match="Unsupported format"):
        dumps(cfg, "txt")

    with pytest.raises(ValueError, match="Unsupported format"):
        loads(MyConfig, "txt", "data")
