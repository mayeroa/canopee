import pytest
from typing import Optional
from pydantic import ValidationError
from canopee.core import ConfigBase
from canopee.serialization import (
    to_dict,
    from_dict,
    to_json_str,
    from_json_str,
    save_json,
    load_json,
    to_yaml_str,
    from_yaml_str,
    save_yaml,
    load_yaml,
    to_toml_str,
    from_toml_str,
    save_toml,
    load_toml,
    save,
    load,
    _DROP,
    _sanitize_toml,
)

# --- Sample Configs for Testing ---


class SubConfig(ConfigBase):
    units: int = 64
    name: str = "layer1"


class MainConfig(ConfigBase):
    epochs: int = 10
    lr: float = 1e-3
    optimizer: SubConfig = SubConfig()
    tags: list[str] = ["base", "v1"]
    nullable_field: Optional[int] = None


# --- Layer 1: Dict tests ---


def test_dict_roundtrip():
    cfg = MainConfig(epochs=20, lr=3e-4, tags=["test"])
    data = to_dict(cfg)
    assert isinstance(data, dict)
    assert data["epochs"] == 20
    assert data["lr"] == 3e-4
    assert data["optimizer"]["units"] == 64
    assert data["tags"] == ["test"]
    assert "fingerprint" not in data

    cfg2 = from_dict(MainConfig, data)
    assert cfg2 == cfg
    assert isinstance(cfg2, MainConfig)


def test_from_dict_validation_error():
    with pytest.raises(ValidationError):
        from_dict(MainConfig, {"epochs": "not_an_int"})


# --- Layer 2: String tests ---


def test_json_string_roundtrip():
    cfg = MainConfig(epochs=5)
    text = to_json_str(cfg, indent=4)
    assert '"epochs": 5' in text

    cfg2 = from_json_str(MainConfig, text)
    assert cfg2 == cfg


def test_yaml_string_roundtrip():
    cfg = MainConfig(epochs=5)
    text = to_yaml_str(cfg)
    assert "epochs: 5" in text
    assert "# canopee config" in text
    assert f"# class: {MainConfig.__name__}" in text

    cfg2 = from_yaml_str(MainConfig, text)
    assert cfg2 == cfg


def test_toml_string_roundtrip():
    cfg = MainConfig(epochs=5)
    text = to_toml_str(cfg)
    assert "epochs = 5" in text

    cfg2 = from_toml_str(MainConfig, text)
    assert cfg2 == cfg


# --- Layer 3: File I/O tests ---


def test_json_file_io(tmp_path):
    cfg = MainConfig(epochs=15)
    path = tmp_path / "config.json"
    save_json(cfg, path)
    assert path.exists()

    cfg2 = load_json(MainConfig, path)
    assert cfg2 == cfg


def test_yaml_file_io(tmp_path):
    cfg = MainConfig(epochs=15)
    path = tmp_path / "config.yaml"
    save_yaml(cfg, path)
    assert path.exists()

    cfg2 = load_yaml(MainConfig, path)
    assert cfg2 == cfg


def test_toml_file_io(tmp_path):
    cfg = MainConfig(epochs=15)
    path = tmp_path / "config.toml"
    save_toml(cfg, path)
    assert path.exists()

    cfg2 = load_toml(MainConfig, path)
    assert cfg2 == cfg


# --- Auto-dispatch tests ---


def test_auto_dispatch_save_load(tmp_path):
    cfg = MainConfig(epochs=100)

    for ext in [".json", ".yaml", ".yml", ".toml"]:
        path = tmp_path / f"test{ext}"
        save(cfg, path)
        assert path.exists()
        cfg2 = load(MainConfig, path)
        assert cfg2 == cfg


def test_auto_dispatch_unsupported_extension():
    cfg = MainConfig()
    with pytest.raises(ValueError, match="Unsupported extension"):
        save(cfg, "config.txt")

    with pytest.raises(ValueError, match="Unsupported extension"):
        load(MainConfig, "config.txt")


# --- TOML None Handling ---


def test_toml_none_drop():
    cfg = MainConfig(nullable_field=None)
    text = to_toml_str(cfg, none_handling="drop")
    assert "nullable_field" not in text


def test_toml_none_raise():
    cfg = MainConfig(nullable_field=None)
    with pytest.raises(ValueError, match="None value at 'nullable_field' cannot be serialised to TOML"):
        to_toml_str(cfg, none_handling="raise")


def test_toml_none_null_str():
    cfg = MainConfig(nullable_field=None)
    text = to_toml_str(cfg, none_handling="null_str")
    assert 'nullable_field = "null"' in text


def test_sanitize_toml_nested_none():
    data = {"a": {"b": None}, "c": [None, 1]}

    # Drop (default)
    res_drop = _sanitize_toml(data, none_handling="drop")
    assert res_drop == {"a": {}, "c": [1]}

    # Raise
    with pytest.raises(ValueError, match="None value at 'a.b'"):
        _sanitize_toml(data, none_handling="raise")

    # Null string
    res_null = _sanitize_toml(data, none_handling="null_str")
    assert res_null == {"a": {"b": "null"}, "c": ["null", 1]}


# --- Internal Helpers / Edge cases ---


def test_require_dependencies(monkeypatch):
    # Test ImportError when PyYAML is missing
    def mock_import_yaml():
        raise ImportError("PyYAML not found")

    # We can't easily mock the import itself without complex mocking,
    # but we can mock the helper function if we want to reach 100% coverage
    # of the error message part if it's not already covered.

    from canopee import serialization

    # Let's try to trigger the ImportError by mocking __import__ or similar
    # Actually, simpler: just call the _require_* functions if they are exposed or
    # mock the import in their module.

    # For tomli-w
    # (Since we are likely running in an env with them, we have to force the failure)

    import builtins

    original_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name in ("yaml", "tomli_w"):
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)

    with pytest.raises(ImportError, match="YAML support requires PyYAML"):
        serialization._require_yaml()

    with pytest.raises(ImportError, match="Writing TOML requires tomli-w"):
        serialization._require_tomli_w()


def test_drop_repr():
    assert repr(_DROP) == "<DROP>"
