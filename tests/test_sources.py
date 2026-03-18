import os
import json
import pytest
from canopee.sources import EnvSource, CLISource, FileSource, DictSource, merge_sources, _coerce_env_value


def test_coerce_env_value():
    assert _coerce_env_value("true") is True
    assert _coerce_env_value("1") is True
    assert _coerce_env_value("yes") is True
    assert _coerce_env_value("false") is False
    assert _coerce_env_value("0") is False
    assert _coerce_env_value("no") is False
    assert _coerce_env_value("123") == 123
    assert _coerce_env_value("1.5") == 1.5
    assert _coerce_env_value("[1, 2]") == [1, 2]
    assert _coerce_env_value('{"a": 1}') == {"a": 1}
    assert _coerce_env_value("not-json") == "not-json"
    assert _coerce_env_value("{invalid") == "{invalid"


def test_env_source(monkeypatch):
    monkeypatch.setenv("APP_VAL", "10")
    monkeypatch.setenv("APP_SUB__X", "1.5")
    monkeypatch.setenv("OTHER", "ignored")

    src = EnvSource(prefix="APP_")
    data = src.load()
    assert data["val"] == 10
    assert data["sub.x"] == 1.5
    assert "other" not in data
    assert "APP_" in repr(src)


def test_env_source_case_sensitive(monkeypatch):
    monkeypatch.setenv("App_Val", "10")
    src = EnvSource(prefix="App_", case_sensitive=True)
    data = src.load()
    assert data["Val"] == 10


def test_cli_source():
    argv = ["--lr=0.01", "--batch_size", "32", "--verbose"]
    src = CLISource(argv=argv)
    data = src.load()
    assert data["lr"] == 0.01
    assert data["batch_size"] == 32
    assert data["verbose"] is True
    assert "CLISource" in repr(src)


def test_dict_source():
    d = {"a": 1, "b.c": 2}
    src = DictSource(d)
    assert src.load() == d
    assert "DictSource" in repr(src)


def test_file_source_json(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    src = FileSource(path)
    assert src.load() == {"a": 1}
    assert "FileSource" in repr(src)


def test_file_source_toml(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text("a = 1\n[sub]\nx = 2", encoding="utf-8")
    src = FileSource(path)
    assert src.load() == {"a": 1, "sub": {"x": 2}}


def test_file_source_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("a: 1\nsub:\n  x: 2", encoding="utf-8")
    src = FileSource(path)
    assert src.load() == {"a": 1, "sub": {"x": 2}}


def test_file_source_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileSource("non-existent.json").load()

    path = tmp_path / "cfg.txt"
    path.write_text("data")
    with pytest.raises(ValueError, match="Unsupported config file format"):
        FileSource(path).load()


def test_merge_sources():
    src1 = DictSource({"a": 1, "b": 2})
    src2 = DictSource({"b": 3, "c.d": 4})

    merged = merge_sources([src1, src2])
    assert merged["a"] == 1
    assert merged["b"] == 3
    assert merged["c"]["d"] == 4


def test_yaml_import_error(monkeypatch):
    import sys
    from canopee.sources import _load_yaml_file

    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ImportError, match="requires PyYAML"):
        _load_yaml_file("data")


def test_cli_source_boolean_flag():
    # Test --verbose without a value
    argv = ["--verbose"]
    src = CLISource(argv=argv)
    data = src.load()
    assert data["verbose"] is True


def test_cli_source_key_value_pair():
    # Test --key value instead of --key=value
    # This should trigger line 166-167
    argv = ["--lr", "0.01", "--batch_size", "64"]
    src = CLISource(argv=argv)
    data = src.load()
    assert data["lr"] == 0.01
    assert data["batch_size"] == 64


def test_cli_source_skip_non_flags():
    # Test skipping arguments that don't start with --
    argv = ["pos_arg", "--lr", "0.01", "other_pos"]
    src = CLISource(argv=argv)
    data = src.load()
    assert data == {"lr": 0.01}


def test_source_base_repr():
    from canopee.sources import Source

    class MySource(Source):
        def load(self):
            return {}

    assert "MySource()" in repr(MySource())
