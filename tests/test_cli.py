import pytest

from canopee import ConfigBase, clify, parse
from pydantic import Field


class AppConfig(ConfigBase):
    name: str = "default_app"
    workers: int = 4


def test_parse_env(monkeypatch):
    monkeypatch.setenv("TEST_NAME", "my_env_app")
    monkeypatch.setenv("TEST_WORKERS", "16")

    cfg = parse(AppConfig, sources=["env:TEST_"])
    assert cfg.name == "my_env_app"
    assert cfg.workers == 16


def test_parse_cli(monkeypatch):
    monkeypatch.setattr("sys.argv", ["script.py", "--name=cli_app", "--workers", "32"])

    cfg = parse(AppConfig, sources=["cli"])
    assert cfg.name == "cli_app"
    assert cfg.workers == 32


def test_parse_cascade(monkeypatch):
    monkeypatch.setenv("TEST_NAME", "env_app")
    monkeypatch.setenv("TEST_WORKERS", "16")

    # CLI should override env
    monkeypatch.setattr("sys.argv", ["script.py", "--workers", "32"])

    cfg = parse(AppConfig, sources=["env:TEST_", "cli"])
    assert cfg.name == "env_app"  # from env
    assert cfg.workers == 32  # from cli override


def test_clify_decorator(monkeypatch):
    monkeypatch.setattr("sys.argv", ["script.py", "--name=cli_app", "--workers=8"])

    @clify(AppConfig, sources=["cli"])
    def main(cfg: AppConfig):
        return cfg.name, cfg.workers

    name, workers = main()
    assert name == "cli_app"
    assert workers == 8
