import pytest
from typing import Iterator
from canopee.core import ConfigBase
from canopee.sweep.distributions import uniform, choice, int_range, log_uniform, normal
from canopee.sweep.strategies import (
    SweepStrategy,
    GridStrategy,
    RandomStrategy,
    OptunaStrategy,
)


class MockConfig(ConfigBase):
    lr: float = 0.1
    optimizer: str = "adam"
    batch_size: int = 32
    log_lr: float = 0.001
    extra: float = 0.0


class ConcreteStrategy(SweepStrategy[MockConfig]):
    def __iter__(self) -> Iterator[MockConfig]:
        yield self.base


def test_sweep_strategy_base():
    base = MockConfig()
    axes = {"lr": uniform(0.01, 0.1)}
    strat = ConcreteStrategy(base, axes)

    assert strat.base == base
    assert strat.axes == axes
    assert strat.results() == []

    configs = list(strat)
    assert len(configs) == 1
    assert configs[0] == base

    strat.report(configs[0], metric=0.5)
    assert strat.results() == [(configs[0], 0.5)]
    assert strat.best(minimize=True) == configs[0]
    assert strat.best(minimize=False) == configs[0]


def test_sweep_strategy_best_empty():
    base = MockConfig()
    strat = ConcreteStrategy(base, {})
    assert strat.best() is None


def test_grid_strategy():
    base = MockConfig()
    axes = {
        "lr": uniform(0.1, 0.2),  # default n_points=5 -> [0.1, 0.125, 0.15, 0.175, 0.2]
        "optimizer": choice("sgd", "adam"),
    }
    # Test with custom n_points
    strat = GridStrategy(base, axes, n_points=2)
    configs = list(strat)

    # 2 (lr) * 2 (optimizer) = 4 configs
    assert len(configs) == 4
    lrs = {c.lr for c in configs}
    opts = {c.optimizer for c in configs}
    assert lrs == {0.1, 0.2}
    assert opts == {"sgd", "adam"}


def test_random_strategy():
    base = MockConfig()
    axes = {"lr": uniform(0, 1)}
    strat = RandomStrategy(base, axes, n_samples=3, seed=42)
    configs = list(strat)
    assert len(configs) == 3
    assert configs[0].lr != configs[1].lr


def test_optuna_strategy():
    base = MockConfig()
    axes = {
        "lr": uniform(0.01, 0.1),
        "log_lr": log_uniform(1e-4, 1e-2),
        "batch_size": int_range(16, 64, step=16),
        "optimizer": choice("sgd", "adam"),
        "extra": normal(0, 1),  # This should hit the fallback branch in OptunaStrategy.__iter__
    }

    strat = OptunaStrategy(base, axes, n_trials=5, direction="minimize")

    # Test lazy study creation
    assert strat._study is None
    configs = []
    for i, cfg in enumerate(strat):
        configs.append(cfg)
        strat.report(cfg, metric=1.0 / (i + 1))

    assert len(configs) == 5
    assert strat._study is not None
    assert len(strat.results()) == 5
    assert strat.best(minimize=True).fingerprint == configs[-1].fingerprint


def test_optuna_strategy_import_error(monkeypatch):
    import sys

    # Simulate optuna not being installed
    monkeypatch.setitem(sys.modules, "optuna", None)

    base = MockConfig()
    strat = OptunaStrategy(base, {"lr": uniform(0, 1)})
    with pytest.raises(ImportError, match="OptunaStrategy requires optuna"):
        strat._get_study()


def test_optuna_report_non_existent_config():
    # If we report a config that wasn't yielded (not in _pending), it should just
    # add it to _results but not tell Optuna (since it can't find the trial_number).
    base = MockConfig()
    strat = OptunaStrategy(base, {"lr": uniform(0, 1)}, n_trials=1)

    # Build study but don't ask/tell yet
    strat._get_study()

    external_cfg = MockConfig(lr=0.5)
    strat.report(external_cfg, metric=0.0)

    assert len(strat.results()) == 1
    assert strat.results()[0] == (external_cfg, 0.0)
    # _pending should still be empty if we haven't iterated
    assert strat._pending == {}
