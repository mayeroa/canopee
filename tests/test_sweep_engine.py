import os
import shutil
import pytest
from canopee.core import ConfigBase
from canopee.sweep.engine import Sweep, SweepResult
from canopee.sweep.distributions import uniform, choice
from pydantic import Field


class MockConfig(ConfigBase):
    lr: float = 0.1
    batch_size: int = 32
    optimizer: str = "adam"


def test_sweep_result():
    cfg1 = MockConfig(lr=0.01)
    cfg2 = MockConfig(lr=0.1)
    history = [(cfg1, 0.5), (cfg2, 0.1)]

    result = SweepResult(history)

    assert result.results() == history
    assert result.best(minimize=True) == cfg2
    assert result.best(minimize=False) == cfg1

    df = result.dataframe()
    assert len(df) == 2
    assert df[0]["lr"] == 0.01
    assert df[0]["metric"] == 0.5
    assert df[1]["lr"] == 0.1
    assert df[1]["metric"] == 0.1


def test_sweep_result_empty():
    result = SweepResult([])
    assert result.best() is None
    assert result.results() == []
    assert result.dataframe() == []


def test_sweep_basic():
    base = MockConfig()
    sweep = Sweep(base)

    sweep.vary("lr", uniform(0.001, 0.1))
    sweep.vary_many(optimizer=choice("sgd", "adam"), batch_size=choice(16, 32))

    assert "lr" in sweep._axes
    assert "optimizer" in sweep._axes
    assert "batch_size" in sweep._axes

    sweep.strategy("grid")
    assert sweep._strategy_name == "grid"

    # Test __repr__
    r = repr(sweep)
    assert "Sweep" in r
    assert "lr" in r


def test_sweep_invalid_field():
    base = MockConfig()
    sweep = Sweep(base)
    with pytest.raises(ValueError, match="Field 'non_existent' not found"):
        sweep.vary("non_existent", uniform(0, 1))


def test_sweep_invalid_strategy():
    base = MockConfig()
    sweep = Sweep(base)
    with pytest.raises(ValueError, match="Unknown strategy 'invalid'"):
        sweep.strategy("invalid")


def test_sweep_run():
    base = MockConfig()
    sweep = Sweep(base).vary("lr", choice(0.1, 0.2)).strategy("grid")

    def train(cfg):
        return cfg.lr * 10

    result = sweep.run(train)
    assert isinstance(result, SweepResult)
    assert len(result.results()) == 2
    assert result.best(minimize=True).lr == 0.1


def test_sweep_export(tmp_path):
    export_dir = tmp_path / "configs"
    base = MockConfig()
    sweep = Sweep(base).vary("lr", choice(0.1, 0.2)).strategy("grid")

    paths = sweep.export(str(export_dir))
    assert len(paths) == 2
    assert os.path.exists(paths[0])
    assert os.path.exists(paths[1])

    # Test export with overrides
    # Seed only works if strategy supports it (like RandomStrategy)
    sweep_random = Sweep(base).vary("lr", uniform(0, 1)).strategy("random", n_samples=5)
    paths_random = sweep_random.export(str(export_dir / "random"), n_samples=2, seed=42)
    assert len(paths_random) == 2
    assert os.path.exists(paths_random[0])


def test_sweep_manual_iteration():
    base = MockConfig()
    sweep = Sweep(base).vary("lr", choice(0.1, 0.2)).strategy("grid")

    assert sweep.best() is None  # No strategy built yet

    configs = list(sweep)
    assert len(configs) == 2

    sweep.report(configs[0], metric=0.5)
    sweep.report(configs[1], metric=0.1)

    assert sweep.best(minimize=True) == configs[1]


def test_sweep_runner():
    base = MockConfig()
    sweep = Sweep(base).vary("lr", choice(0.1)).strategy("grid")
    runner = sweep.runner()
    assert hasattr(runner, "__iter__")
