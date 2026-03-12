import random
import pytest
from canopee.sweep.distributions import (
    UniformDist,
    LogUniformDist,
    IntRangeDist,
    ChoiceDist,
    NormalDist,
    _BaseDist,
    uniform,
    log_uniform,
    int_range,
    choice,
    normal,
)


def test_base_dist_errors():
    base = _BaseDist()
    rng = random.Random(42)
    with pytest.raises(NotImplementedError):
        base.sample(rng)
    with pytest.raises(NotImplementedError):
        base.grid_values()


def test_uniform_dist():
    dist = uniform(0.0, 10.0)
    assert isinstance(dist, UniformDist)
    assert dist.low == 0.0
    assert dist.high == 10.0

    rng = random.Random(42)
    val = dist.sample(rng)
    assert 0.0 <= val <= 10.0

    grid = dist.grid_values(n=5)
    assert grid == [0.0, 2.5, 5.0, 7.5, 10.0]


def test_log_uniform_dist():
    dist = log_uniform(1e-3, 1e-1)
    assert isinstance(dist, LogUniformDist)
    assert dist.low == 1e-3
    assert dist.high == 1e-1

    rng = random.Random(42)
    val = dist.sample(rng)
    assert 1e-3 <= val <= 1e-1

    grid = dist.grid_values(n=3)
    # math.exp(math.log(1e-3)), math.exp((math.log(1e-3)+math.log(1e-1))/2), math.exp(math.log(1e-1))
    # log(1e-3) = -6.907..., log(1e-1) = -2.302...
    # mid = -4.605... math.exp(-4.605) = 0.01
    assert grid[0] == pytest.approx(1e-3)
    assert grid[1] == pytest.approx(1e-2)
    assert grid[2] == pytest.approx(1e-1)


def test_int_range_dist():
    dist = int_range(1, 5, step=2)
    assert isinstance(dist, IntRangeDist)
    assert dist.low == 1
    assert dist.high == 5
    assert dist.step == 2

    rng = random.Random(42)
    val = dist.sample(rng)
    assert val in [1, 3, 5]

    grid = dist.grid_values()
    assert grid == [1, 3, 5]


def test_choice_dist():
    dist = choice("a", "b", "c", weights=[0.1, 0.8, 0.1])
    assert isinstance(dist, ChoiceDist)
    assert dist.options == ["a", "b", "c"]
    assert dist.weights == [0.1, 0.8, 0.1]

    rng = random.Random(42)
    val = dist.sample(rng)
    assert val in ["a", "b", "c"]

    grid = dist.grid_values()
    assert grid == ["a", "b", "c"]


def test_normal_dist():
    dist = normal(0.0, 1.0)
    assert isinstance(dist, NormalDist)
    assert dist.mean == 0.0
    assert dist.std == 1.0

    rng = random.Random(42)
    val = dist.sample(rng)
    # val is approx 0.814... with seed 42
    assert isinstance(val, float)

    with pytest.raises(NotImplementedError, match="NormalDist does not support grid enumeration"):
        dist.grid_values()
