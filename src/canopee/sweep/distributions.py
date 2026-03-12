from __future__ import annotations

import math
import random as _random
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Distributions — discriminated union
# ---------------------------------------------------------------------------


class _BaseDist(BaseModel):
    model_config = ConfigDict(frozen=True)

    def sample(self, rng: _random.Random) -> Any:  # noqa: ARG002
        raise NotImplementedError

    def grid_values(self) -> list[Any]:
        raise NotImplementedError(f"{type(self).__name__} does not support grid enumeration.")


class UniformDist(_BaseDist):
    """Continuous uniform distribution over [low, high]."""

    kind: Literal["uniform"] = "uniform"
    low: float
    high: float

    def sample(self, rng: _random.Random) -> float:
        return rng.uniform(self.low, self.high)

    def grid_values(self, n: int = 5) -> list[float]:
        step = (self.high - self.low) / (n - 1)
        return [self.low + i * step for i in range(n)]


class LogUniformDist(_BaseDist):
    """Log-uniform (log-scale uniform) distribution over [low, high]."""

    kind: Literal["log_uniform"] = "log_uniform"
    low: float
    high: float

    def sample(self, rng: _random.Random) -> float:
        log_low = math.log(self.low)
        log_high = math.log(self.high)
        return math.exp(rng.uniform(log_low, log_high))

    def grid_values(self, n: int = 5) -> list[float]:
        log_low = math.log(self.low)
        log_high = math.log(self.high)
        step = (log_high - log_low) / (n - 1)
        return [math.exp(log_low + i * step) for i in range(n)]


class IntRangeDist(_BaseDist):
    """Uniform integer distribution over [low, high] (inclusive)."""

    kind: Literal["int_range"] = "int_range"
    low: int
    high: int
    step: int = 1

    def sample(self, rng: _random.Random) -> int:
        values = list(range(self.low, self.high + 1, self.step))
        return rng.choice(values)

    def grid_values(self) -> list[int]:
        return list(range(self.low, self.high + 1, self.step))


class ChoiceDist(_BaseDist):
    """Categorical distribution over a fixed set of values."""

    kind: Literal["choice"] = "choice"
    options: list[Any]
    weights: list[float] | None = None

    def sample(self, rng: _random.Random) -> Any:
        return rng.choices(self.options, weights=self.weights, k=1)[0]

    def grid_values(self) -> list[Any]:
        return list(self.options)


class NormalDist(_BaseDist):
    """Normal (Gaussian) distribution."""

    kind: Literal["normal"] = "normal"
    mean: float
    std: float

    def sample(self, rng: _random.Random) -> float:
        return rng.gauss(self.mean, self.std)


# Discriminated union — Pydantic uses `kind` to deserialise the correct type
Distribution = UniformDist | LogUniformDist | IntRangeDist | ChoiceDist | NormalDist


# ---------------------------------------------------------------------------
# Convenience constructors (the public API surface)
# ---------------------------------------------------------------------------


def uniform(low: float, high: float) -> UniformDist:
    """Continuous uniform distribution over [low, high]."""
    return UniformDist(low=low, high=high)


def log_uniform(low: float, high: float) -> LogUniformDist:
    """Log-scale uniform distribution — ideal for learning rates."""
    return LogUniformDist(low=low, high=high)


def int_range(low: int, high: int, step: int = 1) -> IntRangeDist:
    """Integer range distribution."""
    return IntRangeDist(low=low, high=high, step=step)


def choice(*options: Any, weights: list[float] | None = None) -> ChoiceDist:
    """Categorical distribution over explicit options."""
    return ChoiceDist(options=list(options), weights=weights)


def normal(mean: float, std: float) -> NormalDist:
    """Normal (Gaussian) distribution."""
    return NormalDist(mean=mean, std=std)
