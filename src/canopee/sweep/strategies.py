from __future__ import annotations

import random as _random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from canopee.core import ConfigBase
from canopee.sweep.distributions import (
    Distribution,
    IntRangeDist,
    LogUniformDist,
    UniformDist,
    ChoiceDist,
)

T = TypeVar("T", bound=ConfigBase)


# ---------------------------------------------------------------------------
# Strategy protocol / base
# ---------------------------------------------------------------------------


class SweepStrategy(ABC, Generic[T]):
    """
    Abstract base for sweep strategies.

    Strategies receive the base config, the axis specs (field → distribution),
    and yield concrete config instances.
    """

    def __init__(
        self,
        base: T,
        axes: dict[str, Distribution],
    ) -> None:
        self.base = base
        self.axes = axes
        self._results: list[tuple[T, float]] = []  # (config, metric)

    @abstractmethod
    def __iter__(self) -> Iterator[T]: ...

    def report(self, config: T, *, metric: float) -> None:
        """Feed back a metric for a completed trial."""
        self._results.append((config, metric))

    def best(self, *, minimize: bool = True) -> T | None:
        """Return the config with the best (lowest/highest) metric."""
        if not self._results:
            return None
        key = min if minimize else max
        return key(self._results, key=lambda x: x[1])[0]

    def results(self) -> list[tuple[T, float]]:
        """All (config, metric) pairs reported so far."""
        return list(self._results)


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class GridStrategy(SweepStrategy[T]):
    """
    Exhaustive Cartesian product over all axis values.
    """

    def __init__(
        self,
        base: T,
        axes: dict[str, Distribution],
        n_points: int = 5,
    ) -> None:
        super().__init__(base, axes)
        self.n_points = n_points

    def __iter__(self) -> Iterator[T]:
        import itertools

        field_names = list(self.axes.keys())
        value_lists: list[list[Any]] = []

        for field_name in field_names:
            dist = self.axes[field_name]
            # Check if grid_values takes 'n' parameter
            import inspect

            sig = inspect.signature(dist.grid_values)
            if "n" in sig.parameters:
                values = dist.grid_values(n=self.n_points)
            else:
                values = dist.grid_values()
            value_lists.append(values)

        for combination in itertools.product(*value_lists):
            overrides = dict(zip(field_names, combination))
            yield self.base | overrides


class RandomStrategy(SweepStrategy[T]):
    """
    Random sampling from each axis distribution independently.
    """

    def __init__(
        self,
        base: T,
        axes: dict[str, Distribution],
        n_samples: int = 10,
        seed: int | None = None,
    ) -> None:
        super().__init__(base, axes)
        self.n_samples = n_samples
        self._rng = _random.Random(seed)

    def __iter__(self) -> Iterator[T]:
        for _ in range(self.n_samples):
            overrides = {field: dist.sample(self._rng) for field, dist in self.axes.items()}
            yield self.base | overrides


class OptunaStrategy(SweepStrategy[T]):
    """
    Bayesian optimisation via Optuna.
    """

    def __init__(
        self,
        base: T,
        axes: dict[str, Distribution],
        n_trials: int = 20,
        direction: str = "minimize",
        study_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(base, axes)
        self.n_trials = n_trials
        self.direction = direction
        self.study_kwargs = study_kwargs or {}
        self._study: Any = None
        self._pending: dict[int, T] = {}  # trial_number → config

    def _get_study(self) -> Any:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("OptunaStrategy requires optuna. Install it with: pip install canopee[optuna]") from exc
        if self._study is None:
            self._study = optuna.create_study(
                direction=self.direction,
                **self.study_kwargs,
            )
        return self._study

    def __iter__(self) -> Iterator[T]:
        study = self._get_study()

        for _ in range(self.n_trials):
            trial = study.ask()
            overrides: dict[str, Any] = {}

            for field_name, dist in self.axes.items():
                if isinstance(dist, (UniformDist,)):
                    value = trial.suggest_float(field_name, dist.low, dist.high)
                elif isinstance(dist, LogUniformDist):
                    value = trial.suggest_float(field_name, dist.low, dist.high, log=True)
                elif isinstance(dist, IntRangeDist):
                    value = trial.suggest_int(field_name, dist.low, dist.high, step=dist.step)
                elif isinstance(dist, ChoiceDist):
                    value = trial.suggest_categorical(field_name, dist.options)
                else:
                    # Fallback: sample directly
                    rng = _random.Random()
                    value = dist.sample(rng)
                overrides[field_name] = value

            cfg = self.base | overrides
            self._pending[trial.number] = cfg
            yield cfg

    def report(self, config: T, *, metric: float) -> None:
        """Feed metric back to the Optuna study."""
        super().report(config, metric=metric)
        study = self._get_study()
        # Find the trial that produced this config (by fingerprint)
        for trial_number, pending_cfg in self._pending.items():
            if pending_cfg.fingerprint == config.fingerprint:
                study.tell(trial_number, metric)
                del self._pending[trial_number]
                return
