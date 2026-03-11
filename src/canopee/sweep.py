"""
canopee.sweep
~~~~~~~~~~~~~~~~

The Sweep engine: declarative hyperparameter search over ConfigBase subclasses.

Architecture
------------
Distribution types are modelled as a **discriminated union** (via Pydantic's
``Literal`` discriminator pattern).  Each distribution is a frozen Pydantic
model so it can be serialised, logged, and reproduced.

Strategy is a **Strategy Protocol** — concrete strategies (Grid, Random,
Optuna) implement ``__iter__`` and optionally ``report()``.  The ``Sweep``
class is strategy-agnostic and delegates iteration entirely to the strategy.

Example::

    from canopee.sweep import Sweep, uniform, log_uniform, choice
    from canopee.sweep.strategies import GridStrategy, RandomStrategy

    sweep = (
        Sweep(TrainingConfig())
        .vary("lr",         log_uniform(1e-5, 1e-1))
        .vary("batch_size", choice([16, 32, 64, 128]))
        .strategy("random", n_samples=20, seed=42)
    )

    for cfg in sweep:
        result = train(cfg)
        sweep.report(cfg, metric=result.val_loss)

    best = sweep.best()
"""

from __future__ import annotations

import math
import random as _random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

from canopee.core import ConfigBase, to_flat

T = TypeVar("T", bound=ConfigBase)


# ---------------------------------------------------------------------------
# Distributions — discriminated union
# ---------------------------------------------------------------------------
# Each distribution is a frozen Pydantic model with a `kind` Literal field
# used as the discriminator.  This means you can serialise a full sweep
# spec to JSON and reload it exactly.


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
    """Continuous uniform distribution over [low, high].

    Args:
        low (float): Lower bound of the uniform distribution.
        high (float): Upper bound of the uniform distribution.

    Returns:
        UniformDist: A continuous uniform distribution instance.
    """
    return UniformDist(low=low, high=high)


def log_uniform(low: float, high: float) -> LogUniformDist:
    """Log-scale uniform distribution — ideal for learning rates.

    Args:
        low (float): Lower bound of the log-uniform distribution.
        high (float): Upper bound of the log-uniform distribution.

    Returns:
        LogUniformDist: A log-uniform distribution instance.
    """
    return LogUniformDist(low=low, high=high)


def int_range(low: int, high: int, step: int = 1) -> IntRangeDist:
    """Integer range distribution.

    Args:
        low (int): Lower bound of the range.
        high (int): Upper bound of the range (inclusive).
        step (int, optional): Step size between values. Defaults to 1.

    Returns:
        IntRangeDist: An integer range distribution instance.
    """
    return IntRangeDist(low=low, high=high, step=step)


def choice(*options: Any, weights: list[float] | None = None) -> ChoiceDist:
    """Categorical distribution over explicit options.

    Args:
        *options (Any): The available choices.
        weights (list[float] | None, optional): Optional probabilities for each choice. Defaults to None.

    Returns:
        ChoiceDist: A categorical distribution instance.
    """
    return ChoiceDist(options=list(options), weights=weights)


def normal(mean: float, std: float) -> NormalDist:
    """Normal (Gaussian) distribution.

    Args:
        mean (float): Mean (center) of the distribution.
        std (float): Standard deviation (spread or "width") of the distribution.

    Returns:
        NormalDist: A normal distribution instance.
    """
    return NormalDist(mean=mean, std=std)


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
        """Feed back a metric for a completed trial.

        Used by strategies like Optuna that adapt based on results.

        Args:
            config (T): The configuration variant evaluated.
            metric (float): The metric obtained by evaluating the configuration.
        """
        self._results.append((config, metric))

    def best(self, *, minimize: bool = True) -> T | None:
        """Return the config with the best (lowest/highest) metric.

        Args:
            minimize (bool, optional): Whether to look for the minimum metric value. Defaults to True.

        Returns:
            T | None: The configuration yielding the best metric, or None if no results exist.
        """
        if not self._results:
            return None
        key = min if minimize else max
        return key(self._results, key=lambda x: x[1])[0]

    def results(self) -> list[tuple[T, float]]:
        """All (config, metric) pairs reported so far.

        Returns:
            list[tuple[T, float]]: A list of (configuration, metric) pairs.
        """
        return list(self._results)


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class GridStrategy(SweepStrategy[T]):
    """
    Exhaustive Cartesian product over all axis values.

    For continuous distributions, ``n_points`` controls how many evenly
    spaced values are sampled per axis.
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
            try:
                values = dist.grid_values()  # type: ignore[call-arg]
            except TypeError:
                # grid_values with n param
                values = dist.grid_values(self.n_points)  # type: ignore[call-arg]
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

    Requires ``optuna`` to be installed (``pip install canopee[optuna]``).

    The Optuna study is created lazily on first iteration.  After each trial
    you must call ``sweep.report(cfg, metric=val)`` to feed results back.
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

        for trial_num in range(self.n_trials):
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
        """Feed metric back to the Optuna study.

        Args:
            config (T): The configuration variant evaluated.
            metric (float): The metric obtained by evaluating the configuration.
        """
        super().report(config, metric=metric)
        study = self._get_study()
        # Find the trial that produced this config (by fingerprint)
        for trial_number, pending_cfg in self._pending.items():
            if pending_cfg.fingerprint == config.fingerprint:
                study.tell(trial_number, metric)
                del self._pending[trial_number]
                return


# ---------------------------------------------------------------------------
# SweepResult
# ---------------------------------------------------------------------------


class SweepResult(Generic[T]):
    """
    Immutable container holding the results of an executed sweep sequence.
    """

    def __init__(self, run_history: list[tuple[T, float]]) -> None:
        self._history = list(run_history)

    def best(self, *, minimize: bool = True) -> T | None:
        """Return the configuration that yielded the best metric.

        Args:
            minimize (bool, optional): Whether to look for the minimum metric value. Defaults to True.

        Returns:
            T | None: The best configuration, or None if no results exist.
        """
        if not self._history:
            return None
        key = min if minimize else max
        return key(self._history, key=lambda x: x[1])[0]

    def results(self) -> list[tuple[T, float]]:
        """Return the literal list of (config, metric) execution pairs.

        Returns:
            list[tuple[T, float]]: A list of evaluated (configuration, metric) pairs.
        """
        return list(self._history)

    def dataframe(self) -> list[dict[str, Any]]:
        """Return a list of dicts suitable for conversion to pandas DataFrames.

        Flattens the configuration and appends the 'metric' key.

        Returns:
            list[dict[str, Any]]: A list of dictionaries representing the sweep history.
        """
        rows = []
        for cfg, metric in self._history:
            row = to_flat(cfg)
            row["metric"] = metric
            rows.append(row)
        return rows


# ---------------------------------------------------------------------------
# Sweep — the main user-facing class
# ---------------------------------------------------------------------------

_STRATEGY_MAP = {
    "grid": GridStrategy,
    "random": RandomStrategy,
    "optuna": OptunaStrategy,
}


class Sweep(Generic[T]):
    """
    Declarative sweep builder.

    Example::

        sweep = (
            Sweep(TrainingConfig())
            .vary("lr",         log_uniform(1e-5, 1e-1))
            .vary("batch_size", choice(16, 32, 64, 128))
            .strategy("random", n_samples=30, seed=0)
        )

        for cfg in sweep:
            metric = train(cfg)
            sweep.report(cfg, metric=metric)

        best_cfg = sweep.best()
    """

    def __init__(self, base: T) -> None:
        self._base = base
        self._axes: dict[str, Distribution] = {}
        self._strategy: SweepStrategy[T] | None = None
        self._strategy_name: str = "random"
        self._strategy_kwargs: dict[str, Any] = {}

    def vary(self, field: str, distribution: Distribution) -> Sweep[T]:
        """Declare a field to sweep over with a given distribution.

        Validates that *field* exists on the base config.

        Args:
            field (str): The dot-path or top-level field name to vary.
            distribution (Distribution): The distribution to sample values from.

        Returns:
            Sweep[T]: The Sweep instance itself for fluent chaining.

        Raises:
            ValueError: If the specified field is not found on the base config.
        """
        # Validate field exists (dot-path aware)
        flat = to_flat(self._base)
        # Allow top-level or dot-path fields
        all_fields = set(type(self._base).model_fields.keys()) | set(flat.keys())
        if field not in all_fields:
            raise ValueError(
                f"Field '{field}' not found on {type(self._base).__name__}. Available: {sorted(all_fields)}"
            )
        self._axes[field] = distribution
        return self

    def vary_many(self, **kwargs: Distribution) -> Sweep[T]:
        """Declare multiple fields to sweep over with dict-like kwargs.

        Double-underscores (`__`) in keyword arguments are converted to dot-paths
        for deeply nested sweeps.

        Args:
            **kwargs (Distribution): The fields to vary and their corresponding distributions.

        Returns:
            Sweep[T]: The Sweep instance itself for fluent chaining.
        """
        for k, dist in kwargs.items():
            field = k.replace("__", ".") if "__" in k else k
            self.vary(field, dist)
        return self

    def strategy(
        self,
        name: Literal["grid", "random", "optuna"] = "random",
        **kwargs: Any,
    ) -> Sweep[T]:
        """Choose the sweep strategy.

        Args:
            name (Literal["grid", "random", "optuna"], optional): The strategy identifier. Defaults to "random".
            **kwargs (Any): Additional arguments forwarded to the strategy constructor.

        Returns:
            Sweep[T]: The Sweep instance itself for fluent chaining.

        Raises:
            ValueError: If an unknown strategy name is provided.
        """
        if name not in _STRATEGY_MAP:
            raise ValueError(f"Unknown strategy '{name}'. Choose from: {list(_STRATEGY_MAP)}")
        self._strategy_name = name
        self._strategy_kwargs = kwargs
        return self

    def _build_strategy(self) -> SweepStrategy[T]:
        cls = _STRATEGY_MAP[self._strategy_name]
        return cls(self._base, self._axes, **self._strategy_kwargs)  # type: ignore[call-arg]

    def runner(self) -> SweepStrategy[T]:
        """Yields the active stateful strategy iteration engine if manual
        for-loop iteration and reporting is strictly required.

        Returns:
            SweepStrategy[T]: The instantiated and configured sweep strategy.
        """
        return self._build_strategy()

    def run(self, fn: Callable[[T], float]) -> SweepResult[T]:
        """Execute the entire sweep automatically by piping variants into ``fn``.

        Args:
            fn (Callable[[T], float]): The evaluation function taking a configuration and returning a metric.

        Returns:
            SweepResult[T]: A completely detached, immutable object holding the execution history.
        """
        strategy = self._build_strategy()
        for variant in strategy:
            metric = fn(variant)
            strategy.report(variant, metric=metric)
        return SweepResult(strategy.results())

    def export(
        self,
        path: str,
        *,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """Run the sweep and write each config as a JSON file to the given path.

        Args:
            path (str): The directory path where configuration JSON files will be written.
            n_samples (int | None, optional): Override for the number of samples. Defaults to None.
            seed (int | None, optional): Override for the random seed. Defaults to None.

        Returns:
            list[str]: The list of written file paths.
        """
        import os

        os.makedirs(path, exist_ok=True)
        written: list[str] = []
        original_kwargs = dict(self._strategy_kwargs)

        if n_samples is not None:
            self._strategy_kwargs["n_samples"] = n_samples
        if seed is not None:
            self._strategy_kwargs["seed"] = seed

        strategy = self._build_strategy()
        for i, cfg in enumerate(strategy):
            filename = os.path.join(path, f"config_{i:04d}_{cfg.fingerprint}.json")
            cfg.save(filename, indent=4)
            written.append(filename)

        self._strategy_kwargs = original_kwargs
        return written

    def __repr__(self) -> str:
        axes = ", ".join(f"{k}={v!r}" for k, v in self._axes.items())
        return f"Sweep(base={self._base.__class__.__name__}, axes=[{axes}], strategy={self._strategy_name!r})"
