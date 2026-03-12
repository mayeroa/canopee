from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, Literal, TypeVar

from canopee.core import ConfigBase, to_flat
from canopee.sweep.distributions import Distribution
from canopee.sweep.strategies import (
    GridStrategy,
    OptunaStrategy,
    RandomStrategy,
    SweepStrategy,
)

T = TypeVar("T", bound=ConfigBase)

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
        """Return the configuration that yielded the best metric."""
        if not self._history:
            return None
        key = min if minimize else max
        return key(self._history, key=lambda x: x[1])[0]

    def results(self) -> list[tuple[T, float]]:
        """Return the literal list of (config, metric) execution pairs."""
        return list(self._history)

    def dataframe(self) -> list[dict[str, Any]]:
        """Return a list of dicts suitable for conversion to pandas DataFrames."""
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
    """

    def __init__(self, base: T) -> None:
        self._base = base
        self._axes: dict[str, Distribution] = {}
        self._strategy: SweepStrategy[T] | None = None
        self._strategy_name: str = "random"
        self._strategy_kwargs: dict[str, Any] = {}

    def vary(self, field: str, distribution: Distribution) -> Sweep[T]:
        """Declare a field to sweep over with a given distribution."""
        # Validate field exists (dot-path aware)
        flat = to_flat(self._base)
        all_fields = set(type(self._base).model_fields.keys()) | set(flat.keys())
        if field not in all_fields:
            raise ValueError(
                f"Field '{field}' not found on {type(self._base).__name__}. Available: {sorted(all_fields)}"
            )
        self._axes[field] = distribution
        return self

    def vary_many(self, **kwargs: Distribution) -> Sweep[T]:
        """Declare multiple fields to sweep over with dict-like kwargs."""
        for k, dist in kwargs.items():
            field = k.replace("__", ".") if "__" in k else k
            self.vary(field, dist)
        return self

    def strategy(
        self,
        name: Literal["grid", "random", "optuna"] = "random",
        **kwargs: Any,
    ) -> Sweep[T]:
        """Choose the sweep strategy."""
        if name not in _STRATEGY_MAP:
            raise ValueError(f"Unknown strategy '{name}'. Choose from: {list(_STRATEGY_MAP)}")
        self._strategy_name = name
        self._strategy_kwargs = kwargs
        return self

    def _build_strategy(self) -> SweepStrategy[T]:
        cls = _STRATEGY_MAP[self._strategy_name]
        return cls(self._base, self._axes, **self._strategy_kwargs)  # type: ignore[call-arg]

    def runner(self) -> SweepStrategy[T]:
        """Yields the active stateful strategy iteration engine."""
        return self._build_strategy()

    def run(self, fn: Callable[[T], float]) -> SweepResult[T]:
        """Execute the entire sweep automatically by piping variants into ``fn``."""
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
        """Run the sweep and write each config as a JSON file to the given path."""
        import os

        os.makedirs(path, exist_ok=True)
        written: list[str] = []
        original_kwargs = dict(self._strategy_kwargs)

        import inspect

        sig = inspect.signature(_STRATEGY_MAP[self._strategy_name].__init__)

        if n_samples is not None and "n_samples" in sig.parameters:
            self._strategy_kwargs["n_samples"] = n_samples
        if seed is not None and "seed" in sig.parameters:
            self._strategy_kwargs["seed"] = seed

        strategy = self._build_strategy()
        for i, cfg in enumerate(strategy):
            filename = os.path.join(path, f"config_{i:04d}_{cfg.fingerprint}.json")
            cfg.save(filename, indent=4)
            written.append(filename)

        self._strategy_kwargs = original_kwargs
        return written

    def __iter__(self) -> Iterator[T]:
        """Enable direct iteration over the sweep."""
        return iter(self._build_strategy())

    def report(self, config: T, *, metric: float) -> None:
        """Report a metric for a specific config if iterating manually."""
        if self._strategy is None:
            self._strategy = self._build_strategy()
        self._strategy.report(config, metric=metric)

    def best(self, *, minimize: bool = True) -> T | None:
        """Return the best config found so far."""
        if self._strategy is None:
            return None
        return self._strategy.best(minimize=minimize)

    def __repr__(self) -> str:
        axes = ", ".join(f"{k}={v!r}" for k, v in self._axes.items())
        return f"Sweep(base={self._base.__class__.__name__}, axes=[{axes}], strategy={self._strategy_name!r})"
