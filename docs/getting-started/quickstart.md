# Quickstart

<div class="gs-hero" markdown>
From `pip install` to a full hyperparameter sweep in 5 minutes. This page covers every core concept — work through the steps in order and you'll have a solid mental model of the whole library.
</div>

<div class="gs-steps-overview" markdown>

| Step | What you'll learn |
|------|-------------------|
| [1. Define a config](#1-define-a-config) | `ConfigBase`, field annotations, validators |
| [2. Computed fields](#2-computed-fields) | `@computed_field`, derived values, serialisation |
| [3. Merge with `\|`](#3-merge-with) | Immutable overrides, dot-path keys |
| [4. Evolve with keywords](#4-evolve-with-keywords) | `evolve()`, IDE autocomplete |
| [5. Discriminated unions](#5-discriminated-unions) | Type-safe optimizer/model variants |
| [6. Hyperparameter sweep](#6-hyperparameter-sweep) | `Sweep`, distributions, strategies |

</div>

---

## 1. Define a config

Subclass `ConfigBase` and annotate fields exactly as you would with any Pydantic model. canopee adds immutability, the merge operator, and the `evolve()` method on top — nothing else changes.

```python
from canopee import ConfigBase
from pydantic import Field, field_validator

class TrainingConfig(ConfigBase):
    lr: float        = Field(default=1e-3, gt=0.0, le=1.0) # (1)!
    epochs: int      = Field(default=20, gt=0)
    batch_size: int  = 128
    seed: int        = 42

    @field_validator("batch_size")
    @classmethod
    def must_be_power_of_two(cls, v: int) -> int: # (2)!
        if v & (v - 1) != 0:
            raise ValueError(f"batch_size must be a power of 2, got {v}")
        return v

cfg = TrainingConfig()
print(cfg.lr)      # → 0.001
print(cfg.epochs)  # → 20
```

1. Field constraints (`gt`, `le`, …) are standard Pydantic — they run at construction time, not use time.
2. `@field_validator` works identically to Pydantic. The error message surfaces immediately on bad input.

`ConfigBase` enforces three things automatically:

!!! abstract "frozen=True"
    Instances are immutable after `__init__`. Assigning `cfg.lr = 0.1` raises `ValidationError` immediately. Use `cfg | {"lr": 0.1}` to produce a modified copy.

!!! abstract "extra='forbid'"
    Typos in field names are caught at construction time:
    ```python
    TrainingConfig(learing_rate=1e-3)  # ✗ ValidationError: extra inputs not permitted
    ```

!!! abstract "validate_default=True"
    Validators run even on default values, catching bugs in your class definition before any instance is ever created.

---

## 2. Computed fields

Declare derived values as `@computed_field` properties. They are **included in `model_dump()` and `model_dump_json()`**, re-evaluated on every new instance, and supported by Pydantic's JSON schema.

```python
from pydantic import computed_field
import math

class TrainingConfig(ConfigBase):
    lr: float        = Field(default=1e-3, gt=0.0)
    epochs: int      = 20
    batch_size: int  = 128

    @computed_field       # (1)!
    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(54_000 / self.batch_size)

    @computed_field
    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.epochs  # (2)!

    @computed_field
    @property
    def warmup_steps(self) -> int:
        """5 % of total steps, rounded to nearest 10."""
        return round(self.total_steps * 0.05 / 10) * 10

cfg = TrainingConfig(epochs=10)
cfg.steps_per_epoch   # → 422
cfg.total_steps       # → 4220
cfg.warmup_steps      # → 210

# Computed fields are in model_dump() — log them all at once
data = cfg.model_dump()
assert "warmup_steps" in data  # ✓
```

1. `@computed_field` must wrap `@property`. The return type annotation is required — Pydantic uses it for the JSON schema.
2. Computed fields can reference other computed fields freely. Evaluation order is determined by attribute access, not declaration order.

!!! warning "Not constructor parameters"
    Computed fields cannot be passed to the constructor — they are always derived:
    ```python
    TrainingConfig(warmup_steps=500)  # ✗ ValidationError: computed fields cannot be set
    ```

---

## 3. Merge with `|`

The `|` operator returns a **new validated instance** with the given overrides applied. The original is never mutated. All validators and computed fields run on the new instance.

```python
base = TrainingConfig(epochs=20, lr=1e-3)

# Top-level fields
fast = base | {"epochs": 3, "lr": 1e-2}
fast.epochs        # → 3
fast.warmup_steps  # → 42  ← automatically recomputed
base.epochs        # → 20  ← untouched
```

Dot-path keys navigate nested sub-configs:

```python
class OptimizerConfig(ConfigBase):
    lr: float           = 1e-3
    weight_decay: float = 0.0

class ExperimentConfig(ConfigBase):
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    epochs: int                = 20

exp     = ExperimentConfig()
variant = exp | {
    "optimizer.lr":           3e-4,  # (1)!
    "optimizer.weight_decay": 1e-2,
    "epochs":                 50,
}
variant.optimizer.lr   # → 0.0003
exp.optimizer.lr       # → 0.001  ← untouched
```

1. Dot-path depth is unlimited. `"model.encoder.hidden_size"` works for three levels of nesting.

!!! danger "Validation always runs"
    Invalid overrides raise `ValidationError` immediately — you cannot produce a bad config through `|`:
    ```python
    base | {"batch_size": 100}  # ✗ ValidationError: must be a power of 2
    ```

---

## 4. Evolve with keywords

For an IDE-friendly, autocompleting way to modify top-level fields, use the `evolve()` method. It takes keyword arguments, immediately validates them against your Pydantic schema, and returns a new config instance.

```python
cfg = (
    TrainingConfig()
    .evolve(
        lr=3e-4,
        epochs=30,
        batch_size=64
    )  # validates everything here
)
# → TrainingConfig(lr=0.0003, epochs=30, batch_size=64, seed=42)
```

1. Method arguments are keyword-only. Your IDE will autocomplete `.evolve(lr=..., epochs=...)`.
2. All validators run here. Setting an invalid value fails immediately.

Pre-populate a base config and fork it:

```python
base_cfg = TrainingConfig(seed=0, lr=1e-3)

short_run = base_cfg.evolve(epochs=5)
long_run  = base_cfg.evolve(epochs=100)
```

!!! tip "`evolve` vs `|`"
    - Use **`evolve()`** when modifying top-level fields, as you get full IDE autocomplete and type-checking before the code even runs.
    - Use **`|`** when you need to override deeply nested fields via dot-paths (e.g., `cfg | {"model.encoder.layers": 12}`).

---

## 5. Discriminated unions

When a field can be one of several types — different optimizers, schedulers, or model architectures — model it as a **discriminated union**. Pydantic dispatches to the correct class based on a `Literal` tag field. The concrete type is preserved through JSON round-trips.

```python
from typing import Annotated, Literal, Union
from pydantic import Field, computed_field, model_validator

class AdamConfig(ConfigBase):
    name: Literal["adam"] = "adam"  # (1)!
    lr: float           = Field(default=1e-3, gt=0.0)
    weight_decay: float = 0.0

    @computed_field
    @property
    def display_name(self) -> str:
        return f"Adam(lr={self.lr:.2e})"

class SGDConfig(ConfigBase):
    name: Literal["sgd"] = "sgd"
    lr: float       = Field(default=1e-2, gt=0.0)
    momentum: float = 0.9
    nesterov: bool  = True

    @model_validator(mode="after")  # (2)!
    def nesterov_requires_momentum(self) -> "SGDConfig":
        if self.nesterov and self.momentum == 0.0:
            raise ValueError("Nesterov requires momentum > 0")
        return self

# The discriminated union — Pydantic dispatches on "name"
OptimizerConfig = Annotated[
    Union[AdamConfig, SGDConfig],
    Field(discriminator="name"),
]

class ExperimentConfig(ConfigBase):
    optimizer: OptimizerConfig = Field(default_factory=AdamConfig)
    epochs: int                = 20

# Pass a dict — type inferred from "name"
cfg = ExperimentConfig(optimizer={"name": "sgd", "lr": 5e-2})
assert isinstance(cfg.optimizer, SGDConfig)   # ✓

# JSON round-trip: concrete type is preserved
restored = ExperimentConfig.model_validate_json(cfg.model_dump_json())
assert type(restored.optimizer) is SGDConfig  # ✓
```

1. The `Literal` tag field is the discriminator key. Every class in the union must have a unique `Literal` value.
2. `@model_validator(mode="after")` runs after all field validators, with access to the fully-constructed instance. Cross-field constraints live here.

=== "Hydra"

    ```yaml
    # conf/optimizer/sgd.yaml — separate file, separate language
    _target_: torch.optim.SGD
    lr: 1e-2
    momentum: 0.9
    ```
    ```python
    # Plus a matching dataclass...
    @dataclass
    class SGDConf:
        _target_: str = "torch.optim.SGD"
        lr: float = 1e-2
    ```

=== "canopee"

    ```python
    # One class, one language, full validation
    class SGDConfig(ConfigBase):
        name: Literal["sgd"] = "sgd"
        lr: float       = Field(default=1e-2, gt=0.0)
        momentum: float = 0.9
    ```

---

## 6. Hyperparameter sweep

Declare a search space over any config field using **typed distributions**, then iterate with grid, random, or Optuna strategies. Every yielded variant is a fully validated config instance.

```python
from canopee.sweep import Sweep, log_uniform, choice, uniform, int_range

base = TrainingConfig()

def train_and_eval(cfg: TrainingConfig) -> float:
    # ... your training logic ...
    return val_loss

best = (
    Sweep(base)
    .vary("lr",         log_uniform(1e-5, 1e-1))  # (1)!
    .vary("batch_size", choice(32, 64, 128, 256))  # (2)!
    .strategy("random", n_samples=20, seed=42)     # (3)!
    .run(train_and_eval)                           # (4)!
    .best(minimize=True)
)

print(f"Best: lr={best.lr:.2e}")
```

1. `log_uniform` samples uniformly in log space — gives equal probability to `[1e-5, 1e-4]` and `[1e-2, 1e-1]`. Ideal for learning rates.
2. `choice` is categorical. Works with any JSON-serialisable values: strings, ints, floats.
3. `seed` makes the sweep fully reproducible. The same 20 configs are generated on every run.
4. `run()` automatically calls your training function and reports the returned metrics to the sweep strategy, allowing chaining.

Switch strategies with a single line:

=== ":fontawesome-solid-th: Grid"

    Exhaustive Cartesian product. `n_points` controls values per continuous axis.

    ```python
    .strategy("grid", n_points=5)
    # → 5 lr values × 4 batch_sizes = 20 variants
    ```

=== ":fontawesome-solid-shuffle: Random"

    Independent sampling per axis. The default strategy.

    ```python
    .strategy("random", n_samples=50, seed=42)
    ```

=== ":simple-optuna: Optuna"

    Bayesian TPE optimisation. Requires `pip install canopee[optuna]`.

    ```python
    .strategy("optuna", n_trials=50, direction="minimize")
    # Must call sweep.report(cfg, metric=...) after each trial
    ```

Dot-path notation works in `.vary()` too — sweep nested fields directly:

```python
sweep = (
    Sweep(experiment_cfg)
    .vary("optimizer.lr",           log_uniform(1e-5, 1e-1))
    .vary("optimizer.weight_decay", log_uniform(1e-6, 1e-1))
    .vary("training.batch_size",    choice(64, 128, 256))
    .strategy("random", n_samples=100, seed=0)
)

# Export all variants as JSON files for distributed workers
paths = sweep.export("./sweep_configs/")
```

---

## You're ready

You've seen every core feature of canopee. Here's where to go next:

<div class="gs-next-grid" markdown>

<div class="gs-next-card" markdown>
**:fontawesome-solid-flask: MNIST Example**

A complete experiment config with 4 optimizers, 5 schedulers, and 3 model architectures — all wired together and swept.

[View example →](../examples/mnist.md)
</div>

<div class="gs-next-card" markdown>
**:fontawesome-solid-database: ConfigStore**

Register named config variants with parent inheritance. Retrieve them anywhere by name — a global registry for your experiment baselines.

[Learn more →](../concepts/store.md)
</div>

<div class="gs-next-card" markdown>
**:fontawesome-solid-book: API Reference**

Full reference for `ConfigBase`, `ConfigStore`, `Sweep`, and all distribution types.

[Browse docs →](../api/core.md)
</div>

</div>