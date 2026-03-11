# Why canopee?

<div class="gs-hero" markdown>
canopee exists because Python deserves a configuration library that speaks Python — not YAML, not string macros, not brittle dataclass decorators. Here's the case for it.
</div>

---

## The landscape

Managing experiment configurations in Python has historically meant accepting one of three bad trades:

<div class="why-options" markdown>

<div class="why-option why-option--bad" markdown>
**Plain dicts / argparse**

:fontawesome-solid-xmark: No type safety — wrong value type silently accepted  
:fontawesome-solid-xmark: No validation — invalid config caught at use time  
:fontawesome-solid-xmark: No IDE support — no autocomplete, no go-to-definition  
:fontawesome-solid-xmark: No composition — copy-pasting config dicts everywhere
</div>

<div class="why-option why-option--bad" markdown>
**YAML / JSON files**

:fontawesome-solid-xmark: No Python types — everything is a string or a dict  
:fontawesome-solid-xmark: String interpolation for derived values — fragile and untyped  
:fontawesome-solid-xmark: Separate file per variant — config sprawl with no inheritance  
:fontawesome-solid-xmark: No validation until runtime
</div>

<div class="why-option why-option--bad" markdown>
**Hydra**

:fontawesome-solid-xmark: YAML-first — Python is second-class  
:fontawesome-solid-xmark: Two worlds to keep in sync: YAML files and dataclasses  
:fontawesome-solid-xmark: OmegaConf dicts instead of real Python objects  
:fontawesome-solid-xmark: Largely unmaintained since 2022
</div>

</div>

canopee is built on a single observation: **Python itself, plus Pydantic v2, is the best configuration language for Python projects.**

---

## The Hydra problem in detail

Hydra pioneered composable config groups, which was a genuinely good idea. But the execution made it harder than it needed to be:

=== "Hydra"

    ```yaml
    # conf/optimizer/adam.yaml — separate file, separate language
    _target_: torch.optim.Adam
    lr: 1e-3
    betas: [0.9, 0.999]
    weight_decay: 0.0
    ```

    ```python
    # conf/optimizer/adam.yaml has to stay in sync with this dataclass
    from dataclasses import dataclass, field
    from omegaconf import MISSING

    @dataclass
    class AdamConf:
        _target_: str = "torch.optim.Adam"
        lr: float = MISSING             # no default, no constraint
        betas: list = field(default_factory=lambda: [0.9, 0.999])
    ```

    ```yaml
    # Derived values via string interpolation — no type checking
    epochs: 20
    warmup_steps: ${multiply:${epochs},100}
    ```

=== "canopee"

    ```python
    # One file, one language, one source of truth
    from canopee import ConfigBase
    from pydantic import Field, computed_field
    from typing import Literal

    class AdamConfig(ConfigBase):
        name: Literal["adam"] = "adam"
        lr: float           = Field(default=1e-3, gt=0.0)  # constraint enforced
        betas: tuple[float, float] = (0.9, 0.999)
        weight_decay: float = Field(default=0.0, ge=0.0)

    class TrainingConfig(ConfigBase):
        epochs: int = 20

        @computed_field
        @property
        def warmup_steps(self) -> int:  # real Python, type-checked
            return self.epochs * 100
    ```

The delta:

| | Hydra | canopee |
|---|---|---|
| Config language | YAML + Python dataclasses | Pure Python |
| Derived values | `${multiply:${a},${b}}` | `@computed_field` |
| Type validation | On access (slow, partial) | On construction (immediate) |
| JSON round-trip | Loses type info | Preserves concrete type |
| IDE autocomplete | Limited | Full — mypy strict, py.typed |
| Maintenance | :fontawesome-solid-xmark: Stalled | :fontawesome-solid-check: Active |

---

## Design principles

### Configs are values, not state

Mutable configs are a footgun. If a function receives a config and modifies it, you get silent experiment pollution that is nearly impossible to debug:

```python
def train(cfg):
    # hypothetically — if cfg were mutable
    cfg.lr *= scheduler_factor   # ← silent mutation
    ...

# Every subsequent experiment uses the wrong lr
```

canopee configs are **frozen Pydantic models** — they behave like `int` or `str`. Immutable, hashable, safe to share across threads, safe to cache. The only way to get a different config is to produce a new one:

```python
cfg = TrainingConfig(lr=1e-3)
cfg.lr = 1e-2    # ✗ FrozenInstanceError — immediately

fast = cfg | {"lr": 1e-2}   # ✓ new instance, original untouched
```

### Python is the right config DSL

The best config language for a Python project is Python. You get:

- Native types (`int`, `float`, `tuple[float, float]`, `Literal["adam"]`)
- Real logic in validators (`@field_validator`, `@model_validator`)
- Proper derived fields (`@computed_field`)
- The entire Python ecosystem for free

No new syntax to learn. No string macros. No hidden evaluation order. Just Python.

### Type safety all the way down

Discriminated unions let you model "one of these types" in a way that is fully type-checked, serialisable, and validated — without any runtime `isinstance` sprawl:

```python
OptimizerConfig = Annotated[
    Union[AdamConfig, AdamWConfig, SGDConfig, RMSpropConfig],
    Field(discriminator="name"),
]

# Pass a dict — Pydantic infers AdamConfig from "name": "adam"
cfg = ExperimentConfig(optimizer={"name": "adam", "lr": 1e-3})

# Unknown tag raises ValidationError immediately
ExperimentConfig(optimizer={"name": "transformer"})
# ✗ ValidationError: 'transformer' is not a valid discriminator value

# JSON round-trip preserves the exact concrete type
restored = ExperimentConfig.model_validate_json(cfg.model_dump_json())
assert type(restored.optimizer) is AdamConfig  # ✓
```

### Sweep is a first-class citizen

Hyperparameter search is not an afterthought. The same config object you use for a single training run is what you hand to `Sweep`. Distributions are typed Pydantic models — serialisable and reproducible. Strategies are pluggable.

```python
sweep = (
    Sweep(base_cfg)
    .vary("optimizer.lr", log_uniform(1e-5, 1e-1))
    .vary("model.dropout", uniform(0.0, 0.5))
    .strategy("optuna", n_trials=100, direction="minimize")
)
```

---

## When canopee is the right choice

!!! success "Good fit"
    - You train ML models and want reproducible, fingerprinted experiment configs
    - You have multiple component variants (optimizers, schedulers, architectures) and want type-safe dispatch
    - You run hyperparameter sweeps and want them integrated with your config system
    - You care about IDE support, `mypy --strict`, and catching errors at definition time

!!! warning "May not be the right fit"
    - You need configs loaded from many YAML files written by non-engineers — Hydra's composable YAML groups are genuinely good for this
    - You need a config system shared across a non-Python service — canopee is Python-only

---

## Next

<div class="gs-next" markdown>

[Quickstart — build your first config :fontawesome-solid-arrow-right:](quickstart.md){ .md-button .md-button--primary }

[Installation :fontawesome-solid-arrow-right:](installation.md){ .md-button }

</div>