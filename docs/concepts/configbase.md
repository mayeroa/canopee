# ConfigBase & Composition

`ConfigBase` is the heart of canopee. It is a subclass of Pydantic's `BaseModel` that has been explicitly configured for configuration management. 

Every `ConfigBase` instance is **immutable** by default. To modify configurations, you use functional patterns that always return new validated instances.

---

## Immutability (`frozen=True`)

All subclasses of `ConfigBase` are created with `model_config = ConfigDict(frozen=True, extra="forbid")`. 

This brings several massive benefits compared to typical dicts or standard dataclasses:
1. **Thread safety:** You can pass your config to background threads safely.
2. **Hashability:** Configs can be used as cache keys (e.g., `lru_cache`).
3. **No runtime corruption:** It is impossible to accidentally mutate hyperparameter state mid-execution.

```python
from canopee import ConfigBase

class Hyperparams(ConfigBase):
    lr: float = 1e-3

cfg = Hyperparams()
cfg.lr = 1e-2  # ❌ Raises ValidationError: "Instance is frozen"
```

## Creating Modified Copies

Because you cannot mutate configs in-place, you create new modified copies. 

### 1. `evolve(**kwargs)` (Recommended for Python logic)
The `evolve()` method is the IDE-friendly, type-checked way to create a varied config. It accepts exact keyword arguments defined in the class.

```python
fast = cfg.evolve(lr=1e-2)  # ✓ IDE autocompletes `lr`
```

### 2. The `|` Operator (Recommended for dynamic/CLI overrides)
The bitwise OR operator provides a concise syntax for merging dicts. It excels at deeply nested configurations using dot-path notation.

```python
# Assuming a nested config: AppConfig(training=TrainingConfig(lr=1e-3))
app = AppConfig()

fast_app = app | {"training.lr": 1e-2, "dataset.name": "cifar"}
```

Note that the `dict | cfg` symmetry is also perfectly valid:
```python
fast_app = {"training.lr": 1e-2} | app
```

### 3. The `patch` Context Manager
When you need to temporarily override values—most commonly in unit tests—use `patch()`:

```python
with cfg.patch(lr=1e-2) as test_cfg:
    assert test_cfg.lr == 1e-2
    run_test(test_cfg)

# Original is still unchanged outside
assert cfg.lr == 1e-3
```

## Computed Fields

Pydantic's `@computed_field` property decorator is heavily utilized as the native solution to derived configuration variables. Because the `ConfigBase` merge routines explicitly ignore computed fields during deserialisation and revalidation, you will never hit validation errors on computed data.

```python
from pydantic import computed_field

class Training(ConfigBase):
    epochs: int = 20
    
    @computed_field
    @property
    def warmup_steps(self) -> int:
        return self.epochs * 10
```

When you print or serialise the config, `warmup_steps` will be explicitly evaluated and included in the output. If you invoke `cfg.evolve(epochs=50)`, the `warmup_steps` will actively recalculate on the new returned copy.

## Strict Equality and Fingerprinting

A config is globally identity-tracked via its `fingerprint`. The fingerprint is a short 16-character BLAKE3 string deterministically derived from all of its non-computed field properties.

```python
cfg1 = Training(epochs=10)
cfg2 = cfg1.evolve(epochs=10)

assert cfg1 == cfg2
assert cfg1.fingerprint == cfg2.fingerprint
```

When integrating with experiment trackers like Weights & Biases or MLFlow, `cfg.fingerprint` perfectly maps to the unique tag for the run parameters.
