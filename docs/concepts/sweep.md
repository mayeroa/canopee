# Sweep Engine

The Hyperparameter Sweep engine in Canopee is designed to be elegant, fully typed, and closely integrated with the `ConfigBase` immutability guarantees. Instead of treating grid search as an external script, Canopee treats it as an iterator over validated variants.

## Basic Usage

The `Sweep` object takes a base configuration and allows you to vary specific properties using typed distributions.

```python
from canopee.sweep import Sweep, log_uniform, choice

base = TrainingConfig()

# We want to minimize the validation loss
def train_and_eval(cfg: TrainingConfig) -> float:
    return run_experiment(cfg)

best_cfg = (
    Sweep(base)
    .vary("lr",         log_uniform(1e-5, 1e-1))
    .vary("batch_size", choice(32, 64, 128))
    .strategy("random", n_samples=20, seed=42)
    .run(train_and_eval)
    .best(minimize=True)
)
```

The pipeline:
1. `vary("path", distribution)` registers a search space. Dot-paths work securely.
2. `strategy("type", ...)` prepares the iterator.
3. `run(fn)` iterates through all configs, calls your function, and reports metrics.
4. `best(minimize=True)` grabs the config that had the lowest returned metric.

## Distribution Types

Distributions map to the underlying sampling techniques (like Optuna `suggest_float`):

| Distribution | Example | Description |
|--------------|---------|-------------|
| **`choice`** | `choice("relu", "gelu")` | Categorical list of options |
| **`uniform`** | `uniform(0.1, 0.9)` | Float sampled uniformly |
| **`int_range`** | `int_range(1, 10)` | Integer sampled uniformly |
| **`log_uniform`** | `log_uniform(1e-5, 1e-1)` | Log-spaced uniform sampling |

## Strategies

### 1. `grid`
Generates the exhaustive cartesian product of every option.
```python
sweep.strategy("grid", n_points=5)
# A `uniform` distribution above would be discretized to 5 points.
```

### 2. `random` 
Randomly samples each varied field independently. A `seed` value guarantees absolute deterministic reproducibility.
```python
sweep.strategy("random", n_samples=100, seed=123)
```

### 3. `optuna`
Uses the TPE Bayesian optimizer. You must install the `[optuna]` extra. It learns which hyperparameters perform best based on the reported metrics during `.run()`.
```python
sweep.strategy("optuna", n_trials=50, direction="minimize")
```

## Distributed Exporting

Often you run the sweeps on external clusters. You can generate and serialize all variants to files without evaluating them:

```python
sweep = Sweep(base).vary(...).strategy("grid")

# Writes 20 config files: run_00.yaml, run_01.yaml...
sweep.export("sweep_configs/", fmt="yaml")
```
