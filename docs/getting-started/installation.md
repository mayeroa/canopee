# Installation

<div class="gs-hero" markdown>
Get canopee running in under a minute. The core package has a single mandatory dependency — Pydantic v2 — and ships optional extras for Optuna sweep support and experiment tracking.
</div>

---

## Requirements

<div class="req-grid" markdown>

<div class="req-card" markdown>
:fontawesome-brands-python: **Python ≥ 3.11**

Uses `tomllib` from the standard library, `match` statements, and `Self` type — all 3.11+.
</div>

<div class="req-card" markdown>
:simple-pydantic: **Pydantic ≥ 2.6**

Built exclusively on Pydantic v2's `model_validator`, `computed_field`, and discriminated unions. Pydantic v1 is not supported.
</div>

</div>

---

## Install

=== ":fontawesome-solid-bolt: Basic"

    The core package — everything you need to define configs, use the builder, and run sweeps.

    ```bash
    pip install canopee
    ```

=== ":simple-optuna: + Optuna"

    Adds Bayesian hyperparameter optimisation via Optuna's TPE sampler.

    ```bash
    pip install "canopee[optuna]"
    ```

=== ":simple-mlflow: + Tracking"

    Adapters for MLflow and Weights & Biases experiment logging.

    ```bash
    pip install "canopee[tracking]"
    ```

=== ":fontawesome-solid-layer-group: Everything"

    All optional extras in one command.

    ```bash
    pip install "canopee[all]"
    ```

---

## Verify

After installing, confirm everything is working:

```python
import canopee
print(canopee.__version__)  # → 0.1.0

from canopee import ConfigBase
from pydantic import Field

class MyConfig(ConfigBase):
    lr: float = Field(default=1e-3, gt=0)

cfg = MyConfig()
print(cfg.lr)  # → 0.001
```

---

## Development install

To contribute or run the test suite locally:

```bash
git clone https://github.com/your-org/canopee.git
cd canopee
pip install -e ".[dev]"
pytest                    # run the full test suite
pytest -k "mnist" -v      # run only MNIST-related tests
```

!!! tip "Pre-commit hooks"
    The repo ships a `.pre-commit-config.yaml` with `ruff` and `mypy`. Run `pre-commit install` after cloning to enforce formatting and type-checking on every commit.

---

## Extras reference

| Extra | Installs | Use case |
|---|---|---|
| `optuna` | `optuna ≥ 3.0` | Bayesian sweep strategy |
| `tracking` | `mlflow ≥ 2.0` | Experiment logging |
| `toml` | `tomli` (Python < 3.11) | TOML config file loading |
| `dev` | `pytest`, `mypy`, `ruff` | Contributing / testing |
| `all` | All of the above | Everything at once |

!!! note "TOML on Python 3.11+"
    Python 3.11 ships `tomllib` in the standard library. The `toml` extra is only needed on Python 3.10.

---

## Next

<div class="gs-next" markdown>

[Quickstart — define your first config in 5 minutes :fontawesome-solid-arrow-right:](quickstart.md){ .md-button .md-button--primary }

[Why canopee? :fontawesome-solid-arrow-right:](why.md){ .md-button }

</div>