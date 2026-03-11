# MNIST Training Example

This example shows a complete, production-style configuration for MNIST experiments. It demonstrates every major canopee feature working together: discriminated unions, computed fields, model validators, the ConfigStore, and the Sweep engine.

---

## Architecture overview

The config hierarchy looks like this:

```
MNISTExperimentConfig
├── model:      MLPConfig | CNNConfig | ResNetMiniConfig   (discriminated union)
├── optimizer:  AdamConfig | AdamWConfig | SGDConfig | RMSpropConfig
├── scheduler:  ConstantSchedulerConfig | StepLRConfig | CosineAnnealingConfig
│               | OneCycleLRConfig | ReduceLROnPlateauConfig
├── data:       DataConfig
├── training:   TrainingConfig
├── checkpoint: CheckpointConfig
└── logging:    LoggingConfig
```

Every component is a separate `ConfigBase` subclass. The top-level `MNISTExperimentConfig` composes them and adds cross-cutting computed fields.

---

## Optimizer configs

Four optimizers are modelled as a discriminated union on the `name` field:

```python
from typing import Annotated, Literal, Union
from pydantic import Field, computed_field, field_validator, model_validator
from canopee import ConfigBase

class AdamConfig(ConfigBase):
    name: Literal["adam"] = "adam"
    lr: float = Field(default=1e-3, gt=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(default=1e-8, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)

    @computed_field
    @property
    def display_name(self) -> str:
        return f"Adam(lr={self.lr:.2e}, wd={self.weight_decay})"

    @field_validator("betas")
    @classmethod
    def betas_in_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        b1, b2 = v
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {v}")
        return v


class SGDConfig(ConfigBase):
    name: Literal["sgd"] = "sgd"
    lr: float = Field(default=1e-2, gt=0.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    nesterov: bool = True

    @model_validator(mode="after")
    def nesterov_requires_momentum(self) -> "SGDConfig":
        if self.nesterov and self.momentum == 0.0:
            raise ValueError("Nesterov requires momentum > 0")
        return self

# ... AdamWConfig, RMSpropConfig defined similarly

OptimizerConfig = Annotated[
    Union[AdamConfig, AdamWConfig, SGDConfig, RMSpropConfig],
    Field(discriminator="name"),
]
```

Pydantic dispatches on the `name` field — passing `{"name": "sgd"}` produces a `SGDConfig`, and type information is preserved through JSON round-trips.

---

## Scheduler configs

Five schedulers as a discriminated union:

=== "Cosine Annealing"
    ```python
    class CosineAnnealingConfig(ConfigBase):
        name: Literal["cosine"] = "cosine"
        eta_min: float = Field(default=1e-6, ge=0.0)

        @computed_field
        @property
        def display_name(self) -> str:
            return f"CosineAnnealing(η_min={self.eta_min:.1e})"
    ```

=== "OneCycleLR"
    ```python
    class OneCycleLRConfig(ConfigBase):
        name: Literal["one_cycle"] = "one_cycle"
        max_lr_factor: float = Field(default=10.0, gt=1.0)
        pct_start: float = Field(default=0.3, gt=0.0, lt=1.0)
        anneal_strategy: Literal["cos", "linear"] = "cos"
    ```

=== "ReduceOnPlateau"
    ```python
    class ReduceLROnPlateauConfig(ConfigBase):
        name: Literal["reduce_on_plateau"] = "reduce_on_plateau"
        mode: Literal["min", "max"] = "min"
        factor: float = Field(default=0.1, gt=0.0, lt=1.0)
        patience: int = Field(default=5, gt=0)
    ```

---

## Model configs

Three architectures as a discriminated union on `architecture`:

=== "MLP"
    ```python
    class MLPConfig(ConfigBase):
        architecture: Literal["mlp"] = "mlp"
        hidden_dims: list[int] = [512, 256, 128]
        activation: Literal["relu", "gelu", "tanh", "silu"] = "relu"
        dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

        @computed_field
        @property
        def num_layers(self) -> int:
            return len(self.hidden_dims)

        @computed_field
        @property
        def total_params_estimate(self) -> int:
            dims = [784] + self.hidden_dims + [10]
            return sum(
                dims[i] * dims[i+1] + dims[i+1]
                for i in range(len(dims) - 1)
            )
    ```

=== "CNN"
    ```python
    class CNNConfig(ConfigBase):
        architecture: Literal["cnn"] = "cnn"
        channels: list[int] = [32, 64]
        kernel_size: int = Field(default=3, ge=1, le=7)
        dropout: float = Field(default=0.25, ge=0.0, lt=1.0)
        batch_norm: bool = True

        @model_validator(mode="after")
        def kernel_size_odd(self) -> "CNNConfig":
            if self.kernel_size % 2 == 0:
                raise ValueError(f"kernel_size should be odd, got {self.kernel_size}")
            return self
    ```

=== "ResNet Mini"
    ```python
    class ResNetMiniConfig(ConfigBase):
        architecture: Literal["resnet_mini"] = "resnet_mini"
        num_blocks: list[int] = [2, 2]
        base_channels: int = Field(default=16, gt=0)

        @computed_field
        @property
        def total_blocks(self) -> int:
            return sum(self.num_blocks)
    ```

---

## Top-level experiment config

`MNISTExperimentConfig` composes all sub-configs and adds experiment-level computed fields:

```python
class MNISTExperimentConfig(ConfigBase):
    model:      ModelConfig    = Field(default_factory=MLPConfig)
    optimizer:  OptimizerConfig = Field(default_factory=AdamConfig)
    scheduler:  SchedulerConfig = Field(default_factory=CosineAnnealingConfig)
    data:       DataConfig     = Field(default_factory=DataConfig)
    training:   TrainingConfig = Field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging:    LoggingConfig   = Field(default_factory=LoggingConfig)

    experiment_name: str = "mnist-baseline"

    @computed_field
    @property
    def steps_per_epoch(self) -> int:
        n_train = int(60_000 * self.data.train_split)
        return math.ceil(n_train / self.training.batch_size)

    @computed_field
    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.training.epochs

    @computed_field
    @property
    def warmup_steps(self) -> int:
        return round(self.total_steps * 0.05 / 10) * 10

    @computed_field
    @property
    def summary(self) -> str:
        return (
            f"[{self.experiment_name}] "
            f"{self.model.display_name} | "
            f"{self.optimizer.display_name} | "
            f"epochs={self.training.epochs}"
        )
```

---

## Named experiment registry

The global `ConfigStore` provides a dictionary-like API to register and retrieve configurations.

```python
from canopee import ConfigStore

ConfigStore["mlp_baseline"] = MNISTExperimentConfig(
    model=MLPConfig(hidden_dims=[512, 256, 128], dropout=0.2),
    optimizer=AdamConfig(lr=1e-3),
    scheduler=CosineAnnealingConfig(),
)

ConfigStore["cnn_adamw"] = MNISTExperimentConfig(
    model=CNNConfig(channels=[32, 64], batch_norm=True),
    optimizer=AdamWConfig(lr=1e-3, weight_decay=1e-2),
    scheduler=OneCycleLRConfig(max_lr_factor=10.0),
)

# You can inherit from another config to only specify what changes
ConfigStore.register(
    "cnn_augmented",
    MNISTExperimentConfig(
        data=DataConfig(augment_train=True),
    ),
    parent="cnn_adamw",  # inherits cnn_adamw, augmentation added on top
)
```

---

## Hyperparameter sweep

```python
from canopee.sweep import Sweep, log_uniform, choice, uniform

base = ConfigStore["cnn_adamw"]

def train_and_evaluate(cfg: MNISTExperimentConfig) -> float:
    # return accuracy
    ...
    return 0.95

best_cfg = (
    Sweep(base)
    .vary("optimizer.lr",           log_uniform(1e-5, 1e-1))
    .vary("optimizer.weight_decay", log_uniform(1e-6, 1e-1))
    .vary("training.batch_size",    choice(64, 128, 256))
    .vary("model.dropout",          uniform(0.0, 0.5))
    .strategy("random", n_samples=50, seed=42)
    .run(lambda cfg: 1 - train_and_evaluate(cfg)) # minimize 1 - accuracy
    .best(minimize=True)
)

print(f"Best: lr={best_cfg.optimizer.lr:.2e}")
print(f"Summary: {best_cfg.summary}")
```

---

## Running the demo

```bash
cd canopee
python examples/mnist_config.py
```

Expected output:

```
──────────────────────── Registered experiments ────────────────────────
  mlp_baseline          [mlp-baseline] MLP[512→256→128](relu) | Adam(lr=1.00e-03, wd=0.0) | ...
  cnn_adamw             [cnn-adamw] CNN[ch=32→64, k=3] | AdamW(lr=1.00e-03, wd=0.01) | ...
  resnet_sgd            [resnet-sgd] ResNetMini[2+2](base_ch=16) | SGD(lr=1.00e-01, ...) | ...
  fast_dev              [fast-dev] MLP[64→32](relu) | Adam(lr=1.00e-03, wd=0.0) | ...
──────────────────────────────────────────────────────────────────────────
──────────────── MLP baseline — computed fields ────────────────────────
  steps_per_epoch : 422
  total_steps     : 8440
  warmup_steps    : 420
  fingerprint     : a3f9c12e0b7d4158
  model params~   : 534,666
...
```
