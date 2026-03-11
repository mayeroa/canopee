# Standard libraries
from typing import Annotated, Literal

# Third-party libraries
from pydantic import Field
from torch import nn
from torch.optim import Adam, Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Local libraries
from canopee import ConfigBase


# ============================================================
# Data Config
# ============================================================
class DataConfig(ConfigBase):
    """Dataset and dataloader configuration."""

    batch_size: int = 64
    num_workers: int = 4

    def build(self) -> tuple[DataLoader, DataLoader]:
        """Build the dataloader."""
        return self.build_train_loader(), self.build_val_loader()

    def build_train_loader(self) -> DataLoader:
        """Build the training dataloader."""
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def build_val_loader(self) -> DataLoader:
        """Build the validation dataloader."""
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# ============================================================
# Optimizer Configs (Tagged Union)
# ============================================================
class AdamConfig(ConfigBase):
    """Adam optimizer configuration."""

    kind: Literal["adam"] = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0

    def build(self, params) -> Optimizer:
        """Build the optimizer."""
        return Adam(params=params, lr=self.lr, weight_decay=self.weight_decay)


class SGDConfig(ConfigBase):
    """SGD optimizer configuration."""

    kind: Literal["sgd"] = "sgd"
    lr: float = 0.01
    momentum: float = 0.9

    def build(self, params) -> Optimizer:
        """Build the optimizer."""
        return SGD(params=params, lr=self.lr, momentum=self.momentum)


OptimizerConfig = Annotated[AdamConfig | SGDConfig, Field(discriminator="kind")]


# ============================================================
# Scheduler Config
# ============================================================
class StepLRConfig(ConfigBase):
    """StepLR scheduler configuration."""

    kind: Literal["step"] = "step"
    step_size: int = 5
    gamma: float = 0.5

    def build(self, optimizer: Optimizer) -> LRScheduler:
        """Build the scheduler."""
        return StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)


class CosineAnnealingConfig(ConfigBase):
    """CosineAnnealingLR scheduler configuration."""

    kind: Literal["cosine"] = "cosine"
    T_max: int = 10

    def build(self, optimizer: Optimizer) -> LRScheduler:
        """Build the scheduler."""
        return CosineAnnealingLR(optimizer, T_max=self.T_max)


SchedulerConfig = Annotated[StepLRConfig | CosineAnnealingConfig, Field(discriminator="kind")]


# ============================================================
# Model Config
# ============================================================
class MLPConfig(ConfigBase):
    """MLP model configuration."""

    kind: Literal["mlp"] = "mlp"
    hidden_dim: int = 128

    def build(self) -> nn.Module:
        """Build the model."""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 10),
        )


class CNNConfig(ConfigBase):
    """CNN model configuration."""

    kind: Literal["cnn"] = "cnn"

    def build(self) -> nn.Module:
        """Build the model."""
        return nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(26 * 26 * 32, 10),
        )


ModelConfig = Annotated[MLPConfig | CNNConfig, Field(discriminator="kind")]


# ============================================================
# App Config
# ============================================================
class AppConfig(ConfigBase):
    """App configuration."""

    model: ModelConfig
    optimizer: OptimizerConfig
    dataset: str = "cifar10"
    batch_size: int = 32


if __name__ == "__main__":
    # --- Build config directly ---
    print("--- Build config ---")
    config = AppConfig(
        model=MLPConfig(hidden_dim=128),
        optimizer=AdamConfig(lr=1e-3),
        dataset="mnist",
        batch_size=32,
    )
    print(config)

    # --- Serialize config to yaml / json / toml ---
    print("--- Serialize config to files (yaml, json, toml) ---")
    config.save("config.yaml")
    config.save("config.json", indent=4)
    config.save("config.toml")

    # --- Build config from yaml ---
    print("--- Build config from yaml ---")
    reloaded = AppConfig.load("config.yaml")

    print(f"Configs are equal: {config == reloaded}")

    # --- Evolve config ---
    print("--- Evolve config ---")
    fast_config = config.evolve(batch_size=256)
    print(fast_config)
