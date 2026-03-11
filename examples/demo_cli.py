from typing import Literal
from canopee import ConfigBase, clify
from pydantic import Field


class OptimizerConfig(ConfigBase):
    lr: float = Field(default=1e-3, description="Learning rate")
    beta: float = Field(default=0.9, description="Beta value")


class TrainingConfig(ConfigBase):
    epochs: int = Field(default=10, ge=1, le=1000, description="Number of epochs")
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    verbose: bool = Field(default=True, description="Verbose mode")
    batch_size: int = Field(default=32, ge=1, le=1024, description="Batch size")
    dataset: Literal["mnist", "cifar10", "imagenet"] = Field(default="mnist", description="Dataset to use")
    steps: tuple[int, int] = Field(default=(10, 100), description="Steps to train")
    text: dict[str, int] = Field(default_factory=dict)


@clify(TrainingConfig, backend='typer', prog='Trainer', description='Train a model')
def main(cfg: TrainingConfig):
    """Main function to train a model."""
    print("Config Loaded:")
    print(f"  LR: {cfg.optimizer.lr}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Verbose: {cfg.verbose}")
    print(f"  Batch Size: {cfg.batch_size}")
    print(f"  Dataset: {cfg.dataset}")
    print(f"  Steps: {cfg.steps}")
    print(f"  Text: {cfg.text}")


if __name__ == "__main__":
    main()
