from canopee import ConfigBase, clify
from pydantic import Field

class OptimizerConfig(ConfigBase):
    lr: float = Field(default=1e-3, description="Learning rate")
    beta: float = 0.9

class TrainingConfig(ConfigBase):
    epochs: int = 10
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    verbose: bool = True

@clify(TrainingConfig)
def main(cfg: TrainingConfig):
    print(f"LR: {cfg.optimizer.lr}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Verbose: {cfg.verbose}")

if __name__ == "__main__":
    # Test with default values
    print("--- Default values ---")
    main(_argv=[])
    
    # Test with overrides
    print("\n--- Overrides ---")
    main(_argv=["--epochs", "20", "--optimizer.lr", "3e-4", "--no-verbose"])
