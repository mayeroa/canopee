from canopee import ConfigBase, Patch, evolve, wrap, save
from canopee.cli import clify
from pydantic import Field


class OptimizerConfig(ConfigBase):
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


class TrainingConfig(ConfigBase):
    epochs: int = 20
    batch_size: int = 128
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


# --- 1. Evolution ---
cfg = TrainingConfig()

# Keyword style (free function)
cfg2 = evolve(cfg, epochs=50, optimizer__learning_rate=3e-4)

# Merge operator (infix)
cfg3 = cfg | {"epochs": 50}
cfg4 = cfg | Patch({"optimizer.learning_rate": 3e-4})

# Fluent style (opt-in proxy)
cfg5 = wrap(cfg).evolve(epochs=100).apply({"batch_size": 64}).unwrap()


# --- 2. CLI and I/O ---
@clify(TrainingConfig)
def train(cfg: TrainingConfig):
    print(f"Training for {cfg.epochs} epochs")

    # Save using free functions to avoid name shadowing
    save(cfg, "run_config.toml")


if __name__ == "__main__":
    train()
