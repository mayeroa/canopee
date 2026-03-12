from canopee import ConfigBase, pretty_print_error
from pydantic import Field, ValidationError


class OptimizerConfig(ConfigBase):
    lr: float = Field(..., ge=0, le=1)
    beta: float = 0.9


class TrainingConfig(ConfigBase):
    epochs: int = Field(..., gt=0)
    optimizer: OptimizerConfig


def main():
    print("Testing pretty_print_error with invalid data...")
    invalid_data = {
        "epochs": -5,  # Should be > 0
        "optimizer": {
            "lr": 1.5,  # Should be <= 1
            "beta": "not-a-float",
        },
    }

    try:
        TrainingConfig.model_validate(invalid_data)
    except ValidationError as e:
        pretty_print_error(e, title="My App Config Error")


if __name__ == "__main__":
    main()
