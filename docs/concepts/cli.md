# CLI Integration

Canopée provides a powerful and flexible way to expose your `ConfigBase` models as fully-featured Command Line Interfaces (CLI). Using the `@clify` decorator, you can turn any function that takes a configuration instance into a CLI entry point with one flag per configuration field.

---

## The `@clify` Decorator

The `@clify` decorator recursively inspects your Pydantic-based configuration classes and generates corresponding CLI arguments. It handles nested models using dot-path notation (e.g., `--optimizer.lr`), making it an ideal companion for complex experiment configurations.

### Basic Usage (`argparse`)

By default, `@clify` uses the standard library's `argparse` backend, requiring no additional dependencies.

```python
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
    print(f"Running with LR: {cfg.optimizer.lr}, Epochs: {cfg.epochs}")

if __name__ == "__main__":
    main()
```

**CLI Usage:**
```bash
python train.py --epochs 20 --optimizer.lr 3e-4 --no-verbose
```

---

## Multiple Backends

Canopée supports several CLI frameworks as backends. While `argparse` is always available, you can opt-in to `Click` or `Typer` for a more polished experience, including rich help formatting and shell completion.

### Using Click

To use the [Click](https://click.palletsprojects.com/) backend, install it via `pip install click` and specify `backend="click"`.

```python
@clify(TrainingConfig, backend="click")
def main(cfg: TrainingConfig):
    ...
```

### Using Typer

To use the [Typer](https://typer.tiangolo.com/) backend (which provides beautiful, colorized help output), install it via `pip install typer` and specify `backend="typer"`.

```python
@clify(TrainingConfig, backend="typer")
def main(cfg: TrainingConfig):
    ...
```

---

## Nested Configurations

One of the most powerful features of `@clify` is its ability to flatten nested `ConfigBase` models into dot-separated flags. This mirrors the behavior of the `|` merge operator and ensures consistency between your code and your command line.

For a configuration like:
```python
class AppConfig(ConfigBase):
    training: TrainingConfig
```

The generated flags will include:
- `--training.epochs`
- `--training.optimizer.lr`
- `--training.verbose` / `--no-training.verbose`

---

## Support for Complex Types

`@clify` handles a wide variety of Pydantic types out of the box:

| Type | CLI Representation | Example |
| :--- | :--- | :--- |
| `bool` | Boolean flags | `--verbose` / `--no-verbose` |
| `int`, `float`, `str` | Standard flags | `--epochs 10` |
| `Literal` | Choice validation | `--mode fast` (validates against allowed choices) |
| `list[T]`, `tuple[T, ...]` | Space-separated values | `--numbers 1 2 3` |
| `Optional[T]` | Optional flags | `--tag null` (to explicitly unset) |
| `Union`, `dict`, `Any` | JSON-encoded strings | `--extra '{"key": "value"}'` |

---

## Customization

You can customize the generated CLI by passing additional arguments to `@clify`:

- `prog`: Override the program name shown in `--help`.
- `description`: Override the top-level help description (defaults to the function's docstring).

```python
@clify(TrainingConfig, prog="trainer", description="Custom training script")
def main(cfg: TrainingConfig):
    ...
```
