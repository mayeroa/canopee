# Sources & I/O

Canopee makes loading and deserializing configurations highly symmetric and strictly typed. It supports explicit parsing from local files, environment variables, command line arguments, and plain dictionaries.

## File I/O (`save` and `load`)

To persist or reload a config, you can rely on the `.save()` and `.load()` operations directly on the config instances/classes:

```python
cfg = TrainingConfig()

# Export: format is natively determined by the extension!
cfg.save("run.toml")
cfg.save("metrics.json", indent=4)
cfg.save("override.yaml")

# Import: fully validated!
reloaded = TrainingConfig.load("run.toml")
```

Under the hood, this transparently delegates to the `canopee.serialization` module. You don't need any complex serializers or boilerplate.

## String Serialisation (`dumps` and `loads`)

If you work with databases or network APIs, string deserialisation acts precisely the same way.

```python
text_data = cfg.dumps("yaml")

# Deserialize
cfg_new = TrainingConfig.loads("yaml", text_data)
```

## Advanced Loading: The `from_*` APIs

When instantiating configurations dynamically in a production CLI app, you usually want to merge multiple override sources dynamically. Canopee handles this natively:

### Command Line Arguments
```bash
python script.py --epochs=50 --optimizer.lr=3e-4
```
```python
import sys
cfg = TrainingConfig.from_cli(sys.argv[1:])
```

### Environment Variables
Environment variables can override any field. They map to nested properties using double underscores (`__`).

```bash
export APP_EPOCHS=10
export APP_OPTIMIZER__LR=1e-3
```
```python
cfg = TrainingConfig.from_env(prefix="APP_")
```

### Merging Multiple Sources

For extreme composability, you can declare a prioritized list of `Source` providers. Later sources cleanly override earlier sources.

```python
from canopee.sources import FileSource, EnvSource, CLISource

cfg = TrainingConfig.from_sources([
    FileSource("base_config.toml"),
    EnvSource(prefix="APP_"),
    CLISource(sys.argv[1:]),
])
```
This elegantly emulates modern cascade-based hierarchies without polluting your typed class properties!
