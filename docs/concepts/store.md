# Store Registry

The `ConfigStore` provides a global, dictionary-like registry for named configuration instances. It allows you to separate your configuration definitions from your application's execution logic by allowing you to inject configurations purely by a string name.

## Registering Configs

You can register any `ConfigBase` instance into the store via key assignment:

```python
from canopee import ConfigStore

ConfigStore["baseline"] = TrainConfig(lr=1e-3, epochs=20)
ConfigStore["fast"] = TrainConfig(lr=1e-2, epochs=5)
```

Alternatively, you can register default classes purely at definition time using the `@ConfigStore.entry` decorator:

```python
@ConfigStore.entry("production-v2")
class ProductionConfig(ConfigBase):
    workers: int = 16
    retries: int = 5
```

## Retrieving Configs

To retrieve configs, you use the standard dictionary index syntax. You can optionally add a Type parameter to guarantee type-checking and preserve IDE autocomplete for your downstream logic.

```python
# Untyped retrieval
cfg = ConfigStore["baseline"]

# Typed retrieval
cfg = ConfigStore["baseline", TrainConfig]
```

## Inheritance

It's common to have a base configuration, and a dozen variants that only change one or two values. The `ConfigStore` supports this through the `parent` keyword in the `.register()` method.

```python
ConfigStore.register("baseline", TrainConfig(lr=1e-3, epochs=20))

# 'fast' inherits everything from 'baseline', but sets epochs=5
ConfigStore.register("fast", TrainConfig(epochs=5), parent="baseline")
```

When you request `"fast"`, the framework cleanly executes `ConfigStore["baseline"] | TrainConfig(epochs=5)` under the hood and permanently caches the resulting full configuration object.

## Introspection

You can programmatically iterate or check the sizes of all stored configurations:

```python
"baseline" in ConfigStore  # True
len(ConfigStore)           # 2
list(ConfigStore)          # ["baseline", "fast"]

# View parent inheritance chain history!
ConfigStore.lineage("fast") # ["baseline", "fast"]
```
