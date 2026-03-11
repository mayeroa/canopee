# Core API Reference

The Canopee API is designed to be minimal. The following objects are the only primitives you need:

---

## `canopee.ConfigBase`

The foundational class. Subclass this to define your configuration models. Note that all configurations inherently inherit from Pydantic's `BaseModel`, running all `field_validator`, `model_validator`, and enforcing `extra="forbid"`.

### Creation
- `ConfigBase(**kwargs)`: Instantiate and validate a fresh configuration block.

### Instance Methods
- `cfg.evolve(**kwargs)`: Returns a new frozen instance with the keyword arguments mutated. The original is untouched.
- `cfg | {"dot.path": value}`: Bitwise OR operator, creating a new instance by deeply merging a dictionary across nested instances.
- `cfg.patch(**kwargs)`: A context manager for temporarily entering a block with overridden values.
- `cfg.diff(other)`: Returns a clear dictionary mapping `{"field": (self_value, other_value)}`.
- `cfg.fingerprint`: Determines a unique, reproducible 16-character BLAKE3 string based purely on the values of non-computed attributes.

### Class Methods (Sources & Serialization)
- `ConfigBase.load(path: str | Path)`: Determines serialization format based on `.json`, `.toml`, or `.yaml` and validates a returned object from disk.
- `cfg.save(path: str | Path)`: Determines serialization format from path and writes the frozen config instance to disk.
- `cfg.dumps(format)`: Converts the configuration directly to a string representation.
- `ConfigBase.loads(format, text)`: Validates reading an object from string formats.
- `ConfigBase.from_env(prefix="APP_")`: Loads nested dictionary definitions natively from the operative OS environment context.
- `ConfigBase.from_cli(argv)`: Dissects standard sys.argv mapping (like `--optimizer.lr=5`) directly onto a nested class validation tree.

---

## `canopee.ConfigStore`

A module-level mutable singleton storing definitions for rapid access and cross-cutting inheritance tracking across multiple source files.

### Object Registration
```python
ConfigStore["base"] = TrainConfig()
```

- `@ConfigStore.entry(name, **kwargs)`: Bind a definition statically inside a module scope.
- `ConfigStore.register(name, config, parent=None)`: Register the configuration conditionally inheriting all definitions dynamically from the assigned `parent`.

### Retrieval
- `cfg = ConfigStore["name"]`: Returns generic `ConfigBase`.
- `cfg = ConfigStore["name", TrainConfig]`: Returns purely type-safe validated sub-variant.
- `ConfigStore.list()`: Return active loaded names.
- `ConfigStore.lineage("name")`: Prints the full ancestor tree history string array.

---

## `canopee.sweep.Sweep`

A lightweight engine for hyperparameter grid execution mapping.

### Generation
- `Sweep(base)`: Accepts an immutable baseline root.
- `.vary(path, distribution)`: Determines independent variable axis paths. Distribution imports include `choice`, `uniform`, `int_range`, and `log_uniform`.
- `.strategy(kind, **params)`: Set the overarching permutation logic. Valid distributions include `grid` (deterministic cartesian mappings), `random` (deterministic independent variables), and `optuna` (native TPE).

### Execution Pipeline 
- `.run(evaluation_fn)`: Recursively unrolls the active strategy context, feeding each frozen variant dynamically to the target evaluation pipeline. Metrics automatically report back.
- `.results()`: Get the `(variant, accuracy)` pairings explicitly listed.
- `.best(minimize=True)`: Extract the pure optimal frozen configuration found in previous cycles.
