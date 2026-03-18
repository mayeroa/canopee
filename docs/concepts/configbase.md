# ConfigBase & Composition

`ConfigBase` is the heart of Canopée. It is a subclass of Pydantic's `BaseModel` explicitly designed for configuration management with a "collision-free" philosophy.

---

## Namespace Philosophy

In many configuration libraries, the base class is cluttered with methods like `.save()`, `.load()`, or `.evolve()`. This creates a problem: if you want a configuration field named `save` or `load`, it will shadow the library's method, leading to bugs or restricted naming.

**Canopée solves this by exposing almost nothing on the `ConfigBase` instance.** 

The entire API lives in **free functions** or an opt-in **proxy**. This ensures that you can safely use *any* field name in your configuration without risk of collision.

```python
from canopee import ConfigBase, evolve, save

class TrainingConfig(ConfigBase):
    save: str = "checkpoint.pt"   # 🚀 Perfectly safe! No method collision.
    load: str = "pretrained.pt"   # 🚀 Also safe.
    name: str = "baseline"

cfg = TrainingConfig()

# Use free functions instead of methods
cfg2 = evolve(cfg, name="experiment-1")
save(cfg2, "config.toml")
```

---

## Immutability (`frozen=True`)

All `ConfigBase` instances are **immutable**. Once created, they cannot be changed. This brings several massive benefits:

1. **Thread safety:** Pass configs across threads without worry.
2. **Hashability:** Use configs as cache keys or in sets.
3. **Reproducibility:** You can be certain your hyperparameters haven't been mutated mid-run.

```python
cfg = TrainingConfig()
cfg.name = "new-name"  # ❌ Raises ValidationError: "Instance is frozen"
```

---

## Creating Modified Copies

Since configs are immutable, you create new modified copies to change values.

### 1. `evolve(cfg, **kwargs)`
The standard way to create a variant. It uses keyword arguments and supports nested overrides using double-underscores.

```python
from canopee import evolve

# Simple override
fast = evolve(cfg, epochs=5)

# Nested override (optimizer.lr)
tuned = evolve(cfg, optimizer__lr=1e-4)
```

### 2. The `|` Operator
The bitwise OR operator provides a concise syntax for merging. It is the only operation that lives directly on the instance (because infix syntax requires it). It accepts `dict` objects (with dot-paths), `Patch` objects, or other `ConfigBase` instances.

```python
# Deep merge using dot-paths
new_cfg = cfg | {"optimizer.lr": 1e-4, "batch_size": 64}
```

### 3. The `Patch` Class
A `Patch` is a first-class object representing a set of overrides. It is more powerful than a plain dictionary because it can be composed, scoped, and reused.

```python
from canopee import Patch

# Create from dot-paths
p1 = Patch({"optimizer.lr": 1e-4})

# Create from keyword arguments (__ -> .)
p2 = Patch.from_kwargs(optimizer__beta=0.9, epochs=50)

# Compose patches (p2 wins on conflict)
full_patch = p1 & p2

# Apply to a config
cfg2 = cfg | full_patch
```

Patches excel at modular configuration logic:
```python
def get_debug_overrides():
    return Patch({"debug": True, "batch_size": 1})

# Apply a scoped patch (prefixes all keys with 'training.')
opt_patch = Patch({"lr": 1e-3}).scoped("training")

cfg = cfg | get_debug_overrides() | opt_patch
```

### 4. The `wrap()` Proxy (Fluent API)
If you prefer method-chaining (fluent) style, use `wrap()`. It provides a proxy that exposes the full Canopée API as methods without touching your config's namespace.

Every method on the proxy returns a *new* proxy wrapping the updated config, allowing for elegant, readable chains.

```python
from canopee import wrap

cfg2 = (
    wrap(cfg)
    .evolve(epochs=20)
    .apply({"optimizer.lr": 1e-4})
    .unwrap()
)
```

The proxy isn't just for evolution; it exposes the entire feature set:

```python
proxy = wrap(cfg)

# Querying
lr = proxy.select("optimizer.lr")
flat = proxy.to_flat()

# Comparing
diff = proxy.diff(other_cfg)

# Temporary context
with proxy.patched(debug=True) as p_tmp:
    run_val(p_tmp.unwrap())

# I/O
proxy.save("run.toml")
```

---

## Temporary Overrides (`patched`)

When you need a temporarily modified config (e.g., in a unit test or a specific sub-routine), use the `patched` context manager:

```python
from canopee import patched

with patched(cfg, debug=True) as tmp_cfg:
    run_diagnostic(tmp_cfg)

# `cfg` is unchanged here
```

---

## Computed Fields

Use Pydantic's `@computed_field` for derived values. Canopée's merge logic correctly handles these, ensuring they are recalculated when the base fields change.

```python
from pydantic import computed_field

class Training(ConfigBase):
    epochs: int = 20
    
    @computed_field
    @property
    def warmup_steps(self) -> int:
        return self.epochs * 10
```

---

## Equality and Fingerprinting

Configs are compared by value. Every config has a `fingerprint` property—a stable 16-character hex digest of its data.

```python
cfg1 = Training(epochs=10)
cfg2 = evolve(cfg1, epochs=10)

assert cfg1 == cfg2
assert cfg1.fingerprint == cfg2.fingerprint
```

The `fingerprint` is a `@computed_field` (data), not a method, so it behaves like any other field.
