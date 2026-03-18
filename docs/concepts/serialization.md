# Serialisation

Canopée provides a clean, pure-function API for converting your configurations between Python objects, plain dicts, strings, and files.

---

## 1. Dict Round-trip (Layer 1)

At the most basic level, you can convert any `ConfigBase` instance to a JSON-serialisable dictionary and back.

```python
from canopee import to_dict, from_dict

# Config → dict
data = to_dict(cfg)

# dict → Config (re-validated)
cfg2 = from_dict(TrainingConfig, data)
```

**Note:** `to_dict` excludes computed fields (like `fingerprint`) to ensure the dictionary only contains the data necessary to reconstruct the object.

---

## 2. String Round-trip (Layer 2)

Canopée supports JSON, YAML, and TOML formats out-of-the-box.

```python
from canopee import (
    to_json_str, from_json_str,
    to_yaml_str, from_yaml_str,
    to_toml_str, from_toml_str
)

# JSON
text = to_json_str(cfg)
cfg_json = from_json_str(TrainingConfig, text)

# YAML (requires: pip install pyyaml)
# Includes a helpful header with the class name and fingerprint.
text = to_yaml_str(cfg)
cfg_yaml = from_yaml_str(TrainingConfig, text)

# TOML (requires: pip install tomli-w)
text = to_toml_str(cfg)
cfg_toml = from_toml_str(TrainingConfig, text)
```

---

## 3. File I/O & Auto-dispatch (Layer 3)

The most common way to save and load configurations is using the auto-dispatching `save()` and `load()` functions, which infer the format from the file extension.

```python
from canopee import save, load

# Extension determines the format (.json, .yaml, .yml, .toml)
save(cfg, "experiment-1.toml")

# Load and re-validate
cfg2 = load(TrainingConfig, "experiment-1.toml")
```

---

## TOML and `None` Values

Standard TOML has no native "null" or "none" type. When saving to TOML, Canopée provides three ways to handle `None` values via the `none_handling` argument:

1. **`"drop"` (Default):** Silently omit the key from the TOML file.
2. **`"raise"`:** Raise a `ValueError` if a `None` is encountered. Useful for ensuring full configuration presence.
3. **`"null_str"`:** Write the value as the string `"null"`. Useful for debugging.

```python
# If cfg has lr=None
save(cfg, "config.toml", none_handling="raise")  # ❌ Raises ValueError
save(cfg, "config.toml", none_handling="drop")   # ✅ Success (lr is missing in file)
```

---

## Optional Dependencies

To keep the core library lightweight, some serialisation formats require optional dependencies:

*   **YAML:** `pip install pyyaml`
*   **TOML (Write):** `pip install tomli-w`
*   **TOML (Read):** Built-in via the standard library `tomllib` (Python 3.11+).

If you attempt to use a format without its dependency, Canopée will raise a helpful `ImportError` with the correct installation command.
