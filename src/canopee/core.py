"""
canopee.core
~~~~~~~~~~~~

The heart of the library: ``ConfigBase`` — a frozen, immutable Pydantic v2
model with ergonomic config operations.

Design principles
-----------------
- ConfigBase is a frozen Pydantic v2 model → instances are immutable,
  safe to cache, hash, and share across threads.
- Computed / derived fields use Pydantic's native ``@computed_field``.
- Two merge mechanisms:
    cfg | {"training.lr": 3e-4}    dict with dot-path keys
    cfg.evolve(lr=3e-4)            typed keyword arguments
  Both return a new validated instance.
- save() / load() / dumps() / loads() live directly on the class.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, computed_field


T = TypeVar("T", bound="ConfigBase")


# ---------------------------------------------------------------------------
# ConfigBase
# ---------------------------------------------------------------------------


class ConfigBase(BaseModel):
    """
    Base class for all Canopee configs.

    Key behaviours
    --------------
    * **Immutable** — ``model_config = ConfigDict(frozen=True)``.
    * **Merge** — two complementary styles:

        ``cfg | {"training.lr": 3e-4}``
            Dict with dot-path keys for nested overrides.

        ``cfg.evolve(lr=3e-4)``
            Typed keywords — IDE-friendly, no dot-paths.

    * **Serialization** — ``cfg.save("run.toml")`` / ``MyConfig.load("run.toml")``
      and ``cfg.dumps("yaml")`` / ``MyConfig.loads("yaml", text)``.
    * **Fingerprint** — ``cfg.fingerprint`` is a stable SHA-256 hex digest.
    * **Computed fields** — use Pydantic's ``@computed_field`` directly.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # ------------------------------------------------------------------
    # Internal: Dump without computed fields for re-validation
    # ------------------------------------------------------------------

    def _dump_for_validation(self) -> dict[str, Any]:
        """Recursively dump the config to a dict, strictly excluding all
        computed fields so it can be safely passed to ``model_validate``
        even with ``extra='forbid'``.

        Returns:
            dict[str, Any]: A dictionary representation of the configuration without computed fields.
        """
        return self.model_dump(
            mode="json",
            round_trip=True,
            exclude=set(type(self).model_computed_fields.keys()) if type(self).model_computed_fields else None,
        )

    # ------------------------------------------------------------------
    # Merge / evolve
    # ------------------------------------------------------------------

    def __or__(self: T, overrides: dict[str, Any] | ConfigBase) -> T:
        """Return a new config with overrides applied.

        Supports dot-path keys for nested configs.

        Args:
            overrides (dict[str, Any] | ConfigBase): A dictionary of overrides or another ConfigBase to apply.

        Returns:
            T: A new validated configuration instance with the overrides applied.
        """
        current = self._dump_for_validation()
        if isinstance(overrides, ConfigBase):
            overrides_dict = overrides._dump_for_validation()
        else:
            overrides_dict = overrides
        _apply_dotpath(current, overrides_dict)
        return self.__class__.model_validate(current)

    def __ror__(self: T, overrides: dict[str, Any] | ConfigBase) -> T:
        """Support ``dict | cfg`` (same semantics as ``cfg | dict``).

        Args:
            overrides (dict[str, Any] | ConfigBase): A dictionary of overrides or another ConfigBase to apply.

        Returns:
            T: A new validated configuration instance with the overrides applied.
        """
        return self.__or__(overrides)

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self._dump_for_validation().items())
        return f"{self.__class__.__name__}({fields})"

    @computed_field
    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of the configuration.

        Returns:
            str: The hexadecimal digest of the fingerprint.
        """
        data = json.dumps(self._dump_for_validation(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Serialization methods (delegates to canopee.serialization)
    # ------------------------------------------------------------------

    def save(self, path: str | Any, **kwargs: Any) -> Any:
        """Save this configuration to a file. Format is auto-detected from the extension."""
        from canopee.serialization import save as _save

        return _save(self, path, **kwargs)

    @classmethod
    def load(cls: type[T], path: str | Any) -> T:
        """Load a configuration instance from a file. Format is auto-detected."""
        from canopee.serialization import load as _load

        return _load(cls, path)

    def dumps(self, fmt: str = "json", **kwargs: Any) -> str:
        """Serialize this configuration to a string in the given format ('json', 'yaml', 'toml')."""
        from canopee.serialization import dumps as _dumps

        return _dumps(self, fmt=fmt, **kwargs)

    @classmethod
    def loads(cls: type[T], fmt: str, text: str) -> T:
        """Deserialize a configuration instance from a string."""
        from canopee.serialization import loads as _loads

        return _loads(cls, fmt=fmt, text=text)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def evolve(config: T, **kwargs: Any) -> T:
    """Return a new instance with the given top-level or nested fields updated.

    Double-underscores (`__`) in keyword arguments are converted to dot-paths
    for deeply nested overrides.

    Args:
        config (T): The configuration instance to evolve.
        **kwargs (Any): The fields to update and their new values.

    Returns:
        T: A new validated configuration instance with the updates applied.
    """
    # Convert double-underscore keys to dot-paths, preserving top-level keys
    unflattened: dict[str, Any] = {}
    for k, v in kwargs.items():
        if "__" in k:
            unflattened[k.replace("__", ".")] = v
        else:
            unflattened[k] = v

    if not unflattened:
        return config

    base_dict = config._dump_for_validation()
    _apply_dotpath(base_dict, unflattened)
    return type(config).model_validate(base_dict)


def diff(a: ConfigBase, b: ConfigBase) -> dict[str, tuple[Any, Any]]:
    """Return a dictionary of differing fields between two configurations.

    Performs a deep comparison by flattening both configurations and comparing
    leaf nodes.

    Args:
        a (ConfigBase): The first configuration instance.
        b (ConfigBase): The second configuration instance to compare against.

    Returns:
        dict[str, tuple[Any, Any]]: A mapping of field dot-paths to a tuple of (a_value, b_value) for differing fields.
    """
    a_flat = to_flat(a)
    b_flat = to_flat(b)

    return {
        key: (a_flat.get(key), b_flat.get(key))
        for key in set(a_flat) | set(b_flat)
        if a_flat.get(key) != b_flat.get(key)
    }


def to_flat(config: ConfigBase) -> dict[str, Any]:
    """Return a flat dot-path dict of all leaf values.

    Args:
        config (ConfigBase): The configuration instance to flatten.

    Returns:
        dict[str, Any]: A flattened dictionary where keys are dot-paths.
    """
    return dict(_flatten(config.model_dump()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_dotpath(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Mutate the target dictionary in-place, supporting dot-path keys.

    Args:
        target (dict[str, Any]): The target dictionary to mutate.
        overrides (dict[str, Any]): The dictionary of overrides, potentially containing dot-paths.
    """
    for key, value in overrides.items():
        parts = key.split(".")
        d = target
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value


def _flatten(
    d: dict[str, Any],
    prefix: str = "",
) -> Iterator[tuple[str, Any]]:
    """Recursively yield (dot_path, value) pairs for all leaf nodes.

    Args:
        d (dict[str, Any]): The dictionary to flatten.
        prefix (str, optional): The current dot-path prefix. Defaults to "".

    Yields:
        Iterator[tuple[str, Any]]: Tuples of the accumulated dot-path and the leaf value.
    """
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten(value, full_key)
        else:
            yield full_key, value
