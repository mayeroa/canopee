"""
canopee.serialization
~~~~~~~~~~~~~~~~~~~~~

Pure-function serialisation for :class:`~canopee.core.ConfigBase` instances.

Every public name is a free function — no classes, no state.  The module is
organised in three layers:

**Layer 1 — dict round-trip** (format-agnostic, always available)::

    data = to_dict(cfg)                # ConfigBase  → plain dict
    cfg  = from_dict(MyConfig, data)   # plain dict  → ConfigBase

**Layer 2 — string round-trip** (one pair per format)::

    text = to_json_str(cfg)
    text = to_yaml_str(cfg)            # requires: pip install pyyaml
    text = to_toml_str(cfg)            # requires: pip install tomli-w

    cfg  = from_json_str(MyConfig, text)
    cfg  = from_yaml_str(MyConfig, text)
    cfg  = from_toml_str(MyConfig, text)

**Layer 3 — file I/O** (one pair per format + auto-dispatch)::

    save_json(cfg, "run.json")
    save_yaml(cfg, "run.yaml")
    save_toml(cfg, "run.toml")

    cfg = load_json(MyConfig, "run.json")
    cfg = load_yaml(MyConfig, "run.yaml")
    cfg = load_toml(MyConfig, "run.toml")

    # Or let the extension decide:
    save(cfg, "run.toml")
    cfg = load(MyConfig, "run.yaml")

TOML and ``None``
-----------------
TOML has no null type.  By default, ``None`` values are **silently dropped**
when serialising to TOML.  Pass ``none_handling="raise"`` to get an error
instead, or ``none_handling="null_str"`` to serialise them as the string
``"null"`` for debugging::

    to_toml_str(cfg, none_handling="raise")

Optional dependencies
---------------------
- YAML support: ``pip install pyyaml``
- TOML write:   ``pip install tomli-w``
  (TOML read uses the stdlib ``tomllib`` module, Python >= 3.11.)
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Literal, TypeVar

from canopee.core import ConfigBase, _dump

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=ConfigBase)

PathLike = str | Path

#: How ``None`` values are handled when serialising to TOML, which has no
#: native null type.
#:
#: ``"drop"``     — silently omit the key (default).
#: ``"raise"``    — raise :exc:`ValueError` if any ``None`` is encountered.
#: ``"null_str"`` — serialise ``None`` as the string ``"null"``.
NoneHandling = Literal["drop", "raise", "null_str"]

#: Supported file extensions for auto-dispatch in :func:`save` / :func:`load`.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".json", ".yaml", ".yml", ".toml")

#: Supported format names (informational).
SUPPORTED_FORMATS: tuple[str, ...] = ("json", "yaml", "toml")


# ---------------------------------------------------------------------------
# Layer 1 — dict round-trip
# ---------------------------------------------------------------------------


def to_dict(config: ConfigBase) -> dict[str, Any]:
    """Convert *config* to a plain, JSON-serialisable dict.

    Computed fields (e.g. ``fingerprint``) are excluded — the result contains
    only the regular fields needed to reconstruct the config via
    :func:`from_dict`.  Nested ``ConfigBase`` sub-models are recursively
    converted to plain dicts.

    Args:
        config: Any :class:`~canopee.core.ConfigBase` instance.

    Returns:
        A plain ``dict[str, Any]`` with no Pydantic objects at any level.

    Example::

        data = to_dict(cfg)
        # {"epochs": 10, "optimizer": {"lr": 0.001, "beta": 0.9}, ...}
    """
    return _dump(config)


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Reconstruct a *cls* instance from a plain dict.

    Validates the data through Pydantic — field types are coerced and
    unknown keys are rejected (``extra="forbid"`` from
    :class:`~canopee.core.ConfigBase`).

    Args:
        cls:  The :class:`~canopee.core.ConfigBase` subclass to instantiate.
        data: A plain dict, typically produced by :func:`to_dict` or loaded
              from a file.

    Returns:
        A validated, frozen instance of *cls*.

    Raises:
        pydantic.ValidationError: If *data* fails validation against *cls*.

    Example::

        cfg = from_dict(TrainingConfig, {"epochs": 20, "optimizer": {"lr": 3e-4}})
    """
    return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _DropType:
    """Singleton sentinel used by :func:`_sanitize_toml` to mark values for omission."""

    _instance: _DropType | None = None

    def __new__(cls) -> _DropType:
        """Return the single shared instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """Return ``<DROP>``."""
        return "<DROP>"


_DROP = _DropType()


def _require_yaml() -> Any:
    """Import and return PyYAML, raising an :exc:`ImportError` with install hint if absent."""
    try:
        import yaml

        return yaml
    except ImportError as exc:
        raise ImportError("YAML support requires PyYAML.  Install it with:  pip install pyyaml") from exc


def _require_tomli_w() -> Any:
    """Import and return tomli-w, raising an :exc:`ImportError` with install hint if absent."""
    try:
        import tomli_w

        return tomli_w
    except ImportError as exc:
        raise ImportError("Writing TOML requires tomli-w.  Install it with:  pip install tomli-w") from exc


def _sanitize_toml(
    obj: Any,
    *,
    none_handling: NoneHandling,
    path: str = "",
) -> Any:
    """Recursively prepare *obj* for TOML serialisation.

    TOML has no null type, so ``None`` values are handled according to
    *none_handling*:

    - ``"drop"``     — return :data:`_DROP` so the caller omits the key.
    - ``"raise"``    — raise :exc:`ValueError` immediately.
    - ``"null_str"`` — replace with the string ``"null"``.

    Args:
        obj:           The value to sanitise.
        none_handling: Strategy for ``None`` values.
        path:          Dot-path of *obj* in the original structure (for error messages).

    Returns:
        A TOML-safe value, or :data:`_DROP` if the caller should omit this key.

    Raises:
        ValueError: If *none_handling* is ``"raise"`` and a ``None`` is found.
    """
    if obj is None:
        if none_handling == "raise":
            location = f" at {path!r}" if path else ""
            raise ValueError(
                f"None value{location} cannot be serialised to TOML.  "
                "Use none_handling='drop' to omit it silently, or "
                "none_handling='null_str' to keep it as a string."
            )
        return "null" if none_handling == "null_str" else _DROP

    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for k, v in obj.items():
            child_path = f"{path}.{k}" if path else str(k)
            sanitised = _sanitize_toml(v, none_handling=none_handling, path=child_path)
            if sanitised is not _DROP:
                result[str(k)] = sanitised
        return result

    if isinstance(obj, (list, tuple)):
        child_items = []
        for i, v in enumerate(obj):
            child_path = f"{path}.{i}" if path else str(i)
            sanitised = _sanitize_toml(v, none_handling=none_handling, path=child_path)
            if sanitised is not _DROP:
                child_items.append(sanitised)
        return child_items

    return obj


def _to_path(p: PathLike) -> Path:
    """Coerce *p* to a :class:`~pathlib.Path`."""
    return p if isinstance(p, Path) else Path(p)


# ---------------------------------------------------------------------------
# Layer 2 — string round-trip
# ---------------------------------------------------------------------------


def to_json_str(config: ConfigBase, *, indent: int = 2) -> str:
    """Serialise *config* to a JSON string.

    Args:
        config: The config to serialise.
        indent: Number of spaces for indentation. Defaults to ``2``.

    Returns:
        A pretty-printed, UTF-8–safe JSON string.

    Example::

        text = to_json_str(cfg)
        text = to_json_str(cfg, indent=4)
    """
    return json.dumps(to_dict(config), indent=indent, ensure_ascii=False)


def from_json_str(cls: type[T], text: str) -> T:
    """Deserialise a JSON string into an instance of *cls*.

    Args:
        cls:  The :class:`~canopee.core.ConfigBase` subclass to instantiate.
        text: A JSON string produced by :func:`to_json_str`.

    Returns:
        A validated, frozen instance of *cls*.

    Raises:
        pydantic.ValidationError: If the data fails validation.
        json.JSONDecodeError:     If *text* is not valid JSON.
    """
    return cls.model_validate_json(text)


def to_yaml_str(config: ConfigBase) -> str:
    """Serialise *config* to a YAML string.

    A metadata header is prepended as YAML comments so the class and
    fingerprint are visible when the file is inspected::

        # canopee config
        # class: TrainingConfig
        # fingerprint: a3f9c12e4b6d8071

    The header is harmless to the YAML parser and is ignored by
    :func:`from_yaml_str`.

    Requires ``pyyaml`` (``pip install pyyaml``).

    Args:
        config: The config to serialise.

    Returns:
        A YAML string with a Canopée metadata header.

    Raises:
        ImportError: If PyYAML is not installed.
    """
    yaml = _require_yaml()
    header = f"# canopee config\n# class: {type(config).__qualname__}\n# fingerprint: {config.fingerprint}\n"
    body = yaml.dump(
        to_dict(config),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    return header + body


def from_yaml_str(cls: type[T], text: str) -> T:
    """Deserialise a YAML string into an instance of *cls*.

    The Canopée metadata header (comment lines) is safely ignored by the
    YAML parser.

    Requires ``pyyaml`` (``pip install pyyaml``).

    Args:
        cls:  The :class:`~canopee.core.ConfigBase` subclass to instantiate.
        text: A YAML string produced by :func:`to_yaml_str`.

    Returns:
        A validated, frozen instance of *cls*.

    Raises:
        ImportError:              If PyYAML is not installed.
        pydantic.ValidationError: If the data fails validation.
    """
    return from_dict(cls, _require_yaml().safe_load(text))


def to_toml_str(
    config: ConfigBase,
    *,
    none_handling: NoneHandling = "drop",
) -> str:
    """Serialise *config* to a TOML string.

    TOML has no null type, so ``None`` values are handled according to
    *none_handling* (default: silently drop the key).

    Requires ``tomli-w`` (``pip install tomli-w``).

    Args:
        config:        The config to serialise.
        none_handling: Strategy for ``None`` values:

                       - ``"drop"`` *(default)* — omit the key silently.
                       - ``"raise"``             — raise :exc:`ValueError`.
                       - ``"null_str"``          — write ``"null"`` (string).

    Returns:
        A TOML string.

    Raises:
        ImportError:  If tomli-w is not installed.
        ValueError:   If *none_handling* is ``"raise"`` and a ``None`` is found.

    Example::

        text = to_toml_str(cfg)
        text = to_toml_str(cfg, none_handling="raise")
    """
    sanitised = _sanitize_toml(to_dict(config), none_handling=none_handling)
    return _require_tomli_w().dumps(sanitised)


def from_toml_str(cls: type[T], text: str) -> T:
    """Deserialise a TOML string into an instance of *cls*.

    Uses the stdlib ``tomllib`` module — no extra dependency required
    (Python >= 3.11).

    Args:
        cls:  The :class:`~canopee.core.ConfigBase` subclass to instantiate.
        text: A TOML string produced by :func:`to_toml_str`.

    Returns:
        A validated, frozen instance of *cls*.

    Raises:
        pydantic.ValidationError: If the data fails validation.
        tomllib.TOMLDecodeError:  If *text* is not valid TOML.
    """
    return from_dict(cls, tomllib.loads(text))


# ---------------------------------------------------------------------------
# Layer 3 — file I/O
# ---------------------------------------------------------------------------


def save_json(config: ConfigBase, path: PathLike, *, indent: int = 2) -> Path:
    """Write *config* as JSON to *path*, creating parent directories as needed.

    Args:
        config: The config to write.
        path:   Destination file path.
        indent: JSON indentation width. Defaults to ``2``.

    Returns:
        The resolved :class:`~pathlib.Path` that was written.
    """
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_json_str(config, indent=indent), encoding="utf-8")
    return p


def load_json(cls: type[T], path: PathLike) -> T:
    """Load a *cls* instance from the JSON file at *path*.

    Args:
        cls:  The :class:`~canopee.core.ConfigBase` subclass to instantiate.
        path: Path to a JSON file written by :func:`save_json`.

    Returns:
        A validated, frozen instance of *cls*.
    """
    return from_json_str(cls, _to_path(path).read_text(encoding="utf-8"))


def save_yaml(config: ConfigBase, path: PathLike) -> Path:
    """Write *config* as YAML to *path*, creating parent directories as needed.

    Requires ``pyyaml`` (``pip install pyyaml``).

    Args:
        config: The config to write.
        path:   Destination file path.

    Returns:
        The resolved :class:`~pathlib.Path` that was written.

    Raises:
        ImportError: If PyYAML is not installed.
    """
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_yaml_str(config), encoding="utf-8")
    return p


def load_yaml(cls: type[T], path: PathLike) -> T:
    """Load a *cls* instance from the YAML file at *path*.

    Requires ``pyyaml`` (``pip install pyyaml``).

    Args:
        cls:  The :class:`~canopee.core.ConfigBase` subclass to instantiate.
        path: Path to a YAML file written by :func:`save_yaml`.

    Returns:
        A validated, frozen instance of *cls*.

    Raises:
        ImportError: If PyYAML is not installed.
    """
    return from_yaml_str(cls, _to_path(path).read_text(encoding="utf-8"))


def save_toml(
    config: ConfigBase,
    path: PathLike,
    *,
    none_handling: NoneHandling = "drop",
) -> Path:
    """Write *config* as TOML to *path*, creating parent directories as needed.

    Requires ``tomli-w`` (``pip install tomli-w``).

    Args:
        config:        The config to write.
        path:          Destination file path.
        none_handling: How to handle ``None`` values. Defaults to ``"drop"``.

    Returns:
        The resolved :class:`~pathlib.Path` that was written.

    Raises:
        ImportError: If tomli-w is not installed.
        ValueError:  If *none_handling* is ``"raise"`` and a ``None`` is found.
    """
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_toml_str(config, none_handling=none_handling), encoding="utf-8")
    return p


def load_toml(cls: type[T], path: PathLike) -> T:
    """Load a *cls* instance from the TOML file at *path*.

    Args:
        cls:  The :class:`~canopee.core.ConfigBase` subclass to instantiate.
        path: Path to a TOML file written by :func:`save_toml`.

    Returns:
        A validated, frozen instance of *cls*.
    """
    return from_toml_str(cls, _to_path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Auto-dispatch — extension-based save / load
# ---------------------------------------------------------------------------

_SAVERS: dict[str, Any] = {
    ".json": save_json,
    ".yaml": save_yaml,
    ".yml": save_yaml,
    ".toml": save_toml,
}

_LOADERS: dict[str, Any] = {
    ".json": load_json,
    ".yaml": load_yaml,
    ".yml": load_yaml,
    ".toml": load_toml,
}


def save(config: ConfigBase, path: PathLike, **kwargs: Any) -> Path:
    """Save *config* to *path*, inferring the format from the file extension.

    Delegates to :func:`save_json`, :func:`save_yaml`, or :func:`save_toml`
    based on the extension.  Extra keyword arguments are forwarded to the
    format-specific function unchanged.

    Args:
        config:   The config to save.
        path:     Destination file path (extension determines format).
        **kwargs: Forwarded to the format saver — e.g. ``indent=4`` for JSON,
                  ``none_handling="raise"`` for TOML.

    Returns:
        The resolved :class:`~pathlib.Path` that was written.

    Raises:
        ValueError: If the extension is not in :data:`SUPPORTED_EXTENSIONS`.

    Example::

        save(cfg, "run.toml")
        save(cfg, "run.json", indent=4)
        save(cfg, "run.toml", none_handling="raise")
    """
    ext = _to_path(path).suffix.lower()
    saver = _SAVERS.get(ext)
    if saver is None:
        raise ValueError(f"Unsupported extension {ext!r}. Supported: {SUPPORTED_EXTENSIONS}")
    return saver(config, path, **kwargs)


def load(cls: type[T], path: PathLike) -> T:
    """Load a *cls* instance from *path*, inferring the format from the extension.

    Delegates to :func:`load_json`, :func:`load_yaml`, or :func:`load_toml`
    based on the extension.

    Args:
        cls:  The :class:`~canopee.core.ConfigBase` subclass to instantiate.
        path: Path to a config file (extension determines format).

    Returns:
        A validated, frozen instance of *cls*.

    Raises:
        ValueError: If the extension is not in :data:`SUPPORTED_EXTENSIONS`.

    Example::

        cfg = load(TrainingConfig, "run.toml")
        cfg = load(TrainingConfig, "run.yaml")
    """
    ext = _to_path(path).suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported extension {ext!r}. Supported: {SUPPORTED_EXTENSIONS}")
    return loader(cls, path)
