"""
canopee.serialization
~~~~~~~~~~~~~~~~~~~~~

Round-trip serialization for :class:`~canopee.core.ConfigBase` instances.

Supported formats: JSON, TOML, YAML.

Primary API (used by ConfigBase.save/load/dumps/loads)::

    save(config, "run.toml")
    load(MyConfig, "run.toml")
    dumps(config, "yaml")
    loads(MyConfig, "toml", text)

Format-specific functions are available for power users::

    from canopee.serialization import to_json, from_toml, dumps_yaml
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypeVar

from canopee.core import ConfigBase

_T = TypeVar("_T", bound=ConfigBase)

PathLike = str | Path
_Format = Literal["json", "toml", "yaml"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_path(p: PathLike) -> Path:
    if isinstance(p, Path):
        return p
    return Path(p)


def _load_yaml():
    try:
        import yaml

        return yaml
    except ImportError as exc:
        raise ImportError("YAML support requires PyYAML: pip install pyyaml") from exc


def _load_toml_writer():
    try:
        import tomli_w

        return tomli_w
    except ImportError as exc:
        raise ImportError("Writing TOML requires tomli-w: pip install tomli-w") from exc


def _load_toml_reader():
    import tomllib

    return tomllib


class _Omit:
    """Sentinel for values that must be omitted from TOML output (e.g. None)."""


_OMIT = _Omit()


def _sanitize_for_toml(obj: Any) -> Any:
    if obj is None:
        return _OMIT
    if isinstance(obj, dict):
        return {str(k): v2 for k, v in obj.items() if (v2 := _sanitize_for_toml(v)) is not _OMIT}
    if isinstance(obj, (list, tuple)):
        return [item for item in (_sanitize_for_toml(x) for x in obj) if item is not _OMIT]
    return obj


def _dump_data(config: ConfigBase, *, include_computed: bool = False) -> dict[str, Any]:
    if not include_computed:
        return config._dump_for_validation()
    return config.model_dump()


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def dumps_json(config: ConfigBase, *, indent: int = 2, include_computed: bool = False) -> str:
    """Serialise a configuration to a JSON string.

    Args:
        config (ConfigBase): The configuration to serialise.
        indent (int, optional): Indentation level for JSON. Defaults to 2.
        include_computed (bool, optional): Whether to include computed fields. Defaults to False.

    Returns:
        str: The JSON representation of the configuration.
    """
    return json.dumps(_dump_data(config, include_computed=include_computed), indent=indent, ensure_ascii=False)


def loads_json(cls: type[_T], text: str) -> _T:
    """Deserialise a JSON string into a configuration instance.

    Args:
        cls (type[_T]): The configuration class to instantiate.
        text (str): The JSON string to deserialise.

    Returns:
        _T: The deserialised configuration instance.
    """
    return cls.model_validate_json(text)


def to_json(config: ConfigBase, path: PathLike, *, indent: int = 2, include_computed: bool = False) -> Path:
    """Write a configuration to a JSON file.

    Args:
        config (ConfigBase): The configuration to write.
        path (PathLike): The destination file path.
        indent (int, optional): Indentation level for JSON. Defaults to 2.
        include_computed (bool, optional): Whether to include computed fields. Defaults to False.

    Returns:
        Path: The path where the file was written.
    """
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps_json(config, indent=indent, include_computed=include_computed), encoding="utf-8")
    return p


def from_json(cls: type[_T], path: PathLike) -> _T:
    """Load a configuration instance from a JSON file.

    Args:
        cls (type[_T]): The configuration class to instantiate.
        path (PathLike): The file path to load from.

    Returns:
        _T: The loaded configuration instance.
    """
    return loads_json(cls, _to_path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


def dumps_yaml(config: ConfigBase, *, include_computed: bool = False) -> str:
    """Serialise a configuration to a YAML string.

    Args:
        config (ConfigBase): The configuration to serialise.
        include_computed (bool, optional): Whether to include computed fields. Defaults to False.

    Returns:
        str: The YAML representation of the configuration.
    """
    yaml = _load_yaml()
    header = f"# canopee config\n# class: {type(config).__qualname__}\n# fingerprint: {config.fingerprint}\n"
    body = yaml.dump(
        _dump_data(config, include_computed=include_computed),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    return header + body


def loads_yaml(cls: type[_T], text: str) -> _T:
    """Deserialise a YAML string into a configuration instance.

    Args:
        cls (type[_T]): The configuration class to instantiate.
        text (str): The YAML string to deserialise.

    Returns:
        _T: The deserialised configuration instance.
    """
    return cls.model_validate(_load_yaml().safe_load(text))


def to_yaml(config: ConfigBase, path: PathLike, *, include_computed: bool = False) -> Path:
    """Write a configuration to a YAML file.

    Args:
        config (ConfigBase): The configuration to write.
        path (PathLike): The destination file path.
        include_computed (bool, optional): Whether to include computed fields. Defaults to False.

    Returns:
        Path: The path where the file was written.
    """
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps_yaml(config, include_computed=include_computed), encoding="utf-8")
    return p


def from_yaml(cls: type[_T], path: PathLike) -> _T:
    """Load a configuration instance from a YAML file.

    Args:
        cls (type[_T]): The configuration class to instantiate.
        path (PathLike): The file path to load from.

    Returns:
        _T: The loaded configuration instance.
    """
    return loads_yaml(cls, _to_path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# TOML
# ---------------------------------------------------------------------------


def dumps_toml(config: ConfigBase, *, include_computed: bool = False) -> str:
    """Serialise a configuration to a TOML string.

    Args:
        config (ConfigBase): The configuration to serialise.
        include_computed (bool, optional): Whether to include computed fields. Defaults to False.

    Returns:
        str: The TOML representation of the configuration.
    """
    return _load_toml_writer().dumps(_sanitize_for_toml(_dump_data(config, include_computed=include_computed)))


def loads_toml(cls: type[_T], text: str) -> _T:
    """Deserialise a TOML string into a configuration instance.

    Args:
        cls (type[_T]): The configuration class to instantiate.
        text (str): The TOML string to deserialise.

    Returns:
        _T: The deserialised configuration instance.
    """
    return cls.model_validate(_load_toml_reader().loads(text))


def to_toml(config: ConfigBase, path: PathLike, *, include_computed: bool = False) -> Path:
    """Write a configuration to a TOML file.

    Args:
        config (ConfigBase): The configuration to write.
        path (PathLike): The destination file path.
        include_computed (bool, optional): Whether to include computed fields. Defaults to False.

    Returns:
        Path: The path where the file was written.
    """
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps_toml(config, include_computed=include_computed), encoding="utf-8")
    return p


def from_toml(cls: type[_T], path: PathLike) -> _T:
    """Load a configuration instance from a TOML file.

    Args:
        cls (type[_T]): The configuration class to instantiate.
        path (PathLike): The file path to load from.

    Returns:
        _T: The loaded configuration instance.
    """
    return loads_toml(cls, _to_path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Dispatchers — the primary public API
# ---------------------------------------------------------------------------

_SAVERS = {
    ".json": to_json,
    ".toml": to_toml,
    ".yaml": to_yaml,
    ".yml": to_yaml,
}

_LOADERS = {
    ".json": from_json,
    ".toml": from_toml,
    ".yaml": from_yaml,
    ".yml": from_yaml,
}

_DUMPS = {
    "json": dumps_json,
    "toml": dumps_toml,
    "yaml": dumps_yaml,
}

_LOADS = {
    "json": loads_json,
    "toml": loads_toml,
    "yaml": loads_yaml,
}


def save(config: ConfigBase, path: PathLike, **kwargs: Any) -> Path:
    """Save a configuration to a path, auto-detecting format from extension.

    Args:
        config (ConfigBase): The configuration to save.
        path (PathLike): The destination file path.
        **kwargs (Any): Additional keyword arguments passed to the specific format writer.

    Returns:
        Path: The path where the file was saved.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = _to_path(path).suffix.lower()
    writer = _SAVERS.get(ext)
    if writer is None:
        raise ValueError(f"Unsupported extension '{ext}'. Choose from: {list(_SAVERS)}")
    return writer(config, path, **kwargs)


def load(cls: type[_T], path: PathLike) -> _T:
    """Load a configuration instance from a path, auto-detecting format.

    Args:
        cls (type[_T]): The configuration class to instantiate.
        path (PathLike): The file path to load from.

    Returns:
        _T: The loaded configuration instance.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = _to_path(path).suffix.lower()
    reader = _LOADERS.get(ext)
    if reader is None:
        raise ValueError(f"Unsupported extension '{ext}'. Choose from: {list(_LOADERS)}")
    return reader(cls, path)


def dumps(config: ConfigBase, fmt: _Format = "json", **kwargs: Any) -> str:
    """Serialize a configuration to a string in the given format.

    Args:
        config (ConfigBase): The configuration to serialize.
        fmt (_Format, optional): The format to use ('json', 'toml', or 'yaml'). Defaults to "json".
        **kwargs (Any): Additional keyword arguments passed to the specific format serializer.

    Returns:
        str: The serialized configuration string.

    Raises:
        ValueError: If the format is not supported.
    """
    serializer = _DUMPS.get(fmt)
    if serializer is None:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from: {list(_DUMPS)}")
    return serializer(config, **kwargs)


def loads(cls: type[_T], fmt: _Format, text: str) -> _T:
    """Deserialize a configuration instance from a string in the given format.

    Args:
        cls (type[_T]): The configuration class to instantiate.
        fmt (_Format): The format used ('json', 'toml', or 'yaml').
        text (str): The serialized configuration string.

    Returns:
        _T: The deserialized configuration instance.

    Raises:
        ValueError: If the format is not supported.
    """
    deserializer = _LOADS.get(fmt)
    if deserializer is None:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from: {list(_LOADS)}")
    return deserializer(cls, text)
