"""
canopee.sources
~~~~~~~~~~~~~~~~~~

Source loaders: build configs from environment variables, CLI args,
TOML files, JSON files, or any combination with a clear priority chain.

Priority (highest to lowest):
    CLI args > Environment variables > File > Defaults

Design
------
Each source is a pure function returning a ``dict[str, Any]``.  The
``from_sources`` class method on ``ConfigBase`` merges them in priority
order and validates the result through Pydantic's normal pipeline.

Example::

    cfg = TrainingConfig.from_sources(
        sources=[
            FileSource("config.toml"),
            EnvSource(prefix="TRAIN_"),
            CLISource(),            # reads sys.argv
        ]
    )

    # Or individual helpers:
    cfg = TrainingConfig.from_env(prefix="APP_")
    cfg = TrainingConfig.from_file("config.toml")
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------------------
# Source base
# ---------------------------------------------------------------------------


class Source(ABC):
    """Abstract config source.  Returns a flat-ish dict of raw values."""

    @abstractmethod
    def load(self) -> dict[str, Any]:
        """Load and return config values as a dictionary.

        Returns:
            dict[str, Any]: A dictionary containing the loaded configuration values.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Env source
# ---------------------------------------------------------------------------


class EnvSource(Source):
    """Load config values from environment variables.

    Variables are mapped to field names by stripping the prefix, lowercasing,
    and converting double-underscores to dots (for nested fields).

    Args:
        prefix (str, optional): The prefix to filter and strip from keys. Defaults to "".
        case_sensitive (bool, optional): Whether to treat environment variables as case-sensitive. Defaults to False.
    """

    def __init__(
        self,
        prefix: str = "",
        *,
        case_sensitive: bool = False,
    ) -> None:
        self.prefix = prefix
        self.case_sensitive = case_sensitive

    def load(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        prefix = self.prefix if self.case_sensitive else self.prefix.upper()

        for key, value in os.environ.items():
            k = key if self.case_sensitive else key.upper()
            if not k.startswith(prefix):
                continue
            field_key = k[len(prefix) :]
            if not self.case_sensitive:
                field_key = field_key.lower()
            # Double underscore → dot separator for nested fields
            field_key = field_key.replace("__", ".")
            result[field_key] = _coerce_env_value(value)

        return result

    def __repr__(self) -> str:
        return f"EnvSource(prefix={self.prefix!r})"


def _coerce_env_value(value: str) -> Any:
    """Try to coerce an env string to a Python scalar.

    Args:
        value (str): The environment variable value as a string.

    Returns:
        Any: The coerced Python scalar value.
    """
    # bool
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    # int
    try:
        return int(value)
    except ValueError:
        pass
    # float
    try:
        return float(value)
    except ValueError:
        pass
    # JSON array/object
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


# ---------------------------------------------------------------------------
# CLI source
# ---------------------------------------------------------------------------


class CLISource(Source):
    """Parse ``--key=value`` or ``--key value`` arguments from the command line.

    Nested fields use dot notation: ``--training.lr=3e-4``.

    Args:
        argv (list[str] | None, optional): Explicit list of command line arguments to parse. Defaults to None (uses sys.argv[1:]).
    """

    def __init__(self, argv: list[str] | None = None) -> None:
        # Default to sys.argv[1:]; allow injection for testing
        self.argv = argv if argv is not None else sys.argv[1:]

    def load(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        args = list(self.argv)
        i = 0
        while i < len(args):
            arg = args[i]
            if not arg.startswith("--"):
                i += 1
                continue
            arg = arg[2:]  # strip --
            if "=" in arg:
                key, _, raw_value = arg.partition("=")
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                key = arg
                raw_value = args[i + 1]
                i += 1
            else:
                # Boolean flag: --verbose → verbose=True
                key = arg
                raw_value = "true"
            result[key] = _coerce_env_value(raw_value)
            i += 1
        return result

    def __repr__(self) -> str:
        return f"CLISource(argv={self.argv!r})"


# ---------------------------------------------------------------------------
# File sources
# ---------------------------------------------------------------------------


class FileSource(Source):
    """Load config from a TOML, JSON, or YAML file.

    The file format is auto-detected from the extension.
    Nested dicts are preserved (merged with dot-path resolution later).

    Args:
        path (str | pathlib.Path): The path to the configuration file.
    """

    def __init__(self, path: str | pathlib.Path) -> None:
        self.path = pathlib.Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")

        suffix = self.path.suffix.lower()
        text = self.path.read_text(encoding="utf-8")

        if suffix == ".json":
            return json.loads(text)
        if suffix == ".toml":
            return _load_toml(text)
        if suffix in (".yaml", ".yml"):
            return _load_yaml_file(text)
        raise ValueError(f"Unsupported config file format: '{suffix}'. Use .json, .toml, or .yaml")

    def __repr__(self) -> str:
        return f"FileSource({str(self.path)!r})"


def _load_toml(text: str) -> dict[str, Any]:
    import tomllib

    return tomllib.loads(text)


def _load_yaml_file(text: str) -> dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError as exc:
        raise ImportError("YAML support requires PyYAML: pip install pyyaml") from exc


# ---------------------------------------------------------------------------
# DictSource (useful for testing / programmatic overrides)
# ---------------------------------------------------------------------------


class DictSource(Source):
    """Wraps a plain dict as a source. Useful in tests or scripts.

    Args:
        data (dict[str, Any]): The dictionary containing configuration data.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def load(self) -> dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        return f"DictSource({self._data!r})"


# ---------------------------------------------------------------------------
# Source merger
# ---------------------------------------------------------------------------


def merge_sources(sources: list[Source]) -> dict[str, Any]:
    """Merge multiple sources in order, later sources taking priority.

    Handles dot-path keys correctly for nested configs.

    Args:
        sources (list[Source]): A list of configuration sources to merge.

    Returns:
        dict[str, Any]: A single merged dictionary with all overrides applied.
    """
    from canopee.core import _apply_dotpath

    merged: dict[str, Any] = {}
    for source in sources:
        overrides = source.load()
        _apply_dotpath(merged, overrides)
    return merged
