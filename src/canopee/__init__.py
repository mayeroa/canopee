from importlib.metadata import version

from canopee import sweep as sweep
from canopee.cli import clify, ArgparseBackend, ClickBackend, TyperBackend
from canopee.core import ConfigBase, Patch, diff, evolve, to_flat, wrap
from canopee.errors import pretty_print_error
from canopee.sources import (
    CLISource,
    DictSource,
    EnvSource,
    FileSource,
    Source,
    merge_sources,
)
from canopee.serialization import load, save
from canopee.store import ConfigStore, store

__version__ = version("canopee")

__all__ = [
    "ConfigBase",
    "ConfigStore",
    "store",
    "Source",
    "EnvSource",
    "CLISource",
    "FileSource",
    "DictSource",
    "merge_sources",
    "sweep",
    "clify",
    "ArgparseBackend",
    "ClickBackend",
    "TyperBackend",
    "evolve",
    "diff",
    "to_flat",
    "save",
    "load",
    "pretty_print_error",
    "Patch",
    "wrap",
]
