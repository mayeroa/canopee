"""
canopee
~~~~~~~

A type-safe, Pydantic-native configuration library with elegant
composition, immutable configs, and a powerful Sweep engine.

Quick start::

    from canopee import ConfigBase
    from pydantic import Field, computed_field

    class TrainingConfig(ConfigBase):
        lr: float = Field(default=1e-3, ge=0.0, le=1.0)
        epochs: int = 10

        @computed_field
        @property
        def warmup_steps(self) -> int:
            return self.epochs * 100

    cfg = TrainingConfig()

    # Dict / dot-path override:
    fast = cfg | {"lr": 1e-2, "epochs": 3}

    # Typed keyword override:
    fast = cfg.evolve(lr=1e-2, epochs=3)

    # Save / load:
    save(cfg, "run.toml")
    cfg2 = load(TrainingConfig, "run.toml")

    # String serialization:
    text = dumps(cfg, "yaml")
    cfg3 = loads(TrainingConfig, "yaml", text)
"""

from canopee import sweep as sweep
from canopee.cli import clify, ArgparseBackend, ClickBackend, TyperBackend
from canopee.core import ConfigBase, diff, evolve, to_flat
from canopee.errors import pretty_print_error
from canopee.sources import (
    CLISource,
    DictSource,
    EnvSource,
    FileSource,
    Source,
    merge_sources,
)
from canopee.serialization import dumps, load, loads, save
from canopee.store import ConfigStore, global_store

__all__ = [
    "ConfigBase",
    "ConfigStore",
    "global_store",
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
    "dumps",
    "loads",
    "pretty_print_error",
]

__version__ = "0.1.0"
