"""
canopee.store
~~~~~~~~~~~~~

``ConfigStore`` — a global, dict-like registry for named config instances.

Usage::

    from canopee import ConfigStore

    # dict-style access
    ConfigStore["base"] = TrainCfg(lr=1e-3, epochs=10)
    ConfigStore["fast"] = TrainCfg(lr=1e-2, epochs=3)
    cfg = ConfigStore["base"]

    # typed retrieval
    cfg = ConfigStore.get("base", TrainCfg)

    # decorator-based registration
    @ConfigStore.entry("default")
    class TrainCfg(ConfigBase):
        lr: float = 1e-3

    # introspection
    "base" in ConfigStore
    list(ConfigStore)
    len(ConfigStore)
"""

from __future__ import annotations

import threading
from typing import Any, TypeVar, overload

from canopee.core import ConfigBase

T = TypeVar("T", bound=ConfigBase)


class _ConfigStoreEntry:
    __slots__ = ("name", "config", "parent_name")

    def __init__(
        self,
        name: str,
        config: ConfigBase,
        parent_name: str | None,
    ) -> None:
        self.name = name
        self.config = config
        self.parent_name = parent_name


class ConfigStore:
    """
    Instantiable config registry. For simple scripts, use the default `global_store`.
    For complex applications or tests, instantiate your own `ConfigStore`
    to prevent cross-pollution.
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._entries: dict[str, _ConfigStoreEntry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Dict-style access
    # ------------------------------------------------------------------

    def __setitem__(self, name: str, config: ConfigBase) -> None:
        """Register a configuration under a specified name (overwrites silently).

        Args:
            name (str): The name to register the configuration under.
            config (ConfigBase): The configuration instance to register.
        """
        with self._lock:
            self._entries[name] = _ConfigStoreEntry(name, config, parent_name=None)

    def __getitem__(self, key: str) -> ConfigBase:
        """Retrieve a configuration by its registered name.

        Args:
            key (str): The name of the configuration.

        Returns:
            ConfigBase: The retrieved configuration instance.
        """
        return self.get(key)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        config: ConfigBase,
        *,
        parent: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Register a configuration under a specified name.

        Args:
            name (str): Unique identifier for this config variant.
            config (ConfigBase): The config instance to store.
            parent (str | None, optional): Name of a previously registered config to use as base. Defaults to None.
            overwrite (bool, optional): If False, raise on duplicate names. Defaults to False.

        Raises:
            KeyError: If the name is already registered (and overwrite is False), or if the parent is not found.
        """
        with self._lock:
            if name in self._entries and not overwrite:
                raise KeyError(f"Config '{name}' is already registered. Use overwrite=True to replace it.")

            resolved = config
            if parent is not None:
                parent_entry = self._entries.get(parent)
                if parent_entry is None:
                    raise KeyError(f"Parent config '{parent}' not found in store.")
                child_data = config._dump_for_validation()
                resolved = parent_entry.config | child_data

            self._entries[name] = _ConfigStoreEntry(
                name=name,
                config=resolved,
                parent_name=parent,
            )

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def entry(self, name: str, **kwargs: Any) -> Any:
        """Decorator to register a ConfigBase subclass at definition time.

        The class is instantiated with defaults and registered.

        Args:
            name (str): The name to register the configuration under.
            **kwargs (Any): Default values passed to the configuration class constructor.

        Returns:
            Callable[[type[T]], type[T]]: A decorator that registers and returns the class.
        """

        def decorator(cls: type[T]) -> type[T]:
            instance = cls(**kwargs)
            self.register(name, instance, overwrite=True)
            return cls

        return decorator

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @overload
    def get(self, name: str) -> ConfigBase: ...
    @overload
    def get(self, name: str, as_type: type[T]) -> T: ...

    def get(self, name: str, as_type: type[T] | None = None) -> Any:
        """Retrieve a registered configuration by name.

        If a class is provided, the result is type-checked and returned as the concrete type.

        Args:
            name (str): The registered name of the configuration.
            as_type (type[T] | None, optional): The expected type of the configuration. Defaults to None.

        Returns:
            Any: The retrieved configuration instance.

        Raises:
            KeyError: If the configuration name is not found in the store.
            TypeError: If the retrieved configuration does not match the expected type.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                available = list(self._entries.keys())
                raise KeyError(f"Config '{name}' not found. Available: {available}")
            config = entry.config
            if as_type is not None and not isinstance(config, as_type):
                raise TypeError(f"Config '{name}' is a {type(config).__name__}, expected {as_type.__name__}.")
            return config

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list(self) -> list[str]:
        """Return all registered config names.

        Returns:
            list[str]: A list of registered configuration names.
        """
        with self._lock:
            return list(self._entries.keys())

    def lineage(self, name: str) -> list[str]:
        """Return the ancestor chain for a configuration name, oldest ancestor first.

        Args:
            name (str): The registered name of the configuration.

        Returns:
            list[str]: A list of configuration names forming the ancestry chain.

        Raises:
            RuntimeError: If a cycle is detected in the config lineage.
        """
        chain: list[str] = []
        current: str | None = name
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise RuntimeError(f"Cycle detected in config lineage at '{current}'.")
            seen.add(current)
            chain.append(current)
            entry = self._entries.get(current)
            current = entry.parent_name if entry else None
        chain.reverse()
        return chain

    def clear(self) -> None:
        """Remove all registered configs (mainly for testing)."""
        with self._lock:
            self._entries.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self):
        """Iterate over registered config names.

        Yields:
            str: The name of a registered configuration.
        """
        return iter(list(self._entries.keys()))

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        names = ", ".join(f"'{n}'" for n in self._entries)
        return f"ConfigStore([{names}])"


# Module-level default singleton for convenience
global_store = ConfigStore(name="global")
