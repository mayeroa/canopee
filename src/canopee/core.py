"""
canopee.core
~~~~~~~~~~~~

ConfigBase — a frozen, immutable Pydantic v2 model.

Namespace philosophy
--------------------
Every Python method defined on ``ConfigBase`` is a name that users can
never safely use as a field name.  ML configs routinely contain fields
like ``save``, ``load``, ``diff``, ``replace``, ``select``, ``evolve``,
``name``, ``tags`` — all of which would silently shadow methods if we put
the entire API on the class.

The solution: **ConfigBase exposes almost nothing beyond Pydantic's own
``model_*`` namespace**.  The Canopée API lives in free functions.

    # Free-function style — zero collision risk
    import canopee as C

    cfg2  = C.evolve(cfg, epochs=20)
    d     = C.diff(cfg_a, cfg_b)
    flat  = C.to_flat(cfg)
    C.save(cfg, "run.toml")

For users who prefer method-call syntax, an opt-in proxy ``wrap()``
provides the full API without touching the config's own namespace:

    # Proxy style — also zero collision risk
    from canopee import wrap

    wrap(cfg).evolve(epochs=20).diff(other).to_flat()

What *does* live on ConfigBase
------------------------------
Only things that cannot reasonably be free functions:

``cfg | patch``          __or__   — the merge operator (infix syntax requires it)
``cfg.fingerprint``      @computed_field — config data, not a method
``hash(cfg)``            __hash__ — set/dict membership
``repr(cfg)``            __repr__ — debugging

Everything else is a free function or accessed via ``wrap(cfg)``.

Quick reference
---------------

    from canopee.core import ConfigBase, Patch, Diff, wrap
    import canopee.core as cc

    # Define a config
    class TrainingConfig(ConfigBase):
        epochs: int = 10
        save: str = "checkpoint.pt"   # 'save' is safe — no collision
        load: str = "pretrained.pt"   # 'load' is safe — no collision
        diff: float = 1e-4            # even 'diff' is safe

    cfg = TrainingConfig()

    # Operate via free functions
    cfg2 = cc.evolve(cfg, epochs=20)
    cfg3 = cfg | Patch({"epochs": 20})
    cfg3 = cfg | {"epochs": 20}
    d    = cc.diff(cfg, cfg2)
    flat = cc.to_flat(cfg)
    cc.save(cfg, "run.toml")
    cfg4 = cc.load(TrainingConfig, "run.toml")

    # Or via the proxy (opt-in, fluent)
    cfg5 = wrap(cfg).evolve(epochs=20).then.evolve(batch_size=64).unwrap()
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import ItemsView, Iterator
from pathlib import Path
from typing import Any, Generator, Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, computed_field

T = TypeVar("T", bound="ConfigBase")
PathLike = str | Path


# ---------------------------------------------------------------------------
# Patch  ─  first-class override object
# ---------------------------------------------------------------------------


class Patch:
    """
    An explicit, composable set of config overrides.

    A ``Patch`` carries dot-path key→value pairs and is the canonical
    type accepted by the ``|`` operator and all override functions.
    Plain ``dict`` objects are also accepted everywhere as a shorthand.

    Construction
    ------------
    ::

        Patch({"optimizer.lr": 3e-4, "epochs": 20})
        Patch.from_kwargs(optimizer__lr=3e-4, epochs=20)   # __ → .
        Patch.from_flat(flat_dict)
        Patch.identity()                                   # empty / no-op

    Composition
    -----------
    ::

        p1 & p2          →  new Patch; p2 wins on conflict
        p.scoped("opt")  →  prefix every key with "opt."
        p.filtered("opt")→  keep only keys under "opt."
    """

    __slots__ = ("_overrides",)

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        """Initialise with an optional dict of dot-path key→value overrides."""
        self._overrides: dict[str, Any] = dict(overrides or {})

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        """Build from keyword arguments, converting ``__`` to ``.``."""
        return cls({k.replace("__", "."): v for k, v in kwargs.items()})

    @classmethod
    def from_flat(cls, flat: dict[str, Any]) -> Self:
        """Build from an already-flat dot-path dict."""
        return cls(flat)

    @classmethod
    def identity(cls) -> Self:
        """Return an empty (no-op) patch."""
        return cls()

    # ── Dict-like interface ───────────────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        """Return the override value for *key*."""
        return self._overrides[key]

    def __contains__(self, key: str) -> bool:
        """Return ``True`` if *key* is present in this patch."""
        return key in self._overrides

    def __bool__(self) -> bool:
        """Return ``True`` if the patch contains at least one override."""
        return bool(self._overrides)

    def __len__(self) -> int:
        """Return the number of overrides in this patch."""
        return len(self._overrides)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the dot-path keys in this patch."""
        return iter(self._overrides)

    def items(self) -> ItemsView[str, Any]:
        """Return a view of ``(dot_path, value)`` pairs."""
        return self._overrides.items()

    def as_dict(self) -> dict[str, Any]:
        """Return the raw dot-path overrides dict."""
        return dict(self._overrides)

    # ── Composition ───────────────────────────────────────────────────

    def __and__(self, other: Self) -> Self:
        """Compose: ``p1 & p2`` — p2 wins on conflict."""
        return type(self)({**self._overrides, **other._overrides})

    def scoped(self, prefix: str) -> Self:
        """Prefix every key with *prefix*: ``{"lr": v}`` → ``{"opt.lr": v}``."""
        return type(self)({f"{prefix}.{k}": v for k, v in self._overrides.items()})

    def filtered(self, *prefixes: str) -> Self:
        """Keep only keys that fall under any of *prefixes*."""
        return type(self)(
            {k: v for k, v in self._overrides.items() if any(k == p or k.startswith(f"{p}.") for p in prefixes)}
        )

    def __repr__(self) -> str:
        """Return a readable ``Patch({...})`` representation."""
        pairs = ", ".join(f"{k!r}: {v!r}" for k, v in self._overrides.items())
        return f"Patch({{{pairs}}})"

    def __eq__(self, other: object) -> bool:
        """Return ``True`` if *other* is a ``Patch`` with identical overrides."""
        if isinstance(other, Patch):
            return self._overrides == other._overrides
        return NotImplemented


# ---------------------------------------------------------------------------
# Diff  ─  first-class diff object
# ---------------------------------------------------------------------------


class Diff:
    """
    The result of comparing two ``ConfigBase`` instances.

    Every field that differs appears as a ``{dot_path: (old, new)}`` entry
    in ``changes``.

    ::

        d = cc.diff(cfg_a, cfg_b)

        bool(d)           →  True if anything differs
        d["epochs"]       →  (10, 20)
        d.old_values()    →  {"epochs": 10, ...}
        d.new_values()    →  {"epochs": 20, ...}
        d.to_patch()      →  Patch of new values (apply to cfg_a → cfg_b)
        d.invert()        →  Diff with old/new swapped
    """

    __slots__ = ("changes",)

    def __init__(self, changes: dict[str, tuple[Any, Any]]) -> None:
        """Initialise with a mapping of dot-path → ``(old_value, new_value)``."""
        self.changes = changes

    def old_values(self) -> dict[str, Any]:
        """Return a dict of ``{dot_path: old_value}`` for every changed field."""
        return {k: v[0] for k, v in self.changes.items()}

    def new_values(self) -> dict[str, Any]:
        """Return a dict of ``{dot_path: new_value}`` for every changed field."""
        return {k: v[1] for k, v in self.changes.items()}

    def to_patch(self) -> Patch:
        """Return a ``Patch`` of the new values (transforms old → new)."""
        return Patch(self.new_values())

    def invert(self) -> Self:
        """Return a ``Diff`` with old and new swapped."""
        return type(self)({k: (new, old) for k, (old, new) in self.changes.items()})

    def __bool__(self) -> bool:
        """Return ``True`` if at least one field differs."""
        return bool(self.changes)

    def __len__(self) -> int:
        """Return the number of differing fields."""
        return len(self.changes)

    def __contains__(self, key: str) -> bool:
        """Return ``True`` if *key* (dot-path) is among the changed fields."""
        return key in self.changes

    def __getitem__(self, key: str) -> tuple[Any, Any]:
        """Return the ``(old, new)`` tuple for *key*."""
        return self.changes[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the dot-path keys of changed fields."""
        return iter(self.changes)

    def __repr__(self) -> str:
        """Return a human-readable representation with ``old → new`` arrows."""
        if not self.changes:
            return "Diff(∅)"
        lines = [f"  {k!r}: {old!r} → {new!r}" for k, (old, new) in self.changes.items()]
        return "Diff(\n" + "\n".join(lines) + "\n)"

    def __eq__(self, other: object) -> bool:
        """Return ``True`` if *other* is a ``Diff`` with identical changes."""
        if isinstance(other, Diff):
            return self.changes == other.changes
        return NotImplemented


# ---------------------------------------------------------------------------
# ConfigBase  ─  the minimal base class
# ---------------------------------------------------------------------------


class ConfigBase(BaseModel):
    """
    Base class for all Canopée configs.

    **This class intentionally exposes a minimal public interface.**
    The full operation API lives in free functions (``canopee.core.*``)
    and the optional ``wrap()`` proxy, not on the instance, so that
    *any* field name is safe to use without risk of shadowing a method.

    What lives here
    ---------------
    ``cfg | patch``
        Merge operator.  Returns a new validated instance with the patch
        applied.  Accepts a ``Patch``, a plain ``dict``, or another
        ``ConfigBase``.

    ``cfg.fingerprint``
        Stable 16-char SHA-256 hex digest of all field values.
        This is *data* (a ``@computed_field``), not a method, so it
        behaves like any other field.

    ``hash(cfg)``
        Configs are hashable — usable as dict keys and in sets.

    ``repr(cfg)``
        Human-readable field dump.

    Everything else
    ---------------
    Use free functions::

        import canopee.core as cc

        cc.evolve(cfg, epochs=20)
        cc.diff(cfg_a, cfg_b)
        cc.to_flat(cfg)
        cc.select(cfg, "optimizer.lr")
        cc.patched(cfg, epochs=20)      # context manager
        cc.save(cfg, "run.toml")
        cc.load(TrainingConfig, "run.toml")
        cc.dumps(cfg, "yaml")
        cc.loads(TrainingConfig, "yaml", text)
        cc.schema_tree(cfg)

    Or via the proxy for fluent chaining::

        from canopee.core import wrap
        wrap(cfg).evolve(epochs=20).diff(other).to_flat()
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # ------------------------------------------------------------------
    # Merge operator — must live on the instance (infix syntax)
    # ------------------------------------------------------------------

    def __or__(self, other: "Patch | dict[str, Any] | ConfigBase") -> Self:
        """
        Return a new config with *other* applied as an override.

        ``cfg | Patch({"optimizer.lr": 3e-4})``
        ``cfg | {"optimizer.lr": 3e-4}``
        ``cfg | other_cfg``

        The right-hand side wins on conflicts, consistent with Python's
        dict merge convention (``{**a, **b}``).
        """
        if isinstance(other, ConfigBase):
            overrides = _dump(other)
        elif isinstance(other, Patch):
            overrides = other.as_dict()
        elif isinstance(other, dict):
            overrides = other
        else:
            return NotImplemented
        return _copy(self, overrides)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Fingerprint — data, not a method
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def fingerprint(self) -> str:
        """
        Stable 16-character SHA-256 hex digest of all field values.

        Two configs with identical field values always have the same
        fingerprint, regardless of how they were constructed.

        Note: this is a ``@computed_field`` — it is read-only data, not
        a method.  A user field also named ``fingerprint`` would shadow
        this, just as it would shadow any Pydantic computed field.
        """
        data = json.dumps(_dump(self), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Standard dunder
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        """Hash by fingerprint — configs are usable as dict keys and in sets."""
        return hash(self.fingerprint)

    def __repr__(self) -> str:
        """Return a ``ClassName(field=value, ...)`` representation."""
        fields = ", ".join(f"{k}={v!r}" for k, v in _dump(self).items())
        return f"{type(self).__name__}({fields})"


# ---------------------------------------------------------------------------
# Free functions  ─  the full Canopée API, collision-free
# ---------------------------------------------------------------------------


def evolve(config: T, **kwargs: Any) -> T:
    """
    Return a new config instance with the given fields updated.

    Double-underscores in keyword names are converted to dot-paths for
    nested overrides::

        evolve(cfg, epochs=20)
        evolve(cfg, optimizer__lr=3e-4)   # → "optimizer.lr"

    Args:
        config: The config to base the new instance on.
        **kwargs: Fields to update.  Use ``__`` for nested paths.

    Returns:
        A new validated instance of the same type.
    """
    patch = Patch.from_kwargs(**kwargs)
    return _copy(config, patch.as_dict())


def diff(a: ConfigBase, b: ConfigBase) -> Diff:
    """
    Return a ``Diff`` describing every leaf field that differs between
    *a* and *b*.

    ::

        d = diff(cfg_a, cfg_b)
        cfg_b2 = cfg_a | d.to_patch()        # re-derive cfg_b from cfg_a
        cfg_a2 = cfg_b | d.invert().to_patch()  # undo

    Args:
        a: The "before" config.
        b: The "after" config.

    Returns:
        A ``Diff`` object.
    """
    a_flat = _flatten_dict(_dump(a))
    b_flat = _flatten_dict(_dump(b))
    return Diff(
        {k: (a_flat.get(k), b_flat.get(k)) for k in set(a_flat) | set(b_flat) if a_flat.get(k) != b_flat.get(k)}
    )


def to_flat(config: ConfigBase) -> dict[str, Any]:
    """
    Return a flat ``{dot_path: leaf_value}`` dict of all field values.

    ::

        to_flat(cfg)
        # {"optimizer.lr": 1e-3, "optimizer.beta": 0.9, "epochs": 10}

    Args:
        config: The config to flatten.

    Returns:
        A flat dictionary with dot-path keys.
    """
    return _flatten_dict(_dump(config))


def select(config: ConfigBase, dot_path: str) -> Any:
    """
    Return the nested value at *dot_path*.

    ::

        select(cfg, "optimizer")      # → OptimizerConfig(...)
        select(cfg, "optimizer.lr")   # → 1e-3

    Args:
        config: The config to traverse.
        dot_path: Dot-separated path to the target field.

    Returns:
        The value at that path.

    Raises:
        KeyError: If any component of the path does not exist.
    """
    obj: Any = config
    for part in dot_path.split("."):
        if isinstance(obj, (list, tuple)):
            try:
                obj = obj[int(part)]
            except (ValueError, IndexError):
                raise KeyError(
                    f"{part!r} is not a valid index for {type(obj).__name__} (full path: {dot_path!r})"
                ) from None
        elif isinstance(obj, dict):
            try:
                obj = obj[part]
            except KeyError:
                raise KeyError(f"{part!r} not found in dict (full path: {dot_path!r})") from None
        else:
            try:
                obj = getattr(obj, part)
            except AttributeError:
                raise KeyError(f"{part!r} not found on {type(obj).__name__} (full path: {dot_path!r})") from None
    return obj


@contextlib.contextmanager
def patched(config: T, **kwargs: Any) -> Generator[T, None, None]:
    """
    Context manager that yields a temporarily patched config.

    The original is never mutated::

        with patched(cfg, debug=True, batch_size=1) as c:
            inspect_batch(c)
        # cfg is unchanged

    Args:
        config: The config to patch.
        **kwargs: Same semantics as ``evolve``.

    Yields:
        A new validated instance with the overrides applied.
    """
    yield evolve(config, **kwargs)


def schema_tree(config_or_cls: "ConfigBase | type[ConfigBase]") -> dict[str, Any]:
    """
    Return a nested ``{field_name: type}`` tree.

    Accepts either an instance or the class itself::

        schema_tree(cfg)
        schema_tree(TrainingConfig)

    Args:
        config_or_cls: A ``ConfigBase`` instance or subclass.

    Returns:
        A nested dict of field names to their Python type annotations.
    """
    cls = config_or_cls if isinstance(config_or_cls, type) else type(config_or_cls)
    return _build_schema_tree(cls)


# ---------------------------------------------------------------------------
# ConfigProxy  ─  opt-in fluent wrapper, zero methods on ConfigBase
# ---------------------------------------------------------------------------


class ConfigProxy(Generic[T]):
    """
    A thin, opt-in wrapper that exposes the full Canopée API as methods,
    without touching the config's own namespace.

    Obtain via ``wrap(cfg)``.  Every mutating method returns a new
    ``ConfigProxy`` wrapping the updated config, enabling fluent chains.
    Use ``.unwrap()`` to get the underlying ``ConfigBase`` back.

    ::

        from canopee.core import wrap

        cfg2 = (
            wrap(cfg)
            .evolve(epochs=20)
            .evolve(optimizer__lr=3e-4)
            .unwrap()
        )

        # Diff and round-trip
        d = wrap(cfg_a).diff(cfg_b)
        cfg_b2 = wrap(cfg_a).apply(d.to_patch()).unwrap()

    The proxy itself is immutable — each operation returns a *new* proxy
    wrapping the updated config.  The original config is never mutated.
    """

    __slots__ = ("_cfg",)

    def __init__(self, cfg: T) -> None:
        """Wrap *cfg* in a proxy; use ``wrap(cfg)`` rather than calling directly."""
        object.__setattr__(self, "_cfg", cfg)

    # ── Core operations ───────────────────────────────────────────────

    def evolve(self, **kwargs: Any) -> Self:
        """Return a new proxy wrapping ``evolve(cfg, **kwargs)``."""
        return type(self)(evolve(self._cfg, **kwargs))

    def apply(self, patch: "Patch | dict[str, Any]") -> Self:
        """Return a new proxy wrapping ``cfg | patch``."""
        return type(self)(self._cfg | patch)

    def diff(self, other: "ConfigBase | ConfigProxy") -> Diff:
        """Return ``diff(self._cfg, other)``."""
        other_cfg = other._cfg if isinstance(other, ConfigProxy) else other
        return diff(self._cfg, other_cfg)

    def to_flat(self) -> dict[str, Any]:
        """Return ``to_flat(cfg)``."""
        return to_flat(self._cfg)

    def select(self, dot_path: str) -> Any:
        """Return ``select(cfg, dot_path)``."""
        return select(self._cfg, dot_path)

    def save(self, path: PathLike, **kwargs: Any) -> Path:
        """Return ``save(cfg, path)``."""
        from canopee.serialization import save

        return save(self._cfg, path, **kwargs)

    def schema_tree(self) -> dict[str, Any]:
        """Return ``schema_tree(cfg)``."""
        return schema_tree(self._cfg)

    @contextlib.contextmanager
    def patched(self, **kwargs: Any) -> Generator["ConfigProxy[T]", None, None]:
        """Yield a proxy wrapping a temporarily patched config."""
        with patched(self._cfg, **kwargs) as c:
            yield ConfigProxy(c)

    # ── Unwrap ────────────────────────────────────────────────────────

    def unwrap(self) -> T:
        """Return the underlying ``ConfigBase`` instance."""
        return self._cfg

    # ── Pass-through operators ────────────────────────────────────────

    def __or__(self, other: Any) -> Self:
        """Return a new proxy with *other* merged into the wrapped config."""
        return type(self)(self._cfg | other)

    def __repr__(self) -> str:
        """Return ``wrap(<config repr>)``."""
        return f"wrap({self._cfg!r})"

    def __eq__(self, other: object) -> bool:
        """Return ``True`` if the wrapped configs are equal."""
        if isinstance(other, ConfigProxy):
            return self._cfg == other._cfg
        return self._cfg == other

    def __hash__(self) -> int:
        """Delegate hash to the wrapped config."""
        return hash(self._cfg)


def wrap(cfg: T) -> ConfigProxy[T]:
    """
    Wrap *cfg* in a ``ConfigProxy`` for fluent, method-call style access.

    This is a pure convenience — no data is copied, no config is modified.
    The proxy holds a reference to the original immutable config.

    ::

        from canopee.core import wrap

        result = (
            wrap(cfg)
            .evolve(epochs=20, optimizer__lr=3e-4)
            .apply(Patch({"tag": "sweep"}))
            .unwrap()
        )

    Args:
        cfg: Any ``ConfigBase`` instance.

    Returns:
        A ``ConfigProxy`` wrapping *cfg*.
    """
    return ConfigProxy(cfg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dump(config: "ConfigBase") -> dict[str, Any]:
    """
    Return a plain dict representation of *config* suitable for passing to
    ``model_validate``, with all computed fields excluded at every level.

    Delegates entirely to Pydantic's ``model_dump(exclude_computed_fields=True)``,
    which recursively strips computed fields from the model and all nested
    sub-models.  The result contains only regular fields and is safe to pass to
    ``model_validate`` even with ``extra='forbid'``.

    Args:
        config: Any ``ConfigBase`` instance.

    Returns:
        A plain ``dict[str, Any]`` with no computed fields at any level.
    """
    return config.model_dump(exclude_computed_fields=True)


def _copy(config: T, overrides: dict[str, Any]) -> T:
    """
    Return a new validated instance of the same type as *config*, with
    dot-path *overrides* applied.

    Equivalent to the former ``_canopee_copy`` instance method, now a free
    function so ``ConfigBase`` carries no extra methods.

    Args:
        config:    The base config to copy from.
        overrides: Dot-path key → new value mapping.

    Returns:
        A new validated, frozen instance of ``type(config)``.
    """
    base = _apply_dotpath(_dump(config), overrides)
    return type(config).model_validate(base)


def _set_path(
    node: dict[str, Any] | list[Any] | tuple[Any, ...],
    parts: list[str],
    value: Any,
) -> Any:
    """
    Return a new tree with *value* written at *parts*, using structural
    sharing — only the nodes along the path are copied.

    Supports both dict keys and list indices (integer strings like ``"0"``).

    Args:
        node:   The current node (dict or list).
        parts:  Remaining path components.
        value:  The value to write at the leaf.

    Returns:
        A new node of the same type as *node*, with the change applied.

    Raises:
        KeyError:   If an integer index is expected but the key is not an int.
        IndexError: If a list index is out of range.
        TypeError:  If *node* is neither a dict nor a list.
    """
    if not parts:
        return value

    head, *tail = parts

    if isinstance(node, (list, tuple)):
        try:
            idx = int(head)
        except ValueError:
            raise KeyError(
                f"List index must be an integer, got {head!r}. Use e.g. 'layers.0.units' to target list element 0."
            )
        new_list = list(node)  # shallow copy — O(n) but unavoidable for lists
        if idx >= len(new_list):
            new_list.extend([None] * (idx - len(new_list) + 1))

        child = new_list[idx] if new_list[idx] is not None else {}
        new_list[idx] = _set_path(child, tail, value)
        return type(node)(new_list)  # preserve tuple if the original was a tuple

    if isinstance(node, dict):
        new_node = dict(node)  # shallow copy — O(width), not O(subtree)
        child = node.get(head, {})
        new_node[head] = _set_path(child, tail, value)
        return new_node

    raise TypeError(
        f"Cannot index into {type(node).__name__!r} with key {head!r}. Only dicts and lists are traversable."
    )


def _apply_dotpath(
    target: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Return a new dict with all *overrides* applied, using structural sharing.

    Supports dot-path keys for nested dicts and integer indices for lists::

    Round-trip contract
    -------------------
    ``_flatten_dict`` and ``_apply_dotpath`` are inverses *when applied onto
    the original base*::

        base   = _dump(cfg)
        result = _apply_dotpath(base, _flatten_dict(base))
        # result == base  ✓

    Applying a list-containing flat dict onto an *empty* dict is not
    supported — the function has no way to know that ``"layers.0"`` should
    produce a list element rather than a dict key without the existing list
    as context.

        _apply_dotpath(cfg, {"optimizer.lr": 3e-4})
        _apply_dotpath(cfg, {"layers.0.units": 256})
        _apply_dotpath(cfg, {"layers.-1.dropout": 0.1})

    No mutation — *target* is never modified.

    Args:
        target:    The base dict (typically from ``_dump(config)``).
        overrides: Dot-path key → new value.

    Returns:
        A new dict with all overrides applied.
    """
    result: dict[str, Any] = target
    for key, value in overrides.items():
        result = _set_path(result, key.split("."), value)  # type: ignore[assignment]
    return result


def _flatten_dict(
    node: dict[str, Any] | list[Any] | tuple[Any, ...],
    prefix: str = "",
) -> dict[str, Any]:
    """
    Return a flat ``{dot_path: leaf_value}`` dict from a nested structure.

    List elements are indexed by their integer position, so the result
    is round-trippable through ``_apply_dotpath``::

        _flatten_dict({"layers": [{"units": 64}, {"units": 128}]})
        # → {"layers.0.units": 64, "layers.1.units": 128}

    Args:
        node:   A nested dict (or list at inner levels).
        prefix: Accumulated dot-path prefix (used in recursion).

    Returns:
        A flat ``{dot_path: value}`` dict of all leaf nodes.
    """
    result: dict[str, Any] = {}

    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, (list, tuple)):
        items = ((str(i), v) for i, v in enumerate(node))
    else:
        # Leaf — shouldn't normally be called directly on a scalar,
        # but handle it gracefully.
        return {prefix: node} if prefix else {}

    for key, value in items:
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, (dict, list, tuple)):
            result.update(_flatten_dict(value, full_key))
        else:
            result[full_key] = value

    return result


def _build_schema_tree(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Recursively build a ``{field: type_annotation}`` tree."""
    tree: dict[str, Any] = {}
    for name, field_info in model_cls.model_fields.items():
        annotation = field_info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            tree[name] = _build_schema_tree(annotation)
        else:
            tree[name] = annotation
    return tree
