"""
canopee.cli
~~~~~~~~~~~

``clify`` — a decorator that wraps any function taking a ``ConfigBase``
instance and exposes it as a fully-featured CLI, with one flag per config
field (including nested fields via dot-path notation).

Usage
-----

    from canopee import ConfigBase, clify
    from pydantic import Field

    class OptimizerConfig(ConfigBase):
        lr: float = Field(default=1e-3, description="Learning rate")
        beta: float = 0.9

    class TrainingConfig(ConfigBase):
        epochs: int = 10
        optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
        verbose: bool = True

    @clify(TrainingConfig)
    def main(cfg: TrainingConfig):
        print(f"LR: {cfg.optimizer.lr}")
        print(f"Epochs: {cfg.epochs}")

    if __name__ == "__main__":
        main()  # dispatches to the CLI

    # CLI usage:
    #   python train.py --epochs 20 --optimizer.lr 3e-4 --no-verbose

Architecture
------------

    ┌─────────────────────────────────────────────────────────────────┐
    │  clify(config_cls, backend="argparse")                          │
    │   │                                                             │
    │   ├─► FieldInspector.extract(config_cls)                        │
    │   │     Walks the Pydantic v2 JSON schema recursively.          │
    │   │     Returns List[CliParam] — one per leaf field.            │
    │   │                                                             │
    │   ├─► CastRegistry.resolve(cli_param)                           │
    │   │     Maps each CliParam's type tag to a Python cast fn.      │
    │   │                                                             │
    │   ├─► Backend(params).__call__(fn)                              │
    │   │     ArgparseBackend / ClickBackend / TyperBackend           │
    │   │     Registers each CliParam as a flag/option.               │
    │   │                                                             │
    │   └─► On invocation: raw CLI strings                            │
    │         └─► _build_overrides(namespace, params)                 │
    │               Reconstruct nested dict from dot-path flags.      │
    │               └─► ConfigBase.model_validate(overrides)          │
    │                     → validated, frozen cfg instance            │
    └─────────────────────────────────────────────────────────────────┘

Supported field types
---------------------

    Scalars:    int, float, str, bool
    Optional:   Optional[T]  →  default None, accepts "none"/"null"
    Literal:    Literal["a","b"]  →  enum-style choices
    List/Tuple: list[int], tuple[float,...]  →  space- or comma-sep values
    Nested:     ConfigBase sub-models  →  flattened as --parent.child.field
    Union/disc: Union[A,B] with discriminator  →  JSON blob flag
    Any/dict:   dict / Any  →  raw JSON flag

Backends
--------

    "argparse"  stdlib only, no extra dependencies
    "click"     requires pip install click
    "typer"     requires pip install typer
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel
from pydantic.fields import FieldInfo

T = TypeVar("T", bound=BaseModel)

BackendName = Literal["argparse", "click", "typer"]

# ---------------------------------------------------------------------------
# CliParam  ─  backend-agnostic parameter descriptor
# ---------------------------------------------------------------------------

@dataclass
class CliParam:
    """
    A single, backend-agnostic CLI parameter descriptor.

    One ``CliParam`` is produced for every *leaf* field in the config tree
    (nested ``ConfigBase`` sub-models are flattened using dot-path names).

    Attributes
    ----------
    dot_path:
        Full dot-separated path, e.g. ``"optimizer.lr"`` or ``"epochs"``.
        This becomes ``--optimizer.lr`` on the command line.
    type_tag:
        A string identifying the Python type to cast to.  Values:
        ``"int"``, ``"float"``, ``"str"``, ``"bool"``,
        ``"optional_<inner>"``, ``"literal"``, ``"list"``, ``"tuple"``,
        ``"json"`` (unions / dicts / Any).
    inner_type:
        For ``list`` / ``tuple``: the element type as a callable (e.g.
        ``int``).  For ``literal``: unused (choices are in ``choices``).
        For ``optional_<inner>``: the unwrapped inner type callable.
    default:
        The field's default value, or ``None`` if required.
    required:
        ``True`` if the field has no default.
    description:
        Help text, sourced from Pydantic ``Field(description=...)``.
    choices:
        For ``Literal`` fields: the allowed string values.
    is_flag:
        ``True`` for ``bool`` fields that map to ``--flag``/``--no-flag``.
    nargs:
        For list / tuple fields: ``"+"`` or a fixed integer.
    metavar:
        Display name shown in ``--help``.
    """

    dot_path:   str
    type_tag:   str
    inner_type: Callable | None       = None
    default:    Any                   = None
    required:   bool                  = False
    description: str                  = ""
    choices:    list[str] | None      = None
    is_flag:    bool                  = False
    nargs:      str | int | None      = None
    metavar:    str | None            = None

    @property
    def flag(self) -> str:
        """``--optimizer.lr``  (argparse / click style)."""
        return f"--{self.dot_path}"

    @property
    def dest(self) -> str:
        """Argparse ``dest``:  dots replaced by ``__``  →  ``optimizer__lr``."""
        return self.dot_path.replace(".", "__")

    @property
    def env_var(self) -> str:
        """Convention: ``CANOPEE_OPTIMIZER__LR``."""
        return f"CANOPEE_{self.dot_path.replace('.', '__').upper()}"


# ---------------------------------------------------------------------------
# Type resolution helpers
# ---------------------------------------------------------------------------

def _is_config_model(tp: Any) -> bool:
    """Return True if *tp* is a Pydantic BaseModel subclass (not the base)."""
    try:
        return isinstance(tp, type) and issubclass(tp, BaseModel) and tp is not BaseModel
    except TypeError:
        return False


def _unwrap_optional(tp: Any) -> tuple[bool, Any]:
    """
    Detect ``Optional[X]`` / ``X | None`` and return ``(True, X)``.
    Returns ``(False, tp)`` if not Optional.
    """
    origin = get_origin(tp)
    args   = get_args(tp)

    # Union[X, None]  or  X | None  (Python 3.10+)
    if origin is Union or origin is types.UnionType:  # type: ignore[attr-defined]
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == len(args) - 1 and type(None) in args:
            return True, (non_none[0] if len(non_none) == 1 else Union[tuple(non_none)])
    return False, tp


def _unwrap_literal(tp: Any) -> list[str] | None:
    """Return choices list if *tp* is a ``Literal``, else None."""
    if get_origin(tp) is Literal:
        return [str(a) for a in get_args(tp)]
    return None


def _list_element_type(tp: Any) -> Callable | None:
    """Return element cast fn for ``list[X]`` / ``List[X]``, else None."""
    origin = get_origin(tp)
    if origin is list:
        args = get_args(tp)
        return args[0] if args else str
    return None


def _tuple_element_type(tp: Any) -> tuple[Callable | None, int | str | None]:
    """
    Return ``(element_type, nargs)`` for ``tuple`` annotations.

    - ``tuple[int, ...]``  →  ``(int, "+")``
    - ``tuple[int, float, str]``  →  ``(str, 3)``  (mixed → JSON fallback)
    - ``tuple[int, int]``  →  ``(int, 2)``
    """
    origin = get_origin(tp)
    if origin is not tuple:
        return None, None
    args = get_args(tp)
    if not args:
        return str, "+"
    if len(args) == 2 and args[1] is Ellipsis:
        return args[0], "+"
    # Fixed-length — check homogeneous
    unique = set(args)
    if len(unique) == 1:
        return args[0], len(args)
    # Heterogeneous tuple → JSON blob
    return None, None


# ---------------------------------------------------------------------------
# FieldInspector  ─  walks Pydantic schema, yields CliParam objects
# ---------------------------------------------------------------------------

class FieldInspector:
    """
    Recursively inspect a ``ConfigBase`` (Pydantic v2 ``BaseModel``) class
    and yield one ``CliParam`` per *leaf* field.

    Nested sub-models are flattened using dot-path names.
    Discriminated unions and complex types fall back to ``--flag <JSON>``.
    """

    def extract(
        self,
        model_cls: type[BaseModel],
        prefix: str = "",
    ) -> list[CliParam]:
        """
        Walk *model_cls* and return a flat list of ``CliParam`` objects.

        Parameters
        ----------
        model_cls:
            The Pydantic model class to inspect.
        prefix:
            Dot-path prefix for nested models, e.g. ``"optimizer"`` when
            recursing into an optimizer sub-config.
        """
        params: list[CliParam] = []
        hints  = self._get_hints(model_cls)

        for name, field_info in model_cls.model_fields.items():
            dot_path   = f"{prefix}.{name}" if prefix else name
            annotation = hints.get(name, Any)
            param      = self._inspect_field(dot_path, annotation, field_info)

            if param is None:
                # Nested ConfigBase — recurse
                bare = self._bare_type(annotation)
                if _is_config_model(bare):
                    default_factory = (
                        field_info.default_factory
                        if field_info.default_factory not in (None, PydanticUndefined)
                        else None
                    )
                    params.extend(self.extract(bare, prefix=dot_path))
                else:
                    # Unknown compound type → JSON blob
                    params.append(self._json_param(dot_path, field_info))
            else:
                params.append(param)

        return params

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_hints(model_cls: type[BaseModel]) -> dict[str, Any]:
        """Return type hints, falling back gracefully."""
        try:
            import typing
            return typing.get_type_hints(model_cls)
        except Exception:
            return {k: v.annotation for k, v in model_cls.model_fields.items()}

    @staticmethod
    def _default(field_info: FieldInfo) -> tuple[Any, bool]:
        """Return ``(default_value, is_required)``."""
        if field_info.default is not PydanticUndefined:
            return field_info.default, False
        if field_info.default_factory not in (None, PydanticUndefined):  # type: ignore[arg-type]
            return field_info.default_factory(), False  # type: ignore[misc]
        return None, True

    @staticmethod
    def _description(field_info: FieldInfo) -> str:
        return field_info.description or ""

    @staticmethod
    def _bare_type(tp: Any) -> Any:
        """Strip Optional wrapper to get the inner type."""
        _, inner = _unwrap_optional(tp)
        return inner

    def _inspect_field(
        self,
        dot_path: str,
        annotation: Any,
        field_info: FieldInfo,
    ) -> CliParam | None:
        """
        Return a ``CliParam`` for *annotation*, or ``None`` if the field
        is itself a nested model that should be recursed into.
        """
        default, required = self._default(field_info)
        desc              = self._description(field_info)
        is_optional, inner = _unwrap_optional(annotation)

        # ── Nested ConfigBase ──────────────────────────────────────
        bare = get_origin(inner) or inner
        if _is_config_model(bare):
            return None  # caller will recurse

        # ── bool  →  --flag / --no-flag ───────────────────────────
        if inner is bool:
            return CliParam(
                dot_path=dot_path,
                type_tag="bool",
                default=default,
                required=required,
                description=desc,
                is_flag=True,
                metavar=None,
            )

        # ── Literal ───────────────────────────────────────────────
        choices = _unwrap_literal(inner)
        if choices is not None:
            return CliParam(
                dot_path=dot_path,
                type_tag="literal",
                default=str(default) if default is not None else None,
                required=required,
                description=desc + f"  choices: {choices}",
                choices=choices,
                metavar=f"{{{','.join(choices)}}}",
            )

        # ── list[X] ───────────────────────────────────────────────
        elem = _list_element_type(inner)
        if elem is not None:
            return CliParam(
                dot_path=dot_path,
                type_tag="list",
                inner_type=elem,
                default=default,
                required=required,
                description=desc + "  (space-separated values)",
                nargs="+",
                metavar=getattr(elem, "__name__", "VALUE").upper(),
            )

        # ── tuple[X, ...] / tuple[X, X] ──────────────────────────
        tup_elem, tup_nargs = _tuple_element_type(inner)
        if tup_nargs is not None and tup_elem is not None:
            return CliParam(
                dot_path=dot_path,
                type_tag="tuple",
                inner_type=tup_elem,
                default=list(default) if default is not None else None,
                required=required,
                description=desc + f"  (tuple, {tup_nargs} values)",
                nargs=tup_nargs,
                metavar=getattr(tup_elem, "__name__", "VALUE").upper(),
            )

        # ── int / float / str ─────────────────────────────────────
        for scalar_type in (int, float, str):
            if inner is scalar_type:
                tag = scalar_type.__name__
                return CliParam(
                    dot_path=dot_path,
                    type_tag=tag,
                    inner_type=scalar_type,
                    default=default,
                    required=required,
                    description=desc + (" (optional)" if is_optional else ""),
                    metavar=tag.upper(),
                )

        # ── Optional[scalar] that we already unwrapped above ─────
        if is_optional:
            # Try again on the bare inner type
            bare_param = self._inspect_field(dot_path, inner, field_info)
            if bare_param is not None:
                bare_param.description += " (optional, pass 'none' to unset)"
                return bare_param

        # ── Fallback: raw JSON blob ───────────────────────────────
        return self._json_param(dot_path, field_info)

    def _json_param(self, dot_path: str, field_info: FieldInfo) -> CliParam:
        default, required = self._default(field_info)
        return CliParam(
            dot_path=dot_path,
            type_tag="json",
            default=json.dumps(default) if default is not None else None,
            required=required,
            description=self._description(field_info) + "  (JSON string)",
            metavar="JSON",
        )


# Pydantic v2 sentinel for "no default given"
try:
    from pydantic_core import PydanticUndefinedType
    from pydantic.fields import _Unset as PydanticUndefined  # type: ignore[attr-defined]
except ImportError:
    try:
        from pydantic.fields import Undefined as PydanticUndefined  # type: ignore[assignment]
    except ImportError:
        PydanticUndefined = None  # type: ignore[assignment,misc]

# More robust: check via sentinel object
try:
    from pydantic_core import PydanticUndefinedType  # noqa: F811
    def _is_undefined(v: Any) -> bool:
        return isinstance(v, PydanticUndefinedType)
except ImportError:
    def _is_undefined(v: Any) -> bool:  # type: ignore[misc]
        return v is None


# Patch FieldInspector._default to use the robust sentinel
def _field_default(field_info: FieldInfo) -> tuple[Any, bool]:
    from pydantic_core import PydanticUndefinedType
    if isinstance(field_info.default, PydanticUndefinedType):
        # No default — check factory
        if field_info.default_factory is not None:
            try:
                return field_info.default_factory(), False
            except Exception:
                return None, False
        return None, True
    return field_info.default, False

FieldInspector._default = staticmethod(_field_default)  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# CastRegistry  ─  string → Python coercion, centralised
# ---------------------------------------------------------------------------

class CastRegistry:
    """
    Maps a ``CliParam.type_tag`` to a function that converts a raw CLI
    string (or list of strings) to the correct Python value.

    All backends use the same registry, ensuring consistent behaviour
    regardless of which backend is chosen.
    """

    @staticmethod
    def cast(param: CliParam, raw: Any) -> Any:
        """
        Convert *raw* (a string, list of strings, or already-typed value)
        to the Python type indicated by *param*.

        Returns the cast value, ready to be passed into ``model_validate``.
        """
        if raw is None:
            return None

        tag = param.type_tag

        if tag == "bool":
            if isinstance(raw, bool):
                return raw
            return str(raw).lower() not in ("false", "0", "no", "off", "none")

        if tag == "int":
            return int(raw)

        if tag == "float":
            return float(raw)

        if tag == "str":
            s = str(raw)
            return None if s.lower() in ("none", "null") else s

        if tag == "literal":
            return str(raw)

        if tag in ("list", "tuple"):
            cast_fn = param.inner_type or str
            if isinstance(raw, (list, tuple)):
                result = [cast_fn(v) for v in raw]
            else:
                # Comma-separated single string
                result = [cast_fn(v.strip()) for v in str(raw).split(",") if v.strip()]
            return result if tag == "list" else tuple(result)

        if tag == "json":
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return raw
            return raw

        # Unknown tag — return as-is
        return raw


# ---------------------------------------------------------------------------
# Override reconstruction
# ---------------------------------------------------------------------------

def _build_overrides(
    namespace: dict[str, Any],
    params: list[CliParam],
) -> dict[str, Any]:
    """
    Convert a flat ``{dest: value}`` namespace from the CLI parser back
    into a nested dict suitable for ``model_validate``.

    Fields that were not set on the CLI (value is ``None`` and not
    required) are omitted so Pydantic falls back to the field default.

    dot_path ``"optimizer.lr"``  →  ``{"optimizer": {"lr": 0.001}}``
    """
    overrides: dict[str, Any] = {}

    for param in params:
        raw = namespace.get(param.dest)
        if raw is None and not param.required:
            continue

        value = CastRegistry.cast(param, raw)
        if value is None and not param.required:
            continue

        # Build nested dict from dot-path
        parts = param.dot_path.split(".")
        d = overrides
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value

    return overrides


def _merge_with_defaults(
    config_cls: type[T],
    overrides: dict[str, Any],
) -> T:
    """
    Construct a *config_cls* instance by merging CLI overrides on top
    of the schema defaults.  Uses ``model_validate`` so all validators run.
    """
    # Get baseline defaults via a default-constructed instance
    # (works because all fields must have defaults when CLI is partial)
    try:
        baseline = config_cls().model_dump(
            exclude=set(config_cls.model_computed_fields.keys())
        )
    except Exception:
        baseline = {}

    # Deep-merge overrides on top of baseline
    _deep_merge(baseline, overrides)
    return config_cls.model_validate(baseline)


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge *overrides* into *base* in-place."""
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class Backend(ABC):
    """
    Abstract base class for CLI backends.

    Each backend receives the list of ``CliParam`` objects and the
    config class, and wraps the user's function so that invoking it
    (with no arguments) parses the CLI and calls the function with a
    fully-validated config instance.
    """

    def __init__(
        self,
        config_cls: type[T],
        params: list[CliParam],
        prog: str | None = None,
        description: str | None = None,
    ) -> None:
        self.config_cls  = config_cls
        self.params      = params
        self.prog        = prog
        self.description = description or f"CLI for {config_cls.__name__}"

    @abstractmethod
    def wrap(self, fn: Callable) -> Callable:
        """
        Wrap *fn* so that calling it with no arguments triggers CLI
        parsing and invokes *fn* with the resulting config instance.
        """
        ...

    # Shared helper used by all backends
    def _build_config(self, namespace: dict[str, Any]) -> T:
        overrides = _build_overrides(namespace, self.params)
        return _merge_with_defaults(self.config_cls, overrides)


# ---------------------------------------------------------------------------
# ArgparseBackend
# ---------------------------------------------------------------------------

class ArgparseBackend(Backend):
    """
    stdlib ``argparse`` backend.  No extra dependencies required.

    Bool fields become ``--flag`` / ``--no-flag`` argument pairs.
    List/tuple fields accept multiple space-separated values.
    Literal fields produce ``choices=`` validation.
    All other compound types accept a raw JSON string.
    """

    def wrap(self, fn: Callable) -> Callable:
        parser = self._build_parser(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Allow passing explicit argv for testing
            argv = kwargs.pop("_argv", None)
            namespace = vars(parser.parse_args(argv))
            cfg = self._build_config(namespace)
            return fn(cfg)

        wrapper._parser = parser  # type: ignore[attr-defined]
        return wrapper

    def _build_parser(self, fn: Callable) -> argparse.ArgumentParser:
        doc = inspect.getdoc(fn) or ""
        parser = argparse.ArgumentParser(
            prog=self.prog,
            description=self.description + ("\n\n" + doc if doc else ""),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        group = parser.add_argument_group("config fields")

        for param in self.params:
            self._add_argument(group, param)

        return parser

    @staticmethod
    def _add_argument(
        group: argparse._ArgumentGroup,
        param: CliParam,
    ) -> None:
        kwargs: dict[str, Any] = {
            "dest":    param.dest,
            "default": argparse.SUPPRESS,  # omitted → not in namespace
            "help":    param.description or "",
        }

        if param.is_flag:
            # Bool: --verbose / --no-verbose
            group.add_argument(
                param.flag,
                dest=param.dest,
                action="store_true",
                default=argparse.SUPPRESS,
                help=param.description,
            )
            group.add_argument(
                f"--no-{param.dot_path}",
                dest=param.dest,
                action="store_false",
                help=f"Disable {param.dot_path}",
            )
            return

        if param.choices:
            kwargs["choices"] = param.choices

        if param.nargs:
            kwargs["nargs"]      = param.nargs
            kwargs["type"]       = param.inner_type or str
            kwargs["metavar"]    = param.metavar
        else:
            kwargs["type"]       = str   # cast happens in CastRegistry
            kwargs["metavar"]    = param.metavar

        if param.required:
            kwargs.pop("default", None)

        group.add_argument(param.flag, **kwargs)


# ---------------------------------------------------------------------------
# ClickBackend
# ---------------------------------------------------------------------------

class ClickBackend(Backend):
    """
    ``click`` backend.  Requires ``pip install click``.

    Builds a ``click.Command`` declaratively — one ``click.Option``
    instance per ``CliParam`` — using Click's constructor API rather
    than the decorator API.  This makes the command object fully
    introspectable and reusable by subclasses (see ``TyperBackend``).

    The ``click.Command`` callback receives a flat ``**kwargs`` dict
    whose keys are the ``dest`` identifiers (dots replaced by ``__``).
    These are forwarded to ``_build_overrides`` → ``_merge_with_defaults``
    to produce the validated, frozen config instance.

    Dotted flag names (``--optimizer.lr``) are handled by passing a bare
    Python identifier as the second entry in ``param_decls``::

        click.Option(["--optimizer.lr", "optimizer__lr"], ...)

    Click's ``_parse_decls`` treats any decl that ``str.isidentifier()``
    as the explicit dest name, leaving the dotted flag string untouched.
    """

    # Name of the import that must be present for this backend to work.
    # Overridden by TyperBackend to check for typer instead.
    _required_package: str = "click"

    def wrap(self, fn: Callable) -> Callable:
        self._check_import()
        import click

        cmd = self._build_click_command(fn, click)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Support _argv= for testing (mirrors the argparse backend)
            argv       = kwargs.pop("_argv", None)
            standalone = kwargs.pop("_standalone", True)
            cmd.main(args=argv, standalone_mode=standalone,
                     prog_name=self.prog)

        wrapper._click_command = cmd  # type: ignore[attr-defined]
        return wrapper

    # ------------------------------------------------------------------
    # Click command construction — shared with TyperBackend
    # ------------------------------------------------------------------

    def _build_click_command(self, fn: Callable, click: Any) -> Any:
        """
        Build and return a ``click.Command`` for *fn*.

        The command's ``params`` list is constructed by calling
        ``_make_click_option`` for every ``CliParam``.  The callback
        reconstructs the config and invokes *fn*.

        This method is intentionally public so that ``TyperBackend`` can
        call it directly to obtain the ``click.Command`` it wraps.
        """
        click_params = [self._make_click_option(p, click) for p in self.params]
        backend      = self

        def callback(**kwargs: Any) -> None:
            cfg = backend._build_config(kwargs)
            fn(cfg)

        return click.Command(
            name=self.prog or fn.__name__,
            params=click_params,
            callback=callback,
            help=inspect.getdoc(fn) or self.description,
        )

    @staticmethod
    def _make_click_option(param: CliParam, click: Any) -> Any:
        """
        Build a single ``click.Option`` instance from a ``CliParam``.

        Type mapping
        ------------
        ``bool``          → ``--flag/--no-flag`` slash declaration
        ``literal``       → ``click.Choice(choices)``
        ``list``/``tuple``→ ``multiple=True``, typed element
        ``int``           → ``click.INT``  (native error messages)
        ``float``         → ``click.FLOAT``
        ``str``/``json``  → ``click.STRING``  (CastRegistry coerces later)
        """
        tag  = param.type_tag
        dest = param.dest  # e.g. "optimizer__lr"
        flag = param.flag  # e.g. "--optimizer.lr"

        # ── bool: --flag/--no-flag ────────────────────────────────
        if tag == "bool":
            return click.Option(
                [f"{flag}/--no-{param.dot_path}", dest],
                default=param.default,
                show_default=True,
                help=param.description,
            )

        # All other types: [dotted-flag, bare-dest-identifier]
        # The bare identifier is what Click uses as the callback kwarg key.
        decls: list[str] = [flag, dest]

        # ── Literal → click.Choice ────────────────────────────────
        if tag == "literal":
            return click.Option(
                decls,
                type=click.Choice(param.choices or []),
                default=param.default,
                show_default=True,
                required=param.required,
                help=param.description,
            )

        # ── list / tuple → multiple=True ─────────────────────────
        if tag in ("list", "tuple"):
            return click.Option(
                decls,
                type=param.inner_type or str,
                multiple=True,
                default=param.default or (),
                show_default=True,
                required=param.required,
                help=param.description,
                metavar=param.metavar,
            )

        # ── int / float → native Click scalar types ───────────────
        if tag == "int":
            return click.Option(
                decls,
                type=click.INT,
                default=param.default,
                show_default=True,
                required=param.required,
                help=param.description,
                metavar=param.metavar or "INT",
            )

        if tag == "float":
            return click.Option(
                decls,
                type=click.FLOAT,
                default=param.default,
                show_default=True,
                required=param.required,
                help=param.description,
                metavar=param.metavar or "FLOAT",
            )

        # ── str / json / fallback → plain string ─────────────────
        return click.Option(
            decls,
            type=click.STRING,
            default=param.default,
            show_default=param.default is not None,
            required=param.required,
            help=param.description,
            metavar=param.metavar or "TEXT",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_import(self) -> None:
        """Raise a clear ImportError if the required package is missing."""
        try:
            __import__(self._required_package)
        except ImportError as exc:
            raise ImportError(
                f"{self._required_package!r} is required for the "
                f"{type(self).__name__!r} backend.\n"
                f"Install it with: pip install {self._required_package}"
            ) from exc


# ---------------------------------------------------------------------------
# TyperBackend
# ---------------------------------------------------------------------------

class TyperBackend(ClickBackend):
    """
    ``typer`` backend.  Requires ``pip install typer``.

    Inherits from ``ClickBackend`` and reuses ``_build_click_command``
    and ``_make_click_option`` without modification — the ``click.Command``
    built by the parent is identical for both backends.

    The only responsibility added here is wrapping that ``click.Command``
    in a ``typer.Typer`` app so callers get the full Typer experience
    (rich ``--help`` formatting, shell-completion hooks, etc.).

    Inheritance rationale
    ---------------------
    Typer is built on top of Click; every Typer command is ultimately a
    ``click.Command``.  Rather than duplicating ``_make_click_option``
    (which maps ``CliParam`` type tags to ``click.Option`` instances), we
    let ``ClickBackend`` own that logic and ``TyperBackend`` only overrides
    ``wrap`` to add the Typer layer on top.

    Override surface
    ----------------
    ``wrap``              — adds ``typer.Typer`` app around the click command.
    ``_required_package`` — changed to ``"typer"`` for the import check.
    """

    _required_package: str = "typer"

    # def wrap(self, fn: Callable) -> Callable:
    #     self._check_import()
    #     import click
    #     import typer  # type: ignore[import]

    #     # Build the click.Option list and the config-dispatching callback
    #     # exactly as ClickBackend does — but stop before creating the
    #     # click.Command instance.
    #     click_params = [self._make_click_option(p, click) for p in self.params]
    #     backend      = self

    #     def callback(**kwargs: Any) -> None:
    #         cfg = backend._build_config(kwargs)
    #         fn(cfg)

    #     help_text = inspect.getdoc(fn) or self.description

    #     # Typer's app.command(cls=SomeClass) tells Typer to instantiate
    #     # *SomeClass* instead of building its own click.Command from a
    #     # function signature.  This is the public escape hatch for injecting
    #     # a pre-built click.Command into a Typer app.
    #     #
    #     # We create a fresh click.Command subclass per decoration so each
    #     # @clify usage gets its own isolated params list and callback — no
    #     # shared mutable state between decorated functions.
    #     CommandClass = type(
    #         f"{fn.__name__.title()}Command",
    #         (click.Command,),
    #         {
    #             # Class-level attributes read by __init__
    #             "_canopee_params":   click_params,
    #             "_canopee_callback": staticmethod(callback),
    #             "_canopee_help":     help_text,
    #             # Override __init__ to wire everything up
    #             "__init__": _typer_command_init,
    #         },
    #     )

    #     app = typer.Typer(
    #         help=self.description,
    #         add_completion=False,
    #         no_args_is_help=False,
    #     )
    #     # Register a no-op placeholder function so Typer knows the command
    #     # name and associates cls= with it.  Typer will instantiate
    #     # CommandClass instead of building its own command from the placeholder.
    #     app.command(
    #         name=self.prog or fn.__name__,
    #         cls=CommandClass,
    #     )(lambda: None)  # placeholder — never called; CommandClass.callback runs instead

    #     @functools.wraps(fn)
    #     def wrapper(*args: Any, **kwargs: Any) -> Any:
    #         argv       = kwargs.pop("_argv", None)
    #         standalone = kwargs.pop("_standalone", True)
    #         app(
    #             args=argv,
    #             standalone_mode=standalone,
    #             prog_name=self.prog or fn.__name__,
    #         )

    #     wrapper._typer_app = app  # type: ignore[attr-defined]
    #     return wrapper

    def wrap(self, fn: Callable) -> Callable:
        self._check_import()
        import click

        cmd = self._build_typer_command(fn, click)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Support _argv= for testing (mirrors the argparse backend)
            argv       = kwargs.pop("_argv", None)
            standalone = kwargs.pop("_standalone", True)
            cmd.main(args=argv, standalone_mode=standalone,
                     prog_name=self.prog)

        wrapper._typer_command = cmd  # type: ignore[attr-defined]
        return wrapper

    def _build_typer_command(self, fn: Callable, click: Any) -> Any:
        """
        Build and return a ``click.Command`` for *fn*.

        The command's ``params`` list is constructed by calling
        ``_make_click_option`` for every ``CliParam``.  The callback
        reconstructs the config and invokes *fn*.

        This method is intentionally public so that ``TyperBackend`` can
        call it directly to obtain the ``click.Command`` it wraps.
        """
        import typer

        click_params = [self._make_click_option(p, click) for p in self.params]
        backend      = self

        def callback(**kwargs: Any) -> None:
            cfg = backend._build_config(kwargs)
            fn(cfg)

        return typer.core.TyperCommand(
            name=self.prog or fn.__name__,
            params=click_params,
            callback=callback,
            help=inspect.getdoc(fn) or self.description,
        )


def _typer_command_init(self: Any, name: str | None = None, **attrs: Any) -> None:
    """
    ``__init__`` injected into the per-decoration ``CommandClass``.

    Reads the pre-built params list and callback from class attributes
    and passes them to ``click.Command.__init__``.  This fires when Typer
    calls ``CommandClass(name=..., callback=<placeholder>, params=[...], ...)``
    during ``app()``.

    ``callback`` and ``params`` are explicitly popped from ``attrs`` before
    forwarding — Typer always passes them from its own build pipeline, but
    we own both and must not let them conflict with ours.
    """
    import click as _click  # local import — click may not be at module level

    attrs.pop("callback", None)  # discard Typer's placeholder lambda
    attrs.pop("params",   None)  # discard Typer's (empty) params list
    attrs.pop("help",     None)  # discard Typer's generated help string

    _click.Command.__init__(
        self,
        name=name,
        params=type(self)._canopee_params,
        callback=type(self)._canopee_callback,
        help=type(self)._canopee_help,  # always use our help; discard Typer's
        **attrs,
    )


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[Backend]] = {
    "argparse": ArgparseBackend,
    "click":    ClickBackend,
    "typer":    TyperBackend,
}


def register_backend(name: str, cls: type[Backend]) -> None:
    """Register a custom backend under *name*."""
    _BACKENDS[name] = cls


# ---------------------------------------------------------------------------
# clify  ─  the public decorator
# ---------------------------------------------------------------------------

def clify(
    config_cls: type[T],
    *,
    backend: BackendName | str = "argparse",
    prog: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[[T], Any]], Callable[[], Any]]:
    """
    Decorator that turns a function taking a ``ConfigBase`` instance into
    a fully-featured CLI entry point.

    Parameters
    ----------
    config_cls:
        The ``ConfigBase`` subclass that defines the config schema.
        Its fields are automatically exposed as CLI flags.
    backend:
        Which CLI framework to use: ``"argparse"`` (default, no extra deps),
        ``"click"``, or ``"typer"``.
    prog:
        Override the program name shown in ``--help``.
    description:
        Override the help description.  Defaults to the function docstring.

    Returns
    -------
    A decorator.  When applied to a function ``fn(cfg: ConfigCls)``, the
    decorated function can be called with no arguments to parse ``sys.argv``
    and invoke *fn* with the resulting config.

    Examples
    --------

        @clify(TrainingConfig)
        def main(cfg: TrainingConfig):
            print(cfg.epochs)

        if __name__ == "__main__":
            main()

        # Or with a specific backend:
        @clify(TrainingConfig, backend="click")
        def main(cfg: TrainingConfig): ...

        # Or with typer:
        @clify(TrainingConfig, backend="typer")
        def main(cfg: TrainingConfig): ...

    CLI usage:

        python train.py --epochs 20 --optimizer.lr 3e-4 --no-verbose
        python train.py --help
    """

    if config_cls is None or not (
        isinstance(config_cls, type) and issubclass(config_cls, BaseModel)
    ):
        raise TypeError(
            f"clify expects a Pydantic BaseModel subclass, got {config_cls!r}"
        )

    backend_cls = _BACKENDS.get(backend)
    if backend_cls is None:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            f"Available: {list(_BACKENDS)}"
        )

    inspector = FieldInspector()
    params    = inspector.extract(config_cls)

    def decorator(fn: Callable[[T], Any]) -> Callable[[], Any]:
        desc = description or inspect.getdoc(fn) or f"CLI for {config_cls.__name__}"
        bk   = backend_cls(
            config_cls=config_cls,
            params=params,
            prog=prog or fn.__name__,
            description=desc,
        )
        return bk.wrap(fn)

    return decorator