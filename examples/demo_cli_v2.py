"""
tests/test_cli.py
~~~~~~~~~~~~~~~~~

Tests for canopee.cli.  Only the argparse backend is executed here
(no extra deps needed).  The click / typer tests verify construction
without actually invoking the CLI.

Run with:   python -m pytest tests/test_cli.py -v
       or:  python tests/test_cli.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field

# ── Minimal ConfigBase stub (replace with `from canopee import ConfigBase`) ─


class ConfigBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


# ── Import the module under test ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from canopee.cli import (
    CastRegistry,
    CliParam,
    FieldInspector,
    _build_overrides,
    _deep_merge,
    clify,
)

# ===========================================================================
# Fixtures — config classes
# ===========================================================================


class OptimizerConfig(ConfigBase):
    lr: float = Field(default=1e-3, description="Learning rate")
    beta: float = 0.9


class TrainingConfig(ConfigBase):
    epochs: int = 10
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    verbose: bool = True
    tag: str = "baseline"


class AllTypesConfig(ConfigBase):
    """Exercises every supported type tag."""

    count: int = 4
    ratio: float = 0.5
    name: str = "default"
    flag: bool = True
    mode: Literal["fast", "slow"] = "fast"
    opt_val: Optional[int] = None
    numbers: list[int] = Field(default_factory=lambda: [1, 2, 3])
    pair: tuple[float, float] = (0.0, 1.0)
    variadic: tuple[int, ...] = Field(default_factory=tuple)


class NestedDeepConfig(ConfigBase):
    class Inner(ConfigBase):
        class Innermost(ConfigBase):
            value: int = 42

        deep: Innermost = Field(default_factory=Innermost)

    inner: Inner = Field(default_factory=Inner)


# ===========================================================================
# FieldInspector tests
# ===========================================================================


class TestFieldInspector:
    def setup_method(self):
        self.inspector = FieldInspector()

    def test_flat_fields_extracted(self):
        params = self.inspector.extract(TrainingConfig)
        paths = {p.dot_path for p in params}
        assert "epochs" in paths
        assert "verbose" in paths
        assert "tag" in paths
        # Nested fields flattened
        assert "optimizer.lr" in paths
        assert "optimizer.beta" in paths
        # The sub-model itself is NOT a leaf
        assert "optimizer" not in paths

    def test_bool_is_flag(self):
        params = self.inspector.extract(TrainingConfig)
        verbose = next(p for p in params if p.dot_path == "verbose")
        assert verbose.is_flag
        assert verbose.type_tag == "bool"

    def test_int_field(self):
        params = self.inspector.extract(TrainingConfig)
        epochs = next(p for p in params if p.dot_path == "epochs")
        assert epochs.type_tag == "int"
        assert epochs.default == 10
        assert not epochs.required

    def test_float_field(self):
        params = self.inspector.extract(TrainingConfig)
        lr = next(p for p in params if p.dot_path == "optimizer.lr")
        assert lr.type_tag == "float"
        assert lr.default == 1e-3
        assert lr.description == "Learning rate"

    def test_literal_field(self):
        params = self.inspector.extract(AllTypesConfig)
        mode = next(p for p in params if p.dot_path == "mode")
        assert mode.type_tag == "literal"
        assert set(mode.choices or []) == {"fast", "slow"}

    def test_optional_int(self):
        params = self.inspector.extract(AllTypesConfig)
        opt_val = next(p for p in params if p.dot_path == "opt_val")
        assert opt_val.type_tag == "int"
        assert not opt_val.required

    def test_list_int(self):
        params = self.inspector.extract(AllTypesConfig)
        numbers = next(p for p in params if p.dot_path == "numbers")
        assert numbers.type_tag == "list"
        assert numbers.inner_type is int
        assert numbers.nargs == "+"

    def test_fixed_tuple(self):
        params = self.inspector.extract(AllTypesConfig)
        pair = next(p for p in params if p.dot_path == "pair")
        assert pair.type_tag == "tuple"
        assert pair.inner_type is float
        assert pair.nargs == 2

    def test_variadic_tuple(self):
        params = self.inspector.extract(AllTypesConfig)
        variadic = next(p for p in params if p.dot_path == "variadic")
        assert variadic.type_tag == "tuple"
        assert variadic.inner_type is int
        assert variadic.nargs == "+"

    def test_triple_nested(self):
        params = self.inspector.extract(NestedDeepConfig)
        paths = {p.dot_path for p in params}
        assert "inner.deep.value" in paths

    def test_description_forwarded(self):
        params = self.inspector.extract(TrainingConfig)
        lr = next(p for p in params if p.dot_path == "optimizer.lr")
        assert "Learning rate" in lr.description

    def test_required_field(self):
        class RequiredConfig(ConfigBase):
            name: str  # no default

        params = self.inspector.extract(RequiredConfig)
        name = next(p for p in params if p.dot_path == "name")
        assert name.required


# ===========================================================================
# CastRegistry tests
# ===========================================================================


class TestCastRegistry:
    def _param(self, tag, inner=None, choices=None):
        return CliParam(dot_path="x", type_tag=tag, inner_type=inner, choices=choices)

    def test_int(self):
        assert CastRegistry.cast(self._param("int"), "42") == 42
        assert CastRegistry.cast(self._param("int"), "0") == 0

    def test_float(self):
        assert CastRegistry.cast(self._param("float"), "3.14") == pytest.approx(3.14)
        assert CastRegistry.cast(self._param("float"), "1e-3") == pytest.approx(1e-3)

    def test_bool_truthy(self):
        p = self._param("bool")
        for val in (True, "true", "1", "yes", "on"):
            assert CastRegistry.cast(p, val) is True

    def test_bool_falsy(self):
        p = self._param("bool")
        for val in (False, "false", "0", "no", "off", "none"):
            assert CastRegistry.cast(p, val) is False

    def test_str_none_sentinel(self):
        p = self._param("str")
        assert CastRegistry.cast(p, "none") is None
        assert CastRegistry.cast(p, "null") is None
        assert CastRegistry.cast(p, "hello") == "hello"

    def test_literal(self):
        p = self._param("literal", choices=["fast", "slow"])
        assert CastRegistry.cast(p, "fast") == "fast"

    def test_list_of_ints(self):
        p = CliParam(dot_path="x", type_tag="list", inner_type=int, nargs="+")
        assert CastRegistry.cast(p, ["1", "2", "3"]) == [1, 2, 3]

    def test_list_comma_separated(self):
        p = CliParam(dot_path="x", type_tag="list", inner_type=float, nargs="+")
        assert CastRegistry.cast(p, "1.0,2.0,3.0") == pytest.approx([1.0, 2.0, 3.0])

    def test_tuple_fixed(self):
        p = CliParam(dot_path="x", type_tag="tuple", inner_type=float, nargs=2)
        result = CastRegistry.cast(p, ["0.1", "0.9"])
        assert result == pytest.approx((0.1, 0.9))
        assert isinstance(result, tuple)

    def test_json_blob(self):
        p = self._param("json")
        assert CastRegistry.cast(p, '{"key": 1}') == {"key": 1}
        assert CastRegistry.cast(p, "[1, 2, 3]") == [1, 2, 3]

    def test_none_passthrough(self):
        for tag in ("int", "float", "str", "bool", "list"):
            p = self._param(tag)
            assert CastRegistry.cast(p, None) is None


# ===========================================================================
# _build_overrides / _deep_merge
# ===========================================================================


class TestBuildOverrides:
    def _make_params(self):
        return FieldInspector().extract(TrainingConfig)

    def test_top_level_field(self):
        params = self._make_params()
        namespace = {"epochs": "20"}
        result = _build_overrides(namespace, params)
        assert result["epochs"] == 20

    def test_nested_field(self):
        params = self._make_params()
        namespace = {"optimizer__lr": "3e-4"}
        result = _build_overrides(namespace, params)
        assert result["optimizer"]["lr"] == pytest.approx(3e-4)

    def test_omits_none_defaults(self):
        params = self._make_params()
        namespace = {}
        result = _build_overrides(namespace, params)
        assert result == {}

    def test_deep_merge(self):
        base = {"a": {"b": 1, "c": 2}}
        _deep_merge(base, {"a": {"b": 99}})
        assert base == {"a": {"b": 99, "c": 2}}

    def test_deep_merge_new_key(self):
        base = {"a": 1}
        _deep_merge(base, {"b": 2})
        assert base == {"a": 1, "b": 2}


# ===========================================================================
# End-to-end: argparse backend
# ===========================================================================


class TestArgparseBackend:
    def _run(self, fn, argv):
        """Invoke the clified function with explicit argv."""
        return fn(_argv=argv)

    def test_defaults_used_when_no_args(self):
        captured = {}

        @clify(TrainingConfig, backend="argparse")
        def cmd(cfg):
            captured["cfg"] = cfg

        self._run(cmd, [])
        cfg = captured["cfg"]
        assert cfg.epochs == 10
        assert cfg.optimizer.lr == pytest.approx(1e-3)
        assert cfg.verbose is True

    def test_override_int(self):
        captured = {}

        @clify(TrainingConfig, backend="argparse")
        def cmd(cfg):
            captured["cfg"] = cfg

        self._run(cmd, ["--epochs", "50"])
        assert captured["cfg"].epochs == 50

    def test_override_nested_float(self):
        captured = {}

        @clify(TrainingConfig, backend="argparse")
        def cmd(cfg):
            captured["cfg"] = cfg

        self._run(cmd, ["--optimizer.lr", "1e-5"])
        assert captured["cfg"].optimizer.lr == pytest.approx(1e-5)

    def test_override_bool_flag_disable(self):
        captured = {}

        @clify(TrainingConfig, backend="argparse")
        def cmd(cfg):
            captured["cfg"] = cfg

        self._run(cmd, ["--no-verbose"])
        assert captured["cfg"].verbose is False

    def test_override_bool_flag_enable(self):
        captured = {}

        @clify(TrainingConfig, backend="argparse")
        def cmd(cfg):
            captured["cfg"] = cfg

        self._run(cmd, ["--verbose"])
        assert captured["cfg"].verbose is True

    def test_override_str(self):
        captured = {}

        @clify(TrainingConfig, backend="argparse")
        def cmd(cfg):
            captured["cfg"] = cfg

        self._run(cmd, ["--tag", "experiment_42"])
        assert captured["cfg"].tag == "experiment_42"

    def test_override_literal(self):
        captured = {}

        @clify(AllTypesConfig, backend="argparse")
        def cmd(cfg):
            captured["cfg"] = cfg

        self._run(cmd, ["--mode", "slow"])
        assert captured["cfg"].mode == "slow"

    def test_override_list(self):
        captured = {}

        @clify(AllTypesConfig, backend="argparse")
        def cmd(cfg):
            captured["cfg"] = cfg

        self._run(cmd, ["--numbers", "10", "20", "30"])
        assert captured["cfg"].numbers == [10, 20, 30]

    def test_override_fixed_tuple(self):
        captured = {}

        @clify(AllTypesConfig, backend="argparse")
        def cmd(cfg):
            captured["cfg"] = cfg

        self._run(cmd, ["--pair", "0.1", "0.9"])
        assert captured["cfg"].pair == pytest.approx((0.1, 0.9))

    def test_multiple_overrides(self):
        captured = {}

        @clify(TrainingConfig, backend="argparse")
        def cmd(cfg):
            captured["cfg"] = cfg

        self._run(
            cmd,
            [
                "--epochs",
                "30",
                "--optimizer.lr",
                "5e-4",
                "--optimizer.beta",
                "0.99",
                "--no-verbose",
                "--tag",
                "sweep_run",
            ],
        )
        cfg = captured["cfg"]
        assert cfg.epochs == 30
        assert cfg.optimizer.lr == pytest.approx(5e-4)
        assert cfg.optimizer.beta == pytest.approx(0.99)
        assert cfg.verbose is False
        assert cfg.tag == "sweep_run"

    def test_parser_accessible(self):
        @clify(TrainingConfig, backend="argparse")
        def cmd(cfg):
            pass

        assert hasattr(cmd, "_parser")

    def test_help_contains_field_names(self):
        import io

        @clify(TrainingConfig, backend="argparse")
        def cmd(cfg):
            pass

        buf = io.StringIO()
        try:
            cmd._parser.print_help(buf)
        except SystemExit:
            pass
        help_text = buf.getvalue()
        assert "--epochs" in help_text
        assert "--optimizer.lr" in help_text
        assert "--verbose" in help_text


# ===========================================================================
# Backend construction (click / typer) — no invocation needed
# ===========================================================================


class TestBackendConstruction:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):

            @clify(TrainingConfig, backend="magic")
            def cmd(cfg):
                pass

    def test_non_model_raises(self):
        with pytest.raises(TypeError):

            @clify(dict)
            def cmd(cfg):
                pass

    def test_click_construction(self):
        """Click command object is created without invocation."""
        try:
            import click
        except ImportError:
            pytest.skip("click not installed")

        @clify(TrainingConfig, backend="click")
        def cmd(cfg):
            pass

        import click as _click

        assert isinstance(cmd, _click.BaseCommand)

    def test_typer_construction(self):
        """Typer app object is created without invocation."""
        try:
            import typer
        except ImportError:
            pytest.skip("typer not installed")

        @clify(TrainingConfig, backend="typer")
        def cmd(cfg):
            pass

        assert hasattr(cmd, "_typer_app")


# ===========================================================================
# Demo (run directly)
# ===========================================================================


def _demo():
    """Pretty-print inspection results for TrainingConfig."""
    print("\n" + "─" * 60)
    print("  FieldInspector — TrainingConfig")
    print("─" * 60)

    inspector = FieldInspector()
    params = inspector.extract(TrainingConfig)

    for p in params:
        flag = p.flag
        default = f"  default={p.default!r}" if not p.required else "  REQUIRED"
        choices = f"  choices={p.choices}" if p.choices else ""
        nargs = f"  nargs={p.nargs}" if p.nargs else ""
        print(f"  {flag:<30}  [{p.type_tag:<8}]{default}{choices}{nargs}")

    print("\n" + "─" * 60)
    print("  AllTypesConfig")
    print("─" * 60)

    params2 = inspector.extract(AllTypesConfig)
    for p in params2:
        flag = p.flag
        default = f"  default={p.default!r}" if not p.required else "  REQUIRED"
        choices = f"  choices={p.choices}" if p.choices else ""
        nargs = f"  nargs={p.nargs}" if p.nargs else ""
        print(f"  {flag:<30}  [{p.type_tag:<8}]{default}{choices}{nargs}")

    print("\n" + "─" * 60)
    print("  clify demo — argparse backend")
    print("─" * 60)

    captured = {}

    @clify(TrainingConfig, backend="argparse")
    def main(cfg: TrainingConfig):
        """Train a model."""
        captured["cfg"] = cfg

    # Simulate: python script.py --epochs 5 --optimizer.lr 1e-4 --no-verbose
    main(_argv=["--epochs", "5", "--optimizer.lr", "1e-4", "--no-verbose"])
    cfg = captured["cfg"]
    print(f"  epochs         : {cfg.epochs}")
    print(f"  optimizer.lr   : {cfg.optimizer.lr}")
    print(f"  optimizer.beta : {cfg.optimizer.beta}")
    print(f"  verbose        : {cfg.verbose}")
    # print(f"  fingerprint    : {cfg.fingerprint}")
    print()


if __name__ == "__main__":
    # Run demo first
    _demo()

    # Then run all tests via pytest
    sys.exit(pytest.main([__file__, "-v"]))
