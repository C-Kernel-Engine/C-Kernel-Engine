#!/usr/bin/env python3
"""Fail-closed tests for circuit edge to provider-interface joins."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "validate_circuit_interfaces_v8.py"
SPEC = importlib.util.spec_from_file_location("validate_circuit_interfaces_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def port(
    role: str,
    name: str,
    *,
    consumption: str = "required",
    alias_of: str | None = None,
    may_alias: list[str] | None = None,
) -> dict:
    value = {
        "role": role,
        "name": name,
        "dtype": "fp32",
        "shape": ["T", "E"],
        "layout": "token_major_contiguous",
        "access": "read" if role == "input" else "write",
        "storage_class": "activation",
        "consumption": consumption,
    }
    if alias_of is not None:
        value["alias_of"] = alias_of
    if may_alias is not None:
        value["may_alias"] = may_alias
    return value


def interface(*ports: dict) -> dict:
    return {"id": "synthetic.fp32.v1", "op": "synthetic", "ports": list(ports)}


def validate(graph_slots, operation_interface):
    return validator.validate_graph_slots(
        graph_slots=graph_slots,
        interface=operation_interface,
        provider_id="synthetic_provider",
        operation="synthetic",
        phase="prefill",
        context="circuit=test section=body layer=0 op=synthetic instance=0",
    )


class CircuitInterfaceValidationTests(unittest.TestCase):
    def test_complete_explicit_join_is_validated(self) -> None:
        result = validate(
            {"inputs": {"x": "main"}, "outputs": {"y": "next"}},
            interface(port("input", "x"), port("output", "y")),
        )
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["operation_interface"], "synthetic.fp32.v1")
        self.assertEqual(
            result["port_dtypes"],
            {"inputs": {"x": "fp32"}, "outputs": {"y": "fp32"}},
        )

    def test_implicit_dataflow_remains_visible_migration_debt(self) -> None:
        result = validate(None, interface(port("input", "x"), port("output", "y")))
        self.assertEqual(result["status"], "implicit_dataflow")

    def test_missing_required_input_fails_closed(self) -> None:
        with self.assertRaisesRegex(validator.CircuitInterfaceError, "omits required input"):
            validate(
                {"inputs": {}, "outputs": {"y": "next"}},
                interface(port("input", "x"), port("output", "y")),
            )

    def test_unknown_output_fails_closed(self) -> None:
        with self.assertRaisesRegex(validator.CircuitInterfaceError, "unknown output"):
            validate(
                {"inputs": {"x": "main"}, "outputs": {"guessed": "next"}},
                interface(port("input", "x"), port("output", "y")),
            )

    def test_optional_output_may_be_omitted(self) -> None:
        result = validate(
            {"inputs": {"x": "main"}, "outputs": {"y": "next"}},
            interface(
                port("input", "x"),
                port("output", "y"),
                port("output", "rstd", consumption="optional"),
            ),
        )
        self.assertEqual(result["status"], "validated")

    def test_inplace_alias_requires_one_circuit_slot(self) -> None:
        operation_interface = interface(
            port("input", "x"),
            port("output", "out", alias_of="input:x"),
        )
        result = validate(
            {"inputs": {"x": "scratch"}, "outputs": {"out": "scratch"}},
            operation_interface,
        )
        self.assertEqual(result["status"], "validated")
        with self.assertRaisesRegex(validator.CircuitInterfaceError, "splits an in-place"):
            validate(
                {"inputs": {"x": "scratch"}, "outputs": {"out": "other"}},
                operation_interface,
            )

    def test_undeclared_input_output_overlap_fails_closed(self) -> None:
        with self.assertRaisesRegex(validator.CircuitInterfaceError, "undeclared writable"):
            validate(
                {"inputs": {"x": "scratch"}, "outputs": {"y": "scratch"}},
                interface(port("input", "x"), port("output", "y")),
            )

    def test_optional_alias_permits_separate_or_inplace_output(self) -> None:
        operation_interface = interface(
            port("input", "x"),
            port("output", "y", may_alias=["input:x"]),
        )
        separate = validate(
            {"inputs": {"x": "source"}, "outputs": {"y": "destination"}},
            operation_interface,
        )
        inplace = validate(
            {"inputs": {"x": "source"}, "outputs": {"y": "source"}},
            operation_interface,
        )
        self.assertEqual(separate["status"], "validated")
        self.assertEqual(inplace["status"], "validated")

    def test_two_outputs_cannot_own_one_slot(self) -> None:
        with self.assertRaisesRegex(validator.CircuitInterfaceError, "multiple outputs"):
            validate(
                {
                    "inputs": {"x": "main"},
                    "outputs": {"a": "shared", "b": "shared"},
                },
                interface(
                    port("input", "x"),
                    port("output", "a"),
                    port("output", "b"),
                ),
            )

    def test_empty_slot_and_unknown_section_fail_closed(self) -> None:
        operation_interface = interface(port("input", "x"), port("output", "y"))
        with self.assertRaisesRegex(validator.CircuitInterfaceError, "invalid slot"):
            validate(
                {"inputs": {"x": ""}, "outputs": {"y": "next"}},
                operation_interface,
            )
        with self.assertRaisesRegex(validator.CircuitInterfaceError, "unsupported sections"):
            validate(
                {
                    "inputs": {"x": "main"},
                    "outputs": {"y": "next"},
                    "weights": {"w": "guessed"},
                },
                operation_interface,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
