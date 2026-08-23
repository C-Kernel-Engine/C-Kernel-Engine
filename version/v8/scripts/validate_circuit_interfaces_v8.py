#!/usr/bin/env python3
"""Join explicit circuit graph slots to canonical provider interfaces."""

from __future__ import annotations

from typing import Any, Dict


class CircuitInterfaceError(RuntimeError):
    """Raised when circuit dataflow cannot satisfy a selected provider."""


def _fault(summary: str, detail: str, remediation: str) -> CircuitInterfaceError:
    return CircuitInterfaceError(
        f"HARD CIRCUIT INTERFACE FAULT: {summary}\n"
        f"  {detail}\n"
        f"  Fix: {remediation}\n"
        "  Do not add a port-name guess, implicit alias, or model-family bypass."
    )


def _port_table(interface: Dict[str, Any], role: str) -> Dict[str, Dict[str, Any]]:
    return {
        str(port["name"]): port
        for port in interface.get("ports", [])
        if isinstance(port, dict) and port.get("role") == role and port.get("name")
    }


def validate_graph_slots(
    *,
    graph_slots: Any,
    interface: Dict[str, Any],
    provider_id: str,
    operation: str,
    phase: str,
    context: str,
) -> Dict[str, Any]:
    """Validate one explicit circuit edge declaration against one provider.

    Weight ports are intentionally excluded: they are bound from the model
    manifest through the map-owned call ABI. This join covers activation/state
    inputs and outputs owned by circuit dataflow.
    """
    interface_id = str(interface.get("id", "") or "").strip()
    if not interface_id:
        raise _fault(
            "selected provider has no canonical interface identity",
            f"context={context}, provider={provider_id!r}",
            "harden the provider map before validating circuit edges.",
        )

    if graph_slots is None:
        return {
            "schema": "cke.v8.circuit_interface_join",
            "schema_version": 1,
            "status": "implicit_dataflow",
            "operation": operation,
            "phase": phase,
            "provider_id": provider_id,
            "operation_interface": interface_id,
        }
    if not isinstance(graph_slots, dict):
        raise _fault(
            "graph_slots is not an object",
            f"context={context}, value={graph_slots!r}",
            "declare graph_slots.inputs and graph_slots.outputs as objects.",
        )
    unknown_directions = sorted(set(graph_slots) - {"inputs", "outputs"})
    if unknown_directions:
        raise _fault(
            "graph_slots contains unsupported sections",
            f"context={context}, sections={unknown_directions}",
            "use only named inputs and outputs; put parameters and weights in their contracts.",
        )

    declared: Dict[str, Dict[str, str]] = {}
    canonical: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for direction, role in (("inputs", "input"), ("outputs", "output")):
        values = graph_slots.get(direction, {})
        if not isinstance(values, dict):
            raise _fault(
                f"graph_slots.{direction} is not an object",
                f"context={context}, value={values!r}",
                f"map each canonical {role} port to exactly one circuit slot.",
            )
        normalized: Dict[str, str] = {}
        for name, slot in values.items():
            if not isinstance(name, str) or not name.strip():
                raise _fault(
                    f"graph_slots.{direction} has an invalid port name",
                    f"context={context}, port={name!r}",
                    "use the exact non-empty canonical provider port name.",
                )
            if not isinstance(slot, str) or not slot.strip():
                raise _fault(
                    f"graph_slots.{direction} has an invalid slot",
                    f"context={context}, port={name!r}, slot={slot!r}",
                    "bind the port to one non-empty circuit slot or explicit external source.",
                )
            normalized[name] = slot

        ports = _port_table(interface, role)
        required = {
            name for name, port in ports.items()
            if str(port.get("consumption", "required")) == "required"
        }
        unknown = sorted(set(normalized) - set(ports))
        missing = sorted(required - set(normalized))
        if unknown:
            raise _fault(
                f"circuit declares unknown {role} ports",
                f"context={context}, interface={interface_id!r}, unknown={unknown}",
                "rename the circuit ports to the canonical operation interface.",
            )
        if missing:
            raise _fault(
                f"circuit omits required {role} ports",
                f"context={context}, interface={interface_id!r}, missing={missing}",
                "connect every required port or mark a genuinely optional provider port as optional.",
            )
        declared[direction] = normalized
        canonical[direction] = ports

    input_slots = declared["inputs"]
    output_slots = declared["outputs"]
    for output_name, output_slot in output_slots.items():
        port = canonical["outputs"][output_name]
        alias_of = port.get("alias_of")
        if alias_of:
            role, _, target_name = str(alias_of).partition(":")
            if role != "input" or target_name not in input_slots:
                raise _fault(
                    "provider alias is not represented by circuit inputs",
                    f"context={context}, output={output_name!r}, alias_of={alias_of!r}",
                    "bind the aliased input explicitly in graph_slots.inputs.",
                )
            if output_slot != input_slots[target_name]:
                raise _fault(
                    "circuit splits an in-place provider alias",
                    f"context={context}, output={output_name!r} slot={output_slot!r}, "
                    f"alias input={target_name!r} slot={input_slots[target_name]!r}",
                    "map the aliased input and output to the same circuit slot.",
                )
        elif output_slot in input_slots.values():
            owners = sorted(name for name, slot in input_slots.items() if slot == output_slot)
            allowed = {
                str(target).split(":", 1)[1]
                for target in port.get("may_alias", [])
                if isinstance(target, str) and target.startswith("input:")
            }
            if set(owners) <= allowed:
                continue
            raise _fault(
                "circuit creates an undeclared writable input/output overlap",
                f"context={context}, output={output_name!r}, slot={output_slot!r}, inputs={owners}",
                "declare alias_of or may_alias in the canonical provider interface, "
                "or use an independent output slot.",
            )

    seen_outputs: Dict[str, str] = {}
    for output_name, output_slot in output_slots.items():
        previous = seen_outputs.get(output_slot)
        if previous is not None:
            raise _fault(
                "multiple outputs own one writable slot",
                f"context={context}, slot={output_slot!r}, outputs={[previous, output_name]}",
                "use separate output slots or one explicitly modeled view/alias contract.",
            )
        seen_outputs[output_slot] = output_name

    return {
        "schema": "cke.v8.circuit_interface_join",
        "schema_version": 1,
        "status": "validated",
        "operation": operation,
        "phase": phase,
        "provider_id": provider_id,
        "operation_interface": interface_id,
        "inputs": dict(input_slots),
        "outputs": dict(output_slots),
        "port_dtypes": {
            "inputs": {
                name: str(canonical["inputs"][name]["dtype"])
                for name in input_slots
            },
            "outputs": {
                name: str(canonical["outputs"][name]["dtype"])
                for name in output_slots
            },
        },
    }
