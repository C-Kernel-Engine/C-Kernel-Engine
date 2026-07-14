#!/usr/bin/env python3
"""Resolve circuit numerical requirements to one exact kernel implementation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from jsonschema import Draft202012Validator


V8_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V8_ROOT.parents[1]
DEFAULT_CONTRACTS = V8_ROOT / "contracts" / "numerical_execution.json"
DEFAULT_KERNELS = V8_ROOT / "kernel_maps"
SCHEMA_ROOT = V8_ROOT / "schemas"
CONTRACT_SCHEMA = SCHEMA_ROOT / "numerical_execution_contract_registry.schema.json"
REQUIREMENTS_SCHEMA = SCHEMA_ROOT / "numerical_required_contracts.schema.json"
CAPABILITY_SCHEMA = SCHEMA_ROOT / "numerical_kernel_capability.schema.json"
RESOLVED_SCHEMA = SCHEMA_ROOT / "resolved_numerical_execution_contract.schema.json"
VALID_STATES = {"unresolved", "observed", "validated"}
AMBIGUOUS_IDS = {"auto", "default", "fast", "strict", "fp16", "bf16", "fp32"}


class ContractError(RuntimeError):
    pass


def hard_fault(summary: str, detail: str, remediation: str) -> ContractError:
    return ContractError(
        f"HARD CONTRACT FAULT: {summary}\n  {detail}\n  Fix: {remediation}\n"
        "  Do not add ranking, fallback dispatch, or tolerance relaxation."
    )


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot load contract JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Expected JSON object in {path}")
    return value


def validate_schema(value: Dict[str, Any], path: Path, context: str) -> None:
    errors = sorted(
        Draft202012Validator(load_json(path)).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise hard_fault(
            f"{context} violates {path.name}",
            f"At {location}: {error.message}",
            "correct the versioned circuit, contract registry, or kernel capability.",
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_contract_registry(doc: Dict[str, Any]) -> None:
    validate_schema(doc, CONTRACT_SCHEMA, "numerical execution contract registry")
    for contract_id, contract in doc["contracts"].items():
        if contract_id.lower() in AMBIGUOUS_IDS:
            raise hard_fault(
                f"ambiguous numerical contract ID {contract_id!r}",
                "A dtype or policy label does not define storage, rounding, reduction, and threading.",
                "use a stable semantic contract ID with complete execution semantics.",
            )
        transform = contract.get("position_transform")
        if isinstance(transform, dict):
            if transform["position_rank"] != len(transform["axis_order"]):
                raise hard_fault(
                    f"position rank/axis mismatch in {contract_id!r}",
                    f"position_rank={transform['position_rank']} axis_order={transform['axis_order']}",
                    "make the position rank equal the number of named axes.",
                )
            section_semantics = transform["section_interpretation"]
            if transform["pairing"] == "multi_section" and section_semantics not in {
                "axis_selection",
                "interleaved_axis_selection",
            }:
                raise hard_fault(
                    f"multi-section position contract {contract_id!r} redefines rotary width",
                    "Qwen-style M-RoPE sections select position axes; rotary_width remains independent.",
                    "use axis_selection or interleaved_axis_selection and validate the full rotary width separately.",
                )
            rotary_width = transform.get("rotary_width_value")
            head_width = transform.get("head_width_value")
            mrope_width = transform.get("mrope_n_dims_value")
            if rotary_width is not None and head_width is not None and rotary_width > head_width:
                raise hard_fault(
                    f"position contract {contract_id!r} rotates beyond the head width",
                    f"rotary_width={rotary_width}, head_width={head_width}",
                    "declare the actual partial/full rotary width independently of section routing.",
                )
            if mrope_width is not None and rotary_width is not None and mrope_width != rotary_width:
                raise hard_fault(
                    f"position contract {contract_id!r} has inconsistent M-RoPE width",
                    f"mrope_n_dims={mrope_width}, required_rotary_width={rotary_width}",
                    "make mrope_n_dims equal the required full rotary width; sections only select axes.",
                )


def _validate_capability_against_contract(
    kernel_id: str,
    capability: Dict[str, Any],
    contract: Dict[str, Any],
) -> None:
    validate_schema(capability, CAPABILITY_SCHEMA, f"kernel {kernel_id} numerical capability")
    semantics = contract
    threading = semantics["threading"]
    reduction = semantics["reduction"]
    advertised = capability["arithmetic"]
    expected = {
        "partial_accumulator": reduction["partial_accumulator"],
        "merge_order": reduction["merge_order"],
        "deterministic": threading["deterministic"],
        "thread_count_changes_arithmetic_order": threading["thread_count_changes_arithmetic_order"],
        "split_strategy": threading["split_strategy"],
    }
    if advertised != expected:
        raise hard_fault(
            f"kernel {kernel_id!r} arithmetic metadata disagrees with contract {capability['contract_id']!r}",
            f"expected={expected}, advertised={advertised}",
            "correct the kernel capability or bind it to the contract it actually implements.",
        )
    partitions = capability["implementation"]["threading"]["work_partition"]
    if threading["work_partition"] not in partitions:
        raise hard_fault(
            f"kernel {kernel_id!r} cannot satisfy work partition {threading['work_partition']!r}",
            f"advertised partitions={partitions}",
            "advertise the exact partition only after validating the implementation.",
        )


def load_kernel_capabilities(
    root: Path = DEFAULT_KERNELS,
    contracts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    registry = contracts or load_json(DEFAULT_CONTRACTS)
    validate_contract_registry(registry)
    kernels: Dict[str, Any] = {}
    for path in sorted(root.glob("*.json")):
        doc = load_json(path)
        capabilities = doc.get("numerical_capabilities")
        if not isinstance(capabilities, list) or not capabilities:
            continue
        kernel_id = str(doc.get("id", "")).strip()
        operation = str(doc.get("op", "")).strip()
        function = str((doc.get("impl") or {}).get("function", "")).strip()
        if not kernel_id or not operation or not function:
            raise hard_fault(
                "numerical kernel map has incomplete identity",
                f"file={path}, id={kernel_id!r}, op={operation!r}, function={function!r}",
                "declare id, op, and impl.function before advertising numerical capabilities.",
            )
        if kernel_id in kernels:
            raise hard_fault(
                f"duplicate kernel ID {kernel_id!r}",
                f"second provider={path}",
                "give every exact implementation a unique stable kernel ID.",
            )
        checked = []
        for capability in capabilities:
            contract_id = str((capability or {}).get("contract_id", ""))
            contract = registry["contracts"].get(contract_id)
            if contract is None:
                raise hard_fault(
                    f"kernel {kernel_id!r} advertises unknown contract {contract_id!r}",
                    f"source={path}",
                    "register the full contract before binding an implementation.",
                )
            if capability.get("function") != function:
                raise hard_fault(
                    f"kernel {kernel_id!r} capability changes function identity",
                    f"impl.function={function!r}, capability.function={capability.get('function')!r}",
                    "bind the capability to the exact public function in the kernel map.",
                )
            _validate_capability_against_contract(kernel_id, capability, contract)
            checked.append(copy.deepcopy(capability))
        kernels[kernel_id] = {
            "id": kernel_id,
            "op": operation,
            "function": function,
            "capabilities": checked,
            "source": str(path.resolve().relative_to(REPO_ROOT.resolve())),
            "source_hash": sha256_file(path),
        }
    return {
        "schema": "cke.numerical_kernel_capabilities",
        "schema_version": 1,
        "engine_contract_version": "8",
        "kernels": kernels,
    }


def resolve_contract(
    circuit: Dict[str, Any],
    contracts: Dict[str, Any],
    kernels: Dict[str, Any],
    operation: str,
    phase: str,
    mode: str = "bringup",
    source_circuit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    validate_contract_registry(contracts)
    validate_schema(
        {"required_numerical_contracts": circuit.get("required_numerical_contracts")},
        REQUIREMENTS_SCHEMA,
        "circuit numerical requirements",
    )
    if mode not in {"bringup", "production"}:
        raise ContractError(f"Unknown resolution mode: {mode}")
    operation_doc = circuit["required_numerical_contracts"].get(operation)
    if not isinstance(operation_doc, dict):
        raise hard_fault(
            f"circuit has no numerical operation {operation!r}",
            f"circuit={circuit.get('name', '<unnamed>')}",
            "declare the operation and its exact semantic contract.",
        )
    request = (operation_doc.get("phases") or {}).get(phase)
    if not isinstance(request, dict):
        raise hard_fault(
            f"operation {operation!r} has no {phase!r} numerical requirement",
            "Prefill and decode may use different arithmetic contracts.",
            "declare the active phase explicitly.",
        )
    contract_id = request["contract_id"]
    if contract_id.lower() in AMBIGUOUS_IDS:
        raise hard_fault(
            f"ambiguous requested contract {contract_id!r}",
            "The compiler cannot infer execution arithmetic from a dtype label.",
            "request one complete registry contract by stable ID.",
        )
    contract = contracts["contracts"].get(contract_id)
    if contract is None:
        raise hard_fault(
            f"unknown requested contract {contract_id!r}",
            f"operation={operation}.{phase}",
            "register and validate the contract before compiling the circuit.",
        )
    matches = []
    for kernel in kernels.get("kernels", {}).values():
        if kernel.get("op") != operation_doc["op"]:
            continue
        for capability in kernel.get("capabilities", []):
            if capability["contract_id"] == contract_id and phase in capability["phases"]:
                matches.append((kernel, capability))
    if len(matches) != 1:
        raise hard_fault(
            f"numerical requirement resolved to {len(matches)} kernels",
            f"operation={operation}.{phase}, contract={contract_id}, candidates={[item[0]['id'] for item in matches]}",
            "bind exactly one explicit kernel implementation; remove ambiguity or add the missing provider.",
        )
    kernel, capability = matches[0]
    if mode == "production" and any(
        state != "validated"
        for state in (request["validation"], contract["status"], capability["status"])
    ):
        raise hard_fault(
            f"production resolution uses unvalidated contract {contract_id!r}",
            f"request={request['validation']}, contract={contract['status']}, kernel={capability['status']}",
            "produce parity evidence and promote every state to validated.",
        )
    source_hashes = {
        "contract_registry": sha256_json(contracts),
        "kernel_map": kernel["source_hash"],
        "circuit": sha256_json(circuit),
    }
    if source_circuit_path is not None:
        source_hashes["circuit_file"] = sha256_file(source_circuit_path)
    result = {
        "schema": "cke.resolved_numerical_execution_contract",
        "schema_version": 1,
        "engine_contract_version": "8",
        "circuit": str(circuit.get("name") or "embedded"),
        "operation": operation,
        "template_ops": copy.deepcopy(operation_doc["template_ops"]),
        "phase": phase,
        "mode": mode,
        "requirements": copy.deepcopy(request),
        "contract": {"id": contract_id, "status": contract["status"], "semantics": copy.deepcopy(contract)},
        "kernel": {
            "id": kernel["id"],
            "function": kernel["function"],
            "status": capability["status"],
            "explicit_selector": True,
        },
        "implementation": copy.deepcopy(capability["implementation"]),
        "checkpoint": copy.deepcopy(operation_doc["checkpoint"]),
        "source_hashes": source_hashes,
    }
    validate_schema(result, RESOLVED_SCHEMA, "resolved numerical execution contract")
    return result
