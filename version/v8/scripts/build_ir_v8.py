#!/usr/bin/env python3
"""
build_ir_v8.py - Complete IR Pipeline: Circuit + Quant + Contracts → GraphIR → Lowering

PIPELINE (4 stages):
    1. GraphIR Generation: Circuit requirements + quant summary → resolved kernel IDs
    2. Fusion Pass: Combine consecutive kernels using registry-driven patterns
    3. Memory Layout: Plan activation buffers and weight offsets
    4. Output: IR1 JSON + Memory Layout JSON

Stage 1 - IR1 Generation (Direct mapping, no intermediate abstractions):
    1. Parse circuit sequence and required contracts (what math to run)
    2. Read quant summary from manifest (what dtypes for weights)
    3. Resolve contract-bearing ops, then map remaining circuit ops to kernel IDs
    4. Return: List of kernel function names

Stage 2 - Fusion Pass:
    1. Scan kernel registry for kernels with "fuses" field
    2. Match consecutive kernel sequences in IR1
    3. Replace matching sequences with fused kernels
    4. Return: Optimized kernel list + fusion statistics

Stage 3 - Memory Layout:
    1. Calculate activation buffer sizes (based on mode: decode vs prefill)
    2. Plan weight memory layout with explicit offsets
    3. Generate buffer allocation map
    4. Return: Complete memory layout with offsets

REQUIREMENTS:
    1. weights_manifest.json with template and quant_summary
    2. KERNEL_REGISTRY.json

USAGE:
    # Generate IR1 only
    python build_ir_v8.py --manifest=/path/to/weights_manifest.json \\
        --mode=decode --output=ir1_decode.json

    # Generate full pipeline (IR1 + Fusion + Layout)
    python build_ir_v8.py --manifest=/path/to/weights_manifest.json \\
        --mode=decode --output=ir1_decode.json --layout-output=layout_decode.json

OUTPUTS:
    - IR1 JSON: Simple kernel sequence (before fusion)
    - Layout JSON: Fused kernels + memory layout with explicit offsets

LOWERING CONTRACT:
    - The builder must stay model-family agnostic.
    - Circuits declare operations, graph structure, stitch points, and semantics.
    - The resolver selects contract-bearing providers before GraphIR construction.
    - The lowerer consumes resolved providers and may not reselect them.
    - Do not teach the lowerer model names such as MoE, DeepStack, SSM, etc.
    - If a model needs branching, routing, collect, or stitch behavior, that
      contract belongs in the template as explicit operations or graph edges.
"""

import argparse
import copy
import fnmatch
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from jsonschema import Draft202012Validator

# ANSI colors for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

# Import memory planner
from memory_planner_v8 import plan_memory, MemoryPlanner


def _load_numerical_contract_resolver():
    path = Path(__file__).resolve().parent / "resolve_attention_contracts_v8.py"
    spec = importlib.util.spec_from_file_location("resolve_attention_contracts_v8", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load numerical contract resolver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_execution_contract_resolver():
    path = Path(__file__).resolve().parent / "resolve_numerical_execution_contracts_v8.py"
    spec = importlib.util.spec_from_file_location("resolve_numerical_execution_contracts_v8", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load execution contract resolver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config_value(config: Dict[str, Any], dotted_key: str) -> Tuple[bool, Any]:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _contract_selector_matches(selector: Any, config: Dict[str, Any], operation: str) -> bool:
    if selector is None:
        return True
    if not isinstance(selector, dict) or not selector:
        raise RuntimeError(
            f"Numerical contract {operation!r} has an invalid selector. "
            "Declare selector.config_equals and/or selector.config_not_equals."
        )
    supported = {"config_equals", "config_not_equals"}
    unknown = set(selector) - supported
    if unknown:
        raise RuntimeError(
            f"Numerical contract {operation!r} selector has unsupported keys: "
            f"{sorted(unknown)}"
        )
    predicates = 0
    for key, expected in (selector.get("config_equals") or {}).items():
        predicates += 1
        found, current = _config_value(config, str(key))
        if not found or current != expected:
            return False
    for key, rejected in (selector.get("config_not_equals") or {}).items():
        predicates += 1
        found, current = _config_value(config, str(key))
        if found and current == rejected:
            return False
    if predicates == 0:
        raise RuntimeError(
            f"Numerical contract {operation!r} has an invalid selector. "
            "Declare at least one configuration predicate."
        )
    return True


def _resolve_manifest_numerical_contracts(
    manifest: Dict[str, Any],
    mode: str,
) -> List[Dict[str, Any]]:
    circuit = manifest.get("template") if isinstance(manifest.get("template"), dict) else {}
    required = circuit.get("required_contracts")
    if not isinstance(required, dict) or not required:
        return []

    resolver = _load_numerical_contract_resolver()
    contracts = resolver.load_json(resolver.DEFAULT_CONTRACTS)
    kernels = resolver.load_kernel_capabilities()
    circuit_name = str(circuit.get("name", "")).strip()
    source_path = V8_ROOT / "circuits" / f"{circuit_name}.json" if circuit_name else None
    plans: List[Dict[str, Any]] = []
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    for operation, operation_doc in required.items():
        selector = operation_doc.get("selector") if isinstance(operation_doc, dict) else None
        if not _contract_selector_matches(selector, config, str(operation)):
            continue
        phases = operation_doc.get("phases") if isinstance(operation_doc, dict) else None
        if not isinstance(phases, dict) or mode not in phases:
            continue
        try:
            plan = resolver.resolve_contract(
                circuit,
                contracts,
                kernels,
                operation=str(operation),
                phase=mode,
                mode="bringup",
                source_circuit_path=source_path if source_path and source_path.is_file() else None,
            )
        except resolver.ContractError as exc:
            raise RuntimeError(
                f"Numerical contract resolution failed for {circuit_name or '<embedded>'} "
                f"{operation}.{mode}: {exc}"
            ) from exc
        plans.append(plan)
    return plans


def _resolve_manifest_execution_contracts(
    manifest: Dict[str, Any],
    mode: str,
) -> List[Dict[str, Any]]:
    circuit = manifest.get("template") if isinstance(manifest.get("template"), dict) else {}
    required = circuit.get("required_numerical_contracts")
    if not isinstance(required, dict) or not required:
        return []

    resolver = _load_execution_contract_resolver()
    contracts = resolver.load_json(resolver.DEFAULT_CONTRACTS)
    kernels = resolver.load_kernel_capabilities(contracts=contracts)
    circuit_name = str(circuit.get("name", "")).strip()
    source_path = V8_ROOT / "circuits" / f"{circuit_name}.json" if circuit_name else None
    resolution_mode = str((manifest.get("config") or {}).get("numerical_contract_mode", "bringup"))
    plans: List[Dict[str, Any]] = []
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    for operation, operation_doc in required.items():
        selector = operation_doc.get("selector") if isinstance(operation_doc, dict) else None
        if not _contract_selector_matches(selector, config, str(operation)):
            continue
        phases = operation_doc.get("phases") if isinstance(operation_doc, dict) else None
        if not isinstance(phases, dict) or mode not in phases:
            continue
        try:
            plan = resolver.resolve_contract(
                circuit,
                contracts,
                kernels,
                operation=str(operation),
                phase=mode,
                mode=resolution_mode,
                source_circuit_path=source_path if source_path and source_path.is_file() else None,
            )
        except resolver.ContractError as exc:
            raise RuntimeError(
                f"Numerical execution contract resolution failed for "
                f"{circuit_name or '<embedded>'} {operation}.{mode}: {exc}"
            ) from exc
        plans.append(plan)
    return plans


def _load_kernel_execution_capabilities() -> Dict[str, Dict[str, Any]]:
    resolver = _load_numerical_contract_resolver()
    document = resolver.load_kernel_execution_capabilities()
    kernels = document.get("kernels")
    if not isinstance(kernels, dict):
        raise RuntimeError("HARD CONTRACT FAULT: kernel execution capability registry is malformed.")
    return kernels


def _graph_ir_execution_metadata(capability: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        "schema": "cke.graph_ir_execution_contract",
        "schema_version": 1,
        "kernel_id": capability["id"],
        "op": capability["op"],
        "implementation": copy.deepcopy(capability["implementation"]),
    }
    for key in ("numerical_contract", "reference", "production"):
        if capability.get(key) is not None:
            metadata[key] = copy.deepcopy(capability[key])
    return metadata


def _validate_resolved_kernels_are_emitted(
    plans: List[Dict[str, Any]],
    arranged_kernels: List[Dict[str, Any]],
) -> None:
    for plan in plans:
        selected = str((plan.get("kernel") or {}).get("id", ""))
        template_ops = {str(item) for item in plan.get("template_ops", [])}
        governed = [item for item in arranged_kernels if str(item.get("op", "")) in template_ops]
        if not governed:
            raise RuntimeError(
                f"HARD CONTRACT FAULT: resolved operation {plan.get('operation')}.{plan.get('phase')} "
                f"governs {sorted(template_ops)}, but GraphIR emitted none of those operations. "
                "Fix the circuit operation binding; do not bypass contract validation."
            )
        for item in governed:
            emitted = str(item.get("kernel", ""))
            resolved = item.get("resolved_contract") if isinstance(item.get("resolved_contract"), dict) else {}
            if emitted != selected or resolved.get("kernel_id") != selected:
                raise RuntimeError(
                    f"HARD CONTRACT FAULT: GraphIR operation {item.get('op')} selected {emitted!r}, "
                    f"but authoritative resolution requires {selected!r} for "
                    f"{plan.get('operation')}.{plan.get('phase')}. Fix the circuit or kernel map; "
                    "do not add a fallback or validation bypass."
                )


def _index_numerical_contract_plans(
    plans: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for plan in plans:
        for template_op in plan.get("template_ops", []):
            op = str(template_op).strip()
            if not op:
                continue
            existing = indexed.get(op)
            if existing is not None:
                raise RuntimeError(
                    f"HARD CONTRACT FAULT: template op {op!r} is governed by both "
                    f"{existing.get('operation')!r} and {plan.get('operation')!r}. "
                    "Make circuit contract bindings unique; do not rely on declaration order."
                )
            indexed[op] = plan
    return indexed


def _graph_ir_contract_metadata(plan: Dict[str, Any]) -> Dict[str, Any]:
    contract = plan.get("contract") if isinstance(plan.get("contract"), dict) else plan["reduction"]
    metadata = {
        "schema": "cke.graph_ir_numerical_contract",
        "schema_version": 1,
        "operation": plan["operation"],
        "phase": plan["phase"],
        "required_contract_id": plan["requirements"].get("contract_id"),
        "resolved_contract_id": contract["id"],
        "contract_id": contract["id"],
        "kernel_id": plan["kernel"]["id"],
        "function": plan["kernel"]["function"],
        "implementation": copy.deepcopy(plan["implementation"]),
    }
    if isinstance(contract.get("semantics"), dict):
        metadata["semantics"] = copy.deepcopy(contract["semantics"])
    if isinstance(plan.get("checkpoint"), dict):
        metadata["checkpoint"] = copy.deepcopy(plan["checkpoint"])
    return metadata


def _is_vision_mrope_operation(operation: Dict[str, Any]) -> bool:
    """Classify vision M-RoPE from circuit/contract semantics, never kernel names."""
    resolved = operation.get("resolved_contract")
    semantics = resolved.get("semantics") if isinstance(resolved, dict) else None
    position = semantics.get("position_transform") if isinstance(semantics, dict) else None
    if isinstance(semantics, dict):
        return bool(
            semantics.get("operator_family") == "vision_mrope"
            and isinstance(position, dict)
            and position.get("pairing") == "multi_section"
            and int(position.get("position_rank", 0) or 0) == 4
        )

    op_type = str(operation.get("op", ""))
    if op_type == "mrope_qk":
        return True
    if op_type != "rope_qk":
        return False

    params = operation.get("params") if isinstance(operation.get("params"), dict) else {}
    return str(params.get("rope_mode", "")).strip().lower() == "vision"


def _is_text_mrope_operation(operation: Dict[str, Any]) -> bool:
    """Classify text M-RoPE from resolved contract semantics."""
    resolved = operation.get("resolved_contract")
    semantics = resolved.get("semantics") if isinstance(resolved, dict) else None
    position = semantics.get("position_transform") if isinstance(semantics, dict) else None
    if isinstance(semantics, dict):
        return bool(
            semantics.get("operator_family") == "text_mrope"
            and isinstance(position, dict)
            and position.get("pairing") == "multi_section"
        )

    if str(operation.get("op", "")) != "rope_qk":
        return False
    params = operation.get("params") if isinstance(operation.get("params"), dict) else {}
    return str(params.get("rope_mode", "")).strip().lower() == "text_mrope"


def _attach_semantic_checkpoints(
    template: Dict[str, Any],
    arranged_kernels: List[Dict[str, Any]],
    registry: Dict[str, Any],
) -> None:
    contract = template.get("semantic_checkpoints")
    if not isinstance(contract, dict):
        return
    schema_path = V8_ROOT / "schemas" / "semantic_checkpoint_contract.schema.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(contract),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise RuntimeError(
            f"HARD CHECKPOINT ABI FAULT: semantic checkpoint contract is invalid at "
            f"{location}: {error.message}"
        )

    kernel_functions = {
        str(item.get("id", "")): str((item.get("impl") or {}).get("function", ""))
        for item in registry.get("kernels", [])
        if isinstance(item, dict)
    }
    matched: Dict[str, int] = {name: 0 for name in contract["exports"]}
    seen_ids: set[str] = set()
    for arranged in arranged_kernels:
        section = str(arranged.get("section", ""))
        template_op_id = str(arranged.get("template_op_id", ""))
        op = str(arranged.get("op", ""))
        layer = int(arranged.get("layer", -1))
        for declaration_name, declaration in contract["exports"].items():
            if (
                declaration["section"] != section
                or declaration["template_op_id"] != template_op_id
                or declaration["op"] != op
            ):
                continue
            checkpoints = []
            for checkpoint in declaration["checkpoints"]:
                checkpoint_id = str(checkpoint["id"]).replace("{layer}", str(layer))
                if "{layer}" in checkpoint_id or (section == "body" and layer < 0):
                    raise RuntimeError(
                        f"HARD CHECKPOINT ABI FAULT: cannot resolve layer for {checkpoint['id']!r}"
                    )
                if checkpoint_id in seen_ids:
                    raise RuntimeError(
                        f"HARD CHECKPOINT ABI FAULT: duplicate semantic checkpoint {checkpoint_id!r}"
                    )
                seen_ids.add(checkpoint_id)
                item = copy.deepcopy(checkpoint)
                item["id"] = checkpoint_id
                item["phase"] = "prefill" if section in {"header", "body", "footer", "branch"} else section
                item["layer"] = layer
                item["kernel_id"] = str(arranged.get("kernel", ""))
                item["function"] = kernel_functions.get(item["kernel_id"], "")
                if not item["function"]:
                    raise RuntimeError(
                        f"HARD CHECKPOINT ABI FAULT: kernel {item['kernel_id']!r} for "
                        f"{checkpoint_id!r} has no exact public function"
                    )
                resolved = arranged.get("resolved_contract")
                item["resolved_contract_id"] = (
                    str(resolved.get("resolved_contract_id") or resolved.get("contract_id"))
                    if isinstance(resolved, dict)
                    else "unresolved"
                )
                checkpoints.append(item)
            arranged["semantic_checkpoints"] = checkpoints
            matched[declaration_name] += 1

    missing = sorted(name for name, count in matched.items() if count == 0)
    if missing:
        raise RuntimeError(
            "HARD CHECKPOINT ABI FAULT: checkpoint declarations did not bind to generated "
            f"operations: {missing}. Fix section/template_op_id/op; do not add exporter aliases."
        )


def _entry_offset(entry: Dict[str, Any]) -> int:
    """Read manifest offset, accepting both file_offset (v7) and offset (tiny train init)."""
    try:
        return int(entry.get("file_offset", entry.get("offset", 0)) or 0)
    except Exception:
        return 0


def _entry_size(entry: Dict[str, Any]) -> int:
    try:
        return int(entry.get("size", entry.get("size_bytes", 0)) or 0)
    except Exception:
        return 0


def _c_string_literal(text: str) -> str:
    return json.dumps(str(text))


def _collect_chat_marker_strings(chat_contract: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(chat_contract, dict):
        return []

    out: List[str] = []
    seen: set[str] = set()

    for field in ("template_markers", "token_stop_markers", "stop_text_markers"):
        values = chat_contract.get(field)
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    return None


def _has_untied_lm_head_weight(weight_index: Dict[str, Dict[str, Any]]) -> bool:
    return any(
        key in weight_index
        for key in ("output.weight", "lm_head.weight", "lm_head_weight", "lm_head")
    )


def _resolve_logits_weight_source(
    config: Dict[str, Any],
    weight_index: Dict[str, Dict[str, Any]],
) -> str:
    """
    Decide logits weight source for this manifest.

    Returns:
      - "lm_head": untied head (output/lm_head weight must be used)
      - "token_emb": tied head (token embedding shared)

    Rules:
      - tie_word_embeddings=false -> strict untied (must not fallback to token_emb)
      - tie_word_embeddings unknown + untied head present -> treat as untied
      - otherwise -> tied path
    """
    tie_cfg = _coerce_bool(config.get("tie_word_embeddings"))
    has_untied = _has_untied_lm_head_weight(weight_index)

    if tie_cfg is False:
        if not has_untied:
            raise RuntimeError(
                "Logits contract failed: tie_word_embeddings=false but no output/lm_head weight exists in manifest. "
                "Fix conversion/template contract before lowering."
            )
        return "lm_head"

    if tie_cfg is None and has_untied:
        return "lm_head"

    return "token_emb"


def _load_builtin_template_doc(template_name: Optional[str]) -> Optional[Dict[str, Any]]:
    name = str(template_name or "").strip().lower()
    if not name:
        return None
    path = V8_ROOT / "circuits" / f"{name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    _raise_on_forbidden_template_metadata(doc, source=str(path))
    return doc


def _merge_template_defaults(
    default_doc: Dict[str, Any],
    override_doc: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = copy.deepcopy(default_doc)
    if not isinstance(override_doc, dict):
        return merged
    for key, value in override_doc.items():
        if value is None:
            continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_template_defaults(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


_FORBIDDEN_TEMPLATE_FLAG_KEYS: Dict[str, str] = {
    "activation_preference_by_op": "per-op activation dtype policy",
    "prefer_fp32_mlp_matmuls": "MLP activation dtype policy",
    "prefer_q8_0_contract": "quantized activation contract policy",
    "prefer_fp32_logits": "logits activation dtype policy",
}

_FORBIDDEN_TEMPLATE_KEY_NAMES: Dict[str, str] = {
    "dtype": "tensor dtype metadata",
    "weight_dtype": "weight dtype metadata",
    "weight_dtypes": "weight dtype metadata",
    "weight_quant": "weight quant metadata",
    "weight_quant_type": "weight quant metadata",
    "quant_type_by_op": "per-op quant metadata",
}


def _apply_circuit_runtime_defaults(
    config: Dict[str, Any],
    circuit: Optional[Dict[str, Any]],
    *,
    source: str,
) -> Dict[str, Any]:
    contract = circuit.get("contract") if isinstance(circuit, dict) else None
    defaults = contract.get("runtime_defaults") if isinstance(contract, dict) else None
    if defaults is None:
        return dict(config)
    schema_path = V8_ROOT / "schemas" / "circuit_runtime_defaults.schema.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(defaults),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise RuntimeError(
            f"HARD CIRCUIT DEFAULT FAULT: {source} runtime_defaults invalid at "
            f"{location}: {error.message}"
        )

    merged = copy.deepcopy(config)
    for key, value in defaults.items():
        if isinstance(value, dict):
            current = merged.get(key)
            current = dict(current) if isinstance(current, dict) else {}
            for item_key, item_value in value.items():
                current.setdefault(item_key, copy.deepcopy(item_value))
            merged.setdefault(key, current)
            if isinstance(merged.get(key), dict):
                for item_key, item_value in current.items():
                    merged[key].setdefault(item_key, item_value)
        else:
            merged.setdefault(key, copy.deepcopy(value))
    return merged


def _validated_circuit_weight_policy(template: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    contract = template.get("contract") if isinstance(template.get("contract"), dict) else {}
    policy = contract.get("weight_policy") if isinstance(contract, dict) else None
    if policy is None:
        return None
    schema_path = V8_ROOT / "schemas" / "circuit_weight_policy.schema.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(policy),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise RuntimeError(
            f"HARD CIRCUIT WEIGHT POLICY FAULT at {location}: {error.message}. "
            "Fix the circuit; do not add a family branch in lowering."
        )
    return policy


def _ignored_manifest_weights(
    template: Dict[str, Any], config: Dict[str, Any], model_weights: set[str]
) -> Dict[str, str]:
    """Resolve circuit-declared ignored weights; undeclared extras remain hard faults."""
    policy = _validated_circuit_weight_policy(template)
    if policy is None:
        return {}

    ignored: Dict[str, str] = {}
    for rule in policy.get("ignore", []):
        condition = rule.get("when")
        if isinstance(condition, dict):
            current: Any = config
            for part in str(condition["config_key"]).split("."):
                if not isinstance(current, dict) or part not in current:
                    current = None
                    break
                current = current[part]
            if current != condition["equals"]:
                continue
        matches = sorted(name for name in model_weights if fnmatch.fnmatchcase(name, rule["pattern"]))
        for name in matches:
            if name in ignored:
                raise RuntimeError(
                    f"HARD CIRCUIT WEIGHT POLICY FAULT: {name!r} matches multiple ignore rules."
                )
            ignored[name] = str(rule["reason"])
    return ignored


def _circuit_op_weight_keys(
    template: Dict[str, Any], section: str, op: str
) -> Optional[List[str]]:
    policy = _validated_circuit_weight_policy(template)
    if policy is None:
        return None
    matches = [
        rule
        for rule in policy.get("op_bindings", [])
        if rule.get("section") == section and rule.get("op") == op
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"HARD CIRCUIT WEIGHT POLICY FAULT: multiple bindings for {section}.{op}."
        )
    return list(matches[0]["weights"]) if matches else None


def _collect_forbidden_template_metadata(
    node: Any,
    path: Tuple[str, ...] = (),
) -> List[Tuple[str, str]]:
    issues: List[Tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            key_str = str(key)
            next_path = path + (key_str,)
            if len(next_path) == 2 and next_path[0] == "flags" and key_str in _FORBIDDEN_TEMPLATE_FLAG_KEYS:
                issues.append((".".join(next_path), _FORBIDDEN_TEMPLATE_FLAG_KEYS[key_str]))
            if key_str in _FORBIDDEN_TEMPLATE_KEY_NAMES:
                issues.append((".".join(next_path), _FORBIDDEN_TEMPLATE_KEY_NAMES[key_str]))
            issues.extend(_collect_forbidden_template_metadata(value, next_path))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            issues.extend(_collect_forbidden_template_metadata(item, path + (f"[{idx}]",)))
    return issues


def _raise_on_forbidden_template_metadata(
    template_doc: Optional[Dict[str, Any]],
    *,
    source: str,
) -> None:
    if not isinstance(template_doc, dict):
        return
    issues = _collect_forbidden_template_metadata(template_doc)
    if not issues:
        return
    details = "; ".join(f"{path} ({reason})" for path, reason in issues)
    raise RuntimeError(
        f"Template '{source}' declares dtype/quant policy that must not live in templates: {details}. "
        "Keep templates to graph/order metadata; derive weight dtypes from the weights manifest and runtime config."
    )


def _hydrate_manifest_template(manifest: Dict[str, Any]) -> Dict[str, Any]:
    template_doc = manifest.get("template") if isinstance(manifest.get("template"), dict) else None
    cfg = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    template_name = ""
    if isinstance(template_doc, dict):
        template_name = str(template_doc.get("name", "") or "").strip().lower()
    if not template_name:
        template_name = str(cfg.get("model", "") or "").strip().lower()
    built_in = _load_builtin_template_doc(template_name)
    if built_in and isinstance(template_doc, dict):
        manifest["template"] = _merge_template_defaults(built_in, template_doc)
    elif built_in:
        manifest["template"] = copy.deepcopy(built_in)
    _validate_segmented_prefill_contract(
        manifest.get("template") if isinstance(manifest.get("template"), dict) else None,
        source=f"manifest:{template_name or '<embedded>'}",
    )
    _raise_on_forbidden_template_metadata(
        manifest.get("template") if isinstance(manifest.get("template"), dict) else None,
        source=f"manifest:{template_name or '<embedded>'}",
    )
    hydrated_config = _apply_circuit_runtime_defaults(
        dict(cfg),
        manifest.get("template") if isinstance(manifest.get("template"), dict) else None,
        source=f"manifest:{template_name or '<embedded>'}",
    )
    hydrated_template = manifest.get("template") if isinstance(manifest.get("template"), dict) else {}
    hydrated_contract = hydrated_template.get("contract") if isinstance(hydrated_template.get("contract"), dict) else {}
    bridge_contract = hydrated_contract.get("multimodal_bridge")
    if isinstance(bridge_contract, dict):
        hydrated_config["multimodal_bridge_contract"] = copy.deepcopy(bridge_contract)
    manifest["config"] = hydrated_config
    return manifest


def _validate_segmented_prefill_contract(
    circuit: Optional[Dict[str, Any]],
    *,
    source: str,
) -> None:
    if not isinstance(circuit, dict):
        return
    contract = circuit.get("contract") if isinstance(circuit.get("contract"), dict) else {}
    bridge = contract.get("multimodal_bridge") if isinstance(contract.get("multimodal_bridge"), dict) else {}
    schedule = bridge.get("prefill_schedule") if isinstance(bridge.get("prefill_schedule"), dict) else {}
    schedule_is_segmented = bool(schedule) and (
        schedule.get("segments") == ["text_before", "visual", "text_after"]
        and schedule.get("cache_transition") == "append_preserve"
        and schedule.get("position_transition") == "segment_defined"
    )
    if schedule and not schedule_is_segmented:
        raise RuntimeError(
            "HARD CONTRACT FAULT: unsupported mixed-prefill schedule in "
            f"{source}. The segmented append contract requires text_before, visual, text_after; "
            "append_preserve; and segment_defined."
        )

    required = circuit.get("required_contracts")
    required = required if isinstance(required, dict) else {}
    attention = required.get("decoder.attention")
    attention = attention if isinstance(attention, dict) else {}
    phases = attention.get("phases") if isinstance(attention.get("phases"), dict) else {}
    prefill = phases.get("prefill") if isinstance(phases.get("prefill"), dict) else {}
    requirements = prefill.get("requires") if isinstance(prefill.get("requires"), dict) else {}
    attention_is_segmented = requirements.get("execution.prefill_batching") == "segmented_append"

    if schedule_is_segmented != attention_is_segmented:
        raise RuntimeError(
            "HARD CONTRACT FAULT: segmented mixed-prefill schedule and attention provider disagree "
            f"in {source}. Declare both multimodal_bridge.prefill_schedule.cache_transition="
            "append_preserve and decoder.attention.prefill execution.prefill_batching="
            "segmented_append, or declare neither."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DATAFLOW DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
# Each op type defines:
#   - inputs: {input_name: slot_name} - which logical slot this input reads from
#   - outputs: {output_name: slot_name} - which logical slot this output writes to
#   - dtype: output dtype (fp32, q8_0, q8_k, etc.)
#
# Slot names are logical (not physical buffers):
#   - "main_stream"     : Primary activation stream (fp32)
#   - "main_stream_q8"  : Quantized activation stream (q8_0 or q8_k)
#   - "residual"        : Saved residual for skip connection
#   - "q_scratch"       : Q projection output
#   - "k_scratch"       : K projection output
#   - "v_scratch"       : V projection output
#   - "recurrent_*"     : Recurrent packed/split intermediate slots
#   - "attn_scratch"    : Attention output
#   - "mlp_scratch"     : MLP gate_up output
#   - "branch_stream"   : Branch-local merged token stream (fp32)
#   - "branch_normed"   : Branch-local normalized stream (fp32)
#   - "branch_mlp"      : Branch-local MLP scratch (fp32)
#   - "branch_collect"  : Collected branch outputs awaiting stitch
#   - "vision_output"   : Final stitched vision embedding output
#   - "vision_positions": Vision-side position IDs / route metadata (i32)
#   - "kv_cache"        : KV cache (persistent across tokens)
#   - "external:X"      : External input (token_ids, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

OP_DATAFLOW = {
    # Header ops
    "dense_embedding_lookup": {
        "inputs": {"token_ids": "external:token_ids"},
        "outputs": {"out": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "assistant_pre_projection": {
        "inputs": {"x": "backbone_stream"},
        "outputs": {"y": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "gemma4_per_layer_prepare": {
        "inputs": {"input": "main_stream", "tokens": "external:token_ids"},
        "outputs": {"per_layer_input": {"slot": "gemma4_per_layer_stream", "dtype": "fp32"}},
    },
    "patchify": {
        "inputs": {"image": "external:image_input"},
        "outputs": {"patches": {"slot": "patch_scratch", "dtype": "fp32"}},
    },
    "patch_proj": {
        "inputs": {"x": "patch_scratch"},
        "outputs": {"y": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "patch_proj_aux": {
        "inputs": {"x": "patch_scratch"},
        "outputs": {"y": {"slot": "mlp_scratch", "dtype": "fp32"}},
    },
    "add_stream": {
        "inputs": {
            "a": "main_stream",
            "b": "mlp_scratch",
        },
        "outputs": {"out": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "position_embeddings": {
        "inputs": {"x": "main_stream"},
        "outputs": {"x": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "vision_position_ids": {
        "inputs": {},
        "outputs": {"positions": {"slot": "vision_positions", "dtype": "i32"}},
    },
    "position_ids_2d": {
        "inputs": {},
        "outputs": {"positions": {"slot": "vision_positions", "dtype": "i32"}},
    },
    "patch_bias_add": {
        "inputs": {"x": "main_stream"},
        "outputs": {"y": {"slot": "main_stream", "dtype": "fp32"}},
    },

    # Attention block
    "rmsnorm": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "layernorm": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "attn_norm": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "block_rmsnorm": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "layer_input", "dtype": "fp32"}},
    },
    "post_attention_norm": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "ffn_norm": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "post_ffn_norm": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "gemma4_per_layer_embed": {
        "inputs": {"hidden": "main_stream", "per_layer_input": "gemma4_per_layer_stream"},
        "outputs": {"hidden": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "assistant_layer_scale": {
        "inputs": {"hidden": "main_stream"},
        "outputs": {"hidden": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "v_norm": {
        "inputs": {"input": "v_scratch"},
        "outputs": {"output": {"slot": "v_scratch", "dtype": "fp32"}},
    },
    "final_rmsnorm": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "assistant_post_projection": {
        "inputs": {"x": "main_stream"},
        "outputs": {"y": {"slot": "backbone_stream", "dtype": "fp32"}},
    },
    "final_logit_softcap": {
        "inputs": {"logits": "logits"},
        "outputs": {"logits": {"slot": "logits", "dtype": "fp32"}},
    },
    "quantize_input_0": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream_q8", "dtype": "q8_0"}},
    },
    "quantize_input_1": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream_q8", "dtype": "q8_0"}},
    },
    "quantize_input_2": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream_q8", "dtype": "q8_0"}},
    },
    "residual_save": {
        "inputs": {"src": "main_stream"},
        "outputs": {"dst": {"slot": "residual", "dtype": "fp32"}},
    },
    "q_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "q_scratch", "dtype": "fp32"}},
    },
    "qkv_packed_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "qkv_packed", "dtype": "fp32"}},
    },
    "q_gate_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "attn_q_gate_packed", "dtype": "fp32"}},
    },
    "k_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "k_scratch", "dtype": "fp32"}},
    },
    "v_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "v_scratch", "dtype": "fp32"}},
    },
    "recurrent_qkv_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "recurrent_qkv_packed", "dtype": "fp32"}},
    },
    "recurrent_gate_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "recurrent_z", "dtype": "fp32"}},
    },
    "recurrent_alpha_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "recurrent_alpha", "dtype": "fp32"}},
    },
    "recurrent_beta_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "recurrent_beta", "dtype": "fp32"}},
    },
    "recurrent_split_qkv": {
        "inputs": {"packed_qkv": "recurrent_qkv_packed"},
        "outputs": {
            "q": {"slot": "recurrent_q_preconv", "dtype": "fp32"},
            "k": {"slot": "recurrent_k_preconv", "dtype": "fp32"},
            "v": {"slot": "recurrent_v_preconv", "dtype": "fp32"},
        },
    },
    "split_qkv_packed": {
        "inputs": {"packed_qkv": "qkv_packed"},
        "outputs": {
            "q": {"slot": "q_scratch", "dtype": "fp32"},
            "k": {"slot": "k_scratch", "dtype": "fp32"},
            "v": {"slot": "v_scratch", "dtype": "fp32"}
        }
    },
    "split_q_gate": {
        "inputs": {"packed_qg": "attn_q_gate_packed"},
        "outputs": {
            "q": {"slot": "q_scratch", "dtype": "fp32"},
            "gate": {"slot": "attn_gate", "dtype": "fp32"},
        },
    },
    "recurrent_dt_gate": {
        "inputs": {"alpha": "recurrent_alpha"},
        "outputs": {"gate": {"slot": "recurrent_g", "dtype": "fp32"}},
    },
    "recurrent_conv_state_update": {
        "inputs": {
            "state_in": "external:recurrent_conv_state",
            "q": "recurrent_q_preconv",
            "k": "recurrent_k_preconv",
            "v": "recurrent_v_preconv",
        },
        "outputs": {
            "conv_x": {"slot": "recurrent_conv_input", "dtype": "fp32"},
            "state_out": {"slot": "recurrent_conv_state_out", "dtype": "fp32"},
        },
    },
    "recurrent_ssm_conv": {
        "inputs": {"conv_x": "recurrent_conv_input"},
        "outputs": {"out": {"slot": "recurrent_conv_qkv_raw", "dtype": "fp32"}},
    },
    "recurrent_silu": {
        "inputs": {"x": "recurrent_conv_qkv_raw"},
        "outputs": {"out": {"slot": "recurrent_conv_qkv", "dtype": "fp32"}},
    },
    "recurrent_split_conv_qkv": {
        "inputs": {"packed_qkv": "recurrent_conv_qkv"},
        "outputs": {
            "q": {"slot": "recurrent_q", "dtype": "fp32"},
            "k": {"slot": "recurrent_k", "dtype": "fp32"},
            "v": {"slot": "recurrent_v", "dtype": "fp32"},
        },
    },
    "recurrent_qk_l2_norm": {
        "inputs": {"q": "recurrent_q", "k": "recurrent_k"},
        "outputs": {
            "q": {"slot": "recurrent_q", "dtype": "fp32"},
            "k": {"slot": "recurrent_k", "dtype": "fp32"},
        },
    },
    "recurrent_core": {
        "inputs": {
            "q": "recurrent_q",
            "k": "recurrent_k",
            "v": "recurrent_v",
            "g": "recurrent_g",
            "beta": "recurrent_beta",
            "state_in": "external:recurrent_ssm_state",
        },
        "outputs": {
            "out": {"slot": "recurrent_attn_out", "dtype": "fp32"},
            "state_out": {"slot": "recurrent_ssm_state_out", "dtype": "fp32"},
        },
    },
    "recurrent_norm_gate": {
        "inputs": {"x": "recurrent_attn_out", "gate": "recurrent_z"},
        "outputs": {"out": {"slot": "recurrent_normed", "dtype": "fp32"}},
    },
    "bias_add_q": {
        "inputs": {"x": "q_scratch"},
        "outputs": {"x": {"slot": "q_scratch", "dtype": "fp32"}},
    },
    "bias_add_k": {
        "inputs": {"x": "k_scratch"},
        "outputs": {"x": {"slot": "k_scratch", "dtype": "fp32"}},
    },
    "bias_add_v": {
        "inputs": {"x": "v_scratch"},
        "outputs": {"x": {"slot": "v_scratch", "dtype": "fp32"}},
    },
    "qk_norm": {
        "inputs": {"q": "q_scratch", "k": "k_scratch"},
        "outputs": {
            "q": {"slot": "q_scratch", "dtype": "fp32"},
            "k": {"slot": "k_scratch", "dtype": "fp32"},
        },
    },
    "q_norm": {
        "inputs": {"q": "q_scratch"},
        "outputs": {"q": {"slot": "q_scratch", "dtype": "fp32"}},
    },
    "rope_qk": {
        "inputs": {"q": "q_scratch", "k": "k_scratch"},
        "outputs": {
            "q": {"slot": "q_scratch", "dtype": "fp32"},
            "k": {"slot": "k_scratch", "dtype": "fp32"},
        },
    },
    "rope_q": {
        "inputs": {"q": "q_scratch"},
        "outputs": {"q": {"slot": "q_scratch", "dtype": "fp32"}},
    },
    "mrope_qk": {
        "inputs": {"q": "q_scratch", "k": "k_scratch", "positions": "vision_positions"},
        "outputs": {
            "q": {"slot": "q_scratch", "dtype": "fp32"},
            "k": {"slot": "k_scratch", "dtype": "fp32"},
        },
    },
    "kv_cache_store": {
        "inputs": {"k": "k_scratch", "v": "v_scratch"},
        "outputs": {
            "k_cache": {"slot": "kv_cache", "dtype": "fp32"},
            "v_cache": {"slot": "kv_cache", "dtype": "fp32"},
        },
    },
    "kv_cache_store_shared_q": {
        "inputs": {"q": "q_scratch"},
        "outputs": {
            "k_cache": {"slot": "kv_cache", "dtype": "fp32"},
            "v_cache": {"slot": "kv_cache", "dtype": "fp32"},
        },
    },
    "attn": {
        "inputs": {"q": "q_scratch", "k": "kv_cache", "v": "kv_cache"},
        "outputs": {"out": {"slot": "attn_scratch", "dtype": "fp32"}},
    },
    "attn_sliding": {
        "inputs": {"q": "q_scratch", "k": "kv_cache", "v": "kv_cache"},
        "outputs": {"out": {"slot": "attn_scratch", "dtype": "fp32"}},
    },
    "attn_shared_kv": {
        "inputs": {"q": "q_scratch"},
        "outputs": {"out": {"slot": "attn_scratch", "dtype": "fp32"}},
    },
    "attn_sliding_shared_kv": {
        "inputs": {"q": "q_scratch"},
        "outputs": {"out": {"slot": "attn_scratch", "dtype": "fp32"}},
    },
    "attn_gate_sigmoid_mul": {
        "inputs": {"x": "attn_scratch", "gate": "attn_gate"},
        "outputs": {"out": {"slot": "attn_scratch", "dtype": "fp32"}},
    },
    "quantize_recurrent_out_proj_input": {
        "inputs": {"input": "recurrent_normed"},
        "outputs": {"output": {"slot": "main_stream_q8", "dtype": "q8_0"}},
    },
    "quantize_out_proj_input": {
        "inputs": {"input": "attn_scratch"},
        "outputs": {"output": {"slot": "main_stream_q8", "dtype": "q8_0"}},
    },
    "out_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "recurrent_out_proj": {
        # Recurrent output projection is part of the recurrent branch, not the
        # main-stream attention/MLP path. Keep the logical stitch contract
        # anchored to recurrent_normed here; if a selected kernel later needs a
        # quantized activation view, that remap must happen through the generic
        # kernel-activation override path rather than hard-coded family logic.
        "inputs": {"x": "recurrent_normed"},
        "outputs": {"y": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "bias_add": {
        "inputs": {"x": "main_stream"},
        "outputs": {"x": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "residual_add": {
        "inputs": {
            "a": "main_stream",   # Current stream (from out_proj/bias_add)
            "b": "residual",      # Saved residual
        },
        "outputs": {"out": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "moe_router": {
        "inputs": {"x": "layer_input"},
        "outputs": {"y": {"slot": "mlp_scratch", "dtype": "fp32"}},
    },
    "group_limited_topk_router": {
        "inputs": {"scores": "mlp_scratch"},
        "outputs": {
            "indices": {"slot": "q_scratch", "dtype": "i32"},
            "weights": {"slot": "k_scratch", "dtype": "fp32"},
        },
    },
    "moe_relu2_expert_mlp": {
        "inputs": {
            "hidden": "layer_input",
            "indices": "q_scratch",
            "routing_weights": "k_scratch",
        },
        "outputs": {"output": {"slot": "mlp_scratch", "dtype": "fp32"}},
    },
    "shared_relu2_expert_mlp": {
        "inputs": {
            "hidden": "layer_input",
            "routed": "mlp_scratch",
        },
        "outputs": {"output": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "kv_a_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "compressed_kv", "dtype": "fp32"}},
    },
    "kv_a_layernorm": {
        "inputs": {"input": "compressed_kv"},
        "outputs": {"output": {"slot": "compressed_kv_normed", "dtype": "fp32"}},
    },
    "kv_lora_decompress": {
        "inputs": {"compressed_kv": "compressed_kv_normed"},
        "outputs": {
            "k_nope": {"slot": "k_nope", "dtype": "fp32"},
            "value": {"slot": "v_scratch", "dtype": "fp32"},
        },
    },
    "partial_rope_concat": {
        "inputs": {"q_packed": "q_scratch", "k_nope": "k_nope", "k_pe": "compressed_kv"},
        "outputs": {
            "query": {"slot": "q_scratch", "dtype": "fp32"},
            "key": {"slot": "k_scratch", "dtype": "fp32"},
        },
    },
    "mla_attention": {
        "inputs": {"q": "q_scratch", "k": "k_scratch", "v": "v_scratch"},
        "outputs": {"out": {"slot": "attn_scratch", "dtype": "fp32"}},
    },
    "moe_swiglu_expert_mlp": {
        "inputs": {
            "hidden": "layer_input",
            "indices": "q_scratch",
            "routing_weights": "k_scratch",
        },
        "outputs": {"output": {"slot": "mlp_scratch", "dtype": "fp32"}},
    },
    "shared_swiglu_expert_mlp": {
        "inputs": {
            "hidden": "layer_input",
            "routed": "mlp_scratch",
        },
        "outputs": {"output": {"slot": "main_stream", "dtype": "fp32"}},
    },

    "mamba_in_proj": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "recurrent_packed", "dtype": "fp32"}},
    },
    "mamba_in_proj_split": {
        "inputs": {"projected": "recurrent_packed"},
        "outputs": {
            "gate": {"slot": "recurrent_z", "dtype": "fp32"},
            "hidden_bc": {"slot": "recurrent_conv_qkv", "dtype": "fp32"},
            "dt": {"slot": "recurrent_g", "dtype": "fp32"},
        },
    },
    "mamba_dt_softplus": {
        "inputs": {"dt": "recurrent_g"},
        "outputs": {"dt_out": {"slot": "recurrent_g", "dtype": "fp32"}},
    },
    "mamba_conv1d_silu": {
        "inputs": {
            "state_in": "recurrent_conv_state",
            "x": "recurrent_conv_qkv",
        },
        "outputs": {
            "conv_out": {"slot": "recurrent_conv_qkv", "dtype": "fp32"},
            "state_out": {"slot": "recurrent_conv_state", "dtype": "fp32"},
        },
    },
    "mamba_selective_scan": {
        "inputs": {
            "state_init": "recurrent_ssm_state",
            "x": "recurrent_conv_qkv",
            "dt": "recurrent_g",
            "B": "recurrent_conv_qkv",
            "C": "recurrent_conv_qkv",
        },
        "outputs": {
            "state_out": {"slot": "recurrent_ssm_state", "dtype": "fp32"},
            "y": {"slot": "recurrent_v", "dtype": "fp32"},
        },
    },
    "mamba_rmsnorm_gate": {
        "inputs": {"x": "recurrent_v", "gate": "recurrent_z"},
        "outputs": {"out": {"slot": "recurrent_normed", "dtype": "fp32"}},
    },
    "quantize_mamba_out_proj_input": {
        "inputs": {"input": "recurrent_normed"},
        "outputs": {"output": {"slot": "main_stream_q8", "dtype": "q8_k"}},
    },
    "mamba_out_proj": {
        "inputs": {"x": "recurrent_normed"},
        "outputs": {"y": {"slot": "main_stream", "dtype": "fp32"}},
    },

    # MLP block
    "mlp_gate_up": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "mlp_scratch", "dtype": "fp32"}},
    },
    "mlp_up": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "mlp_scratch", "dtype": "fp32"}},
    },
    "bias_add_mlp": {
        "inputs": {"x": "mlp_scratch"},
        "outputs": {"x": {"slot": "mlp_scratch", "dtype": "fp32"}},
    },
    "silu_mul": {
        "inputs": {"x": "mlp_scratch"},
        "outputs": {"out": {"slot": "mlp_scratch", "dtype": "fp32"}},  # In-place
    },
    "geglu": {
        "inputs": {"x": "mlp_scratch"},
        "outputs": {"out": {"slot": "mlp_scratch", "dtype": "fp32"}},  # In-place
    },
    "gelu": {
        "inputs": {"x": "mlp_scratch"},
        "outputs": {"out": {"slot": "mlp_scratch", "dtype": "fp32"}},  # In-place
    },
    "relu2": {
        "inputs": {"x": "mlp_scratch"},
        "outputs": {"out": {"slot": "mlp_scratch", "dtype": "fp32"}},  # In-place
    },
    "quantize_mlp_down_input": {
        "inputs": {"input": "mlp_scratch"},
        "outputs": {"output": {"slot": "main_stream_q8", "dtype": "q8_k"}},
    },
    "mlp_down": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "spatial_merge": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "projector_prep": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "branch_spatial_merge": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "branch_stream", "dtype": "fp32"}},
    },
    "branch_layernorm": {
        "inputs": {"input": "branch_stream"},
        "outputs": {"output": {"slot": "branch_normed", "dtype": "fp32"}},
    },
    "projector_fc1": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "mlp_scratch", "dtype": "fp32"}},
    },
    "projector_gelu": {
        "inputs": {"x": "mlp_scratch"},
        "outputs": {"out": {"slot": "mlp_scratch", "dtype": "fp32"}},
    },
    "projector_fc2": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "main_stream", "dtype": "fp32"}},
    },
    "branch_fc1": {
        "inputs": {"x": "branch_normed"},
        "outputs": {"y": {"slot": "branch_mlp", "dtype": "fp32"}},
    },
    "branch_gelu": {
        "inputs": {"x": "branch_mlp"},
        "outputs": {"out": {"slot": "branch_mlp", "dtype": "fp32"}},
    },
    "branch_fc2": {
        "inputs": {"x": "branch_mlp"},
        "outputs": {"y": {"slot": "branch_collect", "dtype": "fp32"}},
    },
    "branch_concat": {
        "inputs": {
            "main_input": "main_stream",
            "branch_input": "branch_collect",
        },
        "outputs": {"output": {"slot": "vision_output", "dtype": "fp32"}},
    },

    # Footer ops
    "quantize_final_output": {
        "inputs": {"input": "main_stream"},
        "outputs": {"output": {"slot": "main_stream_q8", "dtype": "q8_0"}},
    },
    "logits": {
        "inputs": {"x": "main_stream_q8"},
        "outputs": {"y": {"slot": "logits", "dtype": "fp32"}},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# INIT OPS GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
# Init ops are run ONCE at model load time (not per-token).
# Examples: rope_init (precompute cos/sin tables), ALiBi init, etc.
#
# The init.json file is separate from decode.json and prefill.json because:
#   1. Init ops run once, inference ops run per-token
#   2. Different architectures may need different init ops
#   3. Clean separation of concerns
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_tokenizer_type(
    template: Dict[str, Any],
    config: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Optional[str]:
    flags = template.get("flags", {}) if isinstance(template.get("flags"), dict) else {}
    template_type = str(flags.get("tokenizer") or "").strip().lower()

    explicit_contract = None
    for doc in (manifest, config):
        if not isinstance(doc, dict):
            continue
        tok_contract = doc.get("tokenizer_contract")
        if isinstance(tok_contract, dict):
            explicit_contract = tok_contract
            break
        nested = doc.get("config")
        if isinstance(nested, dict):
            tok_contract = nested.get("tokenizer_contract")
            if isinstance(tok_contract, dict):
                explicit_contract = tok_contract
                break

    if isinstance(explicit_contract, dict):
        explicit_type = str(explicit_contract.get("tokenizer_type") or "").strip().lower()
        if explicit_type:
            return explicit_type

    special_tokens = manifest.get("special_tokens", {}) if isinstance(manifest.get("special_tokens"), dict) else {}
    tok_model = str(special_tokens.get("tokenizer_model") or "").strip().lower()
    if tok_model in {"bpe", "gpt2"}:
        return "bpe"
    if tok_model in {"wordpiece"}:
        return "wordpiece"
    if tok_model in {"llama", "sentencepiece", "spm"}:
        return "sentencepiece"

    return template_type or None


def _generate_tokenizer_c_code(tokenizer_type: str, vocab_size: int, num_merges: int,
                               special_tokens: Optional[Dict] = None,
                               model_type: Optional[str] = None,
                               template_name: Optional[str] = None,
                               chat_contract: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
    """
    Generate tokenizer-specific C code based on tokenizer type from template.

    The tokenizer type comes from template flags (e.g., "bpe", "wordpiece", "sentencepiece").
    This function generates ALL the C code - codegen just emits it blindly.

    For future tokenizer types, add a new elif branch here.
    """
    if tokenizer_type == "bpe":
        add_bos = None
        add_eos = None
        unk_id = None
        bos_id = None
        eos_id = None
        pad_id = None
        if special_tokens:
            add_bos = special_tokens.get("add_bos_token")
            add_eos = special_tokens.get("add_eos_token")
            unk_id = special_tokens.get("unk_token_id")
            bos_id = special_tokens.get("bos_token_id")
            eos_id = special_tokens.get("eos_token_id")
            pad_id = special_tokens.get("pad_token_id")

        bpe_contract_lines = []
        if any(v is not None for v in [unk_id, bos_id, eos_id, pad_id]):
            bpe_contract_lines.append(
                "        ck_true_bpe_set_special_ids(g_model->tokenizer,"
            )
            bpe_contract_lines.append(
                "            " + (str(unk_id) if unk_id is not None else "-1") + ","
            )
            bpe_contract_lines.append(
                "            " + (str(bos_id) if bos_id is not None else "-1") + ","
            )
            bpe_contract_lines.append(
                "            " + (str(eos_id) if eos_id is not None else "-1") + ","
            )
            bpe_contract_lines.append(
                "            " + (str(pad_id) if pad_id is not None else "-1") + ");"
            )
        if add_bos is not None or add_eos is not None:
            bpe_contract_lines.extend(
                [
                    "        {",
                    "            CKBPEConfig cfg = {0};",
                    f"            cfg.add_bos = {'true' if add_bos else 'false'};",
                    f"            cfg.add_eos = {'true' if add_eos else 'false'};",
                    "            cfg.byte_fallback = true;",
                    "            cfg.space_prefix_style = CK_SPACE_PREFIX_AUTO;",
                    "            ck_true_bpe_set_config(g_model->tokenizer, &cfg);",
                    "        }",
                ]
            )
        bpe_contract_block = "\n".join(bpe_contract_lines)
        special_marker_candidates = [
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "<|eot_id|>", "<|begin_of_text|>", "<|end_of_text|>",
            "</s>", "<s>", "<bos>", "<eos>",
            "<start_of_turn>", "<end_of_turn>",
        ]
        for marker in _collect_chat_marker_strings(chat_contract):
            if marker not in special_marker_candidates:
                special_marker_candidates.append(marker)
        special_token_lines = "\n".join(
            f"            {_c_string_literal(marker)},"
            for marker in special_marker_candidates
        )
        return {
            "type": "bpe",
            "include": '#include "tokenizer/true_bpe.h"',
            "struct_field": "CKTrueBPE *tokenizer;    /* BPE tokenizer */",
            "init": f"""
    /* Full BPE encode-side tokenizer initialization is enabled by default so
     * generated runtimes work end-to-end with raw text prompts. Large
     * vocabularies can still run in token-id/display-only mode by setting
     * CK_DISABLE_FULL_BPE_TOKENIZER=1 to skip encode-side merge/rank tables. */
    const char *ck_disable_full_bpe = getenv("CK_DISABLE_FULL_BPE_TOKENIZER");
    if (!ck_disable_full_bpe || strcmp(ck_disable_full_bpe, "0") == 0) {{
        g_model->tokenizer = ck_true_bpe_create();
        if (g_model->tokenizer) {{
            ck_true_bpe_load_binary(
                g_model->tokenizer,
                {vocab_size},
                (const int32_t*)(g_model->bump + W_VOCAB_OFFSETS),
                (const char*)(g_model->bump + W_VOCAB_STRINGS),
                {num_merges},
                (const int32_t*)(g_model->bump + W_VOCAB_MERGES)
            );
{bpe_contract_block}

            /* Register special tokens for pre-BPE matching.
             * Without this, <|im_end|> gets broken into characters by BPE.
             */
            static const char *special_tokens[] = {{
{special_token_lines}
                NULL
            }};
            for (int i = 0; special_tokens[i] != NULL; i++) {{
                int32_t id = ck_true_bpe_lookup(g_model->tokenizer, special_tokens[i]);
                const char *check = ck_true_bpe_id_to_token(g_model->tokenizer, id);
                if (check && strcmp(check, special_tokens[i]) == 0) {{
                    ck_true_bpe_add_special_token(g_model->tokenizer, special_tokens[i], id);
                    #ifdef CK_DEBUG_TOKENIZER
                    printf("[Tokenizer] Registered special: %s -> %d\\n", special_tokens[i], id);
                    #endif
                }}
            }}
        }}
    }}""",
            "free": """
    if (g_model->tokenizer) {
        ck_true_bpe_free(g_model->tokenizer);
        g_model->tokenizer = NULL;
    }""",
            "api_functions": """
/* ============================================================================
 * TOKENIZER API - Encode text to tokens using C tokenizer
 * ============================================================================
 * Full native text encoding is enabled by default for end-to-end raw prompts.
 * Token-id prompts and generated-token display can still use the lightweight
 * BUMP vocab tables if CK_DISABLE_FULL_BPE_TOKENIZER=1 is set.
 */
CK_EXPORT int ck_model_encode_text(const char *text, int text_len) {
    if (!g_model || !g_model->tokenizer || !text) return 0;
    if (text_len < 0) text_len = (int)strlen(text);
    if (text_len == 0) return 0;

    int32_t *token_buf = (int32_t*)(g_model->bump + A_TOKEN_IDS);
    int max_tokens = MAX_SEQ_LEN;
    return ck_true_bpe_encode(g_model->tokenizer, text, text_len, token_buf, max_tokens);
}

static const char *ck_model_vocab_piece(int32_t id, int *len_out) {
#ifndef W_VOCAB_OFFSETS
    (void)id; (void)len_out;
    return NULL;
#else
    if (!g_model || id < 0 || id >= VOCAB_SIZE || !len_out) return NULL;
    const int32_t *offsets = (const int32_t*)(g_model->bump + W_VOCAB_OFFSETS);
    const char *strings = (const char*)(g_model->bump + W_VOCAB_STRINGS);
    int start = offsets[id];
    int end = (id + 1 < VOCAB_SIZE) ? offsets[id + 1] : VOCAB_STRINGS_SIZE;
    if (start < 0 || end < start || end > VOCAB_STRINGS_SIZE) return NULL;
    *len_out = end - start;
    return strings + start;
#endif
}

static int ck_model_append_vocab_piece(int32_t id, char *text, int pos, int max_len) {
    int len = 0;
    const unsigned char *piece = (const unsigned char*)ck_model_vocab_piece(id, &len);
    if (!piece || max_len <= 0) return pos;
    for (int i = 0; i < len && pos < max_len - 1; ) {
        if (i + 1 < len && piece[i] == 0xC4 && piece[i + 1] == 0xA0) {
            text[pos++] = ' ';
            i += 2;
        } else if (i + 1 < len && piece[i] == 0xC4 && piece[i + 1] == 0x8A) {
            text[pos++] = '\\n';
            i += 2;
        } else if (i + 2 < len && piece[i] == 0xE2 && piece[i + 1] == 0x96 && piece[i + 2] == 0x81) {
            text[pos++] = ' ';
            i += 3;
        } else {
            text[pos++] = (char)piece[i++];
        }
    }
    text[pos] = '\\0';
    return pos;
}

/* Decode tokens back to text */
CK_EXPORT int ck_model_decode_tokens(const int32_t *ids, int num_ids, char *text, int max_len) {
    if (!g_model || !ids || !text || max_len <= 0) return 0;
    if (g_model->tokenizer) {
        return ck_true_bpe_decode(g_model->tokenizer, ids, num_ids, text, max_len);
    }
    int pos = 0;
    text[0] = '\\0';
    for (int i = 0; i < num_ids; i++) {
        pos = ck_model_append_vocab_piece(ids[i], text, pos, max_len);
    }
    return pos;
}

/* Check if tokenizer data is available */
CK_EXPORT int ck_model_has_tokenizer(void) {
#ifdef W_VOCAB_OFFSETS
    return g_model ? 1 : 0;
#else
    return (g_model && g_model->tokenizer) ? 1 : 0;
#endif
}

/* Check if raw text encoding is available.
 * BPE models can decode from compact vocab tables without constructing the
 * full tokenizer, but text encoding needs the tokenizer merge/rank state.
 */
CK_EXPORT int ck_model_can_encode_text(void) {
    return (g_model && g_model->tokenizer) ? 1 : 0;
}

/* Get pointer to token buffer (for reading encoded tokens) */
CK_EXPORT const int32_t* ck_model_get_token_buffer(void) {
    return g_model ? (const int32_t*)(g_model->bump + A_TOKEN_IDS) : NULL;
}

/* Lookup single token by text (returns token ID or -1 if not found) */
CK_EXPORT int32_t ck_model_lookup_token(const char *text) {
    if (!g_model || !text) return -1;
    if (g_model->tokenizer) {
        int32_t id = ck_true_bpe_lookup(g_model->tokenizer, text);
        const char *token_str = ck_true_bpe_id_to_token(g_model->tokenizer, id);
        return (token_str && strcmp(token_str, text) == 0) ? id : -1;
    }
#ifdef W_VOCAB_OFFSETS
    int text_len = (int)strlen(text);
    for (int32_t id = 0; id < VOCAB_SIZE; id++) {
        int piece_len = 0;
        const char *piece = ck_model_vocab_piece(id, &piece_len);
        if (piece && piece_len == text_len && memcmp(piece, text, (size_t)text_len) == 0) return id;
    }
#endif
    return -1;
}
"""
        }

    elif tokenizer_type == "sentencepiece":
        add_bos = None
        add_eos = None
        add_space_prefix = None
        tokenizer_model = None
        unk_id = None
        bos_id = None
        eos_id = None
        pad_id = None
        mask_id = None
        if special_tokens:
            add_bos = special_tokens.get("add_bos_token")
            add_eos = special_tokens.get("add_eos_token")
            add_space_prefix = special_tokens.get("add_space_prefix")
            tokenizer_model = special_tokens.get("tokenizer_model")
            unk_id = special_tokens.get("unk_token_id")
            bos_id = special_tokens.get("bos_token_id")
            eos_id = special_tokens.get("eos_token_id")
            pad_id = special_tokens.get("pad_token_id")
            mask_id = special_tokens.get("mask_token_id")

        tokenizer_model_lc = tokenizer_model.strip().lower() if isinstance(tokenizer_model, str) else ""
        model_type_lc = model_type.strip().lower() if isinstance(model_type, str) else ""
        template_name_lc = template_name.strip().lower() if isinstance(template_name, str) else ""
        is_gemma_family = model_type_lc.startswith("gemma") or ("gemma" in template_name_lc)

        # GGUF metadata may report tokenizer_model="llama" for Gemma-family models
        # even though the SentencePiece behavior should be unigram. Keep an
        # explicit override here so codegen does not silently select llama mode.
        effective_spm_model = tokenizer_model_lc
        if tokenizer_model_lc == "llama" and is_gemma_family:
            effective_spm_model = "unigram"

        # IMPORTANT: SPM add_space_prefix is model-family dependent.
        # If metadata is missing this flag, default to:
        # - true for llama-style SPM
        # - false for unigram SPM (Gemma, etc.)
        if add_space_prefix is None:
            add_space_prefix = (effective_spm_model == "llama")

        # Build config setters (only when provided)
        config_lines = []
        if add_bos is not None:
            config_lines.append(
                "            g_model->tokenizer->config.add_bos = %s;" %
                ("true" if add_bos else "false")
            )
        if add_eos is not None:
            config_lines.append(
                "            g_model->tokenizer->config.add_eos = %s;" %
                ("true" if add_eos else "false")
            )
        config_lines.append(
            "            g_model->tokenizer->config.add_space_prefix = %s;" %
            ("true" if add_space_prefix else "false")
        )
        if effective_spm_model:
            if effective_spm_model == "llama":
                config_lines.append(
                    "            g_model->tokenizer->config.spm_mode = CK_SPM_MODE_LLAMA;"
                )
            else:
                config_lines.append(
                    "            g_model->tokenizer->config.spm_mode = CK_SPM_MODE_UNIGRAM;"
                )
        elif is_gemma_family:
            # Defensive default for Gemma when tokenizer_model metadata is absent.
            config_lines.append(
                "            g_model->tokenizer->config.spm_mode = CK_SPM_MODE_UNIGRAM;"
            )

        # Special IDs: fall back to current if missing
        special_ids_lines: List[str] = []
        if any(v is not None for v in [unk_id, bos_id, eos_id, pad_id, mask_id]):
            special_ids_lines.append("            ck_tokenizer_set_special_ids(g_model->tokenizer,")
            special_ids_lines.append("                " + (str(unk_id) if unk_id is not None else "g_model->tokenizer->unk_id") + ",")
            special_ids_lines.append("                " + (str(bos_id) if bos_id is not None else "g_model->tokenizer->bos_id") + ",")
            special_ids_lines.append("                " + (str(eos_id) if eos_id is not None else "g_model->tokenizer->eos_id") + ",")
            special_ids_lines.append("                " + (str(pad_id) if pad_id is not None else "g_model->tokenizer->pad_id") + ",")
            special_ids_lines.append("                " + (str(mask_id) if mask_id is not None else "g_model->tokenizer->mask_id") + ");")
            config_lines.extend(special_ids_lines)

        config_block = "\n".join(config_lines)
        special_ids_reset_block = ""
        if special_ids_lines:
            special_ids_reset_block = "\n".join([
                "            /* Re-apply GGUF special IDs after alias registration. */",
                *special_ids_lines
            ])
        return {
            "type": "spm",
            "include": '#include "tokenizer/tokenizer.h"',
            "struct_field": "CKTokenizer *tokenizer;    /* SPM tokenizer */",
            "init": f"""
    /* Initialize SPM tokenizer from bump data */
    if (getenv("CK_DISABLE_TOKENIZER")) {{
        g_model->tokenizer = NULL;
    }} else {{
        if (getenv("CK_DEBUG_TOKENIZER_INIT")) {{
            fprintf(stderr, "[Tokenizer] SPM init: begin\\n");
        }}
        g_model->tokenizer = ck_tokenizer_create(CK_TOKENIZER_SPM);
        if (g_model->tokenizer) {{
            if (getenv("CK_DEBUG_TOKENIZER_INIT")) {{
                fprintf(stderr, "[Tokenizer] SPM load: begin\\n");
            }}
            #if defined(W_VOCAB_SCORES) && defined(W_VOCAB_TYPES)
            ck_tokenizer_load_binary_with_scores(
                g_model->tokenizer,
                {vocab_size},
                (const int32_t*)(g_model->bump + W_VOCAB_OFFSETS),
                (const char*)(g_model->bump + W_VOCAB_STRINGS),
                (const float*)(g_model->bump + W_VOCAB_SCORES),
                (const uint8_t*)(g_model->bump + W_VOCAB_TYPES),
                0,  /* No BPE merges for SPM */
                NULL
            );
            #else
            ck_tokenizer_load_binary(
                g_model->tokenizer,
                {vocab_size},
                (const int32_t*)(g_model->bump + W_VOCAB_OFFSETS),
                (const char*)(g_model->bump + W_VOCAB_STRINGS),
                0,  /* No BPE merges for SPM */
                NULL
            );
            #endif
            if (getenv("CK_DEBUG_TOKENIZER_INIT")) {{
                fprintf(stderr, "[Tokenizer] SPM load: done\\n");
            }}

{config_block if config_block else ""}

            /* Register special tokens for SPM matching. */
            if (!getenv("CK_SKIP_SPM_SPECIALS")) {{
                static const char *special_tokens[] = {{
                    "<unk>", "<s>", "</s>", "<bos>", "<eos>", "<pad>", "<mask>",
                    "<start_of_turn>", "<end_of_turn>",
                    "<|im_start|>", "<|im_end|>", "<|eot_id|>", "<|endoftext|>",
                    "<think>", "</think>", "<tool_call>", "</tool_call>",
                    NULL
                }};
                for (int i = 0; special_tokens[i] != NULL; i++) {{
                    if (getenv("CK_DEBUG_TOKENIZER_INIT")) {{
                        fprintf(stderr, "[Tokenizer] SPM special: %s\\n", special_tokens[i]);
                    }}
                    int32_t id = ck_tokenizer_lookup(g_model->tokenizer, special_tokens[i]);
                    const char *check = ck_tokenizer_id_to_token(g_model->tokenizer, id);
                    if (check && strcmp(check, special_tokens[i]) == 0) {{
                        ck_tokenizer_add_special_token(g_model->tokenizer, special_tokens[i], id);
                        #ifdef CK_DEBUG_TOKENIZER
                        printf("[Tokenizer] Registered special: %s -> %d\\n", special_tokens[i], id);
                        #endif
                    }}
                }}
            }}
{special_ids_reset_block if special_ids_reset_block else ""}
            if (getenv("CK_DEBUG_TOKENIZER_INIT")) {{
                fprintf(stderr, "[Tokenizer] SPM init: done\\n");
            }}
        }}
    }}""",
            "free": """
    if (g_model->tokenizer) {
        ck_tokenizer_free(g_model->tokenizer);
        g_model->tokenizer = NULL;
    }""",
            "api_functions": """
/* ============================================================================
 * TOKENIZER API - Encode text to tokens using C tokenizer (SPM)
 * ============================================================================
 * Returns: number of tokens written to internal buffer
 * The tokens are written to the same buffer that prefill() reads from.
 * After encoding, call ck_model_prefill() with the returned count.
 */
CK_EXPORT int ck_model_encode_text(const char *text, int text_len) {
    if (!g_model || !g_model->tokenizer || !text) return 0;
    if (text_len < 0) text_len = (int)strlen(text);
    if (text_len == 0) return 0;

    /* Encode directly into the token_ids buffer that prefill uses */
    int32_t *token_buf = (int32_t*)(g_model->bump + A_TOKEN_IDS);
    int max_tokens = MAX_SEQ_LEN;

    int num_tokens = ck_tokenizer_encode(
        g_model->tokenizer,
        text,
        text_len,
        token_buf,
        max_tokens
    );

    return num_tokens;
}

/* Decode tokens back to text */
CK_EXPORT int ck_model_decode_tokens(const int32_t *ids, int num_ids, char *text, int max_len) {
    if (!g_model || !g_model->tokenizer || !ids || !text || max_len <= 0) return 0;
    return ck_tokenizer_decode(g_model->tokenizer, ids, num_ids, text, max_len);
}

/* Check if tokenizer is available */
CK_EXPORT int ck_model_has_tokenizer(void) {
    return (g_model && g_model->tokenizer) ? 1 : 0;
}

/* Check if raw text encoding is available. */
CK_EXPORT int ck_model_can_encode_text(void) {
    return (g_model && g_model->tokenizer) ? 1 : 0;
}

/* Get pointer to token buffer (for reading encoded tokens) */
CK_EXPORT const int32_t* ck_model_get_token_buffer(void) {
    return g_model ? (const int32_t*)(g_model->bump + A_TOKEN_IDS) : NULL;
}

/* Lookup single token by text (returns token ID or -1 if not found)
 * Uses DIRECT vocabulary lookup, not encoding.
 */
CK_EXPORT int32_t ck_model_lookup_token(const char *text) {
    if (!g_model || !g_model->tokenizer || !text) return -1;
    int32_t id = ck_tokenizer_lookup(g_model->tokenizer, text);
    if (id < 0) return -1;
    const char *token_str = ck_tokenizer_id_to_token(g_model->tokenizer, id);
    if (token_str && strcmp(token_str, text) == 0) {
        return id;
    }
    return -1;
}
"""
        }

    # Unknown tokenizer type
    return None


def generate_init_ops(manifest: Dict, config: Dict) -> List[Dict]:
    """
    Generate initialization ops based on model config and template flags.

    Returns list of init ops in IR1 format:
        {
            "op_id": 0,
            "kernel": "rope_precompute_cache",
            "op": "rope_init",
            "section": "init",
            "layer": -1,
            "instance": 0,
            "params": {...},
            "outputs": {...}
        }
    """
    config = _normalize_manifest_config(config)
    init_ops = []
    op_id = 0

    template = manifest.get("template", {})
    flags = template.get("flags", {})
    template_kernels = template.get("kernels", {}) if isinstance(template.get("kernels"), dict) else {}

    # ═══════════════════════════════════════════════════════════
    # ROPE INIT: Precompute cos/sin tables if model uses RoPE
    # ═══════════════════════════════════════════════════════════
    rope_type = flags.get("rope", None)
    if rope_type in ("rope", "rope_qk", "partial_rope_concat", "partial_pairwise_concat", True):
        # Get config values
        rope_theta = config["rope_theta"]
        head_dim = max(int(config["head_dim"]), int(config.get("max_rotary_dim", config["rotary_dim"]) or config["rotary_dim"]))
        rotary_dim = int(config.get("max_rotary_dim", config["rotary_dim"]) or config["rotary_dim"])
        max_seq_len = config["context_length"]

        # RoPE scaling (for extended context models like Llama 3.1)
        rope_scaling_type = config["rope_scaling_type"]
        rope_scaling_factor = config["rope_scaling_factor"]
        rope_layout = config.get("rope_layout", "")
        rope_original_context_length = config.get("rope_original_context_length", max_seq_len)
        rope_beta_fast = config.get("rope_beta_fast", 0.0)
        rope_beta_slow = config.get("rope_beta_slow", 0.0)
        rope_attn_factor = config.get("rope_attn_factor", 1.0)

        rope_init_kernel = template_kernels.get("rope_init", "rope_precompute_cache")
        rope_init_params = {
            "max_seq_len": {"source": "dim:max_seq_len", "value": max_seq_len},
            "head_dim": {"source": "dim:head_dim", "value": head_dim},
            "base": {"source": "config:rope_theta", "value": rope_theta},
        }
        if rope_init_kernel != "rope_precompute_cache_split":
            rope_init_params["rotary_dim"] = {"source": "dim:rotary_dim", "value": rotary_dim}
            rope_init_params["scaling_type"] = {"source": "config:rope_scaling_type", "value": rope_scaling_type}
            rope_init_params["scaling_factor"] = {"source": "config:rope_scaling_factor", "value": rope_scaling_factor}

        init_ops.append({
            "op_id": op_id,
            "kernel": rope_init_kernel,
            "op": "rope_init",
            "section": "init",
            "layer": -1,
            "instance": 0,
            "dataflow": {
                "inputs": {},  # No inputs - pure computation from config
                "outputs": {
                    "cos_cache": {"dtype": "fp32", "buffer": "rope_cache"},
                    "sin_cache": {"dtype": "fp32", "buffer": "rope_cache"},
                }
            },
            "params": rope_init_params,
            "config": {
                "rope_theta": rope_theta,
                "rotary_dim": rotary_dim,
                "rope_scaling_type": rope_scaling_type,
                "rope_scaling_factor": rope_scaling_factor,
                "rope_layout": rope_layout,
                "rope_original_context_length": rope_original_context_length,
                "rope_beta_fast": rope_beta_fast,
                "rope_beta_slow": rope_beta_slow,
                "rope_attn_factor": rope_attn_factor,
            },
            "notes": f"RoPE cache init: theta={rope_theta}, rotary_dim={rotary_dim}, scaling={rope_scaling_type}/{rope_scaling_factor}, max_seq={max_seq_len}"
        })
        op_id += 1

    # ═══════════════════════════════════════════════════════════
    # TOKENIZER INIT: Load tokenizer from bump data
    # ═══════════════════════════════════════════════════════════
    # Prefer the explicit tokenizer contract emitted during conversion. Falling
    # back to template flags keeps older manifests working.
    tokenizer_type = _resolve_tokenizer_type(template, config, manifest)

    # Check if vocab data is in manifest (entries list, not weights dict)
    entries = manifest.get("entries", [])
    entry_names = {e.get("name") for e in entries}
    has_vocab = all(k in entry_names for k in ["vocab_offsets", "vocab_strings", "vocab_merges"])

    if has_vocab and tokenizer_type:
        vocab_size = config.get("vocab_size", 151936)
        special_tokens = manifest.get("special_tokens", {}) or {}
        # Build entry lookup dict
        entry_by_name = {e.get("name"): e for e in entries}
        vocab_offsets_info = entry_by_name.get("vocab_offsets", {})
        vocab_strings_info = entry_by_name.get("vocab_strings", {})
        vocab_merges_info = entry_by_name.get("vocab_merges", {})
        vocab_scores_info = entry_by_name.get("vocab_scores", {})
        vocab_types_info = entry_by_name.get("vocab_types", {})

        # Calculate number of merges from size (each merge is 3 int32s = 12 bytes)
        merges_size = vocab_merges_info.get("size", 0)
        num_merges = merges_size // 12  # 3 * sizeof(int32_t)

        # Generate tokenizer-specific c_code based on type from template
        explicit_chat_contract = config.get("chat_contract") if isinstance(config.get("chat_contract"), dict) else None
        if explicit_chat_contract is None:
            template_contract = template.get("contract") if isinstance(template.get("contract"), dict) else {}
            explicit_chat_contract = (
                template_contract.get("chat_contract")
                if isinstance(template_contract.get("chat_contract"), dict)
                else None
            )

        c_code = _generate_tokenizer_c_code(
            tokenizer_type,
            vocab_size,
            num_merges,
            special_tokens,
            config.get("model_type"),
            template.get("name"),
            explicit_chat_contract,
        )

        if c_code:
            # Build inputs dict - include scores/types if available (for SPM)
            inputs = {
                "vocab_offsets": {"dtype": "i32", "source": "weight:vocab_offsets"},
                "vocab_strings": {"dtype": "u8", "source": "weight:vocab_strings"},
            }
            # Add vocab_merges for BPE
            if "vocab_merges" in entry_names:
                inputs["vocab_merges"] = {"dtype": "i32", "source": "weight:vocab_merges"}
            # Add vocab_scores for SPM
            if "vocab_scores" in entry_names:
                inputs["vocab_scores"] = {"dtype": "f32", "source": "weight:vocab_scores"}
            # Add vocab_types for SPM
            if "vocab_types" in entry_names:
                inputs["vocab_types"] = {"dtype": "u8", "source": "weight:vocab_types"}

            init_ops.append({
                "op_id": op_id,
                "kernel": f"tokenizer_{tokenizer_type}_init",
                "op": "tokenizer_init",
                "section": "init",
                "layer": -1,
                "instance": 0,
                "dataflow": {
                    "inputs": inputs,
                    "outputs": {}
                },
                "params": {
                    "vocab_size": {"source": "config:vocab_size", "value": vocab_size},
                    "num_merges": {"source": "computed", "value": num_merges},
                },
                "c_code": c_code,
                "notes": f"{tokenizer_type.upper()} tokenizer init: vocab_size={vocab_size}, num_merges={num_merges}"
            })
            op_id += 1

    # ═══════════════════════════════════════════════════════════
    # FUTURE: Add other init ops here
    # ═══════════════════════════════════════════════════════════
    # Examples:
    #   - ALiBi slope computation (for models using ALiBi instead of RoPE)
    #   - Learned positional embedding init
    #   - Custom attention bias init

    return init_ops


def generate_init_ir(manifest: Dict, config: Dict) -> Dict:
    """
    Generate the complete init IR (init.json) with all initialization ops.

    Returns:
        {
            "format": "ir1-init-v7",
            "version": 1,
            "config": {...},
            "special_tokens": {...},  # EOS, BOS, etc. from GGUF
            "ops": [...]
        }
    """
    config = _normalize_manifest_config(config)
    init_ops = generate_init_ops(manifest, config)

    # Extract special tokens from manifest for propagation to generated code
    # These come from GGUF metadata (tokenizer.ggml.eos_token_id, etc.)
    special_tokens = manifest.get("special_tokens", {})

    return {
        "format": "ir1-init-v7",
        "version": 1,
        "description": "Model initialization ops (run once at load time)",
        "config": {
            "model": config.get("model", "unknown"),
            "rope_theta": config["rope_theta"],
            "rotary_dim": config["rotary_dim"],
            "rope_scaling_type": config["rope_scaling_type"],
            "rope_scaling_factor": config["rope_scaling_factor"],
            "head_dim": config["head_dim"],
            "max_seq_len": config["context_length"],
            "num_heads": config.get("num_heads", 0),
            "num_kv_heads": config.get("num_kv_heads", 0),
        },
        # Special tokens from GGUF - used by orchestrator for EOS detection
        "special_tokens": special_tokens if special_tokens else None,
        "ops": init_ops,
        "stats": {
            "total_ops": len(init_ops),
            "has_rope_init": any(op["op"] == "rope_init" for op in init_ops),
        }
    }


class DataflowTracker:
    """
    Tracks dataflow during IR1 generation.

    Maintains a mapping of slot_name -> (op_id, output_name, dtype) for each logical slot.
    When an op is added, records its inputs (from current slot state) and outputs (updates slot state).
    """

    def __init__(self):
        # Map slot name -> {op_id, output_name, dtype}
        self.slots: Dict[str, Dict[str, Any]] = {}
        # For residual_save tracking within a layer
        self.layer_residual_sources: Dict[int, Dict[str, Any]] = {}  # layer -> slot info

    def reset_for_layer(self, layer: int):
        """Reset per-layer state (but keep residual from previous residual_save)."""
        # Clear main stream slots but keep residual
        pass  # Slots persist, residual_save will update the residual slot

    def record_op(self, op_id: int, op_type: str, layer: int, instance: int,
                  input_slot_override: Optional[Dict[str, str]] = None,
                  output_slot_override: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Record an op's dataflow and return the dataflow info to embed in IR1.

        Returns:
            {
                "inputs": {input_name: {"from_op": X, "from_output": "Y", "dtype": "Z", "slot": "..." }},
                "outputs": {output_name: {"dtype": "Z", "slot": "..."}}
            }
        """
        dataflow_def = OP_DATAFLOW.get(op_type, {})

        # ═══════════════════════════════════════════════════════════
        # NOTE: Residual saving is now handled by explicit residual_save ops
        # inserted before rmsnorm in IR1 generation. The residual_save op
        # updates the "residual" slot, and residual_add reads from it.
        # ═══════════════════════════════════════════════════════════

        # Build inputs from current slot state
        inputs = {}
        for input_name, slot_name in dataflow_def.get("inputs", {}).items():
            if input_slot_override and input_name in input_slot_override:
                slot_name = input_slot_override[input_name]
            if slot_name.startswith("external:"):
                # External input (token_ids, etc.)
                inputs[input_name] = {
                    "from": slot_name,
                    "dtype": "i32" if "token" in slot_name else "fp32",
                    "slot": slot_name,
                }
            elif slot_name in self.slots:
                # Get from slot
                slot_info = self.slots[slot_name]
                inputs[input_name] = {
                    "from_op": slot_info["op_id"],
                    "from_output": slot_info["output_name"],
                    "dtype": slot_info["dtype"],
                    "slot": slot_name,
                }
            else:
                # Slot not yet written - this is a bug or first use
                inputs[input_name] = {
                    "from": f"uninitialized:{slot_name}",
                    "dtype": "unknown",
                    "slot": slot_name,
                }

        # Build outputs and update slot state
        outputs = {}
        for output_name, output_info in dataflow_def.get("outputs", {}).items():
            if isinstance(output_info, dict):
                slot_name = output_info["slot"]
                dtype = output_info["dtype"]
            else:
                # Legacy format - just slot name
                slot_name = output_info
                dtype = "fp32"
            if output_slot_override and output_name in output_slot_override:
                slot_name = output_slot_override[output_name]

            outputs[output_name] = {"dtype": dtype, "slot": slot_name}

            # Update slot state
            self.slots[slot_name] = {
                "op_id": op_id,
                "output_name": output_name,
                "dtype": dtype,
            }

            # Special handling for residual_save - track per layer
            if op_type == "residual_save":
                self.layer_residual_sources[layer] = self.slots[slot_name].copy()

        return {
            "inputs": inputs,
            "outputs": outputs,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about tracked dataflow."""
        return {
            "slots_active": list(self.slots.keys()),
            "layers_with_residual": list(self.layer_residual_sources.keys()),
        }


def _sanitize_macro(name: str) -> str:
    """Return an ASCII-safe macro suffix for a name."""
    out = []
    prev_us = False
    for ch in name:
        if ch.isalnum():
            out.append(ch.upper())
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    s = "".join(out).strip("_")
    if not s:
        s = "UNNAMED"
    if s[0].isdigit():
        s = f"N_{s}"
    return s


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def _resolve_logits_layout(config: Dict[str, Any], mode: str) -> str:
    """Resolve logits layout policy for this mode: 'last' or 'full'."""
    layout = str(config.get("logits_layout", "auto")).lower()
    if layout not in {"auto", "last", "full"}:
        layout = "auto"
    if layout == "auto":
        return "full" if mode == "prefill" else "last"
    return layout


def _logits_seq_for_layout(layout: str, mode: str, seq_len: int, context_len: int, config: Dict[str, Any]) -> int:
    """Return logits token count for the requested layout."""
    if layout == "full":
        if mode == "decode":
            return int(context_len or config.get("context_length", config.get("context_len", seq_len)))
        return int(seq_len)
    return 1


def build_activation_specs(config: Dict[str, Any], mode: str, context_len: int, num_layers_override: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """Return activation buffer specs keyed by name."""
    embed_dim = int(config.get("embed_dim", 896))
    num_heads = int(config.get("num_heads", 14))
    num_kv_heads = int(config.get("num_kv_heads", 2))
    head_dim = int(config.get("head_dim", 64))
    intermediate_size = int(config.get("intermediate_size", config.get("intermediate_dim", 4864)))
    vocab_size = int(config.get("vocab_size", 151936))
    num_layers = int(num_layers_override or config.get("num_layers", 24))
    recurrent_q = int(config.get("q_dim", 0) or 0)
    recurrent_k = int(config.get("k_dim", 0) or 0)
    recurrent_v = int(config.get("v_dim", 0) or 0)
    recurrent_inner = int(config.get("ssm_inner_size", 0) or 0)
    recurrent_gate = int(config.get("gate_dim", 0) or 0)
    recurrent_conv_history = int(config.get("ssm_conv_history", 0) or 0)
    recurrent_conv_channels = int(config.get("ssm_conv_channels", 0) or 0)
    recurrent_state_size = int(config.get("ssm_state_size", 0) or 0)
    recurrent_state_heads, recurrent_state_rows, recurrent_state_cols = _recurrent_state_shape(config)
    uses_kv_cache = bool(config.get("_template_uses_kv_cache", True))
    uses_rope = bool(config.get("_template_uses_rope", True))
    has_logits = bool(config.get("_template_has_logits", True))
    uses_kv_cache = bool(config.get("_template_uses_kv_cache", True))
    uses_rope = bool(config.get("_template_uses_rope", True))
    has_logits = bool(config.get("_template_has_logits", True))
    decode_kv_cache_dtype = str(config.get("decode_kv_cache_dtype", "fp32") or "fp32").strip().lower()
    kv_cache_dtype = "fp16" if mode == "decode" and decode_kv_cache_dtype in {"fp16", "f16"} else "fp32"
    kv_elem_bytes = _dtype_size_bytes(kv_cache_dtype)

    def _positive_int_config(name: str, default: int) -> int:
        try:
            value = int(config.get(name, default) or default)
        except Exception:
            value = default
        return value if value > 0 else default

    max_q_head_dim = max(head_dim, _positive_int_config("max_q_head_dim", head_dim))
    max_k_head_dim = max(head_dim, _positive_int_config("max_k_head_dim", head_dim))
    max_v_head_dim = max(head_dim, _positive_int_config("max_v_head_dim", head_dim))
    kv_cache_head_dim = max(
        max_k_head_dim,
        max_v_head_dim,
        _positive_int_config("kv_cache_head_dim", head_dim),
    )
    max_attn_head_dim = max(max_q_head_dim, max_k_head_dim, max_v_head_dim, kv_cache_head_dim)
    kv_cache_token_stride_total = int(config.get("kv_cache_token_stride_total", 0) or 0)

    max_context = int(config.get("context_length", 32768))
    if context_len is None:
        context_len = max_context
    else:
        context_len = min(context_len, max_context)

    seq_len = 1 if mode == "decode" else context_len
    image_height = int(config.get("image_height", config.get("image_size", 0)) or 0)
    image_width = int(config.get("image_width", config.get("image_size", 0)) or 0)
    patch_size = int(config.get("patch_size", 0) or 0)
    vision_channels = int(config.get("vision_channels", 3) or 3)
    patch_dim = int(config.get("patch_dim", vision_channels * patch_size * patch_size) or 0)
    vision_grid_h = int(config.get("vision_grid_h", (image_height // patch_size) if image_height and patch_size else 0) or 0)
    vision_grid_w = int(config.get("vision_grid_w", (image_width // patch_size) if image_width and patch_size else 0) or 0)
    vision_num_patches = int(
        config.get(
            "vision_num_patches",
            (vision_grid_h * vision_grid_w) if vision_grid_h and vision_grid_w else 0,
        ) or 0
    )

    specs = {}

    def add(name: str, size: int, shape: str, dtype: str = "fp32") -> None:
        specs[name] = {
            "name": name,
            "size": int(size),
            "shape": shape,
            "dtype": dtype,
        }

    # Text input (optional)
    max_input_bytes = seq_len * 16
    add("text_input", max_input_bytes, f"[{max_input_bytes}]", "u8")

    # Token IDs
    token_ids_size = seq_len * 4
    add("token_ids", token_ids_size, f"[{seq_len}]", "i32")

    if image_height > 0 and image_width > 0:
        image_input_size = vision_channels * image_height * image_width * 4
        add("image_input", image_input_size, f"[{vision_channels}, {image_height}, {image_width}]")
    if vision_num_patches > 0 and patch_dim > 0:
        patch_scratch_size = vision_num_patches * patch_dim * 4
        add("patch_scratch", patch_scratch_size, f"[{vision_num_patches}, {patch_dim}]")
    if vision_num_patches > 0:
        add("vision_positions", vision_num_patches * 4 * 4, f"[4, {vision_num_patches}]", "i32")

    # Embedding + layer buffers
    embedded_size = seq_len * embed_dim * 4
    backbone_hidden_size = int(config.get("backbone_hidden_size", 0) or 0)
    if backbone_hidden_size > 0:
        add("backbone_stream", seq_len * backbone_hidden_size * 4, f"[{seq_len}, {backbone_hidden_size}]")
    add("embedded_input", embedded_size, f"[{seq_len}, {embed_dim}]")
    add("layer_input", embedded_size, f"[{seq_len}, {embed_dim}]")
    add("residual", embedded_size, f"[{seq_len}, {embed_dim}]")

    # KV cache + RoPE
    if uses_kv_cache:
        if kv_cache_token_stride_total > 0:
            total_kv_size = context_len * kv_cache_token_stride_total * kv_elem_bytes
            add("kv_cache", total_kv_size, f"[variable_kv, {context_len}, mixed_head_dim]", kv_cache_dtype)
        else:
            kv_per_layer = num_kv_heads * context_len * kv_cache_head_dim * kv_elem_bytes
            total_kv_size = num_layers * 2 * kv_per_layer
            add("kv_cache", total_kv_size, f"[{num_layers}, 2, {num_kv_heads}, {context_len}, {kv_cache_head_dim}]", kv_cache_dtype)

    rotary_dim = int(config.get("rotary_dim", head_dim) or head_dim)
    layer_rotary = config.get("layer_rotary_dim")
    if isinstance(layer_rotary, list) and layer_rotary:
        rotary_dim = max(rotary_dim, max(int(v or 0) for v in layer_rotary))
    rope_half = int(rotary_dim) // 2
    if uses_rope:
        rope_size = context_len * rope_half * 4 * 2
        add("rope_cache", rope_size, f"[2, {context_len}, {rope_half}]")

    # Scratch buffers
    q_size = num_heads * seq_len * max_q_head_dim * 4
    k_size = num_kv_heads * seq_len * max_k_head_dim * 4
    v_size = num_kv_heads * seq_len * max_v_head_dim * 4
    attn_out_size = num_heads * seq_len * max_q_head_dim * 4
    q_gate_proj_dim = int(config.get("q_gate_proj_dim", config.get("attn_q_gate_proj_dim", 0)) or 0)
    if q_gate_proj_dim <= 0:
        q_gate_proj_dim = 2 * num_heads * max_q_head_dim
    attn_gate_dim = int(config.get("attn_gate_dim", max(q_gate_proj_dim - (num_heads * max_q_head_dim), 0)) or 0)
    if attn_gate_dim <= 0:
        attn_gate_dim = num_heads * max_q_head_dim
    attn_q_gate_packed_size = seq_len * q_gate_proj_dim * 4
    attn_gate_size = seq_len * attn_gate_dim * 4
    add("q_scratch", q_size, f"[{num_heads}, {seq_len}, {max_q_head_dim}]")
    add("k_scratch", k_size, f"[{num_kv_heads}, {seq_len}, {max_k_head_dim}]")
    add("v_scratch", v_size, f"[{num_kv_heads}, {seq_len}, {max_v_head_dim}]")
    add("attn_q_gate_packed", attn_q_gate_packed_size, f"[{seq_len}, {q_gate_proj_dim}]")
    add("attn_gate", attn_gate_size, f"[{seq_len}, {attn_gate_dim}]")
    add("attn_scratch", attn_out_size, f"[{num_heads}, {seq_len}, {max_q_head_dim}]")

    kv_lora_rank = int(config.get("kv_lora_rank", 0) or 0)
    qk_nope_dim = int(config.get("qk_nope_head_dim", 0) or 0)
    qk_rope_dim = int(config.get("qk_rope_head_dim", 0) or 0)
    if kv_lora_rank > 0 and qk_rope_dim > 0:
        add("compressed_kv", seq_len * (kv_lora_rank + qk_rope_dim) * 4, f"[{seq_len}, {kv_lora_rank + qk_rope_dim}]")
        add("compressed_kv_normed", seq_len * kv_lora_rank * 4, f"[{seq_len}, {kv_lora_rank}]")
        if qk_nope_dim > 0:
            add("k_nope", num_heads * seq_len * qk_nope_dim * 4, f"[{num_heads}, {seq_len}, {qk_nope_dim}]")

    if bool(config.get("gemma4_per_layer_embedding", False)):
        per_layer_dim = int(config.get("per_layer_dim", 0) or 0)
        if per_layer_dim > 0 and num_layers > 0:
            per_layer_size = seq_len * num_layers * per_layer_dim * 4
            add("gemma4_per_layer_stream", per_layer_size, f"[{seq_len}, {num_layers}, {per_layer_dim}]")

    mlp_size = seq_len * intermediate_size * 2 * 4
    fused_attn_scratch = max(350 * 1024, 3 * num_heads * seq_len * max_attn_head_dim * 4 + embed_dim * 4 * seq_len * 4)
    # BF16 GeGLU needs 3 * seq_len * dim * 4 (input [a,b] + output)
    geglu_bf16_scratch = seq_len * intermediate_size * 3 * 4
    scratch_size = max(mlp_size, fused_attn_scratch, geglu_bf16_scratch)
    add("mlp_scratch", scratch_size, f"[max({seq_len}*{intermediate_size*2}, fused_attn, geglu_bf16)]")

    # Layer output
    layer_out_size = seq_len * embed_dim * 4
    add("layer_output", layer_out_size, f"[{seq_len}, {embed_dim}]")

    projector_in_dim = int(config.get("projector_in_dim", 0) or 0)
    projector_hidden_dim = int(config.get("projector_hidden_dim", 0) or 0)
    projector_out_dim = int(config.get("projector_out_dim", 0) or 0)
    projector_total_out_dim = int(config.get("projector_total_out_dim", projector_out_dim) or 0)
    num_deepstack_layers = int(config.get("num_deepstack_layers", 0) or 0)
    merged_tokens = int(config.get("vision_merged_tokens", 0) or 0)
    if num_deepstack_layers > 0 and merged_tokens > 0:
        if projector_in_dim > 0:
            add("branch_stream", merged_tokens * projector_in_dim * 4, f"[{merged_tokens}, {projector_in_dim}]")
            add("branch_normed", merged_tokens * projector_in_dim * 4, f"[{merged_tokens}, {projector_in_dim}]")
        if projector_hidden_dim > 0:
            add("branch_mlp", merged_tokens * projector_hidden_dim * 4, f"[{merged_tokens}, {projector_hidden_dim}]")
        if projector_out_dim > 0:
            add(
                "branch_collect",
                merged_tokens * projector_out_dim * num_deepstack_layers * 4,
                f"[{merged_tokens}, {projector_out_dim * num_deepstack_layers}]",
            )
    if projector_total_out_dim > 0 and merged_tokens > 0:
        add("vision_output", merged_tokens * projector_total_out_dim * 4, f"[{merged_tokens}, {projector_total_out_dim}]")

    if any(v > 0 for v in (
        recurrent_q, recurrent_k, recurrent_v, recurrent_inner,
        recurrent_gate, recurrent_conv_channels, recurrent_state_size,
    )):
        packed_dim = max(recurrent_q + recurrent_k + recurrent_v, recurrent_inner, int(config.get("mamba_projection_size", 0) or 0))
        packed_size = seq_len * packed_dim * 4
        recurrent_inner_size = seq_len * recurrent_inner * 4
        gate_size = seq_len * recurrent_gate * 4
        beta_size = seq_len * recurrent_gate * 4
        q_size = seq_len * recurrent_q * 4
        k_size = seq_len * recurrent_k * 4
        v_size = seq_len * recurrent_v * 4
        conv_input_width = max(1, recurrent_conv_history + seq_len)
        conv_input_size = max(1, recurrent_conv_channels) * conv_input_width * 4
        conv_qkv_size = seq_len * max(1, recurrent_conv_channels) * 4
        conv_state_stride = max(1, recurrent_conv_history) * max(1, recurrent_conv_channels) * 4
        ssm_state_stride = max(1, recurrent_state_heads) * max(1, recurrent_state_rows) * max(1, recurrent_state_cols) * 4
        conv_state_size = num_layers * conv_state_stride
        ssm_state_size = num_layers * ssm_state_stride
        add("recurrent_packed", packed_size, f"[{seq_len}, {packed_dim}]")
        add("recurrent_z", recurrent_inner_size, f"[{seq_len}, {recurrent_inner}]")
        add("recurrent_normed", recurrent_inner_size, f"[{seq_len}, {recurrent_inner}]")
        add("recurrent_g", gate_size, f"[{seq_len}, {recurrent_gate}]")
        add("recurrent_beta", beta_size, f"[{seq_len}, {recurrent_gate}]")
        add("recurrent_q", q_size, f"[{seq_len}, {recurrent_q}]")
        add("recurrent_k", k_size, f"[{seq_len}, {recurrent_k}]")
        add("recurrent_v", v_size, f"[{seq_len}, {recurrent_v}]")
        add("recurrent_conv_input", conv_input_size, f"[{recurrent_conv_channels}, {recurrent_conv_history + seq_len}]")
        add("recurrent_conv_qkv_raw", conv_qkv_size, f"[{seq_len}, {recurrent_conv_channels}]")
        add("recurrent_conv_qkv", conv_qkv_size, f"[{seq_len}, {recurrent_conv_channels}]")
        add("recurrent_conv_state", conv_state_size, f"[{num_layers}, {recurrent_conv_history}, {recurrent_conv_channels}]")
        add("recurrent_ssm_state", ssm_state_size, f"[{num_layers}, {recurrent_state_heads}, {recurrent_state_rows}, {recurrent_state_cols}]")

    # Logits
    if has_logits:
        logits_layout = _resolve_logits_layout(config, mode)
        logits_seq = _logits_seq_for_layout(logits_layout, mode, seq_len, context_len, config)
        logits_size = logits_seq * vocab_size * 4
        add("logits", logits_size, f"[{logits_seq}, {vocab_size}]")

    return specs

# Script directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # version/v8
REPO_ROOT = PROJECT_ROOT.parent.parent  # repo root
V8_ROOT = REPO_ROOT / "version" / "v8"


def _validated_kernel_codegen_capability(kernel_id: str, kernel_map: Dict) -> Optional[Dict]:
    capability = kernel_map.get("codegen_capability")
    if capability is None:
        return None
    schema_path = V8_ROOT / "schemas" / "kernel_codegen_capability.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(capability),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise RuntimeError(
            f"HARD CODEGEN CAPABILITY FAULT: kernel {kernel_id!r} capability is invalid "
            f"at {location}: {error.message}"
        )
    function = str((kernel_map.get("impl") or {}).get("function", kernel_id))
    if capability["function"] != function:
        raise RuntimeError(
            f"HARD CODEGEN CAPABILITY FAULT: kernel {kernel_id!r} advertises function "
            f"{capability['function']!r}, but its implementation is {function!r}."
        )
    resolved = copy.deepcopy(capability)
    resolved["kernel_id"] = kernel_id
    return resolved

# Template Op → Kernel Op Mapping
# This is the single source of truth for how template ops map to kernel registry ops.
# Keep this mapping semantic, not architecture-named: the builder should only see
# declared operations to lower and kernels to stitch, regardless of whether the
# source model is dense, recurrent, DeepStack-style, MoE, SSM, or something else.
# Note: "matmul" is a logical op that maps to gemv (decode) or gemm (prefill) based on mode
TEMPLATE_TO_KERNEL_OP = {
    # Header ops
    "tokenizer": None,  # Metadata op - no kernel (deprecated, use bpe_tokenizer)
    "bpe_tokenizer": None,  # BPE tokenizer - init handled separately
    "wordpiece_tokenizer": None,  # WordPiece tokenizer - init handled separately
    "tiktoken_tokenizer": None,  # TikToken tokenizer - init handled separately
    "patch_embeddings": None,  # Vision model patches - init handled separately
    "patchify": "vision_patchify",
    "patch_proj": "matmul",
    "patch_proj_aux": "matmul",
    "add_stream": "add_stream",
    "position_embeddings": "position_embeddings",
    "vision_position_ids": "position_ids",
    "position_ids_2d": "position_ids",
    "patch_bias_add": "rowwise_bias_add",
    "dense_embedding_lookup": "embedding",  # Token embedding lookup
    "residual_save": "residual_save",
    "embedding": "embedding",

    # Attention block
    "rmsnorm": "rmsnorm",
    "layernorm": "layernorm",
    "attn_norm": "rmsnorm",
    "block_rmsnorm": "rmsnorm",
    "post_attention_norm": "rmsnorm",
    "ffn_norm": "rmsnorm",
    "post_ffn_norm": "rmsnorm",
    "gemma4_per_layer_prepare": "gemma4_per_layer_prepare",
    "gemma4_per_layer_embed": "gemma4_per_layer_embed",
    "final_logit_softcap": "final_logit_softcap",
    "v_norm": "v_norm",
    "qkv_proj": "qkv_projection",  # Or fallback to 3x matmul
    "qkv_packed_proj": "matmul",
    "q_proj": "matmul",
    "assistant_pre_projection": "matmul",
    "assistant_post_projection": "matmul",
    "assistant_layer_scale": "assistant_layer_scale",
    "q_gate_proj": "matmul",
    "k_proj": "matmul",
    "v_proj": "matmul",
    "recurrent_qkv_proj": "matmul",
    "recurrent_gate_proj": "matmul",
    "recurrent_alpha_proj": "matmul",
    "recurrent_beta_proj": "matmul",
    "split_q_gate": "split_q_gate",
    "recurrent_split_qkv": "recurrent_split_qkv",
    "split_qkv_packed": "split_qkv_packed_head_major",
    "recurrent_dt_gate": "recurrent_dt_gate",
    "recurrent_conv_state_update": "recurrent_conv_state_update",
    "recurrent_ssm_conv": "ssm_conv1d",
    "recurrent_silu": "recurrent_silu",
    "recurrent_split_conv_qkv": "recurrent_split_conv_qkv",
    "recurrent_qk_l2_norm": "recurrent_qk_l2_norm",
    "recurrent_core": "gated_deltanet",
    "recurrent_norm_gate": "recurrent_norm_gate",
    "attn_gate_sigmoid_mul": "attn_gate_sigmoid_mul",
    "recurrent_out_proj": "matmul",
    "mamba_in_proj": "matmul",
    "mamba_in_proj_split": "mamba_in_proj_split",
    "mamba_dt_softplus": "mamba_dt_softplus",
    "mamba_conv1d_silu": "mamba_conv1d_state_update",
    "mamba_selective_scan": "mamba_selective_scan",
    "mamba_rmsnorm_gate": "mamba_rmsnorm_gate",
    "mamba_out_proj": "matmul",
    "moe_router": "matmul",
    "group_limited_topk_router": "group_limited_topk_router",
    "moe_relu2_expert_mlp": "moe_relu2_expert_mlp",
    "shared_relu2_expert_mlp": "shared_relu2_expert_mlp",
    "moe_swiglu_expert_mlp": "moe_swiglu_expert_mlp",
    "shared_swiglu_expert_mlp": "shared_swiglu_expert_mlp",
    "kv_a_proj": "matmul",
    "kv_a_layernorm": "rmsnorm",
    "kv_lora_decompress": "kv_lora_decompress",
    "partial_rope_concat": "partial_rope_concat",
    "mla_attention": "attention",
    "rope_qk": "rope",
    "mrope_qk": "rope",
    "kv_cache_store": "kv_cache_store",  # Store K,V to KV cache at pos
    "attn": "attention",
    "attn_sliding": "attention_sliding",
    "out_proj": "matmul",  # gemv (decode) or gemm (prefill)

    # Residual
    "residual_add": "residual_add",

    # MLP block
    # NOTE: mega_fused_outproj_mlp_prefill expects head-major attention output,
    # which conflicts with the current pipeline where attention is followed by OutProj.
    # Use simple matmul for mlp_gate_up to avoid the mismatch.
    "mlp_gate_up": "matmul",  # gemv (decode) or gemm (prefill) - use unfused MLP
    "mlp_up": "matmul",
    "relu2": "relu2",
    "silu_mul": "swiglu",
    "geglu": "geglu",
    "gelu": "gelu",
    "mlp_down": "matmul",  # gemv (decode) or gemm (prefill)
    "spatial_merge": "spatial_merge",
    "projector_prep": "projector_prep",
    "branch_spatial_merge": "spatial_merge",
    "branch_layernorm": "layernorm",
    "projector_fc1": "matmul",
    "projector_gelu": "gelu",
    "projector_fc2": "matmul",
    "branch_fc1": "matmul",
    "branch_gelu": "gelu",
    "branch_fc2": "matmul",
    "branch_concat": "feature_concat",

    # QK norm (Qwen3-style: per-head RMSNorm on Q and K after projection)
    "qk_norm": "qk_norm",  # Dedicated kernel wrapping rmsnorm_forward twice
    "q_norm": "q_norm",  # Gemma4 assistant q-only RMSNorm before RoPE

    "rope_q": "rope",
    "attn_shared_kv": "attention",
    "attn_sliding_shared_kv": "attention_sliding",
    "kv_cache_store_shared_q": "kv_cache_store",

    # Footer ops
    "final_rmsnorm": "rmsnorm",
    "weight_tying": None,  # Metadata op - no kernel
    "lm_head": None,  # Metadata op - signals separate lm_head weight (not tied)
    "logits": "matmul",  # gemv (decode) or gemm (prefill)
}

# Map IR1 weight keys to kernel input names
# IR1 uses: wq, wk, wv, wo, w1, w2, ln1_gamma, token_emb, etc.
# Kernel maps use: W, x, gamma, weight, etc.
WEIGHT_TO_KERNEL_INPUT = {
    # Matrix weights → W
    "wq": "W", "wk": "W", "wv": "W", "wo": "W",
    "w1": "W", "w2": "W", "w3": "W",
    "attn_qkv": "W", "attn_gate": "W",
    "ssm_alpha": "W", "ssm_beta": "W", "ssm_out": "W",
    "patch_emb": "W", "patch_emb_aux": "W",
    "mm0_w": "W", "mm1_w": "W",
    "assistant_pre_projection": "W", "assistant_post_projection": "W",
    "layer_output_scale": "scale",
    "branch_fc1_w": "W", "branch_fc2_w": "W",
    # Biases → bias (if kernel has it)
    "bq": "bias", "bk": "bias", "bv": "bias", "bo": "bias",
    "b1": "bias", "b2": "bias",
    "ssm_dt_bias": "bias",
    "patch_bias": "bias", "bqkv": "bias", "mm0_b": "bias", "mm1_b": "bias",
    "branch_fc1_b": "bias", "branch_fc2_b": "bias",
    # Layer norms → gamma/beta
    "ln1_gamma": "gamma", "ln2_gamma": "gamma",
    "ln1_beta": "beta", "ln2_beta": "beta",
    "branch_norm_gamma": "gamma", "branch_norm_beta": "beta",
    "attn_norm": "gamma", "post_attention_norm": "gamma",
    "ffn_norm": "gamma", "post_ffn_norm": "gamma",
    "ssm_norm": "gamma",
    # QK norm weights → q_gamma, k_gamma
    "q_norm": "q_gamma", "k_norm": "k_gamma",
    # Recurrent block special tensors
    "ssm_conv1d": "kernel",
    "ssm_a": "A",
    # Embeddings
    "token_emb": "weight",
    "pos_emb": "pos_emb",
    "lm_head": "W",
    # Footer
    "final_ln_weight": "gamma", "final_ln_bias": "bias",
}


def _resolve_config_layer_kind(
    config: Dict[str, Any],
    layer_idx: int,
    *,
    kind_key: str = "layer_kinds",
    interval_key: Optional[str] = None,
    periodic_kind: Optional[str] = None,
    default_kind: str = "",
) -> str:
    kinds = config.get(kind_key)
    if isinstance(kinds, list) and 0 <= layer_idx < len(kinds):
        kind = str(kinds[layer_idx] or "").strip().lower()
        if kind:
            return kind

    if not interval_key or not periodic_kind:
        return default_kind

    interval_value = config.get(interval_key)
    try:
        interval = int(interval_value)
    except Exception:
        interval = 0
    if interval > 0 and layer_idx >= 0:
        return periodic_kind if ((layer_idx + 1) % interval == 0) else default_kind
    return default_kind


def _template_item_is_active(item: Dict[str, Any]) -> bool:
    lowering = item.get("lowering") if isinstance(item.get("lowering"), dict) else {}
    enabled = lowering.get("enabled")
    if enabled is False:
        return False
    status = str(item.get("status", "active") or "active").strip().lower()
    return status not in {"planned", "disabled", "metadata_only"}


def _normalize_template_op_items(section: Any, include_inactive: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(section, list):
        return out
    for item in section:
        if isinstance(item, str):
            candidate = {"op": item}
        elif isinstance(item, dict):
            candidate = copy.deepcopy(item)
        else:
            continue
        op = candidate.get("op")
        if not isinstance(op, str) or not op:
            continue
        if include_inactive or _template_item_is_active(candidate):
            out.append(candidate)
    return out


def _extract_template_ops(section: Any, include_inactive: bool = False) -> List[str]:
    # Template sections are the graph contract. The lowerer should consume the
    # declared operations exactly as written here; future branching/routing
    # support should surface as explicit template ops/subgraphs rather than
    # family-specific conditionals in the lowerer.
    return [item["op"] for item in _normalize_template_op_items(section, include_inactive=include_inactive)]


def _template_graph_slots(op_item: Dict[str, Any]) -> Dict[str, Any]:
    graph_slots = op_item.get("graph_slots") if isinstance(op_item.get("graph_slots"), dict) else {}
    out: Dict[str, Any] = {}
    for section in ("inputs", "outputs"):
        value = graph_slots.get(section)
        if isinstance(value, dict) and value:
            out[section] = copy.deepcopy(value)
    return out


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


PRE_NORM_OP_NAMES = {"rmsnorm", "layernorm", "attn_norm", "ffn_norm", "post_attention_norm", "block_rmsnorm"}
RESIDUAL_SOURCE_BRANCH_STARTERS = {
    # Attention branches
    "q_proj", "q_gate_proj", "qkv_proj", "qkv_packed_proj",
    "recurrent_qkv_proj", "recurrent_gate_proj",
    "recurrent_alpha_proj", "recurrent_beta_proj", "mamba_in_proj",
    # Feed-forward / routed expert branches
    "mlp_gate_up", "mlp_gate", "mlp_up", "moe_router",
}
PRE_NORM_Q8_DIRECT_CONSUMERS = RESIDUAL_SOURCE_BRANCH_STARTERS


def should_insert_residual_save(layer_ops: List[str], op_idx: int) -> bool:
    """
    Insert residual_save only when the current norm starts a branch whose later
    residual_add must still see the branch input.

    This keeps the rule graph-driven instead of family-specific:
    - attn_norm -> q/k/v branch should preserve layer input
    - ffn_norm  -> MLP branch should preserve sa_out / ffn_inp
    - post_attention_norm must NOT overwrite the saved residual, because the
      following residual_add still needs the original layer input
    """
    if op_idx < 0 or op_idx >= len(layer_ops):
        return False
    if layer_ops[op_idx] not in PRE_NORM_OP_NAMES:
        return False
    if op_idx + 1 >= len(layer_ops):
        return False
    if op_idx > 0 and layer_ops[op_idx - 1] == "residual_save":
        return False
    return layer_ops[op_idx + 1] in RESIDUAL_SOURCE_BRANCH_STARTERS


def _resolve_body_items_for_layer(
    body_def: Dict[str, Any],
    config: Dict[str, Any],
    layer_idx: int,
    include_inactive: bool = False,
) -> List[Dict[str, Any]]:
    ops_by_kind = body_def.get("ops_by_kind")
    if not isinstance(ops_by_kind, dict):
        return _normalize_template_op_items(body_def.get("ops", []), include_inactive=include_inactive)

    # Contract note:
    #   Do not hard-code family-specific graph stitching here.
    #   The template must declare the per-kind body graph explicitly.
    #   The lowerer is only allowed to select the declared variant and then
    #   lower those explicit ops one by one.
    #   This function should not care whether a kind represents dense, MoE,
    #   DeepStack, SSM, or some future block family. It only resolves the
    #   declared operation list for the current layer.
    layer_kind = _resolve_config_layer_kind(
        config,
        layer_idx,
        kind_key=str(body_def.get("kind_config_key", "layer_kinds") or "layer_kinds"),
        interval_key=str(body_def.get("interval_config_key", "") or "") or None,
        periodic_kind=str(body_def.get("periodic_kind", "") or "") or None,
        default_kind=str(body_def.get("default_kind", "") or "").strip().lower(),
    )
    if not layer_kind:
        raise RuntimeError(
            f"Template body with ops_by_kind could not classify layer {layer_idx}. "
            "Declare kind_config_key/layer kinds or interval_config_key/periodic_kind/default_kind in the template."
        )

    ops = ops_by_kind.get(layer_kind)
    if not isinstance(ops, list):
        raise RuntimeError(
            f"Template body missing ops_by_kind['{layer_kind}'] for layer {layer_idx}."
        )
    return _normalize_template_op_items(ops, include_inactive=include_inactive)


def _resolve_body_ops_for_layer(
    body_def: Dict[str, Any],
    config: Dict[str, Any],
    layer_idx: int,
    include_inactive: bool = False,
) -> List[str]:
    return [
        item["op"]
        for item in _resolve_body_items_for_layer(
            body_def,
            config,
            layer_idx,
            include_inactive=include_inactive,
        )
    ]


def _collect_body_items_for_validation(
    body_def: Any,
    config: Dict[str, Any],
    include_inactive: bool = False,
) -> List[Dict[str, Any]]:
    if not isinstance(body_def, dict):
        return _normalize_template_op_items(body_def, include_inactive=include_inactive)

    ops_by_kind = body_def.get("ops_by_kind")
    if not isinstance(ops_by_kind, dict):
        return _normalize_template_op_items(body_def.get("ops", []), include_inactive=include_inactive)

    kinds: List[str] = []
    configured_kinds = config.get(str(body_def.get("kind_config_key", "layer_kinds") or "layer_kinds"))
    if isinstance(configured_kinds, list):
        for raw_kind in configured_kinds:
            kind = str(raw_kind or "").strip().lower()
            if kind and kind in ops_by_kind and kind not in kinds:
                kinds.append(kind)

    if not kinds:
        kinds = [str(k).strip().lower() for k in ops_by_kind.keys()]

    collected: List[str] = []
    for kind in kinds:
        collected.extend(_normalize_template_op_items(ops_by_kind.get(kind, []), include_inactive=include_inactive))

    seen: set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for item in collected:
        key = (
            str(item.get("id", "") or ""),
            str(item.get("op", "") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _collect_body_ops_for_validation(
    body_def: Any,
    config: Dict[str, Any],
    include_inactive: bool = False,
) -> List[str]:
    return [
        item["op"]
        for item in _collect_body_items_for_validation(
            body_def,
            config,
            include_inactive=include_inactive,
        )
    ]


def _normalize_block_branches(block_def: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = block_def.get("branches")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        out.append(copy.deepcopy(item))
    return out


def _template_section_id_map(section: Any, include_inactive: bool = True) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in _normalize_template_op_items(section, include_inactive=include_inactive):
        op_id = str(item.get("id", "") or "").strip()
        if not op_id:
            continue
        out[op_id] = copy.deepcopy(item)
    return out


def _parse_template_value_ref(ref: Any) -> Optional[Dict[str, str]]:
    if not isinstance(ref, str):
        return None
    raw = ref.strip()
    if not raw:
        return None
    parts = raw.split(".")
    if len(parts) < 2:
        return None
    section = str(parts[0] or "").strip().lower()
    if section not in {"header", "body", "footer"}:
        return None
    op_id = str(parts[1] or "").strip()
    if not op_id:
        return None
    output_name = str(parts[2] or "").strip() if len(parts) >= 3 else "out"
    return {
        "section": section,
        "op_id": op_id,
        "output": output_name or "out",
        "ref": raw,
    }


def _lookup_template_output_slot(op_name: str, output_name: str = "out") -> Optional[str]:
    dataflow = OP_DATAFLOW.get(str(op_name or "").strip(), {})
    outputs = dataflow.get("outputs", {}) if isinstance(dataflow, dict) else {}
    info = outputs.get(output_name)
    if isinstance(info, dict):
        return str(info.get("slot", "") or "").strip() or None
    if isinstance(info, str):
        return info.strip() or None
    if outputs:
        first = next(iter(outputs.values()))
        if isinstance(first, dict):
            return str(first.get("slot", "") or "").strip() or None
        if isinstance(first, str):
            return first.strip() or None
    return None


def _resolve_branch_layers(branch_def: Dict[str, Any], config: Dict[str, Any]) -> List[int]:
    tap = branch_def.get("tap") if isinstance(branch_def.get("tap"), dict) else {}
    explicit_layers = tap.get("layers")
    if isinstance(explicit_layers, list):
        out = []
        for raw in explicit_layers:
            try:
                out.append(int(raw))
            except Exception:
                continue
        return sorted({layer for layer in out if layer >= 0})

    cfg_key = str(tap.get("layers_from_config", "") or "").strip()
    if not cfg_key:
        return []
    cfg_value = config.get(cfg_key)
    if isinstance(cfg_value, list):
        if cfg_value and all(isinstance(v, bool) for v in cfg_value):
            return [idx for idx, enabled in enumerate(cfg_value) if enabled]
        out = []
        for raw in cfg_value:
            try:
                out.append(int(raw))
            except Exception:
                continue
        return sorted({layer for layer in out if layer >= 0})
    return []


def _template_int_param(
    params: Dict[str, Any],
    key: str,
    config: Dict[str, Any],
    default: int = 0,
) -> int:
    if not isinstance(params, dict):
        return int(default)
    raw = params.get(key)
    if raw is not None:
        try:
            return int(raw)
        except Exception:
            pass
    cfg_key = str(params.get(f"{key}_from_config", "") or "").strip()
    if cfg_key:
        try:
            return int(config.get(cfg_key, default) or 0)
        except Exception:
            return int(default)
    return int(default)


def _required_template_int_param(
    params: Dict[str, Any],
    key: str,
    config: Dict[str, Any],
    op_name: str,
) -> int:
    if not isinstance(params, dict):
        raise RuntimeError(
            f"Template op '{op_name}' must declare '{key}' or '{key}_from_config'."
        )

    raw = params.get(key)
    if raw is not None:
        try:
            value = int(raw)
        except Exception as exc:
            raise RuntimeError(
                f"Template op '{op_name}' has invalid '{key}' value: {raw!r}"
            ) from exc
    else:
        cfg_key = str(params.get(f"{key}_from_config", "") or "").strip()
        if not cfg_key:
            raise RuntimeError(
                f"Template op '{op_name}' must declare '{key}' or '{key}_from_config'."
            )
        if cfg_key not in config or config.get(cfg_key) is None:
            raise RuntimeError(
                f"Template op '{op_name}' requires config['{cfg_key}'] to resolve '{key}'."
            )
        try:
            value = int(config.get(cfg_key))
        except Exception as exc:
            raise RuntimeError(
                f"Template op '{op_name}' could not parse config['{cfg_key}'] for '{key}'."
            ) from exc

    if value <= 0:
        raise RuntimeError(
            f"Template op '{op_name}' requires positive '{key}', got {value}."
        )
    return value


def _template_str_param(
    params: Dict[str, Any],
    key: str,
    config: Dict[str, Any],
    default: str = "",
) -> str:
    if not isinstance(params, dict):
        return str(default)
    raw = params.get(key)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    cfg_key = str(params.get(f"{key}_from_config", "") or "").strip()
    if cfg_key:
        cfg_value = config.get(cfg_key, default)
        if cfg_value is None:
            return str(default)
        return str(cfg_value)
    return str(default)


def _dtype_size_bytes(dtype: str) -> int:
    return {
        "fp32": 4,
        "f32": 4,
        "bf16": 2,
        "fp16": 2,
        "f16": 2,
        "i32": 4,
        "int32": 4,
        "q8_0": 1,
        "q8_k": 1,
    }.get(str(dtype or "").strip().lower(), 4)


def _resolve_branch_collect_contract(
    branch_def: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    collect = branch_def.get("collect") if isinstance(branch_def.get("collect"), dict) else {}
    layers = branch_def.get("layers") if isinstance(branch_def.get("layers"), list) else []
    default_rows = int(config.get("vision_merged_tokens", config.get("vision_num_patches", 0)) or 0)
    default_slice_dim = int(
        config.get("projector_out_dim", config.get("projection_dim", config.get("embed_dim", 0))) or 0
    )
    dtype = "fp32"
    return {
        "target": _template_str_param(
            collect,
            "target",
            config,
            f"branch.{str(branch_def.get('name', '') or 'collect').strip() or 'collect'}",
        ),
        "mode": _template_str_param(collect, "mode", config, "concat") or "concat",
        "axis": _template_str_param(collect, "axis", config, "feature") or "feature",
        "rows": _template_int_param(collect, "rows", config, default_rows),
        "slice_dim": _template_int_param(collect, "slice_dim", config, default_slice_dim),
        "num_slices": _template_int_param(collect, "num_slices", config, len(layers)),
        "dtype": dtype,
        "bytes_per_elem": _dtype_size_bytes(dtype),
    }


def _resolve_branch_weight_ref_alias(weight_key: str) -> str:
    return {
        "branch_norm_gamma": "gamma",
        "branch_norm_beta": "beta",
        "branch_fc1_w": "W",
        "branch_fc1_b": "bias",
        "branch_fc2_w": "W",
        "branch_fc2_b": "bias",
    }.get(str(weight_key or ""), str(weight_key or ""))


def _build_block_branch_plan(block_def: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    body_ids = _template_section_id_map(block_def.get("body", {}).get("ops", []))
    header_ids = _template_section_id_map(block_def.get("header", []))
    footer_ids = _template_section_id_map(block_def.get("footer", []))
    section_ids = {
        "header": header_ids,
        "body": body_ids,
        "footer": footer_ids,
    }
    footer_stitches: List[Dict[str, Any]] = []
    for item in _normalize_template_op_items(block_def.get("footer", []), include_inactive=True):
        op_name = str(item.get("op", "") or "").strip()
        inputs = item.get("inputs")
        has_branch_input = isinstance(inputs, list) and any(
            isinstance(value, str) and value.strip().startswith("branch.")
            for value in inputs
        )
        if op_name.startswith("branch_") or has_branch_input:
            footer_stitches.append(copy.deepcopy(item))

    plan: List[Dict[str, Any]] = []
    for branch in _normalize_block_branches(block_def):
        producer = branch.get("producer") if isinstance(branch.get("producer"), dict) else {}
        collect = branch.get("collect") if isinstance(branch.get("collect"), dict) else {}
        tap = copy.deepcopy(branch.get("tap", {})) if isinstance(branch.get("tap"), dict) else {}
        resolved_layers = _resolve_branch_layers(branch, config)
        tap_ref = _parse_template_value_ref(tap.get("from"))
        if tap_ref is not None:
            declared = section_ids.get(tap_ref["section"], {})
            if tap_ref["op_id"] not in declared:
                raise RuntimeError(
                    f"Branch '{branch.get('name', '')}' taps '{tap_ref['ref']}', "
                    f"but that op id is not declared in the template."
                )
        plan.append(
            {
                "name": str(branch.get("name", "") or ""),
                "kind": str(branch.get("kind", "fixed_branch") or "fixed_branch"),
                "status": str(branch.get("status", "active") or "active"),
                "tap": tap,
                "tap_ref": tap_ref,
                "layers": resolved_layers,
                "producer_ops": _extract_template_ops(
                    producer.get("ops", []),
                    include_inactive=True,
                ),
                "producer_items": _normalize_template_op_items(
                    producer.get("ops", []),
                    include_inactive=True,
                ),
                "collect": copy.deepcopy(collect),
                "collect_contract": _resolve_branch_collect_contract(
                    {
                        "name": branch.get("name"),
                        "collect": collect,
                        "layers": resolved_layers,
                    },
                    config,
                ),
                "stitches": copy.deepcopy(footer_stitches),
            }
        )
    return plan


def _collect_template_ops(template: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> List[str]:
    if not isinstance(template, dict):
        return []
    cfg = config if isinstance(config, dict) else {}
    block_types = template.get("block_types") if isinstance(template.get("block_types"), dict) else {}
    sequence = template.get("sequence") if isinstance(template.get("sequence"), list) else []
    collected: List[str] = []
    for block_name in sequence:
        block = block_types.get(block_name)
        if not isinstance(block, dict):
            continue
        collected.extend(_extract_template_ops(block.get("header", [])))
        collected.extend(_collect_body_ops_for_validation(block.get("body", {}), cfg))
        for branch in _build_block_branch_plan(block, cfg):
            collected.extend(branch.get("producer_ops", []))
        collected.extend(_extract_template_ops(block.get("footer", [])))
    return _dedupe_preserve_order(collected)


def _template_declares_logits(template: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
    return "logits" in _collect_template_ops(template, config)


def _template_uses_rope(template: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
    contract = template.get("contract") if isinstance(template.get("contract"), dict) else {}
    attention_contract = contract.get("attention_contract") if isinstance(contract.get("attention_contract"), dict) else {}
    rope_layout = _normalize_rope_layout_value(attention_contract.get("rope_layout"))
    if rope_layout == "none":
        return False
    if rope_layout in {"split", "pairwise", "partial_pairwise_concat", "multi_section_1d", "multi_section_2d"}:
        return True
    flags = template.get("flags", {}) if isinstance(template.get("flags"), dict) else {}
    rope_flag = str(flags.get("rope", "") or "").strip().lower()
    if rope_flag in {"rope", "rope_qk", "partial_rope_concat", "partial_pairwise_concat"}:
        return True
    template_ops = _collect_template_ops(template, config)
    return bool({"rope_qk", "rope_q", "mrope_qk", "partial_rope_concat"} & set(template_ops))


def _template_uses_kv_cache(template: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
    contract = template.get("contract") if isinstance(template.get("contract"), dict) else {}
    attention_contract = contract.get("attention_contract") if isinstance(contract.get("attention_contract"), dict) else {}
    kv_layout = str(attention_contract.get("kv_layout", "") or "").strip().lower()
    if kv_layout in {"none", "ephemeral_full_context", "ephemeral", "encoder_context"}:
        return False
    if kv_layout:
        return True
    template_ops = _collect_template_ops(template, config)
    return bool({"attn", "attn_sliding", "attn_shared_kv", "attn_sliding_shared_kv"} & set(template_ops))


def _resolve_decode_kv_cache_dtype(template: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> str:
    cfg = config if isinstance(config, dict) else {}
    explicit = str(cfg.get("decode_kv_cache_dtype", "") or "").strip().lower()
    if explicit in {"fp16", "f16"}:
        return "fp16"
    contract = template.get("contract") if isinstance(template.get("contract"), dict) else {}
    attention_contract = contract.get("attention_contract") if isinstance(contract.get("attention_contract"), dict) else {}
    declared = str(attention_contract.get("decode_kv_cache_dtype", "") or "").strip().lower()
    if declared in {"fp16", "f16"}:
        return "fp16"
    return "fp32"


def _backfill_template_runtime_flags(manifest: Dict[str, Any]) -> None:
    config = manifest.get("config")
    if not isinstance(config, dict):
        config = {}
        manifest["config"] = config
    template = manifest.get("template") if isinstance(manifest.get("template"), dict) else {}
    config.setdefault("_template_has_logits", _template_declares_logits(template, config))
    config.setdefault("_template_uses_kv_cache", _template_uses_kv_cache(template, config))
    config.setdefault("_template_uses_rope", _template_uses_rope(template, config))
    config.setdefault("decode_kv_cache_dtype", _resolve_decode_kv_cache_dtype(template, config))


def _backfill_vision_contract_config(manifest: Dict[str, Any]) -> None:
    config = manifest.get("config")
    if not isinstance(config, dict):
        config = {}
        manifest["config"] = config
    template = manifest.get("template") if isinstance(manifest.get("template"), dict) else {}
    contract = template.get("contract") if isinstance(template.get("contract"), dict) else {}
    vision_contract = contract.get("vision_contract") if isinstance(contract.get("vision_contract"), dict) else {}
    vision_position_contract = contract.get("vision_position_contract") if isinstance(contract.get("vision_position_contract"), dict) else {}
    attention_contract = contract.get("attention_contract") if isinstance(contract.get("attention_contract"), dict) else {}

    image_size = int(vision_contract.get("image_size", 0) or 0)
    patch_size = int(vision_contract.get("patch_size", 0) or 0)
    vision_channels = int(vision_contract.get("channels", 3) or 3)
    rope_layout = (
        attention_contract.get("rope_layout")
        if attention_contract.get("rope_layout") is not None
        else vision_position_contract.get("rope_layout")
    )
    rope_mode = attention_contract.get("rope_mode")
    rope_param_mode = attention_contract.get("rope_param_mode")
    position_rank = vision_position_contract.get("position_rank")

    if image_size > 0:
        config.setdefault("image_size", image_size)
    if patch_size > 0:
        config.setdefault("patch_size", patch_size)
    config.setdefault("vision_channels", vision_channels)
    if rope_layout is not None and str(rope_layout).strip():
        config.setdefault("rope_layout", str(rope_layout))
    if rope_mode is not None and str(rope_mode).strip():
        config.setdefault("rope_mode", str(rope_mode))
    if rope_param_mode is not None and str(rope_param_mode).strip():
        config.setdefault("rope_param_mode", str(rope_param_mode))
    if position_rank is not None:
        config.setdefault("position_rank", int(position_rank))

    if image_size > 0 and patch_size > 0:
        patches_h = image_size // patch_size
        patches_w = image_size // patch_size
        config.setdefault("vision_num_patches", patches_h * patches_w)
        config.setdefault("patch_dim", vision_channels * patch_size * patch_size)


def _resolve_template_quant_aliases(
    body_def: Any,
    config: Dict[str, Any],
    layer_idx: int,
) -> Dict[str, str]:
    if not isinstance(body_def, dict):
        return {}

    aliases: Dict[str, str] = {}
    common = body_def.get("quant_aliases_common")
    if isinstance(common, dict):
        for dst, src in common.items():
            dst_key = str(dst or "").strip()
            src_key = str(src or "").strip()
            if dst_key and src_key:
                aliases[dst_key] = src_key

    by_kind = body_def.get("quant_aliases_by_kind")
    if not isinstance(by_kind, dict):
        return aliases

    layer_kind = _resolve_config_layer_kind(
        config,
        layer_idx,
        kind_key=str(body_def.get("kind_config_key", "layer_kinds") or "layer_kinds"),
        interval_key=str(body_def.get("interval_config_key", "full_attention_interval") or "full_attention_interval"),
        periodic_kind=str(body_def.get("periodic_kind", "full_attention") or "full_attention"),
        default_kind=str(body_def.get("default_kind", "recurrent") or "recurrent"),
    )
    scoped = by_kind.get(layer_kind)
    if isinstance(scoped, dict):
        for dst, src in scoped.items():
            dst_key = str(dst or "").strip()
            src_key = str(src or "").strip()
            if dst_key and src_key:
                aliases[dst_key] = src_key
    return aliases


def _apply_layer_quant_aliases(
    layer_quant: Dict[str, Any],
    body_def: Any,
    config: Dict[str, Any],
    layer_idx: int,
) -> Dict[str, Any]:
    effective = dict(layer_quant or {})
    aliases = _resolve_template_quant_aliases(body_def, config, layer_idx)
    for dst, src in aliases.items():
        if dst not in effective and src in effective:
            effective[dst] = effective[src]

    # Canonical IR weight slots are intentionally older/stable names (w1/w2/w3),
    # while newer safetensors converters often emit semantic names
    # (mlp_gate/mlp_down/mlp_up). Keep this fallback in the lowerer so templates
    # do not need brittle per-source alias variants just to preserve dtype
    # propagation.
    canonical_aliases = {
        "wq": ("attn_q", "mla_q_proj"),
        "w1": ("ffn_gate", "mlp_gate"),
        "w2": ("ffn_down", "mlp_down"),
        "w3": ("ffn_up", "mlp_up"),
        "wo": ("attn_o", "out_proj", "mla_out_proj"),
        "mla_kv_a_proj": ("kv_a_proj",),
        "mla_kv_a_norm": ("kv_a_norm",),
        "mla_kv_b_proj": ("kv_b_proj",),
        "moe_expert_gate": ("expert_gate",),
        "moe_expert_up": ("expert_up",),
        "moe_expert_down": ("expert_down",),
        "moe_shared_gate": ("shared_gate",),
        "moe_shared_up": ("shared_up",),
        "moe_shared_down": ("shared_down",),
    }
    for dst, candidates in canonical_aliases.items():
        if dst in effective:
            continue
        for src in candidates:
            if src in effective:
                effective[dst] = effective[src]
                break

    return effective

def compute_matmul_dims(op_name: str, config: Dict) -> Tuple[Optional[int], Optional[int]]:
    """Compute output/input dims for matmul-like ops (gemv/gemm) and quantize ops."""
    embed = config.get("embed_dim", 896)
    heads = config.get("num_heads", 14)
    kv_heads = config.get("num_kv_heads", 2)
    head_dim = config.get("head_dim", 64)
    inter = config.get("intermediate_size", config.get("intermediate_dim", 4864))
    vocab = config.get("vocab_size", 0)
    patch_dim = int(config.get("patch_dim", 0) or 0)
    projector_in = int(config.get("projector_in_dim", embed * int(config.get("spatial_merge_factor", 1) or 1)) or 0)
    projector_hidden = int(config.get("projector_hidden_dim", projector_in) or 0)
    projector_out = int(config.get("projector_out_dim", config.get("projection_dim", embed)) or 0)

    q_gate_proj = int(config.get("q_gate_proj_dim", config.get("attn_q_gate_proj_dim", 0)) or 0)
    attn_gate_dim = int(config.get("attn_gate_dim", 0) or 0)
    if op_name in ("q_proj",):
        return heads * head_dim, embed
    if op_name in ("assistant_pre_projection",):
        return embed, int(config.get("backbone_hidden_size", embed) or embed)
    if op_name in ("assistant_post_projection",):
        return int(config.get("backbone_hidden_size", embed) or embed), embed
    if op_name in ("qkv_packed_proj",):
        return (heads * head_dim) + 2 * (kv_heads * head_dim), embed
    if op_name in ("q_gate_proj",):
        if q_gate_proj <= 0:
            q_gate_proj = 2 * (heads * head_dim)
        return q_gate_proj, embed
    if op_name in ("k_proj", "v_proj"):
        return kv_heads * head_dim, embed
    recurrent_q = int(config.get("q_dim", 0) or 0)
    recurrent_k = int(config.get("k_dim", 0) or 0)
    recurrent_v = int(config.get("v_dim", 0) or 0)
    recurrent_gate = int(config.get("gate_dim", 0) or 0)
    recurrent_inner = int(config.get("ssm_inner_size", 0) or 0)
    if op_name in ("recurrent_qkv_proj",):
        packed = recurrent_q + recurrent_k + recurrent_v
        return (packed or None), embed
    if op_name in ("recurrent_gate_proj",):
        return (recurrent_inner or None), embed
    if op_name in ("recurrent_alpha_proj", "recurrent_beta_proj"):
        return (recurrent_gate or None), embed
    attn_out = config.get("attn_out_dim", heads * head_dim)
    if int(config.get("kv_lora_rank", 0) or 0) > 0 and int(config.get("v_head_dim", 0) or 0) > 0:
        attn_out = heads * int(config.get("v_head_dim", 0) or 0)
    if op_name in ("out_proj", "attn_proj"):
        return embed, attn_out
    if op_name in ("recurrent_out_proj",):
        return embed, int(config.get("ssm_inner_size", attn_out))
    if op_name in ("mamba_in_proj",):
        return int(config.get("mamba_projection_size", 0) or 0) or None, embed
    if op_name in ("mamba_out_proj",):
        return embed, int(config.get("ssm_inner_size", config.get("mamba_intermediate_size", attn_out)) or attn_out)
    if op_name in ("moe_router",):
        return int(config.get("n_routed_experts", config.get("num_experts", 0)) or 0) or None, embed
    if op_name in ("kv_a_proj",):
        return int(config.get("kv_lora_rank", 0) or 0) + int(config.get("qk_rope_head_dim", 0) or 0), embed
    if op_name in ("moe_relu2_expert_mlp", "moe_swiglu_expert_mlp"):
        return int(config.get("moe_intermediate_size", inter) or inter), embed
    if op_name in ("shared_relu2_expert_mlp", "shared_swiglu_expert_mlp"):
        return int(config.get("moe_shared_expert_intermediate_size", config.get("moe_intermediate_size", inter)) or inter), embed
    if op_name in ("mlp_gate_up",):
        return inter * 2, embed
    if op_name in ("mlp_up",):
        return inter, embed
    if op_name in ("mlp_gate",):
        return inter, embed
    if op_name in ("mlp_down",):
        return embed, inter
    if op_name in ("logits",):
        return vocab, embed
    if op_name in ("patch_proj", "patch_proj_aux"):
        return embed, patch_dim
    if op_name in ("projector_fc1",):
        return projector_hidden, projector_in
    if op_name in ("projector_fc2",):
        return projector_out, projector_hidden
    if op_name in ("branch_fc1",):
        return projector_hidden, projector_in
    if op_name in ("branch_fc2",):
        return projector_out, projector_hidden
    # Quantize ops: _input_dim is the size to quantize
    # quantize_input_0/1/2: quantize embed_dim (rmsnorm output before projections)
    # quantize_out_proj_input: quantize embed_dim (attention output)
    # quantize_mlp_down_input: quantize intermediate_size (swiglu output)
    # quantize_final_output: quantize embed_dim (footer rmsnorm output before logits)
    if op_name in ("quantize_input_0", "quantize_input_1", "quantize_input_2"):
        return embed, embed  # output_dim not used, but _input_dim = embed
    if op_name == "quantize_final_output":
        projector_in_dim = int(config.get("projector_in_dim", 0) or 0)
        if projector_in_dim > 0:
            return projector_in_dim, projector_in_dim
        return embed, embed  # output_dim not used, but _input_dim = embed
    if op_name in ("quantize_out_proj_input",):
        return attn_out, attn_out  # output_dim not used, but _input_dim = attn_out_dim
    if op_name in ("split_q_gate",):
        return attn_out, (attn_gate_dim or attn_out)
    if op_name in ("quantize_recurrent_out_proj_input",):
        recurrent_inner = int(config.get("ssm_inner_size", attn_out))
        return recurrent_inner, recurrent_inner
    if op_name in ("quantize_mamba_out_proj_input",):
        mamba_inner = int(config.get("mamba_intermediate_size", config.get("ssm_inner_size", attn_out)))
        return mamba_inner, mamba_inner
    if op_name in ("quantize_mlp_down_input",):
        return inter, inter  # _input_dim = intermediate_size
    return None, None


def _config_layer_int(config: Dict, key: str, layer: int, default: int) -> int:
    vals = config.get(key)
    if vals is None:
        return int(default)
    if not isinstance(vals, list):
        raise ValueError(f"{key} must be a per-layer integer list, got {type(vals).__name__}")
    if not 0 <= layer < len(vals):
        raise ValueError(f"{key} has no value for layer {layer}; length={len(vals)}")
    try:
        return int(vals[layer])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}[{layer}] must be an integer, got {vals[layer]!r}") from exc


def apply_layer_attention_dims(op_name: str, params: Dict, layer: int, config: Dict) -> None:
    """Apply manifest/circuit dimensions without inferring a model family."""
    if layer < 0:
        return
    embed_dim = int(config.get("embed_dim", 0) or 0)
    num_heads = int(config.get("num_heads", config.get("num_attention_heads", 1)) or 1)
    num_kv_heads = int(config.get("num_kv_heads", config.get("num_key_value_heads", num_heads)) or num_heads)
    q_head_dim = _config_layer_int(config, "layer_q_head_dim", layer, int(config.get("head_dim", 0) or 0))
    k_head_dim = _config_layer_int(config, "layer_k_head_dim", layer, q_head_dim)
    v_head_dim = _config_layer_int(config, "layer_v_head_dim", layer, k_head_dim)
    q_dim = _config_layer_int(
        config,
        "layer_q_dim",
        layer,
        int(config.get("attn_out_dim", num_heads * q_head_dim) or (num_heads * q_head_dim)),
    )
    k_dim = num_kv_heads * k_head_dim
    v_dim = num_kv_heads * v_head_dim
    rotary_default = int(config.get("mrope_n_dims", config.get("rotary_dim", q_head_dim)) or q_head_dim)
    rotary_dim = _config_layer_int(config, "layer_rotary_dim", layer, rotary_default)
    sliding_window = _config_layer_int(config, "layer_sliding_window", layer, int(config.get("sliding_window", 0) or 0))
    rope_kinds = config.get("layer_rope_kind")
    rope_kind = str(rope_kinds[layer]) if isinstance(rope_kinds, list) and 0 <= layer < len(rope_kinds) else ("swa" if sliding_window > 0 else "full")
    if rope_kind == "swa":
        rope_freq_base = float(config.get("rope_theta_swa", 10000.0) or 10000.0)
    else:
        rope_freq_base = float(config.get("rope_theta", 1000000.0) or 1000000.0)

    if op_name == "q_gate_proj":
        params["_output_dim"] = int(
            config.get("q_gate_proj_dim", config.get("attn_q_gate_proj_dim", q_dim * 2))
            or (q_dim * 2)
        )
        params["_input_dim"] = embed_dim
    elif op_name == "q_proj":
        params["_output_dim"] = q_dim
        params["_input_dim"] = embed_dim
        params["output_dim"] = q_dim
    elif op_name == "k_proj":
        params["_output_dim"] = k_dim
        params["_input_dim"] = embed_dim
        params["output_dim"] = k_dim
    elif op_name == "v_proj":
        params["_output_dim"] = v_dim
        params["_input_dim"] = embed_dim
        params["output_dim"] = v_dim
    elif op_name == "out_proj":
        params["_output_dim"] = embed_dim
        params["_input_dim"] = q_dim
        params["input_dim"] = q_dim
    elif op_name == "quantize_out_proj_input":
        params["_output_dim"] = q_dim
        params["_input_dim"] = q_dim
        params["input_dim"] = q_dim
    elif op_name in (
        "qk_norm",
        "q_norm",
        "rope_qk",
        "rope_q",
        "kv_cache_store",
        "attn",
        "attn_sliding",
        "attn_shared_kv",
        "attn_sliding_shared_kv",
    ):
        params["head_dim"] = q_head_dim
        params["q_head_dim"] = q_head_dim
        params["k_head_dim"] = k_head_dim
        params["v_head_dim"] = v_head_dim
        params["q_dim"] = q_dim
        if op_name in ("q_norm", "rope_q", "attn_shared_kv", "attn_sliding_shared_kv"):
            params["k_dim"] = q_dim
            params["v_dim"] = q_dim
            params["num_kv_heads"] = num_heads
        else:
            params["k_dim"] = k_dim
            params["v_dim"] = v_dim
        params["rotary_dim"] = rotary_dim
        params["n_dims"] = rotary_dim
        params["rope_freq_base"] = rope_freq_base
        use_freq_factors = int(config.get("use_rope_freq_factors", 0) or 0)
        if isinstance(rope_kinds, list):
            use_freq_factors = 1 if rope_kind == "full" else 0
        params["use_rope_freq_factors"] = use_freq_factors if op_name in ("rope_qk", "rope_q") else 0
        if sliding_window > 0:
            params["sliding_window"] = sliding_window
    elif op_name == "v_norm":
        params["head_dim"] = v_head_dim
        params["v_head_dim"] = v_head_dim
        params["embed_dim"] = v_dim
        params["d_model"] = v_dim
        params["aligned_embed_dim"] = v_dim
        params["_input_dim"] = v_dim
        params["_output_dim"] = v_dim


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) // alignment * alignment


def load_kernel_registry() -> Dict:
    """Load the v8-local kernel registry."""
    registry_path = V8_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"
    with open(registry_path, 'r') as f:
        registry = json.load(f)

    kernels = registry.get("kernels", [])
    if not isinstance(kernels, list):
        kernels = []
        registry["kernels"] = kernels

    overlay_dir = V8_ROOT / "kernel_maps"
    if not overlay_dir.exists():
        return registry

    by_id = {
        str(kernel.get("id", "") or ""): kernel
        for kernel in kernels
        if isinstance(kernel, dict) and str(kernel.get("id", "") or "").strip()
    }
    for overlay_path in sorted(overlay_dir.glob("*.json")):
        upper_name = overlay_path.name.upper()
        if upper_name.startswith("KERNEL_") or overlay_path.name in {"kernel_bindings.json", "kernel_bindings.overlay.json"}:
            continue
        try:
            with open(overlay_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        kernel_id = str(doc.get("id", "") or "").strip()
        kernel_op = str(doc.get("op", "") or "").strip()
        if not kernel_id or not kernel_op:
            continue
        doc = copy.deepcopy(doc)
        doc.setdefault("name", kernel_id)
        doc.setdefault("_source_file", overlay_path.name)
        by_id[kernel_id] = doc

    registry["kernels"] = [by_id[key] for key in sorted(by_id.keys())]
    return registry


def load_manifest(manifest_path: Path) -> Dict:
    """Load weights manifest with template and quant summary."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def _merge_external_config(manifest: Dict, manifest_path: Path) -> None:
    """Merge optional config.json into manifest["config"] (fill missing keys only)."""
    config = manifest.get("config", {}) or {}
    cfg_path = Path(manifest_path).parent / "config.json"
    if not cfg_path.exists():
        manifest["config"] = config
        return
    try:
        with open(cfg_path, "r") as f:
            external = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load config.json ({cfg_path}): {e}")
        manifest["config"] = config
        return

    # Map HF-style config keys to internal names
    mapped = {
        "embed_dim": external.get("hidden_size"),
        "num_layers": external.get("num_hidden_layers"),
        "num_heads": external.get("num_attention_heads"),
        "num_kv_heads": external.get("num_key_value_heads"),
        "context_length": external.get("max_position_embeddings"),
        "max_seq_len": external.get("max_position_embeddings"),
        "rms_eps": external.get("rms_norm_eps"),
        "rope_theta": external.get("rope_theta"),
        "rotary_dim": external.get("rotary_dim"),
        "rope_scaling_type": external.get("rope_scaling_type"),
        "rope_scaling_factor": external.get("rope_scaling_factor"),
        "rope_layout": external.get("rope_layout"),
        "rope_original_context_length": external.get("rope_original_context_length"),
        "rope_beta_fast": external.get("rope_beta_fast"),
        "rope_beta_slow": external.get("rope_beta_slow"),
        "rope_attn_factor": external.get("rope_attn_factor"),
        "attn_out_dim": external.get("attn_out_dim"),
        "sliding_window": external.get("sliding_window"),
        "intermediate_size": external.get("intermediate_size"),
        "vocab_size": external.get("vocab_size"),
        "model": external.get("model_type"),
        "model_name": external.get("model_name"),
        "finetune": external.get("finetune"),
        "chat_template": external.get("chat_template"),
    }

    for k, v in mapped.items():
        if v is None:
            continue
        config.setdefault(k, v)

    manifest["config"] = config


def _normalize_manifest_config(config: Dict) -> Dict:
    """Normalize aliases and derive canonical dimensions for IR/codegen."""
    out = dict(config or {})

    def _pick(*keys, default=None):
        for key in keys:
            if key in out and out[key] is not None:
                return out[key]
        return default

    embed_dim = _pick("embed_dim", "hidden_size", "n_embd", "d_model")
    num_heads = _pick("num_heads", "num_attention_heads", "n_head")
    num_kv_heads = _pick("num_kv_heads", "num_key_value_heads", "n_kv_head", default=num_heads)
    head_dim = _pick("head_dim")
    context_length = _pick(
        "context_length",
        "context_len",
        "max_seq_len",
        "max_position_embeddings",
        "context_window",
    )

    if embed_dim is not None:
        embed_dim = int(embed_dim)
        out["embed_dim"] = embed_dim
    if num_heads is not None:
        num_heads = int(num_heads)
        out["num_heads"] = num_heads
    if num_kv_heads is not None:
        out["num_kv_heads"] = int(num_kv_heads)

    if head_dim is None and embed_dim is not None and num_heads:
        if embed_dim % int(num_heads) == 0:
            head_dim = embed_dim // int(num_heads)
    if head_dim is not None:
        out["head_dim"] = int(head_dim)

    if context_length is not None:
        out["context_length"] = int(context_length)
        out.setdefault("max_seq_len", int(context_length))

    image_size = _pick("image_size")
    image_height = _pick("image_height", "image_h", default=image_size)
    image_width = _pick("image_width", "image_w", default=image_size)
    if image_size is not None:
        out["image_size"] = int(image_size)
    if image_height is not None:
        out["image_height"] = int(image_height)
    if image_width is not None:
        out["image_width"] = int(image_width)

    # RoPE config (fallbacks remain model-agnostic and are overridden by converter when present)
    out["rope_theta"] = float(_pick("rope_theta", "rope_base", "theta", default=10000.0))
    out["rms_eps"] = float(_pick("rms_eps", "rms_norm_eps", default=1e-6))
    out["rotary_dim"] = int(_pick("rotary_dim", default=out.get("head_dim", 64)))
    out["rope_scaling_type"] = str(_pick("rope_scaling_type", default="none"))
    out["rope_scaling_factor"] = float(_pick("rope_scaling_factor", default=1.0))
    rope_layout_value = _pick("rope_layout")
    if rope_layout_value is not None and str(rope_layout_value).strip():
        out["rope_layout"] = str(rope_layout_value)
    rope_param_mode_value = _pick("rope_param_mode")
    if rope_param_mode_value is not None and str(rope_param_mode_value).strip():
        out["rope_param_mode"] = str(rope_param_mode_value)
    out["rope_original_context_length"] = int(
        _pick("rope_original_context_length", default=out.get("context_length", 0))
    )
    out["rope_beta_fast"] = float(_pick("rope_beta_fast", default=0.0))
    out["rope_beta_slow"] = float(_pick("rope_beta_slow", default=0.0))
    out["rope_attn_factor"] = float(_pick("rope_attn_factor", default=1.0))

    # Clamp rotary_dim to head_dim for safety
    if out.get("head_dim") is not None:
        head_dim = int(out["head_dim"])
        if out.get("rotary_dim", head_dim) > head_dim:
            out["rotary_dim"] = head_dim
    q_gate_proj_dim = _pick("q_gate_proj_dim", "attn_q_gate_proj_dim")
    if q_gate_proj_dim is not None:
        out["q_gate_proj_dim"] = int(q_gate_proj_dim)
    ssm_state = _pick("ssm_state_size")
    ssm_groups = _pick("ssm_group_count", "n_groups")
    ssm_heads = _pick("ssm_time_step_rank", "mamba_num_heads")
    ssm_inner = _pick("ssm_inner_size")
    mamba_head_dim_value = _pick("mamba_head_dim")
    projection_layout = str(out.get("ssm_projection_layout", "qkv_gate") or "qkv_gate").strip().lower()
    if projection_layout not in {"qkv_gate", "mamba2_v_qk_dt"}:
        raise ValueError(f"unsupported ssm_projection_layout: {projection_layout!r}")
    if ssm_inner is None and projection_layout == "mamba2_v_qk_dt" and ssm_heads is not None and mamba_head_dim_value is not None:
        ssm_inner = int(ssm_heads) * int(mamba_head_dim_value)
    ssm_conv_kernel = _pick("ssm_conv_kernel", "conv_kernel")
    if ssm_state is not None:
        out["ssm_state_size"] = int(ssm_state)
    if ssm_groups is not None:
        out["ssm_group_count"] = int(ssm_groups)
    if ssm_heads is not None:
        out["ssm_time_step_rank"] = int(ssm_heads)
        if projection_layout == "mamba2_v_qk_dt":
            out["mamba_num_heads"] = int(ssm_heads)
    if mamba_head_dim_value is not None:
        out["mamba_head_dim"] = int(mamba_head_dim_value)
    if ssm_inner is not None:
        out["ssm_inner_size"] = int(ssm_inner)
    if ssm_conv_kernel is not None:
        out["ssm_conv_kernel"] = int(ssm_conv_kernel)
        history_mode = str(out.get("ssm_conv_history_mode", "kernel_width_minus_one") or "kernel_width_minus_one")
        if history_mode == "kernel_width":
            out["ssm_conv_history"] = max(int(ssm_conv_kernel), 0)
        elif history_mode == "kernel_width_minus_one":
            out["ssm_conv_history"] = max(int(ssm_conv_kernel) - 1, 0)
        else:
            raise ValueError(f"unsupported ssm_conv_history_mode: {history_mode!r}")
    if None not in (ssm_state, ssm_groups, ssm_heads, ssm_inner):
        if projection_layout == "mamba2_v_qk_dt":
            # Runtime dt clamp follows time_step_limit. time_step_min/max are
            # initialization ranges and must not clamp forward activations.
            # Use (0, 0) to disable the CK clamp for PyTorch's default (0, inf).
            limit = out.get("time_step_limit")
            if isinstance(limit, (list, tuple)) and len(limit) >= 2:
                dt_lo = float(limit[0] or 0.0)
                try:
                    dt_hi = float(limit[1])
                except Exception:
                    dt_hi = float("inf")
                if dt_hi == float("inf"):
                    dt_lo, dt_hi = 0.0, 0.0
            else:
                dt_lo, dt_hi = 0.0, 0.0
            out["mamba_dt_min"] = dt_lo
            out["mamba_dt_max"] = dt_hi
            if int(ssm_groups) > 0:
                out["mamba_norm_group_size"] = int(ssm_inner) // int(ssm_groups)
            v_dim = int(ssm_inner)
            q_dim = int(ssm_state) * int(ssm_groups)
            conv_dim = v_dim + q_dim + q_dim
            out["q_dim"] = q_dim
            out["k_dim"] = q_dim
            out["v_dim"] = v_dim
            out["gate_dim"] = v_dim
            out["mamba_intermediate_size"] = v_dim
            out["mamba_conv_dim"] = conv_dim
            out["mamba_projection_size"] = v_dim + conv_dim + int(ssm_heads)
            out["ssm_conv_channels"] = conv_dim
            out.setdefault("ssm_conv_history", max(int(out.get("ssm_conv_kernel", 0) or 0), 0))
            out.setdefault("rope_freq_base", float(out.get("rope_theta", 10000.0) or 10000.0))
            out.setdefault("use_rope_freq_factors", 0)
        else:
            q_dim = int(ssm_state) * int(ssm_groups)
            v_dim = int(ssm_inner)
            out["q_dim"] = q_dim
            out["k_dim"] = q_dim
            out["v_dim"] = v_dim
            out["gate_dim"] = int(ssm_heads)
            out["ssm_conv_channels"] = q_dim + q_dim + v_dim
        out["recurrent_num_heads"] = int(ssm_heads)
        out["recurrent_head_dim"] = int(v_dim // int(ssm_heads)) if int(ssm_heads) else int(ssm_state)
        state_layout = str(out.get("recurrent_state_layout", "grouped_state") or "grouped_state")
        if state_layout == "heads_head_dim_state":
            out["recurrent_state_heads"] = int(ssm_heads)
            out["recurrent_state_rows"] = int(out["recurrent_head_dim"])
            out["recurrent_state_cols"] = int(ssm_state)
        elif state_layout != "grouped_state":
            raise ValueError(f"unsupported recurrent_state_layout: {state_layout!r}")
    attn_out = _pick("attn_out_dim", default=(out.get("num_heads", 0) * out.get("head_dim", 0)))
    if attn_out is not None:
        out["attn_out_dim"] = int(attn_out)
    if out.get("q_gate_proj_dim") is None and out.get("attn_out_dim") is not None:
        out["q_gate_proj_dim"] = int(out["attn_out_dim"]) * 2
    if out.get("attn_gate_dim") is None and out.get("q_gate_proj_dim") is not None and out.get("attn_out_dim") is not None:
        out["attn_gate_dim"] = max(int(out["q_gate_proj_dim"]) - int(out["attn_out_dim"]), 0)
    if not out.get("attn_gate_dim") and out.get("attn_out_dim") is not None:
        out["attn_gate_dim"] = int(out["attn_out_dim"])
    out.setdefault("num_seqs", 1)
    return out


def _template_sequence(template: Dict[str, Any]) -> List[str]:
    sequence = template.get("sequence", [])
    if not isinstance(sequence, list):
        return []
    return [str(name) for name in sequence if str(name).strip()]


def _template_block_def(template: Dict[str, Any], block_name: str) -> Dict[str, Any]:
    block_types = template.get("block_types", {})
    if not isinstance(block_types, dict):
        return {}
    block_def = block_types.get(block_name, {})
    return block_def if isinstance(block_def, dict) else {}


def _single_block_template(template: Dict[str, Any], block_name: str) -> Dict[str, Any]:
    block_def = copy.deepcopy(_template_block_def(template, block_name))
    if not block_def:
        raise RuntimeError(f"Template block '{block_name}' is missing from block_types")

    out = copy.deepcopy(template)
    out["sequence"] = [block_name]
    out["block_types"] = {block_name: block_def}
    return out


def _block_config_overrides(template: Dict[str, Any], block_name: str) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    template_block_configs = template.get("block_configs", {})
    if isinstance(template_block_configs, dict):
        cfg = template_block_configs.get(block_name)
        if isinstance(cfg, dict):
            overrides.update(copy.deepcopy(cfg))

    block_def = _template_block_def(template, block_name)
    block_cfg = block_def.get("config")
    if isinstance(block_cfg, dict):
        overrides.update(copy.deepcopy(block_cfg))

    return overrides


def build_block_manifest(manifest: Dict[str, Any], block_name: str) -> Dict[str, Any]:
    manifest = _hydrate_manifest_template(copy.deepcopy(manifest))
    template = manifest.get("template", {})
    if not isinstance(template, dict):
        raise RuntimeError("Manifest template is missing or invalid")

    block_manifest = copy.deepcopy(manifest)
    block_manifest["template"] = _single_block_template(template, block_name)

    merged_config = _normalize_manifest_config(block_manifest.get("config", {}))
    merged_config.update(_block_config_overrides(template, block_name))
    block_manifest["config"] = _normalize_manifest_config(merged_config)
    block_manifest["block_name"] = block_name

    return block_manifest


def build_block_manifests(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    manifest = _hydrate_manifest_template(copy.deepcopy(manifest))
    template = manifest.get("template", {})
    if not isinstance(template, dict):
        return []

    blocks: List[Dict[str, Any]] = []
    for index, block_name in enumerate(_template_sequence(template)):
        block_manifest = build_block_manifest(manifest, block_name)
        block_manifest["block_index"] = index
        blocks.append(block_manifest)
    return blocks


def build_stitch_plan(manifest: Dict[str, Any]) -> Dict[str, Any]:
    manifest = _hydrate_manifest_template(copy.deepcopy(manifest))
    template = manifest.get("template", {})
    if not isinstance(template, dict):
        raise RuntimeError("Manifest template is missing or invalid")

    sequence = _template_sequence(template)
    template_stitch = template.get("stitch", [])
    edges: List[Dict[str, Any]] = []

    if isinstance(template_stitch, list) and template_stitch:
        for edge in template_stitch:
            if isinstance(edge, dict):
                edges.append(copy.deepcopy(edge))
    else:
        for src, dst in zip(sequence, sequence[1:]):
            edges.append(
                {
                    "from": src,
                    "to": dst,
                    "kind": "sequential",
                    "from_output": "output",
                    "to_input": "input",
                }
            )

    blocks: List[Dict[str, Any]] = []
    for block_manifest in build_block_manifests(manifest):
        blocks.append(
            {
                "name": block_manifest.get("block_name"),
                "index": block_manifest.get("block_index"),
                "config": copy.deepcopy(block_manifest.get("config", {})),
            }
        )

    return {
        "format": "v8-stitch-plan",
        "version": 1,
        "template_name": str(template.get("name", "") or ""),
        "sequence": sequence,
        "blocks": blocks,
        "edges": edges,
    }


def build_template_branch_plan(manifest: Dict[str, Any]) -> Dict[str, Any]:
    manifest = _hydrate_manifest_template(copy.deepcopy(manifest))
    template = manifest.get("template", {})
    if not isinstance(template, dict):
        raise RuntimeError("Manifest template is missing or invalid")

    config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    blocks: List[Dict[str, Any]] = []
    for index, block_name in enumerate(_template_sequence(template)):
        block_def = _template_block_def(template, block_name)
        block_cfg = copy.deepcopy(config)
        block_cfg.update(_block_config_overrides(template, block_name))
        blocks.append(
            {
                "name": block_name,
                "index": index,
                "branches": _build_block_branch_plan(block_def, block_cfg),
            }
        )

    return {
        "format": "v8-template-branch-plan",
        "version": 1,
        "template_name": str(template.get("name", "") or ""),
        "sequence": _template_sequence(template),
        "blocks": blocks,
    }


def _block_artifact_dirname(block_index: int, block_name: str) -> str:
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in block_name)
    return f"{block_index + 1:02d}_{safe_name}"


def write_block_manifests(manifest: Dict[str, Any], output_dir: Path) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Dict[str, Any]] = []
    for block_manifest in build_block_manifests(manifest):
        block_name = str(block_manifest.get("block_name", "") or "")
        block_index = int(block_manifest.get("block_index", 0) or 0)
        block_dir = output_dir / _block_artifact_dirname(block_index, block_name)
        block_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = block_dir / "weights_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(block_manifest, f, indent=2)
        written.append(
            {
                "block_name": block_name,
                "block_index": block_index,
                "artifact_dir": str(block_dir),
                "manifest_path": str(manifest_path),
            }
        )

    return written


def _normalize_rope_layout_value(value: Any) -> str:
    rope_layout = str(value or "").strip().lower()
    aliases = {
        "standard": "split",
        "cos_sin_split": "split",
        "split_half": "split",
        "half": "split",
        "pairwise": "pairwise",
        "interleaved": "pairwise",
        "even_odd": "pairwise",
        "mrope": "multi_section_1d",
        "multi_section": "multi_section_1d",
        "multi_section_text": "multi_section_1d",
    }
    return aliases.get(rope_layout, rope_layout)


def _resolve_rope_qk_kernel(config: Dict, template_kernels: Dict[str, Any]) -> str:
    rope_layout = _normalize_rope_layout_value(config.get("rope_layout"))
    rope_param_mode = str(config.get("rope_param_mode", "") or "").strip().lower()
    override = str(template_kernels.get("rope_qk", "") or "").strip()

    if not override:
        raise RuntimeError(
            "HARD KERNEL RESOLUTION FAULT: rope_qk requires an exact circuit kernel mapping "
            f"for rope_layout={rope_layout!r}, rope_param_mode={rope_param_mode!r}."
        )
    if rope_layout == "pairwise" and "pairwise" not in override.lower():
        raise RuntimeError(
            f"HARD KERNEL RESOLUTION FAULT: pairwise RoPE cannot use {override!r}."
        )
    return override


def _resolve_position_embeddings_kernel(config: Dict, template_kernels: Dict[str, Any]) -> str:
    kernel_spec = template_kernels.get("position_embeddings")
    if isinstance(kernel_spec, dict):
        policy = str(config.get("position_interpolation_policy", "default") or "default").strip().lower()
        selected = kernel_spec.get(policy)
        if not selected:
            raise RuntimeError(
                "HARD KERNEL RESOLUTION FAULT: position_embeddings has no exact circuit "
                f"kernel mapping for interpolation policy {policy!r}."
            )
        return str(selected)
    if not kernel_spec:
        raise RuntimeError(
            "HARD KERNEL RESOLUTION FAULT: position_embeddings requires an exact circuit kernel mapping."
        )
    return str(kernel_spec)


def _attention_contract_is_causal(template: Dict[str, Any], config: Dict[str, Any]) -> bool:
    contract = template.get("contract", {}) if isinstance(template.get("contract"), dict) else {}
    attention_contract = contract.get("attention_contract", {}) if isinstance(contract.get("attention_contract"), dict) else {}

    causal = _coerce_bool(attention_contract.get("causal"))
    if causal is not None:
        return causal

    variant = str(attention_contract.get("attn_variant", "") or "").strip().lower()
    if variant in {"dense_bidirectional", "bidirectional", "full", "full_attention"}:
        return False

    return True


def _resolve_rope_backward_qk_kernel(config: Dict, default_kernel: str = "rope_backward_qk_f32") -> str:
    rope_layout = _normalize_rope_layout_value(config.get("rope_layout"))

    if rope_layout == "pairwise":
        return "rope_backward_qk_pairwise_f32"

    if rope_layout == "split":
        return "rope_backward_qk_f32"

    fallback = str(default_kernel or "").strip()
    if fallback:
        return fallback
    return "rope_backward_qk_f32"


def _resolve_logical_buffer_name(
    planner_buffer: str,
    slot: Any,
    activation_buffers: Dict[str, Dict[str, Any]],
    buffer_name_map: Dict[str, str],
) -> str:
    """
    Preserve template-declared logical slots when they map to concrete lowered
    activation buffers.

    The memory planner tracks physical reuse (for example multiple logical
    scratch slots may alias one physical attention scratch region), but IR/codegen
    still need the logical slot identity declared by the template so graph
    stitching stays template-driven instead of being flattened by Python-side
    alias names.
    """
    if isinstance(slot, str) and slot:
        if slot == "kv_cache":
            return "kv_cache"
        if slot in activation_buffers:
            return slot
    return buffer_name_map.get(planner_buffer, planner_buffer)


def _resolve_planner_io_name(
    io_name: str,
    using_dataflow_io: bool,
    ir_op: Dict[str, Any],
    io_kind: str,
    legacy_name_map: Dict[str, str],
) -> str:
    """
    Resolve the planner lookup name for an op input/output.

    If the IR already exposes canonical dataflow names, preserve them exactly.
    Legacy alias remaps are only for ops that still surface kernel-param names
    directly in IR1.
    """
    if using_dataflow_io:
        return io_name
    declared_slot = _get_declared_dataflow_slot(ir_op, io_kind, io_name, io_name)
    if declared_slot:
        return io_name
    dataflow = ir_op.get("dataflow", {}) if isinstance(ir_op.get("dataflow"), dict) else {}
    declared_ios = dataflow.get(io_kind, {}) if isinstance(dataflow.get(io_kind), dict) else {}
    if len(declared_ios) == 1:
        return next(iter(declared_ios.keys()))
    return legacy_name_map.get(io_name, io_name)


def _get_declared_dataflow_slot(ir_op: Dict, io_kind: str, preferred_name: str, fallback_name: str) -> Optional[str]:
    dataflow = ir_op.get("dataflow", {}) if isinstance(ir_op.get("dataflow"), dict) else {}
    ios = dataflow.get(io_kind, {}) if isinstance(dataflow.get(io_kind), dict) else {}
    for name in (preferred_name, fallback_name):
        entry = ios.get(name)
        if isinstance(entry, dict):
            slot = entry.get("slot")
            if isinstance(slot, str) and slot:
                return slot
    return None


def _bind_recurrent_norm_gate_io(
    lowered_op: Dict[str, Any],
    ir_op: Dict[str, Any],
    activation_buffers: Dict[str, Dict[str, Any]],
) -> None:
    """
    Bind recurrent gated-norm to its declared graph slots.

    This op is a stitch unit, not a model-family special case:
      x    -> recurrent_attn_out -> recurrent_packed
      gate -> recurrent_z
      out  -> recurrent_normed
    """
    input_buf_by_name = {
        "x": "recurrent_packed",
        "gate": "recurrent_z",
    }
    for input_name, input_info in ir_op.get("inputs", {}).items():
        if input_name in ir_op.get("weights", {}):
            continue
        buf_name = input_buf_by_name.get(input_name)
        buf = activation_buffers.get(buf_name) if buf_name else None
        if buf:
            lowered_op["activations"][input_name] = {
                "buffer": buf_name,
                "activation_offset": buf["offset"],
                "dtype": input_info.get("dtype", "fp32"),
                "ptr_expr": f"activations + {buf['offset']}",
            }

    out_buf = activation_buffers.get("recurrent_normed")
    for output_name, output_info in ir_op.get("outputs", {}).items():
        if out_buf:
            lowered_op["outputs"][output_name] = {
                "buffer": "recurrent_normed",
                "activation_offset": out_buf["offset"],
                "dtype": output_info.get("dtype", "fp32"),
                "ptr_expr": f"activations + {out_buf['offset']}",
            }


def _recurrent_state_shape(config: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Derive the per-layer recurrent core state shape from config, not model-family
    branches.

    The recurrent_core op contract owns this shape. Templates or inspectors may
    declare explicit recurrent_state_{heads,rows,cols} keys; otherwise we fall
    back to the common DeltaNet/KDA layout [num_heads, head_dim, head_dim].
    Mamba2-style states must set cols explicitly because their state shape is
    [num_heads, head_dim, state_dim], not square.
    """
    heads = int(
        config.get(
            "recurrent_state_heads",
            config.get("recurrent_num_heads", config.get("gate_dim", 0)),
        ) or 0
    )
    rows = int(
        config.get(
            "recurrent_state_rows",
            config.get("recurrent_head_dim", config.get("ssm_state_size", 0)),
        ) or 0
    )
    cols = int(config.get("recurrent_state_cols", rows) or 0)
    return heads, rows, cols


def _recurrent_state_stride_bytes(config: Dict[str, Any], state_kind: str) -> int:
    if state_kind == "conv":
        history = int(config.get("ssm_conv_history", 0) or 0)
        channels = int(config.get("ssm_conv_channels", 0) or 0)
        return max(1, history) * max(1, channels) * 4
    if state_kind == "ssm":
        heads, rows, cols = _recurrent_state_shape(config)
        return max(1, heads) * max(1, rows) * max(1, cols) * 4
    raise ValueError(f"unknown recurrent state kind: {state_kind}")


def _apply_layer_scoped_recurrent_state_offsets(
    lowered_op: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    External recurrent-state slots are layer-local caches.

    Keep the stitch contract generic by deriving per-layer state slices from the
    declared buffer names instead of model-family branches. Any template that
    uses `external:recurrent_conv_state` / `external:recurrent_ssm_state` gets
    stable layer-scoped bindings.
    """
    layer_idx = int(lowered_op.get("layer", -1))
    if layer_idx < 0:
        return

    stride_by_buffer = {
        "recurrent_conv_state": _recurrent_state_stride_bytes(config, "conv"),
        "recurrent_ssm_state": _recurrent_state_stride_bytes(config, "ssm"),
    }

    for section_name in ("activations", "outputs"):
        section = lowered_op.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for binding in section.values():
            if not isinstance(binding, dict):
                continue
            buf_name = str(binding.get("buffer", ""))
            stride = stride_by_buffer.get(buf_name, 0)
            if stride <= 0:
                continue
            scoped_off = int(binding.get("activation_offset", 0)) + layer_idx * stride
            binding["activation_offset"] = scoped_off
            binding["ptr_expr"] = f"activations + {scoped_off}"


def validate_template_ops(template_ops: List[str]) -> List[str]:
    """
    Validate that all template ops have kernel mappings.
    Returns list of unmapped ops (empty if all valid).
    """
    unmapped = []
    for op in template_ops:
        if op not in TEMPLATE_TO_KERNEL_OP:
            unmapped.append(op)
    return unmapped


def unsupported_template_lowering_reason(manifest: Dict[str, Any]) -> Optional[str]:
    """Return a human-readable reason when a template is known but not lowerable yet."""
    template = manifest.get("template") if isinstance(manifest.get("template"), dict) else {}
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    template_name = str(template.get("name", "") or "").strip().lower()
    model_name = str(config.get("model", "") or config.get("model_type", "") or "").strip().lower()
    arch_name = str(config.get("arch", "") or "").strip().lower()

    seq = template.get("sequence") if isinstance(template.get("sequence"), list) else []
    if not seq:
        return None

    block_name = str(seq[0] or "")
    block_types = template.get("block_types") if isinstance(template.get("block_types"), dict) else {}
    block = block_types.get(block_name) if isinstance(block_types.get(block_name), dict) else {}
    body = block.get("body")
    body_type = str(body.get("type", "")).strip().lower() if isinstance(body, dict) else ""

    if body_type in {"", "dense"}:
        return None

    if isinstance(body, dict) and isinstance(body.get("ops_by_kind"), dict):
        return None

    return (
        f"Template body.type='{body_type}' is not implemented in build_ir_v8 yet. "
        "Only the active flat body graph is lowerable today; non-dense body kernels "
        "and explicit branch/routing execution still need lowering support."
    )


def validate_kernel_availability(registry: Dict, kernel_ops: List[str]) -> Dict[str, bool]:
    """
    Check which kernel ops are available in the registry.
    Returns dict: {kernel_op: is_available}

    Operation families are exact compiler inputs. A missing sliding, quantized, or
    otherwise specialized family must not be accepted because a broader family is
    present; the circuit or kernel registry must provide the required operation.
    """
    available_ops = set()
    for kernel in registry["kernels"]:
        available_ops.add(kernel["op"])

    availability = {}
    for op in kernel_ops:
        availability[op] = op in available_ops

    return availability


def find_kernel(
    registry: Dict,
    op: str,
    quant: Dict[str, str],
    mode: str = "decode",
    prefer_q8_activation: bool = True,  # v7 baseline parity: use Q8_0 activation kernels
    prefer_parallel: bool = False  # Use OpenMP-parallel kernels for decode throughput
) -> Optional[str]:
    """
    Find kernel ID from registry.

    Args:
        registry: Kernel registry
        op: Operation type (e.g., "qkv_projection", "matmul", "attention")
        quant: Quantization dict (e.g., {"weight": "q5_0"})
        mode: Execution mode ("decode" or "prefill")
        prefer_q8_activation: If True, prefer Q8_0 activation kernels (v7 baseline parity).
                              If False, prefer FP32 activation kernels.
        prefer_parallel: If True, prefer _parallel_omp kernel variants for decode mode.
                         These have the same signature as serial kernels but use OpenMP
                         internally — no wrapper code or IR changes needed.

    Returns:
        Kernel ID (C function name) or None if not found

    Note:
        "matmul" is a logical op that maps to:
        - gemv (matrix-vector) for decode mode (single token)
        - gemm (matrix-matrix) for prefill mode (multiple tokens)
    """
    # Map logical "matmul" to concrete gemv/gemm based on mode
    actual_op = op
    if op == "matmul":
        actual_op = "gemv" if mode == "decode" else "gemm"

    candidates = [k for k in registry["kernels"] if k["op"] == actual_op]

    # Filter and collect all matching candidates
    matches = []

    # Treat decode/prefill as inference-mode lookups. Training/backward-only
    # kernels must not be selected in these paths.
    inference_mode = mode in ("decode", "prefill", "inference")

    for candidate in candidates:
        k_quant = candidate.get("quant", {})
        modes = candidate.get("modes", {})

        # Match weight quantization
        if "weight" in quant:
            weight_quant = k_quant.get("weight", "none")
            # Support multi-quant kernels (e.g., "q5_0|q8_0|q4_k")
            allowed_quants = weight_quant.split("|")
            # Skip kernels without a specific weight quant when we need one
            # This prevents meta-kernels like "dense_embedding_lookup" from being selected
            # when we need an actual implementation like "embedding_forward_q8_0"
            if quant["weight"] not in allowed_quants:
                continue

        # Match explicit inference/training/backward mode contract.
        # If modes is absent, treat kernel as inference-eligible.
        if isinstance(modes, dict) and modes:
            if inference_mode and modes.get("inference") is False:
                continue
            if (not inference_mode) and mode == "backward" and modes.get("backward") is False:
                continue
            if (not inference_mode) and mode == "training" and modes.get("training") is False:
                continue

        # Match legacy single "mode" field (if kernel specifies it)
        kernel_mode = candidate.get("mode", "")
        variant = candidate.get("variant", "")

        # If kernel specifies a mode, it must match
        if kernel_mode and kernel_mode != mode:
            continue

        # Also check variant name for mode hints
        if mode == "decode" and "prefill" in variant:
            continue
        if mode == "prefill" and "decode" in variant:
            continue

        # Collect match
        matches.append(candidate)

    if not matches:
        # Fallback: decode with dense fp32/bf16/fp16 weights has no native gemv
        # path in the registry yet. Use GEMM with M=1 for correctness.
        # Use gemm with M=1 for decode correctness.
        if op == "matmul" and mode == "decode" and quant.get("weight") in ("fp32", "bf16", "fp16", "f16"):
            return find_kernel(
                registry,
                op="gemm",
                quant=quant,
                mode=mode,
                prefer_q8_activation=prefer_q8_activation,
                prefer_parallel=prefer_parallel,
            )

        # Fallback: Q4_0 → Q4_K (similar K-quant format)
        # Q4_0 GEMV kernels don't exist in the library, but Q4_K does
        if "weight" in quant and quant["weight"] == "q4_0":
            return find_kernel(registry, op=op, quant={**quant, "weight": "q4_k"}, mode=mode,
                             prefer_q8_activation=prefer_q8_activation, prefer_parallel=prefer_parallel)

        # Fallback: sliding attention → regular attention (if sliding kernel not available)
        if op == "attention_sliding":
            return find_kernel(registry, op="attention", quant=quant, mode=mode,
                             prefer_q8_activation=prefer_q8_activation, prefer_parallel=prefer_parallel)

        return None

    # Sort by forward/backward direction first, then activation preference.
    # Keep this generic: inference/decode should never silently bind a backward
    # variant just because it shares the same logical op family.
    def direction_priority(k):
        variant = str(k.get("variant", "") or "").lower()
        kernel_id = str(k.get("id", "") or "").lower()
        modes = k.get("modes", {})
        if inference_mode:
            if isinstance(modes, dict) and modes:
                if modes.get("inference") is True and modes.get("backward") is False:
                    return 0
                if modes.get("backward") is True or modes.get("inference") is False:
                    return 2
            if "backward" in variant or "backward" in kernel_id:
                return 2
            return 0
        return 0

    # When prefer_q8_activation=True (v7 baseline parity): prefer Q8_0 activation kernels
    # When prefer_q8_activation=False: prefer FP32 activation kernels
    def activation_priority(k):
        act = k.get("quant", {}).get("activation", "fp32")
        if prefer_q8_activation:
            # v7 baseline parity mode: prefer Q8_0 activation (quantized input)
            # Then prefer fp32 over bf16 (bf16 is slower and rarely needed)
            if act == "q8_0":
                return 0  # Prefer Q8_0 activation
            if act == "q8_k":
                return 1  # Q8_K is second choice
            if act == "fp32":
                return 2  # FP32 preferred
            if act == "bf16":
                return 3  # BF16 last choice
            return 4  # Unknown activation types
        else:
            # FP32 mode: prefer FP32 activation, then BF16
            # Explicit ordering to prevent BF16 being chosen over FP32
            if act == "fp32":
                return 0  # Prefer FP32
            if act == "bf16":
                return 1  # BF16 second choice
            return 2  # Quantized last

    matches.sort(key=lambda k: (direction_priority(k), activation_priority(k)))

    # When prefer_parallel=True in decode mode, look for _parallel_omp variant
    # among the top-priority activation matches. These have the same signature
    # as serial kernels — the IR just swaps the function name, no wrapper needed.
    if prefer_parallel and mode == "decode":
        top_act = matches[0].get("quant", {}).get("activation", "fp32")
        same_act = [m for m in matches if m.get("quant", {}).get("activation", "fp32") == top_act]
        for m in same_act:
            if m.get("parallel", False):
                return m["id"]

    return matches[0]["id"]


def kernel_needs_q8_activation(registry: Dict, kernel_id: str) -> bool:
    """
    Check if a kernel requires Q8_0 quantized activation input.

    Args:
        registry: Kernel registry
        kernel_id: Kernel ID to check

    Returns:
        True if kernel expects Q8_0 activation input, False otherwise
    """
    for k in registry.get("kernels", []):
        if k.get("id") == kernel_id:
            act = k.get("quant", {}).get("activation", "fp32")
            return act in ("q8_0", "q8_k")
    return False


def get_kernel_activation_quantization_contract(
    registry: Dict,
    kernel_id: str,
    mode: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Return the Q8 storage and arithmetic contract required by a consumer."""
    for kernel in registry.get("kernels", []):
        if not isinstance(kernel, dict) or kernel.get("id") != kernel_id:
            continue
        quant = kernel.get("quant") if isinstance(kernel.get("quant"), dict) else {}
        activation_dtype = str(quant.get("activation", "fp32") or "fp32")
        if activation_dtype not in {"q8_0", "q8_k"}:
            return None, None
        contract_spec = quant.get("activation_quantization_contract")
        if contract_spec is None:
            if activation_dtype == "q8_k":
                raise RuntimeError(
                    f"HARD NUMERICAL CONTRACT FAULT: Q8_K consumer {kernel_id!r} "
                    "does not declare activation_quantization_contract."
                )
            return activation_dtype, None
        if isinstance(contract_spec, str):
            contract = contract_spec.strip()
        elif isinstance(contract_spec, dict):
            contract = str(contract_spec.get(mode, "") or "").strip()
        else:
            contract = ""
        if not contract:
            raise RuntimeError(
                f"HARD NUMERICAL CONTRACT FAULT: consumer {kernel_id!r} has no "
                f"activation quantization contract for phase {mode!r}."
            )
        return activation_dtype, contract
    raise RuntimeError(f"HARD KERNEL RESOLUTION FAULT: unknown kernel {kernel_id!r}.")


def get_quantize_kernel_for_activation(
    registry: Dict,
    activation_dtype: str,
    rounding_contract: Optional[str] = None,
) -> Optional[str]:
    """
    Get the appropriate quantize kernel for the target activation dtype.

    Args:
        activation_dtype: Target activation dtype (e.g., "q8_0", "q8_k")
        rounding_contract: Required arithmetic/rounding contract, when declared

    Returns:
        Quantize kernel ID or None if no quantization needed
    """
    if activation_dtype not in {"q8_0", "q8_k"}:
        return None
    matches = []
    for kernel in registry.get("kernels", []):
        if (
            not isinstance(kernel, dict)
            or kernel.get("op") != "quantize"
            or (kernel.get("quant") or {}).get("output") != activation_dtype
        ):
            continue
        capability = (
            kernel.get("codegen_capability")
            if isinstance(kernel.get("codegen_capability"), dict)
            else {}
        )
        provider_contract = capability.get("rounding_contract")
        if rounding_contract is not None and provider_contract != rounding_contract:
            continue
        if rounding_contract is None and provider_contract is not None:
            continue
        matches.append(str(kernel.get("id", "")))
    if len(matches) != 1:
        raise RuntimeError(
            f"HARD CODEGEN CAPABILITY FAULT: activation storage {activation_dtype!r} "
            f"with rounding contract {rounding_contract!r} "
            f"resolved {len(matches)} quantization providers: {matches}. "
            "Kernel maps must provide exactly one quantize operator for the requested "
            "storage and arithmetic contract."
        )
    return matches[0]


# Quantization formats that require native kernels (no safe fallback)
# These formats have incompatible memory layouts that cannot be safely fed to other kernels
UNSAFE_QUANT_FALLBACKS = {
    "q4_k",  # Super-block format (8 values per block) - now has native gemm_nt_q4_k, gemv_q4_k
    "q5_k",  # Super-block format; incompatible with Q5_0 simple-block kernels
    "q6_k",  # Super-block format (16 values per block) - now has native gemm_nt_q6_k, gemv_q6_k
}

# Quantization formats that have safe fallbacks (same block structure)
SAFE_QUANT_FALLBACKS = {
    "q5_1": "q5_0",  # Both use 32-value blocks, Q5_1 has min value
    "q4_1": "q4_0",  # Both use 32-value blocks, Q4_1 has min value
}


def validate_quant_safety(manifest: Dict, registry: Dict, allow_fallback: bool = False) -> None:
    """
    Validate that all quantization formats in the model have native kernel support.

    Args:
        manifest: Weights manifest with quant_summary
        registry: Kernel registry
        allow_fallback: If True, allow unsafe fallbacks with warnings

    Raises:
        RuntimeError: If model uses unsupported quant formats without fallback
    """
    quant_summary = manifest.get("quant_summary", {})
    if not isinstance(quant_summary, dict):
        return

    # Collect all quant types used
    used_quants = set()
    for key, value in quant_summary.items():
        if isinstance(value, dict):
            # Layer dict with individual weight dtypes
            for dtype in value.values():
                if isinstance(dtype, str):
                    used_quants.add(dtype.lower())
        elif isinstance(value, str):
            used_quants.add(value.lower())

    # Check for native kernel support
    missing_kernels = []
    for qtype in used_quants:
        if qtype in UNSAFE_QUANT_FALLBACKS:
            # Check if any kernel supports this quant
            has_native = False
            for k in registry.get("kernels", []):
                kq = k.get("quant", {}).get("weight")
                if not kq:
                    continue
                if qtype in str(kq).split("|"):
                    has_native = True
                    break

            if not has_native:
                if allow_fallback:
                    # Check if safe fallback exists
                    fallback = SAFE_QUANT_FALLBACKS.get(qtype)
                    if fallback:
                        print(f"  {YELLOW}WARNING: {qtype.upper()} weights detected but no native kernel.{RESET}")
                        print(f"  {YELLOW}  Falling back to {fallback.upper()} - this may cause accuracy issues.{RESET}")
                        print(f"  {YELLOW}  Use --allow-quant-fallback with caution.{RESET}")
                    else:
                        print(f"  {YELLOW}WARNING: {qtype.upper()} weights detected but no native kernel.{RESET}")
                        print(f"  {YELLOW}  No safe fallback available - this may cause segfaults!{RESET}")
                else:
                    missing_kernels.append(qtype)

    if missing_kernels:
        print(f"\n{RED}ERROR: Model uses quantization formats without native kernel support:{RESET}")
        for qtype in sorted(missing_kernels):
            print(f"  {RED}  - {qtype.upper()}: No kernel map exists for this format{RESET}")
        print()
        print(f"  {YELLOW}Options:{RESET}")
        print(f"    1. Add kernel maps for {', '.join(sorted(missing_kernels))}")
        print(f"    2. Convert weights to a supported format (q5_0, q8_0, fp32)")
        print(f"    3. Use --allow-quant-fallback to attempt unsafe fallback (not recommended)")
        raise RuntimeError(
            f"Unsupported quantization formats: {', '.join(sorted(missing_kernels))}. "
            "Add native kernels or convert weights."
        )


def build_ir1_direct(manifest: Dict, manifest_path: Path, mode: str = "decode",
                     prefer_parallel: bool = False,
                     allow_quant_fallback: bool = False) -> List[str]:
    """
    Direct mapping: Template + Quant Summary → Kernel IDs.

    This is the CORRECT approach - no intermediate abstractions!

    Algorithm:
        1. Validate template ops have kernel mappings
        2. Validate all required kernels exist in registry
        3. For each layer, map template ops → kernel IDs
        4. Return list of kernel IDs (IR1)

    Args:
        manifest: Weights manifest with template and quant_summary
        mode: Execution mode ("decode" or "prefill")
        prefer_parallel: If True, select _parallel_omp kernel variants for decode.
                         These have the same signature as serial kernels but use
                         OpenMP internally — the IR just swaps the function name.
        allow_quant_fallback: If True, allow unsafe quant fallbacks (not recommended)

    Returns:
        List of kernel IDs (C function names) in execution order

    Raises:
        RuntimeError: If validation fails (missing mappings or kernels)
    """
    manifest = _hydrate_manifest_template(manifest)
    _backfill_template_runtime_flags(manifest)
    _backfill_vision_contract_config(manifest)
    template = manifest.get("template", {})
    unsupported_reason = unsupported_template_lowering_reason(manifest)
    if unsupported_reason:
        raise RuntimeError(unsupported_reason)
    quant_summary = manifest.get("quant_summary", {})
    header_quant = {}
    entries = manifest.get("entries", [])
    weight_index = {
        str(e.get("name", "")): e
        for e in entries
        if isinstance(e, dict) and str(e.get("name", "")).strip()
    }
    entry_dtype = {}
    for e in entries:
        n = e.get("name")
        d = e.get("dtype")
        if isinstance(n, str) and isinstance(d, str):
            entry_dtype[n] = d.lower()

    if isinstance(quant_summary, dict):
        # Carry all top-level scalar quant declarations through header/footer
        # lowering. This covers projector weights (`mm0_w` / `mm1_w`) in the
        # vision path in addition to token/patch embeddings.
        for key, value in quant_summary.items():
            if isinstance(key, str) and isinstance(value, str) and value:
                header_quant[key] = value
        token_q = quant_summary.get("token_emb")
        if isinstance(token_q, str) and token_q:
            header_quant["token_emb"] = token_q
        lm_q = quant_summary.get("lm_head")
        if isinstance(lm_q, str) and lm_q:
            header_quant["lm_head"] = lm_q
        patch_q = quant_summary.get("patch_emb")
        if isinstance(patch_q, str) and patch_q:
            header_quant["patch_emb"] = patch_q
    # Fallback to actual manifest entry dtype when top-level quant_summary fields
    # are absent (common for HF fp32 conversions).
    if "token_emb" not in header_quant and "token_emb" in entry_dtype:
        header_quant["token_emb"] = entry_dtype["token_emb"]
    if "lm_head" not in header_quant:
        lm_head_entry = (
            entry_dtype.get("lm_head")
            or entry_dtype.get("lm_head.weight")
            or entry_dtype.get("output.weight")
        )
        if lm_head_entry:
            header_quant["lm_head"] = lm_head_entry
    if "patch_emb" not in header_quant:
        patch_entry = (
            entry_dtype.get("patch_emb.weight")
            or entry_dtype.get("patch_embeddings.weight")
            or entry_dtype.get("vision_model.embeddings.patch_embedding.weight")
            or entry_dtype.get("v.patch_embd.weight")
        )
        if patch_entry:
            header_quant["patch_emb"] = patch_entry
    if "patch_emb_aux" not in header_quant:
        patch_aux_entry = entry_dtype.get("v.patch_embd.weight.1")
        if patch_aux_entry:
            header_quant["patch_emb_aux"] = patch_aux_entry
    if "mm0_w" not in header_quant:
        mm0_entry = entry_dtype.get("mm.0.weight")
        if mm0_entry:
            header_quant["mm0_w"] = mm0_entry
    if "mm1_w" not in header_quant:
        mm1_entry = entry_dtype.get("mm.2.weight")
        if mm1_entry:
            header_quant["mm1_w"] = mm1_entry
    if "assistant_pre_projection" not in header_quant:
        pre_proj_entry = entry_dtype.get("assistant.pre_projection")
        if pre_proj_entry:
            header_quant["assistant_pre_projection"] = pre_proj_entry
    if "assistant_post_projection" not in header_quant:
        post_proj_entry = entry_dtype.get("assistant.post_projection")
        if post_proj_entry:
            header_quant["assistant_post_projection"] = post_proj_entry
    config = manifest.get("config", {})
    template_flags = template.get("flags", {}) if isinstance(template.get("flags"), dict) else {}
    template_contract = template.get("contract", {}) if isinstance(template.get("contract"), dict) else {}
    logits_contract = template_contract.get("logits_contract", {}) if isinstance(template_contract.get("logits_contract"), dict) else {}
    if str(logits_contract.get("lm_head", "")).strip().lower() == "none" or str(logits_contract.get("logits_layout", "")).strip().lower() == "none":
        logits_weight_source = "none"
    else:
        logits_weight_source = _resolve_logits_weight_source(config, weight_index)
    print(f"  [contract/logits] source={logits_weight_source}")
    activation_preference_by_op = config.get("activation_preference_by_op", {})
    if not isinstance(activation_preference_by_op, dict):
        activation_preference_by_op = {}
    # Default to Q8 activation preference for the v7 baseline path.
    # Model-specific overrides can still force FP32 by setting
    # config["prefer_q8_activation"]=false.
    prefer_q8_activation = bool(config.get("prefer_q8_activation", True))
    # Some families still need FP32-activation MLP matmuls for parity.
    # Keep this as an explicit manifest/template opt-in rather than a default.
    prefer_fp32_mlp_matmuls = (
        prefer_q8_activation
        and bool(config.get("prefer_fp32_mlp_matmuls", False))
    )
    registry = load_kernel_registry()

    # Validate quant safety before proceeding
    print(f"\n  [Quant Safety Check]")
    validate_quant_safety(manifest, registry, allow_fallback=allow_quant_fallback)

    if prefer_fp32_mlp_matmuls:
        print("  FP32 MLP matmul override: ON")

    num_layers = config.get("num_layers", 0)

    # Hydration is an explicit pipeline phase. Lowering must never recover a
    # circuit by model name or rerun conversion as a side effect.
    if not template or "sequence" not in template:
        raise RuntimeError(
            "HARD CIRCUIT HYDRATION FAULT: lowering requires a hydrated circuit with a sequence. "
            "Fix conversion or _hydrate_manifest_template; do not add model-specific recovery to the DSL."
        )

    numerical_contract_plans = (
        _resolve_manifest_numerical_contracts(manifest, mode)
        + _resolve_manifest_execution_contracts(manifest, mode)
    )
    numerical_contract_by_template_op = _index_numerical_contract_plans(numerical_contract_plans)
    kernel_execution_capabilities = _load_kernel_execution_capabilities()
    for plan in numerical_contract_plans:
        print(
            "  [contract/numerics] "
            f"{plan['operation']}.{plan['phase']} -> "
            f"{plan['kernel']['id']} / "
            f"{(plan.get('contract') or plan.get('reduction'))['id']}"
        )

    template_flags = template.get("flags", {}) if isinstance(template.get("flags"), dict) else {}
    template_kernels = template.get("kernels", {}) if isinstance(template.get("kernels"), dict) else {}
    # Runtime-config opt-in: use FP32->Q8_0 contract adapters for Q8_0 kernels.
    # Weight dtype still comes from the manifest; this only selects the activation
    # contract path when a model family explicitly requests it.
    prefer_q8_contract = bool(config.get("prefer_q8_0_contract", False))
    q8_contract_ops_raw = config.get("q8_0_contract_ops", [])
    q8_contract_ops = {
        str(op).strip()
        for op in q8_contract_ops_raw
        if str(op).strip()
    } if isinstance(q8_contract_ops_raw, (list, tuple, set)) else set()
    # Gemma parity guardrail: keep logits projection on FP32-activation kernels.
    # This remains a runtime config knob rather than template metadata.
    prefer_fp32_logits = bool(config.get("prefer_fp32_logits", False))
    # Embedding scaling is a circuit semantic. Missing metadata means no scaling;
    # the compiler must never infer it from a model or circuit name.
    scale_embeddings_sqrt_dim = bool(template_flags.get("scale_embeddings_sqrt_dim", False))
    config["scale_embeddings_sqrt_dim"] = scale_embeddings_sqrt_dim

    # Extract active op sequences from the template. Planned branch/stitch ops
    # may already exist in the schema, but only active ops are lowerable here.
    block_name = template["sequence"][0]
    block = template["block_types"][block_name]

    header_items = _normalize_template_op_items(block.get("header", []))
    body_items = _collect_body_items_for_validation(block.get("body", {}), config)
    footer_items = _normalize_template_op_items(block.get("footer", []))
    header_ops = [item["op"] for item in header_items]
    body_ops = [item["op"] for item in body_items]
    footer_ops = [item["op"] for item in footer_items]
    branch_plan = _build_block_branch_plan(block, config)
    branch_tap_targets: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for branch in branch_plan:
        tap_ref = branch.get("tap_ref") if isinstance(branch.get("tap_ref"), dict) else None
        if not isinstance(tap_ref, dict):
            continue
        section_name = str(tap_ref.get("section", "") or "").strip().lower()
        tap_op_id = str(tap_ref.get("op_id", "") or "").strip()
        if not section_name or not tap_op_id:
            continue
        active_layers = branch.get("layers", []) if section_name == "body" else [-1]
        for collect_index, layer_idx in enumerate(active_layers):
            try:
                normalized_layer = int(layer_idx) if section_name == "body" else -1
            except Exception:
                continue
            branch_tap_targets.setdefault((section_name, normalized_layer, tap_op_id), []).append(
                {
                    "name": branch.get("name", ""),
                    "kind": branch.get("kind", "fixed_branch"),
                    "tap": copy.deepcopy(branch.get("tap", {})),
                    "tap_ref": copy.deepcopy(tap_ref),
                    "producer_ops": copy.deepcopy(branch.get("producer_ops", [])),
                    "producer_items": copy.deepcopy(branch.get("producer_items", [])),
                    "collect": copy.deepcopy(branch.get("collect", {})),
                    "collect_contract": copy.deepcopy(branch.get("collect_contract", {})),
                    "stitches": copy.deepcopy(branch.get("stitches", [])),
                    "collect_index": collect_index,
                }
            )

    # For validation, we need all ops
    branch_template_ops: List[str] = []
    for branch in branch_plan:
        branch_template_ops.extend(branch.get("producer_ops", []))
    all_template_ops = _dedupe_preserve_order(header_ops + body_ops + footer_ops + branch_template_ops)

    print(f"\n{'='*60}")
    print("VALIDATION PHASE")
    print(f"{'='*60}")

    # VALIDATION 1: Check template ops have kernel mappings
    print(f"\n[1/2] Validating template ops...")
    print(f"  Header: {header_ops}")
    print(f"  Body: {body_ops}")
    print(f"  Footer: {footer_ops}")
    if branch_plan:
        print("  Branches:")
        for branch in branch_plan:
            producer_ops = ", ".join(branch.get("producer_ops", [])) or "(none)"
            print(
                f"    - {branch.get('name', '')}: status={branch.get('status', 'active')} "
                f"layers={branch.get('layers', [])} producer=[{producer_ops}]"
            )

    unmapped_ops = validate_template_ops(all_template_ops)
    if unmapped_ops:
        print(f"\n❌ HARD FAULT: Template ops have no kernel mapping!")
        for op in unmapped_ops:
            print(f"  - {op}")
        print(f"\nAction required:")
        print(f"  Add mappings to TEMPLATE_TO_KERNEL_OP in build_ir_v8.py")
        raise RuntimeError(f"Missing kernel mappings for: {unmapped_ops}")

    # Get required kernel ops (filter out None for metadata ops)
    required_kernel_ops = set()
    non_kernel_ops = []
    for template_op in all_template_ops:
        kernel_op = TEMPLATE_TO_KERNEL_OP[template_op]
        if kernel_op is None:
            non_kernel_ops.append(template_op)
        else:
            required_kernel_ops.add(kernel_op)

    print(f"  ✅ All {len(all_template_ops)} template ops have mappings")
    if non_kernel_ops:
        print(f"  Graph/metadata ops (no direct kernel): {', '.join(non_kernel_ops)}")
    print(f"  Required kernel ops: {', '.join(sorted(required_kernel_ops))}")

    # VALIDATION 2: Check kernels exist in registry
    print(f"\n[2/2] Validating kernel availability...")

    # Handle "matmul" specially - it maps to gemv (decode) or gemm (prefill)
    validation_kernel_ops = []
    for op in required_kernel_ops:
        if op == "matmul":
            validation_kernel_ops.extend(["gemv", "gemm"])
        else:
            validation_kernel_ops.append(op)

    availability = validate_kernel_availability(registry, validation_kernel_ops)
    missing_kernels = [op for op, avail in availability.items() if not avail]

    if missing_kernels:
        print(f"\n❌ HARD FAULT: Required kernels not found in registry!")
        for op in missing_kernels:
            print(f"  - {op}")
        print(f"\nAction required:")
        print(f"  1. Implement missing kernels")
        print(f"  2. Add to kernel maps and regenerate KERNEL_REGISTRY.json")
        raise RuntimeError(f"Missing kernels: {missing_kernels}")

    print(f"  ✅ All required kernels available (matmul → gemv/gemm)")

    print(f"\n{'='*60}")
    print("IR1 GENERATION PHASE")
    print(f"{'='*60}")

    print(f"\nBuilding IR1 from template...")
    print(f"  Mode: {mode}")
    print(f"  Layers: {num_layers}")
    print(f"  Q8 contract override: {'ON' if prefer_q8_contract else 'OFF'}")
    print(f"  FP32 logits preference: {'ON' if prefer_fp32_logits else 'OFF'}")
    print(f"  Embed sqrt(dim) scale: {'ON' if scale_embeddings_sqrt_dim else 'OFF'}")

    arranged_kernels = []  # Pass 1: list of {kernel, op, section, layer, op_id, instance, dataflow}
    global_op_id = 0  # Global operation ID counter

    # ═══════════════════════════════════════════════════════════
    # IR1 now includes DATAFLOW information:
    #   - Each op has "dataflow" with "inputs" and "outputs"
    #   - Inputs reference the op_id that produced them
    #   - This enables the memory planner to assign physical buffers
    # ═══════════════════════════════════════════════════════════

    # Initialize dataflow tracker
    dataflow_tracker = DataflowTracker()

    # Build activation dtype lookup (kernel_id -> activation dtype)
    kernel_act_dtype = {
        k.get("id"): k.get("quant", {}).get("activation", "fp32")
        for k in registry.get("kernels", [])
    }

    def _input_slot_override_for_kernel(op_type: str, kernel_id: Optional[str]) -> Optional[Dict[str, str]]:
        """Override dataflow input slot based on kernel activation dtype."""
        if not kernel_id:
            return None
        act = kernel_act_dtype.get(kernel_id, "fp32")
        if op_type in ("q_proj", "q_gate_proj", "k_proj", "v_proj", "qkv_packed_proj", "mlp_gate_up", "mlp_up"):
            return {"x": "main_stream" if act == "fp32" else "main_stream_q8"}
        if op_type == "projector_fc1":
            return {"x": "main_stream" if act == "fp32" else "main_stream_q8"}
        if op_type == "out_proj":
            return {"x": "attn_scratch" if act == "fp32" else "main_stream_q8"}
        if op_type == "projector_fc2":
            return {"x": "mlp_scratch" if act == "fp32" else "main_stream_q8"}
        if op_type == "branch_fc1":
            return {"x": "branch_normed" if act == "fp32" else "branch_normed"}
        if op_type == "branch_fc2":
            return {"x": "branch_mlp" if act == "fp32" else "branch_mlp"}
        if op_type == "mamba_in_proj":
            return {"x": "main_stream" if act == "fp32" else "main_stream_q8"}
        if op_type in ("recurrent_qkv_proj", "recurrent_gate_proj", "recurrent_alpha_proj", "recurrent_beta_proj"):
            return {"x": "main_stream" if act == "fp32" else "main_stream_q8"}
        if op_type in ("recurrent_out_proj", "mamba_out_proj"):
            return {"x": "recurrent_normed" if act == "fp32" else "main_stream_q8"}
        if op_type == "mlp_down":
            return {"x": "mlp_scratch" if act == "fp32" else "main_stream_q8"}
        if op_type == "logits":
            return {"x": "main_stream" if act == "fp32" else "main_stream_q8"}
        return None

    def _maybe_apply_q8_contract(
        kernel_id: Optional[str],
        weight_dtype: Optional[str],
        *,
        allow_q8_contract: bool,
    ) -> Optional[str]:
        """Optionally remap standard Q8_0 kernels to explicit contract adapters."""
        if not kernel_id or not allow_q8_contract:
            return kernel_id
        if weight_dtype != "q8_0":
            return kernel_id
        if kernel_id == "gemv_q8_0":
            return "gemv_q8_0_q8_0_contract"
        if kernel_id == "gemm_nt_q8_0":
            return "gemm_nt_q8_0_q8_0_contract"
        return kernel_id

    def _prefer_q8_activation_for_op(op_name: str, default: bool) -> bool:
        """
        Resolve activation preference from template metadata.

        The graph contract belongs in the template, not in architecture-named
        lowerer branches. Any family can declare per-op activation preferences
        here when a reference path requires FP32 inputs for specific matmuls.
        """
        pref = activation_preference_by_op.get(op_name)
        if pref is None:
            return default
        pref_lc = str(pref).strip().lower()
        if pref_lc in {"fp32", "float", "float32"}:
            return False
        if pref_lc in {"q8", "q8_0", "q8_k", "quantized"}:
            return True
        return default

    # Weight entries from manifest (for Pass 2 binding)

    # ═══════════════════════════════════════════════════════════
    # Op → Weight mapping (which weights each op uses for quant lookup)
    # ═══════════════════════════════════════════════════════════
    OP_TO_WEIGHT_KEYS = {
        # Ops with quantized weights - look up in quant_summary
        "patch_proj": ["patch_emb"],
        "patch_proj_aux": ["patch_emb_aux"],
        "patch_bias_add": None,
    "vision_position_ids": None,
    "position_ids_2d": None,
        "qkv_packed_proj": ["attn_qkv"],
        "qkv_proj": ["wq", "wk", "wv"],  # Split into 3 matmuls if no fused kernel
        "q_proj": ["wq"],
        "assistant_pre_projection": ["assistant_pre_projection"],
        "assistant_post_projection": ["assistant_post_projection"],
        "q_gate_proj": ["wq"],
        "k_proj": ["wk"],
        "v_proj": ["wv"],
        "recurrent_qkv_proj": ["attn_qkv"],
        "recurrent_gate_proj": ["attn_gate"],
        "recurrent_alpha_proj": ["ssm_alpha"],
        "recurrent_beta_proj": ["ssm_beta"],
        "recurrent_ssm_conv": ["ssm_conv1d"],
        "recurrent_out_proj": ["ssm_out"],
    "mamba_in_proj": ["mamba_in_proj"],
    "mamba_in_proj_split": [],
    "mamba_dt_softplus": ["mamba_dt_bias"],
    "mamba_conv1d_silu": ["mamba_conv1d", "mamba_conv1d_bias"],
    "mamba_selective_scan": ["mamba_a", "mamba_d"],
    "mamba_rmsnorm_gate": ["mamba_norm"],
    "mamba_out_proj": ["mamba_out_proj"],
    "moe_router": ["moe_router"],
    "group_limited_topk_router": ["moe_router_bias"],
    "moe_relu2_expert_mlp": ["moe_expert_up", "moe_expert_down"],
    "shared_relu2_expert_mlp": ["moe_shared_up", "moe_shared_down"],
    "moe_swiglu_expert_mlp": ["moe_expert_gate", "moe_expert_up", "moe_expert_down"],
    "shared_swiglu_expert_mlp": ["moe_shared_gate", "moe_shared_up", "moe_shared_down"],
    "kv_a_proj": ["mla_kv_a_proj"],
    "kv_a_layernorm": ["mla_kv_a_norm"],
    "kv_lora_decompress": ["mla_kv_b_proj"],
    "partial_rope_concat": [],
    "mla_attention": [],
        "out_proj": ["wo"],
        "mlp_gate_up": ["w1"],
        "mlp_up": ["w3"],
        "mlp_down": ["w2"],
        "projector_fc1": ["mm0_w"],
        "projector_prep": [],
        "projector_fc2": ["mm1_w"],
        "branch_fc1": ["branch_fc1_w"],
        "branch_fc2": ["branch_fc2_w"],
        "dense_embedding_lookup": [],  # Uses token_emb, usually q8_0
        "logits": [],  # Uses lm_head/token_emb, usually q8_0

        # Ops with fp32 weights (no quant lookup needed)
        "rmsnorm": None,  # gamma is always fp32
        "layernorm": None,  # gamma/beta are fp32
        "attn_norm": None,
        "block_rmsnorm": None,
        "post_attention_norm": None,
        "ffn_norm": None,
        "post_ffn_norm": None,
        "gemma4_per_layer_prepare": [
            "per_layer_token_emb",
            "per_layer_model_proj",
            "per_layer_proj_norm",
        ],
        "gemma4_per_layer_embed": [
            "per_layer_inp_gate",
            "per_layer_proj",
            "per_layer_post_norm",
            "layer_output_scale",
        ],
        "assistant_layer_scale": None,
        "final_logit_softcap": None,
        "v_norm": [],
        "final_rmsnorm": None,
        "qk_norm": None,  # Per-head RMSNorm gamma is always fp32
        "q_norm": None,  # Gemma4 assistant per-head Q RMSNorm gamma is fp32

        # Ops without weights (compute-only)
        "patchify": None,
        "position_embeddings": None,
        "vision_position_ids": None,
        "position_ids_2d": None,
        "split_qkv_packed": None,
        "mrope_qk": None,
        "rope_qk": None,
        "rope_q": None,
        "mla_kv_cache_batch_store": None,
        "mla_kv_cache_store": None,
        "kv_cache_store": None,  # Store K,V to KV cache (no weights)
        "kv_cache_store_shared_q": None,
        "attn": None,
        "attn_sliding": None,
        "attn_shared_kv": None,
        "attn_sliding_shared_kv": None,
        "split_q_gate": None,
        "recurrent_split_qkv": None,
        "recurrent_dt_gate": None,
        "recurrent_conv_state_update": None,
        "recurrent_silu": None,
        "recurrent_split_conv_qkv": None,
        "recurrent_qk_l2_norm": None,
        "recurrent_core": None,
        "recurrent_norm_gate": None,
        "attn_gate_sigmoid_mul": None,
        "residual_add": None,
        "add_stream": None,
        "silu_mul": None,
        "geglu": None,
        "spatial_merge": None,
        "projector_prep": None,
        "branch_spatial_merge": None,
        "branch_layernorm": None,
        "projector_gelu": None,
        "branch_gelu": None,
        "branch_concat": None,

        # Metadata ops (no kernel)
        "tokenizer": "metadata",  # Deprecated, use bpe_tokenizer
        "bpe_tokenizer": "metadata",  # BPE tokenizer init
        "wordpiece_tokenizer": "metadata",  # WordPiece tokenizer init
        "tiktoken_tokenizer": "metadata",  # TikToken tokenizer init
        "patch_embeddings": "metadata",  # Vision model patches
        "weight_tying": "metadata",
        "lm_head": "metadata",  # Signals separate lm_head weight (not tied)
    }
    BF16_DENSE_MATMUL_OPS = {
        "patch_proj",
        "patch_proj_aux",
        "qkv_packed_proj",
        "qkv_proj",
        "q_proj",
        "q_gate_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "mlp_gate_up",
        "mlp_up",
        "mlp_down",
        "projector_fc1",
        "projector_fc2",
        "branch_fc1",
        "branch_fc2",
        "assistant_pre_projection",
        "assistant_post_projection",
    }

    def map_op_to_kernel(op: str, layer_quant: Dict, mode: str, header_quant: Dict) -> List[str]:
        """
        Map template op → kernel ID(s).

        Logic:
            1. If metadata op → return []
            2. If has weight keys → lookup quant → find gemv/gemm kernel
            3. If fp32-only → find fp32 kernel

        Note: prefer_parallel is currently DISABLED (always False).
              OpenMP fork/join overhead makes per-kernel parallelism slower than
              serial for inference workloads (tested: 3.1 tok/s parallel vs 5.9
              tok/s serial on Qwen 0.5B). Needs a persistent thread pool instead
              of OpenMP #pragma omp parallel for. See gemv_omp.c for the kernel
              implementations — they are numerically correct but need a different
              threading model.
        """
        # DISABLED: OpenMP fork/join overhead (~50-200us per call) makes parallel
        # kernels slower for inference. Each decode token calls kernels 500+ times,
        # so thread management overhead dominates. Needs persistent thread pool.
        use_parallel = False  # Was: prefer_parallel and op in PARALLEL_OPS

        # Contract-bearing operations are selected before GraphIR is built.
        # Legacy template overrides and heuristic dispatch are not consulted.
        resolved_plan = numerical_contract_by_template_op.get(op)
        if resolved_plan is not None:
            if resolved_plan.get("phase") != mode:
                raise RuntimeError(
                    f"HARD CONTRACT FAULT: resolved {op!r} plan is for "
                    f"{resolved_plan.get('phase')!r}, not active mode {mode!r}."
                )
            return [str(resolved_plan["kernel"]["id"])]
        # Template-specified kernel overrides (keeps IR dumb and data-driven)
        if op == "rope_qk":
            return [_resolve_rope_qk_kernel(config, template_kernels)]
        if op == "rope_q":
            override = str(template_kernels.get("rope_q", "") or "").strip()
            if not override:
                raise RuntimeError("HARD KERNEL RESOLUTION FAULT: rope_q requires an exact circuit kernel mapping.")
            return [override]
        if op == "mrope_qk":
            override = str(template_kernels.get("mrope_qk", "") or "").strip()
            if not override:
                raise RuntimeError("HARD KERNEL RESOLUTION FAULT: mrope_qk requires an exact circuit kernel mapping.")
            return [override]
        if op == "position_embeddings":
            return [_resolve_position_embeddings_kernel(config, template_kernels)]
        if op == "kv_cache_store_shared_q":
            override = str(template_kernels.get(op, "") or "").strip()
            if not override:
                raise RuntimeError(f"HARD KERNEL RESOLUTION FAULT: {op} requires an exact circuit kernel mapping.")
            return [override]
        if op == "assistant_layer_scale":
            override = str(template_kernels.get(op, "") or "").strip()
            if not override:
                raise RuntimeError(f"HARD KERNEL RESOLUTION FAULT: {op} requires an exact circuit kernel mapping.")
            return [override]

        if op in ("attn_shared_kv", "attn_sliding_shared_kv"):
            if op == "attn_sliding_shared_kv":
                mode_key = f"attn_sliding_shared_kv_{mode}"
                attn_kernel = template_kernels.get(mode_key) or template_kernels.get("attn_sliding_shared_kv")
            else:
                mode_key = f"attn_shared_kv_{mode}"
                attn_kernel = template_kernels.get(mode_key) or template_kernels.get("attn_shared_kv")
            if attn_kernel:
                return [attn_kernel]
            raise RuntimeError(
                f"HARD KERNEL RESOLUTION FAULT: {op}.{mode} requires an exact circuit kernel mapping."
            )

        if op in ("attn", "attn_sliding"):
            mode_key = f"{op}_{mode}"
            attn_kernel = template_kernels.get(mode_key) or template_kernels.get(op)
            decode_kv_cache_dtype = str(config.get("decode_kv_cache_dtype", "fp32") or "fp32").strip().lower()
            if (
                op == "attn"
                and mode == "decode"
                and decode_kv_cache_dtype in {"fp16", "f16"}
                and attn_kernel == "attention_forward_decode_head_major_gqa_flash"
            ):
                attn_kernel = "attention_forward_decode_head_major_gqa_flash_f16cache"
            if attn_kernel:
                return [attn_kernel]
            if op == "attn" and mode == "prefill" and not _attention_contract_is_causal(template, config):
                return ["attention_forward_full_head_major_gqa_flash_strided"]

        # Kimi/DeepSeek MLA decompression has the same semantic template op for
        # FP32 and BF16 weights, but the C ABI differs for kv_b_proj.  Resolve it
        # from the manifest-derived layer_quant before honoring a generic template
        # default, otherwise BF16 weights can be cast as float*.
        if op == "kv_lora_decompress":
            kv_b_dtype = str(layer_quant.get("mla_kv_b_proj", "") or "").strip().lower()
            if kv_b_dtype == "bf16":
                return ["deepseek_mla_kv_decompress_bf16"]
            return ["deepseek_mla_kv_decompress_f32"]

        explicit_kernel = str(template_kernels.get(op, "") or "").strip()
        if explicit_kernel:
            explicit_weight_info = OP_TO_WEIGHT_KEYS.get(op)
            if op in BF16_DENSE_MATMUL_OPS and isinstance(explicit_weight_info, list) and explicit_weight_info:
                explicit_weight_dtype = str(layer_quant.get(explicit_weight_info[0], "fp32") or "fp32").lower()
                if explicit_weight_dtype == "fp32":
                    explicit_weight_dtype = str(header_quant.get(explicit_weight_info[0], explicit_weight_dtype) or explicit_weight_dtype).lower()
                if explicit_weight_dtype == "bf16":
                    return ["gemm_nt_bf16"]
            return [explicit_kernel]

        kernel_op = TEMPLATE_TO_KERNEL_OP.get(op)
        if not kernel_op:
            return []

        weight_info = OP_TO_WEIGHT_KEYS.get(op)

        # Metadata ops - no kernel
        if weight_info == "metadata":
            return []

        # Explicit no-weight math ops. These are real kernels, but they must not
        # flow through the header/footer weighted path below, which would invent
        # a q8_0 weight requirement and fail to bind kernels such as Gemma4 V RMSNorm.
        if op == "residual_save":
            return ["memcpy"]
        if op == "partial_rope_concat":
            return ["deepseek_mla_partial_rope_concat_packed_f32"]
        if op == "mla_kv_cache_batch_store":
            return ["deepseek_mla_kv_cache_batch_store_f32"]
        if op == "mla_kv_cache_store":
            return ["deepseek_mla_kv_cache_store_f32"]
        if op == "mla_attention":
            return ["deepseek_mla_attention_f32"]
        if op in {"v_norm", "projector_prep", "group_limited_topk_router", "mamba_in_proj_split"}:
            kernel_id = find_kernel(
                registry,
                op=kernel_op,
                quant={},
                mode=mode,
                prefer_q8_activation=prefer_q8_activation,
                prefer_parallel=use_parallel,
            )
            return [kernel_id] if kernel_id else []

        # Ops with quantized weights
        if isinstance(weight_info, list) and weight_info:
            # NOTE: For v7, qkv_proj uses standard gemm_nt_* (prefill) or gemv_* (decode)
            # The head-major QKV projection kernel (ck_qkv_project_head_major_quant)
            # was from ckernel_orchestration.c which is not used in v7.
            # Fall through to standard matmul handling which splits into q_proj, k_proj, v_proj.

            # NOTE: For v7, out_proj uses standard gemm_nt_* (prefill) or gemv_* (decode)
            # The head-major attention projection kernel (ck_attention_project_head_major_quant)
            # was from ckernel_orchestration.c which is not used in v7.
            # Fall through to standard matmul handling below.

            # Gemma4 per-layer prepare is a structured header op, not a
            # dense projection. It owns BF16/quantized weights internally and
            # must stay on its dedicated kernel regardless of source dtype.
            if op == "kv_a_layernorm":
                return ["rmsnorm_forward_kv_lora"]
            if op == "kv_lora_decompress":
                kv_b_dtype = str(layer_quant.get("mla_kv_b_proj", "")).lower()
                if isinstance(weight_info, list) and weight_info:
                    kv_b_dtype = str(weight_info[0].get("dtype", kv_b_dtype)).lower()
                if kv_b_dtype == "bf16":
                    return ["deepseek_mla_kv_decompress_bf16"]
                return ["deepseek_mla_kv_decompress_f32"]
            if op == "moe_swiglu_expert_mlp":
                gate_dtype = str(layer_quant.get("moe_expert_gate", "")).lower()
                up_dtype = str(layer_quant.get("moe_expert_up", "")).lower()
                down_dtype = str(layer_quant.get("moe_expert_down", "")).lower()
                if gate_dtype == "bf16" and up_dtype == "bf16" and down_dtype == "bf16":
                    return ["moe_swiglu_expert_forward_bf16"]
                return ["moe_swiglu_expert_forward_f32"]
            if op == "shared_swiglu_expert_mlp":
                gate_dtype = str(layer_quant.get("moe_shared_gate", "")).lower()
                up_dtype = str(layer_quant.get("moe_shared_up", "")).lower()
                down_dtype = str(layer_quant.get("moe_shared_down", "")).lower()
                if gate_dtype == "bf16" and up_dtype == "bf16" and down_dtype == "bf16":
                    return ["moe_swiglu_shared_forward_bf16"]
                return ["moe_swiglu_shared_forward_f32"]

            if op == "gemma4_per_layer_prepare":
                token_dtype = layer_quant.get("per_layer_token_emb", header_quant.get("per_layer_token_emb", ""))
                if token_dtype == "bf16":
                    return ["gemma4_per_layer_prepare_bf16_forward"]
                return ["gemma4_per_layer_prepare_forward"]

            if op == "moe_relu2_expert_mlp":
                up_dtype = str(layer_quant.get("moe_expert_up", "")).lower()
                down_dtype = str(layer_quant.get("moe_expert_down", "")).lower()
                if up_dtype == "q5_0" and down_dtype == "q8_0":
                    return ["moe_relu2_expert_forward_q5_0_q8_0"]
                if up_dtype == "q5_0" and down_dtype == "q5_0":
                    return ["moe_relu2_expert_forward_q5_0_q5_0"]
            if op == "shared_relu2_expert_mlp":
                up_dtype = str(layer_quant.get("moe_shared_up", "")).lower()
                down_dtype = str(layer_quant.get("moe_shared_down", "")).lower()
                if up_dtype == "q5_1" and down_dtype == "q8_0":
                    return ["moe_relu2_shared_forward_q5_1_q8_0"]

            # Try fused kernel first (e.g., qkv_projection)
            weight_dtype = layer_quant.get(weight_info[0], "fp32")
            if weight_dtype == "fp32":
                weight_dtype = header_quant.get(weight_info[0], weight_dtype)
            weight_dtype = str(weight_dtype or "fp32").lower()
            force_split_dense_qkv = op == "qkv_proj"
            if op in BF16_DENSE_MATMUL_OPS and weight_dtype == "bf16" and not force_split_dense_qkv:
                return ["gemm_nt_bf16"]
            kernel_prefer_q8_activation = _prefer_q8_activation_for_op(op, prefer_q8_activation)
            if op in ("mlp_gate_up", "mlp_up", "mlp_down") and prefer_fp32_mlp_matmuls:
                kernel_prefer_q8_activation = False
            if op == "projector_fc2":
                # projector_fc2 consumes the fp32 GELU output from projector_fc1.
                # There is no intervening quantize op in the vision footer, so it
                # must stay on an fp32-activation kernel.
                kernel_prefer_q8_activation = False
            allow_q8_contract = bool(
                weight_dtype == "q8_0"
                and (
                    op in q8_contract_ops
                    or (prefer_q8_contract and kernel_prefer_q8_activation)
                )
            )
            if allow_q8_contract:
                # Preserve the reference contract flow: select the FP32-activation
                # kernel first, then remap to the internal Q8 contract adapter.
                kernel_prefer_q8_activation = False
            kernel_id = None if force_split_dense_qkv else find_kernel(
                registry, op=kernel_op, quant={"weight": weight_dtype}, mode=mode,
                prefer_q8_activation=kernel_prefer_q8_activation,
                prefer_parallel=use_parallel
            )
            kernel_id = _maybe_apply_q8_contract(
                kernel_id,
                weight_dtype,
                allow_q8_contract=allow_q8_contract,
            )
            if kernel_id:
                return [kernel_id]

            # Fallback: split into individual matmuls
            # Return list of (kernel_id, split_op_name) tuples
            kernels = []
            # Map weight key to split op name
            weight_to_split_op = {
                "wq": "q_proj", "wk": "k_proj", "wv": "v_proj",
                "w1": "mlp_gate", "w3": "mlp_up", "w2": "mlp_down",
                "attn_qkv": "recurrent_qkv_proj",
                "attn_gate": "recurrent_gate_proj",
                "ssm_alpha": "recurrent_alpha_proj",
                "ssm_beta": "recurrent_beta_proj",
            }
            for w_key in weight_info:
                w_dtype = layer_quant.get(w_key, "fp32")
                if w_dtype == "fp32":
                    w_dtype = header_quant.get(w_key, w_dtype)
                w_dtype = str(w_dtype or "fp32").lower()
                split_op = weight_to_split_op.get(w_key, op)
                if split_op in BF16_DENSE_MATMUL_OPS and w_dtype == "bf16":
                    kernels.append(("gemm_nt_bf16", split_op))
                    continue
                split_prefer_q8_activation = _prefer_q8_activation_for_op(split_op, prefer_q8_activation)
                if split_op in ("mlp_gate_up", "mlp_down", "mlp_gate", "mlp_up") and prefer_fp32_mlp_matmuls:
                    split_prefer_q8_activation = False
                split_allow_q8_contract = bool(
                    prefer_q8_contract
                    and w_dtype == "q8_0"
                    and split_prefer_q8_activation
                )
                if split_allow_q8_contract:
                    split_prefer_q8_activation = False
                k = find_kernel(
                    registry, op="matmul", quant={"weight": w_dtype}, mode=mode,
                    prefer_q8_activation=split_prefer_q8_activation,
                    prefer_parallel=use_parallel
                )
                if k:
                    k = _maybe_apply_q8_contract(
                        k,
                        w_dtype,
                        allow_q8_contract=split_allow_q8_contract,
                    )
                    kernels.append((k, split_op))
            return kernels

        # Header/footer ops with weights (embedding, logits)
        if isinstance(weight_info, list) and not weight_info:
            # Header/footer ops with weights (embedding/logits).
            if op in ("dense_embedding_lookup", "embedding"):
                weight_dtype = header_quant.get("token_emb", "q8_0")
            elif op == "logits":
                if logits_weight_source == "lm_head":
                    weight_dtype = header_quant.get("lm_head")
                    if not weight_dtype:
                        raise RuntimeError(
                            "Logits contract failed: untied lm_head selected but lm_head dtype is missing "
                            "(expected output.weight/lm_head.weight in manifest quant summary)."
                        )
                else:
                    weight_dtype = header_quant.get("token_emb")
                    if not weight_dtype:
                        raise RuntimeError(
                            "Logits contract failed: tied logits selected but token_emb dtype is missing."
                        )
            else:
                weight_dtype = "q8_0"

            kernel_prefer_q8_activation = _prefer_q8_activation_for_op(op, prefer_q8_activation)
            if op == "logits" and prefer_fp32_logits:
                kernel_prefer_q8_activation = False
            allow_q8_contract = bool(
                prefer_q8_contract
                and weight_dtype == "q8_0"
                and kernel_prefer_q8_activation
            )
            if allow_q8_contract:
                kernel_prefer_q8_activation = False

            kernel_id = find_kernel(
                registry, op=kernel_op, quant={"weight": weight_dtype}, mode=mode,
                prefer_q8_activation=kernel_prefer_q8_activation,
                prefer_parallel=use_parallel
            )
            kernel_id = _maybe_apply_q8_contract(
                kernel_id,
                weight_dtype,
                allow_q8_contract=allow_q8_contract,
            )
            if op == "logits":
                print(
                    f"  [debug/logits] mode={mode} weight={weight_dtype} "
                    f"prefer_q8={kernel_prefer_q8_activation} "
                    f"prefer_fp32_logits={prefer_fp32_logits} -> {kernel_id}"
                )
            return [kernel_id] if kernel_id else []

        # Ops with fp32 weights or no weights
        kernel_id = find_kernel(
            registry, op=kernel_op, quant={"weight": "fp32"}, mode=mode,
            prefer_q8_activation=prefer_q8_activation,
            prefer_parallel=use_parallel
        )
        if kernel_id:
            return [kernel_id]

        # Try without weight quant requirement
        kernel_id = find_kernel(
            registry, op=kernel_op, quant={"weight": "none"}, mode=mode,
            prefer_q8_activation=prefer_q8_activation,
            prefer_parallel=use_parallel
        )
        return [kernel_id] if kernel_id else []

    # ═══════════════════════════════════════════════════════════
    # Parse template → Generate IR1
    # The builder walks the declared template graph and lowers explicit ops into
    # kernel calls. Keep the control vocabulary generic: if future templates add
    # branch/collect/stitch or route/dispatch/combine semantics, those should
    # arrive here as declared graph constructs, not as architecture-specific
    # if/else branches keyed on model families.
    # ═══════════════════════════════════════════════════════════

    # Track op instance counts during PASS 1 for data flow lookup
    pass1_instance_counts: Dict[tuple, int] = {}  # (layer, op_type) -> count

    def get_op_info(op_type: str, section: str, layer: int) -> dict:
        """Get op_id and instance for an op. Data flow is handled in IR Lower."""
        nonlocal pass1_instance_counts, global_op_id

        # Track instance for body ops (for repeated ops like rmsnorm, residual_add)
        if section == "body":
            key = (layer, op_type)
            instance = pass1_instance_counts.get(key, 0)
            pass1_instance_counts[key] = instance + 1
        else:
            instance = 0

        # Assign global op_id
        op_id = global_op_id
        global_op_id += 1

        return {
            "op_id": op_id,
            "instance": instance,
        }

    def annotate_branch_taps(emitted_start: int, section: str, layer: int, op_item: Dict[str, Any]) -> None:
        op_id = str(op_item.get("id", "") or "").strip()
        if not op_id or len(arranged_kernels) <= emitted_start:
            return
        taps = branch_tap_targets.get((section, layer, op_id))
        if not taps:
            return
        graph = arranged_kernels[-1].setdefault("graph", {})
        graph["branch_taps"] = copy.deepcopy(taps)

    def emit_branch_producers(section: str, layer_idx: int, op_item: Dict[str, Any], layer_quant: Dict[str, Any]) -> None:
        op_id = str(op_item.get("id", "") or "").strip()
        if not op_id:
            return
        taps = branch_tap_targets.get((section, layer_idx, op_id))
        if not taps:
            return

        merged_tokens = int(config.get("vision_merged_tokens", config.get("vision_num_patches", 0)) or 0)
        projector_out_dim = int(config.get("projector_out_dim", config.get("projection_dim", config.get("embed_dim", 0))) or 0)
        branch_op_alias = {
            "spatial_merge": "branch_spatial_merge",
            "layernorm": "branch_layernorm",
        }

        for tap in taps:
            branch_name = str(tap.get("name", "") or "").strip()
            collect_contract = tap.get("collect_contract") if isinstance(tap.get("collect_contract"), dict) else {}
            collect_target = str(
                collect_contract.get("target", f"branch.{branch_name or 'collect'}") or f"branch.{branch_name or 'collect'}"
            )
            collect_index = int(tap.get("collect_index", 0) or 0)
            collect_rows = int(collect_contract.get("rows", merged_tokens) or 0)
            collect_slice_dim = int(collect_contract.get("slice_dim", projector_out_dim) or 0)
            collect_item_bytes = int(collect_contract.get("bytes_per_elem", 4) or 4)
            collect_offset = collect_rows * collect_slice_dim * collect_index * collect_item_bytes
            producer_items = tap.get("producer_items", []) if isinstance(tap.get("producer_items"), list) else []

            for producer_item in producer_items:
                branch_op = str(producer_item.get("op", "") or "").strip()
                if not branch_op:
                    continue
                lowered_op = branch_op_alias.get(branch_op, branch_op)
                template_weight_refs = (
                    producer_item.get("weight_refs")
                    if isinstance(producer_item.get("weight_refs"), dict)
                    else {}
                )
                branch_quant = dict(layer_quant)
                if lowered_op == "branch_fc1" and "W" in template_weight_refs:
                    resolved = str(template_weight_refs["W"]).replace("{L}", str(layer_idx))
                    entry = weight_index.get(resolved)
                    if isinstance(entry, dict):
                        branch_quant["branch_fc1_w"] = str(entry.get("dtype", "fp32") or "fp32")
                if lowered_op == "branch_fc2" and "W" in template_weight_refs:
                    resolved = str(template_weight_refs["W"]).replace("{L}", str(layer_idx))
                    entry = weight_index.get(resolved)
                    if isinstance(entry, dict):
                        branch_quant["branch_fc2_w"] = str(entry.get("dtype", "fp32") or "fp32")
                kernels = map_op_to_kernel(lowered_op, branch_quant, mode, header_quant)

                params: Dict[str, Any] = copy.deepcopy(
                    producer_item.get("params") if isinstance(producer_item.get("params"), dict) else {}
                )
                if lowered_op == "branch_fc2":
                    params["branch_collect_target"] = collect_target
                    params["branch_collect_offset_bytes"] = collect_offset
                    params.setdefault("branch_collect_rows", collect_rows)
                    params.setdefault("branch_collect_slice_dim", collect_slice_dim)
                    params.setdefault("branch_collect_mode", collect_contract.get("mode", "concat"))
                    params.setdefault("branch_collect_axis", collect_contract.get("axis", "feature"))

                for k in kernels:
                    if isinstance(k, tuple):
                        kernel_id, split_op = k
                    else:
                        kernel_id, split_op = k, lowered_op
                    op_info = get_op_info(split_op, "branch", layer_idx)
                    arranged_kernels.append({
                        "op_id": op_info["op_id"],
                        "kernel": kernel_id,
                        "op": split_op,
                        "template_op_id": f"branch.{branch_name}.{producer_item.get('id', split_op)}",
                        "section": "branch",
                        "layer": layer_idx,
                        "instance": op_info["instance"],
                        "branch_name": branch_name,
                        "branch_source_layer": layer_idx,
                        "branch_collect_index": collect_index,
                        "template_weight_refs": copy.deepcopy(template_weight_refs),
                        "params": params,
                    })
                    print(
                        f"      [{op_info['op_id']:3d}] {split_op:20s} → {kernel_id}  "
                        f"(branch: {branch_name}, layer: {layer_idx})"
                    )

                if not kernels and OP_TO_WEIGHT_KEYS.get(lowered_op) != "metadata":
                    print(f"            {lowered_op:20s} → (no kernel)")

    for block_name in template["sequence"]:
        block_def = template["block_types"][block_name]
        block_sequence = block_def.get("sequence", ["header", "body", "footer"])

        print(f"\n  Block: {block_name}")

        for section_name in block_sequence:
            section_def = block_def.get(section_name)
            if section_def is None:
                continue

            # Get active ops list. Section items may carry ids/metadata even when
            # the lowerer only needs the op names today.
            if isinstance(section_def, dict):
                ops = _normalize_template_op_items(section_def.get("ops", []))
            else:
                ops = _normalize_template_op_items(section_def)

            # Body: loop over layers
            if section_name == "body":
                for layer_idx in range(num_layers):
                    layer_key = f"layer.{layer_idx}"
                    base_layer_quant = quant_summary.get(layer_key, {})
                    if not isinstance(base_layer_quant, dict):
                        base_layer_quant = {}
                    layer_quant = _apply_layer_quant_aliases(
                        base_layer_quant,
                        block["body"],
                        config,
                        layer_idx,
                    )
                    # Reset instance counts for each layer
                    pass1_instance_counts = {k: v for k, v in pass1_instance_counts.items()
                                             if k[0] != layer_idx}

                    print(f"\n    Layer {layer_idx}:")
                    layer_items = _resolve_body_items_for_layer(block["body"], config, layer_idx)
                    layer_ops = [item["op"] for item in layer_items]

                    # Track pre-norm instance for quantize insertion
                    norm_instance = 0
                    active_pre_norm_quant_contract: Optional[str] = None
                    active_pre_norm_quant_op_name: Optional[str] = None
                    active_pre_norm_quant_instance = 0

                    for op_idx, op_item in enumerate(layer_items):
                        op = op_item["op"]
                        emitted_start = len(arranged_kernels)

                        # Check if we need to insert quantize op after rmsnorm
                        # v7 compatibility: quantize activation before Q8_0 activation kernels
                        if op in PRE_NORM_OP_NAMES and op_idx + 1 < len(layer_ops):
                            next_op = layer_ops[op_idx + 1]
                            next_kernels = []
                            next_kernels.extend(
                                map_op_to_kernel(next_op, layer_quant, mode, header_quant)
                            )
                            needs_quantize = False

                            for nk in next_kernels:
                                nk_id = nk[0] if isinstance(nk, tuple) else nk
                                if kernel_needs_q8_activation(registry, nk_id):
                                    needs_quantize = True
                                    break

                            if needs_quantize:
                                # Insert quantize op after pre-norm (will be appended after op below)
                                pass  # Flag is set, handled below

                        # Insert residual_save BEFORE pre-norm to save input for skip connection
                        if should_insert_residual_save(layer_ops, op_idx):
                            residual_save_op_name = f"residual_save"
                            residual_save_info = get_op_info(residual_save_op_name, "body", layer_idx)
                            arranged_kernels.append({
                                "op_id": residual_save_info["op_id"],
                                "kernel": "memcpy",
                                "op": residual_save_op_name,
                                "template_op_id": op_item.get("id"),
                                "section": "body",
                                "layer": layer_idx,
                                "instance": norm_instance,  # Same instance as pre-norm
                                "_auto_inserted": True,
                            })
                            print(f"      [{residual_save_info['op_id']:3d}] {residual_save_op_name:20s} → memcpy  (inst: {norm_instance}) [AUTO-INSERTED before {op}]")

                        kernels = map_op_to_kernel(op, layer_quant, mode, header_quant)

                        # Check if we need to insert quantize op BEFORE out_proj or mlp_down
                        # v7 compatibility: quantize activation output before these projections
                        if op in ("out_proj", "mlp_down", "recurrent_out_proj", "mamba_out_proj") and kernels:
                            first_kernel = kernels[0]
                            fk_id = first_kernel[0] if isinstance(first_kernel, tuple) else first_kernel
                            if kernel_needs_q8_activation(registry, fk_id):
                                for kreg in registry.get("kernels", []):
                                    if kreg.get("id") == fk_id:
                                        act_dtype, quant_contract = (
                                            get_kernel_activation_quantization_contract(
                                                registry, fk_id, mode
                                            )
                                        )
                                        quantize_kernel = get_quantize_kernel_for_activation(
                                            registry, act_dtype, quant_contract
                                        )
                                        if quantize_kernel:
                                            quant_op_name = f"quantize_{op}_input"
                                            quant_op_info = get_op_info(quant_op_name, "body", layer_idx)
                                            arranged_kernels.append({
                                                "op_id": quant_op_info["op_id"],
                                                "kernel": quantize_kernel,
                                                "op": quant_op_name,
                                                "template_op_id": op_item.get("id"),
                                                "section": "body",
                                                "layer": layer_idx,
                                                "instance": 0,
                                            })
                                            print(f"      [{quant_op_info['op_id']:3d}] {quant_op_name:20s} → {quantize_kernel}  (inst: 0) [AUTO-INSERTED]")
                                        break

                        for k in kernels:
                            # Handle both plain kernel ID and (kernel_id, split_op) tuples
                            if isinstance(k, tuple):
                                kernel_id, split_op = k
                            else:
                                kernel_id, split_op = k, op

                            # A normed FP32 stream may feed consumers that share
                            # Q8_K storage but require different arithmetic
                            # contracts. Re-quantize at the exact consumer
                            # boundary instead of treating storage identity as
                            # numerical-contract identity.
                            if (
                                op in PRE_NORM_Q8_DIRECT_CONSUMERS
                                and kernel_needs_q8_activation(registry, kernel_id)
                            ):
                                act_dtype, required_contract = (
                                    get_kernel_activation_quantization_contract(
                                        registry, kernel_id, mode
                                    )
                                )
                                if required_contract != active_pre_norm_quant_contract:
                                    if not active_pre_norm_quant_op_name:
                                        raise RuntimeError(
                                            "HARD NUMERICAL CONTRACT FAULT: Q8 consumer "
                                            f"{kernel_id!r} has no active pre-norm quantizer."
                                        )
                                    quantize_kernel = get_quantize_kernel_for_activation(
                                        registry, act_dtype, required_contract
                                    )
                                    quant_op_info = get_op_info(
                                        active_pre_norm_quant_op_name, "body", layer_idx
                                    )
                                    arranged_kernels.append({
                                        "op_id": quant_op_info["op_id"],
                                        "kernel": quantize_kernel,
                                        "op": active_pre_norm_quant_op_name,
                                        "template_op_id": op_item.get("id"),
                                        "section": "body",
                                        "layer": layer_idx,
                                        "instance": active_pre_norm_quant_instance,
                                        "_auto_inserted": True,
                                    })
                                    print(
                                        f"      [{quant_op_info['op_id']:3d}] "
                                        f"{active_pre_norm_quant_op_name:20s} → "
                                        f"{quantize_kernel}  "
                                        f"(inst: {active_pre_norm_quant_instance}) "
                                        f"[AUTO-INSERTED contract switch]"
                                    )
                                    active_pre_norm_quant_contract = required_contract

                            # Get op_id and instance (data flow is handled in IR Lower)
                            op_info = get_op_info(split_op, "body", layer_idx)

                            arranged = {
                                "op_id": op_info["op_id"],
                                "kernel": kernel_id,
                                "op": split_op,
                                "template_op_id": op_item.get("id"),
                                "section": "body",
                                "layer": layer_idx,
                                "instance": op_info["instance"],
                                "params": copy.deepcopy(
                                    op_item.get("params") if isinstance(op_item.get("params"), dict) else {}
                                ),
                            }
                            graph_slots = _template_graph_slots(op_item)
                            if graph_slots:
                                arranged["graph_slots"] = graph_slots
                            arranged_kernels.append(arranged)
                            print(f"      [{op_info['op_id']:3d}] {split_op:20s} → {kernel_id}  (inst: {op_info['instance']})")

                        annotate_branch_taps(emitted_start, "body", layer_idx, op_item)
                        emit_branch_producers("body", layer_idx, op_item, layer_quant)

                        if not kernels and OP_TO_WEIGHT_KEYS.get(op) != "metadata":
                            print(f"            {op:20s} → (no kernel)")

                        # Insert one quantize op after rmsnorm if any consumer in
                        # this normed section needs a Q8 activation. Qwen3.5
                        # recurrent blocks have a FP32 qkv projection followed by
                        # a Q4_K/Q8_K gate projection, so checking only the
                        # immediate next op misses the required quantize.
                        if op in PRE_NORM_OP_NAMES and op_idx + 1 < len(layer_ops):
                            section_kernels = []
                            for future_op_item in layer_ops[op_idx + 1:]:
                                future_op = (
                                    future_op_item["op"]
                                    if isinstance(future_op_item, dict)
                                    else str(future_op_item)
                                )
                                if future_op in PRE_NORM_OP_NAMES:
                                    break
                                # Only projections that directly consume the
                                # normed stream should trigger quantize_input_N.
                                # Later consumers such as out_proj/mlp_down have
                                # their own quantize insertion after attention
                                # or activation transforms.
                                if future_op not in PRE_NORM_Q8_DIRECT_CONSUMERS:
                                    break
                                section_kernels.extend(
                                    map_op_to_kernel(future_op, layer_quant, mode, header_quant)
                                )

                            for nk in section_kernels:
                                nk_id = nk[0] if isinstance(nk, tuple) else nk
                                if kernel_needs_q8_activation(registry, nk_id):
                                    # Get activation dtype from kernel
                                    for kreg in registry.get("kernels", []):
                                        if kreg.get("id") == nk_id:
                                            act_dtype, quant_contract = (
                                                get_kernel_activation_quantization_contract(
                                                    registry, nk_id, mode
                                                )
                                            )
                                            quantize_kernel = get_quantize_kernel_for_activation(
                                                registry, act_dtype, quant_contract
                                            )
                                            if quantize_kernel:
                                                quant_op_name = f"quantize_input_{norm_instance}"
                                                quant_op_info = get_op_info(quant_op_name, "body", layer_idx)
                                                quant_arranged = {
                                                    "op_id": quant_op_info["op_id"],
                                                    "kernel": quantize_kernel,
                                                    "op": quant_op_name,
                                                    "template_op_id": op_item.get("id"),
                                                    "section": "body",
                                                    "layer": layer_idx,
                                                    "instance": norm_instance,
                                                }
                                                if op == "block_rmsnorm":
                                                    quant_arranged["graph_slots"] = {"inputs": {"input": "layer_input"}}
                                                arranged_kernels.append(quant_arranged)
                                                active_pre_norm_quant_contract = quant_contract
                                                active_pre_norm_quant_op_name = quant_op_name
                                                active_pre_norm_quant_instance = norm_instance
                                                print(f"      [{quant_op_info['op_id']:3d}] {quant_op_name:20s} → {quantize_kernel}  (inst: {norm_instance}) [AUTO-INSERTED]")
                                            break
                                    break
                            norm_instance += 1

            # Header/Footer: run once (no layer quant)
            else:
                print(f"\n    {section_name.capitalize()}:")
                footer_quantize_inserted = False  # Track if we've inserted quantize for footer
                for op_idx, op_item in enumerate(ops):
                    op = op_item["op"]
                    emitted_start = len(arranged_kernels)
                    if op == "patch_embeddings":
                        for patch_op in ("patchify", "patch_proj"):
                            kernels = map_op_to_kernel(patch_op, {}, mode, header_quant)
                            for k in kernels:
                                if isinstance(k, tuple):
                                    kernel_id, split_op = k
                                else:
                                    kernel_id, split_op = k, patch_op
                                op_info = get_op_info(split_op, section_name, -1)
                                arranged_kernels.append({
                                    "op_id": op_info["op_id"],
                                    "kernel": kernel_id,
                                    "op": split_op,
                                    "template_op_id": op_item.get("id"),
                                    "section": section_name,
                                    "layer": -1,
                                    "instance": op_info["instance"],
                                    "params": copy.deepcopy(
                                        op_item.get("params") if isinstance(op_item.get("params"), dict) else {}
                                    ),
                                })
                                print(f"      [{op_info['op_id']:3d}] {split_op:20s} → {kernel_id}  (inst: {op_info['instance']})")
                            if not kernels:
                                print(f"            {patch_op:20s} → (no kernel)")
                        annotate_branch_taps(emitted_start, section_name, -1, op_item)
                        continue
                    kernels = map_op_to_kernel(op, {}, mode, header_quant)

                    # Footer: Insert quantize op BEFORE any op that needs Q8 activation
                    # (after rmsnorm outputs FP32, before logits needs Q8_0)
                    if section_name == "footer" and not footer_quantize_inserted:
                        for k in kernels:
                            k_id = k[0] if isinstance(k, tuple) else k
                            if kernel_needs_q8_activation(registry, k_id):
                                # Get activation dtype from kernel
                                for kreg in registry.get("kernels", []):
                                    if kreg.get("id") == k_id:
                                        act_dtype, quant_contract = (
                                            get_kernel_activation_quantization_contract(
                                                registry, k_id, mode
                                            )
                                        )
                                        quantize_kernel = get_quantize_kernel_for_activation(
                                            registry, act_dtype, quant_contract
                                        )
                                        if quantize_kernel:
                                            quant_op_name = "quantize_final_output"
                                            quant_op_info = get_op_info(quant_op_name, section_name, -1)
                                            arranged_kernels.append({
                                                "op_id": quant_op_info["op_id"],
                                                "kernel": quantize_kernel,
                                                "op": quant_op_name,
                                                "template_op_id": op_item.get("id"),
                                                "section": section_name,
                                                "layer": -1,
                                                "instance": 0,
                                            })
                                            print(f"      [{quant_op_info['op_id']:3d}] {quant_op_name:20s} → {quantize_kernel}  (inst: 0) [AUTO-INSERTED before {op}]")
                                            footer_quantize_inserted = True
                                        break
                                break

                    for k in kernels:
                        # Handle both plain kernel ID and (kernel_id, split_op) tuples
                        if isinstance(k, tuple):
                            kernel_id, split_op = k
                        else:
                            kernel_id, split_op = k, op

                        # Get op_id and instance (data flow is handled in IR Lower)
                        op_info = get_op_info(split_op, section_name, -1)

                        arranged = {
                            "op_id": op_info["op_id"],
                            "kernel": kernel_id,
                            "op": split_op,
                            "template_op_id": op_item.get("id"),
                            "section": section_name,
                            "layer": -1,
                            "instance": op_info["instance"],
                            "params": copy.deepcopy(
                                op_item.get("params") if isinstance(op_item.get("params"), dict) else {}
                            ),
                        }
                        graph_slots = _template_graph_slots(op_item)
                        if graph_slots:
                            arranged["graph_slots"] = graph_slots
                        arranged_kernels.append(arranged)
                        print(f"      [{op_info['op_id']:3d}] {split_op:20s} → {kernel_id}  (inst: {op_info['instance']})")

                    annotate_branch_taps(emitted_start, section_name, -1, op_item)

                    if not kernels:
                        if OP_TO_WEIGHT_KEYS.get(op) == "metadata":
                            print(f"            {op:20s} → (metadata)")
                        else:
                            print(f"            {op:20s} → (no kernel)")

    print(f"\n✓ Pass 1: Generated {len(arranged_kernels)} kernel calls")

    # GraphIR records the selected provider's execution capability for every
    # versioned kernel map. Later stages may carry it but must not reinterpret it.
    for arranged in arranged_kernels:
        kernel_id = str(arranged.get("kernel", ""))
        capability = kernel_execution_capabilities.get(kernel_id)
        if capability is not None:
            arranged["resolved_execution"] = _graph_ir_execution_metadata(capability)
        plan = numerical_contract_by_template_op.get(str(arranged.get("op", "")))
        if plan is None:
            continue
        arranged["required_contract"] = copy.deepcopy(plan["requirements"])
        arranged["resolved_contract"] = _graph_ir_contract_metadata(plan)

    _attach_semantic_checkpoints(template, arranged_kernels, registry)

    # ═══════════════════════════════════════════════════════════
    # PASS 1.5: Add dataflow information
    # For each op, record what it reads from and writes to
    # This enables the memory planner to assign physical buffers
    # ═══════════════════════════════════════════════════════════
    print(f"\n  Pass 1.5: Computing dataflow graph...")

    current_layer = -1
    for ir_op in arranged_kernels:
        op_id = ir_op["op_id"]
        op_type = ir_op["op"]
        layer = ir_op["layer"]
        instance = ir_op.get("instance", 0)

        # Reset tracker for new layer
        if layer != current_layer and layer >= 0:
            dataflow_tracker.reset_for_layer(layer)
            current_layer = layer

        # Record dataflow for this op (override input slot based on kernel activation dtype)
        kernel_id = ir_op.get("kernel")
        input_override = _input_slot_override_for_kernel(op_type, kernel_id)
        graph_slots = ir_op.get("graph_slots", {}) if isinstance(ir_op.get("graph_slots"), dict) else {}
        explicit_input_override = graph_slots.get("inputs") if isinstance(graph_slots.get("inputs"), dict) else {}
        explicit_output_override = graph_slots.get("outputs") if isinstance(graph_slots.get("outputs"), dict) else {}
        # Template graph_slots describe the semantic producer/consumer edge.
        # Kernel activation dtype decides the physical view of that edge. For
        # example GLM4 q_proj semantically consumes main_stream, but a Q4_K x
        # Q8_K kernel must read main_stream_q8 produced by quantize_input_0.
        # Keep the kernel ABI override last so explicit circuit slots cannot
        # accidentally wire a quantized kernel back to the FP32 stream.
        merged_input_override = dict(explicit_input_override or {})
        kernel_act = str(kernel_act_dtype.get(kernel_id, "fp32") or "fp32").lower() if kernel_id else "fp32"
        if input_override:
            # Explicit template graph slots are the semantic edge. For FP32-activation
            # kernels no physical view remap is needed, so preserve explicit sources
            # such as Kimi's block_rmsnorm -> layer_input. Quantized kernels still
            # override to the Q8 physical view produced by an inserted quantize op.
            if not (explicit_input_override and kernel_act == "fp32"):
                merged_input_override.update(input_override)
        dataflow_info = dataflow_tracker.record_op(
            op_id,
            op_type,
            layer,
            instance,
            merged_input_override or None,
            explicit_output_override or None,
        )
        ir_op["dataflow"] = dataflow_info

    # Print dataflow stats
    stats = dataflow_tracker.get_stats()
    print(f"  ✓ Pass 1.5: Added dataflow to {len(arranged_kernels)} ops")
    print(f"    Active slots: {', '.join(stats['slots_active'])}")

    # ═══════════════════════════════════════════════════════════
    # PASS 2: Bind weights from sidecar entries
    # Uses instance counts from PASS 1 (stored in ir_op["instance"])
    # ═══════════════════════════════════════════════════════════
    print(f"\n  Pass 2: Binding weights from sidecar...")

    # Mapping for repeated ops: (op_type, instance_index) -> weight_keys
    # Instance index is 0-based (first occurrence = 0)
    REPEATED_OP_WEIGHTS = {
        # rmsnorm: 1st (pre-attention) uses ln1_gamma, 2nd (pre-MLP) uses ln2_gamma
        ("rmsnorm", 0): ["ln1_gamma"],      # Pre-attention norm
        ("rmsnorm", 1): ["ln2_gamma"],      # Pre-MLP norm
        ("layernorm", 0): ["ln1_gamma", "ln1_beta"],  # Pre-attention norm
        ("layernorm", 1): ["ln2_gamma", "ln2_beta"],  # Pre-MLP norm
        ("attn_norm", 0): ["ln1_gamma"],    # Pre-attention norm (Gemma)
        ("block_rmsnorm", 0): ["ln1_gamma"], # Generic pre-block norm
        ("block_rmsnorm", 1): ["post_attention_norm"], # Second Kimi MLA block norm before MLP/MoE
        ("ffn_norm", 0): ["ln2_gamma"],     # Pre-MLP norm (Gemma)
        ("post_attention_norm", 0): ["post_attention_norm"],
        ("post_ffn_norm", 0): ["post_ffn_norm"],
        ("v_norm", 0): [],
        # residual_add: both instances use same weights (none), but tracked for consistency
        ("residual_add", 0): [],            # Post-attention residual
        ("residual_add", 1): [],            # Post-MLP residual
    }

    # Footer-specific weights (no instance tracking needed)
    FOOTER_OP_WEIGHTS = {
        "rmsnorm": ["final_ln_weight", "final_ln_bias"],
        "layernorm": ["final_ln_weight", "final_ln_bias"],
        "final_rmsnorm": ["final_ln_weight", "final_ln_bias"],
    }

    def _footer_weight_keys(op_name: str) -> List[str]:
        if op_name == "logits":
            if logits_weight_source == "none":
                return []
            return ["lm_head"] if logits_weight_source == "lm_head" else ["token_emb"]
        if op_name in FOOTER_OP_WEIGHTS:
            return FOOTER_OP_WEIGHTS[op_name]
        return TEMPLATE_OP_WEIGHTS.get(op_name, [])

    def _header_weight_keys(op_name: str) -> List[str]:
        if op_name == "patch_proj" and "patch_proj_aux" in header_ops and "patch_bias_add" in header_ops:
            # Qwen3-VL applies the shared patch bias after the dual projection streams
            # are merged, so keep the first projection weight-only here.
            return ["patch_emb"]
        return TEMPLATE_OP_WEIGHTS.get(op_name, [])

    def resolve_weight_name(weight_key: str, op_section: str, op_layer: int) -> Optional[str]:
        patterns = WEIGHT_PATTERNS.get(weight_key, [weight_key])
        candidates: List[str] = []
        for pattern in patterns:
            name = str(pattern)
            if op_section == "body":
                name = name.replace("{L}", str(op_layer))
            candidates.append(name)

        # Back-compat direct fallback.
        direct = f"layer.{op_layer}.{weight_key}" if op_section == "body" else str(weight_key)
        if direct not in candidates:
            candidates.append(direct)

        for cand in candidates:
            if cand in weight_index:
                return cand
        return None

    def resolve_branch_weight_name(ir_op: Dict[str, Any], weight_key: str) -> Optional[str]:
        explicit_refs = ir_op.get("template_weight_refs") if isinstance(ir_op.get("template_weight_refs"), dict) else {}
        branch_layer_raw = ir_op.get("branch_source_layer", ir_op.get("layer", -1))
        try:
            branch_layer = int(branch_layer_raw)
        except Exception:
            branch_layer = -1
        explicit = explicit_refs.get(_resolve_branch_weight_ref_alias(weight_key))
        if isinstance(explicit, str) and explicit.strip():
            cand = explicit.replace("{L}", str(branch_layer))
            if cand in weight_index:
                return cand
        if branch_layer < 0:
            return None
        patterns = WEIGHT_PATTERNS.get(weight_key, [weight_key])
        for pattern in patterns:
            cand = str(pattern).replace("{L}", str(branch_layer))
            if cand in weight_index:
                return cand
        return None

    for ir_op in arranged_kernels:
        op = ir_op["op"]
        layer = ir_op["layer"]
        section = ir_op["section"]

        # Use instance from PASS 1 (already computed with data flow)
        instance_idx = ir_op.get("instance", 0)

        # Get weight keys for this op - check repeated op mapping first
        if section == "branch":
            explicit_refs = ir_op.get("template_weight_refs") if isinstance(ir_op.get("template_weight_refs"), dict) else {}
            branch_weight_map = {
                "layernorm": ["branch_norm_gamma", "branch_norm_beta"],
                "branch_layernorm": ["branch_norm_gamma", "branch_norm_beta"],
                "branch_fc1": ["branch_fc1_w", "branch_fc1_b"],
                "branch_fc2": ["branch_fc2_w", "branch_fc2_b"],
            }
            weight_keys = list(branch_weight_map.get(op, []))
            if not weight_keys and explicit_refs:
                weight_keys = list(explicit_refs.keys())
        elif _circuit_op_weight_keys(template, section, op) is not None:
            weight_keys = _circuit_op_weight_keys(template, section, op) or []
        elif section == "body" and (op, instance_idx) in REPEATED_OP_WEIGHTS:
            weight_keys = REPEATED_OP_WEIGHTS[(op, instance_idx)]
        elif section == "header":
            weight_keys = _header_weight_keys(op)
        elif section == "footer":
            weight_keys = _footer_weight_keys(op)
        else:
            weight_keys = TEMPLATE_OP_WEIGHTS.get(op, [])

        ir_op["weights"] = {}

        for wkey in weight_keys:
            if section == "branch":
                weight_name = resolve_branch_weight_name(ir_op, str(wkey))
            else:
                weight_name = resolve_weight_name(str(wkey), section, int(layer))

            # Look up in manifest entries
            if weight_name and weight_name in weight_index:
                entry = weight_index[weight_name]
                ir_op["weights"][wkey] = {
                    "name": weight_name,
                    "offset": _entry_offset(entry),
                    "size": _entry_size(entry),
                    "dtype": entry.get("dtype", "unknown"),
                }
            else:
                # Weight not found - might be optional (biases)
                pass

    # Count weights bound
    total_weights = sum(len(op["weights"]) for op in arranged_kernels)
    print(f"  ✓ Pass 2: Bound {total_weights} weights to {len(arranged_kernels)} ops")

    # ==========================================================================
    # POST-IR1 COMPLETENESS CHECK
    # Validate that no template ops were silently dropped.
    # This catches the class of bugs where a required kernel returns None
    # and the op is silently skipped (e.g., Gemma logits drop).
    # ==========================================================================
    _check_ir1_completeness(manifest, arranged_kernels)
    _validate_resolved_kernels_are_emitted(numerical_contract_plans, arranged_kernels)

    return arranged_kernels


def _check_ir1_completeness(manifest: Dict, ir1_ops: List[Dict]) -> None:
    """
    Verify that all expected template ops are present in IR1.

    This catches silent kernel drops where find_kernel() returns None
    but the error is not propagated.

    Handles:
    - op splitting (qkv_proj → q_proj + k_proj + v_proj)
    - metadata ops (tokenizer, weight_tying, lm_head - not in IR1)
    - optional ops (post_attention_norm, post_ffn_norm - only if weights exist)

    Raises:
        RuntimeError: If required ops are missing from IR1
    """
    template = manifest.get("template", {})
    if not template or "sequence" not in template:
        return  # Can't validate without template

    # Op groups for validation
    # Only include splits if both parts are real ops that can exist in IR1
    SPLIT_OPS = {
        "patch_embeddings": ["patchify", "patch_proj"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "q_gate_proj": ["q_gate_proj"],
        "recurrent_qkv_proj": ["recurrent_qkv_proj"],
        "recurrent_gate_proj": ["recurrent_gate_proj"],
        "recurrent_alpha_proj": ["recurrent_alpha_proj"],
        "recurrent_beta_proj": ["recurrent_beta_proj"],
        "recurrent_packed_proj": [
            "recurrent_qkv_proj",
            "recurrent_gate_proj",
            "recurrent_alpha_proj",
            "recurrent_beta_proj",
        ],
        # mlp_gate_up -> mlp_gate + mlp_up is not a real split pattern
        # mlp_gate_up produces gate+up tensor, geglu/silu_mul processes it
    }

    NON_KERNEL_OPS = {
        "bpe_tokenizer",
        "wordpiece_tokenizer",
        "tiktoken_tokenizer",
        "tokenizer",
        "weight_tying",
        "lm_head",
        "dense_embedding_lookup",  # Meta-kernel, expanded to embedding_forward
    }

    OPTIONAL_OPS = {
        "post_attention_norm",
        "post_ffn_norm",
    }

    # Parse template structure correctly
    block_name = template["sequence"][0]
    block = template["block_types"].get(block_name, {})

    # Extract ops from header, body, and footer
    header_ops = _extract_template_ops(block.get("header", []))
    body_ops = _collect_body_ops_for_validation(block.get("body", {}), manifest.get("config", {}))
    footer_ops = _extract_template_ops(block.get("footer", []))

    branch_ops: List[str] = []
    for branch in _build_block_branch_plan(block, manifest.get("config", {})):
        branch_ops.extend(branch.get("producer_ops", []))

    template_ops = header_ops + body_ops + branch_ops + footer_ops

    # Determine which optional ops are present in manifest
    manifest_entries = {e.get("name", "") for e in manifest.get("entries", [])}

    def optional_present(op: str) -> bool:
        """Check if optional op should be present based on manifest weights."""
        if op == "post_attention_norm":
            return any("post_attention_norm" in n for n in manifest_entries)
        if op == "post_ffn_norm":
            # Handle both naming conventions (ln3 vs post_ffn_norm)
            return any(x in n for x in ["post_ffn_norm", "post_ffw_norm", "ln3"] for n in manifest_entries)
        return True

    # Collect actual ops from IR1 (use "op" field, not "kernel")
    actual_ops = {op.get("op", "") for op in ir1_ops}

    missing = []

    for op in template_ops:
        # Skip metadata ops - they don't generate IR1 kernels
        if op in NON_KERNEL_OPS:
            continue

        # Skip optional ops that aren't present in manifest
        if op in OPTIONAL_OPS and not optional_present(op):
            continue

        # Handle split ops: accept either fused or split versions
        if op in SPLIT_OPS:
            split_required = SPLIT_OPS[op]
            # Check if fused version exists
            if op in actual_ops:
                continue
            # Check if all split versions exist
            if all(x in actual_ops for x in split_required):
                continue
            # Neither fused nor split found - mark as missing
            missing.append(f"{op} (requires: {split_required})")
        else:
            # Regular op - must exist in IR1
            if op not in actual_ops:
                missing.append(op)

    if missing:
        raise RuntimeError(
            f"\n❌ HARD FAULT: Incomplete IR1 - {len(missing)} ops silently dropped\n"
            f"   Missing ops: {sorted(missing)}\n"
            f"   Template ops: {sorted(set(template_ops))}\n"
            f"   Actual IR1 ops: {sorted(actual_ops)}\n"
            f"   This indicates find_kernel() returned None for required ops.\n"
            f"   Fix: declare and register the exact required kernel operation.\n"
        )

    print(f"  ✓ IR1 completeness check passed ({len(template_ops)} expected, {len(ir1_ops)} generated)")


def apply_fusion_pass(ir1_ops: List[Dict], registry: Dict, mode: str, no_fusion: bool = False) -> tuple[List[Dict], Dict]:
    """
    Apply fusion pass to combine consecutive kernels where fused versions exist.

    Args:
        ir1_ops: List of IR1 ops (each is {kernel, op, section, layer, weights})
        registry: Kernel registry
        mode: Execution mode (decode/prefill)
        no_fusion: If True, skip fusion and return original ops

    Returns:
        (fused_ops, fusion_stats) - New op list after fusion and statistics

    Fusion strategy:
        1. Scan registry for kernels with "fuses" field
        2. Match consecutive kernel sequences
        3. Replace with fused kernel, merge weights
        4. Track fusion statistics
    """
    print(f"\n{'='*60}")
    print("FUSION PASS")
    print(f"{'='*60}")

    # Check for fusion disable flag (parameter only)
    if no_fusion:
        print("  ⚠️ Fusion DISABLED (--no-fusion)")
        return ir1_ops, {"total_fusions": 0, "kernels_removed": 0, "fusions_applied": [], "disabled": True}

    # Build fusion patterns from registry
    fusion_patterns = []
    for kernel in registry["kernels"]:
        if "fuses" not in kernel:
            continue

        # Check if this fused kernel matches the mode
        # NOTE: Allow prefill fused kernels in decode mode (v7 baseline parity)
        # The fused prefill kernels work for tokens=1 (decode) and are more accurate
        # because they handle quantization internally.
        variant = kernel.get("variant", "")
        # Don't skip prefill kernels in decode mode - they work with tokens=1
        # if mode == "decode" and "prefill" in variant and "decode" not in variant:
        #     continue
        if mode == "prefill" and "decode" in variant and "prefill" not in variant:
            continue

        pattern = {
            "fused_kernel": kernel["id"],
            "fused_op": kernel.get("op", ""),
            "sequence": kernel.get("fuses", []),
            "variant": variant,
        }
        fusion_patterns.append(pattern)

    print(f"\nFound {len(fusion_patterns)} fusion patterns in registry for {mode} mode")

    # Apply fusion patterns
    fused_ops = [op.copy() for op in ir1_ops]  # Deep copy to avoid mutation
    for op in fused_ops:
        op["weights"] = op.get("weights", {}).copy()

    fusion_stats = {
        "total_fusions": 0,
        "kernels_removed": 0,
        "fusions_applied": [],
    }

    # Sort patterns by sequence length (longest first) for greedy matching
    fusion_patterns.sort(key=lambda p: -len(p["sequence"]))

    changed = True
    while changed:
        changed = False
        for pattern in fusion_patterns:
            sequence = pattern["sequence"]
            seq_len = len(sequence)

            # Scan for matching sequences
            i = 0
            while i <= len(fused_ops) - seq_len:
                # Check if sequence matches (compare kernel/function IDs)
                match = True
                for j in range(seq_len):
                    # Use "function" field if "kernel" not present
                    op_kernel = fused_ops[i + j].get("kernel") or fused_ops[i + j].get("function", "")
                    if op_kernel != sequence[j]:
                        match = False
                        break

                if match:
                    # Safety: if the first op is a quantize op, check it has
                    # exactly 1 consumer. Shared quantize ops (e.g. quantize_input_0
                    # feeding q/k/v projections) must NOT be fused.
                    first_op = fused_ops[i]
                    first_kernel = first_op.get("kernel", "")
                    if first_kernel.startswith("quantize_row_"):
                        first_op_id = first_op.get("op_id")
                        if first_op_id is not None:
                            consumer_count = sum(
                                1 for op in fused_ops
                                if any(
                                    inp.get("from_op") == first_op_id
                                    for inp in op.get("dataflow", {}).get("inputs", {}).values()
                                )
                            )
                            if consumer_count > 1:
                                print(f"\n  Skipping fusion at position {i}: "
                                      f"{first_kernel} (op_id={first_op_id}) has "
                                      f"{consumer_count} consumers")
                                i += 1
                                continue

                    # Found a match - replace with fused kernel
                    fused_id = pattern["fused_kernel"]
                    removed_ops = fused_ops[i:i+seq_len]
                    removed_kernels = [op.get("kernel") or op.get("function", "?") for op in removed_ops]

                    governed = [op for op in removed_ops if op.get("resolved_contract")]
                    if governed:
                        governed_names = [op.get("op") for op in governed]
                        raise RuntimeError(
                            "HARD CONTRACT FAULT: fusion attempted to replace contract-bearing "
                            f"GraphIR operations {governed_names} with {fused_id!r}. "
                            "Declare and resolve a compatible fused-kernel contract before enabling "
                            "this fusion; do not silently discard the resolved provider."
                        )

                    print(f"\n  Fusion opportunity at position {i}:")
                    print(f"    Replacing: {' + '.join(removed_kernels)}")
                    print(f"    With:      {fused_id}")

                    # Merge weights from all fused ops
                    merged_weights = {}
                    for op in removed_ops:
                        merged_weights.update(op.get("weights", {}))

                    # Build correct dataflow for fused op:
                    # - Input: first op's input (FP32 from rmsnorm, renamed to "x")
                    # - Output: last op's output
                    first_dataflow = removed_ops[0].get("dataflow", {})
                    last_dataflow = removed_ops[-1].get("dataflow", {})
                    # Find the "primary" op (gemv) for op name and instance
                    middle_op = removed_ops[1] if seq_len >= 3 else removed_ops[0]

                    fused_dataflow = {}
                    if first_dataflow.get("inputs"):
                        # Rename first op's input key to "x" for fused kernel
                        first_inputs = first_dataflow["inputs"]
                        # Get the first (and typically only) input
                        first_input_key = next(iter(first_inputs))
                        first_input_val = first_inputs[first_input_key]
                        fused_dataflow["inputs"] = {
                            "x": {**first_input_val, "dtype": "fp32"}
                        }
                    if last_dataflow.get("outputs"):
                        fused_dataflow["outputs"] = last_dataflow["outputs"]

                    # Create fused op preserving the primary op's identity
                    fused_op = {
                        "kernel": fused_id,
                        "op": middle_op.get("op", "fused"),
                        "section": removed_ops[0]["section"],
                        "layer": removed_ops[0]["layer"],
                        "instance": middle_op.get("instance", 0),
                        "weights": merged_weights,
                        "fused_from": removed_kernels,
                    }
                    if fused_dataflow:
                        fused_op["dataflow"] = fused_dataflow

                    # Replace sequence with fused op
                    fused_ops[i:i+seq_len] = [fused_op]

                    # Record fusion
                    fusion_stats["fusions_applied"].append({
                        "position": i,
                        "pattern": pattern["fused_op"],
                        "fused_kernel": fused_id,
                        "replaced": removed_kernels,
                    })
                    fusion_stats["total_fusions"] += 1
                    fusion_stats["kernels_removed"] += seq_len - 1

                    changed = True
                    break  # Restart scan after modification

                i += 1

            if changed:
                break  # Restart with new fusion_patterns iteration

    print(f"\n✓ Fusion complete:")
    print(f"  Total fusions: {fusion_stats['total_fusions']}")
    print(f"  Kernels removed: {fusion_stats['kernels_removed']}")
    print(f"  Final kernel count: {len(fused_ops)} (was {len(ir1_ops)})")

    return fused_ops, fusion_stats


def insert_bias_add_ops(
    ir_ops: List[Dict],
    registry: Dict,
    manifest: Dict,
    mode: str,
    manifest_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Insert explicit bias_add ops after projections when kernels do not apply bias.

    This keeps biases visible in the lowered IR and avoids hiding them in codegen.
    """
    # Only insert if bias_add kernel exists
    if not any(k.get("id") == "bias_add" for k in registry.get("kernels", [])):
        print("  Warning: bias_add kernel not found in registry; skipping bias ops")
        return ir_ops

    kernel_maps_dir = V8_ROOT / "kernel_maps"
    kernel_map_cache: Dict[str, Dict] = {}
    entry_by_name: Dict[str, Dict[str, Any]] = {
        e.get("name"): e for e in (manifest.get("entries", []) or []) if e.get("name")
    }
    bias_zero_cache: Dict[str, bool] = {}
    bump_path: Optional[Path] = None
    if manifest_path:
        candidate = manifest_path.parent / "weights.bump"
        if candidate.exists():
            bump_path = candidate

    def load_kernel_map(kernel_id: str) -> Optional[Dict]:
        if kernel_id in kernel_map_cache:
            return kernel_map_cache[kernel_id]
        kernel_file = kernel_maps_dir / f"{kernel_id}.json"
        if kernel_file.exists():
            with open(kernel_file, "r") as f:
                kernel_map_cache[kernel_id] = json.load(f)
                return kernel_map_cache[kernel_id]
        # fallback to registry entry
        for k in registry.get("kernels", []):
            if k.get("id") == kernel_id:
                kernel_map_cache[kernel_id] = k
                return k
        kernel_map_cache[kernel_id] = {}
        return {}

    def kernel_supports_bias(kernel_id: str) -> bool:
        km = load_kernel_map(kernel_id)
        bias_inputs = {"bias", "bq", "bk", "bv", "bo", "b1", "b2"}
        for inp in km.get("inputs", []):
            name = inp.get("name", "")
            if name in bias_inputs or "bias" in name:
                return True
        for w in km.get("weights", []):
            name = w.get("name", "")
            if name == "bias" or name.startswith("b") or "bias" in name:
                return True
        return False

    def is_zero_bias_tensor(weight_name: str) -> bool:
        """
        Return True if a bias tensor is all zeros in weights.bump.
        Falls back to False on missing metadata/files.
        """
        if weight_name in bias_zero_cache:
            return bias_zero_cache[weight_name]
        if bump_path is None:
            bias_zero_cache[weight_name] = False
            return False
        entry = entry_by_name.get(weight_name)
        if not entry:
            bias_zero_cache[weight_name] = False
            return False
        dtype = str(entry.get("dtype", "")).lower()
        if dtype not in ("fp32", "f32", "float32"):
            bias_zero_cache[weight_name] = False
            return False
        size = _entry_size(entry)
        file_offset = _entry_offset(entry)
        if size <= 0 or file_offset is None:
            bias_zero_cache[weight_name] = False
            return False

        try:
            with open(bump_path, "rb") as f:
                f.seek(int(file_offset))
                remaining = size
                zero = True
                while remaining > 0:
                    chunk = f.read(min(remaining, 16384))
                    if not chunk:
                        break
                    # Fast path: any non-zero byte means tensor has non-zero values.
                    if any(b != 0 for b in chunk):
                        zero = False
                        break
                    remaining -= len(chunk)
                if remaining > 0:
                    zero = False
            bias_zero_cache[weight_name] = zero
            return zero
        except Exception:
            bias_zero_cache[weight_name] = False
            return False

    bias_key_by_op = {
        "qkv_packed_proj": "bqkv",
        "q_proj": "bq",
        "q_gate_proj": "bq",
        "k_proj": "bk",
        "v_proj": "bv",
        "out_proj": "bo",
        "mlp_gate_up": "b1",
        "mlp_up": "b1",
        "mlp_down": "b2",
        "projector_fc1": "mm0_b",
        "projector_fc2": "mm1_b",
    }

    config = manifest.get("config", {})
    out: List[Dict] = []
    inserted = 0
    skipped_zero = 0

    for op in ir_ops:
        out.append(op)
        op_type = op.get("op", "")
        bias_key = bias_key_by_op.get(op_type)
        if not bias_key:
            continue
        if bias_key not in op.get("weights", {}):
            continue
        bias_weight_ref = op["weights"].get(bias_key)
        bias_weight_name = None
        if isinstance(bias_weight_ref, str):
            bias_weight_name = bias_weight_ref
        elif isinstance(bias_weight_ref, dict):
            bias_weight_name = bias_weight_ref.get("name")
        if isinstance(bias_weight_name, str) and is_zero_bias_tensor(bias_weight_name):
            skipped_zero += 1
            continue

        # Prefill GEMM kernels bind optional bias directly, but decode lowering
        # rewrites these projection ops to GEMV kernels that do not take bias.
        # Keep Gemma4/Qwen-style projection biases explicit in decode so the
        # later GEMV specialization cannot silently drop them.
        decode_needs_explicit_bias = mode == "decode" and op_type in {
            "q_proj",
            "q_gate_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "mlp_gate_up",
            "mlp_up",
            "mlp_down",
        }
        if kernel_supports_bias(op.get("kernel", "")) and not decode_needs_explicit_bias:
            continue

        out_dim, _ = compute_matmul_dims(op_type, config)
        bias_op = {
            "kernel": "bias_add",
            "op": "bias_add",
            "layer": op.get("layer", -1),
            "section": op.get("section", ""),
            "weights": {bias_key: op["weights"][bias_key]},
            "params": {},
            "bias_for": op_type,
            "_auto_inserted": True,
        }
        if out_dim is not None:
            bias_op["params"]["_output_dim"] = out_dim
        out.append(bias_op)
        inserted += 1

    if inserted:
        print(f"  Inserted {inserted} bias_add ops (mode={mode})")
    if skipped_zero:
        print(f"  Skipped {skipped_zero} zero bias_add ops (mode={mode})")

    return out


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: IR LOWER 1 (Stitch kernel maps)
# ═══════════════════════════════════════════════════════════════════════════

_DECODE_ATTENTION_OPS = {
    "attn",
    "attn_sliding",
    "attn_shared_kv",
    "attn_sliding_shared_kv",
}


def _require_resolved_decode_attention_kernel(op: Dict[str, Any]) -> str:
    """Validate the resolved decode provider without performing selection."""
    op_name = str(op.get("op", "") or "")
    if op_name not in _DECODE_ATTENTION_OPS:
        raise RuntimeError(
            f"HARD IR LOWERING FAULT: {op_name!r} is not a decode attention operation."
        )
    kernel_id = str(op.get("kernel", "") or "").strip()
    function = str(op.get("function", "") or "").strip()
    if not kernel_id or not function:
        raise RuntimeError(
            f"HARD IR LOWERING FAULT: {op_name} reached Lower 1 without an exact "
            "IR1 kernel and function. Resolve it from circuit requirements and kernel maps."
        )
    resolved = op.get("resolved_contract")
    if isinstance(resolved, dict):
        resolved_kernel = str(resolved.get("kernel_id", "") or "").strip()
        resolved_function = str(resolved.get("function", "") or "").strip()
        if not resolved_kernel or not resolved_function:
            raise RuntimeError(
                f"HARD IR LOWERING FAULT: {op_name} has an incomplete resolved contract."
            )
        if resolved_kernel != kernel_id:
            raise RuntimeError(
                f"HARD IR LOWERING FAULT: {op_name} IR1 kernel {kernel_id!r} "
                f"does not match resolved contract kernel {resolved_kernel!r}."
            )
        if resolved_function != function:
            raise RuntimeError(
                f"HARD IR LOWERING FAULT: {op_name} function {function!r} "
                f"does not match resolved contract function {resolved_function!r}."
            )
    return kernel_id

def generate_ir_lower_1(
    fused_ops: List[Dict],
    registry: Dict,
    manifest: Dict,
    mode: str
) -> List[Dict]:
    """
    IR Lower 1: Stitch kernel maps with IR1 ops.

    For each fused op:
      1. Load the kernel map (inputs, outputs, scratch)
      2. Map IR1 weights to kernel inputs
      3. Track activation flow between kernels

    This creates the buffer requirements that Memory Planner needs.

    Args:
        fused_ops: Fused IR1 ops from fusion pass
        registry: Kernel registry
        manifest: Model manifest
        mode: decode/prefill

    Returns:
        List of lowered ops with input/output/scratch specs
    """
    print(f"\n{'='*60}")
    print("IR LOWER 1 (Stitch kernel maps)")
    print(f"{'='*60}")

    config = manifest.get("config", {})
    logits_layout = _resolve_logits_layout(config, mode)
    template = manifest.get("template", {}) if isinstance(manifest.get("template"), dict) else {}
    template_contract = template.get("contract") if isinstance(template.get("contract"), dict) else {}
    attention_contract = template_contract.get("attention_contract") if isinstance(template_contract.get("attention_contract"), dict) else {}
    decode_cache_contract = attention_contract.get("decode_cache_contract") if isinstance(attention_contract.get("decode_cache_contract"), dict) else {}
    explicit_mla_decode_cache = str(decode_cache_contract.get("type", "") or "").strip().lower() == "mla"
    uses_kv_cache = bool(config.get("_template_uses_kv_cache", _template_uses_kv_cache(template, config)))
    has_logits = bool(config.get("_template_has_logits", _template_declares_logits(template, config)))

    # Build kernel map index by loading individual kernel map files
    # KERNEL_REGISTRY.json is only used for validation, not as source of truth
    kernel_maps_dir = V8_ROOT / "kernel_maps"
    kernel_map_index = {}
    for kernel in registry.get("kernels", []):
        kernel_id = kernel["id"]
        # Try to load individual kernel map file first
        kernel_file = kernel_maps_dir / f"{kernel_id}.json"
        if kernel_file.exists():
            with open(kernel_file, 'r') as f:
                kernel_map_index[kernel_id] = json.load(f)
        else:
            # Fallback to registry entry if no individual file
            kernel_map_index[kernel_id] = kernel
    # Use module-level WEIGHT_TO_KERNEL_INPUT for name mapping

    lowered_ops = []

    # TODO(contract): Carry an explicit semantic model contract through lowering.
    # Current lower stages focus on op wiring + tensor binding. For robust new-model
    # bring-up (e.g. Nanbeige/Llama variants), propagate and validate:
    #   - tokenizer_contract (SP/BPE class, BOS/EOS policy, special IDs, stop IDs)
    #   - attention_contract (rope type/theta/scaling, qk_norm, kv layout policy)
    #   - block_contract (norm kind, residual order, MLP formula, activation, bias)
    #   - logits_contract (final norm/head semantics, clamp/scale policy)
    #   - quant_contract (per-op expected quant family and kernel class)
    # This is additive first, then promoted to strict/fail-fast once model gates stay green.

    # Track activation buffers for dataflow
    # The output of one kernel becomes the input of the next
    current_activation = "input_tokens"  # Start with input token IDs

    for idx, ir_op in enumerate(fused_ops):
        kernel_id = ir_op["kernel"]
        op_name = ir_op["op"]
        layer = ir_op["layer"]
        section = ir_op["section"]
        ir_weights = ir_op.get("weights", {})

        # Get kernel map
        kernel_map = kernel_map_index.get(kernel_id)
        if not kernel_map:
            kernel_file = kernel_maps_dir / f"{kernel_id}.json"
            if kernel_file.exists():
                with open(kernel_file, 'r') as f:
                    kernel_map = json.load(f)
                kernel_map_index[kernel_id] = kernel_map
        if not kernel_map:
            print(f"  Warning: Kernel '{kernel_id}' not in registry, skipping")
            continue

        # Build lowered op - preserve ALL weights from IR1
        # Also preserve op_id and dataflow for memory planner
        lowered_op = {
            "idx": idx,
            "op_id": ir_op.get("op_id", idx),  # Preserve original op_id for memory planner
            "kernel": kernel_id,
            "op": op_name,
            "layer": layer,
            "section": section,
            "function": kernel_map.get("impl", {}).get("function", kernel_id),
            "weights": ir_weights,  # Preserve all IR1 weights
            "inputs": {},  # Activation inputs only
            "outputs": {},
            "scratch": [],
            "params": ir_op.get("params", {}),
            "bias_for": ir_op.get("bias_for"),
            "dataflow": ir_op.get("dataflow", {}),  # Preserve dataflow for memory planner
        }
        codegen_capability = _validated_kernel_codegen_capability(kernel_id, kernel_map)
        if codegen_capability is not None:
            lowered_op["resolved_codegen_capability"] = codegen_capability
        if ir_op.get("required_contract") is not None:
            lowered_op["required_contract"] = copy.deepcopy(ir_op["required_contract"])
        if ir_op.get("resolved_contract") is not None:
            lowered_op["resolved_contract"] = copy.deepcopy(ir_op["resolved_contract"])
            if lowered_op["resolved_contract"].get("kernel_id") != kernel_id:
                raise RuntimeError(
                    f"HARD CONTRACT FAULT: LoweredIR received kernel {kernel_id!r}, but GraphIR "
                    f"resolved {lowered_op['resolved_contract'].get('kernel_id')!r}. "
                    "Lowering must consume the resolved provider without reselection."
                )
        if ir_op.get("resolved_execution") is not None:
            lowered_op["resolved_execution"] = copy.deepcopy(ir_op["resolved_execution"])
            if lowered_op["resolved_execution"].get("kernel_id") != kernel_id:
                raise RuntimeError(
                    f"HARD CONTRACT FAULT: LoweredIR received kernel {kernel_id!r}, but GraphIR "
                    f"execution metadata names {lowered_op['resolved_execution'].get('kernel_id')!r}."
                )
        if ir_op.get("semantic_checkpoints") is not None:
            lowered_op["semantic_checkpoints"] = copy.deepcopy(ir_op["semantic_checkpoints"])
        if ir_op.get("_auto_inserted"):
            lowered_op["_auto_inserted"] = True

        # Special handling for residual_save/memcpy: compute _memcpy_bytes
        if op_name == "residual_save":
            embed_dim = manifest.get("config", {}).get("embed_dim", 896)
            seq_len = int(lowered_op["params"].get(
                "seq_len",
                1 if mode == "decode" else manifest.get("config", {}).get("context_length", 2048),
            ))
            lowered_op["params"]["_memcpy_bytes"] = embed_dim * seq_len * 4  # FP32 = 4 bytes
        elif op_name == "kv_cache_batch_copy":
            # NOTE: "batch" here means a token block in prefill, not multi-request batching.
            # A clearer name would be kv_cache_token_block_copy (keep op ID for v7 compatibility).
            num_kv_heads = int(lowered_op["params"].get("num_kv_heads", manifest.get("config", {}).get("num_kv_heads", 1)))
            head_dim = int(lowered_op["params"].get("head_dim", manifest.get("config", {}).get("head_dim", 1)))
            seq_len = int(lowered_op["params"].get("seq_len", manifest.get("config", {}).get("context_length", 1)))
            lowered_op["params"]["_kv_copy_bytes"] = num_kv_heads * head_dim * seq_len * 4  # FP32 bytes

        # Map kernel input activations from kernel map
        # New format has 4 sections:
        #   - 'inputs': input activations from previous layer
        #   - 'weights': static model parameters (already handled above)
        #   - 'activations': intermediate scratch tensors
        #   - 'outputs': output activations to next layer
        kernel_inputs = kernel_map.get("inputs", [])

        # Build set of weight names from kernel map to filter out weight inputs
        # (handles legacy format where weights were mixed into 'inputs')
        kernel_weight_names = set()
        for kw in (kernel_map.get("weights") or []):
            if isinstance(kw, dict):
                kernel_weight_names.add(kw["name"])
        # Also check IR1 weights mapped to kernel input names
        for wkey in ir_weights.keys():
            kernel_weight_names.add(wkey)
            mapped_name = WEIGHT_TO_KERNEL_INPUT.get(wkey)
            if mapped_name:
                kernel_weight_names.add(mapped_name)

        for kernel_input in kernel_inputs:
            if not isinstance(kernel_input, dict):
                continue
            input_name = kernel_input.get("name")
            if not input_name:
                continue
            input_dtype = kernel_input.get("dtype", "fp32")
            input_shape = kernel_input.get("shape", [])

            # Skip if this is actually a weight parameter
            if input_name in kernel_weight_names:
                continue

            # Activation input (from previous kernel)
            lowered_op["inputs"][input_name] = {
                "type": "activation",
                "source": current_activation,
                "dtype": input_dtype,
                "shape": input_shape,
            }

        # Map kernel outputs
        for kernel_output in (kernel_map.get("outputs") or []):
            if not isinstance(kernel_output, dict):
                continue
            output_name = kernel_output.get("name")
            if not output_name:
                continue
            output_dtype = kernel_output.get("dtype", "fp32")
            output_shape = kernel_output.get("shape", [])

            # Create output buffer name
            output_buffer = f"buf_{idx}_{output_name}"

            lowered_op["outputs"][output_name] = {
                "type": "activation",
                "buffer": output_buffer,
                "dtype": output_dtype,
                "shape": output_shape,
            }

            # Update current activation for next kernel
            current_activation = output_buffer

        # Map scratch buffers
        for scratch in kernel_map.get("scratch", []):
            lowered_op["scratch"].append({
                "name": scratch.get("name", f"scratch_{idx}"),
                "size": scratch.get("size", "dynamic"),
                "dtype": scratch.get("dtype", "fp32"),
            })

        lowered_ops.append(lowered_op)

    # ═══════════════════════════════════════════════════════════════════════════
    # AUTOMATIC KV CACHE INSERTION (decode only)
    # ═══════════════════════════════════════════════════════════════════════════
    # Insert kv_cache_store ops after the last K/V-transforming op to store K,V
    # for subsequent decode. RoPE attention stores after rope_qk; no-RoPE
    # attention (for example Nemotron-H) stores after v_proj.
    # For decode, also update attention to use the decode kernel with KV cache.
    print(f"\n  [{mode.capitalize()} mode] Inserting KV cache operations...")
    final_ops = []
    kv_store_count = 0
    decode_attention_count = 0

    force_decode_attn_regular = str(os.environ.get("CK_V7_DECODE_ATTN_REGULAR", "")).strip().lower() in ("1", "true", "yes", "on")
    decode_kv_cache_dtype = str(config.get("decode_kv_cache_dtype", "fp32") or "fp32").strip().lower()
    decode_uses_fp16_kv = decode_kv_cache_dtype in {"fp16", "f16"}
    layer_kv_source_cfg = config.get("layer_kv_source") if isinstance(config.get("layer_kv_source"), list) else []
    decode_attention_layers = {
        int(op.get("layer", 0))
        for op in lowered_ops
        if str(op.get("op", "")) in {"attn", "attn_sliding", "attn_shared_kv", "attn_sliding_shared_kv"}
    }
    decode_rope_layers = {
        int(op.get("layer", 0))
        for op in lowered_ops
        if str(op.get("op", "")) in {"rope_qk", "rope_q", "mrope_qk"}
    }
    decode_mla_layers = {
        int(op.get("layer", 0))
        for op in lowered_ops
        if str(op.get("op", "")) == "mla_attention"
    }

    def _kv_read_layer_for(layer: int) -> int:
        try:
            if 0 <= int(layer) < len(layer_kv_source_cfg):
                return int(layer_kv_source_cfg[int(layer)])
        except (TypeError, ValueError):
            pass
        return int(layer)

    def _make_decode_kv_store_op(anchor_op: Dict[str, Any]) -> Dict[str, Any]:
        layer = int(anchor_op["layer"])
        return {
            "idx": len(final_ops),  # Will be renumbered
            "kernel": "kv_cache_store_f16" if decode_uses_fp16_kv else "kv_cache_store",
            "op": "kv_cache_store",
            "layer": layer,
            "section": anchor_op["section"],
            "function": "kv_cache_store_f16" if decode_uses_fp16_kv else "kv_cache_store",
            "weights": {},
            "inputs": {
                "k": {"type": "scratch", "source": "k_scratch"},
                "v": {"type": "scratch", "source": "v_scratch"},
            },
            "outputs": {
                "kv_cache_k": {"type": "kv_cache", "buffer": f"kv_cache_k_L{layer}"},
                "kv_cache_v": {"type": "kv_cache", "buffer": f"kv_cache_v_L{layer}"},
            },
            "params": copy.deepcopy(anchor_op.get("params", {})),
            "scratch": [],
            "_auto_inserted": True,
        }

    def _make_decode_shared_q_kv_store_op(anchor_op: Dict[str, Any]) -> Dict[str, Any]:
        layer = int(anchor_op["layer"])
        return {
            "idx": len(final_ops),  # Will be renumbered
            "kernel": "kv_cache_store_shared_q",
            "op": "kv_cache_store_shared_q",
            "layer": layer,
            "section": anchor_op["section"],
            "function": "kv_cache_store_shared_q",
            "weights": {},
            "inputs": {
                "q": {"type": "scratch", "source": "q_scratch"},
            },
            "outputs": {
                "kv_cache_k": {"type": "kv_cache", "buffer": f"kv_cache_k_L{layer}"},
                "kv_cache_v": {"type": "kv_cache", "buffer": f"kv_cache_v_L{layer}"},
            },
            "params": copy.deepcopy(anchor_op.get("params", {})),
            "scratch": [],
            "_auto_inserted": True,
        }

    def _make_mla_kv_store_op(anchor_op: Dict[str, Any], *, batch: bool) -> Dict[str, Any]:
        layer = int(anchor_op["layer"])
        function = "deepseek_mla_kv_cache_batch_store_f32" if batch else "deepseek_mla_kv_cache_store_f32"
        return {
            "idx": len(final_ops),  # Will be renumbered
            "kernel": function,
            "op": "mla_kv_cache_batch_store" if batch else "mla_kv_cache_store",
            "layer": layer,
            "section": anchor_op["section"],
            "function": function,
            "weights": {},
            "inputs": {
                "k": {"type": "scratch", "source": "k_scratch"},
                "v": {"type": "scratch", "source": "v_scratch"},
            },
            "outputs": {
                "kv_cache_k": {"type": "kv_cache", "buffer": f"kv_cache_k_L{layer}"},
                "kv_cache_v": {"type": "kv_cache", "buffer": f"kv_cache_v_L{layer}"},
            },
            "params": copy.deepcopy(anchor_op.get("params", {})),
            "scratch": [],
            "_auto_inserted": True,
        }

    mla_prefill_layers = {
        int(x.get("layer", 0))
        for x in lowered_ops
        if str(x.get("op", "")) in {"partial_rope_concat", "mla_attention", "kv_lora_decompress"}
    }

    for i, op in enumerate(lowered_ops):
        final_ops.append(op)

        if mode == "decode" and uses_kv_cache:
            layer = int(op.get("layer", 0))
            op_name = str(op.get("op", ""))
            should_store_after_rope = op_name in {"rope_qk", "mrope_qk"}
            should_store_after_q_rope = (
                op_name == "rope_q"
                and layer in decode_attention_layers
            )
            should_store_after_v = (
                op_name == "v_proj"
                and layer in decode_attention_layers
                and layer not in decode_rope_layers
            )
            if should_store_after_rope or should_store_after_v:
                final_ops.append(_make_decode_kv_store_op(op))
                kv_store_count += 1
            elif should_store_after_q_rope:
                final_ops.append(_make_decode_shared_q_kv_store_op(op))
                kv_store_count += 1
            elif explicit_mla_decode_cache and op_name == "partial_rope_concat" and layer in decode_mla_layers:
                final_ops.append(_make_mla_kv_store_op(op, batch=False))
                kv_store_count += 1

            # For decode mode, update attention ops to use decode kernel
            if explicit_mla_decode_cache and op["op"] == "mla_attention" and "mla_attention" in op["kernel"]:
                decode_kernel = "deepseek_mla_attention_decode_f32"
                op["kernel"] = decode_kernel
                op["function"] = decode_kernel
                kv_read_layer = _kv_read_layer_for(int(op.get("layer", 0)))
                op["_kv_cache_read_layer"] = kv_read_layer
                op.setdefault("inputs", {})
                op["inputs"]["k_cache"] = {"type": "kv_cache", "source": f"kv_cache_k_L{kv_read_layer}"}
                op["inputs"]["v_cache"] = {"type": "kv_cache", "source": f"kv_cache_v_L{kv_read_layer}"}
                op["inputs"].pop("k", None)
                op["inputs"].pop("v", None)
            elif op["op"] in _DECODE_ATTENTION_OPS:
                decode_kernel = _require_resolved_decode_attention_kernel(op)
                decode_attention_count += 1
                if force_decode_attn_regular:
                    raise RuntimeError(
                        "HARD CONTRACT FAULT: CK_V7_DECODE_ATTN_REGULAR cannot override "
                        f"authoritative IR1 kernel {decode_kernel!r}. Change the circuit or "
                        "kernel-map contract instead."
                    )
                kv_read_layer = _kv_read_layer_for(int(op.get("layer", 0)))
                op["_kv_cache_read_layer"] = kv_read_layer
                # Update inputs to use KV cache instead of scratch
                op.setdefault("inputs", {})
                op["inputs"]["k_cache"] = {"type": "kv_cache", "source": f"kv_cache_k_L{kv_read_layer}"}
                op["inputs"]["v_cache"] = {"type": "kv_cache", "source": f"kv_cache_v_L{kv_read_layer}"}
                # Remove scratch K/V references if present
                op["inputs"].pop("k", None)
                op["inputs"].pop("v", None)

        elif mode == "prefill":
            # Prefill layout bridges are a graph contract, not a decoder/KV-cache
            # special case. Keep the lowerer architecture-agnostic: if one op
            # emits token-major activations but the next declared kernel expects
            # head-major (or vice versa), insert an explicit bridge op here.
            #
            # Standard q/k/v GEMM projections write token-major [T, H*D] while
            # flash attention consumes head-major [H, T, D]. Packed split paths
            # that already emit head-major simply never trigger these bridges.
            if explicit_mla_decode_cache and op.get("op") == "partial_rope_concat" and int(op.get("layer", 0)) in mla_prefill_layers:
                final_ops.append(_make_mla_kv_store_op(op, batch=True))

            if op["op"] in ("q_proj", "split_q_gate"):
                layer = op["layer"]
                if int(layer) in mla_prefill_layers:
                    continue
                transpose_q_op = {
                    "idx": len(final_ops),
                    "kernel": "transpose_qkv_to_head_major",
                    "op": "transpose_qkv_to_head_major",
                    "layer": layer,
                    "section": op["section"],
                    "function": "transpose_inplace",
                    "weights": {},
                    "inputs": {"buf": {"type": "scratch", "source": "q_scratch"}},
                    "outputs": {"buf": {"type": "scratch", "buffer": "q_scratch"}},
                    "scratch": [],
                    "_auto_inserted": True,
                    "_qkv_type": "q",
                }
                final_ops.append(transpose_q_op)

            if op["op"] == "k_proj":
                layer = op["layer"]
                transpose_k_op = {
                    "idx": len(final_ops),
                    "kernel": "transpose_kv_to_head_major",
                    "op": "transpose_kv_to_head_major",
                    "layer": layer,
                    "section": op["section"],
                    "function": "transpose_inplace",
                    "weights": {},
                    "inputs": {"buf": {"type": "scratch", "source": "k_scratch"}},
                    "outputs": {"buf": {"type": "scratch", "buffer": "k_scratch"}},
                    "scratch": [],
                    "_auto_inserted": True,
                    "_is_k": True,
                }
                final_ops.append(transpose_k_op)

            if op["op"] == "v_proj":
                layer = op["layer"]
                transpose_v_op = {
                    "idx": len(final_ops),
                    "kernel": "transpose_kv_to_head_major",
                    "op": "transpose_kv_to_head_major",
                    "layer": layer,
                    "section": op["section"],
                    "function": "transpose_inplace",
                    "weights": {},
                    "inputs": {"buf": {"type": "scratch", "source": "v_scratch"}},
                    "outputs": {"buf": {"type": "scratch", "buffer": "v_scratch"}},
                    "scratch": [],
                    "_auto_inserted": True,
                    "_is_k": False,
                }
                final_ops.append(transpose_v_op)

            # Flash attention writes head-major [H, T, D], but the unfused
            # projection/residual path consumes token-major [T, H*D]. Emit the
            # bridge for all prefill graphs, regardless of whether a KV cache
            # also exists.
            if op["op"] in ("attn", "attn_sliding", "attn_shared_kv", "attn_sliding_shared_kv"):
                layer = op["layer"]
                required_contract = op.get("required_contract") if isinstance(op.get("required_contract"), dict) else {}
                segmented_append = required_contract.get("execution.prefill_batching") == "segmented_append"
                if segmented_append and not uses_kv_cache:
                    raise RuntimeError(
                        "HARD CONTRACT FAULT: segmented_append attention requires a persistent KV cache. "
                        "Fix the circuit cache contract; do not fall back to full-sequence prefill."
                    )
                kv_batch_copy_op = None
                if uses_kv_cache:
                    shared_q_prefill = op["op"] in ("attn_shared_kv", "attn_sliding_shared_kv")
                    copy_src = "q_scratch" if shared_q_prefill else "k_scratch"
                    kv_batch_copy_op = {
                        "idx": len(final_ops),
                        "kernel": "kv_cache_batch_copy",
                        "op": "kv_cache_batch_copy",
                        "layer": layer,
                        "section": op["section"],
                        "function": "kv_cache_batch_copy",
                        "weights": {},
                        "inputs": {
                            "k_src": {"type": "scratch", "source": copy_src},
                            "v_src": {"type": "scratch", "source": "q_scratch" if shared_q_prefill else "v_scratch"},
                        },
                        "outputs": {
                            "k_dst": {"type": "kv_cache", "buffer": f"kv_cache_k_L{layer}"},
                            "v_dst": {"type": "kv_cache", "buffer": f"kv_cache_v_L{layer}"},
                        },
                        "scratch": [],
                        "params": {
                            "num_kv_heads": int(config.get("num_kv_heads", 1)),
                            "head_dim": int(config.get("head_dim", 1)),
                            "seq_len": int(config.get("context_length", config.get("context_len", 1))),
                        },
                        "_auto_inserted": True,
                        "_cache_append": segmented_append,
                    }
                if segmented_append:
                    # The append provider reads current K/V through the persistent
                    # cache, so store this segment immediately before attention.
                    final_ops.insert(len(final_ops) - 1, kv_batch_copy_op)
                    kv_store_count += 1
                transpose_attn_out_op = {
                    "idx": len(final_ops),
                    "kernel": "transpose_attn_out_to_token_major",
                    "op": "transpose_attn_out_to_token_major",
                    "layer": layer,
                    "section": op["section"],
                    "function": "transpose_inplace",
                    "weights": {},
                    "inputs": {"buf": {"type": "scratch", "source": "attn_scratch"}},
                    "outputs": {"buf": {"type": "scratch", "buffer": "attn_scratch"}},
                    "scratch": [],
                    "_auto_inserted": True,
                }
                final_ops.append(transpose_attn_out_op)
                if uses_kv_cache and not segmented_append:
                    # TODO(contract): validate this op against runtime_invariants contract:
                    # _kv_copy_bytes must exist and match
                    # (num_kv_heads * head_dim * seq_len * sizeof(fp32)).
                    final_ops.append(kv_batch_copy_op)
                    kv_store_count += 1

    # ═══════════════════════════════════════════════════════════════════════════
    # AUTOMATIC LOGITS COPY FOR PREFILL
    # ═══════════════════════════════════════════════════════════════════════════
    # In prefill mode, logits are computed for ALL tokens as [num_tokens, vocab_size].
    # But ck_model_forward() expects logits at position 0 (for the LAST token).
    # Insert a copy_last_logits op to copy logits[(n-1)*V : n*V] to logits[0:V].
    copy_last_logits_inserted = False
    if mode == "prefill" and has_logits and logits_layout != "last":
        # Insert copy_last_logits at the very end
        copy_last_logits_op = {
            "idx": len(final_ops),
            "kernel": "copy_last_logits",
            "op": "copy_last_logits",
            "layer": -1,
            "section": "footer",
            "function": "memmove",  # Use memmove for safety (overlapping memory)
            "weights": {},
            "inputs": {
                "src": {"type": "activation", "source": "logits", "offset": "(num_tokens - 1) * vocab_size"},
            },
            "outputs": {
                "dst": {"type": "activation", "buffer": "logits"},
            },
            "scratch": [],
            "_auto_inserted": True,
            "params": {
                "_copy_size": "vocab_size * sizeof(float)",
            },
        }
        final_ops.append(copy_last_logits_op)
        copy_last_logits_inserted = True
        print(f"  Inserted copy_last_logits op for prefill mode")

    # Renumber ops and normalize derived params for auto-inserted kernels.
    # TODO(contract): centralize required-arg derivation/validation for all ops
    # (not only kv_cache_batch_copy/residual_save) and fail in lower stage if any
    # required call arg is missing/invalid.
    for i, op in enumerate(final_ops):
        op["idx"] = i
        if op.get("op") == "kv_cache_batch_copy":
            params = op.setdefault("params", {})
            num_kv_heads = int(params.get("num_kv_heads", config.get("num_kv_heads", 1)))
            head_dim = int(params.get("head_dim", config.get("head_dim", 1)))
            seq_len = int(params.get("seq_len", config.get("context_length", config.get("context_len", 1))))
            params["num_kv_heads"] = num_kv_heads
            params["head_dim"] = head_dim
            params["seq_len"] = seq_len
            params["_kv_copy_bytes"] = num_kv_heads * head_dim * seq_len * 4  # FP32 bytes

    lowered_ops = final_ops
    print(f"  Inserted {kv_store_count} kv_cache_store operations")
    if not uses_kv_cache:
        print("  KV cache insertion skipped: template declares no persistent KV cache")
    if mode == "decode" and uses_kv_cache:
        print(
            f"  Validated {decode_attention_count} authoritative IR1 decode "
            "attention kernels and wired KV-cache inputs"
        )

    # Summary
    total_weight_refs = sum(len(op.get("weights", {})) for op in lowered_ops)
    total_activations = sum(len(op.get("inputs", {})) for op in lowered_ops)

    print(f"\n✓ IR Lower 1 complete:")
    print(f"  Lowered ops: {len(lowered_ops)}")
    print(f"  Weight references: {total_weight_refs}")
    print(f"  Activation inputs: {total_activations}")

    return lowered_ops


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: MEMORY PLANNER (with aggressive validation)
# ═══════════════════════════════════════════════════════════════════════════

# Weight name patterns for matching IR1 ops to manifest entries
# Maps: kernel weight ref → possible manifest entry patterns
WEIGHT_PATTERNS = {
    # QKV projection weights and biases
    "wq": ["layer.{L}.wq", "layers.{L}.attention.wq", "layer.{L}.attn_q_gate", "layer.{L}.attn_q", "layer.{L}.mla_q_proj"],
    "wk": ["layer.{L}.wk", "layers.{L}.attention.wk", "layer.{L}.attn_k"],
    "wv": ["layer.{L}.wv", "layers.{L}.attention.wv", "layer.{L}.attn_v"],
    "bq": ["layer.{L}.bq", "layers.{L}.attention.bq"],
    "bk": ["layer.{L}.bk", "layers.{L}.attention.bk"],
    "bv": ["layer.{L}.bv", "layers.{L}.attention.bv"],

    # QK norm weights (per-head RMSNorm gamma for Q and K)
    "q_norm": ["layer.{L}.q_norm", "layers.{L}.attention.q_norm", "layer.{L}.attn_q_norm"],
    "rope_freqs": ["rope_freqs"],
    "k_norm": ["layer.{L}.k_norm", "layers.{L}.attention.k_norm", "layer.{L}.attn_k_norm"],

    # Recurrent hybrid block weights
    "attn_qkv": ["layer.{L}.attn_qkv"],
    "attn_gate": ["layer.{L}.attn_gate"],
    "ssm_alpha": ["layer.{L}.ssm_alpha"],
    "ssm_beta": ["layer.{L}.ssm_beta"],
    "ssm_conv1d": ["layer.{L}.ssm_conv1d"],
    "ssm_dt_bias": ["layer.{L}.ssm_dt_bias"],
    "ssm_a": ["layer.{L}.ssm_a"],
    "ssm_norm": ["layer.{L}.ssm_norm"],
    "ssm_out": ["layer.{L}.ssm_out"],
    "mamba_in_proj": ["layer.{L}.mamba_in_proj"],
    "mamba_conv1d": ["layer.{L}.mamba_conv1d"],
    "mamba_conv1d_bias": ["layer.{L}.mamba_conv1d_bias"],
    "mamba_dt_bias": ["layer.{L}.mamba_dt_bias"],
    "mamba_a": ["layer.{L}.mamba_a"],
    "mamba_d": ["layer.{L}.mamba_d"],
    "mamba_norm": ["layer.{L}.mamba_norm"],
    "mamba_out_proj": ["layer.{L}.mamba_out_proj"],
    "mla_kv_a_proj": ["layer.{L}.mla_kv_a_proj"],
    "mla_kv_a_norm": ["layer.{L}.mla_kv_a_norm"],
    "mla_kv_b_proj": ["layer.{L}.mla_kv_b_proj"],
    "moe_router": ["layer.{L}.moe_router"],
    "moe_router_bias": ["layer.{L}.moe_router_bias"],
    "moe_expert_gate": ["layer.{L}.moe_expert_gate", "layer.{L}.moe_expert.{E}.gate"],
    "moe_expert_up": ["layer.{L}.moe_expert_up", "layer.{L}.moe_expert.{E}.up"],
    "moe_expert_down": ["layer.{L}.moe_expert_down", "layer.{L}.moe_expert.{E}.down"],
    "moe_shared_gate": ["layer.{L}.moe_shared_gate"],
    "moe_shared_up": ["layer.{L}.moe_shared_up"],
    "moe_shared_down": ["layer.{L}.moe_shared_down"],

    # Output projection
    "wo": ["layer.{L}.wo", "layers.{L}.attention.wo", "layer.{L}.attn_output", "layer.{L}.attn_o", "layer.{L}.mla_out_proj"],
    "bo": ["layer.{L}.bo", "layers.{L}.attention.bo"],

    # MLP weights and biases
    "w1": ["layer.{L}.w1", "layers.{L}.feed_forward.w1", "layer.{L}.ffn_gate", "layer.{L}.mlp_gate"],
    "w2": ["layer.{L}.w2", "layers.{L}.feed_forward.w2", "layer.{L}.ffn_down", "layer.{L}.mlp_down"],
    "w3": ["layer.{L}.w3", "layers.{L}.feed_forward.w3", "layer.{L}.ffn_up", "layer.{L}.mlp_up"],
    "b1": ["layer.{L}.b1", "layers.{L}.feed_forward.b1", "v.blk.{L}.ffn_up.bias"],
    "b2": ["layer.{L}.b2", "layers.{L}.feed_forward.b2", "v.blk.{L}.ffn_down.bias"],

    # Layer norms
    "ln1_gamma": ["layer.{L}.ln1_gamma", "layers.{L}.attention_norm.weight", "layer.{L}.attn_norm", "layer.{L}.block_norm"],
    "ln2_gamma": ["layer.{L}.ln2_gamma", "layers.{L}.ffn_norm.weight", "layer.{L}.post_attention_norm"],
    "ln1_beta": ["layer.{L}.ln1_beta", "layers.{L}.attention_norm.bias", "v.blk.{L}.ln1.bias"],
    "ln2_beta": ["layer.{L}.ln2_beta", "layers.{L}.ffn_norm.bias", "v.blk.{L}.ln2.bias"],
    "post_attention_norm": ["layer.{L}.post_attention_norm", "layers.{L}.post_attention_norm.weight"],
    "post_ffn_norm": ["layer.{L}.post_ffn_norm", "layer.{L}.post_ffw_norm", "layers.{L}.post_ffn_norm.weight"],
    "per_layer_token_emb": ["per_layer_token_emb"],
    "per_layer_model_proj": ["per_layer_model_proj"],
    "per_layer_proj_norm": ["per_layer_proj_norm"],
    "per_layer_inp_gate": ["layer.{L}.per_layer_inp_gate"],
    "per_layer_proj": ["layer.{L}.per_layer_proj"],
    "per_layer_post_norm": ["layer.{L}.per_layer_post_norm"],
    "layer_output_scale": ["layer.{L}.layer_output_scale"],
    "patch_emb": [
        "patch_emb.weight",
        "patch_embeddings.weight",
        "vision_model.embeddings.patch_embedding.weight",
        "v.patch_embd.weight",
    ],
    "patch_emb_aux": [
        "patch_embeddings.weight.1",
        "vision_model.embeddings.patch_embedding.weight.1",
        "v.patch_embd.weight.1",
    ],
    "patch_bias": [
        "patch_bias",
        "patch_emb.bias",
        "patch_embeddings.bias",
        "vision_model.embeddings.patch_embedding.bias",
        "v.patch_embd.bias",
    ],

    # Header weights
    "token_emb": ["token_emb", "token_embd.weight", "embed_tokens.weight"],
    "pos_emb": ["pos_emb", "pos_embd.weight", "position_embedding", "v.position_embd.weight"],

    # Vocab/tokenizer data (not model weights, but need to track)
    "vocab_offsets": ["vocab_offsets"],
    "vocab_strings": ["vocab_strings"],
    "vocab_merges": ["vocab_merges"],

    # Footer weights
    "lm_head": ["lm_head.weight", "output.weight"],
    "final_ln_weight": ["final_ln_weight", "norm.weight", "v.post_ln.weight"],
    "final_ln_bias": ["final_ln_bias", "norm.bias", "v.post_ln.bias"],
    "output_weight": ["output.weight", "lm_head.weight"],
    "bqkv": ["layer.{L}.attn_qkv.bias", "v.blk.{L}.attn_qkv.bias"],
    "mm0_w": ["mm.0.weight"],
    "mm0_b": ["mm.0.bias"],
    "mm1_w": ["mm.2.weight", "mm.input_projection.weight"],
    "mm1_b": ["mm.2.bias"],
    "assistant_pre_projection": ["assistant.pre_projection"],
    "assistant_post_projection": ["assistant.post_projection"],
    "attn_qkv": ["layer.{L}.attn_qkv", "layer.{L}.attn_qkv.weight", "v.blk.{L}.attn_qkv.weight"],
    "ln1_gamma": ["layer.{L}.ln1_gamma", "layers.{L}.attention_norm.weight", "layer.{L}.attn_norm", "layer.{L}.block_norm", "v.blk.{L}.ln1.weight"],
    "ln2_gamma": ["layer.{L}.ln2_gamma", "layers.{L}.ffn_norm.weight", "layer.{L}.post_attention_norm", "v.blk.{L}.ln2.weight"],
    "ln1_beta": ["layer.{L}.ln1_beta", "layers.{L}.attention_norm.bias", "v.blk.{L}.ln1.bias"],
    "ln2_beta": ["layer.{L}.ln2_beta", "layers.{L}.ffn_norm.bias", "v.blk.{L}.ln2.bias"],
    "wo": ["layer.{L}.wo", "layers.{L}.attention.wo", "layer.{L}.attn_output", "layer.{L}.attn_o", "layer.{L}.mla_out_proj", "v.blk.{L}.attn_out.weight"],
    "bo": ["layer.{L}.bo", "layers.{L}.attention.bo", "v.blk.{L}.attn_out.bias"],
    "w1": ["layer.{L}.w1", "layers.{L}.feed_forward.w1", "layer.{L}.ffn_gate", "layer.{L}.mlp_gate", "v.blk.{L}.ffn_gate.weight"],
    "w2": ["layer.{L}.w2", "layers.{L}.feed_forward.w2", "layer.{L}.ffn_down", "layer.{L}.mlp_down", "v.blk.{L}.ffn_down.weight"],
    "w3": ["layer.{L}.w3", "layers.{L}.feed_forward.w3", "layer.{L}.ffn_up", "layer.{L}.mlp_up", "v.blk.{L}.ffn_up.weight"],
    "branch_norm_gamma": ["v.deepstack.{L}.norm.weight"],
    "branch_norm_beta": ["v.deepstack.{L}.norm.bias"],
    "branch_fc1_w": ["v.deepstack.{L}.fc1.weight"],
    "branch_fc1_b": ["v.deepstack.{L}.fc1.bias"],
    "branch_fc2_w": ["v.deepstack.{L}.fc2.weight"],
    "branch_fc2_b": ["v.deepstack.{L}.fc2.bias"],
}

# Template op → weight refs it uses
# This tells us which weights each template op needs
TEMPLATE_OP_WEIGHTS = {
    # Header (tokenizer is metadata, not model weights)
    "tokenizer": [],  # Deprecated, use bpe_tokenizer
    "bpe_tokenizer": [],  # BPE tokenizer data handled separately (not model weights)
    "wordpiece_tokenizer": [],  # WordPiece tokenizer data handled separately
    "patch_embeddings": [],  # Vision model patches handled separately
    "patchify": [],
    "patch_proj": ["patch_emb", "patch_bias"],
    "patch_proj_aux": ["patch_emb_aux"],
    "patch_bias_add": ["patch_bias"],
    "position_embeddings": ["pos_emb"],
    "vision_position_ids": [],
    "position_ids_2d": [],
    "dense_embedding_lookup": ["token_emb"],  # Token embeddings only (pos_emb for non-RoPE)
    "assistant_pre_projection": ["assistant_pre_projection"],

    # Attention block (body + footer)
    # Body: uses ln1_gamma, ln2_gamma (per-layer)
    # Footer: uses final_ln_weight, final_ln_bias (once)
    "rmsnorm": ["ln1_gamma", "ln2_gamma", "final_ln_weight", "final_ln_bias"],
    "layernorm": ["ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta", "final_ln_weight", "final_ln_bias"],
    "attn_norm": ["ln1_gamma"],
    "block_rmsnorm": ["ln1_gamma", "post_attention_norm"],
    "post_attention_norm": ["post_attention_norm"],
    "ffn_norm": ["ln2_gamma"],
    "post_ffn_norm": ["post_ffn_norm"],
    "gemma4_per_layer_prepare": [
        "per_layer_token_emb",
        "per_layer_model_proj",
        "per_layer_proj_norm",
    ],
    "gemma4_per_layer_embed": [
        "per_layer_inp_gate",
        "per_layer_proj",
        "per_layer_post_norm",
        "layer_output_scale",
    ],
    "assistant_layer_scale": ["layer_output_scale"],
    "final_logit_softcap": [],
    "v_norm": [],
    "final_rmsnorm": ["final_ln_weight", "final_ln_bias"],
    "qkv_proj": ["wq", "wk", "wv", "bq", "bk", "bv"],  # QKV + optional biases (for fused kernel)
    "qkv_packed_proj": ["attn_qkv", "bqkv"],
    "q_proj": ["wq", "bq", "wq_input_min", "wq_input_max", "wq_output_min", "wq_output_max"],  # Q projection only (when split)
    "q_gate_proj": ["wq", "bq"],  # Joint Q + gate projection
    "k_proj": ["wk", "bk", "wk_input_min", "wk_input_max", "wk_output_min", "wk_output_max"],  # K projection only (when split)
    "v_proj": ["wv", "bv", "wv_input_min", "wv_input_max", "wv_output_min", "wv_output_max"],  # V projection only (when split)
    "split_q_gate": [],
    "recurrent_packed_proj": ["attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta"],
    "recurrent_qkv_proj": ["attn_qkv"],
    "recurrent_gate_proj": ["attn_gate"],
    "recurrent_alpha_proj": ["ssm_alpha"],
    "recurrent_beta_proj": ["ssm_beta"],
    "recurrent_split_qkv": [],
    "split_qkv_packed": [],
    "recurrent_dt_gate": ["ssm_dt_bias", "ssm_a"],
    "recurrent_conv_state_update": [],
    "recurrent_ssm_conv": ["ssm_conv1d"],
    "recurrent_silu": [],
    "recurrent_split_conv_qkv": [],
    "recurrent_qk_l2_norm": [],
    "recurrent_core": [],
    "recurrent_norm_gate": ["ssm_norm"],
    "recurrent_out_proj": ["ssm_out"],
    "mamba_in_proj": ["mamba_in_proj"],
    "mamba_in_proj_split": [],
    "mamba_dt_softplus": ["mamba_dt_bias"],
    "mamba_conv1d_silu": ["mamba_conv1d", "mamba_conv1d_bias"],
    "mamba_selective_scan": ["mamba_a", "mamba_d"],
    "mamba_rmsnorm_gate": ["mamba_norm"],
    "mamba_out_proj": ["mamba_out_proj"],
    "moe_router": ["moe_router"],
    "group_limited_topk_router": ["moe_router_bias"],
    "moe_relu2_expert_mlp": ["moe_expert_up", "moe_expert_down"],
    "shared_relu2_expert_mlp": ["moe_shared_up", "moe_shared_down"],
    "moe_swiglu_expert_mlp": ["moe_expert_gate", "moe_expert_up", "moe_expert_down"],
    "shared_swiglu_expert_mlp": ["moe_shared_gate", "moe_shared_up", "moe_shared_down"],
    "kv_a_proj": ["mla_kv_a_proj"],
    "kv_a_layernorm": ["mla_kv_a_norm"],
    "kv_lora_decompress": ["mla_kv_b_proj"],
    "partial_rope_concat": [],
    "mla_attention": [],
    "qk_norm": ["q_norm", "k_norm"],  # Per-head RMSNorm gamma weights for Q and K
    "q_norm": ["q_norm"],  # Per-head RMSNorm gamma weights for Q-only shared-KV attention
    "rope_qk": [],  # No model weights (uses precomputed tables)
    "rope_q": [],  # No model weights (uses direct RoPE params/frequency factors)
    "mrope_qk": [],  # No model weights (runtime positions + RoPE params)
    "attn": [],  # No model weights
    "attn_sliding": [],  # No model weights (kernel op handles windowing)
    "attn_shared_kv": [],  # No model weights; q_scratch is used as Q/K/V
    "attn_sliding_shared_kv": [],  # No model weights; q_scratch is used as Q/K/V with SWA
    "mla_kv_cache_batch_store": [],
    "mla_kv_cache_store": [],
    "kv_cache_store_shared_q": [],  # No model weights; writes q_scratch to K/V cache
    "attn_gate_sigmoid_mul": [],  # No model weights
    "out_proj": ["wo", "bo", "wo_input_min", "wo_input_max", "wo_output_min", "wo_output_max"],  # Output projection + optional bias
    "residual_add": [],  # No model weights
    "add_stream": [],

    # MLP block
    "mlp_gate_up": ["w1", "w3", "b1", "w1_input_min", "w1_input_max", "w1_output_min", "w1_output_max", "w3_input_min", "w3_input_max", "w3_output_min", "w3_output_max"],  # Gate + up projection
    "mlp_up": ["w3", "b1"],  # Plain up projection
    "silu_mul": [],  # No model weights
    "geglu": [],  # No model weights
    "gelu": [],  # No model weights
    "mlp_down": ["w2", "b2", "w2_input_min", "w2_input_max", "w2_output_min", "w2_output_max"],  # Down projection
    "spatial_merge": [],
    "projector_prep": [],
    "branch_spatial_merge": [],
    "branch_layernorm": ["branch_norm_gamma", "branch_norm_beta"],
    "projector_fc1": ["mm0_w", "mm0_b"],
    "projector_gelu": [],
    "projector_fc2": ["mm1_w", "mm1_b", "mm1_w_input_min", "mm1_w_input_max", "mm1_w_output_min", "mm1_w_output_max"],
    "branch_fc1": ["branch_fc1_w", "branch_fc1_b"],
    "branch_gelu": [],
    "branch_fc2": ["branch_fc2_w", "branch_fc2_b"],
    "branch_concat": [],

    # Footer
    "weight_tying": [],  # Metadata only
    "assistant_post_projection": ["assistant_post_projection"],
    # logits source is resolved at runtime contract time:
    # - tied -> token_emb
    # - untied -> lm_head/output.weight
    "logits": ["lm_head"],
}


def generate_memory_layout(
    ir_lower_1_ops: List[Dict],
    manifest: Dict,
    registry: Dict,
    mode: str,
    context_len: int = None
) -> Dict:
    """
    Generate memory layout with AGGRESSIVE VALIDATION.

    Args:
        ir_lower_1_ops: List of IR Lower 1 ops (each is {kernel, op, section, layer,
                        inputs, outputs, scratch}). Inputs have type='weight' or
                        type='activation'.

    This function is a VALIDATION GATE:
    - HARD FAULT if any manifest weight is unused
    - HARD FAULT if any required weight is missing

    Steps:
    1. Build weight index from manifest entries (actual sizes)
    2. Extract weight usage from IR Lower 1 inputs (type='weight')
    3. Validate 100% weight coverage
    4. Plan activation buffers from IR Lower 1 outputs
    5. Return complete layout

    Returns:
        Layout dict with memory allocation plan and validation status

    Raises:
        RuntimeError: If validation fails (unused or missing weights)
    """
    print(f"\n{'='*60}")
    print("MEMORY PLANNER (with validation)")
    print(f"{'='*60}")

    config = manifest.get("config", {})
    entries = manifest.get("entries", [])
    template = manifest.get("template", {})
    num_layers = config.get("num_layers", 24)

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Build weight index from manifest entries
    # ═══════════════════════════════════════════════════════════

    if not entries:
        raise RuntimeError(
            "HARD FAULT: Manifest has no 'entries' field!\n"
            "  The manifest must contain weight tensor entries.\n"
            "  Re-run converter with --bump-version=5"
        )

    all_weights = {}  # name -> {dtype, size, offset, ...}
    total_weight_size = 0

    # First pass: collect all entries and find min file_offset/offset (weights base)
    min_file_offset = None
    for entry in entries:
        fo = _entry_offset(entry)
        if min_file_offset is None or fo < min_file_offset:
            min_file_offset = fo

    weights_base_offset = min_file_offset or 0

    for entry in entries:
        name = entry["name"]
        size = _entry_size(entry)
        file_offset = _entry_offset(entry)

        # Compute relative offset from weights base
        relative_offset = file_offset - weights_base_offset

        all_weights[name] = {
            "name": name,
            "dtype": entry.get("dtype", "unknown"),
            "size": size,
            "file_offset": file_offset,
            "relative_offset": relative_offset,  # Offset relative to weights_base_offset
        }
        total_weight_size = max(total_weight_size, relative_offset + size)

    print(f"\n📦 Manifest weights:")
    print(f"  Total entries: {len(all_weights)}")
    print(f"  Weights base offset in file: {weights_base_offset}")
    print(f"  Total size: {total_weight_size / 1024 / 1024:.1f} MB")

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Determine which weights SHOULD be used by template
    # ═══════════════════════════════════════════════════════════

    # Get template ops from template
    block_name = template.get("sequence", ["decoder"])[0]
    block = template.get("block_types", {}).get(block_name, {})

    header_ops = _extract_template_ops(block.get("header", []))
    body_def = block.get("body", {})
    body_ops = _collect_body_ops_for_validation(body_def, config)
    footer_ops = _extract_template_ops(block.get("footer", []))
    branch_plan = _build_block_branch_plan(block, config)

    print(f"\n📋 Template structure:")
    print(f"  Header ops: {header_ops}")
    print(f"  Body ops: {body_ops}")
    print(f"  Footer ops: {footer_ops}")

    # Calculate expected weights based on template
    expected_weights = set()
    weight_to_op = {}  # Track which op uses each weight

    # Header weights (run once)
    for op in header_ops:
        weight_refs = TEMPLATE_OP_WEIGHTS.get(op, [])
        for ref in weight_refs:
            patterns = WEIGHT_PATTERNS.get(ref, [ref])
            for pattern in patterns:
                weight_name = pattern  # Header weights don't have layer index
                if weight_name in all_weights:
                    expected_weights.add(weight_name)
                    weight_to_op[weight_name] = f"header:{op}"

    # Body weights (run per layer)
    for layer_idx in range(num_layers):
        layer_body_ops = _resolve_body_ops_for_layer(body_def, config, layer_idx) if isinstance(body_def, dict) else body_ops
        for op in layer_body_ops:
            weight_refs = TEMPLATE_OP_WEIGHTS.get(op, [])
            for ref in weight_refs:
                patterns = WEIGHT_PATTERNS.get(ref, [ref])
                for pattern in patterns:
                    weight_name = pattern.replace("{L}", str(layer_idx))
                    if weight_name in all_weights:
                        expected_weights.add(weight_name)
                        weight_to_op[weight_name] = f"layer.{layer_idx}:{op}"

    # Branch producer weights (run for selected layers only)
    for branch in branch_plan:
        branch_name = str(branch.get("name", "") or "")
        for layer_idx in branch.get("layers", []):
            for producer_item in branch.get("producer_items", []):
                if not isinstance(producer_item, dict):
                    continue
                op = str(producer_item.get("op", "") or "").strip()
                if not op:
                    continue
                explicit_refs = producer_item.get("weight_refs") if isinstance(producer_item.get("weight_refs"), dict) else {}
                if explicit_refs:
                    refs = explicit_refs.items()
                    for ref_name, pattern in refs:
                        if not isinstance(pattern, str):
                            continue
                        weight_name = pattern.replace("{L}", str(int(layer_idx)))
                        if weight_name in all_weights:
                            expected_weights.add(weight_name)
                            weight_to_op[weight_name] = f"branch:{branch_name}:{layer_idx}:{op}:{ref_name}"
                    continue
                for ref in TEMPLATE_OP_WEIGHTS.get(op, []):
                    patterns = WEIGHT_PATTERNS.get(ref, [ref])
                    for pattern in patterns:
                        weight_name = str(pattern).replace("{L}", str(int(layer_idx)))
                        if weight_name in all_weights:
                            expected_weights.add(weight_name)
                            weight_to_op[weight_name] = f"branch:{branch_name}:{layer_idx}:{op}"

    # Footer weights (run once)
    for op in footer_ops:
        weight_refs = TEMPLATE_OP_WEIGHTS.get(op, [])
        for ref in weight_refs:
            patterns = WEIGHT_PATTERNS.get(ref, [ref])
            for pattern in patterns:
                weight_name = pattern
                if weight_name in all_weights:
                    expected_weights.add(weight_name)
                    weight_to_op[weight_name] = f"footer:{op}"

    # ═══════════════════════════════════════════════════════════
    # STEP 3: Extract weights from IR Lower 1 weights field
    # ═══════════════════════════════════════════════════════════

    # IR Lower 1 ops preserve the original IR1 weights field
    ir1_used_weights = set()

    for ir_op in ir_lower_1_ops:
        weights = ir_op.get("weights", {})
        for wkey, winfo in weights.items():
            if isinstance(winfo, dict) and "name" in winfo:
                ir1_used_weights.add(winfo["name"])

    # ═══════════════════════════════════════════════════════════
    # STEP 4: VALIDATION - Check weight coverage
    # ═══════════════════════════════════════════════════════════

    all_weight_names = set(all_weights.keys())

    # Weights in manifest that are NOT inference model weights.
    # tiny.* entries are parity-harness tensors emitted by tiny train init.
    non_model_weights = {"vocab_offsets", "vocab_strings", "vocab_merges", "vocab_scores", "vocab_types"}
    for wname in all_weight_names:
        if str(wname).startswith("tiny."):
            non_model_weights.add(wname)
    training_cfg = config.get("training") if isinstance(config.get("training"), dict) else {}
    tiny_cfg = training_cfg.get("tiny_parity") if isinstance(training_cfg.get("tiny_parity"), dict) else {}
    state_tensors = tiny_cfg.get("state_tensors") if isinstance(tiny_cfg.get("state_tensors"), dict) else {}
    for v in state_tensors.values():
        if isinstance(v, str) and v:
            non_model_weights.add(v)
    model_weights = all_weight_names - non_model_weights

    ignored_weight_reasons = _ignored_manifest_weights(template, config, model_weights)
    if ignored_weight_reasons:
        model_weights -= set(ignored_weight_reasons)
        reason_counts: Dict[str, int] = {}
        for reason in ignored_weight_reasons.values():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items()):
            print(f"  Ignoring {count} circuit-declared manifest weight(s): {reason}")

    # Weights expected but not used by IR1
    unused_by_ir1 = expected_weights - ir1_used_weights

    # Weights in manifest but not used at all
    completely_unused = model_weights - expected_weights - ir1_used_weights

    coverage = len(ir1_used_weights) / len(model_weights) * 100 if model_weights else 0

    print(f"\n🔍 Weight validation:")
    print(f"  Model weights in manifest: {len(model_weights)}")
    print(f"  Expected by template: {len(expected_weights)}")
    print(f"  Used by IR1 kernels: {len(ir1_used_weights)}")
    print(f"  Coverage: {coverage:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # STEP 5: Report unused weights (potential bugs)
    # ═══════════════════════════════════════════════════════════

    validation_errors = []

    if unused_by_ir1:
        print(f"\n⚠️  WEIGHTS EXPECTED BUT NOT USED BY IR1 ({len(unused_by_ir1)}):")

        # Categorize unused weights
        header_unused = [w for w in unused_by_ir1 if weight_to_op.get(w, "").startswith("header:")]
        footer_unused = [w for w in unused_by_ir1 if weight_to_op.get(w, "").startswith("footer:")]
        body_unused = [w for w in unused_by_ir1 if "layer." in w]

        if header_unused:
            print(f"\n   Header weights (not processed by IR1):")
            for w in sorted(header_unused)[:10]:
                print(f"     - {w} (used by {weight_to_op.get(w, 'unknown')})")
            if len(header_unused) > 10:
                print(f"     ... and {len(header_unused) - 10} more")

            validation_errors.append(
                f"Header weights not used: {len(header_unused)} weights\n"
                f"   FIX: Add header ops to IR1 generation (tokenizer, embedding)"
            )

        if footer_unused:
            print(f"\n   Footer weights (not processed by IR1):")
            for w in sorted(footer_unused)[:10]:
                print(f"     - {w} (used by {weight_to_op.get(w, 'unknown')})")
            if len(footer_unused) > 10:
                print(f"     ... and {len(footer_unused) - 10} more")

            validation_errors.append(
                f"Footer weights not used: {len(footer_unused)} weights\n"
                f"   FIX: Add footer ops to IR1 generation (final_norm, projector, logits)"
            )

        if body_unused:
            print(f"\n   Body weights (not processed by IR1):")
            for w in sorted(body_unused)[:10]:
                print(f"     - {w} (used by {weight_to_op.get(w, 'unknown')})")
            if len(body_unused) > 10:
                print(f"     ... and {len(body_unused) - 10} more")

            validation_errors.append(
                f"Body weights not used: {len(body_unused)} weights\n"
                f"   FIX: Check body ops in IR1 are loading all required weights"
            )

    if completely_unused:
        print(f"\n⚠️  WEIGHTS IN MANIFEST BUT NOT IN TEMPLATE ({len(completely_unused)}):")
        for w in sorted(completely_unused)[:10]:
            print(f"     - {w}")
        if len(completely_unused) > 10:
            print(f"     ... and {len(completely_unused) - 10} more")

        validation_errors.append(
            f"Weights not mapped to any template op: {len(completely_unused)} weights\n"
            f"   FIX: Add TEMPLATE_OP_WEIGHTS mapping for these weight types"
        )

    # ═══════════════════════════════════════════════════════════
    # STEP 6: HARD FAULT if validation fails
    # ═══════════════════════════════════════════════════════════

    if validation_errors:
        print(f"\n{'='*60}")
        print("❌ HARD FAULT: WEIGHT VALIDATION FAILED")
        print(f"{'='*60}")

        for i, err in enumerate(validation_errors, 1):
            print(f"\n[{i}] {err}")

        print(f"\n" + "="*60)
        print("WHY THIS MATTERS:")
        print("  - Unused weights = broken inference (wrong output)")
        print("  - For backprop: gradients will be ZERO for these weights")
        print("  - Model will not learn/work correctly")
        print(f"{'='*60}")

        raise RuntimeError(
            f"Weight validation failed: {len(validation_errors)} issues found.\n"
            f"Fix the issues above before proceeding."
        )

    print(f"\n✅ All {len(model_weights)} model weights are mapped!")

    # ═══════════════════════════════════════════════════════════
    # STEP 7: Plan activation buffers from config + context_len
    # ═══════════════════════════════════════════════════════════

    embed_dim = config.get("embed_dim", 896)
    num_heads = config.get("num_heads", 14)
    num_kv_heads = config.get("num_kv_heads", 2)
    head_dim = config.get("head_dim", 64)
    intermediate_size = config.get("intermediate_size", 4864)
    vocab_size = config.get("vocab_size", 151936)
    uses_kv_cache = bool(config.get("_template_uses_kv_cache", True))
    uses_rope = bool(config.get("_template_uses_rope", True))
    has_logits = bool(config.get("_template_has_logits", True))
    decode_kv_cache_dtype = str(config.get("decode_kv_cache_dtype", "fp32") or "fp32").strip().lower()
    kv_cache_dtype = "fp16" if mode == "decode" and decode_kv_cache_dtype in {"fp16", "f16"} else "fp32"
    kv_elem_bytes = _dtype_size_bytes(kv_cache_dtype)

    head_dim = int(head_dim)
    num_heads = int(num_heads)
    num_kv_heads = int(num_kv_heads)
    def _positive_int_config(name: str, default: int) -> int:
        try:
            value = int(config.get(name, default) or default)
        except Exception:
            value = default
        return value if value > 0 else default

    max_q_head_dim = max(head_dim, _positive_int_config("max_q_head_dim", head_dim))
    max_k_head_dim = max(head_dim, _positive_int_config("max_k_head_dim", head_dim))
    max_v_head_dim = max(head_dim, _positive_int_config("max_v_head_dim", head_dim))
    kv_cache_head_dim = max(
        max_k_head_dim,
        max_v_head_dim,
        _positive_int_config("kv_cache_head_dim", head_dim),
    )
    max_attn_head_dim = max(max_q_head_dim, max_k_head_dim, max_v_head_dim, kv_cache_head_dim)
    kv_cache_token_stride_total = int(config.get("kv_cache_token_stride_total", 0) or 0)

    # Use provided context_len or default from config
    max_context = config.get("context_length", 32768)
    if context_len is None:
        context_len = max_context
    else:
        context_len = min(context_len, max_context)

    # For decode mode, we process 1 token but need full KV cache
    if mode == "decode":
        seq_len = 1  # tokens per forward pass
    else:
        seq_len = context_len  # prefill processes all tokens

    print(f"\n📊 Activation memory planning:")
    print(f"  Mode: {mode}")
    print(f"  Context length: {context_len}")
    print(f"  Sequence length (per pass): {seq_len}")

    # Calculate buffer sizes
    activation_buffers = []
    current_offset = 0

    def add_buffer(name, size, shape_desc, dtype="fp32"):
        nonlocal current_offset
        activation_buffers.append({
            "name": name,
            "size": size,
            "offset": current_offset,
            "shape": shape_desc,
            "dtype": dtype
        })
        current_offset += size

    # ─────────────────────────────────────────────────────────────
    # HEADER buffers: tokenizer → embedding
    # ─────────────────────────────────────────────────────────────

    # Text input buffer: UTF-8 bytes (estimate 4 bytes per token avg)
    # For decode mode, only need 1 token; for prefill, need full context
    max_input_bytes = seq_len * 16  # conservative estimate (avg token ~4 bytes, pad for unicode)
    add_buffer("text_input", max_input_bytes, f"[{max_input_bytes}]", "u8")

    # Token IDs buffer: tokenizer output [seq_len] as int32
    token_ids_size = seq_len * 4  # int32
    add_buffer("token_ids", token_ids_size, f"[{seq_len}]", "i32")

    image_height = int(config.get("image_height", config.get("image_size", 0)) or 0)
    image_width = int(config.get("image_width", config.get("image_size", 0)) or 0)
    patch_size = int(config.get("patch_size", 0) or 0)
    vision_channels = int(config.get("vision_channels", 3) or 3)
    patch_dim = int(config.get("patch_dim", vision_channels * patch_size * patch_size) or 0)
    vision_num_patches = int(config.get("vision_num_patches", 0) or 0)
    if image_height > 0 and image_width > 0:
        add_buffer("image_input", vision_channels * image_height * image_width * 4, f"[{vision_channels}, {image_height}, {image_width}]")
    if vision_num_patches > 0 and patch_dim > 0:
        add_buffer("patch_scratch", vision_num_patches * patch_dim * 4, f"[{vision_num_patches}, {patch_dim}]")
    if vision_num_patches > 0:
        add_buffer("vision_positions", vision_num_patches * 4 * 4, f"[4, {vision_num_patches}]", "i32")

    # Embedded input: embedding lookup output [seq_len, embed_dim]
    # For decode: [1, embed_dim], for prefill: [context_len, embed_dim]
    embedded_size = seq_len * embed_dim * 4
    backbone_hidden_size = int(config.get("backbone_hidden_size", 0) or 0)
    if backbone_hidden_size > 0:
        add_buffer("backbone_stream", seq_len * backbone_hidden_size * 4, f"[{seq_len}, {backbone_hidden_size}]")
    add_buffer("embedded_input", embedded_size, f"[{seq_len}, {embed_dim}]")

    # Layer input buffer (for ping-pong)
    # Must be large enough for Q8_K quantization of MLP intermediate (n_ff elements)
    # Q8_K uses 272 bytes per 256 elements: ceil(n_ff/256) * 272 * seq_len
    q8k_blocks = (intermediate_size + 255) // 256
    q8k_size = q8k_blocks * 272 * seq_len
    layer_input_size = max(embedded_size, q8k_size)
    add_buffer("layer_input", layer_input_size, f"[{seq_len}, max({embed_dim}, Q8_K({intermediate_size}))]")

    # Residual buffer (for residual connections - stores input before layer processing)
    add_buffer("residual", embedded_size, f"[{seq_len}, {embed_dim}]")

    # ─────────────────────────────────────────────────────────────
    # BODY buffers: KV cache + RoPE (shared across all layers)
    # ─────────────────────────────────────────────────────────────

    # KV cache: [num_layers, 2, num_kv_heads, context_len, head_dim]
    # Stores K and V for all layers, indexed by position
    if uses_kv_cache:
        if kv_cache_token_stride_total > 0:
            total_kv_size = context_len * kv_cache_token_stride_total * kv_elem_bytes
            add_buffer("kv_cache", total_kv_size, f"[variable_kv, {context_len}, mixed_head_dim]", kv_cache_dtype)
        else:
            kv_per_layer = num_kv_heads * context_len * kv_cache_head_dim * kv_elem_bytes
            total_kv_size = num_layers * 2 * kv_per_layer
            add_buffer("kv_cache", total_kv_size, f"[{num_layers}, 2, {num_kv_heads}, {context_len}, {kv_cache_head_dim}]", kv_cache_dtype)

    # RoPE tables: precomputed cos/sin [2, context_len, rotary_dim/2]
    rotary_dim = int(config.get("rotary_dim", head_dim) or head_dim)
    layer_rotary = config.get("layer_rotary_dim")
    if isinstance(layer_rotary, list) and layer_rotary:
        rotary_dim = max(rotary_dim, max(int(v or 0) for v in layer_rotary))
    rope_half = int(rotary_dim) // 2
    if uses_rope:
        rope_size = context_len * rope_half * 4 * 2
        add_buffer("rope_cache", rope_size, f"[2, {context_len}, {rope_half}]")

    # Layer scratch buffers (reused across layers)
    # Q output: [num_heads, seq_len, head_dim]
    q_size = num_heads * seq_len * max_q_head_dim * 4
    add_buffer("q_scratch", q_size, f"[{num_heads}, {seq_len}, {max_q_head_dim}]")

    # K output: [num_kv_heads, seq_len, max_k_head_dim]
    k_size = num_kv_heads * seq_len * max_k_head_dim * 4
    add_buffer("k_scratch", k_size, f"[{num_kv_heads}, {seq_len}, {max_k_head_dim}]")

    # V output: [num_kv_heads, seq_len, max_v_head_dim]
    v_size = num_kv_heads * seq_len * max_v_head_dim * 4
    add_buffer("v_scratch", v_size, f"[{num_kv_heads}, {seq_len}, {max_v_head_dim}]")

    # Attention output: [num_heads, seq_len, max_q_head_dim]
    attn_out_size = num_heads * seq_len * max_q_head_dim * 4
    q_gate_proj_dim = int(config.get("q_gate_proj_dim", config.get("attn_q_gate_proj_dim", 0)) or 0)
    if q_gate_proj_dim <= 0:
        q_gate_proj_dim = 2 * num_heads * max_q_head_dim
    attn_gate_dim = int(config.get("attn_gate_dim", max(q_gate_proj_dim - (num_heads * max_q_head_dim), 0)) or 0)
    if attn_gate_dim <= 0:
        attn_gate_dim = num_heads * max_q_head_dim
    add_buffer("attn_q_gate_packed", seq_len * q_gate_proj_dim * 4, f"[{seq_len}, {q_gate_proj_dim}]")
    add_buffer("attn_gate", seq_len * attn_gate_dim * 4, f"[{seq_len}, {attn_gate_dim}]")
    add_buffer("attn_scratch", attn_out_size, f"[{num_heads}, {seq_len}, {max_q_head_dim}]")

    kv_lora_rank = int(config.get("kv_lora_rank", 0) or 0)
    qk_nope_dim = int(config.get("qk_nope_head_dim", 0) or 0)
    qk_rope_dim = int(config.get("qk_rope_head_dim", 0) or 0)
    if kv_lora_rank > 0 and qk_rope_dim > 0:
        add_buffer("compressed_kv", seq_len * (kv_lora_rank + qk_rope_dim) * 4, f"[{seq_len}, {kv_lora_rank + qk_rope_dim}]")
        add_buffer("compressed_kv_normed", seq_len * kv_lora_rank * 4, f"[{seq_len}, {kv_lora_rank}]")
        if qk_nope_dim > 0:
            add_buffer("k_nope", num_heads * seq_len * qk_nope_dim * 4, f"[{num_heads}, {seq_len}, {qk_nope_dim}]")

    if bool(config.get("gemma4_per_layer_embedding", False)):
        per_layer_dim = int(config.get("per_layer_dim", 0) or 0)
        if per_layer_dim > 0 and num_layers > 0:
            per_layer_size = seq_len * num_layers * per_layer_dim * 4
            add_buffer("gemma4_per_layer_stream", per_layer_size, f"[{seq_len}, {num_layers}, {per_layer_dim}]")

    # MLP scratch: [seq_len, intermediate_size * 2]
    mlp_size = seq_len * intermediate_size * 2 * 4
    # Fused attention scratch needs more space (Q, attn_out, proj, qkv_scratch)
    # Formula: 3 * num_heads * seq_len * head_dim * 4 + qkv_scratch (embed_dim * 4 * tokens + overhead)
    # For safety, use at least 350KB for decode fused attention
    fused_attn_scratch = max(350 * 1024, 3 * num_heads * seq_len * max_attn_head_dim * 4 + embed_dim * 4 * seq_len * 4)
    # BF16 GeGLU needs 3 * seq_len * dim * 4 (input [a,b] + output)
    geglu_bf16_scratch = seq_len * intermediate_size * 3 * 4
    scratch_size = max(mlp_size, fused_attn_scratch, geglu_bf16_scratch)
    add_buffer("mlp_scratch", scratch_size, f"[max({seq_len}*{intermediate_size*2}, fused_attn, geglu_bf16)]")

    # Layer output: [seq_len, embed_dim]
    layer_out_size = seq_len * embed_dim * 4
    add_buffer("layer_output", layer_out_size, f"[{seq_len}, {embed_dim}]")

    projector_in_dim = int(config.get("projector_in_dim", 0) or 0)
    projector_hidden_dim = int(config.get("projector_hidden_dim", 0) or 0)
    projector_out_dim = int(config.get("projector_out_dim", 0) or 0)
    projector_total_out_dim = int(config.get("projector_total_out_dim", projector_out_dim) or 0)
    num_deepstack_layers = int(config.get("num_deepstack_layers", 0) or 0)
    merged_tokens = int(config.get("vision_merged_tokens", 0) or 0)
    if num_deepstack_layers > 0 and merged_tokens > 0:
        if projector_in_dim > 0:
            add_buffer("branch_stream", merged_tokens * projector_in_dim * 4, f"[{merged_tokens}, {projector_in_dim}]")
            add_buffer("branch_normed", merged_tokens * projector_in_dim * 4, f"[{merged_tokens}, {projector_in_dim}]")
        if projector_hidden_dim > 0:
            add_buffer("branch_mlp", merged_tokens * projector_hidden_dim * 4, f"[{merged_tokens}, {projector_hidden_dim}]")
        if projector_out_dim > 0:
            add_buffer(
                "branch_collect",
                merged_tokens * projector_out_dim * num_deepstack_layers * 4,
                f"[{merged_tokens}, {projector_out_dim * num_deepstack_layers}]",
            )
    if projector_total_out_dim > 0 and merged_tokens > 0:
        add_buffer("vision_output", merged_tokens * projector_total_out_dim * 4, f"[{merged_tokens}, {projector_total_out_dim}]")

    recurrent_q = int(config.get("q_dim", 0) or 0)
    recurrent_k = int(config.get("k_dim", 0) or 0)
    recurrent_v = int(config.get("v_dim", 0) or 0)
    recurrent_inner = int(config.get("ssm_inner_size", 0) or 0)
    recurrent_gate = int(config.get("gate_dim", 0) or 0)
    recurrent_conv_history = int(config.get("ssm_conv_history", 0) or 0)
    recurrent_conv_channels = int(config.get("ssm_conv_channels", 0) or 0)
    recurrent_state_size = int(config.get("ssm_state_size", 0) or 0)
    recurrent_state_heads, recurrent_state_rows, recurrent_state_cols = _recurrent_state_shape(config)
    if any(v > 0 for v in (
        recurrent_q, recurrent_k, recurrent_v, recurrent_inner,
        recurrent_gate, recurrent_conv_channels, recurrent_state_size,
    )):
        packed_dim = max(recurrent_q + recurrent_k + recurrent_v, recurrent_inner, int(config.get("mamba_projection_size", 0) or 0))
        packed_size = seq_len * packed_dim * 4
        recurrent_inner_size = seq_len * recurrent_inner * 4
        gate_size = seq_len * recurrent_gate * 4
        beta_size = seq_len * recurrent_gate * 4
        rq_size = seq_len * recurrent_q * 4
        rk_size = seq_len * recurrent_k * 4
        rv_size = seq_len * recurrent_v * 4
        conv_input_width = max(1, recurrent_conv_history + seq_len)
        conv_input_size = max(1, recurrent_conv_channels) * conv_input_width * 4
        conv_qkv_size = seq_len * max(1, recurrent_conv_channels) * 4
        conv_state_stride = max(1, recurrent_conv_history) * max(1, recurrent_conv_channels) * 4
        ssm_state_stride = max(1, recurrent_state_heads) * max(1, recurrent_state_rows) * max(1, recurrent_state_cols) * 4
        conv_state_size = num_layers * conv_state_stride
        ssm_state_size = num_layers * ssm_state_stride
        add_buffer("recurrent_packed", packed_size, f"[{seq_len}, {packed_dim}]")
        add_buffer("recurrent_z", recurrent_inner_size, f"[{seq_len}, {recurrent_inner}]")
        add_buffer("recurrent_normed", recurrent_inner_size, f"[{seq_len}, {recurrent_inner}]")
        add_buffer("recurrent_g", gate_size, f"[{seq_len}, {recurrent_gate}]")
        add_buffer("recurrent_beta", beta_size, f"[{seq_len}, {recurrent_gate}]")
        add_buffer("recurrent_q", rq_size, f"[{seq_len}, {recurrent_q}]")
        add_buffer("recurrent_k", rk_size, f"[{seq_len}, {recurrent_k}]")
        add_buffer("recurrent_v", rv_size, f"[{seq_len}, {recurrent_v}]")
        add_buffer("recurrent_conv_input", conv_input_size, f"[{recurrent_conv_channels}, {recurrent_conv_history + seq_len}]")
        add_buffer("recurrent_conv_qkv_raw", conv_qkv_size, f"[{seq_len}, {recurrent_conv_channels}]")
        add_buffer("recurrent_conv_qkv", conv_qkv_size, f"[{seq_len}, {recurrent_conv_channels}]")
        add_buffer("recurrent_conv_state", conv_state_size, f"[{num_layers}, {recurrent_conv_history}, {recurrent_conv_channels}]")
        add_buffer("recurrent_ssm_state", ssm_state_size, f"[{num_layers}, {recurrent_state_heads}, {recurrent_state_rows}, {recurrent_state_cols}]")

    # ─────────────────────────────────────────────────────────────
    # FOOTER buffers: final output
    # ─────────────────────────────────────────────────────────────

    # Logits: [seq_len, vocab_size] - decode can be last-only or full
    logits_layout = _resolve_logits_layout(config, mode)
    if has_logits:
        logits_seq = _logits_seq_for_layout(logits_layout, mode, seq_len, context_len, config)
        logits_size = logits_seq * vocab_size * 4
        add_buffer("logits", logits_size, f"[{logits_seq}, {vocab_size}]")

    total_activation_size = current_offset

    print(f"\n  Buffer breakdown:")
    for buf in activation_buffers:
        size_mb = buf["size"] / (1024 * 1024)
        print(f"    {buf['name']:<20} {buf['shape']:<40} {size_mb:>8.2f} MB")
    print(f"  {'─' * 70}")
    print(f"  {'Total':<20} {'':<40} {total_activation_size / (1024 * 1024):>8.2f} MB")

    # ═══════════════════════════════════════════════════════════
    # STEP 8: Build final layout
    # ═══════════════════════════════════════════════════════════

    # Build weight layout with relative offsets (relative to weights_base_offset)
    weight_layout = []
    for name in sorted(all_weights.keys()):
        w = all_weights[name]
        rel_off = w["relative_offset"]
        abs_off = weights_base_offset + rel_off
        weight_layout.append({
            "name": name,
            "dtype": w["dtype"],
            "size": w["size"],
            "offset": rel_off,  # Offset relative to weights_base_offset
            "abs_offset": abs_off,  # Offset relative to bump base (absolute)
            "define": f"W_{_sanitize_macro(name)}",
        })

    # Add context_len to config for codegen
    layout_config = dict(config)
    if context_len is not None:
        layout_config["context_length"] = context_len
        layout_config["context_len"] = context_len
    elif "context_length" not in layout_config:
        layout_config["context_length"] = layout_config.get("max_seq_len", 32768)
    # Persist resolved logits layout in layout config
    layout_config["logits_layout"] = logits_layout

    # Get bump_layout from manifest (written by converter)
    # This ensures codegen uses the same offsets as the converter
    bump_layout = manifest.get("bump_layout", {
        # Defaults if manifest doesn't have bump_layout (backward compat)
        "header_size": 128,
        "ext_metadata_size": 24,
        "data_start": 152,
        "description": "Offsets: [0..header_size) header, [header_size..data_start) ext_metadata, [data_start..] dtype_table + weights"
    })

    activations_base = weights_base_offset + total_weight_size
    for buf in activation_buffers:
        buf["define"] = f"A_{_sanitize_macro(buf.get('name', 'buffer'))}"
        buf["abs_offset"] = activations_base + buf.get("offset", 0)

    total_size = activations_base + total_activation_size
    total_size = _align_up(total_size, 64)

    layout = {
        "format": "memory-layout-v7",
        "version": 2,
        "mode": mode,
        "config": layout_config,
        # BUMP file layout constants - passed from converter via manifest
        # All downstream (codegen, C runtime) should use these, NOT hardcoded values
        "bump_layout": bump_layout,
        # Note: operations are NOT included here - use generate_ir_lower_2 for lowered ops with offsets
        "validation": {
            "status": "PASS",
            "total_weights": len(model_weights),
            "used_weights": len(ir1_used_weights),
            "coverage_percent": coverage,
        },
        "memory": {
            "weights": {
                "size": total_weight_size,
                "bump_size": total_weight_size,
                "base_offset": weights_base_offset,  # File offset where weights start
                "entries": weight_layout,
            },
            "activations": {
                "size": total_activation_size,
                "buffers": activation_buffers,
            },
            "arena": {
                "mode": "region",
                "weights_base": weights_base_offset,
                "activations_base": activations_base,
                "total_size": total_size,
            },
        },
    }

    print(f"\n✓ Memory layout complete")
    print(f"  Bump (weights): {total_weight_size / 1024 / 1024:.1f} MB")
    print(f"  Activations: {total_activation_size / 1024:.1f} KB")

    return layout


def generate_memory_layout_packed(
    ir_lower_1_ops: List[Dict],
    manifest: Dict,
    registry: Dict,
    mode: str,
    context_len: int = None,
    layer_limit: Optional[int] = None,
) -> Dict:
    """Generate a packed/streamed layout where weights + activations share one arena."""
    print(f"\n{'='*60}")
    print("MEMORY PLANNER (packed/streamed)")
    print(f"{'='*60}")

    config = dict(manifest.get("config", {}))
    if layer_limit:
        config["num_layers"] = int(layer_limit)

    entries = manifest.get("entries", [])
    if not entries:
        raise RuntimeError("Manifest entries missing; cannot build packed layout.")
    entry_by_name = {e["name"]: e for e in entries}

    # Validate: every used weight exists in manifest
    used_weights = set()
    for op in ir_lower_1_ops:
        for w in op.get("weights", {}).values():
            if isinstance(w, dict) and "name" in w:
                used_weights.add(w["name"])
    missing = [w for w in sorted(used_weights) if w not in entry_by_name]
    if missing:
        raise RuntimeError(f"Packed layout: missing {len(missing)} weights in manifest: {missing[:5]}")

    act_specs = build_activation_specs(config, mode, context_len, num_layers_override=layer_limit)

    weight_offset = 0
    act_offset = 0
    weight_layout = []
    weight_offsets = {}
    activation_buffers = []
    act_offsets = {}

    weight_order = []
    seen_weights = set()
    for op in ir_lower_1_ops:
        for w in op.get("weights", {}).values():
            if not isinstance(w, dict):
                continue
            name = w.get("name")
            if not name or name in seen_weights:
                continue
            seen_weights.add(name)
            weight_order.append(name)

    def alloc_weight(name: str) -> None:
        nonlocal weight_offset
        if name in weight_offsets:
            return
        entry = entry_by_name[name]
        size = int(entry.get("size", entry.get("size_bytes", 0)))
        off = _align_up(weight_offset, 64)
        weight_offsets[name] = off
        weight_layout.append({
            "name": name,
            "dtype": entry.get("dtype", "unknown"),
            "size": size,
            "offset": off,
            "abs_offset": off,
            "file_offset": _entry_offset(entry),
            "define": f"W_{_sanitize_macro(name)}",
        })
        weight_offset = off + size

    def alloc_act(name: str) -> None:
        nonlocal act_offset
        if name in act_offsets:
            return
        spec = act_specs.get(name)
        if not spec:
            return
        off = _align_up(act_offset, 64)
        act_offsets[name] = off
        activation_buffers.append({
            "name": name,
            "size": spec["size"],
            "offset": off,
            "abs_offset": off,
            "shape": spec["shape"],
            "dtype": spec["dtype"],
            "define": f"A_{_sanitize_macro(name)}",
        })
        act_offset = off + spec["size"]

    # Allocate all weights first (in order of first use)
    for name in weight_order:
        alloc_weight(name)

    weights_end = _align_up(weight_offset, 64)
    act_offset = weights_end

    # Simulate op order (using same buffer naming logic as IR Lower 2)
    current_input_buffer = "token_ids"
    current_output_buffer = "embedded_input"
    qkv_input_buffer = "token_ids"

    for op in ir_lower_1_ops:
        op_type = op.get("op", "")
        kernel_type = op.get("kernel", "")

        if op_type in ("q_proj", "q_gate_proj", "k_proj", "v_proj"):
            if op_type == "q_proj":
                qkv_input_buffer = current_input_buffer
            alloc_act(qkv_input_buffer)
            if op_type == "q_proj":
                alloc_act("q_scratch")
            elif op_type == "q_gate_proj":
                alloc_act("attn_q_gate_packed")
            elif op_type == "k_proj":
                alloc_act("k_scratch")
            elif op_type == "v_proj":
                alloc_act("v_scratch")
        elif op_type == "split_q_gate":
            alloc_act("attn_q_gate_packed")
            alloc_act("q_scratch")
            alloc_act("attn_gate")
        elif op_type == "qkv_proj":
            qkv_input_buffer = current_input_buffer
            alloc_act(qkv_input_buffer)
            alloc_act("q_scratch")
            alloc_act("k_scratch")
            alloc_act("v_scratch")
        else:
            if "embedding" in kernel_type.lower():
                alloc_act("token_ids")
                alloc_act("embedded_input")
            elif op_type == "logits":
                alloc_act("logits")
            else:
                alloc_act(current_input_buffer)
                alloc_act(current_output_buffer)

            if op_type == "residual_add":
                alloc_act("residual")

        # Scratch buffers
        if op.get("scratch"):
            alloc_act("mlp_scratch")

        if op_type == "rope_qk":
            alloc_act("q_scratch")
            alloc_act("k_scratch")
            alloc_act("rope_cache")
        if op_type == "rope_q":
            alloc_act("q_scratch")
            alloc_act("rope_cache")
        if _is_vision_mrope_operation(op):
            alloc_act("q_scratch")
            alloc_act("k_scratch")
            alloc_act("vision_positions")
        if op_type in ("vision_position_ids", "position_ids_2d"):
            alloc_act("vision_positions")

        if op_type in ("kv_cache_store", "mla_kv_cache_store", "mla_kv_cache_batch_store"):
            alloc_act("k_scratch")
            alloc_act("v_scratch")
            alloc_act("kv_cache")
        if op_type == "kv_cache_store_shared_q":
            alloc_act("q_scratch")
            alloc_act("kv_cache")

        if op_type == "attn" or "attention" in kernel_type:
            alloc_act("q_scratch")
            alloc_act("k_scratch")
            alloc_act("v_scratch")
            alloc_act("attn_scratch")
        if op_type == "attn_gate_sigmoid_mul":
            alloc_act("attn_gate")
            alloc_act("attn_scratch")

        if op_type == "residual_add":
            alloc_act("residual")

        if op.get("section") == "body" and op_type == "rmsnorm":
            alloc_act("layer_input")
            alloc_act("residual")

        # Ping-pong update (same as generate_ir_lower_2)
        if "embedding" in kernel_type.lower():
            current_input_buffer = "embedded_input"
            current_output_buffer = "layer_input"
        elif op_type in ("q_proj", "q_gate_proj", "split_q_gate", "attn_gate_sigmoid_mul", "k_proj", "v_proj", "qkv_proj", "q_norm", "rope_qk", "rope_q", "mrope_qk", "vision_position_ids", "position_ids_2d", "bias_add") or \
                (mode == "prefill" and op_type in ("attn", "attn_sliding", "attn_shared_kv", "attn_sliding_shared_kv")):
            pass
        else:
            current_input_buffer, current_output_buffer = current_output_buffer, current_input_buffer

    # Ensure required buffers exist for runtime pointers
    alloc_act("kv_cache")
    alloc_act("rope_cache")
    alloc_act("logits")

    total_weight_bytes = weights_end
    total_activation_bytes = act_offset - weights_end

    layout_config = dict(config)
    if context_len is not None:
        layout_config["context_length"] = context_len
        layout_config["context_len"] = context_len
    elif "context_length" not in layout_config:
        layout_config["context_length"] = layout_config.get("max_seq_len", 32768)

    layout = {
        "format": "memory-layout-v7",
        "version": 3,
        "mode": mode,
        "config": layout_config,
        "bump_layout": manifest.get("bump_layout", {
            "header_size": 128,
            "ext_metadata_size": 24,
            "data_start": 152,
        }),
        "memory": {
            "weights": {
                "size": total_weight_bytes,
                "bump_size": total_weight_bytes,
                "base_offset": 0,
                "entries": weight_layout,
            },
            "activations": {
                "size": total_activation_bytes,
                "buffers": activation_buffers,
            },
            "arena": {
                "mode": "packed",
                "weights_base": 0,
                "activations_base": weights_end,
                "total_size": total_size,
            },
        },
    }

    print(f"\n✓ Packed layout complete")
    print(f"  Total arena: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Weights (used): {sum(e.get('size', 0) for e in weight_layout) / 1024 / 1024:.1f} MB")
    print(f"  Activations (allocated): {total_activation_bytes / 1024 / 1024:.1f} MB")

    return layout


def write_manifest_map(layout: Dict, manifest: Dict, output_path: Path) -> None:
    """Write weights_manifest.map with runtime offsets from layout."""
    weights = layout.get("memory", {}).get("weights", {}).get("entries", [])
    rt_by_name = {e["name"]: int(e.get("abs_offset", e.get("offset", 0))) for e in weights}
    # Preserve ordering by runtime offset (stream-friendly)
    ordered = sorted(weights, key=lambda e: int(e.get("abs_offset", e.get("offset", 0))))

    entry_by_name = {e["name"]: e for e in manifest.get("entries", [])}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write("# ck-bumpwgt5-manifest-map v1\n")
        f.write("# name|dtype|file_offset|size|runtime_offset\n")
        for w in ordered:
            name = w["name"]
            m = entry_by_name.get(name)
            if not m:
                continue
            file_off = _entry_offset(m)
            size = _entry_size(m)
            dtype = m.get("dtype", w.get("dtype", "unknown"))
            rt_off = rt_by_name.get(name, 0)
            f.write(f"{name}|{dtype}|0x{file_off:016X}|0x{size:016X}|0x{rt_off:016X}\n")


def generate_ir_lower_2(
    ir_lower_1_ops: List[Dict],
    layout: Dict,
    manifest: Dict,
    registry: Dict,
    mode: str
) -> Dict:
    """
    IR Lower 2: Add concrete memory offsets to IR Lower 1 ops.

    IR Lower 1 already has:
    - inputs: {type='weight', name, offset, size} or {type='activation', source}
    - outputs: {buffer, dtype, shape}
    - scratch: [{name, size, dtype}]

    This function adds:
    - Concrete bump_offset for each weight input
    - Concrete activation_offset for each activation input/output
    - Pointer expressions for codegen

    Args:
        ir_lower_1_ops: IR Lower 1 ops with inputs/outputs/scratch
        layout: Memory layout with weight offsets and activation buffers
        manifest: Model manifest
        registry: Kernel registry
        mode: Execution mode

    Returns:
        Final lowered IR with explicit pointer expressions
    """
    print(f"\n{'='*60}")
    print("IR LOWER 2 (Add memory offsets)")
    print(f"{'='*60}")

    config = manifest.get("config", {})
    template_doc = manifest.get("template", {}) if isinstance(manifest.get("template"), dict) else {}
    contract_doc = template_doc.get("contract") if isinstance(template_doc.get("contract"), dict) else None
    if not contract_doc:
        raise RuntimeError(
            "HARD CIRCUIT CONTRACT FAULT: IR Lower 2 requires the hydrated circuit contract. "
            "Fix circuit hydration; do not infer a circuit from the model family during lowering."
        )
    # Carry semantic contract forward so IR Lower 3/codegen can fail-fast on missing semantics.
    if contract_doc:
        config = dict(config)
        config["contract"] = contract_doc

    # Build weight offset lookup from layout
    weight_offsets = {}
    memory = layout.get("memory", {})
    weight_entries = memory.get("weights", {}).get("entries", [])
    for entry in weight_entries:
        weight_offsets[entry["name"]] = {
            "bump_offset": entry["offset"],
            "dtype": entry["dtype"],
            "size": entry["size"],
        }

    # Activation buffer lookup
    activation_buffers = {}
    for buf in memory.get("activations", {}).get("buffers", []):
        activation_buffers[buf["name"]] = buf

    # KV cache slice helper (prefill path may write directly into KV cache)
    layout_config = layout.get("config", {}) if isinstance(layout, dict) else {}
    if layout_config:
        merged = dict(config)
        merged.update(layout_config)
        config = merged
    context_len = config.get("context_length", config.get("max_seq_len", config.get("context_len", 0)))
    num_kv_heads = config.get("num_kv_heads", 0)
    head_dim = config.get("head_dim", 0)
    kv_cache_dtype = str(config.get("decode_kv_cache_dtype", "fp32") or "fp32").strip().lower()
    kv_elem_bytes = _dtype_size_bytes(kv_cache_dtype if kv_cache_dtype in {"fp16", "f16"} else "fp32")

    layer_k_cache_offset = config.get("layer_k_cache_offset") if isinstance(config.get("layer_k_cache_offset"), list) else []
    layer_v_cache_offset = config.get("layer_v_cache_offset") if isinstance(config.get("layer_v_cache_offset"), list) else []

    def kv_layer_offsets(layer: int) -> Optional[Tuple[int, int]]:
        kv_buf = activation_buffers.get("kv_cache")
        if not kv_buf or not context_len or not num_kv_heads or not head_dim:
            return None
        if 0 <= int(layer) < len(layer_k_cache_offset) and 0 <= int(layer) < len(layer_v_cache_offset):
            k_base = kv_buf["offset"] + int(layer_k_cache_offset[int(layer)]) * int(context_len) * kv_elem_bytes
            v_base = kv_buf["offset"] + int(layer_v_cache_offset[int(layer)]) * int(context_len) * kv_elem_bytes
            return k_base, v_base
        kv_per_layer = num_kv_heads * context_len * head_dim * kv_elem_bytes
        base = kv_buf["offset"] + layer * 2 * kv_per_layer
        return base, base + kv_per_layer

    lowered_ops = []

    # ═══════════════════════════════════════════════════════════════════════════════
    # MEMORY PLANNER: Pre-compute buffer assignments based on dataflow
    # This replaces the old ping-pong logic with explicit dataflow-based assignment
    # ═══════════════════════════════════════════════════════════════════════════════
    print("  Running memory planner...")
    buffer_assignments = plan_memory(ir_lower_1_ops)
    print(f"  ✓ Memory planner assigned buffers for {len(buffer_assignments)} ops")

    # Helper to get buffer info from memory planner
    def get_planned_buffer(op_id: int, io_type: str, name: str) -> Optional[Dict]:
        """Get buffer assignment from memory planner.

        Args:
            op_id: Operation ID
            io_type: 'inputs' or 'outputs'
            name: Input/output name (e.g., 'x', 'y', 'input', 'output')

        Returns:
            Buffer info dict with 'buffer' and 'dtype' keys, or None
        """
        assignment = buffer_assignments.get(op_id, {})
        io_assignments = assignment.get(io_type, {})
        return io_assignments.get(name)

    # Map buffer names to activation buffer names
    # Memory planner uses A_EMBEDDED_INPUT, A_LAYER_INPUT, etc.
    # Layout uses embedded_input, layer_input, etc.
    buffer_name_map = {
        "A_EMBEDDED_INPUT": "embedded_input",
        "A_LAYER_INPUT": "layer_input",
        "A_RESIDUAL": "residual",
        "A_ATTN_SCRATCH": "attn_scratch",
        "A_ATTN_Q_GATE_PACKED": "attn_q_gate_packed",
        "A_ATTN_GATE": "attn_gate",
        "A_COMPRESSED_KV": "compressed_kv",
        "A_COMPRESSED_KV_NORMED": "compressed_kv_normed",
        "A_K_NOPE": "k_nope",
        "A_MLP_SCRATCH": "mlp_scratch",
        "A_LAYER_OUTPUT": "layer_output",
        "A_BRANCH_STREAM": "branch_stream",
        "A_BRANCH_NORMED": "branch_normed",
        "A_BRANCH_MLP": "branch_mlp",
        "A_BRANCH_COLLECT": "branch_collect",
        "A_VISION_OUTPUT": "vision_output",
        "A_BACKBONE_STREAM": "backbone_stream",
        "backbone_stream": "backbone_stream",
        "A_LOGITS": "logits",
        "A_RECURRENT_PACKED": "recurrent_packed",
        "A_RECURRENT_Z": "recurrent_z",
        "A_RECURRENT_G": "recurrent_g",
        "A_RECURRENT_NORMED": "recurrent_normed",
        "A_RECURRENT_BETA": "recurrent_beta",
        "A_RECURRENT_Q": "recurrent_q",
        "A_RECURRENT_K": "recurrent_k",
        "A_RECURRENT_V": "recurrent_v",
        "A_RECURRENT_CONV_STATE": "recurrent_conv_state",
        "A_RECURRENT_SSM_STATE": "recurrent_ssm_state",
        "kv_cache": "kv_cache",
    }

    # Legacy ping-pong tracking (kept for fallback, but should not be needed)
    current_input_buffer = "token_ids"
    current_output_buffer = "embedded_input"
    qkv_input_buffer = "token_ids"
    last_output_buffer: Optional[str] = None

    for ir_op in ir_lower_1_ops:
        lowered_op = {
            "idx": ir_op["idx"],
            "kernel": ir_op["kernel"],
            "op": ir_op["op"],
            "layer": ir_op["layer"],
            "section": ir_op["section"],
            "function": ir_op.get("function", ir_op["kernel"]),
            "weights": {},
            "activations": {},
            "outputs": {},
            "params": {},
        }
        if ir_op.get("required_contract") is not None:
            lowered_op["required_contract"] = copy.deepcopy(ir_op["required_contract"])
        if ir_op.get("resolved_contract") is not None:
            lowered_op["resolved_contract"] = copy.deepcopy(ir_op["resolved_contract"])
            if lowered_op["resolved_contract"].get("kernel_id") != ir_op["kernel"]:
                raise RuntimeError(
                    f"HARD CONTRACT FAULT: memory lowering received kernel {ir_op['kernel']!r}, "
                    f"but GraphIR resolved {lowered_op['resolved_contract'].get('kernel_id')!r}."
                )
        if ir_op.get("resolved_execution") is not None:
            lowered_op["resolved_execution"] = copy.deepcopy(ir_op["resolved_execution"])
            if lowered_op["resolved_execution"].get("kernel_id") != ir_op["kernel"]:
                raise RuntimeError(
                    f"HARD CONTRACT FAULT: memory lowering received kernel {ir_op['kernel']!r}, "
                    f"but execution metadata names {lowered_op['resolved_execution'].get('kernel_id')!r}."
                )
        if ir_op.get("resolved_codegen_capability") is not None:
            lowered_op["resolved_codegen_capability"] = copy.deepcopy(
                ir_op["resolved_codegen_capability"]
            )
            if lowered_op["resolved_codegen_capability"].get("kernel_id") != ir_op["kernel"]:
                raise RuntimeError(
                    f"HARD CODEGEN CAPABILITY FAULT: memory lowering received kernel "
                    f"{ir_op['kernel']!r}, but codegen metadata names "
                    f"{lowered_op['resolved_codegen_capability'].get('kernel_id')!r}."
                )
        if ir_op.get("semantic_checkpoints") is not None:
            lowered_op["semantic_checkpoints"] = copy.deepcopy(ir_op["semantic_checkpoints"])
        if "_kv_cache_read_layer" in ir_op:
            lowered_op["_kv_cache_read_layer"] = int(ir_op["_kv_cache_read_layer"])

        # Process weights - add concrete bump offsets
        for wkey, winfo in ir_op.get("weights", {}).items():
            weight_name = winfo.get("name", "")
            weight_entry = weight_offsets.get(weight_name)

            if weight_entry:
                lowered_op["weights"][wkey] = {
                    "name": weight_name,
                    "bump_offset": weight_entry["bump_offset"],
                    "size": weight_entry["size"],
                    "dtype": weight_entry["dtype"],
                    "ptr_expr": f"bump_weights + {weight_entry['bump_offset']}",
                }
            else:
                # Weight not in layout - use file offset from IR1
                lowered_op["weights"][wkey] = {
                    "name": weight_name,
                    "bump_offset": winfo.get("offset", 0),
                    "size": winfo.get("size", 0),
                    "dtype": winfo.get("dtype", "unknown"),
                    "ptr_expr": f"bump_weights + {winfo.get('offset', 0)}",
                }

        # Special handling for Q/K/V projections: all read from same input, write to different outputs
        op_type = ir_op.get("op", "")
        if op_type == "bias_add":
            bias_for = ir_op.get("bias_for")
            target_buf = None
            if bias_for in ("q_proj", "q_gate_proj", "k_proj", "v_proj"):
                target_buf = {
                    "q_proj": "q_scratch",
                    "q_gate_proj": "attn_q_gate_packed",
                    "k_proj": "k_scratch",
                    "v_proj": "v_scratch",
                }.get(bias_for)
            elif bias_for == "qkv_packed_proj":
                target_buf = "mlp_scratch"
            elif bias_for == "projector_fc1":
                target_buf = "mlp_scratch"
            elif bias_for == "projector_fc2":
                target_buf = "embedded_input"
            else:
                target_buf = last_output_buffer or current_output_buffer
            buf = activation_buffers.get(target_buf) if target_buf else None
            if buf:
                lowered_op["activations"]["y"] = {
                    "buffer": target_buf,
                    "activation_offset": buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {buf['offset']}",
                }
                lowered_op["outputs"]["y"] = {
                    "buffer": target_buf,
                    "activation_offset": buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {buf['offset']}",
                }
            if target_buf:
                last_output_buffer = target_buf
        elif op_type == "qkv_proj":
            # QKV fused projection (prefill uses head-major outputs)
            buf = activation_buffers.get(current_input_buffer)
            if buf:
                for input_name, input_info in ir_op.get("inputs", {}).items():
                    lowered_op["activations"][input_name] = {
                        "buffer": current_input_buffer,
                        "activation_offset": buf["offset"],
                        "dtype": input_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {buf['offset']}",
                    }

            layer_idx = int(ir_op.get("layer", 0))
            kv_offs = kv_layer_offsets(layer_idx) if mode == "prefill" else None
            k_off = kv_offs[0] if kv_offs else None
            v_off = kv_offs[1] if kv_offs else None

            for output_name, output_info in ir_op.get("outputs", {}).items():
                if output_name.startswith("q"):
                    q_buf = activation_buffers.get("q_scratch")
                    lowered_op["outputs"][output_name] = {
                        "buffer": "q_scratch",
                        "activation_offset": q_buf["offset"] if q_buf else 0,
                        "dtype": output_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {q_buf['offset'] if q_buf else 0}",
                    }
                    last_output_buffer = "q_scratch"
                elif output_name.startswith("k"):
                    if k_off is not None:
                        lowered_op["outputs"][output_name] = {
                            "buffer": f"kv_cache_k_L{layer_idx}",
                            "activation_offset": k_off,
                            "dtype": output_info.get("dtype", "fp32"),
                            "ptr_expr": f"activations + {k_off}",
                        }
                    else:
                        k_buf = activation_buffers.get("k_scratch")
                        lowered_op["outputs"][output_name] = {
                            "buffer": "k_scratch",
                            "activation_offset": k_buf["offset"] if k_buf else 0,
                            "dtype": output_info.get("dtype", "fp32"),
                            "ptr_expr": f"activations + {k_buf['offset'] if k_buf else 0}",
                        }
                        last_output_buffer = "k_scratch"
                elif output_name.startswith("v"):
                    if v_off is not None:
                        lowered_op["outputs"][output_name] = {
                            "buffer": f"kv_cache_v_L{layer_idx}",
                            "activation_offset": v_off,
                            "dtype": output_info.get("dtype", "fp32"),
                            "ptr_expr": f"activations + {v_off}",
                        }
                    else:
                        v_buf = activation_buffers.get("v_scratch")
                        lowered_op["outputs"][output_name] = {
                            "buffer": "v_scratch",
                            "activation_offset": v_buf["offset"] if v_buf else 0,
                            "dtype": output_info.get("dtype", "fp32"),
                            "ptr_expr": f"activations + {v_buf['offset'] if v_buf else 0}",
                        }
                        last_output_buffer = "v_scratch"
        elif op_type in ("q_proj", "q_gate_proj", "k_proj", "v_proj", "recurrent_qkv_proj", "recurrent_gate_proj", "recurrent_alpha_proj", "recurrent_beta_proj", "mamba_in_proj"):
            # ═══════════════════════════════════════════════════════════════
            # USE MEMORY PLANNER for QKV input buffer assignment
            # The memory planner knows the correct buffer (main_stream_q8)
            #
            # CRITICAL: Buffer selection depends on kernel's activation dtype:
            # - Kernels with fp32 activation (e.g., gemm_nt_q5_1) need FP32 input
            #   → use embedded_input (FP32 buffer)
            # - Kernels with q8_0 activation (e.g., gemm_nt_q8_0_q8_0) need Q8 input
            #   → use layer_input (Q8 buffer, where quantize_input writes)
            # ═══════════════════════════════════════════════════════════════
            op_id = ir_op.get("idx", ir_op.get("op_id", -1))
            kernel_id = ir_op.get("kernel", "")

            # Determine the correct input buffer based on kernel's activation dtype.
            # Body projections consume the pre-norm stream. Quantized kernels read
            # the Q8 view planned for that stream; FP32/BF16 kernels must read the
            # physical FP32 stream. This matters for Qwen3.5 recurrent_qkv_proj:
            # gemv_q5_k consumes FP32 activations, while layer_input is also reused
            # as the Q8 scratch for quantize_input_0 on this layout.
            needs_q8_input = kernel_needs_q8_activation(registry, kernel_id)
            default_buf_name = "layer_input" if needs_q8_input else "embedded_input"
            default_buf = activation_buffers.get(default_buf_name)

            for input_name, input_info in ir_op.get("inputs", {}).items():
                # Skip weight inputs
                if input_name in ir_op.get("weights", {}):
                    continue

                # Use the planner/declared dataflow slot for both Q8 and FP32/BF16
                # paths.  Older code forced FP32 kernels back to embedded_input,
                # bypassing block_rmsnorm for safetensors models.
                dataflow_name = {"A": "x", "x_q8": "x", "x": "x", "input": "x"}.get(input_name, input_name)
                planned = get_planned_buffer(op_id, "inputs", dataflow_name)
                if not planned:
                    planned = get_planned_buffer(op_id, "inputs", input_name)
                if planned:
                    planner_buf = planned.get("buffer", default_buf_name)
                    declared_slot = _get_declared_dataflow_slot(ir_op, "inputs", dataflow_name, input_name)
                    buf_name = _resolve_logical_buffer_name(
                        planner_buf,
                        declared_slot or input_info.get("slot"),
                        activation_buffers,
                        buffer_name_map,
                    )
                    if not needs_q8_input and buf_name == "main_stream_q8":
                        buf_name = "embedded_input"
                    buf = activation_buffers.get(buf_name)
                else:
                    buf_name = default_buf_name
                    buf = default_buf

                if buf:
                    # Set dtype based on kernel's activation requirement (q8_0 for Q8 kernels, fp32 for FP32 kernels)
                    act_dtype = "q8_0" if needs_q8_input else "fp32"
                    lowered_op["activations"][input_name] = {
                        "buffer": buf_name,
                        "activation_offset": buf["offset"],
                        "dtype": input_info.get("dtype", act_dtype),
                        "ptr_expr": f"activations + {buf['offset']}",
                    }
            # Q writes to the template-declared output slot. Standard attention
            # uses q_scratch; Kimi/DeepSeek MLA keeps packed [q_nope|q_pe]
            # in q_scratch for the packed partial-RoPE helper.
            if op_type == "q_proj":
                for output_name, output_info in ir_op.get("outputs", {}).items():
                    planned = get_planned_buffer(op_id, "outputs", output_name) or get_planned_buffer(op_id, "outputs", "y")
                    declared_slot = _get_declared_dataflow_slot(ir_op, "outputs", output_name, "y")
                    buf_name = _resolve_logical_buffer_name(
                        planned.get("buffer", "q_scratch") if planned else "q_scratch",
                        declared_slot or output_info.get("slot"),
                        activation_buffers,
                        buffer_name_map,
                    )
                    q_buf = activation_buffers.get(buf_name)
                    lowered_op["outputs"][output_name] = {
                        "buffer": buf_name,
                        "activation_offset": q_buf["offset"] if q_buf else 0,
                        "dtype": output_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {q_buf['offset'] if q_buf else 0}",
                    }
            elif op_type == "q_gate_proj":
                buf = activation_buffers.get("attn_q_gate_packed")
                for output_name, output_info in ir_op.get("outputs", {}).items():
                    lowered_op["outputs"][output_name] = {
                        "buffer": "attn_q_gate_packed",
                        "activation_offset": buf["offset"] if buf else 0,
                        "dtype": output_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {buf['offset'] if buf else 0}",
                    }
            # K/V write to their respective scratch buffers
            elif op_type == "k_proj":
                buf = activation_buffers.get("k_scratch")
                for output_name, output_info in ir_op.get("outputs", {}).items():
                    lowered_op["outputs"][output_name] = {
                        "buffer": "k_scratch",
                        "activation_offset": buf["offset"] if buf else 0,
                        "dtype": output_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {buf['offset'] if buf else 0}",
                    }
            elif op_type == "v_proj":
                buf = activation_buffers.get("v_scratch")
                for output_name, output_info in ir_op.get("outputs", {}).items():
                    lowered_op["outputs"][output_name] = {
                        "buffer": "v_scratch",
                        "activation_offset": buf["offset"] if buf else 0,
                        "dtype": output_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {buf['offset'] if buf else 0}",
                    }
            else:
                # Recurrent projection ops have semantic destinations that must
                # stay stable regardless of family name. Do not infer these via
                # model-specific Python branches; keep the stitch contract tied
                # to the declared op type so any template reusing these ops
                # lowers the same way.
                output_buf_by_op = {
                    "recurrent_qkv_proj": "recurrent_packed",
                    "recurrent_gate_proj": "recurrent_z",
                    "recurrent_alpha_proj": "recurrent_g",
                    "recurrent_beta_proj": "recurrent_beta",
                    "mamba_in_proj": "recurrent_packed",
                }
                for output_name, output_info in ir_op.get("outputs", {}).items():
                    dataflow_slot = str(output_info.get("slot", ""))
                    slot_to_buf = {
                        "recurrent_qkv_packed": "recurrent_packed",
                        "recurrent_z": "recurrent_z",
                        "recurrent_alpha": "recurrent_g",
                        "recurrent_beta": "recurrent_beta",
                    }
                    buf_name = output_buf_by_op.get(op_type, slot_to_buf.get(dataflow_slot, "recurrent_packed"))
                    buf = activation_buffers.get(buf_name)
                    lowered_op["outputs"][output_name] = {
                        "buffer": buf_name,
                        "activation_offset": buf["offset"] if buf else 0,
                        "dtype": output_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {buf['offset'] if buf else 0}",
                    }
            if op_type == "q_proj":
                last_output_buffer = "q_scratch"
            elif op_type == "q_gate_proj":
                last_output_buffer = "attn_q_gate_packed"
            elif op_type == "k_proj":
                last_output_buffer = "k_scratch"
            elif op_type == "v_proj":
                last_output_buffer = "v_scratch"
            elif lowered_op["outputs"]:
                last_output_buffer = next(iter(lowered_op["outputs"].values())).get("buffer", last_output_buffer)
        elif op_type == "recurrent_norm_gate":
            _bind_recurrent_norm_gate_io(lowered_op, ir_op, activation_buffers)
            if lowered_op["outputs"]:
                last_output_buffer = "recurrent_normed"
        elif op_type == "patchify":
            image_buf = activation_buffers.get("image_input")
            patch_buf = activation_buffers.get("patch_scratch")
            if image_buf:
                lowered_op["activations"]["image"] = {
                    "buffer": "image_input",
                    "activation_offset": image_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {image_buf['offset']}",
                }
            if patch_buf:
                lowered_op["outputs"]["patches"] = {
                    "buffer": "patch_scratch",
                    "activation_offset": patch_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {patch_buf['offset']}",
                }
                last_output_buffer = "patch_scratch"
        elif op_type == "patch_proj":
            patch_buf = activation_buffers.get("patch_scratch")
            embed_buf = activation_buffers.get("embedded_input")
            if patch_buf:
                lowered_op["activations"]["A"] = {
                    "buffer": "patch_scratch",
                    "activation_offset": patch_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {patch_buf['offset']}",
                }
            if embed_buf:
                lowered_op["outputs"]["C"] = {
                    "buffer": "embedded_input",
                    "activation_offset": embed_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {embed_buf['offset']}",
                }
                last_output_buffer = "embedded_input"
        elif op_type == "patch_proj_aux":
            patch_buf = activation_buffers.get("patch_scratch")
            aux_buf = activation_buffers.get("mlp_scratch")
            if patch_buf:
                lowered_op["activations"]["A"] = {
                    "buffer": "patch_scratch",
                    "activation_offset": patch_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {patch_buf['offset']}",
                }
            if aux_buf:
                lowered_op["outputs"]["C"] = {
                    "buffer": "mlp_scratch",
                    "activation_offset": aux_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {aux_buf['offset']}",
                }
                last_output_buffer = "mlp_scratch"
        elif op_type == "add_stream":
            main_buf = activation_buffers.get("embedded_input")
            aux_buf = activation_buffers.get("mlp_scratch")
            if aux_buf:
                lowered_op["activations"]["b"] = {
                    "buffer": "mlp_scratch",
                    "activation_offset": aux_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {aux_buf['offset']}",
                }
            if main_buf:
                lowered_op["outputs"]["out"] = {
                    "buffer": "embedded_input",
                    "activation_offset": main_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {main_buf['offset']}",
                }
                last_output_buffer = "embedded_input"
        elif op_type in ("vision_position_ids", "position_ids_2d"):
            pos_buf = activation_buffers.get("vision_positions")
            if pos_buf:
                lowered_op["outputs"]["positions"] = {
                    "buffer": "vision_positions",
                    "activation_offset": pos_buf["offset"],
                    "dtype": "i32",
                    "ptr_expr": f"activations + {pos_buf['offset']}",
                }
        elif _is_vision_mrope_operation(ir_op):
            q_buf = activation_buffers.get("q_scratch")
            k_buf = activation_buffers.get("k_scratch")
            pos_buf = activation_buffers.get("vision_positions")
            if q_buf:
                lowered_op["activations"]["q"] = {
                    "buffer": "q_scratch",
                    "activation_offset": q_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {q_buf['offset']}",
                }
                lowered_op["outputs"]["q"] = {
                    "buffer": "q_scratch",
                    "activation_offset": q_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {q_buf['offset']}",
                }
            if k_buf:
                lowered_op["activations"]["k"] = {
                    "buffer": "k_scratch",
                    "activation_offset": k_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {k_buf['offset']}",
                }
                lowered_op["outputs"]["k"] = {
                    "buffer": "k_scratch",
                    "activation_offset": k_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {k_buf['offset']}",
                }
            if pos_buf:
                lowered_op["activations"]["positions"] = {
                    "buffer": "vision_positions",
                    "activation_offset": pos_buf["offset"],
                    "dtype": "i32",
                    "ptr_expr": f"activations + {pos_buf['offset']}",
                }
            last_output_buffer = "q_scratch"
        elif op_type == "spatial_merge":
            input_buf_name = last_output_buffer or current_input_buffer
            declared_output = str((ir_op.get("params") or {}).get("output_buffer", "") or "").strip()
            output_buf_name = declared_output or (
                "layer_input" if input_buf_name == "embedded_input" else "embedded_input"
            )
            src_buf = activation_buffers.get(input_buf_name)
            dst_buf = activation_buffers.get(output_buf_name)
            if src_buf:
                lowered_op["activations"]["input"] = {
                    "buffer": input_buf_name,
                    "activation_offset": src_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {src_buf['offset']}",
                }
            if dst_buf:
                lowered_op["outputs"]["output"] = {
                    "buffer": output_buf_name,
                    "activation_offset": dst_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {dst_buf['offset']}",
                }
                last_output_buffer = output_buf_name
                current_input_buffer = output_buf_name
                current_output_buffer = "embedded_input" if output_buf_name == "layer_input" else "layer_input"
        elif op_type == "projector_prep":
            input_buf_name = last_output_buffer or current_input_buffer
            output_buf_name = "layer_input" if input_buf_name == "embedded_input" else "embedded_input"
            src_buf = activation_buffers.get(input_buf_name)
            dst_buf = activation_buffers.get(output_buf_name)
            if src_buf:
                lowered_op["activations"]["input"] = {
                    "buffer": input_buf_name,
                    "activation_offset": src_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {src_buf['offset']}",
                }
            if dst_buf:
                lowered_op["outputs"]["output"] = {
                    "buffer": output_buf_name,
                    "activation_offset": dst_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {dst_buf['offset']}",
                }
                last_output_buffer = output_buf_name
                current_input_buffer = output_buf_name
                current_output_buffer = "embedded_input" if output_buf_name == "layer_input" else "layer_input"
        elif op_type == "projector_fc1":
            input_buf_name = last_output_buffer or "embedded_input"
            src_buf = activation_buffers.get(input_buf_name)
            dst_buf = activation_buffers.get("mlp_scratch")
            if src_buf:
                lowered_op["activations"]["A"] = {
                    "buffer": input_buf_name,
                    "activation_offset": src_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {src_buf['offset']}",
                }
            if dst_buf:
                lowered_op["outputs"]["C"] = {
                    "buffer": "mlp_scratch",
                    "activation_offset": dst_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {dst_buf['offset']}",
                }
                last_output_buffer = "mlp_scratch"
        elif op_type == "projector_gelu":
            mlp_buf = activation_buffers.get("mlp_scratch")
            if mlp_buf:
                lowered_op["activations"]["x"] = {
                    "buffer": "mlp_scratch",
                    "activation_offset": mlp_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {mlp_buf['offset']}",
                }
                lowered_op["outputs"]["out"] = {
                    "buffer": "mlp_scratch",
                    "activation_offset": mlp_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {mlp_buf['offset']}",
                }
                last_output_buffer = "mlp_scratch"
        elif op_type == "projector_fc2":
            kernel_id = str(ir_op.get("kernel", "") or "")
            if kernel_needs_q8_activation(registry, kernel_id):
                raise RuntimeError(
                    "projector_fc2 selected a q8-activation kernel without an explicit quantize stage"
                )
            single_linear_projector = bool(config.get("single_linear_projector") or ir_op.get("params", {}).get("single_linear_projector"))
            src_buf_name = (last_output_buffer or "embedded_input") if single_linear_projector else "mlp_scratch"
            src_buf = activation_buffers.get(src_buf_name)
            dst_buf_name = "vision_output" if "vision_output" in activation_buffers else "embedded_input"
            dst_buf = activation_buffers.get(dst_buf_name)
            if src_buf:
                lowered_op["activations"]["A"] = {
                    "buffer": src_buf_name,
                    "activation_offset": src_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {src_buf['offset']}",
                }
            if dst_buf:
                lowered_op["outputs"]["C"] = {
                    "buffer": dst_buf_name,
                    "activation_offset": dst_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {dst_buf['offset']}",
                }
                last_output_buffer = dst_buf_name
        elif op_type == "assistant_pre_projection":
            src_buf = activation_buffers.get("backbone_stream")
            dst_buf = activation_buffers.get("embedded_input")
            if src_buf:
                lowered_op["activations"]["A"] = {
                    "buffer": "backbone_stream",
                    "activation_offset": src_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {src_buf['offset']}",
                }
            if dst_buf:
                lowered_op["outputs"]["C"] = {
                    "buffer": "embedded_input",
                    "activation_offset": dst_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {dst_buf['offset']}",
                }
                last_output_buffer = "embedded_input"
                current_input_buffer = "embedded_input"
                current_output_buffer = "layer_input"
        elif op_type == "assistant_post_projection":
            src_buf = activation_buffers.get("embedded_input")
            dst_buf = activation_buffers.get("backbone_stream")
            if src_buf:
                lowered_op["activations"]["A"] = {
                    "buffer": "embedded_input",
                    "activation_offset": src_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {src_buf['offset']}",
                }
            if dst_buf:
                lowered_op["outputs"]["C"] = {
                    "buffer": "backbone_stream",
                    "activation_offset": dst_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {dst_buf['offset']}",
                }
                last_output_buffer = "embedded_input"
        elif op_type == "assistant_layer_scale":
            stream_buf = activation_buffers.get("embedded_input")
            if stream_buf:
                lowered_op["activations"]["hidden"] = {
                    "buffer": "embedded_input",
                    "activation_offset": stream_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {stream_buf['offset']}",
                }
                lowered_op["outputs"]["hidden"] = {
                    "buffer": "embedded_input",
                    "activation_offset": stream_buf["offset"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {stream_buf['offset']}",
                }
                last_output_buffer = "embedded_input"
        elif op_type == "logits":
            # Footer logits projection: input buffer must match kernel activation dtype.
            # - fp32 activation kernels (gemv_q8_0 / gemm_nt_q8_0) read embedded_input
            # - q8 activation kernels (gemv_q8_0_q8_0 / gemm_nt_q8_0_q8_0) read layer_input
            op_id = ir_op.get("idx", ir_op.get("op_id", -1))
            kernel_id = ir_op.get("kernel", "")
            needs_q8_input = kernel_needs_q8_activation(registry, kernel_id)
            default_buf_name = "layer_input" if needs_q8_input else "embedded_input"
            default_buf = activation_buffers.get(default_buf_name)

            for input_name, input_info in ir_op.get("inputs", {}).items():
                # Skip weight-style kernel params (B, bias, etc.)
                is_weight_input = input_name in ir_op.get("weights", {})
                if not is_weight_input:
                    for wkey in ir_op.get("weights", {}).keys():
                        if WEIGHT_TO_KERNEL_INPUT.get(wkey) == input_name:
                            is_weight_input = True
                            break
                if is_weight_input:
                    continue

                if not needs_q8_input:
                    # FP32 kernel: force FP32 stream to avoid stale Q8 buffer.
                    buf_name = default_buf_name
                    buf = default_buf
                else:
                    # Q8 kernel: planner assignment is valid.
                    dataflow_name = {"A": "x", "x_q8": "x", "x": "x", "input": "x"}.get(input_name, input_name)
                    planned = get_planned_buffer(op_id, "inputs", dataflow_name)
                    if not planned:
                        planned = get_planned_buffer(op_id, "inputs", input_name)
                    if planned:
                        planner_buf = planned.get("buffer", default_buf_name)
                        declared_slot = _get_declared_dataflow_slot(ir_op, "inputs", dataflow_name, input_name)
                        buf_name = _resolve_logical_buffer_name(
                            planner_buf,
                            declared_slot or input_info.get("slot"),
                            activation_buffers,
                            buffer_name_map,
                        )
                        buf = activation_buffers.get(buf_name)
                    else:
                        buf_name = default_buf_name
                        buf = default_buf

                if buf:
                    act_dtype = "q8_0" if needs_q8_input else "fp32"
                    lowered_op["activations"][input_name] = {
                        "buffer": buf_name,
                        "activation_offset": buf["offset"],
                        "dtype": input_info.get("dtype", act_dtype),
                        "ptr_expr": f"activations + {buf['offset']}",
                    }

            logits_buf = activation_buffers.get("logits")
            if logits_buf:
                for output_name, output_info in ir_op.get("outputs", {}).items():
                    lowered_op["outputs"][output_name] = {
                        "buffer": "logits",
                        "activation_offset": logits_buf["offset"],
                        "dtype": output_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {logits_buf['offset']}",
                    }
                last_output_buffer = "logits"
        else:
            # Process activation inputs - add concrete buffer offsets
            # For header ops (layer=-1), inputs may be in dataflow instead of top-level
            op_inputs = ir_op.get("inputs", {})
            using_dataflow_inputs = False
            if not op_inputs:
                # Fallback to dataflow inputs for header ops
                dataflow = ir_op.get("dataflow", {})
                op_inputs = dataflow.get("inputs", {})
                using_dataflow_inputs = True
            for input_name, input_info in op_inputs.items():
                input_type = str(input_info.get("type", ""))

                # Type-directed fast path: when IR already specifies scratch/KV-cache
                # buffers, preserve that exact contract instead of planner fallback.
                if input_type in ("scratch", "kv_cache"):
                    src_name = input_info.get("source") or input_info.get("buffer")
                    if isinstance(src_name, str) and src_name:
                        buf_name = buffer_name_map.get(src_name, src_name)
                        buf = activation_buffers.get(buf_name)
                        if buf:
                            lowered_op["activations"][input_name] = {
                                "buffer": buf_name,
                                "activation_offset": buf["offset"],
                                "dtype": input_info.get("dtype", "fp32"),
                                "ptr_expr": f"activations + {buf['offset']}",
                            }
                            continue

                if ir_op.get("op") == "gemma4_per_layer_prepare" and input_name in ("tokens", "token_ids"):
                    buf = activation_buffers.get("token_ids")
                    if buf:
                        lowered_op["activations"][input_name] = {
                            "buffer": "token_ids",
                            "activation_offset": buf["offset"],
                            "dtype": "i32",
                            "ptr_expr": f"activations + {buf['offset']}",
                        }
                        continue

                declared_slot = input_info.get("slot")
                declared_from = input_info.get("from")
                if declared_slot == "external:token_ids" or declared_from == "external:token_ids":
                    buf = activation_buffers.get("token_ids")
                    if buf:
                        lowered_op["activations"][input_name] = {
                            "buffer": "token_ids",
                            "activation_offset": buf["offset"],
                            "dtype": "i32",
                            "ptr_expr": f"activations + {buf['offset']}",
                        }
                        continue

                # Special case: embedding operation reads from token_ids, not layer_input
                if "embedding" in ir_op.get("kernel", "").lower() and "token" in input_name.lower():
                    buf = activation_buffers.get("token_ids")
                    if buf:
                        lowered_op["activations"][input_name] = {
                            "buffer": "token_ids",
                            "activation_offset": buf["offset"],
                            "dtype": "i32",
                            "ptr_expr": f"activations + {buf['offset']}",
                        }
                        continue

                # Skip inputs that are actually weight parameters
                # Check both direct match and mapped match via WEIGHT_TO_KERNEL_INPUT
                # e.g., gamma maps to ln1_gamma/ln2_gamma, W maps to wq/wk/wv
                is_weight_input = input_name in ir_op.get("weights", {})
                if not is_weight_input:
                    # Check if any weight key maps to this input name
                    for wkey in ir_op.get("weights", {}).keys():
                        if WEIGHT_TO_KERNEL_INPUT.get(wkey) == input_name:
                            is_weight_input = True
                            break
                if is_weight_input:
                    continue  # Weight is handled via weights dict

                # ═══════════════════════════════════════════════════════════════
                # USE MEMORY PLANNER for buffer assignment
                # ═══════════════════════════════════════════════════════════════
                op_id = ir_op.get("idx", ir_op.get("op_id", -1))

                # Map from kernel input names to dataflow names
                # Kernel maps use: A (input), B (weight), C (output)
                # Dataflow uses: x (input), y (output)
                #
                # IMPORTANT: kernel I/O names MUST map to dataflow names.
                # If a kernel adds new names (e.g., out_token, k_cache, v_cache),
                # update this map or memory planner will silently fall back
                # to main stream buffers (embedded_input/layer_input).
                # This caused a silent correctness bug where attention decode
                # outputs were written to embedded_input instead of attn_scratch.
                kernel_to_dataflow_input = {
                    "A": "x",      # Matrix input for gemm/gemv
                    "x": "x",      # Direct match
                    "input": "x",  # Alternative name
                    "a": "a",      # residual_add input a
                    "b": "b",      # residual_add input b
                    "src": "src",  # memcpy source
                    "gate": "x",   # swiglu gate input -> reads from mlp_scratch
                    "up": "x",     # swiglu up input -> reads from mlp_scratch
                    # Attention decode/prefill kernel names -> dataflow names
                    "q_token": "q",
                    "k_cache": "k",
                    "v_cache": "v",
                    "kv_a_packed": "k_pe",
                }
                dataflow_name = _resolve_planner_io_name(
                    input_name,
                    using_dataflow_inputs,
                    ir_op,
                    "inputs",
                    kernel_to_dataflow_input,
                )
                planned = get_planned_buffer(op_id, "inputs", dataflow_name)
                # Also try the original name if mapping didn't find it
                if not planned:
                    planned = get_planned_buffer(op_id, "inputs", input_name)

                if op_type == "branch_concat" and input_name == "main_input" and last_output_buffer in activation_buffers:
                    # branch_concat's main side is an explicit producer edge
                    # (usually projector_fc2.out).  The logical slot is still
                    # main_stream, but resolving that slot would incorrectly
                    # reopen the pre-projector embedded_input buffer.
                    buf_name = last_output_buffer
                    buf = activation_buffers.get(buf_name)
                elif planned:
                    # Use memory planner's assignment
                    planner_buf = planned.get("buffer", "embedded_input")
                    declared_slot = _get_declared_dataflow_slot(ir_op, "inputs", dataflow_name, input_name)
                    buf_name = _resolve_logical_buffer_name(
                        planner_buf,
                        declared_slot or input_info.get("slot"),
                        activation_buffers,
                        buffer_name_map,
                    )
                    buf = activation_buffers.get(buf_name)
                else:
                    # Fallback to legacy logic for unplanned ops
                    if input_name == "attn_out":
                        buf = activation_buffers.get("attn_scratch")
                        buf_name = "attn_scratch"
                    elif input_name == "scratch":
                        buf = activation_buffers.get("mlp_scratch")
                        buf_name = "mlp_scratch"
                    else:
                        buf = activation_buffers.get(current_input_buffer)
                        buf_name = current_input_buffer

                if buf:
                    lowered_op["activations"][input_name] = {
                        "buffer": buf_name,
                        "activation_offset": buf["offset"],
                        "dtype": input_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {buf['offset']}",
                    }

            # Process outputs - add concrete offsets (for non-QKV ops)
            # For header ops (layer=-1), outputs may be in dataflow instead of top-level
            op_outputs = ir_op.get("outputs", {})
            using_dataflow_outputs = False
            if not op_outputs:
                # Fallback to dataflow outputs for header ops
                dataflow = ir_op.get("dataflow", {})
                op_outputs = dataflow.get("outputs", {})
                using_dataflow_outputs = True
            for output_name, output_info in op_outputs.items():
                output_type = str(output_info.get("type", ""))

                # Type-directed fast path: preserve explicit scratch/KV-cache targets.
                if output_type in ("scratch", "kv_cache"):
                    dst_name = output_info.get("buffer") or output_info.get("source")
                    if isinstance(dst_name, str) and dst_name:
                        output_buf_name = buffer_name_map.get(dst_name, dst_name)
                        buf = activation_buffers.get(output_buf_name)
                        if buf:
                            lowered_op["outputs"][output_name] = {
                                "buffer": output_buf_name,
                                "activation_offset": buf["offset"],
                                "dtype": output_info.get("dtype", "fp32"),
                                "ptr_expr": f"activations + {buf['offset']}",
                            }
                            if not last_output_buffer:
                                last_output_buffer = output_buf_name
                            continue

                # ═══════════════════════════════════════════════════════════════
                # USE MEMORY PLANNER for output buffer assignment
                # ═══════════════════════════════════════════════════════════════
                op_id = ir_op.get("idx", ir_op.get("op_id", -1))

                # Map from kernel output names to dataflow names
                # IMPORTANT: Same rules as input mapping - all kernel output names
                # must be mapped here or fall back to wrong buffer.
                kernel_to_dataflow_output = {
                    "C": "y",       # Matrix output for gemm/gemv
                    "y": "y",       # Direct match
                    "x": "x",       # In-place stream updates (bias/pos add, etc.)
                    "output": "output",  # Quantize output
                    "dst": "dst",   # memcpy destination
                    # Attention decode output name -> dataflow output
                    "out_token": "out",
                    "out": "out",
                }
                dataflow_name = _resolve_planner_io_name(
                    output_name,
                    using_dataflow_outputs,
                    ir_op,
                    "outputs",
                    kernel_to_dataflow_output,
                )
                planned = get_planned_buffer(op_id, "outputs", dataflow_name)
                # Also try the original name if mapping didn't find it
                if not planned:
                    planned = get_planned_buffer(op_id, "outputs", output_name)

                if planned:
                    # Use memory planner's assignment
                    planner_buf = planned.get("buffer", "embedded_input")
                    declared_slot = _get_declared_dataflow_slot(ir_op, "outputs", dataflow_name, output_name)
                    output_buf_name = _resolve_logical_buffer_name(
                        planner_buf,
                        declared_slot or output_info.get("slot"),
                        activation_buffers,
                        buffer_name_map,
                    )
                else:
                    # Fallback to legacy logic for unplanned ops
                    if "embedding" in ir_op.get("kernel", "").lower():
                        output_buf_name = "embedded_input"
                    elif ir_op.get("op") in ("attn", "attn_sliding", "attn_shared_kv", "attn_sliding_shared_kv"):
                        output_buf_name = "attn_scratch"
                    elif ir_op.get("op") == "logits":
                        output_buf_name = "logits"
                    elif ir_op.get("op") in ("mlp_gate_up", "silu_mul"):
                        output_buf_name = "mlp_scratch"
                    else:
                        output_buf_name = current_output_buffer

                buf = activation_buffers.get(output_buf_name)
                if buf:
                    activation_offset = buf["offset"]
                    if op_type == "branch_fc2" and output_buf_name == "branch_collect":
                        activation_offset += int(ir_op.get("params", {}).get("branch_collect_offset_bytes", 0) or 0)
                    lowered_op["outputs"][output_name] = {
                        "buffer": output_buf_name,
                        "activation_offset": activation_offset,
                        "dtype": output_info.get("dtype", "fp32"),
                        "ptr_expr": f"activations + {activation_offset}",
                    }
                    if not last_output_buffer:
                        last_output_buffer = output_buf_name
            if lowered_op["outputs"]:
                last_output_buffer = next(iter(lowered_op["outputs"].values())).get("buffer", last_output_buffer)

        # Process scratch buffers
        lowered_op["scratch"] = []
        scratch_list = ir_op.get("scratch", [])
        if scratch_list:
            mlp_buf = activation_buffers.get("mlp_scratch")
            for i, scratch in enumerate(scratch_list):
                scratch_offset = mlp_buf["offset"] if mlp_buf else 0
                lowered_op["scratch"].append({
                    "name": scratch.get("name", f"scratch_{i}"),
                    "scratch_offset": scratch_offset,
                    "size": scratch.get("size", "dynamic"),
                    "dtype": scratch.get("dtype", "fp32"),
                    "ptr_expr": f"activations + {scratch_offset}",
                })

        # Special handling for QK/Q-only norm: operate in-place on scratch buffers
        # between projection and RoPE.
        if ir_op.get("op", "") in ("qk_norm", "q_norm"):
            scratch_names = ["q_scratch", "k_scratch"] if ir_op.get("op", "") == "qk_norm" else ["q_scratch"]
            for scratch_name in scratch_names:
                buf = activation_buffers.get(scratch_name)
                if buf:
                    lowered_op["scratch"].append({
                        "name": scratch_name,
                        "scratch_offset": buf["offset"],
                        "size": buf["size"],
                        "dtype": "fp32",
                        "ptr_expr": f"activations + {buf['offset']}",
                    })

        # Special handling for RoPE: add q_scratch and k_scratch buffers
        # RoPE always uses the scratch buffers (where k_proj/v_proj just wrote)
        # in both decode and prefill modes
        if ir_op.get("op", "") in ("rope_qk", "rope_q"):
            q_buf = activation_buffers.get("q_scratch")
            if q_buf:
                lowered_op["scratch"].append({
                    "name": "q_scratch",
                    "scratch_offset": q_buf["offset"],
                    "size": q_buf["size"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {q_buf['offset']}",
                })

            if ir_op.get("op", "") == "rope_qk":
                k_buf = activation_buffers.get("k_scratch")
                if k_buf:
                    lowered_op["scratch"].append({
                        "name": "k_scratch",
                        "scratch_offset": k_buf["offset"],
                        "size": k_buf["size"],
                        "dtype": "fp32",
                        "ptr_expr": f"activations + {k_buf['offset']}",
                    })
        if _is_vision_mrope_operation(ir_op):
            for scratch_name in ["q_scratch", "k_scratch"]:
                buf = activation_buffers.get(scratch_name)
                if buf:
                    lowered_op["scratch"].append({
                        "name": scratch_name,
                        "scratch_offset": buf["offset"],
                        "size": buf["size"],
                        "dtype": "fp32",
                        "ptr_expr": f"activations + {buf['offset']}",
                    })

        # Special handling for kv_cache_store: add k_scratch and v_scratch buffers
        if ir_op.get("op", "") in ("kv_cache_store", "mla_kv_cache_store", "mla_kv_cache_batch_store"):
            for scratch_name in ["k_scratch", "v_scratch"]:
                buf = activation_buffers.get(scratch_name)
                if buf:
                    lowered_op["scratch"].append({
                        "name": scratch_name,
                        "scratch_offset": buf["offset"],
                        "size": buf["size"],
                        "dtype": "fp32",
                        "ptr_expr": f"activations + {buf['offset']}",
                    })

        if ir_op.get("op", "") == "kv_cache_store_shared_q":
            q_buf = activation_buffers.get("q_scratch")
            if q_buf:
                lowered_op["scratch"].append({
                    "name": "q_scratch",
                    "scratch_offset": q_buf["offset"],
                    "size": q_buf["size"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {q_buf['offset']}",
                })

        # Special handling for attention: add q_scratch, k_scratch, v_scratch buffers
        # Note: op type is "attn" but kernel contains "attention"
        if ir_op.get("op", "") == "attn" or "attention" in ir_op.get("kernel", ""):
            for scratch_name in ["q_scratch", "k_scratch", "v_scratch"]:
                # For DECODE mode, use KV cache offsets for K and V (they're read from cache)
                # For PREFILL mode, use scratch buffers (K/V are computed fresh each time)
                if mode == "decode" and scratch_name in ("k_scratch", "v_scratch"):
                    layer_idx = int(ir_op.get("_kv_cache_read_layer", ir_op.get("layer", 0)))
                    kv_offs = kv_layer_offsets(layer_idx)
                    if kv_offs:
                        k_off, v_off = kv_offs
                        off = k_off if scratch_name == "k_scratch" else v_off
                        lowered_op["scratch"].append({
                            "name": scratch_name,
                            "scratch_offset": off,
                            "size": activation_buffers.get(scratch_name, {}).get("size", 0),
                            "dtype": "fp32",
                            "ptr_expr": f"activations + {off}",
                            "force_offset": True,
                        })
                        continue
                buf = activation_buffers.get(scratch_name)
                if buf:
                    lowered_op["scratch"].append({
                        "name": scratch_name,
                        "scratch_offset": buf["offset"],
                        "size": buf["size"],
                        "dtype": "fp32",
                        "ptr_expr": f"activations + {buf['offset']}",
                    })

        # Special handling for residual_add: add residual buffer for the saved input
        if ir_op.get("op", "") == "residual_add":
            buf = activation_buffers.get("residual")
            if buf:
                lowered_op["scratch"].append({
                    "name": "residual",
                    "scratch_offset": buf["offset"],
                    "size": buf["size"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {buf['offset']}",
                })

        # Special handling for GeGLU: ensure scratch buffer is allocated
        # GeGLU BF16 variant requires 3 * tokens * dim scratch for input + output
        if ir_op.get("op", "") == "geglu":
            # Use mlp_scratch which is already allocated for MLP operations
            mlp_buf = activation_buffers.get("mlp_scratch")
            if mlp_buf:
                lowered_op["scratch"].append({
                    "name": "geglu_scratch",
                    "scratch_offset": mlp_buf["offset"],
                    "size": mlp_buf["size"],
                    "dtype": "fp32",
                    "ptr_expr": f"activations + {mlp_buf['offset']}",
                })

        _apply_layer_scoped_recurrent_state_offsets(lowered_op, config)

        # Add model config parameters (merge with any op-specific params)
        params = dict(ir_op.get("params", {}) or {})
        params.setdefault("embed_dim", config.get("embed_dim", 896))
        params.setdefault("num_heads", config.get("num_heads", 14))
        params.setdefault("num_kv_heads", config.get("num_kv_heads", 2))
        params.setdefault("head_dim", config.get("head_dim", 64))
        params.setdefault("rotary_dim", config.get("rotary_dim", params.get("head_dim", 64)))
        params.setdefault("intermediate_size", config.get("intermediate_size", config.get("intermediate_dim", 4864)))
        params.setdefault("num_layers", config.get("num_layers", 24))
        params.setdefault("mode", mode)

        # Add sliding_window for attention_sliding operations
        if op_type == "attn_sliding":
            sliding_window = config.get("sliding_window", 0)
            params.setdefault("sliding_window", sliding_window)
        if op_type in ("patchify", "patch_proj"):
            params.setdefault("image_size", config.get("image_size", 0))
            params.setdefault("image_height", config.get("image_height", config.get("image_size", 0)))
            params.setdefault("image_width", config.get("image_width", config.get("image_size", 0)))
            params.setdefault("patch_size", config.get("patch_size", 0))
            params.setdefault("vision_channels", config.get("vision_channels", 3))
            params.setdefault("vision_num_patches", config.get("vision_num_patches", 0))
            params.setdefault("patch_dim", config.get("patch_dim", 0))
        if op_type in (
            "vision_position_ids",
            "position_ids_2d",
            "add_stream",
            "patch_proj_aux",
            "position_embeddings",
            "patch_bias_add",
            "spatial_merge",
            "projector_prep",
            "branch_spatial_merge",
            "branch_layernorm",
            "mrope_qk",
            "projector_fc1",
            "projector_gelu",
            "projector_prep",
            "projector_fc2",
            "branch_fc1",
            "branch_gelu",
            "branch_fc2",
            "branch_concat",
            "gelu",
        ):
            params.setdefault("vision_num_patches", config.get("vision_num_patches", 0))
            params.setdefault("vision_grid_h", config.get("vision_grid_h", 0))
            params.setdefault("vision_grid_w", config.get("vision_grid_w", 0))
            spatial_merge_factor = config.get("spatial_merge_factor")
            if spatial_merge_factor is not None:
                params.setdefault("spatial_merge_factor", spatial_merge_factor)
            params.setdefault("vision_merged_tokens", config.get("vision_merged_tokens", config.get("vision_num_patches", 0)))
            params.setdefault("projector_in_dim", config.get("projector_in_dim", 0))
            params.setdefault("projector_hidden_dim", config.get("projector_hidden_dim", 0))
            params.setdefault("projector_out_dim", config.get("projector_out_dim", 0))
            params.setdefault("projector_total_out_dim", config.get("projector_total_out_dim", config.get("projector_out_dim", 0)))
            params.setdefault("num_deepstack_layers", config.get("num_deepstack_layers", 0))
        if op_type == "branch_layernorm":
            params.setdefault("vision_merged_tokens", config.get("vision_merged_tokens", config.get("vision_num_patches", 0)))
            params.setdefault("projector_in_dim", config.get("projector_in_dim", 0))
        if op_type == "add_stream":
            stream_rows = int(config.get("vision_num_patches", params.get("seq_len", 0)) or 0)
            stream_dim = int(config.get("embed_dim", 0) or 0)
            params.setdefault("stream_elems", stream_rows * stream_dim)
            params["merge_size"] = _required_template_int_param(params, "merge_size", config, op_type)
        if op_type == "patch_bias_add":
            params.setdefault("rows", int(config.get("vision_num_patches", params.get("seq_len", 0)) or 0))
            params.setdefault("dim", int(config.get("embed_dim", 0) or 0))
        if op_type in ("vision_position_ids", "position_ids_2d"):
            params.setdefault("rows", int(config.get("vision_num_patches", 0) or 0))
            params["merge_size"] = _required_template_int_param(params, "merge_size", config, op_type)
        if op_type == "position_embeddings":
            params["merge_size"] = _required_template_int_param(params, "merge_size", config, op_type)
            source_grid_size = int(config.get("position_grid_size", 0) or 0)
            if source_grid_size <= 0:
                source_image_size = int(config.get("image_size", 0) or 0)
                patch_size = int(config.get("patch_size", 0) or 0)
                if source_image_size > 0 and patch_size > 0:
                    source_grid_size = source_image_size // patch_size
            if source_grid_size > 0:
                params.setdefault("source_grid_size", source_grid_size)
        if op_type in ("spatial_merge", "branch_spatial_merge"):
            params["merge_size"] = _required_template_int_param(params, "merge_size", config, op_type)
        if op_type == "projector_prep":
            params.setdefault("rows", int(config.get("vision_merged_tokens", params.get("seq_len", 0)) or 0))
            params.setdefault("dim", int(config.get("projector_in_dim", config.get("embed_dim", 0)) or 0))
            params.setdefault("scale", float(config.get("gemma4_vision_pool_scale", 1.0) or 1.0))
            params.setdefault("eps", float(config.get("gemma4_vision_projector_eps", config.get("rms_eps", 1.0e-6)) or 1.0e-6))
        if op_type in ("projector_gelu", "branch_gelu"):
            merged_tokens = int(params.get("vision_merged_tokens", 0) or 0)
            hidden_dim = int(params.get("projector_hidden_dim", 0) or 0)
            params.setdefault("gelu_elems", merged_tokens * hidden_dim)
        if op_type in ("quantize_input_0", "quantize_input_1", "quantize_input_2", "quantize_out_proj_input", "quantize_mlp_down_input", "quantize_recurrent_out_proj_input", "quantize_mamba_out_proj_input", "quantize_final_output"):
            default_quant_rows = int(params.get("_m", params.get("seq_len", 1)) or 1)
            params.setdefault("rows", default_quant_rows)
        if op_type == "quantize_final_output":
            if int(config.get("projector_in_dim", 0) or 0) > 0:
                params["rows"] = int(params.get("vision_merged_tokens", config.get("vision_merged_tokens", params.get("rows", 1))) or 1)
        if op_type == "branch_concat":
            params.setdefault(
                "rows",
                _template_int_param(params, "rows", config, int(params.get("vision_merged_tokens", 0) or 0)),
            )
            params.setdefault(
                "main_dim",
                _template_int_param(params, "main_dim", config, int(params.get("projector_out_dim", 0) or 0)),
            )
            params.setdefault(
                "branch_slice_dim",
                _template_int_param(params, "branch_slice_dim", config, int(params.get("projector_out_dim", 0) or 0)),
            )
            params.setdefault(
                "num_branch_slices",
                _template_int_param(params, "num_branch_slices", config, int(params.get("num_deepstack_layers", 0) or 0)),
            )
        if op_type == "rope_qk" and ir_op.get("kernel", "") == "rope_forward_qk_gemma4v_vision_xy":
            params.setdefault("vision_grid_w", int(config.get("vision_grid_w", 0) or config.get("image_grid_w", 0) or 0))
            if int(params.get("vision_grid_w", 0) or 0) <= 0:
                params["vision_grid_w"] = int(max(1, round(float(config.get("vision_num_patches", 1) or 1) ** 0.5)))
            params.setdefault("rotary_dim", int(config.get("rotary_dim", params.get("head_dim", 0)) or params.get("head_dim", 0) or 0))
            gemma4v_freq_base = float(config.get("vision_rope_theta", 100.0) or 100.0)
            params["freq_base"] = gemma4v_freq_base
            params["rope_freq_base"] = gemma4v_freq_base
        if _is_vision_mrope_operation(ir_op):
            sections = config.get("vision_mrope_sections")
            if not isinstance(sections, list) or len(sections) != 4:
                default_section = max(1, int(params.get("head_dim", 0) or 0) // 4)
                sections = [default_section, default_section, 0, 0]
            params.setdefault(
                "n_dims",
                int(config.get("vision_mrope_n_dims", max(1, int(params.get("head_dim", 0) or 0)))),
            )
            rotary_width = int(params["n_dims"])
            head_dim = int(params.get("head_dim", 0) or 0)
            if rotary_width <= 0 or (rotary_width & 1) != 0 or rotary_width > head_dim:
                raise ValueError(
                    "vision M-RoPE rotary width must be positive, even, and no larger than head_dim: "
                    f"n_dims={rotary_width} head_dim={head_dim}"
                )
            rope_pairs = rotary_width // 2
            if int(sections[0]) + int(sections[1]) > rope_pairs:
                raise ValueError(
                    "vision M-RoPE sections exceed available frequency pairs: "
                    f"sections={sections} n_dims={params['n_dims']} head_dim={params.get('head_dim')}"
                )
            params.setdefault("section_0", int(sections[0]))
            params.setdefault("section_1", int(sections[1]))
            params.setdefault("section_2", int(sections[2]))
            params.setdefault("section_3", int(sections[3]))
            params.setdefault("freq_base", float(config.get("vision_mrope_freq_base", 10000.0)))
            params.setdefault("freq_scale", float(config.get("vision_mrope_freq_scale", 1.0)))
            params.setdefault("ext_factor", float(config.get("vision_mrope_ext_factor", 0.0)))
            params.setdefault("attn_factor", float(config.get("vision_mrope_attn_factor", 1.0)))
            params.setdefault("beta_fast", float(config.get("vision_mrope_beta_fast", 32.0)))
            params.setdefault("beta_slow", float(config.get("vision_mrope_beta_slow", 1.0)))
            params.setdefault("n_ctx_orig", int(config.get("vision_mrope_original_context_length", 32768)))
        if op_type == "rope_qk" and _is_text_mrope_operation(ir_op):
            sections = config.get("mrope_sections")
            if not isinstance(sections, list) or len(sections) != 4:
                default_section = max(1, int(params.get("head_dim", 0) or 0) // 4)
                sections = [default_section, default_section, default_section, default_section]
            # M-RoPE sections choose the axis pattern; they are not the rotary
            # width. For Qwen3-VL/Qwen3.5, llama.cpp passes n_rot as the full
            # head width. Use the op/config head dim as the fallback so stale
            # configs cannot silently rotate only sum(sections) dimensions.
            default_n_dims = max(1, int(params.get("head_dim", config.get("head_dim", 0)) or 0))
            params.setdefault("n_dims", int(config.get("mrope_n_dims", default_n_dims)))
            params.setdefault("section_0", int(sections[0]))
            params.setdefault("section_1", int(sections[1]))
            params.setdefault("section_2", int(sections[2]))
            params.setdefault("section_3", int(sections[3]))
            params.setdefault("freq_base", float(config.get("rope_theta", 10000.0)))
            params.setdefault(
                "freq_scale",
                float(config.get("rope_scaling_factor", 1.0)) if str(config.get("rope_scaling_type", "none")).strip().lower() != "none" else 1.0,
            )
            params.setdefault("ext_factor", float(config.get("rope_ext_factor", 0.0)))
            params.setdefault("attn_factor", float(config.get("rope_attn_factor", 1.0)))
            params.setdefault("beta_fast", float(config.get("rope_beta_fast", 32.0)))
            params.setdefault("beta_slow", float(config.get("rope_beta_slow", 1.0)))
            params.setdefault("n_ctx_orig", int(config.get("rope_original_context_length", config.get("context_length", 32768))))

        if mode == "decode":
            effective_seq_len = 1
        else:
            # Prefill must follow the effective runtime context length (e.g. --context-len),
            # not the model's training max_seq_len (often 32768+), otherwise kernels
            # run with massively inflated token counts and diverge/slow down.
            effective_seq_len = int(config.get("context_length", config.get("max_seq_len", 2048)))
        # Override stale seq_len injected earlier in the pipeline (IR1 may still carry
        # model max_seq_len). Lowered IR must always reflect runtime-effective length.
        params["seq_len"] = effective_seq_len
        if op_type == "branch_layernorm":
            params["seq_len"] = int(params.get("vision_merged_tokens", params["seq_len"]) or params["seq_len"])
            params["embed_dim"] = int(params.get("projector_in_dim", params.get("embed_dim", 0)) or params.get("embed_dim", 0))

        # Add matmul dims for IR Lower 3 bindings (_input_dim/_output_dim/_m)
        out_dim, in_dim = compute_matmul_dims(op_type, config)
        if out_dim is not None and "_output_dim" not in params:
            params["_output_dim"] = out_dim
        if in_dim is not None and "_input_dim" not in params:
            params["_input_dim"] = in_dim
        apply_layer_attention_dims(op_type, params, int(lowered_op.get("layer", -1)), config)

        if op_type == "bias_add" and "_output_dim" not in params:
            for w in lowered_op.get("weights", {}).values():
                size = int(w.get("size", 0))
                if size > 0 and size % 4 == 0:
                    params["_output_dim"] = size // 4
                    break

        # Keep _m aligned with effective seq_len for token-major kernels.
        params["_m"] = params.get("seq_len", 1)
        if op_type in ("patch_proj", "patch_proj_aux"):
            params["_m"] = int(params.get("vision_num_patches", params.get("_m", 1)) or 1)
        if op_type in (
            "spatial_merge",
            "branch_spatial_merge",
            "projector_fc1",
            "projector_gelu",
            "projector_fc2",
            "branch_fc1",
            "branch_gelu",
            "branch_fc2",
            "branch_concat",
            "quantize_final_output",
        ) or op_type == "branch_layernorm":
            params["_m"] = int(params.get("vision_merged_tokens", params.get("_m", 1)) or 1)
        if op_type in ("vision_position_ids", "position_ids_2d") or _is_vision_mrope_operation(ir_op):
            params["_m"] = int(params.get("vision_num_patches", params.get("_m", 1)) or 1)
        if op_type in ("quantize_input_0", "quantize_input_1", "quantize_input_2", "quantize_out_proj_input", "quantize_mlp_down_input", "quantize_recurrent_out_proj_input", "quantize_mamba_out_proj_input", "quantize_final_output"):
            inferred_quant_rows = int(params.get("_m", params.get("seq_len", params.get("rows", 1))) or 1)
            if int(params.get("rows", 0) or 0) <= 1 and inferred_quant_rows > 1:
                params["rows"] = inferred_quant_rows
        if op_type in ("projector_gelu", "branch_gelu"):
            params["gelu_elems"] = (
                int(params.get("_m", 0) or 0)
                * int(params.get("projector_hidden_dim", 0) or 0)
            )
        if op_type == "gelu":
            params["gelu_elems"] = (
                int(params.get("_m", params.get("seq_len", 0)) or 0)
                * int(params.get("intermediate_size", 0) or 0)
            )
        if op_type == "relu2":
            params["relu2_elems"] = (
                int(params.get("_m", params.get("seq_len", 0)) or 0)
                * int(params.get("intermediate_size", 0) or 0)
            )
        lowered_op["params"] = params

        lowered_ops.append(lowered_op)

        # NOTE: residual_save ops are now explicitly in IR1 (inserted before rmsnorm)
        # The memory planner assigns buffers based on dataflow. No need to auto-insert here.

        # Ping-pong buffers for next op, UNLESS this is a Q/K/V projection
        # Q/K/V all read from the same input (RMSNorm output), so skip ping-pong for K/V
        op_type = ir_op.get("op", "")
        kernel_type = ir_op.get("kernel", "")

        if "embedding" in kernel_type.lower():
            # Embedding: reads from token_ids, outputs to embedded_input
            # Next op (RMSNorm/attention) reads from embedded_input, outputs to layer_input
            current_input_buffer = "embedded_input"
            current_output_buffer = "layer_input"
        elif op_type == "patchify":
            pass
        elif op_type in ("patch_proj", "patch_proj_aux", "position_embeddings", "patch_bias_add", "vision_position_ids", "position_ids_2d", "add_stream"):
            current_input_buffer = "embedded_input"
            current_output_buffer = "layer_input"
        elif op_type in ("q_proj", "q_gate_proj", "split_q_gate", "split_qkv_packed", "attn_gate_sigmoid_mul", "k_proj", "v_proj", "qkv_proj", "qkv_packed_proj", "q_norm", "rope_qk", "rope_q", "mrope_qk",
                         "recurrent_qk_l2_norm",
                         "mlp_gate_up", "mlp_up", "silu_mul", "geglu", "gelu", "mlp_down", "projector_fc1", "projector_gelu", "projector_prep", "projector_fc2", "branch_fc1", "branch_gelu", "branch_fc2", "branch_concat", "spatial_merge", "bias_add") or \
                (ir_op.get("section", "") == "branch" and op_type == "layernorm") or \
                (mode == "prefill" and op_type in ("attn", "attn_sliding", "attn_shared_kv", "attn_sliding_shared_kv")):
            # Ops that don't advance the token-major stream, don't ping-pong
            pass
        else:
            current_input_buffer, current_output_buffer = current_output_buffer, current_input_buffer

    print(f"\n✓ IR Lower 2 complete:")
    print(f"  Lowered ops: {len(lowered_ops)}")
    print(f"  Weight entries resolved: {len(weight_offsets)}")

    lowered_ir = {
        "format": "lowered-ir-v2",
        "version": 2,
        "mode": mode,
        "config": config,
        "memory": memory,
        "operations": lowered_ops,
    }

    # ==========================================================================
    # HARD VALIDATION: Check buffer assignments for decode mode
    # This catches silent mis-routing bugs where kernel I/O names aren't mapped
    # to dataflow names, causing operations to read/write wrong buffers.
    # ==========================================================================
    if mode == "decode":
        validate_buffer_assignments(lowered_ir)

    return lowered_ir


def validate_buffer_assignments(lowered_ir: Dict) -> None:
    """
    Validate that critical operations use the correct buffers in decode mode.

    This prevents a class of silent correctness bugs where kernel I/O names
    aren't mapped to dataflow names, causing operations to read/write wrong
    buffers (e.g., attention output written to embedded_input instead of
    attn_scratch).

    Raises:
        RuntimeError: If a critical mismatch is detected.
    """
    ops = lowered_ir.get("operations", [])
    registry = load_kernel_registry()
    kernel_quant = {
        k.get("id"): k.get("quant", {})
        for k in registry.get("kernels", [])
        if k.get("id")
    }
    kernel_weight_contract = {}
    for k in registry.get("kernels", []):
        kid = k.get("id")
        if not kid:
            continue
        weights = k.get("weights", [])
        if isinstance(weights, list):
            kernel_weight_contract[kid] = {
                str(w.get("name", "")): str(w.get("dtype", "")).lower()
                for w in weights
                if isinstance(w, dict) and w.get("name")
            }

    body_projection_ops = {
        "q_proj",
        "q_gate_proj",
        "k_proj",
        "v_proj",
        "qkv_packed_proj",
        "mlp_gate_up",
        "mlp_up",
        "mamba_in_proj",
        "recurrent_gate_proj",
    }

    for op in ops:
        op_name = op.get("op", op.get("kernel", "unknown"))
        layer = op.get("layer", -1)
        kernel_id = op.get("kernel", "")

        # ===== WEIGHT/KERNEL DTYPE CONTRACT =====
        # Safetensors/BUMP BF16 weights must not silently fall through to FP32
        # matmul kernels. That produces finite output but corrupts the graph by
        # interpreting BF16 payload bytes as FP32 values.
        if kernel_id:
            kernel_weight_dtype = str(kernel_quant.get(kernel_id, {}).get("weight", "")).lower()
            per_weight_contract = kernel_weight_contract.get(kernel_id, {})
            for weight_name, weight_info in op.get("weights", {}).items():
                weight_dtype = str(weight_info.get("dtype", "")).lower()
                expected_weight_dtype = str(per_weight_contract.get(weight_name, kernel_weight_dtype)).lower()
                if weight_dtype == "bf16" and expected_weight_dtype != "bf16":
                    raise RuntimeError(
                        f"\n❌ HARD FAULT: Weight/kernel dtype mismatch\n"
                        f"   op={op_name} layer={layer} kernel={kernel_id}\n"
                        f"   weight={weight_name} dtype=bf16\n"
                        f"   kernel weight dtype: {expected_weight_dtype or '<missing>'}\n"
                        f"   Fix: ensure quant_summary aliases select BF16 kernels for safetensors/BUMP weights\n"
                    )

        # ===== BODY PROJECTION INPUT STREAM =====
        # Projections inside a transformer/recurrent block consume the normalized
        # layer stream. They must not read the raw token embedding stream.
        if op_name in body_projection_ops and str(op.get("expects_layer_input", "")).lower() in ("1", "true", "yes"):
            activations = op.get("activations", {})
            x_in = (
                activations.get("x")
                or activations.get("A")
                or activations.get("x_q8")
                or activations.get("input")
            )
            if x_in:
                x_buf = x_in.get("buffer", "")
                if x_buf != "layer_input":
                    raise RuntimeError(
                        f"\n❌ HARD FAULT: Invalid body projection input\n"
                        f"   op={op_name} layer={layer} kernel={kernel_id}\n"
                        f"   expected input: layer_input (post-norm stream)\n"
                        f"   got: {x_buf}\n"
                        f"   Fix: projection dataflow/lowering must preserve block norm output\n"
                    )

        # ===== MLP ACTIVATION SCRATCH CONTRACT =====
        if op_name in ("relu2", "silu_mul", "geglu", "gelu"):
            activations = op.get("activations", {})
            outputs = op.get("outputs", {})
            x_in = activations.get("x") or activations.get("input") or activations.get("a")
            out = outputs.get("out") or outputs.get("y")
            if x_in and x_in.get("buffer", "") != "mlp_scratch":
                raise RuntimeError(
                    f"\n❌ HARD FAULT: Invalid MLP activation input\n"
                    f"   op={op_name} layer={layer}\n"
                    f"   expected input: mlp_scratch\n"
                    f"   got: {x_in.get('buffer', '')}\n"
                )
            if out and out.get("buffer", "") != "mlp_scratch":
                raise RuntimeError(
                    f"\n❌ HARD FAULT: Invalid MLP activation output\n"
                    f"   op={op_name} layer={layer}\n"
                    f"   expected output: mlp_scratch\n"
                    f"   got: {out.get('buffer', '')}\n"
                )

        # Only enforce the scratch contract on lowered MLP-down kernels that
        # are explicitly part of the activation/MoE scratch path. Existing
        # transformer decode families can lower quantized mlp_down directly
        # from the layer stream, and the v8 family regression suite relies on
        # that contract for Qwen/Gemma/Nanbeige.
        if op_name == "mlp_down" and (
            "relu2" in str(kernel_id).lower()
            or "moe" in str(kernel_id).lower()
            or str(op.get("expects_mlp_scratch", "")).lower() in ("1", "true", "yes")
        ):
            activations = op.get("activations", {})
            x_in = activations.get("x") or activations.get("A") or activations.get("x_q8")
            if x_in and x_in.get("buffer", "") != "mlp_scratch":
                raise RuntimeError(
                    f"\n❌ HARD FAULT: Invalid MLP down input\n"
                    f"   op={op_name} layer={layer} kernel={kernel_id}\n"
                    f"   expected input: mlp_scratch\n"
                    f"   got: {x_in.get('buffer', '')}\n"
                )

        # ===== ATTENTION OPERATIONS =====
        if op_name in ("attn", "attention", "attn_sliding", "attn_shared_kv", "attn_sliding_shared_kv"):
            outputs = op.get("outputs", {})
            activations = op.get("activations", {})

            # Check output buffer: must be attn_scratch
            out_token = outputs.get("out_token") or outputs.get("out")
            if out_token:
                out_buf = out_token.get("buffer", "")
                if out_buf not in ("attn_scratch",):
                    raise RuntimeError(
                        f"\n❌ HARD FAULT: Invalid buffer assignment\n"
                        f"   op={op_name} layer={layer}\n"
                        f"   expected output buffer: attn_scratch\n"
                        f"   got: {out_buf}\n"
                        f"   Fix: Add kernel I/O -> dataflow name mapping in generate_ir_lower_2()\n"
                    )

            # Check KV cache inputs: must come from kv_cache
            k_cache = activations.get("k_cache") or activations.get("k")
            if k_cache:
                k_buf = k_cache.get("buffer", "")
                allowed_k = ("q_scratch", "kv_cache") if op_name in ("attn_shared_kv", "attn_sliding_shared_kv") else ("kv_cache",)
                if k_buf not in allowed_k:
                    raise RuntimeError(
                        f"\n❌ HARD FAULT: Invalid buffer assignment\n"
                        f"   op={op_name} layer={layer}\n"
                        f"   expected k_cache input: {'/'.join(allowed_k)}\n"
                        f"   got: {k_buf}\n"
                        f"   Fix: Add kernel I/O -> dataflow name mapping in generate_ir_lower_2()\n"
                    )

            v_cache = activations.get("v_cache") or activations.get("v")
            if v_cache:
                v_buf = v_cache.get("buffer", "")
                allowed_v = ("q_scratch", "kv_cache") if op_name in ("attn_shared_kv", "attn_sliding_shared_kv") else ("kv_cache",)
                if v_buf not in allowed_v:
                    raise RuntimeError(
                        f"\n❌ HARD FAULT: Invalid buffer assignment\n"
                        f"   op={op_name} layer={layer}\n"
                        f"   expected v_cache input: {'/'.join(allowed_v)}\n"
                        f"   got: {v_buf}\n"
                        f"   Fix: Add kernel I/O -> dataflow name mapping in generate_ir_lower_2()\n"
                    )

        # ===== ROPE QK =====
        elif op_name in ("rope_qk", "rope_q", "rope", "mrope_qk"):
            outputs = op.get("outputs", {})
            activations = op.get("activations", {})

            # Q and K must use scratch buffers, not main stream
            q_out = outputs.get("q") or outputs.get("q_out")
            if q_out:
                q_buf = q_out.get("buffer", "")
                if q_buf in ("embedded_input", "layer_input"):
                    raise RuntimeError(
                        f"\n❌ HARD FAULT: Invalid buffer assignment\n"
                        f"   op={op_name} layer={layer}\n"
                        f"   expected q output: q_scratch or attn_scratch\n"
                        f"   got: {q_buf}\n"
                        f"   Fix: Ensure q_proj/k_proj outputs are assigned scratch buffers\n"
                    )

        # ===== Q/K/V PROJECTIONS =====
        elif op_name in ("q_proj", "q_gate_proj", "k_proj", "v_proj", "qkv_proj", "split_q_gate", "attn_gate_sigmoid_mul"):
            outputs = op.get("outputs", {})
            outputs_to_check = {
                "q_proj": ["q", "q_out"],
                "q_gate_proj": ["y", "out", "C"],
                "k_proj": ["k", "k_out"],
                "v_proj": ["v", "v_out"],
                "split_q_gate": ["q", "gate"],
                "attn_gate_sigmoid_mul": ["out"],
            }
            expected_buffers = {
                "q_proj": "q_scratch",
                "q_gate_proj": "attn_q_gate_packed",
                "k_proj": "k_scratch",
                "v_proj": "v_scratch",
            }
            expected_by_output = {
                "split_q_gate": {"q": "q_scratch", "gate": "attn_gate"},
                "attn_gate_sigmoid_mul": {"out": "attn_scratch"},
            }
            expected = expected_buffers.get(op_name, "scratch")
            for out_name in outputs_to_check.get(op_name, []):
                if out_name in outputs:
                    buf = outputs[out_name].get("buffer", "")
                    expected_out = expected_by_output.get(op_name, {}).get(out_name, expected)
                    if buf not in (expected_out,):
                        raise RuntimeError(
                            f"\n❌ HARD FAULT: Invalid buffer assignment\n"
                            f"   op={op_name} layer={layer}\n"
                            f"   expected output: {expected_out}\n"
                            f"   got: {buf}\n"
                            f"   Fix: Ensure projection outputs use correct scratch buffer\n"
                        )

        # ===== LOGITS =====
        elif op_name == "logits":
            activations = op.get("activations", {})
            outputs = op.get("outputs", {})

            # Input buffer must match logits kernel activation dtype:
            # - fp32 activation kernels read embedded_input (main_stream)
            # - q8 activation kernels read layer_input (main_stream_q8)
            kernel_id = op.get("kernel", "")
            needs_q8_input = kernel_needs_q8_activation(registry, kernel_id)
            expected_input_buf = "layer_input" if needs_q8_input else "embedded_input"

            x_in = (
                activations.get("x")
                or activations.get("A")
                or activations.get("x_q8")
                or activations.get("input")
            )
            if x_in:
                x_buf = x_in.get("buffer", "")
                if x_buf != expected_input_buf:
                    raise RuntimeError(
                        f"\n❌ HARD FAULT: Invalid buffer assignment\n"
                        f"   op={op_name} layer={layer}\n"
                        f"   kernel={kernel_id}\n"
                        f"   expected logits input: {expected_input_buf}\n"
                        f"   got: {x_buf}\n"
                    )

            # Output must be logits
            logits_out = outputs.get("logits") or outputs.get("out")
            if logits_out:
                out_buf = logits_out.get("buffer", "")
                if out_buf != "logits":
                    raise RuntimeError(
                        f"\n❌ HARD FAULT: Invalid buffer assignment\n"
                        f"   op={op_name} layer={layer}\n"
                        f"   expected output: logits\n"
                        f"   got: {out_buf}\n"
                    )

        # ===== QUANTIZE FINAL OUTPUT =====
        elif op_name in ("quantize_final_output", "quantize"):
            outputs = op.get("outputs", {})
            out = outputs.get("output") or outputs.get("y")
            if out:
                buf = out.get("buffer", "")
                # Quantize output goes to layer_input (main stream Q8 buffer)
                if buf not in ("layer_input",):
                    raise RuntimeError(
                        f"\n❌ HARD FAULT: Invalid buffer assignment\n"
                        f"   op={op_name} layer={layer}\n"
                        f"   expected output: layer_input (main_stream_q8)\n"
                        f"   got: {buf}\n"
                    )

    print(f"  ✓ Buffer validation passed for {len(ops)} ops")


def load_kernel_bindings() -> Dict[str, Dict]:
    """Load legacy function-keyed bindings for uncontracted compatibility paths."""
    bindings_path = V8_ROOT / "kernel_maps" / "kernel_bindings.json"
    with open(bindings_path, "r") as f:
        data = json.load(f)
    bindings = data.get("bindings", {})
    if not isinstance(bindings, dict):
        bindings = {}

    overlay_path = V8_ROOT / "kernel_maps" / "kernel_bindings.overlay.json"
    if overlay_path.exists():
        with open(overlay_path, "r", encoding="utf-8") as f:
            overlay_doc = json.load(f)
        overlay_bindings = overlay_doc.get("bindings", overlay_doc)
        if isinstance(overlay_bindings, dict):
            bindings.update(copy.deepcopy(overlay_bindings))
    return bindings


_CALL_ABI_SOURCE_KINDS = {
    "activation",
    "config",
    "const",
    "dim",
    "dtype",
    "dtype_weight",
    "null",
    "output",
    "param",
    "runtime",
    "scratch",
    "weight",
    "weight_f",
}


def _validate_kernel_call_abi(kernel_id: str, function: str, call_abi: Dict, source: Path) -> None:
    if set(call_abi) - {"version", "params"}:
        unknown = sorted(set(call_abi) - {"version", "params"})
        raise RuntimeError(
            f"HARD CALL ABI FAULT: kernel {kernel_id!r} in {source.name} has unknown "
            f"call_abi fields {unknown}. Fix the kernel map; do not add compiler defaults."
        )
    if call_abi.get("version") != 1:
        raise RuntimeError(
            f"HARD CALL ABI FAULT: kernel {kernel_id!r} in {source.name} must declare "
            "call_abi.version=1."
        )
    params = call_abi.get("params")
    if not isinstance(params, list):
        raise RuntimeError(
            f"HARD CALL ABI FAULT: kernel {kernel_id!r} in {source.name} must declare "
            "an ordered call_abi.params array."
        )
    seen = set()
    for index, param in enumerate(params):
        if not isinstance(param, dict):
            raise RuntimeError(
                f"HARD CALL ABI FAULT: {kernel_id!r} call_abi.params[{index}] is not an object."
            )
        unknown = sorted(set(param) - {"name", "source", "cast", "alt"})
        if unknown:
            raise RuntimeError(
                f"HARD CALL ABI FAULT: {kernel_id!r} call parameter {index} has unknown "
                f"fields {unknown}."
            )
        name = str(param.get("name", "") or "").strip()
        source_expr = str(param.get("source", "") or "").strip()
        if not name or not source_expr:
            raise RuntimeError(
                f"HARD CALL ABI FAULT: {kernel_id!r} call parameter {index} requires "
                "non-empty name and source."
            )
        if name in seen:
            raise RuntimeError(
                f"HARD CALL ABI FAULT: {kernel_id!r} repeats call parameter {name!r}."
            )
        seen.add(name)
        source_kind = source_expr.split(":", 1)[0]
        if source_kind not in _CALL_ABI_SOURCE_KINDS:
            raise RuntimeError(
                f"HARD CALL ABI FAULT: {kernel_id!r}.{name} uses unsupported source "
                f"{source_expr!r}."
            )
        if source_kind == "null" and source_expr != "null":
            raise RuntimeError(
                f"HARD CALL ABI FAULT: {kernel_id!r}.{name} null source must be exactly 'null'."
            )
        if source_kind != "null" and ":" not in source_expr:
            raise RuntimeError(
                f"HARD CALL ABI FAULT: {kernel_id!r}.{name} source {source_expr!r} "
                "must include an explicit source namespace."
            )
        if "cast" in param and (not isinstance(param["cast"], str) or not param["cast"].strip()):
            raise RuntimeError(
                f"HARD CALL ABI FAULT: {kernel_id!r}.{name} cast must be a non-empty string."
            )
        if "alt" in param:
            alt = param["alt"]
            if (
                not isinstance(alt, list)
                or not alt
                or any(not isinstance(value, str) or not value.strip() for value in alt)
                or len(set(alt)) != len(alt)
            ):
                raise RuntimeError(
                    f"HARD CALL ABI FAULT: {kernel_id!r}.{name} alt must contain unique, "
                    "non-empty source names."
                )
    if not function:
        raise RuntimeError(
            f"HARD CALL ABI FAULT: kernel {kernel_id!r} in {source.name} has no impl.function."
        )


def load_kernel_call_abis(
    kernel_maps_dir: Optional[Path] = None,
    legacy_bindings: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    """Load exact kernel-ID-owned call ABIs and reject split ownership."""
    maps_dir = kernel_maps_dir or (V8_ROOT / "kernel_maps")
    legacy = load_kernel_bindings() if legacy_bindings is None else legacy_bindings
    result: Dict[str, Dict] = {}
    excluded = {"KERNEL_REGISTRY.json", "kernel_bindings.json", "kernel_bindings.overlay.json"}
    for path in sorted(maps_dir.glob("*.json")):
        if path.name in excluded:
            continue
        with path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
        call_abi = doc.get("call_abi")
        if call_abi is None:
            continue
        kernel_id = str(doc.get("id", "") or "").strip()
        function = str((doc.get("impl") or {}).get("function", "") or "").strip()
        if not kernel_id:
            raise RuntimeError(f"HARD CALL ABI FAULT: {path.name} declares call_abi without id.")
        if kernel_id in result:
            raise RuntimeError(f"HARD CALL ABI FAULT: duplicate call ABI owner for {kernel_id!r}.")
        _validate_kernel_call_abi(kernel_id, function, call_abi, path)
        legacy_keys = [key for key in {kernel_id, function} if key and key in legacy]
        if legacy_keys:
            raise RuntimeError(
                f"HARD CALL ABI FAULT: kernel map {path.name} owns {kernel_id!r}, but legacy "
                f"bindings still define {legacy_keys}. Remove the duplicate legacy entries."
            )
        result[kernel_id] = {
            "function": function,
            "call_abi": copy.deepcopy(call_abi),
            "source_file": path.name,
        }
    return result


def generate_ir_lower_3(lowered_ir: Dict, mode: str) -> Dict:
    """
    IR Lower 3: Emit call-ready ops with ordered args (function + expr list).
    This removes all semantic ambiguity for codegen.
    """
    legacy_bindings = load_kernel_bindings()
    kernel_call_abis = load_kernel_call_abis(legacy_bindings=legacy_bindings)
    ops = lowered_ir.get("operations", lowered_ir.get("ops", []))
    config = lowered_ir.get("config", {})
    dtype_map = {
        "fp32": "0",
        "bf16": "1",
        "fp16": "2",
        "int8": "3",
        "int4": "4",
        "q4_0": "5",
        "q4_1": "6",
        "q4_k": "7",
        "q6_k": "8",
        "q8_0": "9",
        "q8_k": "10",
        "q5_0": "11",
        "q5_1": "12",
    }

    def ptr_expr(base: str, offset: object, cast: Optional[str]) -> str:
        off = str(offset)
        expr = f"{base} + {off}"
        return f"({cast})({expr})" if cast else expr

    def parse_int_literal(expr: object) -> Optional[int]:
        raw = str(expr or "").strip()
        if not raw:
            return None
        # Accept C integer suffixes (e.g. 123u, 456UL).
        while raw and raw[-1] in ("u", "U", "l", "L"):
            raw = raw[:-1]
        try:
            return int(raw, 0)
        except Exception:
            return None

    def arg_by_source(args_list: List[Dict[str, str]], source_key: str) -> Optional[Dict[str, str]]:
        for item in args_list:
            if str(item.get("source", "")) == source_key:
                return item
        return None

    def select_from_dict(name: str, dct: Dict, aliases: Dict[str, List[str]]) -> Optional[Dict]:
        if name in dct:
            return dct[name]
        for alt in aliases.get(name, []):
            if alt in dct:
                return dct[alt]
        if len(dct) == 1:
            return next(iter(dct.values()))
        return None

    def is_bias_weight_binding(key: str, winfo: Dict) -> bool:
        if WEIGHT_TO_KERNEL_INPUT.get(key) == "bias":
            return True
        key_lc = str(key or "").lower()
        explicit_bias_aliases = {
            "bq",
            "bk",
            "bv",
            "bo",
            "b1",
            "b2",
            "bqkv",
            "final_ln_bias",
            "patch_bias",
        }
        if key_lc in explicit_bias_aliases or key_lc.endswith("_b") or "bias" in key_lc:
            return True
        weight_name = str((winfo or {}).get("name", "")).lower()
        return ".bias" in weight_name or weight_name.endswith("bias")

    def select_weight(name: str, weights: Dict, alt: Optional[List[str]] = None) -> Optional[Tuple[str, Dict]]:
        if name in weights:
            return name, weights[name]
        if alt:
            for a in alt:
                if a in weights:
                    return a, weights[a]
        if name == "_first_weight":
            if weights:
                k = next(iter(weights.keys()))
                return k, weights[k]
        if name == "_bias":
            for k, v in weights.items():
                if is_bias_weight_binding(k, v):
                    return k, v
        # Try reverse map: kernel input name -> IR weight key
        for k, v in weights.items():
            mapped = WEIGHT_TO_KERNEL_INPUT.get(k)
            if mapped == name or (name == "_bias" and mapped == "bias"):
                return k, v
        return None

    call_ops = []
    all_errors = []

    memory = lowered_ir.get("memory", {})
    arena = memory.get("arena", {})
    weights_base = int(arena.get("weights_base", 0))
    activations_base = int(arena.get("activations_base", 0))
    layout_mode = arena.get("mode", "region")
    weight_define = {e.get("name"): e.get("define") for e in memory.get("weights", {}).get("entries", [])}
    act_buffers = memory.get("activations", {}).get("buffers", [])
    act_define = {b.get("name"): b.get("define") for b in act_buffers}
    act_offset = {b.get("name"): int(b.get("offset", 0)) for b in act_buffers}
    use_bump_base = bool(arena) or any(weight_define.values()) or any(act_define.values())

    def activation_off_expr(buf_name: str, offset: object) -> str:
        """
        Preserve sub-buffer addressing when a logical activation binding points
        at a scoped slice inside a larger physical buffer.

        The template/lowered graph owns the stitch contract. If a binding
        carries an activation_offset beyond the logical buffer's base offset
        (for example a per-layer recurrent state slice), do not collapse it
        back to the bare buffer macro here.
        """
        offset_i = int(offset)
        macro = act_define.get(buf_name)
        if not macro:
            return str(activations_base + offset_i)
        base_off = act_offset.get(buf_name)
        if base_off is None:
            return macro
        delta = offset_i - int(base_off)
        if delta == 0:
            return macro
        sign = "+" if delta > 0 else "-"
        return f"({macro} {sign} {abs(delta)})"

    for op in ops:
        func = op.get("function", op.get("kernel", "unknown"))
        kernel_id = str(op.get("kernel", "") or "").strip()
        call_abi_entry = kernel_call_abis.get(kernel_id)
        governed = "resolved_contract" in op or "resolved_execution" in op
        binding_owner = "kernel_map" if call_abi_entry is not None else "legacy_compatibility"
        if call_abi_entry is not None:
            if call_abi_entry["function"] != func:
                raise RuntimeError(
                    f"HARD CALL ABI FAULT: kernel map {kernel_id!r} resolves function "
                    f"{call_abi_entry['function']!r}, but LoweredIR carries {func!r}."
                )
            binding = call_abi_entry["call_abi"]
        elif governed:
            binding = None
        else:
            binding = legacy_bindings.get(func)
        op_errors = []
        op_warnings = []

        # Keep explicit transpose placeholder ops in lowered_call.
        # Prefill codegen materializes the data movement based on op type.
        # If we drop these ops here, generated prefill C skips all required
        # token-major <-> head-major transposes and attention consumes wrong layouts.
        op_name = op.get("op", "")
        if op_name in ("transpose_qkv_to_head_major", "transpose_kv_to_head_major", "transpose_attn_out_to_token_major"):
            call_ops.append({
                "idx": op.get("idx", -1),
                "function": func,
                "op": op_name,
                "layer": op.get("layer", -1),
                "section": op.get("section", ""),
                "args": [],
                "errors": [],
                "warnings": [],
            })
            continue

        if not binding:
            if governed:
                op_errors.append(
                    f"Resolved kernel '{kernel_id}' is missing map-owned call_abi; "
                    "fix the kernel map rather than adding a compiler fallback"
                )
            else:
                op_errors.append(f"Missing legacy binding for function '{func}'")
            call_ops.append({
                "idx": op.get("idx", -1),
                "function": func,
                "op": op.get("op", ""),
                "layer": op.get("layer", -1),
                "section": op.get("section", ""),
                "args": [],
                "errors": op_errors,
                "warnings": op_warnings,
            })
            all_errors.append({"idx": op.get("idx", -1), "function": func, "error": op_errors[0]})
            continue

        activations = op.get("activations", {})
        outputs = op.get("outputs", {})
        weights = op.get("weights", {})
        scratch_list = op.get("scratch", [])
        scratch = {s.get("name"): s for s in scratch_list if s.get("name")}
        params = op.get("params", {})

        # Aliases for activation/output key lookups (handles case differences between bindings and IR)
        act_aliases = {
            "tokens": ["token_ids", "tokens"],
            "input": ["input", "a", "A", "x", "X"],
            "a": ["A", "a", "input"],  # GEMM uses "A" in IR, "a" in binding
            "x": ["x", "X", "input"],  # GEMV uses "x"
            "src": ["src", "input", "a", "A"],  # memcpy source
        }
        out_aliases = {
            "out": ["output", "out", "out_token", "C", "c"],
            "c": ["C", "c", "output"],  # GEMM uses "C" in IR, "c" in binding
            "y": ["y", "Y", "output"],  # GEMV output
            "dst": ["dst", "output"],  # memcpy destination
        }

        args = []
        for param in binding.get("params", []):
            src = param.get("source", "")
            name = param.get("name", "")
            cast = param.get("cast")
            resolved_weight_ref = None

            if src.startswith("activation:"):
                key = src.split(":", 1)[1]
                info = select_from_dict(key, activations, act_aliases)
                if not info:
                    op_errors.append(f"{func}.{name}: missing activation '{key}'")
                    expr = "NULL"
                else:
                    offset = info.get("activation_offset", 0)
                    buf_name = info.get("buffer", key)
                    if use_bump_base:
                        off_expr = activation_off_expr(buf_name, offset)
                        expr = ptr_expr("model->bump", off_expr, cast or "const float*")
                    else:
                        expr = ptr_expr("ACT", offset, cast or "const float*")

            elif src.startswith("output:"):
                key = src.split(":", 1)[1]
                info = select_from_dict(key, outputs, out_aliases)
                if not info:
                    op_errors.append(f"{func}.{name}: missing output '{key}'")
                    expr = "NULL"
                else:
                    offset = info.get("activation_offset", 0)
                    buf_name = info.get("buffer", key)
                    if use_bump_base:
                        off_expr = activation_off_expr(buf_name, offset)
                        expr = ptr_expr("model->bump", off_expr, cast or "float*")
                    else:
                        expr = ptr_expr("ACT", offset, cast or "float*")

            elif src.startswith("scratch:"):
                key = src.split(":", 1)[1]
                info = scratch.get(key)
                if not info and len(scratch) == 1:
                    info = next(iter(scratch.values()))
                if not info:
                    op_errors.append(f"{func}.{name}: missing scratch '{key}'")
                    expr = "NULL"
                else:
                    offset = info.get("scratch_offset", 0)
                    buf_name = info.get("name", key)
                    if use_bump_base:
                        if info.get("force_offset"):
                            off_expr = str(activations_base + int(offset))
                        else:
                            off_expr = activation_off_expr(buf_name, offset)
                        expr = ptr_expr("model->bump", off_expr, cast or "float*")
                    else:
                        expr = ptr_expr("ACT", offset, cast or "float*")

            elif src.startswith("weight_f:"):
                key = src.split(":", 1)[1]
                alt = param.get("alt", None)
                sel = select_weight(key, weights, alt)
                if not sel:
                    op_warnings.append(f"{func}.{name}: missing weight_f '{key}', using NULL")
                    expr = "NULL"
                else:
                    _, winfo = sel
                    offset = winfo.get("bump_offset", 0)
                    wname = winfo.get("name")
                    resolved_weight_ref = wname or resolved_weight_ref
                    macro = weight_define.get(wname)
                    if use_bump_base:
                        off_expr = macro if macro else str(weights_base + int(offset))
                        expr = ptr_expr("model->bump", off_expr, cast or "float*")
                    else:
                        expr = ptr_expr("model->bump_weights", offset, cast or "float*")

            elif src.startswith("weight:"):
                key = src.split(":", 1)[1]
                sel = select_weight(key, weights, param.get("alt", None))
                if not sel:
                    op_errors.append(f"{func}.{name}: missing weight '{key}'")
                    expr = "NULL"
                else:
                    _, winfo = sel
                    offset = winfo.get("bump_offset", 0)
                    wname = winfo.get("name")
                    resolved_weight_ref = wname or resolved_weight_ref
                    macro = weight_define.get(wname)
                    if use_bump_base:
                        off_expr = macro if macro else str(weights_base + int(offset))
                        expr = ptr_expr("model->bump", off_expr, cast or "const void*")
                    else:
                        expr = ptr_expr("model->bump_weights", offset, cast or "const void*")

            elif src.startswith("dim:"):
                key = src.split(":", 1)[1]
                if key in params:
                    expr = str(params[key])
                elif key == "max_seq_len":
                    # Prefer context_length override when present (e.g., --context-len)
                    if "context_length" in config:
                        expr = str(config["context_length"])
                    elif "context_len" in config:
                        expr = str(config["context_len"])
                    elif key in config:
                        expr = str(config[key])
                    else:
                        op_errors.append(f"{func}.{name}: missing dim '{key}'")
                        expr = "0"
                elif key == "kv_lora_input_dim":
                    expr = str(int(config.get("kv_lora_rank", 0) or 0) + int(config.get("qk_rope_head_dim", 0) or 0))
                elif key == "kv_cache_head_dim":
                    k_head = int(config.get("mla_k_head_dim", config.get("max_k_head_dim", config.get("head_dim", 1))) or config.get("head_dim", 1) or 1)
                    v_head = int(config.get("mla_v_head_dim", config.get("max_v_head_dim", config.get("v_head_dim", config.get("head_dim", 1)))) or config.get("v_head_dim", config.get("head_dim", 1)) or 1)
                    base = int(config.get("head_dim", 1) or 1)
                    expr = str(max(k_head, v_head, base))
                elif key == "moe_shared_expert_intermediate_size":
                    if key in config:
                        expr = str(config[key])
                    else:
                        expr = str(int(config.get("moe_intermediate_size", 0) or 0) * max(1, int(config.get("n_shared_experts", 1) or 1)))
                elif key in config:
                    expr = str(config[key])
                elif key == "intermediate_size" and "intermediate_dim" in config:
                    expr = str(config["intermediate_dim"])
                else:
                    op_errors.append(f"{func}.{name}: missing dim '{key}'")
                    expr = "0"

            elif src.startswith("param:"):
                key = src.split(":", 1)[1]
                if key in params:
                    expr = str(params[key])
                else:
                    op_errors.append(f"{func}.{name}: missing param '{key}'")
                    expr = "0"

            elif src.startswith("config:"):
                key = src.split(":", 1)[1]
                if key in config:
                    val = config[key]
                    if isinstance(val, str):
                        expr = f'"{val}"'
                    elif isinstance(val, bool):
                        expr = "1" if val else "0"
                    else:
                        expr = str(val)
                else:
                    op_errors.append(f"{func}.{name}: missing config '{key}'")
                    expr = "0"

            elif src.startswith("runtime:"):
                key = src.split(":", 1)[1]
                layer = op.get("layer", 0)
                kv_layer = int(op.get("_kv_cache_read_layer", layer))
                try:
                    kv_cache_head_dim = int(config.get("kv_cache_head_dim", config.get("head_dim", 1)) or config.get("head_dim", 1) or 1)
                except Exception:
                    kv_cache_head_dim = int(config.get("head_dim", 1) or 1)
                if kv_cache_head_dim <= 0:
                    kv_cache_head_dim = 1
                layer_k_offsets = config.get("layer_k_cache_offset") if isinstance(config.get("layer_k_cache_offset"), list) else []
                layer_v_offsets = config.get("layer_v_cache_offset") if isinstance(config.get("layer_v_cache_offset"), list) else []
                if 0 <= kv_layer < len(layer_k_offsets) and 0 <= kv_layer < len(layer_v_offsets):
                    k_expr = f"(model->kv_cache + {int(layer_k_offsets[kv_layer])}ULL*MAX_SEQ_LEN)"
                    v_expr = f"(model->kv_cache + {int(layer_v_offsets[kv_layer])}ULL*MAX_SEQ_LEN)"
                    k_expr_f16 = f"(model->kv_cache_f16 + {int(layer_k_offsets[kv_layer])}ULL*MAX_SEQ_LEN)"
                    v_expr_f16 = f"(model->kv_cache_f16 + {int(layer_v_offsets[kv_layer])}ULL*MAX_SEQ_LEN)"
                else:
                    k_expr = f"(model->kv_cache + ({kv_layer}*2)*NUM_KV_HEADS*MAX_SEQ_LEN*{kv_cache_head_dim})"
                    v_expr = f"(model->kv_cache + ({kv_layer}*2+1)*NUM_KV_HEADS*MAX_SEQ_LEN*{kv_cache_head_dim})"
                    k_expr_f16 = f"(model->kv_cache_f16 + ({kv_layer}*2)*NUM_KV_HEADS*MAX_SEQ_LEN*{kv_cache_head_dim})"
                    v_expr_f16 = f"(model->kv_cache_f16 + ({kv_layer}*2+1)*NUM_KV_HEADS*MAX_SEQ_LEN*{kv_cache_head_dim})"
                if key in ("kv_cache_k_layer", "kv_k"):
                    expr = k_expr
                elif key in ("kv_cache_v_layer", "kv_v"):
                    expr = v_expr
                elif key == "kv_cache_k_layer_f16":
                    expr = k_expr_f16
                elif key == "kv_cache_v_layer_f16":
                    expr = v_expr_f16
                elif key == "rope_cos":
                    expr = "model->rope_cos"
                elif key == "rope_sin":
                    expr = "model->rope_sin"
                elif key == "rope_cos_mla_positioned":
                    rope_half = max(1, int(config.get("qk_rope_head_dim", config.get("head_dim", 2)) or 2) // 2)
                    if mode == "decode":
                        expr = f"(model->rope_cos + (size_t)model->pos * (size_t){rope_half})"
                    else:
                        expr = "model->rope_cos"
                elif key == "rope_sin_mla_positioned":
                    rope_half = max(1, int(config.get("qk_rope_head_dim", config.get("head_dim", 2)) or 2) // 2)
                    if mode == "decode":
                        expr = f"(model->rope_sin + (size_t)model->pos * (size_t){rope_half})"
                    else:
                        expr = "model->rope_sin"
                elif key == "pos":
                    expr = "model->rope_pos" if name == "pos_offset" else "model->pos"
                elif key == "seq_len":
                    expr = str(params.get("seq_len", 1))
                elif key in ("kv_tokens", "cache_len"):
                    if mode == "decode":
                        expr = "model->pos + 1"
                    else:
                        expr = str(params.get("seq_len", 1))
                elif key == "prefill_start_pos":
                    if mode != "prefill":
                        op_errors.append(f"{func}.{name}: prefill_start_pos is invalid in {mode} mode")
                        expr = "0"
                    else:
                        # Call IR names persistent runtime state, not a local
                        # variable owned by one generated prefill function.
                        expr = "model->pos"
                elif key == "layer":
                    expr = str(layer)
                else:
                    op_errors.append(f"{func}.{name}: unknown runtime '{key}'")
                    expr = "0"
                if cast:
                    expr = f"({cast})({expr})"

            elif src.startswith("const:"):
                expr = src.split(":", 1)[1]

            elif src == "null":
                expr = "NULL"

            elif src.startswith("dtype_weight:"):
                key = src.split(":", 1)[1]
                sel = select_weight(key, weights)
                if not sel:
                    op_errors.append(f"{func}.{name}: missing dtype weight '{key}'")
                    expr = "0"
                else:
                    _, winfo = sel
                    dtype_str = str(winfo.get("dtype", "")).lower()
                    resolved_weight_ref = str(winfo.get("name") or resolved_weight_ref or "")
                    if dtype_str in dtype_map:
                        expr = dtype_map[dtype_str]
                    else:
                        op_errors.append(f"{func}.{name}: unknown weight dtype '{dtype_str}'")
                        expr = "0"

            elif src.startswith("dtype:"):
                key = src.split(":", 1)[1]
                if key in dtype_map:
                    expr = dtype_map[key]
                else:
                    op_errors.append(f"{func}.{name}: unknown dtype '{key}'")
                    expr = "0"

            else:
                op_errors.append(f"{func}.{name}: unknown source '{src}'")
                expr = "0"

            arg_doc = {
                "name": name,
                "source": src,
                "expr": expr,
            }
            if src.startswith(("activation:", "output:", "scratch:")):
                info = None
                if src.startswith("activation:"):
                    info = select_from_dict(src.split(":", 1)[1], activations, act_aliases)
                elif src.startswith("output:"):
                    info = select_from_dict(src.split(":", 1)[1], outputs, out_aliases)
                elif src.startswith("scratch:"):
                    info = scratch.get(src.split(":", 1)[1]) or (next(iter(scratch.values())) if len(scratch) == 1 else None)
                if isinstance(info, dict):
                    resolved_buffer_ref = str(info.get("buffer") or info.get("name") or "").strip()
                    if resolved_buffer_ref:
                        arg_doc["buffer_ref"] = resolved_buffer_ref
            if resolved_weight_ref:
                arg_doc["weight_ref"] = resolved_weight_ref
            args.append(arg_doc)

        # Strict runtime invariant checks (lowered-call stage, before codegen).
        if op_name == "kv_cache_batch_copy":
            size_arg = arg_by_source(args, "dim:_kv_copy_bytes")
            if not size_arg:
                op_errors.append(
                    f"{func}: missing required call arg dim:_kv_copy_bytes "
                    "(kv token-block copy size)"
                )
            else:
                size_expr = str(size_arg.get("expr", "")).strip()
                if size_expr in {"", "0", "NULL"}:
                    op_errors.append(f"{func}: invalid _kv_copy_bytes expression '{size_expr or '<empty>'}'")
                size_val = parse_int_literal(size_expr)
                if size_val is not None and size_val <= 0:
                    op_errors.append(f"{func}: _kv_copy_bytes must be > 0 (got {size_val})")

                n_kv_arg = arg_by_source(args, "dim:num_kv_heads")
                hd_arg = arg_by_source(args, "dim:head_dim")
                seq_arg = arg_by_source(args, "dim:seq_len")
                n_kv = parse_int_literal(n_kv_arg.get("expr", "")) if n_kv_arg else None
                hd = parse_int_literal(hd_arg.get("expr", "")) if hd_arg else None
                seq = parse_int_literal(seq_arg.get("expr", "")) if seq_arg else None
                if None not in (n_kv, hd, seq) and size_val is not None:
                    expected = int(n_kv) * int(hd) * int(seq) * 4
                    if expected <= 0:
                        op_errors.append(
                            f"{func}: invalid kv copy dimensions "
                            f"(num_kv_heads={n_kv}, head_dim={hd}, seq_len={seq})"
                        )
                    elif size_val != expected:
                        op_errors.append(
                            f"{func}: _kv_copy_bytes mismatch (expected {expected}, got {size_val})"
                        )

            for src_key, label in (("activation:k_src", "k_src"), ("activation:v_src", "v_src")):
                src_arg = arg_by_source(args, src_key)
                if not src_arg:
                    op_errors.append(f"{func}: missing required call arg {src_key}")
                elif str(src_arg.get("expr", "")).strip() == "NULL":
                    op_errors.append(f"{func}: {label} resolved to NULL")
            for dst_key, label in (("output:k_dst", "k_dst"), ("output:v_dst", "v_dst")):
                dst_arg = arg_by_source(args, dst_key)
                if not dst_arg:
                    op_errors.append(f"{func}: missing required call arg {dst_key}")
                elif str(dst_arg.get("expr", "")).strip() == "NULL":
                    op_errors.append(f"{func}: {label} resolved to NULL")

        if op_errors:
            all_errors.append({
                "idx": op.get("idx", -1),
                "function": func,
                "errors": op_errors,
            })

        resolved_contract = copy.deepcopy(op.get("resolved_contract")) if op.get("resolved_contract") else None
        if resolved_contract is not None and resolved_contract.get("function") != func:
            raise RuntimeError(
                f"HARD CONTRACT FAULT: call-ready IR function {func!r} differs from resolved "
                f"function {resolved_contract.get('function')!r} for {resolved_contract.get('operation')}. "
                "Code generation must emit the resolved decision without reselection."
            )
        call_op = {
            "idx": op.get("idx", -1),
            "function": func,
            "op": op.get("op", ""),
            "layer": op.get("layer", -1),
            "section": op.get("section", ""),
            "args": args,
            "errors": op_errors,
            "warnings": op_warnings,
            "call_abi": {
                "version": int(binding.get("version", 0) or 0),
                "owner": binding_owner,
                "kernel_id": kernel_id,
                "source_file": call_abi_entry["source_file"] if call_abi_entry else "kernel_bindings*.json",
            },
        }
        if op.get("required_contract") is not None:
            call_op["required_contract"] = copy.deepcopy(op["required_contract"])
        if resolved_contract is not None:
            call_op["resolved_contract"] = resolved_contract
        resolved_execution = copy.deepcopy(op.get("resolved_execution")) if op.get("resolved_execution") else None
        if resolved_execution is not None:
            if resolved_execution.get("kernel_id") != op.get("kernel"):
                raise RuntimeError(
                    f"HARD CONTRACT FAULT: call-ready IR kernel {op.get('kernel')!r} differs from "
                    f"execution metadata {resolved_execution.get('kernel_id')!r}."
                )
            call_op["resolved_execution"] = resolved_execution
        resolved_codegen = copy.deepcopy(op.get("resolved_codegen_capability")) if op.get("resolved_codegen_capability") else None
        if resolved_codegen is not None:
            if resolved_codegen.get("kernel_id") != op.get("kernel"):
                raise RuntimeError(
                    f"HARD CODEGEN CAPABILITY FAULT: call-ready IR kernel {op.get('kernel')!r} "
                    f"differs from codegen metadata {resolved_codegen.get('kernel_id')!r}."
                )
            if resolved_codegen.get("function") != func:
                raise RuntimeError(
                    f"HARD CODEGEN CAPABILITY FAULT: call-ready IR function {func!r} differs "
                    f"from codegen metadata {resolved_codegen.get('function')!r}."
                )
            call_op["resolved_codegen_capability"] = resolved_codegen
        if op.get("semantic_checkpoints") is not None:
            call_op["semantic_checkpoints"] = copy.deepcopy(op["semantic_checkpoints"])
        call_ops.append(call_op)

    lowered_call = {
        "format": "lowered-ir-v3",
        "version": 3,
        "mode": mode,
        "config": lowered_ir.get("config", {}),
        "memory": lowered_ir.get("memory", {}),
        "operations": call_ops,
        "errors": all_errors,
    }

    return lowered_call


def generate_init_ir_lower_3(init_ir: Dict, layout: Dict) -> Dict:
    """
    IR Lower 3 for init ops: Emit call-ready ops with ordered args.

    Init ops are simpler than inference ops - they typically just have:
    - Output buffers (rope_cos, rope_sin)
    - Dimension params (max_seq_len, head_dim)
    - Config params (rope_theta)

    Codegen just reads this and emits the function calls sequentially.
    """
    if not init_ir:
        return {"format": "lowered-init-v3", "version": 1, "operations": [], "errors": []}

    ops = init_ir.get("ops", [])
    config = _normalize_manifest_config(init_ir.get("config", {}))
    memory = layout.get("memory", {}) if layout else {}
    act_buffers = {b.get("name"): b for b in memory.get("activations", {}).get("buffers", [])}

    call_ops = []
    all_errors = []

    for op in ops:
        func = op.get("kernel", "unknown")
        op_type = op.get("op", "")
        params = op.get("params", {})
        op_config = op.get("config", {})
        op_errors = []

        args = []

        # Handle rope_init specifically
        if op_type == "rope_init" and func == "rope_precompute_cache":
            # rope_precompute_cache(float *cos_cache, float *sin_cache, int max_seq_len,
            #                      int head_dim, float base, int rotary_dim,
            #                      const char *scaling_type, float scaling_factor)

            # cos_cache output buffer
            rope_cache_buf = act_buffers.get("rope_cache", act_buffers.get("rope_cos_cache", {}))
            rope_cache_define = rope_cache_buf.get("define", "A_ROPE_CACHE")
            args.append({
                "name": "cos_cache",
                "source": "output:rope_cos",
                "expr": f"(float*)(g_model->bump + {rope_cache_define})",
            })

            # sin_cache output buffer (offset by rotary_half)
            # Note: uses ROTARY_DIM for cache sizing, not HEAD_DIM
            args.append({
                "name": "sin_cache",
                "source": "output:rope_sin",
                "expr": f"(float*)(g_model->bump + {rope_cache_define}) + MAX_SEQ_LEN * ROTARY_DIM / 2",
            })

            # max_seq_len from params or config
            max_seq = params.get("max_seq_len", {}).get("value", config["context_length"])
            args.append({
                "name": "max_seq_len",
                "source": "dim:max_seq_len",
                "expr": "MAX_SEQ_LEN",  # Use the #define for consistency
            })

            # head_dim from params or config
            head_dim = params.get("head_dim", {}).get("value", config["head_dim"])
            args.append({
                "name": "head_dim",
                "source": "dim:head_dim",
                "expr": "HEAD_DIM",  # Use the #define for consistency
            })

            # base (rope_theta) from params or config - THIS IS THE KEY VALUE
            rope_theta = params.get("base", {}).get("value", op_config.get("rope_theta", config["rope_theta"]))
            args.append({
                "name": "base",
                "source": "config:rope_theta",
                "expr": f"{rope_theta}f",  # Emit as float literal
            })

            # rotary_dim from params or config
            rotary_dim = params.get("rotary_dim", {}).get("value", op_config.get("rotary_dim", config["rotary_dim"]))
            args.append({
                "name": "rotary_dim",
                "source": "dim:rotary_dim",
                "expr": "ROTARY_DIM",  # Use the #define for consistency
            })

            # scaling_type from params or config
            scaling_type = params.get("scaling_type", {}).get("value", op_config.get("rope_scaling_type", config["rope_scaling_type"]))
            args.append({
                "name": "scaling_type",
                "source": "config:rope_scaling_type",
                "expr": f'"{scaling_type}"',  # Emit as string literal
            })

            # scaling_factor from params or config
            scaling_factor = params.get("scaling_factor", {}).get("value", op_config.get("rope_scaling_factor", config["rope_scaling_factor"]))
            args.append({
                "name": "scaling_factor",
                "source": "config:rope_scaling_factor",
                "expr": f"{scaling_factor}f",  # Emit as float literal
            })

        elif op_type == "rope_init" and func == "rope_precompute_cache_split":
            # rope_precompute_cache_split(float *cos_cache, float *sin_cache,
            #                             int max_seq_len, int head_dim, float base)

            # cos_cache output buffer
            rope_cache_buf = act_buffers.get("rope_cache", act_buffers.get("rope_cos_cache", {}))
            rope_cache_define = rope_cache_buf.get("define", "A_ROPE_CACHE")
            args.append({
                "name": "cos_cache",
                "source": "output:rope_cos",
                "expr": f"(float*)(g_model->bump + {rope_cache_define})",
            })

            head_dim = int(params.get("head_dim", {}).get("value", op_config.get("head_dim", config["head_dim"])))

            # sin_cache output buffer (offset by the init cache row width)
            args.append({
                "name": "sin_cache",
                "source": "output:rope_sin",
                "expr": f"(float*)(g_model->bump + {rope_cache_define}) + MAX_SEQ_LEN * {head_dim} / 2",
            })

            # max_seq_len
            args.append({
                "name": "max_seq_len",
                "source": "dim:max_seq_len",
                "expr": "MAX_SEQ_LEN",
            })

            # head_dim
            args.append({
                "name": "head_dim",
                "source": "dim:head_dim",
                "expr": str(head_dim),
            })

            # base (rope_theta)
            rope_theta = params.get("base", {}).get("value", op_config.get("rope_theta", config["rope_theta"]))
            args.append({
                "name": "base",
                "source": "config:rope_theta",
                "expr": f"{rope_theta}f",
            })

        elif op_type == "tokenizer_init":
            # Tokenizer init has explicit c_code - pass it through directly
            # Codegen will emit the c_code["init"] and c_code["free"] directly
            pass  # No args needed - c_code contains everything

        else:
            # Generic handling for future init ops
            op_errors.append(f"Unknown init op type: {op_type}")

        if op_errors:
            all_errors.append({
                "idx": op.get("op_id", -1),
                "function": func,
                "errors": op_errors,
            })

        call_op = {
            "idx": op.get("op_id", -1),
            "function": func,
            "op": op_type,
            "section": "init",
            "layer": -1,
            "args": args,
            "errors": op_errors,
            "notes": op.get("notes", ""),
        }
        # Pass through c_code for ops that have explicit C code (tokenizer_init, etc.)
        if "c_code" in op:
            call_op["c_code"] = op["c_code"]
        call_ops.append(call_op)

    # Pass through special_tokens from init_ir for code generation
    special_tokens = init_ir.get("special_tokens")

    return {
        "format": "lowered-init-v3",
        "version": 1,
        "config": config,
        # Special tokens (EOS, BOS, etc.) from GGUF - codegen generates stop token API
        "special_tokens": special_tokens,
        "operations": call_ops,
        "errors": all_errors,
        "stats": {
            "total_ops": len(call_ops),
            "errors": len(all_errors),
        }
    }


def main(args: List[str]) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build IR1: Direct template + quant → kernel IDs"
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to weights manifest JSON"
    )
    parser.add_argument(
        "--model",
        type=int,
        help="Use cached model by number (1, 2, ...)"
    )
    parser.add_argument(
        "--mode",
        choices=["decode", "prefill"],
        default="decode",
        help="Execution mode (default: decode)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output IR1 JSON file (just kernel list)"
    )
    parser.add_argument(
        "--block-manifests-dir",
        type=Path,
        help="Write one block-local weights_manifest.json per template sequence entry"
    )
    parser.add_argument(
        "--stitch-output",
        type=Path,
        help="Write the v8 stitch/orchestration plan for a multi-block template"
    )
    parser.add_argument(
        "--layout-output",
        type=Path,
        help="Output memory layout JSON file (after fusion)"
    )
    parser.add_argument(
        "--layout-input",
        type=Path,
        help="Use an existing memory layout JSON instead of generating a new one"
    )
    parser.add_argument(
        "--lowered-output",
        type=Path,
        help="Output lowered IR JSON file (kernel maps stitched with memory layout)"
    )
    parser.add_argument(
        "--manifest-map-output",
        type=Path,
        help="Output weights_manifest.map (uses runtime offsets from layout)"
    )
    parser.add_argument(
        "--call-output",
        type=Path,
        help="Output call-ready IR JSON file (IR Lower 3)"
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help="Context length for buffer allocation (default: from model config)"
    )
    parser.add_argument(
        "--logits-layout",
        choices=["auto", "last", "full"],
        default="auto",
        help="Logits buffer layout (auto=decode last/prefill full)"
    )
    parser.add_argument(
        "--no-fusion",
        action="store_true",
        help="Disable kernel fusion pass (use unfused ops)"
    )
    parser.add_argument(
        "--allow-quant-fallback",
        action="store_true",
        help="Allow unsafe quantization fallbacks (e.g., Q5_K → Q5_0). "
             "Not recommended - may cause accuracy issues or segfaults."
    )
    parser.add_argument(
        "--layout-mode",
        choices=["region", "packed"],
        default="region",
        help="Memory layout mode (region=weights+activations, packed=single arena)"
    )
    parser.add_argument(
        "--layer-limit",
        type=int,
        default=None,
        help="Limit to first N layers (for packed layout prototypes)"
    )
    parser.add_argument(
        "--init-output",
        type=Path,
        help="Output init IR JSON file (one-time initialization ops like rope_init)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable OpenMP parallelization annotations in lowered IR"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-kernel profiling instrumentation in generated code"
    )
    parser.add_argument(
        "--prefer-q8-activation",
        action="store_true",
        help="Prefer Q8-activation matmul kernels (gemv/gemm *_q8_* variants) for speed"
    )

    parsed_args = parser.parse_args(args)

    # Load manifest
    if parsed_args.manifest:
        manifest_path = parsed_args.manifest
    elif parsed_args.model:
        # TODO: Find cached model
        print("Error: --model not implemented yet, use --manifest")
        return 1
    else:
        print("Error: Must specify --manifest or --model")
        parser.print_help()
        return 1

    print(f"Loading manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)
    _merge_external_config(manifest, manifest_path)
    _hydrate_manifest_template(manifest)
    _backfill_template_runtime_flags(manifest)
    manifest["config"] = _normalize_manifest_config(manifest.get("config", {}))
    if parsed_args.prefer_q8_activation:
        manifest.setdefault("config", {})["prefer_q8_activation"] = True
    # Override logits layout if requested (propagates into layout + codegen config)
    manifest.setdefault("config", {})["logits_layout"] = parsed_args.logits_layout

    template = manifest.get("template", {})
    sequence = _template_sequence(template) if isinstance(template, dict) else []
    branch_plan = build_template_branch_plan(manifest) if isinstance(template, dict) else None
    wrote_split_artifacts = False

    if parsed_args.block_manifests_dir:
        written_blocks = write_block_manifests(manifest, parsed_args.block_manifests_dir)
        print(f"✓ Wrote {len(written_blocks)} block manifests to: {parsed_args.block_manifests_dir}")
        for item in written_blocks:
            print(f"  - {item['block_name']}: {item['manifest_path']}")
        wrote_split_artifacts = True

    if parsed_args.stitch_output:
        stitch_plan = build_stitch_plan(manifest)
        with open(parsed_args.stitch_output, "w", encoding="utf-8") as f:
            json.dump(stitch_plan, f, indent=2)
        print(f"✓ Wrote stitch plan to: {parsed_args.stitch_output}")
        wrote_split_artifacts = True

    standard_outputs_requested = any(
        (
            parsed_args.output,
            parsed_args.layout_output,
            parsed_args.layout_input,
            parsed_args.lowered_output,
            parsed_args.manifest_map_output,
            parsed_args.call_output,
            parsed_args.init_output,
        )
    )

    if wrote_split_artifacts and len(sequence) > 1 and not standard_outputs_requested:
        print("✓ Split-only v8 block artifacts generated; skipping flattened IR build")
        return 0

    # Build IR1
    registry = load_kernel_registry()
    ir1 = build_ir1_direct(manifest, manifest_path, mode=parsed_args.mode,
                           prefer_parallel=parsed_args.parallel,
                           allow_quant_fallback=parsed_args.allow_quant_fallback)

    # Insert bias_add ops BEFORE fusion pass so fused kernels can match
    # [quantize + gemv + bias_add] sequences
    ir1_with_bias = insert_bias_add_ops(ir1, registry, manifest, parsed_args.mode, manifest_path)

    # Fusion pass: combine kernels (fused attention, fused MLP, fused GEMV+bias)
    fused_ops, fusion_stats = apply_fusion_pass(ir1_with_bias, registry, parsed_args.mode, no_fusion=parsed_args.no_fusion)

    # IR Lower 1: Stitch kernel maps with fused ops
    # This creates buffer requirements (inputs/outputs/scratch) for each kernel
    ir_lower_1 = generate_ir_lower_1(fused_ops, registry, manifest, parsed_args.mode)

    # Optional: limit to first N layers (keep header ops)
    if parsed_args.layer_limit:
        limit = int(parsed_args.layer_limit)
        filtered = []
        for op in ir_lower_1:
            layer = op.get("layer", -1)
            section = op.get("section", "")
            if section == "header":
                filtered.append(op)
            elif section == "body" and layer >= 0 and layer < limit:
                filtered.append(op)
            elif section == "footer" and limit <= 0:
                filtered.append(op)
        ir_lower_1 = filtered

    # Memory Planner: Plan memory layout using IR Lower 1 buffer requirements
    context_len = parsed_args.context_len  # May be None, will use model default
    if parsed_args.layout_input:
        with open(parsed_args.layout_input, "r") as f:
            layout = json.load(f)
        if not layout.get("memory"):
            raise RuntimeError(f"Invalid layout (missing 'memory'): {parsed_args.layout_input}")
        # Keep layout offsets, but update mode for clarity in per-mode outputs
        layout["mode"] = parsed_args.mode
        if parsed_args.layout_mode:
            arena_mode = layout.get("memory", {}).get("arena", {}).get("mode")
            if arena_mode and arena_mode != parsed_args.layout_mode:
                print(f"Warning: layout_input mode '{arena_mode}' != requested '{parsed_args.layout_mode}'")
    else:
        if parsed_args.layout_mode == "packed":
            layout = generate_memory_layout_packed(
                ir_lower_1, manifest, registry, parsed_args.mode, context_len, parsed_args.layer_limit
            )
        else:
            layout = generate_memory_layout(ir_lower_1, manifest, registry, parsed_args.mode, context_len)

    # IR Lower 2: Add concrete memory offsets to IR Lower 1
    # This produces the final lowered IR with explicit pointer expressions
    lowered_ir = generate_ir_lower_2(ir_lower_1, layout, manifest, registry, parsed_args.mode)

    # CRITICAL: Update context_length in lowered_ir to match layout
    # This ensures codegen uses the correct MAX_SEQ_LEN for KV cache strides
    # Use context_len from layout config if available, otherwise from context_len variable
    effective_context_len = context_len
    if layout and "config" in layout and "context_length" in layout["config"]:
        effective_context_len = layout["config"]["context_length"]
    elif layout and "config" in layout and "context_len" in layout["config"]:
        effective_context_len = layout["config"]["context_len"]

    if effective_context_len:
        if "config" not in lowered_ir:
            lowered_ir["config"] = {}
        lowered_ir["config"]["context_length"] = effective_context_len
        lowered_ir["config"]["context_len"] = effective_context_len

    lowered_call = None
    if parsed_args.call_output:
        lowered_call = generate_ir_lower_3(lowered_ir, parsed_args.mode)

        # NOTE: OpenMP parallel pass is SUPERSEDED by thread pool dispatch.
        #
        # parallel_pass.py annotates ops with #pragma omp parallel for, but
        # codegen_v7.py never reads these annotations. Actual parallelization
        # is handled by ck_parallel_decode.h / ck_parallel_prefill.h which use
        # persistent pthread thread pools via macro redirects:
        #   - Decode: gemv_q5_0_q8_0() → gemv_q5_0_q8_0_parallel_dispatch()
        #   - Prefill: gemm_nt_q5_0_q8_0() → gemm_nt_q5_0_q8_0_parallel_dispatch()
        #
        # Kept commented out for reference. The false-sharing and memory-bandwidth
        # analysis in parallel_pass.py remains useful for planning thread pool work
        # splitting strategies.
        #
        # if getattr(parsed_args, 'parallel', False) and lowered_call:
        #     try:
        #         from parallel_pass import run_parallel_pass
        #         lowered_call, parallel_stats = run_parallel_pass(
        #             lowered_call, parsed_args.mode
        #         )
        #         print(f"  [PARALLEL PASS] {parallel_stats['parallelized_ops']}/{parallel_stats['total_ops']} ops annotated")
        #         for strat, count in parallel_stats.get('strategies', {}).items():
        #             print(f"    - {strat}: {count} ops")
        #     except ImportError as e:
        #         print(f"  [PARALLEL PASS] Warning: parallel_pass.py not found ({e})")
        #     except Exception as e:
        #         print(f"  [PARALLEL PASS] Warning: parallelization failed: {e}")

    # Once we have the right memory and lowered graph then we can do codegen
    # codegen will read the lowered IR and memory layout to emit C code
    # it should just see memory and parse the memory layout - allocate bump
    # The code should then have a load weights and load then to the right bump offset.
    # Then read lowered graph and generate c code sequentially for prefill and decode with
    # right inputs and offset to weights read.
    # and then generate tokens. We have all this working in v5 and v7
    # is the first to have the full pipeline completely generated from template + quant summary
    # and no hardcoded logic for a specific family.

    # Output
    if parsed_args.output:
        output_data = {
            "format": "ir1-dataflow",
            "version": 3,
            "mode": parsed_args.mode,
            "ops": ir1,  # Now a list of {kernel, op, section, layer, weights, dataflow}
            "branch_plan": branch_plan,
        }
        with open(parsed_args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Wrote IR1 to: {parsed_args.output}")

    if parsed_args.layout_output:
        with open(parsed_args.layout_output, 'w') as f:
            json.dump(layout, f, indent=2)
        print(f"✓ Wrote memory layout to: {parsed_args.layout_output}")
    if parsed_args.manifest_map_output:
        write_manifest_map(layout, manifest, parsed_args.manifest_map_output)
        print(f"✓ Wrote manifest map to: {parsed_args.manifest_map_output}")

    if parsed_args.lowered_output:
        with open(parsed_args.lowered_output, 'w') as f:
            json.dump(lowered_ir, f, indent=2)
        print(f"✓ Wrote lowered IR to: {parsed_args.lowered_output}")

    if parsed_args.call_output and lowered_call is not None:
        with open(parsed_args.call_output, 'w') as f:
            json.dump(lowered_call, f, indent=2)
        print(f"✓ Wrote call-ready IR to: {parsed_args.call_output}")

    # Generate and write init IR (rope_init, etc.)
    if parsed_args.init_output:
        config = manifest.get("config", {})
        init_ir = generate_init_ir(manifest, config)
        with open(parsed_args.init_output, 'w') as f:
            json.dump(init_ir, f, indent=2)
        print(f"✓ Wrote init IR to: {parsed_args.init_output}")
        if init_ir["stats"]["has_rope_init"]:
            rope_theta = init_ir["config"].get("rope_theta", 10000.0)
            rotary_dim = init_ir["config"].get("rotary_dim", "head_dim")
            scaling_type = init_ir["config"].get("rope_scaling_type", "none")
            scaling_factor = init_ir["config"].get("rope_scaling_factor", 1.0)
            print(f"  - rope_init: theta={rope_theta}, rotary={rotary_dim}, scaling={scaling_type}/{scaling_factor}")

        # Also generate lowered init IR (init_call.json)
        init_call_path = parsed_args.init_output.parent / "init_call.json"
        init_call = generate_init_ir_lower_3(init_ir, layout)
        with open(init_call_path, 'w') as f:
            json.dump(init_call, f, indent=2)
        print(f"✓ Wrote init call IR to: {init_call_path}")

    if lowered_call is not None:
        # TODO(contract): extend this from structural checks (errors/missing args)
        # to semantic per-op contract validation at lowered_*_call.json generation time.
        # Example: verify rope/norm/logits/kv invariants before codegen.
        ir_errors = lowered_call.get("errors") if isinstance(lowered_call.get("errors"), list) else []
        call_ops = lowered_call.get("operations") if isinstance(lowered_call.get("operations"), list) else []
        op_errors = [op for op in call_ops if isinstance(op, dict) and isinstance(op.get("errors"), list) and op.get("errors")]
        missing_args = [op for op in call_ops if isinstance(op, dict) and "args" not in op]
        if ir_errors or op_errors or missing_args:
            print(
                f"ERROR: IR Lower 3 invalid: {len(ir_errors)} issues, "
                f"{len(op_errors)} ops with errors, {len(missing_args)} ops missing args"
            )
            return 2

    if not parsed_args.output and not parsed_args.layout_output and not parsed_args.lowered_output:
        print(f"\nIR1 (first 10 ops):")
        for i, op in enumerate(ir1[:10]):
            kernel = op["kernel"]
            weights = len(op.get("weights", {}))
            print(f"  {i:3d}: {kernel} ({weights} weights)")
        if len(ir1) > 10:
            print(f"  ... ({len(ir1) - 10} more)")

        print(f"\nFused ops (first 10):")
        for i, op in enumerate(fused_ops[:10]):
            kernel = op["kernel"]
            weights = len(op.get("weights", {}))
            print(f"  {i:3d}: {kernel} ({weights} weights)")
        if len(fused_ops) > 10:
            print(f"  ... ({len(fused_ops) - 10} more)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
