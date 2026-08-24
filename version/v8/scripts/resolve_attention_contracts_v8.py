#!/usr/bin/env python3
"""Resolve v8 attention semantics against kernel capabilities.

This resolver is intentionally model-name blind. It consumes a circuit,
a canonical contract registry, and a kernel capability overlay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from jsonschema import Draft202012Validator


V8_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V8_ROOT.parents[1]
DEFAULT_CONTRACTS = V8_ROOT / "contracts" / "attention_reductions.json"
DEFAULT_LINEAR_CONTRACTS = V8_ROOT / "contracts" / "quantized_linear.json"
DEFAULT_KERNELS = V8_ROOT / "kernel_maps"
SCHEMA_ROOT = V8_ROOT / "schemas"
CONTRACT_REGISTRY_SCHEMA = SCHEMA_ROOT / "attention_reduction_registry.schema.json"
CIRCUIT_REQUIREMENTS_SCHEMA = SCHEMA_ROOT / "attention_required_contracts.schema.json"
KERNEL_CAPABILITY_SCHEMA = SCHEMA_ROOT / "attention_kernel_capability.schema.json"
KERNEL_EXECUTION_SCHEMA = SCHEMA_ROOT / "kernel_execution_capability.schema.json"
RESOLVED_CONTRACT_SCHEMA = SCHEMA_ROOT / "resolved_attention_contract.schema.json"
LINEAR_CONTRACT_REGISTRY_SCHEMA = SCHEMA_ROOT / "quantized_linear_contract_registry.schema.json"
LINEAR_KERNEL_CAPABILITY_SCHEMA = SCHEMA_ROOT / "quantized_linear_kernel_capability.schema.json"
VALID_STATES = {"unresolved", "observed", "validated"}
AMBIGUOUS_IDS = {"fp16", "f16", "bf16", "fp32", "f32", "fast", "strict"}


class ContractError(RuntimeError):
    pass


def hard_contract_fault(summary: str, detail: str, remediation: str) -> ContractError:
    return ContractError(
        "HARD CONTRACT FAULT: " + summary + "\n"
        "  " + detail + "\n"
        "  Fix: " + remediation + "\n"
        "  Do not add a fallback, silent default, tolerance relaxation, or validation bypass."
    )


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except FileNotFoundError as exc:
        raise ContractError(f"Contract input does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ContractError(f"Expected a JSON object in {path}")
    return doc


def validate_schema(instance: Dict[str, Any], schema_path: Path, context: str) -> None:
    schema = load_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(instance),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    location = ".".join(str(part) for part in error.absolute_path) or "<root>"
    raise hard_contract_fault(
        f"{context} violates {schema_path.name}",
        f"At {location}: {error.message}",
        "correct the circuit, reduction registry, or kernel map so it satisfies the versioned schema.",
    )


def load_kernel_capabilities(root: Path = DEFAULT_KERNELS) -> Dict[str, Any]:
    if not root.is_dir():
        raise ContractError(f"Kernel-map directory does not exist: {root}")
    load_kernel_execution_capabilities(root)
    kernels: Dict[str, Any] = {}
    for path in sorted(root.glob("*.json")):
        doc = load_json(path)
        if doc.get("physical_alias_of"):
            continue
        if "supported_reductions" not in doc and "provides" not in doc:
            continue
        kernel_id = str(doc.get("id", "")).strip()
        if not kernel_id:
            raise ContractError(f"Numerical kernel map has no id: {path}")
        capability = dict(doc)
        capability["base_kernel_map"] = str(path.resolve().relative_to(REPO_ROOT.resolve()))
        kernels[kernel_id] = capability
    if not kernels:
        raise ContractError(f"No numerical kernel capabilities found under: {root}")
    return {
        "schema": "cke.kernel_numerical_contracts",
        "schema_version": 1,
        "engine_contract_version": "8",
        "kernels": kernels,
    }


def load_kernel_execution_capabilities(root: Path = DEFAULT_KERNELS) -> Dict[str, Any]:
    if not root.is_dir():
        raise ContractError(f"Kernel-map directory does not exist: {root}")
    kernels: Dict[str, Any] = {}
    linear_contracts = load_json(DEFAULT_LINEAR_CONTRACTS)
    validate_quantized_linear_contract_registry(linear_contracts)
    for path in sorted(root.glob("*.json")):
        doc = load_json(path)
        if doc.get("physical_alias_of"):
            continue
        if "contract_schema_version" not in doc:
            continue
        kernel_id = str(doc.get("id", "")).strip()
        if not kernel_id:
            raise hard_contract_fault(
                "versioned kernel map has no id",
                f"File: {path}",
                "add a stable kernel-map id.",
            )
        if kernel_id in kernels:
            raise hard_contract_fault(
                f"duplicate kernel capability id {kernel_id!r}",
                f"Files: {kernels[kernel_id]['source']} and {path}",
                "give each executable provider a unique id.",
            )
        capability = {
            "id": kernel_id,
            "op": doc.get("op"),
            "contract_schema_version": doc.get("contract_schema_version"),
            "implementation": doc.get("implementation"),
        }
        # Provider lifecycle metadata flows through so GraphIR resolved_execution
        # can disclose it; absent for legacy (unmigrated) providers.
        if doc.get("selection") is not None:
            capability["selection"] = doc.get("selection")
        validate_schema(capability, KERNEL_EXECUTION_SCHEMA, f"kernel execution capability {kernel_id}")
        if capability["op"] in {"gemm", "gemv"}:
            linear_capability = {
                **capability,
                "numerical_contract": doc.get("numerical_contract"),
                "reference": doc.get("reference"),
                "production": doc.get("production"),
                "impl": doc.get("impl"),
            }
            validate_schema(
                linear_capability,
                LINEAR_KERNEL_CAPABILITY_SCHEMA,
                f"quantized linear kernel capability {kernel_id}",
            )
            validate_quantized_linear_kernel_capability(linear_capability, linear_contracts)
            capability.update(
                numerical_contract=linear_capability["numerical_contract"],
                reference=linear_capability["reference"],
                production=linear_capability["production"],
            )
            contract = linear_contracts["contracts"][linear_capability["numerical_contract"]]
            implementation = capability["implementation"]
            weight_storage = implementation.get("weight_storage")
            activation_storage = implementation.get("activation_storage")
            diagnostic_providers = implementation.get("diagnostic_providers")
            if not isinstance(weight_storage, dict) or not isinstance(activation_storage, dict):
                raise hard_contract_fault(
                    f"quantized linear kernel {kernel_id!r} has no explicit storage capability",
                    "Code generation must not recover weight or activation layout from a function name.",
                    "declare implementation.weight_storage and implementation.activation_storage.",
                )
            if (
                weight_storage["format"] != contract["weight"]["format"]
                or weight_storage["block_elements"] != contract["weight"]["block_size"]
                or activation_storage["format"] != contract["activation"]["format"]
                or activation_storage["block_elements"] != contract["activation"]["block_size"]
            ):
                raise hard_contract_fault(
                    f"quantized linear kernel {kernel_id!r} storage capability disagrees with its contract",
                    f"implementation={implementation}, contract={contract}",
                    "make the implementation storage metadata match the numerical contract.",
                )
            if not isinstance(diagnostic_providers, dict):
                raise hard_contract_fault(
                    f"quantized linear kernel {kernel_id!r} has no diagnostic providers",
                    "Code generation must not derive an FP32 or row-quantized provider from a function name.",
                    "declare implementation.diagnostic_providers in the kernel map.",
                )
            if capability["op"] == "gemm" and not diagnostic_providers.get("row_quantized"):
                raise hard_contract_fault(
                    f"quantized GEMM kernel {kernel_id!r} has no row-quantized provider",
                    "The bounded prefill diagnostic path requires an exact map-owned row provider.",
                    "declare implementation.diagnostic_providers.row_quantized in the kernel map.",
                )
        kernels[kernel_id] = {**capability, "source": str(path)}
    if not kernels:
        raise ContractError(f"No versioned kernel execution capabilities found under: {root}")
    return {
        "schema": "cke.kernel_execution_capabilities",
        "schema_version": 1,
        "engine_contract_version": "8",
        "kernels": kernels,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_keys(node: Dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in node]
    if missing:
        raise ContractError(f"{context} is missing required fields: {', '.join(missing)}")


def validate_state(value: Any, context: str) -> str:
    state = str(value or "").strip()
    if state not in VALID_STATES:
        raise ContractError(
            f"{context} has invalid validation state {state!r}; expected one of {sorted(VALID_STATES)}"
        )
    return state


def validate_contract_registry(doc: Dict[str, Any]) -> None:
    validate_schema(doc, CONTRACT_REGISTRY_SCHEMA, "attention reduction registry")
    require_keys(doc, ("contracts", "required_semantic_fields"), "attention contract registry")
    contracts = doc["contracts"]
    fields = doc["required_semantic_fields"]
    if not isinstance(contracts, dict) or not contracts:
        raise ContractError("attention contract registry must define at least one contract")
    if not isinstance(fields, list) or not fields:
        raise ContractError("required_semantic_fields must be a non-empty list")
    for contract_id, contract in contracts.items():
        if contract_id.lower() in AMBIGUOUS_IDS:
            raise ContractError(f"Ambiguous reduction contract ID is forbidden: {contract_id}")
        if not isinstance(contract, dict):
            raise ContractError(f"Reduction contract {contract_id} must be an object")
        require_keys(contract, fields, f"reduction contract {contract_id}")
        validate_state(contract.get("status"), f"reduction contract {contract_id}")
        partition = contract.get("partition")
        if not isinstance(partition, dict) or not partition.get("kind"):
            raise ContractError(f"reduction contract {contract_id}.partition requires kind")
        if partition.get("kind") == "query_tile_threshold":
            for route in ("below_threshold", "at_or_above_threshold"):
                referenced_id = partition[route]
                referenced = contracts.get(referenced_id)
                if not isinstance(referenced, dict):
                    raise hard_contract_fault(
                        f"reduction contract {contract_id!r} references missing {route} contract {referenced_id!r}",
                        "A shape-dependent dispatch route has no registered arithmetic definition.",
                        "Register and validate the exact arithmetic contract before selecting it.",
                    )
                if validate_state(
                    referenced.get("status"),
                    f"reduction contract {referenced_id}",
                ) != "validated":
                    raise hard_contract_fault(
                        f"reduction contract {contract_id!r} references unvalidated {route} contract {referenced_id!r}",
                        "A composite provider cannot be stronger than either selected arithmetic contract.",
                        "Promote the referenced contract only after an independent numerical oracle passes.",
                    )


def validate_quantized_linear_contract_registry(doc: Dict[str, Any]) -> None:
    validate_schema(doc, LINEAR_CONTRACT_REGISTRY_SCHEMA, "quantized linear contract registry")
    for contract_id, contract in doc["contracts"].items():
        validate_state(contract.get("status"), f"quantized linear contract {contract_id}")
        parallel = contract["parallel_reduction"]
        if parallel["kind"] == "independent_outputs" and parallel["reduction_order_effect"] != "none":
            raise hard_contract_fault(
                f"independent-output contract {contract_id!r} changes reduction order",
                "Independent outputs do not share an accumulator.",
                "set reduction_order_effect to none or declare a split_k contract.",
            )
        if parallel["kind"] == "split_k" and (
            "partial_accumulator" not in parallel or "merge_order" not in parallel
        ):
            raise hard_contract_fault(
                f"split-K contract {contract_id!r} is incomplete",
                "partial_accumulator and merge_order are required for split-K.",
                "declare the complete partitioned scalar-oracle semantics.",
            )


def validate_quantized_linear_kernel_capability(
    kernel: Dict[str, Any],
    registry: Dict[str, Any],
) -> None:
    kernel_id = kernel["id"]
    contract_id = kernel["numerical_contract"]
    contract = registry["contracts"].get(contract_id)
    if contract is None:
        raise hard_contract_fault(
            f"kernel {kernel_id!r} names unknown numerical contract {contract_id!r}",
            f"Registry: {DEFAULT_LINEAR_CONTRACTS}",
            "add and validate the complete contract before binding the kernel.",
        )
    production = kernel["production"]
    reference = kernel["reference"]
    if reference["kind"] == "scalar_contract_oracle":
        if not reference["function"].endswith("_ref"):
            raise hard_contract_fault(
                f"kernel {kernel_id!r} scalar oracle is not a reference function",
                f"reference.function={reference['function']!r}",
                "bind a scalar _ref function or declare and validate a graph oracle.",
            )
    elif reference["kind"] == "llama_repacked_graph_oracle":
        external = reference["validation"]["external_oracles"]
        validated_llama = any(
            item.get("backend") == "llama.cpp_ggml_cpu_graph"
            and item.get("status") == "validated"
            for item in external
        )
        comparison = production["reference_comparison"]
        if not validated_llama or comparison["requirement"] != "bit_exact":
            raise hard_contract_fault(
                f"kernel {kernel_id!r} has an unproven loaded-model graph oracle",
                "A repacked provider requires validated llama.cpp graph evidence and bit-exact comparison.",
                "add a real production-provider oracle; do not substitute a leaf tolerance test.",
            )
    impl_function = kernel["impl"]["function"]
    if production["function"] != impl_function:
        raise hard_contract_fault(
            f"kernel {kernel_id!r} production function drifts from impl.function",
            f"production.function={production['function']!r}, impl.function={impl_function!r}",
            "bind both fields to the exact public function emitted by codegen.",
        )
    threading = kernel["implementation"]["threading"]
    if threading["runtime"] == "ck_threadpool" and not production.get("threaded_function"):
        raise hard_contract_fault(
            f"threaded kernel {kernel_id!r} has no threadpool entry point",
            "A threadpool implementation must name the exact dispatch function.",
            "set production.threaded_function and test it against the scalar oracle.",
        )
    if threading["reduction_order_effect"] != contract["parallel_reduction"]["reduction_order_effect"]:
        raise hard_contract_fault(
            f"kernel {kernel_id!r} threading contradicts {contract_id!r}",
            "The kernel map and numerical contract disagree about reduction-order effects.",
            "correct the map or define a distinct numerical contract.",
        )
    runtime_dispatch = kernel["implementation"].get("runtime_dispatch")
    runtime_policies = {}
    if runtime_dispatch is not None:
        runtime_policies = runtime_dispatch["policies"]
        supported = set(threading["work_partition"])
        for policy_id, policy in runtime_policies.items():
            routes = policy["routes"]
            for index, route in enumerate(routes):
                for axis in ("m", "n", "k"):
                    exact = route.get(axis)
                    lo = route.get(f"min_{axis}", 1)
                    hi = route.get(f"max_{axis}")
                    if exact is not None and (
                        f"min_{axis}" in route or f"max_{axis}" in route
                    ):
                        raise hard_contract_fault(
                            f"kernel {kernel_id!r} runtime policy {policy_id!r} mixes exact and ranged {axis.upper()} bounds",
                            f"route_index={index}, route={route}",
                            "declare either the exact dimension or its min/max interval.",
                        )
                    if hi is not None and hi < lo:
                        raise hard_contract_fault(
                            f"kernel {kernel_id!r} runtime policy {policy_id!r} has an invalid {axis.upper()} range",
                            f"route_index={index}, min={lo}, max={hi}",
                            "make every runtime dispatch interval non-empty.",
                        )
            if policy.get("requires_fallback"):
                fallback = routes[-1]
                covers_all_shapes = all(
                    axis not in fallback
                    and f"max_{axis}" not in fallback
                    and int(fallback.get(f"min_{axis}", 1)) == 1
                    for axis in ("m", "n", "k")
                )
                if not covers_all_shapes:
                    raise hard_contract_fault(
                        f"kernel {kernel_id!r} runtime policy {policy_id!r} has no final fallback",
                        f"final_route={fallback}",
                        "end a total runtime schedule with a route covering every positive M/N/K shape.",
                    )
            for route in routes:
                if "output_tiles" in route.get("flags", []) and "output_tiles" not in supported:
                    raise hard_contract_fault(
                        f"kernel {kernel_id!r} runtime policy {policy_id!r} selects unsupported output tiles",
                        f"supported={sorted(supported)}",
                        "advertise and certify output-tile partitioning before selecting it.",
                    )

    routing = kernel["implementation"].get("work_partition_routing")
    if routing is not None and routing["selection"] == "generated_runtime_dispatch":
        if routing["dispatch_function"] != production["threaded_function"]:
            raise hard_contract_fault(
                f"kernel {kernel_id!r} work-partition router names the wrong dispatch function",
                f"routing={routing['dispatch_function']!r}, production={production['threaded_function']!r}",
                "route every performance partition through the exact resolved threaded provider.",
            )
        policy_id = routing.get("policy")
        if policy_id not in runtime_policies:
            raise hard_contract_fault(
                f"kernel {kernel_id!r} work-partition router references an unknown generated policy",
                f"policy={policy_id!r}, available={sorted(runtime_policies)}",
                "bind work-partition routing to one implementation.runtime_dispatch policy.",
            )
        routing = None
    if routing is not None:
        if routing["dispatch_function"] != production["threaded_function"]:
            raise hard_contract_fault(
                f"kernel {kernel_id!r} work-partition router names the wrong dispatch function",
                f"routing={routing['dispatch_function']!r}, production={production['threaded_function']!r}",
                "route every performance partition through the exact resolved threaded provider.",
            )
        supported = set(threading["work_partition"])
        seen_ids = set()
        fallback_indexes = []
        bounded = []
        for index, route in enumerate(routing["routes"]):
            route_id = route["id"]
            if route_id in seen_ids:
                raise hard_contract_fault(
                    f"kernel {kernel_id!r} repeats work-partition route {route_id!r}",
                    "Route IDs must identify one measured implementation choice.",
                    "give every shape route a unique ID.",
                )
            seen_ids.add(route_id)
            if route["work_partition"] not in supported:
                raise hard_contract_fault(
                    f"kernel {kernel_id!r} routes to unsupported work partition {route['work_partition']!r}",
                    f"supported={sorted(supported)}",
                    "advertise and test the partition before making it shape-eligible.",
                )
            predicate = route["predicate"]
            if predicate.get("fallback") is True:
                if len(predicate) != 1:
                    raise hard_contract_fault(
                        f"kernel {kernel_id!r} fallback route also declares shape bounds",
                        f"route={route}",
                        "make fallback unconditional and place it last.",
                    )
                fallback_indexes.append(index)
            else:
                for axis in ("m", "n", "k"):
                    lo = predicate.get(f"min_{axis}", 1)
                    hi = predicate.get(f"max_{axis}")
                    if hi is not None and hi < lo:
                        raise hard_contract_fault(
                            f"kernel {kernel_id!r} has an invalid {axis.upper()} route range",
                            f"route={route_id!r}, min={lo}, max={hi}",
                            "make every bounded shape interval non-empty.",
                        )
                bounded.append(route)
        if fallback_indexes != [len(routing["routes"]) - 1]:
            raise hard_contract_fault(
                f"kernel {kernel_id!r} does not have exactly one final fallback route",
                f"fallback_indexes={fallback_indexes}",
                "end the route table with one unconditional fallback.",
            )
        for left_index, left in enumerate(bounded):
            for right in bounded[left_index + 1:]:
                overlaps = True
                for axis in ("m", "n", "k"):
                    left_lo = left["predicate"].get(f"min_{axis}", 1)
                    left_hi = left["predicate"].get(f"max_{axis}", float("inf"))
                    right_lo = right["predicate"].get(f"min_{axis}", 1)
                    right_hi = right["predicate"].get(f"max_{axis}", float("inf"))
                    if left_hi < right_lo or right_hi < left_lo:
                        overlaps = False
                        break
                if overlaps:
                    raise hard_contract_fault(
                        f"kernel {kernel_id!r} has ambiguous work-partition shape routes",
                        f"routes={left['id']!r},{right['id']!r}",
                        "make non-fallback shape predicates disjoint.",
                    )


def validate_kernel_overlay(doc: Dict[str, Any]) -> None:
    kernels = doc.get("kernels")
    if not isinstance(kernels, dict) or not kernels:
        raise ContractError("kernel numerical contract overlay must define kernels")
    for kernel_id, kernel in kernels.items():
        if not isinstance(kernel, dict):
            raise ContractError(f"Kernel capability {kernel_id} must be an object")
        require_keys(
            kernel,
            ("op", "mode", "base_kernel_map", "provides", "supported_reductions"),
            f"kernel capability {kernel_id}",
        )
        map_path = (REPO_ROOT / str(kernel["base_kernel_map"])).resolve()
        base_map = load_json(map_path)
        if base_map.get("id") != kernel_id:
            raise ContractError(
                f"Kernel capability {kernel_id} points to map with id {base_map.get('id')!r}: {map_path}"
            )
        base_function = (base_map.get("impl") or {}).get("function")
        validate_schema(
            {
                "id": kernel_id,
                "op": kernel.get("op"),
                "contract_schema_version": kernel.get("contract_schema_version"),
                "implementation": kernel.get("implementation"),
            },
            KERNEL_EXECUTION_SCHEMA,
            f"kernel execution capability {kernel_id}",
        )
        validate_schema(
            {
                "id": kernel_id,
                "op": kernel.get("op"),
                "mode": kernel.get("mode"),
                "contract_schema_version": kernel.get("contract_schema_version"),
                "provides": kernel.get("provides"),
                "supported_reductions": kernel.get("supported_reductions"),
                "implementation": kernel.get("implementation"),
                "impl": {"function": base_function},
            },
            KERNEL_CAPABILITY_SCHEMA,
            f"kernel capability {kernel_id}",
        )
        query_tiling = kernel.get("implementation", {}).get("query_tiling")
        if query_tiling is not None:
            routes = query_tiling["routes"]
            expected_min = 1
            functions = set()
            for index, route in enumerate(routes):
                route_min = int(route["min_tokens"])
                route_max = route["max_tokens"]
                if route_min != expected_min:
                    raise hard_contract_fault(
                        f"kernel {kernel_id!r} has a query-tile coverage gap or overlap",
                        f"route {index} starts at {route_min}, expected {expected_min}",
                        "make query-tile token ranges contiguous and start at token 1.",
                    )
                if route_max is None:
                    if index != len(routes) - 1:
                        raise hard_contract_fault(
                            f"kernel {kernel_id!r} has an unbounded non-final query-tile route",
                            f"route {index} has max_tokens=null",
                            "only the final query-tile route may be unbounded.",
                        )
                else:
                    route_max = int(route_max)
                    if route_max < route_min:
                        raise hard_contract_fault(
                            f"kernel {kernel_id!r} has an invalid query-tile range",
                            f"route {index} spans {route_min}..{route_max}",
                            "set max_tokens greater than or equal to min_tokens.",
                        )
                    expected_min = route_max + 1
                function = str(route["function"])
                if function in functions:
                    raise hard_contract_fault(
                        f"kernel {kernel_id!r} repeats a query-tile leaf provider",
                        f"function {function!r} appears more than once",
                        "give each measured query-tile route one exact leaf function.",
                    )
                functions.add(function)
            if routes[-1]["max_tokens"] is not None:
                raise hard_contract_fault(
                    f"kernel {kernel_id!r} leaves large token counts unresolved",
                    "the final query-tile route has a finite max_tokens",
                    "end the final validated route with max_tokens=null.",
                )
        routing = kernel.get("implementation", {}).get("work_partition_routing")
        if routing is not None and routing.get("selection") == "runtime_shape_topology":
            base_routing = (base_map.get("implementation") or {}).get(
                "work_partition_routing"
            )
            if routing != base_routing:
                raise hard_contract_fault(
                    f"kernel {kernel_id!r} runtime route drifts from its kernel map",
                    "The registry overlay and generator-owned map must be identical.",
                    "regenerate the registry from the kernel maps.",
                )
            if routing["dispatch_function"] != base_function:
                raise hard_contract_fault(
                    f"kernel {kernel_id!r} runtime route names the wrong ABI",
                    f"routing={routing['dispatch_function']!r}, impl={base_function!r}",
                    "route through the exact function emitted by codegen.",
                )
            bounded_routes = [
                route
                for route in routing["routes"]
                if route["predicate"].get("fallback") is not True
            ]
            if len(bounded_routes) != 1:
                raise hard_contract_fault(
                    f"kernel {kernel_id!r} has an ambiguous topology ABI",
                    f"bounded_routes={len(bounded_routes)}",
                    "use one bounded route per explicit map-owned ABI.",
                )
            route = bounded_routes[0]
            predicate_fields = (
                "num_heads", "num_kv_heads", "head_dim", "query_tokens",
                "min_kv_tokens", "workers",
            )
            configuration_fields = (
                "query_tile_size", "concurrent_query_tiles",
            )
            missing = [
                name for name in predicate_fields if name not in route["predicate"]
            ] + [
                name
                for name in configuration_fields
                if name not in route.get("configuration", {})
            ]
            if missing:
                raise hard_contract_fault(
                    f"kernel {kernel_id!r} has an incomplete topology route",
                    f"missing={missing}",
                    "declare the complete shape, worker, and tile configuration.",
                )
            expected_sources = {
                "route_num_heads": route["predicate"]["num_heads"],
                "route_num_kv_heads": route["predicate"]["num_kv_heads"],
                "route_head_dim": route["predicate"]["head_dim"],
                "route_query_tokens": route["predicate"]["query_tokens"],
                "route_min_kv_tokens": route["predicate"]["min_kv_tokens"],
                "route_workers": route["predicate"]["workers"],
                "route_query_tile_size": route["configuration"]["query_tile_size"],
                "route_concurrent_query_tiles": route["configuration"][
                    "concurrent_query_tiles"
                ],
            }
            call_sources = {
                param["name"]: param["source"]
                for param in (base_map.get("call_abi") or {}).get("params", [])
            }
            for name, value in expected_sources.items():
                expected = f"const:{value}"
                if call_sources.get(name) != expected:
                    raise hard_contract_fault(
                        f"kernel {kernel_id!r} runtime route is not ABI-owned by its map",
                        f"{name}: expected {expected!r}, got {call_sources.get(name)!r}",
                        "emit every topology predicate and tile choice as an exact map constant.",
                    )
        provides = kernel["provides"]
        if not isinstance(provides, dict) or not provides:
            raise ContractError(f"Kernel capability {kernel_id}.provides must be a non-empty object")
        for capability, values in provides.items():
            if not isinstance(values, list) or not values:
                raise ContractError(
                    f"Kernel capability {kernel_id}.provides[{capability!r}] must be a non-empty list"
                )
        supported = kernel["supported_reductions"]
        if not isinstance(supported, dict) or not supported:
            raise ContractError(f"Kernel capability {kernel_id} must support at least one reduction")
        for reduction_id, implementation in supported.items():
            if not isinstance(implementation, dict):
                raise ContractError(f"Kernel implementation {kernel_id}/{reduction_id} must be an object")
            require_keys(
                implementation,
                ("status", "function", "explicit_selector"),
                f"kernel implementation {kernel_id}/{reduction_id}",
            )
            if implementation["function"] != base_function:
                raise ContractError(
                    f"Kernel implementation {kernel_id}/{reduction_id} names function "
                    f"{implementation['function']!r}, but kernel map names {base_function!r}"
                )
            advertised = provides.get("numerics.attention_reduction", [])
            if reduction_id not in advertised:
                raise ContractError(
                    f"Kernel implementation {kernel_id}/{reduction_id} is not advertised by provides"
                )


def circuit_path(circuit: str) -> Path:
    return V8_ROOT / "circuits" / f"{circuit}.json"


def _capability_satisfies(provides: Dict[str, Any], requires: Dict[str, Any]) -> bool:
    for capability, required in requires.items():
        available = provides.get(capability)
        if not isinstance(available, list) or required not in available:
            return False
    return True


def resolve_contract(
    circuit_doc: Dict[str, Any],
    contract_doc: Dict[str, Any],
    kernel_doc: Dict[str, Any],
    *,
    operation: str,
    phase: str,
    mode: str,
    source_circuit_path: Path | None = None,
) -> Dict[str, Any]:
    validate_contract_registry(contract_doc)
    validate_kernel_overlay(kernel_doc)
    if mode not in {"bringup", "production"}:
        raise ContractError(f"Unknown resolution mode: {mode}")

    operations = circuit_doc.get("required_contracts")
    validate_schema(
        {"required_contracts": operations},
        CIRCUIT_REQUIREMENTS_SCHEMA,
        f"circuit {circuit_doc.get('name') or '<embedded>'} attention requirements",
    )
    if not isinstance(operations, dict) or operation not in operations:
        raise ContractError(f"Circuit does not declare operation contract: {operation}")
    operation_doc = operations[operation]
    if not isinstance(operation_doc, dict):
        raise ContractError(f"Circuit operation {operation} must be an object")
    require_keys(operation_doc, ("op", "template_ops", "phases"), f"circuit operation {operation}")
    template_ops = operation_doc["template_ops"]
    phases = operation_doc["phases"]
    if not isinstance(phases, dict) or phase not in phases:
        raise ContractError(f"Circuit operation {operation} does not declare phase: {phase}")
    request = phases[phase]
    if not isinstance(request, dict):
        raise ContractError(f"Circuit request {operation}.{phase} must be an object")
    require_keys(request, ("requires", "validation"), f"circuit request {operation}.{phase}")
    requires = request["requires"]
    if not isinstance(requires, dict) or not requires:
        raise ContractError(f"Circuit request {operation}.{phase}.requires must be a non-empty object")

    reduction_id = str(requires.get("numerics.attention_reduction", "")).strip()
    if reduction_id.lower() in AMBIGUOUS_IDS:
        raise ContractError(
            f"Circuit requested ambiguous reduction {reduction_id!r}; request a complete registered contract"
        )
    contracts = contract_doc["contracts"]
    if reduction_id not in contracts:
        raise ContractError(f"Unknown reduction contract requested: {reduction_id}")
    contract = contracts[reduction_id]

    kernels = kernel_doc.get("kernels")
    candidates = []
    for candidate_id, candidate in kernels.items():
        if candidate.get("op") != operation_doc["op"]:
            continue
        if str(candidate.get("mode", "")).strip() != phase:
            continue
        if not _capability_satisfies(candidate.get("provides", {}), requires):
            continue
        supported = candidate.get("supported_reductions")
        if isinstance(supported, dict) and reduction_id in supported:
            candidates.append((candidate_id, candidate))
    if not candidates:
        available = [
            f"{candidate_id}: {json.dumps(candidate.get('provides', {}), sort_keys=True)}"
            for candidate_id, candidate in kernels.items()
            if candidate.get("op") == operation_doc["op"]
        ]
        raise hard_contract_fault(
            f"no kernel provides {operation}.{phase}",
            "Required: " + json.dumps(requires, sort_keys=True)
            + ("; available attention providers: " + " | ".join(available) if available else "; no providers exist"),
            "correct the circuit requirement or add and validate a compatible executable kernel map.",
        )
    if len(candidates) > 1:
        raise hard_contract_fault(
            f"multiple kernels provide {operation}.{phase}",
            f"Matching providers: {[item[0] for item in candidates]}",
            "make kernel capabilities mutually exclusive or add an explicit semantic requirement that selects one provider.",
        )
    kernel_id, kernel = candidates[0]
    supported = kernel.get("supported_reductions")
    implementation = supported[reduction_id]

    request_state = validate_state(request.get("validation"), f"circuit request {operation}.{phase}")
    contract_state = validate_state(contract.get("status"), f"reduction contract {reduction_id}")
    implementation_state = validate_state(
        implementation.get("status"), f"kernel implementation {kernel_id}/{reduction_id}"
    )
    explicit_selector = bool(implementation.get("explicit_selector", False))

    blockers = []
    if request_state != "validated":
        blockers.append(f"circuit request is {request_state}")
    if contract_state != "validated":
        blockers.append(f"contract definition is {contract_state}")
    if implementation_state != "validated":
        blockers.append(f"kernel implementation is {implementation_state}")
    if not explicit_selector:
        blockers.append("kernel uses legacy implicit selection")
    if mode == "production" and blockers:
        raise ContractError(
            f"Production contract resolution rejected {operation}.{phase}: " + "; ".join(blockers)
        )

    source_path = source_circuit_path.resolve() if source_circuit_path else None

    result = {
        "schema": "cke.resolved_attention_contract",
        "schema_version": 1,
        "engine_contract_version": "8",
        "circuit": circuit_doc.get("name"),
        "operation": operation,
        "template_ops": template_ops,
        "phase": phase,
        "resolution_mode": mode,
        "kernel": {
            "id": kernel_id,
            "function": implementation.get("function"),
            "implementation_status": implementation_state,
            "explicit_selector": explicit_selector,
            "selector": implementation.get("selector")
        },
        "reduction": {
            "id": reduction_id,
            "definition_status": contract_state,
            "semantics": {key: contract[key] for key in contract_doc["required_semantic_fields"]}
        },
        "implementation": kernel["implementation"],
        "request_status": request_state,
        "requirements": requires,
        "production_blockers": blockers,
        "inputs": {
            "circuit": str(source_path) if source_path else None,
            "circuit_sha256": sha256_file(source_path) if source_path else None
        }
    }
    validate_schema(result, RESOLVED_CONTRACT_SCHEMA, f"resolved contract {operation}.{phase}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", required=True, help="Circuit name, for example qwen3vl")
    parser.add_argument("--operation", default="decoder.attention")
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--mode", choices=("bringup", "production"), default="bringup")
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--kernel-maps", type=Path, default=DEFAULT_KERNELS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_circuit_path = circuit_path(args.circuit)
    try:
        result = resolve_contract(
            load_json(source_circuit_path),
            load_json(args.contracts),
            load_kernel_capabilities(args.kernel_maps),
            operation=args.operation,
            phase=args.phase,
            mode=args.mode,
            source_circuit_path=source_circuit_path,
        )
    except ContractError as exc:
        print(f"v8 contract resolution: FAIL: {exc}")
        return 2

    rendered = json.dumps(result, indent=2 if args.pretty else None, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
