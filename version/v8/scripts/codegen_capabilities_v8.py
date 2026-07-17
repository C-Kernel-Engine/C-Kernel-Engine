"""Read resolved code-generation capabilities without interpreting C symbols."""

from __future__ import annotations

from typing import Any, Dict


def resolved_quantized_linear_emission(op: Dict[str, Any]) -> Dict[str, Any] | None:
    execution = op.get("resolved_execution")
    if not isinstance(execution, dict) or not execution.get("numerical_contract"):
        return None
    implementation = execution.get("implementation")
    if not isinstance(implementation, dict):
        raise RuntimeError("resolved quantized linear operation has no implementation metadata")
    weight = implementation.get("weight_storage")
    activation = implementation.get("activation_storage")
    diagnostics = implementation.get("diagnostic_providers")
    if not all(isinstance(value, dict) for value in (weight, activation, diagnostics)):
        raise RuntimeError(
            "resolved quantized linear operation is missing map-owned storage or diagnostic providers"
        )
    result = {
        "weight_format": str(weight["format"]),
        "weight_block_elements": int(weight["block_elements"]),
        "weight_block_bytes": int(weight["block_bytes"]),
        "activation_format": str(activation["format"]),
        "activation_block_elements": int(activation["block_elements"]),
        "fp32_activation_function": str(diagnostics["fp32_activation"]),
    }
    row_provider = diagnostics.get("row_quantized")
    if row_provider is not None:
        result["row_quantized_function"] = str(row_provider)
    return result


def is_q4_q6_q8_linear(capability: Dict[str, Any] | None) -> bool:
    return bool(
        capability
        and capability["weight_format"] in {"q4_k", "q6_k"}
        and capability["activation_format"] == "q8_k"
    )


def resolved_activation_quantization_emission(op: Dict[str, Any]) -> Dict[str, Any] | None:
    capability = op.get("resolved_codegen_capability")
    if not isinstance(capability, dict):
        return None
    if capability.get("operator_family") != "activation_quantization":
        return None
    function = str(op.get("function", "") or "")
    if capability.get("function") != function:
        raise RuntimeError(
            "resolved activation quantization capability does not match the call-ready function"
        )
    storage = capability.get("output_storage")
    if not isinstance(storage, dict):
        raise RuntimeError("resolved activation quantization capability has no output storage")
    required = {"format", "block_elements", "block_elements_symbol", "c_block_type"}
    if set(storage) != required:
        raise RuntimeError(
            "resolved activation quantization storage must define exact format and block ABI"
        )
    result = {
        "format": str(storage["format"]),
        "block_elements": int(storage["block_elements"]),
        "block_elements_symbol": str(storage["block_elements_symbol"]),
        "c_block_type": str(storage["c_block_type"]),
    }
    rounding_contract = capability.get("rounding_contract")
    if rounding_contract is not None:
        result["rounding_contract"] = str(rounding_contract)
    prefill_batch = capability.get("prefill_batch")
    if prefill_batch is not None:
        if not isinstance(prefill_batch, dict):
            raise RuntimeError("prefill activation quantization capability must be an object")
        required_batch = {
            "function",
            "row_group",
            "tail_function",
            "rounding_contract",
        }
        if set(prefill_batch) != required_batch:
            raise RuntimeError(
                "prefill activation quantization capability must define exact "
                "function, row group, tail function, and rounding contract"
            )
        row_group = int(prefill_batch["row_group"])
        if row_group <= 1:
            raise RuntimeError("prefill activation quantization row_group must exceed one")
        result["prefill_batch"] = {
            "function": str(prefill_batch["function"]),
            "row_group": row_group,
            "tail_function": str(prefill_batch["tail_function"]),
            "rounding_contract": str(prefill_batch["rounding_contract"]),
        }
    return result


def activation_quantized_row_bytes_expr(
    capability: Dict[str, Any], dimension_expr: str
) -> str:
    return (
        f"(size_t)(({dimension_expr}) / {capability['block_elements_symbol']}) * "
        f"sizeof({capability['c_block_type']})"
    )
