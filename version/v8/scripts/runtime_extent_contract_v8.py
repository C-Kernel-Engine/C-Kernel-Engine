"""Model-neutral contract for bounded runtime tensor extents.

The circuit owns names and producer/consumer edges. This module validates and
normalizes those declarations; it does not schedule a model or allocate memory.
"""

from __future__ import annotations

import ctypes
from typing import Any


SIZE_MAX = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1


class RuntimeExtentContractError(ValueError):
    pass


def checked_product(*values: int) -> int:
    result = 1
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeExtentContractError("extent factors must be nonnegative integers")
        if value and result > SIZE_MAX // value:
            raise RuntimeExtentContractError("runtime tensor capacity overflows size_t")
        result *= value
    return result


def _positive(value: Any, where: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeExtentContractError(f"{where} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise RuntimeExtentContractError(f"{where} must be {'nonnegative' if allow_zero else 'positive'}")
    if value > SIZE_MAX:
        raise RuntimeExtentContractError(f"{where} exceeds size_t")
    return value


def normalize_runtime_extents(circuit: dict, operations: list[dict]) -> dict:
    """Validate declared length producers, consumers, and physical tensor views.

    Expected circuit keys:
      runtime_lengths: name -> {producer, result, capacity, allow_zero}
      runtime_views: tensor name -> {length, channels, physical_stride,
                                      element_bytes, buffer_bytes}
    Operations use stable op_id values and optional consumes_runtime_lengths.
    """
    raw_lengths = circuit.get("runtime_lengths", {})
    raw_views = circuit.get("runtime_views", {})
    raw_constants = circuit.get("runtime_constants", {})
    if (not isinstance(raw_lengths, dict) or not isinstance(raw_views, dict)
            or not isinstance(raw_constants, dict)):
        raise RuntimeExtentContractError(
            "runtime_lengths, runtime_views, and runtime_constants must be objects")
    constants = {}
    for name, value in raw_constants.items():
        if not isinstance(name, str) or not name or name in raw_lengths:
            raise RuntimeExtentContractError("runtime constant name is empty or shadows a length")
        constants[name] = _positive(value, f"runtime_constants.{name}", allow_zero=True)
    positions = {}
    by_id = {}
    for index, op in enumerate(operations):
        op_id = op.get("op_id")
        if not isinstance(op_id, str) or not op_id or op_id in positions:
            raise RuntimeExtentContractError("operations require unique nonempty string op_id values")
        positions[op_id] = index
        by_id[op_id] = op

    lengths = {}
    for name, spec in raw_lengths.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            raise RuntimeExtentContractError("runtime length names and declarations must be valid")
        producer = spec.get("producer")
        result = spec.get("result")
        if producer not in positions or not isinstance(result, str) or not result:
            raise RuntimeExtentContractError(f"runtime length {name} has no declared producer/result")
        produced = by_id[producer].get("produces_runtime_lengths", {})
        if not isinstance(produced, dict) or produced.get(name) != result:
            raise RuntimeExtentContractError(
                f"runtime length {name} is not exposed by producer {producer} as {result}")
        if by_id[producer].get("returns_status") is not True:
            raise RuntimeExtentContractError(
                f"runtime length producer {producer} must declare checked status")
        capacity = _positive(spec.get("capacity"), f"runtime_lengths.{name}.capacity")
        if capacity > (1 << 31) - 1:
            raise RuntimeExtentContractError(
                f"runtime_lengths.{name}.capacity exceeds int32 result range")
        allow_zero = spec.get("allow_zero", False)
        if not isinstance(allow_zero, bool):
            raise RuntimeExtentContractError(f"runtime_lengths.{name}.allow_zero must be boolean")
        lengths[name] = {"producer": producer, "result": result,
                         "capacity": capacity, "allow_zero": allow_zero}

    views = {}
    for name, spec in raw_views.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            raise RuntimeExtentContractError("runtime view names and declarations must be valid")
        length = spec.get("length")
        if length not in lengths:
            raise RuntimeExtentContractError(f"runtime view {name} names unknown length {length!r}")
        buffer = spec.get("buffer")
        if not isinstance(buffer, str) or not buffer:
            raise RuntimeExtentContractError(f"runtime view {name} requires an allocated buffer name")
        channels = _positive(spec.get("channels"), f"runtime_views.{name}.channels")
        stride = _positive(spec.get("physical_stride"), f"runtime_views.{name}.physical_stride")
        element_bytes = _positive(spec.get("element_bytes"), f"runtime_views.{name}.element_bytes")
        capacity = lengths[length]["capacity"]
        if stride < capacity:
            raise RuntimeExtentContractError(f"runtime view {name} stride is below capacity")
        required_elements = checked_product(channels - 1, stride) + capacity
        if required_elements > SIZE_MAX:
            raise RuntimeExtentContractError(f"runtime view {name} element extent overflows size_t")
        required_bytes = checked_product(required_elements, element_bytes)
        buffer_bytes = _positive(spec.get("buffer_bytes"), f"runtime_views.{name}.buffer_bytes")
        if buffer_bytes < required_bytes:
            raise RuntimeExtentContractError(f"runtime view {name} allocation is undersized")
        views[name] = {"buffer": buffer, "length": length, "channels": channels,
                       "physical_stride": stride, "element_bytes": element_bytes,
                       "required_bytes": required_bytes, "buffer_bytes": buffer_bytes}

    for op in operations:
        consumed = op.get("consumes_runtime_lengths", [])
        if not isinstance(consumed, list) or len(consumed) != len(set(consumed)):
            raise RuntimeExtentContractError(f"{op['op_id']} has invalid length dependencies")
        for name in consumed:
            if name not in lengths:
                raise RuntimeExtentContractError(f"{op['op_id']} consumes unknown length {name!r}")
            if positions[lengths[name]["producer"]] >= positions[op["op_id"]]:
                raise RuntimeExtentContractError(f"{op['op_id']} executes before length {name} is produced")

    return {"runtime_lengths": lengths, "runtime_views": views,
            "runtime_constants": constants}
