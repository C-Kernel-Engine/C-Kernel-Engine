"""Resolve the bounded decoder state that may be switched between sequences.

This contract is deliberately narrower than general model state. A generated
runtime may switch the declared KV buffer, token positions, and the two
explicitly planned recurrent-state buffers. Unknown persistent mutable state,
encoder state, and multimodal state remain ineligible.
"""

from __future__ import annotations

import re
from math import prod
from typing import Any


_RECURRENT_MACROS = {
    "A_RECURRENT_CONV_STATE": "recurrent_conv_state",
    "A_RECURRENT_SSM_STATE": "recurrent_ssm_state",
}
_RECURRENT_MACRO_PATTERN = re.compile(
    r"\b(?:A_RECURRENT_CONV_STATE|A_RECURRENT_SSM_STATE)\b"
)


def validate_sequence_reset_regions(layout: dict[str, Any]) -> list[dict[str, Any]]:
    """Prove every mutable sequence buffer can be cleared in the model arena.

    Reset safety must not depend on sequence-switch eligibility: hybrid models
    can own persistent state even when switching is deliberately unavailable.
    """
    memory = layout.get("memory", {})
    buffers = memory.get("activations", {}).get("buffers", [])
    if not isinstance(buffers, list):
        raise ValueError("activation buffers must be a list")
    targets = [row for row in buffers if isinstance(row, dict) and
               row.get("lifetime") == "sequence" and row.get("mutable") is True]
    if not targets:
        return []
    target_ids = {id(row) for row in targets}
    arena_size = memory.get("arena", {}).get("total_size")
    if type(arena_size) is not int or arena_size <= 0:
        raise ValueError("sequence reset requires a proven arena size")
    regions: list[tuple[int, int, dict[str, Any]]] = []
    for row in buffers:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("activation buffer has no name")
        start, size = row.get("abs_offset"), row.get("size")
        if type(start) is not int or type(size) is not int or start < 0 or size < 0:
            raise ValueError(f"{row['name']} has no proven arena range")
        if start > arena_size or size > arena_size - start:
            raise ValueError(f"{row['name']} exceeds the planned arena")
        if id(row) in target_ids and size == 0:
            raise ValueError(f"{row['name']} has no proven reset extent")
        regions.append((start, start + size, row))
    for start, end, row in regions:
        if id(row) not in target_ids:
            continue
        for other_start, other_end, other in regions:
            if other is row or other_start == other_end:
                continue
            if start < other_end and other_start < end:
                raise ValueError(f"{row['name']} overlaps {other['name']}")
    return targets


def recurrent_state_operand(arg: dict[str, Any]) -> tuple[str, int, str] | None:
    """Return a checked buffer name, literal byte offset, and active pointer.

    The returned expression has the same pointer type as the lowered operand.
    Arbitrary aliases and computed offsets do not qualify for state switching.
    """
    ref = arg.get("buffer_ref")
    if ref not in ("recurrent_conv_state", "recurrent_ssm_state"):
        return None
    macro = "A_" + ref.upper()
    match = re.fullmatch(
        rf"\((const float|float)\*\)\(model->bump \+ (?:{macro}|\({macro} \+ ([0-9]+)\))\)",
        str(arg.get("expr", "")),
    )
    if match is None:
        return None
    offset = int(match.group(2) or 0)
    tail = f" + {offset}" if offset else ""
    return ref, offset, f"({match.group(1)}*)((uint8_t*)model->{ref}{tail})"


def resolve_sequence_state_contract(
    layout: dict[str, Any], config: dict[str, Any],
    ops: list[dict[str, Any]] | None = None,
) -> dict[str, int] | None:
    """Return checked per-sequence regions for a text-only decoder, or none.

    Offsets and sizes come from the compiler's physical layout.  They are never
    inferred from model names or from the generated C text.
    """
    buffers = layout.get("memory", {}).get("activations", {}).get("buffers", [])
    if not isinstance(buffers, list):
        raise ValueError("activation buffers must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for row in buffers:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("activation buffer has no name")
        name = row["name"]
        if name in by_name:
            raise ValueError(f"duplicate activation buffer {name}")
        by_name[name] = row
    # A missing lifetime is unknown, not a call-local allocation. Older
    # layouts consequently cannot advertise sequence isolation by accident.
    if any(
        row.get("lifetime") not in {"call", "sequence", "model"}
        or not isinstance(row.get("mutable"), bool)
        for row in buffers
    ):
        return None
    if (
        bool(config.get("uses_cross_attention"))
        or bool(config.get("_template_uses_persistent_cross_kv_cache"))
        or bool(config.get("uses_vision"))
        or str(config.get("artifact_scope", "")).lower() == "encoder_only"
    ):
        return None
    kv = by_name.get("kv_cache")
    if kv is None or kv.get("lifetime") != "sequence" or kv.get("mutable") is not True:
        return None
    recurrent_names = ("recurrent_conv_state", "recurrent_ssm_state")
    recurrent = [by_name.get(name) for name in recurrent_names]
    if any(kind not in (None, "none", "full_attention") for kind in
           config.get("layer_recurrent_policy", [])) and recurrent[0] is None:
        return None
    if any(row is not None for row in recurrent) != all(row is not None for row in recurrent):
        return None
    if any(row is not None and (row.get("lifetime") != "sequence" or
                                row.get("mutable") is not True) for row in recurrent):
        return None
    if any(
        all(row is not state for state in (kv, *recurrent)) and (
            row.get("lifetime") == "sequence"
            or (row.get("lifetime") == "model" and row["mutable"])
        ) for row in buffers
    ):
        return None
    offset = kv.get("abs_offset")
    size = kv.get("size")
    if not isinstance(offset, int) or not isinstance(size, int) or offset < 0 or size <= 0:
        raise ValueError("KV cache requires a positive size and absolute offset")
    if offset % 64:
        raise ValueError("KV cache must preserve the planner's 64-byte alignment")
    arena_size = layout.get("memory", {}).get("arena", {}).get("total_size")
    if isinstance(arena_size, int) and offset + size > arena_size:
        raise ValueError("KV cache exceeds the planned arena")
    regions = [kv, *(row for row in recurrent if row is not None)]
    for row in regions:
        start, length = row.get("abs_offset"), row.get("size")
        if not isinstance(start, int) or not isinstance(length, int) or start < 0 or length <= 0:
            raise ValueError(f"{row['name']} requires a positive size and absolute offset")
        if start % 64:
            raise ValueError(f"{row['name']} must preserve the planner's 64-byte alignment")
        if isinstance(arena_size, int) and start + length > arena_size:
            raise ValueError(f"{row['name']} exceeds the planned arena")
        for name, other in by_name.items():
            if other is row:
                continue
            other_offset, other_size = other.get("abs_offset"), other.get("size")
            if isinstance(other_offset, int) and isinstance(other_size, int) and other_size > 0:
                if start < other_offset + other_size and other_offset < start + length:
                    raise ValueError(f"{row['name']} overlaps {name}")
    result = {"kv_offset": offset, "kv_bytes": size, "kv_alignment": 64}
    if recurrent[0] is not None:
        if not ops:
            return None
        layers = config.get("num_layers")
        if not isinstance(layers, int) or layers <= 0:
            return None
        strides: dict[str, int] = {}
        for name in recurrent_names:
            row = by_name[name]
            # The layout describes contiguous [layer, ...] FP32 state. Do not
            # infer a slice solely from total size: a differently shaped arena
            # could have the same byte count but different layer ownership.
            shape = row.get("shape")
            if not isinstance(shape, str) or row.get("dtype") != "fp32":
                return None
            dimensions = re.fullmatch(r"\[\s*([0-9]+(?:\s*,\s*[0-9]+)*)\s*\]", shape)
            if dimensions is None:
                return None
            extents = [int(value.strip()) for value in dimensions.group(1).split(",")]
            if (len(extents) < 2 or any(value <= 0 for value in extents)
                    or extents[0] != layers or prod(extents) * 4 != row["size"]):
                return None
            strides[name] = prod(extents[1:]) * 4
        seen: set[str] = set()
        for op in ops:
            for arg in op.get("args", []):
                ref = arg.get("buffer_ref")
                macros = set(_RECURRENT_MACRO_PATTERN.findall(str(arg.get("expr", ""))))
                # Every lowered access to either state macro must carry the
                # corresponding typed buffer reference, or codegen would
                # leave it attached to the shared model arena.
                if macros and (len(macros) != 1 or
                               ref != _RECURRENT_MACROS[next(iter(macros))]):
                    return None
                if ref not in recurrent_names:
                    continue
                seen.add(ref)
                operand = recurrent_state_operand(arg)
                layer = op.get("layer")
                if (operand is None or not isinstance(layer, int) or
                        not 0 <= layer < layers or
                        operand[1] != layer * strides[ref]):
                    return None
        if seen != set(recurrent_names):
            return None
        # Each caller arena is compact and aligned independently of the model
        # arena's absolute offsets. The compiler's region sizes remain exact.
        cursor = (size + 63) & ~63
        for name, row in zip(recurrent_names, recurrent):
            assert row is not None
            result[f"{name}_offset"] = row["abs_offset"]
            result[f"{name}_bytes"] = row["size"]
            result[f"{name}_arena_offset"] = cursor
            cursor = (cursor + row["size"] + 63) & ~63
        result["arena_bytes"] = cursor
    return result
