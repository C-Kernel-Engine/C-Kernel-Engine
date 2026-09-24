"""Emit a bounded, status-checked C entry point from resolved call IR.

The circuit and canonical kernel maps own the schedule and ABI. This emitter
only validates their resolved calls, carries named valid extents, and stops on
the first checked provider failure.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

from runtime_extent_contract_v8 import (
    RuntimeExtentContractError,
    normalize_runtime_extents,
)


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CheckedCallCodegenError(ValueError):
    pass


def _ident(value: str, where: str) -> str:
    if not isinstance(value, str) or not _IDENT.fullmatch(value):
        raise CheckedCallCodegenError(f"{where} must be a C identifier")
    return value


def _kernel_map(root: Path, kernel_id: str) -> dict:
    _ident(kernel_id, "kernel id")
    path = root / "version" / "v8" / "kernel_maps" / f"{kernel_id}.json"
    if not path.is_file():
        raise CheckedCallCodegenError(f"missing canonical map for {kernel_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def emit_checked_calls(call_ir: dict, root: Path) -> str:
    """Return C for call-ready IR with explicit caller-owned entry buffers.

    The input is the generic call-ready operation representation emitted by
    IR Lower 3, plus a small `entry` declaration for the host ABI. No provider
    selection or model-family behavior occurs here.
    """
    entry = call_ir.get("entry")
    ops = call_ir.get("operations")
    contract = call_ir.get("runtime_extent_contract")
    if not isinstance(entry, dict) or not isinstance(ops, list) or not ops:
        raise CheckedCallCodegenError("entry and nonempty operations are required")
    if not isinstance(contract, dict):
        raise CheckedCallCodegenError("runtime_extent_contract is required")
    function_name = _ident(entry.get("function"), "entry function")
    params = entry.get("params")
    if not isinstance(params, list) or not params:
        raise CheckedCallCodegenError("entry params are required")
    declarations = []
    parameter_types = {}
    for param in params:
        if not isinstance(param, dict):
            raise CheckedCallCodegenError("entry param must be an object")
        name = _ident(param.get("name"), "entry parameter")
        c_type = param.get("c_type")
        if not isinstance(c_type, str) or not re.fullmatch(
            r"(?:const )?(?:void|float|int32_t|size_t|uint8_t|int)\s*\**", c_type
        ):
            raise CheckedCallCodegenError(f"invalid C type for {name}")
        declarations.append(f"{c_type} {name}")
        if name in parameter_types:
            raise CheckedCallCodegenError(f"duplicate entry parameter {name}")
        parameter_types[name] = c_type
    try:
        contract = normalize_runtime_extents(contract, [
            {"op_id": op.get("template_op_id"),
             "produces_runtime_lengths": op.get("produces_runtime_lengths", {}),
             "consumes_runtime_lengths": op.get("consumes_runtime_lengths", []),
             "returns_status": op.get("returns_status", False)}
            for op in ops
        ])
    except RuntimeExtentContractError as error:
        raise CheckedCallCodegenError(str(error)) from error
    lengths = contract.get("runtime_lengths", {})
    if not isinstance(lengths, dict) or not lengths:
        raise CheckedCallCodegenError("named runtime lengths are required")
    length_outputs = entry.get("runtime_length_outputs", {})
    if not isinstance(length_outputs, dict):
        raise CheckedCallCodegenError("runtime_length_outputs must be an object")
    for name, parameter_name in length_outputs.items():
        if name not in lengths or parameter_types.get(parameter_name) != "int32_t *":
            raise CheckedCallCodegenError(f"invalid host output for runtime length {name}")
    lines = [
        "/* Generated from resolved CKE call IR; caller owns all buffers. */",
        '#include <stddef.h>',
        '#include <stdint.h>',
    ]
    arena_spec = entry.get("arena")
    arena_check = []
    if arena_spec is not None:
        if not isinstance(arena_spec, dict):
            raise CheckedCallCodegenError("entry arena must be an object")
        pointer = arena_spec.get("pointer")
        capacity = arena_spec.get("bytes")
        if parameter_types.get(pointer) != "uint8_t *" or parameter_types.get(capacity) != "size_t":
            raise CheckedCallCodegenError("entry arena requires uint8_t pointer and size_t capacity")
        memory = call_ir.get("memory", {})
        arena = memory.get("arena", {}) if isinstance(memory, dict) else {}
        total = arena.get("total_size") if isinstance(arena, dict) else None
        if not isinstance(total, int) or total <= 0:
            raise CheckedCallCodegenError("call IR lacks a positive planned arena size")
        # Offsets are relative to the caller's base. Keep the base aligned to
        # the strongest selected provider requirement, including scratch.
        alignment = arena.get("alignment", 64)
        if not isinstance(alignment, int) or alignment < 1:
            raise CheckedCallCodegenError("arena alignment must be a positive power of two")
        for op in ops:
            kernel = _kernel_map(root, (op.get("call_abi") or {}).get("kernel_id"))
            for port_kind in ("inputs", "outputs", "weights", "scratch"):
                for port in kernel.get(port_kind, []):
                    port_alignment = port.get("alignment", 1)
                    if not isinstance(port_alignment, int) or port_alignment < 1 or port_alignment & (port_alignment - 1):
                        raise CheckedCallCodegenError("provider alignment must be a positive power of two")
                    alignment = max(alignment, port_alignment)
        if not isinstance(alignment, int) or alignment < 1 or alignment & (alignment - 1):
            raise CheckedCallCodegenError("arena alignment must be a positive power of two")
        buffers = memory.get("activations", {}).get("buffers", [])
        if not isinstance(buffers, list):
            raise CheckedCallCodegenError("call IR lacks planned activation buffers")
        seen_defines = set()
        for buffer in buffers:
            define = _ident(buffer.get("define"), "planned buffer define")
            offset = buffer.get("abs_offset")
            size = buffer.get("size")
            if define in seen_defines or not isinstance(offset, int) or not isinstance(size, int):
                raise CheckedCallCodegenError("invalid or duplicate planned buffer")
            if offset < 0 or size < 0 or offset + size > total:
                raise CheckedCallCodegenError(f"planned buffer {define} exceeds arena")
            lines.append(f"#define {define} {offset}u")
            seen_defines.add(define)
        lines.append("typedef struct CKCheckedArenaModel { uint8_t *bump; } CKCheckedArenaModel;")
        arena_check = [
            f"    if (!{pointer} || {capacity} < {total}u) return -2;",
            f"    if (((uintptr_t){pointer} & {alignment - 1}u) != 0u) return -2;",
            f"    CKCheckedArenaModel model_storage = {{{pointer}}};",
            "    CKCheckedArenaModel *model = &model_storage;",
        ]
    declarations_seen = set()
    map_docs = []
    for op in ops:
        if not isinstance(op, dict) or op.get("errors"):
            raise CheckedCallCodegenError("call IR has unresolved provider errors")
        if op.get("returns_status") is not True:
            raise CheckedCallCodegenError("bounded graph requires checked provider status")
        kernel_id = (op.get("call_abi") or {}).get("kernel_id")
        kernel = _kernel_map(root, kernel_id)
        if not kernel.get("modes", {}).get("inference", False):
            raise CheckedCallCodegenError(f"{kernel_id} is not an inference provider")
        if kernel.get("impl", {}).get("function") != op.get("function"):
            raise CheckedCallCodegenError(f"resolved function differs from {kernel_id} map")
        declaration = kernel.get("impl", {}).get("c_declaration", "")
        if not declaration.startswith("int "):
            raise CheckedCallCodegenError(f"{kernel_id} does not return checked status")
        if declaration not in declarations_seen:
            lines.append(declaration)
            declarations_seen.add(declaration)
        map_docs.append(kernel)
    lines += [
        "typedef struct CKRuntimeExtents {",
    ]
    for name in lengths:
        lines.append(f"    int32_t {_ident(name, 'runtime length')};")
    lines += [
        "} CKRuntimeExtents;",
        f"int {function_name}({', '.join(declarations)}) {{",
        "    CKRuntimeExtents runtime_extents = {0};",
        "    int status = 0;",
    ]
    lines.extend(arena_check)
    for parameter_name in length_outputs.values():
        lines.append(f"    if (!{parameter_name}) return -1;")
    produced = set()
    for index, (op, kernel) in enumerate(zip(ops, map_docs)):
        expected = [p["name"] for p in kernel["call_abi"]["params"]]
        expected_sources = [p["source"] for p in kernel["call_abi"]["params"]]
        args = op.get("args")
        if not isinstance(args, list) or [a.get("name") for a in args] != expected:
            raise CheckedCallCodegenError(f"call {index} differs from canonical ABI order")
        if [a.get("source") for a in args] != expected_sources:
            raise CheckedCallCodegenError(f"call {index} differs from canonical ABI sources")
        for name in op.get("consumes_runtime_lengths", []):
            if name not in produced:
                raise CheckedCallCodegenError(f"call {index} consumes unproduced length {name}")
        arg_exprs = []
        for arg in args:
            expr = arg.get("expr")
            if not isinstance(expr, str) or not expr.strip():
                raise CheckedCallCodegenError(f"call {index} has unresolved argument")
            arg_exprs.append(expr)
        lines.append(f"    status = {op['function']}({', '.join(arg_exprs)});")
        lines.append("    if (status != 0) return status;")
        for name, result in op.get("produces_runtime_lengths", {}).items():
            if name not in lengths or name in produced:
                raise CheckedCallCodegenError(f"invalid runtime length producer {name}")
            if not any(a.get("source") == f"output:{result}" and
                       a.get("expr") == f"&runtime_extents.{name}" for a in args):
                raise CheckedCallCodegenError(f"producer {name} is not bound to output {result}")
            capacity = lengths[name]["capacity"]
            lines.append(f"    if (runtime_extents.{name} < 0 || "
                         f"(size_t)runtime_extents.{name} > {capacity}u) return -2;")
            if not lengths[name].get("allow_zero", False):
                lines.append(f"    if (runtime_extents.{name} == 0) return -1;")
            produced.add(name)
    if produced != set(lengths):
        raise CheckedCallCodegenError("not every declared runtime length is produced")
    for name, parameter_name in length_outputs.items():
        lines.append(f"    *{parameter_name} = runtime_extents.{name};")
    lines += ["    return 0;", "}", ""]
    return "\n".join(lines)
