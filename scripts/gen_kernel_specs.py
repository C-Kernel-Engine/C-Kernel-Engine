#!/usr/bin/env python3
"""Generate kernel registry C sources from version/v6.6/legacy_kernel_maps/*.json."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_DIR = os.path.join(ROOT, "version", "v6.6", "legacy_kernel_maps", "kernels")
PLAN_PATH = os.path.join(ROOT, "version", "v6.6", "legacy_kernel_maps", "decoder_layer_plan.json")
PLAN_BWD_PATH = os.path.join(ROOT, "version", "v6.6", "legacy_kernel_maps", "decoder_layer_plan_backward.json")
GLOBAL_PATH = os.path.join(ROOT, "version", "v6.6", "legacy_kernel_maps", "global_buffers.json")
OUT_H = os.path.join(ROOT, "include", "ckernel_kernel_specs.h")
OUT_C = os.path.join(ROOT, "src", "ckernel_kernel_specs.c")

MAX_KERNEL_SOURCES = 8

DIM_ENUM = {
    "tokens": "CK_DIM_TOKENS",
    "embed": "CK_DIM_EMBED",
    "aligned_embed": "CK_DIM_ALIGNED_EMBED",
    "head_dim": "CK_DIM_HEAD_DIM",
    "aligned_head": "CK_DIM_ALIGNED_HEAD",
    "num_heads": "CK_DIM_NUM_HEADS",
    "num_kv_heads": "CK_DIM_NUM_KV_HEADS",
    "aligned_ctx": "CK_DIM_ALIGNED_CTX",
    "intermediate": "CK_DIM_INTERMEDIATE",
    "aligned_intermediate": "CK_DIM_ALIGNED_INTERMEDIATE",
    "vocab": "CK_DIM_VOCAB",
}

ROLE_ENUM = {
    "input": "CK_ROLE_INPUT",
    "output": "CK_ROLE_OUTPUT",
    "activation": "CK_ROLE_ACTIVATION",
    "weight": "CK_ROLE_WEIGHT",
    "scratch": "CK_ROLE_SCRATCH",
    "grad": "CK_ROLE_GRAD",
}

SCOPE_ENUM = {
    "layer": "CK_SCOPE_LAYER",
    "global": "CK_SCOPE_GLOBAL",
}

DEFAULT_BUFFER_DTYPE = "CK_DT_FP32"
DEFAULT_KERNEL_DTYPE_MASK = "CK_DT_MASK(CK_DT_FP32)"
DEFAULT_KERNEL_DEFAULT_DTYPE = "CK_DT_FP32"

DT_ORDER = [
    "fp32",
    "bf16",
    "fp16",
    "int8",
    "int4",
]

DT_ENUM = {
    "fp32": "CK_DT_FP32",
    "bf16": "CK_DT_BF16",
    "fp16": "CK_DT_FP16",
    "int8": "CK_DT_INT8",
    "int4": "CK_DT_INT4",
}

DT_INDEX = {dt: idx for idx, dt in enumerate(DT_ORDER)}


def normalize_dtype(dtype, context):
    if not dtype:
        raise SystemExit(f"missing dtype in {context}")
    key = dtype.lower()
    if key not in DT_ENUM:
        raise SystemExit(f"unknown dtype '{dtype}' in {context}")
    return key


def build_dtype_mask(dtypes):
    if not dtypes:
        return DEFAULT_KERNEL_DTYPE_MASK
    entries = []
    seen = set()
    for dt in dtypes:
        if dt in seen:
            continue
        seen.add(dt)
        entries.append(f"CK_DT_MASK({DT_ENUM[dt]})")
    return " | ".join(entries) if entries else DEFAULT_KERNEL_DTYPE_MASK


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_role(role, context):
    if role not in ROLE_ENUM:
        raise SystemExit(f"unknown role '{role}' in {context}")


def validate_scope(scope, context):
    if scope not in SCOPE_ENUM:
        raise SystemExit(f"unknown scope '{scope}' in {context}")


def load_kernel_maps():
    specs = {}
    for name in sorted(os.listdir(KERNEL_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(KERNEL_DIR, name)
        data = load_json(path)
        kernel_name = data.get("name")
        if not kernel_name:
            raise SystemExit(f"kernel map missing name: {path}")
        if kernel_name in specs:
            raise SystemExit(f"duplicate kernel name: {kernel_name}")
        specs[kernel_name] = data
    return specs


def load_plan(path):
    if not os.path.exists(path):
        return {"steps": []}
    return load_json(path)


def extract_plan_steps(plan, kernels):
    steps = []
    for step in plan.get("steps", []):
        kernel_name = step.get("kernel")
        if not kernel_name:
            raise SystemExit("plan step missing kernel")
        if kernel_name not in kernels:
            raise SystemExit(f"unknown kernel in plan: {kernel_name}")
        condition = step.get("when") or step.get("condition")
        steps.append((kernel_name, condition))
    return steps


def extract_plan_steps_with_bind(plan, kernels):
    steps = []
    for step in plan.get("steps", []):
        kernel_name = step.get("kernel")
        if not kernel_name:
            raise SystemExit("plan step missing kernel")
        if kernel_name not in kernels:
            raise SystemExit(f"unknown kernel in plan: {kernel_name}")
        condition = step.get("when") or step.get("condition")
        bindings = step.get("bind", {}) or {}
        steps.append({"kernel": kernel_name, "condition": condition, "bind": bindings})
    return steps


def select_kernel_buffers(kernel, plan_kind):
    if plan_kind == "backward":
        return kernel.get("buffers_backward") or kernel.get("buffers") or []
    return kernel.get("buffers") or []


def norm_shape(shape):
    tokens = []
    for item in shape:
        if isinstance(item, str):
            dim = item
            mult = 1
            div = 1
        else:
            dim = item.get("dim")
            mult = int(item.get("mult", 1))
            div = int(item.get("div", 1))
        if dim not in DIM_ENUM:
            raise SystemExit(f"unknown dim '{dim}' in shape")
        if mult <= 0 or div <= 0:
            raise SystemExit(f"invalid mult/div for dim '{dim}': {mult}/{div}")
        tokens.append((dim, mult, div))
    return tuple(tokens)


def merge_role(prev_role, new_role):
    if prev_role == new_role:
        return prev_role
    role_rank = {
        "input": 0,
        "output": 1,
        "activation": 2,
        "scratch": 2,
        "grad": 3,
        "weight": 4,
    }
    if prev_role not in role_rank or new_role not in role_rank:
        raise SystemExit(f"unknown role merge: {prev_role} vs {new_role}")
    return prev_role if role_rank[prev_role] >= role_rank[new_role] else new_role


def add_buffer(buffers, order, name, spec):
    if name in buffers:
        prev = buffers[name]
        if prev[0] != spec[0]:
            raise SystemExit(f"buffer scope mismatch for '{name}': {prev[0]} vs {spec[0]}")
        if prev[2] != spec[2]:
            raise SystemExit(f"buffer shape mismatch for '{name}': {prev[2]} vs {spec[2]}")
        merged_role = merge_role(prev[1], spec[1])
        merged_optional = 1 if prev[3] or spec[3] else 0
        merged_alias = prev[4] if prev[4] else spec[4]
        merged_cond = prev[5] if prev[5] else spec[5]
        buffers[name] = (prev[0], merged_role, prev[2], merged_optional, merged_alias, merged_cond)
        return
    buffers[name] = spec
    order.append(name)


def main():
    kernels = load_kernel_maps()
    plan = load_plan(PLAN_PATH)
    plan_bwd = load_plan(PLAN_BWD_PATH)
    globals_spec = load_json(GLOBAL_PATH)

    buffers = {}
    order = []

    for buf in globals_spec.get("buffers", []):
        name = buf.get("name")
        if not name:
            raise SystemExit("global buffer missing name")
        role = buf.get("role", "activation")
        scope = buf.get("scope", "global")
        validate_role(role, f"global buffer '{name}'")
        validate_scope(scope, f"global buffer '{name}'")
        shape = norm_shape(buf.get("shape", []))
        optional = int(bool(buf.get("optional", False)))
        alias_of = buf.get("alias_of")
        condition = buf.get("condition")
        spec = (scope, role, shape, optional, alias_of, condition)
        add_buffer(buffers, order, name, spec)

    def merge_plan_buffers(plan_obj, plan_kind):
        for step in plan_obj.get("steps", []):
            kernel_name = step.get("kernel")
            if kernel_name not in kernels:
                raise SystemExit(f"unknown kernel in plan: {kernel_name}")
            kernel = kernels[kernel_name]
            impl = kernel.get("impl", {}) or {}
            if plan_kind == "backward" and not impl.get("backward"):
                raise SystemExit(f"kernel '{kernel_name}' missing backward impl for backward plan")
            bindings = step.get("bind", {})
            for buf in select_kernel_buffers(kernel, plan_kind):
                buf_id = buf.get("id")
                if not buf_id:
                    raise SystemExit(f"kernel '{kernel_name}' buffer missing id")
                if buf_id not in bindings:
                    if buf.get("optional"):
                        continue
                    raise SystemExit(f"kernel '{kernel_name}' missing bind for '{buf_id}'")
                name = bindings[buf_id]
                if not name:
                    continue
                role = buf.get("role", "activation")
                scope = buf.get("scope", "layer")
                validate_role(role, f"kernel '{kernel_name}' buffer '{buf_id}'")
                validate_scope(scope, f"kernel '{kernel_name}' buffer '{buf_id}'")
                shape = norm_shape(buf.get("shape", []))
                optional = int(bool(buf.get("optional", False)))
                alias_of = buf.get("alias_of")
                condition = buf.get("condition")
                spec = (scope, role, shape, optional, alias_of, condition)
                add_buffer(buffers, order, name, spec)

    merge_plan_buffers(plan, "forward")
    merge_plan_buffers(plan_bwd, "backward")

    kernel_specs = []
    for name in sorted(kernels.keys()):
        kernel = kernels[name]
        impl = kernel.get("impl", {}) or {}
        forward = impl.get("forward")
        backward = impl.get("backward")
        sources = impl.get("sources", []) or []
        if isinstance(sources, str):
            sources = [sources]
        if len(sources) > MAX_KERNEL_SOURCES:
            raise SystemExit(f"kernel '{name}' has too many sources (max {MAX_KERNEL_SOURCES})")

        dtype_names = []
        raw_dtypes = kernel.get("dtypes") or []
        for dt in raw_dtypes:
            norm = normalize_dtype(dt, f"kernel '{name}' dtypes")
            if norm not in dtype_names:
                dtype_names.append(norm)

        variants = impl.get("variants", {}) or {}
        for variant in variants.keys():
            norm = normalize_dtype(variant, f"kernel '{name}' variants")
            if norm not in dtype_names:
                dtype_names.append(norm)

        if not dtype_names:
            dtype_names = ["fp32"]

        default_dtype = kernel.get("default_dtype")
        default_norm = normalize_dtype(default_dtype, f"kernel '{name}' default_dtype") if default_dtype else dtype_names[0]
        if default_norm not in dtype_names:
            dtype_names.insert(0, default_norm)

        forward_by_dtype = [None] * len(DT_ORDER)
        backward_by_dtype = [None] * len(DT_ORDER)
        for dt in dtype_names:
            idx = DT_INDEX[dt]
            forward_by_dtype[idx] = forward
            backward_by_dtype[idx] = backward

        for variant_name, variant_obj in variants.items():
            variant_dt = normalize_dtype(variant_name, f"kernel '{name}' variant")
            idx = DT_INDEX[variant_dt]
            variant_forward = variant_obj.get("forward")
            variant_backward = variant_obj.get("backward")
            if variant_forward:
                forward_by_dtype[idx] = variant_forward
            if variant_backward:
                backward_by_dtype[idx] = variant_backward

        dtype_mask = build_dtype_mask(dtype_names)
        kernel_specs.append({
            "name": name,
            "forward_by_dtype": forward_by_dtype,
            "backward_by_dtype": backward_by_dtype,
            "dtype_mask": dtype_mask,
            "default_dtype": DT_ENUM[default_norm],
            "sources": sources
        })

    plan_fwd_steps = extract_plan_steps(plan, kernels)
    plan_bwd_steps = extract_plan_steps(plan_bwd, kernels)
    plan_fwd_bind = extract_plan_steps_with_bind(plan, kernels)
    plan_bwd_bind = extract_plan_steps_with_bind(plan_bwd, kernels)

    with open(OUT_H, "w", encoding="utf-8") as f:
        f.write("#ifndef CKERNEL_KERNEL_SPECS_H\n")
        f.write("#define CKERNEL_KERNEL_SPECS_H\n\n")
        f.write("#include <stddef.h>\n\n")
        f.write("#include \"ckernel_dtype.h\"\n\n")
        f.write("typedef enum {\n")
        f.write("    CK_DIM_TOKENS = 0,\n")
        f.write("    CK_DIM_EMBED,\n")
        f.write("    CK_DIM_ALIGNED_EMBED,\n")
        f.write("    CK_DIM_HEAD_DIM,\n")
        f.write("    CK_DIM_ALIGNED_HEAD,\n")
        f.write("    CK_DIM_NUM_HEADS,\n")
        f.write("    CK_DIM_NUM_KV_HEADS,\n")
        f.write("    CK_DIM_ALIGNED_CTX,\n")
        f.write("    CK_DIM_INTERMEDIATE,\n")
        f.write("    CK_DIM_ALIGNED_INTERMEDIATE,\n")
        f.write("    CK_DIM_VOCAB,\n")
        f.write("    CK_DIM_END\n")
        f.write("} CKDimKind;\n\n")

        f.write("#define CKERNEL_MAX_KERNEL_SOURCES %d\n\n" % MAX_KERNEL_SOURCES)

        f.write("typedef struct {\n")
        f.write("    CKDimKind dim;\n")
        f.write("    int mult;\n")
        f.write("    int div;\n")
        f.write("} CKDimToken;\n\n")

        f.write("typedef enum {\n")
        f.write("    CK_SCOPE_LAYER = 0,\n")
        f.write("    CK_SCOPE_GLOBAL\n")
        f.write("} CKBufferScope;\n\n")

        f.write("typedef enum {\n")
        f.write("    CK_ROLE_INPUT = 0,\n")
        f.write("    CK_ROLE_OUTPUT,\n")
        f.write("    CK_ROLE_ACTIVATION,\n")
        f.write("    CK_ROLE_WEIGHT,\n")
        f.write("    CK_ROLE_SCRATCH,\n")
        f.write("    CK_ROLE_GRAD\n")
        f.write("} CKBufferRole;\n\n")

        f.write("typedef struct {\n")
        f.write("    const char *name;\n")
        f.write("    CKBufferScope scope;\n")
        f.write("    CKBufferRole role;\n")
        f.write("    CKDimToken shape[4];\n")
        f.write("    int optional;\n")
        f.write("    const char *alias_of;\n")
        f.write("    const char *condition;\n")
        f.write("    CKDataType dtype;\n")
        f.write("} CKBufferSpec;\n\n")

        f.write("typedef struct {\n")
        f.write("    const char *name;\n")
        f.write("    const char *forward[CK_DT_COUNT];\n")
        f.write("    const char *backward[CK_DT_COUNT];\n")
        f.write("    CKDataTypeMask dtype_mask;\n")
        f.write("    CKDataType default_dtype;\n")
        f.write("    const char *sources[CKERNEL_MAX_KERNEL_SOURCES];\n")
        f.write("} CKKernelSpec;\n\n")

        f.write("typedef struct {\n")
        f.write("    const char *kernel;\n")
        f.write("    const char *condition;\n")
        f.write("} CKPlanStep;\n\n")

        f.write("typedef struct {\n")
        f.write("    const char *arg;\n")
        f.write("    const char *buffer;\n")
        f.write("} CKPlanBinding;\n\n")

        f.write("typedef struct {\n")
        f.write("    const char *kernel;\n")
        f.write("    const char *condition;\n")
        f.write("    const CKPlanBinding *bindings;\n")
        f.write("    size_t num_bindings;\n")
        f.write("} CKPlanStepV2;\n\n")

        f.write("extern const CKBufferSpec ck_decoder_buffers[];\n")
        f.write("extern const size_t ck_decoder_buffer_count;\n\n")
        f.write("extern const CKKernelSpec ck_kernel_specs[];\n")
        f.write("extern const size_t ck_kernel_spec_count;\n\n")
        f.write("extern const CKPlanStep ck_decoder_forward_plan[];\n")
        f.write("extern const size_t ck_decoder_forward_plan_count;\n\n")
        f.write("extern const CKPlanStep ck_decoder_backward_plan[];\n")
        f.write("extern const size_t ck_decoder_backward_plan_count;\n\n")
        f.write("extern const CKPlanStepV2 ck_decoder_forward_plan_v2[];\n")
        f.write("extern const size_t ck_decoder_forward_plan_v2_count;\n\n")
        f.write("extern const CKPlanStepV2 ck_decoder_backward_plan_v2[];\n")
        f.write("extern const size_t ck_decoder_backward_plan_v2_count;\n\n")
        f.write("#endif /* CKERNEL_KERNEL_SPECS_H */\n")

    with open(OUT_C, "w", encoding="utf-8") as f:
        f.write('#include "ckernel_kernel_specs.h"\n\n')
        f.write("const CKBufferSpec ck_decoder_buffers[] = {\n")
        for name in order:
            scope, role, shape, optional, alias_of, condition = buffers[name]
            scope_c = SCOPE_ENUM[scope]
            role_c = ROLE_ENUM[role]
            tokens = []
            for dim, mult, div in shape:
                dim_c = DIM_ENUM[dim]
                tokens.append((dim_c, mult, div))
            while len(tokens) < 4:
                tokens.append(("CK_DIM_END", 0, 0))
            shape_c = ", ".join("{ %s, %d, %d }" % t for t in tokens)
            alias_c = f"\"{alias_of}\"" if alias_of else "NULL"
            cond_c = f"\"{condition}\"" if condition else "NULL"
            dtype_c = DEFAULT_BUFFER_DTYPE
            f.write("    {\"%s\", %s, %s, { %s }, %d, %s, %s, %s},\n" % (
                name, scope_c, role_c, shape_c, optional, alias_c, cond_c, dtype_c
            ))
        f.write("};\n\n")
        f.write("const size_t ck_decoder_buffer_count = sizeof(ck_decoder_buffers) / sizeof(ck_decoder_buffers[0]);\n")

        f.write("\nconst CKKernelSpec ck_kernel_specs[] = {\n")
        for spec in kernel_specs:
            name = spec["name"]
            forward_entries = spec["forward_by_dtype"]
            backward_entries = spec["backward_by_dtype"]
            mask_c = spec["dtype_mask"]
            default_c = spec["default_dtype"]
            srcs = list(spec["sources"])
            while len(srcs) < MAX_KERNEL_SOURCES:
                srcs.append(None)
            srcs = srcs[:MAX_KERNEL_SOURCES]
            src_items = []
            for s in srcs:
                src_items.append(f"\"{s}\"" if s else "NULL")
            src_c = ", ".join(src_items)
            forward_c = ", ".join(f"\"{fn}\"" if fn else "NULL" for fn in forward_entries)
            backward_c = ", ".join(f"\"{bn}\"" if bn else "NULL" for bn in backward_entries)
            f.write("    {\"%s\", { %s }, { %s }, %s, %s, { %s }},\n" % (
                name, forward_c, backward_c, mask_c, default_c, src_c
            ))
        f.write("};\n\n")
        f.write("const size_t ck_kernel_spec_count = sizeof(ck_kernel_specs) / sizeof(ck_kernel_specs[0]);\n")

        f.write("\nconst CKPlanStep ck_decoder_forward_plan[] = {\n")
        for kernel_name, condition in plan_fwd_steps:
            cond_c = f"\"{condition}\"" if condition else "NULL"
            f.write("    {\"%s\", %s},\n" % (kernel_name, cond_c))
        f.write("};\n\n")
        f.write("const size_t ck_decoder_forward_plan_count = sizeof(ck_decoder_forward_plan) / sizeof(ck_decoder_forward_plan[0]);\n")

        f.write("\nconst CKPlanStep ck_decoder_backward_plan[] = {\n")
        for kernel_name, condition in plan_bwd_steps:
            cond_c = f"\"{condition}\"" if condition else "NULL"
            f.write("    {\"%s\", %s},\n" % (kernel_name, cond_c))
        f.write("};\n\n")
        f.write("const size_t ck_decoder_backward_plan_count = sizeof(ck_decoder_backward_plan) / sizeof(ck_decoder_backward_plan[0]);\n")

        def emit_plan_bindings(steps, prefix):
            bindings_info = []
            for idx, step in enumerate(steps):
                binds = step.get("bind") or {}
                if not binds:
                    bindings_info.append(("NULL", 0))
                    continue
                arr_name = f"{prefix}_bindings_{idx}"
                f.write(f"\nstatic const CKPlanBinding {arr_name}[] = {{\n")
                for arg, buf in binds.items():
                    f.write(f"    {{\"{arg}\", \"{buf}\"}},\n")
                f.write("};\n")
                bindings_info.append((arr_name, len(binds)))
            return bindings_info

        fwd_bindings = emit_plan_bindings(plan_fwd_bind, "ck_decoder_forward")
        bwd_bindings = emit_plan_bindings(plan_bwd_bind, "ck_decoder_backward")

        f.write("\nconst CKPlanStepV2 ck_decoder_forward_plan_v2[] = {\n")
        for idx, step in enumerate(plan_fwd_bind):
            cond = step.get("condition")
            cond_c = "NULL" if not cond else f"\"{cond}\""
            arr_name, count = fwd_bindings[idx]
            f.write(f"    {{\"{step['kernel']}\", {cond_c}, {arr_name}, {count}}},\n")
        f.write("};\n\n")
        f.write("const size_t ck_decoder_forward_plan_v2_count = sizeof(ck_decoder_forward_plan_v2) / sizeof(ck_decoder_forward_plan_v2[0]);\n")

        f.write("\nconst CKPlanStepV2 ck_decoder_backward_plan_v2[] = {\n")
        for idx, step in enumerate(plan_bwd_bind):
            cond = step.get("condition")
            cond_c = "NULL" if not cond else f"\"{cond}\""
            arr_name, count = bwd_bindings[idx]
            f.write(f"    {{\"{step['kernel']}\", {cond_c}, {arr_name}, {count}}},\n")
        f.write("};\n\n")
        f.write("const size_t ck_decoder_backward_plan_v2_count = sizeof(ck_decoder_backward_plan_v2) / sizeof(ck_decoder_backward_plan_v2[0]);\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
