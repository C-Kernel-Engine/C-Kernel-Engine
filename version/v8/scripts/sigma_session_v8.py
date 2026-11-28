"""Host bindings for the checked generated Sigma arena and native sampler."""
import ctypes as C

SIZE, PTR = C.c_size_t, C.c_void_p
ENTRY = C.CFUNCTYPE(C.c_int, PTR, SIZE)
CANCEL = C.CFUNCTYPE(C.c_int, PTR)


class Plane(C.Structure):
    _fields_ = [(name, SIZE) for name in ("cache", "prefill_rows", "append_rows")]


class Clean(C.Structure):
    _fields_ = [("entry", PTR)] + [(name, SIZE) for name in ("token_ids", "positions", "ranges")]


STEP_FIELDS = ("latent", "next_latent", "noise", "gamma", "self_condition", "cond_mean", "cond_ids")


class Step(C.Structure):
    _fields_ = [(name, SIZE) for name in STEP_FIELDS]


class Session(C.Structure):
    _fields_ = [
        ("arena", PTR), ("arena_bytes", SIZE), ("prefill", Clean), ("append", Clean),
        ("step", PTR), ("step_bindings", Step),
        *((name, SIZE) for name in ("positions", "conditional_ranges", "unconditional_ranges", "guidance")),
        ("planes", C.POINTER(Plane)),
        *((name, SIZE) for name in ("plane_count", "kv_width", "capacity", "prompt_rows", "block_rows",
                                   "latent_elements", "vocabulary", "valid_rows")),
    ]


class Block(C.Structure):
    _fields_ = [
        *((name, PTR) for name in ("latent", "schedule", "noise")),
        *((name, SIZE) for name in ("latent_elements", "schedule_count", "noise_elements")),
        ("guidance", C.c_float), ("staged_ids", PTR), ("output_ids", PTR),
        ("staged_capacity", SIZE), ("output_capacity", SIZE), ("should_cancel", PTR), ("cancel_context", PTR),
    ]


class Seeded(C.Structure):
    _fields_ = [
        ("prompt_ids", PTR), ("prompt_rows", SIZE), ("schedule", PTR), ("schedule_count", SIZE),
        ("seed", C.c_uint64), ("token_budget", SIZE), ("guidance", C.c_float),
        ("workspace", PTR), ("workspace_elements", SIZE), ("staged_ids", PTR), ("output_ids", PTR),
        ("staged_capacity", SIZE), ("output_capacity", SIZE), ("generated_rows", SIZE),
        ("should_cancel", PTR), ("cancel_context", PTR),
    ]


def bind_session(folder, arena, library, geometry):
    """Keep the returned session and its arena, library, and plane storage alive."""
    from version.v8.scripts.checked_execution_exports_v8 import load_exports

    checked = load_exports(folder, library)
    if checked["arena_bytes"] != len(arena):
        raise ValueError("Sigma arena size differs from compiled layout")
    props = checked["properties"]
    prompt, block, capacity = (props[key] for key in ("prompt_rows", "block_rows", "cache_capacity"))
    kv_width = geometry["kv_heads"] * geometry["dim"]
    latent_width = geometry["latent_width"]

    def offset(name, dtype, shape):
        port = checked["ports"][name]
        if port["dtype"] != dtype or port["shape"] != shape:
            raise ValueError(f"Sigma execution port type or shape mismatch: {name}")
        return port["offset"]

    def entry(name):
        return C.cast(getattr(library, checked["entries"][name]), PTR)

    def clean(kind, rows):
        return Clean(entry(kind), offset(kind + "_prefix_tokens", "i32", [rows]),
                     offset(kind + "_positions", "i32", [rows]),
                     offset(kind + "_ranges", "i32", [rows, 4]))

    planes = []
    for layer in range(geometry["layers"]):
        for cache, staged in (("key", "k_rotary"), ("value", "v")):
            cache_name = f"step_cond_layer{layer}_prefix_{cache}"
            prefill_name, append_name = (f"{phase}_layer{layer}_{staged}" for phase in ("prefill", "append"))
            if (any(cache_name in writes for writes in checked["entry_writes"].values())
                    or prefill_name not in checked["entry_writes"][checked["entries"]["prefill"]]
                    or append_name not in checked["entry_writes"][checked["entries"]["append"]]):
                raise ValueError(f"invalid Sigma cache ownership: {cache_name}")

            def cache_offset(name, rows):
                return offset(name, "bf16", [rows, kv_width])

            planes.append(Plane(cache_offset(cache_name, capacity), cache_offset(prefill_name, prompt),
                                cache_offset(append_name, block)))
    plane_array = (Plane * len(planes))(*planes)
    latent = [block, latent_width]
    step_specs = (("fp64", latent), ("fp64", latent), ("fp64", latent), ("fp64", [2]),
                  ("fp32", latent), ("fp32", latent), ("i32", [block]))
    return Session(
        C.addressof(arena), len(arena), clean("prefill", prompt), clean("append", block), entry("step"),
        Step(*(SIZE(-1).value if name == "noise" and "step_noise" not in checked["ports"]
               else offset("step_" + name, dtype, shape)
               for name, (dtype, shape) in zip(STEP_FIELDS, step_specs, strict=True))),
        offset("step_positions", "i32", [block]), offset("step_cond_ranges", "i32", [block, 4]),
        offset("step_uncond_ranges", "i32", [block, 4]), offset("step_guidance", "fp32", [1]),
        plane_array, len(plane_array), kv_width, capacity, prompt, block, block * latent_width,
        geometry["vocabulary"], 0,
    ), checked["loaded_build_identity"]
