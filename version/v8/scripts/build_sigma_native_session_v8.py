"""Author three generated Sigma entries in one arena with shared weights and prefix K/V."""
from version.v8.scripts.checked_circuit_authoring_v8 import execution_port
from version.v8.scripts.build_sigma_native_prefill_v8 import build_circuit as clean_encoder
from version.v8.scripts.build_sigma_native_step_v8 import build_circuit as denoise_step


def build_circuit(prompt_rows, block_rows, *, cache_capacity, latent_width=16, bands=32, **geometry):
    if (any(type(n) is not int or n <= 0 for n in (prompt_rows, block_rows, cache_capacity))
            or prompt_rows + block_rows > cache_capacity or cache_capacity + block_rows > 4096):
        raise ValueError("unsupported Sigma session capacities")
    graph, config, schema = denoise_step(block_rows, prefix_rows=cache_capacity,
                                          latent_width=latent_width, bands=bands, **geometry)
    graph["name"] = "sigma_native_session"
    step_entry = graph["native_entry"]["function"]
    denoise = graph["block_types"]["block"]
    graph["sequence"] = []
    graph["block_types"] = {}
    segments = []

    for kind, rows, prefix_rows in (("prefill", prompt_rows, 0), ("append", block_rows, cache_capacity)):
        clean, clean_config, clean_schema = clean_encoder(rows, prefix_rows=prefix_rows, **geometry)
        for name, spec in clean_schema.items():
            if schema.setdefault(name, spec) != spec:
                raise ValueError(f"shared Sigma weight differs: {name}")
        dtypes = clean_config["activation_buffer_dtypes"]

        def binding(name):
            if kind == "append" and "_prefix_" in name:
                return "step_cond_" + name
            return f"{kind}_{name}"

        for name, spec in clean["activation_buffers"].items():
            target = binding(name)
            if graph["activation_buffers"].setdefault(target, spec) != spec:
                raise ValueError(f"shared Sigma buffer differs: {target}")
            if config["activation_buffer_dtypes"].setdefault(target, dtypes[name]) != dtypes[name]:
                raise ValueError(f"shared Sigma dtype differs: {target}")
            graph["activation_bindings"][target] = target

        def source(tensor):
            name = tensor.removeprefix("external:")
            return ("external:" if name != tensor else "") + binding(name)

        block = clean["block_types"]["block"]
        ops = block["body"]["ops"]
        for op in ops:
            op["id"] = f"{kind}_{op['id']}"
            slots = op["graph_slots"]
            slots["inputs"] = {port: source(name) for port, name in slots["inputs"].items()}
            slots["outputs"] = {port: binding(name) for port, name in slots["outputs"].items()}
        graph["sequence"].append(kind)
        graph["block_types"][kind] = block
        segments.append({"function": f"ck_sigma_session_{kind}", "last_op": ops[-1]["id"]})

    graph["sequence"].append("denoise")
    graph["block_types"]["denoise"] = denoise
    segments.append({"function": step_entry, "last_op": denoise["body"]["ops"][-1]["id"]})
    graph["native_entry"].update(function=segments[0]["function"], segments=segments)
    ports = [f"{phase}_{name}" for phase in ("prefill", "append")
             for name in ("prefix_tokens", "positions", "ranges")]
    ports += ["step_" + name for name in (
        "positions", "cond_ranges", "uncond_ranges", "guidance", "latent",
        "next_latent", "noise", "gamma", "self_condition", "cond_mean", "cond_ids",
    )]
    for layer in range(geometry.get("layers", 26)):
        for cache, staged in (("key", "k_rotary"), ("value", "v")):
            ports.append(f"step_cond_layer{layer}_prefix_{cache}")
            ports.extend(f"{phase}_layer{layer}_{staged}" for phase in ("prefill", "append"))
    graph["execution_exports"] = {
        "schema": "cke.checked.execution_exports.v1",
        "ports": {name: execution_port(graph, name) for name in ports},
        "entries": {name: spec["function"] for name, spec in zip(("prefill", "append", "step"), segments, strict=True)},
        "properties": {"prompt_rows": prompt_rows, "block_rows": block_rows, "cache_capacity": cache_capacity},
    }
    return graph, config, schema
