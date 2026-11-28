"""Connect a bounded Sigma denoising step with shared BF16 denoiser weights."""
from version.v8.scripts.build_sigma_bf16_denoiser_stack_v8 import build_circuit as denoiser
from version.v8.scripts.checked_circuit_authoring_v8 import linear


def build_circuit(rows, *, latent_width=16, bands=32, vocabulary=131072, prefix_rows=0, **geometry):
    if any(type(n) is not int or n < 1 for n in (latent_width, bands)):
        raise ValueError("positive latent and time-feature extents required")
    conditional, cond_config, schema = denoiser(rows, vocabulary=vocabulary, prefix_rows=prefix_rows, **geometry)
    unconditional, uc_config, uc_schema = denoiser(rows, vocabulary=vocabulary, **geometry)
    if schema != uc_schema:
        raise ValueError("guided branches require identical denoiser weights")

    width = cond_config["embed_dim"]
    graph = conditional.copy()
    graph.update(name="sigma_native_ddpm_step", activation_buffers={}, activation_bindings={})
    graph["native_entry"]["function"] = "ck_sigma_native_ddpm_step"
    config = {**cond_config, "activation_buffer_dtypes": {}}
    body = graph["block_types"]["block"]["body"]
    cond_ops = body["ops"]
    ops = body["ops"] = []
    latent_count, hidden_count = rows * latent_width, rows * width
    schema.update({
        "time_frequencies": ([bands], "F32"),
        "normalized_embedding": ([vocabulary, latent_width], "F32"),
    })

    def buffer(name, shape, dtype="fp32"):
        graph["activation_buffers"][name] = {"shape": list(shape)}
        graph["activation_bindings"][name] = name
        config["activation_buffer_dtypes"][name] = dtype

    def emit(name, op, inputs, outputs, geometry, constants, *, weights=None, provider=None):
        node = {
            "id": name, "op": op, "kernel": provider or op, "returns_status": True,
            "graph_slots": {"inputs": inputs, "outputs": outputs},
            "params": {**geometry, "call_constants": constants},
        }
        if weights:
            node["weight_refs"] = weights
        ops.append(node)

    def projection(name, source, dim, target, weights):
        weight_shape = ([target, dim], "F32")
        if schema.setdefault(weights, weight_shape) != weight_shape:
            raise ValueError(f"projection shape differs for {weights}")
        buffer(name, [rows, target])
        node = linear(name, source, name, dim, target, rows, name)
        node.update(op="linear_rows_checked_no_bias", kernel="linear_rows_checked_no_bias_f32",
                    weight_refs={"weight": weights})
        del node["params"]["call_constants"]["linear_bias_elements"]
        ops.append(node)

    def add(name, left, right):
        buffer(name, [rows, width])
        emit(name, "add_rows_checked", {"left": left, "right": right}, {"output": name},
             {"N": hidden_count}, dict.fromkeys(
                 ("left_elements", "right_elements", "output_elements", "elements"), hidden_count),
             provider="add_rows_checked_f32")

    for name in ("step_latent", "step_noise", "step_next_latent"):
        buffer(name, [rows, latent_width], "fp64")
    buffer("step_gamma", [2], "fp64")
    buffer("step_guidance", [1])
    buffer("step_positions", [rows], "i32")
    for name in ("step_self_condition", "step_normalized", "step_prior_input", "step_scaled_self"):
        buffer(name, [rows, latent_width])
    buffer("step_time_features", [rows, 2 * bands])
    emit("vp_inputs", "diffusion_vp_condition_inputs_checked", {
        "latent": "external:step_latent", "gamma": "external:step_gamma",
        "self_condition": "external:step_self_condition",
    }, {
        "normalized": "step_normalized", "prior_input": "step_prior_input",
        "scaled_self": "step_scaled_self", "time_features": "step_time_features",
    }, {"R": rows, "D": latent_width, "B": bands}, {
        "rows": rows, "width": latent_width, "bands": bands, "elements": latent_count,
        "time_elements": rows * 2 * bands, "gamma_elements": 2,
    }, weights={"frequencies": "time_frequencies"})

    projection("step_input_projection", "step_normalized", latent_width, width, "replaid_input_linear.weight")
    projection("step_self_projection", "step_scaled_self", latent_width, width, "replaid_selfcond_linear.weight")
    projection("step_time_projection", "step_time_features", 2 * bands, width, "replaid_gamma_linear.weight")
    projection("step_prior", "step_prior_input", latent_width, vocabulary, "normalized_embedding")
    add("step_latent_plus_self", "step_input_projection", "step_self_projection")
    add("step_cond_float", "step_latent_plus_self", "step_time_projection")
    add("step_uncond_float", "step_input_projection", "step_time_projection")

    def append_branch(label, shapes, dtypes, nodes):
        prefix = f"step_{label}_"
        for name, spec in shapes.items():
            if name != "positions":
                buffer(prefix + name, spec["shape"], dtypes[name])
        emit(prefix + "cast", "narrow_f32_bf16_checked", {"input": prefix + "float"},
             {"out": prefix + "input"}, {"N": hidden_count}, {"elements": hidden_count})

        def source(name):
            if name == "external:positions":
                return "external:step_positions"
            if name == "external:input":
                return prefix + "input"
            if name.startswith("external:"):
                return "external:" + prefix + name.removeprefix("external:")
            return prefix + name

        for node in nodes:
            node["id"] = prefix + node["id"]
            slots = node["graph_slots"]
            slots["inputs"] = {port: source(name) for port, name in slots["inputs"].items()}
            slots["outputs"] = {port: prefix + name for port, name in slots["outputs"].items()}
            ops.append(node)
        buffer(prefix + "logits", [rows, vocabulary])
        buffer(prefix + "mean", [rows, latent_width])
        buffer(prefix + "ids", [rows], "i32")
        emit(prefix + "posterior", "diffusion_embedding_posterior_checked", {
            "model_scores": prefix + "diffusion_scores", "prior_scores": "step_prior",
        }, {"logits": prefix + "logits", "mean": prefix + "mean", "ids": prefix + "ids"},
             {"R": rows, "V": vocabulary, "D": latent_width}, {
                 "rows": rows, "vocabulary": vocabulary, "width": latent_width,
                 "scores_elements": rows * vocabulary, "embedding_elements": vocabulary * latent_width,
                 "mean_elements": latent_count,
             }, weights={"embedding": "normalized_embedding"})

    # The conditional branch can read clean prefix K/V. Unconditional memory
    # contains only the current block and its input excludes self-conditioning.
    append_branch("cond", conditional["activation_buffers"],
                  cond_config["activation_buffer_dtypes"], cond_ops)
    append_branch("uncond", unconditional["activation_buffers"], uc_config["activation_buffer_dtypes"],
                  unconditional["block_types"]["block"]["body"]["ops"])
    buffer("step_guided", [rows, latent_width], "fp64")
    emit("guidance", "diffusion_guidance_checked", {
        "conditional": "step_cond_mean", "unconditional": "step_uncond_mean",
        "scale": "external:step_guidance",
    }, {"output": "step_guided"}, {"N": latent_count}, {
        "elements": latent_count, "scale_elements": 1,
    })
    emit("ddpm", "diffusion_vp_ddpm_step_checked", {
        "latent": "external:step_latent", "noise": "external:step_noise",
        "gamma": "external:step_gamma", "estimate": "step_guided",
    }, {"output": "step_next_latent"}, {"N": latent_count}, {
        "count": latent_count, "gamma_elements": 2,
    }, provider="diffusion_vp_ddpm_step_schedule_f64")
    return graph, config, schema
