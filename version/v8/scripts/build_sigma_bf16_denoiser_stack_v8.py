"""Repeat the Sigma denoiser circuit and connect its BF16 output readout."""
from copy import deepcopy

from version.v8.scripts.build_sigma_bf16_block_v8 import build_circuit as build_block
from version.v8.scripts.replay_bf16_projection_bundle_v8 import projection_graph


def build_circuit(rows, *, layers=26, vocabulary=131072, readout=True, **geometry):
    if type(layers) is not int or layers < 1 or type(vocabulary) is not int or vocabulary < 1:
        raise ValueError("positive layers and vocabulary required")
    graph, config, schema = build_block(rows, **geometry)
    body = graph["block_types"]["block"]["body"]
    block = body["ops"]
    shapes = graph["activation_buffers"]
    dtypes = config["activation_buffer_dtypes"]
    shared = ("input", "positions", "ranges")
    weight_specs = {"inv_freq": schema["inv_freq"]}
    graph["name"] = "sigma_bf16_denoiser"
    graph["native_entry"]["function"] = "ck_sigma_bf16_denoiser"
    graph["activation_buffers"] = {name: shapes[name] for name in shared}
    graph["activation_bindings"] = {name: name for name in shared}
    config["activation_buffer_dtypes"] = {name: dtypes[name] for name in shared}
    body["ops"] = []
    source = "external:input"

    def add_buffer(name, shape, dtype):
        graph["activation_buffers"][name] = {"shape": shape}
        graph["activation_bindings"][name] = name
        config["activation_buffer_dtypes"][name] = dtype

    for index in range(layers):
        prefix = f"layer{index}_"
        for name in sorted(shapes.keys() - set(shared)):
            add_buffer(prefix + name, shapes[name]["shape"], dtypes[name])
        for name, spec in schema.items():
            if name != "inv_freq":
                weight_specs[f"encoder.layers.{index}.{_checkpoint_layer_name(name)}"] = spec
        for operation in deepcopy(block):
            operation["id"] = prefix + operation["id"]
            slots = operation["graph_slots"]
            for port, tensor in slots["inputs"].items():
                if tensor == "external:input":
                    slots["inputs"][port] = source
                elif tensor.startswith("external:prefix_"):
                    slots["inputs"][port] = "external:" + prefix + tensor.removeprefix("external:")
                elif not tensor.startswith("external:"):
                    slots["inputs"][port] = prefix + tensor
            slots["outputs"] = {port: prefix + tensor for port, tensor in slots["outputs"].items()}
            for role, weight in operation.get("weight_refs", {}).items():
                if weight != "inv_freq":
                    operation["weight_refs"][role] = f"encoder.layers.{index}.{_checkpoint_layer_name(weight)}"
            body["ops"].append(operation)
        source = prefix + "output"

    width = config["embed_dim"]
    add_buffer("denoised_hidden", [rows, width], "bf16")
    norm = deepcopy(block[0])
    norm.update(id="final_norm", weight_refs={"gamma": "encoder.norm.weight"},
                graph_slots={"inputs": {"input": source}, "outputs": {"output": "denoised_hidden"}})
    body["ops"].append(norm)
    weight_specs["encoder.norm.weight"] = ([width], "BF16")

    if not readout:
        return graph, config, weight_specs

    add_buffer("diffusion_scores", [rows, vocabulary], "bf16")
    head = projection_graph(rows, width, vocabulary)["block_types"]["block"]["body"]["ops"][0]
    head.update(id="diffusion_readout", weight_refs={"weight": "diffusion_head.weight"},
                graph_slots={"inputs": {"input": "denoised_hidden"}, "outputs": {"output": "diffusion_scores"}})
    if geometry.get("blas"):
        head["kernel"] = "linear_rows_bf16_blas_checked"
        head["params"]["call_constants"]["work_elements"] = rows * width + width * vocabulary + rows * vocabulary
    body["ops"].append(head)
    weight_specs["diffusion_head.weight"] = ([vocabulary, width], "BF16")
    return graph, config, weight_specs


def _checkpoint_layer_name(name):
    """Map authoring-local projection names to published layer parameter names."""
    attention = {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"}
    feedforward = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    if name in attention:
        return "self_attn." + name
    if name in feedforward:
        return "mlp." + name
    return name
