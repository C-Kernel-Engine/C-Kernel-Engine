"""Declare Phonon 31M's normalized-latent to Mimi-feature circuit."""

from version.v8.scripts.checked_circuit_authoring_v8 import linear
from version.v8.scripts.replay_bf16_projection_bundle_v8 import projection_graph


LATENT_WIDTH = 32
CODEC_WIDTH = 512


def build_circuit(frames):
    """Keep denormalization and the 1x1 Mimi projection as explicit edges."""
    if type(frames) is not int or not 1 <= frames <= 2048:
        raise ValueError("expected 1 to 2048 codec input frames")
    graph = projection_graph(frames, LATENT_WIDTH, CODEC_WIDTH)
    graph["name"] = "phonon31m_codec_ingress"
    graph["native_entry"]["function"] = "ck_phonon31m_codec_ingress"
    shape = [frames, LATENT_WIDTH]
    graph["activation_buffers"] = {
        "latents": {"shape": shape},
        "latent_mean": {"shape": [1, LATENT_WIDTH]},
        "scaled": {"shape": shape},
        "denormalized": {"shape": shape},
        "codec_features": {"shape": [frames, CODEC_WIDTH]},
    }
    graph["activation_bindings"] = {name: name for name in graph["activation_buffers"]}
    elements = frames * LATENT_WIDTH
    scale = {
        "id": "scale_latents", "op": "channel_scale_rows_checked",
        "kernel": "channel_scale_rows_checked_f32", "returns_status": True,
        "weight_refs": {"scale": "latent_std"},
        "graph_slots": {"inputs": {"input": "external:latents"}, "outputs": {"output": "scaled"}},
        "params": {"R": frames, "C": LATENT_WIDTH, "call_constants": {
            "input_elements": elements, "input_stride": LATENT_WIDTH, "scale_elements": LATENT_WIDTH,
            "output_elements": elements, "output_stride": LATENT_WIDTH,
            "rows": frames, "channels": LATENT_WIDTH,
        }},
    }
    add = {
        "id": "add_latent_mean", "op": "broadcast_add_rows_checked",
        "kernel": "broadcast_add_rows_checked_f32", "returns_status": True,
        "graph_slots": {"inputs": {"left": "scaled", "right": "external:latent_mean"},
                        "outputs": {"output": "denormalized"}},
        "params": {"R": frames, "C": LATENT_WIDTH, "X": elements, "call_constants": {
            "left_elements": elements, "right_elements": LATENT_WIDTH,
            "output_elements": elements, "scratch_elements": elements,
            "rows": frames, "channels": LATENT_WIDTH,
        }},
    }
    project = linear("mimi_output_projection", "denormalized", "codec_features",
                     LATENT_WIDTH, CODEC_WIDTH, frames, "mimi_output_projection")
    project.update(op="linear_rows_checked_no_bias", kernel="linear_rows_checked_no_bias_f32",
                   weight_refs={"weight": "mimi_output_projection.weight"})
    del project["params"]["call_constants"]["linear_bias_elements"]
    graph["block_types"]["block"]["body"]["ops"] = [scale, add, project]
    config = {
        "num_layers": 1, "embed_dim": LATENT_WIDTH, "num_heads": 1, "num_kv_heads": 1,
        "head_dim": LATENT_WIDTH, "intermediate_size": CODEC_WIDTH,
        "context_length": frames, "max_seq_len": frames,
    }
    return graph, config, {
        "latent_std": ([LATENT_WIDTH], "F32"),
        "mimi_output_projection.weight": ([CODEC_WIDTH, LATENT_WIDTH], "F32"),
    }
