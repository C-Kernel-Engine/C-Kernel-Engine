"""Checked fixed-tile projector stages for Nemotron Nano 12B v2 VL BF16."""

from pathlib import Path

import numpy as np

from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph


PUBLISHER_REVISION = "ca9543b126e8bf3176916d3d305ccc415f89fd4d"
GRID = 32
CHANNELS = 1280
MERGE = 2
PROJECTOR_WIDTH = CHANNELS * MERGE**2
VISUAL_ROWS = (GRID // MERGE)**2
INNER = 20480
DECODER_WIDTH = 5120


def validate_publisher_config(config):
    """Reject other configurations until their feature and bridge layouts are tested."""
    expected = {
        "model_type": "NemotronH_Nano_VL_V2",
        "force_image_size": 512,
        "patch_size": 16,
        "downsample_ratio": 0.5,
        "ps_version": "v2",
        "vit_hidden_size": CHANNELS,
        "projector_hidden_size": INNER,
        "torch_dtype": "bfloat16",
        "img_context_token_id": 131072,
    }
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported Nemotron VL projector configuration")
    vision, decoder = config.get("vision_config"), config.get("llm_config")
    if not isinstance(vision, dict) or vision.get("version") != "radio_v2.5-h":
        raise ValueError("unsupported Nemotron VL vision encoder")
    if (not isinstance(decoder, dict) or decoder.get("model_type") != "nemotron_h"
            or decoder.get("hidden_size") != DECODER_WIDTH):
        raise ValueError("unsupported Nemotron VL decoder geometry")


def projector_patch_indices():
    # Publisher pixel_shuffle v2: output grid (y, x), then (dy, dx, channel).
    return np.fromiter(
        ((y + dy) * GRID + x + dx
         for y in range(0, GRID, MERGE)
         for x in range(0, GRID, MERGE)
         for dy in range(MERGE) for dx in range(MERGE)),
        dtype=np.int32, count=GRID**2,
    )


def projector_input_graph():
    rows = GRID**2
    provider = "gather_rows_bf16_checked"
    buffers = {
        "patches": {"shape": [rows, CHANNELS]},
        "indices": {"shape": [rows]},
        "packed": {"shape": [rows, CHANNELS]},
    }
    return {
        "version": 3, "name": "nemotron_vl_projector_input", "family": "bounded_graph",
        "checked_native_entry": True,
        "contract": {"runtime_invariants": {
            "inference_only": True, "tokens": rows, "width": CHANNELS, "heads": 1, "mult": 1,
        }},
        "activation_buffers": buffers,
        "activation_bindings": {name: name for name in buffers},
        "sequence": ["merge"],
        "block_types": {"merge": {
            "sequence": ["header", "body", "footer"], "header": [], "footer": [],
            "body": {"type": "dense", "ops": [{
                "id": "pack_patches", "op": provider, "kernel": provider, "returns_status": True,
                "graph_slots": {
                    "inputs": {"input": "external:patches", "indices": "external:indices"},
                    "outputs": {"output": "packed"},
                },
                "params": {"R": rows, "S": rows, "D": CHANNELS, "call_constants": {
                    "input_elements": rows * CHANNELS, "index_elements": rows,
                    "output_elements": rows * CHANNELS, "input_rows": rows,
                    "output_rows": rows, "width": CHANNELS,
                }},
            }]},
        }},
        "native_entry": {"function": "ck_nemotron_vl_projector_input", "params": [
            {"c_type": "uint8_t *", "name": "arena"}, {"c_type": "size_t", "name": "arena_bytes"},
        ], "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
    }


class CompiledProjectorInput:
    """Pack one register-free RADIO feature tile, not pixels or decoder embeddings."""

    def __init__(self, directory, config):
        validate_publisher_config(config)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        graph = projector_input_graph()
        rows = GRID**2
        source_config = {
            "model": graph["name"], "arch": graph["name"], "num_layers": 1,
            "embed_dim": CHANNELS, "intermediate_size": CHANNELS, "num_heads": 1,
            "num_kv_heads": 1, "head_dim": CHANNELS, "vocab_size": 1,
            "context_length": rows, "max_seq_len": rows,
            "activation_buffer_dtypes": {"patches": "bf16", "indices": "i32", "packed": "bf16"},
        }
        self._run, self._arena, self._view, self._library = compile_checked_weight_graph(
            directory, graph, {}, {}, source_config=source_config,
        )
        self._view("indices", np.int32)[:] = projector_patch_indices()

    def __call__(self, patches):
        if (not isinstance(patches, np.ndarray) or patches.dtype != np.uint16
                or patches.shape != (GRID**2, CHANNELS)):
            raise ValueError("expected one 1024 x 1280 tile of stored BF16 spatial features")
        self._view("patches", np.uint16)[:] = patches
        status = self._run(self._arena, len(self._arena))
        if status != 0:
            raise RuntimeError(f"generated projector input packing failed with status {status}")
        return self._view("packed", np.uint16).reshape(VISUAL_ROWS, PROJECTOR_WIDTH).copy()


def compile_weighted_projector(directory, weights, *, rows, width, inner, output, cblas=False):
    """Compile RMSNorm, linear, squared ReLU, linear with BF16 stores between ops."""
    if (any(type(n) is not int or n < 1 for n in (rows, width, inner, output))
            or rows > VISUAL_ROWS or width > PROJECTOR_WIDTH or inner > INNER
            or output > DECODER_WIDTH or type(cblas) is not bool):
        raise ValueError("unsupported Nemotron VL weighted projector geometry")
    name = "nemotron_vl_weighted_projector"
    buffers = {key: {"shape": [rows, cols]} for key, cols in (
        ("features", width), ("norm", width), ("expanded", inner),
        ("activated", inner), ("embeddings", output),
    )}
    declarations = {
        "mlp1.0.weight": ([width], "BF16"),
        "mlp1.1.weight": ([inner, width], "BF16"),
        "mlp1.3.weight": ([output, inner], "BF16"),
    }

    def op(key, operation, provider, src, constants, dims, *, port="output", refs=None):
        node = {
            "id": key, "op": operation, "kernel": provider, "returns_status": True,
            "graph_slots": {"inputs": {"input": src}, "outputs": {port: key}},
            "params": {**dims, "call_constants": constants},
        }
        if refs is not None:
            node["weight_refs"] = refs
        return node

    def linear(key, src, k, n, weight):
        operation = "linear_rows_bf16_checked"
        provider = "linear_rows_bf16_cblas_checked" if cblas else operation
        scratch = "work_elements" if cblas else "staged_elements"
        return op(key, operation, provider, src, {
            "input_elements": rows * k, "weight_elements": n * k,
            "output_elements": rows * n, scratch: rows * n,
            "rows": rows, "input_width": k, "output_width": n,
        }, {"M": rows, "K": k, "N": n}, refs={"weight": weight})

    count = rows * width
    ops = [
        op("norm", "rmsnorm_rows_bf16_checked", "rmsnorm_rows_weighted_single_round_bf16_checked",
           "external:features", {
               "input_elements": count, "gamma_elements": width, "output_elements": count,
               "work_elements": count + width, "rows": rows, "width": width,
           }, {"M": rows, "K": width, "epsilon": 1e-5}, refs={"gamma": "mlp1.0.weight"}),
        linear("expanded", "norm", width, inner, "mlp1.1.weight"),
        op("activated", "relu2_bf16_checked", "relu2_bf16_checked", "expanded", {
            "elements": rows * inner,
        }, {"N": rows * inner}, port="out"),
        linear("embeddings", "activated", inner, output, "mlp1.3.weight"),
    ]
    graph = {
        "version": 3, "name": name, "family": "bounded_graph", "checked_native_entry": True,
        "contract": {"runtime_invariants": {"inference_only": True}},
        "activation_buffers": buffers,
        "activation_bindings": {key: key for key in buffers},
        "sequence": ["block"],
        "block_types": {"block": {
            "sequence": ["header", "body", "footer"], "header": [], "footer": [],
            "body": {"type": "dense", "ops": ops},
        }},
        "native_entry": {"function": "ck_" + name, "params": [
            {"c_type": "uint8_t *", "name": "arena"}, {"c_type": "size_t", "name": "arena_bytes"},
        ], "arena": {"pointer": "arena", "bytes": "arena_bytes"}},
    }
    source_config = {
        "num_layers": 1, "embed_dim": width, "num_heads": 1,
        "num_kv_heads": 1, "head_dim": width, "intermediate_size": inner,
        "context_length": rows, "max_seq_len": rows,
        "activation_buffer_dtypes": dict.fromkeys(buffers, "bf16"),
    }
    return compile_checked_weight_graph(
        directory, graph, weights, declarations, source_config=source_config,
    )


class CompiledProjector:
    """Turn register-free spatial RADIO features into one tile of decoder rows."""

    def __init__(self, directory, config, weights, *, cblas=False):
        validate_publisher_config(config)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self._pack = CompiledProjectorInput(directory / "pack", config)
        weighted = directory / "weighted"
        weighted.mkdir()
        self._run, self._arena, self._view, self._library = compile_weighted_projector(
            weighted, weights, rows=VISUAL_ROWS, width=PROJECTOR_WIDTH,
            inner=INNER, output=DECODER_WIDTH, cblas=cblas,
        )

    def __call__(self, patches):
        self._view("features", np.uint16)[:] = self._pack(patches)
        status = self._run(self._arena, len(self._arena))
        if status:
            raise RuntimeError(f"generated weighted projector failed with status {status}")
        return self._view("embeddings", np.uint16).copy()
