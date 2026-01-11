#!/usr/bin/env python3
"""
Verify v6 prefill codegen writes K/V directly into cache stride (no repack).
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPTS_V6 = ROOT / "scripts" / "v6"
sys.path.insert(0, str(SCRIPTS_V6))

import codegen_v6  # noqa: E402


class Buffer:
    def __init__(self, name: str, dtype: str = "fp32") -> None:
        self.name = name
        self.dtype = dtype


class Layer:
    def __init__(self, buffers: list[Buffer]) -> None:
        self.buffers = buffers


class Section:
    def __init__(self,
                 header_buffers: list[Buffer],
                 layers: list[Layer],
                 footer_buffers: list[Buffer],
                 globals: list[Buffer] | None = None) -> None:
        self.header_buffers = header_buffers
        self.layers = layers
        self.footer_buffers = footer_buffers
        self.globals = globals or []


class Layout:
    def __init__(self, name: str, config: dict, section: Section) -> None:
        self.name = name
        self.config = config
        self.sections = [section]
        self.total_bytes = 1024


def build_layout() -> Layout:
    config = {
        "model_type": "qwen2",
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "hidden_size": 64,
        "intermediate_size": 128,
        "vocab_size": 256,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "dtype": "fp32",
    }
    header = [
        Buffer("token_emb"),
        Buffer("embedded_input"),
    ]
    footer = [
        Buffer("final_ln_weight"),
        Buffer("final_output"),
        Buffer("lm_head_weight"),
        Buffer("logits"),
    ]
    globals = [
        Buffer("rope_cos_cache"),
        Buffer("rope_sin_cache"),
    ]
    layer_bufs = [
        Buffer("layer.0.ln1_gamma"),
        Buffer("layer.0.ln1_out"),
        Buffer("layer.0.ln2_gamma"),
        Buffer("layer.0.ln2_out"),
        Buffer("layer.0.wq"),
        Buffer("layer.0.wk"),
        Buffer("layer.0.wv"),
        Buffer("layer.0.wo"),
        Buffer("layer.0.w1"),
        Buffer("layer.0.w2"),
        Buffer("layer.0.q"),
        Buffer("layer.0.k"),
        Buffer("layer.0.v"),
        Buffer("layer.0.attn_out"),
        Buffer("layer.0.proj_tmp"),
        Buffer("layer.0.proj_scratch"),
        Buffer("layer.0.residual1"),
        Buffer("layer.0.fc1_out"),
        Buffer("layer.0.swiglu_out"),
        Buffer("layer.0.mlp_out"),
        Buffer("layer.0.output"),
    ]
    layers = [Layer(layer_bufs)]
    section = Section(header, layers, footer, globals=globals)
    return Layout("test_model", config, section)


def test_codegen_prefill_direct_cache() -> None:
    layout = build_layout()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_c = Path(tmpdir) / "gen.c"
        codegen_v6.emit_c_source_v6(
            layout,
            str(out_c),
            "gen.h",
            mode="prefill",
            emit_main=False,
            emit_debug=False,
            emit_parity=False,
        )
        code = out_c.read_text()

    assert "kv_cache_repack_head_major_inplace" not in code
    assert "attention_forward_causal_head_major_gqa_flash_strided" in code
    assert "rope_forward_qk_strided" in code


if __name__ == "__main__":
    test_codegen_prefill_direct_cache()
    print("v6 prefill codegen direct-to-cache test passed.")
