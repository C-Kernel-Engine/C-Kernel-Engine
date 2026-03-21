#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v7" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import codegen_v7


class TestCodegenLayerOffsetsV7(unittest.TestCase):
    def test_layer_offsets_union_fields_across_mixed_layers(self) -> None:
        layout = {
            "memory": {
                "weights": {
                    "size": 1024,
                    "entries": [
                        {"name": "token_emb", "dtype": "f32", "size": 64, "offset": 0},
                        {"name": "layer.0.attn_qkv", "dtype": "f32", "size": 64, "offset": 64},
                        {"name": "layer.0.ssm_out", "dtype": "f32", "size": 64, "offset": 128},
                        {"name": "layer.1.attn_k", "dtype": "f32", "size": 64, "offset": 192},
                        {"name": "layer.1.attn_k_norm", "dtype": "f32", "size": 64, "offset": 256},
                    ],
                },
                "activations": {"buffers": []},
                "arena": {"activations_base": 1024, "total_size": 1024},
            },
            "bump_layout": {"header_size": 128, "ext_metadata_size": 24, "data_start": 152},
        }
        config = {
            "embed_dim": 64,
            "num_heads": 4,
            "num_kv_heads": 4,
            "intermediate_size": 128,
            "vocab_size": 256,
            "context_length": 32,
            "num_layers": 2,
        }

        c_src = codegen_v7.emit_memory_layout(layout, config)

        self.assertIn("size_t attn_qkv;", c_src)
        self.assertIn("size_t ssm_out;", c_src)
        self.assertIn("size_t attn_k;", c_src)
        self.assertIn("size_t attn_k_norm;", c_src)
        self.assertIn(".attn_qkv = 64,", c_src)
        self.assertIn(".attn_k = 192,", c_src)
        self.assertIn(".attn_k_norm = 256,", c_src)


if __name__ == "__main__":
    unittest.main()
