#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V8_BUILD_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"


def _load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_ir_v8 = _load_module("build_ir_v8_multiblock_tests", V8_BUILD_PATH)


def _multiblock_manifest() -> dict:
    return {
        "config": {
            "model": "qwen3_vl",
            "embed_dim": 2048,
            "num_layers": 40,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 64,
            "context_length": 4096,
            "vocab_size": 151936,
        },
        "entries": [],
        "quant_summary": {},
        "template": {
            "version": 2,
            "name": "qwen3_vl",
            "family": "vision_language",
            "sequence": ["vision_encoder", "decoder"],
            "block_configs": {
                "vision_encoder": {
                    "image_size": 768,
                    "patch_size": 16,
                }
            },
            "block_types": {
                "vision_encoder": {
                    "sequence": ["header", "body", "footer"],
                    "config": {
                        "embed_dim": 1152,
                        "num_layers": 27,
                        "num_heads": 16,
                        "head_dim": 72,
                        "intermediate_size": 4304,
                    },
                    "header": ["patch_embeddings"],
                    "body": {
                        "type": "dense",
                        "ops": [
                            "attn_norm",
                            "qkv_proj",
                            "attn",
                            "out_proj",
                            "residual_add",
                            "ffn_norm",
                            "mlp_gate_up",
                            "geglu",
                            "mlp_down",
                            "residual_add",
                        ],
                    },
                    "footer": ["rmsnorm"],
                },
                "decoder": {
                    "sequence": ["header", "body", "footer"],
                    "config": {
                        "embed_dim": 4096,
                        "num_layers": 36,
                        "num_heads": 32,
                        "head_dim": 128,
                        "intermediate_size": 11008,
                    },
                    "header": ["dense_embedding_lookup"],
                    "body": {
                        "type": "dense",
                        "ops": [
                            "rmsnorm",
                            "qkv_proj",
                            "rope_qk",
                            "attn",
                            "out_proj",
                            "residual_add",
                            "rmsnorm",
                            "mlp_gate_up",
                            "silu_mul",
                            "mlp_down",
                            "residual_add",
                        ],
                    },
                    "footer": ["rmsnorm", "logits"],
                },
            },
        },
    }


class BuildIrV8MultiblockTests(unittest.TestCase):
    def test_build_block_manifests_isolates_each_sequence_entry(self) -> None:
        blocks = build_ir_v8.build_block_manifests(_multiblock_manifest())
        self.assertEqual([b["block_name"] for b in blocks], ["vision_encoder", "decoder"])

        vision = blocks[0]
        decoder = blocks[1]

        self.assertEqual(vision["template"]["sequence"], ["vision_encoder"])
        self.assertEqual(decoder["template"]["sequence"], ["decoder"])
        self.assertEqual(set(vision["template"]["block_types"].keys()), {"vision_encoder"})
        self.assertEqual(set(decoder["template"]["block_types"].keys()), {"decoder"})

        self.assertEqual(vision["config"]["embed_dim"], 1152)
        self.assertEqual(vision["config"]["num_layers"], 27)
        self.assertEqual(vision["config"]["image_size"], 768)
        self.assertEqual(vision["config"]["vocab_size"], 151936)

        self.assertEqual(decoder["config"]["embed_dim"], 4096)
        self.assertEqual(decoder["config"]["num_layers"], 36)
        self.assertEqual(decoder["config"]["vocab_size"], 151936)

    def test_build_stitch_plan_defaults_to_sequential_edges(self) -> None:
        stitch = build_ir_v8.build_stitch_plan(_multiblock_manifest())
        self.assertEqual(stitch["format"], "v8-stitch-plan")
        self.assertEqual(stitch["sequence"], ["vision_encoder", "decoder"])
        self.assertEqual(
            stitch["edges"],
            [
                {
                    "from": "vision_encoder",
                    "to": "decoder",
                    "kind": "sequential",
                    "from_output": "output",
                    "to_input": "input",
                }
            ],
        )

    def test_main_split_only_writes_block_manifests_and_stitch_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "weights_manifest.json"
            blocks_dir = tmp / "blocks"
            stitch_path = tmp / "stitch.json"
            manifest_path.write_text(json.dumps(_multiblock_manifest(), indent=2), encoding="utf-8")

            rc = build_ir_v8.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--block-manifests-dir",
                    str(blocks_dir),
                    "--stitch-output",
                    str(stitch_path),
                ]
            )

            self.assertEqual(rc, 0)
            vision_manifest = blocks_dir / "01_vision_encoder" / "weights_manifest.json"
            decoder_manifest = blocks_dir / "02_decoder" / "weights_manifest.json"
            self.assertTrue(vision_manifest.exists())
            self.assertTrue(decoder_manifest.exists())
            self.assertTrue(stitch_path.exists())

            vision_doc = json.loads(vision_manifest.read_text(encoding="utf-8"))
            decoder_doc = json.loads(decoder_manifest.read_text(encoding="utf-8"))
            stitch_doc = json.loads(stitch_path.read_text(encoding="utf-8"))

            self.assertEqual(vision_doc["template"]["sequence"], ["vision_encoder"])
            self.assertEqual(decoder_doc["template"]["sequence"], ["decoder"])
            self.assertEqual(stitch_doc["sequence"], ["vision_encoder", "decoder"])


if __name__ == "__main__":
    unittest.main()
