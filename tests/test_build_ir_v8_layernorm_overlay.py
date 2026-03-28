#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V8_BUILD_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_ir_v8 = _load_module("build_ir_v8_layernorm_overlay_tests", V8_BUILD_PATH)


def _entry(name: str, dtype: str, shape: list[int], offset: int) -> dict:
    nbytes_per = {"fp32": 4, "q8_0": 1}.get(dtype, 4)
    size = 1
    for dim in shape:
        size *= int(dim)
    size *= nbytes_per
    return {"name": name, "dtype": dtype, "offset": offset, "shape": shape, "size": size}


def _layernorm_manifest() -> dict:
    offset = 0
    entries = []

    def add(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        e = _entry(name, dtype, shape, offset)
        entries.append(e)
        offset += int(e["size"])

    add("token_emb", "q8_0", [64, 128])
    add("v.blk.0.ln1.weight", "fp32", [128])
    add("v.blk.0.ln1.bias", "fp32", [128])

    return {
        "config": {
            "model": "layernorm_overlay_smoke",
            "num_layers": 1,
            "embed_dim": 128,
            "num_heads": 2,
            "num_kv_heads": 2,
            "head_dim": 64,
            "context_length": 16,
            "max_seq_len": 16,
            "vocab_size": 64
        },
        "quant_summary": {
            "token_emb": "q8_0"
        },
        "entries": entries,
        "template": {
            "version": 3,
            "name": "layernorm_overlay_smoke",
            "family": "decoder",
            "sequence": ["decoder"],
            "block_types": {
                "decoder": {
                    "sequence": ["header", "body", "footer"],
                    "header": ["dense_embedding_lookup"],
                    "body": {
                        "type": "dense",
                        "ops": [
                            {"id": "ln1", "op": "layernorm"}
                        ]
                    },
                    "footer": []
                }
            }
        }
    }


class BuildIrV8LayernormOverlayTests(unittest.TestCase):
    def test_registry_overlay_exposes_layernorm_kernel(self) -> None:
        registry = build_ir_v8.load_kernel_registry()
        ids = {kernel["id"] for kernel in registry.get("kernels", [])}
        self.assertIn("layernorm_forward", ids)

    def test_build_ir1_direct_lowers_generic_layernorm(self) -> None:
        manifest = _layernorm_manifest()
        ir1_ops = build_ir_v8.build_ir1_direct(
            manifest,
            ROOT / "tests" / "layernorm_overlay_manifest.synthetic.json",
            mode="prefill",
        )
        ops = [op["op"] for op in ir1_ops]
        self.assertIn("layernorm", ops)
        ln = next(op for op in ir1_ops if op["op"] == "layernorm")
        self.assertEqual(ln["kernel"], "layernorm_forward")
        self.assertEqual(ln["weights"]["ln1_gamma"]["name"], "v.blk.0.ln1.weight")
        self.assertEqual(ln["weights"]["ln1_beta"]["name"], "v.blk.0.ln1.bias")


if __name__ == "__main__":
    unittest.main()
