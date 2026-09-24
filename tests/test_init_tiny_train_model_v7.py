#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v7" / "scripts" / "init_tiny_train_model_v7.py"


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


init_tiny = _load_module("init_tiny_train_model_v7_test", SCRIPT)


class InitTinyTrainModelV7Tests(unittest.TestCase):
    def test_template_aliases_cover_family_names(self) -> None:
        self.assertEqual(init_tiny._resolve_template_name("gemma"), "gemma3")
        self.assertEqual(init_tiny._resolve_template_name("nanbeige"), "nanbeige")
        self.assertEqual(init_tiny._resolve_template_name("qwen35"), "qwen35")
        self.assertEqual(init_tiny._resolve_template_name(""), "qwen3")

    def test_bias_free_semantic_initialization_omits_optional_bias_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            init_tiny.build_tiny_model(
                out_dir=Path(td), seed=42, init="normal_0p02", template_name="qwen3",
                template_doc=json.loads((ROOT / "version/v7/templates/qwen3.json").read_text()),
                n_layers=5, vocab_size=384, embed_dim=32, hidden_dim=64,
                num_heads=4, num_kv_heads=2, context_len=32, rope_theta=10_000.0,
                kernel_policy="fp32_reference_first", adamw_beta1=0.9,
                adamw_beta2=0.999, adamw_eps=1e-8, adamw_weight_decay=0.01,
                omit_linear_biases=True,
            )
            manifest = json.loads((Path(td) / "weights_manifest.json").read_text())
            names = {row["name"] for row in manifest["entries"]}
            self.assertIn("layer.4.q_norm", names)
            self.assertIn("layer.4.k_norm", names)
            self.assertFalse(any(name.endswith((".bq", ".bk", ".bv", ".bo", ".b1", ".b2")) for name in names))


if __name__ == "__main__":
    unittest.main()
