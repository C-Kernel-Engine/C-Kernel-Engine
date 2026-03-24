#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))

try:
    import torch  # noqa: F401
    import train_parity_epochs_v7 as train_parity  # type: ignore
    _TORCH_AVAILABLE = True
    _TORCH_SKIP_REASON = ""
except Exception as exc:  # pragma: no cover - environment-dependent
    train_parity = None  # type: ignore
    _TORCH_AVAILABLE = False
    _TORCH_SKIP_REASON = f"torch not available: {exc}"


def _load_template(name: str) -> dict:
    return json.loads((ROOT / "version" / "v7" / "templates" / f"{name}.json").read_text(encoding="utf-8"))


@unittest.skipUnless(_TORCH_AVAILABLE, _TORCH_SKIP_REASON)
class TrainParityRopeTests(unittest.TestCase):
    def test_manifest_rope_defaults_uses_template_split_layout(self) -> None:
        manifest = {"config": {"num_heads": 4, "head_dim": 16}, "template": _load_template("qwen3")}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "weights_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            defaults = train_parity._manifest_rope_defaults(path)
        self.assertEqual(defaults["rope_layout"], "split")
        self.assertEqual(defaults["rope_heads"], 4)
        self.assertEqual(defaults["rope_rotary_dim"], 16)

    def test_manifest_rope_defaults_uses_template_pairwise_layout(self) -> None:
        manifest = {"config": {"num_heads": 4, "head_dim": 16}, "template": _load_template("llama")}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "weights_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            defaults = train_parity._manifest_rope_defaults(path)
        self.assertEqual(defaults["rope_layout"], "pairwise")
        self.assertEqual(defaults["rope_heads"], 4)
        self.assertEqual(defaults["rope_rotary_dim"], 16)

    def test_stage_order_includes_rope_when_enabled(self) -> None:
        model = train_parity.TinyTorchModel(
            vocab=256,
            d_model=64,
            hidden=128,
            eps=1e-5,
            rope_layout="pairwise",
            rope_heads=4,
            rope_rotary_dim=16,
        )
        self.assertEqual(
            train_parity._stage_order_for_model(model),
            ["embedding", "rope", "rmsnorm", "fc1", "swiglu", "logits", "loss"],
        )
        self.assertEqual(
            train_parity._backward_stage_order_for_model(model),
            ["logits", "swiglu", "fc1", "rmsnorm", "rope", "embedding"],
        )

    def test_stage_order_omits_rope_when_disabled(self) -> None:
        model = train_parity.TinyTorchModel(vocab=256, d_model=64, hidden=128, eps=1e-5)
        self.assertEqual(
            train_parity._stage_order_for_model(model),
            ["embedding", "rmsnorm", "fc1", "swiglu", "logits", "loss"],
        )


if __name__ == "__main__":
    unittest.main()
