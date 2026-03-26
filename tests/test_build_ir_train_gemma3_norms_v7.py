#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))

import build_ir_train_v7  # type: ignore


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(name: str) -> dict:
    return {
        "name": name,
        "dtype": "fp32",
        "shape": [1, 1],
        "offset": 0,
        "file_offset": 0,
        "size": 4,
    }


def _gemma_manifest(*, include_post_norms: bool) -> dict:
    entries = [
        _entry("token_emb"),
        _entry("output.weight"),
        _entry("norm.weight"),
        _entry("layer.0.ln1_gamma"),
        _entry("layer.0.ln2_gamma"),
        _entry("layer.0.wq"),
        _entry("layer.0.wk"),
        _entry("layer.0.wv"),
        _entry("layer.0.q_norm"),
        _entry("layer.0.k_norm"),
        _entry("layer.0.wo"),
        _entry("layer.0.w1"),
        _entry("layer.0.w2"),
        _entry("layer.0.w3"),
    ]
    if include_post_norms:
        entries.extend(
            [
                _entry("layer.0.post_attention_norm"),
                _entry("layer.0.post_ffn_norm"),
            ]
        )
    return {
        "config": {
            "model": "gemma3",
            "num_layers": 1,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_heads": 8,
            "num_kv_heads": 4,
            "head_dim": 16,
            "vocab_size": 512,
            "train_tokens": 2,
        },
        "template": _load_json(ROOT / "version" / "v7" / "templates" / "gemma3.json"),
        "entries": entries,
    }


def _body_rmsnorm_gamma_tensors(ir1: dict) -> list[str]:
    gamma_tensors: list[str] = []
    for op in ir1.get("ops", []):
        if op.get("op") != "rmsnorm" or op.get("section") != "body":
            continue
        gamma = ((op.get("weights") or {}).get("gamma") or {}).get("tensor")
        if gamma is not None:
            gamma_tensors.append(str(gamma))
    return gamma_tensors


class Gemma3TrainNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _load_json(ROOT / "version" / "v7" / "kernel_maps" / "KERNEL_REGISTRY.json")
        cls.bindings = _load_json(ROOT / "version" / "v7" / "kernel_maps" / "kernel_bindings.json")
        cls.grad_rules = _load_json(ROOT / "version" / "v7" / "scripts" / "grad_rules_v7.json")

    def test_gemma3_train_builder_handles_norm_aliases_without_optional_post_norms(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=_gemma_manifest(include_post_norms=False),
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
        )
        self.assertEqual(ir1.get("issues"), [])
        self.assertEqual(ir1.get("warnings"), [])
        self.assertEqual(
            _body_rmsnorm_gamma_tensors(ir1),
            [
                "weight.layer.0.ln1_gamma",
                "weight.layer.0.ln2_gamma",
            ],
        )

    def test_gemma3_train_builder_uses_optional_post_norm_weights_when_present(self) -> None:
        ir1 = build_ir_train_v7.build_ir1_train(
            manifest=_gemma_manifest(include_post_norms=True),
            registry=self.registry,
            bindings_doc=self.bindings,
            grad_rules=self.grad_rules,
            max_layers=1,
            strict=False,
        )
        self.assertEqual(ir1.get("issues"), [])
        self.assertEqual(ir1.get("warnings"), [])
        self.assertEqual(
            _body_rmsnorm_gamma_tensors(ir1),
            [
                "weight.layer.0.ln1_gamma",
                "weight.layer.0.post_attention_norm",
                "weight.layer.0.ln2_gamma",
                "weight.layer.0.post_ffn_norm",
            ],
        )


if __name__ == "__main__":
    unittest.main()
