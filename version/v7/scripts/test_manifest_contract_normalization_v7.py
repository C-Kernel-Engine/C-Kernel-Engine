#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CK_RUN_PATH = ROOT / "version" / "v7" / "scripts" / "ck_run_v7.py"


def _load_ck_run_module():
    spec = importlib.util.spec_from_file_location("ck_run_v7_for_tests", CK_RUN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {CK_RUN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ck_run = _load_ck_run_module()


def _entry(name: str) -> dict[str, object]:
    return {
        "name": name,
        "dtype": "fp32",
        "offset": 0,
        "file_offset": 0,
        "size": 4,
    }


def _manifest(template_name: str, *, tie_word_embeddings, entries: list[str], template=None) -> dict[str, object]:
    if template is None:
        template = ck_run._load_builtin_template_doc(template_name)
    return {
        "config": {
            "model": template_name,
            "num_layers": 1,
            "hidden_size": 4,
            "context_length": 8,
            "tie_word_embeddings": tie_word_embeddings,
        },
        "entries": [_entry(name) for name in entries],
        "template": template,
    }


class ManifestContractNormalizationTests(unittest.TestCase):
    def test_tied_families_stay_weight_tying(self) -> None:
        for template_name in ("qwen2", "qwen3", "gemma3"):
            with self.subTest(template=template_name):
                manifest = _manifest(
                    template_name,
                    tie_word_embeddings=True,
                    entries=["token_emb", "layer.0.wq"],
                )
                normalized = ck_run._normalize_manifest_for_inference(manifest)
                lm_head = normalized["template"]["contract"]["logits_contract"]["lm_head"]
                self.assertEqual(lm_head, "weight_tying")

    def test_stale_untied_manifest_is_repaired(self) -> None:
        stale_template = ck_run._load_builtin_template_doc("llama")
        manifest = _manifest(
            "llama",
            tie_word_embeddings=False,
            entries=["token_emb", "output.weight", "layer.0.wq"],
            template=stale_template,
        )
        normalized = ck_run._normalize_manifest_for_inference(manifest)
        lm_head = normalized["template"]["contract"]["logits_contract"]["lm_head"]
        self.assertEqual(lm_head, "output_weight")

    def test_missing_tie_flag_with_output_head_repairs_to_untied(self) -> None:
        stale_template = ck_run._load_builtin_template_doc("llama")
        manifest = _manifest(
            "llama",
            tie_word_embeddings=None,
            entries=["token_emb", "output.weight", "layer.0.wq"],
            template=stale_template,
        )
        normalized = ck_run._normalize_manifest_for_inference(manifest)
        lm_head = normalized["template"]["contract"]["logits_contract"]["lm_head"]
        self.assertEqual(lm_head, "output_weight")

    def test_stale_template_inherits_builtin_kernels(self) -> None:
        stale_template = ck_run._load_builtin_template_doc("llama")
        stale_template["kernels"] = None
        manifest = _manifest(
            "llama",
            tie_word_embeddings=False,
            entries=["token_emb", "output.weight", "layer.0.wq"],
            template=stale_template,
        )
        normalized = ck_run._normalize_manifest_for_inference(manifest)
        self.assertEqual(
            normalized["template"]["kernels"]["rope_qk"],
            "rope_forward_qk_pairwise",
        )

    def test_semantic_check_accepts_repaired_untied_manifest(self) -> None:
        stale_template = ck_run._load_builtin_template_doc("llama")
        manifest = _manifest(
            "llama",
            tie_word_embeddings=False,
            entries=["token_emb", "output.weight", "layer.0.wq"],
            template=stale_template,
        )
        with tempfile.TemporaryDirectory(prefix="v7_manifest_contract_") as td:
            path = Path(td) / "weights_manifest.json"
            path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            ok, report = ck_run._template_manifest_semantic_check(path)
        self.assertTrue(ok, report)

    def test_tied_manifest_with_output_head_is_rejected(self) -> None:
        manifest = _manifest(
            "qwen3",
            tie_word_embeddings=True,
            entries=["token_emb", "output.weight", "layer.0.wq"],
        )
        with tempfile.TemporaryDirectory(prefix="v7_manifest_contract_") as td:
            path = Path(td) / "weights_manifest.json"
            path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            ok, report = ck_run._template_manifest_semantic_check(path)
        self.assertFalse(ok)
        self.assertTrue(report.get("errors"))


if __name__ == "__main__":
    unittest.main()
