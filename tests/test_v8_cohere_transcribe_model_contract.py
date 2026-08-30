#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import convert_gguf_to_bump_v8 as converter  # type: ignore


class CohereTranscribeModelContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arch = "cohere-transcribe"
        self.contract = converter.load_gguf_ck_map()["architectures"][self.arch]
        self.metadata = {
            "cohere_transcribe.encoder.n_layers": 48,
            "cohere_transcribe.decoder.n_layers": 8,
        }

    def _complete_tensor_names(self) -> set[str]:
        inventory = self.contract["tensor_inventory"]
        names = set(inventory["required_global"])
        for group in inventory["layer_groups"]:
            count = self.metadata[group["count_metadata"]]
            for layer in range(count):
                prefix = group["prefix"].format(L=layer)
                names.update(prefix + suffix for suffix in group["required_suffixes"])
        return names

    def test_model_map_owns_the_complete_real_artifact_inventory(self) -> None:
        inventory = self.contract["tensor_inventory"]
        groups = {group["name"]: group for group in inventory["layer_groups"]}

        self.assertEqual(len(inventory["required_global"]), 24)
        self.assertEqual(len(groups["conformer_encoder"]["required_suffixes"]), 39)
        self.assertEqual(len(groups["cross_attention_decoder"]["required_suffixes"]), 26)
        self.assertIn("attn.pos_bias_u", groups["conformer_encoder"]["required_suffixes"])
        self.assertIn("conv.dw.weight", groups["conformer_encoder"]["required_suffixes"])
        self.assertIn("cross_q.weight", groups["cross_attention_decoder"]["required_suffixes"])

        report = converter.gguf_ck_tensor_inventory_report(
            self.arch,
            self.metadata,
            self._complete_tensor_names(),
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["expected_tensors"], 2104)
        self.assertEqual(report["actual_tensors"], 2104)
        self.assertEqual(report["missing_tensors"], [])
        self.assertEqual(report["undeclared_tensors"], [])

    def test_inventory_fails_closed_for_missing_and_undeclared_tensors(self) -> None:
        names = self._complete_tensor_names()
        names.remove("enc.blk.47.attn.pos_bias_v")
        names.add("enc.blk.48.attn.q.weight")

        report = converter.gguf_ck_tensor_inventory_report(
            self.arch,
            self.metadata,
            names,
        )
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["missing_tensors"], ["enc.blk.47.attn.pos_bias_v"])
        self.assertEqual(report["undeclared_tensors"], ["enc.blk.48.attn.q.weight"])

    def test_inventory_rejects_missing_layer_count_metadata(self) -> None:
        with self.assertRaisesRegex(converter.GGUFError, "decoder.n_layers"):
            converter.gguf_ck_tensor_inventory_report(
                self.arch,
                {"cohere_transcribe.encoder.n_layers": 48},
                self._complete_tensor_names(),
            )

    def test_metadata_and_conversion_boundary_are_explicit(self) -> None:
        metadata_map = self.contract["metadata_map"]
        self.assertEqual(
            metadata_map["audio_sample_rate"],
            "cohere_transcribe.audio.sample_rate",
        )
        self.assertEqual(
            metadata_map["supported_languages"],
            "cohere_transcribe.supported_languages",
        )
        self.assertIn(
            "cohere_transcribe.audio.max_clip_s",
            converter.gguf_ck_declared_metadata_keys(),
        )
        blocker = converter.gguf_ck_conversion_blocker(self.arch)
        self.assertIn("provider_foundation", blocker)
        self.assertIn("decoder cross-attention", blocker)


if __name__ == "__main__":
    unittest.main()
