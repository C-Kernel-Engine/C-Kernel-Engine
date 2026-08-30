#!/usr/bin/env python3
"""Portable contracts for the model-agnostic multimodal OCR certifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "certify_multimodal_ocr_corpus_v8.py"
SPEC = importlib.util.spec_from_file_location("certify_multimodal_ocr_corpus_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MultimodalOcrCorpusCertificationTest(unittest.TestCase):
    def test_extracts_plain_and_fenced_json(self) -> None:
        self.assertEqual(MODULE._extract_json_object('{"name":"Ada"}'), {"name": "Ada"})
        self.assertEqual(
            MODULE._extract_json_object('Result:\n```json\n{"name":"Ada"}\n```'),
            {"name": "Ada"},
        )
        self.assertIsNone(MODULE._extract_json_object("not JSON"))

    def test_scores_alternative_values_without_weakening_field_names(self) -> None:
        metrics = MODULE._score(
            {"sin": ["123-456", "123 456"], "name": "Ada", "blank": ""},
            {"sin": "123456", "name": "Grace", "blank": "", "extra": 1},
        )
        self.assertTrue(metrics["json_valid"])
        self.assertEqual(metrics["exact_fields"], 2)
        self.assertEqual(metrics["mismatched_fields"], ["name"])
        self.assertEqual(metrics["extra_fields"], ["extra"])
        self.assertEqual(metrics["nonempty_exact_fields"], 1)

    def test_loads_existing_private_manifest_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sample.jpg").write_bytes(b"image")
            (root / "sample.json").write_text('{"name":"Ada"}\n', encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "id": "sample",
                                "inputs": [{"path": "sample.jpg", "mimeType": "image/jpeg"}],
                                "groundTruth": [
                                    {"path": "sample.json", "format": "field-key-value-json"}
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = MODULE._load_samples(manifest)
        self.assertEqual(rows[0]["id"], "sample")
        self.assertEqual(rows[0]["truth"], {"name": "Ada"})

    def test_public_row_redacts_private_content(self) -> None:
        private = {
            "image_index": 1,
            "image_sha256": "image-hash",
            "truth_sha256": "truth-hash",
            "status": "complete",
            "output_sha256": "output-hash",
            "token_trace_sha256": "trace-hash",
            "stop_reason": "stop_token",
            "generated_tokens": 2,
            "timings": {"wall_sec": 1.0},
            "metrics": {"json_valid": True},
            "image_path": "/private/image.jpg",
            "truth_path": "/private/truth.json",
            "prompt": "private prompt",
            "generated_text": "private output",
        }
        serialized = json.dumps(MODULE._public_row(private))
        self.assertNotIn("/private", serialized)
        self.assertNotIn("private prompt", serialized)
        self.assertNotIn("private output", serialized)

    def test_make_target_declares_unsupported_llamacpp_oracle(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("test-cohere-compass-private-ocr-auto", makefile)
        self.assertIn("--adapter-id cohere_compass", makefile)
        self.assertIn("--oracle-id llama.cpp", makefile)
        self.assertIn("--oracle-status unsupported", makefile)
        self.assertIn("--adapt-encoder-geometry", makefile)

    def test_bridge_command_uses_shared_geometry_cache_when_requested(self) -> None:
        args = SimpleNamespace(
            decoder_runtime=Path("decoder"),
            encoder_runtime=Path("encoder"),
            composition_circuit="cohere_compass",
            chat_template="auto",
            thinking_mode="suppressed",
            context_len=4096,
            max_new_tokens=32,
            generation_progress_every=8,
            adapt_encoder_geometry=True,
            output_dir=Path("results"),
        )
        sample = {"image": Path("form.jpg")}
        command = MODULE._bridge_command(args, sample, Path("case"), "extract")
        rendered = " ".join(str(item) for item in command)
        self.assertIn("--encoder-geometry-cache-dir", rendered)
        self.assertIn("results/encoder_geometry_cache", rendered)


if __name__ == "__main__":
    unittest.main()
