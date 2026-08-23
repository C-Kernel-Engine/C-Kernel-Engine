#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v8" / "scripts"
sys.path[:0] = [str(SCRIPT_DIR), str(ROOT / "scripts")]
SPEC = importlib.util.spec_from_file_location(
    "certify_text_prompt_parity_v8_tests",
    SCRIPT_DIR / "certify_text_prompt_parity_v8.py",
)
assert SPEC is not None and SPEC.loader is not None
certifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(certifier)


class TextPromptCertificationTests(unittest.TestCase):
    def test_qwen35_fixture_has_strict_promotion_matrix(self) -> None:
        fixture = certifier.load_prompt_set(
            ROOT / "version" / "v8" / "test_assets" / "qwen35_text_parity_prompts.json"
        )
        self.assertEqual(fixture["stages"], [64, 128, 256])
        self.assertEqual([row["id"] for row in fixture["prompts"]], [
            "hello", "c-python-sql", "pure-c", "thanks",
        ])
        self.assertEqual(fixture["stop_token_ids"], [248046])
        self.assertEqual(len(fixture["llama_cpp_commit"]), 40)
        self.assertTrue(all(row["tokens"] for row in fixture["prompts"]))

    def test_invalid_stage_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "stages": [128, 64],
                "prompts": [{"id": "one", "text": "one", "tokens": [1]}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "increasing"):
                certifier.load_prompt_set(path)

    def test_eos_report_satisfies_larger_stage(self) -> None:
        report = {"pass": True, "matched_stop_token": 248046, "steps": [{}] * 12}
        self.assertTrue(certifier.report_satisfies_stage(report, 256))

    def test_unfinished_report_does_not_satisfy_stage(self) -> None:
        report = {"pass": True, "matched_stop_token": None, "steps": [{}] * 63}
        self.assertFalse(certifier.report_satisfies_stage(report, 64))

    def test_quality_failure_does_not_discard_complete_trajectory(self) -> None:
        report = {
            "pass": False,
            "first_divergence": None,
            "matched_stop_token": None,
            "steps": [{}] * 64,
            "quality": {"pass": False},
        }
        self.assertTrue(certifier.report_satisfies_stage(report, 64))

    def test_utf8_corruption_markers_are_rejected(self) -> None:
        self.assertTrue(certifier.decoded_text_is_clean("Hello \u2014 \u4f60\u597d \U0001f60a"))
        for text in ("bad \\uFFFD", "bad \ufffd", "bad \u00c3\u00a9", "bad \ufffd\u0141"):
            self.assertFalse(certifier.decoded_text_is_clean(text))

    def test_eos_report_is_reused_for_larger_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hello-64.json"
            source.write_text(json.dumps({
                "pass": True, "matched_stop_token": 248046, "steps": [{}] * 14,
            }), encoding="utf-8")
            self.assertEqual(
                certifier.reusable_report_path(root, "hello", [64, 128, 256], 256),
                source,
            )

    def test_failure_handoff_uses_recurrent_xray(self) -> None:
        command = certifier.xray_handoff(
            Path("/model"), Path("/model.gguf"), Path("/out/fail.json"), Path("/out"), 20
        )
        self.assertIn("xray_text_recurrent_v8.py", command)
        self.assertIn("--parity-report /out/fail.json", command)
        self.assertIn("--ck-prefill-mode hybrid", command)

    def test_standalone_svg_quality_contract_is_fail_closed(self) -> None:
        contract = {
            "kind": "standalone_svg.v1",
            "min_graphic_elements": 3,
            "required_labels": ["decode"],
        }
        valid = certifier.evaluate_quality_contract(
            '<svg viewBox="0 0 100 100"><title>Pipeline</title>'
            '<desc>Four stages</desc><rect width="10" height="10"/>'
            '<line x1="0" y1="0" x2="10" y2="10"/><text>decode</text></svg>',
            contract,
        )
        self.assertTrue(valid["pass"])
        self.assertEqual(valid["graphic_element_count"], 3)

        scripted = certifier.evaluate_quality_contract(
            '<svg viewBox="0 0 1 1"><title>T</title><desc>D</desc>'
            '<script>alert(1)</script><rect/></svg>',
            contract,
        )
        self.assertFalse(scripted["pass"])
        self.assertTrue(scripted["has_script"])

    def test_svg_quality_requires_requested_labels_and_used_arrow_marker(self) -> None:
        contract = {
            "kind": "standalone_svg.v1",
            "required_labels": ["tokenize", "prefill", "decode", "detokenize"],
            "require_arrow_marker": True,
        }
        valid = certifier.evaluate_quality_contract(
            '<svg viewBox="0 0 100 20"><title>T</title><desc>D</desc>'
            '<defs><marker id="arrow"><path d="M0 0L5 2L0 4Z"/></marker></defs>'
            '<text>Tokenize Prefill Decode Detokenize</text>'
            '<line marker-end="url(#arrow)"/></svg>',
            contract,
        )
        self.assertTrue(valid["pass"])
        self.assertTrue(valid["has_arrow_marker"])

        missing = certifier.evaluate_quality_contract(
            '<svg viewBox="0 0 100 20"><title>T</title><desc>D</desc><circle/></svg>',
            contract,
        )
        self.assertFalse(missing["pass"])
        self.assertIn("tokenize", missing["missing_labels"])

    def test_svg_quality_rejects_wrappers_and_external_assets(self) -> None:
        contract = {"kind": "standalone_svg.v1"}
        result = certifier.evaluate_quality_contract(
            '```svg\n<svg viewBox="0 0 1 1"><title>T</title><desc>D</desc>'
            '<image href="https://example.com/a.png"/></svg>\n```',
            contract,
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["output_only"])
        self.assertTrue(result["has_external_reference"])

    def test_svg_quality_preserves_complete_visible_reasoning(self) -> None:
        result = certifier.evaluate_quality_contract(
            '<think>Plan the accessible diagram.</think>\n'
            '<svg viewBox="0 0 10 10"><title>T</title><desc>D</desc><rect/></svg>',
            {"kind": "standalone_svg.v1"},
        )
        self.assertTrue(result["pass"])
        self.assertTrue(result["reasoning_present"])
        self.assertTrue(result["reasoning_complete"])
        self.assertGreater(result["reasoning_characters"], 0)

    def test_svg_quality_rejects_incomplete_visible_reasoning(self) -> None:
        result = certifier.evaluate_quality_contract(
            '<think>Still planning the diagram',
            {"kind": "standalone_svg.v1"},
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["reasoning_complete"])
        self.assertEqual(result["answer_characters"], 0)

    def test_xray_completion_rejects_missing_boundaries(self) -> None:
        self.assertTrue(certifier.xray_report_is_complete({
            "first_divergence": None,
            "rows": [{"status": "exact"}, {"status": "exact"}],
        }))
        self.assertFalse(certifier.xray_report_is_complete({
            "first_divergence": None,
            "rows": [{"status": "exact"}, {"status": "missing_or_incompatible"}],
        }))

    def test_trajectory_numerical_contract_requires_every_exact_row(self) -> None:
        report = {
            "first_divergence": None,
            "steps": [
                {"top1_match": True, "bit_exact": True},
                {"top1_match": True, "bit_exact": False},
            ],
        }
        self.assertTrue(certifier.trajectory_numerical_contract(report, False)["pass"])
        strict = certifier.trajectory_numerical_contract(report, True)
        self.assertFalse(strict["pass"])
        self.assertEqual(strict["bit_exact_rows"], 1)

    def test_large_vocabulary_trajectory_uses_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps({"vocab_size": 248320}), encoding="utf-8"
            )
            self.assertFalse(certifier.trajectory_uses_streaming(root, 64))
            self.assertTrue(certifier.trajectory_uses_streaming(root, 512))

    def test_runtime_context_capacity_reads_generated_abi(self) -> None:
        class Getter:
            argtypes = None
            restype = None

            def __call__(self) -> int:
                return 2048

        class Runtime:
            ck_model_get_context_window = Getter()

        original = certifier.ctypes.CDLL
        certifier.ctypes.CDLL = lambda _path: Runtime()
        try:
            self.assertEqual(certifier.runtime_context_capacity(Path("/model")), 2048)
        finally:
            certifier.ctypes.CDLL = original


if __name__ == "__main__":
    unittest.main()
