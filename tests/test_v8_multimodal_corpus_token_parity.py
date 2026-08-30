#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/compare_multimodal_corpus_runs_v8.py"
SPEC = importlib.util.spec_from_file_location("compare_multimodal_corpus_runs_v8", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class MultimodalCorpusTokenParityTests(unittest.TestCase):
    def _run(self, subject_tokens: list[int], oracle_tokens: list[int]):
        temporary = tempfile.TemporaryDirectory(prefix="cke-corpus-token-parity-")
        root = Path(temporary.name)
        for lane, tokens in (("subject", subject_tokens), ("oracle", oracle_tokens)):
            _write_json(root / lane / "config.json", {"manifest_sha256": "manifest"})
            _write_json(
                root / lane / "image01" / "case_result.json",
                {
                    "image_index": 1,
                    "image_sha256": "image",
                    "case_config": {
                        "prompt_sha256": "prompt",
                        "max_new_tokens": 128,
                    },
                    "generated_token_ids": tokens,
                    "token_trace_sha256": f"{lane}-trace",
                },
            )
        return temporary, MODULE.compare(root / "subject", root / "oracle")

    def test_exact_private_token_traces_pass(self) -> None:
        temporary, report = self._run([1, 2, 3], [1, 2, 3])
        with temporary:
            self.assertEqual(report["status"], "pass")
            self.assertIsNone(report["rows"][0]["first_divergence"])

    def test_first_token_divergence_is_redacted_and_precise(self) -> None:
        temporary, report = self._run([1, 7, 3], [1, 8, 3])
        with temporary:
            self.assertEqual(report["status"], "fail")
            self.assertEqual(
                report["rows"][0]["first_divergence"],
                {"step": 1, "subject_token": 7, "oracle_token": 8},
            )
            self.assertNotIn("generated_text", json.dumps(report))

    def test_manifest_mismatch_is_rejected(self) -> None:
        temporary, _ = self._run([1], [1])
        with temporary:
            root = Path(temporary.name)
            _write_json(root / "oracle" / "config.json", {"manifest_sha256": "other"})
            with self.assertRaisesRegex(ValueError, "identical corpus manifest"):
                MODULE.compare(root / "subject", root / "oracle")

    def test_generation_budget_mismatch_is_rejected(self) -> None:
        temporary, _ = self._run([1], [1])
        with temporary:
            root = Path(temporary.name)
            oracle_path = root / "oracle" / "image01" / "case_result.json"
            oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
            oracle["case_config"]["max_new_tokens"] = 256
            _write_json(oracle_path, oracle)
            with self.assertRaisesRegex(ValueError, "generation budget differs"):
                MODULE.compare(root / "subject", root / "oracle")


if __name__ == "__main__":
    unittest.main(verbosity=2)
