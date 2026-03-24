#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v7" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_v7_stabilization_nightly_v7 as stab  # noqa: E402


class StabilizationNightlyTests(unittest.TestCase):
    def test_parse_csv_strings_dedupes_and_preserves_order(self) -> None:
        got = stab._parse_csv_strings("qwen2, qwen3, qwen2, llama", name="family_templates")
        self.assertEqual(got, ["qwen2", "qwen3", "llama"])

    def test_pick_head_layout_prefers_even_divisors(self) -> None:
        self.assertEqual(stab._pick_head_layout(64), (8, 4))
        self.assertEqual(stab._pick_head_layout(12), (4, 4))
        self.assertEqual(stab._pick_head_layout(10), (1, 1))

    def test_render_markdown_includes_template_and_case_kind(self) -> None:
        payload = {
            "generated_at": "2026-03-24T00:00:00Z",
            "summary": {"passed": True, "parity_pass_rate": 1.0, "tokenizer_gate_pass_rate": 1.0},
            "tokenizer_gates": [],
            "matrix_cases": [
                {
                    "case_id": "family_qwen2",
                    "template": "qwen2",
                    "case_kind": "family",
                    "status": "PASS",
                    "layers": 4,
                    "token_budget": 4096,
                    "metrics": {
                        "final_ck_loss": 5.123456,
                        "replay_determinism_pass": True,
                        "failed_stage_ids": [],
                    },
                }
            ],
        }
        md = stab._render_markdown(payload)
        self.assertIn("| Case | Template | Kind | Status |", md)
        self.assertIn("| family_qwen2 | qwen2 | family | PASS | 4 | 4096 | 5.123456 | True | - |", md)


if __name__ == "__main__":
    unittest.main()
