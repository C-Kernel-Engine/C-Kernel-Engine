#!/usr/bin/env python3
"""Regression tests for v7 training telemetry payload materialization."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from version.v7.scripts import ck_run_v7  # noqa: E402


class TrainTelemetryPipelinePayloadTest(unittest.TestCase):
    def test_build_training_pipeline_payload_handles_missing_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)

            payload = ck_run_v7._build_training_pipeline_payload(  # noqa: SLF001
                {
                    "train_mode": "pretrain",
                    "backend": "ck",
                    "epochs": 1,
                    "steps": 1,
                    "seq_len": 8,
                    "grad_accum": 2,
                    "total_tokens": 64,
                },
                run_dir,
            )

            self.assertEqual(payload["active_stage"], "pretrain")
            self.assertEqual(payload["pipeline"]["stages"][0]["stage"], "data_preparation")
            self.assertEqual(payload["pipeline"]["stages"][1]["stage"], "tokenizer")
            self.assertEqual(payload["pipeline"]["stages"][2]["stage"], "pretrain")


if __name__ == "__main__":
    unittest.main()
