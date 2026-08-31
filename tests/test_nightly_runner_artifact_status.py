#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/nightly_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("nightly_runner_artifact_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NightlyArtifactStatusTests(unittest.TestCase):
    def test_json_report_records_the_runner_python(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "nightly.json"
            with mock.patch.object(
                runner, "capture_runner_hardware", return_value={"available": False}
            ):
                runner.save_json_report([], report, datetime(2026, 8, 26, 1, 2, 3))
            payload = json.loads(report.read_text(encoding="utf-8"))

        self.assertEqual(payload["runner_python"]["executable"], sys.executable)
        self.assertTrue(payload["runner_python"]["version"])

    def test_makefile_reuses_primary_checkout_venv_from_linked_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primary = Path(tmp) / "primary"
            shared_python = primary / ".venv" / "bin" / "python"
            shared_python.parent.mkdir(parents=True)
            shared_python.touch()
            command = [
                "make",
                "--no-print-directory",
                f"CK_LOCAL_VENV_PYTHON={primary / 'missing-python'}",
                f"CK_GIT_COMMON_DIR={primary / '.git'}",
                '--eval=print-python: ; @printf "%s\\n" "$(PYTHON)"',
                "print-python",
            ]
            # Scrub PYTHON from the inherited environment: the nightly runner
            # exports PYTHON=<its interpreter>, and the Makefile's `PYTHON ?=`
            # would then yield to it instead of resolving the primary
            # checkout's venv — which is exactly what this test verifies.
            env = {k: v for k, v in os.environ.items() if k != "PYTHON"}
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.splitlines()[-1], str(shared_python))

    def test_make_target_inherits_the_runner_python(self) -> None:
        runner = _load_runner()
        target = {
            "name": "interpreter contract",
            "category": "inference",
            "target": "fake-target",
            "timeout_sec": 10,
            "env": {"PYTHON": "/tmp/wrong-python", "CK_TEST_FLAG": "1"},
        }
        completed = subprocess.CompletedProcess(
            ["make", "fake-target"], 0, stdout="", stderr=""
        )
        with mock.patch.object(runner.subprocess, "run", return_value=completed) as run:
            result = runner.run_make_target(target)

        self.assertEqual(result.status, "pass")
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["PYTHON"], sys.executable)
        self.assertEqual(env["CK_TEST_FLAG"], "1")
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(ROOT))

    def test_xeon_cache_refuses_nonempty_unmarked_directory(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            cache.mkdir()
            retained = cache / "private-weight.gguf"
            retained.write_text("do not delete", encoding="utf-8")
            with self.assertRaises(ValueError):
                runner._prune_xeon_sweep_cache(cache)
            self.assertTrue(retained.exists())

    def test_xeon_cache_prunes_only_sentinel_marked_children(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = runner._prepare_xeon_sweep_cache(root / "managed")
            disposable = cache / "downloaded-model" / "weights.bump"
            disposable.parent.mkdir()
            disposable.write_text("temporary", encoding="utf-8")
            private = root / "private-corpus" / "image01.png"
            private.parent.mkdir()
            private.write_text("private", encoding="utf-8")

            runner._prune_xeon_sweep_cache(cache)

            self.assertFalse(disposable.exists())
            self.assertTrue(private.exists())
            self.assertTrue((cache / runner.XEON_SWEEP_CACHE_SENTINEL).exists())

    def test_private_corpus_output_is_redacted_from_nightly_result(self) -> None:
        runner = _load_runner()
        private_path = "/confidential/corpus/customer-01.png"
        target = {
            "name": "private corpus",
            "category": "parity",
            "target": "fake-private",
            "timeout_sec": 10,
            "redact_output": True,
        }
        completed = subprocess.CompletedProcess(
            ["make", "fake-private"],
            1,
            stdout=f"processing {private_path}\n",
            stderr=f"failure for {private_path}\n",
        )
        with mock.patch.object(runner.subprocess, "run", return_value=completed):
            result = runner.run_make_target(target)
        serialized = f"{result.stdout}\n{result.stderr}\n{result.error_msg}"
        self.assertNotIn(private_path, serialized)
        self.assertIn("Private corpus lane failed", result.error_msg)

    def test_local_model_path_avoids_remote_download_disk_reservation(self) -> None:
        runner = _load_runner()
        target = {
            "sweep_storage": {
                "required_gb": 45,
                "model_env": "V8_QWEN36_MODEL",
            }
        }
        with mock.patch.dict(
            runner.os.environ,
            {"V8_QWEN36_MODEL": "/local/runtime"},
            clear=False,
        ):
            self.assertEqual(runner._sweep_required_bytes(target), 1 << 30)
        with mock.patch.dict(
            runner.os.environ,
            {"V8_QWEN36_MODEL": "hf://repo/model.gguf"},
            clear=False,
        ):
            self.assertEqual(runner._sweep_required_bytes(target), 45 * (1 << 30))

    def test_xeon_e2e_profile_registers_real_artifact_lanes(self) -> None:
        runner = _load_runner()
        profile = runner.NIGHTLY_PROFILES["xeon-e2e"]
        expected = {
            "v8_qwen36_highmem",
            "v8_qwen3vl_vision_smoke",
            "qwen3vl_private_corpus_parity",
            "qwen36vl_private_corpus_parity",
            "qwen3vl_bf16_private_corpus_parity",
            "v8_glm4_highmem",
            "v8_kimi_highmem",
            "v8_gemma4_highmem",
            "v8_xeon_decoder_family_sweep",
        }
        self.assertTrue(expected.issubset(profile))
        self.assertTrue(all(key in runner.MAKE_TARGETS for key in profile))
        qwen36vl = runner.MAKE_TARGETS["qwen36vl_private_corpus_parity"]
        self.assertEqual(
            qwen36vl["target"], "test-qwen36vl-private-corpus-parity-auto"
        )
        self.assertTrue(qwen36vl["redact_output"])
        self.assertEqual(qwen36vl["timeout_sec"], 21600)
        sweep = runner.MAKE_TARGETS["v8_xeon_decoder_family_sweep"]
        self.assertEqual(sweep["profile_only"], "xeon-e2e")

    def test_demo_readiness_profile_covers_models_and_private_ocr(self) -> None:
        runner = _load_runner()
        profile = runner.NIGHTLY_PROFILES["demo-readiness"]
        expected = {
            "v8_regression_fast",
            "v8_audio_contracts",
            "v8_cohere_laguna_contracts",
            "v8_qwen3vl_vision_smoke",
            "v8_gemma4_vision_smoke",
            "qwen3vl_private_corpus_parity",
            "qwen36vl_private_corpus_parity",
            "v8_gemma4_highmem",
            "v8_nemotron9_highmem",
            "v8_glm4_highmem",
            "v8_kimi_highmem",
            "v8_qwen36_highmem",
        }
        self.assertTrue(expected.issubset(profile))
        self.assertTrue(all(key in runner.MAKE_TARGETS for key in profile))
        self.assertIn(
            "v8_instella_moe_circuit_contracts",
            runner.NIGHTLY_PROFILE_TESTS["demo-readiness"],
        )
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        runbook = (ROOT / "docs" / "site" / "_pages" / "v8-runbook.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("nightly-demo-readiness:", makefile)
        self.assertIn("--profile demo-readiness", makefile)
        self.assertIn("make nightly-demo-readiness", runbook)

    def test_markdown_report_preserves_fail_and_skip_reasons(self) -> None:
        runner = _load_runner()
        results = [
            runner.TestResult("decoder", "inference", "pass", 1.25),
            runner.TestResult("private OCR", "parity", "skip", 0.1, error_msg="not configured"),
            runner.TestResult("vision", "inference", "fail", 2.0, error_msg="hash mismatch"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "nested" / "report.md"
            runner.save_markdown_report(results, report, datetime(2026, 8, 25, 1, 2, 3))
            text = report.read_text(encoding="utf-8")
        self.assertIn("**Overall:** FAIL", text)
        self.assertIn("| SKIP | private OCR | parity | 0.1s | not configured |", text)
        self.assertIn("| FAIL | vision | inference | 2.0s | hash mismatch |", text)

    def test_profile_only_sweep_is_not_part_of_default_or_category_nightly(self) -> None:
        runner = _load_runner()
        default_targets = runner._select_make_targets()
        inference_targets = runner._select_make_targets(category="inference")
        self.assertNotIn("v8_xeon_decoder_family_sweep", default_targets)
        self.assertNotIn("v8_xeon_decoder_family_sweep", inference_targets)
        self.assertIn(
            "v8_xeon_decoder_family_sweep",
            runner.NIGHTLY_PROFILES["xeon-e2e"],
        )
        self.assertIn("v8_gemma4_highmem", default_targets)
        self.assertIn("v8_gemma4_highmem", inference_targets)
        self.assertEqual(
            runner.NIGHTLY_PROFILES["gemma4-e2e"],
            ["v8_gemma4_highmem"],
        )
        gemma4 = runner.MAKE_TARGETS["v8_gemma4_highmem"]
        self.assertEqual(gemma4["status_artifact"], "build/v8_gemma4_certification/summary.json")

    def test_make_target_explicit_skip_is_not_reported_as_pass(self) -> None:
        runner = _load_runner()
        target = {
            "name": "private artifact gate",
            "category": "parity",
            "target": "fake-private-gate",
            "timeout_sec": 10,
        }
        completed = subprocess.CompletedProcess(
            ["make", "fake-private-gate"],
            0,
            stdout="SKIP: private corpus is not configured\n",
            stderr="",
        )
        with mock.patch.object(runner.subprocess, "run", return_value=completed):
            result = runner.run_make_target(target)
        self.assertEqual(result.status, "skip")
        self.assertEqual(result.error_msg, "SKIP: private corpus is not configured")

    def test_fresh_artifact_status_overrides_zero_exit(self) -> None:
        runner = _load_runner()
        target = {
            "name": "artifact gate",
            "category": "bf16",
            "target": "fake-gate",
            "timeout_sec": 10,
            "status_artifact": "build/fake/summary.json",
        }
        completed = subprocess.CompletedProcess(["make", "fake-gate"], 0, stdout="", stderr="")
        for artifact_status in ("pass", "skip", "fail"):
            with self.subTest(status=artifact_status):
                with mock.patch.object(runner.subprocess, "run", return_value=completed):
                    with mock.patch.object(
                        runner,
                        "_load_json_if_fresh",
                        return_value={"status": artifact_status},
                    ):
                        result = runner.run_make_target(target)
                self.assertEqual(result.status, artifact_status)

    def test_methodical_qwen3vl_stage_lines_are_visible_subtests(self) -> None:
        runner = _load_runner()
        names = [
            "qwen3vl_checkpoint_coverage",
            "qwen3vl_circuit_codegen",
            "qwen3vl_frontend_mrope",
            "qwen3vl_attention_contract",
            "qwen3vl_q8_projection_matrix",
            "qwen3vl_eos_contract",
        ]
        output = "\n".join(
            f"{name} max_diff=0 tol=0 [PASS]" for name in names
        )
        parsed = runner.parse_sub_tests(output)
        self.assertEqual([row.name for row in parsed], names)
        self.assertTrue(all(row.status == "pass" for row in parsed))

    def test_phase_status_prevents_q8_pass_from_masking_bf16_skip(self) -> None:
        runner = _load_runner()
        target = {
            "name": "BF16 artifact gate",
            "category": "bf16",
            "target": "fake-gate",
            "timeout_sec": 10,
            "status_artifact": "build/fake/summary.json",
            "status_phase": "bf16_pytorch",
        }
        artifact = {
            "status": "pass",
            "phases": {
                "q8_mmproj_llamacpp": {"status": "pass"},
                "bf16_pytorch": {
                    "status": "skip",
                    "reason": "missing BF16 checkpoint",
                },
            },
        }
        completed = subprocess.CompletedProcess(
            ["make", "fake-gate"], 0, stdout="", stderr=""
        )
        with mock.patch.object(runner.subprocess, "run", return_value=completed):
            with mock.patch.object(
                runner, "_load_json_if_fresh", return_value=artifact
            ):
                result = runner.run_make_target(target)
        self.assertEqual(result.status, "skip")
        self.assertEqual(result.error_msg, "missing BF16 checkpoint")


if __name__ == "__main__":
    unittest.main()
