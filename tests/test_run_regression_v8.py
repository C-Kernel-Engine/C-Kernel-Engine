#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v8" / "scripts"))

import run_regression_v8 as regression  # type: ignore


class RegressionHarnessV8Tests(unittest.TestCase):
    def test_extract_assistant_output(self) -> None:
        text = (
            "You: Hello\n"
            "Assistant: Hello! How can I assist you today?\n"
            "prompt eval: 123.0 ms / 20 tokens\n"
        )
        self.assertEqual(
            regression._extract_assistant_output(text),
            "Hello! How can I assist you today?",
        )

    def test_normalize_assistant_output_trims_markers_and_think(self) -> None:
        text = "<think>\ninternal\n</think>\nHello!<|im_end|>\nprompt eval: 1.0 ms"
        cleaned = regression.normalize_assistant_output(
            text,
            {
                "strip_think_blocks": True,
                "stop_text_markers": ["<|im_end|>"],
                "strip_trailing_metrics": True,
                "trim_whitespace": True,
            },
        )
        self.assertEqual(cleaned, "Hello!")

    def test_greeting_coherence_passes(self) -> None:
        heuristics = {
            "min_chars": 8,
            "min_words": 3,
            "max_chars": 240,
            "max_words": 48,
            "max_lines": 8,
            "min_printable_ratio": 0.95,
            "max_replacement_chars": 0,
            "max_repeated_4gram": 2,
            "max_duplicate_lines": 1,
            "expected_keywords": ["hello", "help", "assist"],
            "min_keyword_hits": 1,
        }
        result = regression.assess_coherence("Hello! How can I assist you today?", heuristics)
        self.assertEqual(result["status"], regression.PASS)

    def test_coherence_rejects_actual_and_literal_replacement_markers(self) -> None:
        heuristics = {"max_replacement_chars": 0}
        for text in ("broken \ufffd text", r"broken \uFFFD text", r"broken \ufffd text"):
            with self.subTest(text=text):
                result = regression.assess_coherence(text, heuristics)
                self.assertEqual(result["status"], regression.FAIL)
                self.assertGreater(result["metrics"]["replacement_chars"], 0)

    def test_manifest_files_are_consistent(self) -> None:
        prompts = regression.load_prompts(ROOT / "version" / "v8" / "regression" / "prompts.json")
        families = regression.load_families(ROOT / "version" / "v8" / "regression" / "families.json", prompts)
        ids = {family.family_id for family in families}
        self.assertEqual(ids, {"gemma", "qwen2", "qwen3", "qwen35", "nanbeige"})
        by_id = {family.family_id: family for family in families}
        self.assertIn("--thinking-mode", by_id["qwen3"].runtime_args)
        self.assertIn("suppressed", by_id["qwen3"].runtime_args)
        arm_manifest = json.loads(
            (ROOT / "version" / "v8" / "regression" / "families_arm.json").read_text(
                encoding="utf-8"
            )
        )
        arm_by_id = {family["id"]: family for family in arm_manifest["families"]}
        self.assertNotIn("qwen3vl", by_id)
        self.assertIn("qwen3vl", arm_by_id)
        self.assertFalse(arm_by_id["qwen3vl"]["enabled"])
        self.assertIn("--image-path", arm_by_id["qwen3vl"]["runtime_args"])
        self.assertEqual(arm_by_id["qwen3vl"]["smoke_prompts"], ["vision_doc_card"])
        self.assertIn("--image-path", prompts["vision_doc_card"].runtime_args)
        self.assertIn("v8_vision_doc_card_72.ppm", " ".join(prompts["vision_doc_card"].runtime_args))
        self.assertIn("--image-path", prompts["vision_mamba2_card"].runtime_args)
        self.assertIn("v8_vision_mamba2_card_144.png", " ".join(prompts["vision_mamba2_card"].runtime_args))
        self.assertEqual(arm_by_id["qwen3vl"]["runtime_expect"].get("manifest", {}).get("config.model"), "qwen3vl")
        self.assertEqual(by_id["nanbeige"].runtime_expect.get("config", {}).get("chat_contract.name"), "llama_chatml")
        self.assertEqual(by_id["gemma"].runtime_expect.get("config", {}).get("rope_layout"), "split")
        self.assertEqual(by_id["gemma"].repeatability.get("prompt"), "hello")
        self.assertEqual(by_id["gemma"].repeatability.get("runs"), 5)
        qwen35_lowered = by_id["qwen35"].runtime_expect.get("lowered_ops", [])
        self.assertEqual(qwen35_lowered[0].get("function_prefix"), "mrope_qk_text")

    def test_family_model_env_overrides_portable_default(self) -> None:
        prompts = regression.load_prompts(
            ROOT / "version" / "v8" / "regression" / "prompts.json"
        )
        manifest = ROOT / "version" / "v8" / "regression" / "families_gemma4_certification.json"
        with mock.patch.dict(
            os.environ,
            {"V8_GEMMA4_MODEL": "/models/local-gemma4.gguf"},
            clear=False,
        ):
            families = regression.load_families(manifest, prompts)
        self.assertEqual(len(families), 1)
        self.assertEqual(families[0].family_id, "gemma4")
        self.assertEqual(families[0].model, "/models/local-gemma4.gguf")
        self.assertTrue(families[0].coherence_gate)
        self.assertEqual(families[0].repeatability["runs"], 3)

    def test_resolve_gguf_path_accepts_direct_cache_root_layout(self) -> None:
        cache_root = Path("/tmp/test_run_regression_v8_cache")
        repo_dir = cache_root / "unsloth--gemma-3-270m-it-GGUF"
        repo_dir.mkdir(parents=True, exist_ok=True)
        gguf = repo_dir / "gemma-3-270m-it-Q5_K_M.gguf"
        gguf.write_bytes(b"gguf")
        old_cache_dir = os.environ.get("CK_CACHE_DIR")
        try:
            os.environ["CK_CACHE_DIR"] = str(cache_root)
            resolved = regression._resolve_gguf_path(
                "hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf"
            )
            self.assertEqual(resolved, gguf)
        finally:
            if old_cache_dir is None:
                os.environ.pop("CK_CACHE_DIR", None)
            else:
                os.environ["CK_CACHE_DIR"] = old_cache_dir
            if gguf.exists():
                gguf.unlink()
            if repo_dir.exists():
                repo_dir.rmdir()
            if cache_root.exists():
                cache_root.rmdir()

    def test_resolve_gguf_path_checks_default_cache_root_after_env_override(self) -> None:
        with tempfile.TemporaryDirectory(prefix="reg_v8_cache_roots_") as tmpdir:
            tmp = Path(tmpdir)
            env_root = tmp / "env"
            default_root = tmp / "default"
            legacy_root = tmp / "legacy"
            repo_dir = default_root / "models" / "unsloth--gemma-3-270m-it-GGUF"
            repo_dir.mkdir(parents=True, exist_ok=True)
            gguf = repo_dir / "gemma-3-270m-it-Q5_K_M.gguf"
            gguf.write_bytes(b"gguf")

            with mock.patch.object(regression, "DEFAULT_CACHE_ROOT", default_root), \
                 mock.patch.object(regression, "LEGACY_CACHE_ROOT", legacy_root):
                old_cache_dir = os.environ.get("CK_CACHE_DIR")
                try:
                    os.environ["CK_CACHE_DIR"] = str(env_root / "models")
                    resolved = regression._resolve_gguf_path(
                        "hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf"
                    )
                finally:
                    if old_cache_dir is None:
                        os.environ.pop("CK_CACHE_DIR", None)
                    else:
                        os.environ["CK_CACHE_DIR"] = old_cache_dir

            self.assertEqual(resolved, gguf)

    def test_run_prompt_passes_cache_dir_to_child_runner(self) -> None:
        family = regression.FamilySpec(
            family_id="qwen3",
            label="Qwen3",
            model="hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
            context_len=1024,
            runtime_args=["--chat-template", "auto"],
            smoke_prompts=["hello"],
            response_contract={"trim_whitespace": True},
            coherence_gate=False,
            runtime_expect={},
        )
        prompt = regression.PromptSpec(
            prompt_id="hello",
            label="Hello",
            prompt="Hello",
            max_tokens=32,
            heuristics={},
        )
        cache_dir = Path("/tmp/ck-v8-test-cache/models")

        with mock.patch.object(
            regression,
            "_run_stream",
            return_value=subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="Response: Hello there\n",
                stderr="",
            ),
        ) as run_stream:
            row = regression.run_prompt(
                family,
                prompt,
                Path("/tmp/ck-v8-test-run"),
                force_rebuild=False,
                cache_dir=cache_dir,
            )

        self.assertEqual(row["status"], regression.PASS)
        self.assertEqual(run_stream.call_args.kwargs["env"]["CK_CACHE_DIR"], str(cache_dir))

    def test_run_prompt_appends_prompt_runtime_args(self) -> None:
        family = regression.FamilySpec(
            family_id="qwen3vl",
            label="Qwen3-VL",
            model="hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
            context_len=1024,
            runtime_args=["--thinking-mode", "suppressed"],
            smoke_prompts=["vision_doc_card"],
            response_contract={"trim_whitespace": True},
            coherence_gate=True,
            runtime_expect={},
        )
        prompt = regression.PromptSpec(
            prompt_id="vision_doc_card",
            label="Vision Doc Card",
            prompt="Explain this image.",
            max_tokens=24,
            heuristics={},
            runtime_args=["--image-path", "version/v8/test_assets/v8_vision_doc_card_72.ppm"],
        )

        with mock.patch.object(
            regression,
            "_run_stream",
            return_value=subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="Response: This image is a logo.\n",
                stderr="",
            ),
        ) as run_stream:
            row = regression.run_prompt(
                family,
                prompt,
                Path("/tmp/ck-v8-test-run"),
                force_rebuild=False,
                cache_dir=Path("/tmp/ck-v8-test-cache/models"),
            )

        self.assertEqual(row["status"], regression.PASS)
        cmd = run_stream.call_args.args[0]
        self.assertIn("--thinking-mode", cmd)
        self.assertIn("--image-path", cmd)
        self.assertIn("version/v8/test_assets/v8_vision_doc_card_72.ppm", cmd)

    def test_run_prompt_prefers_multimodal_bridge_generated_text(self) -> None:
        family = regression.FamilySpec(
            family_id="qwen3vl",
            label="Qwen3-VL",
            model="hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
            context_len=1024,
            runtime_args=["--image-path", "version/v8/test_assets/v8_vision_doc_card_72.ppm"],
            smoke_prompts=["vision_doc_card"],
            response_contract={"trim_whitespace": True},
            coherence_gate=True,
            runtime_expect={},
        )
        prompt = regression.PromptSpec(
            prompt_id="vision_doc_card",
            label="Vision Doc Card",
            prompt="Explain this image.",
            max_tokens=24,
            heuristics={},
        )

        with tempfile.TemporaryDirectory(prefix="ck_reg_v8_bridge_prompt_") as tmp:
            run_dir = Path(tmp)
            bridge_dir = run_dir / "multimodal_bridge"
            bridge_dir.mkdir(parents=True, exist_ok=True)
            (bridge_dir / "bridge_report.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "generated_text": "This image is a logo for C-earned Engineer.",
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                regression,
                "_run_stream",
                return_value=subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout='{"status":"ok","top_logits":[{"token_text":"This"}]}',
                    stderr="",
                ),
            ):
                row = regression.run_prompt(
                    family,
                    prompt,
                    run_dir,
                    force_rebuild=False,
                    cache_dir=Path("/tmp/ck-v8-test-cache/models"),
                )

        self.assertEqual(row["assistant"], "This image is a logo for C-earned Engineer.")
        self.assertEqual(row["bridge_report"]["status"], "ok")

    def test_repeatability_rejects_empty_or_changed_output(self) -> None:
        family = regression.FamilySpec(
            family_id="gemma",
            label="Gemma",
            model="model.gguf",
            context_len=1024,
            runtime_args=[],
            smoke_prompts=["hello"],
            response_contract={},
            coherence_gate=False,
            runtime_expect={},
            repeatability={"prompt": "hello", "runs": 3, "require_nonempty": True},
        )
        prompt = regression.PromptSpec("hello", "Hello", "Hello", 8, {})
        baseline = {"status": regression.PASS, "assistant": "stable"}
        repeated = [
            {"status": regression.PASS, "assistant": "stable"},
            {"status": regression.PASS, "assistant": "changed"},
        ]
        with mock.patch.object(regression, "run_prompt", side_effect=repeated):
            result = regression.run_repeatability(
                family,
                prompt,
                baseline,
                Path("/tmp/repeatability"),
                cache_dir=Path("/tmp/cache"),
            )
        self.assertEqual(result["status"], regression.FAIL)
        self.assertIn("output_mismatch", result["reasons"])

        with mock.patch.object(
            regression,
            "run_prompt",
            side_effect=[
                {"status": regression.PASS, "assistant": ""},
                {"status": regression.PASS, "assistant": ""},
            ],
        ):
            result = regression.run_repeatability(
                family,
                prompt,
                {"status": regression.PASS, "assistant": ""},
                Path("/tmp/repeatability"),
                cache_dir=Path("/tmp/cache"),
            )
        self.assertIn("empty_output", result["reasons"])

    def test_runtime_contract_audit_checks_config_manifest_lowered_and_stdout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_reg_contract_v8_") as tmp:
            run_dir = Path(tmp)
            runtime_dir = run_dir
            (run_dir / "config.json").write_text(
                json.dumps({"rope_layout": "split", "chat_contract": {"name": "gemma"}}),
                encoding="utf-8",
            )
            (run_dir / "weights_manifest.json").write_text(
                json.dumps({"config": {"rope_layout": "split", "chat_contract": {"name": "gemma"}}}),
                encoding="utf-8",
            )
            (run_dir / "lowered_decode_call.json").write_text(
                json.dumps(
                    {
                        "operations": [
                            {"op": "rope_qk", "function": "rope_forward_qk_with_rotary_dim"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = regression.audit_runtime_contract(
                run_dir,
                runtime_dir,
                [{"stdout": "Loaded HuggingFace tokenizer from /tmp/tokenizer.json\n"}],
                {
                    "stdout_contains_any_of": [
                        "Using built-in C tokenizer",
                        "Loaded HuggingFace tokenizer",
                    ],
                    "stdout_not_contains": ["Python tokenizer"],
                    "config": {"rope_layout": "split", "chat_contract.name": "gemma"},
                    "manifest": {"config.rope_layout": "split", "config.chat_contract.name": "gemma"},
                    "lowered_ops": [
                        {"op": "rope_qk", "function_prefix": "rope_forward_qk_with_rotary_dim"}
                    ],
                },
                run_dir / "contract_audit.json",
            )
            self.assertEqual(result["status"], regression.PASS)

    def test_failure_classification_handles_contract_and_coherence(self) -> None:
        failure_class, detail = regression.classify_family_result(
            build_status=regression.PASS,
            smoke_status=regression.PASS,
            coherence_status=regression.FAIL,
            coherence_gate=True,
            contract_result={"status": regression.FAIL},
            failure_reason="coherence_failed:hello",
        )
        self.assertEqual(failure_class, "contract_failure")
        self.assertIn("contract", detail)

        failure_class, detail = regression.classify_family_result(
            build_status=regression.PASS,
            smoke_status=regression.PASS,
            coherence_status=regression.FAIL,
            coherence_gate=False,
            contract_result={"status": regression.SKIP},
            failure_reason="coherence_failed:hello",
        )
        self.assertEqual(failure_class, "pass")
        self.assertEqual(detail, "")

    def test_environment_marker_detection(self) -> None:
        rows = [
            {
                "status": regression.FAIL,
                "stdout": "",
                "stderr": (
                    "Error: V8DownloadError: CK_V8_DOWNLOAD_FAILED: download failed for "
                    "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf: HTTP error after retries: "
                    "HTTP Error 429 (HuggingFace rate limit (HTTP 429) or error page received)"
                ),
            }
        ]
        self.assertTrue(regression._is_environment_failure(rows))

        corrupt_cache_rows = [
            {
                "status": regression.FAIL,
                "stdout": "",
                "stderr": "GGUFError: /tmp/model.gguf: invalid magic b'<!DO' (expected b'GGUF')",
            }
        ]
        self.assertFalse(regression._is_environment_failure(corrupt_cache_rows))
        self.assertFalse(regression._is_environment_failure([{"status": regression.PASS}]))

    def test_environment_failure_classification_names_rate_limit(self) -> None:
        failure_class, detail = regression.classify_family_result(
            build_status=regression.FAIL,
            smoke_status=regression.FAIL,
            coherence_status=regression.PASS,
            coherence_gate=False,
            contract_result={"status": regression.SKIP},
            failure_reason="smoke_failed:hello:rc=1",
            environment_failure=True,
        )
        self.assertEqual(failure_class, regression.ENVIRONMENT_FAILURE_CLASS)
        self.assertEqual(failure_class, "environment_unavailable")
        self.assertIn("429", detail)
        self.assertIn("rate limit", detail)

    def test_aggregate_status_distinguishes_infrastructure_from_regressions(self) -> None:
        cases = [
            ([{"status": regression.PASS}], regression.PASS),
            ([{"status": regression.PASS}, {"status": regression.SKIP}], regression.SKIP),
            ([{"status": regression.SKIP}, {"status": regression.SKIP}], regression.SKIP),
            ([{"status": regression.SKIP}, {"status": regression.FAIL}], regression.FAIL),
        ]
        for rows, expected in cases:
            with self.subTest(rows=rows):
                self.assertEqual(regression.aggregate_family_status(rows), expected)

    def _run_family_with_prompt_row(self, row: dict) -> dict:
        family = regression.FamilySpec(
            family_id="qwen3",
            label="Qwen3",
            model="hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
            context_len=1024,
            runtime_args=[],
            smoke_prompts=["hello"],
            response_contract={},
            coherence_gate=False,
            runtime_expect={},
        )
        prompts = {
            "hello": regression.PromptSpec(
                prompt_id="hello",
                label="Hello",
                prompt="Hello",
                max_tokens=32,
                heuristics={},
            )
        }
        with tempfile.TemporaryDirectory(prefix="ck_reg_v8_env_") as tmp:
            with mock.patch.object(regression, "run_prompt", return_value=row):
                return regression.run_family(
                    family,
                    prompts,
                    mode="fast",
                    run_root=Path(tmp) / "runs",
                    report_dir=Path(tmp) / "reports",
                    force_rebuild=False,
                    cache_dir=Path(tmp) / "cache",
                )

    def test_run_family_classifies_rate_limit_download_as_environment(self) -> None:
        row = {
            "status": regression.FAIL,
            "command": [],
            "model_arg": "hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
            "returncode": 1,
            "stdout": "",
            "stderr": (
                "Error: V8DownloadError: CK_V8_DOWNLOAD_FAILED: download failed for "
                "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf: HTTP error after retries: "
                "HTTP Error 429 (HuggingFace rate limit (HTTP 429) or error page received)"
            ),
            "assistant_raw": "",
            "assistant": "",
            "coherence": {"status": regression.SKIP, "metrics": {}, "reasons": []},
            "bridge_report_path": None,
            "bridge_report": None,
        }
        result = self._run_family_with_prompt_row(row)
        self.assertEqual(result["failure_class"], "environment_unavailable")
        self.assertIn("rate limit", result["failure_detail"])
        self.assertEqual(result["build_status"], regression.FAIL)
        self.assertEqual(result["status"], regression.SKIP)

    def test_run_family_corrupt_local_gguf_stays_build_failure(self) -> None:
        row = {
            "status": regression.FAIL,
            "command": [],
            "model_arg": "/tmp/corrupt-model.gguf",
            "returncode": 1,
            "stdout": "",
            "stderr": "GGUFError: /tmp/corrupt-model.gguf: invalid magic b'<!DO' (expected b'GGUF')",
            "assistant_raw": "",
            "assistant": "",
            "coherence": {"status": regression.SKIP, "metrics": {}, "reasons": []},
            "bridge_report_path": None,
            "bridge_report": None,
        }
        result = self._run_family_with_prompt_row(row)
        self.assertEqual(result["failure_class"], "build_failure")
        self.assertEqual(result["status"], regression.FAIL)


if __name__ == "__main__":
    unittest.main()
