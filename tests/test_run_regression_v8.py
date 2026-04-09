#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
import unittest
import json
from pathlib import Path


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

    def test_manifest_files_are_consistent(self) -> None:
        prompts = regression.load_prompts(ROOT / "version" / "v8" / "regression" / "prompts.json")
        families = regression.load_families(ROOT / "version" / "v8" / "regression" / "families.json", prompts)
        ids = {family.family_id for family in families}
        self.assertEqual(ids, {"gemma", "qwen2", "qwen3", "qwen35", "nanbeige"})
        by_id = {family.family_id: family for family in families}
        self.assertIn("--thinking-mode", by_id["qwen3"].runtime_args)
        self.assertIn("suppressed", by_id["qwen3"].runtime_args)
        self.assertEqual(by_id["nanbeige"].runtime_expect.get("config", {}).get("chat_contract.name"), "llama_chatml")
        self.assertEqual(by_id["gemma"].runtime_expect.get("config", {}).get("rope_layout"), "split")

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


if __name__ == "__main__":
    unittest.main()
