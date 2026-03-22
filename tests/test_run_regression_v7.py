#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))

import run_regression_v7 as regression  # type: ignore


class RegressionHarnessTests(unittest.TestCase):
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

    def test_extract_response_output_ignores_runner_chatter(self) -> None:
        text = (
            "make[1]: Entering directory '/tmp/work'\n"
            "Loading model from /tmp/model...\n"
            "Prompt: Hello\n"
            "Response: Hello there!\n\n\n"
            "\x1b[90mprompt eval: 1.0 ms / 1 tokens\x1b[0m\n"
        )
        self.assertEqual(regression._extract_assistant_output(text), "Hello there!")

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

    def test_greeting_coherence_accepts_short_valid_reply(self) -> None:
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
            "expected_keywords": ["hello", "hi", "help", "assist", "here", "today", "how can"],
            "min_keyword_hits": 1,
        }
        result = regression.assess_coherence("I'm here, I am a human", heuristics)
        self.assertEqual(result["status"], regression.PASS)

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

    def test_normalize_assistant_output_drops_unterminated_think_prefix(self) -> None:
        text = "<think>\ninternal planning that never closes"
        cleaned = regression.normalize_assistant_output(
            text,
            {
                "strip_think_blocks": True,
                "trim_whitespace": True,
            },
        )
        self.assertEqual(cleaned, "")

    def test_greeting_keyword_match_is_token_aware(self) -> None:
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
            "expected_keywords": ["hello", "hi", "help", "assist"],
            "min_keyword_hits": 1,
        }
        result = regression.assess_coherence("What is the best thing?", heuristics)
        self.assertEqual(result["status"], regression.FAIL)
        self.assertIn("keyword_hits:0<1", result["reasons"])

    def test_gibberish_coherence_fails(self) -> None:
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
        text = "حدىحدىحدىحدىardiaحدىحدىحدىحدى\uFFFD\uFFFD"
        result = regression.assess_coherence(text, heuristics)
        self.assertEqual(result["status"], regression.FAIL)
        self.assertTrue(result["reasons"])

    def test_code_prompt_requires_real_code_markers(self) -> None:
        heuristics = {
            "min_chars": 48,
            "min_words": 12,
            "min_printable_ratio": 0.96,
            "max_replacement_chars": 0,
            "max_repeated_4gram": 4,
            "max_duplicate_lines": 6,
            "expected_keywords": ["c", "python", "sql"],
            "min_keyword_hits": 2,
            "required_substrings_any_of": ["```", "#include", "def ", "select "],
            "min_required_substrings_any_of_hits": 1,
        }
        text = (
            "I can explain C, Python, and SQL for you. "
            "These languages are useful in different situations."
        )
        result = regression.assess_coherence(text, heuristics)
        self.assertEqual(result["status"], regression.FAIL)
        self.assertIn("required_markers:0<1", result["reasons"])

    def test_manifest_files_are_consistent(self) -> None:
        prompts = regression.load_prompts(ROOT / "version" / "v7" / "regression" / "prompts.json")
        families = regression.load_families(ROOT / "version" / "v7" / "regression" / "families.json", prompts)
        ids = {family.family_id for family in families}
        self.assertIn("qwen35", ids)
        self.assertIn("gemma", ids)
        by_id = {family.family_id: family for family in families}
        self.assertIn("<|im_end|>", by_id["qwen35"].response_contract.get("stop_text_markers", []))
        self.assertIn("<end_of_turn>", by_id["gemma"].response_contract.get("stop_text_markers", []))
        self.assertTrue(by_id["nanbeige"].response_contract.get("strip_think_blocks"))
        self.assertTrue(by_id["qwen2"].coherence_gate)
        self.assertFalse(by_id["gemma"].coherence_gate)
        self.assertFalse(by_id["nanbeige"].coherence_gate)
        self.assertIn("--repeat-penalty", by_id["qwen35"].runtime_args)
        self.assertIn("1.1", by_id["gemma"].runtime_args)
        self.assertIn("--thinking-mode", by_id["qwen3"].runtime_args)
        self.assertIn("suppressed", by_id["qwen3"].runtime_args)
        self.assertEqual(by_id["gemma"].smoke_prompts, ["hello", "france_capital"])
        self.assertEqual(by_id["qwen2"].smoke_prompts, ["hello", "france_capital", "code_bundle"])
        self.assertEqual(by_id["qwen3"].smoke_prompts, ["hello", "france_capital", "code_bundle"])
        self.assertEqual(by_id["nanbeige"].smoke_prompts, ["hello"])
        self.assertTrue(by_id["gemma"].parity.get("always_run"))
        self.assertTrue(by_id["nanbeige"].parity.get("always_run"))
        self.assertEqual(by_id["gemma"].runtime_expect.get("config", {}).get("rope_layout"), "split")
        self.assertEqual(by_id["nanbeige"].runtime_expect.get("config", {}).get("rope_layout"), "pairwise")
        self.assertEqual(by_id["qwen35"].runtime_expect.get("config", {}).get("chat_contract.name"), "qwen35")

    def test_resolve_gguf_path_accepts_direct_cache_root_layout(self) -> None:
        cache_root = Path("/tmp/test_run_regression_v7_cache")
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
        with tempfile.TemporaryDirectory(prefix="ck_reg_contract_") as tmp:
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
                [{"stdout": "Using built-in C tokenizer (llama)\n"}],
                {
                    "stdout_contains": ["Using built-in C tokenizer"],
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

    def test_failure_classification_prefers_stitch_and_parity(self) -> None:
        failure_class, detail = regression.classify_family_result(
            build_status=regression.PASS,
            smoke_status=regression.PASS,
            coherence_status=regression.FAIL,
            coherence_gate=True,
            contract_result={"status": regression.SKIP},
            stitch_result={"status": regression.FAIL},
            kernel_result={"status": regression.SKIP},
            first_token_result={"status": regression.SKIP},
            divergence_result={"status": regression.SKIP},
            failure_reason="coherence_failed:hello",
        )
        self.assertEqual(failure_class, "stitch_failure")
        self.assertIn("lowered IR", detail)

        failure_class, detail = regression.classify_family_result(
            build_status=regression.PASS,
            smoke_status=regression.PASS,
            coherence_status=regression.FAIL,
            coherence_gate=True,
            contract_result={"status": regression.SKIP},
            stitch_result={"status": regression.PASS},
            kernel_result={"status": regression.PASS},
            first_token_result={"status": regression.FAIL},
            divergence_result={"status": regression.FAIL},
            failure_reason="coherence_failed:hello",
        )
        self.assertEqual(failure_class, "parity_divergence")
        self.assertIn("parity", detail)

        failure_class, detail = regression.classify_family_result(
            build_status=regression.PASS,
            smoke_status=regression.PASS,
            coherence_status=regression.PASS,
            coherence_gate=True,
            contract_result={"status": regression.FAIL},
            stitch_result={"status": regression.SKIP},
            kernel_result={"status": regression.SKIP},
            first_token_result={"status": regression.SKIP},
            divergence_result={"status": regression.SKIP},
            failure_reason="",
        )
        self.assertEqual(failure_class, "contract_failure")
        self.assertIn("contract", detail)

    def test_failure_classification_ignores_coherence_when_gate_disabled(self) -> None:
        failure_class, detail = regression.classify_family_result(
            build_status=regression.PASS,
            smoke_status=regression.PASS,
            coherence_status=regression.FAIL,
            coherence_gate=False,
            contract_result={"status": regression.SKIP},
            stitch_result={"status": regression.SKIP},
            kernel_result={"status": regression.SKIP},
            first_token_result={"status": regression.SKIP},
            divergence_result={"status": regression.SKIP},
            failure_reason="coherence_failed:hello",
        )
        self.assertEqual(failure_class, "pass")
        self.assertEqual(detail, "")


if __name__ == "__main__":
    unittest.main()
