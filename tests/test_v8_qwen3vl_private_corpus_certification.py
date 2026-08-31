from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "certify_qwen3vl_llamacpp_corpus_v8.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("qwen3vl_private_corpus", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Qwen3VLCorpusCertificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_module()

    def test_model_profiles_select_architecture_explicitly(self) -> None:
        qwen3 = SimpleNamespace(
            model_profile="qwen3vl",
            model_label=None,
            chat_template=None,
            composition_circuit=None,
        )
        self.module._apply_model_profile(qwen3)
        self.assertEqual(qwen3.model_label, "Qwen3-VL")
        self.assertEqual(qwen3.chat_template, "qwen3vl")
        self.assertIsNone(qwen3.composition_circuit)

        qwen36 = SimpleNamespace(
            model_profile="qwen36vl",
            model_label=None,
            chat_template=None,
            composition_circuit=None,
        )
        self.module._apply_model_profile(qwen36)
        self.assertEqual(qwen36.model_label, "Qwen3.6-VL")
        self.assertEqual(qwen36.chat_template, "auto")
        self.assertEqual(qwen36.composition_circuit, "qwen36vl")

    def test_model_profile_preserves_explicit_overrides(self) -> None:
        args = SimpleNamespace(
            model_profile="qwen36vl",
            model_label="private-label",
            chat_template="qwen35",
            composition_circuit="explicit-circuit",
        )
        self.module._apply_model_profile(args)
        self.assertEqual(args.model_label, "private-label")
        self.assertEqual(args.chat_template, "qwen35")
        self.assertEqual(args.composition_circuit, "explicit-circuit")

    def test_private_corpus_size_gate_requires_full_manifest(self) -> None:
        rows = [{"index": index} for index in range(1, 41)]
        self.module._require_corpus_size(rows, 40)
        with self.assertRaisesRegex(ValueError, "contains 40 images"):
            self.module._require_corpus_size(rows, 41)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.module._require_corpus_size(rows, 0)

    def test_makefile_exposes_both_private_model_profiles(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("test-qwen3vl-private-corpus-parity-auto:", makefile)
        self.assertIn("--model-profile qwen3vl", makefile)
        self.assertIn("test-qwen36vl-private-corpus-parity-auto:", makefile)
        self.assertIn("--model-profile qwen36vl", makefile)
        self.assertIn(
            "V8_PRIVATE_VISION_CORPUS_MANIFEST ?= $(CK_QWEN3VL_OCR_MANIFEST)",
            makefile,
        )
        self.assertIn(
            "QWEN36VL_PRIVATE_CORPUS_MANIFEST ?= "
            "$(V8_PRIVATE_VISION_CORPUS_MANIFEST)",
            makefile,
        )
        self.assertIn(
            "test-qwen-vl-private-corpus-parity-auto: "
            "test-qwen3vl-private-corpus-parity-auto "
            "test-qwen36vl-private-corpus-parity-auto",
            makefile,
        )

    def test_manifest_order_and_hashes_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "first.jpg").write_bytes(b"first")
            (root / "second.jpg").write_bytes(b"second")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {"id": "private-name-1", "inputs": [{"path": "first.jpg"}]},
                            {"id": "private-name-2", "inputs": [{"path": "second.jpg"}]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = self.module._load_corpus(manifest)
            self.assertEqual([row["index"] for row in rows], [1, 2])
            self.assertEqual(rows[0]["image_sha256"], self.module._sha256_file(root / "first.jpg"))

    def test_redacted_summary_excludes_paths_text_and_sample_ids(self) -> None:
        report = {
            "pass": True,
            "max_new_tokens": 128,
            "ctx_len": 4096,
            "steps": [
                {
                    "generated_prefix": [1],
                    "ck_next": 7,
                    "llama_next": 7,
                    "ck_next_text": "private",
                }
            ],
            "stop_reason": None,
            "first_divergence": None,
            "prefix": {"grid": [36, 28], "tokens": 1008},
            "prompt_tokens_before_image": [1, 2],
            "prompt_tokens_after_image": [3],
            "ck_runtime": {
                "shared_library": {"sha256": "decoder"},
                "engine_library": {"sha256": "engine"},
            },
            "compiler_provenance": {
                "status": "pass",
                "decoder_family": "gcc",
                "engine_family": "gcc",
            },
            "llama_oracle": {"commit": self.module.PINNED_LLAMA_COMMIT},
            "generated_shared_text": "private generated document text",
        }
        row = self.module._redacted_row(
            index=1,
            image_sha256="image",
            prefix_sha256="prefix",
            report=report,
            elapsed={"bridge": 1.0, "parity": 2.0},
            requested_tokens=128,
        )
        encoded = json.dumps(row)
        self.assertNotIn("private", encoded)
        self.assertNotIn("path", encoded)
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["steps"], 1)
        self.assertEqual(row["matched_tokens"], 1)
        self.assertEqual(row["requested_tokens"], 128)
        self.assertEqual(row["prefill_tokens"], 1011)
        self.assertEqual(row["context_tokens_after_comparison"], 1012)
        self.assertEqual(row["elapsed_sec"]["total"], 3.0)
        self.assertEqual(row["elapsed_sec"]["comparison_per_token"], 2.0)

    def test_resume_requires_exact_case_configuration_and_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "case_result.json"
            config = {
                "global_config_sha256": "config",
                "image_index": 1,
                "image_sha256": "image",
            }
            result.write_text(
                json.dumps(
                    {
                        "case_config": config,
                        "redacted_row": {
                            "image_index": 1,
                            "status": "pass",
                            "native_cli": {"pass": True},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (Path(temporary) / "native_comparison.json").write_text(
                json.dumps({"pass": True}),
                encoding="utf-8",
            )
            self.assertIsNotNone(self.module._resumed_row(result, config))
            changed = dict(config, image_sha256="changed")
            self.assertIsNone(self.module._resumed_row(result, changed))

    def test_resume_rejects_missing_native_comparison_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "case_result.json"
            config = {
                "global_config_sha256": "config",
                "image_index": 1,
                "image_sha256": "image",
            }
            result.write_text(
                json.dumps(
                    {
                        "case_config": config,
                        "redacted_row": {
                            "image_index": 1,
                            "status": "pass",
                            "native_cli": {"pass": True},
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(self.module._resumed_row(result, config))

    def test_summary_fails_for_any_divergence(self) -> None:
        selected = [{"index": 1}, {"index": 2}]
        config = self._config()
        summary = self.module._summary(
            selected=selected,
            rows=[
                {"image_index": 1, "status": "pass"},
                {"image_index": 2, "status": "fail"},
            ],
            config=config,
        )
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        encoded = json.dumps(summary)
        self.assertNotIn("/private", encoded)
        self.assertNotIn("llama_root", encoded)
        self.assertNotIn('"decoder"', encoded)
        self.assertNotIn('"mmproj"', encoded)

    def test_redacted_match_count_uses_common_parity_prefix(self) -> None:
        report = {
            "pass": False,
            "ctx_len": 1400,
            "steps": [
                {"ck_next": 7, "llama_next": 7},
                {"ck_next": 8, "llama_next": 9},
                {"ck_next": 10, "llama_next": 10},
            ],
            "first_divergence": {"step": 1},
            "prefix": {"tokens": 1008},
            "prompt_tokens_before_image": [],
            "prompt_tokens_after_image": [],
        }
        row = self.module._redacted_row(
            index=1,
            image_sha256="image",
            prefix_sha256="prefix",
            report=report,
            elapsed={},
            requested_tokens=128,
            native_comparison={"pass": False, "native_tokens": 128},
        )
        self.assertEqual(row["steps"], 3)
        self.assertEqual(row["matched_tokens"], 1)

    def test_localization_summary_discloses_native_cli_was_skipped(self) -> None:
        config = dict(self._config(), skip_native_cli=True)
        summary = self.module._summary(
            selected=[{"index": 1}],
            rows=[{"image_index": 1, "status": "pass"}],
            config=config,
        )
        self.assertEqual(summary["certification_scope"], "localization")
        self.assertIn("native CLI not run", summary["comparison"])

    def test_progress_line_discloses_skipped_native_cli(self) -> None:
        line = self.module._progress_line(
            {
                "image_index": 1,
                "status": "fail",
                "matched_tokens": 8,
                "requested_tokens": 128,
                "native_cli": None,
            },
            completed=1,
            requested=5,
        )
        self.assertIn("matched=8/128", line)
        self.assertIn("native=skipped", line)

    def test_summary_treats_execution_errors_as_failures(self) -> None:
        summary = self.module._summary(
            selected=[{"index": 1}],
            rows=[{"image_index": 1, "status": "error", "error_sha256": "hash"}],
            config=self._config(),
        )
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["failed"], 1)

    def test_resume_can_skip_redundant_native_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "case_result.json"
            expected = {"global_config_sha256": "cfg", "image_index": 1, "image_sha256": "hash"}
            result.write_text(
                json.dumps({"case_config": expected, "redacted_row": {"status": "pass", "native_cli": None}}),
                encoding="utf-8",
            )
            self.assertIsNotNone(
                self.module._resumed_row(result, expected, require_native=False)
            )
            self.assertIsNone(
                self.module._resumed_row(result, expected, require_native=True)
            )

    def test_production_commands_use_batched_exact_runtime_parity(self) -> None:
        parser_values = type(
            "Args",
            (),
            {
                "context_len": 4096,
                "max_new_tokens": 128,
                "top_k": 16,
                "threads": 20,
                "ck_threads": 20,
                "llama_required_isa": "avx2",
                "append_on_divergence": "llama",
            },
        )()
        command = self.module._parity_command(
            parser_values,
            bridge_report=Path("bridge.json"),
            prefix_path=Path("prefix.f32"),
            workdir=Path("work"),
            report_path=Path("report.json"),
        )
        rendered = " ".join(map(str, command))
        self.assertIn("--reuse-bridge-decoder-runtime-exact", rendered)
        self.assertIn("--llama-decode-mode batched", rendered)
        self.assertIn("--append-on-divergence llama", rendered)
        self.assertIn("--max-new-tokens 128", rendered)
        self.assertIn("--gemm-schedule auto", rendered)

        parser_values.encoder_runtime = Path("encoder-runtime")
        command = self.module._parity_command(
            parser_values,
            bridge_report=Path("bridge.json"),
            prefix_path=Path("prefix.f32"),
            workdir=Path("work"),
            report_path=Path("report.json"),
        )
        self.assertIn(
            "--ck-engine-so encoder-runtime/libckernel_engine.so",
            " ".join(map(str, command)),
        )

    def test_bridge_command_accepts_prebuilt_encoder_runtime(self) -> None:
        args = SimpleNamespace(
            decoder_gguf=Path("decoder.gguf"),
            mmproj_gguf=None,
            encoder_runtime=Path("encoder-runtime"),
            prompt="Extract visible form fields as compact JSON.",
            chat_template="auto",
            image_max_tokens=1024,
            context_len=4096,
            top_k=16,
            gemm_schedule="auto",
        )
        command = self.module._bridge_command(
            args,
            image=Path("image.jpg"),
            runtime_dir=Path("runtime"),
            prefix_path=Path("prefix.f32"),
        )
        rendered = " ".join(map(str, command))
        self.assertIn("--encoder-runtime encoder-runtime", rendered)
        self.assertNotIn("--encoder-gguf", command)
        self.assertIn("--chat-template auto", rendered)

    def test_bridge_command_uses_requested_chat_template(self) -> None:
        args = SimpleNamespace(
            decoder_gguf=Path("decoder.gguf"),
            mmproj_gguf=Path("mmproj.gguf"),
            encoder_runtime=None,
            prompt="Extract text.",
            chat_template="auto",
            composition_circuit="",
            image_max_tokens=1024,
            context_len=2048,
            top_k=16,
            gemm_schedule="auto",
        )
        command = self.module._bridge_command(
            args,
            image=Path("image.png"),
            runtime_dir=Path("runtime"),
            prefix_path=Path("prefix.f32"),
        )
        rendered = " ".join(map(str, command))
        self.assertIn("--chat-template auto", rendered)

    def test_bridge_command_forwards_explicit_composition_circuit(self) -> None:
        args = SimpleNamespace(
            decoder_gguf=Path("decoder.gguf"),
            mmproj_gguf=Path("mmproj.gguf"),
            encoder_runtime=None,
            prompt="Extract text.",
            chat_template="auto",
            composition_circuit="qwen36vl",
            image_max_tokens=1024,
            context_len=2048,
            top_k=16,
            gemm_schedule="auto",
        )
        command = self.module._bridge_command(
            args,
            image=Path("image.png"),
            runtime_dir=Path("runtime"),
            prefix_path=Path("prefix.f32"),
        )
        rendered = " ".join(map(str, command))
        self.assertIn("--composition-circuit qwen36vl", rendered)

    def test_native_cli_command_requires_generated_abi_and_exact_trace(self) -> None:
        args = SimpleNamespace(
            native_cli=Path("build/ck-cli-v8"),
            max_new_tokens=128,
            context_len=4096,
        )
        command = self.module._native_cli_command(
            args,
            bridge_report=Path("case/bridge_report.json"),
            runtime_dir=Path("runtime"),
            trace_path=Path("case/native_token_trace.json"),
        )
        rendered = " ".join(map(str, command))
        self.assertIn("runtime/decoder/libdecoder_v8.so", rendered)
        self.assertIn("--bridge-report case/bridge_report.json", rendered)
        self.assertIn("--require-generated-abi", command)
        self.assertIn("--token-trace-json case/native_token_trace.json", rendered)
        self.assertIn("--gemm-schedule auto", rendered)

    def test_native_trace_must_match_python_and_llama_pre_eos_tokens(self) -> None:
        report = {
            "pass": True,
            "stop_token_ids": [99],
            "steps": [
                {"ck_next": 7, "llama_next": 7},
                {"ck_next": 8, "llama_next": 8},
                {"ck_next": 99, "llama_next": 99},
            ],
        }
        trace = {
            "schema": "cke.native_token_trace",
            "schema_version": 1,
            "token_ids": [7, 8],
        }
        comparison = self.module._compare_native_trace(report, trace)
        self.assertTrue(comparison["pass"])
        self.assertEqual(comparison["native_tokens"], 2)
        trace["token_ids"] = [7, 9]
        comparison = self.module._compare_native_trace(report, trace)
        self.assertFalse(comparison["pass"])
        self.assertEqual(
            comparison["native_vs_python_first_divergence"],
            {"step": 1, "native_token": 9, "reference_token": 8},
        )

    def test_native_trace_attributes_a_truncated_oracle_divergence(self) -> None:
        report = {
            "pass": False,
            "steps": [
                {"ck_next": 7, "llama_next": 7},
                {"ck_next": 606, "llama_next": 627},
            ],
        }
        trace = {
            "schema": "cke.native_token_trace",
            "schema_version": 1,
            "token_ids": [7, 606, 42],
        }
        comparison = self.module._compare_native_trace(report, trace)
        self.assertFalse(comparison["pass"])
        self.assertTrue(comparison["native_matches_python_captured_prefix"])
        self.assertFalse(comparison["three_way_comparison_complete"])
        self.assertIsNone(comparison["native_vs_python_first_divergence"])
        self.assertEqual(
            comparison["native_vs_llama_first_divergence"],
            {"step": 1, "native_token": 606, "reference_token": 627},
        )

    def test_logged_command_accepts_completed_numerical_failure_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            elapsed = self.module._run_logged(
                [sys.executable, "-c", "raise SystemExit(3)"],
                env={},
                log_path=Path(temporary) / "numerical.log",
                dry_run=False,
                accepted_returncodes=(0, 3),
            )
            self.assertGreaterEqual(elapsed, 0.0)

    def test_redacted_dry_run_does_not_print_private_paths(self) -> None:
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            self.module._run_logged(
                ["runner", "--image", "/private/confidential-image.jpg"],
                env={},
                log_path=Path("unused.log"),
                dry_run=True,
                show_dry_run_command=False,
            )
        rendered = output.getvalue()
        self.assertIn("private command redacted", rendered)
        self.assertNotIn("confidential-image", rendered)

    def test_redacted_row_excludes_matched_eos_decision_from_context(self) -> None:
        report = {
            "pass": True,
            "ctx_len": 4096,
            "stop_token_ids": [99],
            "steps": [
                {"ck_next": 7, "llama_next": 7},
                {"ck_next": 8, "llama_next": 8},
                {"ck_next": 99, "llama_next": 99},
            ],
            "prefix": {"tokens": 10},
            "prompt_tokens_before_image": [1],
            "prompt_tokens_after_image": [2],
        }
        native = {
            "pass": True,
            "native_tokens": 2,
            "python_ck_tokens": 2,
            "llama_tokens": 2,
        }
        row = self.module._redacted_row(
            index=1,
            image_sha256="image",
            prefix_sha256="prefix",
            report=report,
            elapsed={"parity": 3.0},
            requested_tokens=128,
            native_comparison=native,
        )
        self.assertEqual(row["steps"], 3)
        self.assertEqual(row["matched_tokens"], 2)
        self.assertEqual(row["context_tokens_after_comparison"], 14)

    def test_progress_line_distinguishes_tokens_context_and_time(self) -> None:
        row = {
            "image_index": 7,
            "status": "pass",
            "matched_tokens": 128,
            "requested_tokens": 128,
            "prefix_tokens": 1008,
            "prompt_tokens": 27,
            "prefill_tokens": 1035,
            "context_tokens_after_comparison": 1163,
            "context_capacity": 4096,
            "elapsed_sec": {
                "bridge": 75.5,
                "parity": 100.0,
                "total": 175.5,
                "comparison_per_token": 0.78125,
            },
        }
        line = self.module._progress_line(
            row,
            completed=7,
            requested=40,
            resumed=True,
        )
        self.assertIn("matched=128/128", line)
        self.assertIn("prefix=1008", line)
        self.assertIn("prefill=1035", line)
        self.assertIn("context=1163/4096", line)
        self.assertIn("total=175.50s", line)
        self.assertIn("compare=0.781s/token-pair", line)
        self.assertTrue(line.endswith("resumed"))

    def test_timing_summary_reports_corpus_total_and_mean(self) -> None:
        timing = self.module._timing_summary(
            [
                {"elapsed_sec": {"total": 10.0}},
                {"elapsed_sec": {"total": 20.0}},
            ]
        )
        self.assertEqual(timing["total_sec"], 30.0)
        self.assertEqual(timing["mean_sec_per_image"], 15.0)
        self.assertEqual(timing["min_sec_per_image"], 10.0)
        self.assertEqual(timing["max_sec_per_image"], 20.0)

    def test_private_console_requires_tty_or_explicit_opt_in(self) -> None:
        with mock.patch.dict("os.environ", {"CI": "true"}):
            self.assertFalse(
                self.module._private_console_enabled(
                    SimpleNamespace(show_private_details=None)
                )
            )
        self.assertTrue(
            self.module._private_console_enabled(
                SimpleNamespace(show_private_details=True)
            )
        )
        self.assertFalse(
            self.module._private_console_enabled(
                SimpleNamespace(show_private_details=False)
            )
        )

    def test_private_console_renders_local_details_without_changing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case_dir = Path(temporary)
            (case_dir / "bridge_report.json").write_text(
                json.dumps(
                    {
                        "encoder_report": {
                            "source_image_size": [2200, 1700],
                            "image_width": 1152,
                            "image_height": 896,
                        },
                        "timings": {
                            "encoder_execute_ms": 20000.0,
                            "decoder_forward_mixed_ms": 85000.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (case_dir / "parity.json").write_text(
                json.dumps(
                    {
                        "generated_shared_text": '{"title":"Monthly Report"}',
                        "first_divergence": None,
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "image_index": 1,
                "image_sha256": "image-hash",
                "grid": [36, 28],
                "status": "pass",
                "matched_tokens": 128,
                "requested_tokens": 128,
                "prefix_tokens": 1008,
                "prefill_tokens": 1035,
                "context_tokens_after_comparison": 1163,
                "context_capacity": 4096,
                "stop_reason": None,
                "native_cli": {
                    "pass": True,
                    "native_tokens": 128,
                    "python_ck_tokens": 128,
                },
                "elapsed_sec": {
                    "bridge": 110.0,
                    "parity": 90.0,
                    "total": 200.0,
                },
            }
            output = io.StringIO()
            with mock.patch("sys.stdout", output):
                self.module._print_private_case_details(
                    sample={
                        "index": 1,
                        "image": Path("/private/form.jpg"),
                        "image_sha256": "image-hash",
                    },
                    row=row,
                    case_dir=case_dir,
                    prompt="Extract fields.",
                )
            rendered = output.getvalue()
            self.assertIn("EXACT MATCH", rendered)
            self.assertIn("/private/form.jpg", rendered)
            self.assertIn("source 2200x1700 -> processed 1152x896", rendered)
            self.assertIn("128/128 exact native/Python/llama pre-EOS tokens", rendered)
            self.assertIn('Output (native CLI == Python CKE == llama.cpp', rendered)
            self.assertIn('{"title":"Monthly Report"}', rendered)
            self.assertNotIn("/private/form.jpg", json.dumps(row))

    def _config(self) -> dict[str, object]:
        return {
            "version": 1,
            "cke_commit": "cke",
            "model_profile": "qwen3vl",
            "model_label": "Qwen3-VL",
            "composition_circuit": None,
            "manifest_sha256": "manifest",
            "decoder": {"path": "/private/decoder.gguf"},
            "mmproj": {"path": "/private/mmproj.gguf"},
            "llama_root": "/private/llama.cpp",
            "llama_commit": self.module.PINNED_LLAMA_COMMIT,
            "expected_llama_commit": self.module.PINNED_LLAMA_COMMIT,
            "compiler": "gcc",
            "prompt_sha256": "prompt",
            "context_len": 4096,
            "image_max_tokens": 1024,
            "max_new_tokens": 128,
            "require_images": 40,
            "append_on_divergence": "stop",
            "chat_template": "qwen3vl",
            "threads": 20,
            "ck_threads": 20,
            "top_k": 16,
            "llama_required_isa": "avx2",
            "native_cli_sha256": "native-cli",
        }


if __name__ == "__main__":
    unittest.main()
