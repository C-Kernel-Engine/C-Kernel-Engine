#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v8" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "compare_multitoken_logits_v8_tests",
    SCRIPT_DIR / "compare_multitoken_logits_v8.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class MultitokenParityEOSContractTests(unittest.TestCase):
    def test_trajectory_rejects_a_runtime_without_batched_prefill_before_init(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "libmodel.so"
            runtime.write_bytes(b"test runtime")
            library = mock.Mock()
            library.ck_model_get_capabilities.return_value = 0
            with mock.patch.object(runner.ctypes, "CDLL", return_value=library):
                with self.assertRaisesRegex(RuntimeError, "without generated prefill"):
                    runner.load_ck_greedy_trajectory(
                        model_dir=Path(directory), prompt_tokens=[7],
                        max_new_tokens=1, runtime_so=runtime,
                    )
            library.ck_model_init.assert_not_called()

    def test_exact_acceptance_rejects_matching_tokens_with_logit_drift(self):
        report = {"pass": True, "steps": [{"step": 0, "top1_match": True, "bit_exact": False}]}
        runner._apply_acceptance_contract(report, True)
        self.assertFalse(report["pass"])
        self.assertEqual(report["numerical_summary"]["first_nonexact_step"], 0)

    def test_exact_acceptance_requires_nonempty_measured_evidence(self):
        for steps in ([], [{"step": 0, "top1_match": True}]):
            report = {"pass": True, "steps": steps}
            runner._apply_acceptance_contract(report, True)
            self.assertFalse(report["pass"])

    def test_exact_acceptance_accepts_exact_rows_and_keeps_top1_mode_explicit(self):
        report = {"pass": True, "steps": [{"step": 0, "bit_exact": True}]}
        runner._apply_acceptance_contract(report, True)
        self.assertTrue(report["pass"])
        report = {"pass": True, "steps": [{"step": 0, "bit_exact": False}]}
        runner._apply_acceptance_contract(report, False)
        self.assertTrue(report["pass"])
        self.assertEqual(report["acceptance_contract"], "greedy_top1")

    def _run(self, ck_logits: np.ndarray, llama_logits: np.ndarray) -> dict:
        with mock.patch.object(runner, "run_llama_logits", return_value={"logits": llama_logits}), \
             mock.patch.object(runner, "load_ck_logits", return_value={"logits": ck_logits}):
            return runner.run_multitoken_parity(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=64,
                ctx_len=128,
                top_k=3,
                threads=1,
                append_on_divergence="stop",
                ck_prefill_mode="batched",
                llama_decode_mode="batched",
                llama_no_repack=False,
                stop_token_ids={2},
            )

    def test_matched_declared_stop_ends_without_post_eos_comparison(self) -> None:
        logits = np.asarray([0.0, 1.0, 4.0], dtype=np.float32)
        report = self._run(logits, logits)
        self.assertTrue(report["pass"])
        self.assertEqual(report["matched_stop_token"], 2)
        self.assertEqual(len(report["steps"]), 1)

    def test_unmatched_stop_candidate_is_still_a_failure(self) -> None:
        ck_logits = np.asarray([0.0, 1.0, 4.0], dtype=np.float32)
        llama_logits = np.asarray([0.0, 5.0, 1.0], dtype=np.float32)
        report = self._run(ck_logits, llama_logits)
        self.assertFalse(report["pass"])
        self.assertIsNone(report["matched_stop_token"])
        self.assertEqual(report["first_divergence"]["ck_next"], 2)

    def test_replay_configures_ck_threads_before_loading_runtime(self) -> None:
        logits = np.asarray([0.0, 1.0, 4.0], dtype=np.float32)
        configured = {
            "CK_NUM_THREADS": "7",
            "CK_THREADPOOL_THREADS": "7",
            "OMP_NUM_THREADS": "7",
        }
        with mock.patch.object(runner, "run_llama_logits", return_value={"logits": logits}), \
             mock.patch.object(runner, "load_ck_logits", return_value={"logits": logits}), \
             mock.patch.object(
                 runner, "_configure_ck_threads", return_value=configured
             ) as configure:
            report = runner.run_multitoken_parity(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=1,
                ctx_len=128,
                top_k=3,
                threads=7,
                append_on_divergence="stop",
                ck_prefill_mode="batched",
                llama_decode_mode="batched",
                llama_no_repack=False,
            )
        configure.assert_called_once_with(7)
        self.assertEqual(report["ck_thread_environment"], configured)


class PersistentTrajectoryParityTests(unittest.TestCase):
    def test_hidden_capture_retains_float_and_integer_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            f32 = root / "tok_0002_layer_006_moe_routing_weights.f32"
            i32 = root / "tok_0002_layer_006_moe_selected_experts.i32"
            unrelated = root / "tok_0002_layer_006_notes.txt"
            empty = root / "tok_0002_layer_006_empty.f32"
            f32.write_bytes(b"\0" * 32)
            i32.write_bytes(b"\0" * 32)
            unrelated.write_text("ignored", encoding="utf-8")
            empty.touch()

            self.assertEqual(runner._hidden_capture_paths(root), [f32, i32])

    def test_llama_trajectory_rejects_stale_file_backed_logits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "llama_logits_sequence.f32"
            output.write_bytes(b"stale")
            function_globals = runner.run_llama_greedy_trajectory.__globals__
            with mock.patch.dict(
                function_globals,
                {"ensure_llama_helper": lambda: Path("/bin/true")},
            ):
                with self.assertRaisesRegex(ValueError, "refusing stale evidence"):
                    runner.run_llama_greedy_trajectory(
                        Path("/tmp/model.gguf"),
                        [7],
                        2,
                        128,
                        3,
                        1,
                        logits_sequence_out=output,
                        load_logits=False,
                    )

    def test_trajectory_size_estimate_reads_runtime_vocab(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "config.json").write_text(
                '{"vocab_size": 248320}', encoding="utf-8"
            )
            estimated = runner._trajectory_logits_bytes(model_dir, 1200)

        self.assertEqual(estimated, 1200 * 248320 * 4)
        self.assertGreater(estimated, runner._AUTO_STREAM_LOGITS_BYTES)
        self.assertEqual(
            runner._select_trajectory_storage(
                "auto",
                capture_requested=False,
                estimated_logits_bytes=estimated,
            ),
            "stream",
        )

    def test_auto_storage_preserves_memory_mode_for_boundary_capture(self) -> None:
        self.assertEqual(
            runner._select_trajectory_storage(
                "auto",
                capture_requested=True,
                estimated_logits_bytes=1024 * 1024 * 1024,
            ),
            "memory",
        )
        with self.assertRaisesRegex(ValueError, "does not support boundary capture"):
            runner._select_trajectory_storage(
                "stream",
                capture_requested=True,
                estimated_logits_bytes=1024 * 1024 * 1024,
            )

    def test_streaming_trajectory_returns_compact_metrics_and_removes_oracle(self) -> None:
        seen: dict = {}

        def llama_run(*_args, **kwargs):
            path = kwargs["logits_sequence_out"]
            path.write_bytes(b"oracle")
            seen["path"] = path
            seen["load_logits"] = kwargs["load_logits"]
            return {
                "logits": None,
                "generated_tokens": [1, 2],
                "capture": {},
                "layer_profile": None,
            }

        def ck_run(**kwargs):
            self.assertTrue(kwargs["reference_logits_path"].is_file())
            self.assertEqual(kwargs["forced_tokens"], [1, 2])
            self.assertTrue(kwargs["stop_on_top1_divergence"])
            return {
                "logits": None,
                "generated_tokens": [1, 2],
                "stream_steps": [
                    {
                        "step": 0, "prefix_len": 1, "ck_next": 1,
                        "llama_next": 1, "top1_match": True, "bit_exact": True,
                    },
                    {
                        "step": 1, "prefix_len": 2, "ck_next": 2,
                        "llama_next": 2, "top1_match": True, "bit_exact": True,
                    },
                ],
                "thread_environment": {"CK_NUM_THREADS": "7"},
            }

        with mock.patch.object(
            runner, "run_llama_greedy_trajectory", side_effect=llama_run
        ), mock.patch.object(
            runner, "load_ck_greedy_trajectory_isolated", side_effect=ck_run
        ):
            report = runner.run_multitoken_trajectory_parity_streaming(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=2,
                ctx_len=128,
                top_k=3,
                threads=7,
                llama_no_repack=False,
                stop_token_ids={2},
            )

        self.assertTrue(report["pass"])
        self.assertEqual(report["execution_mode"], "persistent_greedy_trajectory_streaming")
        self.assertEqual(report["logits_storage"]["exact_steps"], 2)
        self.assertEqual(report["logits_storage"]["compared_steps"], 2)
        self.assertFalse(report["logits_storage"]["temporary_artifact_retained"])
        self.assertFalse(seen["load_logits"])
        self.assertFalse(seen["path"].exists())

    def test_capture_identity_compares_float32_bits_not_only_values(self) -> None:
        positive_zero = np.asarray([[0.0, 1.0]], dtype=np.float32)
        negative_zero = positive_zero.copy()
        negative_zero[0, 0] = np.float32(-0.0)

        result = runner._compare_ck_trajectory_identity(
            {"logits": positive_zero},
            {"logits": negative_zero},
            top_k=2,
        )

        self.assertFalse(result["exact"])
        self.assertEqual(result["first_different_step"], 0)

    def test_capture_identity_accepts_identical_nan_payload_bits(self) -> None:
        bits = np.asarray([[0x7FC01234, 0x3F800000]], dtype=np.uint32)
        first = bits.view(np.float32)
        second = bits.copy().view(np.float32)

        result = runner._compare_ck_trajectory_identity(
            {"logits": first},
            {"logits": second},
            top_k=2,
        )

        self.assertTrue(result["exact"])

    def test_capture_neutrality_accepts_bit_exact_aggregate_capture(self) -> None:
        rows = np.asarray([[0.0, 4.0, 1.0], [0.0, 1.0, 4.0]], dtype=np.float32)
        calls = []

        def run(**kwargs):
            calls.append(kwargs)
            result = {"logits": rows, "generated_tokens": [1, 2], "vocab": 3}
            if kwargs.get("dump_step") is not None:
                result["capture"] = {"artifacts": [{"path": "/tmp/aggregate.f32"}]}
            return result

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            runner, "load_ck_greedy_trajectory_isolated", side_effect=run
        ):
            result = runner.run_ck_capture_with_neutrality(
                model_dir=Path("/tmp/model"), prompt_tokens=[7], max_new_tokens=2,
                top_k=3, threads=1, runtime_so=None, dump_step=1,
                dump_dir=Path(td) / "capture", dump_layer=63,
                dump_names="attn_out,layer_out", dump_format="hidden",
                dump_kv_layer=None, stop_token_ids=set(),
            )

        neutrality = result["capture"]["neutrality"]
        self.assertEqual(len(calls), 3)
        self.assertEqual(neutrality["status"], "accepted")
        self.assertEqual(neutrality["accepted_mode"], "aggregate")
        self.assertTrue(neutrality["baseline_repeatability"]["exact"])
        self.assertTrue(neutrality["aggregate_capture"]["exact"])

    def test_capture_neutrality_rejects_nondeterministic_control(self) -> None:
        first = np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32)
        second = np.asarray([[0.0, 3.0, 1.0]], dtype=np.float32)
        results = [
            {"logits": first, "generated_tokens": [1], "vocab": 3},
            {"logits": second, "generated_tokens": [1], "vocab": 3},
        ]
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            runner, "load_ck_greedy_trajectory_isolated", side_effect=results
        ) as run:
            result = runner.run_ck_capture_with_neutrality(
                model_dir=Path("/tmp/model"), prompt_tokens=[7], max_new_tokens=1,
                top_k=3, threads=1, runtime_so=None, dump_step=0,
                dump_dir=Path(td) / "capture", dump_layer=63,
                dump_names="layer_out", dump_format="hidden",
                dump_kv_layer=None, stop_token_ids=set(),
            )

        neutrality = result["capture"]["neutrality"]
        self.assertEqual(run.call_count, 2)
        self.assertEqual(neutrality["status"], "rejected")
        self.assertEqual(neutrality["reason"], "uncaptured_runtime_is_not_repeatable")
        self.assertEqual(result["capture"]["artifacts"], [])

    def test_parallel_nondeterminism_skips_expensive_reference_by_default(self) -> None:
        parallel_a = np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32)
        parallel_b = np.asarray([[0.0, 3.0, 1.0]], dtype=np.float32)
        results = [
            {"logits": parallel_a, "generated_tokens": [1], "vocab": 3},
            {"logits": parallel_b, "generated_tokens": [1], "vocab": 3},
        ]
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            runner, "load_ck_greedy_trajectory_isolated", side_effect=results
        ) as run:
            result = runner.run_ck_capture_with_neutrality(
                model_dir=Path("/tmp/model"), prompt_tokens=[7], max_new_tokens=1,
                top_k=3, threads=24, runtime_so=None, dump_step=0,
                dump_dir=Path(td) / "capture", dump_layer=63,
                dump_names="layer_out", dump_format="hidden",
                dump_kv_layer=None, stop_token_ids=set(),
            )

        diagnostic = result["capture"]["neutrality"]["single_thread_simd_reference"]
        self.assertEqual(run.call_count, 2)
        self.assertFalse(diagnostic["attempted"])
        self.assertEqual(diagnostic["reason"], "not_requested")
        self.assertEqual(result["capture"]["artifacts"], [])

    def test_parallel_nondeterminism_runs_one_thread_simd_reference(self) -> None:
        parallel_a = np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32)
        parallel_b = np.asarray([[0.0, 3.0, 1.0]], dtype=np.float32)
        single = np.asarray([[0.0, 3.5, 1.0]], dtype=np.float32)
        results = [
            {"logits": parallel_a, "generated_tokens": [1], "vocab": 3},
            {"logits": parallel_b, "generated_tokens": [1], "vocab": 3},
            {"logits": single, "generated_tokens": [1], "vocab": 3},
            {"logits": single.copy(), "generated_tokens": [1], "vocab": 3},
        ]
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            runner, "load_ck_greedy_trajectory_isolated", side_effect=results
        ) as run:
            result = runner.run_ck_capture_with_neutrality(
                model_dir=Path("/tmp/model"), prompt_tokens=[7], max_new_tokens=1,
                top_k=3, threads=24, runtime_so=None, dump_step=0,
                dump_dir=Path(td) / "capture", dump_layer=63,
                dump_names="layer_out", dump_format="hidden",
                dump_kv_layer=None, stop_token_ids=set(),
                diagnose_single_thread=True,
            )

        diagnostic = result["capture"]["neutrality"]["single_thread_simd_reference"]
        self.assertEqual(run.call_count, 4)
        self.assertTrue(diagnostic["attempted"])
        self.assertEqual(diagnostic["threads"], 1)
        self.assertEqual(diagnostic["simd"], "enabled_by_runtime_build")
        self.assertTrue(diagnostic["repeatability"]["exact"])
        self.assertFalse(diagnostic["parallel_vs_reference"]["exact"])
        self.assertEqual(result["capture"]["artifacts"], [])

    def test_non_neutral_aggregate_falls_back_to_isolated_boundaries(self) -> None:
        exact = np.asarray([[0.0, 4.0, 1.0], [0.0, 1.0, 4.0]], dtype=np.float32)
        perturbed = exact.copy()
        perturbed[1, 1] += np.float32(0.25)
        calls = []

        def run(**kwargs):
            calls.append(kwargs)
            names = str(kwargs.get("dump_names") or "")
            logits = perturbed if names == "attn_out,layer_out" else exact
            result = {"logits": logits, "generated_tokens": [1, 2], "vocab": 3}
            if kwargs.get("dump_step") is not None:
                result["capture"] = {
                    "artifacts": [{"path": f"/tmp/{names}.f32"}],
                }
            return result

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            runner, "load_ck_greedy_trajectory_isolated", side_effect=run
        ):
            result = runner.run_ck_capture_with_neutrality(
                model_dir=Path("/tmp/model"), prompt_tokens=[7], max_new_tokens=2,
                top_k=3, threads=1, runtime_so=None, dump_step=1,
                dump_dir=Path(td) / "capture", dump_layer=63,
                dump_names="attn_out,layer_out", dump_format="hidden",
                dump_kv_layer=None, stop_token_ids=set(),
            )

        neutrality = result["capture"]["neutrality"]
        self.assertEqual(len(calls), 5)
        self.assertEqual(neutrality["status"], "accepted")
        self.assertEqual(neutrality["accepted_mode"], "isolated_boundaries")
        self.assertFalse(neutrality["aggregate_capture"]["exact"])
        self.assertEqual(
            [row["status"] for row in neutrality["fallback"]["boundaries"]],
            ["accepted", "accepted"],
        )
        self.assertEqual(len(result["capture"]["artifacts"]), 2)
        self.assertEqual(len(result["capture"]["rejected_artifacts"]), 1)

    def test_llama_layer_profiler_is_a_persistent_public_callback_hook(self) -> None:
        source = (
            ROOT / "version" / "v8" / "scripts" / "llama_token_replay_v8.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("--profile-layers-out", source)
        self.assertIn('static constexpr const char prefix[] = "l_out-";', source)
        self.assertIn("cparams.cb_eval = dump_eval_callback", source)
        self.assertIn("write_layer_profile(dump_state)", source)

    def test_trajectory_passes_llama_layer_profile_to_oracle(self) -> None:
        rows = np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32)
        profile = Path("/tmp/llama-layers.csv")
        with mock.patch.object(runner, "run_llama_greedy_trajectory", return_value={
            "logits": rows,
            "generated_tokens": [1],
            "meta": {},
            "layer_profile": {"path": str(profile)},
        }) as llama_run, mock.patch.object(
            runner,
            "load_ck_greedy_trajectory_isolated",
            return_value={"logits": rows, "generated_tokens": [1], "vocab": 3},
        ):
            report = runner.run_multitoken_trajectory_parity(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=1,
                ctx_len=128,
                top_k=3,
                threads=1,
                llama_no_repack=False,
                llama_profile_layers_out=profile,
            )
        self.assertEqual(
            llama_run.call_args.kwargs["profile_layers_out"],
            profile,
        )
        self.assertEqual(
            report["llama_layer_profile"],
            {"path": str(profile)},
        )

    def test_llama_helper_rejects_partially_matched_named_capture(self) -> None:
        source = (
            ROOT / "version" / "v8" / "scripts" / "llama_token_replay_v8.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("matched_names.insert(base_name)", source)
        self.assertIn(
            "requested dump tensor names were not observed at greedy step",
            source,
        )
        self.assertIn("return 22;", source)

    def test_thread_configuration_is_applied_before_runtime_load(self) -> None:
        with mock.patch.dict(runner.os.environ, {}, clear=True):
            configured = runner._configure_ck_threads(12)
            self.assertEqual(
                configured,
                {
                    "CK_NUM_THREADS": "12",
                    "CK_THREADPOOL_THREADS": "12",
                    "OMP_NUM_THREADS": "12",
                },
            )
            self.assertEqual(runner.os.environ["CK_NUM_THREADS"], "12")
            self.assertEqual(runner.os.environ["CK_THREADPOOL_THREADS"], "12")
            self.assertEqual(runner.os.environ["OMP_NUM_THREADS"], "12")

    def test_exact_trajectory_stops_at_shared_eos(self) -> None:
        rows = np.asarray([
            [0.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
        ], dtype=np.float32)
        llama = {"logits": rows, "generated_tokens": [1, 2], "meta": {}}
        ck = {"logits": rows, "generated_tokens": [1, 2], "vocab": 3}
        with mock.patch.object(runner, "run_llama_greedy_trajectory", return_value=llama), \
             mock.patch.object(runner, "load_ck_greedy_trajectory_isolated", return_value=ck):
            report = runner.run_multitoken_trajectory_parity(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=64,
                ctx_len=128,
                top_k=3,
                threads=1,
                llama_no_repack=False,
                stop_token_ids={2},
            )
        self.assertTrue(report["pass"])
        self.assertEqual(report["matched_stop_token"], 2)
        self.assertEqual(report["final_prefix"], [7, 1])
        self.assertEqual(report["execution_mode"], "persistent_greedy_trajectory")

    def test_trajectory_reports_first_top1_divergence(self) -> None:
        ck_rows = np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32)
        llama_rows = np.asarray([[0.0, 1.0, 4.0]], dtype=np.float32)
        with mock.patch.object(runner, "run_llama_greedy_trajectory", return_value={
            "logits": llama_rows, "generated_tokens": [2], "meta": {},
        }), mock.patch.object(runner, "load_ck_greedy_trajectory_isolated", return_value={
            "logits": ck_rows, "generated_tokens": [1], "vocab": 3,
        }):
            report = runner.run_multitoken_trajectory_parity(
                model_dir=Path("/tmp/model"), gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7], max_new_tokens=64, ctx_len=128, top_k=3,
                threads=1, llama_no_repack=False, stop_token_ids={2},
            )
        self.assertFalse(report["pass"])
        self.assertEqual(report["first_divergence"]["step"], 0)
        self.assertEqual(report["first_divergence"]["ck_next"], 1)
        self.assertEqual(report["first_divergence"]["llama_next"], 2)
        self.assertEqual(report["final_prefix"], [7])

    def test_trajectory_can_continue_on_llama_teacher_forced_prefix(self) -> None:
        llama_rows = np.asarray([
            [0.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
            [0.0, 1.0, 4.0],
        ], dtype=np.float32)
        ck_rows = np.asarray([
            [0.0, 4.0, 1.0],
            [4.0, 1.0, 0.0],
            [0.0, 1.0, 4.0],
        ], dtype=np.float32)
        captured = {}

        def ck_teacher(**kwargs):
            captured.update(kwargs)
            return {"logits": ck_rows, "generated_tokens": [1, 0, 2], "vocab": 3}

        with mock.patch.object(runner, "run_llama_greedy_trajectory", return_value={
            "logits": llama_rows, "generated_tokens": [1, 2, 2], "meta": {},
        }), mock.patch.object(
            runner, "load_ck_greedy_trajectory_isolated", side_effect=ck_teacher
        ):
            report = runner.run_multitoken_trajectory_parity(
                model_dir=Path("/tmp/model"), gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7], max_new_tokens=3, ctx_len=128, top_k=3,
                threads=1, llama_no_repack=False,
                append_on_divergence="llama",
            )

        self.assertEqual(captured["forced_tokens"], [1, 2, 2])
        self.assertEqual(len(report["steps"]), 3)
        self.assertEqual(report["first_divergence"]["step"], 1)
        self.assertEqual(report["trajectory_policy"], "llama_teacher_forced")

    def test_trajectory_captures_isolated_ck_before_llama(self) -> None:
        rows = np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32)
        calls = []

        def ck_capture(**_kwargs):
            calls.append("ck")
            return {"logits": rows, "generated_tokens": [1], "vocab": 3}

        def llama_capture(*_args, **_kwargs):
            calls.append("llama")
            return {"logits": rows, "generated_tokens": [1], "meta": {}}

        with mock.patch.object(
            runner, "load_ck_greedy_trajectory_isolated", side_effect=ck_capture
        ), mock.patch.object(
            runner, "run_llama_greedy_trajectory", side_effect=llama_capture
        ):
            runner.run_multitoken_trajectory_parity(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=1,
                ctx_len=128,
                top_k=3,
                threads=1,
                llama_no_repack=False,
            )
        self.assertEqual(calls, ["ck", "llama"])

    def test_requested_threads_reach_isolated_ck_capture(self) -> None:
        rows = np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32)
        captured: dict = {}

        def ck_capture(**kwargs):
            captured.update(kwargs)
            return {
                "logits": rows,
                "generated_tokens": [1],
                "vocab": 3,
                "thread_environment": {
                    "CK_NUM_THREADS": "7",
                    "CK_THREADPOOL_THREADS": "7",
                    "OMP_NUM_THREADS": "7",
                },
            }

        with mock.patch.object(
            runner, "load_ck_greedy_trajectory_isolated", side_effect=ck_capture
        ), mock.patch.object(
            runner,
            "run_llama_greedy_trajectory",
            return_value={"logits": rows, "generated_tokens": [1], "meta": {}},
        ):
            report = runner.run_multitoken_trajectory_parity(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=1,
                ctx_len=128,
                top_k=3,
                threads=7,
                llama_no_repack=False,
            )

        self.assertEqual(captured["threads"], 7)
        self.assertEqual(report["threads"], 7)
        self.assertEqual(report["ck_thread_environment"]["CK_NUM_THREADS"], "7")

    def test_capture_contract_reaches_isolated_persistent_runtime(self) -> None:
        rows = np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32)
        captured: dict = {}
        runtime = Path("/tmp/libmodel-parity.so")
        dump_dir = Path("/tmp/glm-step-39")

        def ck_capture(**kwargs):
            captured.update(kwargs)
            return {
                "logits": rows,
                "generated_tokens": [1],
                "vocab": 3,
                "runtime": {"path": str(runtime), "sha256": "a" * 64},
                "capture": {
                    "execution_mode": "persistent_greedy_trajectory",
                    "step": 39,
                    "layer": 0,
                    "op_filter": "q_proj,k_proj",
                    "format": "hidden",
                    "artifacts": [{
                        "path": str(dump_dir / "tok_0059_layer_000_q_proj.f32"),
                        "sha256": "b" * 64,
                        "size": 4096,
                    }],
                },
            }

        with mock.patch.object(
            runner, "load_ck_greedy_trajectory_isolated", side_effect=ck_capture
        ), mock.patch.object(
            runner,
            "run_llama_greedy_trajectory",
            return_value={"logits": rows, "generated_tokens": [1], "meta": {}},
        ):
            report = runner.run_multitoken_trajectory_parity(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=64,
                ctx_len=128,
                top_k=3,
                threads=7,
                llama_no_repack=False,
                ck_runtime_so=runtime,
                ck_dump_step=39,
                ck_dump_dir=dump_dir,
                ck_dump_layer=0,
                ck_dump_names="q_proj,k_proj",
                ck_dump_kv_layer=0,
            )

        self.assertEqual(captured["runtime_so"], runtime)
        self.assertEqual(captured["dump_step"], 39)
        self.assertEqual(captured["dump_dir"], dump_dir)
        self.assertEqual(captured["dump_layer"], 0)
        self.assertEqual(captured["dump_names"], "q_proj,k_proj")
        self.assertEqual(captured["dump_format"], "hidden")
        self.assertEqual(captured["dump_kv_layer"], 0)
        self.assertEqual(
            report["ck_capture"]["execution_mode"],
            "persistent_greedy_trajectory",
        )
        self.assertEqual(report["ck_runtime"]["sha256"], "a" * 64)

    def test_capture_step_requires_dump_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "dump_dir is required"):
            runner.load_ck_greedy_trajectory(
                model_dir=Path("/tmp/model"),
                prompt_tokens=[7],
                max_new_tokens=64,
                dump_step=39,
            )

    def test_prefill_capture_is_enabled_before_embedding(self) -> None:
        source = (
            ROOT / "version" / "v8" / "scripts" / "compare_multitoken_logits_v8.py"
        ).read_text(encoding="utf-8")
        capture_enable = source.index(
            "if capture_step == 0:\n"
            "            # Batched prefill executes inside embed_tokens."
        )
        embed_call = source.index(
            "if lib.ck_model_embed_tokens(token_array, len(prompt)) != 0:"
        )
        forward_call = source.index("if lib.ck_model_forward(None) != 0:")
        self.assertLess(capture_enable, embed_call)
        self.assertLess(embed_call, forward_call)

    def test_llama_capture_contract_reaches_same_trajectory_step(self) -> None:
        rows = np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32)
        captured: dict = {}

        def llama_capture(*_args, **kwargs):
            captured.update(kwargs)
            return {
                "logits": rows,
                "generated_tokens": [1],
                "meta": {"flash_attention_mode": "auto"},
                "capture": {
                    "step": 0,
                    "attention_mode": "auto",
                    "artifacts": [{"sha256": "c" * 64}],
                },
            }

        with mock.patch.object(
            runner,
            "load_ck_greedy_trajectory_isolated",
            return_value={"logits": rows, "generated_tokens": [1], "vocab": 3},
        ), mock.patch.object(
            runner, "run_llama_greedy_trajectory", side_effect=llama_capture
        ):
            report = runner.run_multitoken_trajectory_parity(
                model_dir=Path("/tmp/model"),
                gguf_path=Path("/tmp/model.gguf"),
                prompt_tokens=[7],
                max_new_tokens=1,
                ctx_len=128,
                top_k=3,
                threads=7,
                llama_no_repack=False,
                llama_dump_step=0,
                llama_dump_dir=Path("/tmp/llama-step-0"),
                llama_dump_names="__fattn__-0",
                llama_dump_flash_inputs=True,
            )

        self.assertEqual(captured["dump_step"], 0)
        self.assertEqual(captured["dump_dir"], Path("/tmp/llama-step-0"))
        self.assertEqual(captured["dump_names"], "__fattn__-0")
        self.assertTrue(captured["dump_flash_inputs"])
        self.assertEqual(report["llama_capture"]["attention_mode"], "auto")


if __name__ == "__main__":
    unittest.main()
