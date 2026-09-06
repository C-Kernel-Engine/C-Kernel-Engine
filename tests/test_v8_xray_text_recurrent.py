from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "version" / "v8" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("xray_text_recurrent_v8", SCRIPT_DIR / "xray_text_recurrent_v8.py")
XRAY = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(XRAY)


class TextRecurrentXRayTests(unittest.TestCase):
    @staticmethod
    def _write_runtime_bundle(root: Path, *, linked_engine_hash: str | None = None) -> None:
        outputs = {}
        for name, payload in (
            ("libmodel.so", b"model-runtime"),
            ("libckernel_engine.so", b"engine-runtime"),
            ("libckernel_tokenizer.so", b"tokenizer-runtime"),
        ):
            path = root / name
            path.write_bytes(payload)
            outputs[name] = {
                "path": str(path),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        stamp = {
            "inputs": {
                "schema": XRAY.RUNTIME_BUNDLE_SCHEMA,
                "linked_libraries": {
                    "engine": {
                        "sha256": linked_engine_hash
                        or outputs["libckernel_engine.so"]["sha256"]
                    },
                    "tokenizer": {
                        "sha256": outputs["libckernel_tokenizer.so"]["sha256"]
                    },
                },
            },
            "outputs": outputs,
        }
        (root / XRAY.RUNTIME_STAMP_NAME).write_text(
            json.dumps(stamp), encoding="utf-8"
        )

    def test_runtime_provenance_accepts_one_stamped_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_runtime_bundle(root)

            result = XRAY.validate_runtime_provenance(root)

            self.assertEqual(result["status"], "verified")
            self.assertEqual(
                result["outputs"]["libmodel.so"]["sha256"],
                hashlib.sha256(b"model-runtime").hexdigest(),
            )

    def test_runtime_provenance_rejects_replaced_model_binary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_runtime_bundle(root)
            (root / "libmodel.so").write_bytes(b"manually-recompiled-model")

            with self.assertRaisesRegex(RuntimeError, "does not match its stamp"):
                XRAY.validate_runtime_provenance(root)

    def test_runtime_provenance_rejects_unpaired_engine(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_runtime_bundle(root, linked_engine_hash="0" * 64)

            with self.assertRaisesRegex(RuntimeError, "was linked against engine"):
                XRAY.validate_runtime_provenance(root)

    def test_boundary_selection_follows_layer_kind(self) -> None:
        config = {"layer_kinds": ["recurrent", "full_attention"]}
        recurrent = XRAY.boundaries_for_layer(config, 0)
        attention = XRAY.boundaries_for_layer(config, 1)

        self.assertIn("new_state", recurrent)
        self.assertNotIn("q_proj", recurrent)
        self.assertIn("q_proj", attention)
        self.assertNotIn("new_state", attention)
        self.assertIn("layer_out", recurrent)
        self.assertIn("layer_out", attention)

    def test_hyper_moe_recurrent_profile_uses_declared_topology(self) -> None:
        boundaries = XRAY.boundaries_for_layer(
            {
                "arch": "future_hyper_moe",
                "hc_count": 4,
                "num_experts": 512,
                "layer_kinds": ["recurrent"],
            },
            0,
        )
        self.assertEqual(
            boundaries[:3],
            (
                "attn_hyper_norm",
                "attn_hyper_gate",
                "attn_mixed_input",
            ),
        )
        self.assertNotIn("attn_norm", boundaries)
        self.assertEqual(
            XRAY.ORACLE_BOUNDARY_NAMES["attn_hyper_norm"],
            "hc_norm",
        )
        self.assertEqual(
            XRAY.ORACLE_BOUNDARY_NAMES["attn_hyper_gate"],
            "hc_gate",
        )
        self.assertEqual(
            XRAY.ORACLE_BOUNDARY_NAMES["attn_mixed_input"],
            "hc_mixed",
        )
        self.assertIn("after_attn_hyper", boundaries)
        self.assertIn("mlp_mixed_input", boundaries)
        self.assertIn("moe_router_logits", boundaries)
        self.assertIn("moe_routing_weights", boundaries)
        self.assertIn("moe_routed_output", boundaries)
        self.assertIn("moe_combined_output", boundaries)
        self.assertIn("layer_out_hyper", boundaries)
        self.assertEqual(
            XRAY.ORACLE_BOUNDARY_OCCURRENCES["mlp_mixed_input"], 1
        )
        self.assertEqual(
            XRAY.ORACLE_BOUNDARY_NAMES["layer_out_hyper"], "l_last"
        )
        self.assertNotIn("layer_out_hyper", XRAY.ORACLE_BOUNDARY_OCCURRENCES)
        self.assertEqual(
            XRAY.ORACLE_BOUNDARY_NAMES["moe_router_logits"],
            "ffn_moe_logits",
        )
        self.assertEqual(
            XRAY.ORACLE_BOUNDARY_NAMES["moe_routing_weights"],
            "ffn_moe_weights_norm",
        )
        self.assertEqual(
            XRAY.ck_capture_names(("layer_out_hyper",)), ("layer_out",)
        )

    def test_moe_profiles_do_not_require_hyper_connections(self) -> None:
        recurrent = XRAY.boundaries_for_layer(
            {
                "num_experts": 256,
                "layer_kinds": ["recurrent", "full_attention"],
            },
            0,
        )
        attention = XRAY.boundaries_for_layer(
            {
                "num_experts": 256,
                "layer_kinds": ["recurrent", "full_attention"],
            },
            1,
        )

        for boundaries in (recurrent, attention):
            self.assertIn("moe_router_logits", boundaries)
            self.assertIn("moe_routed_output", boundaries)
            self.assertIn("moe_combined_output", boundaries)
            self.assertNotIn("mlp_gate", boundaries)
            self.assertNotIn("attn_hyper_norm", boundaries)
        self.assertIn("new_state", recurrent)
        self.assertIn("q_proj", attention)

    def test_ple_hyper_moe_profile_observes_ple_before_hyper_mix(self) -> None:
        boundaries = XRAY.boundaries_for_layer(
            {
                "hc_count": 4,
                "num_experts": 512,
                "layer_kinds": ["recurrent", "recurrent"],
                "ple_owner_layers": [1],
            },
            1,
        )
        self.assertEqual(
            boundaries[:8],
            (
                "ple_key_projected",
                "ple_value_projected",
                "ple_key_normed",
                "ple_query_normed",
                "ple_gated_value",
                "ple_conv_normed",
                "ple_conv_out",
                "ple_layer_out",
            ),
        )
        self.assertEqual(boundaries[8], "attn_hyper_norm")

    def test_ple_profile_is_not_applied_to_other_recurrent_layers(self) -> None:
        boundaries = XRAY.boundaries_for_layer(
            {
                "hc_count": 4,
                "num_experts": 512,
                "layer_kinds": ["recurrent", "recurrent"],
                "ple_owner_layers": [1],
            },
            0,
        )
        self.assertNotIn("ple_gated_value", boundaries)

    def test_sparse_attention_profile_observes_hyper_attention_and_moe(self) -> None:
        boundaries = XRAY.boundaries_for_layer(
            {
                "hc_count": 4,
                "num_experts": 512,
                "layer_kinds": ["sparse_attention"],
            },
            0,
        )
        self.assertEqual(
            boundaries[:3],
            (
                "attn_hyper_norm",
                "attn_hyper_gate",
                "attn_mixed_input",
            ),
        )
        self.assertIn("q_proj", boundaries)
        self.assertIn("rope_q", boundaries)
        self.assertIn("attn_pregate", boundaries)
        self.assertIn("moe_routing_weights", boundaries)
        self.assertIn("layer_out_hyper", boundaries)
        self.assertNotIn("new_state", boundaries)

    def test_sparse_attention_without_hyper_moe_uses_attention_profile(self) -> None:
        boundaries = XRAY.boundaries_for_layer(
            {"layer_kinds": ["sparse_attention"]},
            0,
        )
        self.assertEqual(boundaries, XRAY.FULL_ATTENTION_BOUNDARIES)

    def test_boundary_selection_rejects_unknown_or_missing_layer_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer_kinds"):
            XRAY.boundaries_for_layer({}, 0)
        with self.assertRaisesRegex(ValueError, "outside"):
            XRAY.boundaries_for_layer({"layer_kinds": ["recurrent"]}, 1)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            XRAY.boundaries_for_layer({"layer_kinds": ["mystery"]}, 0)

    def test_capture_plan_includes_physical_fused_gate_up_dependency(self) -> None:
        names = XRAY.ck_capture_names(("attn_norm", "mlp_gate", "mlp_up"))
        self.assertEqual(
            names,
            ("attn_norm", "mlp_gate", "mlp_up", "mlp_gate_up"),
        )

    def test_ck_capture_runs_in_short_lived_worker(self) -> None:
        process = mock.Mock()
        process.exitcode = 0
        context = mock.Mock()
        context.Process.return_value = process
        with mock.patch.object(
            XRAY.multiprocessing, "get_context", return_value=context
        ) as get_context:
            XRAY._run_isolated_ck_capture(
                Path("/runtime"),
                [1, 2],
                [3],
                "hybrid",
                {"CK_DEBUG_EXPORT_HIDDEN": "/captures"},
            )
        get_context.assert_called_once_with("fork")
        process.start.assert_called_once_with()
        process.join.assert_called_once_with()

    def test_ck_capture_worker_failure_is_reported(self) -> None:
        process = mock.Mock()
        process.exitcode = -9
        context = mock.Mock()
        context.Process.return_value = process
        with mock.patch.object(XRAY.multiprocessing, "get_context", return_value=context):
            with self.assertRaisesRegex(RuntimeError, "exit code -9"):
                XRAY._run_isolated_ck_capture(
                    Path("/runtime"), [1], [], "hybrid", {}
                )

    def test_reuse_ck_capture_requires_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            self._write_runtime_bundle(model)
            (model / "config.json").write_text(
                '{"ssm_state_size": 4, "num_heads": 2, "num_kv_heads": 1, '
                '"layer_kinds": ["recurrent"]}'
            )
            parity = root / "parity.json"
            parity.write_text('{"initial_tokens": [1], "final_prefix": [1]}')
            with self.assertRaisesRegex(ValueError, "no requested checkpoint"):
                XRAY.capture_and_analyze(
                    model,
                    root / "model.gguf",
                    parity,
                    root / "captures",
                    0,
                    16,
                    1,
                    "hybrid",
                    True,
                )

    def test_reuse_ck_capture_accepts_profile_specific_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "tok_0000_layer_000_attn_mixed_input.f32"
            checkpoint.write_bytes(b"\x00\x00\x00\x00")
            self.assertEqual(
                XRAY._validate_reused_ck_capture(
                    root,
                    ("attn_hyper_norm", "attn_mixed_input"),
                    layer=0,
                    logical_token=0,
                ),
                checkpoint,
            )

    def test_fresh_ck_capture_rejects_stale_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "tok_0000_layer_003_attn_mixed_input.f32"
            checkpoint.write_bytes(b"\x00\x00\x00\x00")
            with self.assertRaisesRegex(ValueError, "checkpoint already exists"):
                XRAY._reject_existing_ck_capture(
                    root,
                    ("attn_hyper_norm", "attn_mixed_input"),
                    layer=3,
                )

    def test_fresh_ck_capture_accepts_unrelated_layer_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tok_0000_layer_002_attn_mixed_input.f32").write_bytes(
                b"\x00\x00\x00\x00"
            )
            XRAY._reject_existing_ck_capture(
                root,
                ("attn_hyper_norm", "attn_mixed_input"),
                layer=3,
            )

    def test_reuse_oracle_capture_requires_requested_layer_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(ValueError, "no requested checkpoint"):
                XRAY._validate_reused_oracle_capture(
                    root,
                    ("linear_attn_qkv_mixed",),
                    layer=3,
                    physical_token=7,
                )

            checkpoint = (
                root / "linear_attn_qkv_mixed-3-token-000007-occ-000.bin"
            )
            checkpoint.write_bytes(b"\x00\x00\x00\x00")
            self.assertEqual(
                XRAY._validate_reused_oracle_capture(
                    root,
                    ("linear_attn_qkv_mixed",),
                    layer=3,
                    physical_token=7,
                ),
                checkpoint,
            )

    def test_reuse_oracle_capture_skips_llama_execution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            self._write_runtime_bundle(model)
            (model / "config.json").write_text(
                '{"ssm_state_size": 4, "num_heads": 2, "num_kv_heads": 1, '
                '"layer_kinds": ["recurrent"]}'
            )
            parity = root / "parity.json"
            parity.write_text('{"initial_tokens": [1], "final_prefix": [1]}')
            ck_root = root / "captures" / "ck"
            ck_root.mkdir(parents=True)
            (ck_root / "tok_0000_layer_000_attn_norm.f32").write_bytes(
                b"\x00\x00\x00\x00"
            )
            oracle_root = root / "captures" / "llama"
            oracle_root.mkdir()
            (
                oracle_root
                / "linear_attn_qkv_mixed-0-token-000000-occ-000.bin"
            ).write_bytes(b"\x00\x00\x00\x00")

            with mock.patch.object(
                XRAY, "_run_llama_capture"
            ) as run_llama, mock.patch.object(
                XRAY,
                "analyze_capture",
                return_value={"first_divergence": None},
            ):
                report = XRAY.capture_and_analyze(
                    model,
                    root / "model.gguf",
                    parity,
                    root / "captures",
                    0,
                    16,
                    1,
                    "hybrid",
                    True,
                    True,
                )

            run_llama.assert_not_called()
            self.assertTrue(report["llama_capture"]["reused"])

    def test_named_axis_state_transform_prevents_flat_layout_false_positive(self) -> None:
        ck = np.arange(2 * 128 * 128, dtype=np.float32).reshape(2, 128, 128)
        oracle = ck.transpose(0, 2, 1).copy()
        result = XRAY.compare_arrays("new_state", ck.reshape(-1), oracle.reshape(-1))
        self.assertEqual(result["status"], "exact")
        self.assertIn("head,value,key", result["axis_transform"])

    def test_llama_physical_state_layout_is_compared_without_transpose(self) -> None:
        values = np.arange(2 * 4 * 4, dtype=np.float32)
        result = XRAY.compare_arrays(
            "new_state",
            values,
            values.copy(),
            state_size=4,
            recurrent_state_physical_layout="head_value_key_contiguous",
        )
        self.assertEqual(result["status"], "exact")
        self.assertEqual(result["axis_transform"], "identity:[head,value,key]")

    def test_unknown_state_layout_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "unsupported recurrent_state_physical_layout"
        ):
            XRAY.compare_arrays(
                "new_state",
                np.zeros(16, dtype=np.float32),
                np.zeros(16, dtype=np.float32),
                state_size=4,
                recurrent_state_physical_layout="ambiguous",
            )

    def test_selected_provider_owns_recurrent_state_layout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            maps = root / "maps"
            model.mkdir()
            maps.mkdir()
            (model / "lowered_decode_call.json").write_text(
                json.dumps({
                    "operations": [{
                        "layer": 0,
                        "op": "recurrent_core",
                        "function": "grouped_forward",
                        "call_abi": {
                            "owner": "kernel_map",
                            "kernel_id": "grouped",
                            "source_file": "grouped.json",
                        },
                    }],
                }),
                encoding="utf-8",
            )
            (maps / "grouped.json").write_text(
                json.dumps({
                    "id": "grouped",
                    "op": "gated_deltanet",
                    "inputs": [{
                        "name": "state_in",
                        "layout": "head_value_key_contiguous",
                    }],
                    "outputs": [{
                        "name": "state_out",
                        "layout": "head_value_key_contiguous",
                    }],
                    "impl": {"function": "grouped_forward"},
                }),
                encoding="utf-8",
            )

            layout, provenance = XRAY.recurrent_state_layout_from_selected_provider(
                model,
                {"recurrent_state_physical_layout": "head_key_value_contiguous"},
                0,
                kernel_maps_dir=maps,
            )

            self.assertEqual(layout, "head_value_key_contiguous")
            self.assertEqual(provenance["kernel_id"], "grouped")

    def test_selected_provider_layout_validation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            maps = root / "maps"
            model.mkdir()
            maps.mkdir()
            (model / "lowered_decode_call.json").write_text(
                json.dumps({
                    "operations": [{
                        "layer": 0,
                        "op": "recurrent_core",
                        "function": "wrong_forward",
                        "call_abi": {
                            "owner": "kernel_map",
                            "kernel_id": "wrong",
                            "source_file": "wrong.json",
                        },
                    }],
                }),
                encoding="utf-8",
            )
            (maps / "wrong.json").write_text(
                json.dumps({
                    "id": "wrong",
                    "op": "memcpy",
                    "inputs": [{
                        "name": "state_in",
                        "layout": "head_value_key_contiguous",
                    }],
                    "outputs": [{
                        "name": "state_out",
                        "layout": "head_value_key_contiguous",
                    }],
                    "impl": {"function": "wrong_forward"},
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "incompatible operation class"):
                XRAY.recurrent_state_layout_from_selected_provider(
                    model,
                    {},
                    0,
                    kernel_maps_dir=maps,
                )

    def test_exact_input_then_projection_difference_is_provider_mismatch(self) -> None:
        schedules = {"ck": "sequential_decode", "oracle_prefix": "batched", "oracle_decode": "sequential"}
        rows = [
            {"logical_token": 3, "layer": 0, "boundary": "attn_norm", "status": "exact"},
            {"logical_token": 3, "layer": 0, "boundary": "linear_attn_qkv_mixed", "status": "different", "max_abs_diff": 0.0047},
        ]
        result = XRAY.classify(rows, schedules)
        self.assertEqual(result["classification"], "PROJECTION_PROVIDER_MISMATCH")
        self.assertEqual(result["previous_exact_boundary"], "attn_norm")

    def test_exact_hyper_mixed_input_then_projection_difference_is_provider_mismatch(self) -> None:
        schedules = {"ck": "sequential_decode", "oracle_prefix": "batched", "oracle_decode": "sequential"}
        rows = [
            {"logical_token": 0, "layer": 0, "boundary": "attn_mixed_input", "status": "exact"},
            {"logical_token": 0, "layer": 0, "boundary": "linear_attn_qkv_mixed", "status": "different", "max_abs_diff": 0.0047},
        ]
        result = XRAY.classify(rows, schedules)
        self.assertEqual(result["classification"], "PROJECTION_PROVIDER_MISMATCH")
        self.assertEqual(result["previous_exact_boundary"], "attn_mixed_input")

    def test_missing_checkpoint_does_not_hide_previous_exact_boundary(self) -> None:
        schedules = {"ck": "sequential_decode", "oracle_prefix": "batched", "oracle_decode": "sequential"}
        rows = [
            {"logical_token": 3, "layer": 7, "boundary": "attn_pregate", "status": "exact"},
            {"logical_token": 3, "layer": 7, "boundary": "attn_out", "status": "missing_or_incompatible"},
            {"logical_token": 3, "layer": 7, "boundary": "out_proj", "status": "different", "max_abs_diff": 0.0047},
        ]
        result = XRAY.classify(rows, schedules)
        self.assertEqual(result["boundary"], "out_proj")
        self.assertEqual(result["previous_exact_boundary"], "attn_pregate")

    def test_previous_boundary_is_scoped_to_same_token_and_layer(self) -> None:
        schedules = {"ck": "sequential_decode", "oracle_prefix": "batched", "oracle_decode": "sequential"}
        rows = [
            {"logical_token": 2, "layer": 3, "boundary": "layer_out", "status": "exact"},
            {"logical_token": 3, "layer": 3, "boundary": "attn_norm", "status": "different", "max_abs_diff": 0.0047},
        ]
        result = XRAY.classify(rows, schedules)
        self.assertIsNone(result["previous_exact_boundary"])

    def test_small_normalization_difference_then_gate_drift_is_amplification(self) -> None:
        schedules = {"ck": "batched_then_sequential", "oracle_prefix": "batched", "oracle_decode": "sequential"}
        rows = [
            {"logical_token": 16, "layer": 0, "boundary": "attn_norm", "status": "different", "max_abs_diff": 9.54e-7},
            {"logical_token": 16, "layer": 0, "boundary": "linear_attn_qkv_mixed", "status": "different", "max_abs_diff": 3.81e-6},
            {"logical_token": 16, "layer": 0, "boundary": "alpha", "status": "different", "max_abs_diff": 1.60e-3},
        ]
        result = XRAY.classify(rows, schedules)
        self.assertEqual(result["classification"], "NORMALIZATION_TO_QUANTIZATION_AMPLIFICATION")
        self.assertEqual(result["amplification_source"], "attn_norm")

    def test_batched_prompt_rows_are_compared_by_logical_token(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = np.arange(12, dtype=np.float32).reshape(3, 4)
            values.tofile(root / "attn_norm-0-token-000002-occ-000.bin")
            row = XRAY._load_oracle_row(root, "attn_norm", 0, 1, 3, 4)
            np.testing.assert_array_equal(row, values[1])

    def test_selected_oracle_row_requires_shape_and_final_position(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "ffn_moe_logits-47-token-000039-occ-000.bin"
            values = np.arange(512, dtype=np.float32)
            values.tofile(path)
            path.with_suffix(".json").write_text(json.dumps({"type": 0, "ne": [512, 1, 1, 1]}))
            self.assertEqual(XRAY._infer_oracle_row_count(root, "moe_router_logits", 47, 39, 40), 512)
            np.testing.assert_array_equal(
                XRAY._load_oracle_row(root, "moe_router_logits", 47, 39, 40, 512), values)
            with self.assertRaisesRegex(ValueError, "only for the final"):
                XRAY._infer_oracle_row_count(root, "moe_router_logits", 47, 0, 40)
            path.with_suffix(".json").write_text(json.dumps({"type": 0, "ne": [256, 1, 1, 1]}))
            with self.assertRaisesRegex(ValueError, "shape/storage mismatch"):
                XRAY._infer_oracle_row_count(root, "moe_router_logits", 47, 39, 40)

    def test_selected_hyper_row_not_divided_by_prompt_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "l_last-47-token-000039-occ-000.bin"
            np.zeros(10240, dtype=np.float32).tofile(path)
            path.with_suffix(".json").write_text(json.dumps({"type": 0, "ne": [2560, 4, 1, 1]}))
            self.assertEqual(XRAY._infer_oracle_row_count(root, "layer_out_hyper", 47, 39, 40), 10240)

    def test_normalized_hyper_stream_has_flattened_feature_axis(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "hc_norm-0-token-000039-occ-000.bin"
            values = np.arange(10240 * 40, dtype=np.float32).reshape(40, 10240)
            values.tofile(path)
            path.with_suffix(".json").write_text(json.dumps({"type": 0, "ne": [10240, 40, 1, 1]}))
            self.assertEqual(XRAY._infer_oracle_row_count(root, "attn_hyper_norm", 0, 17, 40), 10240)
            np.testing.assert_array_equal(XRAY._load_oracle_row(root, "attn_hyper_norm", 0, 17, 40, 10240), values[17])

    def test_schedule_metadata_is_mandatory_in_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = XRAY.analyze_capture(root / "ck", root / "llama", 3, 3, 0)
            self.assertEqual(report["schedules"]["ck"], "sequential_decode")
            self.assertEqual(report["schedules"]["oracle_prefix"], "batched")
            self.assertEqual(report["schedules"]["oracle_decode"], "sequential")
            self.assertIn("only prioritizes attribution", report["acceptance_policy"])

    def test_hybrid_ck_prompt_rows_are_loaded_from_full_prefill_capture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = np.arange(12, dtype=np.float32).reshape(3, 4)
            values.tofile(root / "tok_0000_layer_000_z.f32")
            row = XRAY._load_ck_row(root, "z", 0, 2, 3, 4, "hybrid")
            np.testing.assert_array_equal(row, values[2])

    def test_hybrid_ck_prompt_rows_prefer_token_scoped_granular_capture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            expected = np.arange(4, dtype=np.float32) + 20
            expected.tofile(root / "tok_0002_layer_000_z.f32")
            np.arange(12, dtype=np.float32).tofile(
                root / "tok_0000_layer_000_z.f32"
            )

            row = XRAY._load_ck_row(root, "z", 0, 2, 3, 4, "hybrid")

            np.testing.assert_array_equal(row, expected)

    def test_hybrid_ck_token_scoped_extent_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            np.arange(3, dtype=np.float32).tofile(
                root / "tok_0002_layer_000_z.f32"
            )

            with self.assertRaisesRegex(ValueError, "token-scoped batched CK extent"):
                XRAY._load_ck_row(root, "z", 0, 2, 3, 4, "hybrid")

    def test_hybrid_ck_gate_and_up_rows_are_reconstructed_from_fused_capture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = np.arange(24, dtype=np.float32).reshape(3, 8)
            values.tofile(root / "tok_0000_layer_000_mlp_gate_up.f32")

            gate = XRAY._load_ck_row(root, "mlp_gate", 0, 1, 3, 4, "hybrid")
            up = XRAY._load_ck_row(root, "mlp_up", 0, 1, 3, 4, "hybrid")

            np.testing.assert_array_equal(gate, values[1, :4])
            np.testing.assert_array_equal(up, values[1, 4:])

    def test_hybrid_ck_gate_and_up_use_token_scoped_fused_capture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = np.arange(8, dtype=np.float32)
            values.tofile(root / "tok_0001_layer_000_mlp_gate_up.f32")

            gate = XRAY._load_ck_row(root, "mlp_gate", 0, 1, 3, 4, "hybrid")
            up = XRAY._load_ck_row(root, "mlp_up", 0, 1, 3, 4, "hybrid")

            np.testing.assert_array_equal(gate, values[:4])
            np.testing.assert_array_equal(up, values[4:])

    def test_hybrid_ck_fused_gate_up_extent_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            np.arange(23, dtype=np.float32).tofile(
                root / "tok_0000_layer_000_mlp_gate_up.f32"
            )
            with self.assertRaisesRegex(ValueError, "fused gate/up extent mismatch"):
                XRAY._load_ck_row(root, "mlp_gate", 0, 1, 3, 4, "hybrid")

    def test_hybrid_ck_attention_rows_are_canonicalized_from_head_major(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
            values.tofile(root / "tok_0000_layer_000_attn_pregate.f32")
            row = XRAY._load_ck_row(
                root, "attn_pregate", 0, 1, 3, 4, "hybrid", attention_heads=2
            )
            np.testing.assert_array_equal(row, values[:, 1, :].reshape(-1))

    def test_hybrid_ck_gated_attention_rows_remain_token_major_after_transpose(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = np.arange(12, dtype=np.float32).reshape(3, 4)
            values.tofile(root / "tok_0000_layer_000_attn_out.f32")
            row = XRAY._load_ck_row(root, "attn_out", 0, 1, 3, 4, "hybrid", attention_heads=2)
            np.testing.assert_array_equal(row, values[1])

    def test_hybrid_schedule_metadata_matches_batched_oracle_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = XRAY.analyze_capture(
                root / "ck", root / "llama", 3, 3, 0, ck_prefill_mode="hybrid"
            )
            self.assertEqual(report["schedules"]["ck_prefix"], "batched")
            self.assertEqual(report["schedules"]["oracle_prefix"], "batched")
            self.assertEqual(report["schedules"]["ck_decode"], "sequential")

    def test_circuit_checkpoint_maps_to_oracle_graph_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = np.arange(12, dtype=np.float32).reshape(3, 4)
            values.tofile(root / "attn_residual-0-token-000002-occ-000.bin")
            row = XRAY._load_oracle_row(root, "after_attn", 0, 1, 3, 4)
            np.testing.assert_array_equal(row, values[1])

    def test_layer_composition_edges_are_part_of_xray_contract(self) -> None:
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["after_attn"], "attn_residual")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["out_proj"], "attn_output")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["q_proj"], "Qcur_full")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["attn_gate"], "gate_reshaped")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["attn_pregate"], "attn_pregate")
        self.assertLess(XRAY.BOUNDARIES.index("attn_pregate"), XRAY.BOUNDARIES.index("attn_out"))
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["attn_out"], "attn_gated")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["post_attn_norm"], "attn_post_norm")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["mlp_swiglu"], "ffn_swiglu")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["mlp_down"], "ffn_out")
        self.assertEqual(XRAY.ORACLE_BOUNDARY_NAMES["layer_out"], "l_out")
        for boundary in XRAY.ORACLE_BOUNDARY_NAMES:
            self.assertIn(boundary, XRAY.BOUNDARIES)

    def test_reused_oracle_graph_label_selects_declared_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = np.arange(4, dtype=np.float32)
            second = first + np.float32(10.0)
            first.tofile(root / "Kcur-3-token-000004-occ-000.bin")
            second.tofile(root / "Kcur-3-token-000004-occ-001.bin")
            row = XRAY._load_oracle_row(root, "rope_k", 3, 4, 4, 4)
            np.testing.assert_array_equal(row, second)

    def test_state_axis_extent_is_not_hardcoded_to_one_model(self) -> None:
        ck = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        oracle = ck.transpose(0, 2, 1).copy()
        result = XRAY.compare_arrays("new_state", ck.reshape(-1), oracle.reshape(-1), state_size=4)
        self.assertEqual(result["status"], "exact")


if __name__ == "__main__":
    unittest.main()
