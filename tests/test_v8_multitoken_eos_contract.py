#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import ctypes
import json
import struct
import tempfile
import unittest
from array import array
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/compare_multimodal_multitoken_logits_v8.py"


def load_module():
    spec = importlib.util.spec_from_file_location("multitoken_eos_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MultitokenEOSContractTests(unittest.TestCase):
    def test_hidden_xray_configures_matching_llama_boundary_capture(self) -> None:
        args = Namespace(
            hidden_state_step=8,
            hidden_state_layer=3,
            hidden_state_names="layer_input,after_attn,layer_out",
            hidden_state_dir=Path("capture"),
            llama_persistent_dump_dir=None,
            workdir=Path("work"),
        )
        inputs = {"runtime": {"manifest": {"config": {"num_layers": 8}}}}

        self.runner._configure_hidden_oracle_capture(inputs, args)

        self.assertEqual(args.llama_persistent_dump_step, 8)
        self.assertEqual(
            args.llama_persistent_dump_names,
            "l_out-2,attn_residual-3,l_out-3",
        )
        self.assertEqual(args.llama_persistent_dump_dir, Path("capture").resolve() / "llama")

        args.llama_persistent_dump_step = 7
        with self.assertRaisesRegex(ValueError, "conflicts.*dump step"):
            self.runner._configure_hidden_oracle_capture(inputs, args)

    def test_hidden_xray_accepts_prefill_trajectory_step_zero(self) -> None:
        args = Namespace(
            hidden_state_step=0,
            hidden_state_layer=1,
            hidden_state_names="new_state",
            hidden_state_dir=Path("capture"),
            llama_persistent_dump_dir=None,
            workdir=Path("work"),
        )
        inputs = {"runtime": {"manifest": {"config": {"num_layers": 8}}}}

        self.runner._configure_hidden_oracle_capture(inputs, args)

        self.assertEqual(args.llama_persistent_dump_step, 0)
        self.assertEqual(args.llama_persistent_dump_names, "new_state-1")

    def test_hidden_xray_accepts_explicit_dense_oracle_boundaries(self) -> None:
        args = Namespace(
            hidden_state_step=0,
            hidden_state_layer=0,
            hidden_state_names="attn_out,after_attn,layer_out",
            hidden_state_oracle_map="attn_out=kqv_out,after_attn=ffn_inp",
            hidden_state_dir=Path("capture"),
            llama_persistent_dump_dir=None,
            workdir=Path("work"),
        )
        inputs = {"runtime": {"manifest": {"config": {"num_layers": 4}}}}

        self.runner._configure_hidden_oracle_capture(inputs, args)

        self.assertEqual(
            args.llama_persistent_dump_names,
            "kqv_out-0,ffn_inp-0,l_out-0",
        )
        self.assertEqual(
            args.hidden_state_oracle_name_map,
            {"attn_out": "kqv_out", "after_attn": "ffn_inp"},
        )

    def test_hidden_xray_rejects_invalid_oracle_boundary_map(self) -> None:
        with self.assertRaisesRegex(ValueError, "unrequested hidden boundary"):
            self.runner._parse_hidden_oracle_name_map(
                "layer_out=l_out", requested_names=["attn_out"]
            )
        with self.assertRaisesRegex(ValueError, "more than one boundary"):
            self.runner._parse_hidden_oracle_name_map(
                "attn_out=kqv_out,after_attn=kqv_out",
                requested_names=["attn_out", "after_attn"],
            )

    def test_hidden_xray_relabels_explicit_oracle_boundaries(self) -> None:
        dump = self.runner.first_token.parity_test_v7.ParityDump(
            2,
            "kqv_out",
            np.array([1.0, 2.0], dtype=np.float32),
            7,
            "fp32",
            source_token_id=6,
            source_name="kqv_out-2-occ-0",
        )

        relabeled = self.runner._apply_hidden_oracle_name_map(
            [dump], {"attn_out": "kqv_out"}
        )

        self.assertEqual(relabeled[0].op_name, "attn_out")
        self.assertEqual(relabeled[0].token_id, 7)
        self.assertEqual(relabeled[0].source_token_id, 6)
        self.assertEqual(relabeled[0].source_name, "kqv_out-2-occ-0")
        np.testing.assert_array_equal(relabeled[0].data, dump.data)

    def test_load_ck_hidden_exports_preserves_layer_and_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "tok_1042_layer_003_layer_out.f32"
            np.array([1.0, 2.0], dtype=np.float32).tofile(path)

            dumps = self.runner._load_ck_hidden_exports(root, ["layer_out"], 3)

            self.assertEqual(len(dumps), 1)
            self.assertEqual(dumps[0].layer_id, 3)
            self.assertEqual(dumps[0].op_name, "layer_out")
            self.assertEqual(dumps[0].token_id, 1042)
            np.testing.assert_array_equal(dumps[0].data, [1.0, 2.0])

    def test_segmented_ck_loader_preserves_all_position_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for position, values in (
                (0, [1.0, 2.0]),
                (9, [3.0]),
                (1017, [4.0, 5.0]),
            ):
                np.asarray(values, dtype=np.float32).tofile(
                    root / f"tok_{position:04d}_layer_003_layer_out.f32"
                )

            dumps = self.runner._load_ck_hidden_export_occurrences(
                root, ["layer_out"], 3
            )

            self.assertEqual([dump.token_id for dump in dumps], [0, 9, 1017])
            np.testing.assert_array_equal(
                self.runner._coalesce_segmented_prefill_dumps(dumps)[0].data,
                [1.0, 2.0, 3.0, 4.0, 5.0],
            )

    def test_segmented_ck_loader_keeps_only_final_persistent_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for position, value in ((0, 10.0), (9, 20.0), (1017, 30.0)):
                np.asarray([value], dtype=np.float32).tofile(
                    root / f"tok_{position:04d}_layer_003_new_state.f32"
                )

            dumps = self.runner._load_ck_hidden_export_occurrences(
                root, ["new_state"], 3
            )
            result = self.runner._coalesce_segmented_prefill_dumps(dumps)

            self.assertEqual(result[0].token_id, 1017)
            np.testing.assert_array_equal(result[0].data, [30.0])

    def test_segmented_ck_loader_splits_pairwise_mlp_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Two rows of [gate(2), up(2)] in the standalone-prefill layout.
            np.asarray(
                [1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 13.0, 14.0],
                dtype=np.float32,
            ).tofile(root / "tok_0009_layer_003_mlp_gate_up.f32")

            dumps = self.runner._load_ck_hidden_export_occurrences(
                root,
                ["mlp_gate", "mlp_up"],
                3,
                runtime_config={"intermediate_size": 2},
            )

            self.assertEqual([dump.op_name for dump in dumps], ["mlp_gate", "mlp_up"])
            np.testing.assert_array_equal(dumps[0].data, [1.0, 2.0, 3.0, 4.0])
            np.testing.assert_array_equal(dumps[1].data, [11.0, 12.0, 13.0, 14.0])

    def test_segmented_ck_loader_requires_width_for_pairwise_mlp_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.asarray([1.0, 2.0], dtype=np.float32).tofile(
                root / "tok_0000_layer_000_mlp_gate_up.f32"
            )

            with self.assertRaisesRegex(RuntimeError, "positive intermediate_size"):
                self.runner._load_ck_hidden_export_occurrences(
                    root, ["mlp_gate"], 0, runtime_config={}
                )

    def test_oracle_hidden_loader_retains_present_exports_when_one_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.array([1.0, 2.0], dtype=np.float32).tofile(
                root / "tok_0000_layer_003_layer_out.f32"
            )

            dumps = self.runner._load_ck_hidden_exports(
                root,
                ["conv_input", "layer_out"],
                3,
                allow_missing=True,
            )

            self.assertEqual(
                [(dump.layer_id, dump.op_name) for dump in dumps],
                [(3, "layer_out")],
            )

    def test_hidden_loader_remains_fail_closed_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "expected hidden dump"):
                self.runner._load_ck_hidden_exports(Path(tmp), ["conv_input"], 3)

    def test_hidden_export_name_matching_is_not_suffix_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.array([1.0], dtype=np.float32).tofile(
                root / "tok_0000_layer_000_attn_norm.f32"
            )
            np.array([2.0], dtype=np.float32).tofile(
                root / "tok_0000_layer_000_post_attn_norm.f32"
            )

            selected = self.runner._hidden_named_files(root, "attn_norm")

            self.assertEqual(
                [path.name for path in selected],
                ["tok_0000_layer_000_attn_norm.f32"],
            )

    def test_recurrent_state_layout_is_normalized_to_llama_axes(self) -> None:
        dump = self.runner.first_token.parity_test_v7.ParityDump(
            1,
            "new_state",
            np.arange(8, dtype=np.float32),
            7,
            "fp32",
        )

        result = self.runner._normalize_ck_recurrent_state_layout(
            [dump], {"recurrent_num_heads": 2, "recurrent_head_dim": 2}
        )

        expected = np.arange(8, dtype=np.float32).reshape(2, 2, 2).transpose(0, 2, 1)
        np.testing.assert_array_equal(result[0].data, expected)

    def test_llama_physical_recurrent_state_layout_is_not_transposed(self) -> None:
        values = np.arange(8, dtype=np.float32)
        dump = self.runner.first_token.parity_test_v7.ParityDump(
            1, "new_state", values.copy(), 7, "fp32"
        )

        result = self.runner._normalize_ck_recurrent_state_layout(
            [dump],
            {
                "recurrent_num_heads": 2,
                "recurrent_head_dim": 2,
                "recurrent_state_physical_layout": "head_value_key_contiguous",
            },
        )

        np.testing.assert_array_equal(result[0].data, values)

    def test_unknown_recurrent_state_layout_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported recurrent_state_physical_layout"):
            self.runner._normalize_ck_recurrent_state_layout(
                [], {"recurrent_state_physical_layout": "ambiguous"}
            )

    def test_segmented_prefill_oracle_concatenates_rows_and_keeps_final_state(self) -> None:
        dump_type = self.runner.first_token.parity_test_v7.ParityDump
        row_dumps = [
            dump_type(1, "attn_norm", np.asarray(values, dtype=np.float32), token, "fp32")
            for values, token in (([1.0, 2.0], 8), ([3.0], 1016), ([4.0, 5.0], 62))
        ]
        state_dumps = [
            dump_type(1, "new_state", np.asarray([value], dtype=np.float32), token, "fp32")
            for value, token in ((10.0, 8), (20.0, 1016), (30.0, 62))
        ]
        state_in_dumps = [
            dump_type(1, "state_predelta", np.asarray([value], dtype=np.float32), token, "fp32")
            for value, token in ((40.0, 8), (50.0, 1016), (60.0, 62))
        ]

        result = self.runner._coalesce_segmented_prefill_oracle_dumps(
            row_dumps + state_in_dumps + state_dumps
        )

        self.assertEqual([(row.op_name, row.token_id) for row in result], [
            ("attn_norm", 62),
            ("state_predelta", 8),
            ("new_state", 62),
        ])
        np.testing.assert_array_equal(result[0].data, [1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(result[1].data, [40.0])
        np.testing.assert_array_equal(result[2].data, [30.0])

    def test_structurally_mismatched_dumps_are_identified_before_comparison(self) -> None:
        dump_type = self.runner.first_token.parity_test_v7.ParityDump
        ck = [dump_type(1, "layer_out", np.zeros(18, dtype=np.float32), 0, "fp32")]
        llama = [dump_type(1, "layer_out", np.zeros(1035, dtype=np.float32), 0, "fp32")]

        result = self.runner._dump_element_count_mismatches(ck, llama)

        self.assertEqual(
            result,
            [
                {
                    "layer": 1,
                    "op": "layer_out",
                    "ck_element_counts": [18],
                    "llama_element_counts": [1035],
                }
            ],
        )

    def test_structural_check_accepts_one_compatible_alias_candidate(self) -> None:
        dump_type = self.runner.first_token.parity_test_v7.ParityDump
        ck = [
            dump_type(1, "layer_out", np.zeros(18, dtype=np.float32), 0, "fp32"),
            dump_type(1, "layer_out", np.zeros(36, dtype=np.float32), 0, "fp32"),
        ]
        llama = [dump_type(1, "layer_out", np.zeros(36, dtype=np.float32), 0, "fp32")]

        self.assertEqual(self.runner._dump_element_count_mismatches(ck, llama), [])

    def test_hidden_xray_ignores_unrequested_oracle_aliases(self) -> None:
        dump_type = self.runner.first_token.parity_test_v7.ParityDump
        dumps = [
            dump_type(0, "q_proj", np.zeros(4, dtype=np.float32), 0, "fp32"),
            dump_type(0, "qcur_rope", np.zeros(4, dtype=np.float32), 0, "fp32"),
            dump_type(0, "v_proj_view", np.zeros(4, dtype=np.float32), 0, "fp32"),
        ]

        filtered = self.runner._filter_requested_dump_semantics(
            dumps, {"q_proj"}
        )

        self.assertEqual([dump.op_name for dump in filtered], ["q_proj"])

    def test_llama_persistent_dump_uses_trajectory_logits_step_index(self) -> None:
        observed = {}

        def run_helper(command, **kwargs):
            observed["command"] = command
            seq_path = Path(command[command.index("--logits-seq-out") + 1])
            logits_path = Path(command[command.index("--logits-out") + 1])
            np.zeros((2, 3), dtype=np.float32).tofile(seq_path)
            np.zeros(3, dtype=np.float32).tofile(logits_path)
            payload = {"ok": True, "n_vocab": 3, "greedy_steps": 2, "greedy_generated": [1, 2]}
            return mock.Mock(returncode=0, stdout=json.dumps(payload) + "\n", stderr="")

        args = Namespace(
            top_k=3,
            llama_decode_mode="batched",
            llama_flash_attention="enabled",
            max_new_tokens=2,
            threads=4,
            llama_no_repack=False,
            llama_persistent_dump_step=0,
            llama_persistent_dump_dir=Path("persistent-dump"),
            llama_persistent_dump_names="l_out-0",
            llama_persistent_dump_flash_inputs=True,
            workdir=Path("work"),
        )
        inputs = {
            "gguf_path": Path("model.gguf"),
            "ctx_len": 4096,
            "tokens_before": [1],
            "tokens_after": [2],
            "llama_prefix_path": None,
            "prefix_grid": None,
            "prefix_row_dim": 0,
            "prefix_text_pos": 0,
        }
        with mock.patch.object(
            self.runner.first_token.compare_first_token_logits_v7,
            "ensure_llama_helper",
            return_value=Path("llama-helper"),
        ), mock.patch.object(
            self.runner,
            "_llama_oracle_evidence",
            return_value={"commit": "a" * 40},
        ), mock.patch.object(self.runner.subprocess, "run", side_effect=run_helper):
            result = self.runner._run_llama_greedy_sequence(inputs, args)

        command = observed["command"]
        index = command.index("--dump-greedy-decode-step")
        self.assertEqual(command[index + 1], "0")
        self.assertEqual(command[command.index("--dump-names") + 1], "l_out-0")
        self.assertIn("--dump-flash-inputs", command)
        flash_index = command.index("--flash-attn")
        self.assertEqual(command[flash_index + 1], "enabled")
        self.assertEqual(result["oracle_evidence"]["commit"], "a" * 40)

    def test_llama_flash_input_capture_rejects_unfused_oracle(self) -> None:
        args = Namespace(
            top_k=3,
            llama_decode_mode="batched",
            llama_flash_attention="disabled",
            max_new_tokens=1,
            threads=1,
            llama_no_repack=False,
            llama_persistent_dump_step=0,
            llama_persistent_dump_dir=Path("persistent-dump"),
            llama_persistent_dump_names="kqv_out-0",
            llama_persistent_dump_flash_inputs=True,
            workdir=Path("work"),
        )
        inputs = {
            "gguf_path": Path("model.gguf"),
            "ctx_len": 128,
            "tokens_before": [1],
            "tokens_after": [],
            "llama_prefix_path": None,
            "prefix_grid": None,
            "prefix_row_dim": 0,
            "prefix_text_pos": 0,
        }
        with mock.patch.object(
            self.runner.first_token.compare_first_token_logits_v7,
            "ensure_llama_helper",
            return_value=Path("llama-helper"),
        ), mock.patch.object(
            self.runner,
            "_llama_oracle_evidence",
            return_value={"commit": "a" * 40},
        ):
            with self.assertRaisesRegex(ValueError, "requires --llama-flash-attention enabled"):
                self.runner._run_llama_greedy_sequence(inputs, args)

    def test_kv_first_difference_decodes_semantic_location(self) -> None:
        header = struct.pack("<8I", 0x564B5843, 1, 3, 5, 2, 4096, 4, 0)
        values = [0] * (2 * 2 * 5 * 4)
        persistent = bytearray(header + struct.pack(f"<{len(values)}H", *values))
        replay = bytearray(persistent)
        # V, head 1, token 2, channel 3.
        element = (2 * 5 * 4) + (1 * 5 * 4) + (2 * 4) + 3
        offset = len(header) + element * 2
        struct.pack_into("<H", persistent, offset, 0x3C00)
        struct.pack_into("<H", replay, offset, 0x3C01)

        row = self.runner._describe_kv_f16_difference(bytes(persistent), bytes(replay), offset)

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["kind"], "V")
        self.assertEqual(row["layer"], 3)
        self.assertEqual(row["head"], 1)
        self.assertEqual(row["token"], 2)
        self.assertEqual(row["channel"], 3)
        self.assertEqual(row["fp16_bit_distance"], 1)

    def test_compiler_provenance_accepts_matching_runtime(self) -> None:
        evidence = {
            "shared_library": {
                "build": {"compiler": {"command": ["gcc"], "version": "gcc 15"}},
                "elf_compiler_comments": ["GCC: 15"],
            },
            "engine_library": {
                "build": {"compiler": {"command": ["gcc"], "version": "gcc 15"}},
                "elf_compiler_comments": ["GCC: 15"],
            },
        }
        result = self.runner._validate_runtime_compiler_provenance(evidence)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["decoder_family"], "gcc")

    def test_compiler_provenance_rejects_gcc_decoder_with_icx_engine(self) -> None:
        evidence = {
            "shared_library": {
                "build": {"compiler": {"command": "cc", "version": "gcc 15"}},
                "elf_compiler_comments": ["GCC: 15"],
            },
            "engine_library": {
                "elf_compiler_comments": [
                    "GCC: 15",
                    "Intel(R) oneAPI DPC++/C++ Compiler 2026.0.0",
                ],
            },
        }
        with self.assertRaisesRegex(RuntimeError, "compiler provenance"):
            self.runner._validate_runtime_compiler_provenance(evidence)

    def test_compiler_provenance_rejects_unknown_family(self) -> None:
        evidence = {
            "shared_library": {"elf_compiler_comments": []},
            "engine_library": {"elf_compiler_comments": []},
        }
        with self.assertRaisesRegex(RuntimeError, "unknown_compiler_family"):
            self.runner._validate_runtime_compiler_provenance(evidence)

    def test_llama_oracle_isa_accepts_matching_avx512(self) -> None:
        observed = {"avx2": True, "avx512": True}
        self.assertEqual(
            self.runner._validate_llama_oracle_isa(observed, "avx512"),
            "avx512",
        )

    def test_llama_oracle_isa_rejects_avx2_for_avx512_claim(self) -> None:
        observed = {"avx2": True, "avx512": False}
        with self.assertRaisesRegex(RuntimeError, "oracle ISA mismatch"):
            self.runner._validate_llama_oracle_isa(observed, "avx512")

    def test_llama_oracle_isa_rejects_avx512_for_avx2_only_claim(self) -> None:
        observed = {"avx2": True, "avx512": True}
        with self.assertRaisesRegex(RuntimeError, "avx2-only"):
            self.runner._validate_llama_oracle_isa(observed, "avx2")

    def test_dump_first_divergence_resolves_observed_step(self) -> None:
        report = {"first_divergence": {"step": 60}}
        self.assertEqual(self.runner._resolve_dump_step(report, None, True), 60)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.runner._resolve_dump_step(report, 119, True)
        with self.assertRaisesRegex(RuntimeError, "did not diverge"):
            self.runner._resolve_dump_step({}, None, True)

    def test_diagnostic_failure_preserves_coarse_report_and_continues(self) -> None:
        report = {"status": "fail", "first_divergence": {"step": 4}}
        args = Namespace(
            dump_step=None,
            dump_first_divergence=True,
            full_replay_step=4,
            hidden_state_step=None,
        )
        with mock.patch.object(
            self.runner,
            "_capture_step_dump",
            side_effect=RuntimeError("wrong engine"),
        ), mock.patch.object(
            self.runner,
            "_capture_full_replay_step",
            return_value={"step": 4, "status": "ok"},
        ):
            passed = self.runner._run_requested_diagnostics(report, args)

        self.assertFalse(passed)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["full_replay_step"]["status"], "ok")
        self.assertEqual(report["diagnostic_errors"][0]["diagnostic"], "step_dump")
        self.assertEqual(report["diagnostic_errors"][0]["error_type"], "RuntimeError")
        self.assertIn("wrong engine", report["diagnostic_errors"][0]["message"])

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_module()

    def test_bridge_stop_token_list_is_authoritative(self) -> None:
        self.assertEqual(
            self.runner._resolve_stop_token_ids(
                {"stop_token_ids": [151645, 151643], "eos_token_id": 7}
            ),
            {151645, 151643},
        )

    def test_eos_token_is_used_when_bridge_has_no_stop_list(self) -> None:
        self.assertEqual(
            self.runner._resolve_stop_token_ids({"eos_token_id": 151645}),
            {151645},
        )

    def test_only_a_matched_declared_token_stops_parity(self) -> None:
        stops = {151645}
        self.assertTrue(self.runner._is_matched_stop_token(151645, 151645, stops))
        self.assertFalse(self.runner._is_matched_stop_token(151645, 4, stops))
        self.assertFalse(self.runner._is_matched_stop_token(4, 4, stops))

    def test_segmented_append_auto_selects_batched_oracle(self) -> None:
        bridge = {
            "bridge_contract": {
                "prefill_schedule": {
                    "segments": ["text_before", "visual", "text_after"],
                    "cache_transition": "append_preserve",
                }
            }
        }
        result = self.runner._resolve_oracle_prefill_mode("auto", bridge)
        self.assertEqual(result["resolved"], "batched")
        self.assertTrue(result["compatible"])
        self.assertEqual(result["scope"], "production")

    def test_current_bridge_report_identifies_executed_segmented_prefill(self) -> None:
        bridge = {
            "multimodal_prompt_segmented": True,
            "bridge_runtime_policy": "decode-staged",
            "prefix_tokens": 1008,
            "bridge_contract": {"prefill_schedules": {"segmented_append": {}}},
        }

        self.assertTrue(self.runner._uses_segmented_append_prefill(bridge))
        result = self.runner._resolve_oracle_prefill_mode("auto", bridge)
        self.assertEqual(result["required"], "batched")

    def test_generated_layout_identifies_segmented_prefill(self) -> None:
        config = {
            "multimodal_bridge_contract": {
                "prefill_batching": "segmented_append",
                "prefill_schedule": {
                    "segments": ["text_before", "visual", "text_after"],
                    "cache_transition": "append_preserve",
                },
            }
        }

        self.assertTrue(
            self.runner._uses_segmented_append_prefill({}, runtime_config=config)
        )

    def test_unified_prefill_auto_also_selects_concrete_batched_oracle(self) -> None:
        result = self.runner._resolve_oracle_prefill_mode(
            "auto", {"bridge_contract": {}}
        )

        self.assertEqual(result["resolved"], "batched")
        self.assertIsNone(result["required"])
        self.assertTrue(result["compatible"])

    def test_segmented_append_rejects_sequential_oracle(self) -> None:
        bridge = {
            "bridge_contract": {
                "prefill_schedule": {
                    "segments": ["text_before", "visual", "text_after"],
                    "cache_transition": "append_preserve",
                }
            }
        }
        with self.assertRaisesRegex(RuntimeError, "HARD PARITY CONTRACT FAULT"):
            self.runner._resolve_oracle_prefill_mode("sequential", bridge)

    def test_exact_runtime_reuse_loads_only_declared_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            decoder_dir = Path(tmp)
            (decoder_dir / "layout_decode.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "embed_dim": 4096,
                            "num_deepstack_layers": 3,
                            "context_length": 4096,
                            "vocab_size": 151936,
                        }
                    }
                ),
                encoding="utf-8",
            )
            for name in ("weights.bump", "weights_manifest.map", "libdecoder_v8.so"):
                (decoder_dir / name).touch()
            (decoder_dir / "libdecoder_v8.so.build.json").write_text(
                json.dumps({"compiler": {"command": "cc", "version": "gcc test"}}),
                encoding="utf-8",
            )

            runtime = self.runner._load_exact_decoder_runtime(
                Path("decoder.gguf"),
                decoder_dir,
                so_override=None,
                manifest_map_override=None,
            )

            self.assertEqual(runtime["embed_dim"], 4096)
            self.assertEqual(runtime["input_embed_dim"], 16384)
            self.assertEqual(runtime["context_length"], 4096)
            self.assertEqual(runtime["so_path"], decoder_dir / "libdecoder_v8.so")
            evidence = self.runner._runtime_evidence(runtime, exact_reuse=True)
            self.assertTrue(evidence["exact_reuse"])
            self.assertEqual(
                evidence["shared_library"]["build"]["compiler"]["version"],
                "gcc test",
            )
            self.assertEqual(
                evidence["shared_library"]["sha256"],
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            )

            explicit_engine = decoder_dir / "libckernel_engine_gcc.so"
            explicit_engine.write_bytes(b"gcc-engine")
            evidence = self.runner._runtime_evidence(
                runtime,
                exact_reuse=True,
                engine_so=explicit_engine,
            )
            self.assertEqual(evidence["engine_library"]["path"], str(explicit_engine.resolve()))
            self.assertEqual(
                evidence["engine_library"]["sha256"],
                "a6fa04e316b06ba1c57d027077def545c4f53954cc7005713e3edffc74255ca5",
            )

    def test_exact_runtime_reuse_fails_instead_of_regenerating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "refusing to regenerate or guess"):
                self.runner._load_exact_decoder_runtime(
                    Path("decoder.gguf"),
                    Path(tmp),
                    so_override=None,
                    manifest_map_override=None,
                )

    def test_explicit_runtime_tuple_is_exact_reuse(self) -> None:
        args = Namespace(
            ck_runtime_so=Path("/tmp/runtime/libdecoder_v8.so"),
            ck_runtime_manifest_map=Path("/tmp/runtime/weights_manifest.map"),
            reuse_bridge_decoder_runtime=False,
            reuse_bridge_decoder_runtime_exact=False,
        )
        decoder_dir, exact = self.runner._resolve_decoder_runtime_request(
            args, {}, Path("/tmp/output")
        )
        self.assertEqual(decoder_dir, Path("/tmp/runtime").resolve())
        self.assertTrue(exact)

    def test_partial_explicit_runtime_tuple_hard_fails(self) -> None:
        args = Namespace(
            ck_runtime_so=Path("/tmp/runtime/libdecoder_v8.so"),
            ck_runtime_manifest_map=None,
            reuse_bridge_decoder_runtime=False,
            reuse_bridge_decoder_runtime_exact=False,
        )
        with self.assertRaisesRegex(ValueError, "requires both"):
            self.runner._resolve_decoder_runtime_request(args, {}, Path("/tmp/output"))

    def test_segmented_hidden_capture_selects_final_physical_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for token in (0, 5, 1013):
                (root / f"tok_{token:04d}_layer_000_layer_out_last.f32").touch()
                (root / f"tok_{token:04d}_layer_001_layer_out_last.f32").touch()

            selected = self.runner._hidden_files_by_layer(root)

            self.assertEqual(selected[0].name, "tok_1013_layer_000_layer_out_last.f32")
            self.assertEqual(selected[1].name, "tok_1013_layer_001_layer_out_last.f32")

    def test_segmented_hidden_capture_rejects_ambiguous_final_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a_tok_1013_layer_000_layer_out_last.f32"
            second = root / "b_tok_1013_layer_000_layer_out_last.f32"
            first.touch()
            second.touch()

            with self.assertRaisesRegex(RuntimeError, "share final token position 1013"):
                self.runner._hidden_files_by_layer(root)

    def test_hidden_capture_preflight_accepts_exact_decode_and_replay_exporters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "decoder_v8.c"
            source.write_text(
                '\n'.join(
                    [
                        'ck_debug_export_hidden(model, 0, "rope_q", data, 128);',
                        'ck_debug_export_hidden(model, 0, "rope_q_last", data, 128);',
                        'ck_debug_export_hidden(model, 1, "rope_q", data, 128);',
                        'ck_debug_export_hidden(model, 1, "rope_q_last", data, 128);',
                    ]
                ),
                encoding="utf-8",
            )

            catalog = self.runner._validate_hidden_capture_request(
                {"workdir": root, "c_path": source}, ["rope_q"], 1
            )

            self.assertEqual(catalog["rope_q"], [0, 1])
            self.assertEqual(catalog["rope_q_last"], [0, 1])

    def test_hidden_capture_preflight_rejects_alias_before_model_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "decoder_v8.c"
            source.write_text(
                '\n'.join(
                    [
                        'ck_debug_export_hidden(model, 0, "qk_norm_q", data, 128);',
                        'ck_debug_export_hidden(model, 0, "qk_norm_q_last", data, 128);',
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "qcur_normed.*does not exist.*Valid base names: qk_norm_q",
            ):
                self.runner._validate_hidden_capture_request(
                    {"workdir": root, "c_path": source}, ["qcur_normed"], 0
                )

    def test_recurrent_semantic_names_resolve_to_exact_exporters(self) -> None:
        self.assertEqual(
            self.runner._resolve_ck_hidden_export_names(
                ["qkv", "q_predelta", "k_predelta", "v_predelta", "new_state"]
            ),
            [
                "linear_attn_qkv_mixed",
                "q_conv_predelta",
                "k_conv_predelta",
                "v_conv_predelta",
                "new_state",
            ],
        )

    def test_hidden_capture_preflight_rejects_missing_replay_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "decoder_v8.c"
            source.write_text(
                'ck_debug_export_hidden(model, 0, "attn_out", data, 128);\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "attn_out_last.*does not exist"):
                self.runner._validate_hidden_capture_request(
                    {"workdir": root, "c_path": source}, ["attn_out"], 0
                )

            catalog = self.runner._validate_hidden_capture_request(
                {"workdir": root, "c_path": source},
                ["attn_out"],
                0,
                require_replay=False,
            )
            self.assertEqual(catalog["attn_out"], [0])

    def test_hidden_capture_preflight_uses_prefill_source_for_step_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decode_source = root / "decoder_v8.c"
            prefill_source = root / "decoder_v8_prefill.c"
            decode_source.write_text(
                'ck_debug_export_hidden(model, 0, "decode_only", data, 1);\n',
                encoding="utf-8",
            )
            prefill_source.write_text(
                'ck_debug_export_hidden(model, 0, "new_state", data, 1);\n',
                encoding="utf-8",
            )

            catalog = self.runner._validate_hidden_capture_request(
                {
                    "c_path": decode_source,
                    "prefill_c_path": prefill_source,
                    "prefill_so_path": root / "libdecoder_v8_prefill.so",
                },
                ["new_state"],
                0,
                require_replay=False,
                execution_phase="prefill",
            )

            self.assertEqual(catalog["new_state"], [0])

    def test_hidden_capture_batches_multiple_names_into_two_model_executions(self) -> None:
        class Library:
            def ck_model_decode(self, token, logits):
                return 0

            def ck_model_free(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "decoder_v8.c"
            names = ["rope_q", "rope_k", "attn_out"]
            source.write_text(
                "\n".join(
                    f'ck_debug_export_hidden(model, 0, "{name}{suffix}", data, 128);'
                    for name in names
                    for suffix in ("", "_last")
                ),
                encoding="utf-8",
            )
            inputs = {
                "runtime": {
                    "workdir": root,
                    "c_path": source,
                    "prefill_c_path": source,
                },
                "tokens_after": [1, 2],
            }
            report = {
                "steps": [
                    {},
                    {
                        "generated_prefix": [10, 11],
                        "ck_next": 12,
                        "llama_next": 12,
                        "ck_logits_sha256": self.runner._logits_sha256(
                            np.zeros(3, dtype=np.float32)
                        ),
                    },
                ]
            }
            args = Namespace(
                hidden_state_step=1,
                hidden_state_layer=0,
                hidden_state_names=",".join(names),
                hidden_state_dir=root / "capture",
                hidden_state_atol=1.0e-5,
                workdir=root,
                ck_strict_parity=False,
            )
            logits = (ctypes.c_float * 3)(0.0, 0.0, 0.0)
            init = mock.Mock(side_effect=[(Library(), logits, 3), (Library(), logits, 3)])

            def hidden_file(_directory, name):
                if name == "rope_k_last":
                    raise RuntimeError("missing rope_k replay checkpoint")
                return root / f"{name}.f32"

            with mock.patch.object(self.runner, "_prepare_inputs", return_value=inputs), \
                 mock.patch.object(self.runner, "_init_ck_state", init), \
                 mock.patch.object(self.runner, "_single_hidden_file", side_effect=hidden_file), \
                 mock.patch.object(self.runner, "_hidden_compare", return_value={"status": "ok", "max_abs_diff": 0.0}):
                result = self.runner._capture_hidden_state_step(report, args)

            self.assertEqual(init.call_count, 2)
            self.assertEqual(result["ck_execution_count"], 2)
            self.assertEqual(result["preflight"]["requested_names"], names)
            self.assertEqual(len(result["results"]), len(names))
            self.assertEqual([row["name"] for row in result["results"]], names)
            self.assertEqual([row["status"] for row in result["results"]], ["ok", "error", "ok"])
            self.assertIn("missing rope_k", result["results"][1]["error"])
            self.assertEqual(result["observational_neutrality"]["status"], "accepted")

    def test_prefill_hidden_capture_arms_export_before_initialization_without_replay(self) -> None:
        class Library:
            def ck_model_decode(self, token, logits):
                return 0

            def ck_model_free(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "decoder_v8.c"
            source.write_text(
                'ck_debug_export_hidden(model, 0, "rope_q", data, 128);\n',
                encoding="utf-8",
            )
            inputs = {
                "runtime": {
                    "workdir": root,
                    "c_path": source,
                    "prefill_c_path": source,
                },
                "tokens_after": [1, 2],
            }
            logits = (ctypes.c_float * 3)(0.0, 0.0, 0.0)
            report = {
                "steps": [{
                    "generated_prefix": [],
                    "ck_next": 12,
                    "llama_next": 12,
                    "ck_logits_sha256": self.runner._logits_sha256(
                        np.zeros(3, dtype=np.float32)
                    ),
                }]
            }
            args = Namespace(
                hidden_state_step=0,
                hidden_state_layer=0,
                hidden_state_names="rope_q",
                hidden_state_dir=root / "capture",
                hidden_state_atol=1.0e-5,
                hidden_state_skip_full_replay=False,
                llama_persistent_dump_dir=None,
                workdir=root,
                ck_strict_parity=False,
            )

            def init_state(*_args, **_kwargs):
                self.assertEqual(
                    self.runner.os.environ.get("CK_DEBUG_EXPORT_HIDDEN_NAMES"),
                    "rope_q",
                )
                return Library(), logits, 3

            with mock.patch.object(self.runner, "_prepare_inputs", return_value=inputs), \
                 mock.patch.object(self.runner, "_init_ck_state", side_effect=init_state) as init:
                result = self.runner._capture_hidden_state_step(report, args)

            self.assertEqual(init.call_count, 1)
            self.assertEqual(result["ck_execution_count"], 1)
            self.assertEqual(result["full_replay_control"], "not_applicable_prefill")
            self.assertEqual(result["results"], [])
            self.assertIsNone(result["first_issue"])
            self.assertEqual(result["observational_neutrality"]["status"], "accepted")

    def test_hidden_capture_exports_bounded_kv_for_every_requested_layer(self) -> None:
        class ExportKV:
            argtypes = None
            restype = None

            def __call__(self, path, layer):
                Path(path.decode("utf-8")).write_bytes(bytes([int(layer), 0, 1, 2]))
                return 0

        class Library:
            def __init__(self):
                self.ck_model_debug_export_kv_f16 = ExportKV()

            def ck_model_decode(self, token, logits):
                return 0

            def ck_model_free(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "decoder_v8.c"
            source.write_text(
                "\n".join(
                    f'ck_debug_export_hidden(model, {layer}, "{name}", data, 128);'
                    for layer in (0, 1)
                    for name in ("rope_q", "rope_q_last", "attn_out", "attn_out_last")
                ),
                encoding="utf-8",
            )
            inputs = {
                "runtime": {"workdir": root, "c_path": source},
                "tokens_after": [1, 2],
            }
            report = {
                "steps": [
                    {},
                    {"generated_prefix": [10, 11], "ck_next": 12, "llama_next": 12},
                ]
            }
            args = Namespace(
                hidden_state_step=1,
                hidden_state_layer=-1,
                hidden_state_names="rope_q,attn_out",
                hidden_state_dir=root / "capture",
                hidden_state_atol=1.0e-5,
                workdir=root,
                ck_strict_parity=False,
            )
            init = mock.Mock(side_effect=[(Library(), object(), 3), (Library(), object(), 3)])
            compared = [
                {"layer": 0, "status": "ok", "max_abs_diff": 0.0},
                {"layer": 1, "status": "ok", "max_abs_diff": 0.0},
            ]

            with mock.patch.object(self.runner, "_prepare_inputs", return_value=inputs), \
                 mock.patch.object(self.runner, "_init_ck_state", init), \
                 mock.patch.object(self.runner, "_hidden_compare_many", return_value=compared):
                result = self.runner._capture_hidden_state_step(report, args)

            self.assertEqual(result["preflight"]["capture_layers"], [0, 1])
            self.assertEqual(len(result["kv_cache"]["layers"]), 2)
            self.assertEqual(result["kv_cache"]["status"], "ok")
            for phase in ("persistent", "full_replay"):
                for layer in (0, 1):
                    self.assertTrue(
                        (root / "capture" / phase / f"kv_cache_f16_layer_{layer:03d}.bin").is_file()
                    )

    def test_granular_capture_rejects_uninstrumented_runtime_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "decoder_v8.c"
            source.write_text("/* production runtime */\n", encoding="utf-8")
            inputs = {
                "runtime": {"c_path": source},
                "tokens_after": [],
            }
            report = {"steps": [{"generated_prefix": []}]}
            args = Namespace(
                dump_step=0,
                dump_dir=root / "dumps",
                workdir=root,
            )

            with mock.patch.object(self.runner, "_prepare_inputs", return_value=inputs), \
                 mock.patch.object(self.runner.first_token, "_capture_dump_compare") as capture:
                with self.assertRaisesRegex(RuntimeError, "compiled with CK_PARITY_DUMP"):
                    self.runner._capture_step_dump(report, args)

            capture.assert_not_called()

    def test_multimodal_prefill_segments_use_extent_and_execution_order(self) -> None:
        dump_type = self.runner.first_token.parity_test_v7.ParityDump
        dumps = [
            dump_type(0, "q_proj", np.zeros(rows * 4, dtype=np.float32), 0, "fp32")
            for rows in (5, 1008, 14)
        ]
        specs = {(0, "q_proj"): (4, (4,))}
        segments = [
            ("text_before", 5, 0),
            ("visual", 1008, 5),
            ("text_after", 14, 1013),
        ]

        labeled = self.runner.first_token._coalesce_multimodal_prefill_segments(
            dumps, specs, segments
        )

        self.assertEqual(
            [(row.op_name, row.token_id, row.data.shape) for row in labeled],
            [
                ("q_proj", 0, (1027, 4)),
            ],
        )

    def test_llama_qk_middle_occurrence_is_not_guessed_as_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for occurrence, values in enumerate(([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])):
                name = f"Qcur-0-token-000000-occ-{occurrence:03d}"
                np.asarray(values, dtype=np.float32).tofile(root / f"{name}.bin")
                rows.append({
                    "name": name,
                    "base_name": "Qcur-0",
                    "token_id": 0,
                    "occurrence": occurrence,
                    "elem_count": 2,
                    "nbytes": 8,
                    "rank": 1,
                    "shape": [2],
                })
            (root / "index.json").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            dumps = self.runner.first_token._load_llama_dump_dir(root)

        self.assertEqual([dump.op_name for dump in dumps], ["q_proj", "qcur_rope"])
        self.assertNotIn("qcur_normed", [dump.op_name for dump in dumps])

    def test_prepare_inputs_preserves_explicit_engine_for_nested_dumps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = root / "selected-engine.so"
            engine.write_bytes(b"selected")
            runtime = {
                "so_path": root / "decoder" / "libdecoder_v8.so",
                "embed_dim": 4,
                "input_embed_dim": 4,
                "context_length": 32,
            }
            runtime["so_path"].parent.mkdir(parents=True)
            runtime["so_path"].touch()
            bridge_report = {
                "decoder_runtime": {"gguf": str(root / "model.gguf")},
                "decoder_context_len": 32,
            }
            args = Namespace(
                bridge_report=root / "bridge.json",
                workdir=root / "work",
                reuse_bridge_decoder_runtime=False,
                reuse_bridge_decoder_runtime_exact=False,
                ctx_len=32,
                prefix_f32=None,
                prefix_row_dim=None,
                prefix_grid_x=None,
                prefix_grid_y=None,
                prefix_text_pos=None,
                ck_engine_so=engine,
            )
            with mock.patch.object(self.runner.first_token, "_load_bridge_report", return_value=bridge_report), \
                 mock.patch.object(self.runner.first_token.bridge_runner_v8, "_prepare_decoder_runtime", return_value=runtime), \
                 mock.patch.object(self.runner, "GGUFTokenizer") as tokenizer_cls, \
                 mock.patch.object(self.runner.first_token, "_resolve_prompt_token_segments", return_value=(None, [], [], {})), \
                 mock.patch.object(self.runner.first_token, "_load_prefix_embeddings", return_value=(array("f"), 0, 4, "none")), \
                 mock.patch.object(self.runner.first_token.bridge_runner_v8, "_sync_runtime_engine") as sync:
                tokenizer_cls.from_gguf.return_value = object()
                inputs = self.runner._prepare_inputs(args)

            self.assertEqual(inputs["runtime"]["engine_so"], str(engine.resolve()))
            sync.assert_called_once_with(engine.resolve(), runtime["so_path"])

    def test_runner_records_eos_step_without_decoding_past_it(self) -> None:
        class Tokenizer:
            def decode(self, tokens, skip_special=False):
                return ",".join(str(token) for token in tokens)

        class Library:
            decode_calls = 0

            def ck_model_decode(self, token, logits):
                self.decode_calls += 1
                return 0

            def ck_model_free(self):
                return None

        library = Library()
        inputs = {
            "tokenizer": Tokenizer(),
            "bridge_report": {"stop_token_ids": [151645]},
            "gguf_path": Path("decoder.gguf"),
            "workdir": Path("work"),
            "ctx_len": 4096,
            "requested_ctx_len": 4096,
            "tokens_before": [1],
            "tokens_after": [2],
            "prefix_source": "fixture",
            "prefix_tokens": 1008,
            "prefix_row_dim": 16384,
            "prefix_path": Path("prefix.f32"),
            "prefix_grid": (36, 28),
            "prefix_text_pos": 41,
            "runtime": {},
        }
        comparison = {
            "top1_ck": 151645,
            "top1_llama": 151645,
            "cosine": 1.0,
            "rmse": 0.0,
            "mean_abs_diff": 0.0,
            "max_abs_diff": 0.0,
            "ck_top1_margin": 1.0,
            "llama_top1_margin": 1.0,
            "topk_overlap_count": 1,
            "topk_overlap_ratio": 1.0,
            "ck_topk_ids": [151645],
            "llama_topk_ids": [151645],
            "topk_logits": [],
        }
        args = Namespace(
            ck_strict_parity=False,
            max_new_tokens=64,
            top_k=1,
            llama_no_repack=False,
            append_on_divergence="stop",
            bridge_report=Path("bridge_report.json"),
            threads=20,
            llama_decode_mode="persistent",
        )
        llama_sequence = {
            "meta": {"greedy_generated": [151645, 7]},
            "logits": np.zeros((2, 3), dtype=np.float32),
            "oracle_evidence": {
                "root": "/oracle/llama.cpp",
                "commit": "a" * 40,
                "helper": {"sha256": "b" * 64},
                "libraries": [{"sha256": "c" * 64}],
            },
        }
        with mock.patch.object(self.runner, "_prepare_inputs", return_value=inputs), \
             mock.patch.object(self.runner, "_run_llama_greedy_sequence", return_value=llama_sequence), \
             mock.patch.object(self.runner, "_init_ck_state", return_value=(library, object(), 3)), \
             mock.patch.object(self.runner, "_ck_logits_from_buffer", return_value=np.zeros(3, dtype=np.float32)), \
             mock.patch.object(
                 self.runner,
                 "_runtime_evidence",
                 return_value={
                     "shared_library": {"elf_compiler_comments": ["GCC: 15"]},
                     "engine_library": {"elf_compiler_comments": ["GCC: 15"]},
                 },
             ), \
             mock.patch.object(self.runner.first_token.compare_first_token_logits_v7, "compare_logits", return_value=comparison), \
             mock.patch.object(self.runner, "_decode_topk", return_value=[]):
            report = self.runner.run_multimodal_multitoken_parity(args)

        self.assertEqual(report["status"], "pass")
        self.assertFalse(report["execution_modes"]["ck_strict_parity"])
        self.assertEqual(report["execution_modes"]["llama_decode_mode"], "persistent")
        self.assertTrue(report["execution_modes"]["llama_tensor_repack"])
        self.assertFalse(report["execution_modes"]["diagnostic_tensor_dump"])
        self.assertIsInstance(report["execution_modes"]["ck_environment"], dict)
        self.assertEqual(report["stop_reason"], "matched_stop_token")
        self.assertEqual(report["llama_oracle"]["commit"], "a" * 40)
        self.assertEqual(len(report["steps"]), 1)
        self.assertEqual(report["steps"][0]["ck_next"], 151645)
        self.assertEqual(
            report["steps"][0]["ck_logits_sha256"],
            self.runner._logits_sha256(np.zeros(3, dtype=np.float32)),
        )
        self.assertEqual(library.decode_calls, 0)

    def test_execution_evidence_marks_hidden_state_capture_as_diagnostic(self) -> None:
        args = Namespace(hidden_state_step=0)
        self.assertTrue(self.runner._diagnostic_tensor_dump_requested(args))


if __name__ == "__main__":
    unittest.main(verbosity=2)
