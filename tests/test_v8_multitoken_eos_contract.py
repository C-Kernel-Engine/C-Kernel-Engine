#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
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
    def test_llama_persistent_dump_targets_decode_call_before_logits_step(self) -> None:
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
            max_new_tokens=2,
            threads=4,
            llama_no_repack=False,
            llama_persistent_dump_step=5,
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
        self.assertEqual(command[index + 1], "4")
        self.assertEqual(command[command.index("--dump-names") + 1], "l_out-0")
        self.assertIn("--dump-flash-inputs", command)
        self.assertEqual(result["oracle_evidence"]["commit"], "a" * 40)

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
                "runtime": {"workdir": root, "c_path": source},
                "tokens_after": [1, 2],
            }
            report = {
                "steps": [
                    {},
                    {
                        "generated_prefix": [10, 11],
                        "ck_next": 12,
                        "llama_next": 12,
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
            init = mock.Mock(side_effect=[(Library(), object(), 3), (Library(), object(), 3)])

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

        labeled = self.runner.first_token._label_multimodal_prefill_segments(
            dumps, specs, segments
        )

        self.assertEqual(
            [(row.op_name, row.token_id, row.data.size) for row in labeled],
            [
                ("q_proj@text_before", 0, 20),
                ("q_proj@visual", 5, 4032),
                ("q_proj@text_after", 1013, 56),
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
        self.assertEqual(library.decode_calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
