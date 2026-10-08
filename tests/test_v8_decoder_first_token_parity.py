#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V8_DECODER_PARITY_PATH = ROOT / "version" / "v8" / "scripts" / "decoder_first_token_parity_v8.py"


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


decoder_parity_v8 = _load_module("decoder_first_token_parity_v8_tests", V8_DECODER_PARITY_PATH)


class V8DecoderFirstTokenParityTests(unittest.TestCase):
    def test_capture_numerical_failure_cannot_be_hidden_by_logit_pass(self) -> None:
        status = decoder_parity_v8._capture_aware_status
        self.assertEqual(status("pass", None), "pass")
        self.assertEqual(
            status("pass", {"status": "fail", "summary": {"fail": 1}}), "fail"
        )
        self.assertEqual(
            status("pass", {"status": "fail", "summary": {"error": 1}}), "incomplete"
        )
        self.assertEqual(
            status("pass", {"status": "ok", "summary": {"missing": 1}}), "incomplete"
        )
        self.assertEqual(
            status("incomplete", {"status": "fail", "summary": {"fail": 1}}), "fail"
        )

    def test_failed_replay_removes_stale_pass_without_overwriting_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gguf = root / "decoder.gguf"
            gguf.write_bytes(b"fixture")
            output = root / "report.json"
            output.write_text('{"status":"pass"}', encoding="utf-8")
            argv = ["--gguf", str(gguf), "--workdir", str(root / "work"),
                    "--json-out", str(output)]
            with mock.patch.object(
                decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime",
                side_effect=RuntimeError("stale runtime"),
            ), self.assertRaisesRegex(RuntimeError, "stale runtime"):
                decoder_parity_v8.main(argv)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                decoder_parity_v8.main([*argv[:-1], str(gguf)])
            self.assertEqual(gguf.read_bytes(), b"fixture")

    def test_bridge_prefix_decode_policy_fails_closed(self) -> None:
        self.assertEqual(decoder_parity_v8._prefix_decode_policy(None), "causal_mixed_prefix")
        self.assertEqual(
            decoder_parity_v8._prefix_decode_policy({"prefix_decode_policy": "non_causal_visual_chunk"}),
            "non_causal_visual_chunk",
        )
        for invalid in (None, "", "unknown", [], True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "prefix_decode_policy"):
                decoder_parity_v8._prefix_decode_policy({"prefix_decode_policy": invalid})

    def test_encoder_prefix_report_binds_complete_independent_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ck = root / "ck.f32"
            llama = root / "llama.f32"
            artifact = root / "model.so"
            report_path = root / "encoder.json"
            ck.write_bytes(bytes(48))
            llama.write_bytes(bytes(range(48)))
            artifact.write_bytes(b"library")

            def identity(path: Path) -> dict:
                return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

            names = ("mmproj_gguf", "ck_model_library", "ck_generated_source",
                     "ck_engine_library", "ck_weights", "llama_shim_library", "llama_mtmd_library")
            report = {
                "status": "complete",
                "input_provenance": "independently_preprocessed_from_shared_decoded_rgb8",
                "preprocess_evidence": {"verdict": "pass"},
                "strict_mtmd_oracle": False,
                "ck_resolved_output": "vision_output",
                "llama_reference_output": "clip_encode_float_image",
                "row_slices": {"ck": None, "llama": None},
                "feature_slices": {"ck": None, "llama": None},
                "raw_num_values": {"ck": 12, "llama": 12},
                "decoder_prefix_exports": {"contract": "cke.decoder_prefix_f32.v1",
                                           "ck_output_role": "vision_output",
                                           "oracle_output_role": "clip_encode_float_image",
                                           "tokens": 4, "row_dim": 3, "grid": [2, 2],
                                           "ck": identity(ck), "llama": identity(llama)},
                "artifact_identity": {name: identity(artifact) for name in names},
            }

            def verify(candidate: dict) -> dict:
                report_path.write_text(json.dumps(candidate), encoding="utf-8")
                return decoder_parity_v8._verify_encoder_prefix_report(report_path, ck, llama, 4, 3)

            self.assertEqual(verify(report)["input_provenance"], report["input_provenance"])
            alternate_roles = copy.deepcopy(report)
            alternate_roles["ck_resolved_output"] = "bridge_embeddings"
            alternate_roles["llama_reference_output"] = "projected_image_embeddings"
            alternate_roles["decoder_prefix_exports"]["ck_output_role"] = "bridge_embeddings"
            alternate_roles["decoder_prefix_exports"]["oracle_output_role"] = "projected_image_embeddings"
            self.assertEqual(verify(alternate_roles)["input_provenance"], report["input_provenance"])
            for change in (
                {"status": "fail"},
                {"input_provenance": "shared_processed_tensor"},
                {"preprocess_evidence": "pass"},
                {"preprocess_evidence": {"verdict": "fail"}},
                {"strict_mtmd_oracle": True},
                {"ck_resolved_output": "attention_output"},
                {"raw_num_values": {"ck": True, "llama": 12}},
                {"row_slices": {"ck": [0, 1], "llama": None}},
                {"artifact_identity": {}},
            ):
                candidate = copy.deepcopy(report)
                candidate.update(change)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    verify(candidate)
            stale = copy.deepcopy(report)
            stale["decoder_prefix_exports"]["ck"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "export identity"):
                verify(stale)
            wrong_role = copy.deepcopy(report)
            wrong_role["decoder_prefix_exports"]["oracle_output_role"] = "attention_output"
            with self.assertRaisesRegex(ValueError, "inconsistent oracle_output_role"):
                verify(wrong_role)
            artifact.write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                verify(report)

    def test_mtmd_prefix_report_binds_independent_image_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ck = root / "ck.f32"
            oracle = root / "oracle.f32"
            image = root / "image.ppm"
            bridge_path = root / "bridge.json"
            report_path = root / "prefix-parity.json"
            ck.write_bytes(array("f", [1.0, 2.0, 3.0, 4.0]).tobytes())
            oracle.write_bytes(array("f", [1.0, 2.0, 3.0, 4.0]).tobytes())
            image.write_bytes(b"P6\n2 1\n255\n" + bytes((255, 0, 0, 0, 0, 255)))
            names = ("cke_model_library", "oracle_model", "oracle_mmproj", "oracle_probe",
                     "libmtmd", "libllama", "libggml", "libggml-base", "libggml-cpu")
            files = {name: root / name for name in names}
            for name, path in files.items():
                path.write_bytes(name.encode())

            def identity(path: Path) -> dict:
                return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

            pixels_hash = hashlib.sha256(bytes((255, 0, 0, 0, 0, 255))).hexdigest()
            bridge = {
                "status": "ok", "prefix_source": "encoder", "prefix_dump_path": str(ck.resolve()),
                "prefix_dump_sha256": identity(ck)["sha256"],
                "encoder_report": {
                    "image_sha256": identity(image)["sha256"],
                    "decoded_rgb8_sha256": pixels_hash,
                    "model_library_sha256": identity(files["cke_model_library"])["sha256"],
                },
                "encoder_runtime": {"so_path": str(files["cke_model_library"].resolve())},
                "decoder_runtime": {"gguf": str(files["oracle_model"].resolve())},
            }
            bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
            revision = decoder_parity_v8.subprocess.check_output(
                ["git", "-C", str(ROOT), "ls-tree", "HEAD", "llama.cpp"], text=True,
            ).split()[2]
            artifacts = {name: identity(path) for name, path in files.items()}
            artifacts.update({"image": identity(image), "cke_bridge_report": identity(bridge_path)})
            report = {
                "status": "pass", "lane": "independent_mtmd_encoder_prefix",
                "input_provenance": "independently_decoded_and_preprocessed_from_same_p6",
                "metrics": {"tokens": 2, "embed_dim": 2, "rmse": 0.0, "max_abs": 0.0},
                "thresholds": {"max_rmse": 0.1, "max_abs": 0.1},
                "decoder_prefix_exports": {
                    "contract": "cke.decoder_prefix_f32.v1", "tokens": 2, "row_dim": 2,
                    "ck": identity(ck), "llama": identity(oracle),
                },
                "artifact_identity": artifacts,
                "provenance": {
                    "cke_prefix_sha256": identity(ck)["sha256"],
                    "oracle_prefix_sha256": identity(oracle)["sha256"],
                    "image_sha256": identity(image)["sha256"],
                    "cke_encoder_library_sha256": artifacts["cke_model_library"]["sha256"],
                    "cke_decoded_rgb8_sha256": pixels_hash,
                    "oracle_decoded_rgb8_sha256": pixels_hash,
                    "oracle_model_sha256": artifacts["oracle_model"]["sha256"],
                    "oracle_mmproj_sha256": artifacts["oracle_mmproj"]["sha256"],
                    "oracle_probe_sha256": artifacts["oracle_probe"]["sha256"],
                    "oracle_library_sha256": {
                        name: artifacts[name]["sha256"] for name in names if name.startswith("lib")
                    },
                    "oracle_revision": revision,
                },
            }

            def verify(candidate: dict) -> dict:
                report_path.write_text(json.dumps(candidate), encoding="utf-8")
                return decoder_parity_v8._verify_encoder_prefix_report(
                    report_path, ck, oracle, 2, 2, bridge_path,
                )

            self.assertEqual(verify(report)["input_provenance"], report["input_provenance"])
            for mutate in (
                lambda row: row.update(status="fail"),
                lambda row: row["metrics"].update(rmse=0.2),
                lambda row: row["metrics"].update(rmse=0.01),
                lambda row: row["metrics"].update(max_abs=float("nan")),
                lambda row: row["decoder_prefix_exports"]["llama"].update(sha256="0" * 64),
                lambda row: row["artifact_identity"]["image"].update(sha256="0" * 64),
                lambda row: row["provenance"].update(cke_prefix_sha256="0" * 64),
                lambda row: row["provenance"].update(oracle_decoded_rgb8_sha256="0" * 64),
                lambda row: row["provenance"].update(oracle_revision="0" * 40),
            ):
                candidate = copy.deepcopy(report)
                mutate(candidate)
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    verify(candidate)
            verify(report)
            with self.assertRaisesRegex(ValueError, "bridge run"):
                decoder_parity_v8._verify_encoder_prefix_report(
                    report_path, ck, oracle, 2, 2, root / "other-bridge.json",
                )

    def test_xray_verifies_planner_memory_contract_without_recomputing_extent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_memory_contract_") as tmpdir:
            root = Path(tmpdir)
            paths = {}
            for phase in ("prefill", "decode"):
                path = root / f"layout_{phase}.json"
                path.write_text(
                    json.dumps(
                        {
                            "validation": {
                                "activation_memory": {
                                    "status": "PASS",
                                    "arena_bytes": 4096,
                                    "writes": [
                                        {
                                            "op": "quantize_input",
                                            "layer": 0,
                                            "provider": "quantize_row_q8_k",
                                            "buffer": "layer_input",
                                            "required_bytes": 2920,
                                            "available_bytes": 2920,
                                        }
                                    ],
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                paths[phase] = path

            report = decoder_parity_v8._verify_runtime_memory_contracts(
                {
                    "prefill_layout_path": paths["prefill"],
                    "decode_layout_path": paths["decode"],
                }
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["phases"]["prefill"]["checked_writes"], 1)

    def test_xray_rejects_planner_memory_contract_overflow(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_memory_contract_bad_") as tmpdir:
            path = Path(tmpdir) / "layout.json"
            path.write_text(
                json.dumps(
                    {
                        "validation": {
                            "activation_memory": {
                                "status": "PASS",
                                "writes": [
                                    {
                                        "op": "quantize_input",
                                        "layer": 3,
                                        "provider": "quantize_row_q8_k",
                                        "buffer": "layer_input",
                                        "required_bytes": 292000,
                                        "available_bytes": 272000,
                                    }
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "phase=prefill.*required=292000 available=272000",
            ):
                decoder_parity_v8._verify_runtime_memory_contracts(
                    {
                        "prefill_layout_path": path,
                        "decode_layout_path": path,
                    }
                )

    def test_llama_capture_preserves_explicit_prefix_and_decode_schedules(self) -> None:
        captured: list[str] = []
        actual_mode = ["enabled"]

        def fake_run(command: list[str]):
            captured.extend(command)
            logits_path = Path(command[command.index("--logits-out") + 1])
            np.array([0.0], dtype=np.float32).tofile(logits_path)
            return mock.Mock(
                returncode=0,
                stdout=json.dumps({"ok": True, "n_vocab": 1, "flash_attention_mode": actual_mode[0]}),
                stderr="",
            )

        with mock.patch.object(
            decoder_parity_v8.compare_first_token_logits_v7,
            "ensure_llama_helper",
            return_value=Path("/tmp/llama-token-replay"),
        ), mock.patch.object(decoder_parity_v8, "_run", side_effect=fake_run):
            decoder_parity_v8._run_llama_capture(
                Path("model.gguf"),
                [3],
                128,
                1,
                1,
                tokens_before=[1, 2],
                prefix_decode_mode="batched",
                decode_mode="sequential",
                flash_attention="enabled",
            )
            actual_mode[0] = "disabled_for_internal_dump"
            with self.assertRaisesRegex(RuntimeError, "did not execute requested flash attention"):
                decoder_parity_v8._run_llama_capture(
                    Path("model.gguf"), [3], 128, 1, 1, flash_attention="enabled",
                )

        self.assertEqual(captured[captured.index("--prefix-decode-mode") + 1], "batched")
        self.assertEqual(captured[captured.index("--decode-mode") + 1], "sequential")
        self.assertEqual(captured[captured.index("--flash-attn") + 1], "enabled")

    def test_llama_helper_fingerprint_tracks_root_source_and_library_content(self) -> None:
        helper_module = decoder_parity_v8.compare_first_token_logits_v7
        with tempfile.TemporaryDirectory(prefix="v8_llama_helper_identity_") as tmpdir:
            root = Path(tmpdir)
            source = root / "llama_token_replay_v8.cpp"
            source.write_text("int main() { return 0; }\n", encoding="utf-8")
            lib_dir = root / "build" / "bin"
            lib_dir.mkdir(parents=True)
            for name in ("libllama.so", "libggml.so", "libggml-cpu.so", "libggml-base.so"):
                (lib_dir / name).write_bytes(name.encode("ascii"))

            with mock.patch.object(helper_module, "LLAMA_CPP", root), mock.patch.object(
                helper_module, "HELPER_SRC", source
            ):
                original = helper_module._llama_helper_fingerprint()
                (lib_dir / "libggml-cpu.so").write_bytes(b"different provider")
                changed_library = helper_module._llama_helper_fingerprint()
                source.write_text("int main() { return 1; }\n", encoding="utf-8")
                changed_source = helper_module._llama_helper_fingerprint()

            self.assertNotEqual(original, changed_library)
            self.assertNotEqual(changed_library, changed_source)

    def test_ck_dump_filter_names_expands_llama_aliases(self) -> None:
        self.assertEqual(
            decoder_parity_v8._ck_dump_filter_names("Qcur-0,Kcur_normed-2,ffn_inp-0,l_out-3"),
            "Qcur-0,q_proj-0,Qcur_normed-0,qcur_normed-0,Qcur_rope-0,qcur_rope-0,"
            "Kcur_normed-2,kcur_normed-2,ffn_inp-0,l_out-3,layer_out-3",
        )
        self.assertEqual(
            decoder_parity_v8._ck_dump_filter_names("Kcur-1"),
            "Kcur-1,k_proj-1,Kcur_normed-1,kcur_normed-1,Kcur_rope-1,kcur_rope-1",
        )
        self.assertEqual(
            decoder_parity_v8._ck_dump_filter_names("layer_input,after_attn,layer_out"),
            "layer_input-0,layer_out,after_attn,attn_residual,ffn_inp",
        )
        self.assertEqual(
            decoder_parity_v8._ck_dump_filter_names("layer_input-3,after_attn-3"),
            "layer_out-2,after_attn-3,attn_residual-3,ffn_inp-3",
        )
        self.assertEqual(
            decoder_parity_v8._ck_dump_filter_names("result_norm,result_output"),
            "result_norm,final_norm,final_hidden,final_hidden_last,"
            "result_output,logits",
        )
        self.assertEqual(
            decoder_parity_v8._ck_dump_filter_names("post_attn_norm-0,mlp_down-0"),
            "post_attn_norm-0,attn_post_norm-0,mlp_down-0,down_proj-0",
        )

    def test_resolve_llama_dump_names_expands_semantic_boundaries(self) -> None:
        self.assertEqual(
            decoder_parity_v8._resolve_llama_dump_names(
                "layer_input,after_attn,layer_out", layer_count=3
            ),
            "model.input_embed,l_out-0,l_out-1,"
            "attn_residual-0,attn_residual-1,attn_residual-2,"
            "l_out-2",
        )

    def test_resolve_llama_dump_names_supports_layer_suffix_and_native_names(self) -> None:
        self.assertEqual(
            decoder_parity_v8._resolve_llama_dump_names(
                "layer_input-2,after_attn-1,Qcur-4", layer_count=5
            ),
            "l_out-1,attn_residual-1,Qcur-4",
        )
        with self.assertRaisesRegex(ValueError, "outside decoder layers"):
            decoder_parity_v8._resolve_llama_dump_names("layer_out-5", layer_count=5)

    def test_post_layer_embedding_boundaries_follow_resolved_ir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_ir_boundary_") as tmpdir:
            tmp = Path(tmpdir)
            ir_path = tmp / "ir1_decode.json"
            runtime = {"decode_layout_path": tmp / "layout_decode.json"}
            with self.assertRaisesRegex(RuntimeError, "decode IR required"):
                decoder_parity_v8._post_layer_embedding_layers(runtime)
            ir_path.write_text(
                json.dumps({"ops": [
                    {"op": "layer_out", "layer": 0},
                    {"op": "gemma4_per_layer_embed", "layer": 0},
                    {"op": "layer_out", "layer": 1},
                ]}),
                encoding="utf-8",
            )
            layers = decoder_parity_v8._post_layer_embedding_layers(runtime)

        self.assertEqual(layers, frozenset({0}))
        self.assertEqual(
            decoder_parity_v8._resolve_llama_dump_names(
                "layer_out-0,gemma4_per_layer_embed-0,layer_input-1,layer_out-1",
                layer_count=2,
                post_layer_embedding_layers=layers,
            ),
            "pe_in-0,l_out-0,l_out-1",
        )
        self.assertEqual(
            decoder_parity_v8._ck_dump_filter_names(
                "layer_input-1", post_layer_embedding_layers=layers
            ),
            "gemma4_per_layer_embed-0",
        )
        self.assertEqual(
            decoder_parity_v8._ck_dump_filter_names(
                "l_out-0", post_layer_embedding_layers=layers
            ),
            "gemma4_per_layer_embed-0",
        )
        with self.assertRaisesRegex(ValueError, "absent from decode IR"):
            decoder_parity_v8._resolve_llama_dump_names(
                "gemma4_per_layer_embed-1",
                layer_count=2,
                post_layer_embedding_layers=layers,
            )

        dump = decoder_parity_v8.parity_test_v7.ParityDump
        before = np.array([10.0, 20.0], dtype=np.float32)
        after = np.array([30.0, 40.0], dtype=np.float32)
        rows = decoder_parity_v8._augment_layer_input_aliases(
            [dump(0, "layer_out", before, 3, "fp32"),
             dump(0, "gemma4_per_layer_embed", after, 3, "fp32")],
            layer_count=2,
            post_layer_embedding_layers=layers,
        )
        self.assertEqual(
            [(row.layer_id, row.op_name) for row in rows],
            [(0, "layer_out"), (0, "gemma4_per_layer_embed"), (1, "layer_input")],
        )
        self.assertIs(rows[-1].data, after)

    def test_llama_post_layer_embedding_dump_keeps_distinct_stages(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_llama_stages_") as tmpdir:
            tmp = Path(tmpdir)
            entries = []
            for name, value in (("pe_in-0", 1.0), ("l_out-0", 2.0)):
                stem = f"{name}-token-000003-occ-000"
                (tmp / f"{stem}.bin").write_bytes(np.array([value], dtype=np.float32).tobytes())
                entries.append({
                    "name": stem, "base_name": name, "token_id": 3,
                    "occurrence": 0, "dtype": 0, "rank": 1,
                    "shape": [1], "elem_count": 1, "nbytes": 4,
                })
            (tmp / "index.json").write_text(
                "".join(json.dumps(row) + "\n" for row in entries), encoding="utf-8"
            )
            gemma = decoder_parity_v8._load_llama_dump_dir(
                tmp, post_layer_embedding_layers=frozenset({0})
            )
            ordinary = decoder_parity_v8._load_llama_dump_dir(tmp)

        self.assertEqual([row.op_name for row in gemma],
                         ["layer_out", "gemma4_per_layer_embed"])
        self.assertEqual([row.op_name for row in ordinary], ["pe_in", "layer_out"])

        dump = decoder_parity_v8.parity_test_v7.ParityDump
        ck = [
            dump(0, "layer_out", np.array([1.0], dtype=np.float32), 3, "fp32"),
            dump(0, "gemma4_per_layer_embed", np.array([2.0], dtype=np.float32), 3, "fp32"),
        ]
        aligned = decoder_parity_v8._compare_dump_sets(
            ck, gemma, atol=0.0, rtol=0.0, pass_filter="all"
        )
        wrong_edge = decoder_parity_v8._compare_dump_sets(
            ck, ordinary, atol=0.0, rtol=0.0, pass_filter="all"
        )
        self.assertEqual(aligned["summary"]["pass"], 2)
        self.assertGreater(wrong_edge["summary"]["fail"], 0)

    def test_resolve_llama_dump_names_maps_recurrent_mlp_boundaries(self) -> None:
        self.assertEqual(
            decoder_parity_v8._resolve_llama_dump_names(
                "post_attn_norm-0,mlp_gate-0,mlp_up-0,mlp_swiglu-0,mlp_down-0",
                layer_count=3,
            ),
            "attn_post_norm-0,ffn_gate-0,ffn_up-0,ffn_swiglu-0,ffn_out-0",
        )
        self.assertEqual(
            [
                decoder_parity_v8._canonical_dump_op_name(name)
                for name in (
                    "attn_post_norm", "gate_proj", "up_proj", "ffn_swiglu", "down_proj"
                )
            ],
            ["post_attn_norm", "mlp_gate", "mlp_up", "mlp_swiglu", "mlp_down"],
        )

    def test_resolve_llama_dump_names_maps_full_attention_boundaries(self) -> None:
        self.assertEqual(
            decoder_parity_v8._resolve_llama_dump_names(
                "q_proj-3,v_proj-3,qk_norm_q-3,qk_norm_k-3,"
                "rope_q-3,rope_k-3,attn_gate-3,attn_pregate-3,attn_out-3",
                layer_count=4,
            ),
            "Qcur_full-3,Vcur-3,Qcur_normed-3,Kcur_normed-3,"
            "Qcur-3,Kcur-3,gate_reshaped-3,attn_pregate-3,attn_gated-3",
        )
        self.assertEqual(
            decoder_parity_v8._canonical_dump_op_name("gate_reshaped"),
            "attn_gate",
        )
        self.assertEqual(
            decoder_parity_v8._canonical_dump_op_name("attn_gated"),
            "attn_out",
        )
        self.assertEqual(
            decoder_parity_v8._canonical_dump_op_name("Qcur_full"),
            "q_proj",
        )
        self.assertEqual(
            decoder_parity_v8._canonical_dump_op_name("Qcur_normed"),
            "qk_norm_q",
        )

    def test_ck_attention_head_major_capture_is_canonicalized_token_major(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump(
            3,
            "qk_norm_q",
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
            0,
            "fp32",
        )
        normalized = decoder_parity_v8._normalize_ck_attention_head_major_layout(
            [dump],
            {"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4},
        )
        expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(1, 0, 2)
        np.testing.assert_array_equal(normalized[0].data, expected)

        pregate = decoder_parity_v8.parity_test_v7.ParityDump(
            3,
            "attn_pregate",
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
            0,
            "fp32",
        )
        decoder_parity_v8._normalize_ck_attention_head_major_layout(
            [pregate],
            {"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4},
        )
        np.testing.assert_array_equal(pregate.data, expected)

        qwen_named = decoder_parity_v8.parity_test_v7.ParityDump(
            3,
            "attn_pregate",
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
            0,
            "fp32",
        )
        decoder_parity_v8._normalize_ck_attention_head_major_layout(
            [qwen_named],
            {"num_heads": 2, "num_kv_heads": 1, "head_dim": 4},
        )
        np.testing.assert_array_equal(qwen_named.data, expected)

        variable = decoder_parity_v8.parity_test_v7.ParityDump(
            1,
            "qk_norm_q",
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
            0,
            "fp32",
        )
        decoder_parity_v8._normalize_ck_attention_head_major_layout(
            [variable],
            {
                "num_heads": 2,
                "num_kv_heads": 1,
                "head_dim": 8,
                "layer_q_head_dim": [8, 4],
            },
        )
        np.testing.assert_array_equal(variable.data, expected)

    def test_requested_rope_semantics_disambiguate_llama_kcur_occurrences(self) -> None:
        def dump(name: str) -> object:
            return decoder_parity_v8.parity_test_v7.ParityDump(
                3, "k_proj", np.array([1.0], dtype=np.float32), 0, "fp32",
                source_name=name,
            )

        rows = decoder_parity_v8._apply_requested_oracle_attention_semantics(
            [
                dump("Kcur-3-token-000008-occ-000.bin"),
                dump("Kcur-3-token-000008-occ-001.bin"),
            ],
            {"k_proj", "rope_k"},
        )
        self.assertEqual([row.op_name for row in rows], ["k_proj", "rope_k"])

        loaded_rows = decoder_parity_v8._apply_requested_oracle_attention_semantics(
            [
                decoder_parity_v8.parity_test_v7.ParityDump(
                    3, "kcur_rope", np.array([2.0], dtype=np.float32), 0, "fp32",
                    source_name="Kcur-3-token-000008-occ-002.bin",
                )
            ],
            {"rope_k"},
        )
        self.assertEqual([row.op_name for row in loaded_rows], ["rope_k"])

    def test_requested_rope_q_uses_only_post_rope_occurrence(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        rows = decoder_parity_v8._apply_requested_oracle_attention_semantics(
            [
                dump(3, "q_proj", np.array([1.0], dtype=np.float32), 0, "fp32",
                     source_name="Qcur-3-token-000008-occ-000.bin"),
                dump(3, "qcur_rope", np.array([2.0], dtype=np.float32), 0, "fp32",
                     source_name="Qcur-3-token-000008-occ-002.bin"),
            ],
            {"rope_q"},
        )
        self.assertEqual([row.op_name for row in rows], ["q_proj", "rope_q"])

    def test_requested_v_projection_ignores_llama_tensor_view_occurrence(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        rows = decoder_parity_v8._apply_requested_oracle_attention_semantics(
            [
                dump(3, "v_proj", np.array([1.0], dtype=np.float32), 0, "fp32",
                     source_name="Vcur-3-token-000008-occ-000.bin"),
                dump(3, "v_proj", np.array([1.0], dtype=np.float32), 0, "fp32",
                     source_name="Vcur-3-token-000008-occ-001.bin"),
            ],
            {"v_proj"},
        )
        self.assertEqual([row.op_name for row in rows], ["v_proj", "v_proj_view"])

    def test_resolve_llama_dump_names_maps_recurrent_internal_boundaries(self) -> None:
        self.assertEqual(
            decoder_parity_v8._resolve_llama_dump_names(
                "v_predelta-4,conv_output_raw-4,attn_output-4,new_state-4,"
                "final_output-4,linear_attn_out-4",
                layer_count=64,
            ),
            "v_conv_predelta-4,conv_output_raw-4,attn_output-4,new_state-4,"
            "final_output-4,linear_attn_out-4",
        )

    def test_augment_llama_layer_input_aliases_preserves_shared_edge_identity(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        embedded = np.array([1.0, 2.0], dtype=np.float32)
        layer_zero_out = np.array([3.0, 4.0], dtype=np.float32)
        augmented = decoder_parity_v8._augment_layer_input_aliases(
            [
                dump(-1, "model.input_embed", embedded, 7, "fp32"),
                dump(0, "layer_out", layer_zero_out, 7, "fp32"),
            ],
            layer_count=2,
        )

        self.assertEqual(
            [(row.layer_id, row.op_name) for row in augmented],
            [(0, "layer_input"), (0, "layer_out"), (1, "layer_input")],
        )
        self.assertIs(augmented[0].data, embedded)
        self.assertIs(augmented[2].data, layer_zero_out)

    def test_augment_semantic_after_attn_uses_ck_ffn_input_checkpoint(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        augmented = decoder_parity_v8._augment_layer_input_aliases(
            [dump(3, "ffn_inp", np.array([1.0], dtype=np.float32), 8, "fp32")],
            layer_count=4,
            alias_after_attn=True,
        )
        self.assertEqual([(row.layer_id, row.op_name) for row in augmented], [(3, "after_attn")])

    def test_load_llama_dump_dir_parses_jsonl_index(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_llama_dump_") as tmpdir:
            tmp = Path(tmpdir)
            raw = np.array([1.0, -2.0, 3.5, 4.0], dtype=np.float32)
            bin_name = "Qcur-0-token-000001-occ-000"
            (tmp / f"{bin_name}.bin").write_bytes(raw.tobytes())
            (tmp / "index.json").write_text(
                json.dumps(
                    {
                        "name": bin_name,
                        "base_name": "Qcur-0",
                        "token_id": 1,
                        "occurrence": 0,
                        "dtype": 0,
                        "rank": 2,
                        "shape": [2, 2, 1, 1],
                        "elem_count": 4,
                        "nbytes": 16,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dumps = decoder_parity_v8._load_llama_dump_dir(tmp)

            self.assertEqual(len(dumps), 1)
            self.assertEqual(dumps[0].layer_id, 0)
            self.assertEqual(dumps[0].op_name, "q_proj")
            self.assertEqual(dumps[0].token_id, 1)
            np.testing.assert_allclose(dumps[0].data, raw.reshape(2, 2))

    def test_load_llama_dump_dir_removes_physical_stride_padding(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_llama_strided_dump_") as tmpdir:
            tmp = Path(tmpdir)
            name = "v_conv_predelta-1-token-000008-occ-000"
            # Logical ggml shape [2, 2] with dimension zero contiguous and
            # one padded float between rows: physical [1, 2, pad, 3, 4].
            physical = np.array([1.0, 2.0, 99.0, 3.0, 4.0], dtype=np.float32)
            (tmp / f"{name}.bin").write_bytes(physical.tobytes())
            row = {
                "name": name,
                "base_name": "v_conv_predelta-1",
                "token_id": 8,
                "occurrence": 0,
                "dtype": 0,
                "rank": 2,
                "shape": [2, 2, 1, 1],
                "elem_count": 4,
                "nbytes": 20,
            }
            (tmp / "index.json").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (tmp / f"{name}.json").write_text(
                json.dumps({**row, "type": 0, "ne": [2, 2], "nb": [4, 12]}),
                encoding="utf-8",
            )

            dumps = decoder_parity_v8._load_llama_dump_dir(tmp)

            self.assertEqual(len(dumps), 1)
            np.testing.assert_array_equal(
                np.asarray(dumps[0].data).reshape(-1),
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            )

    def test_load_llama_dump_dir_labels_post_rope_occurrence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_llama_rope_dump_") as tmpdir:
            tmp = Path(tmpdir)
            rows = []
            for occurrence in (0, 1, 2):
                name = f"Qcur-0-token-000001-occ-{occurrence:03d}"
                raw = np.full(8, float(occurrence), dtype=np.float32)
                (tmp / f"{name}.bin").write_bytes(raw.tobytes())
                rows.append(
                    {
                        "name": name,
                        "base_name": "Qcur-0",
                        "token_id": 1,
                        "occurrence": occurrence,
                        "dtype": 0,
                        "rank": 2,
                        "shape": [4, 2, 1, 1],
                        "elem_count": 8,
                        "nbytes": 32,
                    }
                )
            (tmp / "index.json").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            dumps = decoder_parity_v8._load_llama_dump_dir(tmp)

            self.assertEqual(
                [dump.op_name for dump in dumps],
                ["q_proj", "qcur_rope"],
            )

            # Occurrence 1 is an ambiguous graph alias in current llama.cpp,
            # not a stable post-QK-normalization checkpoint. X-ray must omit
            # it rather than manufacture a semantic identity.
            self.assertEqual([dump.source_name for dump in dumps], [rows[0]["name"], rows[2]["name"]])


    def test_compare_dump_sets_reports_failures(self) -> None:
        ck_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "Qcur",
                np.array([1.0, 2.0], dtype=np.float32),
                1,
                "fp32",
            )
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "Qcur",
                np.array([1.0, 2.5], dtype=np.float32),
                1,
                "fp32",
            )
        ]

        report = decoder_parity_v8._compare_dump_sets(
            ck_dumps,
            llama_dumps,
            atol=1.0e-4,
            rtol=1.0e-3,
            pass_filter="decode",
        )

        self.assertEqual(report["summary"]["fail"], 1)
        self.assertEqual(report["first_issue"]["op"], "Qcur")
        self.assertEqual(report["first_issue"]["status"], "FAIL")

    def test_compare_dump_sets_preserves_graph_order_for_first_issue(self) -> None:
        def dump(op_name: str, value: float):
            return decoder_parity_v8.parity_test_v7.ParityDump(
                0, op_name, np.array([value], dtype=np.float32), 0, "fp32"
            )

        report = decoder_parity_v8._compare_dump_sets(
            [dump("ffn_inp", 2.0), dump("down_proj", 4.0)],
            [dump("ffn_inp", 1.0), dump("down_proj", 3.0)],
            atol=1.0e-4,
            rtol=1.0e-3,
            pass_filter="decode",
        )

        self.assertEqual([row["op"] for row in report["results"]], ["ffn_inp", "mlp_down"])
        self.assertEqual(report["first_issue"]["op"], "ffn_inp")

    def test_compare_dump_sets_rejects_distinct_ambiguous_occurrences(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        report = decoder_parity_v8._compare_dump_sets(
            [dump(0, "v_proj", np.array([1.0], dtype=np.float32), 7, "fp32")],
            [
                dump(0, "v_proj", np.array([1.0], dtype=np.float32), 7, "fp32"),
                dump(0, "v_proj", np.array([2.0], dtype=np.float32), 7, "fp32"),
            ],
            atol=0.0,
            rtol=0.0,
            pass_filter="decode",
        )

        self.assertEqual(report["summary"]["error"], 1)
        self.assertEqual(report["first_issue"]["reason"], "ambiguous_alignment")

    def test_compare_dump_sets_accepts_identical_reshape_occurrences(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        values = np.array([1.0, 2.0], dtype=np.float32)
        report = decoder_parity_v8._compare_dump_sets(
            [dump(0, "v_proj", values.copy(), 7, "fp32")],
            [
                dump(0, "v_proj", values.copy(), 7, "fp32"),
                dump(0, "v_proj", values.reshape(2, 1).copy(), 7, "fp32"),
            ],
            atol=0.0,
            rtol=0.0,
            pass_filter="decode",
        )

        self.assertEqual(report["summary"]["pass"], 1)
        self.assertFalse(report["results"][0]["alignment_ambiguous"])

    def test_expand_ck_prefill_decode_dumps_splits_prompt_rows(self) -> None:
        ck_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.arange(24, dtype=np.float32),
                0,
                "fp32",
            ),
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "kqv_out",
                np.arange(100, 124, dtype=np.float32),
                0,
                "fp32",
            ),
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.zeros(6, dtype=np.float32),
                0,
                "fp32",
            ),
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "kqv_out",
                np.zeros(6, dtype=np.float32),
                0,
                "fp32",
            ),
        ]

        expanded = decoder_parity_v8._expand_ck_prefill_decode_dumps(
            ck_dumps,
            llama_dumps,
            prompt_start_token=0,
            prompt_token_count=2,
        )

        q_rows = [d for d in expanded if d.layer_id == 0 and d.op_name == "q_proj"]
        self.assertEqual([d.token_id for d in q_rows], [0, 1])
        np.testing.assert_allclose(q_rows[0].data, np.arange(12, 18, dtype=np.float32))
        np.testing.assert_allclose(q_rows[1].data, np.arange(18, 24, dtype=np.float32))

        attn_rows = [d for d in expanded if d.layer_id == 0 and d.op_name == "kqv_out"]
        self.assertEqual([d.token_id for d in attn_rows], [0, 1])
        np.testing.assert_allclose(attn_rows[0].data, np.arange(112, 118, dtype=np.float32))
        np.testing.assert_allclose(attn_rows[1].data, np.arange(118, 124, dtype=np.float32))

    def test_expand_ck_prefill_decode_dumps_selects_exact_trailing_segment(self) -> None:
        row_elems = 4

        def ck_segment(rows: int, base: float):
            data = np.arange(rows * row_elems, dtype=np.float32) + base
            return decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                data,
                1083,
                "fp32",
            )

        # Reproduces Qwen3-VL segmented mixed prefill: text-before, visual,
        # then the requested text-after/generated-token replay window.
        ck_dumps = [
            ck_segment(5, 1000.0),
            ck_segment(1008, 2000.0),
            ck_segment(71, 3000.0),
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.zeros(row_elems, dtype=np.float32),
                token_id,
                "fp32",
            )
            for token_id in range(71)
        ]

        expanded = decoder_parity_v8._expand_ck_prefill_decode_dumps(
            ck_dumps,
            llama_dumps,
            prompt_start_token=1013,
            prompt_token_count=71,
        )

        rows = [d for d in expanded if d.layer_id == 0 and d.op_name == "q_proj"]
        self.assertEqual([d.token_id for d in rows], list(range(71)))
        self.assertTrue(all(d.data.size == row_elems for d in rows))
        np.testing.assert_allclose(rows[0].data, np.arange(4, dtype=np.float32) + 3000.0)
        np.testing.assert_allclose(rows[-1].data, np.arange(280, 284, dtype=np.float32) + 3000.0)

    def test_expand_ck_prefill_decode_dumps_extracts_head_major_norm_rows(self) -> None:
        # CK stores qk_norm_q head-major as [heads, tokens, dim].
        # The parity comparator flattens both sides, so expansion must preserve
        # CK's native flat order rather than transposing into dim-major form.
        ck_tensor = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        ck_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "qk_norm_q",
                ck_tensor.reshape(-1),
                0,
                "fp32",
            )
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "qk_norm_q",
                np.zeros((4, 2), dtype=np.float32),
                0,
                "fp32",
            )
        ]

        expanded = decoder_parity_v8._expand_ck_prefill_decode_dumps(
            ck_dumps,
            llama_dumps,
            prompt_start_token=0,
            prompt_token_count=2,
        )

        rows = [d for d in expanded if d.layer_id == 0 and d.op_name == "qk_norm_q"]
        self.assertEqual([d.token_id for d in rows], [0, 1])
        np.testing.assert_allclose(rows[0].data, ck_tensor[:, 1, :])
        np.testing.assert_allclose(rows[1].data, ck_tensor[:, 2, :])

    def test_expand_ck_prefill_decode_dumps_extracts_token_major_rope_rows(self) -> None:
        # The dedicated CK post-RoPE dumper writes [tokens, heads, dim], unlike
        # the raw qcur_normed scratch tensor above.
        ck_tensor = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
        ck_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "qcur_rope",
                ck_tensor.reshape(-1),
                0,
                "fp32",
            )
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "qcur_rope",
                np.zeros((4, 2), dtype=np.float32),
                0,
                "fp32",
            )
        ]

        expanded = decoder_parity_v8._expand_ck_prefill_decode_dumps(
            ck_dumps,
            llama_dumps,
            prompt_start_token=0,
            prompt_token_count=2,
        )

        rows = [d for d in expanded if d.layer_id == 0 and d.op_name == "qcur_rope"]
        self.assertEqual([d.token_id for d in rows], [0, 1])
        np.testing.assert_allclose(rows[0].data, ck_tensor[1])
        np.testing.assert_allclose(rows[1].data, ck_tensor[2])

    def test_build_llama_row_specs_prefers_ranked_norm_shape_on_tie(self) -> None:
        specs = decoder_parity_v8._build_llama_row_specs(
            [
                decoder_parity_v8.parity_test_v7.ParityDump(
                    0,
                    "qk_norm_q",
                    np.zeros(8, dtype=np.float32),
                    0,
                    "fp32",
                ),
                decoder_parity_v8.parity_test_v7.ParityDump(
                    0,
                    "qk_norm_q",
                    np.zeros((4, 2), dtype=np.float32),
                    0,
                    "fp32",
                ),
            ]
        )

        self.assertEqual(specs[(0, "qk_norm_q")], (8, (4, 2)))

    def test_trim_llama_prefill_decode_dumps_preserves_duplicate_occurrences(self) -> None:
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.array([1.0], dtype=np.float32),
                1,
                "fp32",
            ),
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.array([2.0], dtype=np.float32),
                1,
                "fp32",
            ),
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.array([3.0], dtype=np.float32),
                0,
                "fp32",
            ),
        ]

        trimmed = decoder_parity_v8._trim_llama_prefill_decode_dumps(
            llama_dumps,
            prompt_start_token=1,
            prompt_token_count=1,
        )

        q_rows = [d for d in trimmed if d.layer_id == 0 and d.op_name == "q_proj"]
        self.assertEqual(len(q_rows), 2)
        self.assertEqual([d.token_id for d in q_rows], [0, 0])
        np.testing.assert_allclose(q_rows[0].data, np.array([1.0], dtype=np.float32))
        np.testing.assert_allclose(q_rows[1].data, np.array([2.0], dtype=np.float32))

    def test_expand_llama_prefill_decode_dumps_selects_execution_tail_not_largest_position(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        row_elems = 4
        llama_dumps = [
            dump(0, "q_proj", np.arange(5 * row_elems, dtype=np.float32), 4, "fp32"),
            dump(0, "q_proj", np.arange(1008 * row_elems, dtype=np.float32), 1012, "fp32"),
            dump(
                0,
                "q_proj",
                np.arange(59 * row_elems, dtype=np.float32) + 10000.0,
                99,
                "fp32",
            ),
        ]

        expanded = decoder_parity_v8._expand_llama_prefill_decode_dumps(
            llama_dumps,
            prompt_token_count=59,
        )

        self.assertEqual([item.token_id for item in expanded], list(range(59)))
        self.assertTrue(all(item.data.size == row_elems for item in expanded))
        np.testing.assert_allclose(
            expanded[-1].data,
            np.arange(232, 236, dtype=np.float32) + 10000.0,
        )

    def test_resolve_decode_prompt_start_tokens_uses_stage_and_rope_windows(self) -> None:
        ck_start, llama_start = decoder_parity_v8._resolve_decode_prompt_start_tokens(
            tokens_before_count=4,
            prefix_tokens=9,
            prefix_text_pos=7,
            llama_meta={"prefix_text_pos": 7},
        )

        self.assertEqual(ck_start, 13)
        self.assertEqual(llama_start, 7)

    def test_build_multimodal_position_contract_matches_qwen3vl_grid_contract(self) -> None:
        contract = decoder_parity_v8._build_multimodal_position_contract(
            tokens_before_count=4,
            prefix_tokens=9,
            prefix_grid=(3, 3),
            prefix_text_pos=7,
            llama_meta={"prefix_start_pos": 4, "prefix_text_pos": 7},
        )

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract["rows_match"])
        self.assertTrue(contract["text_pos_match"])
        self.assertEqual(contract["ck"]["rows"][0], [4, 4, 4, 0])
        self.assertEqual(contract["ck"]["rows"][1], [4, 4, 5, 0])
        self.assertEqual(contract["ck"]["rows"][3], [4, 5, 4, 0])
        self.assertEqual(contract["llama"]["rows"], contract["ck"]["rows"])

    def test_compare_dump_sets_keeps_native_kqv_out_distinct_from_attn_output(self) -> None:
        ck_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "kqv_out",
                np.array([1.0, 2.0], dtype=np.float32),
                0,
                "fp32",
            ),
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "attn_output",
                np.array([9.0, 9.0], dtype=np.float32),
                0,
                "fp32",
            ),
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "kqv_out",
                np.array([1.0, 2.0], dtype=np.float32),
                0,
                "fp32",
            )
        ]

        report = decoder_parity_v8._compare_dump_sets(
            ck_dumps,
            llama_dumps,
            atol=1.0e-4,
            rtol=1.0e-3,
            pass_filter="decode",
        )

        self.assertEqual(report["summary"]["pass"], 1)
        self.assertEqual(report["summary"]["warn"], 1)
        self.assertIsNone(report["first_issue"])
        by_op = {row["op"]: row for row in report["results"]}
        self.assertEqual(by_op["kqv_out"]["status"], "PASS")
        self.assertEqual(by_op["attn_output"]["status"], "WARN")

    def test_compare_dump_sets_legacy_attn_output_falls_back_to_kqv_out(self) -> None:
        ck_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "attn_output",
                np.array([1.0, 2.0], dtype=np.float32),
                0,
                "fp32",
            )
        ]
        llama_dumps = [
            decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "kqv_out",
                np.array([1.0, 2.0], dtype=np.float32),
                0,
                "fp32",
            )
        ]

        report = decoder_parity_v8._compare_dump_sets(
            ck_dumps,
            llama_dumps,
            atol=1.0e-4,
            rtol=1.0e-3,
            pass_filter="decode",
        )

        self.assertEqual(report["summary"]["pass"], 1)
        self.assertEqual(report["summary"]["warn"], 1)
        self.assertIsNone(report["first_issue"])
        by_op = {row["op"]: row for row in report["results"]}
        self.assertEqual(by_op["kqv_out"]["status"], "PASS")
        self.assertEqual(by_op["attn_output"]["status"], "WARN")

    def test_report_passes_when_top1_and_overlap_match(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_parity_pass_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            fake_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    self.last_text = text
                    return [11, 22]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}), \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 0,
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
                         },
                         "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
                     },
                 ):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--gguf",
                            str(fake_gguf),
                            "--workdir",
                            str(tmp / "work"),
                            "--prompt",
                            "Hello",
                            "--top-k",
                            "2",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["pass"])
            self.assertEqual(report["tokens"], [11, 22])
            self.assertEqual(report["prefix"]["source"], "none")
            self.assertEqual(report["compare"]["top1_ck"], 1)
            self.assertEqual(report["compare"]["top1_llama"], 1)
            self.assertTrue(report["compare"]["top1_match"])
            self.assertGreaterEqual(report["compare"]["topk_overlap_ratio"], 0.5)

    def test_report_fails_when_top1_mismatches(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_parity_fail_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            fake_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [7, 8]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return "|".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.7, 0.1, 0.2, -0.4])}), \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 0,
                             "topk": [{"id": 2, "logit": 0.8}, {"id": 0, "logit": 0.2}],
                         },
                         "logits": np.array([0.1, 0.2, 0.8, -0.1], dtype=np.float32),
                     },
                 ):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--gguf",
                            str(fake_gguf),
                            "--workdir",
                            str(tmp / "work"),
                            "--prompt",
                            "Hello",
                            "--top-k",
                            "2",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 3)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["pass"])
            self.assertFalse(report["compare"]["top1_match"])
            self.assertEqual(report["compare"]["top1_ck"], 0)
            self.assertEqual(report["compare"]["top1_llama"], 2)

    def test_report_replays_prefix_file_on_llama_side(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_parity_prefix_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            fake_gguf = tmp / "decoder.gguf"
            prefix_path = tmp / "prefix.f32"
            prefix = array("f", [0.0] * (3 * 16))
            prefix_path.write_bytes(prefix.tobytes())

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [101, 202]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}), \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 3,
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
                         },
                         "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
                     },
                 ) as llama_capture:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--gguf",
                            str(fake_gguf),
                            "--workdir",
                            str(tmp / "work"),
                            "--prompt",
                            "Hello",
                            "--prefix-f32",
                            str(prefix_path),
                            "--top-k",
                            "2",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            _, kwargs = llama_capture.call_args
            self.assertEqual(kwargs["prefix_path"], prefix_path.resolve())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["prefix"]["source"], "file")
            self.assertEqual(report["prefix"]["tokens"], 3)

    def test_report_replays_synthetic_prefix_on_llama_side(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_parity_synth_prefix_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            fake_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [101, 202]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}), \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 3,
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
                         },
                         "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
                     },
                 ) as llama_capture:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--gguf",
                            str(fake_gguf),
                            "--workdir",
                            str(tmp / "work"),
                            "--prompt",
                            "Hello",
                            "--synthetic-prefix-tokens",
                            "3",
                            "--top-k",
                            "2",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            _, kwargs = llama_capture.call_args
            prefix_path = Path(kwargs["prefix_path"])
            self.assertTrue(prefix_path.exists())
            self.assertEqual(kwargs["prefix_row_dim"], 64)
            self.assertEqual(prefix_path.stat().st_size, 3 * 64 * 4)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["prefix"]["source"], "synthetic_zero")
            self.assertEqual(report["prefix"]["tokens"], 3)
            self.assertEqual(report["prefix"]["row_dim"], 64)

    def test_main_passes_explicit_prefix_grid_through_replay(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_parity_explicit_grid_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            prefix_path = tmp / "prefix.f32"
            prefix_path.write_bytes(array("f", [0.0] * (6 * 64)).tobytes())
            fake_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [101, 202]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}) as run_decoder, \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 6,
                             "prefix_position_count": 3,
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
                         },
                         "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
                     },
                 ) as llama_capture:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--gguf",
                            str(fake_gguf),
                            "--workdir",
                            str(tmp / "work"),
                            "--prompt",
                            "Hello",
                            "--prefix-f32",
                            str(prefix_path),
                            "--prefix-row-dim",
                            "64",
                            "--prefix-grid-x",
                            "2",
                            "--prefix-grid-y",
                            "3",
                            "--prefix-text-pos",
                            "7",
                            "--top-k",
                            "2",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            _, llama_kwargs = llama_capture.call_args
            self.assertEqual(llama_kwargs["prefix_grid"], (2, 3))
            _, decoder_kwargs = run_decoder.call_args
            self.assertEqual(decoder_kwargs["prefix_grid"], (2, 3))
            self.assertEqual(decoder_kwargs["prefix_text_pos"], 7)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["prefix"]["grid"], [2, 3])
            self.assertEqual(report["prefix"]["text_pos"], 7)

    def test_main_separate_prefixes_are_labeled_and_geometry_checked(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_own_prefix_") as tmpdir:
            tmp = Path(tmpdir)
            ck_prefix = tmp / "ck.f32"
            llama_prefix = tmp / "llama.f32"
            ck_prefix.write_bytes(array("f", [1.0, 2.0, 3.0, 4.0]).tobytes())
            llama_prefix.write_bytes(array("f", [4.0, 3.0, 2.0, 1.0]).tobytes())
            report_path = tmp / "report.json"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [1, 2]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            runtime = {
                "embed_dim": 2,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }
            llama_result = {
                "meta": {"ok": True, "n_vocab": 4, "token_count": 2,
                         "prefix_token_count": 2, "topk": [{"id": 1, "logit": 0.95}]},
                "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
            }
            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}) as run_decoder, \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(decoder_parity_v8, "_run_llama_capture", return_value=llama_result) as run_llama:
                argv = [
                    "--gguf", str(tmp / "decoder.gguf"), "--workdir", str(tmp / "work"),
                    "--tokens", "1,2", "--prefix-f32", str(ck_prefix),
                    "--llama-prefix-f32", str(llama_prefix), "--prefix-row-dim", "2",
                    "--json-out", str(report_path),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(decoder_parity_v8.main(argv), 4)
                self.assertEqual(run_llama.call_args.kwargs["prefix_path"], llama_prefix)
                self.assertEqual(list(run_decoder.call_args.args[1]), [1.0, 2.0, 3.0, 4.0])
                self.assertEqual(run_decoder.call_args.kwargs["prefix_decode_policy"], "causal_mixed_prefix")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "incomplete")
                self.assertEqual(report["prefix_decode_policy"], "causal_mixed_prefix")
                self.assertFalse(report["pass"])
                self.assertTrue(report["diagnostic_comparison_pass"])
                self.assertEqual(report["prefix_input_scope"], "separate_files_unverified_producers")
                self.assertNotEqual(
                    report["prefix_input_identity"]["ck_sha256"],
                    report["prefix_input_identity"]["llama_sha256"],
                )
                verified_argv = [*argv, "--encoder-prefix-report", str(tmp / "producer.json")]
                with mock.patch.object(
                    decoder_parity_v8, "_verify_encoder_prefix_report",
                    return_value={"input_provenance": "independently_decoded_and_preprocessed_from_same_p6"},
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(decoder_parity_v8.main(verified_argv), 4)
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(decoder_parity_v8.main([
                            *verified_argv, "--max-abs-threshold", "0.2",
                            "--max-rmse-threshold", "0.2",
                        ]), 0)
                    verified_report = json.loads(report_path.read_text(encoding="utf-8"))
                    self.assertEqual(verified_report["status"], "pass")
                    self.assertEqual(verified_report["comparison_scope"], "independent_image_to_first_logit")
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(decoder_parity_v8.main([
                            *verified_argv, "--max-abs-threshold", "0.01",
                            "--max-rmse-threshold", "0.01",
                        ]), 3)
                    self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["status"], "fail")
                llama_prefix.write_bytes(array("f", [1.0, 2.0]).tobytes())
                run_llama.reset_mock()
                run_decoder.reset_mock()
                with self.assertRaisesRegex(ValueError, "separate prefix geometry mismatch"):
                    decoder_parity_v8.main(argv)
                run_llama.assert_not_called()
                run_decoder.assert_not_called()
                with self.assertRaisesRegex(ValueError, "different files"):
                    decoder_parity_v8.main([
                        *argv[:argv.index("--llama-prefix-f32") + 1], str(ck_prefix),
                        *argv[argv.index("--prefix-row-dim"):],
                    ])
                llama_prefix.write_bytes(array("f", [4.0, 3.0, 2.0, 1.0]).tobytes())

                def change_oracle_prefix(*_args, **_kwargs):
                    llama_prefix.write_bytes(array("f", [4.0, 3.0, 2.0, 1.1]).tobytes())
                    return llama_result

                run_llama.side_effect = change_oracle_prefix
                with self.assertRaisesRegex(RuntimeError, "changed during capture"):
                    decoder_parity_v8.main(argv)
                run_decoder.assert_not_called()
                llama_prefix.write_bytes(array("f", [4.0, 3.0, 2.0, 1.0]).tobytes())
                run_llama.side_effect = None
                run_llama.return_value = {
                    **llama_result,
                    "logits": np.array([2.0, 0.0, 0.1, -0.5], dtype=np.float32),
                }
                with mock.patch.object(
                    decoder_parity_v8, "_verify_encoder_prefix_report",
                    return_value={"input_provenance": "independently_decoded_and_preprocessed_from_same_p6"},
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(decoder_parity_v8.main(verified_argv), 4)
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(decoder_parity_v8.main([
                            *verified_argv, "--max-abs-threshold", "0.2",
                            "--max-rmse-threshold", "0.2",
                        ]), 3)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(decoder_parity_v8.main(argv), 4)
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "incomplete")
                self.assertFalse(report["diagnostic_comparison_pass"])

    def test_main_passes_ctx_len_into_decoder_runtime_prep(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_ctx_len_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            fake_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [1, 2]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime) as prepare_runtime, \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}), \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 0,
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
                         },
                         "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
                     },
                 ):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--gguf",
                            str(fake_gguf),
                            "--workdir",
                            str(tmp / "work"),
                            "--prompt",
                            "Hello",
                            "--ctx-len",
                            "123",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            _, kwargs = prepare_runtime.call_args
            self.assertEqual(kwargs["context_override"], 123)


    def test_main_auto_bumps_ctx_len_for_prefix_budget(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_ctx_autobump_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            fake_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [1, 2]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 16,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime) as prepare_runtime,                  mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}),                  mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()),                  mock.patch.object(decoder_parity_v8, "_load_prefix_embeddings", return_value=(array("f", [0.0] * 3 * 16), 3, 16, "synthetic_zero")),                  mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 2,
                             "prefix_token_count": 3,
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
                         },
                         "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
                     },
                 ) as llama_capture:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--gguf",
                            str(fake_gguf),
                            "--workdir",
                            str(tmp / "work"),
                            "--prompt",
                            "Hello",
                            "--ctx-len",
                            "1",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            self.assertEqual(prepare_runtime.call_count, 2)
            self.assertEqual(prepare_runtime.call_args_list[0].kwargs["context_override"], 1)
            self.assertEqual(prepare_runtime.call_args_list[1].kwargs["context_override"], 5)
            self.assertEqual(llama_capture.call_args.args[2], 5)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["requested_ctx_len"], 1)
            self.assertEqual(report["ctx_len"], 5)

    def test_capture_dump_compare_replays_prefix_for_decode_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_dump_prefix_") as tmpdir:
            tmp = Path(tmpdir)
            prefix = array("f", [1.0, 2.0, 3.0, 4.0])
            ck_dump = decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.array([1.0, 2.0], dtype=np.float32),
                0,
                "fp32",
            )
            llama_dump = decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.array([1.0, 2.0], dtype=np.float32),
                0,
                "fp32",
            )
            with mock.patch.object(
                decoder_parity_v8,
                "_run_llama_capture",
                return_value={"meta": {"decode_mode": "sequential", "dumped": 1}, "logits": np.array([0.0], dtype=np.float32)},
            ) as llama_capture, \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_capture_ck_dump",
                     return_value={"vocab_size": 1, "logits": array("f", [0.0])},
                 ), \
                 mock.patch.object(decoder_parity_v8.parity_test_v7, "read_dump_file", return_value=[ck_dump]), \
                 mock.patch.object(decoder_parity_v8, "_load_llama_dump_dir", return_value=[llama_dump]):
                ck, report = decoder_parity_v8._capture_dump_compare(
                    Path("/tmp/model.gguf"),
                    {"embed_dim": 2},
                    prefix,
                    2,
                    [11, 22],
                    prefix_row_dim=2,
                    ctx_len=6,
                    top_k=2,
                    threads=1,
                    dump_root=tmp,
                    dump_names="Qcur-0",
                    dump_pass="decode",
                    dump_atol=1.0e-4,
                    dump_rtol=1.0e-3,
                )

            self.assertEqual(ck["vocab_size"], 1)
            _, kwargs = llama_capture.call_args
            self.assertEqual(kwargs["decode_mode"], "sequential")
            self.assertTrue(Path(kwargs["prefix_path"]).exists())
            self.assertEqual(report["status"], "ok")
            self.assertTrue(str(report["prefix_path"]).endswith("prefix.f32"))

    def test_capture_dump_compare_expands_prefill_batch_rows_for_decode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_dump_expand_") as tmpdir:
            tmp = Path(tmpdir)
            prefix = array("f", [1.0, 2.0, 3.0, 4.0])
            ck_dump = decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.arange(8, dtype=np.float32),
                0,
                "fp32",
            )
            llama_dump = decoder_parity_v8.parity_test_v7.ParityDump(
                0,
                "q_proj",
                np.zeros(4, dtype=np.float32),
                0,
                "fp32",
            )

            captured: dict[str, object] = {}

            def _fake_compare(ck_dumps, llama_dumps, *, atol, rtol, pass_filter):
                captured["ck_dumps"] = ck_dumps
                return {"summary": {"total": 1, "pass": 1, "fail": 0, "error": 0, "warn": 0, "missing": 0}, "first_issue": None, "results": []}

            with mock.patch.object(
                decoder_parity_v8,
                "_run_llama_capture",
                return_value={"meta": {"decode_mode": "sequential", "dumped": 1}, "logits": np.array([0.0], dtype=np.float32)},
            ), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_capture_ck_dump",
                     return_value={"vocab_size": 1, "logits": array("f", [0.0])},
                 ), \
                 mock.patch.object(decoder_parity_v8.parity_test_v7, "read_dump_file", return_value=[ck_dump]), \
                 mock.patch.object(decoder_parity_v8, "_load_llama_dump_dir", return_value=[llama_dump]), \
                 mock.patch.object(decoder_parity_v8, "_compare_dump_sets", side_effect=_fake_compare):
                _, report = decoder_parity_v8._capture_dump_compare(
                    Path("/tmp/model.gguf"),
                    {"embed_dim": 2},
                    prefix,
                    2,
                    [11, 22],
                    prefix_row_dim=2,
                    ctx_len=6,
                    top_k=2,
                    threads=1,
                    dump_root=tmp,
                    dump_names="Qcur-0",
                    dump_pass="decode",
                    dump_atol=1.0e-4,
                    dump_rtol=1.0e-3,
                )

            self.assertEqual(report["status"], "ok")
            ck_rows = captured["ck_dumps"]
            self.assertIsInstance(ck_rows, list)
            self.assertEqual([d.token_id for d in ck_rows], [0, 1])
            np.testing.assert_allclose(ck_rows[0].data, np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
            np.testing.assert_allclose(ck_rows[1].data, np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))

    def test_capture_dump_compare_rebases_segmented_prompt_window(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_dump_segmented_") as tmpdir:
            tmp = Path(tmpdir)
            prefix = array("f", [1.0, 2.0, 3.0, 4.0])
            ck_dumps = [
                decoder_parity_v8.parity_test_v7.ParityDump(
                    0,
                    "q_proj",
                    np.array([1.0, 2.0], dtype=np.float32),
                    11,
                    "fp32",
                ),
                decoder_parity_v8.parity_test_v7.ParityDump(
                    0,
                    "q_proj",
                    np.array([3.0, 4.0], dtype=np.float32),
                    12,
                    "fp32",
                ),
            ]
            llama_dumps = [
                decoder_parity_v8.parity_test_v7.ParityDump(
                    0,
                    "q_proj",
                    np.array([1.0, 2.0], dtype=np.float32),
                    5,
                    "fp32",
                ),
                decoder_parity_v8.parity_test_v7.ParityDump(
                    0,
                    "q_proj",
                    np.array([3.0, 4.0], dtype=np.float32),
                    6,
                    "fp32",
                ),
            ]

            captured: dict[str, object] = {}

            def _fake_compare(ck_rows, llama_rows, *, atol, rtol, pass_filter):
                captured["ck_rows"] = ck_rows
                captured["llama_rows"] = llama_rows
                return {"summary": {"total": 1, "pass": 1, "fail": 0, "error": 0, "warn": 0, "missing": 0}, "first_issue": None, "results": []}

            with mock.patch.object(
                decoder_parity_v8,
                "_run_llama_capture",
                return_value={
                    "meta": {"decode_mode": "sequential", "dumped": 2, "prefix_text_pos": 5},
                    "logits": np.array([0.0], dtype=np.float32),
                },
            ), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_capture_ck_dump",
                     return_value={"vocab_size": 1, "logits": array("f", [0.0])},
                 ), \
                 mock.patch.object(decoder_parity_v8.parity_test_v7, "read_dump_file", return_value=ck_dumps), \
                 mock.patch.object(decoder_parity_v8, "_load_llama_dump_dir", return_value=llama_dumps), \
                 mock.patch.object(decoder_parity_v8, "_compare_dump_sets", side_effect=_fake_compare):
                _, report = decoder_parity_v8._capture_dump_compare(
                    Path("/tmp/model.gguf"),
                    {"embed_dim": 2},
                    prefix,
                    9,
                    [33, 44],
                    tokens_before=[11, 22],
                    prefix_row_dim=2,
                    ctx_len=16,
                    top_k=2,
                    threads=1,
                    dump_root=tmp,
                    dump_names="Qcur-0",
                    dump_pass="decode",
                    dump_atol=1.0e-4,
                    dump_rtol=1.0e-3,
                    prefix_grid=(3, 3),
                    prefix_text_pos=5,
                )

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["ck_prompt_start_token"], 11)
            self.assertEqual(report["llama_prompt_start_token"], 5)
            ck_rows = captured["ck_rows"]
            llama_rows = captured["llama_rows"]
            self.assertIsInstance(ck_rows, list)
            self.assertIsInstance(llama_rows, list)
            self.assertEqual([d.token_id for d in ck_rows], [0, 1])
            self.assertEqual([d.token_id for d in llama_rows], [0, 1])

    def test_capture_dump_compare_runs_segmented_multimodal_prefill_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_dump_prefix_skip_") as tmpdir:
            tmp = Path(tmpdir)
            prefix = array("f", [0.0] * 8)
            dump = decoder_parity_v8.parity_test_v7.ParityDump
            ck_dumps = [
                dump(0, "q_proj", np.zeros((1, 4), dtype=np.float32), 0, "fp32"),
                dump(0, "q_proj", np.zeros((2, 4), dtype=np.float32), 0, "fp32"),
                dump(0, "q_proj", np.zeros((2, 4), dtype=np.float32), 0, "fp32"),
            ]
            llama_dumps = [
                dump(0, "q_proj", np.zeros((1, 4), dtype=np.float32), 0, "fp32"),
                dump(0, "q_proj", np.zeros((2, 4), dtype=np.float32), 0, "fp32"),
                dump(0, "q_proj", np.zeros((2, 4), dtype=np.float32), 0, "fp32"),
            ]
            with mock.patch.object(
                decoder_parity_v8.bridge_runner_v8,
                "_run_decoder",
                return_value={"vocab_size": 4, "logits": array("f", [0.0, 0.0, 0.0, 0.0])},
            ), mock.patch.object(
                decoder_parity_v8,
                "_run_llama_capture",
                return_value={"meta": {"dumped": 1, "decode_mode": "sequential", "flash_attention_mode": "enabled"}},
            ) as llama_capture, mock.patch.object(
                decoder_parity_v8,
                "_capture_ck_dump",
                return_value={"vocab_size": 4, "logits": array("f", [0.0, 0.0, 0.0, 0.0])},
            ) as ck_capture, mock.patch.object(
                decoder_parity_v8.parity_test_v7,
                "read_dump_file",
                return_value=ck_dumps,
            ), mock.patch.object(
                decoder_parity_v8,
                "_load_llama_dump_dir",
                return_value=llama_dumps,
            ):
                ck, report = decoder_parity_v8._capture_dump_compare(
                    Path("/tmp/model.gguf"),
                    {"embed_dim": 4},
                    prefix,
                    2,
                    [1, 2],
                    tokens_before=[9],
                    prefix_row_dim=4,
                    ctx_len=8,
                    top_k=2,
                    threads=1,
                    dump_root=tmp,
                    dump_names="Qcur-0",
                    dump_pass="prefill",
                    dump_atol=1.0e-4,
                    dump_rtol=1.0e-3,
                    prefix_decode_policy="non_causal_visual_chunk",
                    llama_flash_attention="enabled",
                )

            self.assertEqual(ck["vocab_size"], 4)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["comparison_pass_filter"], "all")
            self.assertEqual(report["prefill_segments"], [
                {"name": "text_before", "rows": 1, "physical_start": 0},
                {"name": "visual", "rows": 2, "physical_start": 1},
                {"name": "text_after", "rows": 2, "physical_start": 3},
            ])
            ck_capture.assert_called_once()
            self.assertEqual(ck_capture.call_args.kwargs["prefix_decode_policy"], "non_causal_visual_chunk")
            llama_capture.assert_called_once()
            self.assertEqual(llama_capture.call_args.kwargs["flash_attention"], "enabled")

    def test_coalesce_multimodal_prefill_segments_joins_token_rows(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        segmented = [
            dump(4, "qkv", np.full((1, 2), 1.0, dtype=np.float32), 8, "fp32"),
            dump(4, "qkv", np.full((3, 2), 2.0, dtype=np.float32), 1016, "fp32"),
            dump(4, "qkv", np.full((2, 2), 3.0, dtype=np.float32), 62, "fp32"),
        ]
        row_specs = {(4, "qkv"): (2, (2,))}
        segments = [("text_before", 1, 0), ("visual", 3, 1), ("text_after", 2, 4)]

        merged = decoder_parity_v8._coalesce_multimodal_prefill_segments(
            segmented, row_specs, segments
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].op_name, "qkv")
        np.testing.assert_array_equal(
            merged[0].data,
            np.array(
                [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [3.0, 3.0], [3.0, 3.0]],
                dtype=np.float32,
            ),
        )

    def test_coalesce_ignores_deceptive_ggml_axis_order(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        segmented = [
            dump(0, "qk_norm_q", np.arange(4, dtype=np.float32).reshape(2, 2, 1), 0, "fp32"),
            dump(0, "qk_norm_q", np.arange(4, 12, dtype=np.float32).reshape(2, 2, 2), 1, "fp32"),
        ]
        merged = decoder_parity_v8._coalesce_multimodal_prefill_segments(
            segmented,
            {(0, "qk_norm_q"): (4, (2, 2))},
            [("text", 1, 0), ("visual", 2, 1)],
        )
        np.testing.assert_array_equal(
            merged[0].data,
            np.arange(12, dtype=np.float32).reshape(3, 2, 2),
        )

    def test_coalesce_multimodal_prefill_with_sequential_text_tokens(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        captures = [
            dump(0, "mlp_down", np.arange(10, dtype=np.float32), 4, "fp32"),
            dump(0, "mlp_down", np.arange(10, 154, dtype=np.float32), 76, "fp32"),
        ]
        captures.extend(
            dump(0, "mlp_down", np.arange(154 + 2 * i, 156 + 2 * i, dtype=np.float32), 77 + i, "fp32")
            for i in range(15)
        )
        segments = [("text_before", 5, 0), ("visual", 72, 5), ("text_after", 15, 77)]
        row_specs = {(0, "mlp_down"): (2, (2,))}

        merged = decoder_parity_v8._coalesce_multimodal_prefill_segments(
            captures, row_specs, segments
        )
        self.assertEqual(len(merged), 1)
        np.testing.assert_array_equal(
            merged[0].data, np.arange(184, dtype=np.float32).reshape(92, 2)
        )

        captures[4].token_id = 99
        unresolved = decoder_parity_v8._coalesce_multimodal_prefill_segments(
            captures, row_specs, segments
        )
        self.assertEqual(len(unresolved), len(captures))

    def test_coalesce_multimodal_prefill_segments_uses_state_endpoints(self) -> None:
        dump = decoder_parity_v8.parity_test_v7.ParityDump
        captures = []
        for value in (1.0, 2.0, 3.0):
            captures.append(
                dump(4, "state_predelta", np.full(8, value, dtype=np.float32), 0, "fp32")
            )
        for value in (4.0, 5.0, 6.0):
            captures.append(
                dump(4, "new_state", np.full(8, value, dtype=np.float32), 0, "fp32")
            )
        row_specs = {
            (4, "state_predelta"): (8, (8,)),
            (4, "new_state"): (8, (8,)),
        }
        segments = [("text_before", 1, 0), ("visual", 3, 1), ("text_after", 2, 4)]

        merged = decoder_parity_v8._coalesce_multimodal_prefill_segments(
            captures, row_specs, segments
        )

        self.assertEqual([item.op_name for item in merged], ["state_predelta", "new_state"])
        np.testing.assert_array_equal(merged[0].data, np.full(8, 1.0, dtype=np.float32))
        np.testing.assert_array_equal(merged[1].data, np.full(8, 6.0, dtype=np.float32))

    def test_main_replays_segmented_prompt_from_bridge_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_decoder_bridge_report_") as tmpdir:
            tmp = Path(tmpdir)
            report_path = tmp / "report.json"
            bridge_report_path = tmp / "bridge_report.json"
            prefix_path = tmp / "prefix.f32"
            fake_gguf = tmp / "decoder.gguf"
            prefix_path.write_bytes(array("f", [0.0] * (9 * 64)).tobytes())
            bridge_report_path.write_text(
                json.dumps(
                    {
                        "decoder_runtime": {"gguf": str(fake_gguf)},
                        "decoder_context_len": 57,
                        "prompt": "Explain this image.",
                        "formatted_prompt": "<|im_start|>user\n<|vision_start|><image_embeds><|vision_end|>Explain this image.<|im_end|>\n<|im_start|>assistant\n",
                        "prompt_tokens_before_image": [11, 22],
                        "prompt_tokens_after_image": [33, 44, 55],
                        "multimodal_prompt_segmented": True,
                        "prefix_dump_path": str(prefix_path),
                        "prefix_grid_x": 3,
                        "prefix_grid_y": 3,
                        "prefix_text_pos": 5,
                        "prefix_decode_policy": "non_causal_visual_chunk",
                    }
                ),
                encoding="utf-8",
            )

            class FakeTokenizer:
                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return ",".join(str(x) for x in ids)

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": tmp / "libdecoder_v8.so",
                "c_path": tmp / "decoder_v8.c",
            }

            with mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(decoder_parity_v8.bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": array("f", [0.1, 0.9, 0.2, -0.4])}) as run_decoder, \
                 mock.patch.object(decoder_parity_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()), \
                 mock.patch.object(
                     decoder_parity_v8,
                     "_run_llama_capture",
                     return_value={
                         "meta": {
                             "ok": True,
                             "n_vocab": 4,
                             "token_count": 5,
                             "token_count_before": 2,
                             "token_count_after": 3,
                             "prefix_token_count": 9,
                             "prefix_position_count": 3,
                             "prefix_start_pos": 2,
                             "prefix_text_pos": 5,
                             "flash_attention_mode": "enabled",
                             "topk": [{"id": 1, "logit": 0.95}, {"id": 2, "logit": 0.18}],
                         },
                         "logits": np.array([0.0, 1.0, 0.1, -0.5], dtype=np.float32),
                     },
                 ) as llama_capture:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = decoder_parity_v8.main(
                        [
                            "--bridge-report",
                            str(bridge_report_path),
                            "--workdir",
                            str(tmp / "work"),
                            "--llama-flash-attention",
                            "enabled",
                            "--json-out",
                            str(report_path),
                        ]
                    )

            self.assertEqual(rc, 0)
            self.assertEqual(llama_capture.call_args.args[0], fake_gguf.resolve())
            self.assertEqual(llama_capture.call_args.args[1], [33, 44, 55])
            self.assertEqual(llama_capture.call_args.args[2], 57)
            self.assertEqual(llama_capture.call_args.kwargs["tokens_before"], [11, 22])
            self.assertEqual(llama_capture.call_args.kwargs["prefix_grid"], (3, 3))
            self.assertEqual(llama_capture.call_args.kwargs["prefix_text_pos"], 5)
            self.assertEqual(llama_capture.call_args.kwargs["flash_attention"], "enabled")
            self.assertEqual(Path(llama_capture.call_args.kwargs["prefix_path"]), prefix_path.resolve())
            _, decoder_kwargs = run_decoder.call_args
            self.assertEqual(decoder_kwargs["tokens_before"], [11, 22])
            self.assertEqual(decoder_kwargs["prefix_grid"], (3, 3))
            self.assertEqual(decoder_kwargs["prefix_text_pos"], 5)
            self.assertEqual(decoder_kwargs["prefix_decode_policy"], "non_causal_visual_chunk")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["prefix_decode_policy"], "non_causal_visual_chunk")
            self.assertEqual(report["llama_flash_attention_requested"], "enabled")
            self.assertEqual(report["llama_flash_attention_actual"], "enabled")
            self.assertTrue(report["multimodal_prompt_segmented"])
            self.assertEqual(report["formatted_prompt"], "<|im_start|>user\n<|vision_start|><image_embeds><|vision_end|>Explain this image.<|im_end|>\n<|im_start|>assistant\n")
            self.assertEqual(report["prompt_tokens_before_image"], [11, 22])
            self.assertEqual(report["prompt_tokens_after_image"], [33, 44, 55])
            self.assertEqual(report["prefix"]["path"], str(prefix_path.resolve()))
            self.assertEqual(report["position_contract"]["ck"]["rows"][0], [2, 2, 2, 0])
            self.assertEqual(report["position_contract"]["ck"]["rows"][4], [2, 3, 3, 0])
            self.assertTrue(report["position_contract"]["rows_match"])
            self.assertTrue(report["position_contract"]["text_pos_match"])


if __name__ == "__main__":
    unittest.main()
