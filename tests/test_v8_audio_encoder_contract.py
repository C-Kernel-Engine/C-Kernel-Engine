#!/usr/bin/env python3
"""Fail-closed circuit and kernel-map tests for reusable audio transformers."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "version" / "v8"
RESOLVER_PATH = V8 / "scripts" / "resolve_numerical_execution_contracts_v8.py"
BUILD_IR_PATH = V8 / "scripts" / "build_ir_v8.py"
CODEGEN_CORE_PATH = V8 / "scripts" / "codegen_core_v8.py"
CODEGEN_PATH = V8 / "scripts" / "codegen_v8.py"
WHISPER_XRAY_PATH = V8 / "scripts" / "compare_whisper_encoder_pytorch_v8.py"
NIGHTLY_PATH = ROOT / "scripts" / "nightly_runner.py"
if str(BUILD_IR_PATH.parent) not in sys.path:
    sys.path.insert(0, str(BUILD_IR_PATH.parent))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


resolver = _load_module("audio_encoder_contract_resolver", RESOLVER_PATH)
build_ir = _load_module("audio_encoder_build_ir", BUILD_IR_PATH)
codegen_core = _load_module("audio_encoder_codegen_core", CODEGEN_CORE_PATH)
codegen = _load_module("audio_encoder_codegen", CODEGEN_PATH)
whisper_xray = _load_module("audio_encoder_whisper_xray", WHISPER_XRAY_PATH)
nightly = _load_module("audio_encoder_nightly", NIGHTLY_PATH)


def _fp32_entry(name: str, shape: list[int]) -> dict:
    elements = 1
    for extent in shape:
        elements *= extent
    return {
        "name": name,
        "dtype": "fp32",
        "offset": 0,
        "shape": shape,
        "size": elements * 4,
        "nbytes": elements * 4,
    }


def _make_audio_encoder_manifest() -> dict:
    config = {
        "model": "audio_transformer_encoder",
        "artifact_scope": "encoder_only",
        "num_layers": 1,
        "embed_dim": 8,
        "num_heads": 2,
        "num_kv_heads": 2,
        "head_dim": 4,
        "intermediate_size": 16,
        "context_length": 4,
        "audio_sample_rate": 16000,
        "audio_sample_extent": 1280,
        "audio_max_source_frames": 3840,
        "audio_resample_radius": 16,
        "audio_n_fft": 400,
        "audio_hop_length": 160,
        "audio_power_bins": 201,
        "audio_feature_channels": 4,
        "audio_feature_frames": 8,
        "audio_conv1_output_channels": 8,
        "audio_conv2_output_channels": 8,
        "audio_conv1_kernel_size": 3,
        "audio_conv1_stride": 1,
        "audio_conv1_padding": 1,
        "audio_conv1_output_frames": 8,
        "audio_conv1_elements": 64,
        "audio_conv2_kernel_size": 3,
        "audio_conv2_stride": 2,
        "audio_conv2_padding": 1,
        "audio_conv2_output_frames": 4,
        "audio_conv2_elements": 32,
        "attention_scale": 0.5,
        "rms_eps": 1.0e-5,
        "prefer_q8_activation": False,
        "numerical_contract_mode": "production",
    }
    entries = [
        _fp32_entry("audio_conv1_weight", [8, 4, 3]),
        _fp32_entry("audio_conv1_bias", [8]),
        _fp32_entry("audio_conv2_weight", [8, 8, 3]),
        _fp32_entry("audio_conv2_bias", [8]),
        _fp32_entry("pos_emb", [4, 8]),
        _fp32_entry("layer.0.ln1_gamma", [8]),
        _fp32_entry("layer.0.ln1_beta", [8]),
        _fp32_entry("layer.0.ln2_gamma", [8]),
        _fp32_entry("layer.0.ln2_beta", [8]),
        _fp32_entry("layer.0.wq", [8, 8]),
        _fp32_entry("layer.0.wk", [8, 8]),
        _fp32_entry("layer.0.wv", [8, 8]),
        _fp32_entry("layer.0.wo", [8, 8]),
        _fp32_entry("layer.0.w3", [16, 8]),
        _fp32_entry("layer.0.w2", [8, 16]),
        _fp32_entry("final_ln_weight", [8]),
        _fp32_entry("final_ln_bias", [8]),
    ]
    return {
        "config": config,
        "quant_summary": {
            "layer.0": {
                "wq": "fp32",
                "wk": "fp32",
                "wv": "fp32",
                "wo": "fp32",
                "w3": "fp32",
                "w2": "fp32",
            }
        },
        "entries": entries,
        "template": build_ir._load_builtin_template_doc("audio_transformer_encoder"),
    }


class AudioEncoderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.circuit_path = V8 / "circuits" / "audio_transformer_encoder.json"
        cls.circuit = resolver.load_json(cls.circuit_path)
        cls.frontend = resolver.load_json(
            V8 / "circuits" / "whisper_audio_frontend.json"
        )
        cls.contracts = resolver.load_json(resolver.DEFAULT_CONTRACTS)
        cls.kernels = resolver.load_kernel_capabilities(contracts=cls.contracts)

    def test_audio_encoder_contracts_resolve_exact_providers(self):
        expected = {
            "audio.frontend.wav_decode": "audio_wav_decode_memory_pcm16_mono_window_f32",
            "audio.frontend.resample": "audio_resample_windowed_sinc_f32",
            "audio.frontend.pad": "audio_pad_or_truncate_f32",
            "audio.frontend.stft_tables": "audio_stft_precompute_tables_f32",
            "audio.frontend.stft": "audio_stft_power_fft400_f32",
            "audio.frontend.mel_filters": "audio_whisper_mel_filters_slaney_f32",
            "audio.frontend.log_mel": "audio_whisper_log_mel_from_power_f32",
            "audio.encoder.stem.conv1": "audio_conv1d_channel_major_f32",
            "audio.encoder.stem.conv2": "audio_conv1d_channel_major_f32",
            "audio.encoder.layout": "audio_transpose_channel_to_token_f32",
            "audio.encoder.position": "position_embeddings_add",
            "audio.encoder.attention.fp32": "attention_forward_query_key_head_major_f32_packed_k",
        }
        for requirement, kernel_id in expected.items():
            with self.subTest(requirement=requirement):
                plan = resolver.resolve_contract(
                    self.circuit,
                    self.contracts,
                    self.kernels,
                    requirement,
                    "prefill",
                    mode="production",
                )
                self.assertEqual(plan["kernel"]["id"], kernel_id)

    def test_fp16_audio_attention_contract_resolves_tiled_provider(self):
        plan = resolver.resolve_contract(
            self.circuit,
            self.contracts,
            self.kernels,
            "audio.encoder.attention.fp16_tiled",
            "prefill",
            mode="production",
        )
        self.assertEqual(
            plan["kernel"]["id"],
            "attention_forward_query_key_head_major_tiled_f16kv_fp32",
        )

    def test_whisper_xray_maps_every_generated_operation(self):
        config = _make_audio_encoder_manifest()["config"]
        config["num_layers"] = 4
        table = whisper_xray.checkpoint_table(config)
        self.assertEqual(sorted(table), list(range(79)))
        self.assertEqual(table[0].buffer, "audio_conv_1")
        self.assertEqual(table[14].shape, (2, 4, 4))
        self.assertEqual(table[21].shape, (4, 16))
        self.assertEqual(table[78].name, "encoder.final_layer_norm")

    def test_whisper_xray_rejects_inconsistent_layout_dimensions(self):
        config = _make_audio_encoder_manifest()["config"]
        config["audio_conv2_output_frames"] = 3
        with self.assertRaisesRegex(ValueError, "layout contract"):
            whisper_xray.checkpoint_table(config)

    def test_whisper_xray_head_major_round_trip(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        token_major = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
        head_major = whisper_xray._head_major(token_major, 2, 4)
        restored = whisper_xray._token_major(head_major)
        self.assertTrue(torch.equal(restored, token_major[0]))

    def test_audio_primitive_contracts_resolve_all_exact_providers(self):
        cases = {
            "audio_pcm_s16_mono_fp32": (
                "audio_pcm_decode", "audio_pcm_s16_to_mono_f32"
            ),
            "audio_resample_linear_rational_fp32": (
                "audio_resample", "audio_resample_linear_f32"
            ),
            "audio_resample_windowed_sinc_fp64_accum_fp32_output": (
                "audio_resample", "audio_resample_windowed_sinc_f32"
            ),
            "audio_stft_centered_hann_precomputed_fp32": (
                "audio_stft", "audio_stft_power_precomputed_f32"
            ),
            "audio_stft_centered_hann_fft400_radix20_fp32": (
                "audio_stft", "audio_stft_power_fft400_f32"
            ),
            "audio_conv1d_channel_major_ascending_fp32": (
                "audio_conv1d", "audio_conv1d_channel_major_f32"
            ),
            "layout_channel_to_token_copy_fp32": (
                "layout_transform", "audio_transpose_channel_to_token_f32"
            ),
            "position_embeddings_add_token_major_fp32": (
                "position_embeddings", "position_embeddings_add"
            ),
            "attention_query_key_scaled_ordered_fp32": (
                "attention", "attention_forward_query_key_head_major_f32"
            ),
            "attention_query_key_scaled_ordered_fp32_packed_k": (
                "attention", "attention_forward_query_key_head_major_f32_packed_k"
            ),
        }
        for contract_id, (operator, kernel_id) in cases.items():
            circuit = {
                "required_numerical_contracts": {
                    "test": {
                        "op": operator,
                        "template_ops": ["test_op"],
                        "phases": {
                            "prefill": {
                                "contract_id": contract_id,
                                "validation": "validated",
                                "evidence": "synthetic exact-provider resolution test",
                            }
                        },
                        "checkpoint": {
                            "id": "test.output",
                            "producer": "test_op",
                            "logical_layout": "test_layout",
                            "axis_names": ["element"],
                        },
                    }
                }
            }
            with self.subTest(contract=contract_id):
                plan = resolver.resolve_contract(
                    circuit,
                    self.contracts,
                    self.kernels,
                    "test",
                    "prefill",
                    mode="production",
                )
                self.assertEqual(plan["kernel"]["id"], kernel_id)

    def test_frontend_and_encoder_do_not_name_concrete_kernels(self):
        self.assertNotIn("kernels", self.frontend)
        self.assertNotIn("kernels", self.circuit)

    def test_audio_runtime_topology_policy_is_circuit_declared(self):
        policy = self.circuit["contract"]["audio_encoder"][
            "runtime_topology_policy"
        ]
        self.assertEqual(policy["config_key"], "audio_runtime_topology_policy")
        self.assertEqual(policy["fallback"], "all_allowed_cpus")
        self.assertIn("performance_core_smt_on_hybrid", policy["supported"])

    def test_reusable_frontend_and_encoder_prefix_cannot_drift(self):
        frontend_requirements = self.frontend["required_numerical_contracts"]
        encoder_requirements = {
            name: requirement
            for name, requirement in self.circuit[
                "required_numerical_contracts"
            ].items()
            if name.startswith("audio.frontend.")
        }
        self.assertEqual(frontend_requirements, encoder_requirements)
        reusable_sequence = self.frontend["block_types"]["audio_frontend"]["sequence"]
        encoder_header = self.circuit["block_types"]["audio_encoder"]["header"]
        self.assertEqual(
            [row["op"] for row in reusable_sequence],
            [row["op"] for row in encoder_header[: len(reusable_sequence)]],
        )

    def test_audio_encoder_generates_complete_noncausal_call_ir(self):
        manifest = _make_audio_encoder_manifest()
        ir1 = build_ir.build_ir1_direct(
            manifest,
            ROOT / "tests" / "audio_encoder_manifest.synthetic.json",
            mode="prefill",
        )
        by_op = {}
        for operation in ir1:
            by_op.setdefault(operation["op"], []).append(operation)
        self.assertEqual(
            [row["kernel"] for row in by_op["audio_conv1d_stem_1"]],
            ["audio_conv1d_channel_major_f32"],
        )
        self.assertEqual(by_op["audio_conv1d_stem_1"][0]["params"]["kernel_size"], 3)
        self.assertEqual(by_op["audio_conv1d_stem_2"][0]["params"]["stride"], 2)
        self.assertEqual(
            [row["kernel"] for row in by_op["position_embeddings"]],
            ["position_embeddings_add"],
        )
        self.assertEqual(
            [row["kernel"] for row in by_op["attn"]],
            ["attention_forward_query_key_head_major_f32_packed_k"],
        )
        self.assertFalse(manifest["config"]["_template_uses_kv_cache"])
        self.assertFalse(manifest["config"]["_template_uses_rope"])

        registry = build_ir.load_kernel_registry()
        lower1 = build_ir.generate_ir_lower_1(ir1, registry, manifest, "prefill")
        layout = build_ir.generate_memory_layout(
            lower1, manifest, registry, mode="prefill", context_len=4
        )
        buffers = {
            row["name"]: row
            for row in layout["memory"]["activations"]["buffers"]
        }
        for name in (
            "audio_samples",
            "audio_resampled",
            "audio_normalized",
            "audio_window",
            "audio_cos_table",
            "audio_sin_table",
            "audio_power",
            "audio_mel_filters",
            "audio_fft_scratch",
        ):
            with self.subTest(frontend_buffer=name):
                self.assertIn(name, buffers)
        self.assertEqual(buffers["audio_features"]["shape"], "[4, 8]")
        self.assertEqual(buffers["audio_conv_1"]["shape"], "[8, 8]")
        self.assertEqual(buffers["audio_conv_2"]["shape"], "[8, 4]")
        lower2 = build_ir.generate_ir_lower_2(
            lower1, layout, manifest, registry, mode="prefill"
        )
        call_ir = build_ir.generate_ir_lower_3(lower2, "prefill")
        errors = [
            (row["op"], row.get("errors"))
            for row in call_ir["operations"]
            if row.get("errors")
        ]
        self.assertEqual(errors, [])
        attention_call = next(
            row for row in call_ir["operations"]
            if row.get("function") == "attention_forward_query_key_head_major_f32_packed_k"
        )
        scratch_args = {
            arg["name"]: arg["expr"] for arg in attention_call["args"]
            if arg["name"] in {"score_scratch", "key_transpose_scratch"}
        }
        self.assertEqual(set(scratch_args), {"score_scratch", "key_transpose_scratch"})
        self.assertNotEqual(
            scratch_args["score_scratch"], scratch_args["key_transpose_scratch"]
        )
        frontend_calls = {
            row["op"]: row
            for row in call_ir["operations"]
            if row.get("op")
            in {
                "audio_wav_decode",
                "audio_resample",
                "audio_pad_or_truncate",
                "audio_stft_tables",
                "audio_stft",
                "audio_mel_filters",
                "audio_log_mel",
                "audio_feature_window",
            }
        }
        expected_frontend_functions = {
            "audio_wav_decode": "audio_wav_decode_memory_pcm16_mono_window_f32",
            "audio_resample": "audio_resample_windowed_sinc_f32",
            "audio_pad_or_truncate": "audio_pad_or_truncate_f32",
            "audio_stft_tables": "audio_stft_precompute_tables_f32",
            "audio_stft": "audio_stft_power_fft400_f32",
            "audio_mel_filters": "audio_whisper_mel_filters_slaney_f32",
            "audio_log_mel": "audio_whisper_log_mel_from_power_reference_f32",
            "audio_feature_window": "audio_whisper_log_mel_window_wav_pcm16_f32",
        }
        self.assertEqual(set(frontend_calls), set(expected_frontend_functions))
        for op, function in expected_frontend_functions.items():
            with self.subTest(frontend_call=op):
                self.assertEqual(frontend_calls[op]["function"], function)
                self.assertTrue(frontend_calls[op]["args"])
        wav_args = {
            arg["name"]: arg["expr"]
            for arg in frontend_calls["audio_wav_decode"]["args"]
        }
        self.assertEqual(wav_args["byte_count"], "audio_wav_byte_count")
        resample_args = {
            arg["name"]: arg["expr"]
            for arg in frontend_calls["audio_resample"]["args"]
        }
        self.assertEqual(resample_args["input_frames"], "audio_source_frames")
        self.assertEqual(resample_args["input_rate"], "audio_source_rate")

        entrypoint = codegen._emit_audio_wav_entrypoint(
            call_ir["operations"], manifest["config"]
        )
        self.assertIn("CK_EXPORT int ck_model_run_audio_wav(", entrypoint)
        self.assertIn(
            "CK_EXPORT int ck_model_prepare_audio_wav_window(", entrypoint
        )
        for function in expected_frontend_functions.values():
            self.assertIn(function + "(", entrypoint)
        descriptor = codegen._emit_runtime_capability_api(
            call_ir, layout, None, None
        )
        self.assertIn("CK_MODEL_CAP_AUDIO_WAV_ENCODER", descriptor)
        self.assertIn("CK_MODEL_CAP_ENCODER_OUTPUT", descriptor)
        self.assertIn("CK_EXPORT int ck_model_get_encoder_output(", descriptor)
        self.assertNotIn("CK_MODEL_CAP_AUTOREGRESSIVE_DECODE,", descriptor)
        missing = [
            row
            for row in call_ir["operations"]
            if row.get("op") != "audio_mel_filters"
        ]
        with self.assertRaisesRegex(
            RuntimeError, "did not lower every required operation"
        ):
            codegen._emit_audio_wav_entrypoint(missing, manifest["config"])
        gelu_calls = [
            row for row in call_ir["operations"]
            if row.get("op") == "gelu" and int(row.get("layer", -1)) == -1
        ]
        self.assertEqual(len(gelu_calls), 2)
        gelu_args = [
            {arg["name"]: arg for arg in row["args"]}
            for row in gelu_calls
        ]
        self.assertEqual(gelu_args[0]["data"]["buffer_ref"], "audio_conv_1")
        self.assertEqual(gelu_args[0]["n"]["expr"], "64")
        self.assertEqual(gelu_args[1]["data"]["buffer_ref"], "audio_conv_2")
        self.assertEqual(gelu_args[1]["n"]["expr"], "32")
        attention_call = next(
            row for row in call_ir["operations"] if row.get("op") == "attn"
        )
        attention_args = {
            arg["name"]: arg for arg in attention_call["args"]
        }
        self.assertEqual(attention_args["query"]["buffer_ref"], "q_scratch")
        self.assertEqual(attention_args["key"]["buffer_ref"], "k_scratch")
        self.assertEqual(attention_args["value"]["buffer_ref"], "v_scratch")

    def test_encoder_only_codegen_contract_is_capability_scoped_and_fail_closed(self):
        manifest = _make_audio_encoder_manifest()
        config = copy.deepcopy(manifest["config"])
        config["artifact_scope"] = "encoder_only"
        config["contract"] = copy.deepcopy(manifest["template"]["contract"])
        self.assertEqual(codegen_core._validate_codegen_contract(config), [])

        missing_output = copy.deepcopy(config)
        del missing_output["contract"]["audio_encoder"]["output"]
        self.assertIn(
            "missing contract field: audio_encoder.output",
            codegen_core._validate_codegen_contract(missing_output),
        )

        decoder = copy.deepcopy(config)
        decoder["artifact_scope"] = "decoder"
        decoder_issues = codegen_core._validate_codegen_contract(decoder)
        self.assertIn("missing contract section: tokenizer_contract", decoder_issues)
        self.assertIn("missing contract section: quant_contract", decoder_issues)

    def test_audio_encoder_geometry_mismatch_is_a_hard_failure(self):
        manifest = _make_audio_encoder_manifest()
        manifest["config"]["context_length"] = 5
        with self.assertRaisesRegex(ValueError, "post-Conv1D token extent"):
            build_ir.build_activation_specs(
                manifest["config"], mode="prefill", context_len=5
            )

    def test_audio_ops_are_generic_dsl_vocabulary(self):
        expected = {
            "audio_wav_decode": "audio_wav_decode",
            "audio_pcm_decode": "audio_pcm_decode",
            "audio_resample": "audio_resample",
            "audio_pad_or_truncate": "audio_pad_or_truncate",
            "audio_stft_tables": "audio_stft_tables",
            "audio_stft": "audio_stft",
            "audio_mel_filters": "audio_mel_filters",
            "audio_log_mel": "audio_log_mel",
            "audio_feature_window": "audio_feature_window",
            "audio_conv1d_stem_1": "audio_conv1d",
            "audio_conv1d_stem_2": "audio_conv1d",
            "layout_channel_to_token": "layout_transform",
            "cross_attn": "attention",
        }
        for op, family in expected.items():
            self.assertEqual(build_ir.TEMPLATE_TO_KERNEL_OP.get(op), family)
            self.assertIn(op, build_ir.OP_DATAFLOW)

    def test_shared_cross_attention_provider_is_not_audio_named(self):
        kernel = json.loads(
            (V8 / "kernel_maps" / "attention_forward_query_key_head_major_f32_packed_k.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(kernel["op"], "attention")
        identity = f"{kernel['id']} {kernel['impl']['function']}".lower()
        self.assertNotIn("audio", identity)
        self.assertNotIn("whisper", identity)
        self.assertEqual(kernel["scratch"][0]["shape"], ["Tq", "Tk"])
        self.assertEqual(kernel["scratch"][1]["shape"], ["H", "D", "Tk"])
        threading = kernel["numerical_capabilities"][0]["implementation"]["threading"]
        self.assertEqual(threading["work_partition"], ["independent_rows"])

    def test_unknown_resampling_semantics_are_a_hard_failure(self):
        circuit = copy.deepcopy(self.frontend)
        request = circuit["required_numerical_contracts"]["audio.frontend.log_mel"]
        request["op"] = "audio_resample"
        request["template_ops"] = ["audio_resample"]
        request["phases"]["prefill"]["contract_id"] = (
            "audio_resample_unknown_bandlimited_fp32"
        )
        with self.assertRaises(resolver.ContractError):
            resolver.resolve_contract(
                circuit,
                self.contracts,
                self.kernels,
                "audio.frontend.log_mel",
                "prefill",
                mode="production",
            )

    def test_audio_primitive_matrix_is_a_visible_nightly_row(self):
        suite = nightly.TEST_SUITES["audio_transformer_primitives"]
        self.assertEqual(suite.name, "Audio Transformer Primitives")
        self.assertEqual(suite.category, "kernels")
        self.assertEqual(suite.test_file.name, "test_audio_encoder.py")
        source = suite.test_file.read_text(encoding="utf-8")
        capability_pin = source.index(
            'os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")'
        )
        torch_import = source.index("import torch")
        self.assertLess(capability_pin, torch_import)
        self.assertIn('{"DEFAULT", "NO AVX"}', source)
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("pytest", requirements.splitlines())
        constraints = (
            ROOT / ".github" / "requirements-nightly-constraints.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("torch==2.12.1", constraints.splitlines())
        workflow = (
            ROOT / ".github" / "workflows" / "nightly.yml"
        ).read_text(encoding="utf-8")
        dependency_installs = [
            line.strip()
            for line in workflow.splitlines()
            if "pip install" in line and "--upgrade pip" not in line
        ]
        self.assertTrue(
            dependency_installs,
            "nightly must install its Python dependency sets",
        )
        self.assertTrue(
            all(
                "-c .github/requirements-nightly-constraints.txt" in line
                for line in dependency_installs
            ),
            "every nightly dependency install must use the pinned constraints file",
        )
        parsed = nightly.parse_sub_tests(
            "audio_encoder_self_attention_equal "
            "max_diff=2.98e-08 tol=2.0e-06 [PASS]\n"
        )
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].status, "pass")
        failed = nightly.parse_sub_tests(
            "audio_erf_gelu_fp64_scalar "
            "max_diff=1.19209290e-06 tol=0 [FAIL] "
            "rmse=1.3e-07 rmse_tol=1.25e-07\n"
        )
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].name, "audio_erf_gelu_fp64_scalar")
        self.assertEqual(failed[0].status, "fail")
        self.assertEqual(failed[0].max_diff, 1.1920929e-6)
        self.assertEqual(failed[0].tolerance, 0.0)

    def test_standalone_attention_library_links_its_bf16_gemm_dependency(self):
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        rule = makefile.split("$(LIB_ATTENTION):", 1)[1].split("\n\n", 1)[0]
        self.assertIn("src/kernels/gemm_kernels_bf16.c", rule)

    def test_audio_erf_gelu_uses_fp64_scalar_libm(self):
        kernel = json.loads(
            (V8 / "kernel_maps" / "gelu_erf_fp64_f32_inplace.json")
            .read_text(encoding="utf-8")
        )
        capability = kernel["numerical_capabilities"][0]
        self.assertEqual(capability["implementation"]["isa_dispatch"], "scalar")
        source = (ROOT / "src" / "kernels" / "gelu_kernels.c").read_text(
            encoding="utf-8"
        )
        function = source.split(
            "void gelu_erf_fp64_f32_inplace(float *data, size_t n)", 1
        )[1].split("\n}", 1)[0]
        self.assertIn("ck_gelu_system_erf()", function)
        self.assertIn("reference_erf(", function)
        self.assertIn("const double scaled", function)
        self.assertIn("data[i] = (float)", function)

    def test_bf16_erf_gelu_oracle_is_pytorch_version_scoped(self):
        source = (
            ROOT
            / "unittest"
            / "bf16"
            / "test_gelu_pytorch_erf_sleef_storage_bf16.py"
        ).read_text(encoding="utf-8")
        self.assertIn('torch_version.startswith("2.8.")', source)
        kernel = json.loads(
            (
                V8
                / "kernel_maps"
                / "gelu_pytorch_erf_sleef_bf16_storage.json"
            ).read_text(encoding="utf-8")
        )
        self.assertIn("PyTorch 2.8", kernel["notes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
