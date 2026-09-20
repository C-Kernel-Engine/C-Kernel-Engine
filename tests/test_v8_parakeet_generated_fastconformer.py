from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codegen = _load("parakeet_fastconformer_codegen", SCRIPTS / "codegen_v8.py")
build_ir = _load("parakeet_fastconformer_build_ir", SCRIPTS / "build_ir_v8.py")
certifier = _load(
    "parakeet_fastconformer_certifier",
    SCRIPTS / "certify_parakeet_generated_fastconformer_v8.py",
)


def _operation(layer: int) -> dict:
    sources = [
        "activation:input",
        "activation:relative_positions",
        *(f"weight:w{index}" for index in range(39)),
        "output:output",
        "scratch:workspace",
        "scratch_size:workspace",
        "dim:frames",
    ]
    return {
        "op": "audio_fastconformer_block",
        "layer": layer,
        "function": f"resolved_block_{layer}",
        "args": [
            {"name": f"arg_{index}", "source": source, "expr": f"expr_{index}"}
            for index, source in enumerate(sources)
        ],
    }


class GeneratedParakeetFastConformerTests(unittest.TestCase):
    def test_relative_position_entrypoint_uses_resolved_call(self) -> None:
        operation = {
            "op": "audio_relative_position",
            "function": "resolved_relative_position",
            "args": [
                {"name": "output", "source": "output:output", "expr": "old_output"},
                {"name": "frames", "source": "dim:frames", "expr": "93"},
                {"name": "channels", "source": "dim:channels", "expr": "1024"},
            ],
        }
        generated = codegen._emit_audio_relative_position_entrypoint(
            [operation],
            {"audio_subsampling_output_frames": 93, "hidden_size": 1024},
        )
        self.assertIn("ck_model_prepare_audio_relative_positions", generated)
        self.assertIn("return resolved_relative_position(output, frames, 1024)", generated)
        self.assertIn("output_elements < required", generated)

    def test_compiler_uses_each_block_definition_in_multi_block_sequence(self) -> None:
        manifest = {
            "config": {
                "model": "multi_block_test",
                "num_layers": 1,
                "embed_dim": 8,
                "hidden_size": 8,
                "num_heads": 2,
                "num_kv_heads": 2,
                "head_dim": 4,
                "intermediate_size": 16,
                "context_length": 4,
                "audio_window_length": 8,
                "audio_subsampling_output_frames": 4,
                "prefer_q8_activation": False,
            },
            "entries": [],
            "quant_summary": {},
            "template": {
                "name": "multi_block_test",
                "kernels": {
                    "audio_hann_window": "audio_hann_window_f32",
                    "audio_relative_position": "audio_relative_sinusoidal_position_f32",
                },
                "sequence": ["first", "second"],
                "block_types": {
                    "first": {
                        "sequence": ["header"],
                        "header": [{
                            "id": "window",
                            "op": "audio_hann_window",
                            "params": {
                                "frames_from_config": "audio_window_length",
                                "periodic": 0,
                            },
                        }],
                    },
                    "second": {
                        "sequence": ["header"],
                        "header": [{
                            "id": "position",
                            "op": "audio_relative_position",
                            "params": {
                                "frames_from_config": "audio_subsampling_output_frames",
                                "channels_from_config": "hidden_size",
                            },
                        }],
                    },
                },
            },
        }
        operations = build_ir.build_ir1_direct(
            manifest, ROOT / "tests/multi_block.synthetic.json", mode="prefill")
        self.assertEqual(
            [operation["op"] for operation in operations],
            ["audio_hann_window", "audio_relative_position"],
        )

    def test_entrypoint_dispatches_contiguous_layers_and_runtime_frames(self) -> None:
        generated = codegen._emit_audio_fastconformer_block_entrypoint(
            [_operation(0), _operation(1)],
            {
                "audio_subsampling_output_frames": 93,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_attention_heads": 8,
            },
        )
        self.assertIn("CK_EXPORT int ck_model_run_audio_fastconformer_block", generated)
        self.assertIn("case 0: return resolved_block_0(input, relative_positions", generated)
        self.assertIn("case 1: return resolved_block_1(input, relative_positions", generated)
        self.assertIn("frames > 93", generated)

    def test_entrypoint_rejects_noncontiguous_layers(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "contiguous zero-based"):
            codegen._emit_audio_fastconformer_block_entrypoint(
                [_operation(0), _operation(2)],
                {
                    "audio_subsampling_output_frames": 4,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                },
            )

    def test_encoder_projection_entrypoint_uses_resolved_provider(self) -> None:
        operation = {
            "op": "audio_encoder_projection",
            "function": "resolved_projection",
            "args": [
                {"name": "A", "source": "activation:a", "expr": "old_input"},
                {"name": "B", "source": "weight:_first_weight", "expr": "projector_weight"},
                {"name": "bias", "source": "weight_f:_bias", "expr": "projector_bias"},
                {"name": "C", "source": "output:c", "expr": "old_output"},
                {"name": "M", "source": "runtime:seq_len", "expr": "93"},
                {"name": "N", "source": "dim:_output_dim", "expr": "640"},
                {"name": "K", "source": "dim:_input_dim", "expr": "1024"},
            ],
        }
        generated = codegen._emit_audio_encoder_projection_entrypoint(
            [operation],
            {
                "audio_subsampling_output_frames": 93,
                "hidden_size": 1024,
                "audio_encoder_projection_size": 640,
            },
        )
        self.assertIn("ck_model_run_audio_encoder_projection", generated)
        self.assertIn(
            "resolved_projection(input, projector_weight, projector_bias, output, frames, 640, 1024)",
            generated,
        )
        self.assertIn("frames > 93", generated)

    def test_native_encoder_schedule_is_derived_from_resolved_operations(self) -> None:
        operations = [
            {"op": "audio_fastconformer_subsampling"},
            {"op": "audio_relative_position"},
            _operation(0),
            _operation(1),
            {"op": "audio_encoder_projection"},
        ]
        generated = codegen._emit_audio_encoder_entrypoint(
            operations,
            {
                "audio_feature_frames": 32,
                "audio_subsampling_output_frames": 4,
                "audio_feature_channels": 6,
                "hidden_size": 8,
                "audio_encoder_projection_size": 5,
            },
        )
        self.assertIn("CK_EXPORT int ck_model_run_audio_encoder", generated)
        self.assertLess(
            generated.index("\n        0, hidden_a, relative_positions"),
            generated.index("\n        1, hidden_b, relative_positions"),
        )
        self.assertIn(
            "ck_model_run_audio_encoder_projection(\n        hidden_a",
            generated,
        )

    def test_native_encoder_rejects_incomplete_resolved_schedule(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "resolved sequence"):
            codegen._emit_audio_encoder_entrypoint(
                [
                    {"op": "audio_fastconformer_subsampling"},
                    _operation(0),
                    {"op": "audio_encoder_projection"},
                ],
                {
                    "audio_feature_frames": 32,
                    "audio_subsampling_output_frames": 4,
                    "audio_feature_channels": 6,
                    "hidden_size": 8,
                    "audio_encoder_projection_size": 5,
                },
            )

    def test_circuit_produces_relative_positions_before_repeated_blocks(self) -> None:
        circuit = json.loads(
            (ROOT / "version/v8/circuits/parakeet_tdt.json").read_text(
                encoding="utf-8"))
        encoder = circuit["block_types"]["encoder"]
        self.assertEqual(
            encoder["sequence"],
            ["subsampling", "relative_position", "body", "encoder_projection"],
        )
        self.assertEqual(
            encoder["relative_position"][0]["graph_slots"]["outputs"]["output"],
            "audio_relative_positions",
        )
        block = encoder["body"]["ops"][0]
        self.assertEqual(block["op"], "audio_fastconformer_block")
        self.assertEqual(
            block["graph_slots"]["inputs"]["relative_positions"],
            "audio_relative_positions",
        )
        projection = encoder["encoder_projection"][0]
        self.assertEqual(projection["op"], "audio_encoder_projection")
        self.assertEqual(
            projection["when"],
            {"config_key": "audio_include_encoder_projection", "equals": True},
        )
        self.assertEqual(
            projection["weight_refs"]["weight"], "encoder_projector.weight")

    def test_projection_is_excluded_from_diagnostic_component_scopes(self) -> None:
        circuit = json.loads(
            (ROOT / "version/v8/circuits/parakeet_tdt.json").read_text(
                encoding="utf-8"))
        projection = circuit["block_types"]["encoder"]["encoder_projection"]
        self.assertEqual(
            build_ir._normalize_template_op_items(
                projection, config={"audio_include_encoder_projection": True}),
            projection,
        )
        self.assertEqual(
            build_ir._normalize_template_op_items(
                projection, config={"audio_include_encoder_projection": False}),
            [],
        )

    def test_fixture_validation_rejects_mismatched_layers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fixture = Path(td) / "fixture.npz"
            np.savez(
                fixture,
                **{
                    "encoder.subsampling": np.zeros((1, 4, 8), np.float32),
                    "frontend.input_features": np.zeros((1, 32, 6), np.float32),
                    "frontend.attention_mask": np.ones((1, 32), np.bool_),
                    "encoder.layer.0": np.zeros((1, 4, 8), np.float32),
                    "encoder.layer.23": np.zeros((1, 3, 8), np.float32),
                    "encoder.projected": np.zeros((1, 4, 6), np.float32),
                },
            )
            with self.assertRaisesRegex(ValueError, "shapes must agree"):
                certifier._checked_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
