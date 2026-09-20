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
CIRCUIT = ROOT / "version" / "v8" / "circuits" / "cohere_transcribe.json"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


converter = _load(
    "cohere_generated_encoder_converter",
    SCRIPTS / "convert_cohere_transcribe_gguf_to_bump_v8.py",
)
codegen_core = _load(
    "cohere_generated_encoder_codegen_core", SCRIPTS / "codegen_core_v8.py"
)
build_ir = _load(
    "cohere_generated_encoder_build_ir", SCRIPTS / "build_ir_v8.py"
)
certifier = _load(
    "cohere_generated_encoder_certifier",
    SCRIPTS / "certify_cohere_generated_encoder_components_v8.py",
)


class GeneratedCohereEncoderTests(unittest.TestCase):
    def test_circuit_uses_shared_contract_matched_encoder_operations(self) -> None:
        circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
        encoder = circuit["block_types"]["encoder"]
        self.assertEqual(
            encoder["sequence"],
            ["subsampling", "relative_position", "body", "encoder_projection"],
        )
        self.assertEqual(
            encoder["subsampling"][0]["op"], "audio_fastconformer_subsampling"
        )
        block = encoder["body"]["ops"][0]
        self.assertEqual(block["op"], "audio_fastconformer_block")
        self.assertEqual(
            block["weight_refs"]["relative_weight"],
            "enc.blk.{L}.attn.pos.weight",
        )
        self.assertEqual(
            block["weight_refs"]["conv_bn_variance"],
            "enc.blk.{L}.conv.bn.var",
        )
        self.assertNotIn("cohere_mode", block.get("params", {}))

    def test_component_scopes_have_explicit_weight_boundaries(self) -> None:
        circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
        ignored = circuit["contract"]["weight_policy"]["ignore"]
        by_scope: dict[str, set[str]] = {}
        for row in ignored:
            by_scope.setdefault(row["when"]["equals"], set()).add(row["pattern"])
        self.assertEqual(by_scope["audio_frontend"], {"enc.*", "dec.*"})
        self.assertEqual(
            by_scope["audio_subsampling"],
            {"enc.blk.*", "enc.proj.*", "dec.*"},
        )
        self.assertEqual(
            by_scope["audio_encoder_block"],
            {"fe.*", "enc.pre.*", "enc.proj.*", "dec.*"},
        )
        self.assertEqual(by_scope["audio_encoder"], {"dec.*"})

    def test_fp16_promotion_is_component_scoped(self) -> None:
        f16 = converter.c.GGML_TYPE_F16
        f32 = converter.c.GGML_TYPE_F32
        self.assertTrue(converter._promote_to_fp32("fe.mel_fb", f16, "audio_frontend"))
        self.assertTrue(
            converter._promote_to_fp32(
                "enc.pre.out.weight", f16, "audio_subsampling"
            )
        )
        self.assertFalse(
            converter._promote_to_fp32(
                "enc.blk.0.attn.q.weight", f16, "audio_subsampling"
            )
        )
        self.assertTrue(
            converter._promote_to_fp32(
                "enc.blk.0.attn.q.weight", f16, "audio_encoder_block"
            )
        )
        self.assertFalse(
            converter._promote_to_fp32(
                "enc.pre.out.weight", f16, "audio_encoder_block"
            )
        )
        self.assertTrue(
            converter._promote_to_fp32("enc.proj.weight", f16, "audio_encoder")
        )
        self.assertFalse(
            converter._promote_to_fp32("enc.proj.weight", f32, "audio_encoder")
        )

    def test_projection_is_conditioned_by_generated_encoder_scope(self) -> None:
        circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
        projection = circuit["block_types"]["encoder"]["encoder_projection"][0]
        self.assertEqual(
            projection["when"],
            {"config_key": "audio_include_encoder_projection", "equals": True},
        )

    def test_audio_component_scope_uses_encoder_contract_not_text_contract(self) -> None:
        circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
        issues = codegen_core._validate_codegen_contract(
            {
                "artifact_scope": "audio_encoder_block",
                "contract": circuit["contract"],
            }
        )
        self.assertEqual(issues, [])

    def test_component_input_is_explicitly_external_not_uninitialized(self) -> None:
        tracker = build_ir.DataflowTracker({"audio_encoder_tokens"})
        dataflow = tracker.record_op(
            0,
            "audio_fastconformer_block",
            0,
            0,
            {"input": "audio_encoder_tokens"},
            {"output": "audio_encoder_tokens"},
        )
        self.assertEqual(
            dataflow["inputs"]["input"]["from"],
            "external:audio_encoder_tokens",
        )

    def test_external_activation_contract_rejects_unknown_slot(self) -> None:
        manifest = {
            "config": {
                "model": "external_slot_test",
                "num_layers": 1,
                "embed_dim": 8,
                "hidden_size": 8,
                "num_heads": 2,
                "num_kv_heads": 2,
                "head_dim": 4,
                "intermediate_size": 16,
                "context_length": 4,
                "external_activation_slots": ["not_declared"],
            },
            "entries": [],
            "quant_summary": {},
            "template": {
                "name": "external_slot_test",
                "kernels": {},
                "activation_buffers": {},
                "sequence": ["empty"],
                "block_types": {"empty": {"sequence": []}},
            },
        }
        with self.assertRaisesRegex(RuntimeError, "not declared by the circuit"):
            build_ir.build_ir1_direct(manifest, ROOT / "tests/external.synthetic.json")

    def test_certifier_rejects_wrong_relative_position_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.npz"
            np.savez(
                path,
                **{
                    "audio.frontend.log_mel.output": np.zeros((8, 128), np.float32),
                    "audio.encoder.subsampling.output": np.zeros((2, 1280), np.float32),
                    "audio.encoder.relative_position.output": np.zeros((2, 1280), np.float32),
                    "audio.encoder.layer.0.output": np.zeros((2, 1280), np.float32),
                    "audio.encoder.projected.output": np.zeros((2, 1024), np.float32),
                },
            )
            with self.assertRaisesRegex(ValueError, "relative-position"):
                certifier._fixture(path)

    def test_first_weight_selector_never_uses_leading_bias(self) -> None:
        selected = build_ir._select_first_non_bias_call_weight(
            {
                "bias": {"name": "enc.proj.bias"},
                "weight": {"name": "enc.proj.weight"},
            }
        )
        self.assertEqual(selected, ("weight", {"name": "enc.proj.weight"}))


if __name__ == "__main__":
    unittest.main()
