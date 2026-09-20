from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
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


codegen = _load("parakeet_tdt_codegen", SCRIPTS / "codegen_v8.py")
certifier = _load(
    "parakeet_tdt_certifier",
    SCRIPTS / "certify_parakeet_generated_tdt_v8.py",
)


OP_IDS = {
    "tdt_embedding": "audio_tdt_embedding",
    "tdt_lstm_0": "audio_tdt_lstm",
    "tdt_lstm_1": "audio_tdt_lstm",
    "tdt_decoder_projection": "audio_tdt_projection",
    "tdt_joint_add": "audio_tdt_joint_add",
    "tdt_joint_relu": "audio_tdt_relu",
    "tdt_joint_head": "audio_tdt_joint_head",
    "tdt_token_argmax": "audio_tdt_argmax",
    "tdt_duration_argmax": "audio_tdt_argmax",
}


def _resolved_operations() -> list[dict]:
    circuit = json.loads(
        (ROOT / "version/v8/circuits/parakeet_tdt.json").read_text(
            encoding="utf-8"))
    maps = circuit["kernels"]
    operations = []
    for semantic_id, operation in OP_IDS.items():
        kernel_id = maps[operation]
        kernel_map = json.loads(
            (ROOT / f"version/v8/kernel_maps/{kernel_id}.json").read_text(
                encoding="utf-8"))
        operations.append({
            "op": operation,
            "template_op_id": semantic_id,
            "function": kernel_map["impl"]["function"],
            "args": [
                {
                    "name": parameter["name"],
                    "source": parameter["source"],
                    "expr": parameter["name"],
                }
                for parameter in kernel_map["call_abi"]["params"]
            ],
        })
    return operations


class GeneratedParakeetTDTTests(unittest.TestCase):
    def test_certifier_rejects_misaligned_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.npz"
            np.savez(
                fixture,
                **{
                    "encoder.projected": np.zeros((1, 2, 640), dtype=np.float32),
                    "joint.first_logits": np.zeros((1, 1, 1, 8), dtype=np.float32),
                    "decode.sequences": np.zeros((1, 2), dtype=np.int64),
                    "decode.durations": np.zeros((1, 3), dtype=np.int64),
                },
            )
            with self.assertRaisesRegex(ValueError, "must align"):
                certifier._fixture(fixture)

    def test_standalone_host_builds_and_queries_generated_capabilities(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler is unavailable")
        source = ROOT / "version/v8/src/ck_audio_transcribe_v8.c"
        text = source.read_text(encoding="utf-8")
        self.assertIn("ck_model_audio_blank_token_id", text)
        self.assertIn("ck_model_audio_frame_stride_samples", text)
        self.assertIn("ck_model_audio_transcription_output_capacity", text)
        self.assertNotIn("tokens[i] == 8192", text)
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                 str(source), "-ldl", "-o", str(Path(directory) / "host")],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_circuit_declares_complete_ordered_tdt_graph(self) -> None:
        circuit = json.loads(
            (ROOT / "version/v8/circuits/parakeet_tdt.json").read_text(
                encoding="utf-8"))
        decode = circuit["block_types"]["transducer"]["decode"]
        self.assertEqual([item["id"] for item in decode], list(OP_IDS))
        self.assertEqual([item["op"] for item in decode], list(OP_IDS.values()))
        self.assertTrue(all(
            item["when"] == {
                "config_key": "audio_include_tdt_decode", "equals": True}
            for item in decode
        ))

    def test_host_math_inventory_has_no_unresolved_production_debt(self) -> None:
        inventory = json.loads(
            (ROOT / "version/v8/contracts/parakeet_tdt_host_math_inventory.json").read_text(
                encoding="utf-8"))
        self.assertEqual(inventory["summary"]["open_entries"], 0)
        self.assertFalse(any(
            entry["status"] == "open" for entry in inventory["entries"]
        ))
        self.assertEqual(
            inventory["target_acceptance"]["production_model_math_in_python"], 0)

    def test_generated_tdt_uses_resolved_calls_and_owns_state_loop(self) -> None:
        generated = codegen._emit_audio_tdt_entrypoint(
            _resolved_operations(),
            {
                "decoder_hidden_size": 640,
                "num_decoder_layers": 2,
                "vocab_size": 8193,
                "blank_token_id": 8192,
                "pad_token_id": 2,
                "max_symbols_per_step": 10,
                "durations": [0, 1, 2, 3, 4],
                "audio_subsampling_factor": 8,
                "audio_tdt_joint_output_size": 8198,
                "audio_feature_channels": 128,
                "audio_hop_length": 160,
                "audio_subsampling_output_frames": 93,
                "audio_encoder_projection_size": 640,
                "audio_sample_rate": 16000,
                "audio_max_source_frames": 6399840,
            },
        )
        self.assertIn("CK_EXPORT int ck_model_run_audio_tdt_decode", generated)
        self.assertIn("CK_EXPORT int ck_model_transcribe_audio_wav", generated)
        self.assertIn("CK_EXPORT int ck_model_audio_blank_token_id", generated)
        self.assertIn("CK_EXPORT int ck_model_audio_transcription_output_capacity", generated)
        self.assertIn("CKModel *model = g_model", generated)
        self.assertIn("memset(hidden_state, 0, state_bytes)", generated)
        self.assertIn("while (frame < encoder_frames", generated)
        self.assertIn("token == 8192 && duration == 0", generated)
        for operation in _resolved_operations():
            self.assertIn(operation["function"] + "(", generated)

    def test_generated_tdt_rejects_incomplete_or_invalid_graph(self) -> None:
        operations = _resolved_operations()
        config = {
            "decoder_hidden_size": 640,
            "num_decoder_layers": 2,
            "vocab_size": 8193,
            "blank_token_id": 8192,
            "pad_token_id": 2,
            "max_symbols_per_step": 10,
            "durations": [0, 1, 2, 3, 4],
            "audio_subsampling_factor": 8,
            "audio_tdt_joint_output_size": 8198,
        }
        with self.assertRaisesRegex(RuntimeError, "complete resolved"):
            codegen._emit_audio_tdt_entrypoint(operations[:-1], config)
        with self.assertRaisesRegex(RuntimeError, "state geometry"):
            codegen._emit_audio_tdt_entrypoint(
                operations, {**config, "num_decoder_layers": 3})
        with self.assertRaisesRegex(RuntimeError, "joint size"):
            codegen._emit_audio_tdt_entrypoint(
                operations, {**config, "audio_tdt_joint_output_size": 1})


if __name__ == "__main__":
    unittest.main()
