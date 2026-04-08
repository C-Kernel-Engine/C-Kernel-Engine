import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "version" / "v8" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numeric_parity_qwen3vl_mmproj_v8 as npv8  # type: ignore  # noqa: E402


class NumericParityQwen3VLMmprojV8Tests(unittest.TestCase):
    def test_llama_reference_output_name_uses_public_encode_for_qwen3vl(self) -> None:
        config = {
            "projector_out_dim": 4096,
            "projector_total_out_dim": 16384,
        }
        self.assertIsNone(npv8._llama_reference_output_name(config))

    def test_llama_reference_output_name_uses_public_encode_when_dims_match(self) -> None:
        config = {
            "projector_out_dim": 4096,
            "projector_total_out_dim": 4096,
        }
        self.assertIsNone(npv8._llama_reference_output_name(config))

    def test_resolve_llama_reference_output_name_honors_explicit_named_dump(self) -> None:
        config = {
            "projector_out_dim": 4096,
            "projector_total_out_dim": 16384,
        }
        self.assertEqual(npv8._resolve_llama_reference_output_name(config, "projector_out"), "projector_out")
        self.assertIsNone(npv8._resolve_llama_reference_output_name(config, "clip_encode_float_image"))

    def test_resolve_ck_output_contract_supports_explicit_bridge_alias(self) -> None:
        layout = {
            "config": {
                "projection_dim": 4096,
                "projector_out_dim": 4096,
                "projector_total_out_dim": 16384,
                "vision_merged_tokens": 576,
            }
        }
        offsets = {
            "embedded_input": {"size_bytes": 576 * 4096 * 4},
            "vision_output": {"size_bytes": 576 * 16384 * 4},
        }
        contract = npv8._resolve_ck_output_contract(layout, offsets, "vision_bridge_output")
        self.assertEqual(contract["named_activation"], "vision_bridge_output")
        self.assertEqual(contract["fallback_buffer_name"], "embedded_input")
        self.assertEqual(contract["used_nbytes"], 576 * 4096 * 4)
        self.assertEqual(contract["resolved_output"], "vision_bridge_output")

    def test_load_runtime_metadata_recovers_config_and_weights_bump(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.json").write_text(json.dumps({"image_size": 768}), encoding="utf-8")
            (root / "weights.bump").write_bytes(b"stub")
            report = npv8._load_runtime_metadata({"metrics": {"max_abs": 1.0}}, root)
        self.assertEqual(report["config"]["image_size"], 768)
        self.assertTrue(str(report["weights_bump"]).endswith("weights.bump"))

    def test_read_named_llama_dump_tensor_flattens_named_record(self) -> None:
        fake_dump = SimpleNamespace(op_name="projector_out", data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        with mock.patch.object(npv8.parity_test, "read_dump_file", return_value=[fake_dump]):
            values = npv8._read_named_llama_dump_tensor(Path("/tmp/fake.bin"), "projector_out")
        self.assertEqual(list(values), [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
