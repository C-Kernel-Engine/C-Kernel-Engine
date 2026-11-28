"""Exercise the generated Nemotron VL projector input with opaque BF16 words."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from version.v8.scripts.nemotron_vl_checked_v8 import (
    CompiledProjectorInput, validate_publisher_config,
)


def config():
    return {
        "model_type": "NemotronH_Nano_VL_V2",
        "force_image_size": 512, "patch_size": 16, "vit_hidden_size": 1280,
        "downsample_ratio": 0.5, "ps_version": "v2", "projector_hidden_size": 20480,
        "torch_dtype": "bfloat16", "img_context_token_id": 131072,
        "vision_config": {"version": "radio_v2.5-h"},
        "llm_config": {"model_type": "nemotron_h", "hidden_size": 5120},
    }


class NemotronProjectorInputTest(unittest.TestCase):
    def test_config_rejects_other_geometry(self):
        validate_publisher_config(config())
        for change in (
            {"ps_version": "v1"}, {"patch_size": 14}, {"downsample_ratio": 1},
            {"vision_config": None}, {"llm_config": {"hidden_size": 4096}},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_publisher_config(dict(config(), **change))

    def test_generated_packing_matches_publisher_view_permute_order(self):
        data = np.random.default_rng(17).integers(0, 65536, (1024, 1280), dtype=np.uint16)

        def reference(tile):
            # Mirrors the publisher's view/permute sequence, independent of gather indices.
            return (tile.reshape(32, 16, 2560).transpose(1, 0, 2)
                    .reshape(16, 16, 5120).transpose(1, 0, 2).reshape(256, 5120))

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "packed"
            pack = CompiledProjectorInput(directory, config())
            self.assertIn("ck_gather_rows_bf16_checked", (directory / "generated.c").read_text())
            first = pack(data)
            np.testing.assert_array_equal(first, reference(data))
            self.assertFalse(np.array_equal(first[0, 1280:2560], data[32]))
            second = np.bitwise_xor(data, np.uint16(0x8000))
            with self.assertRaises(ValueError):
                pack(data[:-1])
            with self.assertRaises(ValueError):
                pack(data.astype(np.float32))
            np.testing.assert_array_equal(pack(second), reference(second))
            np.testing.assert_array_equal(pack(data), first)
            indices = pack._view("indices", np.int32)
            last = indices[-1]
            indices[-1] = len(data)
            with self.assertRaisesRegex(RuntimeError, "packing failed"):
                pack(second)
            np.testing.assert_array_equal(pack._view("packed", np.uint16).reshape(first.shape), first)
            indices[-1] = last
            np.testing.assert_array_equal(pack(second), reference(second))


if __name__ == "__main__":
    unittest.main()
