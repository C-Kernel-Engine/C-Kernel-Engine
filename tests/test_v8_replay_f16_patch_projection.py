from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
from replay_f16_patch_projection_v8 import patch_rows, replay


class PatchProjectionReplayTests(unittest.TestCase):
    def test_native_replay_preserves_layout_and_fp32_activations(self):
        cc = shutil.which("cc")
        if cc is None:
            self.skipTest("C compiler unavailable")
        pixels = (np.arange(2 * 3 * 4 * 9, dtype=np.float32) + 1) / 137
        pixels = pixels.reshape(2, 3, 4, 9)
        weights = np.arange(2 * 3 * 2 * 3, dtype=np.float16).reshape(2, 3, 2, 3) / 16 - 1
        reference = np.empty((2, 2, 2, 3), dtype=np.float32)
        rounded = np.empty_like(reference)
        for b, out, h, w in np.ndindex(reference.shape):
            window = pixels[b, :, h * 2:h * 2 + 2, w * 3:w * 3 + 3]
            reference[b, out, h, w] = np.sum(window.astype(np.float64) * weights[out])
            rounded[b, out, h, w] = np.sum(window.astype(np.float16).astype(np.float64) * weights[out])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            symbols = root / "exports.map"
            symbols.write_text("{ global: gemm_f16; local: *; };\n", encoding="ascii")
            engine = root / "patch.so"
            subprocess.run([
                cc, "-O2", "-fPIC", "-shared", "-ffunction-sections", "-fdata-sections",
                "-I", str(ROOT / "include"), str(ROOT / "src/kernels/gemm_kernels_f16.c"),
                f"-Wl,--gc-sections,--version-script={symbols},-z,defs",
                "-lm", "-ldl", "-o", str(engine),
            ], check=True, capture_output=True, text=True)
            actual = replay(pixels, weights, engine)
        np.testing.assert_allclose(actual, reference, rtol=0, atol=3e-6)
        self.assertGreater(float(np.max(np.abs(actual - rounded))), 1e-4)

    def test_rejects_geometry_dtype_and_channel_mismatch_before_loading_engine(self):
        pixels = np.ones((1, 2, 4, 9), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "exactly tile"):
            patch_rows(pixels, 2, 4)
        with self.assertRaisesRegex(ValueError, "FP32 NCHW"):
            patch_rows(pixels.astype(np.float16), 2, 3)
        weights = np.ones((1, 3, 2, 3), dtype=np.float16)
        with self.assertRaisesRegex(ValueError, "mismatched channels"):
            replay(pixels, weights, Path("not-loaded.so"))


if __name__ == "__main__":
    unittest.main()
