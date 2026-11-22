"""Operation replay packets reject stale or semantically inconsistent evidence."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from version.v8.scripts.xray_operation_bundle_v8 import BundleError, load_bundle, write_bundle
from version.v8.scripts.replay_audio_stft_bundle_v8 import validate_stft_bundle, replay


class OperationBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = self.root / "libtest.so"
        self.runtime.write_bytes(b"fixed test runtime")

    def tearDown(self):
        self.temp.cleanup()

    def bundle(self):
        return write_bundle(
            self.root / "bundle", operation="projection", provider="test_provider",
            contract="bf16_storage_fp32_compute", runtime=self.runtime,
            context={"layer": 0, "valid_rows": 2},
            tensors={
                "input": (np.array([0x3f80, 0x4000], dtype=np.uint16),
                          "bf16_bits", ("reduction",)),
                "packed_weight": (np.array([[0, 60, 128, 255]], dtype=np.uint8),
                                  "u8", ("channel", "packed_byte")),
                "output": (np.array([[1.0], [2.0]], dtype=np.float32),
                           "fp32", ("row", "channel")),
            },
        )

    def test_round_trip_and_byte_tampering(self):
        bundle = self.bundle()
        manifest, tensors = load_bundle(bundle)
        self.assertEqual(bundle.parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual(bundle.stat().st_mode & 0o777, 0o600)
        self.assertEqual((bundle.parent / 'input.bin').stat().st_mode & 0o777, 0o600)
        self.assertEqual(manifest["provider"], "test_provider")
        np.testing.assert_array_equal(tensors["input"], [0x3f80, 0x4000])
        self.assertEqual(tensors["packed_weight"].dtype, np.dtype("u1"))
        np.testing.assert_array_equal(tensors["packed_weight"], [[0, 60, 128, 255]])
        path = bundle.parent / "packed_weight.bin"
        changed = bytearray(path.read_bytes())
        changed[0] ^= 1
        path.write_bytes(changed)
        with self.assertRaisesRegex(BundleError, "tensor bytes changed"):
            load_bundle(bundle)

    def test_stale_runtime_and_invalid_geometry(self):
        bundle = self.bundle()
        self.runtime.write_bytes(b"different runtime")
        with self.assertRaisesRegex(BundleError, "runtime library identity changed"):
            load_bundle(bundle)
        self.runtime.write_bytes(b"fixed test runtime")
        manifest = json.loads(bundle.read_text())
        manifest["tensors"]["output"]["shape"] = [3, 1]
        bundle.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(BundleError, "inconsistent byte extent"):
            load_bundle(bundle)

    def test_embedded_runtime_survives_packet_relocation(self):
        packet = write_bundle(
            self.root / 'embedded', operation='projection',
            provider='test_provider', contract='fp32', runtime=self.runtime,
            context={}, embed_runtime=True,
            tensors={'input': (np.array([1.], dtype=np.float32),
                               'fp32', ('channel',))})
        self.runtime.unlink()
        relocated = self.root / 'moved'
        shutil.move(str(packet.parent), relocated)
        manifest, _ = load_bundle(relocated / 'bundle.json')
        self.assertEqual(manifest['runtime']['path'], 'runtime.so')
        self.assertEqual((relocated / 'runtime.so').stat().st_mode & 0o777, 0o600)
        manifest['runtime']['path'] = '../libtest.so'
        (relocated / 'bundle.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(BundleError, 'invalid embedded runtime path'):
            load_bundle(relocated / 'bundle.json')

    def test_no_implicit_storage_cast_or_overwrite(self):
        with self.assertRaisesRegex(BundleError, "storage dtype"):
            write_bundle(
                self.root / "bad", operation="projection", provider="test_provider",
                contract="fp16", runtime=self.runtime, context={},
                tensors={"input": (np.array([1.0], dtype=np.float32),
                                   "fp16", ("channel",))},
            )
        bundle = self.bundle()
        with self.assertRaisesRegex(BundleError, "must be empty"):
            self.bundle()
        self.assertTrue(bundle.exists())

    def test_public_empty_packet_directory_rejected(self):
        directory = self.root / 'public'
        directory.mkdir(mode=0o755)
        os.chmod(directory, 0o755)
        with self.assertRaisesRegex(BundleError, 'must be private'):
            write_bundle(directory, operation='test', provider='test',
                         contract='test', runtime=self.runtime, context={},
                         tensors={'input': (np.array([1.], dtype=np.float32),
                                            'fp32', ('channel',))})

    def stft_bundle(self):
        fft, frames, stride = 20, 5, 7
        z = lambda shape: np.zeros(shape, dtype=np.float32)
        return write_bundle(
            self.root / "stft", operation="audio_stft_mag_phase_checked",
            provider="audio_stft_mag_phase_checked_f32",
            contract="audio_stft_mag_phase_direct_f64_accum_fp32",
            runtime=self.runtime,
            context={"samples_capacity": 20, "samples_count": 20,
                     "window_capacity": fft, "table_capacity": 220,
                     "output_capacity": 22 * stride, "output_stride": stride,
                     "scratch_capacity": 22 * frames, "n_fft": fft,
                     "hop": 5, "frames": frames, "state": "stateless",
                     "reference_identity": {"model_pin": "test", "code_pin": "test",
                                            "asset_sha256": {"config.json": "a" * 64}}},
            tensors={"samples": (z(20), "fp32", ("sample",)),
                     "reference_samples": (z(20), "fp32", ("sample",)),
                     "window": (z(20), "fp32", ("tap",)),
                     "cos_table": (z((11, 20)), "fp32", ("bin", "tap")),
                     "sin_table": (z((11, 20)), "fp32", ("bin", "tap")),
                     "cke_output": (z((22, stride)), "fp32",
                                    ("channel", "physical_frame")),
                     "reference_output": (z((22, frames)), "fp32",
                                          ("channel", "valid_frame"))})

    def test_stft_packet_validates_abi_before_library_load(self):
        packet = self.stft_bundle()
        validate_stft_bundle(packet)
        original = json.loads(packet.read_text())
        for location, value, message in (
                (("provider",), "wrong", "provider or numerical contract"),
                (("contract",), "wrong", "provider or numerical contract"),
                (("context", "frames"), 4, "length, stride or capacity"),
                (("context", "output_stride"), 4, "length, stride or capacity"),
                (("context", "state"), "dirty", "no recurrent or RNG state"),
                (("context", "reference_identity", "asset_sha256"), "test",
                 "lacks reference identity"),
                (("context", "reference_identity", "asset_sha256"), {},
                 "lacks reference identity"),
                (("context", "reference_identity", "asset_sha256"),
                 {"config.json": "invalid"}, "lacks reference identity"),
                (("tensors", "cke_output", "axes"), ["frame", "channel"],
                 "physical layout labels")):
            altered = json.loads(json.dumps(original))
            field = altered
            for key in location[:-1]:
                field = field[key]
            field[location[-1]] = value
            packet.write_text(json.dumps(altered))
            with self.assertRaisesRegex(BundleError, message):
                replay(packet)
        packet.write_text(json.dumps(original))
        validate_stft_bundle(packet)
        with self.assertRaisesRegex(BundleError, 'trusted runtime path'):
            replay(packet)
        different = self.root / 'different.so'
        different.write_bytes(b'not the captured library')
        with self.assertRaisesRegex(BundleError, 'chosen runtime differs'):
            replay(packet, runtime=different)


if __name__ == "__main__":
    unittest.main()
