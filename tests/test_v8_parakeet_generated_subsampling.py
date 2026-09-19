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


codegen = _load("parakeet_subsampling_codegen", SCRIPTS / "codegen_v8.py")
certifier = _load(
    "parakeet_subsampling_certifier",
    SCRIPTS / "certify_parakeet_generated_subsampling_v8.py",
)


class GeneratedParakeetSubsamplingTests(unittest.TestCase):
    def test_component_scope_uses_dedicated_entrypoint(self) -> None:
        ir = {"config": {"artifact_scope": "audio_subsampling", "prefill_policy": "batched"}}
        self.assertFalse(codegen._uses_generated_batched_prefill(ir, ir))

    def test_entrypoint_uses_resolved_call_and_runtime_extents(self) -> None:
        sources = [
            "activation:features",
            *(f"weight:w{index}" for index in range(12)),
            "output:output",
            "scratch:workspace",
            "scratch_size:workspace",
            "dim:feature_frames",
            "dim:live_frames",
            "dim:feature_channels",
            "dim:conv_channels",
            "dim:hidden_size",
            "dim:kernel_size",
            "dim:stride",
            "dim:output_capacity_frames",
            "runtime:audio_subsampling_output_frames",
        ]
        operation = {
            "op": "audio_fastconformer_subsampling",
            "function": "resolved_subsampling",
            "args": [
                {"name": f"arg_{index}", "source": source, "expr": f"expr_{index}"}
                for index, source in enumerate(sources)
            ],
        }
        generated = codegen._emit_audio_subsampling_entrypoint(
            [operation],
            {
                "audio_feature_frames": 744,
                "audio_feature_channels": 128,
                "audio_subsampling_conv_channels": 256,
                "audio_subsampling_kernel_size": 3,
                "audio_subsampling_stride": 2,
            },
        )
        self.assertIn("CK_EXPORT int ck_model_run_audio_subsampling", generated)
        self.assertIn("return resolved_subsampling(audio_features", generated)
        self.assertIn("audio_feature_frames > 744", generated)
        self.assertIn("output_capacity_frames", generated)

    def test_fixture_validation_rejects_noncontiguous_live_mask(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture.npz"
            np.savez(
                path,
                **{
                    "frontend.input_features": np.zeros((1, 4, 8), dtype=np.float32),
                    "frontend.attention_mask": np.array([[True, False, True, False]]),
                    "encoder.subsampling": np.zeros((1, 1, 6), dtype=np.float32),
                },
            )
            with self.assertRaisesRegex(ValueError, "contiguous live prefix"):
                certifier._checked_fixture(path)

    def test_bundle_validation_rejects_stale_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = Path(td)
            output = runtime / "model_v8.c"
            output.write_text("first\n", encoding="utf-8")
            identity = certifier.frontend._identity(output)
            bundle = {
                "outputs": {
                    "model_v8.c": {
                        "path": str(output),
                        "size": identity["bytes"],
                        "sha256": identity["sha256"],
                    }
                }
            }
            (runtime / ".ck_codegen_bundle.json").write_text(
                json.dumps(bundle), encoding="utf-8"
            )
            certifier._validate_bundle_outputs(runtime, ".ck_codegen_bundle.json")
            output.write_text("second\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale generated runtime output"):
                certifier._validate_bundle_outputs(runtime, ".ck_codegen_bundle.json")


if __name__ == "__main__":
    unittest.main()
