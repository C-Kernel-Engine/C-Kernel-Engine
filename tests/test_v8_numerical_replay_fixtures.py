from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from version.v8.scripts.run_numerical_replay_fixtures_v8 import (
    ReplayError,
    adapter_environment,
    load_manifest,
    run_suite,
    verify_artifacts,
)


class NumericalReplayFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cke_numerical_replay_")
        self.root = Path(self.temporary.name)
        self.activation = self._artifact("activation.bin", b"activation")
        self.weights = self._artifact("weights.bin", b"weights")
        self.expected = self._artifact("expected.bin", b"expected")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _artifact(self, name: str, data: bytes) -> dict[str, object]:
        path = self.root / name
        path.write_bytes(data)
        return {
            "path": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    def _manifest(self, adapter: str = "q4_k_q8_k_production") -> dict[str, object]:
        fixture: dict[str, object] = {
            "id": "image5.layer0.attn_out",
            "adapter": adapter,
            "artifacts": {
                "activations": "activation",
                "weights": "weights",
                "expected_output": "expected",
            },
            "shape": {"m": 1, "n": 1024, "k": 4096},
            "threads": [1, 20],
            "expected": {
                "oracle": "llama.cpp_production_graph",
                "comparison": "bit_exact",
            },
        }
        if adapter == "q4_k_q8_k_production":
            fixture["weight_offset"] = 4096
        return {
            "schema_version": 1,
            "suite_id": "qwen3vl-private-boundaries",
            "artifacts": {
                "activation": self.activation,
                "weights": self.weights,
                "expected": self.expected,
            },
            "fixtures": [fixture],
        }

    def _write_manifest(self, payload: dict[str, object]) -> Path:
        path = self.root / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _args(self, manifest: Path | None) -> Namespace:
        binary = self.root / "oracle"
        binary.write_bytes(b"oracle")
        binary.chmod(0o700)
        return Namespace(
            manifest=manifest,
            q4_binary=binary,
            q6_binary=binary,
            report=self.root / "report.json",
        )

    def test_unconfigured_suite_is_truthful_skip(self) -> None:
        result = run_suite(self._args(None))
        report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "SKIP")

    def test_unknown_fields_fail_schema_validation(self) -> None:
        payload = self._manifest()
        payload["command"] = "never execute manifest commands"
        with self.assertRaisesRegex(ReplayError, "schema violation"):
            load_manifest(self._write_manifest(payload))

    def test_duplicate_fixture_ids_fail(self) -> None:
        payload = self._manifest()
        payload["fixtures"].append(dict(payload["fixtures"][0]))  # type: ignore[index,union-attr]
        with self.assertRaisesRegex(ReplayError, "unique"):
            load_manifest(self._write_manifest(payload))

    def test_unknown_artifact_reference_fails(self) -> None:
        payload = self._manifest()
        payload["fixtures"][0]["artifacts"]["weights"] = "missing"  # type: ignore[index]
        with self.assertRaisesRegex(ReplayError, "unknown artifacts"):
            load_manifest(self._write_manifest(payload))

    def test_hash_mismatch_fails_before_execution(self) -> None:
        path = self._write_manifest(self._manifest())
        manifest = load_manifest(path)
        (self.root / "activation.bin").write_bytes(b"changed-data")
        with self.assertRaisesRegex(ReplayError, "size mismatch|SHA-256 mismatch"):
            verify_artifacts(path, manifest)

    def test_q6_rejects_multirow_fixture(self) -> None:
        payload = self._manifest("q6_k_q8_k_production")
        payload["fixtures"][0]["shape"]["m"] = 4  # type: ignore[index]
        with self.assertRaisesRegex(ReplayError, "requires m=1"):
            load_manifest(self._write_manifest(payload))

    def test_q6_rejects_q4_container_offsets(self) -> None:
        payload = self._manifest("q6_k_q8_k_production")
        payload["fixtures"][0]["weight_offset"] = 12  # type: ignore[index]
        with self.assertRaisesRegex(ReplayError, "cannot use offsets"):
            load_manifest(self._write_manifest(payload))

    def test_adapter_exports_only_registered_environment(self) -> None:
        path = self._write_manifest(self._manifest())
        manifest = load_manifest(path)
        artifacts = verify_artifacts(path, manifest)
        env = adapter_environment(manifest["fixtures"][0], artifacts)
        self.assertEqual(env["CK_Q4K_Q8K_REAL_M"], "1")
        self.assertEqual(env["CK_Q4K_Q8K_REAL_WEIGHT_OFFSET"], "4096")
        self.assertNotIn("command", env)

    @patch("version.v8.scripts.run_numerical_replay_fixtures_v8.subprocess.run")
    def test_thread_matrix_and_sanitized_report(self, run_mock) -> None:
        run_mock.return_value.returncode = 0
        with patch.dict(os.environ, {"CK_Q4K_Q8K_REAL_BIAS_OFFSET": "stale"}):
            result = run_suite(self._args(self._write_manifest(self._manifest())))
        report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual([run["threads"] for run in report["fixtures"][0]["runs"]], [1, 20])
        self.assertNotIn(str(self.root), json.dumps(report))
        self.assertEqual(run_mock.call_count, 2)
        self.assertNotIn("CK_Q4K_Q8K_REAL_BIAS_OFFSET", run_mock.call_args.kwargs["env"])

    @patch("version.v8.scripts.run_numerical_replay_fixtures_v8.subprocess.run")
    def test_oracle_failure_fails_suite_without_tolerance(self, run_mock) -> None:
        run_mock.return_value.returncode = 1
        result = run_suite(self._args(self._write_manifest(self._manifest())))
        report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 1)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["fixtures"][0]["runs"][0]["returncode"], 1)


if __name__ == "__main__":
    unittest.main()
