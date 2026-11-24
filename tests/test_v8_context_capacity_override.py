"""A build request must not silently shrink its declared context capacity."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"


def _load_builder():
    sys.path.insert(0, str(BUILDER.parent))
    spec = importlib.util.spec_from_file_location("build_ir_v8_context_test", BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ContextCapacityOverrideTests(unittest.TestCase):
    def test_context_outside_manifest_capacity_fails_before_lowering(self) -> None:
        for requested in (0, -1, 8):
            with self.subTest(requested=requested), tempfile.TemporaryDirectory() as root:
                tmp_path = Path(root)
                manifest = tmp_path / "weights_manifest.json"
                manifest.write_text(json.dumps({
                    "version": 5,
                    "config": {"context_length": 4, "max_seq_len": 4},
                    "template": {},
                    "entries": [],
                }))
                output = tmp_path / "ir.json"
                result = subprocess.run(
                    [sys.executable, str(BUILDER), "--manifest", str(manifest),
                     "--mode", "decode", "--context-len", str(requested),
                     "--output", str(output)],
                    cwd=ROOT, text=True, capture_output=True, timeout=30,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())
                diagnostic = json.loads((tmp_path / "build_diagnostic.json").read_text())
                self.assertEqual(diagnostic["failure"]["code"], "CKE-V8-CONTEXT-CAPACITY")
                self.assertEqual(diagnostic["failure"]["observed"], {
                    "requested_context_len": requested,
                    "declared_context_length": 4,
                })


class RecurrentRuntimeWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = _load_builder()

    def _manifest(self, **config) -> dict:
        return {"config": dict(config), "template": {"contract": {"runtime_defaults": {
            k: config[k] for k in ("context_capacity_policy", "hard_context_limit") if k in config
        }}}}

    def _resolve(self, manifest: dict, requested):
        return self.builder._resolve_context_capacity(manifest, requested)

    def test_manifest_cannot_raise_circuit_ceiling(self) -> None:
        manifest = self._manifest(context_length=64, context_capacity_policy="recurrent_runtime_window", hard_context_limit=128)
        manifest["config"]["hard_context_limit"] = 256
        with self.assertRaisesRegex(RuntimeError, "disagrees with circuit"):
            self._resolve(manifest, 256)

    def test_default_comes_from_manifest(self) -> None:
        manifest = self._manifest(context_length=25600, max_seq_len=25600)
        resolved = self._resolve(manifest, None)
        self.assertEqual(resolved["effective"], 25600)
        self.assertEqual(resolved["source"], "manifest")
        self.assertEqual(manifest["config"]["context_length"], 25600)

    def test_top_level_context_is_honored(self) -> None:
        manifest = {"config": {}, "context_length": 25600}
        self.assertEqual(self.builder._resolve_declared_context(manifest), 25600)

    def test_declared_policy_rejects_above_declared(self) -> None:
        manifest = self._manifest(context_length=25600, max_seq_len=25600)
        with self.assertRaises(self.builder.BuildDiagnosticError) as ctx:
            self._resolve(manifest, 25601)
        self.assertEqual(ctx.exception.diagnostic["code"], "CKE-V8-CONTEXT-CAPACITY")
        self.assertEqual(ctx.exception.diagnostic["observed"], {
            "requested_context_len": 25601,
            "declared_context_length": 25600,
        })

    def test_recurrent_window_accepts_up_to_hard_limit(self) -> None:
        manifest = self._manifest(
            context_length=25600,
            max_seq_len=25600,
            context_capacity_policy="recurrent_runtime_window",
            hard_context_limit=1048576,
        )
        resolved = self._resolve(manifest, 1048576)
        self.assertEqual(resolved["effective"], 1048576)
        self.assertEqual(resolved["source"], "cli_override")
        self.assertEqual(resolved["hard_limit"], 1048576)
        self.assertEqual(manifest["config"]["context_length"], 1048576)
        self.assertEqual(manifest["config"]["max_seq_len"], 1048576)

    def test_recurrent_window_rejects_above_hard_limit(self) -> None:
        manifest = self._manifest(
            context_length=25600,
            max_seq_len=25600,
            context_capacity_policy="recurrent_runtime_window",
            hard_context_limit=1048576,
        )
        with self.assertRaises(self.builder.BuildDiagnosticError) as ctx:
            self._resolve(manifest, 1048577)
        self.assertEqual(ctx.exception.diagnostic["code"], "CKE-V8-CONTEXT-CAPACITY")
        self.assertEqual(
            ctx.exception.diagnostic["observed"]["hard_context_limit"], 1048576
        )

    def test_recurrent_window_requires_hard_limit(self) -> None:
        manifest = self._manifest(
            context_length=25600,
            max_seq_len=25600,
            context_capacity_policy="recurrent_runtime_window",
        )
        with self.assertRaises(RuntimeError):
            self._resolve(manifest, 25600)

    def test_missing_declaration_fails_closed(self) -> None:
        with self.assertRaises(self.builder.BuildDiagnosticError) as ctx:
            self._resolve(self._manifest(), None)
        self.assertEqual(
            ctx.exception.diagnostic["code"], "CKE-V8-CONTEXT-UNDECLARED"
        )

    def test_conflicting_declarations_fail_closed(self) -> None:
        manifest = self._manifest(context_length=25600, max_seq_len=4096)
        with self.assertRaises(self.builder.BuildDiagnosticError) as ctx:
            self._resolve(manifest, None)
        self.assertEqual(
            ctx.exception.diagnostic["code"], "CKE-V8-CONTEXT-CONFLICT"
        )

    def test_declared_above_hard_limit_fails_closed(self) -> None:
        manifest = self._manifest(
            context_length=1048577,
            max_seq_len=1048577,
            hard_context_limit=1048576,
        )
        with self.assertRaises(self.builder.BuildDiagnosticError) as ctx:
            self._resolve(manifest, None)
        self.assertEqual(ctx.exception.diagnostic["code"], "CKE-V8-CONTEXT-CAPACITY")

    def test_sequential_decode_prefill_extent_is_one(self) -> None:
        extent = self.builder._resolve_prefill_extent(
            {"prefill_chunk_length": 1}, 1048576
        )
        self.assertEqual(extent, 1)


if __name__ == "__main__":
    unittest.main()
