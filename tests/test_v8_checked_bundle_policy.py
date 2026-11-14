"""Shared native build policy rejects undeclared support before lowering."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph


class CheckedBundlePolicy(unittest.TestCase):
    def test_invalid_policy_rejected_before_lowering(self):
        source = {"template": {"native_entry": {"function": "checked_entry"}}}
        invalid = [
            {"openmp": 1},
            {"fp_contract": "unsafe"},
            {"provider_warnings": "ignore"},
            {"c_standard": "c++"},
            {"runtime_sources": ["src/kernels/linear_rows_checked.c"]},
            {"runtime_sources": ["src/runtime/../../tests/missing.c"]},
            {"runtime_sources": ["src/runtime/missing.c"]},
            {"extra_exports": ["entry; *"]},
        ]
        with tempfile.TemporaryDirectory() as temp:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), patch(
                    "version.v8.scripts.compile_checked_bundle_v8.build_ir_v8.load_kernel_registry"
                ) as registry:
                    with self.assertRaises(ValueError):
                        compile_native_graph(Path(temp), source, {}, **kwargs)
                    registry.assert_not_called()


class CheckedBundleEvidence(unittest.TestCase):
    def test_warning_exceptions_are_exact_and_bounded(self):
        from version.v8.scripts.checked_bundle_evidence_v8 import (
            check_provider_warnings,
            ROOT,
        )

        source = str(ROOT / "src/kernels/attention_kernels.c")
        line = (
            source
            + ":502:12: warning: 'ck_attention_ggml_out_graph_enabled' defined but not used [-Wunused-function]"
        )
        self.assertEqual(check_provider_warnings(line)[0]["count"], 1)
        for diagnostics in (
            line + "\n" + line,
            line.replace("ck_attention_ggml_out_graph_enabled", "new_unused_helper"),
            line.replace("-Wunused-function", "-Wuninitialized"),
            "cc1: warning: unreviewed option",
            line.replace(source, "/tmp/unreviewed.c"),
        ):
            with self.subTest(diagnostics=diagnostics), self.assertRaises(ValueError):
                check_provider_warnings(diagnostics)

    def test_loaded_identity_relocation_and_negative_controls(self):
        import ctypes
        import hashlib
        import json
        import shutil
        from tests.test_v8_checked_multi_entry import circuit
        from version.v8.scripts.checked_bundle_evidence_v8 import (
            load_verified_checked_bundle,
            verify_loaded_identity,
        )

        graph = circuit()
        source = dict(
            config=dict(
                model=graph["name"],
                arch=graph["name"],
                num_layers=1,
                embed_dim=4,
                num_heads=1,
                num_kv_heads=1,
                head_dim=4,
                intermediate_size=4,
                context_length=1,
                max_seq_len=1,
                vocab_size=4,
                activation_buffer_dtypes={
                    name: "i32"
                    for name in graph["activation_buffers"]
                    if name != "scores"
                },
            ),
            entries=[],
            quant_summary={},
            template=graph,
        )
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "build"
            folder.mkdir()
            _, _, library, loaded, _ = compile_native_graph(folder, source, graph)
            identity_path = folder / "build-identity.json"
            identity = json.loads(identity_path.read_text())
            self.assertEqual(identity["embedded_runtime_build_identity"], "VERIFIED")
            sha = hashlib.sha256(identity_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                verify_loaded_identity(loaded, "0" * 64)
            old_library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
            with patch(
                "version.v8.scripts.compile_checked_bundle_v8.check_provider_warnings",
                side_effect=ValueError("unreviewed warning"),
            ):
                with self.assertRaisesRegex(ValueError, "unreviewed warning"):
                    compile_native_graph(folder, source, graph)
            self.assertEqual(
                hashlib.sha256(library.read_bytes()).hexdigest(), old_library_hash
            )
            self.assertEqual(
                hashlib.sha256(identity_path.read_bytes()).hexdigest(), sha
            )
            self.assertEqual(list(folder.glob("checked-build-*.so")), [])
            moved = Path(temp) / "relocated"
            moved.mkdir()
            shutil.copy2(library, moved / "model.so")
            shutil.copy2(identity_path, moved / "identity.json")
            shutil.rmtree(folder)
            native = load_verified_checked_bundle(
                moved / "model.so", moved / "identity.json", sha
            )
            verify_loaded_identity(native, identity["build_identity_sha256"])
            with self.assertRaisesRegex(ValueError, "metadata hash"):
                load_verified_checked_bundle(
                    moved / "model.so", moved / "identity.json", "0" * 64
                )
            changed = dict(identity, library_sha256="0" * 64)
            (moved / "bad.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "library hash"):
                load_verified_checked_bundle(
                    moved / "model.so",
                    moved / "bad.json",
                    hashlib.sha256((moved / "bad.json").read_bytes()).hexdigest(),
                )
            changed = dict(identity, build_identity_sha256="0" * 64)
            (moved / "bad.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "identity inputs"):
                load_verified_checked_bundle(
                    moved / "model.so",
                    moved / "bad.json",
                    hashlib.sha256((moved / "bad.json").read_bytes()).hexdigest(),
                )
