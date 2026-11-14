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
            {"openmp": 1}, {"fp_contract": "unsafe"},
            {"provider_warnings": "ignore"}, {"c_standard": "c++"},
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
