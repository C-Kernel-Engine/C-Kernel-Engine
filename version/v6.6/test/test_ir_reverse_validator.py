#!/usr/bin/env python3
"""
test_ir_reverse_validator.py - Unit tests for IR reverse validator

Tests the validation logic that works backwards from IR Lower 3 to check:
- Buffer completeness
- Manifest coverage
- Bias accounting
- Op sequence (data flow)
- Size consistency
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Add scripts dir to path
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ir_reverse_validator import (
    IRReverseValidator,
    compute_size_from_shape_dtype,
    run_validation,
)


class TestSizeComputation(unittest.TestCase):
    """Test size computation from shape and dtype."""

    def test_fp32_size(self):
        """FP32: 1 element = 4 bytes."""
        self.assertEqual(compute_size_from_shape_dtype([100], "fp32"), 400)
        self.assertEqual(compute_size_from_shape_dtype([32, 32], "fp32"), 4096)

    def test_q4_0_size(self):
        """Q4_0: 32 values = 18 bytes (1 block)."""
        # 32 values = 1 block = 18 bytes
        self.assertEqual(compute_size_from_shape_dtype([32], "q4_0"), 18)
        # 64 values = 2 blocks = 36 bytes
        self.assertEqual(compute_size_from_shape_dtype([64], "q4_0"), 36)
        # 33 values = 2 blocks (rounded up) = 36 bytes
        self.assertEqual(compute_size_from_shape_dtype([33], "q4_0"), 36)

    def test_q8_0_size(self):
        """Q8_0: 32 values = 34 bytes."""
        self.assertEqual(compute_size_from_shape_dtype([32], "q8_0"), 34)
        self.assertEqual(compute_size_from_shape_dtype([64], "q8_0"), 68)

    def test_q4_k_size(self):
        """Q4_K: 256 values = 144 bytes."""
        self.assertEqual(compute_size_from_shape_dtype([256], "q4_k"), 144)
        self.assertEqual(compute_size_from_shape_dtype([512], "q4_k"), 288)

    def test_unknown_dtype(self):
        """Unknown dtype returns -1."""
        self.assertEqual(compute_size_from_shape_dtype([100], "unknown"), -1)


class TestBufferCompleteness(unittest.TestCase):
    """Test buffer completeness validation."""

    def test_all_buffers_defined(self):
        """All referenced buffers are defined - should pass."""
        ir = {
            "operations": [
                {
                    "kernel": "test_op",
                    "inputs": [{"buffer": "input_buf"}],
                    "outputs": [{"buffer": "output_buf"}],
                    "weights": [],
                    "biases": [],
                }
            ],
            "buffers": {
                "input_buf": {"size": 1024},
                "output_buf": {"size": 1024},
            },
        }
        validator = IRReverseValidator(ir)
        result = validator.validate_buffer_completeness()
        self.assertTrue(result.passed)

    def test_missing_buffer(self):
        """Missing buffer definition - should fail."""
        ir = {
            "operations": [
                {
                    "kernel": "test_op",
                    "inputs": [{"buffer": "missing_buf"}],
                    "outputs": [],
                    "weights": [],
                    "biases": [],
                }
            ],
            "buffers": {},
        }
        validator = IRReverseValidator(ir)
        result = validator.validate_buffer_completeness()
        self.assertFalse(result.passed)
        self.assertTrue(any("missing_buf" in e for e in result.errors))

    def test_weight_in_manifest_ok(self):
        """Weight buffer in manifest is OK even if not in buffers dict."""
        ir = {
            "operations": [
                {
                    "kernel": "test_op",
                    "inputs": [],
                    "outputs": [],
                    "weights": [{"buffer": "layer.0.wq"}],
                    "biases": [],
                }
            ],
            "buffers": {},
        }
        manifest = {
            "entries": [
                {"name": "layer.0.wq", "dtype": "q5_0", "size": 1000}
            ]
        }
        validator = IRReverseValidator(ir, manifest)
        result = validator.validate_buffer_completeness()
        self.assertTrue(result.passed)


class TestManifestCoverage(unittest.TestCase):
    """Test manifest coverage validation."""

    def test_all_manifest_used(self):
        """All manifest entries are used - should pass."""
        ir = {
            "operations": [
                {
                    "kernel": "gemv",
                    "inputs": [],
                    "outputs": [],
                    "weights": [{"buffer": "layer.0.wq"}],
                    "biases": [],
                }
            ],
            "buffers": {},
        }
        manifest = {
            "entries": [
                {"name": "layer.0.wq", "dtype": "q5_0", "size": 1000}
            ]
        }
        validator = IRReverseValidator(ir, manifest)
        result = validator.validate_manifest_coverage()
        self.assertTrue(result.passed)

    def test_unused_weight_warning(self):
        """Unused manifest entry - should warn (not fail)."""
        ir = {
            "operations": [
                {
                    "kernel": "gemv",
                    "layer": 0,  # Specify layer
                    "inputs": [],
                    "outputs": [],
                    "weights": [{"buffer": "layer.0.wq"}],
                    "biases": [],
                }
            ],
            "buffers": {},
            "config": {"num_layers": 1},  # Single layer config
        }
        manifest = {
            "entries": [
                {"name": "layer.0.wq", "dtype": "q5_0", "size": 1000},
                {"name": "layer.0.wk", "dtype": "q5_0", "size": 1000},  # Unused
            ]
        }
        validator = IRReverseValidator(ir, manifest)
        result = validator.validate_manifest_coverage()
        # Unused weights are warnings, not failures
        self.assertTrue(result.passed)
        self.assertTrue(any("layer.0.wk" in w for w in result.warnings))


class TestBiasAccounting(unittest.TestCase):
    """Test bias accounting validation."""

    def test_all_biases_used(self):
        """All biases in manifest are used - should pass."""
        ir = {
            "operations": [
                {
                    "kernel": "gemv",
                    "inputs": [],
                    "outputs": [],
                    "weights": [],
                    "biases": [{"buffer": "layer.0.bq"}],
                }
            ],
            "buffers": {},
        }
        manifest = {
            "entries": [
                {"name": "layer.0.bq", "dtype": "fp32", "size": 256}
            ]
        }
        validator = IRReverseValidator(ir, manifest)
        result = validator.validate_bias_accounting()
        self.assertTrue(result.passed)

    def test_all_biases_missing_warns(self):
        """All biases in manifest not used - should warn (no-bias model)."""
        ir = {
            "operations": [
                {
                    "kernel": "gemv",
                    "layer": 0,  # Specify layer
                    "inputs": [],
                    "outputs": [],
                    "weights": [],
                    "biases": [],  # No biases used
                }
            ],
            "buffers": {},
            "config": {"num_layers": 1},  # Single layer config
        }
        manifest = {
            "entries": [
                {"name": "layer.0.bq", "dtype": "fp32", "size": 256}
            ]
        }
        validator = IRReverseValidator(ir, manifest)
        result = validator.validate_bias_accounting()
        # When ALL biases are missing, it's treated as a no-bias model (warning, not failure)
        self.assertTrue(result.passed)
        self.assertTrue(any("NULL biases" in w for w in result.warnings))

    def test_some_biases_missing_fails(self):
        """Some biases used, some not - should fail."""
        ir = {
            "operations": [
                {
                    "kernel": "gemv",
                    "layer": 0,  # Specify layer
                    "inputs": [],
                    "outputs": [],
                    "weights": [],
                    "biases": [{"buffer": "layer.0.bq"}],  # Only bq used
                }
            ],
            "buffers": {},
            "config": {"num_layers": 1},  # Single layer config
        }
        manifest = {
            "entries": [
                {"name": "layer.0.bq", "dtype": "fp32", "size": 256},
                {"name": "layer.0.bk", "dtype": "fp32", "size": 256},  # Not used
            ]
        }
        validator = IRReverseValidator(ir, manifest)
        result = validator.validate_bias_accounting()
        self.assertFalse(result.passed)
        self.assertTrue(any("layer.0.bk" in e for e in result.errors))


class TestOpSequence(unittest.TestCase):
    """Test operation sequence (data flow) validation."""

    def test_valid_sequence(self):
        """Valid data flow - should pass."""
        ir = {
            "operations": [
                {
                    "kernel": "op1",
                    "inputs": [{"buffer": "input"}],  # External input
                    "outputs": [{"buffer": "temp"}],
                    "weights": [],
                    "biases": [],
                },
                {
                    "kernel": "op2",
                    "inputs": [{"buffer": "temp"}],  # Written by op1
                    "outputs": [{"buffer": "output"}],
                    "weights": [],
                    "biases": [],
                },
            ],
            "buffers": {"input": {}, "temp": {}, "output": {}},
        }
        validator = IRReverseValidator(ir)
        result = validator.validate_op_sequence()
        self.assertTrue(result.passed)


class TestSizeConsistency(unittest.TestCase):
    """Test size consistency validation."""

    def test_consistent_sizes(self):
        """Size matches shape+dtype - should pass."""
        manifest = {
            "entries": [
                {
                    "name": "weight",
                    "dtype": "fp32",
                    "shape": [100, 100],
                    "file_size": 40000,  # 10000 * 4 bytes
                }
            ]
        }
        validator = IRReverseValidator({}, manifest)
        result = validator.validate_size_consistency()
        self.assertTrue(result.passed)

    def test_inconsistent_size(self):
        """Size doesn't match - should fail."""
        manifest = {
            "entries": [
                {
                    "name": "weight",
                    "dtype": "fp32",
                    "shape": [100, 100],
                    "file_size": 1000,  # Wrong! Should be 40000
                }
            ]
        }
        validator = IRReverseValidator({}, manifest)
        result = validator.validate_size_consistency()
        self.assertFalse(result.passed)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""

    def test_run_validation_with_files(self):
        """Test run_validation with temporary files."""
        ir = {
            "operations": [
                {
                    "kernel": "test_kernel",
                    "inputs": [{"buffer": "input"}],
                    "outputs": [{"buffer": "output"}],
                    "weights": [{"buffer": "layer.0.w"}],
                    "biases": [],
                }
            ],
            "buffers": {"input": {}, "output": {}},
            "config": {},
        }
        manifest = {
            "entries": [
                {"name": "layer.0.w", "dtype": "fp32", "shape": [64], "file_size": 256}
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ir_path = Path(tmpdir) / "lowered.json"
            manifest_path = Path(tmpdir) / "manifest.json"

            with open(ir_path, "w") as f:
                json.dump(ir, f)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            passed, report = run_validation(ir_path, manifest_path)
            self.assertTrue(passed)
            self.assertIn("PASS", report)


if __name__ == "__main__":
    unittest.main()
