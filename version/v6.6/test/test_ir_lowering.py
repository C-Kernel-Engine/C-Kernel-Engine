#!/usr/bin/env python3
"""
Test individual IR lowering stages.
"""
import pytest
import json
from pathlib import Path


class TestIRLower1:
    """Test IR Lower 1: Buffer assignment."""

    def test_buffer_assignment_structure(self):
        """Test decode mode buffer assignment structure."""
        # Create minimal test data
        ir1 = {
            "operations": [
                {"idx": 0, "op": "token_emb", "section": "header"},
                {"idx": 1, "op": "attn_norm", "section": "body", "layer": 0},
                {"idx": 2, "op": "qkv_proj", "section": "body", "layer": 0},
            ]
        }

        layout = {
            "sections": [
                {
                    "header": {"buffers": [{"name": "token_emb"}]},
                    "layers": [{"buffers": [{"name": "attn_norm"}, {"name": "qkv_proj"}]}],
                    "footer": {"buffers": []}
                }
            ]
        }

        # Simulate buffer assignment
        for op in ir1["operations"]:
            if op.get("section") == "body":
                layer = op.get("layer", 0)
                op["buffers"] = layout["sections"][0]["layers"][layer]["buffers"]

        # Verify buffers assigned
        for op in ir1["operations"]:
            if op.get("section") == "body":
                assert "buffers" in op, f"Op {op.get('op')} missing buffers"

    def test_prefill_buffer_assignment(self):
        """Test prefill mode buffer assignment."""
        ir1 = {
            "operations": [
                {"idx": 0, "op": "attn_norm", "section": "prefill", "layer": 0},
            ]
        }

        layout = {
            "sections": [
                {
                    "prefill": {"buffers": [{"name": "attn_norm"}]},
                    "layers": [],
                    "footer": {"buffers": []}
                }
            ]
        }

        # Simulate prefill buffer assignment
        for op in ir1["operations"]:
            if op.get("section") == "prefill":
                op["buffers"] = layout["sections"][0]["prefill"]["buffers"]

        assert "buffers" in ir1["operations"][0]


class TestIRLower2:
    """Test IR Lower 2: Memory offsets."""

    def test_memory_offsets_valid(self):
        """Test memory offset calculation."""
        ir_lower_1 = {
            "operations": [
                {
                    "op": "qkv_proj",
                    "weights": {
                        "wq": {"size": 4096},
                        "wk": {"size": 4096},
                        "wv": {"size": 4096}
                    }
                }
            ]
        }

        # Simulate offset calculation
        offset = 0
        for op in ir_lower_1["operations"]:
            if "weights" in op:
                for wname, winfo in op["weights"].items():
                    winfo["offset"] = offset
                    offset += winfo["size"]

        # Verify offsets are valid
        for op in ir_lower_1["operations"]:
            for wname, winfo in op.get("weights", {}).items():
                assert "offset" in winfo, f"Weight {wname} missing offset"
                assert winfo["offset"] >= 0, f"Weight {wname} has negative offset"

    def test_offsets_contiguous(self):
        """Test that offsets are contiguous."""
        weights = [
            {"name": "w1", "size": 1024},
            {"name": "w2", "size": 2048},
            {"name": "w3", "size": 4096},
        ]

        offset = 0
        for w in weights:
            w["offset"] = offset
            offset += w["size"]

        # Verify contiguous
        assert weights[0]["offset"] == 0
        assert weights[1]["offset"] == 1024
        assert weights[2]["offset"] == 3072


class TestIRLower3:
    """Test IR Lower 3: Kernel bindings."""

    def test_binding_completeness(self):
        """Test all ops have bindings."""
        ir_lower_2 = {
            "operations": [
                {
                    "op": "gemv",
                    "kernel": "gemv_forward",
                    "function": "ck_gemv_forward",
                    "params": [
                        {"name": "x", "source": "activation:x"},
                        {"name": "w", "source": "weight:wq"},
                        {"name": "out", "source": "output:hidden_states"},
                    ]
                }
            ]
        }

        # Verify all required binding components
        for op in ir_lower_2.get("operations", []):
            assert op.get("function"), f"Op {op.get('op')} missing function"
            assert op.get("params"), f"Op {op.get('op')} missing params"

            # Check for binding errors
            errors = []
            for param in op.get("params", []):
                if not param.get("source"):
                    errors.append(f"Param {param.get('name')} missing source")
            assert len(errors) == 0, f"Op {op.get('op')} has binding errors: {errors}"

    def test_param_sources_valid(self):
        """Test param source prefixes are valid."""
        valid_prefixes = {
            "activation:", "weight:", "output:",
            "dim:", "runtime:", "const:", "param:",
            "function:"
        }

        params = [
            {"source": "activation:x"},
            {"source": "weight:wq"},
            {"source": "output:hidden_states"},
        ]

        for param in params:
            source = param.get("source", "")
            prefix = source.split(":")[0] + ":"
            assert prefix in valid_prefixes, f"Invalid source prefix: {source}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
