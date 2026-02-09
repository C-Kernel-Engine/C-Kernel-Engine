#!/usr/bin/env python3
"""
Test kernel bindings generate correct C code.
"""
import pytest
import json
from pathlib import Path


class TestKernelBindings:
    """Test kernel bindings."""

    @pytest.fixture
    def bindings_data(self):
        """Load kernel bindings."""
        bindings_path = Path(__file__).parent.parent / "kernel_maps" / "kernel_bindings.json"
        if not bindings_path.exists():
            pytest.skip("kernel_bindings.json not found")
        with open(bindings_path) as f:
            return json.load(f)

    @pytest.fixture
    def bindings(self, bindings_data):
        """Return the bindings sub-dictionary (skipping metadata)."""
        # bindings_data has _meta at top level, actual bindings are in bindings_data["bindings"]
        if "bindings" in bindings_data:
            return bindings_data["bindings"]
        # If no "bindings" key, skip the _meta key and return remaining items
        return {k: v for k, v in bindings_data.items() if not k.startswith("_")}

    def test_all_bindings_have_params(self, bindings):
        """Every binding should have params."""
        for name, binding in bindings.items():
            # Skip metadata keys
            if name.startswith("_"):
                continue
            assert "params" in binding, f"Binding {name} missing params"
            assert len(binding["params"]) > 0, f"Binding {name} has empty params"

    def test_param_names_unique(self, bindings):
        """Param names within binding should be unique."""
        for name, binding in bindings.items():
            # Skip metadata keys
            if name.startswith("_"):
                continue
            param_names = [p.get("name") for p in binding.get("params", [])]
            assert len(param_names) == len(set(param_names)), \
                f"Binding {name} has duplicate param names"

    def test_source_prefixes_valid(self, bindings):
        """All source prefixes should be recognized."""
        valid_prefixes = {
            "activation:", "weight:", "output:",
            "dim:", "runtime:", "const:", "param:",
            "function:", "scratch:", "null:", "dtype:",
            "dtype_weight:", "weight_f:"
        }
        for name, binding in bindings.items():
            # Skip metadata keys
            if name.startswith("_"):
                continue
            for param in binding.get("params", []):
                source = param.get("source", "")
                prefix = source.split(":")[0] + ":"
                assert prefix in valid_prefixes, \
                    f"Binding {name} has invalid source prefix: {source}"

    def test_cast_types_valid(self, bindings):
        """All cast types should be valid C types."""
        valid_casts = {
            "float*", "const float*", "void*", "const void*",
            "int32_t*", "const int32_t*",
            "uint16_t*", "const uint16_t*",
            "float", "int32_t", "uint16_t", "void"
        }
        for name, binding in bindings.items():
            # Skip metadata keys
            if name.startswith("_"):
                continue
            for param in binding.get("params", []):
                cast = param.get("cast", "")
                if cast:
                    assert cast in valid_casts, \
                        f"Binding {name} has invalid cast: {cast}"

    def test_binding_ids_valid(self, bindings):
        """All binding IDs should be valid strings."""
        for name in bindings.keys():
            # Skip metadata keys
            if name.startswith("_"):
                continue
            assert isinstance(name, str), f"Binding ID {name} is not a string"
            assert len(name) > 0, "Binding ID is empty"

    def test_output_bindings_have_dst(self, bindings):
        """Output bindings should have dst field."""
        for name, binding in bindings.items():
            # Skip metadata keys
            if name.startswith("_"):
                continue
            for param in binding.get("params", []):
                source = param.get("source", "")
                if source.startswith("output:"):
                    assert "dst" in param or "cast" in param, \
                        f"Output binding {name} missing dst/cast"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
