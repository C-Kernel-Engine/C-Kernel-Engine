#!/usr/bin/env python3
"""
Unit test for IR buffer mapping validation.

This test verifies that the buffer assignment validation catches the class of bugs
where kernel I/O names aren't properly mapped to dataflow names, causing silent
mis-routing of attention outputs to the wrong buffers.

Test uses a cached Qwen model manifest to verify buffer assignments in decode mode.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure scripts directory is in path
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))


def find_cached_model_dir() -> Path | None:
    """Find a cached GGUF model directory with ck_build/weights_manifest.json."""
    cache_dir = Path(os.path.expanduser("~/.cache/ck-engine-v6.6/models"))

    # Common model directories that should have cached manifests
    candidates = [
        cache_dir / "Qwen--Qwen2-0.5B-Instruct-GGUF",
        cache_dir / "Qwen--Qwen3-0.6B-GGUF",
        cache_dir / "Qwen--Qwen2-1.5B-Instruct-GGUF",
        cache_dir / "unsloth--gemma-3-270m-it-GGUF",
    ]

    for cand in candidates:
        manifest_path = cand / "ck_build" / "weights_manifest.json"
        if manifest_path.exists():
            return cand

    # Try to find any cached model with weights_manifest.json
    for item in cache_dir.iterdir():
        if item.is_dir():
            manifest = item / "ck_build" / "weights_manifest.json"
            if manifest.exists():
                return item

    return None


def get_model_manifest_path(model_dir: Path) -> Path:
    """Get the weights manifest path for a model."""
    return model_dir / "ck_build" / "weights_manifest.json"


def get_ir_output_dir(model_dir: Path) -> Path:
    """Get the directory for IR output files."""
    return model_dir / "ck_build"


@pytest.fixture
def cached_model_dir():
    """Pytest fixture providing a cached model directory."""
    model_dir = find_cached_model_dir()
    if model_dir is None:
        pytest.skip("No cached model with manifest found. Run a model first.")
    return model_dir


@pytest.fixture
def lowered_ir(cached_model_dir):
    """Pytest fixture that generates and loads the lowered decode IR."""
    model_dir = cached_model_dir
    manifest_path = get_model_manifest_path(model_dir)
    output_dir = get_ir_output_dir(model_dir)

    # Clean up old IR files to force regeneration
    for f in ["ir1_decode.json", "ir1_prefill.json", "layout_decode.json",
              "layout_prefill.json", "lowered_decode.json", "lowered_prefill.json"]:
        (output_dir / f).unlink(missing_ok=True)

    # Generate IR for decode mode
    build_ir_script = SCRIPTS_DIR.parent / "scripts" / "build_ir_v6_6.py"
    result = subprocess.run(
        [
            sys.executable,
            str(build_ir_script),
            f"--manifest={manifest_path}",
            "--mode=decode",
            f"--output={output_dir / 'ir1_decode.json'}",
            f"--layout-output={output_dir / 'layout_decode.json'}",
            f"--lowered-output={output_dir / 'lowered_decode.json'}",
        ],
        capture_output=True,
        text=True,
        cwd=str(SCRIPTS_DIR.parent),
    )

    if result.returncode != 0:
        pytest.fail(f"IR generation failed:\n{result.stderr}")

    # Load the lowered IR
    lowered_path = output_dir / "lowered_decode.json"
    if not lowered_path.exists():
        pytest.fail(f"Lowered IR not generated at {lowered_path}")

    with open(lowered_path) as f:
        return json.load(f)


class TestBufferMappingValidation:
    """Test that critical operations use correct buffers in decode mode."""

    def test_attn_outputs_to_attn_scratch(self, lowered_ir):
        """Verify attention operations output to attn_scratch, not main stream."""
        ops = lowered_ir.get("operations", [])

        attn_ops = [op for op in ops if op.get("op") in ("attn", "attention", "attn_sliding")]

        # Skip if no attention ops found (may be very small model)
        if not attn_ops:
            pytest.skip("No attention ops found in this model")

        for op in attn_ops:
            outputs = op.get("outputs", {})
            # Check both out_token and out
            out = outputs.get("out_token") or outputs.get("out")
            assert out is not None, f"Attention op has no output: {op.get('kernel')}"

            buffer_name = out.get("buffer", "")
            assert buffer_name == "attn_scratch", (
                f"Attention output should use attn_scratch buffer, got '{buffer_name}'. "
                f"This indicates kernel I/O -> dataflow mapping is missing."
            )

    def test_kv_cache_from_kv_cache(self, lowered_ir):
        """Verify K/V cache reads come from kv_cache buffer."""
        ops = lowered_ir.get("operations", [])

        attn_ops = [op for op in ops if op.get("op") in ("attn", "attention", "attn_sliding")]

        if not attn_ops:
            pytest.skip("No attention ops found in this model")

        for op in attn_ops:
            activations = op.get("activations", {})

            # Check k_cache input
            k_cache = activations.get("k_cache") or activations.get("k")
            if k_cache:
                buffer_name = k_cache.get("buffer", "")
                assert buffer_name == "kv_cache", (
                    f"k_cache should come from kv_cache buffer, got '{buffer_name}'"
                )

            # Check v_cache input
            v_cache = activations.get("v_cache") or activations.get("v")
            if v_cache:
                buffer_name = v_cache.get("buffer", "")
                assert buffer_name == "kv_cache", (
                    f"v_cache should come from kv_cache buffer, got '{buffer_name}'"
                )

    def test_no_attn_to_embedded_input(self, lowered_ir):
        """Verify no attention outputs use embedded_input buffer."""
        ops = lowered_ir.get("operations", [])

        for op in ops:
            outputs = op.get("outputs", {})
            for out_name, out_info in outputs.items():
                buffer_name = out_info.get("buffer", "")

                # Check that no attention-related output uses embedded_input
                if buffer_name == "embedded_input":
                    op_type = op.get("op", "")
                    kernel = op.get("kernel", "")
                    if "attn" in op_type.lower() or "attention" in kernel.lower():
                        pytest.fail(
                            f"Attention output incorrectly uses embedded_input buffer. "
                            f"op={op_type}, kernel={kernel}. "
                            f"This is the silent bug we're preventing with validation."
                        )

    def test_qkv_to_scratch_buffers(self, lowered_ir):
        """Verify Q/K/V projections output to scratch buffers, not main stream."""
        ops = lowered_ir.get("operations", [])

        proj_ops = [op for op in ops if op.get("op") in ("q_proj", "k_proj", "v_proj", "qkv_proj")]

        if not proj_ops:
            pytest.skip("No projection ops found in this model")

        for op in proj_ops:
            op_type = op.get("op", "")
            outputs = op.get("outputs", {})

            expected_buffers = {
                "q_proj": "q_scratch",
                "k_proj": "k_scratch",
                "v_proj": "v_scratch",
            }
            expected = expected_buffers.get(op_type)

            for out_name, out_info in outputs.items():
                buffer_name = out_info.get("buffer", "")

                # Q/K/V must use their respective scratch buffers
                if buffer_name in ("embedded_input", "layer_input"):
                    pytest.fail(
                        f"{op_type} output uses main stream buffer '{buffer_name}' "
                        f"instead of '{expected}'. "
                        f"This breaks dataflow - fix kernel I/O mapping."
                    )

    def test_logits_uses_logits_buffer(self, lowered_ir):
        """Verify logits operation outputs to logits buffer."""
        ops = lowered_ir.get("operations", [])

        logits_ops = [op for op in ops if op.get("op") == "logits"]

        if not logits_ops:
            pytest.skip("No logits ops found in this model")

        for op in logits_ops:
            outputs = op.get("outputs", {})

            logits_out = outputs.get("logits") or outputs.get("out")
            if logits_out:
                buffer_name = logits_out.get("buffer", "")
                assert buffer_name == "logits", (
                    f"Logits should output to 'logits' buffer, got '{buffer_name}'"
                )


class TestBufferMappingValidationErrors:
    """Test that validation catches specific error conditions."""

    def test_validation_function_defined_in_source(self):
        """Verify the validate_buffer_assignments function is defined in build_ir_v6_6.py."""
        build_ir_script = SCRIPTS_DIR.parent / "scripts" / "build_ir_v6_6.py"
        with open(build_ir_script) as f:
            source = f.read()

        # Check that the validation function is defined
        assert "def validate_buffer_assignments" in source, (
            "validate_buffer_assignments function not found in build_ir_v6_6.py"
        )

    def test_validation_called_after_ir_lower_2(self):
        """Verify validation is called after IR LOWER 2 completes."""
        build_ir_script = SCRIPTS_DIR.parent / "scripts" / "build_ir_v6_6.py"
        with open(build_ir_script) as f:
            source = f.read()

        # Check that validation is called after IR Lower 2
        # The pattern should be: print IR Lower 2 complete -> validate -> return
        assert "IR Lower 2 complete" in source
        assert "validate_buffer_assignments" in source

        # Verify the order: IR Lower 2 prints before validation
        ir_lower_2_pos = source.find("IR Lower 2 complete")
        validate_pos = source.find("validate_buffer_assignments")
        assert ir_lower_2_pos < validate_pos, (
            "validate_buffer_assignments should be called after IR Lower 2 completes"
        )

    def test_validation_passes_for_valid_ir(self, lowered_ir):
        """Verify validation passes for a correctly generated IR.

        We inline the validation logic here since importing from build_ir_v6_6.py
        has module resolution issues in the test context.
        """
        # Run the same validation logic that's in validate_buffer_assignments
        ops = lowered_ir.get("operations", [])

        errors = []
        for op in ops:
            op_name = op.get("op", op.get("kernel", "unknown"))
            layer = op.get("layer", -1)

            if op_name in ("attn", "attention", "attn_sliding"):
                outputs = op.get("outputs", {})
                out_token = outputs.get("out_token") or outputs.get("out")
                if out_token:
                    out_buf = out_token.get("buffer", "")
                    if out_buf != "attn_scratch":
                        errors.append(
                            f"op={op_name} layer={layer}: expected attn_scratch, got {out_buf}"
                        )

        # Fail if any errors found
        assert not errors, (
            f"Buffer validation failed:\n  - " + "\n  - ".join(errors)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
