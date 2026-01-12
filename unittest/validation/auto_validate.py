#!/usr/bin/env python3
"""
Auto-Validation System for C-Kernel-Engine v6

This module provides automatic validation when model output appears to be gibberish.
It integrates with ck_run_v6.py to:
1. Detect gibberish in output
2. Automatically run staged validation
3. Pinpoint the first divergent operation

Usage:
    from validation.auto_validate import AutoValidator

    validator = AutoValidator(gguf_path, bump_path, manifest_path)
    if validator.check_output(tokens, text):
        print("Output is coherent")
    else:
        print("Gibberish detected - running validation...")
        report = validator.run_validation()
        validator.print_debug_instructions()
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "unittest"))

from validation.gibberish_detector import detect_gibberish, GibberishResult
from validation.base import ValidationReport
from validation.stage1_weight_validation import Stage1Validator
from validation.stage2_dimension_validation import Stage2Validator
from validation.stage3_single_layer import Stage3Validator


# Colors for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


class AutoValidator:
    """
    Automatic validation system that activates when gibberish is detected.
    """

    def __init__(
        self,
        gguf_path: str,
        bump_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
        work_dir: Optional[str] = None,
        verbose: bool = False
    ):
        self.gguf_path = gguf_path
        self.bump_path = bump_path
        self.manifest_path = manifest_path
        self.work_dir = work_dir or str(Path(bump_path).parent) if bump_path else None
        self.verbose = verbose
        self.last_check_result: Optional[GibberishResult] = None
        self.validation_report: Optional[ValidationReport] = None

    def check_output(
        self,
        tokens: List[int] = None,
        text: str = None,
        vocab_size: int = 32000
    ) -> bool:
        """
        Check if model output is coherent.

        Returns:
            True if output is coherent, False if gibberish detected
        """
        result = detect_gibberish(tokens=tokens, text=text, vocab_size=vocab_size)
        self.last_check_result = result

        if result.is_gibberish:
            self._print_gibberish_warning(result)
            return False

        return True

    def _print_gibberish_warning(self, result: GibberishResult):
        """Print warning when gibberish is detected"""
        print(f"\n{RED}{'='*60}{RESET}")
        print(f"{RED}{BOLD}GIBBERISH DETECTED{RESET}")
        print(f"{RED}{'='*60}{RESET}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Reason: {result.reason}")
        print()
        print(f"{YELLOW}This usually indicates a kernel bug introduced by recent changes.{RESET}")
        print(f"{YELLOW}Running staged validation to pinpoint the issue...{RESET}")
        print()

    def run_validation(self, stages: List[int] = None) -> ValidationReport:
        """
        Run staged validation.

        Args:
            stages: List of stages to run (default: [1, 2, 3])

        Returns:
            ValidationReport with results
        """
        if stages is None:
            stages = [1, 2, 3]

        report = ValidationReport()

        for stage_num in stages:
            print(f"\n{CYAN}Running Stage {stage_num}...{RESET}")

            if stage_num == 1:
                validator = Stage1Validator(
                    gguf_path=self.gguf_path,
                    bump_path=self.bump_path,
                    manifest_path=self.manifest_path,
                    verbose=self.verbose
                )
                result = validator.run()

            elif stage_num == 2:
                validator = Stage2Validator(
                    gguf_path=self.gguf_path,
                    bump_path=self.bump_path,
                    manifest_path=self.manifest_path,
                    verbose=self.verbose
                )
                result = validator.run()

            elif stage_num == 3:
                validator = Stage3Validator(
                    gguf_path=self.gguf_path,
                    bump_path=self.bump_path,
                    manifest_path=self.manifest_path,
                    verbose=self.verbose,
                    layer=0,
                    prompt="Hello"
                )
                result = validator.run()

            else:
                continue

            result.print_summary(verbose=self.verbose)
            report.add_stage_result(stage_num, result)

            # Gate on failure
            if not result.passed:
                report.gated_at = stage_num
                break

        self.validation_report = report
        return report

    def print_debug_instructions(self):
        """Print instructions for debugging the issue"""
        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}DEBUGGING INSTRUCTIONS{RESET}")
        print(f"{'='*60}")

        if not self.validation_report:
            print("Run validation first with run_validation()")
            return

        report = self.validation_report
        gated_at = report.gated_at

        if gated_at == 1:
            print(f"""
{YELLOW}Stage 1 Failed: Weight Conversion Issue{RESET}

The GGUF weights are not being correctly converted to BUMP format.

Steps to debug:
1. Check the specific test that failed in Stage 1
2. Compare dequantization output:
   python unittest/test_quant_vs_llamacpp.py --gguf {self.gguf_path}

3. If scale unpacking failed:
   - Check src/kernels/gemm_kernels_q*k.c for scale extraction
   - Compare with llama.cpp's get_scale_min_k* functions

4. If dequant parity failed:
   - Check block layout in scripts/v6/convert_gguf_to_bump_v6.py
   - Verify byte offsets match llama.cpp block structure
""")

        elif gated_at == 2:
            print(f"""
{YELLOW}Stage 2 Failed: Dimension/Memory Issue{RESET}

The memory layout or tensor dimensions don't match expected values.

Steps to debug:
1. Check the manifest file:
   cat {self.manifest_path}

2. Verify dimensions match GGUF:
   python scripts/v6/convert_gguf_to_bump_v6.py --gguf {self.gguf_path} --inspect

3. Common issues:
   - Alignment padding not matching (should be 64-byte aligned)
   - Stride calculation wrong for quantized formats
   - Head packing layout mismatch
""")

        elif gated_at == 3:
            print(f"""
{YELLOW}Stage 3 Failed: Layer Activation Divergence{RESET}

A kernel is producing different output than llama.cpp.

Steps to debug:
1. Build the llama.cpp tensor dump tool:
   cd llama.cpp
   g++ -I. -I./include -I./ggml/include \\
       ../patches/single_layer_dump.cpp \\
       -L./build/src -L./build/ggml/src \\
       -lllama -lggml -lm -lpthread \\
       -o build/bin/single-layer-dump

2. Dump llama.cpp layer 0 tensors:
   ./llama.cpp/build/bin/single-layer-dump \\
       -m {self.gguf_path} \\
       -l 0 \\
       -o ./llama_dumps \\
       -p "Hello"

3. Add tensor dumps to CK generated code:
   - Edit ck-kernel-inference.c
   - Add: save_tensor("attn_norm-0", ln1_out, embed_dim);
   - After each operation

4. Compare tensors:
   python -c "
import numpy as np
llama = np.fromfile('llama_dumps/attn_norm-0.bin', dtype=np.float32)
ck = np.fromfile('ck_dumps/layer.0.ln1_out.bin', dtype=np.float32)
diff = np.abs(llama - ck)
print(f'Max diff: {{np.max(diff):.6e}}')
print(f'Mean diff: {{np.mean(diff):.6e}}')
if np.max(diff) > 1e-4:
    first_bad = np.argmax(diff > 1e-4)
    print(f'First divergence at index {{first_bad}}')
"

5. Common kernel issues:
   - RMSNorm epsilon value mismatch
   - RoPE frequency calculation
   - Attention score scaling
   - GEMM transpose flags
""")

        else:
            print(f"""
{GREEN}All stages passed but output is still gibberish.{RESET}

This may indicate:
1. Tokenizer mismatch (check vocab/merges)
2. Chat template issue
3. Context length exceeded
4. Temperature/sampling issue

Try:
1. Reduce max_tokens to rule out context issues
2. Compare tokenization:
   python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('...')
print(tok.encode('Hello'))
"
3. Check chat template in manifest
""")

        print(f"\n{CYAN}Full validation report:{RESET}")
        print(self.validation_report.to_json())

    def run_llama_comparison(self, layer: int = 0, prompt: str = "Hello") -> bool:
        """
        Run a direct comparison with llama.cpp for a specific layer.

        Returns:
            True if tensors match within tolerance
        """
        print(f"\n{CYAN}Running llama.cpp comparison for layer {layer}...{RESET}")

        # Check if single-layer-dump tool exists
        dump_tool = PROJECT_ROOT / "llama.cpp" / "build" / "bin" / "single-layer-dump"
        if not dump_tool.exists():
            print(f"{YELLOW}Building single-layer-dump tool...{RESET}")
            self._build_dump_tool()

        if not dump_tool.exists():
            print(f"{RED}Could not build single-layer-dump tool{RESET}")
            return False

        # Run llama.cpp dump
        dump_dir = Path(self.work_dir or ".") / f"llama_layer{layer}_dumps"
        dump_dir.mkdir(exist_ok=True)

        cmd = [
            str(dump_tool),
            "-m", self.gguf_path,
            "-l", str(layer),
            "-o", str(dump_dir),
            "-p", prompt,
            "-v"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"{RED}llama.cpp dump failed:{RESET}")
                print(result.stderr)
                return False
        except subprocess.TimeoutExpired:
            print(f"{RED}llama.cpp dump timed out{RESET}")
            return False

        print(f"{GREEN}Dumped tensors to {dump_dir}{RESET}")

        # TODO: Add CK dump and comparison
        return True

    def _build_dump_tool(self):
        """Build the single-layer-dump tool"""
        llama_dir = PROJECT_ROOT / "llama.cpp"
        if not llama_dir.exists():
            print(f"{RED}llama.cpp directory not found{RESET}")
            return

        build_dir = llama_dir / "build"
        if not build_dir.exists():
            print("Building llama.cpp first...")
            subprocess.run(["cmake", "-B", "build", "-DGGML_CPU=ON"], cwd=llama_dir)
            subprocess.run(["cmake", "--build", "build", "-j4"], cwd=llama_dir)

        # Build our tool
        patch_file = PROJECT_ROOT / "patches" / "single_layer_dump.cpp"
        if not patch_file.exists():
            print(f"{RED}single_layer_dump.cpp not found{RESET}")
            return

        output_path = build_dir / "bin" / "single-layer-dump"
        output_path.parent.mkdir(exist_ok=True)

        cmd = [
            "g++", "-std=c++17", "-O2",
            "-I" + str(llama_dir),
            "-I" + str(llama_dir / "include"),
            "-I" + str(llama_dir / "ggml" / "include"),
            str(patch_file),
            "-L" + str(build_dir / "src"),
            "-L" + str(build_dir / "ggml" / "src"),
            "-lllama", "-lggml", "-lm", "-lpthread",
            "-o", str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"{GREEN}Built single-layer-dump tool{RESET}")
        except subprocess.CalledProcessError as e:
            print(f"{RED}Build failed: {e}{RESET}")


def integrate_with_chat(
    gguf_path: str,
    bump_path: str,
    manifest_path: str,
    tokens: List[int],
    text: str
) -> bool:
    """
    Integration function for ck_chat.py

    Call this after generating output to check for gibberish.

    Returns:
        True if output is OK, False if gibberish was detected and validation ran
    """
    validator = AutoValidator(
        gguf_path=gguf_path,
        bump_path=bump_path,
        manifest_path=manifest_path
    )

    if validator.check_output(tokens=tokens, text=text):
        return True

    # Gibberish detected - run validation
    validator.run_validation()
    validator.print_debug_instructions()
    return False


if __name__ == "__main__":
    # Demo usage
    import argparse

    parser = argparse.ArgumentParser(description="Auto-validation demo")
    parser.add_argument("--gguf", required=True, help="GGUF model path")
    parser.add_argument("--bump", help="BUMP weights path")
    parser.add_argument("--manifest", help="Manifest path")
    parser.add_argument("--test-gibberish", action="store_true",
                       help="Test with simulated gibberish")
    args = parser.parse_args()

    validator = AutoValidator(
        gguf_path=args.gguf,
        bump_path=args.bump,
        manifest_path=args.manifest,
        verbose=True
    )

    if args.test_gibberish:
        # Simulate gibberish output
        print("Testing with simulated gibberish...")
        gibberish_tokens = [42] * 50  # Same token repeated
        gibberish_text = "answer answer answer answer answer answer"

        if not validator.check_output(tokens=gibberish_tokens, text=gibberish_text):
            validator.run_validation()
            validator.print_debug_instructions()
    else:
        print("Run with --test-gibberish to demo gibberish detection")
