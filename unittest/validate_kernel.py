#!/usr/bin/env python3
"""
Staged Kernel Validation System for C-Kernel-Engine v6

This script provides a systematic, layered approach to validate kernels
after code changes. Run it to pinpoint exactly where things go wrong.

STAGES:
  Stage 1: Weight Conversion (GGUF -> BUMP)
           - Block structure, scale unpacking, dequant parity
  Stage 2: Dimension & Memory Planning
           - Manifest dimensions, alignment, stride calculations
  Stage 3: Single Layer Activation Validation
           - Per-operation comparison vs llama.cpp

USAGE:
  # Run all stages
  python validate_kernel.py --gguf model.gguf --bump weights.bump --manifest manifest.json

  # Run only Stage 1
  python validate_kernel.py --stage 1 --gguf model.gguf

  # Run Stages 1 and 2
  python validate_kernel.py --stage 1,2 --gguf model.gguf --bump weights.bump

  # Run Stage 3 for specific layer
  python validate_kernel.py --stage 3 --layer 0 --gguf model.gguf --bump weights.bump

  # Quick validation after kernel change
  python validate_kernel.py --quick --gguf model.gguf --bump weights.bump

EXAMPLE DEBUG WORKFLOW (when output becomes gibberish):
  1. First check weights are correct:
     python validate_kernel.py --stage 1 --gguf model.gguf -v

  2. Then check dimensions:
     python validate_kernel.py --stage 2 --gguf model.gguf --bump weights.bump

  3. Then find which operation diverges:
     python validate_kernel.py --stage 3 --layer 0 --gguf model.gguf --bump weights.bump
"""

import argparse
import os
import sys
import json
from typing import List, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.base import ValidationReport, StageResult, TestStatus
from validation.stage1_weight_validation import Stage1Validator
from validation.stage2_dimension_validation import Stage2Validator
from validation.stage3_single_layer import Stage3Validator


# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


@dataclass
class ValidationConfig:
    """Configuration for validation run"""
    stages: List[int]
    gguf_path: str
    bump_path: Optional[str] = None
    manifest_path: Optional[str] = None
    kernel_type: Optional[str] = None
    layer: int = 0
    prompt: str = "Hello"
    tolerance: float = 1e-5
    verbose: bool = False
    quick: bool = False
    json_output: Optional[str] = None


class KernelValidator:
    """Main orchestrator for staged validation"""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.report = ValidationReport()

    def _print_header(self):
        """Print validation header"""
        print("\n" + "=" * 70)
        print(f"{BOLD}C-KERNEL-ENGINE v6 - STAGED KERNEL VALIDATION{RESET}")
        print("=" * 70)
        print(f"GGUF:     {self.config.gguf_path}")
        if self.config.bump_path:
            print(f"BUMP:     {self.config.bump_path}")
        if self.config.manifest_path:
            print(f"Manifest: {self.config.manifest_path}")
        print(f"Stages:   {self.config.stages}")
        print("=" * 70 + "\n")

    def run_stage1(self) -> StageResult:
        """Run Stage 1: Weight Conversion Validation"""
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{CYAN}STAGE 1: Weight Conversion Validation{RESET}")
        print(f"{CYAN}{'='*60}{RESET}")

        validator = Stage1Validator(
            gguf_path=self.config.gguf_path,
            bump_path=self.config.bump_path,
            manifest_path=self.config.manifest_path,
            verbose=self.config.verbose,
            kernel_type=self.config.kernel_type,
        )
        return validator.run()

    def run_stage2(self) -> StageResult:
        """Run Stage 2: Dimension & Memory Validation"""
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{CYAN}STAGE 2: Dimension & Memory Planning{RESET}")
        print(f"{CYAN}{'='*60}{RESET}")

        validator = Stage2Validator(
            gguf_path=self.config.gguf_path,
            bump_path=self.config.bump_path,
            manifest_path=self.config.manifest_path,
            verbose=self.config.verbose,
        )
        return validator.run()

    def run_stage3(self) -> StageResult:
        """Run Stage 3: Single Layer Activation Validation"""
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{CYAN}STAGE 3: Single Layer Activation (layer={self.config.layer}){RESET}")
        print(f"{CYAN}{'='*60}{RESET}")

        validator = Stage3Validator(
            gguf_path=self.config.gguf_path,
            bump_path=self.config.bump_path,
            manifest_path=self.config.manifest_path,
            verbose=self.config.verbose,
            layer=self.config.layer,
            prompt=self.config.prompt,
        )
        return validator.run()

    def run(self) -> ValidationReport:
        """Execute all requested stages with gating"""
        self._print_header()

        for stage_num in sorted(self.config.stages):
            if stage_num == 1:
                result = self.run_stage1()
            elif stage_num == 2:
                result = self.run_stage2()
            elif stage_num == 3:
                result = self.run_stage3()
            else:
                print(f"{YELLOW}Unknown stage: {stage_num}{RESET}")
                continue

            # Print stage summary
            result.print_summary(verbose=self.config.verbose)

            # Add to report
            self.report.add_stage_result(stage_num, result)

            # Gate on failure (unless quick mode)
            if not result.passed and not self.config.quick:
                self.report.gated_at = stage_num
                print(f"\n{RED}GATED at Stage {stage_num} - stopping validation{RESET}")
                print(f"{YELLOW}Fix the issues above before proceeding to next stage{RESET}")
                break

        return self.report


def parse_stages(stages_str: str) -> List[int]:
    """Parse comma-separated stages string"""
    stages = []
    for s in stages_str.split(','):
        s = s.strip()
        if s:
            try:
                stages.append(int(s))
            except ValueError:
                pass
    return sorted(set(stages)) if stages else [1, 2, 3]


def find_manifest_for_bump(bump_path: str) -> Optional[str]:
    """Try to find manifest file for a bump file"""
    if not bump_path:
        return None

    # Try common patterns
    base = os.path.splitext(bump_path)[0]
    candidates = [
        f"{base}_manifest.json",
        f"{base}.manifest.json",
        os.path.join(os.path.dirname(bump_path), "weights_manifest.json"),
        os.path.join(os.path.dirname(bump_path), "manifest.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Staged Kernel Validation for C-Kernel-Engine v6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Run all stages
  python validate_kernel.py --gguf model.gguf --bump weights.bump

  # Run only Stage 1 (weight validation)
  python validate_kernel.py --stage 1 --gguf model.gguf

  # Find where layer 0 diverges
  python validate_kernel.py --stage 3 --layer 0 --gguf model.gguf --bump weights.bump

  # Quick validation (don't stop on failures)
  python validate_kernel.py --quick --gguf model.gguf --bump weights.bump
"""
    )

    parser.add_argument("--gguf", type=str, required=True,
                       help="Path to GGUF model file")
    parser.add_argument("--bump", type=str, default=None,
                       help="Path to BUMP weights file")
    parser.add_argument("--manifest", type=str, default=None,
                       help="Path to weights manifest JSON")
    parser.add_argument("--stage", type=str, default="1,2,3",
                       help="Stages to run (comma-separated: 1,2,3)")
    parser.add_argument("--kernel", type=str, default=None,
                       help="Kernel type to focus on (q4_k, q6_k, etc.)")
    parser.add_argument("--layer", type=int, default=0,
                       help="Layer index for Stage 3")
    parser.add_argument("--prompt", type=str, default="Hello",
                       help="Prompt for Stage 3 activation comparison")
    parser.add_argument("--tol", type=float, default=1e-5,
                       help="Tolerance for numerical comparison")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode - don't stop on failures")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--json", type=str, default=None,
                       help="Save report as JSON to this path")

    args = parser.parse_args()

    # Validate GGUF exists
    if not os.path.exists(args.gguf):
        print(f"{RED}Error: GGUF file not found: {args.gguf}{RESET}")
        return 1

    # Auto-find manifest if not provided
    manifest_path = args.manifest
    if not manifest_path and args.bump:
        manifest_path = find_manifest_for_bump(args.bump)
        if manifest_path:
            print(f"Auto-detected manifest: {manifest_path}")

    # Parse stages
    stages = parse_stages(args.stage)

    config = ValidationConfig(
        stages=stages,
        gguf_path=args.gguf,
        bump_path=args.bump,
        manifest_path=manifest_path,
        kernel_type=args.kernel,
        layer=args.layer,
        prompt=args.prompt,
        tolerance=args.tol,
        verbose=args.verbose,
        quick=args.quick,
        json_output=args.json,
    )

    # Run validation
    validator = KernelValidator(config)
    report = validator.run()

    # Print summary
    report.print_summary()

    # Save JSON if requested
    if args.json:
        report.save_json(args.json)
        print(f"\nReport saved to: {args.json}")

    # Return exit code
    return 0 if report.all_passed() else 1


if __name__ == "__main__":
    sys.exit(main())
