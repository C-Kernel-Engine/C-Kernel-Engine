#!/usr/bin/env python3
"""
Main Build Pipeline Orchestrator

Usage:
    python scripts/build_model.py --model model.gguf
    python scripts/build_model.py --model model.gguf --ir-dir build/ir/
    python scripts/build_model.py --model model.gguf --dry-run

Pipeline:
    1. gen_kernel_registry  -> KERNEL_REGISTRY.json
    2. check_model_coverage -> MISSING_KERNELS.txt
    3. build_ir             -> ir/layer_XX.json
    4. fusion_pass          -> fused_ir/layer_XX.json + fusion_log.json
    5. gen_memory_layout    -> src/memory_layout.{c,h}
    6. codegen              -> src/inference.c
    7. build                -> ck_engine
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PipelineConfig:
    """Configuration for the build pipeline."""
    model: Path
    kernel_maps_dir: Path = Path("version/v6.6/kernel_maps")
    ir_dir: Path = Path("build/ir")
    fused_ir_dir: Path = Path("build/fused_ir")
    output_dir: Path = Path("build")
    dry_run: bool = False
    verbose: bool = False


class PipelineRunner:
    """Run the build pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.steps = []

    def log(self, msg: str):
        if self.config.verbose:
            print(f"[info] {msg}")

    def run_step(self, name: str, cmd: List[str], cwd: Optional[Path] = None) -> bool:
        """Run a single pipeline step."""
        self.steps.append(name)
        print(f"[step {len(self.steps)}] {name}")

        if self.config.dry_run:
            print(f"       (dry-run) {' '.join(cmd)}")
            return True

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.config.model.parent,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                print(f"       [fail] {result.stderr or result.stdout}")
                return False
            print(f"       [ok]")
            return True
        except subprocess.TimeoutExpired:
            print(f"       [fail] timeout")
            return False
        except Exception as e:
            print(f"       [fail] {e}")
            return False

    def run(self) -> bool:
        """Run the complete pipeline."""
        print("=" * 60)
        print("CK-Engine Build Pipeline")
        print("=" * 60)
        print(f"Model: {self.config.model}")
        print(f"Output: {self.config.output_dir}")
        print("=" * 60)
        print()

        # Step 1: Generate kernel registry
        print("Phase 1: Kernel Registry")
        if not self.run_step(
            "gen_kernel_registry",
            ["python", "scripts/gen_kernel_registry.py",
             "--output", str(self.config.kernel_maps_dir / "KERNEL_REGISTRY.json"),
             "--verbose"]
        ):
            return False

        # Step 2: Check model coverage
        print("\nPhase 2: Model Analysis")
        if not self.run_step(
            "check_model_coverage",
            ["python", "scripts/check_model_coverage.py",
             "--model", str(self.config.model),
             "--registry", str(self.config.kernel_maps_dir / "KERNEL_REGISTRY.json")]
        ):
            return False

        # Step 3: Build IR
        print("\nPhase 3: IR Generation")
        self.config.ir_dir.mkdir(parents=True, exist_ok=True)
        if not self.run_step(
            "build_ir",
            ["python", "scripts/build_ir.py",
             "--model", str(self.config.model),
             "--output-dir", str(self.config.ir_dir)]
        ):
            return False

        # Step 4: Fusion pass
        print("\nPhase 4: Fusion")
        self.config.fused_ir_dir.mkdir(parents=True, exist_ok=True)
        if not self.run_step(
            "fusion_pass",
            ["python", "scripts/fusion_pass.py",
             "--ir-dir", str(self.config.ir_dir),
             "--kernel-maps", str(self.config.kernel_maps_dir),
             "--output", str(self.config.fused_ir_dir)]
        ):
            return False

        # Step 5: Memory layout
        print("\nPhase 5: Memory Layout")
        if not self.run_step(
            "gen_memory_layout",
            ["python", "scripts/gen_memory_layout.py",
             "--ir-dir", str(self.config.fused_ir_dir),
             "--output", "src/memory_layout.c",
             "--header", "src/memory_layout.h",
             "--max-tokens", "8192"]
        ):
            return False

        # Step 6: Code generation
        print("\nPhase 6: Code Generation")
        if not self.run_step(
            "codegen",
            ["python", "scripts/codegen.py",
             "--ir-dir", str(self.config.fused_ir_dir),
             "--output", "src/inference.c"]
        ):
            return False

        # Step 7: Build
        print("\nPhase 7: Compilation")
        if not self.run_step(
            "make",
            ["make", "clean", "&&", "make", "-j$(nproc)"],
        ):
            return False

        # Summary
        print()
        print("=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"Steps run: {len(self.steps)}")
        for i, step in enumerate(self.steps, 1):
            print(f"  {i}. {step}")
        print()
        print("Output artifacts:")
        print(f"  - {self.config.ir_dir}/")
        print(f"  - {self.config.fused_ir_dir}/")
        print(f"  - src/memory_layout.{{c,h}}")
        print(f"  - src/inference.c")
        print(f"  - ck_engine binary")

        return True


def main():
    parser = argparse.ArgumentParser(description="CK-Engine Build Pipeline")
    parser.add_argument("--model", "-m", required=True, help="Model file (GGUF/Safetensors)")
    parser.add_argument("--kernel-maps", "-k", default="version/v6.6/kernel_maps",
                        help="Kernel maps directory")
    parser.add_argument("--ir-dir", default="build/ir", help="IR output directory")
    parser.add_argument("--fused-ir-dir", default="build/fused_ir",
                        help="Fused IR output directory")
    parser.add_argument("--output", "-o", default="build", help="Output directory")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show commands without running")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    config = PipelineConfig(
        model=Path(args.model),
        kernel_maps_dir=Path(args.kernel_maps),
        ir_dir=Path(args.ir_dir),
        fused_ir_dir=Path(args.fused_ir_dir),
        output_dir=Path(args.output),
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    runner = PipelineRunner(config)
    success = runner.run()

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
