#!/usr/bin/env python3
"""
Sync core v7 inference stack into version/v7.

Why:
- v7 training work still needs a stable inference path for forward checks,
  parity probes, and mixed inference+backprop workflows.
- Keep a reproducible "known-good" inference baseline inside v7 while
  preserving v7 training contracts.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "version" / "v6.6"
DST_ROOT = ROOT / "version" / "v7"

IGNORE_NAMES = {"__pycache__", ".pytest_cache"}
IGNORE_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True)
class CopyItem:
    src: str
    dst: str


CORE_FILES: List[CopyItem] = [
    CopyItem("README.md", "INFERENCE_BASELINE_FROM_V7.md"),
    CopyItem("scripts/ck_run_v7.py", "scripts/ck_run_v7.py"),
    CopyItem("scripts/build_ir_v7.py", "scripts/build_ir_v7.py"),
    CopyItem("scripts/codegen_v7.py", "scripts/codegen_v7.py"),
    CopyItem("scripts/codegen_prefill_v7.py", "scripts/codegen_prefill_v7.py"),
    CopyItem("scripts/convert_gguf_to_bump_v7.py", "scripts/convert_gguf_to_bump_v7.py"),
    CopyItem("scripts/convert_hf_to_bump_v7.py", "scripts/convert_hf_to_bump_v7.py"),
    CopyItem("scripts/inspect_weights_v7.py", "scripts/inspect_weights_v7.py"),
    CopyItem("scripts/ir_core_v7.py", "scripts/ir_core_v7.py"),
    CopyItem("scripts/ir_types_v7.py", "scripts/ir_types_v7.py"),
    CopyItem("scripts/memory_planner_v7.py", "scripts/memory_planner_v7.py"),
    CopyItem("scripts/generate_memory_map_v7.py", "scripts/generate_memory_map_v7.py"),
    CopyItem("scripts/memory_signoff_v7.py", "scripts/memory_signoff_v7.py"),
    CopyItem("scripts/parallel_planner.py", "scripts/parallel_planner.py"),
    CopyItem("scripts/parallel_pass.py", "scripts/parallel_pass.py"),
    CopyItem("scripts/quant_types.py", "scripts/quant_types.py"),
    CopyItem("scripts/fusion_patterns.py", "scripts/fusion_patterns.py"),
    CopyItem("scripts/op_builders_v7.py", "scripts/op_builders_v7.py"),
    CopyItem("scripts/op_builders_hybrid_v7.py", "scripts/op_builders_hybrid_v7.py"),
    CopyItem("scripts/op_builders_auto.py", "scripts/op_builders_auto.py"),
    CopyItem("scripts/gen_kernel_registry_from_maps.py", "scripts/gen_kernel_registry_from_maps.py"),
    CopyItem("scripts/validate_kernel_registry.py", "scripts/validate_kernel_registry.py"),
    CopyItem("scripts/resolve_model_dir_v7.py", "scripts/resolve_model_dir_v7.py"),
    CopyItem("scripts/generate_profile_summary_v7.py", "scripts/generate_profile_summary_v7.py"),
    CopyItem("scripts/perf_artifacts_v7.py", "scripts/perf_artifacts_v7.py"),
    CopyItem("scripts/perf_gate_v7.py", "scripts/perf_gate_v7.py"),
    CopyItem("scripts/vtune_artifacts_v7.py", "scripts/vtune_artifacts_v7.py"),
    CopyItem("scripts/validate_tooling_contracts.py", "scripts/validate_tooling_contracts.py"),
    CopyItem("scripts/validate_model_matrix_v7.py", "scripts/validate_model_matrix_v7.py"),
    CopyItem("scripts/validate_parity_matrix_v7.py", "scripts/validate_parity_matrix_v7.py"),
    CopyItem("scripts/validate_long_decode_stability_v7.py", "scripts/validate_long_decode_stability_v7.py"),
    CopyItem("scripts/ck_model_smoke_v7.py", "scripts/ck_model_smoke_v7.py"),
    CopyItem("scripts/ir_reverse_validator.py", "scripts/ir_reverse_validator.py"),
    CopyItem("tools/ir_visualizer.html", "tools/ir_visualizer.html"),
    CopyItem("tools/open_ir_visualizer.py", "tools/open_ir_visualizer.py"),
    CopyItem("src/ck_parallel_decode.c", "src/ck_parallel_decode.c"),
    CopyItem("src/ck_parallel_decode.h", "src/ck_parallel_decode.h"),
    CopyItem("src/ck_parallel_prefill.c", "src/ck_parallel_prefill.c"),
    CopyItem("src/ck_parallel_prefill.h", "src/ck_parallel_prefill.h"),
    CopyItem("src/ckernel_model_load_v7.c", "src/ckernel_model_load_v7.c"),
    CopyItem("src/ckernel_model_load_v7.h", "src/ckernel_model_load_v7.h"),
    CopyItem("src/ck_cli_v7.c", "src/ck_cli_v7.c"),
    CopyItem("include/ck_parity_dump.h", "include/ck_parity_dump.h"),
    CopyItem("include/ckernel_bump_v5.h", "include/ckernel_bump_v5.h"),
    CopyItem("test/Makefile", "test/Makefile"),
    CopyItem("test/TEST_README.py", "test/TEST_README.py"),
    CopyItem("test/ck_test_runner.py", "test/ck_test_runner.py"),
    CopyItem("test/test_layer_by_layer.py", "test/test_layer_by_layer.py"),
    CopyItem("test/test_numerical_parity.py", "test/test_numerical_parity.py"),
    CopyItem("test/trace_divergence.py", "test/trace_divergence.py"),
    CopyItem("test/v7_comprehensive_debug.py", "test/v7_comprehensive_debug.py"),
    CopyItem("test/test_memory_planner.py", "test/test_memory_planner.py"),
    CopyItem("test/advanced_memory_validator.py", "test/advanced_memory_validator.py"),
    CopyItem("test/test_kv_cache.py", "test/test_kv_cache.py"),
]

CORE_DIRS: List[CopyItem] = [
    CopyItem("kernel_maps", "kernel_maps"),
    CopyItem("templates", "templates"),
    CopyItem("scripts/parity", "scripts/parity"),
]


def _should_skip(path: Path) -> bool:
    if path.name in IGNORE_NAMES:
        return True
    if path.suffix in IGNORE_SUFFIXES:
        return True
    return False


def _iter_dir_files(path: Path) -> Iterable[Path]:
    for p in path.rglob("*"):
        if p.is_dir():
            continue
        if _should_skip(p):
            continue
        if any(part in IGNORE_NAMES for part in p.parts):
            continue
        yield p


def _copy_file(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists():
        raise FileNotFoundError(f"missing source file: {src}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def _source_candidates(rel_path: str) -> List[Path]:
    """
    Resolve v7-named sync entries against v6.6 source filenames.

    The destination in v7 is intentionally v7-named, but source files in
    version/v6.6 still use mixed v6_6/v6.6 suffix styles.
    """
    candidates: List[str] = [rel_path]

    # Python script suffix style: *_v7.py -> *_v6_6.py
    if "_v7.py" in rel_path:
        candidates.append(rel_path.replace("_v7.py", "_v6_6.py"))

    # C source/header suffix styles in v6.6 tree.
    if "_v7.c" in rel_path:
        candidates.append(rel_path.replace("_v7.c", "_v6_6.c"))
    if "_v7.h" in rel_path:
        candidates.append(rel_path.replace("_v7.h", "_v6_6.h"))
        candidates.append(rel_path.replace("_v7.h", "_v6.6.h"))

    # Special filename variants used in v6.6.
    candidates.append(rel_path.replace("ck_cli_v7.c", "ck_cli_v6.6.c"))
    candidates.append(
        rel_path.replace("v7_comprehensive_debug.py", "v6_6_comprehensive_debug.py")
    )

    # De-duplicate while preserving order.
    deduped: List[Path] = []
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        deduped.append(SRC_ROOT / c)
    return deduped


def _resolve_source_path(rel_path: str) -> Path:
    for candidate in _source_candidates(rel_path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"missing source file for {rel_path}. tried: "
        + ", ".join(str(p) for p in _source_candidates(rel_path))
    )


def _copy_dir(src: Path, dst: Path, dry_run: bool) -> int:
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"missing source dir: {src}")
    count = 0
    for file_path in _iter_dir_files(src):
        rel = file_path.relative_to(src)
        _copy_file(file_path, dst / rel, dry_run=dry_run)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync v7 inference baseline into v7.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files.")
    args = parser.parse_args()

    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"v6.6 source root not found: {SRC_ROOT}")

    copied_files = 0
    copied_dirs_files = 0
    print(f"[sync] source: {SRC_ROOT}")
    print(f"[sync] dest:   {DST_ROOT}")

    for item in CORE_FILES:
        src = _resolve_source_path(item.src)
        dst = DST_ROOT / item.dst
        _copy_file(src, dst, dry_run=args.dry_run)
        copied_files += 1
        print(f"[file] {src.relative_to(SRC_ROOT)} -> {item.dst}")

    for item in CORE_DIRS:
        src = SRC_ROOT / item.src
        dst = DST_ROOT / item.dst
        n = _copy_dir(src, dst, dry_run=args.dry_run)
        copied_dirs_files += n
        print(f"[dir ] {item.src} -> {item.dst} ({n} files)")

    readme_path = DST_ROOT / "INFERENCE_BASELINE_SYNC.md"
    readme_text = (
        "# v7 Inference Baseline Sync\n\n"
        "This tree is a curated sync of v6.6 inference assets into v7 names.\n\n"
        "Purpose:\n"
        "- keep inference parity tooling available while building backprop/training in v7\n"
        "- avoid rewriting stable v6.6 runtime pieces during early v7 work\n\n"
        "Update workflow:\n"
        "1. modify source inference in version/v6.6\n"
        "2. run `make v7-sync-inference`\n"
        "3. re-run v7 validation gates\n"
    )
    if not args.dry_run:
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(readme_text, encoding="utf-8")

    print(
        f"[done] copied {copied_files} explicit files + {copied_dirs_files} files from synced directories"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
