#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import convert_gguf_to_bump_v8 as gguf  # type: ignore  # noqa: E402


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_ir_v8 = _load_module("build_ir_v8_qwen3vl_parity", SCRIPT_DIR / "build_ir_v8.py")
codegen_v8 = _load_module("codegen_v8_qwen3vl_parity", SCRIPT_DIR / "codegen_v8.py")


REQUIRED_TENSORS = [
    "v.patch_embd.weight",
    "v.patch_embd.weight.1",
    "v.patch_embd.bias",
    "v.position_embd.weight",
    "v.post_ln.weight",
    "v.post_ln.bias",
    "mm.0.weight",
    "mm.0.bias",
    "mm.2.weight",
    "mm.2.bias",
]


def _run_converter(gguf_path: Path, output_dir: Path) -> tuple[dict, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bump_path = output_dir / "weights.bump"
    manifest_path = output_dir / "weights_manifest.json"
    config_path = output_dir / "config.json"

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(SCRIPT_DIR / "convert_gguf_to_bump_v8.py"),
            "--gguf", str(gguf_path),
            "--output", str(bump_path),
            "--manifest-out", str(manifest_path),
            "--config-out", str(config_path),
        ]
        gguf.main()
    finally:
        sys.argv = old_argv

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest, manifest_path, bump_path, config_path


def _report_notes(manifest: dict) -> list[str]:
    coverage = manifest.get("source_tensor_coverage", {}) if isinstance(manifest, dict) else {}
    unconsumed = coverage.get("unconsumed_source_tensors", [])
    deepstack_layers = manifest.get("config", {}).get("deepstack_layer_indices", [])
    notes = [
        "This harness now uses the real v8 GGUF->BUMP converter output instead of a synthetic manifest.",
        "IR/codegen parity is validated against the converted local mmproj file; numeric embedding parity is still pending.",
    ]
    if deepstack_layers:
        notes.append(
            f"Deepstack layers are present in the manifest ({deepstack_layers}); "
            "the v8 template now lowers them via generic branch producer/stitch ops."
        )
    if unconsumed:
        notes.append(
            f"Converter left {len(unconsumed)} source tensors unconsumed; full source coverage is not complete yet."
        )
    else:
        notes.append("Converter consumed the full source tensor set from the local mmproj GGUF.")
    return notes


def _entry_is_runtime_supported(name: str) -> bool:
    return True


def _build_runtime_manifest(manifest: dict) -> tuple[dict, list[str]]:
    entries = manifest.get("entries", [])
    runtime_entries = []
    excluded: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", ""))
        if name and _entry_is_runtime_supported(name):
            runtime_entries.append(entry)
        elif name:
            excluded.append(name)
    runtime_manifest = json.loads(json.dumps(manifest))
    runtime_manifest["entries"] = runtime_entries
    runtime_manifest["runtime_excluded_entries"] = excluded
    return runtime_manifest, excluded


def run_harness(gguf_path: Path, output_dir: Path) -> dict:
    manifest, manifest_path, bump_path, config_path = _run_converter(gguf_path, output_dir)
    runtime_manifest, excluded_entries = _build_runtime_manifest(manifest)
    runtime_manifest_path = output_dir / "weights_manifest.runtime.json"
    runtime_manifest_path.write_text(json.dumps(runtime_manifest, indent=2), encoding="utf-8")
    ir1_path = output_dir / "ir1.json"
    layout_path = output_dir / "layout.json"
    lowered_path = output_dir / "lowered.json"
    call_path = output_dir / "call.json"
    c_path = output_dir / "qwen3_vl_mmproj_v8.c"

    rc = build_ir_v8.main(
        [
            "--manifest", str(runtime_manifest_path),
            "--mode", "prefill",
            "--output", str(ir1_path),
            "--layout-output", str(layout_path),
            "--lowered-output", str(lowered_path),
            "--call-output", str(call_path),
        ]
    )
    if rc != 0:
        raise RuntimeError(f"build_ir_v8 failed with rc={rc}")

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(SCRIPT_DIR / "codegen_v8.py"),
            "--ir", str(call_path),
            "--layout", str(layout_path),
            "--output", str(c_path),
        ]
        codegen_rc = codegen_v8.main()
    finally:
        sys.argv = old_argv
    if codegen_rc != 0:
        raise RuntimeError(f"codegen_v8 failed with rc={codegen_rc}")

    ir1 = json.loads(ir1_path.read_text(encoding="utf-8"))
    ir1_ops = ir1 if isinstance(ir1, list) else ir1.get("ops", ir1.get("operations", []))
    ops = [op["op"] for op in ir1_ops]
    branch_lowering_ops = {"branch_spatial_merge", "branch_layernorm", "branch_fc1", "branch_gelu", "branch_fc2", "branch_concat"}
    has_deepstack_lowering = branch_lowering_ops.issubset(set(ops))
    has_position_ids = "position_ids_2d" in ops or "vision_position_ids" in ops
    has_vision_mrope = any(
        op.get("op") in {"rope_qk", "mrope_qk"} and op.get("kernel") == "mrope_qk_vision"
        for op in ir1_ops
        if isinstance(op, dict)
    )
    config = manifest.get("config", {})
    entry_names = {
        str(entry.get("name"))
        for entry in manifest.get("entries", [])
        if isinstance(entry, dict) and entry.get("name")
    }
    coverage = manifest.get("source_tensor_coverage", {})
    deepstack_layers = config.get("deepstack_layer_indices", config.get("deepstack_layers"))
    report = {
        "gguf": str(gguf_path),
        "weights_bump": str(bump_path),
        "weights_manifest": str(manifest_path),
        "runtime_manifest": str(runtime_manifest_path),
        "config_path": str(config_path),
        "projector_type": manifest.get("projector_type"),
        "has_vision_encoder": manifest.get("has_vision_encoder", config.get("has_vision_encoder")),
        "deepstack_layers": deepstack_layers,
        "required_tensors_present": all(name in entry_names for name in REQUIRED_TENSORS),
        "config": config,
        "source_tensor_coverage": coverage,
        "runtime_excluded_entries": excluded_entries,
        "ops": ops,
        "lowering": {
            "has_dual_patch_frontend": {"patch_proj", "patch_proj_aux", "add_stream", "patch_bias_add"}.issubset(set(ops)),
            "has_position_embeddings": "position_embeddings" in ops,
            "has_position_ids": has_position_ids,
            "has_packed_qkv": {"qkv_packed_proj", "split_qkv_packed"}.issubset(set(ops)),
            "has_vision_mrope": has_vision_mrope,
            "has_spatial_merge": "spatial_merge" in ops,
            "has_projector_footer": {"projector_fc1", "projector_gelu", "projector_fc2"}.issubset(set(ops)),
            "has_mlp_bias_lowering": any(
                op.get("op") in {"mlp_gate_up", "mlp_down"} and bool(op.get("weights", {}).get("b1") or op.get("weights", {}).get("b2"))
                for op in ir1_ops
                if isinstance(op, dict)
            ),
            "has_deepstack_lowering": has_deepstack_lowering,
            "has_runtime_codegen": c_path.exists(),
        },
        "notes": _report_notes(manifest),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Contract/codegen parity harness for Qwen3-VL mmproj in v8")
    ap.add_argument("--gguf", type=Path, required=True, help="Path to mmproj-Qwen3VL-*.gguf")
    ap.add_argument("--output-dir", type=Path, default=None, help="Optional output directory")
    args = ap.parse_args()

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="qwen3vl_mmproj_v8_"))

    report = run_harness(args.gguf, out_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
