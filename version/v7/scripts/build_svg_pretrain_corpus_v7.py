#!/usr/bin/env python3
"""
build_svg_pretrain_corpus_v7.py

Stitch existing v7 dataset scripts into one operator command:
1) Build ASCII SVG corpus from docs/site/assets/*.svg
2) Generate synthetic Stage-A and Stage-B SVG rows
3) Build Stage-A bridge pack from missing syntax features
4) Emit ready-to-train corpora:
   - <prefix>_stage_a_plus_bridge.txt
   - <prefix>_stage_b.txt
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_rows(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        row = raw.strip()
        if row:
            out.append(row)
    return out


def _merge_rows(
    sources: Iterable[Path],
    max_line_chars: int,
    dedupe: bool,
    shuffle: bool,
    seed: int,
) -> list[str]:
    rows: list[str] = []
    for src in sources:
        for row in _read_rows(src):
            if len(row) <= max_line_chars:
                rows.append(row)
    if dedupe:
        seen: set[str] = set()
        uniq: list[str] = []
        for row in rows:
            if row in seen:
                continue
            seen.add(row)
            uniq.append(row)
        rows = uniq
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    return rows


def _write_rows(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _stats(path: Path) -> dict:
    rows = _read_rows(path)
    chars = sum(len(r) + 1 for r in rows)  # newline-inclusive estimate
    return {
        "path": str(path),
        "rows": len(rows),
        "chars_est": chars,
    }


def main() -> int:
    root = _repo_root()
    py = Path(sys.executable).resolve()
    scripts = root / "version" / "v7" / "scripts"

    ap = argparse.ArgumentParser(description="Build Stage-A+Bridge and Stage-B SVG corpora in one command.")
    ap.add_argument("--out-dir", default=str(root / "version" / "v7" / "data"), help="Output directory")
    ap.add_argument("--prefix", default="svg_pretrain_pack", help="Output prefix")
    ap.add_argument("--assets-glob", default=str(root / "docs" / "site" / "assets" / "*.svg"), help="SVG assets glob")
    ap.add_argument("--stage-a-samples", type=int, default=24000, help="Synthetic Stage-A sample count")
    ap.add_argument("--stage-b-samples", type=int, default=28000, help="Synthetic Stage-B sample count")
    ap.add_argument("--holdout-ratio", type=float, default=0.10, help="Synthetic holdout ratio")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--stage-a-types",
        default="line,ellipse,triangle,rounded_triangle,polygon,path,arrow,double_arrow,polyline,rect_circle,text",
        help="Comma-separated types for Stage-A synthetic generator",
    )
    ap.add_argument(
        "--stage-b-types",
        default="all",
        help="Comma-separated types for Stage-B synthetic generator",
    )
    ap.add_argument("--fill-mode-a", choices=["mixed", "filled", "outline"], default="mixed")
    ap.add_argument("--fill-mode-b", choices=["mixed", "filled", "outline"], default="mixed")
    ap.add_argument("--bridge-per-feature-cap", type=int, default=10)
    ap.add_argument("--bridge-max-total", type=int, default=192)
    ap.add_argument("--max-line-chars", type=int, default=4096)
    ap.set_defaults(dedupe=True, shuffle=True)
    ap.add_argument("--dedupe", dest="dedupe", action="store_true")
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    ap.add_argument("--shuffle", dest="shuffle", action="store_true")
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    args = ap.parse_args()

    if not (0.0 <= float(args.holdout_ratio) < 1.0):
        raise SystemExit("--holdout-ratio must be in [0,1)")
    if int(args.stage_a_samples) < 1 or int(args.stage_b_samples) < 1:
        raise SystemExit("--stage-a-samples and --stage-b-samples must be >= 1")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = str(args.prefix).strip()
    if not prefix:
        raise SystemExit("--prefix must be non-empty")

    assets_ascii = out_dir / f"{prefix}_assets_ascii.txt"
    assets_manifest = out_dir / f"{prefix}_assets_ascii_manifest.json"
    stage_a_syn_prefix = f"{prefix}_stage_a_syn"
    stage_b_syn_prefix = f"{prefix}_stage_b_syn"
    stage_a_syn_train = out_dir / f"{stage_a_syn_prefix}_svg_train.txt"
    stage_b_syn_train = out_dir / f"{stage_b_syn_prefix}_svg_train.txt"

    stage_a_base_raw = out_dir / f"{prefix}_stage_a_base_raw.txt"
    stage_a_base = out_dir / f"{prefix}_stage_a_base.txt"
    bridge_txt = out_dir / f"{prefix}_stage_a_bridge.txt"
    bridge_manifest = out_dir / f"{prefix}_stage_a_bridge_manifest.json"
    stage_a_plus_bridge_raw = out_dir / f"{prefix}_stage_a_plus_bridge_raw.txt"
    stage_a_plus_bridge = out_dir / f"{prefix}_stage_a_plus_bridge.txt"
    stage_b_raw = out_dir / f"{prefix}_stage_b_raw.txt"
    stage_b = out_dir / f"{prefix}_stage_b.txt"
    manifest_out = out_dir / f"{prefix}_manifest.json"

    build_assets = scripts / "build_svg_corpus_from_assets_v7.py"
    gen_synth = scripts / "generate_svg_instruction_dataset_v7.py"
    build_bridge = scripts / "build_stage_a_bridge_svg_v7.py"
    prep_ascii = scripts / "prepare_ascii_dataset_v7.py"

    _run(
        [
            str(py),
            str(build_assets),
            "--assets-glob",
            str(args.assets_glob),
            "--output",
            str(assets_ascii),
            "--manifest",
            str(assets_manifest),
            "--ascii-map-common",
            "--ascii-mode",
            "xml_escape",
            "--no-dedupe",
        ]
    )

    _run(
        [
            str(py),
            str(gen_synth),
            "--out-dir",
            str(out_dir),
            "--prefix",
            stage_a_syn_prefix,
            "--num-samples",
            str(int(args.stage_a_samples)),
            "--holdout-ratio",
            str(float(args.holdout_ratio)),
            "--seed",
            str(int(args.seed)),
            "--types",
            str(args.stage_a_types),
            "--fill-mode",
            str(args.fill_mode_a),
        ]
    )

    _run(
        [
            str(py),
            str(gen_synth),
            "--out-dir",
            str(out_dir),
            "--prefix",
            stage_b_syn_prefix,
            "--num-samples",
            str(int(args.stage_b_samples)),
            "--holdout-ratio",
            str(float(args.holdout_ratio)),
            "--seed",
            str(int(args.seed) + 101),
            "--types",
            str(args.stage_b_types),
            "--fill-mode",
            str(args.fill_mode_b),
        ]
    )

    stage_a_rows = _merge_rows(
        sources=[stage_a_syn_train],
        max_line_chars=int(args.max_line_chars),
        dedupe=bool(args.dedupe),
        shuffle=bool(args.shuffle),
        seed=int(args.seed),
    )
    _write_rows(stage_a_base_raw, stage_a_rows)
    _run(
        [
            str(py),
            str(prep_ascii),
            "--input",
            str(stage_a_base_raw),
            "--output",
            str(stage_a_base),
            "--input-format",
            "text",
            "--ascii-map-common",
            "--ascii-mode",
            "xml_escape",
            "--svg-only",
        ]
    )

    _run(
        [
            str(py),
            str(build_bridge),
            "--stage-a",
            str(stage_a_base),
            "--stage-b",
            str(assets_ascii),
            "--out",
            str(bridge_txt),
            "--manifest",
            str(bridge_manifest),
            "--per-feature-cap",
            str(int(args.bridge_per_feature_cap)),
            "--max-total",
            str(int(args.bridge_max_total)),
            "--seed",
            str(int(args.seed)),
        ]
    )

    stage_a_plus_bridge_rows = _merge_rows(
        sources=[stage_a_base, bridge_txt],
        max_line_chars=int(args.max_line_chars),
        dedupe=bool(args.dedupe),
        shuffle=bool(args.shuffle),
        seed=int(args.seed) + 7,
    )
    _write_rows(stage_a_plus_bridge_raw, stage_a_plus_bridge_rows)
    _run(
        [
            str(py),
            str(prep_ascii),
            "--input",
            str(stage_a_plus_bridge_raw),
            "--output",
            str(stage_a_plus_bridge),
            "--input-format",
            "text",
            "--ascii-map-common",
            "--ascii-mode",
            "xml_escape",
            "--svg-only",
        ]
    )

    stage_b_rows = _merge_rows(
        sources=[stage_a_plus_bridge, assets_ascii, stage_b_syn_train],
        max_line_chars=int(args.max_line_chars),
        dedupe=bool(args.dedupe),
        shuffle=bool(args.shuffle),
        seed=int(args.seed) + 19,
    )
    _write_rows(stage_b_raw, stage_b_rows)
    _run(
        [
            str(py),
            str(prep_ascii),
            "--input",
            str(stage_b_raw),
            "--output",
            str(stage_b),
            "--input-format",
            "text",
            "--ascii-map-common",
            "--ascii-mode",
            "xml_escape",
            "--svg-only",
        ]
    )

    manifest = {
        "format": "v7-svg-pretrain-corpus-pack",
        "seed": int(args.seed),
        "assets_glob": str(args.assets_glob),
        "stage_a_types": str(args.stage_a_types),
        "stage_b_types": str(args.stage_b_types),
        "fill_mode_a": str(args.fill_mode_a),
        "fill_mode_b": str(args.fill_mode_b),
        "max_line_chars": int(args.max_line_chars),
        "paths": {
            "assets_ascii": str(assets_ascii),
            "stage_a_syn_train": str(stage_a_syn_train),
            "stage_b_syn_train": str(stage_b_syn_train),
            "stage_a_base": str(stage_a_base),
            "stage_a_bridge": str(bridge_txt),
            "stage_a_plus_bridge": str(stage_a_plus_bridge),
            "stage_b": str(stage_b),
            "bridge_manifest": str(bridge_manifest),
        },
        "stats": {
            "assets_ascii": _stats(assets_ascii),
            "stage_a_syn_train": _stats(stage_a_syn_train),
            "stage_b_syn_train": _stats(stage_b_syn_train),
            "stage_a_base": _stats(stage_a_base),
            "stage_a_bridge": _stats(bridge_txt),
            "stage_a_plus_bridge": _stats(stage_a_plus_bridge),
            "stage_b": _stats(stage_b),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[OK] stage_a_plus_bridge:", stage_a_plus_bridge)
    print("[OK] stage_b:", stage_b)
    print("[OK] manifest:", manifest_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
