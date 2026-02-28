#!/usr/bin/env python3
"""
build_svg_pretrain_corpus_v7.py

Builds training-ready SVG corpora for v7 in one command.

Outputs (canonical):
  - <prefix>_stage_a_plus_bridge.txt
  - <prefix>_stage_b.txt
  - <prefix>_tokenizer_corpus.txt

Row format:
  - default mode: <task>...</task><svg ...>...</svg><eos>
  - --spec-catalog mode: [tag][tag]...<svg ...>...</svg><eos>
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


_INSTR_RE = re.compile(
    r"^\s*<task>(?P<task>.*?)</task>(?P<svg><svg.*</svg>)(?:<eos>)?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_TAG_SVG_RE = re.compile(
    r"^\s*(?:\[[^\]]+\])+\s*(?P<svg><svg.*</svg>)(?:<eos>)?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)

_TASK_TEMPLATES_BRIDGE = (
    "create an svg sample that demonstrates advanced vector syntax cleanly",
    "draw a structured svg composition with valid layout and readable labels",
    "generate a compact infographic-style svg with balanced spacing and contrast",
)

_TASK_TEMPLATES_ASSET = (
    "recreate this svg design preserving composition and readability",
    "generate a polished svg card with clear visual hierarchy",
    "draw an infographic element with clean spacing and balanced color use",
)

_TAG_TEMPLATES_BRIDGE = (
    "[infographic][palette:neutral][style:mixed][layout:grid]",
    "[diagram][palette:cool][style:outlined][layout:stacked]",
    "[multi-shape][palette:pastel][style:filled][layout:offset]",
)

_TAG_TEMPLATES_ASSET = (
    "[infographic][palette:bold][style:mixed][layout:grid]",
    "[table][palette:neutral][style:outlined][layout:stacked]",
    "[multi-shape][palette:cool][style:gradient][layout:offset]",
)


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


def _write_rows(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _stats(path: Path) -> dict:
    rows = _read_rows(path)
    chars = sum(len(r) + 1 for r in rows)
    return {"path": str(path), "rows": len(rows), "chars_est": chars}


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _prepare_ascii(prep_ascii: Path, py: Path, src: Path, out: Path, svg_only: bool) -> None:
    cmd = [
        str(py),
        str(prep_ascii),
        "--input",
        str(src),
        "--output",
        str(out),
        "--input-format",
        "text",
        "--ascii-map-common",
        "--ascii-mode",
        "xml_escape",
    ]
    if svg_only:
        cmd.append("--svg-only")
    _run(cmd)


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
        rows = list(dict.fromkeys(rows))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    return rows


def _merge_row_lists(
    sources: Iterable[list[str]],
    max_line_chars: int,
    dedupe: bool,
    shuffle: bool,
    seed: int,
) -> list[str]:
    rows: list[str] = []
    for src in sources:
        for row in src:
            if row and len(row) <= max_line_chars:
                rows.append(row)
    if dedupe:
        rows = list(dict.fromkeys(rows))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    return rows


def _instruction_rows(path: Path, max_line_chars: int) -> list[str]:
    out: list[str] = []
    for row in _read_rows(path):
        if len(row) > max_line_chars:
            continue
        if _INSTR_RE.match(row) or _TAG_SVG_RE.match(row):
            out.append(row)
    return out


def _rows_by_ratio(rows: list[str], ratio: float, seed: int) -> list[str]:
    if not rows or ratio <= 0.0:
        return []
    if ratio >= 1.0:
        return list(rows)
    target = int(round(len(rows) * ratio))
    if target <= 0:
        return []
    if target >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, target)


def _wrap_svg_rows_as_task(
    svg_rows: list[str],
    task_templates: tuple[str, ...],
    seed: int,
    max_rows: int,
    max_line_chars: int,
) -> list[str]:
    candidates = [r for r in svg_rows if r.lstrip().startswith("<svg") and len(r) <= max_line_chars]
    if not candidates:
        return []
    if max_rows > 0 and len(candidates) > max_rows:
        rng = random.Random(seed + 17)
        candidates = rng.sample(candidates, max_rows)
    out: list[str] = []
    rng = random.Random(seed)
    for svg in candidates:
        task = task_templates[rng.randrange(len(task_templates))]
        row = f"<task>{task}</task>{svg}<eos>"
        if len(row) <= max_line_chars:
            out.append(row)
    return out


def _wrap_svg_rows_as_tags(
    svg_rows: list[str],
    tag_prefixes: tuple[str, ...],
    seed: int,
    max_rows: int,
    max_line_chars: int,
) -> list[str]:
    candidates = [r for r in svg_rows if r.lstrip().startswith("<svg") and len(r) <= max_line_chars]
    if not candidates:
        return []
    if max_rows > 0 and len(candidates) > max_rows:
        rng = random.Random(seed + 117)
        candidates = rng.sample(candidates, max_rows)
    out: list[str] = []
    rng = random.Random(seed)
    for svg in candidates:
        prefix = tag_prefixes[rng.randrange(len(tag_prefixes))]
        row = f"{prefix}{svg}<eos>"
        if len(row) <= max_line_chars:
            out.append(row)
    return out


def main() -> int:
    root = _repo_root()
    py = Path(sys.executable).resolve()
    scripts = root / "version" / "v7" / "scripts"

    ap = argparse.ArgumentParser(description="Build Stage-A/Stage-B task-conditioned SVG corpora.")
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
    ap.add_argument("--stage-b-types", default="all", help="Comma-separated types for Stage-B synthetic generator")
    ap.add_argument("--fill-mode-a", choices=["mixed", "filled", "outline"], default="mixed")
    ap.add_argument("--fill-mode-b", choices=["mixed", "filled", "outline"], default="mixed")
    ap.add_argument("--bridge-per-feature-cap", type=int, default=10)
    ap.add_argument("--bridge-max-total", type=int, default=192)
    ap.add_argument("--stage-a-task-ratio", type=float, default=0.70, help="Fraction of Stage-A instruction rows kept for task-conditioned corpus")
    ap.add_argument("--stage-b-task-ratio", type=float, default=0.85, help="Fraction of Stage-B instruction rows kept for task-conditioned corpus")
    # Backward-compat alias from earlier experimental flag naming.
    ap.add_argument("--stage-a-desc-ratio", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--stage-b-desc-ratio", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--assets-task-max", type=int, default=1200, help="Max wrapped assets rows included in Stage-B task-conditioned corpus")
    ap.add_argument("--bridge-task-max", type=int, default=320, help="Max wrapped bridge rows included in Stage-A task-conditioned corpus")
    ap.add_argument("--color-theory-samples", type=int, default=0, help="Optional extra color-theory instruction rows for Stage-B task-conditioned corpus")
    ap.add_argument("--color-palette-file", action="append", default=[], help="Optional palette file(s) passed to color-theory generator")
    ap.add_argument("--color-external-palette-prob", type=float, default=0.0, help="External palette usage probability for color-theory generator [0,1]")
    ap.add_argument("--color-min-contrast", type=float, default=4.5, help="Minimum contrast target for color-theory generation")
    ap.add_argument("--spec-catalog", default=None, help="Optional spec catalog JSON for spec-driven synthetic generation")
    ap.add_argument("--strict-coverage", action="store_true", help="Fail if spec-driven coverage gate fails")
    ap.add_argument("--max-line-chars", type=int, default=4096)
    ap.set_defaults(dedupe=True, shuffle=True)
    ap.add_argument("--dedupe", dest="dedupe", action="store_true")
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    ap.add_argument("--shuffle", dest="shuffle", action="store_true")
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    args = ap.parse_args()

    stage_a_task_ratio = float(args.stage_a_desc_ratio) if args.stage_a_desc_ratio is not None else float(args.stage_a_task_ratio)
    stage_b_task_ratio = float(args.stage_b_desc_ratio) if args.stage_b_desc_ratio is not None else float(args.stage_b_task_ratio)

    if not (0.0 <= float(args.holdout_ratio) < 1.0):
        raise SystemExit("--holdout-ratio must be in [0,1)")
    if int(args.stage_a_samples) < 1 or int(args.stage_b_samples) < 1:
        raise SystemExit("--stage-a-samples and --stage-b-samples must be >= 1")
    if not (0.0 <= stage_a_task_ratio <= 1.0):
        raise SystemExit("--stage-a-task-ratio must be in [0,1]")
    if not (0.0 <= stage_b_task_ratio <= 1.0):
        raise SystemExit("--stage-b-task-ratio must be in [0,1]")
    if not (0.0 <= float(args.color_external_palette_prob) <= 1.0):
        raise SystemExit("--color-external-palette-prob must be in [0,1]")
    if int(args.color_theory_samples) < 0:
        raise SystemExit("--color-theory-samples must be >= 0")
    spec_catalog_path: Path | None = None
    if args.spec_catalog:
        spec_catalog_path = Path(str(args.spec_catalog)).expanduser().resolve()
        if not spec_catalog_path.exists():
            raise SystemExit(f"--spec-catalog not found: {spec_catalog_path}")
        # Keep exact per-spec counts stable when catalog mode is enabled.
        stage_a_task_ratio = 1.0
        stage_b_task_ratio = 1.0

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix).strip()
    if not prefix:
        raise SystemExit("--prefix must be non-empty")

    build_assets = scripts / "build_svg_corpus_from_assets_v7.py"
    gen_synth = scripts / "generate_svg_instruction_dataset_v7.py"
    gen_color = scripts / "generate_svg_color_theory_dataset_v7.py"
    build_bridge = scripts / "build_stage_a_bridge_svg_v7.py"
    prep_ascii = scripts / "prepare_ascii_dataset_v7.py"

    assets_ascii = out_dir / f"{prefix}_assets_ascii.txt"
    assets_manifest = out_dir / f"{prefix}_assets_ascii_manifest.json"
    stage_a_syn_prefix = f"{prefix}_stage_a_syn"
    stage_b_syn_prefix = f"{prefix}_stage_b_syn"
    stage_a_syn_train = out_dir / f"{stage_a_syn_prefix}_svg_train.txt"
    stage_b_syn_train = out_dir / f"{stage_b_syn_prefix}_svg_train.txt"
    stage_a_syn_instruction = out_dir / f"{stage_a_syn_prefix}_instruction_train.txt"
    stage_b_syn_instruction = out_dir / f"{stage_b_syn_prefix}_instruction_train.txt"

    stage_a_base_raw = out_dir / f"{prefix}_stage_a_base_raw.txt"
    stage_a_base = out_dir / f"{prefix}_stage_a_base.txt"
    bridge_txt = out_dir / f"{prefix}_stage_a_bridge.txt"
    bridge_manifest = out_dir / f"{prefix}_stage_a_bridge_manifest.json"
    stage_a_plus_bridge_raw = out_dir / f"{prefix}_stage_a_plus_bridge_raw.txt"
    stage_a_plus_bridge = out_dir / f"{prefix}_stage_a_plus_bridge.txt"
    stage_b_raw = out_dir / f"{prefix}_stage_b_raw.txt"
    stage_b = out_dir / f"{prefix}_stage_b.txt"
    tokenizer_corpus_raw = out_dir / f"{prefix}_tokenizer_corpus_raw.txt"
    tokenizer_corpus = out_dir / f"{prefix}_tokenizer_corpus.txt"

    color_prefix = f"{prefix}_color_theory"
    color_instruction_train = out_dir / f"{color_prefix}_instruction_train.txt"
    color_manifest = out_dir / f"{color_prefix}_manifest.json"
    manifest_out = out_dir / f"{prefix}_manifest.json"

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

    stage_a_cmd = [
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
    stage_b_cmd = [
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
    if spec_catalog_path is not None:
        stage_a_cmd.extend(["--spec-catalog", str(spec_catalog_path), "--spec-stage", "pretrain_a"])
        stage_b_cmd.extend(["--spec-catalog", str(spec_catalog_path), "--spec-stage", "pretrain_b"])
    _run(stage_a_cmd)
    _run(stage_b_cmd)

    if int(args.color_theory_samples) > 0:
        cmd = [
            str(py),
            str(gen_color),
            "--out-dir",
            str(out_dir),
            "--prefix",
            color_prefix,
            "--num-samples",
            str(int(args.color_theory_samples)),
            "--holdout-ratio",
            str(float(args.holdout_ratio)),
            "--seed",
            str(int(args.seed) + 303),
            "--min-contrast",
            str(float(args.color_min_contrast)),
            "--external-palette-prob",
            str(float(args.color_external_palette_prob)),
        ]
        for pf in args.color_palette_file:
            cmd.extend(["--palette-file", str(pf)])
        _run(cmd)

    coverage_checks: list[dict[str, Any]] = []
    for cov_path in (
        out_dir / f"{stage_a_syn_prefix}_coverage_manifest.json",
        out_dir / f"{stage_b_syn_prefix}_coverage_manifest.json",
    ):
        if not cov_path.exists():
            continue
        payload = _load_json(cov_path)
        gate = payload.get("gate") if isinstance(payload, dict) else {}
        passed = bool(gate.get("passed")) if isinstance(gate, dict) else False
        failures = list(gate.get("failures") or []) if isinstance(gate, dict) else []
        coverage_checks.append(
            {
                "path": str(cov_path),
                "passed": passed,
                "failures": failures,
            }
        )
        if args.strict_coverage and (not passed):
            raise SystemExit(
                "coverage gate failed for synthetic dataset:\n"
                f"  manifest: {cov_path}\n"
                "  failures:\n  - "
                + "\n  - ".join([str(x) for x in failures])
            )

    # Build bridge from strict svg rows, then emit canonical task-conditioned outputs.
    stage_a_base_rows = _merge_rows(
        sources=[stage_a_syn_train],
        max_line_chars=int(args.max_line_chars),
        dedupe=bool(args.dedupe),
        shuffle=bool(args.shuffle),
        seed=int(args.seed),
    )
    _write_rows(stage_a_base_raw, stage_a_base_rows)
    _prepare_ascii(prep_ascii, py, stage_a_base_raw, stage_a_base, svg_only=True)

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

    stage_a_task_syn_all = _instruction_rows(stage_a_syn_instruction, int(args.max_line_chars))
    stage_a_task_syn = _rows_by_ratio(stage_a_task_syn_all, stage_a_task_ratio, int(args.seed) + 21)
    if spec_catalog_path is not None:
        bridge_task_rows = _wrap_svg_rows_as_tags(
            svg_rows=_read_rows(bridge_txt),
            tag_prefixes=_TAG_TEMPLATES_BRIDGE,
            seed=int(args.seed) + 22,
            max_rows=int(args.bridge_task_max),
            max_line_chars=int(args.max_line_chars),
        )
    else:
        bridge_task_rows = _wrap_svg_rows_as_task(
            svg_rows=_read_rows(bridge_txt),
            task_templates=_TASK_TEMPLATES_BRIDGE,
            seed=int(args.seed) + 22,
            max_rows=int(args.bridge_task_max),
            max_line_chars=int(args.max_line_chars),
        )
    stage_a_plus_bridge_task_rows = _merge_row_lists(
        sources=[stage_a_task_syn, bridge_task_rows],
        max_line_chars=int(args.max_line_chars),
        dedupe=bool(args.dedupe),
        shuffle=bool(args.shuffle),
        seed=int(args.seed) + 23,
    )
    _write_rows(stage_a_plus_bridge_raw, stage_a_plus_bridge_task_rows)
    _prepare_ascii(prep_ascii, py, stage_a_plus_bridge_raw, stage_a_plus_bridge, svg_only=False)

    stage_b_task_syn_all = _instruction_rows(stage_b_syn_instruction, int(args.max_line_chars))
    stage_b_task_syn = _rows_by_ratio(stage_b_task_syn_all, stage_b_task_ratio, int(args.seed) + 31)
    if spec_catalog_path is not None:
        assets_task_rows = _wrap_svg_rows_as_tags(
            svg_rows=_read_rows(assets_ascii),
            tag_prefixes=_TAG_TEMPLATES_ASSET,
            seed=int(args.seed) + 32,
            max_rows=int(args.assets_task_max),
            max_line_chars=int(args.max_line_chars),
        )
    else:
        assets_task_rows = _wrap_svg_rows_as_task(
            svg_rows=_read_rows(assets_ascii),
            task_templates=_TASK_TEMPLATES_ASSET,
            seed=int(args.seed) + 32,
            max_rows=int(args.assets_task_max),
            max_line_chars=int(args.max_line_chars),
        )
    color_task_rows = _instruction_rows(color_instruction_train, int(args.max_line_chars))
    stage_b_rows = _merge_row_lists(
        sources=[_read_rows(stage_a_plus_bridge), stage_b_task_syn, assets_task_rows, color_task_rows],
        max_line_chars=int(args.max_line_chars),
        dedupe=bool(args.dedupe),
        shuffle=bool(args.shuffle),
        seed=int(args.seed) + 33,
    )
    _write_rows(stage_b_raw, stage_b_rows)
    _prepare_ascii(prep_ascii, py, stage_b_raw, stage_b, svg_only=False)

    tokenizer_corpus_rows = _merge_row_lists(
        sources=[_read_rows(stage_a_plus_bridge), _read_rows(stage_b)],
        max_line_chars=int(args.max_line_chars),
        dedupe=bool(args.dedupe),
        shuffle=bool(args.shuffle),
        seed=int(args.seed) + 41,
    )
    _write_rows(tokenizer_corpus_raw, tokenizer_corpus_rows)
    _prepare_ascii(prep_ascii, py, tokenizer_corpus_raw, tokenizer_corpus, svg_only=False)

    manifest = {
        "format": "v7-svg-pretrain-corpus-pack",
        "seed": int(args.seed),
        "assets_glob": str(args.assets_glob),
        "stage_a_types": str(args.stage_a_types),
        "stage_b_types": str(args.stage_b_types),
        "fill_mode_a": str(args.fill_mode_a),
        "fill_mode_b": str(args.fill_mode_b),
        "stage_a_task_ratio": float(stage_a_task_ratio),
        "stage_b_task_ratio": float(stage_b_task_ratio),
        "max_line_chars": int(args.max_line_chars),
        "conditioning_format": (
            "task_plus_svg"
            if spec_catalog_path is None
            else ("mixed_conditioning" if int(args.color_theory_samples) > 0 else "tag_plus_svg")
        ),
        "spec_catalog": str(spec_catalog_path) if spec_catalog_path is not None else None,
        "coverage_checks": coverage_checks,
        "color_theory_samples": int(args.color_theory_samples),
        "color_external_palette_prob": float(args.color_external_palette_prob),
        "paths": {
            "assets_ascii": str(assets_ascii),
            "stage_a_syn_train": str(stage_a_syn_train),
            "stage_a_syn_instruction": str(stage_a_syn_instruction),
            "stage_b_syn_train": str(stage_b_syn_train),
            "stage_b_syn_instruction": str(stage_b_syn_instruction),
            "stage_a_bridge": str(bridge_txt),
            "stage_a_plus_bridge": str(stage_a_plus_bridge),
            "stage_b": str(stage_b),
            "tokenizer_corpus": str(tokenizer_corpus),
            "color_theory_instruction_train": str(color_instruction_train),
            "color_theory_manifest": str(color_manifest),
            "bridge_manifest": str(bridge_manifest),
        },
        "stats": {
            "assets_ascii": _stats(assets_ascii),
            "stage_a_syn_train": _stats(stage_a_syn_train),
            "stage_a_syn_instruction": _stats(stage_a_syn_instruction),
            "stage_b_syn_train": _stats(stage_b_syn_train),
            "stage_b_syn_instruction": _stats(stage_b_syn_instruction),
            "stage_a_plus_bridge": _stats(stage_a_plus_bridge),
            "stage_b": _stats(stage_b),
            "tokenizer_corpus": _stats(tokenizer_corpus),
            "color_theory_instruction_train": _stats(color_instruction_train),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[OK] stage_a_plus_bridge:", stage_a_plus_bridge)
    print("[OK] stage_b:", stage_b)
    print("[OK] tokenizer_corpus:", tokenizer_corpus)
    print("[OK] manifest:", manifest_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
