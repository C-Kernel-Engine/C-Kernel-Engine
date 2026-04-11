#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

TOKENIZER_MAX_SOURCE_CHARS = 40000
TOKENIZER_DROP_LOCAL_TAGS = frozenset({"style", "script", "namedview", "title", "desc", "path-effect"})
STRUCTURAL_MAX_ROW_CHARS = 12000
STRUCTURAL_DROP_LOCAL_TAGS = frozenset({"style", "script", "namedview", "title", "desc", "path-effect"})
STRUCTURAL_REJECT_SUBSTRINGS = (
    "<style",
    "inkscape",
    "sodipodi",
    "xmlns:ns",
)

SHAPE_TAGS = ("[circle]", "[rect]", "[line]", "[triangle]", "[ellipse]", "[polygon]")
CHART_TAGS = ("[bar-chart]", "[line-chart]", "[scatter]", "[table]")
INFO_TAGS = ("[infographic]", "[card]", "[legend]", "[flow]", "[timeline]")
COUNT_TAGS = ("[bars:3]", "[bars:5]", "[bars:7]", "[points:4]", "[points:6]", "[points:8]", "[shapes:2]", "[shapes:3]")
ORDER_TAGS = ("[ascending]", "[descending]", "[mixed]", "[axes]", "[trend-line]", "[labeled]", "[values]")
PALETTE_TAGS = ("[palette:neutral]", "[palette:bold]", "[palette:warm]", "[palette:cool]", "[palette:pastel]", "[palette:dark]")
STYLE_TAGS = ("[style:gradient]", "[style:minimal]", "[style:filled]", "[style:outlined]")
LAYOUT_TAGS = ("[layout:center]", "[layout:tiled]", "[layout:grid]", "[layout:stacked]", "[layout:horizontal]", "[layout:header]", "[layout:flow]")
COMPLEXITY_TAGS = ("[complexity:simple]", "[complexity:moderate]", "[complexity:rich]")
TAG_SEED_REPEAT = 8

COMMON_ASCII_MAP: tuple[tuple[str, str], ...] = (
    ("→", "->"),
    ("←", "<-"),
    ("↔", "<->"),
    ("⇒", "=>"),
    ("⇐", "<="),
    ("±", "+/-"),
    ("×", "x"),
    ("÷", "/"),
    ("≤", "<="),
    ("≥", ">="),
    ("≠", "!="),
    ("≈", "~"),
    ("∞", "inf"),
    ("—", "-"),
    ("–", "-"),
    ("−", "-"),
    ("…", "..."),
    ("•", "-"),
    ("●", "o"),
    ("○", "o"),
    ("◆", "<>"),
    ("■", "[]"),
    ("▁", "_"),
    ("µ", "u"),
    ("°", "deg"),
    ("α", "alpha"),
    ("β", "beta"),
    ("γ", "gamma"),
    ("δ", "delta"),
    ("Δ", "Delta"),
    ("π", "pi"),
    ("λ", "lambda"),
    ("Ω", "Ohm"),
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _ascii_map_common(text: str) -> str:
    out = text
    for src, dst in COMMON_ASCII_MAP:
        if src in out:
            out = out.replace(src, dst)
    return out


def _ascii_escape(text: str) -> str:
    return "".join(ch if ord(ch) < 128 else f"&#x{ord(ch):X};" for ch in text)


def _ascii_clean(text: str) -> str:
    return _ascii_escape(_ascii_map_common(text))


def _svg_bytes(elem: ET.Element) -> str:
    return _ascii_clean(ET.tostring(elem, encoding="unicode"))


def _drop_local_tags_in_place(elem: ET.Element, drop_local_tags: set[str] | frozenset[str]) -> None:
    if not drop_local_tags:
        return
    kept: list[ET.Element] = []
    for child in list(elem):
        if _local(child.tag) in drop_local_tags:
            continue
        _drop_local_tags_in_place(child, drop_local_tags)
        kept.append(child)
    elem[:] = kept


def _wrap_svg(root: ET.Element, children: list[ET.Element]) -> str:
    wrapped = ET.Element(root.tag, dict(root.attrib))
    for child in children:
        wrapped.append(copy.deepcopy(child))
    return _svg_bytes(wrapped)


def _extract_full_svg(svg_path: Path, *, drop_local_tags: set[str] | frozenset[str] | None = None) -> str:
    root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    _drop_local_tags_in_place(root, drop_local_tags or frozenset())
    return _svg_bytes(root)


def _extract_structural_rows(
    svg_path: Path,
    *,
    drop_local_tags: set[str] | frozenset[str] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    _drop_local_tags_in_place(root, drop_local_tags or frozenset())
    top_children = list(root)
    defs_children = [child for child in top_children if _local(child.tag) == "defs"]
    if defs_children:
        text = _wrap_svg(root, defs_children[:1])
        rows.append({"kind": "defs", "text": text})
    for idx, child in enumerate(top_children):
        if _local(child.tag) != "g":
            continue
        if len(list(child)) < 2:
            continue
        text = _wrap_svg(root, defs_children[:1] + [child] if defs_children else [child])
        rows.append({"kind": f"group_{idx}", "text": text})
    return rows


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        h = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append({**row, "sha256": h})
    return out


def _keep_structural_row(text: str) -> bool:
    if not text:
        return False
    if len(text) > STRUCTURAL_MAX_ROW_CHARS:
        return False
    lowered = text.lower()
    return not any(bad in lowered for bad in STRUCTURAL_REJECT_SUBSTRINGS)


def _build_tag_seed_rows() -> list[dict]:
    rows: list[dict] = []
    all_tags = (
        SHAPE_TAGS
        + CHART_TAGS
        + INFO_TAGS
        + COUNT_TAGS
        + ORDER_TAGS
        + PALETTE_TAGS
        + STYLE_TAGS
        + LAYOUT_TAGS
        + COMPLEXITY_TAGS
    )

    def add(prompt: str) -> None:
        for _ in range(TAG_SEED_REPEAT):
            rows.append({"source_path": "tag_seed", "kind": "tag_seed", "text": prompt})

    for tag in all_tags:
        add(f"{tag}<svg width=\"64\" height=\"64\"></svg><eos>")

    for tag in SHAPE_TAGS:
        add(f"{tag}[palette:cool][style:minimal][layout:center]<svg width=\"64\" height=\"64\"></svg><eos>")
    for tag in CHART_TAGS:
        count = "[bars:5]" if tag == "[bar-chart]" else "[points:6]"
        add(f"{tag}{count}[palette:warm][style:filled][axes][labeled]<svg width=\"96\" height=\"64\"></svg><eos>")
    for tag in INFO_TAGS:
        add(f"{tag}[palette:dark][style:gradient][layout:horizontal][complexity:moderate]<svg width=\"120\" height=\"80\"></svg><eos>")
    for tag in COUNT_TAGS:
        add(f"[infographic]{tag}[palette:neutral][style:minimal][layout:grid]<svg width=\"96\" height=\"64\"></svg><eos>")
    for tag in ORDER_TAGS:
        add(f"[bar-chart][bars:5]{tag}[palette:bold][style:filled]<svg width=\"96\" height=\"64\"></svg><eos>")
    for tag in PALETTE_TAGS:
        add(f"[rect]{tag}[style:minimal][layout:center]<svg width=\"64\" height=\"64\"></svg><eos>")
    for tag in STYLE_TAGS:
        add(f"[circle][palette:cool]{tag}[layout:center]<svg width=\"64\" height=\"64\"></svg><eos>")
    for tag in LAYOUT_TAGS:
        add(f"[infographic][palette:cool][style:minimal]{tag}<svg width=\"120\" height=\"80\"></svg><eos>")
    for tag in COMPLEXITY_TAGS:
        add(f"[infographic][palette:dark][style:gradient][layout:grid]{tag}<svg width=\"120\" height=\"80\"></svg><eos>")

    return rows


def _reserved_control_tokens() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for group in (
        SHAPE_TAGS,
        CHART_TAGS,
        INFO_TAGS,
        COUNT_TAGS,
        ORDER_TAGS,
        PALETTE_TAGS,
        STYLE_TAGS,
        LAYOUT_TAGS,
        COMPLEXITY_TAGS,
    ):
        for token in group:
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
    return ordered


def _write_rows(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(_ascii_clean(r.replace("\n", " ").strip()) for r in rows if r.strip())
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize spec03 SVG pretrain/tokenizer/holdout artifacts from classification manifest")
    ap.add_argument("--workspace", required=True, help="Dataset workspace root, e.g. version/v7/data/spec03")
    ap.add_argument("--force", action="store_true", help="Overwrite existing materialized outputs")
    args = ap.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    manifest_path = workspace / "manifests" / "asset_classification_manifest.json"
    manifest = _load_json(manifest_path)
    suggested = manifest.get("suggested_splits", {})
    entries = manifest.get("entries", [])
    entry_by_path = {str(e["normalized_path"]): e for e in entries}

    pretrain_dir = workspace / "pretrain"
    holdout_dir = workspace / "holdout"
    tokenizer_dir = workspace / "tokenizer"

    outputs = [
        pretrain_dir / "spec03_small_full_pretrain.txt",
        pretrain_dir / "spec03_structural_pretrain.txt",
        holdout_dir / "spec03_holdout_manifest.json",
        holdout_dir / "spec03_holdout_paths.txt",
        tokenizer_dir / "spec03_tokenizer_corpus.txt",
        tokenizer_dir / "spec03_tag_seed_rows.txt",
        tokenizer_dir / "spec03_reserved_control_tokens.txt",
        tokenizer_dir / "spec03_tokenizer_corpus_manifest.json",
        pretrain_dir / "spec03_pretrain_materialization_manifest.json",
    ]
    if not args.force:
        existing = [str(p) for p in outputs if p.exists()]
        if existing:
            raise SystemExit("ERROR: outputs already exist; rerun with --force\n  " + "\n  ".join(existing))

    small_full_paths = [Path(p) for p in suggested.get("pretrain_small_full", [])]
    structural_paths = [Path(p) for p in suggested.get("pretrain_structural", [])]
    holdout_paths = [Path(p) for p in suggested.get("holdout_candidates", [])]

    small_rows = []
    for path in small_full_paths:
        text = _extract_full_svg(path).replace("\n", " ").strip()
        small_rows.append({"source_path": str(path), "kind": "full_svg", "text": text})
    small_rows = _dedupe_rows(small_rows)

    structural_rows = []
    for path in structural_paths:
        for row in _extract_structural_rows(path, drop_local_tags=STRUCTURAL_DROP_LOCAL_TAGS):
            if not _keep_structural_row(row["text"]):
                continue
            structural_rows.append({"source_path": str(path), **row})
    structural_rows = _dedupe_rows(structural_rows)

    tokenizer_excluded_large_sources: list[str] = []
    tokenizer_source_paths = sorted({str(path) for path in small_full_paths + structural_paths})
    tokenizer_allowed_sources: set[str] = set()
    for src in tokenizer_source_paths:
        meta = entry_by_path.get(str(src), {})
        chars = int(meta.get("chars") or 0)
        if chars > TOKENIZER_MAX_SOURCE_CHARS:
            tokenizer_excluded_large_sources.append(str(src))
            continue
        tokenizer_allowed_sources.add(str(src))

    tokenizer_small_rows = []
    for path in small_full_paths:
        if str(path) not in tokenizer_allowed_sources:
            continue
        text = _extract_full_svg(path, drop_local_tags=TOKENIZER_DROP_LOCAL_TAGS).replace("\n", " ").strip()
        tokenizer_small_rows.append({"source_path": str(path), "kind": "full_svg", "text": text})
    tokenizer_small_rows = _dedupe_rows(tokenizer_small_rows)

    tokenizer_structural_rows = []
    for path in structural_paths:
        if str(path) not in tokenizer_allowed_sources:
            continue
        for row in _extract_structural_rows(path, drop_local_tags=TOKENIZER_DROP_LOCAL_TAGS):
            tokenizer_structural_rows.append({"source_path": str(path), **row})
    tokenizer_structural_rows = _dedupe_rows(tokenizer_structural_rows)
    tag_seed_rows = _build_tag_seed_rows()
    reserved_control_tokens = _reserved_control_tokens()

    tokenizer_rows = _dedupe_rows(
        [{"source_path": row["source_path"], "kind": row["kind"], "text": row["text"]} for row in tokenizer_small_rows]
        + [{"source_path": row["source_path"], "kind": row["kind"], "text": row["text"]} for row in tokenizer_structural_rows]
        + [{"source_path": row["source_path"], "kind": row["kind"], "text": row["text"]} for row in tag_seed_rows]
    )

    _write_rows(pretrain_dir / "spec03_small_full_pretrain.txt", [r["text"] for r in small_rows])
    _write_rows(pretrain_dir / "spec03_structural_pretrain.txt", [r["text"] for r in structural_rows])
    _write_rows(holdout_dir / "spec03_holdout_paths.txt", [str(p) for p in holdout_paths])
    _write_rows(tokenizer_dir / "spec03_tokenizer_corpus.txt", [r["text"] for r in tokenizer_rows])
    _write_rows(tokenizer_dir / "spec03_tag_seed_rows.txt", [r["text"] for r in tag_seed_rows])
    _write_rows(tokenizer_dir / "spec03_reserved_control_tokens.txt", reserved_control_tokens)

    holdout_manifest = {
        "schema": "ck.spec03_holdout_manifest.v1",
        "workspace": str(workspace),
        "count": len(holdout_paths),
        "entries": [
            {
                "normalized_path": str(path),
                "family": entry_by_path.get(str(path), {}).get("family"),
                "roles": entry_by_path.get(str(path), {}).get("roles", []),
                "chars": entry_by_path.get(str(path), {}).get("chars"),
            }
            for path in holdout_paths
        ],
    }
    (holdout_dir / "spec03_holdout_manifest.json").write_text(json.dumps(holdout_manifest, indent=2), encoding="utf-8")

    tokenizer_manifest = {
        "schema": "ck.spec03_tokenizer_corpus_manifest.v1",
        "workspace": str(workspace),
        "small_full_rows": len(small_rows),
        "structural_rows": len(structural_rows),
        "tokenizer_rows": len(tokenizer_rows),
        "tag_seed_rows": len(tag_seed_rows),
        "reserved_control_tokens": len(reserved_control_tokens),
        "kind_counts": dict(Counter(r["kind"] for r in tokenizer_rows)),
        "sources": {
            "small_full_paths": len(small_full_paths),
            "structural_paths": len(structural_paths),
            "tokenizer_allowed_sources": len(tokenizer_allowed_sources),
            "tokenizer_excluded_large_sources": len(tokenizer_excluded_large_sources),
        },
        "tokenizer_filters": {
            "max_source_chars": TOKENIZER_MAX_SOURCE_CHARS,
            "drop_local_tags": sorted(TOKENIZER_DROP_LOCAL_TAGS),
            "structural_max_row_chars": STRUCTURAL_MAX_ROW_CHARS,
            "structural_drop_local_tags": sorted(STRUCTURAL_DROP_LOCAL_TAGS),
            "structural_reject_substrings": list(STRUCTURAL_REJECT_SUBSTRINGS),
            "excluded_large_sources": tokenizer_excluded_large_sources,
            "tokenizer_small_full_rows": len(tokenizer_small_rows),
            "tokenizer_structural_rows": len(tokenizer_structural_rows),
            "tokenizer_tag_seed_rows": len(tag_seed_rows),
        },
        "artifacts": {
            "small_full_pretrain": str(pretrain_dir / "spec03_small_full_pretrain.txt"),
            "structural_pretrain": str(pretrain_dir / "spec03_structural_pretrain.txt"),
            "tokenizer_corpus": str(tokenizer_dir / "spec03_tokenizer_corpus.txt"),
            "tag_seed_rows": str(tokenizer_dir / "spec03_tag_seed_rows.txt"),
            "reserved_control_tokens": str(tokenizer_dir / "spec03_reserved_control_tokens.txt"),
        },
    }
    (tokenizer_dir / "spec03_tokenizer_corpus_manifest.json").write_text(json.dumps(tokenizer_manifest, indent=2), encoding="utf-8")

    materialization_manifest = {
        "schema": "ck.spec03_pretrain_materialization.v1",
        "workspace": str(workspace),
        "source_classification_manifest": str(manifest_path),
        "counts": {
            "small_full_rows": len(small_rows),
            "structural_rows": len(structural_rows),
            "tokenizer_rows": len(tokenizer_rows),
            "tag_seed_rows": len(tag_seed_rows),
            "holdout_rows": len(holdout_paths),
        },
        "artifacts": {
            "small_full_pretrain": str(pretrain_dir / "spec03_small_full_pretrain.txt"),
            "structural_pretrain": str(pretrain_dir / "spec03_structural_pretrain.txt"),
            "holdout_manifest": str(holdout_dir / "spec03_holdout_manifest.json"),
            "holdout_paths": str(holdout_dir / "spec03_holdout_paths.txt"),
            "tokenizer_corpus": str(tokenizer_dir / "spec03_tokenizer_corpus.txt"),
            "tag_seed_rows": str(tokenizer_dir / "spec03_tag_seed_rows.txt"),
            "reserved_control_tokens": str(tokenizer_dir / "spec03_reserved_control_tokens.txt"),
            "tokenizer_manifest": str(tokenizer_dir / "spec03_tokenizer_corpus_manifest.json"),
        },
    }
    (pretrain_dir / "spec03_pretrain_materialization_manifest.json").write_text(json.dumps(materialization_manifest, indent=2), encoding="utf-8")

    print(f"[OK] small_full_rows={len(small_rows)} structural_rows={len(structural_rows)} tokenizer_rows={len(tokenizer_rows)} holdout_rows={len(holdout_paths)}")
    print(f"[OK] pretrain_dir={pretrain_dir}")
    print(f"[OK] tokenizer_dir={tokenizer_dir}")
    print(f"[OK] holdout_dir={holdout_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
