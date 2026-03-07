#!/usr/bin/env python3
"""Build a self-contained interactive SVG dataset viewer from spec03 manifests.

Uses the Antsand/IR-Visualizer brand system. All data is embedded as JSON so the
viewer works offline with full search, sort, gallery, quality checks, and provenance.
"""
from __future__ import annotations

import argparse
import base64
import json
from collections import Counter
from pathlib import Path
import re
from typing import Any

PROMPT_TAG_RE = re.compile(r"\[[^\]]+\]")
SVG_VIEWBOX_RE = re.compile(r'viewBox=["\']([^"\']+)')
SVG_WIDTH_RE = re.compile(r'width=["\']([^"\']+)')
SVG_HEIGHT_RE = re.compile(r'height=["\']([^"\']+)')

SHAPE_PROMPT_TAGS = frozenset({"[circle]", "[rect]", "[line]", "[triangle]", "[ellipse]", "[polygon]"})
CHART_PROMPT_TAGS = frozenset({"[bar-chart]", "[line-chart]", "[scatter]", "[table]"})
INFO_PROMPT_TAGS = frozenset({"[infographic]", "[card]", "[legend]", "[flow]", "[timeline]"})
MODIFIER_PROMPT_TAGS = frozenset({"[ascending]", "[descending]", "[mixed]", "[axes]", "[trend-line]", "[labeled]", "[values]"})


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _json_for_embed(obj: Any) -> str:
    """Compact JSON safe for embedding in a <script> tag."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")


def _build_gallery_items(classified: dict, workspace: Path) -> list[dict]:
    """Build gallery items with embedded SVG data-URIs from classification manifest."""
    items = []
    for e in classified.get("entries", []):
        svg_path = e.get("normalized_path", "")
        # Try relative to workspace if absolute path doesn't exist
        if not Path(svg_path).exists():
            # Try normalized/ subdir
            nm = Path(svg_path).name
            for sub in ("normalized/repo", "normalized/bcgov_citz", "normalized/bcgov_hub", "normalized"):
                candidate = workspace / sub / nm
                if candidate.exists():
                    svg_path = str(candidate)
                    break
        try:
            svg_text = Path(svg_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
        items.append({
            "name": Path(e.get("source_path", svg_path)).name,
            "family": e.get("family", "unknown"),
            "source": e.get("source_name", "unknown"),
            "size_band": e.get("size_band", "unknown"),
            "chars": e.get("chars", 0),
            "features": [k for k, v in e.get("features", {}).items() if v],
            "roles": e.get("roles", []),
            "tags": e.get("tag_counts", {}),
            "placeholders": e.get("placeholders", {}),
            "data_uri": f"data:image/svg+xml;base64,{b64}",
            "sha": e.get("normalized_sha256", "")[:12],
        })
    items.sort(key=lambda g: (g["family"], g["name"]))
    return items


def _extract_prompt_tags(text: str) -> list[str]:
    svg_idx = text.lower().find("<svg")
    prefix = text[:svg_idx] if svg_idx >= 0 else text
    return PROMPT_TAG_RE.findall(prefix)


def _prompt_tag_family(tag: str) -> str:
    if tag in SHAPE_PROMPT_TAGS:
        return "shape"
    if tag in CHART_PROMPT_TAGS:
        return "chart"
    if tag in INFO_PROMPT_TAGS:
        return "info"
    if tag in MODIFIER_PROMPT_TAGS:
        return "modifier"
    inner = tag[1:-1]
    if ":" in inner:
        head = inner.split(":", 1)[0]
        if head in {"palette", "style", "layout", "complexity", "bars", "points", "shapes"}:
            return head
    return "other"


def _parse_svg_row_meta(text: str) -> dict[str, Any]:
    """Extract metadata columns from a single SVG text row."""
    vb = SVG_VIEWBOX_RE.search(text)
    width = SVG_WIDTH_RE.search(text)
    height = SVG_HEIGHT_RE.search(text)
    prompt_tags = _extract_prompt_tags(text)
    tags_count = text.count("<") - text.count("</") - text.count("<!") - text.count("<?")
    stripped = text.strip()
    starts_with_svg = stripped.startswith("<svg") or stripped.startswith("<ns0:svg")
    kind = "full_svg" if starts_with_svg else "structural"
    if prompt_tags and "<svg" in text.lower():
        kind = "prompt_svg"
    # Detect sub-kind from content patterns
    if "<defs" in text.lower() or "<ns0:defs" in text.lower():
        if len(text) < 3000:
            kind = "defs_fragment"
    root_size = ""
    if width or height:
        root_size = f"{width.group(1) if width else '?'}x{height.group(1) if height else '?'}"
    eos_token = "<|eos|>" if "<|eos|>" in text else "<eos>" if "<eos>" in text else ""
    return {
        "viewBox": vb.group(1) if vb else "",
        "root_size": root_size,
        "element_count": max(0, tags_count),
        "kind": kind,
        "prompt_tag_count": len(prompt_tags),
        "eos_token": eos_token,
    }


def _build_text_rows(workspace: Path) -> dict[str, Any]:
    """Build per-row structured data for each split, HuggingFace-style."""
    splits: dict[str, Any] = {}
    scan = [
        ("pretrain", "pretrain", "*full_pretrain*.txt"),
        ("structural", "pretrain", "*structural*.txt"),
        ("sft", "sft", "*.txt"),
        ("holdout", "holdout", "*.txt"),
        ("tokenizer", "tokenizer", "*.txt"),
    ]
    for split_name, subdir, glob_pat in scan:
        split_dir = workspace / subdir
        if not split_dir.exists():
            continue
        for txt_file in sorted(split_dir.glob(glob_pat)):
            if "manifest" in txt_file.name:
                continue
            try:
                lines = txt_file.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            total = len(lines)
            char_counts = [len(l) for l in lines]
            # Build per-row data (all rows — up to a cap for embedding size)
            MAX_ROWS = 200
            rows = []
            for i, line in enumerate(lines[:MAX_ROWS]):
                meta = _parse_svg_row_meta(line)
                rows.append({
                    "row": i,
                    "chars": len(line),
                    "kind": meta["kind"],
                    "prompt_tags": meta["prompt_tag_count"],
                    "root_size": meta["root_size"],
                    "viewBox": meta["viewBox"],
                    "elements": meta["element_count"],
                    "eos": meta["eos_token"],
                    "text": line[:400],  # truncated preview
                })
            splits[f"{split_name}/{txt_file.name}"] = {
                "split": split_name,
                "file": txt_file.name,
                "total_rows": total,
                "total_chars": sum(char_counts),
                "avg_chars": round(sum(char_counts) / total) if total else 0,
                "min_chars": min(char_counts) if char_counts else 0,
                "max_chars": max(char_counts) if char_counts else 0,
                "rows": rows,
                "capped": total > MAX_ROWS,
            }
    return splits


def _build_tokenizer_info(workspace: Path) -> dict[str, Any]:
    """Read tokenizer manifest and corpus stats."""
    info: dict[str, Any] = {"available": False}
    manifest_path = workspace / "tokenizer"
    if not manifest_path.exists():
        return info
    manifest: dict[str, Any] = {}
    for mf in sorted(manifest_path.glob("*manifest*.json")):
        try:
            manifest = _load_json(mf)
            info["manifest"] = manifest
            info["available"] = True
        except Exception:
            pass
    text_files: dict[str, Any] = {}
    corpus_lines: list[str] = []
    corpus_content_chars = 0
    tag_seed_lines: list[str] = []
    tag_seed_content_chars = 0
    for cf in sorted(manifest_path.glob("*.txt")):
        try:
            text = cf.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            content_chars = sum(len(line) for line in lines)
            text_files[cf.name] = {
                "rows": len(lines),
                "chars": len(text),
                "content_chars": content_chars,
                "samples": [line[:220] for line in lines[:6]],
            }
            info["available"] = True
            if "tokenizer_corpus" in cf.name:
                info["corpus_file"] = cf.name
                info["corpus_rows"] = len(lines)
                info["corpus_chars"] = len(text)
                info["corpus_content_chars"] = content_chars
                info["corpus_samples"] = [line[:500] for line in lines[:10]]
                corpus_lines = lines
                corpus_content_chars = content_chars
            if "tag_seed" in cf.name:
                info["tag_seed_file"] = cf.name
                info["tag_seed_rows"] = len(lines)
                info["tag_seed_chars"] = len(text)
                info["tag_seed_content_chars"] = content_chars
                info["tag_seed_samples"] = [line[:220] for line in lines[:10]]
                tag_seed_lines = lines
                tag_seed_content_chars = content_chars
        except Exception:
            pass
    if text_files:
        info["files"] = text_files

    snapshot = _load_json_if_exists(workspace / "dataset_snapshot.json") or {}
    run_dir_text = snapshot.get("run_dir")
    run_dir = Path(str(run_dir_text)).expanduser() if run_dir_text else None
    if run_dir and run_dir.exists():
        tokenizer_json = _load_json_if_exists(run_dir / "tokenizer.json") or {}
        added_tokens = tokenizer_json.get("added_tokens") if isinstance(tokenizer_json, dict) else None
        protected_lookup: dict[str, dict[str, Any]] = {}
        if isinstance(added_tokens, list):
            for row in added_tokens:
                if not isinstance(row, dict):
                    continue
                content = row.get("content")
                if not isinstance(content, str):
                    continue
                protected_lookup[content] = {
                    "id": row.get("id"),
                    "content": content,
                    "special": bool(row.get("special")),
                }
        reserved_tokens_path = manifest_path / "spec03_reserved_control_tokens.txt"
        protected_tokens: list[dict[str, Any]] = []
        if reserved_tokens_path.exists():
            reserved_tokens = [
                line.strip()
                for line in reserved_tokens_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
            for token in reserved_tokens:
                row = protected_lookup.get(token, {})
                protected_tokens.append({
                    "token": token,
                    "id": row.get("id"),
                    "family": _prompt_tag_family(token),
                    "protected": bool(row.get("special")),
                    "present": token in protected_lookup,
                })
        info["protected_tokens"] = protected_tokens
        info["protected_tokens_present"] = sum(1 for row in protected_tokens if row.get("present"))
        info["protected_tokens_expected"] = len(protected_tokens)

    if corpus_lines:
        prompt_rows = [line for line in corpus_lines if _extract_prompt_tags(line)]
        prompt_row_chars = sum(len(line) for line in prompt_rows)
        corpus_prompt_tag_counts: Counter[str] = Counter()
        eos_counts: Counter[str] = Counter()
        for line in prompt_rows:
            corpus_prompt_tag_counts.update(_extract_prompt_tags(line))
            if "<|eos|>" in line:
                eos_counts["<|eos|>"] += 1
            elif "<eos>" in line:
                eos_counts["<eos>"] += 1
            else:
                eos_counts["missing"] += 1

        canonical_tag_counts: Counter[str] = Counter()
        tag_family_counts: Counter[str] = Counter()
        prefix_tag_counts: list[int] = []
        prefix_lengths: list[int] = []
        prefix_samples: list[str] = []
        seen_prefixes: set[str] = set()
        for line in tag_seed_lines:
            tags = _extract_prompt_tags(line)
            canonical_tag_counts.update(tags)
            tag_family_counts.update(_prompt_tag_family(tag) for tag in tags)
            prefix_tag_counts.append(len(tags))
            svg_idx = line.lower().find("<svg")
            prefix = line[:svg_idx] if svg_idx >= 0 else line
            prefix_lengths.append(len(prefix))
            if prefix and prefix not in seen_prefixes and len(prefix_samples) < 8:
                seen_prefixes.add(prefix)
                prefix_samples.append(prefix)

        corpus_set = set(corpus_lines)
        missing_seed_rows = [line[:220] for line in tag_seed_lines if line not in corpus_set][:12]
        info["prompt_contract"] = {
            "prompt_rows": len(prompt_rows),
            "prompt_chars": prompt_row_chars,
            "prompt_row_share": (len(prompt_rows) / len(corpus_lines)) if corpus_lines else 0.0,
            "prompt_char_share": (prompt_row_chars / corpus_content_chars) if corpus_content_chars else 0.0,
            "tag_seed_rows_actual": len(tag_seed_lines),
            "tag_seed_chars_actual": tag_seed_content_chars,
            "canonical_tags": [
                {"tag": tag, "count": count, "family": _prompt_tag_family(tag)}
                for tag, count in sorted(canonical_tag_counts.items())
            ],
            "tag_family_counts": dict(sorted(tag_family_counts.items())),
            "corpus_prompt_tag_counts": dict(sorted(corpus_prompt_tag_counts.items())),
            "all_seed_rows_in_corpus": not missing_seed_rows,
            "missing_seed_rows": missing_seed_rows,
            "prefix_tag_avg": (sum(prefix_tag_counts) / len(prefix_tag_counts)) if prefix_tag_counts else 0.0,
            "prefix_tag_max": max(prefix_tag_counts) if prefix_tag_counts else 0,
            "prefix_len_avg": (sum(prefix_lengths) / len(prefix_lengths)) if prefix_lengths else 0.0,
            "prefix_len_max": max(prefix_lengths) if prefix_lengths else 0,
            "prefix_samples": prefix_samples,
            "eos_counts": dict(sorted(eos_counts.items())),
            "eos_variant_count": len([key for key, value in eos_counts.items() if value]),
        }
        drift_entries = []
        if manifest:
            drift_checks = (
                ("tokenizer_rows", manifest.get("tokenizer_rows"), len(corpus_lines)),
                ("tag_seed_rows", manifest.get("tag_seed_rows"), len(tag_seed_lines)),
                ("tag_seed_kind_count", (manifest.get("kind_counts") or {}).get("tag_seed"), len(prompt_rows)),
            )
            for field, manifest_value, actual_value in drift_checks:
                if manifest_value is None:
                    continue
                drift_entries.append({
                    "field": field,
                    "manifest": manifest_value,
                    "actual": actual_value,
                    "match": manifest_value == actual_value,
                })
        info["manifest_drift"] = {
            "entries": drift_entries,
            "count": sum(1 for entry in drift_entries if not entry["match"]),
        }
    return info


def _resolve_run_dir(workspace: Path) -> Path | None:
    snapshot = _load_json_if_exists(workspace / "dataset_snapshot.json")
    if not snapshot:
        return None
    run_dir = snapshot.get("run_dir")
    if not run_dir:
        return None
    path = Path(str(run_dir)).expanduser()
    return path if path.exists() else None


def _load_vocab_pieces(run_dir: Path | None) -> list[str]:
    if run_dir is None:
        return []
    pieces: list[str] = []
    seen: set[str] = set()
    vocab_path = run_dir / "tokenizer_vocab_index.jsonl"
    try:
        if vocab_path.exists():
            with vocab_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    piece = obj.get("piece")
                    if isinstance(piece, str) and piece not in seen:
                        seen.add(piece)
                        pieces.append(piece)
        tokenizer_json = run_dir / "tokenizer.json"
        if tokenizer_json.exists():
            doc = _load_json(tokenizer_json)
            added = doc.get("added_tokens") if isinstance(doc, dict) else None
            if isinstance(added, list):
                for row in added:
                    if not isinstance(row, dict):
                        continue
                    piece = row.get("content")
                    if isinstance(piece, str) and piece not in seen:
                        seen.add(piece)
                        pieces.append(piece)
    except Exception:
        return pieces
    return pieces


def _build_preflight_info(workspace: Path, raw_inventory: dict[str, Any],
                          normalized: dict[str, Any], classified: dict[str, Any],
                          tokenizer_info: dict[str, Any]) -> dict[str, Any]:
    run_dir = _resolve_run_dir(workspace)
    roundtrip = _load_json_if_exists(run_dir / "tokenizer_roundtrip.json") if run_dir else None
    fit_audit = _load_json_if_exists(workspace / "manifests" / "spec03_fit_audit_manifest.json") or {}
    holdout_manifest = _load_json_if_exists(workspace / "holdout" / "spec03_holdout_manifest.json") or {}
    vocab_pieces = _load_vocab_pieces(run_dir)
    piece_set = set(vocab_pieces)
    canonical_tags = [
        entry.get("tag", "")
        for entry in (tokenizer_info.get("prompt_contract") or {}).get("canonical_tags", [])
        if isinstance(entry, dict) and entry.get("tag")
    ]
    exact_control_coverage = {
        "expected": len(canonical_tags),
        "covered": sum(1 for tag in canonical_tags if tag in piece_set),
        "missing": [tag for tag in canonical_tags if tag not in piece_set][:24],
    }
    control_ratio = (
        exact_control_coverage["covered"] / exact_control_coverage["expected"]
        if exact_control_coverage["expected"] else 0.0
    )

    cross_patterns = ("</svg>\n<svg", "<eos>\n<svg", "<eos>\n[", "</svg>\n[")
    cross_hits = sorted({piece for piece in vocab_pieces if any(pat in piece for pat in cross_patterns)})

    structural_path = workspace / "pretrain" / "spec03_structural_pretrain.txt"
    structural_text = structural_path.read_text(encoding="utf-8", errors="replace") if structural_path.exists() else ""
    style_hits = {
        "<style": structural_text.count("<style"),
        "inkscape": structural_text.count("inkscape"),
        "sodipodi": structural_text.count("sodipodi"),
        "xmlns:ns": structural_text.count("xmlns:ns"),
    }
    structural_rows = structural_text.splitlines() if structural_text else []
    structural_max_chars = max((len(row) for row in structural_rows), default=0)

    fit_lookup: dict[str, dict[str, Any]] = {}
    for entry in fit_audit.get("entries", []):
        split = str(entry.get("split") or "")
        seq_len = entry.get("seq_len")
        if split and seq_len is not None:
            fit_lookup[f"{split}_{seq_len}"] = entry

    holdout_count = 0
    if isinstance(holdout_manifest.get("selected_holdout"), list):
        holdout_count = len(holdout_manifest["selected_holdout"])
    elif isinstance((holdout_manifest.get("summary") or {}).get("selected_holdout"), int):
        holdout_count = int(holdout_manifest["summary"]["selected_holdout"])

    canary_files = [
        workspace / "holdout" / "spec03_canary_prompts.json",
        workspace / "holdout" / "spec03_canary_prompts.jsonl",
        workspace / "holdout" / "spec03_canary_prompts.txt",
        workspace / "holdout" / "spec03_canary_manifest.json",
    ]
    canary_ready = any(path.exists() for path in canary_files)

    contract_files = [
        workspace / "contracts" / "TAG_CONTRACT.md",
        workspace / "contracts" / "spec03_eval_contract.svg.v1.json",
    ]
    contract_ready = all(path.exists() for path in contract_files)
    raw_ready = int(raw_inventory.get("imported_files") or 0) > 0
    normalized_ready = int(normalized.get("normalized_entries") or 0) > 0
    structural_ready = all(v == 0 for v in style_hits.values()) and structural_max_chars <= 10000
    tag_seed_rows = int(tokenizer_info.get("tag_seed_rows") or 0)
    tag_seed_ready = tag_seed_rows > 0
    roundtrip_ready = bool(roundtrip and roundtrip.get("status") == "pass" and roundtrip.get("exact_match") is True)
    cross_ready = len(cross_hits) == 0
    fit_ready = bool(fit_lookup.get("small_full_512") and fit_lookup.get("small_full_2048") and fit_lookup.get("structural_512") and fit_lookup.get("structural_2048"))
    canary_status = holdout_count > 0 and canary_ready
    fmt_i = lambda value: f"{int(value):,}"

    checklist = [
        {
            "key": "contract_present",
            "label": "Contract present",
            "status": "ok" if contract_ready else "err",
            "detail": f"{sum(1 for p in contract_files if p.exists())}/{len(contract_files)} contract files",
        },
        {
            "key": "raw_import",
            "label": "Raw import present",
            "status": "ok" if raw_ready else "err",
            "detail": f"{fmt_i(raw_inventory.get('imported_files') or 0)} imported assets",
        },
        {
            "key": "normalized_corpus",
            "label": "Normalized corpus present",
            "status": "ok" if normalized_ready else "err",
            "detail": f"{fmt_i(normalized.get('normalized_entries') or 0)} normalized assets",
        },
        {
            "key": "structural_cleanup",
            "label": "Structural cleanup status",
            "status": "ok" if structural_ready else "warn",
            "detail": f"style={style_hits['<style']} · inkscape={style_hits['inkscape']} · max_chars={fmt_i(structural_max_chars)}",
        },
        {
            "key": "tag_seed",
            "label": "Tag seed present",
            "status": "ok" if tag_seed_ready else "err",
            "detail": f"{fmt_i(tag_seed_rows)} tag-seed rows",
        },
        {
            "key": "tokenizer_roundtrip",
            "label": "Tokenizer roundtrip status",
            "status": "ok" if roundtrip_ready else ("warn" if roundtrip else "err"),
            "detail": (
                f"exact={roundtrip.get('exact_match')} · lines={fmt_i(roundtrip.get('input_lines') or 0)}"
                if roundtrip else "tokenizer_roundtrip.json missing"
            ),
        },
        {
            "key": "cross_row_merges",
            "label": "Cross-row merge status",
            "status": "ok" if cross_ready else "err",
            "detail": "0 bad cross-row tokens" if cross_ready else f"{fmt_i(len(cross_hits))} bad vocab pieces",
        },
        {
            "key": "control_coverage",
            "label": "Exact control-tag coverage",
            "status": "ok" if control_ratio >= 0.8 else ("warn" if control_ratio > 0 else "err"),
            "detail": f"{fmt_i(exact_control_coverage['covered'])}/{fmt_i(exact_control_coverage['expected'])} exact DSL tags as atoms",
        },
        {
            "key": "fit_audit",
            "label": "Fit audit summary for 512 and 2048",
            "status": "ok" if fit_ready else "warn",
            "detail": "small_full + structural recorded" if fit_ready else "fit audit incomplete",
        },
        {
            "key": "canary_readiness",
            "label": "Canary readiness",
            "status": "ok" if canary_status else ("warn" if holdout_count > 0 else "err"),
            "detail": f"holdout={fmt_i(holdout_count)} · prompts={'ready' if canary_ready else 'missing'}",
        },
    ]

    return {
        "available": True,
        "run_dir": str(run_dir) if run_dir else "",
        "roundtrip": roundtrip or {},
        "checklist": checklist,
        "ready_count": sum(1 for item in checklist if item["status"] == "ok"),
        "total_count": len(checklist),
        "cross_row_merge_hits": cross_hits[:12],
        "style_hits": style_hits,
        "structural_max_chars": structural_max_chars,
        "exact_control_coverage": exact_control_coverage,
        "fit_audit": fit_lookup,
        "holdout_count": holdout_count,
        "canary_prompt_files_present": canary_ready,
        "recommendation": {
            "dsl_atoms_required": exact_control_coverage["expected"] > 0,
            "reserved_control_tokens_recommended": control_ratio < 0.8,
            "reason": (
                "This prompt interface is a DSL. Control tags should be protected atoms, not left to frequency-based BPE merges."
                if control_ratio < 0.8
                else "Exact control-tag coverage is strong enough to proceed without reserved control tokens."
            ),
        },
    }


def build_html(workspace: Path, raw_inventory: dict[str, Any],
               normalized: dict[str, Any], classified: dict[str, Any]) -> str:
    gallery_items = _build_gallery_items(classified, workspace)
    text_rows = _build_text_rows(workspace)
    tokenizer_info = _build_tokenizer_info(workspace)
    preflight_info = _build_preflight_info(workspace, raw_inventory, normalized, classified, tokenizer_info)
    return (
        _HTML_PREFIX
        + "\n<script>\n"
        + f"const CK_WORKSPACE = {_json_for_embed(str(workspace))};\n"
        + f"const CK_RAW_INVENTORY = {_json_for_embed(raw_inventory)};\n"
        + f"const CK_NORMALIZED = {_json_for_embed(normalized)};\n"
        + f"const CK_CLASSIFIED = {_json_for_embed(classified)};\n"
        + f"const CK_GALLERY = {_json_for_embed(gallery_items)};\n"
        + f"const CK_TEXT_ROWS = {_json_for_embed(text_rows)};\n"
        + f"const CK_TOKENIZER = {_json_for_embed(tokenizer_info)};\n"
        + f"const CK_PREFLIGHT = {_json_for_embed(preflight_info)};\n"
        + "</script>\n"
        + _HTML_SUFFIX
    )


# ── HTML template (Antsand brand, IR-Visualizer style) ───────────────────────

_HTML_PREFIX = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Viewer | C-Kernel-Engine</title>
<style>
:root {
    --orange: #ffb400;
    --orange-dark: #e5a200;
    --orange-light: #ffc933;
    --dark: #2a2a2a;
    --dark-lighter: #363636;
    --dark-card: #323232;
    --grey: #454545;
    --grey-light: #555555;
    --text-primary: #f5f5f5;
    --text-secondary: #b0b0b0;
    --text-muted: #808080;
    --bg-dark: #232323;
    --white: #ffffff;
    --code-bg: #1a1a1a;
    --green: #47b475;
    --blue: #07adf8;
    --red: #e74c3c;
    --purple: #9b59b6;
    --teal: #1abc9c;
    --gradient-header: linear-gradient(135deg, #1a1c22 0%, #22252c 50%, #1a1c22 100%);
    --shadow-sm: 0 4px 12px rgba(0,0,0,0.25);
    --shadow-md: 0 8px 24px rgba(0,0,0,0.35);
    --shadow-lg: 0 20px 60px rgba(0,0,0,0.28);
    --transition: all 0.3s ease;
    --radius: 16px;
    --backdrop: blur(12px) saturate(1.4);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Space Grotesk', 'Avenir Next', 'Segoe UI', sans-serif;
    line-height: 1.6; color: var(--text-primary);
    background:
        radial-gradient(ellipse at 15% 0%, rgba(255,180,0,0.07), transparent 45%),
        radial-gradient(ellipse at 85% 5%, rgba(7,173,248,0.05), transparent 40%),
        radial-gradient(ellipse at 50% 100%, rgba(71,180,117,0.04), transparent 35%),
        var(--bg-dark);
    min-height: 100vh; display: flex; flex-direction: column;
    -webkit-font-smoothing: antialiased;
}
/* ── Header ─────────────────────────────────────────────────────── */
.site-header {
    background: var(--gradient-header); border-bottom: 3px solid var(--orange);
    padding: 0.75rem 2rem; display: flex; align-items: center;
    justify-content: space-between; flex-wrap: wrap; gap: 1rem;
    box-shadow: var(--shadow-md); position: sticky; top: 0; z-index: 100;
    backdrop-filter: var(--backdrop);
}
.header-brand { display: flex; align-items: center; gap: 0.75rem; text-decoration: none; }
.header-logo {
    width: 36px; height: 36px; background: var(--orange); border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; color: var(--dark); font-size: 0.9rem;
    font-family: 'JetBrains Mono', monospace;
}
.header-title { color: var(--white); font-size: 1.2rem; font-weight: 700; }
.header-subtitle { color: var(--orange); font-size: 0.75rem; font-weight: 500; }
.header-meta {
    display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;
}
.chip {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 10px; border-radius: 999px; font-size: 0.72rem;
    border: 1px solid var(--grey); background: rgba(255,255,255,0.03);
    color: var(--text-muted); font-family: 'JetBrains Mono', monospace;
}
.chip code { color: var(--orange); background: none; padding: 0; }
/* ── Tabs ───────────────────────────────────────────────────────── */
.tabs {
    display: flex; background: var(--dark); border-bottom: 1px solid var(--grey);
    overflow-x: auto;
}
.tab {
    padding: 0.65rem 1.25rem; cursor: pointer; color: var(--text-secondary);
    font-weight: 500; font-size: 0.85rem; border: none;
    border-bottom: 3px solid transparent; transition: var(--transition);
    white-space: nowrap; background: none; font-family: inherit;
}
.tab:hover { color: var(--text-primary); background: rgba(255,255,255,0.03); }
.tab.active { color: var(--orange); border-bottom-color: var(--orange); background: var(--dark-card); }
/* ── Main ───────────────────────────────────────────────────────── */
main {
    max-width: 1600px; margin: 0 auto; padding: 1.25rem 2rem 3rem;
    flex: 1; width: 100%;
}
.panel { display: none; }
.panel.active { display: block; }
/* ── Search bar ─────────────────────────────────────────────────── */
.search-strip {
    background: var(--dark-card); padding: 0.6rem 2rem;
    display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: center;
    border-bottom: 1px solid var(--grey);
}
.search-box {
    padding: 0.4rem 0.85rem; width: 280px; border-radius: 4px;
    border: 1px solid var(--grey); background: var(--code-bg);
    color: var(--text-primary); font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
}
.search-box:focus { outline: none; border-color: var(--orange); box-shadow: 0 0 0 2px rgba(255,180,0,0.15); }
.search-strip select {
    padding: 0.4rem 0.6rem; border-radius: 4px; border: 1px solid var(--grey);
    background: var(--code-bg); color: var(--text-primary); font-size: 0.82rem;
    font-family: inherit; cursor: pointer;
}
/* ── Stats grid ─────────────────────────────────────────────────── */
.stats-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
    gap: 0.7rem; margin: 0.85rem 0 1.25rem;
}
.stat-card {
    background: rgba(255,255,255,0.03); padding: 0.9rem 1rem; border-radius: var(--radius);
    text-align: center; border: 1px solid rgba(255,255,255,0.06);
    box-shadow: var(--shadow-sm); backdrop-filter: var(--backdrop);
    position: relative; overflow: hidden;
}
.stat-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--orange); opacity: 0; transition: opacity 0.3s;
}
.stat-card:hover::before { opacity: 1; }
.stat-value {
    font-size: 1.5rem; font-weight: 700; color: var(--orange); line-height: 1.2;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label {
    color: var(--text-muted); font-size: 0.7rem; margin-top: 0.15rem;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.stat-note { color: var(--text-muted); font-size: 0.68rem; margin-top: 0.25rem; line-height: 1.3; }
.stat-value.green { color: var(--green); }
.stat-value.blue  { color: var(--blue); }
.stat-value.red   { color: var(--red); }
/* ── Section cards ──────────────────────────────────────────────── */
.section-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: var(--radius); margin-bottom: 1rem; overflow: hidden;
    box-shadow: var(--shadow-sm); backdrop-filter: var(--backdrop);
}
.section-header {
    padding: 0.75rem 1rem; background: rgba(255,255,255,0.02); display: flex;
    align-items: center; justify-content: space-between;
    cursor: pointer; user-select: none;
}
.section-header:hover { background: rgba(255,255,255,0.05); }
.section-title {
    font-weight: 600; font-size: 0.92rem; display: flex; align-items: center; gap: 0.5rem;
}
.section-badge {
    padding: 2px 8px; border-radius: 999px; font-size: 0.66rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.badge-orange { background: rgba(255,180,0,0.12); color: var(--orange); border: 1px solid rgba(255,180,0,0.25); }
.badge-green  { background: rgba(71,180,117,0.12); color: var(--green); border: 1px solid rgba(71,180,117,0.25); }
.badge-blue   { background: rgba(7,173,248,0.12); color: var(--blue); border: 1px solid rgba(7,173,248,0.25); }
.badge-red    { background: rgba(231,76,60,0.12); color: var(--red); border: 1px solid rgba(231,76,60,0.25); }
.badge-purple { background: rgba(155,89,182,0.12); color: var(--purple); border: 1px solid rgba(155,89,182,0.25); }
.section-arrow { color: var(--orange); font-size: 0.7rem; transition: transform 0.2s; }
.section-body { padding: 0.9rem 1rem; }
/* ── Distribution bars ──────────────────────────────────────────── */
.dist-row { display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.45rem; }
.dist-label {
    width: 150px; min-width: 130px; font-size: 0.76rem; color: var(--text-secondary);
    text-align: right; font-family: 'JetBrains Mono', monospace;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.dist-bar-bg {
    flex: 1; height: 20px; background: var(--dark); border-radius: 4px;
    overflow: hidden; border: 1px solid rgba(255,255,255,0.04);
}
.dist-bar-fill {
    height: 100%; border-radius: 3px; transition: width 0.5s ease;
    display: flex; align-items: center; padding-left: 6px;
}
.dist-bar-fill span {
    font-size: 0.6rem; font-weight: 600; color: var(--white);
    font-family: 'JetBrains Mono', monospace; text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    white-space: nowrap;
}
.dist-count {
    width: 55px; min-width: 55px; font-size: 0.76rem; color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace; text-align: right;
}
/* ── Tables ─────────────────────────────────────────────────────── */
table {
    width: 100%; border-collapse: collapse; background: rgba(255,255,255,0.02);
    border-radius: var(--radius); overflow: hidden; border: 1px solid rgba(255,255,255,0.06);
}
th {
    background: rgba(255,255,255,0.03); color: var(--orange); padding: 0.6rem 0.8rem;
    text-align: left; font-weight: 600; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.5px; cursor: pointer;
    user-select: none; white-space: nowrap;
}
th:hover { color: var(--orange-light); }
th.sort-active { text-decoration: underline; text-underline-offset: 3px; }
td {
    padding: 0.5rem 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.04);
    color: var(--text-secondary); font-size: 0.8rem;
}
tr:hover { background: rgba(255,255,255,0.02); }
tr:last-child td { border-bottom: none; }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; }
/* ── Health dots ────────────────────────────────────────────────── */
.health-row { display: flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0; font-size: 0.8rem; }
.health-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.health-dot.ok   { background: var(--green); box-shadow: 0 0 4px rgba(71,180,117,0.5); }
.health-dot.warn { background: #f39c12; box-shadow: 0 0 4px rgba(243,156,18,0.5); }
.health-dot.err  { background: var(--red); box-shadow: 0 0 4px rgba(231,76,60,0.5); }
/* ── Grid layouts ───────────────────────────────────────────────── */
.grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
.grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem; }
/* ── Candidate lists ────────────────────────────────────────────── */
.candidate-list { list-style: none; padding: 0; }
.candidate-list li {
    padding: 0.3rem 0; border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.76rem; font-family: 'JetBrains Mono', monospace;
    color: var(--blue);
}
.candidate-list li:last-child { border-bottom: none; }
/* ── Color swatches ─────────────────────────────────────────────── */
.swatch-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem; }
.swatch {
    width: 34px; height: 34px; border-radius: 6px; border: 2px solid var(--grey);
    cursor: help; position: relative;
}
.swatch-label {
    font-size: 0.58rem; text-align: center; color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace; margin-top: 1px;
}
/* ── Pagination ─────────────────────────────────────────────────── */
.pagination { display: flex; gap: 0.3rem; align-items: center; justify-content: center; margin-top: 0.85rem; }
.page-btn {
    padding: 0.25rem 0.55rem; border-radius: 4px; border: 1px solid var(--grey);
    background: var(--dark); color: var(--text-secondary); font-size: 0.76rem;
    cursor: pointer; font-family: 'JetBrains Mono', monospace;
}
.page-btn:hover { border-color: var(--orange); color: var(--orange); }
.page-btn.active { background: var(--orange); color: var(--dark); border-color: var(--orange); font-weight: 700; }
.page-btn:disabled { opacity: 0.3; cursor: not-allowed; }
.page-info { font-size: 0.72rem; color: var(--text-muted); margin: 0 0.4rem; font-family: 'JetBrains Mono', monospace; }
/* ── Modal ──────────────────────────────────────────────────────── */
.modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.75);
    z-index: 200; align-items: center; justify-content: center; padding: 2rem;
}
.modal-overlay.open { display: flex; }
.modal {
    background: var(--dark-card); border: 1px solid var(--grey); border-radius: 12px;
    max-width: 900px; width: 100%; max-height: 85vh; overflow-y: auto; box-shadow: var(--shadow-lg);
}
.modal-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.85rem 1.1rem; border-bottom: 1px solid var(--grey);
}
.modal-header h3 { font-size: 1rem; color: var(--white); }
.modal-close {
    background: none; border: none; color: var(--text-muted); font-size: 1.4rem;
    cursor: pointer; padding: 0 0.25rem;
}
.modal-close:hover { color: var(--white); }
.modal-body { padding: 1.1rem; }
.source-block {
    background: var(--code-bg); border: 1px solid var(--grey); border-radius: var(--radius);
    padding: 0.75rem; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    color: var(--text-secondary); overflow-x: auto; white-space: pre-wrap;
    word-break: break-all; max-height: 220px; overflow-y: auto; line-height: 1.5;
}
/* ── Subhead styling ────────────────────────────────────────────── */
.subhead {
    font-size: 0.88rem; font-weight: 600; color: var(--text-primary);
    margin: 0 0 0.3rem; display: flex; align-items: center; gap: 0.4rem;
}
.subnote { color: var(--text-muted); font-size: 0.78rem; margin-bottom: 0.65rem; line-height: 1.4; }
/* ── SVG preview gallery ────────────────────────────────────────── */
.gallery-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 0.7rem;
}
.gallery-card {
    background: var(--dark); border: 1px solid var(--grey); border-radius: var(--radius);
    overflow: hidden; cursor: pointer; transition: var(--transition);
}
.gallery-card:hover { border-color: var(--orange); box-shadow: 0 0 0 1px var(--orange), var(--shadow-md); }
.gallery-render {
    width: 100%; aspect-ratio: 4/3; background: var(--white);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden; padding: 6px;
}
.gallery-render svg { max-width: 100%; max-height: 100%; }
.gallery-meta {
    padding: 0.4rem 0.6rem; font-size: 0.68rem; color: var(--text-muted);
}
.gallery-meta .idx { color: var(--orange); font-weight: 600; font-family: 'JetBrains Mono', monospace; }
/* ── Scrollbar ──────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--dark); }
::-webkit-scrollbar-thumb { background: var(--grey); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--grey-light); }
/* ── Responsive ─────────────────────────────────────────────────── */
@media (max-width: 1100px) { .grid-3 { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 768px) {
    main { padding: 0.75rem 1rem; }
    .site-header { padding: 0.5rem 1rem; }
    .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
    .search-box { width: 100%; }
    .stats-grid { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
    .gallery-grid { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }
}
/* ── SVG Gallery Viewer (full-screen) ──────────────────────────── */
.gal-toolbar {
    display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.75rem;
}
.gal-toolbar .gal-search {
    flex: 1; min-width: 180px; padding: 0.45rem 0.8rem; border-radius: 8px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    color: var(--text-primary); font-size: 0.82rem; font-family: var(--font-display, 'Space Grotesk', sans-serif);
    outline: none;
}
.gal-toolbar .gal-search:focus { border-color: var(--orange); }
.gal-pill {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    color: var(--text-secondary); padding: 0.3rem 0.65rem; border-radius: 999px;
    font-size: 0.72rem; cursor: pointer; transition: all 0.2s; white-space: nowrap;
}
.gal-pill:hover { border-color: rgba(255,180,0,0.4); color: var(--text-primary); }
.gal-pill.active { background: rgba(255,180,0,0.15); border-color: rgba(255,180,0,0.5); color: var(--orange); font-weight: 600; }
.gal-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 0.8rem;
}
.gal-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; overflow: hidden; cursor: pointer; transition: all 0.25s;
}
.gal-card:hover { border-color: rgba(255,180,0,0.35); box-shadow: 0 12px 40px rgba(0,0,0,0.35); transform: translateY(-2px); }
.gal-card .gal-thumb {
    height: 160px; background: #fff; display: flex; align-items: center; justify-content: center;
    padding: 10px; overflow: hidden; position: relative;
}
.gal-card .gal-thumb img { max-width: 100%; max-height: 100%; object-fit: contain; }
.gal-card .gal-thumb .gal-badge {
    position: absolute; top: 6px; left: 6px; padding: 0.12rem 0.45rem; border-radius: 999px;
    font-size: 0.6rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
}
.gal-card .gal-info { padding: 0.55rem 0.7rem; }
.gal-card .gal-info .gal-name {
    font-size: 0.78rem; font-weight: 600; color: var(--text-primary);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.gal-card .gal-info .gal-sub {
    font-size: 0.65rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;
}
.fam-chart       { background: rgba(255,180,0,0.85); color: #1a1a1a; }
.fam-flow        { background: rgba(7,173,248,0.85); color: #fff; }
.fam-architecture { background: rgba(160,124,248,0.85); color: #fff; }
.fam-technical   { background: rgba(71,180,117,0.85); color: #fff; }
.fam-infographic { background: rgba(240,80,80,0.85); color: #fff; }
.fam-other       { background: rgba(107,112,128,0.85); color: #fff; }
/* Gallery full-screen overlay */
.gal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.92);
    z-index: 300; flex-direction: column; backdrop-filter: blur(8px);
}
.gal-overlay.open { display: flex; }
.gal-overlay .gal-ov-bar {
    padding: 0.5rem 1.2rem; background: rgba(0,0,0,0.8); border-bottom: 1px solid rgba(255,255,255,0.08);
    display: flex; align-items: center; justify-content: space-between; flex-shrink: 0;
}
.gal-overlay .gal-ov-bar .gal-ov-title { font-size: 0.85rem; font-weight: 600; color: var(--text-primary); }
.gal-overlay .gal-ov-bar button {
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12);
    color: var(--orange); padding: 0.35rem 0.7rem; border-radius: 8px; cursor: pointer;
    font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;
}
.gal-overlay .gal-ov-bar button:hover { background: rgba(255,180,0,0.15); border-color: var(--orange); }
.gal-overlay .gal-ov-body { flex: 1; display: flex; overflow: hidden; }
.gal-overlay .gal-ov-canvas {
    flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden;
    cursor: grab; position: relative;
}
.gal-overlay .gal-ov-canvas:active { cursor: grabbing; }
.gal-overlay .gal-ov-canvas img {
    max-width: 90%; max-height: 90%; background: #fff; border-radius: 8px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5); transform-origin: center;
}
.gal-overlay .gal-ov-sidebar {
    width: 300px; background: rgba(26,28,34,0.95); border-left: 1px solid rgba(255,255,255,0.06);
    overflow-y: auto; padding: 1rem; flex-shrink: 0;
}
.gal-ov-sidebar h4 {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1px;
    color: var(--orange); margin: 0.8rem 0 0.4rem; font-weight: 700;
}
.gal-ov-sidebar h4:first-child { margin-top: 0; }
.gal-ov-meta-row {
    display: flex; justify-content: space-between; padding: 0.25rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.76rem;
}
.gal-ov-meta-row .mkey { color: var(--text-muted); }
.gal-ov-meta-row .mval { color: var(--text-primary); font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; }
.gal-ov-pill { display: inline-block; background: rgba(255,180,0,0.12); border: 1px solid rgba(255,180,0,0.25); color: var(--orange); padding: 0.1rem 0.4rem; border-radius: 999px; font-size: 0.65rem; font-family: 'JetBrains Mono', monospace; margin: 0.15rem 0.15rem 0 0; }
.token-pill { display: inline-block; background: rgba(7,173,248,0.12); border: 1px solid rgba(7,173,248,0.25); color: var(--blue); padding: 0.14rem 0.45rem; border-radius: 999px; font-size: 0.65rem; font-family: 'JetBrains Mono', monospace; margin: 0.15rem 0.15rem 0 0; }
.token-pill.family-palette,
.token-pill.family-style,
.token-pill.family-layout,
.token-pill.family-complexity { background: rgba(155,89,182,0.12); border-color: rgba(155,89,182,0.25); color: var(--purple); }
.token-pill.family-shape,
.token-pill.family-chart,
.token-pill.family-info { background: rgba(255,180,0,0.12); border-color: rgba(255,180,0,0.25); color: var(--orange); }
.token-pill.family-bars,
.token-pill.family-points,
.token-pill.family-shapes,
.token-pill.family-modifier { background: rgba(71,180,117,0.12); border-color: rgba(71,180,117,0.25); color: var(--green); }
.gal-ov-nav {
    position: absolute; top: 50%; transform: translateY(-50%);
    background: rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.1);
    color: var(--orange); font-size: 1.6rem; padding: 0.8rem 0.6rem;
    cursor: pointer; border-radius: 8px; z-index: 10;
}
.gal-ov-nav:hover { background: rgba(255,180,0,0.15); border-color: var(--orange); }
.gal-ov-prev { left: 10px; }
.gal-ov-next { right: 10px; }
/* ── Data Table (HuggingFace-style) ─────────────────────────────── */
.dt-header {
    display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap;
    margin-bottom: 0.75rem;
}
.dt-split-select {
    padding: 0.45rem 0.85rem; border-radius: 10px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    color: var(--text-primary); font-size: 0.82rem; font-family: 'Space Grotesk', sans-serif;
    outline: none; cursor: pointer; min-width: 200px;
}
.dt-split-select:focus { border-color: var(--orange); }
.dt-split-pills {
    display: flex; gap: 0.35rem; flex-wrap: wrap;
}
.dt-split-pill {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    color: var(--text-secondary); padding: 0.3rem 0.7rem; border-radius: 999px;
    font-size: 0.72rem; cursor: pointer; transition: all 0.2s; white-space: nowrap;
    font-family: 'JetBrains Mono', monospace;
}
.dt-split-pill:hover { border-color: rgba(255,180,0,0.4); color: var(--text-primary); }
.dt-split-pill.active { background: rgba(255,180,0,0.15); border-color: rgba(255,180,0,0.5); color: var(--orange); font-weight: 600; }
.dt-search {
    flex: 1; min-width: 160px; padding: 0.45rem 0.8rem; border-radius: 8px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    color: var(--text-primary); font-size: 0.82rem; outline: none;
}
.dt-search:focus { border-color: var(--orange); }
.dt-meta {
    font-size: 0.74rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;
}
.dt-stats {
    display: flex; gap: 1.2rem; margin-bottom: 0.6rem; font-size: 0.72rem;
    color: var(--text-muted); font-family: 'JetBrains Mono', monospace;
}
.dt-stats strong { color: var(--text-secondary); }
.dt-wrap {
    overflow-x: auto; border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
}
.dt-table {
    width: 100%; border-collapse: collapse; min-width: 700px;
    table-layout: fixed;
}
.dt-table thead th {
    position: sticky; top: 0;
    background: rgba(20,22,27,0.95); backdrop-filter: blur(6px);
    padding: 0.55rem 0.75rem; text-align: left; font-weight: 600;
    font-size: 0.72rem; color: var(--text-primary); cursor: pointer;
    user-select: none; border-bottom: 2px solid rgba(255,255,255,0.08);
    white-space: nowrap; transition: color 0.15s;
    position: relative; overflow: visible;
}
.dt-table thead th .dt-resize {
    position: absolute; right: -3px; top: 0; bottom: 0; width: 6px;
    cursor: col-resize; z-index: 2;
}
.dt-table thead th .dt-resize::after {
    content: ''; position: absolute; right: 2px; top: 25%; height: 50%;
    width: 2px; border-radius: 1px; background: rgba(255,255,255,0.08);
    transition: background 0.15s;
}
.dt-table thead th .dt-resize:hover::after,
.dt-table thead th .dt-resize.active::after {
    background: var(--orange);
}
body.dt-resizing, body.dt-resizing * { cursor: col-resize !important; user-select: none !important; }
.dt-table thead th:hover { color: var(--orange); }
.dt-table thead th .col-type {
    display: block; font-size: 0.6rem; font-weight: 400;
    color: var(--text-muted); font-family: 'JetBrains Mono', monospace;
    margin-top: 1px;
}
.dt-table thead th .sort-icon { color: var(--orange); margin-left: 0.3rem; font-size: 0.65rem; }
.dt-table tbody td {
    padding: 0.5rem 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.03);
    font-size: 0.78rem; color: var(--text-secondary); vertical-align: top;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.dt-table tbody tr:hover { background: rgba(255,255,255,0.025); }
.dt-table tbody tr:last-child td { border-bottom: none; }
.dt-table .col-row { color: var(--text-muted); font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; width: 55px; min-width: 55px; }
.dt-table .col-chars { color: var(--orange); font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; width: 70px; }
.dt-table .col-kind { width: 110px; }
.dt-table .col-kind span {
    display: inline-block; padding: 0.1rem 0.45rem; border-radius: 999px;
    font-size: 0.65rem; font-weight: 600; font-family: 'JetBrains Mono', monospace;
}
.dt-table .col-kind .kind-full_svg       { background: rgba(255,180,0,0.15); color: var(--orange); border: 1px solid rgba(255,180,0,0.25); }
.dt-table .col-kind .kind-structural     { background: rgba(7,173,248,0.15); color: var(--blue); border: 1px solid rgba(7,173,248,0.25); }
.dt-table .col-kind .kind-prompt_svg     { background: rgba(71,180,117,0.15); color: var(--green); border: 1px solid rgba(71,180,117,0.25); }
.dt-table .col-kind .kind-defs_fragment  { background: rgba(160,124,248,0.15); color: #a07cf8; border: 1px solid rgba(160,124,248,0.25); }
.dt-table .col-tags,
.dt-table .col-size,
.dt-table .col-vb,
.dt-table .col-eos,
.dt-table .col-els {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted);
}
.dt-table .col-tags { width: 60px; text-align: right; }
.dt-table .col-size { width: 90px; }
.dt-table .col-vb { font-size: 0.7rem; width: 130px; }
.dt-table .col-eos { width: 84px; }
.dt-table .col-els { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; width: 60px; text-align: right; }
.dt-table .col-text {
    font-family: 'JetBrains Mono', monospace; font-size: 0.66rem;
    color: var(--text-muted); overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; max-width: 320px; cursor: pointer;
}
.dt-table .col-text:hover { color: var(--text-secondary); white-space: normal; word-break: break-all; }
.dt-pag {
    display: flex; gap: 0.35rem; align-items: center; justify-content: center;
    margin-top: 0.75rem;
}
.dt-pag button {
    padding: 0.25rem 0.55rem; border-radius: 6px; border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04); color: var(--text-secondary); font-size: 0.74rem;
    cursor: pointer; font-family: 'JetBrains Mono', monospace;
}
.dt-pag button:hover { border-color: var(--orange); color: var(--orange); }
.dt-pag button.active { background: var(--orange); color: var(--dark); border-color: var(--orange); font-weight: 700; }
.dt-pag button:disabled { opacity: 0.3; cursor: not-allowed; }
.dt-pag .dt-pag-info { font-size: 0.7rem; color: var(--text-muted); margin: 0 0.4rem; font-family: 'JetBrains Mono', monospace; }
/* ── Tokenizer ─────────────────────────────────────────────────── */
.tok-stat-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.6rem; margin-bottom: 1rem;
}
.tok-stat {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 0.7rem 0.9rem; text-align: center;
}
.tok-stat .tok-val { font-size: 1.3rem; font-weight: 700; color: var(--orange); font-family: 'JetBrains Mono', monospace; }
.tok-stat .tok-lbl { font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
.tok-metric-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.6rem; margin-top: 0.6rem;
}
.tok-metric {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px; padding: 0.65rem 0.75rem;
}
.tok-metric .mkey {
    color: var(--text-muted); font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 0.2rem;
}
.tok-metric .mval {
    color: var(--text-primary); font-size: 0.86rem; font-family: 'JetBrains Mono', monospace;
}
</style>
</head>
<body>
"""

_HTML_SUFFIX = r"""
<!-- ═══ Header ══════════════════════════════════════════════════════ -->
<header class="site-header">
    <div class="header-brand">
        <div class="header-logo">CK</div>
        <div>
            <div class="header-title">Dataset Viewer</div>
            <div class="header-subtitle">C-Kernel-Engine · SVG Data Operations</div>
        </div>
    </div>
    <div class="header-meta" id="headerMeta"></div>
</header>

<!-- ═══ Search strip ════════════════════════════════════════════════ -->
<div class="search-strip">
    <input type="text" class="search-box" id="searchBox" placeholder="🔍  Search assets, tags, families…">
    <select id="filterFamily"><option value="">All families</option></select>
    <select id="filterRole"><option value="">All roles</option></select>
    <span style="margin-left:auto;font-size:0.75rem;color:var(--text-muted);font-family:'JetBrains Mono',monospace" id="resultCount"></span>
</div>

<!-- ═══ Tabs ════════════════════════════════════════════════════════ -->
<nav class="tabs">
    <button class="tab active" data-tab="overview">Overview</button>
    <button class="tab" data-tab="preflight">Preflight</button>
    <button class="tab" data-tab="gallery">SVG Gallery</button>
    <button class="tab" data-tab="text">Text Samples</button>
    <button class="tab" data-tab="tokenizer">Tokenizer</button>
    <button class="tab" data-tab="vocabulary">Vocabulary</button>
    <button class="tab" data-tab="classification">Classification</button>
    <button class="tab" data-tab="browse">Browse</button>
    <button class="tab" data-tab="candidates">Candidates</button>
    <button class="tab" data-tab="quality">Quality</button>
</nav>

<main>
    <div class="panel active" id="panel-overview"></div>
    <div class="panel" id="panel-preflight"></div>
    <div class="panel" id="panel-gallery"></div>
    <div class="panel" id="panel-text"></div>
    <div class="panel" id="panel-tokenizer"></div>
    <div class="panel" id="panel-vocabulary"></div>
    <div class="panel" id="panel-classification"></div>
    <div class="panel" id="panel-browse"></div>
    <div class="panel" id="panel-candidates"></div>
    <div class="panel" id="panel-quality"></div>
</main>

<!-- ═══ Gallery Full-Screen Overlay ══════════════════════════════ -->
<div class="gal-overlay" id="galOverlay">
    <div class="gal-ov-bar">
        <span class="gal-ov-title" id="galOvTitle">—</span>
        <div style="display:flex;gap:0.4rem;align-items:center;">
            <span id="galOvNav" style="font-size:0.72rem;color:var(--text-muted);font-family:'JetBrains Mono',monospace;margin-right:0.5rem;"></span>
            <button onclick="galViewer.zoomOut()">−</button>
            <span id="galOvZoom" style="font-size:0.72rem;color:var(--text-muted);font-family:'JetBrains Mono',monospace;min-width:40px;text-align:center;">100%</span>
            <button onclick="galViewer.zoomIn()">+</button>
            <button onclick="galViewer.resetZoom()">Reset</button>
            <button onclick="galViewer.close()" style="font-size:1.1rem;padding:0.2rem 0.6rem;">✕</button>
        </div>
    </div>
    <div class="gal-ov-body">
        <div class="gal-ov-canvas" id="galOvCanvas">
            <button class="gal-ov-nav gal-ov-prev" onclick="galViewer.prev()">‹</button>
            <img id="galOvImg" src="" alt="">
            <button class="gal-ov-nav gal-ov-next" onclick="galViewer.next()">›</button>
        </div>
        <div class="gal-ov-sidebar" id="galOvSidebar"></div>
    </div>
</div>

<!-- ═══ Modal ═══════════════════════════════════════════════════════ -->
<div class="modal-overlay" id="modalOverlay" onclick="if(event.target===this)closeModal()">
    <div class="modal">
        <div class="modal-header">
            <h3 id="modalTitle">Detail</h3>
            <button class="modal-close" onclick="closeModal()">×</button>
        </div>
        <div class="modal-body" id="modalBody"></div>
    </div>
</div>

<script>
/* ════════════════════════════════════════════════════════════════════
   Dataset Viewer — C-Kernel-Engine (Antsand brand)
   All data pre-embedded by build_svg_dataset_visualizer_v7.py
   ════════════════════════════════════════════════════════════════════ */

const PAL = ['#ffb400','#07adf8','#47b475','#e74c3c','#9b59b6','#1abc9c',
             '#f39c12','#3498db','#e67e22','#2ecc71','#e91e63','#00bcd4',
             '#ff5722','#8bc34a','#673ab7','#ffc107'];

/* ── Helpers ────────────────────────────────────────────────────── */
const fmt = n => n == null ? '—' : typeof n === 'number' ? n.toLocaleString() : String(n);
const pct = (n,t) => t ? (n/t*100).toFixed(1)+'%' : '0%';
const fmtPct = (n, digits) => n == null ? '—' : (n*100).toFixed(digits == null ? 1 : digits) + '%';
const esc = s => { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; };
const trunc = (s,n) => { s=String(s||''); return s.length>n ? s.slice(0,n)+'…' : s; };
const pathName = p => String(p||'').split('/').pop();

function toggleSection(hdr) {
    const body = hdr.nextElementSibling, arrow = hdr.querySelector('.section-arrow');
    if (body.style.display === 'none') { body.style.display = ''; arrow.textContent = '▼'; }
    else { body.style.display = 'none'; arrow.textContent = '▶'; }
}
function closeModal() { document.getElementById('modalOverlay').classList.remove('open'); }
function openModal(title, html) {
    document.getElementById('modalTitle').textContent = title;
    document.getElementById('modalBody').innerHTML = html;
    document.getElementById('modalOverlay').classList.add('open');
}

function distBarsHtml(countsObj, total, baseColor) {
    const entries = Object.entries(countsObj).sort((a,b) => b[1]-a[1]);
    if (!entries.length) return '<div style="color:var(--text-muted);font-size:0.8rem">No data</div>';
    const max = entries[0][1] || 1;
    return entries.map(([name,count],i) => {
        const w = (count/max*100).toFixed(1);
        const c = PAL[i % PAL.length] || baseColor;
        return `<div class="dist-row">
            <span class="dist-label" title="${esc(name)}">${esc(name)}</span>
            <div class="dist-bar-bg"><div class="dist-bar-fill" style="width:${w}%;background:${c}"><span>${pct(count,total)}</span></div></div>
            <span class="dist-count">${fmt(count)}</span>
        </div>`;
    }).join('');
}

function statCardHtml(value, label, note, cls) {
    return `<div class="stat-card"><div class="stat-value ${cls||''}">${value}</div><div class="stat-label">${label}</div>${note ? '<div class="stat-note">'+note+'</div>' : ''}</div>`;
}

function sectionHtml(icon, title, badgeText, badgeCls, bodyId) {
    return `<div class="section-card">
        <div class="section-header" onclick="toggleSection(this)">
            <div class="section-title">${icon} ${title} <span class="section-badge ${badgeCls}">${badgeText}</span></div>
            <span class="section-arrow">▼</span>
        </div>
        <div class="section-body" id="${bodyId}"></div>
    </div>`;
}

function healthRowHtml(status, label, detail) {
    return `<div class="health-row"><span class="health-dot ${status}"></span><span style="flex:1">${label}</span><span style="color:var(--text-muted);font-size:0.76rem">${detail}</span></div>`;
}

function candidateListHtml(paths, limit) {
    if (!paths || !paths.length) return '<div style="color:var(--text-muted);font-size:0.8rem">None</div>';
    let html = '<ul class="candidate-list">';
    paths.slice(0, limit || 15).forEach(p => { html += `<li>${esc(pathName(p))}</li>`; });
    if (paths.length > (limit || 15)) html += `<li style="color:var(--text-muted)">… and ${paths.length-(limit||15)} more</li>`;
    html += '</ul>';
    return html;
}

/* ── Data extraction ───────────────────────────────────────────── */
const raw   = typeof CK_RAW_INVENTORY !== 'undefined' ? CK_RAW_INVENTORY : {};
const norm  = typeof CK_NORMALIZED !== 'undefined' ? CK_NORMALIZED : {};
const cls   = typeof CK_CLASSIFIED !== 'undefined' ? CK_CLASSIFIED : {};
const ws    = typeof CK_WORKSPACE !== 'undefined' ? CK_WORKSPACE : '';
const preflight = typeof CK_PREFLIGHT !== 'undefined' ? CK_PREFLIGHT : { available: false, checklist: [] };

const rawEntries    = Array.isArray(raw.entries) ? raw.entries : [];
const normFailures  = Array.isArray(norm.failures) ? norm.failures : [];
const classEntries  = Array.isArray(cls.entries) ? cls.entries : [];
const tagTotals     = norm.tag_totals || {};
const placeholders  = norm.placeholder_totals || {};
const familyCounts  = cls.family_counts || {};
const roleCounts    = cls.counts || {};
const suggested     = (typeof cls.suggested_splits === 'object' && cls.suggested_splits) ? cls.suggested_splits : {};

const sourceCounts = {};
(raw.sources || []).forEach(s => { sourceCounts[s.name || 'unknown'] = s.matched_files || 0; });

const sizeBands = {};
classEntries.forEach(e => { const b = e.size_band || 'unknown'; sizeBands[b] = (sizeBands[b]||0)+1; });

let state = { searchTerm: '', filterFamily: '', filterRole: '', browsePage: 0, sortCol: null, sortDir: 1 };

function filteredEntries() {
    let rows = classEntries;
    if (state.searchTerm) {
        const q = state.searchTerm.toLowerCase();
        rows = rows.filter(r =>
            (pathName(r.normalized_path)||'').toLowerCase().includes(q) ||
            (r.family||'').toLowerCase().includes(q) ||
            (r.roles||[]).some(role => role.toLowerCase().includes(q))
        );
    }
    if (state.filterFamily) rows = rows.filter(r => r.family === state.filterFamily);
    if (state.filterRole)   rows = rows.filter(r => (r.roles||[]).includes(state.filterRole));
    return rows;
}

/* ── Header meta chips ─────────────────────────────────────────── */
function renderHeader() {
    const el = document.getElementById('headerMeta');
    const chips = [];
    chips.push(`Workspace: <code>${esc(pathName(ws))}</code>`);
    chips.push(`Raw: <code>${esc(pathName(raw.raw_assets_root||''))}</code>`);
    chips.push(`Normalized: <code>${esc(pathName(norm.normalized_root||''))}</code>`);
    el.innerHTML = chips.map(c => `<span class="chip">${c}</span>`).join('');
}

/* ── Tab: Overview ─────────────────────────────────────────────── */
function renderOverview() {
    const el = document.getElementById('panel-overview');
    const imported   = raw.imported_files || 0;
    const normalized = norm.normalized_entries || 0;
    const unique     = norm.unique_normalized_hashes || 0;
    const holdout    = (suggested.holdout_candidates||[]).length;
    const sftSeed    = (suggested.sft_seed_candidates||[]).length;
    const pretrain   = (suggested.pretrain_structural||[]).length;
    const dupes      = norm.duplicate_normalized_entries || 0;

    let html = '<div class="stats-grid">';
    html += statCardHtml(fmt(imported), 'Raw Imports', 'source SVG assets collected');
    html += statCardHtml(fmt(normalized), 'Normalized', 'usable SVGs after cleanup', 'green');
    html += statCardHtml(fmt(unique), 'Unique', 'dedupe identity count', 'blue');
    html += statCardHtml(fmt(holdout), 'Holdout', 'reserved for canary/eval');
    html += statCardHtml(fmt(sftSeed), 'SFT Seed', 'prompt→SVG supervision targets');
    html += statCardHtml(fmt(pretrain), 'Pretrain', 'structural grammar sources');
    html += '</div>';

    // Health checks
    const checks = [];
    checks.push({ ok: imported > 0, label: 'Raw assets imported', detail: `${imported} files` });
    checks.push({ ok: normalized > 0, label: 'Normalization succeeded', detail: `${normalized} output` });
    checks.push({ ok: dupes === 0, label: 'No duplicate normalized hashes', detail: dupes ? `${dupes} dupes` : 'All unique' });
    checks.push({ ok: normFailures.length === 0, warn: true, label: 'No normalization failures', detail: normFailures.length ? `${normFailures.length} failures` : 'Clean' });
    checks.push({ ok: holdout > 0, label: 'Holdout candidates identified', detail: `${holdout} assets` });
    checks.push({ ok: sftSeed > 0, label: 'SFT seed candidates identified', detail: `${sftSeed} assets` });

    const passCount = checks.filter(c => c.ok).length;
    html += sectionHtml('🏥', 'Dataset Health', `${passCount}/${checks.length} pass`, passCount===checks.length?'badge-green':'badge-red', 'healthBody');

    // Source mix
    html += sectionHtml('📦', 'Imported Asset Sources', fmt(Object.keys(sourceCounts).length)+' sources', 'badge-orange', 'sourceBody');

    // Size bands
    html += sectionHtml('📐', 'Size Distribution', '', 'badge-blue', 'sizeBody');

    el.innerHTML = html;

    // Fill health
    document.getElementById('healthBody').innerHTML = checks.map(c => {
        const dot = c.ok ? 'ok' : (c.warn ? 'warn' : 'err');
        return `<div class="health-row"><span class="health-dot ${dot}"></span><span style="flex:1">${c.label}</span><span style="color:var(--text-muted);font-size:0.76rem">${c.detail}</span></div>`;
    }).join('');

    // Source
    const srcTotal = Object.values(sourceCounts).reduce((s,v) => s+v, 0);
    document.getElementById('sourceBody').innerHTML =
        '<div class="grid-2"><div>' +
        '<div class="subhead">Source counts</div>' +
        '<div class="subnote">How many source SVG files were pulled from each asset collection.</div>' +
        distBarsHtml(sourceCounts, srcTotal, '#47b475') +
        '</div><div>' +
        '<div class="subhead">Size bands</div>' +
        '<div class="subnote">Normalized SVG character-count bands — proxy for tokenizer/context fit.</div>' +
        distBarsHtml(sizeBands, classEntries.length, '#07adf8') +
        '</div></div>';

    document.getElementById('sizeBody').style.display = 'none';
    document.getElementById('sizeBody').parentElement.style.display = 'none';
}

/* ── Tab: Preflight ─────────────────────────────────────────────── */
function renderPreflight() {
    const el = document.getElementById('panel-preflight');
    if (!preflight.available) {
        el.innerHTML = '<div class="subnote" style="padding:2rem;text-align:center;">No preflight data available for this workspace.</div>';
        return;
    }

    const coverage = preflight.exact_control_coverage || { expected: 0, covered: 0, missing: [] };
    const roundtrip = preflight.roundtrip || {};
    const fit = preflight.fit_audit || {};
    const recommendation = preflight.recommendation || {};

    let html = '<div class="subhead">Preflight Checklist</div>';
    html += '<div class="subnote">Live dataset-side gate before pretraining. This is the source of truth for corpus readiness, tokenizer quality, fit risk, and whether the DSL control interface is actually represented as stable atoms.</div>';

    html += '<div class="stats-grid">';
    html += statCardHtml(`${fmt(preflight.ready_count || 0)}/${fmt(preflight.total_count || 0)}`, 'Checklist Ready', 'dataset-side steps complete', (preflight.ready_count === preflight.total_count) ? 'green' : 'blue');
    html += statCardHtml(fmt(coverage.covered || 0), 'Exact DSL Atoms', `${fmt(coverage.expected || 0)} expected control tags`, (coverage.covered || 0) > 0 ? 'blue' : 'red');
    html += statCardHtml(roundtrip.status === 'pass' ? 'PASS' : '—', 'Roundtrip', roundtrip.exact_match ? 'exact encode/decode' : 'tokenizer artifacts missing', roundtrip.status === 'pass' ? 'green' : '');
    html += statCardHtml((preflight.cross_row_merge_hits || []).length ? fmt((preflight.cross_row_merge_hits || []).length) : '0', 'Cross-Row Merges', 'bad vocab pieces should be 0', (preflight.cross_row_merge_hits || []).length ? 'red' : 'green');
    html += statCardHtml(preflight.canary_prompt_files_present ? 'READY' : 'PENDING', 'Canary', `${fmt(preflight.holdout_count || 0)} holdout assets`, preflight.canary_prompt_files_present ? 'green' : 'red');
    html += '</div>';

    html += sectionHtml('🧭', 'Repeatable Pretraining Gate', `${fmt(preflight.ready_count || 0)}/${fmt(preflight.total_count || 0)}`, (preflight.ready_count === preflight.total_count) ? 'badge-green' : 'badge-orange', 'preflightChecklistBody');
    html += sectionHtml('🧬', 'DSL Atom Coverage', `${fmt(coverage.covered || 0)}/${fmt(coverage.expected || 0)}`, (coverage.covered || 0) === (coverage.expected || 0) ? 'badge-green' : 'badge-red', 'preflightControlBody');
    html += sectionHtml('📏', 'Fit Audit (512 vs 2048)', 'context fit', 'badge-blue', 'preflightFitBody');
    html += sectionHtml('🧪', 'Canary Readiness', preflight.canary_prompt_files_present ? 'ready' : 'pending', preflight.canary_prompt_files_present ? 'badge-green' : 'badge-orange', 'preflightCanaryBody');

    el.innerHTML = html;

    document.getElementById('preflightChecklistBody').innerHTML =
        '<div class="subnote">This is the repeatable dataset checklist: contract, import, normalization, structural cleanup, tokenizer quality, fit audit, and canary readiness. Reuse this exact sequence for SVG, C, SQL, Bash, and similar specialist datasets.</div>' +
        (preflight.checklist || []).map(item => {
            const tone = item.status === 'ok' ? 'ok' : (item.status === 'warn' ? 'warn' : 'err');
            return healthRowHtml(tone, item.label, item.detail);
        }).join('');

    const ratio = coverage.expected ? coverage.covered / coverage.expected : 0;
    let controlHtml = '<div class="subnote">This prompt interface is a DSL. The important question is not whether the tokenizer is reversible; it is whether the control tags themselves exist as stable atoms. Right now they do not.</div>';
    controlHtml += '<div class="tok-metric-grid">';
    [
        ['Exact control tags', `${fmt(coverage.covered || 0)} / ${fmt(coverage.expected || 0)}`],
        ['Coverage ratio', fmtPct(ratio, 1)],
        ['Recommendation', recommendation.reserved_control_tokens_recommended ? 'reserve/protect tags' : 'frequency weighting may suffice'],
        ['Run-local artifacts', preflight.run_dir ? 'present' : 'missing'],
    ].forEach(([k, v]) => {
        controlHtml += `<div class="tok-metric"><div class="mkey">${k}</div><div class="mval">${esc(String(v))}</div></div>`;
    });
    controlHtml += '</div>';
    if ((coverage.missing || []).length) {
        controlHtml += '<div class="subhead" style="margin-top:0.9rem">Missing exact DSL atoms</div><div>';
        controlHtml += coverage.missing.map(tag => `<span class="token-pill">${esc(tag)}</span>`).join('');
        controlHtml += '</div>';
    }
    controlHtml += `<div class="source-block" style="margin-top:0.8rem">${esc(recommendation.reason || '')}</div>`;
    document.getElementById('preflightControlBody').innerHTML = controlHtml;

    const fitRows = [
        ['small_full', fit.small_full_512, fit.small_full_2048],
        ['structural', fit.structural_512, fit.structural_2048],
    ];
    let fitHtml = '<div class="subnote">Use this to decide which artifacts belong in the 512 mainline and which should move to a 2048 branch. Small full SVGs are the first thing that benefit from larger context; structural chunks are already mostly safe at 512.</div>';
    fitHtml += '<table><thead><tr><th>Split</th><th>512 Fit</th><th>2048 Fit</th><th>P95 Tokens</th><th>Max Tokens</th></tr></thead><tbody>';
    fitRows.forEach(([name, a512, a2048]) => {
        fitHtml += `<tr>
            <td class="mono">${esc(name)}</td>
            <td class="mono">${a512 ? `${fmtPct(a512.fit_rate || 0, 1)} · ${fmt(a512.rows_kept || 0)}/${fmt(a512.rows_total || 0)}` : '—'}</td>
            <td class="mono">${a2048 ? `${fmtPct(a2048.fit_rate || 0, 1)} · ${fmt(a2048.rows_kept || 0)}/${fmt(a2048.rows_total || 0)}` : '—'}</td>
            <td class="mono">${a2048 ? fmt(a2048.row_tokens_p95 || 0) : (a512 ? fmt(a512.row_tokens_p95 || 0) : '—')}</td>
            <td class="mono">${a2048 ? fmt(a2048.row_tokens_max || 0) : (a512 ? fmt(a512.row_tokens_max || 0) : '—')}</td>
        </tr>`;
    });
    fitHtml += '</tbody></table>';
    document.getElementById('preflightFitBody').innerHTML = fitHtml;

    let canaryHtml = '<div class="subnote">Holdout assets exist, but prompt-level canaries are not frozen yet. Do not treat holdout asset presence as equivalent to an eval contract. Freeze a small prompt set before the first overnight run.</div>';
    canaryHtml += healthRowHtml(preflight.holdout_count > 0 ? 'ok' : 'err', 'Holdout assets reserved', `${fmt(preflight.holdout_count || 0)} assets`);
    canaryHtml += healthRowHtml(preflight.canary_prompt_files_present ? 'ok' : 'warn', 'Prompt-level canary file', preflight.canary_prompt_files_present ? 'present' : 'missing');
    document.getElementById('preflightCanaryBody').innerHTML = canaryHtml;
}

/* ── Tab: Vocabulary ───────────────────────────────────────────── */
function renderVocabulary() {
    const el = document.getElementById('panel-vocabulary');
    const tagTotal = Object.values(tagTotals).reduce((s,v) => s+v, 0);
    const phTotal  = Object.values(placeholders).reduce((s,v) => s+v, 0);

    let html = '<div class="stats-grid">';
    html += statCardHtml(fmt(Object.keys(tagTotals).length), 'SVG Tag Types');
    html += statCardHtml(fmt(tagTotal), 'Total Tag Instances', null, 'blue');
    html += statCardHtml(fmt(Object.keys(placeholders).length), 'Placeholder Types');
    html += statCardHtml(fmt(phTotal), 'Placeholder Instances', null, 'green');
    html += '</div>';

    html += sectionHtml('🔤', 'SVG Vocabulary Histogram', fmt(Object.keys(tagTotals).length)+' tags', 'badge-blue', 'vocabBody');
    html += sectionHtml('📝', 'Placeholder Histogram', '', 'badge-green', 'phBody');

    el.innerHTML = html;

    document.getElementById('vocabBody').innerHTML =
        '<div class="subnote">Top tag counts across the normalized corpus — the visual grammar the model learns from.</div>' +
        distBarsHtml(tagTotals, tagTotal, '#07adf8');

    document.getElementById('phBody').innerHTML =
        '<div class="subnote">Human text removed during normalization. Keeps composition/layout signal while reducing English memorization.</div>' +
        distBarsHtml(placeholders, phTotal, '#47b475');
}

/* ── Tab: Classification ───────────────────────────────────────── */
function renderClassification() {
    const el = document.getElementById('panel-classification');
    const total = classEntries.length;

    let html = '<div class="stats-grid">';
    html += statCardHtml(fmt(Object.keys(familyCounts).length), 'Families');
    html += statCardHtml(fmt(Object.keys(roleCounts).length), 'Roles');
    html += statCardHtml(fmt(total), 'Classified Assets', null, 'blue');
    html += '</div>';

    html += sectionHtml('🏷️', 'Family Distribution', '', 'badge-blue', 'famDistBody');
    html += sectionHtml('🎭', 'Role Distribution', '', 'badge-purple', 'roleDistBody');

    // Split candidates summary
    html += sectionHtml('✂️', 'Split Candidates', '', 'badge-orange', 'splitBody');

    el.innerHTML = html;

    document.getElementById('famDistBody').innerHTML =
        '<div class="subnote">Heuristic family labels from normalized filenames and structure.</div>' +
        distBarsHtml(familyCounts, total, '#07adf8');

    document.getElementById('roleDistBody').innerHTML =
        '<div class="subnote">Roles assigned by the classifier. Assets can carry multiple roles.</div>' +
        distBarsHtml(roleCounts, total, '#ffb400');

    const splits = {
        'Pretrain: small full': (suggested.pretrain_small_full||[]).length,
        'Pretrain: structural': (suggested.pretrain_structural||[]).length,
        'Midtrain: transform': (suggested.midtrain_transform_candidates||[]).length,
        'SFT: seed candidates': (suggested.sft_seed_candidates||[]).length,
        'Holdout: canary': (suggested.holdout_candidates||[]).length,
    };
    document.getElementById('splitBody').innerHTML =
        '<div class="subnote">Manifest-driven candidate lists — not final corpora yet.</div>' +
        distBarsHtml(splits, total, '#9b59b6');
}

/* ── Tab: Browse (sortable table with pagination) ──────────────── */
const PAGE_SIZE = 40;

function renderBrowse() {
    const el = document.getElementById('panel-browse');
    let rows = filteredEntries();
    document.getElementById('resultCount').textContent = `${rows.length} assets`;

    if (state.sortCol) {
        rows = [...rows].sort((a,b) => {
            let va = a[state.sortCol], vb = b[state.sortCol];
            if (va == null) va = '';
            if (vb == null) vb = '';
            if (typeof va === 'number' && typeof vb === 'number') return (va-vb)*state.sortDir;
            return String(va).localeCompare(String(vb))*state.sortDir;
        });
    }

    const start = state.browsePage * PAGE_SIZE;
    const page = rows.slice(start, start + PAGE_SIZE);

    if (!page.length) {
        el.innerHTML = '<div style="text-align:center;padding:3rem;color:var(--text-muted)">No assets match the current filters.</div>';
        return;
    }

    const cols = [
        { key: 'idx',    label: '#',      render: (r,i) => `<span class="mono">${start+i}</span>` },
        { key: 'normalized_path', label: 'Asset', render: r => `<span class="mono" style="color:var(--blue)">${esc(trunc(pathName(r.normalized_path),40))}</span>` },
        { key: 'family', label: 'Family', render: r => `<span style="color:var(--purple)">${esc(r.family||'—')}</span>` },
        { key: 'chars',  label: 'Chars',  render: r => `<span class="mono">${fmt(r.chars||0)}</span>` },
        { key: 'size_band', label: 'Size Band', render: r => `<span class="mono">${esc(r.size_band||'—')}</span>` },
        { key: 'roles',  label: 'Roles',  render: r => (r.roles||[]).map(role => `<span class="mono" style="color:var(--orange);font-size:0.68rem">${esc(role)}</span>`).join(', ') || '—' },
    ];

    let html = '<table><thead><tr>';
    cols.forEach(col => {
        const active = state.sortCol === col.key;
        html += `<th class="${active?'sort-active':''}" onclick="sortBrowse('${col.key}')">${col.label}${active ? (state.sortDir>0?' ↑':' ↓') : ''}</th>`;
    });
    html += '</tr></thead><tbody>';
    page.forEach((row, i) => {
        html += `<tr style="cursor:pointer" onclick="showAssetDetail(${start+i})">`;
        cols.forEach(col => { html += '<td>' + col.render(row, i) + '</td>'; });
        html += '</tr>';
    });
    html += '</tbody></table>';
    html += '<div class="pagination" id="browsePag"></div>';
    el.innerHTML = html;

    renderPagination('browsePag', rows.length, PAGE_SIZE, state.browsePage, p => { state.browsePage = p; renderBrowse(); });
}

window.sortBrowse = function(key) {
    if (state.sortCol === key) state.sortDir *= -1;
    else { state.sortCol = key; state.sortDir = 1; }
    renderBrowse();
};

window.showAssetDetail = function(idx) {
    const row = classEntries[idx];
    if (!row) return;
    let html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem 1rem;margin-bottom:1rem">';
    const fields = { 'Family': row.family, 'Size Band': row.size_band, 'Chars': fmt(row.chars), 'Roles': (row.roles||[]).join(', ') };
    for (const [k,v] of Object.entries(fields)) {
        html += `<span class="mono" style="color:var(--text-muted)">${k}</span><span class="mono">${esc(String(v||''))}</span>`;
    }
    html += '</div>';
    html += '<div class="subhead">Normalized Path</div>';
    html += `<div class="source-block">${esc(row.normalized_path||'')}</div>`;
    if (row.raw_path) {
        html += '<div class="subhead" style="margin-top:0.75rem">Raw Path</div>';
        html += `<div class="source-block">${esc(row.raw_path)}</div>`;
    }
    openModal(pathName(row.normalized_path), html);
};

function renderPagination(containerId, total, pageSize, current, onPage) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const totalPages = Math.ceil(total / pageSize);
    if (totalPages <= 1) { container.innerHTML = ''; return; }
    let html = `<button class="page-btn" ${current===0?'disabled':''} data-p="${current-1}">◀</button>`;
    const maxV = 7; let s = Math.max(0, current - Math.floor(maxV/2));
    let e = Math.min(totalPages, s + maxV);
    if (e - s < maxV) s = Math.max(0, e - maxV);
    if (s > 0) html += `<button class="page-btn" data-p="0">1</button><span class="page-info">…</span>`;
    for (let i = s; i < e; i++) html += `<button class="page-btn ${i===current?'active':''}" data-p="${i}">${i+1}</button>`;
    if (e < totalPages) html += `<span class="page-info">…</span><button class="page-btn" data-p="${totalPages-1}">${totalPages}</button>`;
    html += `<button class="page-btn" ${current>=totalPages-1?'disabled':''} data-p="${current+1}">▶</button>`;
    html += `<span class="page-info">${fmt(total)} assets</span>`;
    container.innerHTML = html;
    container.querySelectorAll('.page-btn[data-p]').forEach(btn => {
        btn.addEventListener('click', () => { const p = parseInt(btn.dataset.p); if (p >= 0 && p < totalPages) onPage(p); });
    });
}

/* ── Tab: Candidates ───────────────────────────────────────────── */
function renderCandidates() {
    const el = document.getElementById('panel-candidates');
    let html = '<div class="grid-3" style="margin-bottom:1rem">';

    html += '<div class="section-card"><div class="section-header" onclick="toggleSection(this)"><div class="section-title">🎯 Holdout Candidates <span class="section-badge badge-red">' + (suggested.holdout_candidates||[]).length + '</span></div><span class="section-arrow">▼</span></div>';
    html += '<div class="section-body"><div class="subnote">Good medium-sized assets to reserve for canary and stage evaluation.</div>' + candidateListHtml(suggested.holdout_candidates) + '</div></div>';

    html += '<div class="section-card"><div class="section-header" onclick="toggleSection(this)"><div class="section-title">🌱 SFT Seed Candidates <span class="section-badge badge-green">' + (suggested.sft_seed_candidates||[]).length + '</span></div><span class="section-arrow">▼</span></div>';
    html += '<div class="section-body"><div class="subnote">Manageable, structurally rich assets for the first strict tag→SVG pairs.</div>' + candidateListHtml(suggested.sft_seed_candidates) + '</div></div>';

    html += '<div class="section-card"><div class="section-header" onclick="toggleSection(this)"><div class="section-title">🏗️ Structural Pretrain <span class="section-badge badge-blue">' + (suggested.pretrain_structural||[]).length + '</span></div><span class="section-arrow">▼</span></div>';
    html += '<div class="section-body"><div class="subnote">Assets likely to contribute gradients, groups, markers, filters, and panel composition.</div>' + candidateListHtml(suggested.pretrain_structural) + '</div></div>';

    html += '</div>';

    // Largest assets table
    const biggest = [...classEntries].sort((a,b) => (b.chars||0)-(a.chars||0)).slice(0, 12);
    html += sectionHtml('📏', 'Largest Normalized Assets', 'top '+biggest.length, 'badge-orange', 'bigBody');
    el.innerHTML = html;

    let tbl = '<div class="subnote">Likely candidates for structural chunking rather than full-row training at short context lengths.</div>';
    tbl += '<table><thead><tr><th>Asset</th><th>Family</th><th>Chars</th><th>Roles</th></tr></thead><tbody>';
    biggest.forEach(r => {
        tbl += `<tr><td class="mono" style="color:var(--blue)">${esc(trunc(pathName(r.normalized_path),40))}</td><td>${esc(r.family||'')}</td><td class="mono">${fmt(r.chars||0)}</td><td class="mono" style="font-size:0.7rem">${esc((r.roles||[]).join(', '))}</td></tr>`;
    });
    tbl += '</tbody></table>';
    document.getElementById('bigBody').innerHTML = tbl;
}

/* ── Tab: Quality ──────────────────────────────────────────────── */
function renderQuality() {
    const el = document.getElementById('panel-quality');
    const dupes = norm.duplicate_normalized_entries || 0;
    const failCount = normFailures.length;
    const tok = CK_TOKENIZER || {};
    const prompt = tok.prompt_contract || {};
    const drift = tok.manifest_drift || { count: 0, entries: [] };

    let html = '<div class="stats-grid">';
    html += statCardHtml(fmt(dupes), 'Duplicates', null, dupes === 0 ? 'green' : 'red');
    html += statCardHtml(fmt(failCount), 'Parse Failures', null, failCount === 0 ? 'green' : 'red');
    html += statCardHtml(fmt(norm.normalized_entries||0), 'Normalized OK', null, 'green');
    html += statCardHtml(fmt(norm.input_non_ascii_chars_total||0), 'Non-ASCII Input', null, (norm.input_non_ascii_chars_total||0) === 0 ? 'green' : '');
    html += statCardHtml(fmt(norm.output_non_ascii_chars_total||0), 'Non-ASCII Output', null, (norm.output_non_ascii_chars_total||0) === 0 ? 'green' : 'red');
    html += statCardHtml(fmt(norm.mapped_common_symbols_total||0), 'Mapped Symbols');
    html += '</div>';

    // Checks
    const checks = [];
    checks.push({ ok: dupes === 0, label: 'No duplicate hashes after normalization', detail: dupes ? `${dupes} duplicates` : 'All unique' });
    checks.push({ ok: failCount === 0, label: 'No normalization parse failures', detail: failCount ? `${failCount} failures` : 'Clean parse' });
    checks.push({ ok: (norm.output_non_ascii_chars_total||0) === 0, label: 'Output is ASCII-only', detail: `${norm.output_non_ascii_chars_total||0} non-ASCII chars in output` });
    checks.push({ ok: (raw.duplicate_files||0) === 0, label: 'No duplicate raw imports', detail: `${raw.duplicate_files||0} raw dupes` });

    html += sectionHtml('✅', 'Quality Checks', `${checks.filter(c=>c.ok).length}/${checks.length}`, checks.every(c=>c.ok)?'badge-green':'badge-red', 'qcBody');
    html += sectionHtml('🧠', 'Representation Checks', tok.available ? `${prompt.prompt_rows||0} prompt rows` : 'n/a', tok.available ? ((drift.count||0) ? 'badge-red' : 'badge-blue') : 'badge-purple', 'reprBody');
    html += sectionHtml('⚠️', 'Parse Failures', fmt(failCount), failCount?'badge-red':'badge-green', 'failBody');

    el.innerHTML = html;

    document.getElementById('qcBody').innerHTML = checks.map(c => healthRowHtml(c.ok ? 'ok' : 'err', c.label, c.detail)).join('');

    if (!tok.available) {
        document.getElementById('reprBody').innerHTML = '<div class="subnote">Tokenizer artifacts were not staged into this workspace.</div>';
    } else {
        const eosCounts = prompt.eos_counts || {};
        const reprChecks = [];
        reprChecks.push({ status: (prompt.prompt_rows||0) > 0 ? 'ok' : 'err', label: 'Prompt rows present in tokenizer corpus', detail: `${fmt(prompt.prompt_rows||0)} rows · ${fmtPct(prompt.prompt_char_share||0, 2)} byte share` });
        reprChecks.push({ status: prompt.all_seed_rows_in_corpus ? 'ok' : 'err', label: 'All tag-seed rows are in the tokenizer corpus', detail: prompt.all_seed_rows_in_corpus ? 'Seed rows covered' : `${(prompt.missing_seed_rows||[]).length} seed rows missing` });
        reprChecks.push({ status: (prompt.prompt_char_share||0) >= 0.10 ? 'ok' : ((prompt.prompt_char_share||0) > 0 ? 'warn' : 'err'), label: 'Prompt bytes reach the recommended floor', detail: `target >= 10% · actual ${fmtPct(prompt.prompt_char_share||0, 2)}` });
        reprChecks.push({ status: (prompt.eos_variant_count||0) <= 1 ? 'ok' : 'warn', label: 'EOS convention is consistent', detail: Object.entries(eosCounts).map(([k,v]) => `${k}:${v}`).join(' · ') || 'no EOS markers' });
        reprChecks.push({ status: (drift.count||0) === 0 ? 'ok' : 'warn', label: 'Manifest counts match staged files', detail: (drift.count||0) === 0 ? 'No drift detected' : `${drift.count} drift fields` });

        let reprHtml = '<div class="subnote">Representation-oriented checks for prompt atoms, corpus weighting, and staging consistency.</div>';
        reprHtml += reprChecks.map(c => healthRowHtml(c.status, c.label, c.detail)).join('');
        if ((prompt.missing_seed_rows||[]).length) {
            reprHtml += '<div class="subhead" style="margin-top:0.9rem">Missing seed rows</div>';
            reprHtml += '<div class="source-block">' + esc(prompt.missing_seed_rows.join('\n')) + '</div>';
        }
        document.getElementById('reprBody').innerHTML = reprHtml;
    }

    if (failCount) {
        let ftbl = '<div class="subnote">Parser/cleanup edge cases. Drive this list toward zero or quarantine the bad assets.</div>';
        ftbl += '<table><thead><tr><th>Asset</th><th>Error</th></tr></thead><tbody>';
        normFailures.forEach(f => {
            ftbl += `<tr><td class="mono" style="color:var(--blue)">${esc(pathName(f.imported_path||''))}</td><td style="color:var(--red);font-size:0.78rem">${esc(f.error||'')}</td></tr>`;
        });
        ftbl += '</tbody></table>';
        document.getElementById('failBody').innerHTML = ftbl;
    } else {
        document.getElementById('failBody').innerHTML = '<div style="color:var(--green);font-size:0.82rem">✓ No normalization failures.</div>';
    }
}

/* ── Filter dropdowns ──────────────────────────────────────────── */
function populateFilters() {
    const families = new Set(), roles = new Set();
    classEntries.forEach(e => {
        if (e.family) families.add(e.family);
        (e.roles||[]).forEach(r => roles.add(r));
    });
    const famEl = document.getElementById('filterFamily');
    const roleEl = document.getElementById('filterRole');
    famEl.innerHTML = '<option value="">All families</option>' + [...families].sort().map(f => `<option value="${f}">${f}</option>`).join('');
    roleEl.innerHTML = '<option value="">All roles</option>' + [...roles].sort().map(r => `<option value="${r}">${r}</option>`).join('');
}

/* ── Wiring ────────────────────────────────────────────────────── */

// ── Gallery state ─────────────────────────────────────────────────
const galState = { filtered: [...CK_GALLERY], family: null, search: '' };

function galApplyFilters() {
    const q = galState.search.toLowerCase();
    galState.filtered = CK_GALLERY.filter(item => {
        if (galState.family && item.family !== galState.family) return false;
        if (q) {
            const hay = (item.name + ' ' + item.family + ' ' + item.source + ' ' + item.features.join(' ')).toLowerCase();
            if (!hay.includes(q)) return false;
        }
        return true;
    });
    renderGalleryGrid();
}

function renderGallery() {
    if (!CK_GALLERY.length) {
        document.getElementById('panel-gallery').innerHTML = '<div class="subnote" style="padding:2rem;text-align:center;">No SVG assets found in this dataset.</div>';
        return;
    }
    const families = {};
    CK_GALLERY.forEach(g => { families[g.family] = (families[g.family] || 0) + 1; });

    let html = '<div class="gal-toolbar">';
    html += '<input type="text" class="gal-search" id="galSearch" placeholder="Search SVGs by name, family…">';
    html += '<button class="gal-pill active" data-galfam="">All (' + CK_GALLERY.length + ')</button>';
    Object.entries(families).sort((a,b) => b[1]-a[1]).forEach(([fam, cnt]) => {
        html += `<button class="gal-pill" data-galfam="${esc(fam)}">${esc(fam)} (${cnt})</button>`;
    });
    html += '</div>';
    html += '<div class="gal-grid" id="galGrid"></div>';
    document.getElementById('panel-gallery').innerHTML = html;

    // Wire events
    document.getElementById('galSearch').addEventListener('input', e => {
        galState.search = e.target.value.trim();
        galApplyFilters();
    });
    document.querySelectorAll('[data-galfam]').forEach(btn => {
        btn.addEventListener('click', () => {
            const val = btn.dataset.galfam || null;
            galState.family = val;
            document.querySelectorAll('[data-galfam]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            galApplyFilters();
        });
    });
    renderGalleryGrid();
}

function renderGalleryGrid() {
    const grid = document.getElementById('galGrid');
    if (!grid) return;
    if (!galState.filtered.length) {
        grid.innerHTML = '<div class="subnote" style="padding:2rem;text-align:center;">No SVGs match your filter.</div>';
        return;
    }
    grid.innerHTML = galState.filtered.map((item, i) => `
        <div class="gal-card" onclick="galViewer.open(${i})" title="${esc(item.name)}">
            <div class="gal-thumb">
                <img src="${item.data_uri}" alt="${esc(item.name)}" loading="lazy">
                <span class="gal-badge fam-${item.family}">${item.family}</span>
            </div>
            <div class="gal-info">
                <div class="gal-name">${esc(item.name)}</div>
                <div class="gal-sub">${item.source} · ${item.size_band} · ${fmt(item.chars)} chars</div>
            </div>
        </div>
    `).join('');
}

// ── Gallery full-screen viewer ────────────────────────────────────
const galViewer = {
    idx: 0, zoom: 1, panX: 0, panY: 0, dragging: false,
    dragSX: 0, dragSY: 0, panSX: 0, panSY: 0,
    open(i) {
        this.idx = i; this.zoom = 1; this.panX = 0; this.panY = 0;
        this.show();
        document.getElementById('galOverlay').classList.add('open');
    },
    close() { document.getElementById('galOverlay').classList.remove('open'); },
    show() {
        const item = galState.filtered[this.idx];
        if (!item) return;
        document.getElementById('galOvTitle').textContent = item.name;
        document.getElementById('galOvImg').src = item.data_uri;
        document.getElementById('galOvNav').textContent = `${this.idx+1} / ${galState.filtered.length}`;
        this.updateTransform();
        this.renderSidebar(item);
    },
    prev() { if (this.idx > 0) { this.idx--; this.zoom=1; this.panX=0; this.panY=0; this.show(); } },
    next() { if (this.idx < galState.filtered.length-1) { this.idx++; this.zoom=1; this.panX=0; this.panY=0; this.show(); } },
    zoomIn()  { this.zoom = Math.min(8, this.zoom*1.3); this.updateTransform(); },
    zoomOut() { this.zoom = Math.max(0.2, this.zoom/1.3); this.updateTransform(); },
    resetZoom() { this.zoom=1; this.panX=0; this.panY=0; this.updateTransform(); },
    updateTransform() {
        document.getElementById('galOvImg').style.transform = `translate(${this.panX}px,${this.panY}px) scale(${this.zoom})`;
        document.getElementById('galOvZoom').textContent = Math.round(this.zoom*100)+'%';
    },
    renderSidebar(item) {
        let h = '<h4>Classification</h4>';
        [['Family',item.family],['Source',item.source],['Size Band',item.size_band],['Characters',fmt(item.chars)],['SHA-256',item.sha]].forEach(([k,v]) => {
            h += `<div class="gal-ov-meta-row"><span class="mkey">${k}</span><span class="mval">${esc(String(v))}</span></div>`;
        });
        if (item.features.length || item.roles.length) {
            h += '<h4>Features & Roles</h4><div>';
            [...item.features, ...item.roles].forEach(f => { h += `<span class="gal-ov-pill">${esc(f)}</span>`; });
            h += '</div>';
        }
        const tags = Object.entries(item.tags||{}).filter(([,v])=>v>0).sort((a,b)=>b[1]-a[1]);
        if (tags.length) {
            h += '<h4>SVG Tags</h4>';
            tags.forEach(([name,cnt]) => { h += `<div class="gal-ov-meta-row"><span class="mkey">&lt;${esc(name)}&gt;</span><span class="mval">${cnt}</span></div>`; });
        }
        document.getElementById('galOvSidebar').innerHTML = h;
    }
};

// Gallery keyboard & mouse
document.addEventListener('keydown', e => {
    const ov = document.getElementById('galOverlay');
    if (!ov.classList.contains('open')) return;
    switch(e.key) {
        case 'Escape': galViewer.close(); e.stopPropagation(); break;
        case 'ArrowLeft': galViewer.prev(); break;
        case 'ArrowRight': galViewer.next(); break;
        case '+': case '=': galViewer.zoomIn(); break;
        case '-': galViewer.zoomOut(); break;
        case '0': galViewer.resetZoom(); break;
    }
});
const galCanvas = document.getElementById('galOvCanvas');
if (galCanvas) {
    galCanvas.addEventListener('wheel', e => { e.preventDefault(); galViewer.zoom = Math.max(0.2, Math.min(8, galViewer.zoom+(e.deltaY>0?-0.15:0.15))); galViewer.updateTransform(); });
    galCanvas.addEventListener('mousedown', e => { if(e.target.tagName==='BUTTON')return; galViewer.dragging=true; galViewer.dragSX=e.clientX; galViewer.dragSY=e.clientY; galViewer.panSX=galViewer.panX; galViewer.panSY=galViewer.panY; });
    document.addEventListener('mousemove', e => { if(!galViewer.dragging)return; galViewer.panX=galViewer.panSX+(e.clientX-galViewer.dragSX); galViewer.panY=galViewer.panSY+(e.clientY-galViewer.dragSY); galViewer.updateTransform(); });
    document.addEventListener('mouseup', () => { galViewer.dragging=false; });
}

// ── Text Samples tab ──────────────────────────────────────────────
// ── Data Table (HuggingFace-style) for Text & Tokenizer ──────────
const dtState = { splitKey: null, page: 0, search: '', sortCol: null, sortAsc: true };
const DT_PAGE_SIZE = 30;

function dtGetSplits() { return CK_TEXT_ROWS; }

function renderTextSamples() {
    const panel = document.getElementById('panel-text');
    const splits = dtGetSplits();
    const keys = Object.keys(splits).sort();
    if (!keys.length) {
        panel.innerHTML = '<div class="subnote" style="padding:2rem;text-align:center;">No training text data found.</div>';
        return;
    }
    if (!dtState.splitKey || !splits[dtState.splitKey]) dtState.splitKey = keys[0];

    // Build header: split pills + search + row count
    let html = '<div class="dt-header">';
    html += '<div class="dt-split-pills">';
    keys.forEach(k => {
        const s = splits[k];
        const active = k === dtState.splitKey ? 'active' : '';
        html += `<button class="dt-split-pill ${active}" data-dtkey="${esc(k)}">${esc(s.split)} <span style="opacity:0.6">${fmt(s.total_rows)}</span></button>`;
    });
    html += '</div>';
    html += '<input type="text" class="dt-search" id="dtSearch" placeholder="Search rows…" value="' + esc(dtState.search) + '">';
    html += '<div class="dt-meta" id="dtMeta"></div>';
    html += '</div>';

    // Stats bar
    const cur = splits[dtState.splitKey];
    html += '<div class="dt-stats">';
    html += `<span><strong>${esc(cur.split)}</strong> · ${esc(cur.file)}</span>`;
    html += `<span>${fmt(cur.total_rows)} rows</span>`;
    html += `<span>${fmt(cur.total_chars)} chars</span>`;
    html += `<span>avg ${fmt(cur.avg_chars)}</span>`;
    html += `<span>range ${fmt(cur.min_chars)}–${fmt(cur.max_chars)}</span>`;
    if (cur.capped) html += `<span style="color:var(--orange)">⚠ showing first 200</span>`;
    html += '</div>';

    // Table
    html += '<div class="dt-wrap"><table class="dt-table"><thead><tr>';
    const cols = [
        { key: 'row', label: 'row', type: 'int', cls: 'col-row' },
        { key: 'chars', label: 'chars', type: 'int', cls: 'col-chars' },
        { key: 'kind', label: 'kind', type: 'string', cls: 'col-kind' },
        { key: 'prompt_tags', label: 'tags', type: 'int', cls: 'col-tags' },
        { key: 'root_size', label: 'size', type: 'string', cls: 'col-size' },
        { key: 'viewBox', label: 'viewBox', type: 'string', cls: 'col-vb' },
        { key: 'elements', label: 'elements', type: 'int', cls: 'col-els' },
        { key: 'eos', label: 'eos', type: 'string', cls: 'col-eos' },
        { key: 'text', label: 'text', type: 'string', cls: 'col-text' },
    ];
    cols.forEach(c => {
        const arrow = dtState.sortCol === c.key ? (dtState.sortAsc ? '▲' : '▼') : '';
        html += `<th data-dtsort="${c.key}">${c.label}${arrow ? `<span class="sort-icon">${arrow}</span>` : ''}<span class="col-type">${c.type}</span><div class="dt-resize" data-dtcol="${c.key}"></div></th>`;
    });
    html += '</tr></thead><tbody id="dtBody"></tbody></table></div>';
    html += '<div class="dt-pag" id="dtPag"></div>';

    panel.innerHTML = html;

    // Wire events
    document.querySelectorAll('[data-dtkey]').forEach(btn => {
        btn.addEventListener('click', () => {
            dtState.splitKey = btn.dataset.dtkey;
            dtState.page = 0; dtState.search = ''; dtState.sortCol = null;
            renderTextSamples();
        });
    });
    document.getElementById('dtSearch').addEventListener('input', e => {
        dtState.search = e.target.value.trim();
        dtState.page = 0;
        dtRenderRows();
    });
    document.querySelectorAll('[data-dtsort]').forEach(th => {
        th.addEventListener('click', (e) => {
            if (e.target.closest('.dt-resize')) return; // ignore resize handle clicks
            const col = th.dataset.dtsort;
            if (dtState.sortCol === col) dtState.sortAsc = !dtState.sortAsc;
            else { dtState.sortCol = col; dtState.sortAsc = true; }
            dtState.page = 0;
            renderTextSamples();
        });
    });
    // Column resize handles
    dtInitResize();
    dtRenderRows();
}

function dtInitResize() {
    document.querySelectorAll('.dt-resize').forEach(handle => {
        handle.addEventListener('mousedown', e => {
            e.preventDefault(); e.stopPropagation();
            const th = handle.parentElement;
            const table = th.closest('.dt-table');
            const startX = e.clientX;
            const startW = th.offsetWidth;
            handle.classList.add('active');
            document.body.classList.add('dt-resizing');
            const onMove = ev => {
                const newW = Math.max(40, startW + ev.clientX - startX);
                th.style.width = newW + 'px';
            };
            const onUp = () => {
                handle.classList.remove('active');
                document.body.classList.remove('dt-resizing');
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
            };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
    });
}

function dtRenderRows() {
    const splits = dtGetSplits();
    const cur = splits[dtState.splitKey];
    if (!cur) return;
    let rows = [...(cur.rows || [])];

    // Search filter
    const q = dtState.search.toLowerCase();
    if (q) {
        rows = rows.filter(r => {
            return String(r.row).includes(q) || String(r.chars).includes(q) ||
                   r.kind.toLowerCase().includes(q) || String(r.prompt_tags).includes(q) ||
                   r.root_size.toLowerCase().includes(q) || r.viewBox.toLowerCase().includes(q) ||
                   r.eos.toLowerCase().includes(q) || r.text.toLowerCase().includes(q);
        });
    }

    // Sort
    if (dtState.sortCol) {
        const k = dtState.sortCol;
        const dir = dtState.sortAsc ? 1 : -1;
        rows.sort((a, b) => {
            const av = a[k], bv = b[k];
            if (typeof av === 'number') return (av - bv) * dir;
            return String(av).localeCompare(String(bv)) * dir;
        });
    }

    // Paginate
    const total = rows.length;
    const pages = Math.ceil(total / DT_PAGE_SIZE);
    if (dtState.page >= pages) dtState.page = Math.max(0, pages - 1);
    const start = dtState.page * DT_PAGE_SIZE;
    const pageRows = rows.slice(start, start + DT_PAGE_SIZE);

    // Render body
    const tbody = document.getElementById('dtBody');
    tbody.innerHTML = pageRows.map(r => {
        const kindCls = 'kind-' + r.kind.replace(/\s+/g, '_');
        return `<tr>
            <td class="col-row">${r.row}</td>
            <td class="col-chars">${fmt(r.chars)}</td>
            <td class="col-kind"><span class="${kindCls}">${esc(r.kind)}</span></td>
            <td class="col-tags">${r.prompt_tags ? fmt(r.prompt_tags) : '—'}</td>
            <td class="col-size">${esc(r.root_size || '—')}</td>
            <td class="col-vb">${esc(r.viewBox)}</td>
            <td class="col-els">${fmt(r.elements)}</td>
            <td class="col-eos">${esc(r.eos || '—')}</td>
            <td class="col-text" title="Click to expand">${esc(r.text)}</td>
        </tr>`;
    }).join('');

    // Meta
    document.getElementById('dtMeta').textContent = `${fmt(total)} rows`;

    // Pagination
    const pag = document.getElementById('dtPag');
    if (pages <= 1) { pag.innerHTML = ''; return; }
    let ph = `<button ${dtState.page===0?'disabled':''} onclick="dtState.page=0;dtRenderRows()">«</button>`;
    ph += `<button ${dtState.page===0?'disabled':''} onclick="dtState.page--;dtRenderRows()">‹</button>`;
    const windowSize = 5;
    let pStart = Math.max(0, dtState.page - Math.floor(windowSize/2));
    let pEnd = Math.min(pages, pStart + windowSize);
    if (pEnd - pStart < windowSize) pStart = Math.max(0, pEnd - windowSize);
    for (let i = pStart; i < pEnd; i++) {
        ph += `<button class="${i===dtState.page?'active':''}" onclick="dtState.page=${i};dtRenderRows()">${i+1}</button>`;
    }
    ph += `<button ${dtState.page>=pages-1?'disabled':''} onclick="dtState.page++;dtRenderRows()">›</button>`;
    ph += `<button ${dtState.page>=pages-1?'disabled':''} onclick="dtState.page=${pages-1};dtRenderRows()">»</button>`;
    ph += `<span class="dt-pag-info">${start+1}–${Math.min(start+DT_PAGE_SIZE, total)} of ${fmt(total)}</span>`;
    pag.innerHTML = ph;
}

// ── Tokenizer tab ─────────────────────────────────────────────────
function renderTokenizer() {
    const panel = document.getElementById('panel-tokenizer');
    const tok = CK_TOKENIZER;
    const pre = CK_PREFLIGHT || {};
    if (!tok.available) {
        panel.innerHTML = '<div class="subnote" style="padding:2rem;text-align:center;">No tokenizer data found in this dataset.</div>';
        return;
    }
    const manifest = tok.manifest || {};
    const prompt = tok.prompt_contract || {};
    const drift = tok.manifest_drift || { count: 0, entries: [] };
    const canonicalTags = prompt.canonical_tags || [];
    const exactCoverage = (pre.exact_control_coverage || { expected: canonicalTags.length || 0, covered: 0, missing: [] });
    const kinds = manifest.kind_counts || {};
    const promptShare = prompt.prompt_char_share || 0;
    const promptBadge = promptShare >= 0.10 ? 'badge-green' : (promptShare > 0 ? 'badge-red' : 'badge-purple');

    let html = '<div class="subhead">Tokenizer Corpus</div>';
    html += '<div class="subnote">Merged tokenizer corpus staged from full SVG rows, structural fragments, and prompt-seed rows. The key question here is whether the prompt contract is visible enough to win merge budget.</div>';

    html += '<div class="tok-stat-grid">';
    [
        ['Corpus Rows', fmt(tok.corpus_rows || manifest.tokenizer_rows || 0)],
        ['Corpus Size', tok.corpus_chars ? (tok.corpus_chars / 1024).toFixed(0) + ' KB' : '—'],
        ['Prompt Rows', fmt(prompt.prompt_rows || 0)],
        ['Prompt Share', fmtPct(promptShare, 2)],
        ['Seed Rows', fmt(prompt.tag_seed_rows_actual || tok.tag_seed_rows || manifest.tag_seed_rows || 0)],
        ['Canonical Tags', fmt(canonicalTags.length)],
    ].forEach(([lbl, val]) => {
        html += `<div class="tok-stat"><div class="tok-val">${val}</div><div class="tok-lbl">${lbl}</div></div>`;
    });
    html += '</div>';

    html += sectionHtml('🧭', 'Prompt Contract Coverage', fmtPct(promptShare, 2), promptBadge, 'tokPromptBody');
    html += sectionHtml('🧱', 'Canonical Prompt Atoms', `${fmt(canonicalTags.length)} atoms`, 'badge-blue', 'tokAtomsBody');
    html += sectionHtml('🔐', 'Protected DSL Tokens', `${fmt(tok.protected_tokens_present || 0)}/${fmt(tok.protected_tokens_expected || 0)}`, ((tok.protected_tokens_present || 0) === (tok.protected_tokens_expected || 0) && (tok.protected_tokens_expected || 0) > 0) ? 'badge-green' : 'badge-orange', 'tokProtectedBody');
    html += sectionHtml('⚠️', 'Manifest Drift', drift.count ? `${drift.count} mismatches` : 'clean', drift.count ? 'badge-red' : 'badge-green', 'tokDriftBody');
    html += sectionHtml('📊', 'Token Kind Distribution', fmt(Object.keys(kinds).length) + ' kinds', 'badge-purple', 'tokKindsBody');
    html += sectionHtml('🧪', 'Prompt Prefix Samples', fmt((prompt.prefix_samples || []).length) + ' samples', 'badge-orange', 'tokSamplesBody');

    panel.innerHTML = html;

    const eosCounts = prompt.eos_counts || {};
    const promptChecks = [];
    promptChecks.push({ status: (prompt.prompt_rows || 0) > 0 ? 'ok' : 'err', label: 'Prompt rows exist in the tokenizer corpus', detail: `${fmt(prompt.prompt_rows || 0)} rows` });
    promptChecks.push({ status: prompt.all_seed_rows_in_corpus ? 'ok' : 'err', label: 'All tag-seed rows are included in the corpus', detail: prompt.all_seed_rows_in_corpus ? 'seed rows covered' : `${(prompt.missing_seed_rows || []).length} rows missing` });
    promptChecks.push({ status: promptShare >= 0.10 ? 'ok' : (promptShare > 0 ? 'warn' : 'err'), label: 'Prompt byte share reaches the recommended floor', detail: `target >= 10% · actual ${fmtPct(promptShare, 2)}` });
    promptChecks.push({ status: exactCoverage.covered === exactCoverage.expected && exactCoverage.expected > 0 ? 'ok' : (exactCoverage.covered > 0 ? 'warn' : 'err'), label: 'Exact control-tag atoms exist in the tokenizer vocab', detail: `${fmt(exactCoverage.covered || 0)} / ${fmt(exactCoverage.expected || 0)} exact DSL tags` });
    promptChecks.push({ status: (prompt.eos_variant_count || 0) <= 1 ? 'ok' : 'warn', label: 'EOS spelling is consistent inside prompt rows', detail: Object.entries(eosCounts).map(([k,v]) => `${k}:${v}`).join(' · ') || 'no EOS markers' });
    promptChecks.push({ status: drift.count ? 'warn' : 'ok', label: 'Manifest matches the staged tokenizer files', detail: drift.count ? `${drift.count} mismatched counts` : 'no drift detected' });

    let promptHtml = '<div class="subnote">These checks are the representation-level view of the tokenizer corpus: visibility of prompt atoms, EOS consistency, and whether the manifest still describes the files on disk.</div>';
    promptHtml += promptChecks.map(c => healthRowHtml(c.status, c.label, c.detail)).join('');
    promptHtml += '<div class="tok-metric-grid">';
    [
        ['Prompt rows', `${fmt(prompt.prompt_rows || 0)} / ${fmt(tok.corpus_rows || 0)}`],
        ['Prompt chars', `${fmt(prompt.prompt_chars || 0)} / ${fmt(tok.corpus_content_chars || 0)}`],
        ['Avg tags / seed row', (prompt.prefix_tag_avg || 0).toFixed(2)],
        ['Max tags / seed row', fmt(prompt.prefix_tag_max || 0)],
        ['Avg prompt prefix chars', (prompt.prefix_len_avg || 0).toFixed(1)],
        ['Max prompt prefix chars', fmt(prompt.prefix_len_max || 0)],
    ].forEach(([k, v]) => {
        promptHtml += `<div class="tok-metric"><div class="mkey">${k}</div><div class="mval">${v}</div></div>`;
    });
    promptHtml += '</div>';
    document.getElementById('tokPromptBody').innerHTML = promptHtml;

    let atomsHtml = '<div class="grid-2">';
    atomsHtml += '<div><div class="subhead">Atom families</div><div class="subnote">Canonical prompt atoms extracted from the tag-seed rows. This is the control surface you want the tokenizer to preserve compactly.</div>';
    atomsHtml += distBarsHtml(prompt.tag_family_counts || {}, canonicalTags.length || 1, '#07adf8');
    atomsHtml += '</div>';
    atomsHtml += '<div><div class="subhead">Canonical tags</div><div class="subnote">Exact prompt atoms currently seeded into the corpus.</div><div>';
    atomsHtml += canonicalTags.map(tag => `<span class="token-pill family-${esc(tag.family)}">${esc(tag.tag)}</span>`).join('');
    atomsHtml += '</div></div></div>';
    document.getElementById('tokAtomsBody').innerHTML = atomsHtml;

    const protectedTokens = tok.protected_tokens || [];
    if (!protectedTokens.length) {
        document.getElementById('tokProtectedBody').innerHTML = '<div class="subnote">No protected DSL token inventory was staged for this workspace.</div>';
    } else {
        let protectedHtml = '<div class="subnote">These are the reserved DSL/control tokens baked into the tokenizer. This is the operator-facing proof that the prompt language is atomic rather than left to ordinary BPE merges.</div>';
        protectedHtml += '<table><thead><tr><th>Token</th><th>ID</th><th>Family</th><th>Status</th></tr></thead><tbody>';
        protectedTokens.forEach(row => {
            const ok = !!row.present && !!row.protected;
            protectedHtml += `<tr>
                <td class="mono">${esc(row.token)}</td>
                <td class="mono">${row.id != null ? fmt(row.id) : '—'}</td>
                <td><span class="token-pill family-${esc(row.family || 'other')}">${esc(row.family || 'other')}</span></td>
                <td style="color:${ok ? 'var(--green)' : 'var(--red)'}">${ok ? 'protected' : 'missing'}</td>
            </tr>`;
        });
        protectedHtml += '</tbody></table>';
        document.getElementById('tokProtectedBody').innerHTML = protectedHtml;
    }

    if ((drift.count || 0) === 0) {
        document.getElementById('tokDriftBody').innerHTML = '<div style="color:var(--green);font-size:0.82rem">✓ Manifest counts match the staged tokenizer files.</div>';
    } else {
        let driftHtml = '<div class="subnote">Compares manifest-declared counts against the tokenizer files currently on disk. This catches stale staging after prompt-seed edits.</div>';
        driftHtml += '<table><thead><tr><th>Field</th><th>Manifest</th><th>Actual</th><th>Status</th></tr></thead><tbody>';
        drift.entries.forEach(entry => {
            driftHtml += `<tr><td class="mono">${esc(entry.field)}</td><td class="mono">${fmt(entry.manifest)}</td><td class="mono">${fmt(entry.actual)}</td><td style="color:${entry.match ? 'var(--green)' : 'var(--red)'}">${entry.match ? 'match' : 'mismatch'}</td></tr>`;
        });
        driftHtml += '</tbody></table>';
        document.getElementById('tokDriftBody').innerHTML = driftHtml;
    }

    document.getElementById('tokKindsBody').innerHTML =
        '<div class="subnote">How the staged corpus is currently split across full SVGs, defs fragments, groups, and prompt rows.</div>' +
        distBarsHtml(kinds, Object.values(kinds).reduce((s, v) => s + v, 0), '#9b59b6');

    const sampleBlocks = [];
    (prompt.prefix_samples || []).forEach(sample => sampleBlocks.push(sample));
    (tok.corpus_samples || []).slice(0, 4).forEach(sample => sampleBlocks.push(sample));
    if (!sampleBlocks.length) {
        document.getElementById('tokSamplesBody').innerHTML = '<div class="subnote">No prompt-prefix samples found.</div>';
    } else {
        document.getElementById('tokSamplesBody').innerHTML =
            '<div class="subnote">Representative prompt prefixes and corpus rows. Use the <strong>Text Samples</strong> tab for sortable row-level inspection.</div>' +
            sampleBlocks.map(sample => `<div class="source-block" style="margin-bottom:0.6rem">${esc(sample)}</div>`).join('');
    }
}

function renderAll() {
    renderHeader();
    renderOverview();
    renderPreflight();
    renderGallery();
    renderTextSamples();
    renderTokenizer();
    renderVocabulary();
    renderClassification();
    renderBrowse();
    renderCandidates();
    renderQuality();
    populateFilters();
}

// Tabs
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        const panel = document.getElementById('panel-' + tab.dataset.tab);
        if (panel) panel.classList.add('active');
    });
});

// Search
let searchTimeout;
document.getElementById('searchBox').addEventListener('input', e => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => { state.searchTerm = e.target.value; state.browsePage = 0; renderBrowse(); }, 200);
});

// Filters
document.getElementById('filterFamily').addEventListener('change', e => { state.filterFamily = e.target.value; state.browsePage = 0; renderBrowse(); });
document.getElementById('filterRole').addEventListener('change', e => { state.filterRole = e.target.value; state.browsePage = 0; renderBrowse(); });

// Keyboard
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeModal();
    if (e.key === '/' && !e.ctrlKey && !e.metaKey && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault(); document.getElementById('searchBox').focus();
    }
});

renderAll();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a standalone SVG dataset visualizer from spec03 manifests")
    ap.add_argument("--workspace", required=True, help="Spec workspace root, e.g. version/v7/data/spec03")
    ap.add_argument("--output", required=True, help="Output HTML file")
    args = ap.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    raw_inventory = _load_json(workspace / "manifests" / "raw_assets_inventory.json")
    normalized = _load_json(workspace / "manifests" / "normalized_assets_manifest.json")
    classified = _load_json(workspace / "manifests" / "asset_classification_manifest.json")

    html_doc = build_html(workspace, raw_inventory, normalized, classified)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")
    print(f"[OK] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
