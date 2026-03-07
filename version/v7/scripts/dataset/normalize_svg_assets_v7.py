#!/usr/bin/env python3
"""
Normalize imported SVG assets into a clean spec workspace normalized/ tree.

Normalization is the stage-agnostic cleanup layer between raw imports and train splits.
It handles placeholder text replacement, structural cleanup, and dedupe identity.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import string
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

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


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


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


def _extract_svg(raw: str) -> str:
    m_open = re.search(r"<svg\b", raw, flags=re.IGNORECASE)
    m_close = list(re.finditer(r"</svg\s*>", raw, flags=re.IGNORECASE))
    if not m_open or not m_close:
        raise ValueError("missing_svg_root")
    snippet = raw[m_open.start():m_close[-1].end()]
    snippet = snippet.replace("\r\n", "\n").replace("\r", "\n")
    snippet = re.sub(r"<!--.*?-->", "", snippet, flags=re.DOTALL)
    snippet = re.sub(r">\s+<", "><", snippet, flags=re.DOTALL)
    return snippet.strip()


def _sanitize_text_nodes(svg_text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        inner = match.group(1)
        if not inner or not inner.strip():
            return f">{inner}<"
        return f">{html.escape(_ascii_clean(inner), quote=False)}<"

    # Escape literal text-node content without touching tag markup.
    return re.sub(r">([^<]+)<", repl, svg_text)


def _alpha(n: int) -> str:
    letters = string.ascii_uppercase
    s = ""
    x = n
    while True:
        s = letters[x % 26] + s
        x = x // 26 - 1
        if x < 0:
            break
    return s


def _placeholder_for(tag: str, text: str, counts: Counter[str]) -> str:
    stripped = " ".join(text.split())
    if not stripped:
        return text
    lower = stripped.lower()
    if tag == "title":
        counts["TITLE"] += 1
        return f"TITLE_{_alpha(counts['TITLE'] - 1)}"
    if tag == "desc":
        counts["PARA"] += 1
        return f"PARA_{_alpha(counts['PARA'] - 1)}"
    if "axis" in lower:
        counts["AXIS"] += 1
        return f"AXIS_{_alpha(counts['AXIS'] - 1)}"
    if len(stripped) > 48:
        counts["PARA"] += 1
        return f"PARA_{_alpha(counts['PARA'] - 1)}"
    if len(stripped) > 18:
        counts["SUBTITLE"] += 1
        return f"SUBTITLE_{_alpha(counts['SUBTITLE'] - 1)}"
    counts["LABEL"] += 1
    return f"LABEL_{counts['LABEL']}"


def _normalize_tree(svg_text: str) -> tuple[str, dict[str, int], dict[str, int]]:
    root = ET.fromstring(_sanitize_text_nodes(svg_text))
    placeholders: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()

    for elem in root.iter():
        tag = _local(elem.tag)
        tag_counts[tag] += 1

        if tag in {"metadata", "sodipodi:namedview"}:
            continue

        # Strip obvious editor noise.
        for attr in list(elem.attrib.keys()):
            attr_local = _local(attr)
            if attr_local.startswith("inkscape") or attr_local.startswith("sodipodi"):
                del elem.attrib[attr]
                continue
            elem.attrib[attr] = _ascii_clean(elem.attrib[attr])

        if tag in {"title", "desc", "text", "tspan"} and elem.text and elem.text.strip():
            elem.text = _placeholder_for(tag, elem.text, placeholders)
        if tag in {"text", "tspan"} and elem.tail and elem.tail.strip():
            elem.tail = ""
        elif elem.text:
            elem.text = _ascii_clean(elem.text)
        if elem.tail:
            elem.tail = _ascii_clean(elem.tail)

    normalized = ET.tostring(root, encoding="unicode", method="xml")
    normalized = re.sub(r">\s+<", "><", normalized, flags=re.DOTALL)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized, dict(placeholders), dict(tag_counts)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize imported SVG assets into normalized/ and write dedupe manifest")
    ap.add_argument("--workspace", required=True, help="Spec workspace root, e.g. version/v7/data/spec03")
    ap.add_argument("--inventory", default=None, help="Optional raw inventory path (default: WORKSPACE/manifests/raw_assets_inventory.json)")
    ap.add_argument("--manifest", default=None, help="Optional output manifest path (default: WORKSPACE/manifests/normalized_assets_manifest.json)")
    ap.add_argument("--force", action="store_true", help="Overwrite normalized outputs")
    args = ap.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    inventory_path = Path(args.inventory).expanduser().resolve() if args.inventory else (workspace / "manifests" / "raw_assets_inventory.json")
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else (workspace / "manifests" / "normalized_assets_manifest.json")
    norm_root = workspace / "normalized"
    norm_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    entries = inventory.get("entries") if isinstance(inventory.get("entries"), list) else []
    if not entries:
        raise SystemExit(f"ERROR: no entries in inventory: {inventory_path}")

    norm_rows: list[dict[str, object]] = []
    first_by_hash: dict[str, str] = {}
    failures: list[dict[str, str]] = []
    placeholder_totals: Counter[str] = Counter()
    tag_totals: Counter[str] = Counter()

    for row in entries:
        imported_path = Path(str(row.get("imported_path") or ""))
        source_name = str(row.get("source_name") or "unknown")
        try:
            raw = imported_path.read_text(encoding="utf-8", errors="replace")
            svg = _extract_svg(raw)
            normalized, placeholders, tag_counts = _normalize_tree(svg)
            norm_hash = _sha256_text(normalized)
            duplicate_of = first_by_hash.get(norm_hash)
            if duplicate_of is None:
                first_by_hash[norm_hash] = str(imported_path)

            out_dir = norm_root / source_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{imported_path.stem}--{norm_hash[:12]}.svg"
            out_path = out_dir / out_name
            if out_path.exists() and not args.force:
                pass
            else:
                out_path.write_text(normalized, encoding="utf-8")

            placeholder_totals.update(placeholders)
            tag_totals.update(tag_counts)
            norm_rows.append({
                "source_name": source_name,
                "source_path": row.get("source_path"),
                "imported_path": str(imported_path),
                "normalized_path": str(out_path.resolve()),
                "normalized_sha256": norm_hash,
                "duplicate_of_imported_path": duplicate_of,
                "placeholders": placeholders,
                "tag_counts": tag_counts,
                "chars": len(normalized),
            })
        except Exception as exc:
            failures.append({"imported_path": str(imported_path), "error": str(exc)})

    manifest = {
        "schema": "ck.svg_normalized_assets_manifest.v1",
        "workspace": str(workspace),
        "inventory_path": str(inventory_path),
        "normalized_root": str(norm_root),
        "normalized_entries": len(norm_rows),
        "unique_normalized_hashes": len(first_by_hash),
        "duplicate_normalized_entries": sum(1 for r in norm_rows if r.get("duplicate_of_imported_path")),
        "failures": failures,
        "placeholder_totals": dict(placeholder_totals),
        "tag_totals": dict(tag_totals),
        "entries": norm_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"[OK] normalized={len(norm_rows)} unique={len(first_by_hash)} "
        f"duplicates={sum(1 for r in norm_rows if r.get('duplicate_of_imported_path'))} failures={len(failures)}"
    )
    print(f"[OK] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
