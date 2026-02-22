#!/usr/bin/env python3
"""
Build one-SVG-per-line training text from SVG asset files.

This is intended for v7 SVG training pipelines where each line is one
standalone <svg ...>...</svg> sample.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path


COMMON_ASCII_MAP: tuple[tuple[str, str], ...] = (
    ("→", "->"),
    ("←", "&lt;-"),
    ("↔", "&lt;-&gt;"),
    ("⇒", "=>"),
    ("⇐", "&lt;="),
    ("±", "+/-"),
    ("×", "x"),
    ("÷", "/"),
    ("≤", "&lt;="),
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
    ("◆", "&lt;&gt;"),
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
    ("✅", "ok"),
    ("❌", "x"),
    ("⚠", "warn"),
    ("✨", "*"),
    ("⭐", "*"),
    ("🙂", ":)"),
    ("😊", ":)"),
    ("😉", ";)"),
    ("😐", ":|"),
    ("😕", ":/"),
    ("😢", ":("),
    ("😭", ":'("),
    ("😡", ">:("),
    ("🔥", "fire"),
)


def _ascii_map_common(text: str) -> tuple[str, int]:
    out = text
    changed = 0
    for src, dst in COMMON_ASCII_MAP:
        if src not in out:
            continue
        count = out.count(src)
        out = out.replace(src, dst)
        changed += int(count)
    return out, changed


def _ascii_xml_escape(text: str) -> tuple[str, int]:
    out: list[str] = []
    changed = 0
    for ch in text:
        cp = ord(ch)
        if cp < 128:
            out.append(ch)
        else:
            out.append(f"&#x{cp:X};")
            changed += 1
    return "".join(out), changed


def _ascii_drop(text: str) -> tuple[str, int]:
    out = text.encode("ascii", errors="ignore").decode("ascii")
    changed = sum(1 for ch in text if ord(ch) >= 128)
    return out, changed


def _ascii_replace(text: str) -> tuple[str, int]:
    out = text.encode("ascii", errors="replace").decode("ascii")
    changed = sum(1 for ch in text if ord(ch) >= 128)
    return out, changed


def _extract_svg_snippet(raw: str) -> str:
    open_m = re.search(r"<svg\b", raw, flags=re.IGNORECASE)
    if not open_m:
        raise ValueError("missing_svg_open_tag")
    close_matches = list(re.finditer(r"</svg\s*>", raw, flags=re.IGNORECASE))
    if not close_matches:
        raise ValueError("missing_svg_close_tag")

    start = open_m.start()
    end = close_matches[-1].end()
    snippet = raw[start:end]

    # Drop XML comments and flatten to one line.
    snippet = snippet.replace("\r\n", "\n").replace("\r", "\n")
    snippet = re.sub(r"<!--.*?-->", "", snippet, flags=re.DOTALL)
    snippet = re.sub(r">\s+<", "><", snippet, flags=re.DOTALL)
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if not snippet.lower().startswith("<svg"):
        raise ValueError("snippet_does_not_start_with_svg")
    return snippet


def _validate_svg(snippet: str) -> None:
    try:
        root = ET.fromstring(snippet)
    except ET.ParseError as exc:
        raise ValueError(f"xml_parse_error: {exc}") from exc
    tag = root.tag.split("}", 1)[-1] if "}" in root.tag else root.tag
    if str(tag).lower() != "svg":
        raise ValueError(f"root_not_svg:{tag}")


def _expand_inputs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in patterns:
        if not raw.strip():
            continue
        expanded = Path(raw).expanduser()
        candidates: list[Path] = []
        if expanded.exists() and expanded.is_dir():
            candidates = sorted(expanded.glob("*.svg"))
        else:
            matches = sorted(glob.glob(str(expanded), recursive=True))
            candidates = [Path(m) for m in matches if Path(m).suffix.lower() == ".svg"]

        for p in candidates:
            try:
                key = str(p.resolve())
            except Exception:
                key = str(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build one-SVG-per-line corpus from SVG files")
    ap.add_argument(
        "--assets-glob",
        action="append",
        required=True,
        help="Glob pattern or directory for source SVG files (repeatable)",
    )
    ap.add_argument("--output", required=True, help="Output text file (one SVG per line)")
    ap.add_argument("--manifest", default=None, help="Optional JSON manifest path")
    ap.add_argument(
        "--ascii-mode",
        choices=["none", "xml_escape", "drop", "replace"],
        default="none",
        help="Optional non-ASCII conversion on output rows",
    )
    ap.add_argument(
        "--ascii-map-common",
        action="store_true",
        help="Map common Unicode symbols to keyboard-style ASCII before --ascii-mode conversion",
    )
    ap.add_argument("--min-chars", type=int, default=16, help="Drop rows shorter than this")
    ap.add_argument("--max-samples", type=int, default=0, help="Cap output rows (0 = no cap)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle rows before writing")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    ap.set_defaults(dedupe=True)
    ap.add_argument("--dedupe", dest="dedupe", action="store_true", help="Deduplicate rows (default)")
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false", help="Keep duplicates")
    args = ap.parse_args()

    files = _expand_inputs(args.assets_glob)
    if not files:
        raise SystemExit("ERROR: no SVG files matched --assets-glob inputs")

    convert = {
        "none": lambda s: (s, 0),
        "xml_escape": _ascii_xml_escape,
        "drop": _ascii_drop,
        "replace": _ascii_replace,
    }[args.ascii_mode]

    rows: list[str] = []
    seen_rows: set[str] = set()
    failures: list[dict[str, str]] = []
    converted_chars_total = 0
    mapped_chars_total = 0
    input_non_ascii_chars_total = 0
    output_non_ascii_chars_total = 0

    for path in files:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            input_non_ascii_chars_total += sum(1 for ch in raw if ord(ch) >= 128)
            svg_line = _extract_svg_snippet(raw)
            if len(svg_line) < int(args.min_chars):
                raise ValueError(f"too_short:{len(svg_line)}")
            mapped = 0
            if args.ascii_map_common:
                svg_line, mapped = _ascii_map_common(svg_line)
            converted, changed = convert(svg_line)
            converted_chars_total += int(changed)
            mapped_chars_total += int(mapped)
            if not converted.strip():
                raise ValueError("empty_after_conversion")
            _validate_svg(converted)
            output_non_ascii_chars_total += sum(1 for ch in converted if ord(ch) >= 128)
            if args.dedupe:
                if converted in seen_rows:
                    continue
                seen_rows.add(converted)
            rows.append(converted)
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})

    if args.shuffle:
        rnd = random.Random(int(args.seed))
        rnd.shuffle(rows)

    if args.max_samples and args.max_samples > 0:
        rows = rows[: int(args.max_samples)]

    if not rows:
        raise SystemExit("ERROR: no valid SVG rows produced")

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
    manifest = {
        "format": "svg_assets_corpus.v1",
        "source_patterns": list(args.assets_glob),
        "source_file_count": int(len(files)),
        "output_path": str(out_path),
        "output_rows": int(len(rows)),
        "ascii_mode": args.ascii_mode,
        "ascii_map_common": bool(args.ascii_map_common),
        "dedupe": bool(args.dedupe),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "min_chars": int(args.min_chars),
        "max_samples": int(args.max_samples),
        "input_non_ascii_chars_total": int(input_non_ascii_chars_total),
        "output_non_ascii_chars_total": int(output_non_ascii_chars_total),
        "mapped_common_symbols_total": int(mapped_chars_total),
        "converted_non_ascii_chars_total": int(converted_chars_total),
        "failed_file_count": int(len(failures)),
        "failed_files": failures[:200],
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[OK] manifest: {manifest_path}")

    print(f"[OK] output: {out_path}")
    print(f"[INFO] source_files={len(files)} output_rows={len(rows)} failed={len(failures)}")
    print(f"[INFO] ascii_mode={args.ascii_mode} ascii_map_common={bool(args.ascii_map_common)} dedupe={bool(args.dedupe)} shuffle={bool(args.shuffle)}")
    print(f"[INFO] mapped_common_symbols_total={mapped_chars_total}")
    print(f"[INFO] output_non_ascii_chars={output_non_ascii_chars_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
