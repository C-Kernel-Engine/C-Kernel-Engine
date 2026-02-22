#!/usr/bin/env python3
"""
Prepare v7 training text as strict ASCII.

Supports plain text or JSONL input. Output is one sample per line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


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
            continue
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


def _iter_text_rows(raw: str) -> Iterable[str]:
    for line in raw.splitlines():
        s = line.strip()
        if s:
            yield s


def _iter_jsonl_rows(raw: str, text_key: str) -> Iterable[str]:
    for i, line in enumerate(raw.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"ERROR: invalid JSONL at line {i}: {exc}") from exc
        if not isinstance(obj, dict):
            continue
        value = obj.get(text_key)
        if isinstance(value, str):
            v = value.strip()
            if v:
                yield v


def _detect_input_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    return "text"


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare ASCII-only v7 dataset text")
    ap.add_argument("--input", required=True, help="Input path (text or JSONL)")
    ap.add_argument("--output", required=True, help="Output text path")
    ap.add_argument(
        "--input-format",
        choices=["auto", "text", "jsonl"],
        default="auto",
        help="Input format; auto uses file extension",
    )
    ap.add_argument(
        "--jsonl-text-key",
        default="text",
        help="JSONL key to extract when --input-format jsonl",
    )
    ap.add_argument(
        "--ascii-mode",
        choices=["xml_escape", "drop", "replace"],
        default="xml_escape",
        help="How to convert non-ASCII chars",
    )
    ap.add_argument(
        "--ascii-map-common",
        action="store_true",
        help="Map common Unicode symbols to keyboard-style ASCII before --ascii-mode conversion",
    )
    ap.add_argument(
        "--svg-only",
        action="store_true",
        help="Keep only rows that start with <svg",
    )
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"ERROR: input not found: {in_path}")

    raw = in_path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    fmt = _detect_input_format(in_path, args.input_format)
    if fmt == "jsonl":
        rows = list(_iter_jsonl_rows(raw, args.jsonl_text_key))
    else:
        rows = list(_iter_text_rows(raw))

    convert = {
        "xml_escape": _ascii_xml_escape,
        "drop": _ascii_drop,
        "replace": _ascii_replace,
    }[args.ascii_mode]

    kept: list[str] = []
    dropped_non_svg = 0
    changed_rows = 0
    changed_chars = 0
    mapped_chars = 0
    for row in rows:
        if args.svg_only and not row.lstrip().startswith("<svg"):
            dropped_non_svg += 1
            continue
        mapped = 0
        if args.ascii_map_common:
            row, mapped = _ascii_map_common(row)
        conv, changed = convert(row)
        if conv.strip():
            kept.append(conv.strip())
            total_changed = int(mapped) + int(changed)
            if total_changed > 0:
                changed_rows += 1
                changed_chars += total_changed
                mapped_chars += int(mapped)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_text = ("\n".join(kept) + "\n") if kept else ""
    out_path.write_text(out_text, encoding="utf-8")

    non_ascii_after = sum(1 for ch in out_text if ord(ch) >= 128)
    print(f"[OK] input:  {in_path}")
    print(f"[OK] output: {out_path}")
    print(f"[INFO] input_format={fmt} ascii_mode={args.ascii_mode} ascii_map_common={bool(args.ascii_map_common)} svg_only={bool(args.svg_only)}")
    print(f"[INFO] rows_in={len(rows)} rows_out={len(kept)} dropped_non_svg={dropped_non_svg}")
    print(f"[INFO] rows_changed={changed_rows} non_ascii_chars_converted={changed_chars} mapped_common_symbols={mapped_chars}")
    print(f"[INFO] output_non_ascii_chars={non_ascii_after}")
    if not kept:
        raise SystemExit("ERROR: output is empty after filtering/conversion")
    if non_ascii_after != 0:
        raise SystemExit("ERROR: output still contains non-ASCII characters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
