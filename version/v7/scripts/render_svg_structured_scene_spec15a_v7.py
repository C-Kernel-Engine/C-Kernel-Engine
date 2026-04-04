#!/usr/bin/env python3
"""Render strict family-generic spec15a memory_map scenes into SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec12_v7 import _background_motif, _defs, _memory_map, _palette, _canvas_size
from spec15a_scene_canonicalizer_v7 import canonicalize_scene_text


def render_structured_scene_spec15a_svg(text: str, content: dict[str, Any] | None = None) -> str:
    scene_doc = canonicalize_scene_text(text)
    scene = scene_doc.to_runtime()
    scene["_content"] = content or {}
    width, height = _canvas_size(scene)
    palette = _palette(scene)
    body = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#bgGrad)"/>',
        _background_motif(scene, palette, width, height),
        _memory_map(scene, palette, width, height),
    ]
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f"{_defs(scene, palette)}"
        f'{"".join(body)}'
        "</svg>"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Render strict spec15a memory_map scene DSL to SVG.")
    ap.add_argument("--scene", default=None, help="Inline scene document.")
    ap.add_argument("--scene-file", default=None, help="Path to a compact scene document file.")
    ap.add_argument("--content-json", default=None, help="Optional content JSON payload path.")
    ap.add_argument("--out", default=None, help="Optional output SVG path.")
    args = ap.parse_args()

    if bool(args.scene) == bool(args.scene_file):
        raise SystemExit("ERROR: pass exactly one of --scene or --scene-file")
    text = args.scene if args.scene is not None else Path(args.scene_file).read_text(encoding="utf-8")
    content = None
    if args.content_json:
        content = json.loads(Path(args.content_json).read_text(encoding="utf-8"))
    svg = render_structured_scene_spec15a_svg(text, content=content)
    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(svg, encoding="utf-8")
        print(f"[OK] wrote: {out}")
    else:
        print(svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
