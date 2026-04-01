#!/usr/bin/env python3
"""Render strict spec16 shared scene bundles to SVG via family lowerers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from render_svg_structured_scene_spec14b_v7 import render_structured_scene_spec14b_svg
    from render_svg_structured_scene_spec15a_v7 import render_structured_scene_spec15a_svg
    from render_svg_structured_scene_spec15b_v7 import render_structured_scene_spec15b_svg
    from spec16_bundle_lowering_v7 import lower_scene_bundle_to_scene_dsl
    from spec16_scene_bundle_canonicalizer_v7 import canonicalize_scene_bundle_text
except ModuleNotFoundError:  # pragma: no cover
    from version.v7.scripts.render_svg_structured_scene_spec14b_v7 import render_structured_scene_spec14b_svg
    from version.v7.scripts.render_svg_structured_scene_spec15a_v7 import render_structured_scene_spec15a_svg
    from version.v7.scripts.render_svg_structured_scene_spec15b_v7 import render_structured_scene_spec15b_svg
    from version.v7.scripts.spec16_bundle_lowering_v7 import lower_scene_bundle_to_scene_dsl
    from version.v7.scripts.spec16_scene_bundle_canonicalizer_v7 import canonicalize_scene_bundle_text


def render_structured_scene_spec16_svg(text: str, content: dict[str, Any] | None = None) -> str:
    bundle = canonicalize_scene_bundle_text(text)
    scene_dsl = lower_scene_bundle_to_scene_dsl(bundle)
    if bundle.family == "memory_map":
        return render_structured_scene_spec15a_svg(scene_dsl, content=content)
    if bundle.family == "timeline":
        return render_structured_scene_spec14b_svg(scene_dsl, content=content)
    if bundle.family == "system_diagram":
        return render_structured_scene_spec15b_svg(scene_dsl, content=content)
    raise ValueError(f"unsupported spec16 family: {bundle.family!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Render strict spec16 scene bundles to SVG.")
    ap.add_argument("--bundle", default=None, help="Inline bundle document.")
    ap.add_argument("--bundle-file", default=None, help="Path to a compact bundle document file.")
    ap.add_argument("--content-json", default=None, help="Optional content JSON payload path.")
    ap.add_argument("--out", default=None, help="Optional output SVG path.")
    args = ap.parse_args()

    if bool(args.bundle) == bool(args.bundle_file):
        raise SystemExit("ERROR: pass exactly one of --bundle or --bundle-file")
    text = args.bundle if args.bundle is not None else Path(args.bundle_file).read_text(encoding="utf-8")
    content = None
    if args.content_json:
        content = json.loads(Path(args.content_json).read_text(encoding="utf-8"))
    svg = render_structured_scene_spec16_svg(text, content=content)
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
