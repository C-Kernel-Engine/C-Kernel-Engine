#!/usr/bin/env python3
"""Build a cross-family compiler smoke report for spec16 shared scene bundles."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Callable

from render_svg_structured_scene_spec14b_v7 import render_structured_scene_spec14b_svg
from render_svg_structured_scene_spec15a_v7 import render_structured_scene_spec15a_svg
from render_svg_structured_scene_spec15b_v7 import render_structured_scene_spec15b_svg
from spec16_bundle_lowering_v7 import lower_scene_bundle_to_scene_dsl


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "version" / "v7" / "reports"

_CASE_MANIFEST = {
    "memory_map_bundle.json": {
        "family": "memory_map",
        "content_json": REPORTS / "spec15a_gold_mappings" / "bump_allocator_quant.content.json",
        "renderer": render_structured_scene_spec15a_svg,
        "title": "Spec16 Memory Map Foundation Bundle",
    },
    "timeline_bundle.json": {
        "family": "timeline",
        "content_json": REPORTS / "spec14b_gold_mappings" / "ir-timeline-why.content.json",
        "renderer": render_structured_scene_spec14b_svg,
        "title": "Spec16 Timeline Foundation Bundle",
    },
    "system_diagram_bundle.json": {
        "family": "system_diagram",
        "content_json": REPORTS / "spec15b_gold_mappings" / "ir-pipeline-flow-system.content.json",
        "renderer": render_structured_scene_spec15b_svg,
        "title": "Spec16 System Diagram Foundation Bundle",
    },
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_html(report: dict[str, Any]) -> str:
    rows: list[str] = []
    for case in report.get("cases", []):
        status = "pass" if case.get("compiled") else "fail"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(case.get('name') or ''))}</td>"
            f"<td>{html.escape(str(case.get('family') or ''))}</td>"
            f"<td>{html.escape(str(case.get('content_json') or ''))}</td>"
            f"<td class=\"{status}\">{html.escape(str(case.get('status') or ''))}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8"/>',
            "  <title>Spec16 Compiler Smoke</title>",
            "  <style>",
            "    body { margin: 0; padding: 28px; font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; background: #f7f3ea; color: #1f2933; }",
            "    main { max-width: 1240px; margin: 0 auto; }",
            "    .card { background: #fffdf8; border: 1px solid #d8cfbf; border-radius: 18px; padding: 18px; box-shadow: 0 10px 22px rgba(31,41,51,0.06); }",
            "    table { width: 100%; border-collapse: collapse; }",
            "    th, td { text-align: left; vertical-align: top; padding: 10px 12px; border-bottom: 1px solid #e7dece; }",
            "    .pass { color: #116149; font-weight: 700; }",
            "    .fail { color: #b42318; font-weight: 700; }",
            "  </style>",
            "</head>",
            "<body>",
            "<main>",
            "<h1>Spec16 Compiler Smoke</h1>",
            "<div class=\"card\">",
            f"<p>Compiled bundles: <strong>{int(report.get('compiled_count', 0))}/{int(report.get('count', 0))}</strong></p>",
            "<table>",
            "<thead><tr><th>Name</th><th>Family</th><th>Content</th><th>Status</th></tr></thead>",
            f"<tbody>{''.join(rows)}</tbody>",
            "</table>",
            "</div>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def build_report(bundle_dir: Path, out_dir: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(bundle_dir.glob("*.json")):
        manifest = _CASE_MANIFEST.get(path.name)
        if manifest is None:
            cases.append(
                {
                    "name": path.name,
                    "family": "",
                    "content_json": "",
                    "compiled": False,
                    "status": "no manifest entry",
                }
            )
            continue
        content_json_path = Path(manifest["content_json"]).resolve()
        renderer: Callable[[str, dict[str, Any] | None], str] = manifest["renderer"]
        try:
            bundle_doc = _load_json(path)
            scene_dsl = lower_scene_bundle_to_scene_dsl(bundle_doc)
            content_json = _load_json(content_json_path)
            svg = renderer(scene_dsl, content=content_json)
            stem = path.name.replace("_bundle.json", "")
            scene_path = out_dir / f"{stem}.scene.dsl"
            svg_path = out_dir / f"{stem}.svg"
            scene_path.write_text(scene_dsl + "\n", encoding="utf-8")
            svg_path.write_text(svg + "\n", encoding="utf-8")
            cases.append(
                {
                    "name": path.name,
                    "family": manifest["family"],
                    "content_json": str(content_json_path),
                    "compiled": True,
                    "status": "ok",
                    "scene_dsl": str(scene_path),
                    "svg": str(svg_path),
                }
            )
        except Exception as exc:
            cases.append(
                {
                    "name": path.name,
                    "family": manifest["family"],
                    "content_json": str(content_json_path),
                    "compiled": False,
                    "status": str(exc),
                }
            )
    compiled_count = sum(1 for case in cases if case.get("compiled"))
    return {
        "schema": "ck.spec16_compiler_smoke.v1",
        "bundle_dir": str(bundle_dir),
        "output_dir": str(out_dir),
        "count": len(cases),
        "compiled_count": compiled_count,
        "cases": cases,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--json-out", required=True, type=Path)
    ap.add_argument("--html-out", required=True, type=Path)
    args = ap.parse_args()

    report = build_report(args.bundle_dir.expanduser().resolve(), args.out_dir.expanduser().resolve())
    json_out = args.json_out.expanduser().resolve()
    html_out = args.html_out.expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    html_out.write_text(_render_html(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
