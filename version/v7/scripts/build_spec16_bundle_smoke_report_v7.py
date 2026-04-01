#!/usr/bin/env python3
"""Validate spec16 shared scene bundles and emit a small smoke report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from spec16_scene_bundle_v7 import canonicalize_scene_bundle


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_html(report: dict[str, Any]) -> str:
    rows: list[str] = []
    for case in report.get("cases", []):
        status = "pass" if case.get("valid") else "fail"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(case.get('name') or ''))}</td>"
            f"<td>{html.escape(str(case.get('family') or ''))}</td>"
            f"<td>{html.escape(str(case.get('form') or ''))}</td>"
            f"<td>{html.escape(str(case.get('prompt_tags') or ''))}</td>"
            f"<td class=\"{status}\">{html.escape(str(case.get('status') or ''))}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8"/>',
            "  <title>Spec16 Bundle Smoke</title>",
            "  <style>",
            "    body { margin: 0; padding: 28px; font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; background: #f7f3ea; color: #1f2933; }",
            "    main { max-width: 1200px; margin: 0 auto; }",
            "    .card { background: #fffdf8; border: 1px solid #d8cfbf; border-radius: 18px; padding: 18px; box-shadow: 0 10px 22px rgba(31,41,51,0.06); }",
            "    table { width: 100%; border-collapse: collapse; }",
            "    th, td { text-align: left; vertical-align: top; padding: 10px 12px; border-bottom: 1px solid #e7dece; }",
            "    .pass { color: #116149; font-weight: 700; }",
            "    .fail { color: #b42318; font-weight: 700; }",
            "    code { font-family: 'IBM Plex Mono', monospace; font-size: 12px; }",
            "  </style>",
            "</head>",
            "<body>",
            "<main>",
            "<h1>Spec16 Shared Bundle Smoke</h1>",
            "<div class=\"card\">",
            f"<p>Validated bundles: <strong>{int(report.get('valid_count', 0))}/{int(report.get('count', 0))}</strong></p>",
            "<table>",
            "<thead><tr><th>Name</th><th>Family</th><th>Form</th><th>Prompt Tags</th><th>Status</th></tr></thead>",
            f"<tbody>{''.join(rows)}</tbody>",
            "</table>",
            "</div>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def build_report(bundle_dir: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for path in sorted(bundle_dir.glob("*.json")):
        doc = _load_json(path)
        try:
            bundle = canonicalize_scene_bundle(doc)
            cases.append(
                {
                    "name": path.name,
                    "valid": True,
                    "status": "ok",
                    "family": bundle.family,
                    "form": bundle.form,
                    "normalized_bundle": bundle.to_dict(),
                    "prompt_tags": bundle.to_prompt_tags(),
                }
            )
        except Exception as exc:
            cases.append(
                {
                    "name": path.name,
                    "valid": False,
                    "status": str(exc),
                    "family": str(doc.get("family") or doc.get("layout") or ""),
                    "form": str(doc.get("form") or ""),
                    "normalized_bundle": None,
                    "prompt_tags": "",
                }
            )
    valid_count = sum(1 for case in cases if case.get("valid"))
    return {
        "schema": "ck.spec16_bundle_smoke.v1",
        "bundle_dir": str(bundle_dir),
        "count": len(cases),
        "valid_count": valid_count,
        "cases": cases,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", required=True, type=Path)
    ap.add_argument("--json-out", required=True, type=Path)
    ap.add_argument("--html-out", required=True, type=Path)
    args = ap.parse_args()

    bundle_dir = args.bundle_dir.expanduser().resolve()
    report = build_report(bundle_dir)

    json_out = args.json_out.expanduser().resolve()
    html_out = args.html_out.expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    html_out.write_text(_render_html(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
