#!/usr/bin/env python3
"""Build a spec19 bundle-to-DSL-to-SVG smoke report with visual previews."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec16_v7 import render_structured_scene_spec16_svg
from spec16_bundle_lowering_v7 import lower_scene_bundle_to_scene_dsl
from spec16_scene_bundle_canonicalizer_v7 import canonicalize_scene_bundle_text, serialize_scene_bundle


PREFERRED_SURFACES = (
    "explicit_bundle_anchor",
    "routebook_direct",
    "style_topology_bridge",
)


def _find_catalog(run_dir: Path, prefix: str) -> Path:
    candidates = [
        run_dir / "dataset" / "manifests" / "generated" / "structured_atoms" / f"{prefix}_render_catalog.json",
        run_dir / "manifests" / "generated" / "structured_atoms" / f"{prefix}_render_catalog.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"render catalog not found for prefix {prefix!r} under {run_dir}")


def _load_catalog(run_dir: Path, prefix: str) -> list[dict[str, Any]]:
    doc = json.loads(_find_catalog(run_dir, prefix).read_text(encoding="utf-8"))
    if not isinstance(doc, list):
        raise ValueError("spec19 render catalog must be a JSON list")
    return [row for row in doc if isinstance(row, dict)]


def _select_cases(catalog_rows: list[dict[str, Any]], max_per_family: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    families = sorted({str(row.get("family") or row.get("layout") or "") for row in catalog_rows if str(row.get("family") or row.get("layout") or "")})
    for family in families:
        family_rows = [row for row in catalog_rows if str(row.get("family") or row.get("layout") or "") == family]
        picked = 0
        for surface in PREFERRED_SURFACES:
            row = next(
                (
                    candidate
                    for candidate in family_rows
                    if str(candidate.get("prompt_surface") or "") == surface
                    and str(candidate.get("split") or "") == "train"
                ),
                None,
            )
            if row is None:
                continue
            key = (
                family,
                str(row.get("profile_id") or row.get("case_id") or ""),
                str(row.get("prompt_surface") or ""),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(row)
            picked += 1
            if picked >= max(1, int(max_per_family)):
                break
        if picked >= max(1, int(max_per_family)):
            continue
        for row in family_rows:
            if str(row.get("split") or "") != "train":
                continue
            key = (
                family,
                str(row.get("profile_id") or row.get("case_id") or ""),
                str(row.get("prompt_surface") or ""),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(row)
            picked += 1
            if picked >= max(1, int(max_per_family)):
                break
    return selected


def _slug(text: str) -> str:
    out = [ch.lower() if ch.isalnum() else "_" for ch in str(text or "").strip()]
    return "".join(out).strip("_") or "case"


def _svg_rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), start=base.resolve())).as_posix()


def _render_html(report: dict[str, Any], out_dir: Path) -> str:
    cards: list[str] = []
    for row in report.get("cases") or []:
        status = "ok" if row.get("compiled") else "fail"
        svg_rel = _svg_rel(Path(str(row.get("svg_path") or "")), out_dir) if row.get("svg_path") else ""
        cards.append(
            f"""
            <section class="card">
              <div class="meta">
                <div>
                  <h2>{html.escape(str(row.get("family") or ""))} · {html.escape(str(row.get("prompt_surface") or ""))}</h2>
                  <p class="sub">{html.escape(str(row.get("profile_id") or row.get("case_id") or ""))}</p>
                </div>
                <div class="pill {status}">{html.escape(str(row.get("status") or ""))}</div>
              </div>
              <div class="checks">
                <span>bundle={html.escape(str(row.get("bundle_valid") or False).lower())}</span>
                <span>dsl={html.escape(str(row.get("dsl_emitted") or False).lower())}</span>
                <span>svg={html.escape(str(row.get("compiled") or False).lower())}</span>
                <span>exact={html.escape(str(row.get("svg_exact") or False).lower())}</span>
                <span>rects={html.escape(str(row.get("rects") or 0))}</span>
                <span>paths={html.escape(str(row.get("paths") or 0))}</span>
              </div>
              <div class="paths">
                <a href="{html.escape(_svg_rel(Path(str(row.get('bundle_path') or '')), out_dir))}">bundle</a>
                <a href="{html.escape(_svg_rel(Path(str(row.get('scene_dsl_path') or '')), out_dir))}">scene DSL</a>
                <a href="{html.escape(svg_rel)}">compiled SVG</a>
              </div>
              <p class="prompt">{html.escape(str(row.get("prompt") or ""))}</p>
              <div class="frame">
                <img src="{html.escape(svg_rel)}" alt="{html.escape(str(row.get('profile_id') or row.get('case_id') or 'spec19 smoke'))}"/>
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Spec19 Compiler Smoke Report</title>
  <style>
    :root {{
      --bg: #0b1722;
      --ink: #eef6ff;
      --muted: #99b5cd;
      --card: rgba(14, 26, 39, 0.92);
      --line: rgba(124, 208, 255, 0.15);
      --ok: #72e0aa;
      --fail: #ff7f7f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(86,174,255,0.18), transparent 28%),
        linear-gradient(180deg, #07131f 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 40px 24px 64px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 14px;
      margin: 24px 0 32px;
    }}
    .box, .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 16px 40px rgba(4, 10, 16, 0.26);
    }}
    .box {{ padding: 18px 20px; }}
    .k {{ font: 700 28px/1.1 "IBM Plex Sans", "Segoe UI", sans-serif; }}
    .v {{ color: var(--muted); margin-top: 6px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    .card {{ padding: 18px; }}
    .meta {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; }}
    .meta h2 {{ margin: 0 0 6px; font-size: 24px; }}
    .sub {{ margin: 0; color: var(--muted); }}
    .pill {{ border-radius: 999px; padding: 7px 12px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; border: 1px solid var(--line); }}
    .pill.ok {{ color: var(--ok); }}
    .pill.fail {{ color: var(--fail); }}
    .checks, .paths {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; color: var(--muted); font-size: 13px; }}
    .checks span, .paths a, .paths span {{
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.09);
      padding: 6px 10px;
      text-decoration: none;
      color: inherit;
    }}
    .prompt {{ margin: 14px 0 0; color: var(--ink); font-family: "IBM Plex Mono", monospace; font-size: 13px; line-height: 1.5; }}
    .frame {{ margin-top: 16px; border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.02); }}
    .frame img {{ display: block; width: 100%; height: auto; background: white; }}
  </style>
</head>
<body>
<main>
  <h1>Spec19 Compiler Smoke Report</h1>
  <p>Pre-training contract gate for the shared bundle line: canonical bundle parse, bundle-to-scene lowering, compiler SVG generation, and exact SVG reproduction on representative spec19 rows.</p>
  <div class="summary">
    <div class="box"><div class="k">{int(report.get("compiled_count", 0))}/{int(report.get("count", 0))}</div><div class="v">Compiled Cases</div></div>
    <div class="box"><div class="k">{int(report.get("svg_exact_count", 0))}/{int(report.get("count", 0))}</div><div class="v">SVG Exact Cases</div></div>
    <div class="box"><div class="k">{len(report.get("families", []))}</div><div class="v">Families Covered</div></div>
  </div>
  <div class="grid">
    {''.join(cards)}
  </div>
</main>
</body>
</html>
"""


def build_report(run_dir: Path, prefix: str, out_dir: Path, *, max_per_family: int = 3) -> dict[str, Any]:
    catalog_rows = _load_catalog(run_dir, prefix)
    selected = _select_cases(catalog_rows, max_per_family=max_per_family)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for row in selected:
        family = str(row.get("family") or row.get("layout") or "")
        prompt_surface = str(row.get("prompt_surface") or "")
        profile_id = str(row.get("profile_id") or row.get("case_id") or "")
        stem = _slug(f"{family}_{prompt_surface}_{profile_id}")
        bundle_path = out_dir / f"{stem}.bundle.txt"
        scene_dsl_path = out_dir / f"{stem}.scene.dsl"
        svg_path = out_dir / f"{stem}.svg"
        output_tokens = str(row.get("output_tokens") or "")
        expected_svg = str(row.get("svg_xml") or "")
        content_json = row.get("content_json") if isinstance(row.get("content_json"), dict) else None
        try:
            bundle = canonicalize_scene_bundle_text(output_tokens)
            canonical_bundle = serialize_scene_bundle(bundle)
            scene_dsl = lower_scene_bundle_to_scene_dsl(bundle)
            rendered_svg = render_structured_scene_spec16_svg(canonical_bundle, content=content_json)
            bundle_path.write_text(canonical_bundle + "\n", encoding="utf-8")
            scene_dsl_path.write_text(scene_dsl + "\n", encoding="utf-8")
            svg_path.write_text(rendered_svg + "\n", encoding="utf-8")
            cases.append(
                {
                    "profile_id": profile_id,
                    "case_id": str(row.get("case_id") or ""),
                    "family": family,
                    "prompt_surface": prompt_surface,
                    "prompt": str(row.get("prompt") or ""),
                    "status": "ok",
                    "bundle_valid": True,
                    "dsl_emitted": bool(scene_dsl.strip()),
                    "compiled": bool(rendered_svg.strip()),
                    "svg_exact": rendered_svg.strip() == expected_svg.strip(),
                    "bundle_path": str(bundle_path),
                    "scene_dsl_path": str(scene_dsl_path),
                    "svg_path": str(svg_path),
                    "rects": rendered_svg.count("<rect "),
                    "paths": rendered_svg.count("<path "),
                }
            )
        except Exception as exc:
            cases.append(
                {
                    "profile_id": profile_id,
                    "case_id": str(row.get("case_id") or ""),
                    "family": family,
                    "prompt_surface": prompt_surface,
                    "prompt": str(row.get("prompt") or ""),
                    "status": str(exc),
                    "bundle_valid": False,
                    "dsl_emitted": False,
                    "compiled": False,
                    "svg_exact": False,
                    "bundle_path": str(bundle_path),
                    "scene_dsl_path": str(scene_dsl_path),
                    "svg_path": str(svg_path),
                    "rects": 0,
                    "paths": 0,
                }
            )

    compiled_count = sum(1 for case in cases if case.get("compiled"))
    svg_exact_count = sum(1 for case in cases if case.get("svg_exact"))
    return {
        "schema": "ck.spec19_compiler_smoke.v1",
        "run": str(run_dir),
        "prefix": prefix,
        "output_dir": str(out_dir),
        "count": len(cases),
        "compiled_count": compiled_count,
        "svg_exact_count": svg_exact_count,
        "families": sorted({str(case.get("family") or "") for case in cases if str(case.get("family") or "")}),
        "cases": cases,
        "pass": compiled_count == len(cases) and svg_exact_count == len(cases),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--prefix", default="spec19_scene_bundle")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--json-out", required=True, type=Path)
    ap.add_argument("--html-out", required=True, type=Path)
    ap.add_argument("--max-per-family", type=int, default=3)
    args = ap.parse_args()

    run_dir = args.run.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    report = build_report(run_dir, str(args.prefix), out_dir, max_per_family=max(1, int(args.max_per_family)))
    json_out = args.json_out.expanduser().resolve()
    html_out = args.html_out.expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    html_out.write_text(_render_html(report, out_dir), encoding="utf-8")
    return 0 if bool(report.get("pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
