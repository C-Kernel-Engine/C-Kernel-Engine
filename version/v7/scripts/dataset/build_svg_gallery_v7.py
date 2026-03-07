#!/usr/bin/env python3
"""Generate a self-contained SVG gallery HTML for the dataset.

Usage:
    python3 build_svg_gallery_v7.py --workspace <spec-dir> [--output gallery.html]

Reads:
  manifests/asset_classification_manifest.json
  manifests/raw_assets_inventory.json
  normalized/**/*.svg  (only non -assets- variants)

Produces:
  A single self-contained HTML file with all SVGs embedded as data-URIs,
  filterable gallery grid, full-screen viewer, and Antsand branding.
"""
from __future__ import annotations
import argparse, base64, json, os, sys, html as _html
from pathlib import Path


def _load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def _read_svg(p: str) -> str | None:
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def _svg_to_data_uri(svg_text: str) -> str:
    b = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b}"


def _basename(p: str) -> str:
    return os.path.basename(p)


def build_gallery(workspace: Path, output: Path) -> None:
    manifests = workspace / "manifests"
    class_path = manifests / "asset_classification_manifest.json"
    raw_path = manifests / "raw_assets_inventory.json"

    if not class_path.exists():
        print(f"ERROR: {class_path} not found", file=sys.stderr)
        sys.exit(1)

    classified = _load_json(class_path)
    raw_inv = _load_json(raw_path) if raw_path.exists() else {"entries": []}

    # Build raw lookup by source_path
    raw_by_source = {e["source_path"]: e for e in raw_inv.get("entries", [])}

    entries = classified.get("entries", [])
    # Collect family/source counts
    families = {}
    sources = {}
    size_bands = {}

    gallery_items = []
    for e in entries:
        svg_path = e.get("normalized_path", "")
        svg_text = _read_svg(svg_path)
        if not svg_text:
            continue

        family = e.get("family", "unknown")
        source = e.get("source_name", "unknown")
        band = e.get("size_band", "unknown")
        families[family] = families.get(family, 0) + 1
        sources[source] = sources.get(source, 0) + 1
        size_bands[band] = size_bands.get(band, 0) + 1

        features = e.get("features", {})
        active_features = [k for k, v in features.items() if v]
        roles = e.get("roles", [])
        tags = e.get("tag_counts", {})
        placeholders = e.get("placeholders", {})
        chars = e.get("chars", 0)

        raw_entry = raw_by_source.get(e.get("source_path", ""), {})
        raw_bytes = raw_entry.get("bytes", 0)

        name = _basename(e.get("source_path", svg_path))

        gallery_items.append({
            "name": name,
            "family": family,
            "source": source,
            "size_band": band,
            "chars": chars,
            "raw_bytes": raw_bytes,
            "features": active_features,
            "roles": roles,
            "tags": tags,
            "placeholders": placeholders,
            "data_uri": _svg_to_data_uri(svg_text),
            "sha": e.get("normalized_sha256", "")[:12],
        })

    total = len(gallery_items)
    total_kb = sum(g["chars"] for g in gallery_items) / 1024

    # Sort by family then name
    gallery_items.sort(key=lambda g: (g["family"], g["name"]))

    # Build JSON for embedding
    items_json = json.dumps(gallery_items, separators=(",", ":")).replace("</", "<\\/")
    families_json = json.dumps(families, separators=(",", ":"))
    sources_json = json.dumps(sources, separators=(",", ":"))
    bands_json = json.dumps(size_bands, separators=(",", ":"))

    html = _HTML_TEMPLATE.replace("__ITEMS_JSON__", items_json)
    html = html.replace("__FAMILIES_JSON__", families_json)
    html = html.replace("__SOURCES_JSON__", sources_json)
    html = html.replace("__BANDS_JSON__", bands_json)
    html = html.replace("__TOTAL__", str(total))
    html = html.replace("__TOTAL_KB__", f"{total_kb:.0f}")
    html = html.replace("__WORKSPACE__", str(workspace))
    html = html.replace("__FAMILY_COUNT__", str(len(families)))
    html = html.replace("__SOURCE_COUNT__", str(len(sources)))

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(html)
    print(f"Gallery written: {output}  ({total} SVGs, {total_kb:.0f} KB embedded)")


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SVG Dataset Gallery | CK-Engine v7</title>
<style>
/* ── Antsand Brand System ─────────────────────────────────────────────── */
:root {
    --orange: #ffb400;
    --orange-dark: #e5a200;
    --orange-glow: rgba(255,180,0,0.25);
    --dark: #2a2a2a;
    --dark-lighter: #363636;
    --dark-card: #1e1f23;
    --grey: #454545;
    --grey-light: #6b6b6b;
    --text-primary: #f0f0f0;
    --text-secondary: #a0a4b0;
    --text-muted: #6b7080;
    --bg-dark: #14161b;
    --bg-surface: #1a1c22;
    --white: #ffffff;
    --green: #47b475;
    --blue: #07adf8;
    --red: #f05050;
    --purple: #a07cf8;
    --radius: 16px;
    --radius-sm: 10px;
    --shadow-card: 0 8px 32px rgba(0,0,0,0.35);
    --shadow-hover: 0 16px 48px rgba(0,0,0,0.45);
    --font-display: 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    --backdrop: blur(12px) saturate(1.4);
}

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: var(--font-display);
    color: var(--text-primary);
    background:
        radial-gradient(ellipse at 15% 0%, rgba(255,180,0,0.08), transparent 45%),
        radial-gradient(ellipse at 85% 5%, rgba(7,173,248,0.06), transparent 40%),
        radial-gradient(ellipse at 50% 100%, rgba(71,180,117,0.04), transparent 35%),
        var(--bg-dark);
    min-height: 100vh;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
}

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--grey); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--grey-light); }

/* ── Header ───────────────────────────────────────────────────────────── */
.site-header {
    background: linear-gradient(135deg, #1a1c22 0%, #22252c 50%, #1a1c22 100%);
    border-bottom: 3px solid var(--orange);
    padding: 0.75rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: var(--backdrop);
}
.site-header .brand {
    display: flex; align-items: center; gap: 0.75rem;
    font-weight: 700; font-size: 1rem; color: var(--text-primary);
    text-decoration: none;
}
.site-header .brand .logo {
    width: 28px; height: 28px; border-radius: 6px;
    background: linear-gradient(135deg, var(--orange) 0%, var(--orange-dark) 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 800; color: var(--bg-dark);
}
.header-meta {
    display: flex; align-items: center; gap: 1rem;
    font-size: 0.78rem; color: var(--text-muted);
}
.header-meta .pill {
    background: rgba(255,180,0,0.12);
    border: 1px solid rgba(255,180,0,0.25);
    color: var(--orange);
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-family: var(--font-mono);
    font-weight: 600;
}

/* ── Hero ─────────────────────────────────────────────────────────────── */
.hero {
    padding: 2.5rem 2rem 1.8rem;
    max-width: 1400px; margin: 0 auto;
}
.hero h1 {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, var(--text-primary) 0%, var(--orange) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.5rem;
}
.hero .subtitle {
    font-size: 0.95rem; color: var(--text-secondary); max-width: 700px;
    line-height: 1.6;
}
.hero-stats {
    display: flex; flex-wrap: wrap; gap: 0.65rem; margin-top: 1.2rem;
}
.hero-stat {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: var(--radius-sm);
    padding: 0.8rem 1.2rem;
    backdrop-filter: var(--backdrop);
    min-width: 120px;
}
.hero-stat .label {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.8px;
    color: var(--text-muted); margin-bottom: 0.3rem;
}
.hero-stat .value {
    font-size: 1.4rem; font-weight: 800;
    font-family: var(--font-mono); color: var(--orange);
}
.hero-stat .value small {
    font-size: 0.65rem; color: var(--text-secondary); font-weight: 400;
}

/* ── Toolbar ──────────────────────────────────────────────────────────── */
.toolbar {
    max-width: 1400px; margin: 0 auto;
    padding: 0 2rem 1rem;
    display: flex; flex-wrap: wrap; gap: 0.7rem; align-items: center;
}
.search-box {
    flex: 1; min-width: 220px;
    background: var(--bg-surface);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: var(--radius-sm);
    padding: 0.55rem 1rem 0.55rem 2.4rem;
    color: var(--text-primary);
    font-size: 0.85rem;
    font-family: var(--font-display);
    outline: none;
    transition: border-color 0.2s;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%236b7080' viewBox='0 0 16 16'%3E%3Cpath d='M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85zm-5.242.156a5 5 0 1 1 0-10 5 5 0 0 1 0 10z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: 0.75rem center;
    background-size: 14px;
}
.search-box:focus { border-color: var(--orange); }
.search-box::placeholder { color: var(--text-muted); }

.filter-group {
    display: flex; gap: 0.35rem; align-items: center;
}
.filter-group label {
    font-size: 0.72rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-right: 0.15rem;
}
.filter-btn {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: var(--text-secondary);
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-family: var(--font-display);
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}
.filter-btn:hover { border-color: rgba(255,180,0,0.4); color: var(--text-primary); }
.filter-btn.active {
    background: rgba(255,180,0,0.15);
    border-color: rgba(255,180,0,0.5);
    color: var(--orange);
    font-weight: 600;
}
.filter-btn .count {
    font-family: var(--font-mono); font-size: 0.68rem;
    opacity: 0.7; margin-left: 0.25rem;
}

.view-toggle {
    display: flex; gap: 2px;
    background: rgba(255,255,255,0.04);
    border-radius: 8px; padding: 3px;
}
.view-toggle button {
    background: none; border: none; color: var(--text-muted);
    padding: 0.35rem 0.65rem; border-radius: 6px;
    cursor: pointer; font-size: 0.85rem; transition: all 0.2s;
}
.view-toggle button.active {
    background: rgba(255,180,0,0.15); color: var(--orange);
}

.result-count {
    font-size: 0.78rem; color: var(--text-muted);
    font-family: var(--font-mono);
    margin-left: auto;
}

/* ── Gallery Grid ─────────────────────────────────────────────────────── */
.gallery {
    max-width: 1400px; margin: 0 auto;
    padding: 0 2rem 3rem;
}

.gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 1rem;
}
.gallery-grid.view-large {
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
}
.gallery-grid.view-list {
    grid-template-columns: 1fr;
    gap: 0.5rem;
}

/* ── Gallery Card ─────────────────────────────────────────────────────── */
.svg-card {
    background: var(--bg-surface);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: var(--radius);
    overflow: hidden;
    cursor: pointer;
    transition: all 0.25s ease;
    position: relative;
}
.svg-card:hover {
    border-color: rgba(255,180,0,0.35);
    box-shadow: var(--shadow-hover);
    transform: translateY(-2px);
}
.svg-card .thumb {
    height: 180px;
    background: #ffffff;
    display: flex; align-items: center; justify-content: center;
    padding: 12px;
    overflow: hidden;
    position: relative;
}
.svg-card .thumb img {
    max-width: 100%; max-height: 100%;
    object-fit: contain;
}
.svg-card .thumb .family-badge {
    position: absolute; top: 8px; left: 8px;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.svg-card .thumb .source-badge {
    position: absolute; top: 8px; right: 8px;
    background: rgba(0,0,0,0.65);
    color: #fff;
    padding: 0.15rem 0.45rem;
    border-radius: 999px;
    font-size: 0.6rem;
    font-family: var(--font-mono);
    backdrop-filter: blur(4px);
}
.svg-card .meta {
    padding: 0.75rem 0.9rem;
}
.svg-card .meta .name {
    font-size: 0.82rem; font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    margin-bottom: 0.35rem;
}
.svg-card .meta .details {
    display: flex; gap: 0.5rem; flex-wrap: wrap;
    font-size: 0.68rem; color: var(--text-muted);
    font-family: var(--font-mono);
}
.svg-card .meta .feature-pill {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-size: 0.62rem;
    color: var(--text-secondary);
    margin-top: 0.3rem;
    margin-right: 0.25rem;
}

/* List view overrides */
.gallery-grid.view-list .svg-card {
    display: grid;
    grid-template-columns: 140px 1fr;
    border-radius: var(--radius-sm);
}
.gallery-grid.view-list .svg-card .thumb {
    height: 90px;
    border-radius: var(--radius-sm) 0 0 var(--radius-sm);
}
.gallery-grid.view-list .svg-card .meta {
    display: flex; flex-direction: column; justify-content: center;
}

/* Family badge colors */
.family-chart      { background: rgba(255,180,0,0.85); color: #1a1a1a; }
.family-flow       { background: rgba(7,173,248,0.85); color: #fff; }
.family-architecture { background: rgba(160,124,248,0.85); color: #fff; }
.family-technical  { background: rgba(71,180,117,0.85); color: #fff; }
.family-infographic { background: rgba(240,80,80,0.85); color: #fff; }
.family-other      { background: rgba(107,112,128,0.85); color: #fff; }

/* ── Full-Screen Viewer ───────────────────────────────────────────────── */
.viewer-overlay {
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.92);
    z-index: 9999;
    backdrop-filter: blur(8px);
}
.viewer-overlay.open { display: flex; flex-direction: column; }

.viewer-toolbar {
    padding: 0.6rem 1.5rem;
    background: rgba(0,0,0,0.8);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    display: flex; align-items: center; justify-content: space-between;
    flex-shrink: 0;
}
.viewer-toolbar .title {
    font-size: 0.88rem; font-weight: 600; color: var(--text-primary);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-width: 50%;
}
.viewer-toolbar .controls {
    display: flex; gap: 0.5rem; align-items: center;
}
.viewer-toolbar button {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    color: var(--orange);
    padding: 0.4rem 0.8rem;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.82rem;
    font-family: var(--font-mono);
    transition: all 0.15s;
}
.viewer-toolbar button:hover {
    background: rgba(255,180,0,0.15);
    border-color: var(--orange);
}
.viewer-toolbar .close-btn {
    font-size: 1.2rem; padding: 0.25rem 0.7rem; margin-left: 0.5rem;
}
.viewer-toolbar .zoom-label {
    font-size: 0.72rem; color: var(--text-muted);
    font-family: var(--font-mono); min-width: 40px; text-align: center;
}
.viewer-toolbar .nav-info {
    font-size: 0.72rem; color: var(--text-muted);
    font-family: var(--font-mono);
    margin-right: 0.5rem;
}

.viewer-body {
    flex: 1; display: flex; overflow: hidden;
}
.viewer-canvas {
    flex: 1; display: flex; align-items: center; justify-content: center;
    overflow: hidden; cursor: grab; position: relative;
}
.viewer-canvas:active { cursor: grabbing; }
.viewer-canvas img {
    max-width: 90%; max-height: 90%;
    transform-origin: center center;
    transition: transform 0.1s;
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}

.viewer-sidebar {
    width: 320px;
    background: var(--bg-surface);
    border-left: 1px solid rgba(255,255,255,0.06);
    overflow-y: auto;
    padding: 1.2rem;
    flex-shrink: 0;
}
.viewer-sidebar h3 {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px;
    color: var(--orange); margin-bottom: 0.6rem; font-weight: 700;
}
.viewer-sidebar .meta-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.8rem;
}
.viewer-sidebar .meta-row .key { color: var(--text-muted); }
.viewer-sidebar .meta-row .val {
    color: var(--text-primary); font-family: var(--font-mono);
    font-size: 0.78rem; text-align: right; max-width: 55%;
    word-break: break-all;
}
.viewer-sidebar .tag-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.25rem; margin-top: 0.3rem;
}
.viewer-sidebar .tag-item {
    display: flex; justify-content: space-between;
    font-size: 0.72rem; padding: 0.2rem 0.4rem;
    background: rgba(255,255,255,0.03);
    border-radius: 4px;
}
.viewer-sidebar .tag-item .tname { color: var(--text-secondary); }
.viewer-sidebar .tag-item .tcount {
    color: var(--orange); font-family: var(--font-mono);
}
.viewer-sidebar .pill-list {
    display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.4rem;
}
.viewer-sidebar .pill {
    background: rgba(255,180,0,0.12);
    border: 1px solid rgba(255,180,0,0.25);
    color: var(--orange);
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    font-size: 0.68rem;
    font-family: var(--font-mono);
}
.viewer-sidebar section { margin-bottom: 1.4rem; }

/* Navigation arrows */
.viewer-nav {
    position: absolute;
    top: 50%; transform: translateY(-50%);
    background: rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.1);
    color: var(--orange);
    font-size: 1.8rem;
    padding: 1rem 0.8rem;
    cursor: pointer;
    border-radius: 8px;
    z-index: 10;
    transition: all 0.15s;
}
.viewer-nav:hover {
    background: rgba(255,180,0,0.15);
    border-color: var(--orange);
}
.viewer-prev { left: 12px; }
.viewer-next { right: 12px; }

/* ── Empty State ──────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-muted);
}
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.5; }
.empty-state p { font-size: 0.9rem; }

/* ── Responsive ───────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .hero { padding: 1.5rem 1rem 1rem; }
    .hero h1 { font-size: 1.5rem; }
    .toolbar { padding: 0 1rem 0.8rem; }
    .gallery { padding: 0 1rem 2rem; }
    .gallery-grid { grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
    .viewer-sidebar { display: none; }
    .viewer-toolbar .title { max-width: 30%; }
}
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>

<!-- Header -->
<header class="site-header">
    <a class="brand" href="#">
        <div class="logo">CK</div>
        <span>SVG Dataset Gallery</span>
    </a>
    <div class="header-meta">
        <span class="pill">v7 / spec03</span>
        <span>CK-Engine Training Data</span>
    </div>
</header>

<!-- Hero -->
<section class="hero">
    <h1>Raw SVG Dataset</h1>
    <p class="subtitle">
        Visual browser for the complete normalized SVG training corpus.
        Every image in the dataset rendered at a glance — filter by family, source, or structural features.
    </p>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="label">Total Assets</div>
            <div class="value">__TOTAL__ <small>SVGs</small></div>
        </div>
        <div class="hero-stat">
            <div class="label">Corpus Size</div>
            <div class="value">__TOTAL_KB__ <small>KB</small></div>
        </div>
        <div class="hero-stat">
            <div class="label">Families</div>
            <div class="value">__FAMILY_COUNT__</div>
        </div>
        <div class="hero-stat">
            <div class="label">Sources</div>
            <div class="value">__SOURCE_COUNT__</div>
        </div>
    </div>
</section>

<!-- Toolbar -->
<div class="toolbar" id="toolbar">
    <input type="text" class="search-box" id="searchBox" placeholder="Search by filename, family, or feature…">
    <div class="filter-group" id="familyFilters">
        <label>Family</label>
    </div>
    <div class="filter-group" id="sourceFilters">
        <label>Source</label>
    </div>
    <div class="view-toggle">
        <button class="active" data-view="grid" title="Grid view">▦</button>
        <button data-view="large" title="Large grid">⊞</button>
        <button data-view="list" title="List view">☰</button>
    </div>
    <span class="result-count" id="resultCount"></span>
</div>

<!-- Gallery -->
<div class="gallery">
    <div class="gallery-grid" id="galleryGrid"></div>
    <div class="empty-state" id="emptyState" style="display:none;">
        <div class="icon">🔍</div>
        <p>No SVGs match your filters.</p>
    </div>
</div>

<!-- Full-Screen Viewer -->
<div class="viewer-overlay" id="viewerOverlay">
    <div class="viewer-toolbar">
        <span class="title" id="viewerTitle">—</span>
        <div class="controls">
            <span class="nav-info" id="viewerNavInfo"></span>
            <button onclick="viewer.zoomOut()">−</button>
            <span class="zoom-label" id="zoomLabel">100%</span>
            <button onclick="viewer.zoomIn()">+</button>
            <button onclick="viewer.resetZoom()">Reset</button>
            <button class="close-btn" onclick="viewer.close()">✕</button>
        </div>
    </div>
    <div class="viewer-body">
        <div class="viewer-canvas" id="viewerCanvas">
            <button class="viewer-nav viewer-prev" onclick="viewer.prev()">‹</button>
            <img id="viewerImg" src="" alt="">
            <button class="viewer-nav viewer-next" onclick="viewer.next()">›</button>
        </div>
        <div class="viewer-sidebar" id="viewerSidebar"></div>
    </div>
</div>

<script>
// ── Embedded Data ────────────────────────────────────────────────────────
const ALL_ITEMS = __ITEMS_JSON__;
const FAMILIES = __FAMILIES_JSON__;
const SOURCES  = __SOURCES_JSON__;
const BANDS    = __BANDS_JSON__;

// ── State ────────────────────────────────────────────────────────────────
let filtered = [...ALL_ITEMS];
let activeFamily = null;
let activeSource = null;
let searchQuery = '';

// ── Init ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    buildFilters();
    renderGallery();
    setupSearch();
    setupViewToggle();
    setupKeyboard();
});

function buildFilters() {
    const fGroup = document.getElementById('familyFilters');
    const sGroup = document.getElementById('sourceFilters');

    // Family pills
    const allBtn = mkBtn('All', ALL_ITEMS.length, () => { activeFamily = null; applyFilters(); });
    allBtn.classList.add('active');
    allBtn.id = 'famAll';
    fGroup.appendChild(allBtn);

    Object.entries(FAMILIES).sort((a,b) => b[1]-a[1]).forEach(([name, count]) => {
        const btn = mkBtn(name, count, () => {
            activeFamily = activeFamily === name ? null : name;
            applyFilters();
        });
        btn.dataset.family = name;
        fGroup.appendChild(btn);
    });

    // Source pills
    const sAllBtn = mkBtn('All', ALL_ITEMS.length, () => { activeSource = null; applyFilters(); });
    sAllBtn.classList.add('active');
    sAllBtn.id = 'srcAll';
    sGroup.appendChild(sAllBtn);

    Object.entries(SOURCES).sort((a,b) => b[1]-a[1]).forEach(([name, count]) => {
        const btn = mkBtn(name, count, () => {
            activeSource = activeSource === name ? null : name;
            applyFilters();
        });
        btn.dataset.source = name;
        sGroup.appendChild(btn);
    });
}

function mkBtn(label, count, onclick) {
    const btn = document.createElement('button');
    btn.className = 'filter-btn';
    btn.innerHTML = `${label}<span class="count">${count}</span>`;
    btn.onclick = onclick;
    return btn;
}

function applyFilters() {
    const q = searchQuery.toLowerCase();
    filtered = ALL_ITEMS.filter(item => {
        if (activeFamily && item.family !== activeFamily) return false;
        if (activeSource && item.source !== activeSource) return false;
        if (q) {
            const haystack = (item.name + ' ' + item.family + ' ' + item.source + ' ' + item.features.join(' ') + ' ' + item.roles.join(' ')).toLowerCase();
            if (!haystack.includes(q)) return false;
        }
        return true;
    });

    // Update active states on family buttons
    document.querySelectorAll('#familyFilters .filter-btn').forEach(b => {
        if (b.id === 'famAll') b.classList.toggle('active', !activeFamily);
        else b.classList.toggle('active', b.dataset.family === activeFamily);
    });
    document.querySelectorAll('#sourceFilters .filter-btn').forEach(b => {
        if (b.id === 'srcAll') b.classList.toggle('active', !activeSource);
        else b.classList.toggle('active', b.dataset.source === activeSource);
    });

    renderGallery();
}

function setupSearch() {
    const box = document.getElementById('searchBox');
    let timer;
    box.addEventListener('input', () => {
        clearTimeout(timer);
        timer = setTimeout(() => {
            searchQuery = box.value.trim();
            applyFilters();
        }, 200);
    });
}

function setupViewToggle() {
    document.querySelectorAll('.view-toggle button').forEach(btn => {
        btn.onclick = () => {
            document.querySelectorAll('.view-toggle button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const grid = document.getElementById('galleryGrid');
            grid.classList.remove('view-large', 'view-list');
            if (btn.dataset.view !== 'grid') grid.classList.add('view-' + btn.dataset.view);
        };
    });
}

// ── Render Gallery ───────────────────────────────────────────────────────
function renderGallery() {
    const grid = document.getElementById('galleryGrid');
    const empty = document.getElementById('emptyState');
    const count = document.getElementById('resultCount');

    count.textContent = `${filtered.length} of ${ALL_ITEMS.length}`;

    if (filtered.length === 0) {
        grid.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';

    // Build cards
    let html = '';
    filtered.forEach((item, i) => {
        const familyClass = 'family-' + item.family;
        const featurePills = item.features.slice(0, 3).map(f =>
            `<span class="feature-pill">${f}</span>`
        ).join('');

        html += `
        <div class="svg-card" onclick="viewer.open(${i})" title="${esc(item.name)}">
            <div class="thumb">
                <img src="${item.data_uri}" alt="${esc(item.name)}" loading="lazy">
                <span class="family-badge ${familyClass}">${item.family}</span>
                <span class="source-badge">${item.source}</span>
            </div>
            <div class="meta">
                <div class="name">${esc(item.name)}</div>
                <div class="details">
                    <span>${formatSize(item.chars)}</span>
                    <span>·</span>
                    <span>${item.size_band}</span>
                </div>
                <div>${featurePills}</div>
            </div>
        </div>`;
    });
    grid.innerHTML = html;
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function formatSize(chars) {
    if (chars < 1024) return chars + ' B';
    return (chars / 1024).toFixed(1) + ' KB';
}

// ── Full-Screen Viewer ───────────────────────────────────────────────────
const viewer = {
    idx: 0,
    zoom: 1,
    panX: 0, panY: 0,
    dragging: false,
    dragStartX: 0, dragStartY: 0,
    panStartX: 0, panStartY: 0,

    open(i) {
        this.idx = i;
        this.zoom = 1; this.panX = 0; this.panY = 0;
        this.show();
        document.getElementById('viewerOverlay').classList.add('open');
    },
    close() {
        document.getElementById('viewerOverlay').classList.remove('open');
    },
    show() {
        const item = filtered[this.idx];
        if (!item) return;
        document.getElementById('viewerTitle').textContent = item.name;
        document.getElementById('viewerImg').src = item.data_uri;
        document.getElementById('viewerNavInfo').textContent = `${this.idx + 1} / ${filtered.length}`;
        this.updateTransform();
        this.renderSidebar(item);
    },
    prev() {
        if (this.idx > 0) { this.idx--; this.zoom = 1; this.panX = 0; this.panY = 0; this.show(); }
    },
    next() {
        if (this.idx < filtered.length - 1) { this.idx++; this.zoom = 1; this.panX = 0; this.panY = 0; this.show(); }
    },
    zoomIn()  { this.zoom = Math.min(8, this.zoom * 1.3); this.updateTransform(); },
    zoomOut() { this.zoom = Math.max(0.2, this.zoom / 1.3); this.updateTransform(); },
    resetZoom() { this.zoom = 1; this.panX = 0; this.panY = 0; this.updateTransform(); },
    updateTransform() {
        const img = document.getElementById('viewerImg');
        img.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
        document.getElementById('zoomLabel').textContent = Math.round(this.zoom * 100) + '%';
    },
    renderSidebar(item) {
        let html = '';

        // Classification
        html += `<section>
            <h3>Classification</h3>
            <div class="meta-row"><span class="key">Family</span><span class="val">${item.family}</span></div>
            <div class="meta-row"><span class="key">Source</span><span class="val">${item.source}</span></div>
            <div class="meta-row"><span class="key">Size Band</span><span class="val">${item.size_band}</span></div>
            <div class="meta-row"><span class="key">Characters</span><span class="val">${item.chars.toLocaleString()}</span></div>
            <div class="meta-row"><span class="key">Raw Bytes</span><span class="val">${item.raw_bytes.toLocaleString()}</span></div>
            <div class="meta-row"><span class="key">SHA-256</span><span class="val">${item.sha}</span></div>
        </section>`;

        // Features & Roles
        if (item.features.length || item.roles.length) {
            html += `<section><h3>Features & Roles</h3><div class="pill-list">`;
            item.features.forEach(f => { html += `<span class="pill">${f}</span>`; });
            item.roles.forEach(r => { html += `<span class="pill">${r}</span>`; });
            html += `</div></section>`;
        }

        // Tag Counts
        const tags = Object.entries(item.tags || {}).filter(([,v]) => v > 0).sort((a,b) => b[1]-a[1]);
        if (tags.length) {
            html += `<section><h3>SVG Tag Counts</h3><div class="tag-grid">`;
            tags.forEach(([name, count]) => {
                html += `<div class="tag-item"><span class="tname">&lt;${name}&gt;</span><span class="tcount">${count}</span></div>`;
            });
            html += `</div></section>`;
        }

        // Placeholders
        const ph = Object.entries(item.placeholders || {}).filter(([,v]) => v > 0);
        if (ph.length) {
            html += `<section><h3>Placeholders</h3><div class="tag-grid">`;
            ph.forEach(([name, count]) => {
                html += `<div class="tag-item"><span class="tname">${name}</span><span class="tcount">${count}</span></div>`;
            });
            html += `</div></section>`;
        }

        document.getElementById('viewerSidebar').innerHTML = html;
    }
};

// ── Keyboard & Mouse ─────────────────────────────────────────────────────
function setupKeyboard() {
    document.addEventListener('keydown', e => {
        const overlay = document.getElementById('viewerOverlay');
        if (!overlay.classList.contains('open')) return;
        switch(e.key) {
            case 'Escape': viewer.close(); break;
            case 'ArrowLeft': viewer.prev(); break;
            case 'ArrowRight': viewer.next(); break;
            case '+': case '=': viewer.zoomIn(); break;
            case '-': viewer.zoomOut(); break;
            case '0': viewer.resetZoom(); break;
        }
    });

    // Pan & zoom on canvas
    const canvas = document.getElementById('viewerCanvas');
    canvas.addEventListener('wheel', e => {
        e.preventDefault();
        viewer.zoom = Math.max(0.2, Math.min(8, viewer.zoom + (e.deltaY > 0 ? -0.15 : 0.15)));
        viewer.updateTransform();
    });
    canvas.addEventListener('mousedown', e => {
        if (e.target.tagName === 'BUTTON') return;
        viewer.dragging = true;
        viewer.dragStartX = e.clientX;
        viewer.dragStartY = e.clientY;
        viewer.panStartX = viewer.panX;
        viewer.panStartY = viewer.panY;
    });
    document.addEventListener('mousemove', e => {
        if (!viewer.dragging) return;
        viewer.panX = viewer.panStartX + (e.clientX - viewer.dragStartX);
        viewer.panY = viewer.panStartY + (e.clientY - viewer.dragStartY);
        viewer.updateTransform();
    });
    document.addEventListener('mouseup', () => { viewer.dragging = false; });
}
</script>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description="Build SVG dataset gallery")
    ap.add_argument("--workspace", required=True, help="spec workspace directory")
    ap.add_argument("--output", default=None, help="output HTML path (default: <workspace>/svg_gallery.html)")
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    out = Path(args.output) if args.output else ws / "svg_gallery.html"
    build_gallery(ws, out)


if __name__ == "__main__":
    main()
