#!/usr/bin/env python3
"""Build a self-contained interactive SVG dataset viewer from split-aware workspaces.

Uses the Antsand/IR-Visualizer brand system. All data is embedded as JSON so the
viewer works offline with full search, sort, gallery, quality checks, and provenance.

┌──────────────────────────────────────────────────────────────────────────────┐
│  ⚠️  FEATURE MANIFEST — DO NOT DELETE OR REGRESS ANY OF THESE FEATURES  ⚠️   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  This file generates the run-local Dataset Viewer HTML. Every feature below  │
│  is operator-critical and was deliberately built for sprint demos + ongoing   │
│  research. Future agents: you may ENHANCE any of these, but NEVER remove     │
│  or break them. If you must refactor, ensure every feature still works.      │
│                                                                              │
│  TABS (12 tabs, each with its own panel and renderer):                       │
│   1. Overview        — renderOverview()       — run summary, split stats     │
│   2. Preflight       — renderPreflight()      — data quality checks          │
│   3. SVG Gallery     — renderGallery()        — visual grid of SVG assets    │
│   4. Text Samples    — renderTextSamples()    — sortable per-row text view   │
│   5. Tokenizer       — renderTokenizer()      — corpus stats, prompt atoms,  │
│                        renderTokenizerVocabOnly() (fallback when no corpus)   │
│                        NEVER delete the tokenizer table — it shows the        │
│                        operator what tokens the model was trained on.         │
│   6. Vocabulary      — renderVocabulary()     — vocab frequency analysis     │
│   7. Classification  — renderClassification() — asset family breakdown       │
│   8. Browse          — renderBrowse()         — paginated raw data browser   │
│   9. Candidates      — renderCandidates()     — training candidate review    │
│  10. Quality         — renderQuality()        — data quality audit           │
│  11. 🧬 Embeddings   — renderEmbeddings()     — dense embedding heatmap,     │
│                        cosine similarity, group legend, sort/norm controls    │
│  12. 🔭 Attention    — renderAttention()      — per-head/per-layer attention │
│                        matrices, sequence picker, entropy analysis            │
│                                                                              │
│  CROSS-CUTTING FEATURES:                                                     │
│   • CKTable reusable ES6 class (sort, search, pagination, column resize)     │
│   • Search/filter bar with family + role dropdowns                           │
│   • Column resize (initColumnResize) on data tables                          │
│   • Collapsible section-cards with badges                                    │
│   • Distribution bar charts (distBarsHtml)                                   │
│   • Pagination (renderPagination)                                            │
│   • Dark theme with CSS variables (--bg, --surface, --border, etc.)          │
│   • Gallery overlay with SVG preview                                         │
│   • Text Samples DataTable with resize + sort                                │
│   • Tokenizer fallback: reads vocab from run-level tokenizer.json            │
│     when the corpus isn't staged (vocab_source='run_tokenizer_json')         │
│   • Embeddings: auto-embeds if ≤1MB, file picker fallback if larger          │
│   • Attention: auto-embeds if ≤1MB, file picker fallback if larger           │
│                                                                              │
│  PYTHON DATA BUILDERS:                                                       │
│   • _build_text_rows()       — per-split structured row data                 │
│   • _build_tokenizer_info()  — corpus + vocab + protected tokens + drift     │
│   • _parse_svg_row_meta()    — extracts prompt tags, EOS, coords from rows   │
│   • _iter_workspace_text_files() — recursive split-aware file discovery      │
│   • _prompt_tag_family()     — classifies prompt tags into families           │
│                                                                              │
│  If you're adding a new tab or feature, follow the existing pattern:         │
│   1. Add a <button class="tab"> and <div class="panel"> in _HTML_SUFFIX      │
│   2. Write a renderXxx() function                                            │
│   3. Call it from renderAll()                                                │
│   4. Update this manifest                                                    │
│   5. Update version/v8/tests/contracts/dataset_viewer_contract.json:         │
│      — Add tab entry with id, label, render_function, panel_id              │
│      — Add render function to required_functions list                        │
│   6. If adding a pure utility function, also add to:                         │
│      version/v8/tests/fixtures/ds_pure_functions.js (with module.exports)    │
│      + add test vectors to the contract JSON                                 │
│   7. Run: make v8-visualizer-health  (must pass before pushing)              │
│                                                                              │
│  WHAT NOT TO BREAK:                                                          │
│   • attnColor — was missing before, caused ReferenceError (the original bug) │
│   • embColor, embNormalise, cosineSim — L2-tested pure functions             │
│   • renderAll — entry point, wraps each section in try/catch                 │
│   • esc() — HTML escaping, used in every render function                     │
│   • CKTable — reusable sort/search/pagination class, used by 4+ tabs        │
│   • Null guards: every render function must handle missing/null data safely  │
│                                                                              │
│  TESTING:                                                                    │
│   make v8-visualizer-health          # L1 static + L2 JS units (~3s)        │
│   make v8-visualizer-generated-e2e   # L3 generate + validate (~10s)        │
│   Contracts: version/v8/tests/contracts/dataset_viewer_contract.json        │
│   Fixtures:  version/v8/tests/fixtures/ds_pure_functions.js                 │
│   Pre-push:  .githooks/pre-push step [0.5/6] runs health checks            │
│   Nightly:   nightly_runner.py runs v8_visualizer_health + generated_e2e    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
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


def _synthesize_structured_atoms(workspace: Path) -> tuple[dict, dict]:
    """Detect structured-atoms workspace and synthesize normalized/classified equivalents.

    Structured-atoms workspaces (spec06+) store training data as DSL text in
    render_catalog.json rather than raw SVG files.  This function extracts
    vocabulary (DSL tag frequency) and classification (layout/topic families)
    data so the DV tabs render meaningful content.

    Returns (normalized_equiv, classified_equiv) — empty dicts if not applicable.
    """
    # Find the generated manifest with render catalog
    gen_dir = workspace / "manifests" / "generated" / "structured_atoms"
    if not gen_dir.is_dir():
        return {}, {}

    catalog_files = list(gen_dir.glob("*_render_catalog.json"))
    if not catalog_files:
        return {}, {}

    catalog = _load_json_if_exists(catalog_files[0])
    if not catalog or not isinstance(catalog, list):
        return {}, {}

    # --- Synthesize normalized-like data (tag_totals, placeholder_totals) ---
    tag_totals: dict[str, int] = {}
    placeholder_totals: dict[str, int] = {}
    for entry in catalog:
        out = entry.get("output_tokens", "")
        # Count DSL tag families: [tag:value] → tag as tag_totals, value as placeholder
        for m in re.finditer(r"\[(\w+):([^\]]*)\]", out):
            tag_name = m.group(1)
            tag_value = m.group(2).strip()
            tag_totals[tag_name] = tag_totals.get(tag_name, 0) + 1
            if tag_value:
                placeholder_totals[tag_value] = placeholder_totals.get(tag_value, 0) + 1
        # Count SVG tags if svg_xml present
        svg_xml = entry.get("svg_xml", "")
        if svg_xml:
            for m in re.finditer(r"<(\w+)[\s/>]", svg_xml):
                svg_tag = m.group(1).lower()
                tag_totals[svg_tag] = tag_totals.get(svg_tag, 0) + 1

    normalized = {
        "schema": "ck.structured_atoms_synthesized.v1",
        "normalized_entries": len(catalog),
        "unique_normalized_hashes": len(catalog),
        "duplicate_normalized_entries": 0,
        "tag_totals": dict(sorted(tag_totals.items(), key=lambda kv: -kv[1])),
        "placeholder_totals": placeholder_totals,
        "normalized_root": str(gen_dir),
    }

    # --- Synthesize classified-like data (entries with family, roles) ---
    # Human-readable descriptions for DSL layout families
    layout_descriptions = {
        "bullet-panel": "Hero title + bullet list + callout — classic infographic card",
        "compare-panels": "Side-by-side comparison panels — A vs B layouts",
        "flow-steps": "Sequential step flow — process/pipeline diagrams",
        "spectrum-band": "Gradient spectrum with labeled positions — range/scale visuals",
        "stat-cards": "Grid of metric cards — KPI dashboard style",
        "timeline": "Chronological event sequence",
        "hierarchy": "Tree/org-chart structure",
    }
    topic_descriptions = {
        "capacity_math": "GPU/compute capacity calculations and sizing",
        "eval_discipline": "Model evaluation methodology and metrics",
        "governance_path": "Policy, compliance, and governance workflows",
        "gpu_readiness": "Hardware readiness checks and provisioning",
        "platform_rollout": "Platform deployment and launch planning",
        "structured_outputs": "Structured output formats and schemas",
    }

    entries = []
    for i, entry in enumerate(catalog):
        layout = entry.get("layout", "unknown")
        topic = entry.get("topic", "unknown")
        split = entry.get("split", "train")
        prompt = entry.get("prompt", "")
        out_tokens = entry.get("output_tokens", "")

        # Extract DSL tag counts from output
        entry_tags: dict[str, int] = {}
        for m in re.finditer(r"<(\w+)[\s/>]", entry.get("svg_xml", "")):
            t = m.group(1).lower()
            entry_tags[t] = entry_tags.get(t, 0) + 1

        chars = len(entry.get("svg_xml", "") or out_tokens)
        entries.append({
            "source_path": f"{layout}_{topic}_{i:04d}",
            "normalized_path": "",
            "family": layout,
            "source_name": topic,
            "split": split,
            "chars": chars,
            "size_band": "small" if chars < 500 else "medium" if chars < 2000 else "large",
            "roles": [topic],
            "features": {"has_text": True, "has_color": True},
            "tag_counts": entry_tags,
            "placeholders": {},
            "normalized_sha256": "",
        })

    classified = {
        "schema": "ck.structured_atoms_classification.v1",
        "entries": entries,
        "counts": {
            "total": len(entries),
            "by_family": {},
            "by_source": {},
        },
        "family_counts": {},
        "family_descriptions": {k: v for k, v in layout_descriptions.items()
                                if k in {e["family"] for e in entries}},
        "source_descriptions": {k: v for k, v in topic_descriptions.items()
                                if k in {e["source_name"] for e in entries}},
    }
    # Compute by_family / by_source counts
    for e in entries:
        f = e["family"]
        s = e["source_name"]
        classified["counts"]["by_family"][f] = classified["counts"]["by_family"].get(f, 0) + 1
        classified["counts"]["by_source"][s] = classified["counts"]["by_source"].get(s, 0) + 1
        classified["family_counts"][f] = classified["family_counts"].get(f, 0) + 1

    return normalized, classified


def _build_pipeline_map(workspace: Path, raw_inventory: dict, normalized: dict,
                        classified: dict, training_data: dict | None,
                        emb_data: dict | None, attn_data: dict | None) -> list[dict]:
    """Build a script → artifact → tab mapping with live status.

    Each entry tells the operator:
      - which tab uses this artifact
      - which script generates it
      - whether the artifact is present, synthesized, or missing
      - the command to generate it if missing
    """
    run_dir = _resolve_run_dir(workspace)
    if not run_dir:
        candidate = workspace.parent
        if (candidate / "weights_manifest.json").exists():
            run_dir = candidate
    run_str = str(run_dir) if run_dir else str(workspace.parent)
    ws_str = str(workspace)
    has_run_brief = bool(run_dir and any((run_dir / name).exists() for name in ("run_scope.json", "agent.md", "training.md")))

    is_synth = normalized.get("schema", "").startswith("ck.structured_atoms_synthesized")

    entries = [
        {
            "tab": "Vocabulary",
            "artifact": "normalized_assets_manifest.json",
            "script": "normalize_svg_assets_v7.py",
            "status": "synthesized" if is_synth else ("present" if normalized.get("normalized_entries") else "missing"),
            "detail": (f"Synthesized from render catalog ({normalized.get('normalized_entries', 0)} entries)"
                       if is_synth
                       else f"{normalized.get('normalized_entries', 0)} entries" if normalized.get("normalized_entries")
                       else "Trained without SVG normalization step"),
            "command": f"python3 version/v7/scripts/dataset/normalize_svg_assets_v7.py --workspace {ws_str}",
        },
        {
            "tab": "Classification",
            "artifact": "asset_classification_manifest.json",
            "script": "classify_svg_assets_v7.py",
            "status": "synthesized" if (is_synth and classified.get("entries")) else ("present" if classified.get("entries") else "missing"),
            "detail": (f"Synthesized from render catalog ({len(classified.get('entries', []))} entries)"
                       if is_synth
                       else f"{len(classified.get('entries', []))} entries" if classified.get("entries")
                       else "Trained without asset classification step"),
            "command": f"python3 version/v7/scripts/dataset/classify_svg_assets_v7.py --workspace {ws_str}",
        },
        {
            "tab": "Gallery",
            "artifact": "asset_classification_manifest.json + SVG files",
            "script": "classify_svg_assets_v7.py",
            "status": "present" if classified.get("entries") and not is_synth else ("na" if is_synth else "missing"),
            "detail": ("N/A — structured-atoms workspace uses DSL text, not rendered SVG files"
                       if is_synth
                       else f"{len(classified.get('entries', []))} assets" if classified.get("entries")
                       else "No classified SVG assets found"),
            "command": f"python3 version/v7/scripts/dataset/classify_svg_assets_v7.py --workspace {ws_str}",
        },
        {
            "tab": "Training",
            "artifact": "training_loss_curve.json",
            "script": "train_e2e_v7.py (generated during training)",
            "status": "present" if (training_data and training_data.get("available")) else "missing",
            "detail": (f"{training_data.get('summary', {}).get('total_steps', 0)} steps recorded"
                       if training_data and training_data.get("available")
                       else "No training curve recorded — re-run training with loss logging enabled"),
            "command": f"python3 version/v7/scripts/train_e2e_v7.py --run-dir {run_str}",
        },
        {
            "tab": "Overview",
            "artifact": "run_scope.json + agent.md + training.md",
            "script": "init_run_scope_v7.py",
            "status": "present" if has_run_brief else "missing",
            "detail": ("Run-local brief files are present"
                       if has_run_brief
                       else "Missing run-local brief files for future agents/operators"),
            "command": f"python3 version/v7/scripts/init_run_scope_v7.py --run {run_str}",
        },
        {
            "tab": "Embeddings",
            "artifact": "embeddings.json",
            "script": "export_embeddings.py",
            "status": "present" if emb_data else "missing",
            "detail": (f"{emb_data.get('num_tokens', '?')} tokens × {emb_data.get('embed_dim', '?')} dims"
                       if emb_data
                       else "Export token embeddings from trained weights"),
            "command": f"python3 version/v7/tools/export_embeddings.py {run_str}",
        },
        {
            "tab": "Attention",
            "artifact": "attention.json",
            "script": "export_attention.py",
            "status": "present" if attn_data else "missing",
            "detail": (f"{attn_data.get('num_sequences', '?')} sequences, {attn_data.get('num_layers', '?')} layers"
                       if attn_data
                       else "Export attention matrices from probe sequences"),
            "command": f"python3 version/v7/tools/export_attention.py {run_str} --probe",
        },
        {
            "tab": "Quality",
            "artifact": "normalized_assets_manifest.json",
            "script": "normalize_svg_assets_v7.py",
            "status": "synthesized" if is_synth else ("present" if normalized.get("normalized_entries") else "missing"),
            "detail": (f"Synthesized from render catalog"
                       if is_synth
                       else f"{normalized.get('normalized_entries', 0)} entries" if normalized.get("normalized_entries")
                       else "Requires normalization corpus"),
            "command": f"python3 version/v7/scripts/dataset/normalize_svg_assets_v7.py --workspace {ws_str}",
        },
    ]
    return entries


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


def _iter_workspace_text_files(stage_dir: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for txt_file in sorted(stage_dir.rglob("*.txt")):
        if not txt_file.is_file() or txt_file in seen:
            continue
        seen.add(txt_file)
        rel_parent = txt_file.parent.relative_to(stage_dir)
        split_label = stage_dir.name if str(rel_parent) == "." else f"{stage_dir.name}/{rel_parent.as_posix()}"
        files.append((split_label, txt_file))
    return files


def _build_text_rows(workspace: Path) -> dict[str, Any]:
    """Build per-row structured data for each split, HuggingFace-style."""
    splits: dict[str, Any] = {}
    for stage_name in ("pretrain", "midtrain", "sft", "holdout", "tokenizer"):
        split_dir = workspace / stage_name
        if not split_dir.exists():
            continue
        for split_name, txt_file in _iter_workspace_text_files(split_dir):
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
            rel_path = txt_file.relative_to(split_dir)
            splits[f"{stage_name}/{rel_path.as_posix()}"] = {
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
            if any(token in cf.name for token in ("tag_seed", "seen_prompts", "holdout_prompts", "canary_prompts")):
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

    # ── Fallback: read run-level tokenizer.json vocab when corpus isn't staged ──
    # IMPORTANT: The tokenizer table is critical for operators to understand what
    # tokens the model was trained on. Never remove this fallback — only enhance it.
    # Even when the full corpus isn't available, showing the vocabulary gives the
    # operator visibility into tokenization.
    snapshot = _load_json_if_exists(workspace / "dataset_snapshot.json") or {}
    run_dir_text = snapshot.get("run_dir")
    run_dir = Path(str(run_dir_text)).expanduser() if run_dir_text else None

    # Also try workspace parent as run_dir (workspace is often run_dir/dataset)
    if not run_dir or not run_dir.exists():
        candidate = workspace.parent
        if (candidate / "tokenizer.json").exists() or (candidate / "weights_manifest.json").exists():
            run_dir = candidate

    if run_dir and run_dir.exists():
        tokenizer_json = _load_json_if_exists(run_dir / "tokenizer.json") or {}

        # Extract vocab from the tokenizer file when we don't already have corpus data
        if isinstance(tokenizer_json, dict) and not info.get("available"):
            model_block = tokenizer_json.get("model", {})
            vocab = model_block.get("vocab") if isinstance(model_block, dict) else None
            if not vocab and isinstance(tokenizer_json.get("vocab"), dict):
                vocab = tokenizer_json["vocab"]
            if isinstance(vocab, dict) and vocab:
                info["available"] = True
                info["vocab_source"] = "run_tokenizer_json"
                info["vocab_size"] = len(vocab)
                info["ck_mode"] = tokenizer_json.get("ck_mode", "unknown")
                # Build a compact vocab table: [{token, id, length, family}]
                vocab_table = []
                for token_str, token_id in sorted(vocab.items(), key=lambda x: x[1]):
                    vocab_table.append({
                        "token": token_str,
                        "id": token_id,
                        "len": len(token_str),
                        "family": _prompt_tag_family(token_str),
                    })
                info["vocab_table"] = vocab_table
                # Count by family
                family_counts: dict[str, int] = {}
                for row in vocab_table:
                    fam = row["family"]
                    family_counts[fam] = family_counts.get(fam, 0) + 1
                info["vocab_family_counts"] = family_counts

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
        reserved_tokens_path = None
        for candidate in sorted(manifest_path.glob("*reserved_control_tokens*.txt")):
            reserved_tokens_path = candidate
            break
        protected_tokens: list[dict[str, Any]] = []
        if reserved_tokens_path and reserved_tokens_path.exists():
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


def _downsample_steps(steps: list[dict], max_pts: int = 500) -> list[dict]:
    """Downsample a time-series to max_pts keeping first, last, and evenly spaced points."""
    n = len(steps)
    if n <= max_pts:
        return steps
    indices = set()
    indices.add(0)
    indices.add(n - 1)
    stride = (n - 1) / (max_pts - 1)
    for i in range(max_pts):
        indices.add(min(int(i * stride), n - 1))
    return [steps[i] for i in sorted(indices)]


def _load_training_data(run_dir: Path) -> dict[str, Any] | None:
    """Load training telemetry: loss curve, grad norms, step profile.

    Downsamples to ~500 points to keep embedded HTML reasonable (~200KB).
    Returns structured dict with 'loss_curve', 'grad_norms', 'step_profile', 'summary'.
    """
    if not run_dir or not run_dir.exists():
        return None

    result: dict[str, Any] = {"available": False}

    # Loss curve (the richest source — has loss, lr, grad_norm, timing per step)
    for name in ("training_loss_curve_latest.json", "training_loss_curve.json"):
        lc_path = run_dir / name
        if lc_path.exists():
            try:
                lc = json.loads(lc_path.read_text(encoding="utf-8"))
                steps = lc.get("steps", [])
                if steps:
                    ds = _downsample_steps(steps)
                    # Extract only fields we need to keep size small
                    compact = []
                    for s in ds:
                        compact.append({
                            "step": s.get("step"),
                            "loss_ck": s.get("loss_ck"),
                            "loss_pt": s.get("loss_pt"),
                            "lr": s.get("lr"),
                            "grad_norm": s.get("grad_norm"),
                            "epoch": s.get("epoch"),
                            "source_stage": s.get("source_stage"),
                            "forward_ms": s.get("forward_ms"),
                            "backward_ms": s.get("backward_ms"),
                            "optimizer_ms": s.get("optimizer_ms"),
                        })
                    result["loss_curve"] = compact
                    result["total_steps"] = len(steps)
                    result["available"] = True
                    # Summary metrics
                    first = steps[0]
                    last = steps[-1]
                    result["summary"] = {
                        "start_loss": first.get("loss_ck"),
                        "final_loss": last.get("loss_ck"),
                        "final_loss_pt": last.get("loss_pt"),
                        "total_steps": len(steps),
                        "final_lr": last.get("lr"),
                        "source": lc.get("source"),
                    }
            except Exception:
                pass
            break

    # Grad norms (separate file, may have per-param detail)
    for name in ("training_grad_norms.json",):
        gn_path = run_dir / name
        if gn_path.exists():
            try:
                gn = json.loads(gn_path.read_text(encoding="utf-8"))
                gn_steps = gn.get("steps", [])
                gn_global = gn.get("global", [])
                if gn_steps and gn_global and len(gn_steps) == len(gn_global):
                    ds_idx = set()
                    stride = max(1, len(gn_steps) // 500)
                    for i in range(0, len(gn_steps), stride):
                        ds_idx.add(i)
                    ds_idx.add(len(gn_steps) - 1)
                    result["grad_norms"] = [
                        {"step": gn_steps[i], "norm": gn_global[i]}
                        for i in sorted(ds_idx)
                    ]
                    result["available"] = True
            except Exception:
                pass

    # Step profile (timing per step)
    for name in ("training_step_profile.json",):
        sp_path = run_dir / name
        if sp_path.exists():
            try:
                sp = json.loads(sp_path.read_text(encoding="utf-8"))
                sp_steps = sp.get("steps", [])
                ck_ms = sp.get("ck_total_ms", [])
                pt_ms = sp.get("torch_total_ms", [])
                tok_s = sp.get("train_tok_s", [])
                if sp_steps:
                    n = len(sp_steps)
                    ds_idx = set()
                    stride = max(1, n // 500)
                    for i in range(0, n, stride):
                        ds_idx.add(i)
                    ds_idx.add(n - 1)
                    profile = []
                    for i in sorted(ds_idx):
                        profile.append({
                            "step": sp_steps[i],
                            "ck_ms": ck_ms[i] if i < len(ck_ms) else None,
                            "pt_ms": pt_ms[i] if i < len(pt_ms) else None,
                            "tok_s": tok_s[i] if i < len(tok_s) else None,
                        })
                    result["step_profile"] = profile
                    result["available"] = True
            except Exception:
                pass

    return result if result["available"] else None


def _load_embeddings_data(run_dir: Path) -> dict[str, Any] | None:
    """Load embeddings.json if it exists and is ≤ 1MB, otherwise return None."""
    if not run_dir or not run_dir.exists():
        return None
    
    emb_path = run_dir / "embeddings.json"
    if not emb_path.exists():
        return None
        
    # Check file size (≤ 1MB)
    try:
        if emb_path.stat().st_size > 1024 * 1024:
            return None
        return _load_json_if_exists(emb_path)
    except Exception:
        return None


def _load_attention_data(run_dir: Path) -> dict[str, Any] | None:
    """Load attention.json if it exists and is ≤ 1MB, otherwise return None."""
    if not run_dir or not run_dir.exists():
        return None
    
    attn_path = run_dir / "attention.json"
    if not attn_path.exists():
        return None
        
    # Check file size (≤ 1MB)
    try:
        if attn_path.stat().st_size > 1024 * 1024:
            return None
        return _load_json_if_exists(attn_path)
    except Exception:
        return None


def _load_run_scope_bundle(run_dir: Path | None) -> dict[str, Any]:
    bundle: dict[str, Any] = {"available": False, "scope": {}, "agent_md": "", "training_md": ""}
    if not run_dir or not run_dir.exists():
        return bundle

    scope = _load_json_if_exists(run_dir / "run_scope.json")
    if not isinstance(scope, dict):
        training_plan = _load_json_if_exists(run_dir / "training_plan.json")
        if isinstance(training_plan, dict) and isinstance(training_plan.get("run_scope"), dict):
            scope = training_plan.get("run_scope")
    if isinstance(scope, dict) and scope:
        bundle["scope"] = scope
        bundle["available"] = True

    for key, name in (("agent_md", "agent.md"), ("training_md", "training.md")):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if text:
            bundle[key] = text
            bundle["available"] = True
    return bundle


def build_html(workspace: Path, raw_inventory: dict[str, Any],
               normalized: dict[str, Any], classified: dict[str, Any]) -> str:
    gallery_items = _build_gallery_items(classified, workspace)
    text_rows = _build_text_rows(workspace)
    tokenizer_info = _build_tokenizer_info(workspace)
    preflight_info = _build_preflight_info(workspace, raw_inventory, normalized, classified, tokenizer_info)
    
    # Load embeddings and attention data if available
    run_dir = _resolve_run_dir(workspace)
    # Fallback: workspace parent is often the run dir (workspace = run_dir/dataset)
    if not run_dir:
        candidate = workspace.parent
        if (candidate / "embeddings.json").exists() or (candidate / "attention.json").exists() or (candidate / "weights_manifest.json").exists():
            run_dir = candidate
    # Last resort: workspace itself might be the run dir (toy_svg runs)
    if not run_dir:
        if (workspace / "embeddings.json").exists() or (workspace / "attention.json").exists():
            run_dir = workspace
    emb_data = _load_embeddings_data(run_dir) if run_dir else None
    attn_data = _load_attention_data(run_dir) if run_dir else None
    training_data = _load_training_data(run_dir) if run_dir else None
    run_scope_bundle = _load_run_scope_bundle(run_dir) if run_dir else {"available": False, "scope": {}, "agent_md": "", "training_md": ""}

    pipeline_map = _build_pipeline_map(workspace, raw_inventory, normalized,
                                       classified, training_data, emb_data, attn_data)
    
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
        + f"const CK_EMBEDDINGS = {_json_for_embed(emb_data)};\n"
        + f"const CK_ATTENTION = {_json_for_embed(attn_data)};\n"
        + f"const CK_PREFLIGHT = {_json_for_embed(preflight_info)};\n"
        + f"const CK_TRAINING = {_json_for_embed(training_data)};\n"
        + f"const CK_RUN_SCOPE = {_json_for_embed(run_scope_bundle)};\n"
        + f"const CK_PIPELINE_MAP = {_json_for_embed(pipeline_map)};\n"
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
.ck-sort-icon { font-size: 0.65em; margin-left: 4px; opacity: 0.5; }
th.sort-active .ck-sort-icon { opacity: 1; }
th .ck-col-grip { position: absolute; right: 0; top: 0; bottom: 0; width: 5px; cursor: col-resize; user-select: none; }
th { position: relative; }
.ck-table-search { width: 100%; padding: 0.5rem 0.75rem; margin-bottom: 0.75rem; background: var(--dark-card); border: 1px solid var(--grey); border-radius: 6px; color: var(--text-primary); font-size: 0.85rem; }
.ck-table-pager { display: flex; align-items: center; justify-content: space-between; padding: 0.5rem 0; font-size: 0.78rem; color: var(--text-muted); }
.ck-table-pager button { background: var(--dark-card); border: 1px solid var(--grey); color: var(--text-primary); padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 0.78rem; }
.ck-table-pager button:disabled { opacity: 0.3; cursor: default; }
/* ─── Embedding Heatmap tab ── */
.emb-load-bar { display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:14px 16px;background:var(--dark-card);border-radius:10px;border:1px solid var(--grey);margin-bottom:18px; }
.emb-stats-grid { display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px;margin-bottom:16px; }
.emb-controls { display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px; }
.emb-controls select { padding:4px 10px; }
.emb-legend { display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px;align-items:center; }
.emb-legend-item { display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text-secondary);cursor:pointer; }
.emb-legend-dot { width:11px;height:11px;border-radius:3px;flex-shrink:0; }
.emb-scroll-wrap { overflow:auto;max-height:65vh;border:1px solid var(--grey);border-radius:8px;background:#111; }
#embCanvas { display:block;image-rendering:pixelated;cursor:crosshair; }
.emb-colorbar-row { display:flex;align-items:center;gap:10px;margin-top:10px; }
.emb-cb-label { font-size:11px;color:var(--text-muted);min-width:52px; }
#embColorbar { flex:1;max-width:400px;height:14px;border-radius:4px; }
.emb-tooltip { position:fixed;pointer-events:none;z-index:2000;background:var(--dark-card);border:1px solid var(--grey);border-radius:8px;padding:9px 13px;font-size:12px;line-height:1.6;max-width:300px;display:none;box-shadow:0 4px 12px rgba(0,0,0,0.4); }
.emb-sim-grid { display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:6px;padding:12px; }
.emb-sim-item { display:flex;align-items:center;gap:8px;padding:6px 10px;background:var(--dark-card);border-radius:6px; }
.emb-sim-bar-track { width:56px;height:4px;background:var(--grey);border-radius:2px;flex-shrink:0; }
.emb-sim-bar-fill { height:100%;border-radius:2px;background:var(--orange); }
/* ─── Attention Viewer tab ── */
.attn-load-bar { display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:14px 16px;background:var(--dark-card);border-radius:10px;border:1px solid var(--grey);margin-bottom:18px; }
.attn-seq-list { display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px; }
.attn-seq-btn { padding:5px 12px;border-radius:20px;border:1px solid var(--grey);background:var(--dark-card);color:var(--text-secondary);cursor:pointer;font-size:12px;transition:all 0.2s; }
.attn-seq-btn:hover { border-color:var(--orange);color:var(--orange); }
.attn-seq-btn.active { background:var(--orange);color:var(--bg-dark);border-color:var(--orange);font-weight:700; }
.attn-split-seen { border-left:3px solid var(--green); }
.attn-split-holdout { border-left:3px solid var(--orange); }
.attn-split-canary { border-left:3px solid var(--red); }
.attn-split-custom { border-left:3px solid var(--blue); }
.attn-controls { display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:14px; }
.attn-controls select { padding:4px 10px; }
.attn-grid { display:grid;gap:10px;margin-bottom:18px; }
.attn-head-card { background:var(--dark-card);border:1px solid var(--grey);border-radius:8px;overflow:hidden;cursor:pointer;transition:all 0.2s; }
.attn-head-card:hover { border-color:var(--orange); }
.attn-head-card.active { border-color:var(--orange);box-shadow:0 0 0 2px rgba(255,180,0,0.3); }
.attn-head-label { font-size:10px;font-weight:700;text-align:center;padding:3px 6px;background:rgba(0,0,0,0.4);color:var(--text-muted);letter-spacing:0.05em; }
.attn-head-label.l0 { color:#07adf8; } .attn-head-label.l1 { color:#47b475; }
.attn-head-label.l2 { color:#9b59b6; } .attn-head-label.l3 { color:#e74c3c; }
.attn-mini-wrap { display:flex;justify-content:center;padding:4px; }
.attn-detail-panel { background:var(--dark-card);border:1px solid var(--orange);border-radius:10px;padding:16px;margin-bottom:18px;display:none; }
.attn-detail-title { font-size:13px;font-weight:700;color:var(--orange);margin-bottom:12px;display:flex;align-items:center;gap:10px; }
.attn-canvas-scroll { overflow:auto; }
#attnDetailCanvas { display:block;image-rendering:pixelated; }
.attn-avg-row { display:grid;gap:10px;margin-bottom:18px; }
.attn-avg-card { background:var(--dark-card);border:1px solid var(--grey);border-radius:8px;overflow:hidden; }
.attn-entropy-bar { height:6px;background:var(--grey);border-radius:3px;margin-top:4px; }
.attn-entropy-fill { height:100%;border-radius:3px;background:var(--blue); }
.attn-tok-chip { display:inline-block;padding:2px 7px;border-radius:4px;font-family:monospace;font-size:11px;background:rgba(255,255,255,0.06);color:var(--text-secondary);margin:2px; }
.attn-tooltip { position:fixed;pointer-events:none;z-index:2001;background:var(--dark-card);border:1px solid var(--grey);border-radius:8px;padding:8px 12px;font-size:12px;line-height:1.6;max-width:320px;display:none;box-shadow:0 4px 12px rgba(0,0,0,0.4); }
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
    <button class="tab" data-tab="embeddings">🧬 Embeddings</button>
    <button class="tab" data-tab="attention">🔭 Attention</button>
    <button class="tab" data-tab="training">📈 Training</button>
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
    <div class="panel" id="panel-embeddings"></div>
    <div class="panel" id="panel-attention"></div>
    <div class="panel" id="panel-training"></div>
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
   ════════════════════════════════════════════════════════════════════

   ⚠️  DO NOT DELETE OR REGRESS ANY FEATURES IN THIS VIEWER  ⚠️

   This viewer contains 12 tabs, each operator-critical:
     Overview, Preflight, SVG Gallery, Text Samples, Tokenizer,
     Vocabulary, Classification, Browse, Candidates, Quality,
     🧬 Embeddings, 🔭 Attention

   Key JS functions (all must be preserved):
     CKTable (reusable sort/search/pagination/resize table class),
     renderHeader, renderOverview, renderPreflight, renderGallery,
     renderTextSamples, renderTokenizer, renderTokenizerVocabOnly,
     renderVocabulary, renderClassification, renderBrowse,
     renderCandidates, renderQuality, populateFilters,
     renderEmbeddings, renderAttention, loadEmbData, loadAttnData,
     drawEmbHeatmap, renderSimPanel, cosineSim, embColor, embNormalise,
     renderAttnMain, renderAttnSeqList, renderAttnTokenChips,
     renderPagination, renderGalleryGrid, distBarsHtml,
     dtGetSplits, dtInitResize, dtRenderRows

   The tokenizer table must ALWAYS be present — it is essential for
   operators to see what tokens the model was trained on. When the
   full corpus isn't staged, renderTokenizerVocabOnly() shows the
   vocabulary from the run-level tokenizer.json as a fallback.

   You may ENHANCE any feature. You may NOT remove any feature.
   If refactoring, ensure every tab and function still works.
   ════════════════════════════════════════════════════════════════════ */

const PAL = ['#ffb400','#07adf8','#47b475','#e74c3c','#9b59b6','#1abc9c',
             '#f39c12','#3498db','#e67e22','#2ecc71','#e91e63','#00bcd4',
             '#ff5722','#8bc34a','#673ab7','#ffc107'];

/* ── Helpers ────────────────────────────────────────────────────── */
const fmt = n => n == null ? '—' : typeof n === 'number' ? n.toLocaleString() : String(n);
const pct = (n,t) => t ? (n/t*100).toFixed(1)+'%' : '0%';
const fmtPct = (n, digits) => n == null ? '—' : (n*100).toFixed(digits == null ? 1 : digits) + '%';
const esc = s => { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; };
// escAttr: safe for HTML attribute values — also escapes quotes
const escAttr = s => esc(String(s||'')).replace(/"/g,'&quot;').replace(/'/g,'&#39;');
const trunc = (s,n) => { s=String(s||''); return s.length>n ? s.slice(0,n)+'…' : s; };
const pathName = p => String(p||'').split('/').pop();

function toggleSection(hdr) {
    const body = hdr.nextElementSibling, arrow = hdr.querySelector('.section-arrow');
    if (body.style.display === 'none') { body.style.display = ''; arrow.textContent = '▼'; }
    else { body.style.display = 'none'; arrow.textContent = '▶'; }
}
function closeModal() { var el = document.getElementById('modalOverlay'); if (el) el.classList.remove('open'); }
function openModal(title, html) {
    var overlay = document.getElementById('modalOverlay');
    var titleEl = document.getElementById('modalTitle');
    var bodyEl  = document.getElementById('modalBody');
    if (!overlay || !titleEl || !bodyEl) return;
    titleEl.textContent = title;
    bodyEl.innerHTML = html;
    overlay.classList.add('open');
}

function distBarsHtml(countsObj, total, baseColor) {
    const entries = Object.entries(countsObj).sort((a,b) => b[1]-a[1]);
    if (!entries.length) return '<div style="color:var(--text-muted);font-size:0.8rem">No data</div>';
    const max = entries[0][1] || 1;
    return entries.map(([name,count],i) => {
        const w = (count/max*100).toFixed(1);
        const c = PAL[i % PAL.length] || baseColor;
        return `<div class="dist-row">
            <span class="dist-label" title="${escAttr(name)}">${esc(name)}</span>
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

/* ── CKTable: Reusable table component with search, sort, pagination ── */
class CKTable {
    constructor(config) {
        this.config = {
            containerId: '',
            columns: [], // [{key, label, width?, sortable?, mono?}]
            data: [],
            pageSize: 50,
            searchKeys: [],
            ...config
        };
        this.state = {
            sortColumn: null,
            sortDir: 'asc',
            currentPage: 0,
            filteredData: [...this.config.data],
            searchQuery: ''
        };
        this.render();
    }

    render() {
        const container = document.getElementById(this.config.containerId);
        if (!container) return;

        // Build search input
        let html = `<input type="text" class="ck-table-search" placeholder="Search..." value="${escAttr(this.state.searchQuery)}">`;
        
        // Build table
        html += '<div style="overflow-x:auto;border:1px solid var(--grey);border-radius:6px;"><table class="table"><thead><tr>';
        
        this.config.columns.forEach(col => {
            const sortable = col.sortable !== false;
            const sortIcon = this.state.sortColumn === col.key ? 
                (this.state.sortDir === 'asc' ? '▲' : '▼') : '';
            const activeClass = this.state.sortColumn === col.key ? 'sort-active' : '';
            
            html += `<th class="${activeClass}" style="${col.width ? 'width:' + col.width + ';' : ''}">`;
            html += col.label;
            if (sortable) {
                html += `<span class="ck-sort-icon">${sortIcon}</span>`;
            }
            html += '<div class="ck-col-grip"></div></th>';
        });
        html += '</tr></thead><tbody>';

        // Build table rows
        const startIdx = this.state.currentPage * this.config.pageSize;
        const endIdx = startIdx + this.config.pageSize;
        const pageData = this.state.filteredData.slice(startIdx, endIdx);

        pageData.forEach(row => {
            html += '<tr>';
            this.config.columns.forEach(col => {
                const value = row[col.key] != null ? row[col.key] : '—';
                const className = col.mono ? 'mono' : '';
                html += `<td class="${className}">${esc(String(value))}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table></div>';

        // Build pagination
        const totalPages = Math.ceil(this.state.filteredData.length / this.config.pageSize);
        const start = startIdx + 1;
        const end = Math.min(endIdx, this.state.filteredData.length);
        
        html += `<div class="ck-table-pager">`;
        html += `<span>${start}-${end} of ${this.state.filteredData.length}</span>`;
        html += '<div>';
        html += `<button ${this.state.currentPage === 0 ? 'disabled' : ''}>‹ Prev</button>`;
        html += `<button ${this.state.currentPage >= totalPages - 1 ? 'disabled' : ''}>Next ›</button>`;
        html += '</div></div>';

        container.innerHTML = html;
        this.attachEvents();
    }

    attachEvents() {
        const container = document.getElementById(this.config.containerId);
        
        // Search input
        const searchInput = container.querySelector('.ck-table-search');
        searchInput.addEventListener('input', (e) => {
            this.state.searchQuery = e.target.value.toLowerCase();
            this.filter();
        });

        // Column sorting
        const ths = container.querySelectorAll('th');
        ths.forEach((th, i) => {
            const col = this.config.columns[i];
            if (col.sortable !== false) {
                th.style.cursor = 'pointer';
                th.addEventListener('click', (e) => {
                    if (e.target.classList.contains('ck-col-grip')) return;
                    this.sort(col.key);
                });
            }
        });

        // Column resize
        const grips = container.querySelectorAll('.ck-col-grip');
        grips.forEach((grip, i) => {
            let startX, startW;
            grip.addEventListener('mousedown', (e) => {
                startX = e.pageX;
                startW = grip.parentElement.offsetWidth;
                const onMove = (ev) => {
                    grip.parentElement.style.width = Math.max(30, startW + ev.pageX - startX) + 'px';
                };
                const onUp = () => {
                    document.removeEventListener('mousemove', onMove);
                    document.removeEventListener('mouseup', onUp);
                };
                document.addEventListener('mousemove', onMove);
                document.addEventListener('mouseup', onUp);
                e.preventDefault();
            });
        });

        // Pagination
        const prevBtn = container.querySelector('.ck-table-pager button');
        const nextBtn = container.querySelector('.ck-table-pager button:last-child');
        
        prevBtn.addEventListener('click', () => {
            if (this.state.currentPage > 0) {
                this.state.currentPage--;
                this.render();
            }
        });
        
        nextBtn.addEventListener('click', () => {
            const totalPages = Math.ceil(this.state.filteredData.length / this.config.pageSize);
            if (this.state.currentPage < totalPages - 1) {
                this.state.currentPage++;
                this.render();
            }
        });
    }

    filter() {
        if (!this.state.searchQuery) {
            this.state.filteredData = [...this.config.data];
        } else {
            this.state.filteredData = this.config.data.filter(row => {
                const searchKeys = this.config.searchKeys.length ? this.config.searchKeys : 
                    this.config.columns.map(col => col.key);
                return searchKeys.some(key => {
                    const val = String(row[key] || '').toLowerCase();
                    return val.includes(this.state.searchQuery);
                });
            });
        }
        this.state.currentPage = 0;
        this.sort(this.state.sortColumn, true);
    }

    sort(column, skipRender = false) {
        if (!column) return;
        
        if (this.state.sortColumn === column) {
            this.state.sortDir = this.state.sortDir === 'asc' ? 'desc' : 'asc';
        } else {
            this.state.sortColumn = column;
            this.state.sortDir = 'asc';
        }

        this.state.filteredData.sort((a, b) => {
            const aVal = a[column];
            const bVal = b[column];
            let result = 0;
            
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                result = aVal - bVal;
            } else {
                result = String(aVal || '').localeCompare(String(bVal || ''));
            }
            
            return this.state.sortDir === 'desc' ? -result : result;
        });

        if (!skipRender) this.render();
    }
}

/* ── Data extraction ───────────────────────────────────────────── */
const raw   = typeof CK_RAW_INVENTORY !== 'undefined' ? CK_RAW_INVENTORY : {};
const norm  = typeof CK_NORMALIZED !== 'undefined' ? CK_NORMALIZED : {};
const cls   = typeof CK_CLASSIFIED !== 'undefined' ? CK_CLASSIFIED : {};
const ws    = typeof CK_WORKSPACE !== 'undefined' ? CK_WORKSPACE : '';
const preflight = typeof CK_PREFLIGHT !== 'undefined' ? CK_PREFLIGHT : { available: false, checklist: [] };
const runScopeBundle = typeof CK_RUN_SCOPE !== 'undefined' ? CK_RUN_SCOPE : { available: false, scope: {}, agent_md: '', training_md: '' };
const runScope = (runScopeBundle && typeof runScopeBundle.scope === 'object') ? runScopeBundle.scope : {};

const rawEntries    = Array.isArray(raw.entries) ? raw.entries : [];
const normFailures  = Array.isArray(norm.failures) ? norm.failures : [];
const classEntries  = Array.isArray(cls.entries) ? cls.entries : [];
const tagTotals     = norm.tag_totals || {};
const placeholders  = norm.placeholder_totals || {};
const familyCounts  = cls.family_counts || {};
const roleCounts    = cls.counts || {};
const suggested     = (typeof cls.suggested_splits === 'object' && cls.suggested_splits) ? cls.suggested_splits : {};
const familyDescs   = cls.family_descriptions || {};
const sourceDescs   = cls.source_descriptions || {};

// ── Empty-state helper ──────────────────────────────────────────
// ws = CK_WORKSPACE (dataset dir), runDir = parent of ws (run dir)
var runDir = ws ? ws.replace(/\/dataset\/?$/, '') : '';
// Data provenance: detect if vocab/classification was synthesized from render catalog
var isSynthesized = (norm.schema || '').indexOf('synthesized') >= 0;

function provenanceBanner(source) {
    return '<div style="background:rgba(255,165,0,0.08);border:1px solid rgba(255,165,0,0.2);border-radius:6px;padding:0.6rem 1rem;margin-bottom:1rem;font-size:0.8rem;color:var(--orange);">'
        + '🔄 <strong>Data source:</strong> ' + source
        + '</div>';
}

function emptyTabHtml(icon, title, reason, cmds) {
    var h = '<div style="text-align:center;padding:2.5rem;color:var(--text-muted);">'
        + '<div style="font-size:2rem;margin-bottom:0.8rem">' + icon + '</div>'
        + '<div style="font-size:1rem;font-weight:600;color:var(--orange);margin-bottom:0.4rem">' + title + '</div>'
        + '<div style="font-size:0.85rem;margin-bottom:1rem">' + reason + '</div>';
    if (cmds && cmds.length) {
        h += '<div style="text-align:left;max-width:700px;margin:0 auto;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:1rem;">';
        h += '<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;color:var(--orange);">How to populate</div>';
        for (var i = 0; i < cmds.length; i++) {
            h += '<div style="margin-bottom:0.4rem;font-size:0.8rem;">' + (i+1) + '. <code style="background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:3px;font-size:0.75rem;user-select:all;cursor:text;">' + cmds[i] + '</code></div>';
        }
        h += '</div>';
    }
    h += '</div>';
    return h;
}

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
    if (state.filterRole) {
        if (isSynthesized) {
            rows = rows.filter(r => r.source_name === state.filterRole);
        } else {
            rows = rows.filter(r => (r.roles||[]).includes(state.filterRole));
        }
    }
    return rows;
}

/* ── Header meta chips ─────────────────────────────────────────── */
function renderHeader() {
    const el = document.getElementById('headerMeta');
    const chips = [];
    chips.push(`Workspace: <code>${esc(pathName(ws))}</code>`);
    chips.push(`Raw: <code>${esc(pathName(raw.raw_assets_root||''))}</code>`);
    chips.push(`Normalized: <code>${esc(pathName(norm.normalized_root||''))}</code>`);
    if (runScope.spec) chips.push(`Spec: <code>${esc(String(runScope.spec))}</code>`);
    if (runScope.rung) chips.push(`Rung: <code>${esc(String(runScope.rung))}</code>`);
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

    if (runScopeBundle.available) {
        html += sectionHtml('🧭', 'Run Brief', runScope.rung || runScope.spec || 'loaded', 'badge-green', 'runBriefBody');
    }

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

    if (runScopeBundle.available) {
        const body = document.getElementById('runBriefBody');
        if (body) {
            const runTitle = runScope.title || runScope.run_name || 'Run brief';
            let briefHtml = `<div class="source-block"><strong>${esc(String(runTitle))}</strong>`;
            if (runScope.objective) briefHtml += `<div style="margin-top:0.45rem;">${esc(String(runScope.objective))}</div>`;
            if (runScope.prompt_contract) briefHtml += `<div style="margin-top:0.55rem;color:var(--orange);"><strong>Prompt contract:</strong> ${esc(String(runScope.prompt_contract))}</div>`;
            if (runScope.output_contract) briefHtml += `<div style="margin-top:0.3rem;color:var(--green);"><strong>Output contract:</strong> ${esc(String(runScope.output_contract))}</div>`;
            const mkList = (title, values) => Array.isArray(values) && values.length
                ? `<div style="margin-top:0.65rem;"><div class="subhead">${esc(title)}</div>${values.map(v => `<span class="token-pill">${esc(String(v))}</span>`).join('')}</div>`
                : '';
            briefHtml += mkList('In Scope', runScope.in_scope);
            briefHtml += mkList('Out Of Scope', runScope.out_of_scope);
            briefHtml += mkList('Success Gates', runScope.success_gates);
            if (runScopeBundle.agent_md) {
                briefHtml += `<div class="subhead" style="margin-top:0.8rem">agent.md</div><div class="source-block" style="white-space:pre-wrap;">${esc(runScopeBundle.agent_md)}</div>`;
            }
            if (runScopeBundle.training_md) {
                briefHtml += `<div class="subhead" style="margin-top:0.8rem">training.md</div><div class="source-block" style="white-space:pre-wrap;">${esc(runScopeBundle.training_md)}</div>`;
            }
            briefHtml += '</div>';
            body.innerHTML = briefHtml;
        }
    }

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

    // ── Pipeline Map: script → artifact → tab ──
    var pmap = (typeof CK_PIPELINE_MAP !== 'undefined') ? CK_PIPELINE_MAP : [];
    if (pmap.length) {
        var pmDiv = document.createElement('div');
        pmDiv.style.marginTop = '1.5rem';
        var pmHtml = '<div class="subhead">🗺️ Pipeline Map — Script → Artifact → Tab</div>';
        pmHtml += '<div class="subnote">Which scripts produce which artifacts for which tabs. Status shows whether the artifact was found, synthesized from the render catalog, or is missing. Copy commands to generate missing artifacts.</div>';
        pmHtml += '<table><thead><tr><th>Tab</th><th>Artifact</th><th>Script</th><th>Status</th><th>Detail</th></tr></thead><tbody>';
        pmap.forEach(function(row) {
            var tone = row.status === 'present' ? 'color:var(--green)' : (row.status === 'synthesized' ? 'color:var(--orange)' : (row.status === 'na' ? 'color:var(--text-muted)' : 'color:var(--red)'));
            var badge = row.status === 'present' ? '✅' : (row.status === 'synthesized' ? '🔄' : (row.status === 'na' ? '➖' : '❌'));
            pmHtml += '<tr>';
            pmHtml += '<td class="mono">' + esc(row.tab) + '</td>';
            pmHtml += '<td class="mono" style="font-size:0.75rem">' + esc(row.artifact) + '</td>';
            pmHtml += '<td class="mono" style="font-size:0.75rem">' + esc(row.script) + '</td>';
            pmHtml += '<td style="' + tone + '">' + badge + ' ' + esc(row.status) + '</td>';
            pmHtml += '<td style="font-size:0.75rem">' + esc(row.detail) + '</td>';
            pmHtml += '</tr>';
            if (row.status === 'missing') {
                pmHtml += '<tr><td colspan="5" style="padding:0.2rem 1rem 0.6rem 1rem;border-top:none;"><code style="background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:3px;font-size:0.72rem;user-select:all;cursor:text;">' + esc(row.command) + '</code></td></tr>';
            }
        });
        pmHtml += '</tbody></table>';
        pmDiv.innerHTML = pmHtml;
        el.appendChild(pmDiv);
    }
}

/* ── Tab: Vocabulary ───────────────────────────────────────────── */
function renderVocabulary() {
    const el = document.getElementById('panel-vocabulary');
    if (!Object.keys(tagTotals).length && !Object.keys(placeholders).length) {
        el.innerHTML = emptyTabHtml('🔤', 'No Vocabulary Data', 'Normalized SVG corpus not found. The vocabulary tab shows tag frequency analysis from normalized assets.', [
            'python3 version/v7/scripts/dataset/normalize_svg_assets_v7.py --workspace ' + ws,
            'python3 version/v7/tools/prepare_run_viewer.py ' + runDir + ' --force'
        ]);
        return;
    }
    const tagTotal = Object.values(tagTotals).reduce((s,v) => s+v, 0);
    const phTotal  = Object.values(placeholders).reduce((s,v) => s+v, 0);

    let html = '';
    if (isSynthesized) {
        html += provenanceBanner('Synthesized from structured-atoms render catalog (' + fmt(norm.normalized_entries || 0) + ' DSL entries). This workspace uses DSL text training, not raw SVG files. Tag frequencies reflect DSL tokens and rendered SVG elements.');
    }
    html += '<div class="stats-grid">';
    html += statCardHtml(fmt(Object.keys(tagTotals).length), isSynthesized ? 'DSL + SVG Tags' : 'SVG Tag Types');
    html += statCardHtml(fmt(tagTotal), 'Total Instances', null, 'blue');
    html += statCardHtml(fmt(Object.keys(placeholders).length), isSynthesized ? 'DSL Value Types' : 'Placeholder Types');
    html += statCardHtml(fmt(phTotal), isSynthesized ? 'Value Instances' : 'Placeholder Instances', null, 'green');
    html += '</div>';

    html += sectionHtml('🔤', isSynthesized ? 'DSL + SVG Tag Histogram' : 'SVG Vocabulary Histogram', fmt(Object.keys(tagTotals).length)+' tags', 'badge-blue', 'vocabBody');
    html += sectionHtml('📝', isSynthesized ? 'DSL Value Histogram' : 'Placeholder Histogram', '', 'badge-green', 'phBody');

    el.innerHTML = html;

    document.getElementById('vocabBody').innerHTML =
        (isSynthesized
            ? '<div class="subnote">Tag frequencies from DSL tokens (layout, theme, canvas…) and rendered SVG elements (rect, text, line…). Hover bars for counts.</div>'
            : '<div class="subnote">Top tag counts across the normalized corpus — the visual grammar the model learns from.</div>') +
        distBarsHtml(tagTotals, tagTotal, '#07adf8');

    document.getElementById('phBody').innerHTML =
        (isSynthesized
            ? '<div class="subnote">DSL tag values — the variable slots in [tag:value] tokens. Shows which values appear most frequently across the training corpus.</div>'
            : '<div class="subnote">Human text removed during normalization. Keeps composition/layout signal while reducing English memorization.</div>') +
        distBarsHtml(placeholders, phTotal, '#47b475');
}

/* ── Tab: Classification ───────────────────────────────────────── */
function renderClassification() {
    const el = document.getElementById('panel-classification');
    const total = classEntries.length;
    if (!total) {
        el.innerHTML = emptyTabHtml('🏷️', 'No Classification Data', 'No classified assets found. Run the asset classifier to populate families, roles, and split candidates.', [
            'python3 version/v7/scripts/dataset/classify_svg_assets_v7.py --workspace ' + ws,
            'python3 version/v7/tools/prepare_run_viewer.py ' + runDir + ' --force'
        ]);
        return;
    }

    let html = '';
    if (isSynthesized) {
        html += provenanceBanner('Synthesized from render catalog. Families = DSL layout types, sources = topic categories. ' + fmt(total) + ' training entries.');
    }
    html += '<div class="stats-grid">';
    html += statCardHtml(fmt(Object.keys(familyCounts).length), isSynthesized ? 'Layouts' : 'Families');
    html += statCardHtml(isSynthesized ? fmt(Object.keys(cls.counts && cls.counts.by_source ? cls.counts.by_source : {}).length) : fmt(Object.keys(roleCounts).length), isSynthesized ? 'Topics' : 'Roles');
    html += statCardHtml(fmt(total), isSynthesized ? 'Training Entries' : 'Classified Assets', null, 'blue');
    html += '</div>';

    html += sectionHtml('🏷️', isSynthesized ? 'Layout Distribution' : 'Family Distribution', '', 'badge-blue', 'famDistBody');
    html += sectionHtml('🎭', isSynthesized ? 'Topic Distribution' : 'Role Distribution', '', 'badge-purple', 'roleDistBody');

    // Split candidates summary
    html += sectionHtml('✂️', 'Split Candidates', '', 'badge-orange', 'splitBody');

    el.innerHTML = html;

    document.getElementById('famDistBody').innerHTML =
        (isSynthesized
            ? '<div class="subnote">DSL layout families from the render catalog. Each family is a distinct infographic layout type the model learns to generate.</div>'
            : '<div class="subnote">Heuristic family labels from normalized filenames and structure.</div>') +
        distBarsHtml(familyCounts, total, '#07adf8') +
        (Object.keys(familyDescs).length
            ? '<div style="margin-top:0.8rem;border-top:1px solid rgba(255,255,255,0.06);padding-top:0.6rem;">' +
              '<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--text-muted);margin-bottom:0.4rem;">Layout Legend</div>' +
              Object.entries(familyDescs).map(function(kv) {
                  return '<div style="font-size:0.78rem;margin-bottom:0.3rem;"><span style="color:var(--purple);font-weight:600;">' + esc(kv[0]) + '</span> — <span style="color:var(--text-muted)">' + esc(kv[1]) + '</span></div>';
              }).join('') +
              '</div>'
            : '');

    document.getElementById('roleDistBody').innerHTML =
        (isSynthesized
            ? '<div class="subnote">Topic categories from the render catalog. Each topic provides domain-specific content for the layouts.</div>'
            : '<div class="subnote">Roles assigned by the classifier. Assets can carry multiple roles.</div>') +
        (isSynthesized ? distBarsHtml(cls.counts && cls.counts.by_source ? cls.counts.by_source : roleCounts, total, '#ffb400') : distBarsHtml(roleCounts, total, '#ffb400')) +
        (Object.keys(sourceDescs).length
            ? '<div style="margin-top:0.8rem;border-top:1px solid rgba(255,255,255,0.06);padding-top:0.6rem;">' +
              '<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--text-muted);margin-bottom:0.4rem;">Topic Legend</div>' +
              Object.entries(sourceDescs).map(function(kv) {
                  return '<div style="font-size:0.78rem;margin-bottom:0.3rem;"><span style="color:var(--orange);font-weight:600;">' + esc(kv[0]) + '</span> — <span style="color:var(--text-muted)">' + esc(kv[1]) + '</span></div>';
              }).join('') +
              '</div>'
            : '');

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
    if (!classEntries.length) {
        el.innerHTML = emptyTabHtml('📋', 'No Assets to Browse', 'The browse tab shows classified and normalized assets. No classified entries found in this workspace.', [
            'python3 version/v7/scripts/dataset/classify_svg_assets_v7.py --workspace ' + ws,
            'python3 version/v7/tools/prepare_run_viewer.py ' + runDir + ' --force'
        ]);
        return;
    }
    document.getElementById('resultCount').textContent = `${rows.length} assets`;

    var browseProvenance = '';
    if (isSynthesized) {
        browseProvenance = provenanceBanner('Synthesized from render catalog. Each row = one DSL training entry. Family = layout type, source = topic.');
    }

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
        { key: 'normalized_path', label: isSynthesized ? 'Entry' : 'Asset', render: r => {
            var name = isSynthesized ? (r.source_path || r.family + '_' + (r.source_name||'')) : pathName(r.normalized_path);
            return `<span class="mono" style="color:var(--blue)">${esc(trunc(name,40))}</span>`;
        }},
        { key: 'family', label: isSynthesized ? 'Layout' : 'Family', render: r => {
            var desc = familyDescs[r.family] || '';
            return `<span style="color:var(--purple)" ${desc ? 'title="'+escAttr(desc)+'"' : ''}>${esc(r.family||'—')}${desc ? ' ℹ️' : ''}</span>`;
        }},
        { key: 'source_name', label: 'Topic', render: r => {
            var desc = sourceDescs[r.source_name] || '';
            return `<span style="color:var(--orange)" ${desc ? 'title="'+escAttr(desc)+'"' : ''}>${esc(r.source_name||'—')}</span>`;
        }},
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
    el.innerHTML = browseProvenance + html;

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
    if (!classEntries.length) {
        el.innerHTML = emptyTabHtml('🎯', 'No Candidates Available', 'Split candidates require classified assets. Run the classifier first.', [
            'python3 version/v7/scripts/dataset/classify_svg_assets_v7.py --workspace ' + ws
        ]);
        return;
    }
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
    if (!Object.keys(norm).length || (!norm.normalized_entries && !norm.duplicate_normalized_entries)) {
        el.innerHTML = emptyTabHtml('🔍', 'No Quality Data', 'Quality analysis requires a normalized corpus. No normalization data found.', [
            'python3 version/v7/scripts/dataset/normalize_svg_assets_v7.py --workspace ' + ws
        ]);
        return;
    }
    const dupes = norm.duplicate_normalized_entries || 0;
    const failCount = normFailures.length;
    const tok = CK_TOKENIZER || {};
    const prompt = tok.prompt_contract || {};
    const drift = tok.manifest_drift || { count: 0, entries: [] };

    let html = '';
    if (isSynthesized) {
        html += provenanceBanner('Synthesized from render catalog — no SVG normalization or deduplication was performed. Metrics below reflect catalog counts, not measured quality gates.');
        html += '<div class="stats-grid">';
        html += statCardHtml(fmt(norm.normalized_entries||0), 'Catalog Entries', 'from render catalog (not normalized SVGs)', 'blue');
        html += statCardHtml('N/A', 'Duplicates', 'deduplication not run on this workspace', '');
        html += statCardHtml('N/A', 'Parse Failures', 'SVG parse step not applicable for DSL text', '');
        html += '</div>';
    } else {
        html += '<div class="stats-grid">';
        html += statCardHtml(fmt(dupes), 'Duplicates', null, dupes === 0 ? 'green' : 'red');
        html += statCardHtml(fmt(failCount), 'Parse Failures', null, failCount === 0 ? 'green' : 'red');
        html += statCardHtml(fmt(norm.normalized_entries||0), 'Normalized OK', null, 'green');
        html += statCardHtml(fmt(norm.input_non_ascii_chars_total||0), 'Non-ASCII Input', null, (norm.input_non_ascii_chars_total||0) === 0 ? 'green' : '');
        html += statCardHtml(fmt(norm.output_non_ascii_chars_total||0), 'Non-ASCII Output', null, (norm.output_non_ascii_chars_total||0) === 0 ? 'green' : 'red');
        html += statCardHtml(fmt(norm.mapped_common_symbols_total||0), 'Mapped Symbols');
        html += '</div>';
    }

    // Checks
    // Checks — skip SVG-specific checks for synthesized data
    const checks = [];
    if (isSynthesized) {
        checks.push({ ok: true, label: 'Render catalog loaded', detail: fmt(norm.normalized_entries||0) + ' DSL entries from structured-atoms pipeline' });
        checks.push({ ok: true, label: 'Catalog integrity', detail: 'Entries parsed without errors' });
    } else {
        checks.push({ ok: dupes === 0, label: 'No duplicate hashes after normalization', detail: dupes ? `${dupes} duplicates` : 'All unique' });
        checks.push({ ok: failCount === 0, label: 'No normalization parse failures', detail: failCount ? `${failCount} failures` : 'Clean parse' });
        checks.push({ ok: (norm.output_non_ascii_chars_total||0) === 0, label: 'Output is ASCII-only', detail: `${norm.output_non_ascii_chars_total||0} non-ASCII chars in output` });
        checks.push({ ok: (raw.duplicate_files||0) === 0, label: 'No duplicate raw imports', detail: `${raw.duplicate_files||0} raw dupes` });
    }

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
        if (isSynthesized) {
            // For synthesized: use source_name (topic) for the topic filter, skip split labels
            if (e.source_name) roles.add(e.source_name);
        } else {
            (e.roles||[]).forEach(r => roles.add(r));
        }
    });
    const famEl = document.getElementById('filterFamily');
    const roleEl = document.getElementById('filterRole');
    var famLabel = isSynthesized ? 'All layouts' : 'All families';
    var roleLabel = isSynthesized ? 'All topics' : 'All roles';
    famEl.innerHTML = '<option value="">' + famLabel + '</option>' + [...families].sort().map(function(f) {
        var desc = familyDescs[f];
        var label = f + (desc ? ' \u2014 ' + desc.split('\u2014')[0].trim() : '');
        return '<option value="' + escAttr(f) + '"' + (desc ? ' title="' + escAttr(desc) + '"' : '') + '>' + esc(label) + '</option>';
    }).join('');
    roleEl.innerHTML = '<option value="">' + roleLabel + '</option>' + [...roles].sort().map(function(r) {
        var desc = sourceDescs[r];
        var label = r + (desc ? ' \u2014 ' + desc : '');
        return '<option value="' + escAttr(r) + '">' + esc(label) + '</option>';
    }).join('');
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
        var galleryMsg = isSynthesized
            ? 'This workspace uses structured-atoms DSL text training. SVG gallery is not available — visual previews require raw SVG file assets. See the Browse tab for entry-level data.'
            : 'No SVG assets found in this dataset. Run the classifier to populate gallery items.';
        document.getElementById('panel-gallery').innerHTML = emptyTabHtml('🖼️', 'No Gallery Data', galleryMsg,
            isSynthesized ? [] : [
                'python3 version/v7/scripts/dataset/classify_svg_assets_v7.py --workspace ' + ws,
                'python3 version/v7/tools/prepare_run_viewer.py ' + runDir + ' --force'
            ]);
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
        <div class="gal-card" onclick="galViewer.open(${i})" title="${escAttr(item.name)}">
            <div class="gal-thumb">
                <img src="${item.data_uri}" alt="${escAttr(item.name)}" loading="lazy">
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
    html += '<input type="text" class="dt-search" id="dtSearch" placeholder="Search rows…" value="' + escAttr(dtState.search) + '">';
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
/* ── Vocab-only fallback renderer for runs without staged corpus ── */
function renderTokenizerVocabOnly(panel, tok) {
    const vocabTable = tok.vocab_table || [];
    const familyCounts = tok.vocab_family_counts || {};
    const vocabSize = tok.vocab_size || vocabTable.length;
    const ckMode = tok.ck_mode || 'unknown';
    const protTokens = tok.protected_tokens || [];

    let html = '<div class="subhead">Tokenizer Vocabulary</div>';
    html += '<div class="subnote">Vocabulary extracted from the run-level <code>tokenizer.json</code>. Full corpus staging was not found — showing the learned vocabulary directly.</div>';

    /* ── Summary metrics ── */
    html += '<div class="tok-stat-grid">';
    html += `<div class="tok-stat"><div class="tok-stat-val">${fmt(vocabSize)}</div><div class="tok-stat-lbl">Vocab Size</div></div>`;
    html += `<div class="tok-stat"><div class="tok-stat-val">${esc(ckMode)}</div><div class="tok-stat-lbl">Mode</div></div>`;
    html += `<div class="tok-stat"><div class="tok-stat-val">${Object.keys(familyCounts).length}</div><div class="tok-stat-lbl">Token Families</div></div>`;
    html += `<div class="tok-stat"><div class="tok-stat-val">${protTokens.length}</div><div class="tok-stat-lbl">Protected Tokens</div></div>`;
    html += '</div>';

    /* ── Family distribution bar ── */
    if (Object.keys(familyCounts).length > 0) {
        html += '<div class="section-card"><div class="section-header" onclick="this.parentElement.classList.toggle(\'collapsed\')"><span>📊 Token Family Distribution</span><span class="badge badge-blue">' + Object.keys(familyCounts).length + ' families</span><span class="caret">▼</span></div><div class="section-body">';
        html += '<div class="subnote">Distribution of tokens across structural families.</div>';
        html += distBarsHtml(familyCounts, vocabSize, '#9b59b6');
        html += '</div></div>';
    }

    /* ── Protected tokens ── */
    if (protTokens.length) {
        html += '<div class="section-card"><div class="section-header" onclick="this.parentElement.classList.toggle(\'collapsed\')"><span>🛡️ Protected DSL Tokens</span><span class="badge badge-green">' + protTokens.filter(r=>r.present).length + '/' + protTokens.length + '</span><span class="caret">▼</span></div><div class="section-body">';
        html += '<table><thead><tr><th>Token</th><th>ID</th><th>Family</th><th>Protected</th><th>Present</th></tr></thead><tbody>';
        protTokens.forEach(r => {
            html += `<tr><td class="mono">${esc(r.token)}</td><td class="mono">${r.id != null ? r.id : '—'}</td><td>${esc(r.family||'—')}</td><td>${r.protected ? '✅' : '—'}</td><td style="color:${r.present?'var(--green)':'var(--red)'}">${r.present?'yes':'missing'}</td></tr>`;
        });
        html += '</tbody></table></div></div>';
    }

    /* ── Full vocab table (searchable, sortable, resizable columns) ── */
    html += '<div class="section-card"><div class="section-header" onclick="this.parentElement.classList.toggle(\'collapsed\')"><span>📖 Full Vocabulary</span><span class="badge badge-purple">' + fmt(vocabSize) + ' tokens</span><span class="caret">▼</span></div><div class="section-body">';
    html += '<div class="subnote">Complete vocabulary sorted by token ID. Use the search box to filter. Drag column borders to resize.</div>';
    html += '<div id="vocabTableContainer"></div>';
    html += '</div></div>';

    panel.innerHTML = html;

    // Create CKTable for vocabulary
    new CKTable({
        containerId: 'vocabTableContainer',
        columns: [
            {key: 'id', label: 'ID', width: '60px', sortable: true, mono: true},
            {key: 'token', label: 'Token', sortable: true, mono: true},
            {key: 'len', label: 'Len', width: '50px', sortable: true, mono: true},
            {key: 'family', label: 'Family', width: '120px', sortable: true}
        ],
        data: vocabTable.map(r => ({
            ...r,
            token: r.token.replace(/ /g, '·').replace(/\\n/g, '↵')
        })),
        pageSize: 50,
        searchKeys: ['token', 'family']
    });
}

/* ── Column resize helper: drag column borders to resize (DEPRECATED: use CKTable instead) ── */
function initColumnResize(tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;
    const ths = table.querySelectorAll('thead th');
    ths.forEach(th => {
        const grip = document.createElement('div');
        grip.style.cssText = 'position:absolute;right:0;top:0;bottom:0;width:5px;cursor:col-resize;user-select:none;';
        th.style.position = 'relative';
        th.appendChild(grip);
        let startX, startW;
        grip.addEventListener('mousedown', function(e) {
            startX = e.pageX;
            startW = th.offsetWidth;
            const onMove = ev => { th.style.width = Math.max(30, startW + ev.pageX - startX) + 'px'; };
            const onUp = () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
            e.preventDefault();
        });
    });
}

/*
 * IMPORTANT: Never remove the tokenizer table. It is essential for operators
 * to understand what tokens the model was trained on. Only enhance this tab
 * — never delete it. When the full corpus isn't staged, we fall back to
 * displaying the vocabulary from the run-level tokenizer.json.
 */
function renderTokenizer() {
    const panel = document.getElementById('panel-tokenizer');
    const tok = CK_TOKENIZER;
    const pre = CK_PREFLIGHT || {};
    if (!tok.available) {
        panel.innerHTML = '<div class="subnote" style="padding:2rem;text-align:center;">No tokenizer data found in this dataset. To populate this tab, stage the tokenizer corpus or ensure a <code>tokenizer.json</code> exists in the run directory.</div>';
        return;
    }
    /* ── Fallback: vocab-only mode from run tokenizer.json ── */
    if (tok.vocab_source === 'run_tokenizer_json') {
        renderTokenizerVocabOnly(panel, tok);
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

/* ═══════════════════════════════════════════════════════════════════════
   🧬 Embeddings Viewer — ported from repo-root dataset_viewer.html
   ═══════════════════════════════════════════════════════════════════════ */

const EMB_GROUP_COLORS = {
    system:        '#e74c3c',
    prompt:        '#07adf8',
    svg_structure: '#47b475',
    svg_style:     '#9b59b6',
    svg_attr:      '#ffb400',
    dsl_other:     '#1abc9c',
    ascii:         '#555555',
};

/* Diverging colormap: blue (#07adf8) → near-white → orange (#ffb400) */
function embColor(t) {
    const blue   = [7, 173, 248];
    const mid    = [195, 200, 208];
    const orange = [255, 180, 0];
    let r, g, b;
    if (t < 0.5) {
        const s = t * 2;
        r = blue[0] + (mid[0] - blue[0]) * s;
        g = blue[1] + (mid[1] - blue[1]) * s;
        b = blue[2] + (mid[2] - blue[2]) * s;
    } else {
        const s = (t - 0.5) * 2;
        r = mid[0] + (orange[0] - mid[0]) * s;
        g = mid[1] + (orange[1] - mid[1]) * s;
        b = mid[2] + (orange[2] - mid[2]) * s;
    }
    return [Math.round(r), Math.round(g), Math.round(b)];
}

function embNormalise(matrix, mode) {
    if (!matrix || !matrix.length || !matrix[0]) return { norm: [], vmin: 0, vmax: 0, note: '' };
    const V = matrix.length, D = matrix[0].length;
    const out = matrix.map(r => Float32Array.from(r));
    if (mode === 'global') {
        let mn = Infinity, mx = -Infinity;
        for (const r of matrix) for (const v of r) { if (v < mn) mn = v; if (v > mx) mx = v; }
        const rng = (mx - mn) || 1;
        for (let i = 0; i < V; i++) for (let j = 0; j < D; j++) out[i][j] = (matrix[i][j] - mn) / rng;
        return { norm: out, vmin: mn, vmax: mx, note: '' };
    } else if (mode === 'col') {
        for (let j = 0; j < D; j++) {
            let s = 0, s2 = 0;
            for (let i = 0; i < V; i++) { s += matrix[i][j]; s2 += matrix[i][j] ** 2; }
            const mean = s / V, std = Math.sqrt(s2 / V - mean * mean) || 1;
            for (let i = 0; i < V; i++) out[i][j] = Math.max(0, Math.min(1, (matrix[i][j] - mean) / (3 * std) * 0.5 + 0.5));
        }
        return { norm: out, vmin: -3, vmax: 3, note: 'per-column z (±3σ)' };
    } else {
        for (let i = 0; i < V; i++) {
            let s = 0, s2 = 0;
            for (const v of matrix[i]) { s += v; s2 += v ** 2; }
            const mean = s / D, std = Math.sqrt(s2 / D - mean * mean) || 1;
            for (let j = 0; j < D; j++) out[i][j] = Math.max(0, Math.min(1, (matrix[i][j] - mean) / (3 * std) * 0.5 + 0.5));
        }
        return { norm: out, vmin: -3, vmax: 3, note: 'per-row z (±3σ)' };
    }
}

function cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
    return dot / (Math.sqrt(na * nb) || 1);
}

/* State */
const embSt = { data: null, selectedOrig: -1, sortedIndices: [], rowPx: 14, colPx: 9, labelW: 180 };

function embSortedIndices() {
    if (!embSt.data) return [];
    const mode = document.getElementById('embSort') ? document.getElementById('embSort').value : 'id';
    const V = embSt.data.vocab.length;
    let idx = Array.from({ length: V }, (_, i) => i);
    const groupOrder = ['system', 'prompt', 'svg_structure', 'svg_style', 'svg_attr', 'dsl_other', 'ascii'];
    if (mode === 'group') {
        idx.sort((a, b) => {
            const ga = groupOrder.indexOf(embSt.data.vocab[a].group);
            const gb = groupOrder.indexOf(embSt.data.vocab[b].group);
            return ga !== gb ? ga - gb : a - b;
        });
    } else if (mode === 'norm') {
        idx.sort((a, b) => {
            const na = embSt.data.matrix[a].reduce((s, v) => s + v * v, 0);
            const nb = embSt.data.matrix[b].reduce((s, v) => s + v * v, 0);
            return nb - na;
        });
    } else if (mode === 'sim' && embSt.selectedOrig >= 0) {
        const ref = embSt.data.matrix[embSt.selectedOrig];
        idx.sort((a, b) => cosineSim(embSt.data.matrix[b], ref) - cosineSim(embSt.data.matrix[a], ref));
    }
    return idx;
}

function drawEmbColorbar(vmin, vmax, note) {
    const cb = document.getElementById('embColorbar');
    if (!cb) return;
    const W = Math.max(cb.offsetWidth, 200);
    cb.width = W; cb.height = 14;
    const ctx = cb.getContext('2d');
    for (let x = 0; x < W; x++) {
        const [r, g, b] = embColor(x / W);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x, 0, 1, 14);
    }
    if (document.getElementById('embCbMin')) document.getElementById('embCbMin').textContent = vmin.toFixed(4);
    if (document.getElementById('embCbMax')) document.getElementById('embCbMax').textContent = vmax.toFixed(4);
    if (document.getElementById('embCbNote')) document.getElementById('embCbNote').textContent = note;
}

function drawEmbHeatmap() {
    if (!embSt.data) return;
    const rowPx = document.getElementById('embRowPx') ? parseInt(document.getElementById('embRowPx').value) : embSt.rowPx;
    const colPx = embSt.colPx;
    const labelW = embSt.labelW;
    const normMode = document.getElementById('embNorm') ? document.getElementById('embNorm').value : 'global';

    const indices = embSortedIndices();
    embSt.sortedIndices = indices;
    embSt.rowPx = rowPx;

    const V = indices.length;
    if (!V || !embSt.data.matrix[0]) return;
    const D = embSt.data.matrix[0].length;

    const orderedMatrix = indices.map(i => embSt.data.matrix[i]);
    const { norm, vmin, vmax, note } = embNormalise(orderedMatrix, normMode);

    const W = labelW + D * colPx;
    const H = V * rowPx;

    const canvas = document.getElementById('embCanvas');
    if (!canvas) return;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');

    /* Fill heatmap pixels via ImageData for speed */
    const img = ctx.createImageData(W, H);
    const px = img.data;

    for (let vi = 0; vi < V; vi++) {
        const row = norm[vi];
        const y0 = vi * rowPx;
        for (let di = 0; di < D; di++) {
            const [r, g, b] = embColor(row[di]);
            const x0 = labelW + di * colPx;
            for (let py = y0; py < y0 + rowPx; py++) {
                for (let px_ = x0; px_ < x0 + colPx; px_++) {
                    const i4 = (py * W + px_) * 4;
                    px[i4] = r; px[i4 + 1] = g; px[i4 + 2] = b; px[i4 + 3] = 255;
                }
            }
        }
    }
    ctx.putImageData(img, 0, 0);

    /* Draw label column on top */
    const fontSize = Math.min(rowPx - 3, 11);
    ctx.font = `${fontSize}px "JetBrains Mono", ui-monospace, monospace`;
    ctx.textBaseline = 'middle';
    for (let vi = 0; vi < V; vi++) {
        const tok = embSt.data.vocab[indices[vi]];
        const gc  = EMB_GROUP_COLORS[tok.group] || '#808080';
        const y0  = vi * rowPx;
        const yMid = y0 + rowPx / 2;

        /* dark label background */
        ctx.fillStyle = '#0e0e0e';
        ctx.fillRect(0, y0, labelW - 1, rowPx);

        /* group colour bar (4px strip) */
        ctx.fillStyle = gc;
        ctx.fillRect(0, y0, 4, rowPx);

        /* row highlight if this is the selected token */
        if (indices[vi] === embSt.selectedOrig) {
            ctx.fillStyle = 'rgba(255,180,0,0.12)';
            ctx.fillRect(0, y0, W, rowPx);
            ctx.strokeStyle = 'rgba(255,180,0,0.5)';
            ctx.lineWidth = 1;
            ctx.strokeRect(0, y0, W, rowPx);
        }

        /* token label text */
        ctx.fillStyle = gc;
        const maxW = labelW - 14;
        let label = tok.token;
        /* rough character truncation (monospace ~6.5px/char at 10px) */
        const maxChars = Math.floor(maxW / (fontSize * 0.65));
        if (label.length > maxChars) label = label.slice(0, maxChars - 1) + '…';
        ctx.fillText(label, 8, yMid);
    }

    drawEmbColorbar(vmin, vmax, note);
}

/* Tooltip */
const embTip = (() => {
    const el = document.createElement('div');
    el.className = 'emb-tooltip';
    document.body.appendChild(el);
    return el;
})();

function renderSimPanel(origIdx) {
    const tok = embSt.data.vocab[origIdx];
    const ref = embSt.data.matrix[origIdx];
    if (document.getElementById('simTargetLabel')) {
        document.getElementById('simTargetLabel').textContent = tok.token;
    }
    if (document.getElementById('simPanel')) {
        document.getElementById('simPanel').style.display = '';
    }

    const sims = embSt.data.vocab.map((t, i) => ({
        i, tok: t, sim: cosineSim(embSt.data.matrix[i], ref)
    }));
    sims.sort((a, b) => b.sim - a.sim);
    const top = sims.slice(0, 24);

    const html = '<div class="emb-sim-grid">' + top.map(s => {
        const gc  = EMB_GROUP_COLORS[s.tok.group] || '#808080';
        const pct = Math.round(((s.sim + 1) / 2) * 100);
        const valColor = s.sim > 0.85 ? 'var(--green)' : s.sim > 0.5 ? 'var(--orange)' : 'var(--text-secondary)';
        return `<div class="emb-sim-item">
            <div class="emb-legend-dot" style="background:${gc}"></div>
            <span style="font-family:monospace;font-size:11px;flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;color:var(--text-primary)">${escHtml(s.tok.token)}</span>
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px;flex-shrink:0">
                <span style="font-size:12px;font-weight:700;color:${valColor}">${s.sim.toFixed(3)}</span>
                <div class="emb-sim-bar-track"><div class="emb-sim-bar-fill" style="width:${pct}%"></div></div>
            </div>
        </div>`;
    }).join('') + '</div>';
    if (document.getElementById('simBody')) {
        document.getElementById('simBody').innerHTML = html;
    }
}

/* Create alias for escHtml to match repo-root viewer */
const escHtml = esc;

function loadEmbData(data) {
    embSt.data = data;
    embSt.selectedOrig = -1;
    if (document.getElementById('embEmpty')) document.getElementById('embEmpty').style.display = 'none';
    if (document.getElementById('embContent')) document.getElementById('embContent').style.display = '';
    if (document.getElementById('simPanel')) document.getElementById('simPanel').style.display = 'none';

    const s = data.stats || {};
    if (document.getElementById('embStats')) {
        document.getElementById('embStats').innerHTML =
            statCardHtml(escHtml(data.run_id || '—'), 'Run', '') +
            statCardHtml(data.step != null ? data.step : '—', 'Step', '') +
            statCardHtml(s.vocab_size != null ? s.vocab_size : '—', 'Vocab', '') +
            statCardHtml(s.embed_dim != null ? s.embed_dim : '—', 'Dim', '') +
            statCardHtml(s.nonzero_rows != null ? s.nonzero_rows : '—', 'Non-zero', '') +
            statCardHtml(s.min != null ? s.min.toFixed(4) : '—', 'Min', '') +
            statCardHtml(s.max != null ? s.max.toFixed(4) : '—', 'Max', '') +
            statCardHtml(s.std != null ? s.std.toFixed(4) : '—', 'Std', '');
    }

    const groups = [...new Set(data.vocab.map(v => v.group))];
    if (document.getElementById('embLegend')) {
        document.getElementById('embLegend').innerHTML =
            '<span style="font-size:12px;color:var(--text-muted)">Groups:</span>' +
            groups.map(g => {
                const cnt = data.vocab.filter(v => v.group === g).length;
                return `<div class="emb-legend-item">
                    <div class="emb-legend-dot" style="background:${EMB_GROUP_COLORS[g] || '#808080'}"></div>
                    <span>${g}</span>
                    <span style="color:var(--text-muted)">(${cnt})</span>
                </div>`;
            }).join('');
    }

    drawEmbHeatmap();
}

/* ═══════════════════════════════════════════════════════════════════════
   🔭 Attention Viewer — ported from repo-root dataset_viewer.html
   ═══════════════════════════════════════════════════════════════════════ */

const ATTN_LAYER_COLORS = ['#07adf8', '#47b475', '#9b59b6', '#e74c3c', '#ffb400', '#1abc9c'];

/* Map attention weight (0–1) to [r, g, b] for the chosen colourmap.
   Supports: 'orange', 'blue', 'green', 'heatmap'. */
function attnColor(v, cmap) {
    const t = Math.max(0, Math.min(1, v));
    if (cmap === 'heatmap') {
        const blue   = [7, 100, 248];
        const mid    = [240, 240, 240];
        const orange = [255, 160, 0];
        if (t < 0.5) {
            const s = t * 2;
            return [Math.round(blue[0] + (mid[0] - blue[0]) * s),
                    Math.round(blue[1] + (mid[1] - blue[1]) * s),
                    Math.round(blue[2] + (mid[2] - blue[2]) * s)];
        }
        const s = (t - 0.5) * 2;
        return [Math.round(mid[0] + (orange[0] - mid[0]) * s),
                Math.round(mid[1] + (orange[1] - mid[1]) * s),
                Math.round(mid[2] + (orange[2] - mid[2]) * s)];
    }
    const targets = {
        orange: [255, 180, 0],
        blue:   [7, 173, 248],
        green:  [71, 180, 117],
    };
    const col = targets[cmap] || targets.orange;
    return [Math.round(col[0] * t), Math.round(col[1] * t), Math.round(col[2] * t)];
}

const attnSt = {
    data: null,         // loaded attention.json
    seqIdx: 0,          // selected sequence index
    activeCard: null,   // currently expanded layer+head key
};

function loadAttnData(data) {
    attnSt.data = data;
    attnSt.seqIdx = 0;
    attnSt.activeCard = null;

    if (document.getElementById('attnEmpty')) document.getElementById('attnEmpty').style.display = 'none';
    if (document.getElementById('attnContent')) document.getElementById('attnContent').style.display = '';

    const cfg = data.config || {};
    const seq0 = (data.sequences && data.sequences[0]) || {};
    const seq0L0 = (seq0.layers && seq0.layers[0]) || {};
    if (document.getElementById('attnMeta')) {
        document.getElementById('attnMeta').innerHTML =
            statCardHtml(esc(data.run_id || '—'), 'Run', '') +
            statCardHtml(data.step != null ? data.step : '—', 'Step', '') +
            statCardHtml(cfg.num_layers || (seq0.layers ? seq0.layers.length : '—'), 'Layers', '') +
            statCardHtml(cfg.num_heads || (seq0L0.heads ? seq0L0.heads.length : '—'), 'Heads', '') +
            statCardHtml(cfg.num_kv_heads || '—', 'KV Heads', '') +
            statCardHtml(cfg.head_dim || '—', 'Head Dim', '') +
            statCardHtml(data.sequences ? data.sequences.length : 0, 'Sequences', '');
    }
    renderAttnSeqList();
    renderAttnTokenChips();
    renderAttnMain();
}

function renderAttnSeqList() {
    if (!attnSt.data) return;
    const el = document.getElementById('attnSeqList');
    if (!el) return;
    const seqs = attnSt.data.sequences;
    el.innerHTML = seqs.map((s, i) => {
        const cls = 'attn-seq-btn' + (s.split ? ' attn-split-' + s.split : '') + (i === attnSt.seqIdx ? ' active' : '');
        const label = (s.label || s.id || 'Seq ' + (i+1));
        return '<button class="' + cls + '" data-idx="' + i + '"><strong>' + esc(s.split || '') + '</strong> · ' + esc(label.slice(0,40)) + ' <span style="color:var(--text-muted)">(L=' + s.tokens.length + ')</span></button>';
    }).join('');
    el.querySelectorAll('.attn-seq-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            attnSt.seqIdx = parseInt(btn.dataset.idx);
            attnSt.activeCard = null;
            document.getElementById('attnDetailPanel').style.display = 'none';
            renderAttnSeqList();
            renderAttnTokenChips();
            renderAttnMain();
        });
    });
}

function renderAttnTokenChips() {
    const el = document.getElementById('attnTokenChips');
    if (!el || !attnSt.data) return;
    const seq = attnSt.data.sequences[attnSt.seqIdx];
    if (!seq || !seq.tokens) return;
    const chips = seq.tokens.map((t, i) => {
        const isPred = seq.top_preds && i > 0 && seq.top_preds[i-1] === seq.token_ids[i];
        const dot = isPred ? '<span style="color:var(--green);margin-right:3px" title="next-tok correct">✓</span>' : '';
        return '<span class="attn-tok-chip" title="id=' + escAttr(seq.token_ids ? seq.token_ids[i] : i) + '">' + dot + esc(t) + '</span>';
    }).join('');
    el.innerHTML = '<div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">Tokens (L=' + seq.tokens.length + '):</div>' + chips;
}

/* ── Draw a single attention matrix onto a canvas ── */
function drawAttnMatrix(canvas, matrix, tokens, opts) {
    if (!canvas || !matrix || !tokens) return {};
    opts = opts || {};
    const cmap = opts.cmap || 'orange';
    const showBos = opts.showBos !== false;
    const labels = opts.labels !== false;
    const labelPx = opts.labelPx || 12;
    const maxSize = opts.maxSize || 600;

    const startTok = showBos ? 0 : 1;
    const toks = tokens.slice(startTok);
    const L = toks.length;
    if (!L) return {};
    const matSlice = matrix.slice(startTok).map(r => r.slice(startTok));

    const labelW = labels ? Math.min(labelPx * 7, 140) : 0;
    const labelH = labels ? labelPx + 4 : 0;
    const cellPx = opts.cellPx || Math.max(4, Math.min(24, Math.floor((maxSize - labelW) / (L || 1))));
    const W = labelW + L * cellPx;
    const H = labelH + L * cellPx;

    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#111'; ctx.fillRect(0, 0, W, H);

    const img = ctx.createImageData(W, H);
    const d = img.data;
    for (let qi = 0; qi < L; qi++) {
        for (let ki = 0; ki < L; ki++) {
            const v = matSlice[qi] && matSlice[qi][ki] || 0;
            const [r, g, b] = attnColor(v, cmap);
            const x0 = labelW + ki * cellPx;
            const y0 = labelH + qi * cellPx;
            for (let py = y0; py < y0 + cellPx; py++) {
                for (let px = x0; px < x0 + cellPx; px++) {
                    const i4 = (py * W + px) * 4;
                    d[i4] = r; d[i4+1] = g; d[i4+2] = b; d[i4+3] = 255;
                }
            }
        }
    }
    ctx.putImageData(img, 0, 0);

    if (labels && labelPx >= 8) {
        const fs = Math.min(labelPx - 1, 11);
        ctx.font = fs + 'px "JetBrains Mono", ui-monospace, monospace';
        for (let i = 0; i < L; i++) {
            const y = labelH + i * cellPx + cellPx / 2;
            const maxCh = Math.floor((labelW - 6) / (fs * 0.62));
            const label = toks[i].length > maxCh ? toks[i].slice(0, maxCh - 1) + '…' : toks[i];
            ctx.fillStyle = '#444'; ctx.fillRect(0, labelH + i * cellPx, labelW - 1, cellPx);
            ctx.fillStyle = '#c0c8d0'; ctx.textBaseline = 'middle'; ctx.fillText(label, 3, y);
        }
        ctx.save();
        for (let i = 0; i < L; i++) {
            const x = labelW + i * cellPx + cellPx / 2;
            const maxCh = Math.floor((labelH - 3) / (fs * 0.62));
            const label = toks[i].length > maxCh ? toks[i].slice(0, maxCh - 1) + '…' : toks[i];
            ctx.save(); ctx.translate(x, labelH - 2); ctx.rotate(-Math.PI / 2);
            ctx.fillStyle = '#c0c8d0'; ctx.textBaseline = 'bottom'; ctx.fillText(label, 0, 0);
            ctx.restore();
        }
        ctx.restore();
    }
    return { L, cellPx, labelW, labelH };
}

function drawAttnThumb(canvas, matrix, tokens, cmap, thumbPx) {
    const L = matrix.length;
    const px = Math.max(1, Math.floor(thumbPx / L));
    const side = L * px;
    canvas.width = side; canvas.height = side;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(side, side);
    const d = img.data;
    for (let qi = 0; qi < L; qi++) {
        for (let ki = 0; ki < L; ki++) {
            const v = matrix[qi] && matrix[qi][ki] || 0;
            const [r, g, b] = attnColor(v, cmap);
            const x0 = ki * px, y0 = qi * px;
            for (let py = y0; py < y0 + px; py++) {
                for (let px_ = x0; px_ < x0 + px; px_++) {
                    const i4 = (py * side + px_) * 4;
                    d[i4] = r; d[i4+1] = g; d[i4+2] = b; d[i4+3] = 255;
                }
            }
        }
    }
    ctx.putImageData(img, 0, 0);
}

function attnEntropy(row) {
    let h = 0;
    for (const v of row) { if (v > 1e-9) h -= v * Math.log2(v); }
    return h;
}

function avgMatrices(matrices) {
    if (!matrices || !matrices.length || !matrices[0]) return [[]];
    const L = matrices[0].length;
    const out = Array.from({length:L}, () => new Float32Array(L));
    for (const m of matrices)
        for (let i = 0; i < L; i++)
            for (let j = 0; j < L; j++)
                out[i][j] += m[i][j] / matrices.length;
    return out.map(r => Array.from(r));
}

function attnGetMatrix(layerData, headIdx, showBos) {
    if (!layerData || !layerData.heads || !layerData.heads[headIdx]) return [[]];
    let mat = layerData.heads[headIdx].attn;
    if (!mat || !mat.length) return [[]];
    if (!showBos) { mat = mat.slice(1).map(r => r.slice(1)); }
    return mat;
}

function renderAttnMain() {
    if (!attnSt.data) return;
    const seq = attnSt.data.sequences[attnSt.seqIdx];
    if (!seq || !seq.layers || !seq.layers.length) return;
    const view = document.getElementById('attnView') ? document.getElementById('attnView').value : 'grid';
    const cmap = document.getElementById('attnCmap') ? document.getElementById('attnCmap').value : 'orange';
    const thumbPx = document.getElementById('attnThumbPx') ? parseInt(document.getElementById('attnThumbPx').value) : 100;
    const showBos = document.getElementById('attnShowBos') ? document.getElementById('attnShowBos').checked : true;
    const tokens = showBos ? seq.tokens : seq.tokens.slice(1);
    const numL = seq.layers.length;
    const numH = (seq.layers[0] && seq.layers[0].heads) ? seq.layers[0].heads.length : 0;
    const wrap = document.getElementById('attnMainView');
    if (!wrap || !numH) return;

    if (view === 'grid') {
        wrap.innerHTML = '';
        const grid = document.createElement('div');
        grid.className = 'attn-grid';
        grid.style.gridTemplateColumns = 'repeat(' + numH + ', 1fr)';
        wrap.appendChild(grid);
        for (let li = 0; li < numL; li++) {
            for (let hi = 0; hi < numH; hi++) {
                const key = li + '-' + hi;
                const card = document.createElement('div');
                card.className = 'attn-head-card' + (attnSt.activeCard === key ? ' active' : '');
                card.innerHTML = '<div class="attn-head-label l' + li + '">L' + li + ' · H' + hi + '</div><div class="attn-mini-wrap"><canvas></canvas></div>';
                const cv = card.querySelector('canvas');
                const mat = attnGetMatrix(seq.layers[li], hi, showBos);
                drawAttnThumb(cv, mat, tokens, cmap, thumbPx);
                card.addEventListener('click', () => {
                    attnSt.activeCard = (attnSt.activeCard === key) ? null : key;
                    if (attnSt.activeCard) openAttnDetail(seq, li, hi, tokens, cmap, showBos);
                    else document.getElementById('attnDetailPanel').style.display = 'none';
                    renderAttnMain();
                });
                grid.appendChild(card);
            }
        }
        return;
    }

    if (view === 'avg') {
        wrap.innerHTML = '';
        const row = document.createElement('div');
        row.className = 'attn-avg-row';
        row.style.gridTemplateColumns = 'repeat(' + numL + ', 1fr)';
        wrap.appendChild(row);
        for (let li = 0; li < numL; li++) {
            const matrices = seq.layers[li].heads.map((_, hi) => attnGetMatrix(seq.layers[li], hi, showBos));
            const avg = avgMatrices(matrices);
            const lc = ATTN_LAYER_COLORS[li % ATTN_LAYER_COLORS.length];
            const card = document.createElement('div');
            card.className = 'attn-avg-card';
            card.innerHTML = '<div class="attn-head-label" style="color:' + lc + '">Layer ' + li + ' — avg ' + numH + ' heads</div><div class="attn-mini-wrap"><canvas></canvas></div>';
            const cv = card.querySelector('canvas');
            drawAttnThumb(cv, avg, tokens, cmap, 200);
            card.style.cursor = 'pointer';
            card.addEventListener('click', () => openAttnDetailMatrix(avg, tokens, 'Layer ' + li + ' — averaged', cmap));
            row.appendChild(card);
        }
        return;
    }

    if (view === 'entropy') {
        let html = '<div style="display:grid;gap:10px">';
        const maxLogL = Math.log2(tokens.length || 1);
        for (let li = 0; li < numL; li++) {
            const lc = ATTN_LAYER_COLORS[li % ATTN_LAYER_COLORS.length];
            html += '<div style="background:var(--surface);border-radius:8px;padding:14px;border:1px solid var(--border)"><div style="font-size:12px;font-weight:700;color:' + lc + ';margin-bottom:10px">Layer ' + li + '</div><div style="display:grid;grid-template-columns:repeat(' + numH + ',1fr);gap:8px">';
            for (let hi = 0; hi < numH; hi++) {
                const mat = attnGetMatrix(seq.layers[li], hi, showBos);
                const entropies = mat.map(row => attnEntropy(row));
                const avgEnt = entropies.reduce((a, b) => a + b, 0) / entropies.length;
                const pct = Math.min(100, (avgEnt / maxLogL) * 100);
                html += '<div style="background:var(--bg);border-radius:6px;padding:10px"><div style="font-size:11px;font-weight:700;color:var(--text-secondary);margin-bottom:6px">H' + hi + '</div><div style="font-size:16px;font-weight:800;color:' + (pct > 70 ? 'var(--blue)' : pct > 40 ? 'var(--orange)' : 'var(--green)') + '">' + avgEnt.toFixed(2) + '</div><div style="font-size:10px;color:var(--text-muted)">bits / max ' + maxLogL.toFixed(1) + '</div><div class="attn-entropy-bar"><div class="attn-entropy-fill" style="width:' + pct + '%;background:' + (pct > 70 ? 'var(--blue)' : 'var(--orange)') + '"></div></div><div style="margin-top:6px">';
                for (let ti = 0; ti < tokens.length; ti++) {
                    const ep = Math.min(1, entropies[ti] / maxLogL);
                    const c = attnColor(1 - ep, 'orange');
                    html += '<div title="' + escAttr(tokens[ti]) + ': ' + entropies[ti].toFixed(3) + ' bits" style="display:inline-block;width:10px;height:10px;margin:1px;border-radius:2px;background:rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')"></div>';
                }
                html += '</div></div>';
            }
            html += '</div></div>';
        }
        html += '</div>';
        wrap.innerHTML = html;
    }
}

function openAttnDetail(seq, li, hi, tokens, cmap, showBos) {
    const mat = attnGetMatrix(seq.layers[li], hi, showBos);
    openAttnDetailMatrix(mat, tokens, 'Layer ' + li + ' · Head ' + hi, cmap);
}

function openAttnDetailMatrix(mat, tokens, label, cmap) {
    const panel = document.getElementById('attnDetailPanel');
    if (!panel) return;
    const labelEl = document.getElementById('attnDetailLabel');
    if (labelEl) labelEl.textContent = label;
    const cv = document.getElementById('attnDetailCanvas');
    if (!cv || !mat || !mat.length) return;
    const L = mat.length;
    const cellPx = L <= 8 ? 48 : L <= 16 ? 32 : L <= 32 ? 18 : L <= 64 ? 12 : 8;
    const labelPx = cellPx >= 14 ? 12 : 9;
    drawAttnMatrix(cv, mat, tokens, { cmap: cmap, cellPx: cellPx, labels: true, labelPx: labelPx });

    cv.onmousemove = function(e) {
        const rect = cv.getBoundingClientRect();
        const x = e.clientX - rect.left, y = e.clientY - rect.top;
        const lw = Math.min(labelPx * 7, 140), lh = labelPx + 4;
        if (x < lw || y < lh) { attnTip.style.display = 'none'; return; }
        const ki = Math.floor((x - lw) / cellPx), qi = Math.floor((y - lh) / cellPx);
        if (qi >= 0 && qi < L && ki >= 0 && ki < L && mat[qi]) {
            const w = mat[qi][ki] != null ? mat[qi][ki] : 0;
            attnTip.innerHTML = '<span style="color:var(--orange)">from</span> <strong>' + esc(tokens[qi] || '') + '</strong><br><span style="color:var(--orange)">to</span> <strong>' + esc(tokens[ki] || '') + '</strong><br><span style="color:var(--text-muted)">weight:</span> <strong>' + w.toFixed(5) + '</strong>';
            attnTip.style.display = 'block';
            attnTip.style.left = (e.clientX + 14) + 'px';
            attnTip.style.top = (e.clientY - 20) + 'px';
        }
    };
    cv.onmouseleave = function() { attnTip.style.display = 'none'; };
    panel.style.display = '';
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

const attnTip = (function() {
    const el = document.createElement('div');
    el.className = 'attn-tooltip';
    document.body.appendChild(el);
    return el;
})();

function renderEmbeddings() {
    const panel = document.getElementById('panel-embeddings');
    let html = `
        <div class="subhead">🧬 Token Embeddings</div>
        <div class="subnote">Interactive embedding heatmaps with similarity search and clustering analysis.</div>
        
        <div class="emb-load-bar">
            <span>📁 Load embeddings.json:</span>
            <input type="file" id="embFileInput" accept=".json" style="font-size:0.8rem">
            <button id="btnEmbDemo" style="padding:4px 12px;font-size:0.8rem">Demo</button>
            <button id="btnEmbClear" style="padding:4px 12px;font-size:0.8rem">Clear</button>
        </div>
        
        <div id="embEmpty">
            <div style="text-align:center;padding:3rem;color:var(--text-muted)">
                <div style="font-size:2rem;margin-bottom:1rem">📊</div>
                <div>No embeddings loaded</div>
                <div style="font-size:0.8rem;margin-top:0.5rem">Load embeddings.json or try the demo</div>
            </div>
        </div>
        
        <div id="embContent" style="display:none">
            <div class="emb-stats-grid" id="embStats"></div>
            
            <div class="emb-controls">
                <label>Sort: <select id="embSort"><option value="id">ID</option><option value="group">Group</option><option value="norm">L2 Norm</option><option value="sim">Similarity</option></select></label>
                <label>Norm: <select id="embNorm"><option value="global">Global</option><option value="col">Column</option><option value="row">Row</option></select></label>
                <label>Row px: <input type="range" id="embRowPx" min="8" max="32" value="14" style="width:80px"></label>
            </div>
            
            <div class="emb-legend" id="embLegend"></div>
            
            <div class="emb-scroll-wrap">
                <canvas id="embCanvas"></canvas>
            </div>
            
            <div class="emb-colorbar-row">
                <span class="emb-cb-label" id="embCbMin">0</span>
                <canvas id="embColorbar"></canvas>
                <span class="emb-cb-label" id="embCbMax">1</span>
                <span class="emb-cb-label" id="embCbNote" style="margin-left:10px;flex:1"></span>
            </div>
            
            <div id="simPanel" style="display:none;margin-top:1rem;padding:1rem;background:var(--dark-card);border-radius:8px">
                <div style="font-weight:700;margin-bottom:0.5rem">Most similar to: <span id="simTargetLabel" style="color:var(--orange)">—</span></div>
                <div id="simBody"></div>
            </div>
        </div>
    `;
    
    panel.innerHTML = html;
    
    // Wire up events
    if (document.getElementById('embFileInput')) {
        document.getElementById('embFileInput').addEventListener('change', async e => {
            const f = e.target.files[0]; if (!f) return;
            try {
                const text = await f.text();
                const data = JSON.parse(text);
                if (!data.matrix || !data.vocab) throw new Error('Not a valid embeddings.json');
                loadEmbData(data);
                document.querySelector('.tab[data-tab="embeddings"]').click();
            } catch (err) {
                alert('Failed to load embeddings.json:\\n' + err.message);
            }
            e.target.value = '';
        });
    }
    
    if (document.getElementById('btnEmbDemo')) {
        document.getElementById('btnEmbDemo').addEventListener('click', () => {
            loadEmbData(generateEmbDemo());
        });
    }
    
    if (document.getElementById('btnEmbClear')) {
        document.getElementById('btnEmbClear').addEventListener('click', () => {
            embSt.data = null; embSt.selectedOrig = -1;
            if (document.getElementById('embEmpty')) document.getElementById('embEmpty').style.display = '';
            if (document.getElementById('embContent')) document.getElementById('embContent').style.display = 'none';
        });
    }
    
    // Wire up controls
    ['embSort', 'embNorm', 'embRowPx'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', drawEmbHeatmap);
            el.addEventListener('input', drawEmbHeatmap);
        }
    });
    
    // Auto-load if data is embedded
    if (window.CK_EMBEDDINGS && CK_EMBEDDINGS !== null) {
        loadEmbData(CK_EMBEDDINGS);
    }

    // Wire up canvas events after rendering
    setTimeout(() => {
        const canvas = document.getElementById('embCanvas');
        if (!canvas) return;

        canvas.addEventListener('mousemove', e => {
            if (!embSt.data || !embSt.sortedIndices.length) return;
            const rect = e.target.getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;
            const vi = Math.floor(cy / embSt.rowPx);
            if (vi < 0 || vi >= embSt.sortedIndices.length) { embTip.style.display = 'none'; return; }
            const origIdx = embSt.sortedIndices[vi];
            const tok = embSt.data.vocab[origIdx];
            const gc  = EMB_GROUP_COLORS[tok.group] || '#808080';
            const row = embSt.data.matrix[origIdx];
            const l2  = Math.sqrt(row.reduce((s, v) => s + v * v, 0));
            let dimInfo = '';
            if (cx >= embSt.labelW) {
                const di = Math.floor((cx - embSt.labelW) / embSt.colPx);
                if (di >= 0 && di < row.length) {
                    dimInfo = `<br><span style="color:var(--text-muted)">dim[${di}]:</span> <strong>${row[di].toFixed(5)}</strong>`;
                }
            }
            embTip.innerHTML =
                `<span style="color:${gc};font-weight:700;font-family:monospace">${escHtml(tok.token)}</span><br>` +
                `<span style="color:var(--text-muted)">id:</span> ${tok.id} &nbsp; <span style="color:var(--text-muted)">group:</span> ${tok.group}<br>` +
                `<span style="color:var(--text-muted)">L2 norm:</span> ${l2.toFixed(4)}` + dimInfo;
            embTip.style.display = 'block';
            embTip.style.left = (e.clientX + 14) + 'px';
            embTip.style.top  = (e.clientY - 20) + 'px';
        });

        canvas.addEventListener('mouseleave', () => { embTip.style.display = 'none'; });

        canvas.addEventListener('click', e => {
            if (!embSt.data || !embSt.sortedIndices.length) return;
            const rect = e.target.getBoundingClientRect();
            const vi = Math.floor((e.clientY - rect.top) / embSt.rowPx);
            if (vi < 0 || vi >= embSt.sortedIndices.length) return;
            const origIdx = embSt.sortedIndices[vi];
            embSt.selectedOrig = origIdx;
            renderSimPanel(origIdx);
            drawEmbHeatmap();
        });
    }, 100);
}

function renderAttention() {
    const panel = document.getElementById('panel-attention');
    let html = `
        <div class="subhead">🔭 Attention Patterns</div>
        <div class="subnote">Visualize attention heads across layers with per-sequence analysis and entropy metrics.</div>
        <div class="attn-load-bar">
            <span>📁 Load attention.json:</span>
            <input type="file" id="attnFileInput" accept=".json" style="font-size:0.8rem">
            <button id="btnAttnDemo" style="padding:4px 12px;font-size:0.8rem">Demo</button>
            <button id="btnAttnClear" style="padding:4px 12px;font-size:0.8rem">Clear</button>
        </div>
        <div id="attnEmpty">
            <div style="text-align:center;padding:3rem;color:var(--text-muted)">
                <div style="font-size:2rem;margin-bottom:1rem">🔭</div>
                <div>No attention data loaded</div>
                <div style="font-size:0.8rem;margin-top:0.5rem">Load attention.json or click Demo</div>
            </div>
        </div>
        <div id="attnContent" style="display:none">
            <div class="emb-stats-grid" id="attnMeta"></div>
            <div style="margin-bottom:6px;font-size:12px;color:var(--text-muted)">Sequences — click to select:</div>
            <div class="attn-seq-list" id="attnSeqList"></div>
            <div class="attn-controls">
                <span style="color:var(--text-secondary);font-size:12px">View:</span>
                <select id="attnView" class="search-box"><option value="grid">All heads (grid)</option><option value="avg">Average per layer</option><option value="entropy">Attention entropy</option></select>
                <span style="color:var(--text-secondary);font-size:12px">Colour:</span>
                <select id="attnCmap" class="search-box"><option value="orange">Dark → Orange</option><option value="blue">Dark → Blue</option><option value="green">Dark → Green</option><option value="heatmap">Blue → White → Orange</option></select>
                <span style="color:var(--text-secondary);font-size:12px">Thumb px:</span>
                <select id="attnThumbPx" class="search-box"><option value="80">80</option><option value="100" selected>100</option><option value="130">130</option><option value="160">160</option></select>
                <label style="font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:5px;cursor:pointer"><input type="checkbox" id="attnShowBos" checked> show BOS</label>
            </div>
            <div id="attnTokenChips" style="margin-bottom:14px"></div>
            <div class="attn-detail-panel" id="attnDetailPanel">
                <div class="attn-detail-title"><span id="attnDetailLabel"></span>
                    <button id="btnAttnDetailClose" style="margin-left:auto;padding:2px 10px;font-size:11px;background:var(--surface);border:1px solid var(--border);color:var(--text-primary);border-radius:4px;cursor:pointer">✕</button>
                </div>
                <div class="attn-canvas-scroll"><canvas id="attnDetailCanvas"></canvas></div>
                <div style="margin-top:8px;font-size:11px;color:var(--text-muted)">Rows = query (from) · Columns = key (to) · Upper triangle is masked (causal)</div>
            </div>
            <div id="attnMainView"></div>
        </div>`;
    panel.innerHTML = html;

    // Wire up view/cmap/thumb controls to re-render
    ['attnView','attnCmap','attnThumbPx'].forEach(function(id) {
        var el = document.getElementById(id);
        if (el) el.addEventListener('change', function() { renderAttnMain(); });
    });
    var bosEl = document.getElementById('attnShowBos');
    if (bosEl) bosEl.addEventListener('change', function() { renderAttnMain(); });

    var closeBtn = document.getElementById('btnAttnDetailClose');
    if (closeBtn) closeBtn.addEventListener('click', function() {
        document.getElementById('attnDetailPanel').style.display = 'none';
        attnSt.activeCard = null;
        renderAttnMain();
    });
    
    // Wire up events
    if (document.getElementById('attnFileInput')) {
        document.getElementById('attnFileInput').addEventListener('change', async e => {
            const f = e.target.files[0]; if (!f) return;
            try {
                const text = await f.text();
                const data = JSON.parse(text);
                if (!data.sequences) throw new Error('Not a valid attention.json');
                loadAttnData(data);
                document.querySelector('.tab[data-tab="attention"]').click();
            } catch (err) {
                alert('Failed to load attention.json:\\n' + err.message);
            }
            e.target.value = '';
        });
    }
    
    if (document.getElementById('btnAttnDemo')) {
        document.getElementById('btnAttnDemo').addEventListener('click', () => {
            loadAttnData(generateAttnDemo());
        });
    }
    
    if (document.getElementById('btnAttnClear')) {
        document.getElementById('btnAttnClear').addEventListener('click', () => {
            attnSt.data = null;
            if (document.getElementById('attnEmpty')) document.getElementById('attnEmpty').style.display = '';
            if (document.getElementById('attnContent')) document.getElementById('attnContent').style.display = 'none';
        });
    }
    
    // Auto-load if data is embedded
    if (window.CK_ATTENTION && CK_ATTENTION !== null) {
        loadAttnData(CK_ATTENTION);
    }
}

/* Demo data generators */
function generateEmbDemo() {
    const vocab = [
        {id:0,token:'<|unk|>',   group:'system'},
        {id:1,token:'<|bos|>',   group:'system'},
        {id:2,token:'<|eos|>',   group:'system'},
        {id:3,token:'<|pad|>',   group:'system'},
        {id:4,token:'[shape:circle]',   group:'prompt'},
        {id:5,token:'[shape:rect]',     group:'prompt'},
        {id:6,token:'[shape:triangle]', group:'prompt'},
        {id:7,token:'[color:red]',      group:'prompt'},
        {id:8,token:'[color:blue]',     group:'prompt'},
        {id:9,token:'[color:green]',    group:'prompt'},
        {id:10,token:'[size:small]',    group:'prompt'},
        {id:11,token:'[size:big]',      group:'prompt'},
        {id:12,token:'[OUT]',           group:'prompt'},
        {id:13,token:'[svg]',           group:'svg_structure'},
        {id:14,token:'[/svg]',          group:'svg_structure'},
        {id:15,token:'[circle]',        group:'svg_structure'},
        {id:16,token:'[rect]',          group:'svg_structure'},
        {id:17,token:'[polygon]',       group:'svg_structure'},
        {id:18,token:'[fill:red]',      group:'svg_style'},
        {id:19,token:'[fill:blue]',     group:'svg_style'},
        {id:20,token:'[fill:green]',    group:'svg_style'},
        {id:21,token:'[stroke:black]',  group:'svg_style'},
        {id:22,token:'[sw:2]',          group:'svg_style'},
        {id:23,token:'[cx:64]',         group:'svg_attr'},
        {id:24,token:'[cy:64]',         group:'svg_attr'},
        {id:25,token:'[r:18]',          group:'svg_attr'},
        {id:26,token:'[r:36]',          group:'svg_attr'},
        {id:27,token:'[x:42]',          group:'svg_attr'},
        {id:28,token:'[y:48]',          group:'svg_attr'},
        {id:29,token:'[width:44]',      group:'svg_attr'},
        {id:30,token:'[height:32]',     group:'svg_attr'},
        {id:31,token:'[points:64,34|36,86|92,86]', group:'svg_attr'},
    ];
    const D = 32;
    const V = vocab.length;
    /* Each group occupies a subspace; noise added for realism */
    const groupDims = {
        system:        [0,1,2,3],
        prompt:        [4,5,6,7,8,9],
        svg_structure: [10,11,12,13,14,15],
        svg_style:     [16,17,18,19,20,21],
        svg_attr:      [22,23,24,25,26,27,28,29],
    };
    function randn() {
        return Math.sqrt(-2*Math.log(Math.random()+1e-9)) * Math.cos(2*Math.PI*Math.random());
    }
    const matrix = vocab.map(tok => {
        const dims = groupDims[tok.group] || [];
        return Array.from({length:D}, (_,j) =>
            (dims.includes(j) ? randn() * 0.45 + 0.3 : randn() * 0.08) * 0.6
        );
    });
    const flat = matrix.flat();
    return {
        format: 'ck-embeddings.v1', run_id: 'demo', step: null,
        tensor: 'token_emb', shape: [V, D],
        vocab, matrix,
        stats: {
            min: Math.min(...flat), max: Math.max(...flat),
            mean: flat.reduce((a,b)=>a+b,0)/flat.length,
            std: Math.sqrt(flat.reduce((a,b)=>a+b*b,0)/flat.length - (flat.reduce((a,b)=>a+b,0)/flat.length)**2),
            vocab_size: V, embed_dim: D, nonzero_rows: V,
        },
    };
}

function generateAttnDemo() {
    function softmax(arr) { const m = Math.max(...arr); const e = arr.map(x => Math.exp(x-m)); const s = e.reduce((a,b)=>a+b,0); return e.map(x => x/s); }
    function randn() { return Math.sqrt(-2*Math.log(Math.random()+1e-9)) * Math.cos(2*Math.PI*Math.random()); }
    const tokens = ['<|bos|>', '[shape:circle]', '[color:red]', '[OUT]', '[svg]', '[circle]', '[/svg]', '<|eos|>'];
    const L = tokens.length;
    const layers = [];
    for (let li = 0; li < 2; li++) {
        const heads = [];
        for (let hi = 0; hi < 4; hi++) {
            const attn = [];
            for (let qi = 0; qi < L; qi++) {
                const logits = [];
                for (let ki = 0; ki < L; ki++) {
                    logits.push(ki <= qi ? randn() * (li === 0 ? 1.5 : 2.0) + (ki === qi ? 1.0 : 0) + (ki === 0 ? 0.5 : 0) : -1e9);
                }
                attn.push(softmax(logits));
            }
            heads.push({head: hi, attn: attn});
        }
        layers.push({layer: li, heads: heads});
    }
    return {
        format: 'ck-attention.v1', run_id: 'demo', step: null,
        config: {num_layers: 2, num_heads: 4, num_kv_heads: 2, head_dim: 16},
        sequences: [{
            id: 'demo_seq_1', label: 'circle red small', split: 'seen',
            tokens: tokens, token_ids: [1,4,7,12,13,15,14,2], top_preds: null,
            layers: layers
        }]
    };
}

// ── Canvas Chart Engine (zero-dependency) ────────────────────────────────────

function drawCanvasChart(canvas, series, opts) {
    opts = opts || {};
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width || 400;
    const H = rect.height || 180;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';

    var margin = { top: 24, right: 14, bottom: 32, left: 56 };
    var pw = W - margin.left - margin.right;
    var ph = H - margin.top - margin.bottom;
    if (pw <= 0 || ph <= 0) return;

    var xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (var si = 0; si < series.length; si++) {
        var sd = series[si].data;
        for (var pi = 0; pi < sd.length; pi++) {
            var pt = sd[pi];
            if (!Number.isFinite(pt.x) || !Number.isFinite(pt.y)) continue;
            if (pt.x < xMin) xMin = pt.x;
            if (pt.x > xMax) xMax = pt.x;
            if (pt.y < yMin) yMin = pt.y;
            if (pt.y > yMax) yMax = pt.y;
        }
    }
    if (!Number.isFinite(xMin) || xMin === xMax) return;
    if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }
    if (opts.yIncludeZero && yMin > 0) yMin = 0;

    function toX(v) { return margin.left + ((v - xMin) / (xMax - xMin)) * pw; }
    function toY(v) { return margin.top + (1 - (v - yMin) / (yMax - yMin)) * ph; }

    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 0.5;
    var yTicks = 5, xTicks = 5;
    for (var i = 0; i <= yTicks; i++) {
        var y = margin.top + (i / yTicks) * ph;
        ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + pw, y); ctx.stroke();
    }
    for (var i = 0; i <= xTicks; i++) {
        var x = margin.left + (i / xTicks) * pw;
        ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + ph); ctx.stroke();
    }

    // Axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    for (var i = 0; i <= yTicks; i++) {
        var v = yMax - (i / yTicks) * (yMax - yMin);
        ctx.fillText(fmtAxisVal(v), margin.left - 4, margin.top + (i / yTicks) * ph + 3);
    }
    ctx.textAlign = 'center';
    for (var i = 0; i <= xTicks; i++) {
        var v = xMin + (i / xTicks) * (xMax - xMin);
        ctx.fillText(fmtAxisVal(v), margin.left + (i / xTicks) * pw, H - margin.bottom + 16);
    }

    if (opts.title) {
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        ctx.font = 'bold 11px JetBrains Mono, monospace';
        ctx.textAlign = 'left';
        ctx.fillText(opts.title, margin.left, margin.top - 8);
    }
    if (opts.xLabel) {
        ctx.fillStyle = 'rgba(255,255,255,0.4)';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(opts.xLabel, margin.left + pw / 2, H - 2);
    }

    // Draw series lines
    for (var si = 0; si < series.length; si++) {
        var s = series[si];
        var pts = s.data.filter(function(p) { return Number.isFinite(p.x) && Number.isFinite(p.y); });
        if (pts.length < 2) continue;
        ctx.strokeStyle = s.color || '#ffb400';
        ctx.lineWidth = s.width || 1.5;
        ctx.globalAlpha = s.alpha || 1.0;
        ctx.beginPath();
        ctx.moveTo(toX(pts[0].x), toY(pts[0].y));
        for (var pi = 1; pi < pts.length; pi++) {
            ctx.lineTo(toX(pts[pi].x), toY(pts[pi].y));
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;
    }

    // Legend
    if (series.length > 1) {
        var lx = margin.left + pw - 10;
        ctx.textAlign = 'right';
        ctx.font = '9px JetBrains Mono, monospace';
        for (var si = series.length - 1; si >= 0; si--) {
            var s = series[si];
            if (!s.label) continue;
            var tw = ctx.measureText(s.label).width;
            ctx.fillStyle = s.color || '#ffb400';
            ctx.fillRect(lx - tw - 14, margin.top + 2, 8, 8);
            ctx.fillStyle = 'rgba(255,255,255,0.7)';
            ctx.fillText(s.label, lx, margin.top + 10);
            lx -= tw + 24;
        }
    }
}

function fmtAxisVal(v) {
    if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(1) + 'K';
    if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(1);
    if (Number.isInteger(v)) return v.toString();
    return v.toPrecision(3);
}

// ── Training Dynamics Tab ────────────────────────────────────────────────────

function renderTraining() {
    var panel = document.getElementById('panel-training');
    if (!panel) return;
    var td = (typeof CK_TRAINING !== 'undefined') ? CK_TRAINING : null;

    if (!td || !td.available) {
        panel.innerHTML = '<div class="subnote" style="padding:2rem;text-align:center;">'
            + '<h3 style="color:var(--orange);">📈 Training Dynamics</h3>'
            + '<p>No training telemetry found for this run.</p>'
            + '<p style="font-size:0.8rem;color:var(--text-muted);">Generate with: '
            + '<code>python3 version/v7/scripts/ck_run_v7.py --run &lt;DIR&gt;</code></p>'
            + '</div>';
        return;
    }

    var summary = td.summary || {};
    var lossCurve = td.loss_curve || [];
    var gradNorms = td.grad_norms || [];
    var stepProfile = td.step_profile || [];
    var totalSteps = td.total_steps || lossCurve.length;

    var html = '<div style="padding:1rem;">';
    html += '<h3 style="color:var(--orange);margin:0 0 0.5rem 0;">📈 Training Dynamics</h3>';
    html += '<p class="subnote" style="margin-bottom:1rem;">Loss convergence, gradient health, parity drift, and step timing from <code>training_loss_curve.json</code></p>';

    // Stat cards
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:0.5rem;margin-bottom:1.5rem;">';
    var cards = [
        { label: 'Start Loss', value: summary.start_loss != null ? summary.start_loss.toFixed(4) : '\u2014', color: '#f87171' },
        { label: 'Final Loss', value: summary.final_loss != null ? summary.final_loss.toFixed(4) : '\u2014', color: '#4ade80' },
        { label: 'PT Final', value: summary.final_loss_pt != null ? summary.final_loss_pt.toFixed(4) : '\u2014', color: '#60a5fa' },
        { label: 'Total Steps', value: totalSteps.toLocaleString(), color: '#fbbf24' },
        { label: 'Final LR', value: summary.final_lr != null ? summary.final_lr.toExponential(2) : '\u2014', color: '#a78bfa' }
    ];
    if (summary.start_loss && summary.final_loss && summary.final_loss > 0) {
        var ratio = summary.start_loss / summary.final_loss;
        cards.push({ label: 'Reduction', value: ratio.toFixed(1) + '\u00d7', color: '#2dd4bf' });
    }
    for (var ci = 0; ci < cards.length; ci++) {
        var c = cards[ci];
        html += '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:0.6rem;text-align:center;">'
            + '<div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">' + c.label + '</div>'
            + '<div style="font-size:1.2rem;font-weight:700;color:' + c.color + ';font-family:JetBrains Mono,monospace;">' + c.value + '</div>'
            + '</div>';
    }
    html += '</div>';

    // Chart canvases (only create if data exists)
    var chartDefs = [];
    if (lossCurve.length > 1) {
        chartDefs.push({ id: 'ck-loss-chart', h: 220 });
        chartDefs.push({ id: 'ck-parity-drift-chart', h: 160 });
    }
    if (lossCurve.length > 1 && lossCurve.some(function(s) { return s.lr != null && s.lr > 0; })) {
        chartDefs.push({ id: 'ck-lr-chart', h: 160 });
    }
    if (gradNorms.length > 1 || lossCurve.some(function(s) { return s.grad_norm != null && s.grad_norm > 0; })) {
        chartDefs.push({ id: 'ck-grad-chart', h: 160 });
    }
    if (lossCurve.some(function(s) { return s.forward_ms > 0; }) || stepProfile.length > 1) {
        chartDefs.push({ id: 'ck-timing-chart', h: 160 });
    }

    html += '<div style="display:grid;grid-template-columns:1fr;gap:1rem;">';
    for (var di = 0; di < chartDefs.length; di++) {
        var cd = chartDefs[di];
        html += '<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:6px;overflow:hidden;">'
            + '<canvas id="' + cd.id + '" style="width:100%;height:' + cd.h + 'px;display:block;"></canvas>'
            + '</div>';
    }
    html += '</div>';

    // Per-stage breakdown table
    var stages = {};
    var stageOrder = [];
    for (var i = 0; i < lossCurve.length; i++) {
        var s = lossCurve[i];
        var st = s.source_stage || 'unknown';
        if (!stages[st]) { stages[st] = { count: 0, first_loss: null, last_loss: null }; stageOrder.push(st); }
        stages[st].count++;
        if (stages[st].first_loss == null) stages[st].first_loss = s.loss_ck;
        stages[st].last_loss = s.loss_ck;
    }
    if (stageOrder.length > 0) {
        html += '<h4 style="color:var(--orange);margin:1.5rem 0 0.5rem 0;">Stage Breakdown</h4>';
        html += '<table style="width:100%;border-collapse:collapse;font-size:0.8rem;">';
        html += '<thead><tr style="border-bottom:1px solid rgba(255,255,255,0.1);">'
            + '<th style="text-align:left;padding:0.4rem;">Stage</th>'
            + '<th style="text-align:right;padding:0.4rem;">Steps</th>'
            + '<th style="text-align:right;padding:0.4rem;">Start Loss</th>'
            + '<th style="text-align:right;padding:0.4rem;">End Loss</th>'
            + '<th style="text-align:right;padding:0.4rem;">Reduction</th>'
            + '</tr></thead><tbody>';
        for (var si = 0; si < stageOrder.length; si++) {
            var sn = stageOrder[si];
            var stg = stages[sn];
            var red = (stg.first_loss && stg.last_loss && stg.last_loss > 0) ? (stg.first_loss / stg.last_loss).toFixed(1) + '\u00d7' : '\u2014';
            html += '<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">'
                + '<td style="padding:0.4rem;color:var(--orange);">' + esc(sn) + '</td>'
                + '<td style="text-align:right;padding:0.4rem;">' + stg.count + '</td>'
                + '<td style="text-align:right;padding:0.4rem;">' + (stg.first_loss != null ? stg.first_loss.toFixed(4) : '\u2014') + '</td>'
                + '<td style="text-align:right;padding:0.4rem;">' + (stg.last_loss != null ? stg.last_loss.toFixed(4) : '\u2014') + '</td>'
                + '<td style="text-align:right;padding:0.4rem;color:#4ade80;">' + red + '</td>'
                + '</tr>';
        }
        html += '</tbody></table>';
    }

    html += '</div>';
    panel.innerHTML = html;

    // ── Draw charts after DOM is ready ───────────────────────────────

    // 1. Loss Curve (CK red, PT blue)
    if (lossCurve.length > 1) {
        drawCanvasChart(document.getElementById('ck-loss-chart'),
            [{ label: 'CK', color: '#f87171', data: lossCurve.map(function(s) { return {x: s.step, y: s.loss_ck}; }) },
             { label: 'PyTorch', color: '#60a5fa', alpha: 0.7, data: lossCurve.map(function(s) { return {x: s.step, y: s.loss_pt}; }) }],
            { title: 'Loss Curve \u2014 CK (red) vs PyTorch (blue)', xLabel: 'step' });

        // 2. Parity Drift |CK - PT|
        var driftData = lossCurve
            .filter(function(s) { return s.loss_ck != null && s.loss_pt != null; })
            .map(function(s) { return {x: s.step, y: Math.abs(s.loss_ck - s.loss_pt)}; });
        if (driftData.length > 1) {
            drawCanvasChart(document.getElementById('ck-parity-drift-chart'),
                [{ label: '|CK \u2212 PT|', color: '#fbbf24', data: driftData }],
                { title: 'Parity Drift \u2014 |loss_ck \u2212 loss_pt| per step', xLabel: 'step', yIncludeZero: true });
        }
    }

    // 3. LR Schedule
    var lrData = lossCurve.filter(function(s) { return s.lr != null && s.lr > 0; }).map(function(s) { return {x: s.step, y: s.lr}; });
    if (lrData.length > 1) {
        drawCanvasChart(document.getElementById('ck-lr-chart'),
            [{ label: 'lr', color: '#a78bfa', data: lrData }],
            { title: 'Learning Rate Schedule', xLabel: 'step' });
    }

    // 4. Gradient Norms (prefer dedicated file, fallback to loss_curve field)
    var gnData = gradNorms.map(function(g) { return {x: g.step, y: g.norm}; }).filter(function(p) { return Number.isFinite(p.y) && p.y > 0; });
    if (gnData.length < 2) {
        gnData = lossCurve.filter(function(s) { return s.grad_norm != null && s.grad_norm > 0; }).map(function(s) { return {x: s.step, y: s.grad_norm}; });
    }
    if (gnData.length > 1) {
        drawCanvasChart(document.getElementById('ck-grad-chart'),
            [{ label: 'grad_norm', color: '#fbbf24', data: gnData }],
            { title: 'Gradient Norms over Training', xLabel: 'step', yIncludeZero: true });
    }

    // 5. Step Timing (CK ms green, PT ms blue)
    var timingCk = stepProfile.map(function(s) { return {x: s.step, y: s.ck_ms}; }).filter(function(p) { return Number.isFinite(p.y) && p.y > 0; });
    var timingPt = stepProfile.map(function(s) { return {x: s.step, y: s.pt_ms}; }).filter(function(p) { return Number.isFinite(p.y) && p.y > 0; });
    if (timingCk.length < 2) {
        timingCk = lossCurve.filter(function(s) { return s.forward_ms > 0; }).map(function(s) { return {x: s.step, y: (s.forward_ms || 0) + (s.backward_ms || 0) + (s.optimizer_ms || 0)}; });
    }
    var timingSeries = [];
    if (timingCk.length > 1) timingSeries.push({ label: 'CK ms', color: '#4ade80', data: timingCk });
    if (timingPt.length > 1) timingSeries.push({ label: 'PT ms', color: '#60a5fa', data: timingPt });
    if (timingSeries.length > 0) {
        drawCanvasChart(document.getElementById('ck-timing-chart'), timingSeries,
            { title: 'Step Timing \u2014 CK (green) vs PyTorch (blue)', xLabel: 'step', yIncludeZero: true });
    }
}

function renderAll() {
    const sections = [
        renderHeader, renderOverview, renderPreflight, renderGallery,
        renderTextSamples, renderTokenizer, renderVocabulary,
        renderClassification, renderBrowse, renderCandidates,
        renderQuality, renderEmbeddings, renderAttention, renderTraining, populateFilters
    ];
    for (const fn of sections) {
        try { fn(); } catch (e) { console.error('[dataset-viewer] ' + fn.name + ' failed:', e); }
    }
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
    ap = argparse.ArgumentParser(description="Build a standalone SVG dataset visualizer from a split-aware workspace")
    ap.add_argument("--workspace", required=True, help="Workspace root, e.g. version/v7/data/spec04 or RUN/dataset")
    ap.add_argument("--output", required=True, help="Output HTML file")
    args = ap.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    raw_inventory = _load_json_if_exists(workspace / "manifests" / "raw_assets_inventory.json") or {}
    normalized = _load_json_if_exists(workspace / "manifests" / "normalized_assets_manifest.json") or {}
    classified = _load_json_if_exists(workspace / "manifests" / "asset_classification_manifest.json") or {}

    # Structured-atoms fallback: synthesize from render catalog when no SVG manifests
    if not normalized.get("normalized_entries") and not classified.get("entries"):
        synth_norm, synth_cls = _synthesize_structured_atoms(workspace)
        if synth_norm:
            normalized = synth_norm
        if synth_cls:
            classified = synth_cls

    html_doc = build_html(workspace, raw_inventory, normalized, classified)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")
    print(f"[OK] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
