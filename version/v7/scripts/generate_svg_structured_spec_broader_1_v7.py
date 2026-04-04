#!/usr/bin/env python3
"""Generate a broader scene-DSL bootstrap dataset from current supported families."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bootstrap_spec_broader_1_comparison_gold_pack_v7 import CASES as COMPARISON_GOLD_CASES
from build_spec09_asset_alignment_report_v7 import GOLD_CASES as ASSET_GOLD_CASES
from build_spec09_compiler_validation_report_v7 import VALIDATION_CASES
from generate_svg_structured_spec06_v7 import (
    _build_tokenizer_payload,
    _ordered_domain_tokens,
    _write_lines,
    _write_tokenizer_artifacts,
)
from render_svg_structured_scene_spec09_v7 import render_structured_scene_spec09_svg


LAYOUTS = (
    "poster_stack",
    "comparison_span_chart",
    "pipeline_lane",
    "dashboard_cards",
    "dual_panel_compare",
    "timeline_flow",
    "table_analysis",
)
THEMES = ("infra_dark", "paper_editorial", "signal_glow")
TONES = ("amber", "green", "blue", "purple", "mixed")
DENSITIES = ("compact", "balanced", "airy")
SCENE_ATTR_KEYS = {
    "canvas",
    "layout",
    "theme",
    "tone",
    "frame",
    "density",
    "inset",
    "gap",
    "hero",
    "columns",
    "emphasis",
    "rail",
    "background",
    "connector",
    "topic",
}


@dataclass(frozen=True)
class SceneSeed:
    case_id: str
    source_asset: str
    title: str
    attrs: dict[str, str]
    components: tuple[str, ...]
    content_json: dict[str, Any] | None = None

    @property
    def layout(self) -> str:
        return str(self.attrs.get("layout") or "")

    @property
    def topic(self) -> str:
        return str(self.attrs.get("topic") or "")


@dataclass(frozen=True)
class Scene:
    seed: SceneSeed
    theme: str
    tone: str
    density: str

    @property
    def layout(self) -> str:
        return self.seed.layout

    @property
    def topic(self) -> str:
        return self.seed.topic

    @property
    def prompt_tokens(self) -> list[str]:
        base_attrs = self.seed.attrs
        return [
            "[task:svg]",
            f"[layout:{self.layout}]",
            f"[topic:{self.topic}]",
            f"[theme:{self.theme}]",
            f"[tone:{self.tone}]",
            f"[density:{self.density}]",
            f"[frame:{_frame(self, base_attrs)}]",
            f"[background:{_background(self, base_attrs)}]",
            "[OUT]",
        ]

    @property
    def prompt_text(self) -> str:
        return " ".join(self.prompt_tokens)


def _parse_case_scene(case_dsl: str) -> dict[str, Any]:
    attrs: dict[str, str] = {}
    components: list[str] = []
    for raw in str(case_dsl or "").splitlines():
        token = raw.strip()
        if not token or token in {"[scene]", "[/scene]"}:
            continue
        if token.startswith("[") and token.endswith("]") and ":" in token:
            key, value = token[1:-1].split(":", 1)
            if key in SCENE_ATTR_KEYS:
                attrs[key] = value
                continue
        components.append(token)
    return {"attrs": attrs, "components": components}


def _seed_from_asset_case(case: dict[str, Any]) -> SceneSeed | None:
    parsed = _parse_case_scene(str(case.get("dsl") or ""))
    attrs = dict(parsed["attrs"])
    layout = str(attrs.get("layout") or "")
    if layout not in LAYOUTS:
        return None
    return SceneSeed(
        case_id=str(case.get("asset") or case.get("title") or layout).replace(".svg", "").replace("-", "_"),
        source_asset=str(case.get("asset") or ""),
        title=str(case.get("title") or ""),
        attrs=attrs,
        components=tuple(str(item) for item in parsed["components"]),
        content_json=case.get("content") if isinstance(case.get("content"), dict) else None,
    )


def _comparison_pack_seed(case) -> SceneSeed:
    parsed = _parse_case_scene(str(case.scene_text))
    topic_map = {
        "performance-balance": "performance_balance",
        "cpu-gpu-analysis": "cpu_gpu_cost",
        "theory-of-constraints": "ethernet_equalizer",
    }
    attrs = dict(parsed["attrs"])
    attrs["topic"] = topic_map.get(str(case.case_id), attrs.get("topic", "comparison_span_chart_generic"))
    return SceneSeed(
        case_id=str(case.case_id).replace("-", "_"),
        source_asset=str(case.asset),
        title=str(case.asset).replace(".svg", "").replace("-", " "),
        attrs=attrs,
        components=tuple(str(item) for item in parsed["components"]),
        content_json=json.loads(json.dumps(case.content_json)),
    )


def _selected_validation_ids() -> set[str]:
    return {"dashboard_light", "dual_panel_glow", "timeline_flow", "table_analysis", "content_bound_compare"}


def _build_seeds() -> list[SceneSeed]:
    seeds: list[SceneSeed] = []
    seen_ids: set[str] = set()

    selected_asset_names = {
        "memory-reality-infographic.svg",
        "pipeline-overview.svg",
        "training-intuition-map.svg",
    }
    for row in ASSET_GOLD_CASES:
        if str(row.get("asset") or "") not in selected_asset_names:
            continue
        seed = _seed_from_asset_case(row)
        if seed is None or seed.case_id in seen_ids:
            continue
        seen_ids.add(seed.case_id)
        seeds.append(seed)

    for row in VALIDATION_CASES:
        if str(row.get("id") or "") not in _selected_validation_ids():
            continue
        seed = _seed_from_asset_case(row)
        if seed is None or seed.case_id in seen_ids:
            continue
        seen_ids.add(seed.case_id)
        seeds.append(seed)

    for row in COMPARISON_GOLD_CASES:
        seed = _comparison_pack_seed(row)
        if seed.case_id in seen_ids:
            continue
        seen_ids.add(seed.case_id)
        seeds.append(seed)

    return sorted(seeds, key=lambda seed: (seed.layout, seed.topic, seed.case_id))


SEEDS = tuple(_build_seeds())
TOPICS = tuple(sorted({seed.topic for seed in SEEDS}))


def _scene_sort_key(scene: Scene) -> tuple[str, str, str, str]:
    return (scene.layout, scene.topic, scene.theme, scene.tone, scene.density)


def _is_holdout(scene: Scene) -> bool:
    score = (
        LAYOUTS.index(scene.layout)
        + TOPICS.index(scene.topic)
        + THEMES.index(scene.theme)
        + TONES.index(scene.tone)
        + DENSITIES.index(scene.density)
    )
    return (score % 7) == 0


def _frame(scene: Scene, base_attrs: dict[str, str]) -> str:
    frame = str(base_attrs.get("frame") or "").strip()
    return frame or ("panel" if scene.layout in {"pipeline_lane", "dashboard_cards", "table_analysis"} else "card")


def _inset(scene: Scene, base_attrs: dict[str, str]) -> str:
    default = str(base_attrs.get("inset") or "md")
    return {
        "compact": default if default in {"sm", "md"} else "md",
        "balanced": "md",
        "airy": "lg",
    }.get(scene.density, default)


def _gap(scene: Scene, base_attrs: dict[str, str]) -> str:
    default = str(base_attrs.get("gap") or "md")
    return {
        "compact": "sm",
        "balanced": default if default in {"sm", "md"} else "md",
        "airy": "lg",
    }.get(scene.density, default)


def _rail(scene: Scene, base_attrs: dict[str, str]) -> str:
    rail = str(base_attrs.get("rail") or "none")
    if scene.layout == "poster_stack" and rail == "none":
        return "accent"
    return rail


def _background(scene: Scene, base_attrs: dict[str, str]) -> str:
    base = str(base_attrs.get("background") or "none")
    if base == "none":
        return "none"
    return {
        "paper_editorial": "grid",
        "signal_glow": "mesh",
        "infra_dark": "rings",
    }.get(scene.theme, base)


def _valid_style_combo(theme: str, tone: str) -> bool:
    allowed = {
        "infra_dark": {"amber", "green", "blue", "purple", "mixed"},
        "paper_editorial": {"amber", "green", "blue", "mixed"},
        "signal_glow": {"green", "blue", "purple"},
    }
    return tone in allowed.get(theme, set())


def _scene_output_tokens(scene: Scene) -> list[str]:
    base_attrs = dict(scene.seed.attrs)
    return [
        "[scene]",
        f"[canvas:{base_attrs.get('canvas', 'wide')}]",
        f"[layout:{scene.layout}]",
        f"[theme:{scene.theme}]",
        f"[tone:{scene.tone}]",
        f"[frame:{_frame(scene, base_attrs)}]",
        f"[density:{scene.density}]",
        f"[inset:{_inset(scene, base_attrs)}]",
        f"[gap:{_gap(scene, base_attrs)}]",
        f"[hero:{base_attrs.get('hero', 'left')}]",
        f"[columns:{base_attrs.get('columns', '1')}]",
        f"[emphasis:{base_attrs.get('emphasis', 'top')}]",
        f"[rail:{_rail(scene, base_attrs)}]",
        f"[background:{_background(scene, base_attrs)}]",
        f"[connector:{base_attrs.get('connector', 'line')}]",
        f"[topic:{scene.topic}]",
        *list(scene.seed.components),
        "[/scene]",
    ]


def _row(scene: Scene) -> str:
    return " ".join(scene.prompt_tokens + _scene_output_tokens(scene))


def _render_svg(scene: Scene) -> str:
    return render_structured_scene_spec09_svg(
        " ".join(_scene_output_tokens(scene)),
        content=json.loads(json.dumps(scene.seed.content_json)) if isinstance(scene.seed.content_json, dict) else None,
    )


def _build_scenes() -> list[Scene]:
    scenes: list[Scene] = []
    for seed in SEEDS:
        for theme in THEMES:
            for tone in TONES:
                if not _valid_style_combo(theme, tone):
                    continue
                for density in DENSITIES:
                    scenes.append(Scene(seed=seed, theme=theme, tone=tone, density=density))
    return sorted(scenes, key=_scene_sort_key)


def _write_probe_report_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_probe_report_contract.json"
    payload = {
        "schema": "ck.probe_report_contract.v1",
        "title": "Spec Broader 1 Scene DSL Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec09.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {"max_tokens": 384, "temperature": 0.0},
        "catalog": {
            "format": "json_rows",
            "path": f"{prefix}_render_catalog.json",
            "prompt_key": "prompt",
            "output_key": "output_tokens",
            "rendered_key": "svg_xml",
            "rendered_mime": "image/svg+xml",
            "split_key": "split",
        },
        "splits": [
            {"name": "train", "label": "Train prompts", "format": "lines", "path": f"{prefix}_seen_prompts.txt", "limit": 14},
            {"name": "test", "label": "Holdout prompts", "format": "lines", "path": f"{prefix}_holdout_prompts.txt", "limit": 14},
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _write_eval_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_eval_contract.json"
    payload = {
        "schema": "ck.eval_contract.v1",
        "dataset_type": "svg",
        "goal": "Emit a broader scene DSL that compiles into production-style SVG with optional bound content_json.",
        "notes": [
            "The broader bootstrap line reuses only currently supported scene families.",
            "Some families bind visible copy and values through content_json while the scene DSL stays structural.",
            "This branch is a breadth bootstrap, not yet the full site-asset coverage line.",
        ],
        "scorer": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec09.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {"max_tokens": 384, "temperature": 0.0},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a broader scene-DSL bootstrap dataset")
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec_broader_1_scene_dsl", help="Output prefix")
    ap.add_argument("--train-repeats", type=int, default=3, help="Repeats for seen combos")
    ap.add_argument("--holdout-repeats", type=int, default=1, help="Repeats for holdout combos")
    args = ap.parse_args()

    if args.train_repeats < 1:
        raise SystemExit("--train-repeats must be >= 1")
    if args.holdout_repeats < 1:
        raise SystemExit("--holdout-repeats must be >= 1")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = _build_scenes()
    holdout = [scene for scene in scenes if _is_holdout(scene)]
    train = [scene for scene in scenes if not _is_holdout(scene)]

    train_rows = [_row(scene) for scene in train for _ in range(int(args.train_repeats))]
    holdout_rows = [_row(scene) for scene in holdout for _ in range(int(args.holdout_repeats))]
    tokenizer_corpus = [_row(scene) for scene in scenes] + [scene.prompt_text for scene in scenes]
    seen_prompts = [scene.prompt_text for scene in train]
    holdout_prompts = [scene.prompt_text for scene in holdout]
    render_rows = []
    for scene in scenes:
        content_json = json.loads(json.dumps(scene.seed.content_json)) if isinstance(scene.seed.content_json, dict) else None
        render_rows.append(
            {
                "prompt": scene.prompt_text,
                "output_tokens": " ".join(_scene_output_tokens(scene)),
                "svg_xml": _render_svg(scene),
                "split": "holdout" if scene in holdout else "train",
                "layout": scene.layout,
                "topic": scene.topic,
                "source_asset": scene.seed.source_asset,
                "case_id": scene.seed.case_id,
                "title": scene.seed.title,
                "theme": scene.theme,
                "tone": scene.tone,
                "density": scene.density,
                "content_json": content_json,
            }
        )

    domain_tokens = _ordered_domain_tokens(tokenizer_corpus)
    tokenizer, tokenizer_meta = _build_tokenizer_payload(domain_tokens)
    tokenizer_json, tokenizer_bin = _write_tokenizer_artifacts(
        tokenizer,
        tokenizer_meta,
        out_dir / f"{args.prefix}_tokenizer",
    )

    _write_lines(out_dir / f"{args.prefix}_train.txt", train_rows)
    _write_lines(out_dir / f"{args.prefix}_holdout.txt", holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_tokenizer_corpus.txt", tokenizer_corpus)
    _write_lines(out_dir / f"{args.prefix}_seen_prompts.txt", seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_holdout_prompts.txt", holdout_prompts)
    _write_lines(out_dir / f"{args.prefix}_reserved_control_tokens.txt", domain_tokens)

    vocab_spec = {
        "format": "svg-structured-fixed-vocab.v7",
        "prefix": args.prefix,
        "system_tokens": tokenizer.get("added_tokens", []),
        "tokenizer_json": str(tokenizer_json),
        "tokenizer_bin": str(tokenizer_bin),
        "vocab_size": int(tokenizer_meta["vocab_size"]),
        "num_merges": int(tokenizer_meta["num_merges"]),
    }
    (out_dir / f"{args.prefix}_vocab.json").write_text(
        json.dumps(vocab_spec, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"{args.prefix}_render_catalog.json").write_text(
        json.dumps(render_rows, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    probe_contract_path = _write_probe_report_contract(out_dir, args.prefix)
    eval_contract_path = _write_eval_contract(out_dir, args.prefix)

    manifest = {
        "schema": "ck.generated_dataset.v1",
        "line_name": "spec_broader_1_scene_dsl",
        "prefix": args.prefix,
        "out_dir": str(out_dir),
        "topics": list(TOPICS),
        "layouts": list(LAYOUTS),
        "source_assets": {seed.topic: seed.source_asset for seed in SEEDS},
        "content_bound_cases": sorted(seed.case_id for seed in SEEDS if isinstance(seed.content_json, dict)),
        "artifacts": {
            "train": str(out_dir / f"{args.prefix}_train.txt"),
            "holdout": str(out_dir / f"{args.prefix}_holdout.txt"),
            "seen_prompts": str(out_dir / f"{args.prefix}_seen_prompts.txt"),
            "holdout_prompts": str(out_dir / f"{args.prefix}_holdout_prompts.txt"),
            "render_catalog": str(out_dir / f"{args.prefix}_render_catalog.json"),
            "tokenizer_corpus": str(out_dir / f"{args.prefix}_tokenizer_corpus.txt"),
            "reserved_control_tokens": str(out_dir / f"{args.prefix}_reserved_control_tokens.txt"),
            "vocab": str(out_dir / f"{args.prefix}_vocab.json"),
            "probe_report_contract": str(probe_contract_path),
            "eval_contract": str(eval_contract_path),
        },
        "counts": {
            "seed_cases": len(SEEDS),
            "all": len(scenes),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "holdout_prompts": len(holdout_prompts),
            "tokenizer_rows": len(tokenizer_corpus),
        },
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
