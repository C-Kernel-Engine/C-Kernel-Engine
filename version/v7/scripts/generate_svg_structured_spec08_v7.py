#!/usr/bin/env python3
"""
Generate a richer scene-DSL dataset for spec08.

Spec08 keeps the spec06 control grid, but the model target is now a richer
compiler-owned scene language that renders directly to SVG with gradients,
groups, markers, wrapped text, and background motifs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_svg_structured_spec06_v7 import (
    Scene,
    _build_tokenizer_payload,
    _is_holdout,
    _ordered_domain_tokens,
    _scene_sort_key,
    _write_lines,
    _write_tokenizer_artifacts,
)
from render_svg_structured_scene_rich_v7 import render_structured_scene_rich_svg


def _theme(scene: Scene) -> str:
    if scene.bg == "paper":
        return "paper_editorial"
    if scene.bg == "mint":
        return "signal_glow" if scene.accent in {"blue", "purple"} else "paper_editorial"
    return "infra_dark" if scene.accent in {"green", "gray"} else "signal_glow"


def _frame(scene: Scene) -> str:
    # Compare and stats layouts currently share one framed compiler treatment.
    if scene.layout in {"compare-panels", "stat-cards"}:
        return "panel"
    if scene.frame == "card":
        return "card"
    return "none"


def _density(scene: Scene) -> str:
    if scene.density == "airy":
        return "airy"
    if scene.layout in {"compare-panels", "stat-cards"}:
        return "balanced"
    return "compact"


def _inset(scene: Scene) -> str:
    if scene.density == "airy":
        return "lg"
    if scene.layout in {"compare-panels", "flow-steps"}:
        return "md"
    return "sm"


def _gap(scene: Scene) -> str:
    if scene.density == "airy":
        return "lg"
    if scene.layout in {"stat-cards", "compare-panels"}:
        return "md"
    return "sm"


def _hero(scene: Scene) -> str:
    return {
        "bullet-panel": "left",
        "compare-panels": "split",
        "stat-cards": "center",
        "spectrum-band": "left",
        "flow-steps": "center",
    }[scene.layout]


def _columns(scene: Scene) -> str:
    return {
        "bullet-panel": "1",
        "compare-panels": "2",
        "stat-cards": "3",
        "spectrum-band": "1",
        "flow-steps": "3",
    }[scene.layout]


def _emphasis(scene: Scene) -> str:
    return {
        "bullet-panel": "left",
        "compare-panels": "left",
        "stat-cards": "center",
        "spectrum-band": "top",
        "flow-steps": "top",
    }[scene.layout]


def _rail(scene: Scene) -> str:
    if scene.layout == "bullet-panel":
        return "muted" if scene.accent == "gray" else "accent"
    return "none"


def _background(scene: Scene) -> str:
    return {
        "paper": "grid",
        "mint": "mesh",
        "slate": "rings",
    }[scene.bg]


def _connector(scene: Scene) -> str:
    if scene.layout == "flow-steps":
        return "arrow"
    if scene.layout == "spectrum-band":
        return "bracket"
    if scene.layout == "compare-panels":
        return "arrow" if scene.accent in {"blue", "purple"} else "line"
    return "line"


def _component_tokens(scene: Scene) -> list[str]:
    hero = "[hero:title|subtitle]"
    if scene.layout == "bullet-panel":
        return [
            hero,
            "[bullet_list:bullet_panel_b1|bullet_panel_b2|bullet_panel_b3]",
            "[callout:bullet_panel_callout]",
        ]
    if scene.layout == "compare-panels":
        return [
            hero,
            "[compare_block:compare_panels_left_title|compare_panels_left_line1|compare_panels_left_line2|compare_panels_right_title|compare_panels_right_line1|compare_panels_right_line2]",
            "[footer:compare_panels_footer]",
        ]
    if scene.layout == "stat-cards":
        return [
            hero,
            "[stats:stat_cards_value1|stat_cards_label1|stat_cards_value2|stat_cards_label2|stat_cards_value3|stat_cards_label3]",
            "[callout:stat_cards_footer]",
        ]
    if scene.layout == "spectrum-band":
        return [
            hero,
            "[band:spectrum_band_segment1|spectrum_band_segment2|spectrum_band_segment3]",
            "[footer:spectrum_band_footer]",
        ]
    if scene.layout == "flow-steps":
        return [
            hero,
            "[step_flow:flow_steps_title1|flow_steps_caption1|flow_steps_title2|flow_steps_caption2|flow_steps_title3|flow_steps_caption3]",
            "[badge:flow_steps_badge]",
        ]
    raise RuntimeError(f"unsupported layout: {scene.layout}")


def _scene_output_tokens(scene: Scene) -> list[str]:
    return [
        "[scene]",
        "[canvas:wide]",
        f"[layout:{scene.layout}]",
        f"[theme:{_theme(scene)}]",
        f"[tone:{scene.accent}]",
        f"[frame:{_frame(scene)}]",
        f"[density:{_density(scene)}]",
        f"[inset:{_inset(scene)}]",
        f"[gap:{_gap(scene)}]",
        f"[hero_align:{_hero(scene)}]",
        f"[columns:{_columns(scene)}]",
        f"[emphasis:{_emphasis(scene)}]",
        f"[rail:{_rail(scene)}]",
        f"[background:{_background(scene)}]",
        f"[connector:{_connector(scene)}]",
        f"[topic:{scene.topic}]",
        *_component_tokens(scene),
        "[/scene]",
    ]


def _row(scene: Scene) -> str:
    return " ".join(scene.prompt_tokens + _scene_output_tokens(scene))


def _render_svg(scene: Scene) -> str:
    return render_structured_scene_rich_svg(" ".join(_scene_output_tokens(scene)))


def _build_scenes() -> list[Scene]:
    from generate_svg_structured_spec06_v7 import (
        ACCENTS,
        BACKGROUNDS,
        DENSITIES,
        FRAMES,
        LAYOUTS,
        TOPIC_LIBRARY,
    )

    scenes: list[Scene] = []
    for layout in LAYOUTS:
        for topic in sorted(TOPIC_LIBRARY):
            for accent in ACCENTS:
                for bg in BACKGROUNDS:
                    for frame in FRAMES:
                        for density in DENSITIES:
                            scenes.append(
                                Scene(
                                    layout=layout,
                                    topic=topic,
                                    accent=accent,
                                    bg=bg,
                                    frame=frame,
                                    density=density,
                                )
                            )
    return sorted(scenes, key=_scene_sort_key)


def _write_probe_report_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_probe_report_contract.json"
    payload = {
        "schema": "ck.probe_report_contract.v1",
        "title": "Spec08 Rich Scene DSL Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_rich.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 160,
            "temperature": 0.0,
            "stop_on_text": "<|eos|>",
        },
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
            {"name": "train", "label": "Train prompts", "format": "lines", "path": f"{prefix}_seen_prompts.txt", "limit": 10},
            {"name": "test", "label": "Holdout prompts", "format": "lines", "path": f"{prefix}_holdout_prompts.txt", "limit": 10},
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _write_eval_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_eval_contract.json"
    payload = {
        "schema": "ck.eval_contract.v1",
        "dataset_type": "svg",
        "goal": "Emit a richer scene DSL that compiles directly into infographic SVG.",
        "notes": [
            "Spec08 removes low-level geometry from the model target.",
            "The compiler owns gradients, markers, wrapped text, grouping, and layout geometry.",
            "Probe scoring remains exact-match on the scene DSL plus exact-match on the compiled SVG.",
        ],
        "scorer": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_rich.v1",
            "preview_mime": "image/svg+xml",
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate spec08 rich scene DSL dataset with compiler-backed SVG renders")
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec08_rich_scene_dsl", help="Output file prefix")
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
    render_rows = [
        {
            "prompt": scene.prompt_text,
            "output_tokens": " ".join(_scene_output_tokens(scene)),
            "svg_xml": _render_svg(scene),
            "split": "holdout" if scene in holdout else "train",
            "layout": scene.layout,
            "topic": scene.topic,
        }
        for scene in scenes
    ]

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
        "format": "svg-structured-fixed-vocab.v6",
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
        "line_name": "spec08_rich_scene_dsl",
        "prefix": args.prefix,
        "out_dir": str(out_dir),
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
