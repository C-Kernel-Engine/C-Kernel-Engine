#!/usr/bin/env python3
"""
Generate a scene-DSL dataset for spec07.

Spec07 keeps the spec06 infographic control space and deterministic slot packs,
but moves the model target up one layer: the model emits a compact scene DSL and
the compiler lowers that DSL into the existing spec06 structured SVG atom IR.
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
from render_svg_structured_scene_v7 import render_structured_scene_svg


def _component_tokens(scene: Scene) -> list[str]:
    hero = "[hero:title|subtitle]"
    if scene.layout == "bullet-panel":
        return [
            hero,
            "[rail:accent]",
            "[bullet_list:bullet_panel_b1|bullet_panel_b2|bullet_panel_b3]",
            "[callout:bullet_panel_callout]",
        ]
    if scene.layout == "compare-panels":
        return [
            hero,
            "[compare_pair:compare_panels_left_title|compare_panels_left_line1|compare_panels_left_line2|compare_panels_right_title|compare_panels_right_line1|compare_panels_right_line2]",
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
            "[marker:focus]",
            "[footer:spectrum_band_footer]",
        ]
    if scene.layout == "flow-steps":
        return [
            hero,
            "[steps:flow_steps_title1|flow_steps_caption1|flow_steps_title2|flow_steps_caption2|flow_steps_title3|flow_steps_caption3]",
            "[badge:flow_steps_badge]",
        ]
    raise RuntimeError(f"unsupported layout: {scene.layout}")


def _scene_output_tokens(scene: Scene) -> list[str]:
    return [
        "[scene]",
        "[canvas:wide]",
        f"[source:{scene.source}]",
        f"[theme:{scene.bg}]",
        f"[layout:{scene.layout}]",
        f"[topic:{scene.topic}]",
        f"[accent:{scene.accent}]",
        f"[frame:{scene.frame}]",
        f"[density:{scene.density}]",
        *_component_tokens(scene),
        "[/scene]",
    ]


def _row(scene: Scene) -> str:
    return " ".join(scene.prompt_tokens + _scene_output_tokens(scene))


def _render_svg(scene: Scene) -> str:
    return render_structured_scene_svg(" ".join(_scene_output_tokens(scene)))


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
        "title": "Spec07 Scene DSL Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 128,
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
            {
                "name": "train",
                "label": "Train prompts",
                "format": "lines",
                "path": f"{prefix}_seen_prompts.txt",
                "limit": 10,
            },
            {
                "name": "test",
                "label": "Holdout prompts",
                "format": "lines",
                "path": f"{prefix}_holdout_prompts.txt",
                "limit": 10,
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _write_eval_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_eval_contract.json"
    payload = {
        "schema": "ck.eval_contract.v1",
        "dataset_type": "svg",
        "goal": "Emit a compact scene DSL that compiles deterministically into infographic SVG.",
        "notes": [
            "Spec07 lifts the model target from atom geometry to a compiler-owned scene description.",
            "The compiler owns exact coordinates, slot materialization, and low-level SVG atom expansion.",
            "Probe scoring remains exact-match on the emitted DSL plus exact-match on the compiled SVG.",
        ],
        "scorer": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene.v1",
            "preview_mime": "image/svg+xml",
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate spec07 scene DSL dataset with deterministic compiler-backed SVG renders")
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec07_scene_dsl", help="Output file prefix")
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
            "source": scene.source,
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
        "format": "svg-structured-fixed-vocab.v5",
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

    layout_counts = {
        layout: {
            "train": sum(1 for scene in train if scene.layout == layout),
            "holdout": sum(1 for scene in holdout if scene.layout == layout),
        }
        for layout in sorted({scene.layout for scene in scenes})
    }

    manifest = {
        "format": "structured-svg-scene-manifest.v1",
        "prefix": args.prefix,
        "line": "spec07_scene_dsl",
        "out_dir": str(out_dir),
        "train_repeats": int(args.train_repeats),
        "holdout_repeats": int(args.holdout_repeats),
        "counts": {
            "unique_total": len(scenes),
            "unique_train": len(train),
            "unique_holdout": len(holdout),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "tokenizer_rows": len(tokenizer_corpus),
            "vocab_size": int(tokenizer_meta["vocab_size"]),
            "domain_token_count": len(domain_tokens),
            "layout_counts": layout_counts,
        },
        "artifacts": {
            "train": str(out_dir / f"{args.prefix}_train.txt"),
            "holdout": str(out_dir / f"{args.prefix}_holdout.txt"),
            "tokenizer_corpus": str(out_dir / f"{args.prefix}_tokenizer_corpus.txt"),
            "seen_prompts": str(out_dir / f"{args.prefix}_seen_prompts.txt"),
            "holdout_prompts": str(out_dir / f"{args.prefix}_holdout_prompts.txt"),
            "reserved_control_tokens": str(out_dir / f"{args.prefix}_reserved_control_tokens.txt"),
            "vocab": str(out_dir / f"{args.prefix}_vocab.json"),
            "render_catalog": str(out_dir / f"{args.prefix}_render_catalog.json"),
            "probe_report_contract": str(probe_contract_path),
            "eval_contract": str(eval_contract_path),
            "tokenizer_json": str(tokenizer_json),
            "tokenizer_bin": str(tokenizer_bin),
        },
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "prefix": args.prefix,
                "unique_train": len(train),
                "unique_holdout": len(holdout),
                "train_rows": len(train_rows),
                "holdout_rows": len(holdout_rows),
                "layout_counts": layout_counts,
                "vocab_size": int(tokenizer_meta["vocab_size"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

