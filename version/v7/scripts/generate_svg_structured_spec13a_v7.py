#!/usr/bin/env python3
"""
Generate a spec13a intent-prompt bridge dataset on top of the spec12 scene DSL.

Spec13a changes the prompt side:
- prompt: topic + goal + audience (+ optional future ablations)
- target: same spec12 scene DSL
- content remains external and compiler-backed

For r1, keep the output/control vocabulary frozen to the working spec12 line
whenever possible.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generate_svg_structured_spec06_v7 import _write_lines

# Reuse the frozen-vocab alias cases from spec12 by default.
os.environ.setdefault("CK_SPEC12_FREEZE_VOCAB_ALIAS", "1")
from generate_svg_structured_spec12_v7 import CASES_BY_ID  # noqa: E402
from render_svg_structured_scene_spec12_v7 import render_structured_scene_spec12_svg  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
ENABLE_GOALLESS_BRIDGE = os.environ.get("CK_SPEC13A_ENABLE_GOALLESS_BRIDGE", "0") == "1"
PROMPT_FIELD_SURFACE = os.environ.get("CK_SPEC13A_PROMPT_FIELD_SURFACE", "plain_labels")
INCLUDE_LEGACY_INTENT = os.environ.get("CK_SPEC13A_INCLUDE_LEGACY_INTENT", "0") == "1"
CASE_IDS_OVERRIDE = tuple(
    part.strip()
    for part in os.environ.get("CK_SPEC13A_CASE_IDS", "").split(",")
    if part.strip()
)

AUDIENCE_STYLE = {
    "technical": {"theme": "paper_editorial", "tone": "blue", "density": "balanced"},
    "operator": {"theme": "infra_dark", "tone": "amber", "density": "balanced"},
    "engineer": {"theme": "infra_dark", "tone": "green", "density": "airy"},
}

CASE_GOAL = {
    "failure_triage": "route_debug",
    "quantization_formats": "compare_options",
    "edge_case_matrix": "compare_edge_cases",
    "model_memory_layout": "explain_structure",
    "training_memory_canary": "diagnose_training",
}

CASE_AUDIENCES = {case_id: tuple(AUDIENCE_STYLE) for case_id in CASE_GOAL}

OUTPUT_ATTR_KEYS = {
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
class IntentScene:
    case_id: str
    prompt_topic: str
    goal: str
    audience: str
    layout: str
    topic_token: str
    theme: str
    tone: str
    density: str

    @property
    def scene_id(self) -> str:
        return f"{self.case_id}::{self.audience}"

    @property
    def anchor_prompt(self) -> str:
        return " ".join(
            [
                "[task:svg]",
                f"[layout:{self.layout}]",
                f"[topic:{self.topic_token}]",
                f"[theme:{self.theme}]",
                f"[tone:{self.tone}]",
                f"[density:{self.density}]",
                "[OUT]",
            ]
        )

    @property
    def intent_prompt(self) -> str:
        return self._intent_prompt(topic_first=True)

    def _surface_token(self, field: str, value: str) -> str:
        if PROMPT_FIELD_SURFACE == "bracket":
            return f"[{field}:{value}]"
        return f"{field} {value}"

    def _intent_prompt(self, *, order: tuple[str, ...] = ("topic", "goal", "audience"), topic_first: bool = False) -> str:
        ordered = ("topic", "goal", "audience") if topic_first else order
        tokens = ["[task:svg]"]
        for field in ordered:
            if field == "topic":
                tokens.append(self._surface_token("topic", self.prompt_topic))
            elif field == "goal":
                tokens.append(self._surface_token("goal", self.goal))
            elif field == "audience":
                tokens.append(self._surface_token("audience", self.audience))
            else:
                raise ValueError(f"unsupported prompt field: {field}")
        tokens.append("[OUT]")
        return " ".join(tokens)

    def _legacy_bracket_prompt(self, *, order: tuple[str, ...]) -> str:
        tokens = ["[task:svg]"]
        for field in order:
            if field == "topic":
                tokens.append(f"[topic:{self.prompt_topic}]")
            elif field == "goal":
                tokens.append(f"[goal:{self.goal}]")
            elif field == "audience":
                tokens.append(f"[audience:{self.audience}]")
            else:
                raise ValueError(f"unsupported prompt field: {field}")
        tokens.append("[OUT]")
        return " ".join(tokens)

    @property
    def train_intent_prompts(self) -> list[tuple[str, str]]:
        return [
            (
                "intent_primary",
                self._intent_prompt(order=("topic", "goal", "audience")),
            ),
            (
                "intent_order_train_1",
                self._intent_prompt(order=("audience", "goal", "topic")),
            ),
            (
                "intent_order_train_2",
                self._intent_prompt(order=("goal", "topic", "audience")),
            ),
            (
                "intent_order_train_3",
                self._intent_prompt(order=("goal", "audience", "topic")),
            ),
        ]

    @property
    def bridge_prompts(self) -> list[tuple[str, str]]:
        return [
            (
                "intent_bridge_topic_audience",
                self._intent_prompt(order=("topic", "audience")),
            ),
            (
                "intent_bridge_audience_topic",
                self._intent_prompt(order=("audience", "topic")),
            ),
        ]

    @property
    def legacy_train_intent_prompts(self) -> list[tuple[str, str]]:
        return [
            ("legacy_intent_primary", self._legacy_bracket_prompt(order=("topic", "goal", "audience"))),
            ("legacy_intent_order_train_1", self._legacy_bracket_prompt(order=("audience", "goal", "topic"))),
            ("legacy_intent_order_train_2", self._legacy_bracket_prompt(order=("goal", "topic", "audience"))),
            ("legacy_intent_order_train_3", self._legacy_bracket_prompt(order=("goal", "audience", "topic"))),
        ]

    @property
    def hidden_prompts(self) -> list[str]:
        return [
            self._intent_prompt(order=("topic", "audience", "goal")),
            self._intent_prompt(order=("audience", "topic", "goal")),
        ]


def _dedupe_preserve(rows: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        text = str(row or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _scene_output_tokens(scene: IntentScene) -> list[str]:
    case = CASES_BY_ID[scene.case_id]
    attrs = dict(case.attrs)
    tokens = [
        "[scene]",
        f"[layout:{scene.layout}]",
        f"[theme:{scene.theme}]",
        f"[tone:{scene.tone}]",
        f"[density:{scene.density}]",
        f"[topic:{scene.topic_token}]",
    ]
    for optional in ("canvas", "frame", "inset", "gap", "background", "connector"):
        value = str(attrs.get(optional) or "").strip()
        if value:
            tokens.append(f"[{optional}:{value}]")
    tokens.extend(list(case.components))
    tokens.append("[/scene]")
    return tokens


def _render_svg(scene: IntentScene) -> str:
    case = CASES_BY_ID[scene.case_id]
    return render_structured_scene_spec12_svg(" ".join(_scene_output_tokens(scene)), content=dict(case.content_json))


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_frozen_tokenizer(run_dir: Path, out_dir: Path, prefix: str) -> tuple[Path, Path, list[str], dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    tokenizer_json_src = next(
        (
            path
            for path in (
                run_dir / "tokenizer.json",
                run_dir / "dataset" / "tokenizer" / "tokenizer.json",
            )
            if path.exists()
        ),
        None,
    )
    tokenizer_bin_src = next(
        (
            path
            for path in (
                run_dir / "tokenizer_bin",
                run_dir / "dataset" / "tokenizer" / "tokenizer_bin",
            )
            if path.exists()
        ),
        None,
    )
    if tokenizer_json_src is None or tokenizer_bin_src is None:
        raise SystemExit(f"frozen tokenizer artifacts not found under {run_dir}")

    tokenizer_dir = out_dir / f"{prefix}_tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json_dst = tokenizer_dir / "tokenizer.json"
    tokenizer_bin_dst = tokenizer_dir / "tokenizer_bin"
    shutil.copy2(tokenizer_json_src, tokenizer_json_dst)
    _copy_tree(tokenizer_bin_src, tokenizer_bin_dst)

    doc = json.loads(tokenizer_json_dst.read_text(encoding="utf-8"))
    reserved = [
        str(row.get("content") or "")
        for row in (doc.get("added_tokens") or [])
        if isinstance(row, dict) and row.get("special") is True and str(row.get("content") or "")
    ]
    vocab_doc = {
        "format": "svg-structured-frozen-vocab.v7",
        "prefix": prefix,
        "tokenizer_json": str(tokenizer_json_dst),
        "tokenizer_bin": str(tokenizer_bin_dst),
        "frozen_from_run": str(run_dir),
        "vocab_size": int(((doc.get("model") or {}).get("vocab") or {}).__len__()),
        "domain_tokens_frozen": True,
    }
    return tokenizer_json_dst, tokenizer_bin_dst, reserved, vocab_doc


def _build_scenes() -> list[IntentScene]:
    scenes: list[IntentScene] = []
    active_case_ids = _active_case_ids()
    for case_id in active_case_ids:
        goal = CASE_GOAL[case_id]
        case = CASES_BY_ID[case_id]
        for audience in CASE_AUDIENCES[case_id]:
            style = dict(AUDIENCE_STYLE[audience])
            scenes.append(
                IntentScene(
                    case_id=case_id,
                    prompt_topic=case_id,
                    goal=goal,
                    audience=audience,
                    layout=str(case.layout),
                    topic_token=str(case.topic_token),
                    theme=str(style["theme"]),
                    tone=str(style["tone"]),
                    density=str(style["density"]),
                )
            )
    return sorted(scenes, key=lambda scene: (scene.layout, scene.case_id, scene.audience))


def _active_case_ids() -> tuple[str, ...]:
    return CASE_IDS_OVERRIDE or tuple(CASE_GOAL)


def _is_holdout(scene: IntentScene) -> bool:
    audience_order = tuple(AUDIENCE_STYLE)
    score = (
        sorted(CASE_GOAL).index(scene.case_id)
        + audience_order.index(scene.audience)
    )
    return (score % 4) == 0


def _write_probe_report_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_probe_report_contract.json"
    payload = {
        "schema": "ck.probe_report_contract.v1",
        "title": "Spec13a Intent Prompt Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec12.v1",
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
            {"name": "train", "label": "Train prompts", "format": "lines", "path": f"{prefix}_seen_prompts.txt", "limit": 12},
            {"name": "test", "label": "Holdout prompts", "format": "lines", "path": f"{prefix}_holdout_prompts.txt", "limit": 12},
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _write_eval_contract(out_dir: Path, prefix: str) -> Path:
    path = out_dir / f"{prefix}_eval_contract.json"
    payload = {
        "schema": "ck.eval_contract.v1",
        "dataset_type": "svg",
        "goal": "Infer a spec12 scene plan from bounded intent prompts while keeping content external and compiler-backed.",
        "notes": [
            "Spec13a removes explicit layout/theme/tone/density from the primary prompt surface.",
            "The output contract remains spec12 scene DSL plus external content_json.",
            "Probe scoring stays exact-match on scene DSL plus exact-match on compiled SVG.",
        ],
        "scorer": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec12.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {"max_tokens": 384, "temperature": 0.0},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec13a_scene_dsl", help="Output prefix")
    ap.add_argument("--freeze-tokenizer-run", required=True, type=Path, help="Existing run whose tokenizer should be copied unchanged")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = _build_scenes()

    render_rows: list[dict[str, Any]] = []
    train_rows: list[str] = []
    holdout_rows: list[str] = []
    seen_prompts: list[str] = []
    holdout_prompts: list[str] = []
    hidden_seen_prompts: list[str] = []
    hidden_holdout_prompts: list[str] = []

    for scene in scenes:
        split = "holdout" if _is_holdout(scene) else "train"
        output_tokens = " ".join(_scene_output_tokens(scene))
        svg_xml = _render_svg(scene)
        case = CASES_BY_ID[scene.case_id]

        def add(prompt: str, prompt_family: str, prompt_split: str, *, training_prompt: bool) -> None:
            row = {
                "prompt": prompt,
                "output_tokens": output_tokens,
                "content_json": dict(case.content_json),
                "svg_xml": svg_xml,
                "scene_id": scene.scene_id,
                "split": prompt_split,
                "layout": scene.layout,
                "topic": scene.topic_token,
                "case_id": scene.case_id,
                "prompt_topic": scene.prompt_topic,
                "goal": scene.goal,
                "audience": scene.audience,
                "theme": scene.theme,
                "tone": scene.tone,
                "density": scene.density,
                "source_asset": case.asset,
                "prompt_family": prompt_family,
                "training_prompt": bool(training_prompt),
            }
            render_rows.append(row)
            row_text = f"{prompt} {output_tokens}".strip()
            if prompt_split == "train":
                train_rows.append(row_text)
                seen_prompts.append(prompt)
            elif prompt_split == "holdout":
                holdout_rows.append(row_text)
                holdout_prompts.append(prompt)
            elif prompt_split == "probe_hidden_train":
                hidden_seen_prompts.append(prompt)
            elif prompt_split == "probe_hidden_holdout":
                hidden_holdout_prompts.append(prompt)

        add(scene.anchor_prompt, "explicit_anchor", split, training_prompt=True)
        for prompt_family, prompt in scene.train_intent_prompts:
            add(prompt, prompt_family, split, training_prompt=True)
        if INCLUDE_LEGACY_INTENT:
            for prompt_family, prompt in scene.legacy_train_intent_prompts:
                add(prompt, prompt_family, split, training_prompt=True)
        if ENABLE_GOALLESS_BRIDGE:
            for prompt_family, prompt in scene.bridge_prompts:
                add(prompt, prompt_family, split, training_prompt=True)
        hidden_split = "probe_hidden_train" if split == "train" else "probe_hidden_holdout"
        for idx, prompt in enumerate(scene.hidden_prompts, start=1):
            add(prompt, f"hidden_order_{idx}", hidden_split, training_prompt=False)

    seen_prompts = _dedupe_preserve(seen_prompts)
    holdout_prompts = _dedupe_preserve(holdout_prompts)
    hidden_seen_prompts = _dedupe_preserve(hidden_seen_prompts)
    hidden_holdout_prompts = _dedupe_preserve(hidden_holdout_prompts)

    tokenizer_json, tokenizer_bin, reserved_tokens, vocab_doc = _copy_frozen_tokenizer(
        args.freeze_tokenizer_run,
        out_dir,
        args.prefix,
    )
    _write_lines(out_dir / f"{args.prefix}_train.txt", train_rows)
    _write_lines(out_dir / f"{args.prefix}_holdout.txt", holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_seen_prompts.txt", seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_holdout_prompts.txt", holdout_prompts)
    _write_lines(out_dir / f"{args.prefix}_hidden_seen_prompts.txt", hidden_seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_hidden_holdout_prompts.txt", hidden_holdout_prompts)
    _write_lines(out_dir / f"{args.prefix}_reserved_control_tokens.txt", reserved_tokens)
    (out_dir / f"{args.prefix}_vocab.json").write_text(json.dumps(vocab_doc, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (out_dir / f"{args.prefix}_render_catalog.json").write_text(json.dumps(render_rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    probe_contract_path = _write_probe_report_contract(out_dir, args.prefix)
    eval_contract_path = _write_eval_contract(out_dir, args.prefix)

    manifest = {
        "schema": "ck.generated_dataset.v1",
        "line_name": "spec13a_scene_dsl",
        "prefix": args.prefix,
        "out_dir": str(out_dir),
        "case_ids": sorted(CASE_GOAL),
        "active_case_ids": list(_active_case_ids()),
        "goals": sorted(set(CASE_GOAL.values())),
        "audiences": sorted(AUDIENCE_STYLE),
        "frozen_tokenizer_run": str(args.freeze_tokenizer_run.expanduser().resolve()),
        "artifacts": {
            "train": str(out_dir / f"{args.prefix}_train.txt"),
            "holdout": str(out_dir / f"{args.prefix}_holdout.txt"),
            "seen_prompts": str(out_dir / f"{args.prefix}_seen_prompts.txt"),
            "holdout_prompts": str(out_dir / f"{args.prefix}_holdout_prompts.txt"),
            "hidden_seen_prompts": str(out_dir / f"{args.prefix}_hidden_seen_prompts.txt"),
            "hidden_holdout_prompts": str(out_dir / f"{args.prefix}_hidden_holdout_prompts.txt"),
            "render_catalog": str(out_dir / f"{args.prefix}_render_catalog.json"),
            "reserved_control_tokens": str(out_dir / f"{args.prefix}_reserved_control_tokens.txt"),
            "probe_report_contract": str(probe_contract_path),
            "eval_contract": str(eval_contract_path),
            "tokenizer_json": str(tokenizer_json),
            "tokenizer_bin": str(tokenizer_bin),
            "vocab": str(out_dir / f"{args.prefix}_vocab.json"),
        },
        "counts": {
            "all_scenes": len(scenes),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "train_prompts": len(seen_prompts),
            "holdout_prompts": len(holdout_prompts),
            "hidden_seen_prompts": len(hidden_seen_prompts),
            "hidden_holdout_prompts": len(hidden_holdout_prompts),
        },
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
