#!/usr/bin/env python3
"""
Generate an asset-inspired infographic SVG atoms dataset for spec06.

Spec06 keeps the deterministic structured SVG atoms renderer, but moves from
toy scene controls toward infographic template filling. Each prompt selects:
- a layout family derived from an existing site asset style
- a topic pack that fills fixed content slots
- a small style envelope (accent, background, frame, density)

The model still predicts the same low-level renderer-backed SVG atom DSL.
What changes is the curriculum target: composition is represented as choosing
the correct template and slot values, not inventing geometry token by token.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from render_svg_structured_atoms_v7 import render_structured_svg_atoms


SYSTEM_TOKENS = ["<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>"]
ASCII_BASE = ["\t", "\n", "\r"] + [chr(i) for i in range(32, 127)]

LAYOUTS = ("bullet-panel", "compare-panels", "stat-cards", "spectrum-band", "flow-steps")
ACCENTS = ("orange", "green", "blue", "purple", "gray")
BACKGROUNDS = ("paper", "mint", "slate")
DENSITIES = ("compact", "airy")
FRAMES = ("plain", "card")

SOURCE_BY_LAYOUT = {
    "bullet-panel": "memory-reality-infographic",
    "compare-panels": "performance-balance",
    "stat-cards": "activation-memory-infographic",
    "spectrum-band": "operator-spectrum-map",
    "flow-steps": "pipeline-overview",
}

TOPIC_LIBRARY = {
    "structured_outputs": {
        "title": "Structured Outputs",
        "subtitle": "Reliable layout before open style",
        "bullet-panel": {
            "bullets": ["Score exact prompts", "Compile fixed boxes", "Track holdout drift"],
            "callout": "Benchmark before scale",
        },
        "compare-panels": {
            "left_title": "Raw text",
            "left_lines": ["layout drifts", "scoring is weak"],
            "right_title": "Template fill",
            "right_lines": ["slots stay bound", "regressions show up"],
            "footer": "Structure turns demos into tests",
        },
        "stat-cards": {
            "cards": [("5", "layouts"), ("26", "probe cases"), ("1", "compiler path")],
            "footer": "Each slot is deterministic",
        },
        "spectrum-band": {
            "segments": ["free text", "scene dsl", "compiled svg"],
            "footer": "Move right to reduce ambiguity",
        },
        "flow-steps": {
            "steps": [("Prompt", "pick topic"), ("Compile", "fill slots"), ("Probe", "score exact")],
            "badge": "spec06 ready",
        },
    },
    "platform_rollout": {
        "title": "Platform Rollout",
        "subtitle": "Pilot safely before wide intake",
        "bullet-panel": {
            "bullets": ["start with guardrails", "publish support path", "measure active use"],
            "callout": "adoption needs operations",
        },
        "compare-panels": {
            "left_title": "Fast launch",
            "left_lines": ["licenses sit idle", "support gets noisy"],
            "right_title": "Staged launch",
            "right_lines": ["guardrails land first", "usage stays visible"],
            "footer": "Operational model beats blanket rollout",
        },
        "stat-cards": {
            "cards": [("1", "pilot group"), ("2", "support lanes"), ("Apr", "next review")],
            "footer": "Scale follows intake discipline",
        },
        "spectrum-band": {
            "segments": ["trial", "guardrails", "supported service"],
            "footer": "Move from experiment to deliverable",
        },
        "flow-steps": {
            "steps": [("Pilot", "small cohort"), ("Refine", "fix intake"), ("Scale", "support teams")],
            "badge": "phase two",
        },
    },
    "gpu_readiness": {
        "title": "GPU Readiness",
        "subtitle": "Capacity only helps when it is usable",
        "bullet-panel": {
            "bullets": ["track sku limits", "document install path", "benchmark real loads"],
            "callout": "hardware needs runbooks",
        },
        "compare-panels": {
            "left_title": "Raw capacity",
            "left_lines": ["gpus arrive late", "teams still wait"],
            "right_title": "Ready platform",
            "right_lines": ["docs ship first", "benchmarks guide use"],
            "footer": "Provisioning is not enablement",
        },
        "stat-cards": {
            "cards": [("RTX", "pro line"), ("2", "cost paths"), ("POC", "azure compare")],
            "footer": "Benchmark on prem before promises",
        },
        "spectrum-band": {
            "segments": ["sku choice", "platform setup", "supported jobs"],
            "footer": "Usable compute beats raw headlines",
        },
        "flow-steps": {
            "steps": [("Install", "base stack"), ("Document", "run path"), ("Support", "user jobs")],
            "badge": "emerald path",
        },
    },
    "governance_path": {
        "title": "Governance Path",
        "subtitle": "Classification must match deployment",
        "bullet-panel": {
            "bullets": ["tag data class", "choose run location", "record approval trail"],
            "callout": "governance is part of delivery",
        },
        "compare-panels": {
            "left_title": "Loose policy",
            "left_lines": ["scope stays fuzzy", "ownership slips"],
            "right_title": "Clear classing",
            "right_lines": ["risk is visible", "routing is consistent"],
            "footer": "Policy must shape the runtime choice",
        },
        "stat-cards": {
            "cards": [("3", "sensitivity tiers"), ("2", "deploy zones"), ("1", "approval trail")],
            "footer": "Governed models need explicit paths",
        },
        "spectrum-band": {
            "segments": ["local data", "managed cloud", "public edge"],
            "footer": "Place the model where the data allows",
        },
        "flow-steps": {
            "steps": [("Classify", "data risk"), ("Route", "pick zone"), ("Review", "keep record")],
            "badge": "policy live",
        },
    },
    "capacity_math": {
        "title": "Capacity Math",
        "subtitle": "Tokens memory and throughput all count",
        "bullet-panel": {
            "bullets": ["measure batch cost", "watch kv growth", "compare local spend"],
            "callout": "throughput is not free",
        },
        "compare-panels": {
            "left_title": "Peak demos",
            "left_lines": ["ignore memory", "hide queue cost"],
            "right_title": "Real budgets",
            "right_lines": ["track token load", "show full price"],
            "footer": "Capacity claims need operating math",
        },
        "stat-cards": {
            "cards": [("512", "ctx target"), ("1M", "token budget"), ("2", "cost models")],
            "footer": "Budget the workload before scaling",
        },
        "spectrum-band": {
            "segments": ["token cost", "memory headroom", "fleet throughput"],
            "footer": "Each axis changes the final envelope",
        },
        "flow-steps": {
            "steps": [("Measure", "token load"), ("Model", "cost path"), ("Decide", "scale fit")],
            "badge": "budget first",
        },
    },
    "eval_discipline": {
        "title": "Eval Discipline",
        "subtitle": "Tests keep research from drifting",
        "bullet-panel": {
            "bullets": ["hold out prompt slices", "watch exact rates", "keep parity visible"],
            "callout": "measurement keeps trust",
        },
        "compare-panels": {
            "left_title": "Loose eval",
            "left_lines": ["demos look fine", "failures hide"],
            "right_title": "Probe suite",
            "right_lines": ["slices stay clear", "regressions surface"],
            "footer": "Capability reports should tell the truth",
        },
        "stat-cards": {
            "cards": [("80.8%", "exact"), ("92.3%", "renderable"), ("1", "narrow fail slice")],
            "footer": "Good metrics narrow the next patch",
        },
        "spectrum-band": {
            "segments": ["syntax pass", "semantic bind", "holdout generalize"],
            "footer": "Do not stop at valid shells",
        },
        "flow-steps": {
            "steps": [("Train", "fit rows"), ("Probe", "slice failures"), ("Patch", "target gaps")],
            "badge": "regressions on",
        },
    },
}


def _accent_alt(accent: str) -> str:
    return {
        "orange": "blue",
        "blue": "green",
        "green": "purple",
        "purple": "orange",
        "gray": "blue",
    }.get(accent, "blue")


def _accent_soft(accent: str) -> str:
    return {
        "orange": "paper",
        "blue": "paper",
        "green": "mint",
        "purple": "paper",
        "gray": "paper",
    }.get(accent, "paper")


@dataclass(frozen=True)
class Scene:
    layout: str
    topic: str
    accent: str = "blue"
    bg: str = "paper"
    frame: str = "plain"
    density: str = "compact"

    @property
    def source(self) -> str:
        return SOURCE_BY_LAYOUT[self.layout]

    @property
    def prompt_tokens(self) -> list[str]:
        return [
            "[task:svg]",
            f"[layout:{self.layout}]",
            f"[topic:{self.topic}]",
            f"[accent:{self.accent}]",
            f"[bg:{self.bg}]",
            f"[frame:{self.frame}]",
            f"[density:{self.density}]",
            "[OUT]",
        ]

    @property
    def prompt_text(self) -> str:
        return " ".join(self.prompt_tokens)


def _rect(x: int, y: int, width: int, height: int, rx: int, fill: str, stroke: str = "black", sw: int = 2) -> list[str]:
    return [
        "[rect]",
        f"[x:{x}]",
        f"[y:{y}]",
        f"[width:{width}]",
        f"[height:{height}]",
        f"[rx:{rx}]",
        f"[fill:{fill}]",
        f"[stroke:{stroke}]",
        f"[sw:{sw}]",
    ]


def _circle(cx: int, cy: int, r: int, fill: str, stroke: str = "black", sw: int = 2) -> list[str]:
    return [
        "[circle]",
        f"[cx:{cx}]",
        f"[cy:{cy}]",
        f"[r:{r}]",
        f"[fill:{fill}]",
        f"[stroke:{stroke}]",
        f"[sw:{sw}]",
    ]


def _triangle(points: str, fill: str, stroke: str = "black", sw: int = 2) -> list[str]:
    return [
        "[polygon]",
        f"[points:{points}]",
        f"[fill:{fill}]",
        f"[stroke:{stroke}]",
        f"[sw:{sw}]",
    ]


def _text(tx: int, ty: int, font: int, anchor: str, fill: str, content: str) -> list[str]:
    return [
        "[text]",
        f"[tx:{tx}]",
        f"[ty:{ty}]",
        f"[font:{font}]",
        f"[anchor:{anchor}]",
        f"[fill:{fill}]",
        content,
        "[/text]",
    ]


def _stroke_for_bg(bg: str) -> str:
    return "white" if bg == "slate" else "black"


def _base_fill(bg: str) -> str:
    if bg == "slate":
        return "gray"
    if bg == "mint":
        return "white"
    return "paper"


def _text_fill(bg: str) -> str:
    return "white" if bg == "slate" else "black"


def _muted_fill(bg: str) -> str:
    return "gray" if bg != "slate" else "white"


def _content(scene: Scene) -> dict[str, object]:
    return TOPIC_LIBRARY[scene.topic][scene.layout]  # type: ignore[index]


def _slot(scene: Scene, field: str) -> str:
    return f"[slot:{field}]"


def _title(scene: Scene) -> str:
    return _slot(scene, "title")


def _subtitle(scene: Scene) -> str:
    return _slot(scene, "subtitle")


def _outer_frame(scene: Scene, stroke: str) -> list[str]:
    if scene.frame != "card":
        return []
    return _rect(8, 8, 240, 144, 18, _base_fill(scene.bg), stroke)


def _bullet_panel_tokens(scene: Scene, stroke: str) -> list[str]:
    content = dict(_content(scene))
    bullets = list(content["bullets"])
    text_fill = _text_fill(scene.bg)
    muted = _muted_fill(scene.bg)
    left = 18 if scene.frame == "card" else 12
    title_y = 26 if scene.frame == "card" else 22
    subtitle_y = title_y + 16
    bullet_y = 64 if scene.density == "compact" else 68
    bullet_step = 18 if scene.density == "compact" else 20
    callout_y = 118 if scene.density == "compact" else 122
    tokens: list[str] = []
    tokens.extend(_rect(left, 52, 6, 68 if scene.density == "compact" else 72, 3, scene.accent, scene.accent, 1))
    tokens.extend(_text(left + 12, title_y, 16, "start", scene.accent, _title(scene)))
    tokens.extend(_text(left + 12, subtitle_y, 10, "start", muted, _subtitle(scene)))
    for idx, bullet in enumerate(bullets):
        cy = bullet_y + (idx * bullet_step)
        tokens.extend(_circle(left + 18, cy - 4, 3, scene.accent, scene.accent, 1))
        tokens.extend(_text(left + 30, cy, 10, "start", text_fill, _slot(scene, f"bullet_panel_b{idx + 1}")))
    tokens.extend(_rect(left + 12, callout_y, 208, 20, 8, _accent_soft(scene.accent), stroke, 1))
    tokens.extend(_text(left + 20, callout_y + 14, 10, "start", "black", _slot(scene, "bullet_panel_callout")))
    return tokens


def _compare_panel_tokens(scene: Scene, stroke: str) -> list[str]:
    text_fill = _text_fill(scene.bg)
    muted = _muted_fill(scene.bg)
    left_x = 18 if scene.frame == "card" else 12
    panel_y = 54 if scene.density == "compact" else 58
    panel_h = 74 if scene.density == "compact" else 70
    panel_w = 98
    right_x = left_x + 120
    header_h = 18
    partner = _accent_alt(scene.accent)
    tokens: list[str] = []
    tokens.extend(_text(left_x, 26 if scene.frame == "card" else 22, 16, "start", scene.accent, _title(scene)))
    tokens.extend(_text(left_x, 42 if scene.frame == "card" else 38, 10, "start", muted, _subtitle(scene)))

    tokens.extend(_rect(left_x, panel_y, panel_w, panel_h, 10, _base_fill(scene.bg), stroke))
    tokens.extend(_rect(left_x, panel_y, panel_w, header_h, 10, scene.accent, scene.accent, 1))
    tokens.extend(_text(left_x + 10, panel_y + 13, 10, "start", "black", _slot(scene, "compare_panels_left_title")))
    tokens.extend(_text(left_x + 10, panel_y + 38, 10, "start", text_fill, _slot(scene, "compare_panels_left_line1")))
    tokens.extend(_text(left_x + 10, panel_y + 54, 10, "start", text_fill, _slot(scene, "compare_panels_left_line2")))

    tokens.extend(_rect(right_x, panel_y, panel_w, panel_h, 10, _base_fill(scene.bg), stroke))
    tokens.extend(_rect(right_x, panel_y, panel_w, header_h, 10, partner, partner, 1))
    tokens.extend(_text(right_x + 10, panel_y + 13, 10, "start", "black", _slot(scene, "compare_panels_right_title")))
    tokens.extend(_text(right_x + 10, panel_y + 38, 10, "start", text_fill, _slot(scene, "compare_panels_right_line1")))
    tokens.extend(_text(right_x + 10, panel_y + 54, 10, "start", text_fill, _slot(scene, "compare_panels_right_line2")))
    tokens.extend(_text(left_x, 144, 10, "start", muted, _slot(scene, "compare_panels_footer")))
    return tokens


def _stat_card_tokens(scene: Scene, stroke: str) -> list[str]:
    content = dict(_content(scene))
    cards = list(content["cards"])
    muted = _muted_fill(scene.bg)
    base_x = 18 if scene.frame == "card" else 12
    card_y = 56 if scene.density == "compact" else 60
    card_w = 64
    card_h = 54
    gap = 14
    tokens: list[str] = []
    tokens.extend(_text(base_x, 26 if scene.frame == "card" else 22, 16, "start", scene.accent, _title(scene)))
    tokens.extend(_text(base_x, 42 if scene.frame == "card" else 38, 10, "start", muted, _subtitle(scene)))
    for idx, (value, label) in enumerate(cards):
        x = base_x + (idx * (card_w + gap))
        fill = scene.accent if idx == 0 else (_accent_alt(scene.accent) if idx == 1 else _base_fill(scene.bg))
        value_fill = "black" if idx < 2 else _text_fill(scene.bg)
        label_fill = "black" if idx < 2 else muted
        tokens.extend(_rect(x, card_y, card_w, card_h, 10, fill, stroke))
        tokens.extend(_text(x + 10, card_y + 22, 16, "start", value_fill, _slot(scene, f"stat_cards_value{idx + 1}")))
        tokens.extend(_text(x + 10, card_y + 40, 10, "start", label_fill, _slot(scene, f"stat_cards_label{idx + 1}")))
    tokens.extend(_rect(base_x, 124, 206, 18, 8, _accent_soft(scene.accent), stroke, 1))
    tokens.extend(_text(base_x + 10, 137, 10, "start", "black", _slot(scene, "stat_cards_footer")))
    return tokens


def _spectrum_band_tokens(scene: Scene, stroke: str) -> list[str]:
    content = dict(_content(scene))
    segments = list(content["segments"])
    muted = _muted_fill(scene.bg)
    base_x = 18 if scene.frame == "card" else 12
    bar_y = 74 if scene.density == "compact" else 80
    bar_w = 68
    colors = (scene.accent, _accent_alt(scene.accent), "gray")
    tokens: list[str] = []
    tokens.extend(_text(base_x, 26 if scene.frame == "card" else 22, 16, "start", scene.accent, _title(scene)))
    tokens.extend(_text(base_x, 42 if scene.frame == "card" else 38, 10, "start", muted, _subtitle(scene)))
    for idx, segment in enumerate(segments):
        x = base_x + (idx * bar_w)
        tokens.extend(_rect(x, bar_y, bar_w, 22, 6, colors[idx], colors[idx], 1))
        tokens.extend(_text(x + (bar_w // 2), bar_y + 38, 10, "middle", _text_fill(scene.bg), _slot(scene, f"spectrum_band_segment{idx + 1}")))
    tokens.extend(_triangle(f"{base_x + 136},{bar_y - 10}|{base_x + 130},{bar_y}|{base_x + 142},{bar_y}", scene.accent, scene.accent, 1))
    tokens.extend(_rect(base_x, 118, 206, 18, 8, _base_fill(scene.bg), stroke, 1))
    tokens.extend(_text(base_x + 10, 131, 10, "start", _text_fill(scene.bg), _slot(scene, "spectrum_band_footer")))
    return tokens


def _flow_step_tokens(scene: Scene, stroke: str) -> list[str]:
    content = dict(_content(scene))
    steps = list(content["steps"])
    muted = _muted_fill(scene.bg)
    base_x = 18 if scene.frame == "card" else 12
    card_y = 60 if scene.density == "compact" else 64
    card_w = 64
    card_h = 48
    gap = 14
    tokens: list[str] = []
    tokens.extend(_text(base_x, 26 if scene.frame == "card" else 22, 16, "start", scene.accent, _title(scene)))
    tokens.extend(_text(base_x, 42 if scene.frame == "card" else 38, 10, "start", muted, _subtitle(scene)))
    for idx, (title, caption) in enumerate(steps):
        x = base_x + (idx * (card_w + gap))
        if idx < len(steps) - 1:
            tokens.extend(_rect(x + card_w, card_y + 20, gap, 4, 2, scene.accent, scene.accent, 1))
        tokens.extend(_circle(x + 12, card_y - 6, 8, scene.accent, stroke, 1))
        tokens.extend(_text(x + 12, card_y - 2, 9, "middle", "black", str(idx + 1)))
        tokens.extend(_rect(x, card_y, card_w, card_h, 10, _base_fill(scene.bg), stroke))
        tokens.extend(_text(x + 10, card_y + 18, 11, "start", _text_fill(scene.bg), _slot(scene, f"flow_steps_title{idx + 1}")))
        tokens.extend(_text(x + 10, card_y + 34, 10, "start", muted, _slot(scene, f"flow_steps_caption{idx + 1}")))
    tokens.extend(_rect(base_x + 156, 124, 64, 18, 9, _accent_alt(scene.accent), _accent_alt(scene.accent), 1))
    tokens.extend(_text(base_x + 188, 137, 10, "middle", "black", _slot(scene, "flow_steps_badge")))
    return tokens


def _scene_output_tokens(scene: Scene) -> list[str]:
    stroke = _stroke_for_bg(scene.bg)
    tokens = [
        "[svg]",
        "[w:256]",
        "[h:160]",
        f"[bg:{scene.bg}]",
        f"[layout:{scene.layout}]",
        f"[topic:{scene.topic}]",
        f"[accent:{scene.accent}]",
        f"[frame:{scene.frame}]",
        f"[density:{scene.density}]",
    ]
    tokens.extend(_outer_frame(scene, stroke))
    if scene.layout == "bullet-panel":
        tokens.extend(_bullet_panel_tokens(scene, stroke))
    elif scene.layout == "compare-panels":
        tokens.extend(_compare_panel_tokens(scene, stroke))
    elif scene.layout == "stat-cards":
        tokens.extend(_stat_card_tokens(scene, stroke))
    elif scene.layout == "spectrum-band":
        tokens.extend(_spectrum_band_tokens(scene, stroke))
    elif scene.layout == "flow-steps":
        tokens.extend(_flow_step_tokens(scene, stroke))
    else:
        raise RuntimeError(f"unsupported layout: {scene.layout}")
    tokens.append("[/svg]")
    return tokens


def _row(scene: Scene) -> str:
    return " ".join(scene.prompt_tokens + _scene_output_tokens(scene))


def _render_svg(scene: Scene) -> str:
    return render_structured_svg_atoms(" ".join(_scene_output_tokens(scene)))


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _build_tokenizer_payload(domain_tokens: list[str]) -> tuple[dict, dict]:
    vocab_list: list[str] = []
    seen: set[str] = set()

    def _push(token: str) -> None:
        if token in seen:
            return
        seen.add(token)
        vocab_list.append(token)

    for token in SYSTEM_TOKENS:
        _push(token)
    for token in domain_tokens:
        _push(token)
    for token in ASCII_BASE:
        _push(token)

    vocab = {token: idx for idx, token in enumerate(vocab_list)}
    added_tokens = [
        {
            "id": vocab[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for token in (SYSTEM_TOKENS + domain_tokens)
    ]
    tokenizer = {
        "version": "1.0",
        "ck_mode": "ascii_bpe",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<|unk|>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": True,
            "vocab": vocab,
            "merges": [],
        },
    }
    meta = {
        "schema": "ck.bpe.binary.v1",
        "mode": "ascii_bpe",
        "vocab_size": len(vocab_list),
        "num_merges": 0,
        "num_reserved_special_tokens": len(domain_tokens),
        "max_piece_bytes": 0,
    }
    return tokenizer, meta


def _write_tokenizer_artifacts(tokenizer: dict, meta: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json = out_dir / "tokenizer.json"
    tokenizer_json.write_text(json.dumps(tokenizer, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    vocab = tokenizer["model"]["vocab"]
    sorted_vocab = sorted(vocab.items(), key=lambda item: int(item[1]))
    offsets: list[int] = []
    strings_blob = b""
    current_offset = 0
    for token, _ in sorted_vocab:
        token_bytes = token.encode("utf-8")
        offsets.append(current_offset)
        strings_blob += token_bytes + b"\0"
        current_offset += len(token_bytes) + 1

    bin_dir = out_dir / "tokenizer_bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "vocab_offsets.bin").write_bytes(struct.pack("<" + ("i" * len(offsets)), *offsets))
    (bin_dir / "vocab_strings.bin").write_bytes(strings_blob)
    (bin_dir / "vocab_merges.bin").write_bytes(b"")

    meta_doc = dict(meta)
    meta_doc["offsets"] = str((bin_dir / "vocab_offsets.bin").resolve())
    meta_doc["strings"] = str((bin_dir / "vocab_strings.bin").resolve())
    meta_doc["merges"] = str((bin_dir / "vocab_merges.bin").resolve())
    (bin_dir / "tokenizer_meta.json").write_text(
        json.dumps(meta_doc, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return tokenizer_json, bin_dir


def _ordered_domain_tokens(rows: list[str]) -> list[str]:
    domain_tokens: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for token in row.split():
            if not (token.startswith("[") and token.endswith("]")):
                continue
            if token in seen:
                continue
            seen.add(token)
            domain_tokens.append(token)
    return domain_tokens


def _scene_sort_key(scene: Scene) -> tuple[str, str, str, str, str, str]:
    return (scene.layout, scene.topic, scene.accent, scene.bg, scene.frame, scene.density)


def _is_holdout(scene: Scene) -> bool:
    if scene.layout == "bullet-panel":
        return (
            scene.topic in {"gpu_readiness", "governance_path"} and scene.bg == "slate" and scene.frame == "card"
        ) or (
            scene.topic in {"governance_path", "eval_discipline"} and scene.bg == "mint" and scene.density == "airy"
        )
    if scene.layout == "compare-panels":
        return scene.accent in {"purple", "gray"} and scene.density == "airy"
    if scene.layout == "stat-cards":
        return scene.topic in {"capacity_math", "eval_discipline"} and scene.bg == "mint" and scene.frame == "card"
    if scene.layout == "spectrum-band":
        return scene.topic in {"structured_outputs", "platform_rollout"} and scene.accent == "orange" and scene.density == "airy"
    if scene.layout == "flow-steps":
        return scene.topic in {"platform_rollout", "governance_path"} and scene.bg == "mint" and scene.frame == "card"
    return False


def _build_scenes() -> list[Scene]:
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
        "title": "Spec06 Structured Infographics Probe Report",
        "dataset_type": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/svg]"],
            "renderer": "structured_svg_atoms.v1",
            "preview_mime": "image/svg+xml",
        },
        "decode": {
            "max_tokens": 384,
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
        "goal": "Populate fixed infographic templates with deterministic slot content while preserving renderability.",
        "notes": [
            "Spec06 uses asset-inspired infographic layouts rendered through the structured SVG atoms DSL.",
            "The model is scored on exact slot binding and materialized SVG validity.",
            "Template geometry is deterministic; learning target is layout plus slot selection.",
        ],
        "scorer": "svg",
        "output_adapter": {
            "name": "text_renderer",
            "stop_markers": ["[/svg]"],
            "renderer": "structured_svg_atoms.v1",
            "preview_mime": "image/svg+xml",
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate spec06 infographic SVG atoms dataset with fixed vocab tokenizer")
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="spec06_structured_infographics", help="Output file prefix")
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
        "format": "svg-structured-fixed-vocab.v4",
        "prefix": args.prefix,
        "system_tokens": SYSTEM_TOKENS,
        "domain_tokens": domain_tokens,
        "ascii_base_count": len(ASCII_BASE),
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
        for layout in LAYOUTS
    }

    manifest = {
        "format": "structured-svg-atoms-manifest.v4",
        "prefix": args.prefix,
        "line": "spec06_infographics",
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
