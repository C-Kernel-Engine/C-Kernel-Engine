#!/usr/bin/env python3
"""Shared topic content and text-slot expansion for spec06 infographics."""

from __future__ import annotations


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


def _slot_id(topic: str, field: str) -> str:
    return f"{topic}__{field}"


def build_text_slot_map() -> dict[str, str]:
    slot_map: dict[str, str] = {}
    for topic, payload in TOPIC_LIBRARY.items():
        slot_map[_slot_id(topic, "title")] = str(payload["title"])
        slot_map[_slot_id(topic, "subtitle")] = str(payload["subtitle"])

        bullet_panel = payload["bullet-panel"]
        for idx, bullet in enumerate(bullet_panel["bullets"], start=1):
            slot_map[_slot_id(topic, f"bullet_panel_b{idx}")] = str(bullet)
        slot_map[_slot_id(topic, "bullet_panel_callout")] = str(bullet_panel["callout"])

        compare = payload["compare-panels"]
        slot_map[_slot_id(topic, "compare_panels_left_title")] = str(compare["left_title"])
        slot_map[_slot_id(topic, "compare_panels_left_line1")] = str(compare["left_lines"][0])
        slot_map[_slot_id(topic, "compare_panels_left_line2")] = str(compare["left_lines"][1])
        slot_map[_slot_id(topic, "compare_panels_right_title")] = str(compare["right_title"])
        slot_map[_slot_id(topic, "compare_panels_right_line1")] = str(compare["right_lines"][0])
        slot_map[_slot_id(topic, "compare_panels_right_line2")] = str(compare["right_lines"][1])
        slot_map[_slot_id(topic, "compare_panels_footer")] = str(compare["footer"])

        stat_cards = payload["stat-cards"]
        for idx, (value, label) in enumerate(stat_cards["cards"], start=1):
            slot_map[_slot_id(topic, f"stat_cards_value{idx}")] = str(value)
            slot_map[_slot_id(topic, f"stat_cards_label{idx}")] = str(label)
        slot_map[_slot_id(topic, "stat_cards_footer")] = str(stat_cards["footer"])

        spectrum = payload["spectrum-band"]
        for idx, segment in enumerate(spectrum["segments"], start=1):
            slot_map[_slot_id(topic, f"spectrum_band_segment{idx}")] = str(segment)
        slot_map[_slot_id(topic, "spectrum_band_footer")] = str(spectrum["footer"])

        flow_steps = payload["flow-steps"]
        for idx, (title, caption) in enumerate(flow_steps["steps"], start=1):
            slot_map[_slot_id(topic, f"flow_steps_title{idx}")] = str(title)
            slot_map[_slot_id(topic, f"flow_steps_caption{idx}")] = str(caption)
        slot_map[_slot_id(topic, "flow_steps_badge")] = str(flow_steps["badge"])

    return slot_map


_TEXT_SLOT_MAP = build_text_slot_map()


def resolve_text_slot(slot: str, *, topic: str | None = None) -> str:
    raw_slot = str(slot or "").strip()
    if not raw_slot:
        return ""
    if raw_slot in _TEXT_SLOT_MAP:
        return _TEXT_SLOT_MAP[raw_slot]
    topic_name = str(topic or "").strip()
    if topic_name:
        scoped = _slot_id(topic_name, raw_slot)
        if scoped in _TEXT_SLOT_MAP:
            return _TEXT_SLOT_MAP[scoped]
    return raw_slot.replace("__", " ").replace("_", " ")
