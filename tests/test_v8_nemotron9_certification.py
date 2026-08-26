from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_nemotron9_certification_manifest_is_coherent_and_repeatable() -> None:
    manifest = json.loads(
        (ROOT / "version/v8/regression/families_nemotron9_certification.json").read_text(
            encoding="utf-8"
        )
    )
    families = manifest["families"]
    assert len(families) == 1
    family = families[0]
    assert family["id"] == "nemotron9"
    assert family["model_env"] == "V8_NEMOTRON9_MODEL"
    assert family["coherence_gate"] is True
    assert family["repeatability"]["runs"] >= 3
    assert set(family["smoke_prompts"]) == {"hello", "france_capital", "code_bundle"}
    assert family["runtime_args"][family["runtime_args"].index("--thinking-mode") + 1] == "auto"
    assert family["runtime_expect"]["config"]["model"] == "nemotron_h"
    assert family["runtime_expect"]["config"]["tokenizer_contract.tokenizer_type"] == "bpe"

    prompts = json.loads(
        (ROOT / "version/v8/regression/prompts_nemotron9_certification.json").read_text(
            encoding="utf-8"
        )
    )["prompts"]
    assert prompts["hello"]["max_tokens"] >= 128
    assert prompts["france_capital"]["max_tokens"] >= 512
    assert prompts["code_bundle"]["max_tokens"] >= 512


def test_nemotron9_chat_contract_matches_embedded_gguf_protocol() -> None:
    circuit = json.loads(
        (ROOT / "version/v8/circuits/nemotron_h.json").read_text(encoding="utf-8")
    )
    chat = circuit["contract"]["chat_contract"]
    assert chat["raw_prompt_allowed"] is False
    assert chat["conversation_prefix"] == "<SPECIAL_10>System\n"
    assert chat["turn_prefix_by_role"]["system"] == ""
    assert chat["turn_prefix"] == "<SPECIAL_11>{role}\n"
    assert chat["assistant_generation_prefix_by_thinking_mode"]["visible"].endswith(
        "<think>\n"
    )
    assert chat["assistant_generation_prefix_by_thinking_mode"]["suppressed"].endswith(
        "<think></think>"
    )
    assert chat["stop_text_markers"] == ["<SPECIAL_12>"]
    assert chat["template_markers"] == ["<SPECIAL_10>", "<SPECIAL_11>", "<SPECIAL_12>"]


def test_nemotron9_high_memory_target_uses_certification_manifest() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "V8_NEMOTRON9_MIN_MEM_GB ?= 16" in makefile
    assert "families_nemotron9_certification.json" in makefile
    assert "prompts_nemotron9_certification.json" in makefile
    assert "--family nemotron9" in makefile
    assert "--mode full" in makefile
    assert "--force-rebuild" in makefile
