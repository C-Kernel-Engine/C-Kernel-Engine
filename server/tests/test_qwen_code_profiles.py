"""Static contracts for the checked Qwen Code launch profiles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


PROFILE_DIR = Path(__file__).resolve().parents[1] / "qwen-code"


@pytest.mark.parametrize(
    ("name", "context_length", "max_tokens", "wall_seconds"),
    [
        ("interactive", 16384, 2048, 1800),
        ("overnight", 262144, 32768, 64800),
    ],
)
def test_qwen_code_profile_contract(name, context_length, max_tokens, wall_seconds):
    profile = json.loads(
        (PROFILE_DIR / f"{name}.settings.json").read_text(encoding="utf-8")
    )
    model = profile["model"]
    generation = model["generationConfig"]
    assert profile["$version"] == 4
    assert generation["contextWindowSize"] == context_length
    assert generation["samplingParams"]["max_tokens"] == max_tokens
    assert max_tokens < context_length
    assert model["maxWallTimeSeconds"] == wall_seconds
    assert profile["privacy"]["usageStatisticsEnabled"] is False
