#!/usr/bin/env python3
"""Central defaults for parity/autocheck probe inputs.

Keeping probe token defaults in one place avoids hidden drift across scripts.
"""

from __future__ import annotations

import os


DEFAULT_PREFILL_TOKENS_CSV = "2,9259"
DEFAULT_DECODE_TOKEN_ID = 9259
DEFAULT_AUTOCHECK_PROBE_TOKEN_ID = 5


def getenv_prefill_tokens_csv() -> str:
    return os.environ.get("CK_V7_PROBE_PREFILL_TOKENS", DEFAULT_PREFILL_TOKENS_CSV)


def getenv_decode_token_id() -> int:
    raw = os.environ.get("CK_V7_PROBE_DECODE_TOKEN")
    if raw is None:
        return DEFAULT_DECODE_TOKEN_ID
    try:
        return int(raw)
    except Exception:
        return DEFAULT_DECODE_TOKEN_ID


def getenv_autocheck_probe_token_id() -> int:
    raw = os.environ.get("CK_V7_AUTOCHECK_PROBE_TOKEN")
    if raw is None:
        return DEFAULT_AUTOCHECK_PROBE_TOKEN_ID
    try:
        return int(raw)
    except Exception:
        return DEFAULT_AUTOCHECK_PROBE_TOKEN_ID

