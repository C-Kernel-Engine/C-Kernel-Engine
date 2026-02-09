#!/usr/bin/env python3
"""
Tokenizer codegen sync regression test.

Ensures tokenizer config emitted in generated C matches lowered init IR
(`init_call.json`) for add_bos/add_eos/add_space_prefix.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


ASSIGN_RE = re.compile(
    r"g_model->tokenizer->config\.add_(bos|eos|space_prefix)\s*=\s*(true|false)\s*;"
)
SPM_MODE_RE = re.compile(
    r"g_model->tokenizer->config\.spm_mode\s*=\s*(CK_SPM_MODE_LLAMA|CK_SPM_MODE_UNIGRAM)\s*;"
)


def _log(prefix: str, msg: str) -> None:
    print(f"{prefix} {msg}{RESET}")


def _bool_from_word(word: str) -> bool:
    return word == "true"


def _find_default_model_dir() -> Optional[Path]:
    roots = [
        Path.home() / ".cache" / "ck-engine-v6.6" / "models",
        Path.home() / ".cache" / "ck-engine-v6" / "models",
    ]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for base in root.iterdir():
            if not base.is_dir():
                continue
            ck_dir = base / "ck_build"
            if not (ck_dir / "model_v6_6.c").exists():
                continue
            if not (ck_dir / "init_call.json").exists():
                continue
            mtime = (ck_dir / "model_v6_6.c").stat().st_mtime
            candidates.append((mtime, ck_dir))
    if not candidates:
        return None
    _, ck_dir = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
    return ck_dir


def _parse_expected_from_init_call(init_call_path: Path) -> Dict[str, object]:
    with open(init_call_path, "r", encoding="utf-8") as f:
        init_obj = json.load(f)

    expected: Dict[str, object] = {}

    # Prefer explicit assignments in c_code.init.
    for op in init_obj.get("operations", []):
        if op.get("op") != "tokenizer_init":
            continue
        c_code = op.get("c_code", {})
        if not isinstance(c_code, dict):
            continue
        init_src = c_code.get("init", "")
        for key, val in ASSIGN_RE.findall(init_src):
            expected[key] = _bool_from_word(val)
        spm_mode = SPM_MODE_RE.search(init_src)
        if spm_mode:
            expected["spm_mode"] = spm_mode.group(1)

    if expected:
        return expected

    # Fallback to special_tokens metadata.
    special = init_obj.get("special_tokens", {}) or {}
    if "add_bos_token" in special:
        expected["bos"] = bool(special["add_bos_token"])
    if "add_eos_token" in special:
        expected["eos"] = bool(special["add_eos_token"])
    if "add_space_prefix" in special:
        expected["space_prefix"] = bool(special["add_space_prefix"])
    tokenizer_model = special.get("tokenizer_model")
    if isinstance(tokenizer_model, str):
        if tokenizer_model.strip().lower() == "llama":
            expected["spm_mode"] = "CK_SPM_MODE_LLAMA"
        else:
            expected["spm_mode"] = "CK_SPM_MODE_UNIGRAM"

    return expected


def _parse_actual_from_model_c(model_c_path: Path) -> Dict[str, object]:
    src = model_c_path.read_text(encoding="utf-8", errors="replace")
    actual: Dict[str, object] = {}
    # Keep the last occurrence for each key.
    for key, val in ASSIGN_RE.findall(src):
        actual[key] = _bool_from_word(val)
    for mode in SPM_MODE_RE.findall(src):
        actual["spm_mode"] = mode
    return actual


def _diff(expected: Dict[str, object], actual: Dict[str, object]) -> Tuple[bool, str]:
    missing = []
    mismatched = []
    for key, exp in expected.items():
        if key not in actual:
            missing.append(key)
            continue
        act = actual[key]
        if act != exp:
            mismatched.append((key, exp, act))

    if not missing and not mismatched:
        return True, ""

    parts = []
    if missing:
        parts.append(f"missing in model_v6_6.c: {missing}")
    if mismatched:
        parts.extend(
            [f"{k}: expected={e}, actual={a}" for (k, e, a) in mismatched]
        )
    return False, "; ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenizer codegen sync test")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to ck_build containing model_v6_6.c and init_call.json",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if model dir cannot be resolved",
    )
    args = parser.parse_args()

    model_dir = args.model_dir or _find_default_model_dir()
    if model_dir is None:
        msg = "No ck_build with model_v6_6.c + init_call.json found"
        if args.strict_missing:
            _log(RED + "[FAIL]", msg)
            return 1
        _log(YELLOW + "[SKIP]", msg)
        return 0

    init_call_path = model_dir / "init_call.json"
    model_c_path = model_dir / "model_v6_6.c"
    if not init_call_path.exists() or not model_c_path.exists():
        _log(
            RED + "[FAIL]",
            f"Missing required files under {model_dir} (need init_call.json and model_v6_6.c)",
        )
        return 1

    _log(CYAN + "[INFO]", f"Model dir: {model_dir}")
    expected = _parse_expected_from_init_call(init_call_path)
    actual = _parse_actual_from_model_c(model_c_path)

    if not expected:
        _log(YELLOW + "[SKIP]", "No tokenizer config assignments found in init_call.json")
        return 0

    ok, message = _diff(expected, actual)
    if ok:
        _log(GREEN + "[PASS]", f"Tokenizer codegen config matches init_call: {expected}")
        return 0

    _log(RED + "[FAIL]", "Tokenizer codegen config drift detected")
    print(f"       expected (init_call): {expected}")
    print(f"       actual (model_v6_6.c): {actual}")
    print(f"       details: {message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
