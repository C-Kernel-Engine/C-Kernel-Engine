#!/usr/bin/env python3
"""Optional activation buffers must not alias the bump arena base.

Models without a KV cache, RoPE cache, or logits buffer get ``#define
A_* 0`` fallbacks so generated code still compiles. The init/reset paths
must distinguish "absent" from "present at offset 0"; otherwise the
fallback makes ``g_model->kv_cache`` point at the weights and a KV reset
zeroes the model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _core_source() -> str:
    return (SCRIPTS / "codegen_core_v8.py").read_text(encoding="utf-8")


def test_absent_optional_buffers_are_never_treated_as_present() -> None:
    source = _core_source()
    for macro in ("A_KV_CACHE", "A_ROPE_CACHE", "A_LOGITS"):
        assert f"#if defined({macro}) && {macro} > 0" in source, macro
        assert f"#ifdef {macro}" not in source, macro
        assert f"#ifndef {macro}" not in source, macro


def test_fallback_injector_emits_zero_offsets_for_missing_buffers() -> None:
    codegen = _load("codegen_v8_fallback_test", "codegen_v8.py")
    layout = {
        "memory": {
            "activations": {
                "buffers": [{"name": "embedded_input"}],
            }
        }
    }
    generated = codegen._inject_vision_only_fallbacks("#include <math.h>\n", layout)
    for macro in ("A_KV_CACHE", "A_ROPE_CACHE", "A_LOGITS"):
        assert f"#ifndef {macro}\n#define {macro} 0\n#endif" in generated


def test_present_buffers_are_not_overridden() -> None:
    codegen = _load("codegen_v8_fallback_present_test", "codegen_v8.py")
    layout = {
        "memory": {
            "activations": {
                "buffers": [
                    {"name": "kv_cache"},
                    {"name": "rope_cache"},
                    {"name": "logits"},
                ],
            }
        }
    }
    code = "#include <math.h>\n"
    assert codegen._inject_vision_only_fallbacks(code, layout) == code
