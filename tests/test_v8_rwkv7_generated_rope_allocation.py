"""Non-RoPE models must not allocate a context-sized RoPE fallback table."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODEGEN = ROOT / "version" / "v8" / "scripts" / "codegen_core_v8.py"


def _load_codegen():
    sys.path.insert(0, str(CODEGEN.parent))
    spec = importlib.util.spec_from_file_location(
        "codegen_core_v8_rwkv7_rope_allocation_test", CODEGEN
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {CODEGEN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def codegen():
    return _load_codegen()


def _layout() -> dict:
    return {
        "memory": {"weights": {"size": 1000}, "activations": {"buffers": []}},
        "bump_layout": {},
    }


def _config(*, uses_rope: bool | None) -> dict:
    config = {
        "embed_dim": 2048,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 64,
        "intermediate_size": 8192,
        "num_layers": 24,
        "vocab_size": 65536,
        "context_length": 1048576,
        "rotary_dim": 64,
        "rope_scaling_type": "none",
        "rope_scaling_factor": 1.0,
        "num_seqs": 1,
    }
    if uses_rope is not None:
        config["_template_uses_rope"] = uses_rope
    return config


def _fragment(codegen, config: dict) -> str:
    layout = _layout()
    return codegen.emit_memory_layout(layout, config) + codegen.emit_model_and_api(
        layout=layout, config=config
    )


def _preprocess(tmp_path: Path, source: str) -> str:
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        pytest.skip("no C compiler available for preprocessor assertion")
    path = tmp_path / "fragment.c"
    path.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [compiler, "-E", "-P", "-x", "c", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_non_rope_model_leaves_rope_tables_null(codegen, tmp_path: Path) -> None:
    source = _fragment(codegen, _config(uses_rope=False))
    assert "#define MODEL_USES_ROPE 0" in source

    expanded = _preprocess(tmp_path, source)
    assert "calloc(rope_cache_elems" not in expanded
    assert "g_model->rope_cos = NULL;" in expanded
    assert "g_model->rope_sin = NULL;" in expanded


def test_rope_model_keeps_the_fallback_allocation(codegen, tmp_path: Path) -> None:
    source = _fragment(codegen, _config(uses_rope=True))
    assert "#define MODEL_USES_ROPE 1" in source

    expanded = _preprocess(tmp_path, source)
    assert "calloc(rope_cache_elems" in expanded
    assert "g_model->rope_cos = NULL;" not in expanded


def test_missing_rope_flag_preserves_legacy_allocation(codegen, tmp_path: Path) -> None:
    source = _fragment(codegen, _config(uses_rope=None))
    assert "#define MODEL_USES_ROPE 1" in source

    expanded = _preprocess(tmp_path, source)
    assert "calloc(rope_cache_elems" in expanded
