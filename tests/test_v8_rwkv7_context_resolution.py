"""RWKV-7 conversion must resolve context capacity from checkpoint metadata."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = ROOT / "version" / "v8" / "scripts" / "convert_safetensors_to_bump_v8.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location(
        "convert_safetensors_to_bump_v8_rwkv7_context_test", CONVERTER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _release_manifest(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "release-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_config_field_is_authoritative(tmp_path: Path) -> None:
    converter = _load_converter()
    _release_manifest(
        tmp_path,
        {"metadata": {"context_length": 999}, "identity": {"training_context_length": 999}},
    )
    length, source, hard_limit = converter._rwkv7_resolve_context(
        tmp_path,
        {"inference_ctx_len": 25600},
        {"model_type": "rwkv7"},
        "rwkv7",
    )
    assert length == 25600
    assert source == "config:inference_ctx_len"
    assert hard_limit == 1048576


def test_release_manifest_metadata_is_used_when_config_has_no_context(
    tmp_path: Path,
) -> None:
    converter = _load_converter()
    _release_manifest(tmp_path, {"metadata": {"context_length": 25600}})
    length, source, hard_limit = converter._rwkv7_resolve_context(
        tmp_path,
        {"hidden_size": 2048},
        {"model_type": "rwkv7"},
        "rwkv7",
    )
    assert length == 25600
    assert source == "release-manifest.metadata.context_length"
    assert hard_limit == 1048576


def test_release_manifest_identity_is_the_fallback(tmp_path: Path) -> None:
    converter = _load_converter()
    _release_manifest(
        tmp_path,
        {"identity": {"training_context_length": 25600}},
    )
    length, source, hard_limit = converter._rwkv7_resolve_context(
        tmp_path, {}, {"model_type": "rwkv7"}, "rwkv7"
    )
    assert length == 25600
    assert source == "release-manifest.identity.training_context_length"
    assert hard_limit == 1048576


def test_conflicting_config_fields_fail_closed(tmp_path: Path) -> None:
    converter = _load_converter()
    with pytest.raises(SystemExit, match="conflicting context lengths"):
        converter._rwkv7_resolve_context(
            tmp_path,
            {"context_length": 4096, "max_position_embeddings": 25600},
            {"model_type": "rwkv7"},
            "rwkv7",
        )


def test_missing_declaration_fails_closed(tmp_path: Path) -> None:
    converter = _load_converter()
    with pytest.raises(SystemExit, match="declares no context length"):
        converter._rwkv7_resolve_context(
            tmp_path, {"hidden_size": 2048}, {"model_type": "rwkv7"}, "rwkv7"
        )


def test_declared_context_above_hard_limit_fails_closed(tmp_path: Path) -> None:
    converter = _load_converter()
    with pytest.raises(SystemExit, match="exceeds the circuit hard_context_limit"):
        converter._rwkv7_resolve_context(
            tmp_path,
            {"context_length": 1048577},
            {"model_type": "rwkv7"},
            "rwkv7",
        )


def test_release_manifest_without_context_metadata_fails_closed(tmp_path: Path) -> None:
    converter = _load_converter()
    _release_manifest(tmp_path, {"metadata": {"version": "1"}, "identity": {}})
    with pytest.raises(SystemExit, match="declares no context length"):
        converter._rwkv7_resolve_context(
            tmp_path, {}, {"model_type": "rwkv7"}, "rwkv7"
        )
