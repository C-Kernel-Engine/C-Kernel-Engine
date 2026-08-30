from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v8" / "scripts" / "run_multimodal_bridge_v8.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prebuilt_encoder_bridge_v8", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_prebuilt_encoder_runtime_requires_and_resolves_artifacts() -> None:
    bridge = _load_module()
    with tempfile.TemporaryDirectory(prefix="v8_prebuilt_encoder_") as temporary:
        runtime = Path(temporary)
        (runtime / "weights.bump").write_bytes(b"weights")
        (runtime / "weights_manifest.map").write_text("", encoding="utf-8")
        (runtime / "libckernel_engine.so").write_bytes(b"engine")
        (runtime / "libqwen_encoder.so").write_bytes(b"model")
        (runtime / "layout.json").write_text(
            json.dumps({"config": {"embed_dim": 5120}}), encoding="utf-8"
        )

        loaded = bridge._load_prebuilt_encoder_runtime(runtime)

        assert loaded["runtime_dir"] == runtime.resolve()
        assert loaded["so_path"] == runtime / "libqwen_encoder.so"
        assert loaded["engine_so"] == runtime / "libckernel_engine.so"
        assert loaded["embed_dim"] == 5120


def test_load_prebuilt_encoder_runtime_rejects_missing_provenance() -> None:
    bridge = _load_module()
    with tempfile.TemporaryDirectory(prefix="v8_prebuilt_encoder_missing_") as temporary:
        runtime = Path(temporary)
        (runtime / "libqwen_encoder.so").write_bytes(b"model")
        try:
            bridge._load_prebuilt_encoder_runtime(runtime)
        except RuntimeError as exc:
            assert "incomplete" in str(exc)
            assert "weights.bump" in str(exc)
        else:
            raise AssertionError("incomplete prebuilt encoder runtime was accepted")


def test_explicit_composition_accepts_prebuilt_encoder_runtime() -> None:
    bridge = _load_module()
    bridge._validate_composition_encoder_source(
        {"name": "qwen36vl"},
        encoder_gguf=None,
        encoder_runtime=Path("encoder-runtime"),
    )


def test_explicit_composition_accepts_encoder_gguf() -> None:
    bridge = _load_module()
    bridge._validate_composition_encoder_source(
        {"name": "qwen36vl"},
        encoder_gguf=Path("encoder.gguf"),
        encoder_runtime=None,
    )


def test_explicit_composition_accepts_explicit_synthetic_prefix() -> None:
    bridge = _load_module()
    bridge._validate_composition_encoder_source(
        {"name": "qwen36vl"},
        encoder_gguf=None,
        encoder_runtime=None,
        synthetic_prefix_tokens=128,
    )


def test_explicit_composition_rejects_missing_encoder_source() -> None:
    bridge = _load_module()
    try:
        bridge._validate_composition_encoder_source(
            {"name": "qwen36vl"},
            encoder_gguf=None,
            encoder_runtime=None,
        )
    except RuntimeError as exc:
        assert "--encoder-gguf or --encoder-runtime" in str(exc)
    else:
        raise AssertionError("explicit composition accepted no encoder source")
