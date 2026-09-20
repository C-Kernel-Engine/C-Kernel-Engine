from __future__ import annotations

import shutil
import subprocess
import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "version" / "v8" / "src" / "ck_audio_encoder_decoder_transcribe_v8.c"


def _load_certifier():
    path = ROOT / "version" / "v8" / "scripts" / "certify_cohere_generated_standalone_v8.py"
    spec = importlib.util.spec_from_file_location("cohere_standalone_certifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_native_host_builds_without_python_runtime() -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        return
    source = HOST.read_text(encoding="utf-8")
    assert "Python" not in source
    assert "system(" not in source
    assert "popen(" not in source
    assert "ck_model_run_audio_encoder" in source
    assert "ck_model_set_encoder_memory" in source
    assert "ck_model_audio_prompt_token_id" in source
    assert "<|startoftranscript|>" not in source
    assert "tokens[i] ==" not in source
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(HOST),
                "-ldl",
                "-o",
                str(Path(directory) / "cohere-transcribe"),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


def test_certifier_requires_explicit_native_token_trajectory() -> None:
    certifier = _load_certifier()
    assert certifier._tokens("noise\ntoken_ids=2,17,3\n") == [2, 17, 3]
    try:
        certifier._tokens("tokens=2,17,3\n")
    except ValueError as error:
        assert "token trajectory" in str(error)
    else:
        raise AssertionError("missing native token evidence was accepted")
