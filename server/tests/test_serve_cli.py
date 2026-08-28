"""Argument-parsing tests for the ``cks-v8-run serve`` entrypoint.

These exercise only the argparse surface of ``ck_serve_v8`` plus the runtime
build command construction; no native library, model, or network is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "version" / "v8" / "scripts"))

import pytest

from ck_serve_v8 import _build_arg_parser, _build_runtime, main

HF_MODEL = "hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"


def test_parser_exposes_serve_command():
    parser = _build_arg_parser()
    ns = parser.parse_args([HF_MODEL])
    assert ns.model == HF_MODEL
    assert ns.host == "127.0.0.1"
    assert ns.port == 8080
    assert ns.model_name == "ck-v8"
    assert ns.no_build is False


def test_parser_flag_defaults():
    parser = _build_arg_parser()
    ns = parser.parse_args(["model", "--host", "0.0.0.0", "--port", "9000"])
    assert ns.host == "0.0.0.0"
    assert ns.port == 9000


def test_parser_forwarded_build_flags():
    parser = _build_arg_parser()
    ns = parser.parse_args(
        [
            "model",
            "--run",
            "/tmp/run",
            "--context-len",
            "512",
            "--force-convert",
            "--force-compile",
            "--force-download",
            "--python-tokenizer",
            "--profile",
        ]
    )
    assert ns.run_dir == "/tmp/run"
    assert ns.context_len == 512
    assert ns.force_convert is True
    assert ns.force_compile is True
    assert ns.force_download is True
    assert ns.python_tokenizer is True
    assert ns.profile is True


def test_parser_accepts_hugging_face_model():
    parser = _build_arg_parser()
    ns = parser.parse_args(
        [
            HF_MODEL,
            "--logits-layout",
            "full",
            "--gemm-schedule",
            "dynamic",
        ]
    )
    assert ns.model == HF_MODEL
    assert ns.logits_layout == "full"
    assert ns.gemm_schedule == "dynamic"


def test_parser_requires_model():
    parser = _build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([])
    assert excinfo.value.code in (1, 2)


def test_parser_new_serve_flags():
    parser = _build_arg_parser()
    ns = parser.parse_args(
        [
            "model",
            "--temperature",
            "0.9",
            "--top-p",
            "0.95",
            "--max-tokens",
            "256",
            "--top-k",
            "20",
            "--min-p",
            "0.05",
            "--repeat-penalty",
            "1.1",
            "--repeat-last-n",
            "64",
            "--no-repeat-ngram-size",
            "3",
            "--stop-on-text",
            "<|im_end|>",
            "--stop-on-text",
            "END",
            "--stop-at-eos",
            "--allow-raw-prompt",
            "--no-stats",
            "--no-viz",
        ]
    )
    assert ns.temperature == 0.9
    assert ns.top_p == 0.95
    assert ns.max_tokens == 256
    assert ns.top_k == 20
    assert ns.min_p == 0.05
    assert ns.repeat_penalty == 1.1
    assert ns.repeat_last_n == 64
    assert ns.no_repeat_ngram_size == 3
    assert ns.stop_on_text == ["<|im_end|>", "END"]
    assert ns.stop_at_eos is True
    assert ns.allow_raw_prompt is True
    assert ns.stats is False
    assert ns.no_viz is True


def test_parser_new_serve_flag_defaults():
    parser = _build_arg_parser()
    ns = parser.parse_args(["model"])
    assert ns.temperature == 0.7
    assert ns.top_p == 1.0
    assert ns.max_tokens == 512
    assert ns.top_k is None
    assert ns.min_p is None
    assert ns.repeat_penalty is None
    assert ns.repeat_last_n is None
    assert ns.no_repeat_ngram_size is None
    assert ns.stop_on_text == []
    assert ns.stop_at_eos is False
    assert ns.allow_raw_prompt is False
    assert ns.stats is True
    assert ns.no_viz is False


def test_main_strips_serve_prefix(monkeypatch):
    captured: dict[str, list[str]] = {}

    class Parser:
        def parse_args(self, argv):
            captured["argv"] = list(argv)
            raise SystemExit(0)

    monkeypatch.setattr("ck_serve_v8._build_arg_parser", lambda: Parser())
    with pytest.raises(SystemExit):
        main(["serve", HF_MODEL, "--no-build"])
    assert captured["argv"] == [HF_MODEL, "--no-build"]


def test_main_without_serve_prefix_passes_through(monkeypatch):
    captured: dict[str, list[str]] = {}

    class Parser:
        def parse_args(self, argv):
            captured["argv"] = list(argv)
            raise SystemExit(0)

    monkeypatch.setattr("ck_serve_v8._build_arg_parser", lambda: Parser())
    with pytest.raises(SystemExit):
        main([HF_MODEL, "--no-build"])
    assert captured["argv"] == [HF_MODEL, "--no-build"]


def test_build_runtime_constructs_ck_run_pipeline_command(monkeypatch):
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(list(cmd))
        result = type("P", (), {"returncode": 0, "stderr": "", "stdout": ""})()
        return result

    monkeypatch.setattr("ck_serve_v8.subprocess.run", fake_run)
    run_dir = _build_runtime(
        HF_MODEL,
        Path("/tmp/run"),
        ctx_len=1024,
        force_convert=False,
        force_compile=True,
        force_download=False,
        logits_layout="full",
        chat_template=None,
        no_chat_template=True,
        allow_raw_prompt=True,
        python_tokenizer=True,
        profile=False,
        gemm_schedule="dynamic",
    )

    assert run_dir == Path("/tmp/run")
    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[0] == sys.executable
    assert "--generate-only" in cmd
    assert HF_MODEL in cmd
    assert "--context-len" in cmd and "1024" in cmd
    assert "--logits-layout" in cmd and "full" in cmd
    assert "--no-chat-template" in cmd
    assert "--allow-raw-prompt" in cmd
    assert "--python-tokenizer" in cmd
    assert "--gemm-schedule" in cmd and "dynamic" in cmd


def test_build_runtime_raises_on_failure(monkeypatch):
    result = type("P", (), {"returncode": 1, "stderr": "boom", "stdout": ""})()

    def fake_run(cmd, **kwargs):
        return result

    monkeypatch.setattr("ck_serve_v8.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="boom"):
        _build_runtime(
            HF_MODEL,
            Path("/tmp/run"),
            None,
            False,
            False,
            False,
            None,
            None,
            False,
            False,
            False,
            False,
            None,
        )


def test_parser_thinking_mode_flag():
    parser = _build_arg_parser()
    ns = parser.parse_args([HF_MODEL, "--thinking-mode", "suppressed"])
    assert ns.thinking_mode == "suppressed"


def test_parser_thinking_mode_default():
    parser = _build_arg_parser()
    ns = parser.parse_args([HF_MODEL])
    assert ns.thinking_mode == "auto"


def test_parser_thinking_mode_visible():
    parser = _build_arg_parser()
    ns = parser.parse_args([HF_MODEL, "--thinking-mode", "visible"])
    assert ns.thinking_mode == "visible"


def test_parser_thinking_mode_invalid():
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([HF_MODEL, "--thinking-mode", "invalid"])