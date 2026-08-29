#!/usr/bin/env python3
"""v8 model preparation entrypoint.

Owns the model-ready side of ``cks-v8-run serve``:

  download -> convert -> build IR -> codegen -> compile

Reuses the :mod:`ck_run_v8` pipeline with ``--generate-only`` so the
resulting ``run_dir`` contains:

  libmodel.so, weights.bump, weights_manifest.{json,map}, config.json

The server side (``ck_serve_v8.py``) imports helpers from this module and
then opens a native session and starts FastAPI/SSE.

This module is intentionally free of :mod:`fastapi`, :mod:`ctypes` session
binding, and HTTP state — it only prepares artifacts on disk.

It can also run standalone::

  python version/v8/scripts/ck_serve_runtime_v8.py hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --context-len 1024 --run /tmp/ck-run

"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[2]
BUILD_DIR = PROJECT_ROOT / "build"
SESSION_LIB_PATH = BUILD_DIR / "libck_session_v8.so"

# Ensure both the project root and this scripts directory are on sys.path so
# ``import ck_run_v8`` works whether this file is run as a script
# (``python version/v8/scripts/ck_serve_runtime_v8.py`` / ``cks-v8-run``) or
# imported as ``version.v8.scripts.ck_serve_runtime_v8``.
for _p in (str(SCRIPTS_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if __package__:
    from . import ck_run_v8
else:
    import ck_run_v8

C_GREEN = "\033[38;5;114m"
C_ORANGE = "\033[38;5;208m"
C_RED = "\033[38;5;203m"
C_GRAY = "\033[38;5;242m"
C_RESET = "\033[0m"


def log(msg: str, color: str = "") -> None:
    if color:
        print(f"{color}{msg}{C_RESET}")
    else:
        print(msg)


def log_error(msg: str) -> None:
    print(f"{C_RED}Error:{C_RESET} {msg}", file=sys.stderr)


# -----------------------------------------------------------------------------
# Runtime helpers — used by ck_serve_v8.py and standalone __main__
# -----------------------------------------------------------------------------


def _ensure_native_session_lib() -> None:
    if SESSION_LIB_PATH.is_file():
        return
    log("Building native session library (make ck-session-v8) ...")
    subprocess.run(["make", "ck-session-v8"], cwd=str(PROJECT_ROOT), check=True)


def _resolve_run_dir(model: str, run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir).expanduser().resolve()
    input_type, info = ck_run_v8.detect_input_type(model)
    if input_type == "hf_gguf":
        return ck_run_v8.CACHE_DIR / info["repo_id"].replace("/", "--")
    if input_type == "hf_id":
        return ck_run_v8.CACHE_DIR / info["model_id"].replace("/", "--")
    return Path(info["path"])


def _build_runtime(
    model: str,
    run_dir: Path,
    ctx_len: int | None,
    force_convert: bool,
    force_compile: bool,
    force_download: bool,
    logits_layout: str | None,
    chat_template: str | None,
    no_chat_template: bool,
    allow_raw_prompt: bool,
    python_tokenizer: bool,
    profile: bool,
    gemm_schedule: str | None,
) -> Path:
    args = [
        "run",
        model,
        "--run",
        str(run_dir),
        "--generate-only",
    ]
    if ctx_len:
        args += ["--context-len", str(int(ctx_len))]
    if force_convert:
        args.append("--force-convert")
    if force_compile:
        args.append("--force-compile")
    if force_download:
        args.append("--force-download")
    if logits_layout:
        args.extend(["--logits-layout", logits_layout])
    if no_chat_template:
        args.append("--no-chat-template")
    elif chat_template:
        args.extend(["--chat-template", chat_template])
    if allow_raw_prompt:
        args.append("--allow-raw-prompt")
    if python_tokenizer:
        args.append("--python-tokenizer")
    if profile:
        args.append("--profile")
    if gemm_schedule:
        args.extend(["--gemm-schedule", gemm_schedule])

    log("Building runtime via ck_run pipeline ...", C_ORANGE)
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "ck_run_v8.py"), *args],
        cwd=str(PROJECT_ROOT),
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "ck_run pipeline failed to build the runtime.\n"
            + (proc.stderr or proc.stdout or "").strip()
        )
    return run_dir


def add_build_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach download/convert/compile flags shared with ``ck_serve_v8``.

    Mutates and returns *parser* so both the standalone runtime CLI and the
    server CLI share identical build-flag definitions.
    """
    build = parser.add_argument_group(
        "build / tokenizer flags (mirrors cks-v8-run run)"
    )
    build.add_argument("--context-len", type=int, default=None)
    build.add_argument(
        "--logits-layout", choices=["auto", "last", "full"], default=None
    )
    build.add_argument(
        "--chat-template", default=None, help="Chat template to compile in"
    )
    build.add_argument("--no-chat-template", action="store_true")
    build.add_argument(
        "--allow-raw-prompt",
        action="store_true",
        help="Forward to the build; with --no-chat-template also set the RAW_PROMPT request flag",
    )
    build.add_argument("--python-tokenizer", action="store_true")
    build.add_argument(
        "--profile", action="store_true", help="Emit CK_PROFILE timing wrappers"
    )
    build.add_argument(
        "--gemm-schedule", choices=("auto", "static", "dynamic"), default=None
    )
    build.add_argument("--force-download", action="store_true")
    build.add_argument("--force-convert", action="store_true")
    build.add_argument("--force-compile", action="store_true")
    return parser


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ck_serve_runtime_v8",
        description="Prepare a v8 runtime (download/convert/build C artifacts, no server).",
        epilog="Example:\n  python version/v8/scripts/ck_serve_runtime_v8.py hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --context-len 1024",
    )
    parser.add_argument("model", help="GGUF source or pre-built runtime directory")
    parser.add_argument(
        "--run", dest="run_dir", default=None, help="Explicit run directory"
    )
    add_build_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_arg_parser().parse_args(argv)

    _ensure_native_session_lib()

    run_dir = _resolve_run_dir(args.model, args.run_dir)
    run_dir = _build_runtime(
        args.model,
        run_dir,
        args.context_len,
        args.force_convert,
        args.force_compile,
        args.force_download,
        args.logits_layout,
        args.chat_template,
        args.no_chat_template,
        args.allow_raw_prompt,
        args.python_tokenizer,
        args.profile,
        args.gemm_schedule,
    )
    log(f"Runtime ready at {run_dir}", C_GREEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
