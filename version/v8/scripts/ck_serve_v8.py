#!/usr/bin/env python3
"""v8 inference serving entrypoint.

Owns the model lifecycle for a ``cks-v8-run serve`` invocation:

  build runtime (reuse ck_run_v8 pipeline) -> open native session -> FastAPI

The Responses + tools lifecycle (prompt rendering, tool parsing, SSE
streaming, history, single-flight, cancel, models, health) is canonical in
``server.live`` (native binding in ``server.session_v8``, artifact helpers in
``server.runtime``). This module keeps the ``serve`` CLI, the ``/viz`` page,
and the Chat Completions compatibility route, and re-exports the shared names
so ``from ck_serve_v8 import ...`` keeps working.

**Single-flight**: only one ``generate`` request is processed at a time.
Concurrent requests receive HTTP 429 (Too Many Requests) or an SSE
``response.failed`` event with ``code: "session_busy"``.  The native C layer
enforces this via ``pthread_mutex_trylock``; the Python server adds an
additional Python-level guard for duck-typed sessions.

The production host is a dedicated C or Rust server. The Python/ctypes binding
in ``server.session_v8`` is a reference implementation of the same ABI binding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# SCRIPTS_DIR/PROJECT_ROOT must be on sys.path before importing server/ or
# ck_* helpers (supports both ``python version/v8/scripts/ck_serve_v8.py`` and
# ``python -m version.v8.scripts.ck_serve_v8`` / pytest shims).
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[2]

for _p in (str(SCRIPTS_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if __package__:
    from . import ck_serve_runtime_v8
    from .ck_chat_completions_v8 import add_chat_completions_route
else:
    import ck_serve_runtime_v8
    from ck_chat_completions_v8 import add_chat_completions_route

# Reuse color/logging constants from the runtime module so both entrypoints
# share identical console styling.
C_GREEN = ck_serve_runtime_v8.C_GREEN
C_ORANGE = ck_serve_runtime_v8.C_ORANGE
C_RED = ck_serve_runtime_v8.C_RED
C_GRAY = ck_serve_runtime_v8.C_GRAY
C_RESET = ck_serve_runtime_v8.C_RESET

# Canonical lifecycle: server/ owns the native binding, artifact helpers, and
# the Responses + tools app factory. This module re-exports those names so
# existing ``from ck_serve_v8 import ...`` imports (tests, tooling) keep
# working with identical objects. NOTE: keep this list complete — tests and
# main() import these names from this module.
from server.live import (
    _FLIGHT_WAIT_SECONDS,
    _FUNCTION_LIKE_TYPES,
    _IGNORED_TOOL_TYPES,
    _StreamThinkSplitter,
    _classify_stream_mode,
    _content_text,
    _detect_template_tool_syntax,
    _effective_tools,
    _extract_prompt,
    _extract_tool_calls_from_text,
    _format_prompt_with_chat_contract,
    _has_function_tools,
    _has_tool_support,
    _harness_error,
    _input_chat_messages,
    _load_builtin_chat_contract,
    _load_runtime_chat_contract,
    _load_runtime_templates,
    _log_performance,
    _log_rejection,
    _performance_profile,
    _render_with_chat_templates,
    _resolve_contract_thinking_overrides,
    _resolve_thinking_mode,
    _sse,
    _strip_tool_json_from_text,
    _truncate_stop_markers,
    _usage,
    create_app as _live_create_app,
    split_thinking,
)
from server.runtime import load_manifest_templates, resolve_runtime_context_length
from server.session_v8 import (
    CK_SESSION_REQUEST_RAW_PROMPT,
    _Config,
    _GenerateRequest,
    _GenerateResult,
    _SESSION_STATUS_EXCEPTIONS,
    _SESSION_STATUS_NAMES,
    _TOKEN_CALLBACK,
    _configure_abi,
    _configure_lib,
    _detect_threads,
    _last_error,
    _resolve_num_threads,
    _stop_reason_name,
    SessionBusyError,
    SessionError,
    SessionV8,
    stop_reason_name,
    truncate_stop_markers,
)

log = ck_serve_runtime_v8.log
log_error = ck_serve_runtime_v8.log_error

# Public / compat surface. Every name here is re-exported from server/ for
# ``from ck_serve_v8 import ...`` consumers (tests, tooling); listing them in
# __all__ also keeps ruff F401 from flagging the re-exports as unused.
__all__ = [
    "CK_SESSION_REQUEST_RAW_PROMPT",
    "C_GRAY",
    "C_GREEN",
    "C_ORANGE",
    "C_RED",
    "C_RESET",
    "SessionBusyError",
    "SessionError",
    "SessionV8",
    "_Config",
    "_FUNCTION_LIKE_TYPES",
    "_GenerateRequest",
    "_GenerateResult",
    "_IGNORED_TOOL_TYPES",
    "_SESSION_STATUS_EXCEPTIONS",
    "_SESSION_STATUS_NAMES",
    "_StreamThinkSplitter",
    "_TOKEN_CALLBACK",
    "_build_arg_parser",
    "_build_runtime",
    "_classify_stream_mode",
    "_configure_abi",
    "_configure_lib",
    "_content_text",
    "_detect_template_tool_syntax",
    "_detect_threads",
    "_effective_tools",
    "_ensure_native_session_lib",
    "_extract_prompt",
    "_extract_tool_calls_from_text",
    "_format_prompt_with_chat_contract",
    "_has_function_tools",
    "_has_tool_support",
    "_harness_error",
    "_input_chat_messages",
    "_last_error",
    "_load_builtin_chat_contract",
    "_load_runtime_chat_contract",
    "_load_runtime_templates",
    "_log_performance",
    "_log_rejection",
    "_performance_profile",
    "_render_with_chat_templates",
    "_resolve_contract_thinking_overrides",
    "_resolve_thinking_mode",
    "_resolve_num_threads",
    "_resolve_run_dir",
    "_sse",
    "_stop_reason_name",
    "_strip_tool_json_from_text",
    "_truncate_stop_markers",
    "_usage",
    "add_chat_completions_route",
    "create_app",
    "load_manifest_templates",
    "log",
    "log_error",
    "main",
    "resolve_runtime_context_length",
    "split_thinking",
    "stop_reason_name",
    "truncate_stop_markers",
]


def create_app(session, *, viz: bool = True, **kwargs):
    """Live Responses app with viz page + Chat Completions compat route.

    The shared lifecycle (validation, tools, streaming, history,
    single-flight, cancel, models, health) is owned by
    :func:`server.live.create_app`; this wrapper adds the serve-CLI concerns:
    the ``/viz`` page (``ck_serve_viz.html`` next to this file) and the Chat
    Completions compatibility route from ``ck_chat_completions_v8``.
    """
    viz_html = None
    if viz:
        viz_path = SCRIPTS_DIR / "ck_serve_viz.html"
        if viz_path.is_file():
            viz_html = viz_path.read_text(encoding="utf-8")
        else:
            log_error(f"visualizer file missing: {viz_path}; disabling /viz")

    def _register_chat_route(router, create_response):
        add_chat_completions_route(router, create_response)

    return _live_create_app(
        session,
        viz_html=viz_html,
        extra_route_registrar=_register_chat_route,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# CLI — runtime helpers re-exported from ck_serve_runtime_v8 (model-ready side)
# -----------------------------------------------------------------------------

# The Responses + tools lifecycle is canonical in server.live; artifact
# preparation (download → convert → build IR → codegen → compile libmodel.so)
# lives in ck_serve_runtime_v8. Re-export here for backward compat so
# ``from ck_serve_v8 import _build_runtime`` keeps working.
_ensure_native_session_lib = ck_serve_runtime_v8._ensure_native_session_lib
_resolve_run_dir = ck_serve_runtime_v8._resolve_run_dir
_build_runtime = ck_serve_runtime_v8._build_runtime


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cks-v8-run serve",
        description="Build and serve a v8 model over HTTP/SSE (OpenAI Responses shape).",
        epilog="Example:\n  cks-v8-run serve hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --context-len 1024 --port 8080",
    )
    parser.add_argument("model", help="GGUF source or pre-built runtime directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--model-name", default="ck-v8", help="Model string reported in responses"
    )
    parser.add_argument(
        "--run", dest="run_dir", default=None, help="Explicit run directory"
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip building; require an existing run directory",
    )

    sampler = parser.add_argument_group(
        "sampling (server-level defaults; request body overrides)"
    )
    sampler.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    sampler.add_argument(
        "--top-p", type=float, default=1.0, help="Top-p nucleus sampling (default: 1.0)"
    )
    sampler.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default max output tokens when the request omits max_output_tokens",
    )
    sampler.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling size (accepted for parity; NOT applied by the native ABI)",
    )
    sampler.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p filter as fraction of max prob (accepted for parity; NOT applied by the native ABI)",
    )
    sampler.add_argument(
        "--repeat-penalty",
        type=float,
        default=None,
        help="Repeat penalty >1.0 reduces looping (accepted for parity; NOT applied by the native ABI)",
    )
    sampler.add_argument(
        "--repeat-last-n",
        type=int,
        default=None,
        help="Window size for repeat penalty (accepted for parity; NOT applied by the native ABI)",
    )
    sampler.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=None,
        help="Block tokens that repeat an n-gram of this size (accepted for parity; NOT applied by the native ABI)",
    )

    stop = parser.add_argument_group("stop markers (honored via the token callback)")
    stop.add_argument(
        "--stop-on-text",
        action="append",
        default=[],
        help="Stop generation when this decoded text marker appears (repeatable)",
    )
    stop.add_argument(
        "--stop-at-eos",
        action="store_true",
        help="Stop generation when '<eos>' appears in decoded text",
    )

    display = parser.add_argument_group("metrics / visualizer")
    display.add_argument(
        "--stats",
        action="store_true",
        default=True,
        help="Print per-request performance stats (default: on)",
    )
    display.add_argument(
        "--no-stats",
        action="store_false",
        dest="stats",
        help="Disable per-request performance stats",
    )
    display.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable the live HTML visualizer page at /viz",
    )

    ck_serve_runtime_v8.add_build_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        argv = argv[1:]

    args = _build_arg_parser().parse_args(argv)

    _ensure_native_session_lib()

    run_dir = _resolve_run_dir(args.model, args.run_dir)
    if not args.no_build:
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

    ignored = []
    if args.top_k is not None:
        ignored.append(f"--top-k {args.top_k}")
    if args.min_p is not None:
        ignored.append(f"--min-p {args.min_p}")
    if args.repeat_penalty is not None:
        ignored.append(f"--repeat-penalty {args.repeat_penalty}")
    if args.repeat_last_n is not None:
        ignored.append(f"--repeat-last-n {args.repeat_last_n}")
    if args.no_repeat_ngram_size is not None:
        ignored.append(f"--no-repeat-ngram-size {args.no_repeat_ngram_size}")
    if ignored:
        log(
            "Warning: the native session ABI (ck_session_v8_generate) only supports "
            "temperature/top_p/max_tokens. Ignored until a native ABI extension exists: "
            + ", ".join(ignored),
            C_ORANGE,
        )

    runtime_context_length = ck_serve_runtime_v8.resolve_runtime_context_length(
        run_dir, args.context_len
    )
    if args.context_len is None and runtime_context_length is not None:
        log(f"Using generated runtime context length: {runtime_context_length}")
    elif args.context_len is None:
        log(
            "Warning: generated context capacity is unavailable; using the "
            "native session default. Rebuild or pass --context-len explicitly.",
            C_ORANGE,
        )

    session = SessionV8.open(
        run_dir,
        context_length=runtime_context_length,
    )

    chat_template, chat_templates, chat_contract = _load_runtime_templates(run_dir)
    if args.no_chat_template:
        chat_template = None
        chat_templates = None
        log("Chat templates disabled via --no-chat-template", C_GRAY)
    elif args.chat_template:
        override = str(args.chat_template)
        override_path = Path(override).expanduser()
        if override_path.is_file():
            try:
                override = override_path.read_text(encoding="utf-8")
            except OSError as exc:
                log_error(f"cannot read --chat-template file {override_path}: {exc}")
                override = ""
        if override.strip():
            chat_template = override
            log(
                f"Using explicit --chat-template override ({len(chat_template)} chars); "
                "chat_template.jinja sidecar ignored",
                C_GRAY,
            )
        else:
            chat_template = None
            chat_templates = None
    elif chat_template:
        log(
            f"Loaded chat_template ({len(chat_template)} chars) from chat_template.jinja",
            C_GRAY,
        )
    else:
        log(
            "chat_template.jinja missing or empty in run dir; "
            "continuing without native Jinja template",
            C_ORANGE,
        )
    if chat_templates:
        log(
            f"Loaded chat_templates variants {list(chat_templates.keys())} from additional_chat_templates/",
            C_GRAY,
        )
    log("Chat contract disabled; prompt rendering is pure Jinja", C_GRAY)

    app = create_app(
        session,
        model=args.model_name,
        context_length=runtime_context_length,
        stats=args.stats,
        viz=not args.no_viz,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_on_text=args.stop_on_text,
        stop_at_eos=args.stop_at_eos,
        flags=(
            CK_SESSION_REQUEST_RAW_PROMPT
            if (args.no_chat_template and args.allow_raw_prompt)
            else 0
        ),
        chat_contract=chat_contract,
        chat_template=chat_template,
        chat_templates=chat_templates,
    )

    try:
        import uvicorn
    except ImportError as exc:
        session.close()
        raise ImportError(
            "uvicorn is required to run the server. Install server requirements:\n"
            "  python3 -m pip install -r server/requirements.txt"
        ) from exc

    log(
        f"Serving on http://{args.host}:{args.port}  (mode=live, inference=True)",
        C_GREEN,
    )
    if not args.no_viz:
        log(f"Visualizer: http://{args.host}:{args.port}/viz", C_GREEN)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
