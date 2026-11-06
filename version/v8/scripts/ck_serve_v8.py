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
    _content_parts,
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
    _prompt_opens_thinking,
    _render_with_chat_templates,
    _resolve_contract_thinking_overrides,
    _resolve_thinking_mode,
    _select_template,
    _sse,
    _strip_tool_json_from_text,
    _truncate_stop_markers,
    _usage,
    create_app as _live_create_app,
    split_thinking,
)
from server.serving_bundle import PROTOCOLS
from server.runtime import load_manifest_templates, load_tool_protocol, resolve_runtime_context_length, resolve_runtime_vision_capability
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
    "_content_parts",
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
    "_prompt_opens_thinking",
    "_render_with_chat_templates",
    "_resolve_contract_thinking_overrides",
    "_resolve_thinking_mode",
    "_select_template",
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
    "load_tool_protocol",
    "resolve_runtime_vision_capability",
    "log",
    "log_error",
    "main",
    "resolve_runtime_context_length",
    "split_thinking",
    "stop_reason_name",
    "truncate_stop_markers",
    "_undeclared_protocol_warning",
]


def _undeclared_protocol_warning(
    chat_template: str | None,
    chat_templates: dict[str, str] | None,
    protocol: str | None,
) -> str | None:
    """Startup warning when tool requests are doomed to 501."""
    if protocol in (None, "none") and (chat_template or chat_templates):
        return (
            "Warning: tool protocol is undeclared; tool requests will fail with "
            "501 tool_protocol_undeclared until --tool-protocol "
            "tagged_json|qwen_xml|bare_json is passed or a matching "
            "tool_protocol.json sidecar is present"
        )
    return None


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
        "--chat-template-file", default=None,
        help="Explicit UTF-8 Jinja file for serving; replaces native and tool variants",
    )
    parser.add_argument(
        "--chat-template-inline", default=None,
        help="Explicit inline Jinja source for serving; replaces native and tool variants",
    )
    parser.add_argument(
        "--run", dest="run_dir", default=None, help="Explicit run directory"
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip building; require an existing run directory",
    )
    parser.add_argument("--serving-variant", default=None,
                        help="Explicit circuit-linked serving variant to resolve during bundle build")
    parser.add_argument(
        "--tool-protocol",
        choices=sorted(PROTOCOLS),
        default=None,
        help="Explicit tool output protocol; defaults to the resolved bundle or hash-bound legacy sidecar",
    )
    parser.add_argument(
        "--request-output-cap", type=int, default=None,
        help="Explicit server maximum for each request's output allowance; default rejects oversized requests",
    )
    parser.add_argument(
        "--request-prompt-byte-cap", type=int, default=None,
        help="Optional rendered-prompt UTF-8 byte limit, independent of model token capacity",
    )
    parser.add_argument(
        "--prefix-checkpoint-cache-bytes", type=int, default=None,
        help="Opt in to one exact-token prefix checkpoint with this total byte budget; requires a certified append bundle",
    )
    parser.add_argument(
        "--batch-http-extra-bytes", type=int, default=None,
        help="Opt in to two-slot generated HTTP decode with this native+host memory budget",
    )
    parser.add_argument(
        "--batch-http-prompt-cap", type=int, default=256,
        help="Maximum rendered prompt tokens in experimental batch HTTP mode",
    )
    parser.add_argument(
        "--batch-http-decode-replay-chunk-tokens", type=int, default=None,
        help="Experimental staged prefill: replay at most this many prompt tokens via generated decode per scheduling turn; requires --batch-http-extra-bytes and artifact-specific parity certification",
    )
    parser.add_argument(
        "--batch-http-mixed-one-row", action="store_true",
        help="Experimental: pair one staged prompt token with one ready decode token in the same generated batch2 step; requires staged prefill",
    )
    parser.add_argument(
        "--batch-http-prefill-phase", action="store_true",
        help="Experimental: use map-resolved prefill-phase arithmetic for standalone and mixed staged rows; requires staged prefill and an artifact certified for this mode",
    )
    parser.add_argument(
        "--batch-http-delivery-consumers", type=int, default=4,
        help="Maximum admitted batch HTTP responses awaiting delivery",
    )
    parser.add_argument(
        "--batch-http-output-byte-cap", type=int, default=1 << 20,
        help="Maximum decoded output bytes buffered per batch HTTP response",
    )
    parser.add_argument(
        "--batch-http-output-token-cap", type=int, default=4096,
        help="Maximum output token events buffered per batch HTTP response",
    )
    parser.add_argument(
        "--batch-http-delivery-timeout", type=float, default=30.0,
        help="Maximum seconds allowed for each batch HTTP stream write",
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

    if args.no_build and args.serving_variant is not None:
        raise ValueError("--serving-variant requires bundle construction; use the explicit serving resolver for an existing bundle")

    if args.chat_template is not None:
        raise ValueError(
            "--chat-template is ambiguous for serving; use --chat-template-file "
            "or --chat-template-inline"
        )
    if args.chat_template_file is not None and args.chat_template_inline is not None:
        raise ValueError("select only one serving template override")
    if args.no_chat_template and (args.chat_template_file or args.chat_template_inline):
        raise ValueError("--no-chat-template conflicts with a serving template override")
    if args.no_chat_template and not args.allow_raw_prompt:
        raise ValueError("--no-chat-template requires explicit --allow-raw-prompt")
    override: str | None = None
    if args.chat_template_file is not None:
        override_path = Path(args.chat_template_file).expanduser()
        try:
            override = override_path.read_bytes().decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"cannot read --chat-template-file {override_path}: {exc}") from exc
    elif args.chat_template_inline is not None:
        override = args.chat_template_inline
    if override is not None and not override.strip():
        raise ValueError("serving template override is empty")

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
            None,
            args.no_chat_template,
            args.allow_raw_prompt,
            args.python_tokenizer,
            args.profile,
            args.gemm_schedule,
            **({"serving_variant": args.serving_variant} if args.serving_variant is not None else {}),
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

    from server.serving_bundle import load_resolved_serving, resolved_templates
    resolved_serving = load_resolved_serving(run_dir)
    if args.no_chat_template:
        chat_template, chat_templates, chat_contract = None, None, None
        log("Using explicitly requested untemplated raw-prompt serving", C_GRAY)
    else:
        if resolved_serving is not None:
            chat_template, chat_templates, chat_contract = resolved_templates(run_dir, resolved_serving)
        else:
            chat_template, chat_templates, chat_contract = _load_runtime_templates(run_dir)
    if override is not None:
        if resolved_serving is not None:
            raise ValueError("template override conflicts with resolved serving bundle; rebuild with the circuit profile variant")
        chat_template = override
        chat_templates = None
        log(
            f"Using explicit serving Jinja override ({len(chat_template)} chars); "
            "native sidecar and tool variants ignored",
            C_GRAY,
        )
    elif chat_template:
        log(
            f"Loaded chat_template ({len(chat_template)} chars) from "
            + ("resolved serving bundle" if resolved_serving is not None else "chat_template.jinja"),
            C_GRAY,
        )
    elif not args.no_chat_template:
        raise ValueError(
            f"normal chat serving requires {run_dir / 'chat_template.jinja'}; "
            "use --no-chat-template --allow-raw-prompt only for intentional raw serving"
        )
    if chat_templates:
        log(
            f"Loaded chat_templates variants {list(chat_templates.keys())} from "
            + ("resolved serving bundle" if resolved_serving is not None else "additional_chat_templates/"),
            C_GRAY,
        )
    log("Chat contract disabled; prompt rendering is pure Jinja", C_GRAY)
    sidecar_protocol = (None if args.no_chat_template else
                        resolved_serving["output_protocol"] if resolved_serving is not None else
                        load_tool_protocol(run_dir, chat_template, chat_templates))
    if args.tool_protocol is not None and sidecar_protocol is not None and args.tool_protocol != sidecar_protocol:
        raise ValueError(
            f"--tool-protocol {args.tool_protocol!r} conflicts with tool_protocol.json "
            f"declaration {sidecar_protocol!r}"
        )
    selected_tool_protocol = args.tool_protocol or sidecar_protocol
    if selected_tool_protocol not in (None, "none") and not (chat_template or chat_templates):
        raise ValueError("tool protocol requires a nonempty selected chat template")
    warning = _undeclared_protocol_warning(
        chat_template, chat_templates, selected_tool_protocol
    )
    if warning is not None:
        log(warning, C_ORANGE)

    session = SessionV8.open(
        run_dir,
        context_length=runtime_context_length,
    )
    from server.serving_bundle import (
        _loaded_model_token_decoder, _loaded_model_token_lookup, resolved_renderer_tokens,
        runtime_auto_bos_token, runtime_renderer_tokens,
    )
    loaded_identity = None
    try:
        if resolved_serving is not None:
            from server.serving_bundle import loaded_serving_identity, verify_loaded_libraries, verify_runtime_dependencies
            verify_loaded_libraries(resolved_serving)
            verify_runtime_dependencies(resolved_serving)
            loaded_identity = loaded_serving_identity(
                resolved_serving,
                model=args.model_name,
                session_library=session.lib,
            )
            renderer_tokens = resolved_renderer_tokens(
                run_dir, resolved_serving,
                decode_ids=_loaded_model_token_decoder(run_dir),
                lookup_token=_loaded_model_token_lookup(run_dir),
            )
        else:
            renderer_tokens = runtime_renderer_tokens(run_dir) if chat_template else None
        auto_bos_token_text = runtime_auto_bos_token(run_dir)
    except Exception:
        session.close()
        raise

    log("Typed media disabled: this native serving path is text-only", C_GRAY)

    batch_http = None
    if args.batch_http_decode_replay_chunk_tokens is not None and args.batch_http_extra_bytes is None:
        raise ValueError("staged batch HTTP prefill requires --batch-http-extra-bytes")
    if args.batch_http_mixed_one_row and args.batch_http_decode_replay_chunk_tokens is None:
        raise ValueError("mixed batch HTTP execution requires --batch-http-decode-replay-chunk-tokens")
    if args.batch_http_prefill_phase and args.batch_http_decode_replay_chunk_tokens is None:
        raise ValueError("phase-aware batch HTTP prefill requires --batch-http-decode-replay-chunk-tokens")
    if args.batch_http_extra_bytes is not None:
        from server.batch_http import Batch2HTTPOwner
        if args.batch_http_extra_bytes <= 0 or args.batch_http_prompt_cap <= 0:
            raise ValueError("batch HTTP memory budget and prompt cap must be positive")
        if args.request_output_cap is None:
            raise ValueError("batch HTTP requires an explicit --request-output-cap")
        try:
            batch_http = Batch2HTTPOwner(
                session,
                max_extra_bytes=args.batch_http_extra_bytes,
                max_prompt_tokens=args.batch_http_prompt_cap,
                stop_ids=session.generated_stop_ids(),
                max_delivery_consumers=args.batch_http_delivery_consumers,
                max_output_tokens=args.batch_http_output_token_cap,
                max_output_bytes=args.batch_http_output_byte_cap,
                delivery_timeout=args.batch_http_delivery_timeout,
                decode_replay_chunk_tokens=args.batch_http_decode_replay_chunk_tokens,
                mixed_one_row=args.batch_http_mixed_one_row,
                prefill_phase_staged=args.batch_http_prefill_phase,
            )
        except Exception:
            session.close()
            raise

    try:
        app = create_app(
            session,
            model=args.model_name,
            context_length=runtime_context_length,
            stats=args.stats,
            viz=not args.no_viz,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            request_output_cap=args.request_output_cap,
            request_prompt_byte_cap=args.request_prompt_byte_cap,
            stop_on_text=list(dict.fromkeys([
                *(resolved_serving.get("stop_text", []) if resolved_serving else []),
                *args.stop_on_text,
            ])),
            stop_at_eos=args.stop_at_eos,
            flags=(
                CK_SESSION_REQUEST_RAW_PROMPT
                if (args.no_chat_template and args.allow_raw_prompt)
                else 0
            ),
            chat_contract=chat_contract,
            chat_template=chat_template,
            chat_templates=chat_templates,
            tool_protocol=selected_tool_protocol,
            loaded_identity=loaded_identity,
            renderer_tokens=renderer_tokens,
            auto_bos_token_text=auto_bos_token_text,
            allow_untemplated=args.no_chat_template and args.allow_raw_prompt,
            batch_http=batch_http,
            prefix_cache_bytes=args.prefix_checkpoint_cache_bytes,
        )
    except Exception:
        if batch_http is not None:
            batch_http.close()
        session.close()
        raise

    try:
        import uvicorn
    except ImportError as exc:
        if batch_http is not None:
            batch_http.close()
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
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if batch_http is not None:
            batch_http.close()
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
