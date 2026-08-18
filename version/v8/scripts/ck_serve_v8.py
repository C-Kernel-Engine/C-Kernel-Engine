#!/usr/bin/env python3
"""v8 inference serving entrypoint.

Owns the full model lifecycle for a ``cks-v8-run serve`` invocation:

  build runtime (reuse ck_run_v8 pipeline) -> open native session -> FastAPI

The model is loaded once through the stable native host boundary in
``include/ck_session_v8.h`` (``build/libck_session_v8.so``) and invoked per
request with ``ck_session_v8_generate``. Each generated token is delivered as an
SSE ``response.output_text.delta`` event. Responses keep an in-process store so
the scaffold ``GET /responses/{id}`` surface resolves at runtime; conversation
state remains the scaffold's in-memory store.

The production host is a dedicated C or Rust server. This Python/ctypes module
is a reference implementation of the same ABI binding.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import ck_run_v8


SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[2]
BUILD_DIR = PROJECT_ROOT / "build"
SESSION_LIB_PATH = BUILD_DIR / "libck_session_v8.so"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

C_GREEN = "\033[38;5;114m"
C_ORANGE = "\033[38;5;208m"
C_RED = "\033[38;5;203m"
C_GRAY = "\033[38;5;242m"
C_RESET = "\033[0m"

CK_SESSION_REQUEST_RAW_PROMPT = 1 << 0

_STOP_REASON_NAMES = {
    0: "none",
    1: "eos",
    2: "token_limit",
    3: "cancelled",
    4: "callback",
    5: "runtime_error",
}

# Third-party and local schema imports are optional at module load so the CLI
# (--help / pipeline) works even without the server extras. They must live at
# module scope so FastAPI/Pydantic can resolve route annotations.
try:
    from fastapi import APIRouter, FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, StreamingResponse

    _FASTAPI_AVAILABLE = True
except ImportError:
    APIRouter = FastAPI = HTTPException = StreamingResponse = None
    _FASTAPI_AVAILABLE = False

try:
    from server.schemas.common import ItemStatus, ResponseStatus
    from server.schemas.content import ResponseOutputText
    from server.schemas.output_items import (
        ReasoningItem,
        ReasoningTextContent,
        ResponseOutputMessage,
    )
    from server.schemas.response import CreateResponseRequest
    from server.routes.conversations import router as conversations_router

    _SCHEMAS_AVAILABLE = True
except ImportError:
    _SCHEMAS_AVAILABLE = False


def log(msg: str, color: str = "") -> None:
    if color:
        print(f"{color}{msg}{C_RESET}")
    else:
        print(msg)


def log_error(msg: str) -> None:
    print(f"{C_RED}Error:{C_RESET} {msg}", file=sys.stderr)


def _detect_threads() -> int:
    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    return max(1, os.cpu_count() or 1)


# --- Native session binding ---------------------------------------------------
#
# ``create_app`` depends only on this duck-typed surface, so the tests inject a
# fake session and never touch the native library. A live session provides:
#
#   session.generate(system, user, *, max_tokens, temperature, top_p, on_token) -> dict
#   session.cancel() -> None
#   session.close() -> None


class _Config(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("model_library_path", ctypes.c_char_p),
        ("weights_path", ctypes.c_char_p),
        ("manifest_path", ctypes.c_char_p),
        ("context_length", ctypes.c_int32),
        ("num_threads", ctypes.c_int32),
        ("required_capabilities", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _GenerateRequest(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("system_text", ctypes.c_char_p),
        ("user_text", ctypes.c_char_p),
        ("max_tokens", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("flags", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _GenerateResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("prompt_tokens", ctypes.c_int32),
        ("generated_tokens", ctypes.c_int32),
        ("stop_reason", ctypes.c_int32),
        ("reserved0", ctypes.c_int32),
        ("prefill_time_ms", ctypes.c_double),
        ("decode_time_ms", ctypes.c_double),
        ("reserved", ctypes.c_uint64 * 8),
    ]


_TOKEN_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.c_int32,
)


def _configure_abi(lib: Any, name: str) -> None:
    fn = getattr(lib, name)
    if name == "ck_session_v8_open":
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    elif name == "ck_session_v8_generate":
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(_GenerateResult),
        ]
    elif name in ("ck_session_v8_cancel", "ck_session_v8_close"):
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p]
    elif name == "ck_session_v8_last_error":
        fn.restype = ctypes.c_char_p
        fn.argtypes = [ctypes.c_void_p]


def _last_error(lib: Any, session: Any) -> str:
    try:
        value = lib.ck_session_v8_last_error(session)
        return value.decode("utf-8", "replace") if value else "unknown error"
    except Exception:
        return "unknown error"


def _configure_lib(lib: Any) -> None:
    for name in (
        "ck_session_v8_open",
        "ck_session_v8_generate",
        "ck_session_v8_cancel",
        "ck_session_v8_close",
        "ck_session_v8_last_error",
    ):
        _configure_abi(lib, name)


class SessionV8:
    """ctypes binding for the session ABI (one process-local model load)."""

    CK_ABI_VERSION = 1
    CK_OK = 0

    @classmethod
    def open(
        cls,
        work_dir: Path,
        *,
        context_length: int | None = None,
        num_threads: int | None = None,
    ) -> "SessionV8":
        work_dir = work_dir.resolve()
        for name in ("libmodel.so", "weights.bump", "weights_manifest.map"):
            if not (work_dir / name).is_file():
                missing = work_dir / name
                log_error(
                    f"missing runtime artifact {name} in {work_dir}\n"
                    f"  The v8 native loader requires the pipe-delimited "
                    f"weights_manifest.map (name|dtype|file_offset|size|runtime_offset); "
                    f"weights_manifest.json is not accepted by the C loader."
                )
                raise FileNotFoundError(
                    f"missing runtime artifact {name} in {work_dir}"
                )
        if not SESSION_LIB_PATH.is_file():
            raise RuntimeError(
                f"missing native session library {SESSION_LIB_PATH}; "
                "run `make ck-session-v8`"
            )

        lib = ctypes.CDLL(str(SESSION_LIB_PATH))
        _configure_lib(lib)

        cfg = _Config(
            struct_size=ctypes.sizeof(_Config),
            abi_version=cls.CK_ABI_VERSION,
            model_library_path=str(work_dir / "libmodel.so").encode(),
            weights_path=str(work_dir / "weights.bump").encode(),
            manifest_path=str(work_dir / "weights_manifest.map").encode(),
            context_length=int(context_length or 2048),
            num_threads=int(num_threads or _detect_threads()),
        )
        session = ctypes.c_void_p()
        status = lib.ck_session_v8_open(ctypes.byref(cfg), ctypes.byref(session))
        if status != cls.CK_OK or not session:
            log_error(
                f"ck_session_v8_open failed with manifest {cfg.manifest_path.decode('utf-8', 'replace')}\n"
                f"  The v8 native loader parses the pipe-delimited weights_manifest.map "
                f"(name|dtype|file_offset|size|runtime_offset); weights_manifest.json is not accepted."
            )
            raise RuntimeError(f"ck_session_v8_open failed: {_last_error(lib, session)}")

        self = cls.__new__(cls)
        self.lib = lib
        self.session = session
        return self

    def generate(
        self,
        system: str | None,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        on_token: Callable[[int, str], None],
        flags: int = 0,
        stop_on_text: Sequence[str] = (),
        stop_at_eos: bool = False,
    ) -> dict[str, Any]:
        request = _GenerateRequest(
            struct_size=ctypes.sizeof(_GenerateRequest),
            abi_version=self.CK_ABI_VERSION,
            system_text=(system or "").encode(),
            user_text=(user or "").encode(),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            flags=int(flags),
        )
        result = _GenerateResult()
        result.struct_size = ctypes.sizeof(_GenerateResult)
        result.abi_version = self.CK_ABI_VERSION

        stop_markers = [str(m) for m in (stop_on_text or ()) if str(m)]
        if stop_at_eos:
            stop_markers.append("<eos>")

        emitted: list[str] = []

        @_TOKEN_CALLBACK
        def callback(_user_data, token_id, text_bytes, text_len, _sequence_index):
            text = ctypes.string_at(text_bytes, text_len).decode("utf-8", "replace")
            emitted.append(text)
            if stop_markers:
                lowered = "".join(emitted).lower()
                for marker in stop_markers:
                    if marker.lower() in lowered:
                        on_token(int(token_id), text)
                        return 1
            on_token(int(token_id), text)
            return 0

        status = self.lib.ck_session_v8_generate(
            self.session,
            ctypes.byref(request),
            callback,
            None,
            ctypes.byref(result),
        )
        if status != self.CK_OK:
            raise RuntimeError(
                f"ck_session_v8_generate failed: {_last_error(self.lib, self.session)}"
            )
        return {
            "prompt_tokens": int(result.prompt_tokens),
            "generated_tokens": int(result.generated_tokens),
            "stop_reason": int(result.stop_reason),
            "prefill_time_ms": float(result.prefill_time_ms),
            "decode_time_ms": float(result.decode_time_ms),
        }

    def cancel(self) -> None:
        self.lib.ck_session_v8_cancel(self.session)

    def close(self) -> None:
        self.lib.ck_session_v8_close(self.session)
        self.session = None


# --- HTTP application ---------------------------------------------------------


def _extract_prompt(body: Any) -> str:
    if body.input is None:
        return ""
    if isinstance(body.input, str):
        return body.input
    parts: list[str] = []
    for item in body.input:
        if isinstance(item, str):
            parts.append(item)
            continue
        content = getattr(item, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)


def _usage(
    input_tokens: int, output_tokens: int, reasoning_tokens: int = 0
) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {
            "cache_write_tokens": 0,
            "cached_tokens": 0,
        },
        "output_tokens_details": {
            "reasoning_tokens": reasoning_tokens,
        },
    }


def _sse(event_type: str, data: Any) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


def _stop_reason_name(value: Any) -> str:
    try:
        return _STOP_REASON_NAMES.get(int(value), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def _truncate_stop_markers(text: str, markers: Sequence[str]) -> str:
    """Cut the final text at the first decoded stop marker (exclusive), matching
    ck_chat's ``--stop-on-text`` / ``--stop-at-eos`` response semantics."""
    active = [str(m) for m in (markers or ()) if str(m)]
    if not active or not text:
        return text
    lowered = text.lower()
    hits = [lowered.find(str(m).lower()) for m in active]
    hits = [i for i in hits if i != -1]
    if not hits:
        return text
    return text[: min(hits)]


# --- Thinking separation (text-generation only) -------------------------------

_THINK_OPEN = " thinking"
_THINK_CLOSE = " response"


def _marker_index(lowered: str, marker: str) -> int:
    """First marker occurrence, preferring a line-anchored (start-of-string or
    after ``\\n``) hit over an inline one."""
    anchored = lowered.find("\n" + marker)
    if anchored != -1:
        return anchored + 1
    if lowered.startswith(marker):
        return 0
    return lowered.find(marker)


def split_thinking(text: str) -> tuple[str, str]:
    """Split a ` thinking ... response ` block out of generated text.

    Returns ``(thinking, answer)``. Detects the open marker ` thinking` and the
    close marker ` response` (case-insensitive, preferring line-anchored
    occurrences). Qwen3-style generations already carry the open prefix in the
    prompt, so a lone close marker means everything before it is thinking. With
    no markers the whole text is the answer; an open marker without a close
    marker leaves everything after it as thinking.
    """
    if not text:
        return "", text
    lowered = text.lower()
    open_idx = _marker_index(lowered, _THINK_OPEN)
    close_idx = _marker_index(lowered, _THINK_CLOSE)
    if open_idx != -1:
        if close_idx == -1:
            return text[open_idx + len(_THINK_OPEN) :].strip(), ""
        if open_idx + len(_THINK_OPEN) <= close_idx:
            return (
                text[open_idx + len(_THINK_OPEN) : close_idx].strip(),
                text[close_idx + len(_THINK_CLOSE) :].lstrip(),
            )
    if close_idx != -1:
        return text[:close_idx].strip(), text[close_idx + len(_THINK_CLOSE) :].lstrip()
    return "", text


class _StreamThinkSplitter:
    """Streaming counterpart of ``split_thinking`` for SSE routing.

    Emits every pre-close token as ``("thinking", ...)`` immediately, holding
    back only a short lookahead tail so a close marker split across chunk
    boundaries is still detected. A leading open marker is stripped once so the
    accumulated deltas join to exactly what ``split_thinking`` reports. On the
    close marker the stream switches to ``("answer", ...)``. ``flush`` releases
    any held-back tail; if the stream ends with no close marker the caller
    reconciles the final snapshot (see ``stream_events``).
    """

    _KEEP = len(_THINK_CLOSE) + 2

    def __init__(self) -> None:
        self._look = ""
        self._mode = "thinking"
        self._open_stripped = False
        self._thinking_lstrip = True
        self._answer_lstrip = True

    def feed(self, chunk: str):
        if self._mode == "answer":
            if self._answer_lstrip:
                chunk = chunk.lstrip()
                if not chunk:
                    return
                self._answer_lstrip = False
            yield ("answer", chunk)
            return

        buf = self._look + chunk

        if not self._open_stripped:
            lowered = buf.lower()
            open_idx = _marker_index(lowered, _THINK_OPEN)
            if open_idx == 0 or (open_idx == 1 and buf.startswith("\n")):
                buf = buf[open_idx + len(_THINK_OPEN) :]
                self._open_stripped = True

        close_idx = _marker_index(buf.lower(), _THINK_CLOSE)
        if close_idx != -1:
            pre = buf[:close_idx]
            if pre:
                if self._thinking_lstrip:
                    pre = pre.lstrip()
                    self._thinking_lstrip = False
                pre = pre.rstrip()
                if pre:
                    yield ("thinking", pre)
            self._mode = "answer"
            rest = buf[close_idx + len(_THINK_CLOSE) :]
            self._look = ""
            if rest:
                if self._answer_lstrip:
                    rest = rest.lstrip()
                    if not rest:
                        return
                    self._answer_lstrip = False
                yield ("answer", rest)
            return

        emit_len = len(buf) - self._KEEP
        if emit_len > 0:
            head, buf = buf[:emit_len], buf[emit_len:]
            if self._thinking_lstrip:
                head = head.lstrip()
                if head:
                    self._thinking_lstrip = False
            if head:
                yield ("thinking", head)
        self._look = buf

    def flush(self):
        if self._mode == "answer":
            return
        if not self._look:
            return
        text = self._look
        self._look = ""
        if self._thinking_lstrip:
            text = text.lstrip()
            if not text:
                return
            self._thinking_lstrip = False
        if self._open_stripped:
            text = text.rstrip()
            if not text:
                return
        yield ("thinking", text)


# --- Performance metrics ------------------------------------------------------


def _performance_profile(result: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = int(result.get("prompt_tokens") or 0)
    generated_tokens = int(result.get("generated_tokens") or 0)
    prefill_ms = float(result.get("prefill_time_ms") or 0.0)
    decode_ms = float(result.get("decode_time_ms") or 0.0)
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prefill_ms": prefill_ms,
        "prefill_ms_per_token": round(prefill_ms / prompt_tokens, 2) if prompt_tokens > 0 else 0.0,
        "prefill_tokens_per_sec": round(1000 * prompt_tokens / prefill_ms, 2) if prefill_ms > 0 else 0.0,
        "decode_ms": decode_ms,
        "decode_ms_per_token": round(decode_ms / generated_tokens, 2) if generated_tokens > 0 else 0.0,
        "decode_tokens_per_sec": round(1000 * generated_tokens / decode_ms, 2) if decode_ms > 0 else 0.0,
        "total_ms": round(prefill_ms + decode_ms, 2),
        "stop_reason": _stop_reason_name(result.get("stop_reason")),
    }


def _log_performance(model: str, perf: dict[str, Any] | None) -> None:
    if not perf:
        return
    line = (
        f"eval time = {perf['prefill_ms']:.1f} ms prompt, "
        f"{perf['decode_ms']:.1f} ms decode "
        f"({perf['decode_ms_per_token']:.2f} ms/token, "
        f"{perf['decode_tokens_per_sec']:.1f} tokens/s), "
        f"total {perf['total_ms']:.1f} ms, stop: {perf['stop_reason']}"
    )
    print(f"{C_GRAY}{model} - {line}{C_RESET}", flush=True)


def create_app(
    session,
    *,
    model: str = "ck-v8",
    stats: bool = True,
    viz: bool = True,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 512,
    stop_on_text: Sequence[str] = (),
    stop_at_eos: bool = False,
    flags: int = 0,
):
    """Build the FastAPI app around a live session (real or injected fake)."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "fastapi is required for `serve`. Install server requirements:\n"
            "  python3 -m pip install -r server/requirements.txt"
        )
    if not _SCHEMAS_AVAILABLE:
        raise ImportError(
            "server schemas are required for `serve`; run from the C-Kernel-Engine "
            "project root or add it to PYTHONPATH."
        )

    router = APIRouter()
    response_store: dict[str, dict[str, Any]] = {}

    stop_markers = [str(m) for m in (stop_on_text or ()) if str(m)]
    all_stop_markers = list(stop_markers)
    if stop_at_eos:
        all_stop_markers.append("<eos>")

    viz_html = None
    if viz:
        viz_path = SCRIPTS_DIR / "ck_serve_viz.html"
        if viz_path.is_file():
            viz_html = viz_path.read_text(encoding="utf-8")
        else:
            log_error(f"visualizer file missing: {viz_path}; disabling /viz")

    def _conversation_echo(body) -> dict[str, Any] | None:
        conv = body.conversation
        if isinstance(conv, str):
            return {"id": conv}
        if conv is not None:
            return {"id": conv.id}
        return None

    def build_response(
        body,
        *,
        response_id: str,
        message_id: str,
        created_at: int,
        status,
        text: str,
        input_tokens: int,
        output_tokens: int,
        thinking: str | None = None,
        reasoning_tokens: int = 0,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        completed_at: int | None = None,
        item_status: str = "completed",
        reasoning_item_id: str | None = None,
        include_empty_reasoning: bool = False,
    ):
        item_status_value = (
            "in_progress" if status == ResponseStatus.in_progress else item_status
        )
        output: list[dict[str, Any]] = []
        if status != ResponseStatus.in_progress:
            if thinking is not None or include_empty_reasoning:
                output.append(
                    ReasoningItem(
                        id=reasoning_item_id or f"rsn_{uuid.uuid4().hex[:24]}",
                        status=item_status_value,
                        content=(
                            [ReasoningTextContent(text=thinking)]
                            if thinking is not None
                            else []
                        ),
                        summary=[],
                    ).model_dump()
                )
            message = ResponseOutputMessage(
                id=message_id,
                content=[ResponseOutputText(text=text)],
                role="assistant",
                status=item_status_value,
            )
            output.append(message.model_dump())

        reasoning_echo = (
            body.reasoning.model_dump() if body.reasoning is not None else None
        )

        resp: dict[str, Any] = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "completed_at": completed_at,
            "status": status,
            "error": error,
            "incomplete_details": None,
            "instructions": body.instructions,
            "metadata": body.metadata or {},
            "model": body.model or model,
            "output": output,
            "output_text": text,
            "parallel_tool_calls": (
                body.parallel_tool_calls
                if body.parallel_tool_calls is not None
                else True
            ),
            "temperature": (
                body.temperature if body.temperature is not None else temperature
            ),
            "top_p": body.top_p if body.top_p is not None else top_p,
            "top_logprobs": body.top_logprobs,
            "tool_choice": body.tool_choice,
            "tools": [t.model_dump() for t in body.tools] if body.tools else [],
            "truncation": body.truncation,
            "text": body.text.model_dump() if body.text is not None else None,
            "user": body.user,
            "background": body.background,
            "conversation": _conversation_echo(body),
            "max_output_tokens": (
                body.max_output_tokens
                if body.max_output_tokens is not None
                else max_tokens
            ),
            "max_tool_calls": body.max_tool_calls,
            "moderation": (
                body.moderation.model_dump() if body.moderation is not None else None
            ),
            "previous_response_id": body.previous_response_id,
            "prompt": body.prompt.model_dump() if body.prompt is not None else None,
            "prompt_cache_key": body.prompt_cache_key,
            "prompt_cache_options": (
                body.prompt_cache_options.model_dump()
                if body.prompt_cache_options is not None
                else None
            ),
            "prompt_cache_retention": body.prompt_cache_retention,
            "reasoning": reasoning_echo,
            "safety_identifier": body.safety_identifier,
            "service_tier": body.service_tier,
            "usage": _usage(input_tokens, output_tokens, reasoning_tokens),
        }
        if result is not None:
            resp["performance"] = _performance_profile(result)
        response_store[response_id] = resp
        return resp

    def stream_events(body, prompt, *, max_tokens, temperature, top_p):
        think_enabled = body.reasoning is not None
        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        reasoning_item_id = f"rsn_{uuid.uuid4().hex[:24]}" if think_enabled else None
        created_at = int(time.time())

        pending = build_response(
            body,
            response_id=response_id,
            message_id=message_id,
            created_at=created_at,
            status=ResponseStatus.in_progress,
            text="",
            input_tokens=0,
            output_tokens=0,
        )

        seq = 0
        yield _sse(
            "response.created",
            {"type": "response.created", "response": pending, "sequence_number": seq},
        )
        seq += 1
        yield _sse(
            "response.in_progress",
            {
                "type": "response.in_progress",
                "response": pending,
                "sequence_number": seq,
            },
        )
        seq += 1

        complete: list[str] = []
        events: "queue.Queue" = queue.Queue()
        cancelled = threading.Event()
        splitter = _StreamThinkSplitter() if think_enabled else None

        def on_token(_tid, text):
            complete.append(text)
            if splitter is not None:
                for state, delta in splitter.feed(text):
                    events.put(
                        ("reasoning_text" if state == "thinking" else "text", delta)
                    )
            else:
                events.put(("text", text))
            return -1 if cancelled.is_set() else 0

        def worker():
            try:
                result = session.generate(
                    None,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    on_token=on_token,
                    flags=flags,
                    stop_on_text=stop_markers,
                    stop_at_eos=stop_at_eos,
                )
                if splitter is not None:
                    for state, delta in splitter.flush():
                        events.put(
                            ("reasoning_text" if state == "thinking" else "text", delta)
                        )
                events.put(("done", result))
            except Exception as exc:
                events.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

        reasoning_started = False
        message_started = False

        def emit(kind, data):
            nonlocal seq
            data["sequence_number"] = seq
            seq += 1
            yield _sse(kind, data)

        while True:
            kind, payload = events.get()
            if kind == "reasoning_text":
                if not reasoning_started:
                    yield from emit(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": ReasoningItem(
                                id=reasoning_item_id,
                                status="in_progress",
                                content=[],
                                summary=[],
                            ).model_dump(),
                        },
                    )
                    reasoning_started = True
                yield from emit(
                    "response.reasoning_text.delta",
                    {
                        "type": "response.reasoning_text.delta",
                        "item_id": reasoning_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": payload,
                    },
                )
            elif kind == "text":
                if not message_started:
                    yield from emit(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": 1 if reasoning_started else 0,
                            "item": ResponseOutputMessage(
                                id=message_id,
                                content=[],
                                role="assistant",
                                status="in_progress",
                            ).model_dump(),
                        },
                    )
                    message_started = True
                yield from emit(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": message_id,
                        "output_index": 1 if reasoning_started else 0,
                        "content_index": 0,
                        "delta": payload,
                    },
                )
            elif kind == "done":
                result = payload or {}
                text = _truncate_stop_markers("".join(complete), all_stop_markers)
                thinking = None
                reasoning_tokens = 0
                if think_enabled:
                    thinking, text = split_thinking(text)
                    thinking = thinking or None
                    reasoning_tokens = len(thinking) // 4 if thinking else 0
                input_tokens = int(result.get("prompt_tokens") or 0)
                output_tokens = int(result.get("generated_tokens") or len(complete))
                message_index = 1 if reasoning_started else 0

                if reasoning_started:
                    if thinking is not None:
                        yield from emit(
                            "response.reasoning_text.done",
                            {
                                "type": "response.reasoning_text.done",
                                "item_id": reasoning_item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "text": thinking,
                            },
                        )
                    yield from emit(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": ReasoningItem(
                                id=reasoning_item_id,
                                status="completed",
                                content=(
                                    [ReasoningTextContent(text=thinking)]
                                    if thinking is not None
                                    else []
                                ),
                                summary=[],
                            ).model_dump(),
                        },
                    )
                if not message_started:
                    yield from emit(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": message_index,
                            "item": ResponseOutputMessage(
                                id=message_id,
                                content=[],
                                role="assistant",
                                status="in_progress",
                            ).model_dump(),
                        },
                    )
                yield from emit(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": message_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "text": text,
                    },
                )
                yield from emit(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": message_index,
                        "item": ResponseOutputMessage(
                            id=message_id,
                            content=[ResponseOutputText(text=text)],
                            role="assistant",
                            status="completed",
                        ).model_dump(),
                    },
                )
                final = build_response(
                    body,
                    response_id=response_id,
                    message_id=message_id,
                    created_at=created_at,
                    completed_at=int(time.time()),
                    status=ResponseStatus.completed,
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    thinking=thinking,
                    reasoning_tokens=reasoning_tokens,
                    result=result,
                    reasoning_item_id=reasoning_item_id,
                    include_empty_reasoning=reasoning_started,
                )
                if stats:
                    _log_performance(model, final.get("performance"))
                yield from emit(
                    "response.completed",
                    {"type": "response.completed", "response": final},
                )
                return
            elif kind == "error":
                final = build_response(
                    body,
                    response_id=response_id,
                    message_id=message_id,
                    created_at=created_at,
                    completed_at=int(time.time()),
                    status=ResponseStatus.failed,
                    text="".join(complete),
                    input_tokens=0,
                    output_tokens=len(complete),
                    error={"code": "server_error", "message": str(payload)},
                )
                yield from emit(
                    "response.failed",
                    {
                        "type": "response.failed",
                        "response": final,
                        "error": {"code": "server_error", "message": str(payload)},
                    },
                )
                return

    @router.post("/responses", response_model=None)
    def create_response(body: CreateResponseRequest):
        prompt = _extract_prompt(body)
        if body.instructions is not None and isinstance(body.instructions, str):
            prompt = f"{body.instructions}\n{prompt}".strip()
        tok_limit = body.max_output_tokens if body.max_output_tokens is not None else max_tokens
        temperature_eff = body.temperature if body.temperature is not None else temperature
        top_p_eff = body.top_p if body.top_p is not None else top_p
        think_enabled = body.reasoning is not None

        if body.stream:
            return StreamingResponse(
                stream_events(
                    body,
                    prompt,
                    max_tokens=tok_limit,
                    temperature=temperature_eff,
                    top_p=top_p_eff,
                ),
                media_type="text/event-stream",
            )

        chunks: list[str] = []
        result = session.generate(
            None,
            prompt,
            max_tokens=tok_limit,
            temperature=temperature_eff,
            top_p=top_p_eff,
            on_token=lambda _tid, text: chunks.append(text),
            flags=flags,
            stop_on_text=stop_markers,
            stop_at_eos=stop_at_eos,
        )
        text = _truncate_stop_markers("".join(chunks), all_stop_markers)
        thinking = None
        reasoning_tokens = 0
        if think_enabled:
            thinking, text = split_thinking(text)
            thinking = thinking or None
            reasoning_tokens = len(thinking) // 4 if thinking else 0
        input_tokens = int(result.get("prompt_tokens") or 0)
        output_tokens = int(result.get("generated_tokens") or len(chunks))
        resp = build_response(
            body,
            response_id=f"resp_{uuid.uuid4().hex[:24]}",
            message_id=f"msg_{uuid.uuid4().hex[:24]}",
            created_at=int(time.time()),
            completed_at=int(time.time()),
            status=ResponseStatus.completed,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking=thinking,
            reasoning_tokens=reasoning_tokens,
            result=result,
        )
        if stats:
            _log_performance(model, resp.get("performance"))
        return resp

    @router.get("/responses/{response_id}")
    def get_response(response_id: str):
        resp = response_store.get(response_id)
        if not resp:
            raise HTTPException(status_code=404, detail="Response not found")
        return resp

    @router.post("/responses/{response_id}/cancel")
    def cancel_response(response_id: str):
        if response_id not in response_store:
            raise HTTPException(status_code=404, detail="Response not found")
        session.cancel()
        response_store[response_id]["status"] = ResponseStatus.cancelled
        return response_store[response_id]

    @router.get("/health")
    def health():
        return {"status": "ok", "mode": "live", "inference": True}

    app = FastAPI(title="CKE v2 live inference server", version="0.2.0")
    app.include_router(router, prefix="/v1")
    app.include_router(conversations_router, prefix="/v1")

    if viz_html is not None:

        @app.get("/viz", response_class=HTMLResponse)
        def viz_page():
            return HTMLResponse(viz_html)

    return app


# -----------------------------------------------------------------------------
# CLI
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
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "ck_run pipeline failed to build the runtime.\n"
            + (proc.stderr or proc.stdout or "").strip()
        )
    return run_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cks-v8-run serve",
        description="Build and serve a v8 model over HTTP/SSE (OpenAI Responses shape).",
        epilog="Example:\n  cks-v8-run serve hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --context-len 1024 --port 8080",
    )
    parser.add_argument("model", help="GGUF source or pre-built runtime directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-name", default="ck-v8", help="Model string reported in responses")
    parser.add_argument("--run", dest="run_dir", default=None, help="Explicit run directory")
    parser.add_argument("--no-build", action="store_true", help="Skip building; require an existing run directory")

    sampler = parser.add_argument_group("sampling (server-level defaults; request body overrides)")
    sampler.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    sampler.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling (default: 1.0)")
    sampler.add_argument("--max-tokens", type=int, default=512, help="Default max output tokens when the request omits max_output_tokens")
    sampler.add_argument("--top-k", type=int, default=None, help="Top-k sampling size (accepted for parity; NOT applied by the native ABI)")
    sampler.add_argument("--min-p", type=float, default=None, help="Min-p filter as fraction of max prob (accepted for parity; NOT applied by the native ABI)")
    sampler.add_argument("--repeat-penalty", type=float, default=None, help="Repeat penalty >1.0 reduces looping (accepted for parity; NOT applied by the native ABI)")
    sampler.add_argument("--repeat-last-n", type=int, default=None, help="Window size for repeat penalty (accepted for parity; NOT applied by the native ABI)")
    sampler.add_argument("--no-repeat-ngram-size", type=int, default=None, help="Block tokens that repeat an n-gram of this size (accepted for parity; NOT applied by the native ABI)")

    stop = parser.add_argument_group("stop markers (honored via the token callback)")
    stop.add_argument("--stop-on-text", action="append", default=[], help="Stop generation when this decoded text marker appears (repeatable)")
    stop.add_argument("--stop-at-eos", action="store_true", help="Stop generation when '<eos>' appears in decoded text")

    display = parser.add_argument_group("metrics / visualizer")
    display.add_argument("--stats", action="store_true", default=True, help="Print per-request performance stats (default: on)")
    display.add_argument("--no-stats", action="store_false", dest="stats", help="Disable per-request performance stats")
    display.add_argument("--no-viz", action="store_true", help="Disable the live HTML visualizer page at /viz")

    build = parser.add_argument_group("build / tokenizer flags (mirrors cks-v8-run run)")
    build.add_argument("--context-len", type=int, default=None)
    build.add_argument("--logits-layout", choices=["auto", "last", "full"], default=None)
    build.add_argument("--chat-template", default=None, help="Chat template to compile in")
    build.add_argument("--no-chat-template", action="store_true")
    build.add_argument("--allow-raw-prompt", action="store_true", help="Forward to the build; with --no-chat-template also set the RAW_PROMPT request flag")
    build.add_argument("--python-tokenizer", action="store_true")
    build.add_argument("--profile", action="store_true", help="Emit CK_PROFILE timing wrappers")
    build.add_argument("--gemm-schedule", choices=("auto", "static", "dynamic"), default=None)
    build.add_argument("--force-download", action="store_true")
    build.add_argument("--force-convert", action="store_true")
    build.add_argument("--force-compile", action="store_true")
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
            f"temperature/top_p/max_tokens. Ignored until a native ABI extension exists: "
            + ", ".join(ignored),
            C_ORANGE,
        )

    log(f"Opening session {run_dir} ...")
    session = SessionV8.open(
        run_dir,
        context_length=args.context_len,
    )
    app = create_app(
        session,
        model=args.model_name,
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
    )

    try:
        import uvicorn
    except ImportError as exc:
        session.close()
        raise ImportError(
            "uvicorn is required to run the server. Install server requirements:\n"
            "  python3 -m pip install -r server/requirements.txt"
        ) from exc

    log(f"Serving on http://{args.host}:{args.port}  (mode=live, inference=True)", C_GREEN)
    if not args.no_viz:
        log(f"Visualizer: http://{args.host}:{args.port}/viz", C_GREEN)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
