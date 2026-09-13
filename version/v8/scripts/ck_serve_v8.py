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

**Single-flight**: only one ``generate`` request is processed at a time.
Concurrent requests receive HTTP 429 (Too Many Requests) or an SSE
``response.failed`` event with ``code: "session_busy"``.  The native C layer
enforces this via ``pthread_mutex_trylock``; the Python server adds an
additional Python-level guard for duck-typed sessions.

The production host is a dedicated C or Rust server. This Python/ctypes module
is a reference implementation of the same ABI binding.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence, Callable
from pathlib import Path
from typing import Any

try:
    import jinja2
    import jinja2.sandbox  # noqa: F401 ensure sandbox submodule loaded

    _JINJA_AVAILABLE = True
except ImportError:
    jinja2 = None  # type: ignore
    _JINJA_AVAILABLE = False

# SCRIPTS_DIR/PROJECT_ROOT must be on sys.path before importing ck_* helpers
# (supports both ``python version/v8/scripts/ck_serve_v8.py`` and
# ``python -m version.v8.scripts.ck_serve_v8`` / pytest shims).
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[2]
BUILD_DIR = PROJECT_ROOT / "build"
SESSION_LIB_PATH = BUILD_DIR / "libck_session_v8.so"

for _p in (str(SCRIPTS_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if __package__:
    from . import ck_serve_runtime_v8
else:
    import ck_serve_runtime_v8

# Reuse color/logging constants from the runtime module so both entrypoints
# share identical console styling.
C_GREEN = ck_serve_runtime_v8.C_GREEN
C_ORANGE = ck_serve_runtime_v8.C_ORANGE
C_RED = ck_serve_runtime_v8.C_RED
C_GRAY = ck_serve_runtime_v8.C_GRAY
C_RESET = ck_serve_runtime_v8.C_RESET

CK_SESSION_REQUEST_RAW_PROMPT = 1 << 0

_STOP_REASON_NAMES = {
    0: "none",
    1: "eos",
    2: "token_limit",
    3: "cancelled",
    4: "callback",
    5: "runtime_error",
}

# --- Native session status codes -----------------------------------------------

_SESSION_STATUS_NAMES: dict[int, str] = {
    0: "ok",
    -1: "invalid_argument",
    -2: "abi",
    -3: "load",
    -4: "init",
    -5: "capability",
    -6: "busy",
    -7: "runtime",
    -8: "buffer_too_small",
}


class SessionError(RuntimeError):
    """Base exception for native ck_session_v8 errors."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.status_name = _SESSION_STATUS_NAMES.get(status, f"unknown({status})")
        super().__init__(message)


class SessionBusyError(SessionError):
    """Raised when the session already has an active generate request (status -6).

    The server is single-flight: only one generate request is processed at a
    time.  Concurrent requests receive HTTP 429 (Too Many Requests).
    """


_SESSION_STATUS_EXCEPTIONS: dict[int, type[SessionError]] = {
    -6: SessionBusyError,
}


# Third-party and local schema imports are optional at module load so the CLI
# (--help / pipeline) works even without the server extras. They must live at
# module scope so FastAPI/Pydantic can resolve route annotations.
try:
    from fastapi import APIRouter, FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, StreamingResponse

    _FASTAPI_AVAILABLE = True
except ImportError:
    APIRouter = FastAPI = HTTPException = StreamingResponse = Request = None
    _FASTAPI_AVAILABLE = False

try:
    from server.schemas.common import ResponseStatus
    from server.schemas.content import ResponseOutputText
    from server.schemas.output_items import (
        FunctionCall,
        ReasoningItem,
        ReasoningTextContent,
        ResponseOutputMessage,
    )
    from server.schemas.response import CreateResponseRequest
    from server.routes.conversations import router as conversations_router

    _SCHEMAS_AVAILABLE = True
except ImportError:
    _SCHEMAS_AVAILABLE = False


log = ck_serve_runtime_v8.log
log_error = ck_serve_runtime_v8.log_error


def _detect_threads() -> int:
    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, len(os.sched_getaffinity(0)))
    except OSError:
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
        return value.decode("utf-8", "replace") if value else "Unknown error occured"
    except Exception as e:
        return f"Unknown error occured during session. Error -> {e}"


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
    """ctypes binding for the session ABI (one process-local model load).

    The native layer serialises concurrent ``ck_session_v8_generate`` calls via
    a pthread mutex try-lock.  When the lock is held the C function returns
    ``CK_SESSION_V8_ERROR_BUSY`` (-6), which surfaces as
    :class:`SessionBusyError` in Python.
    """

    CK_ABI_VERSION = 1
    CK_OK = 0

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @classmethod
    def open(
        cls,
        work_dir: Path,
        *,
        context_length: int | None = None,
        num_threads: int | None = None,
    ) -> SessionV8:
        work_dir = work_dir.resolve()
        for name in ("libmodel.so", "weights.bump", "weights_manifest.map"):
            if not (work_dir / name).is_file():
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
            msg = f"ck_session_v8_open failed ({_SESSION_STATUS_NAMES.get(status, f'status={status}')})"
            native_msg = _last_error(lib, session)
            if native_msg:
                msg += f": {native_msg}"
            exc_cls = _SESSION_STATUS_EXCEPTIONS.get(status, SessionError)
            raise exc_cls(status, msg)

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
        on_token: Callable[[int, str], int],
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
        visible_len = 0
        callback_stopped = False
        last_token_id = -1

        @_TOKEN_CALLBACK
        def callback(_user_data, token_id, text_bytes, text_len, _sequence_index):
            nonlocal callback_stopped, last_token_id, visible_len
            last_token_id = int(token_id)
            text = ctypes.string_at(text_bytes, text_len).decode("utf-8", "replace")
            emitted.append(text)
            joined = "".join(emitted)
            truncated = (
                _truncate_stop_markers(joined, stop_markers) if stop_markers else joined
            )
            hit = len(truncated) < len(joined)
            if hit:
                delta = truncated[visible_len:]
            else:
                hold = (
                    _longest_marker_prefix_len(joined, stop_markers)
                    if stop_markers
                    else 0
                )
                # Don't hold more than the not-yet-emitted tail
                hold = min(hold, len(joined) - visible_len)
                target = len(joined) - hold
                delta = joined[visible_len:target] if target > visible_len else ""
            rc = 0
            if delta:
                try:
                    rc = int(on_token(int(token_id), delta) or 0)
                except Exception:
                    rc = 0
            else:
                # No visible delta (buffered prefix or marker consumed the whole token);
                # still allow cancellation to propagate via on_token.
                try:
                    rc = int(on_token(int(token_id), "") or 0)
                    # Empty delta should not enqueue SSE; caller handles empty string as no-op.
                except Exception:
                    rc = 0
                if rc == 0 and hit:
                    return 1
            visible_len += len(delta)
            if rc != 0:
                callback_stopped = True
                return rc
            if hit:
                return 1
            return 0

        status = self.lib.ck_session_v8_generate(
            self.session,
            ctypes.byref(request),
            callback,
            None,
            ctypes.byref(result),
        )
        if status == self.CK_OK and not callback_stopped:
            final_text = _truncate_stop_markers("".join(emitted), stop_markers)
            pending = final_text[visible_len:]
            if pending:
                try:
                    on_token(last_token_id, pending)
                except Exception:
                    pass
        if status != self.CK_OK:
            msg = f"ck_session_v8_generate failed ({_SESSION_STATUS_NAMES.get(status, f'status={status}')})"
            native_msg = _last_error(self.lib, self.session)
            if native_msg:
                msg += f": {native_msg}"
            exc_cls = _SESSION_STATUS_EXCEPTIONS.get(status, SessionError)
            raise exc_cls(status, msg)
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


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        text = getattr(part, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _input_chat_messages(value: Any) -> list[dict[str, Any]]:
    """Convert Responses input items to role-preserving chat-template messages."""
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]

    messages: list[dict[str, Any]] = []
    for item in value:
        item_type = getattr(item, "type", None)
        if item_type == "message" or (
            item_type is None and getattr(item, "role", None) is not None
        ):
            role = getattr(item, "role", "user")
            role = getattr(role, "value", role)
            messages.append(
                {"role": str(role), "content": _content_text(item.content)}
            )
        elif item_type == "function_call":
            try:
                arguments = json.loads(item.arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"function_call {item.call_id!r} has invalid JSON arguments"
                ) from exc
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": item.name,
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "content": _content_text(item.output),
                    "tool_call_id": item.call_id,
                }
            )
    return messages


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


def _longest_marker_prefix_len(text: str, markers: Sequence[str]) -> int:
    """Longest suffix of text that is a proper prefix of any marker (case-insensitive).

    Used to buffer a split-boundary stop marker so the prefix is not
    emitted as a delta before the next token proves whether the marker
    completes.
    """
    if not markers or not text:
        return 0
    low = text.lower()
    best = 0
    for raw in markers:
        m = str(raw).lower()
        if not m:
            continue
        # proper prefix only (k < len(m)); full match is a hit, not a hold
        upper = len(m) - 1
        # cap by text length
        upper = min(upper, len(low))
        for k in range(1, upper + 1):
            if low.endswith(m[:k]) and k > best:
                best = k
    return best


# --- Thinking separation (text-generation only) -------------------------------

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


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

    Before a marker is seen, text is buffered because it may be either a plain
    answer or Qwen-style close-only reasoning. This avoids emitting reasoning
    deltas that the completed response later reclassifies as answer text.
    """

    _KEEP = len(_THINK_CLOSE) + 2

    def __init__(self) -> None:
        self._look = ""
        self._mode = "undetermined"
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
        if self._mode == "undetermined":
            open_idx = _marker_index(buf.lower(), _THINK_OPEN)
            close_idx = _marker_index(buf.lower(), _THINK_CLOSE)
            if open_idx != -1 and (close_idx == -1 or open_idx < close_idx):
                buf = buf[open_idx + len(_THINK_OPEN) :]
                self._mode = "thinking"
            elif close_idx != -1:
                pre = buf[:close_idx].strip()
                if pre:
                    yield ("thinking", pre)
                self._mode = "answer"
                rest = buf[close_idx + len(_THINK_CLOSE) :].lstrip()
                self._look = ""
                if rest:
                    self._answer_lstrip = False
                    yield ("answer", rest)
                return
            else:
                self._look = buf
                return

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
        if self._mode == "undetermined":
            yield ("answer", text)
            return
        if self._thinking_lstrip:
            text = text.lstrip()
            if not text:
                return
            self._thinking_lstrip = False
        if self._mode == "thinking":
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
        "prefill_ms_per_token": round(prefill_ms / prompt_tokens, 2)
        if prompt_tokens > 0
        else 0.0,
        "prefill_tokens_per_sec": round(1000 * prompt_tokens / prefill_ms, 2)
        if prefill_ms > 0
        else 0.0,
        "decode_ms": decode_ms,
        "decode_ms_per_token": round(decode_ms / generated_tokens, 2)
        if generated_tokens > 0
        else 0.0,
        "decode_tokens_per_sec": round(1000 * generated_tokens / decode_ms, 2)
        if decode_ms > 0
        else 0.0,
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


# ---------------------------------------------------------------------------
# Python-side chat contract loading and prompt formatting
# ---------------------------------------------------------------------------


def _load_runtime_chat_contract(run_dir: Path) -> dict[str, Any] | None:
    """Load the exact chat contract exported by the built runtime.

    Delegates to ``ck_serve_runtime_v8.load_manifest_templates`` so the manifest
    is the single source of truth. Kept for backward compatibility with
    ``server/tests/test_live_app.py``.
    """
    try:
        _, _, contract = ck_serve_runtime_v8.load_manifest_templates(run_dir)
        return contract
    except RuntimeError:
        raise
    except Exception:
        return None


def _load_runtime_templates(
    run_dir: Path,
) -> tuple[str | None, dict[str, str] | None, dict[str, Any] | None]:
    """Return (chat_template, chat_templates, chat_contract) from manifest."""
    return ck_serve_runtime_v8.load_manifest_templates(run_dir)


# Backward-compat stub: previously loaded from version/v8/circuits/*.json.
# Now removed — manifest is single source. Keep symbol so imports do not break.
def _load_builtin_chat_contract(
    template_name: str | None,
    *,
    _seen: set[str] | None = None,
) -> dict[str, Any] | None:
    return None


def _resolve_contract_thinking_overrides(
    contract: dict[str, Any],
    thinking_mode: str | None,
) -> tuple[str, str]:
    assistant_generation_prefix = str(contract.get("assistant_generation_prefix") or "")
    last_user_prefix = str(contract.get("last_user_prefix") or "")
    requested_mode = str(thinking_mode or "auto").strip().lower()
    default_mode = str(contract.get("thinking_mode_default") or "").strip().lower()
    resolved_mode = default_mode if requested_mode in {"", "auto"} else requested_mode

    assistant_by_mode = contract.get("assistant_generation_prefix_by_thinking_mode")
    if isinstance(assistant_by_mode, dict):
        override = assistant_by_mode.get(resolved_mode)
        if isinstance(override, str):
            assistant_generation_prefix = override

    last_user_prefix_by_mode = contract.get("last_user_prefix_by_thinking_mode")
    if isinstance(last_user_prefix_by_mode, dict):
        override = last_user_prefix_by_mode.get(resolved_mode)
        if isinstance(override, str):
            last_user_prefix = override

    return assistant_generation_prefix, last_user_prefix


def _format_prompt_with_chat_contract(
    prompt: str,
    contract: dict[str, Any] | None,
    *,
    thinking_mode: str = "auto",
    system_prompt: str | None = None,
) -> str:
    if not isinstance(contract, dict):
        return str(prompt or "")

    role_labels = (
        contract.get("role_labels")
        if isinstance(contract.get("role_labels"), dict)
        else {}
    )
    turn_prefix = str(contract.get("turn_prefix") or "")
    turn_suffix = str(contract.get("turn_suffix") or "")
    system_prompt_mode = (
        str(contract.get("system_prompt_mode") or "disabled").strip().lower()
    )
    system_prompt_separator = str(contract.get("system_prompt_separator") or "\n\n")
    default_system_prompt = str(contract.get("default_system_prompt") or "")
    inject_default_system_prompt = bool(contract.get("inject_default_system_prompt"))
    bos_prefix = str(contract.get("force_bos_text_if_tokenizer_add_bos_false") or "")
    suppression_markers = [
        str(marker).lower()
        for marker in list(contract.get("last_user_prefix_suppression_markers") or [])
        if str(marker or "").strip()
    ]
    assistant_generation_prefix, last_user_prefix = (
        _resolve_contract_thinking_overrides(contract, thinking_mode)
    )

    user_text = str(prompt or "")
    if last_user_prefix:
        lowered = user_text.lower()
        if last_user_prefix.lower() not in lowered and not any(
            marker in lowered for marker in suppression_markers
        ):
            user_text = f"{last_user_prefix}{user_text}"

    system_text = str(system_prompt or "")
    if not system_text and inject_default_system_prompt:
        system_text = default_system_prompt

    if system_text and system_prompt_mode == "prepend_first_user":
        user_text = (
            f"{system_text}{system_prompt_separator}{user_text}"
            if user_text
            else system_text
        )
        system_text = ""

    def _render_turn(role: str, content: str) -> str:
        label = str(role_labels.get(role) or role)
        prefix = turn_prefix.replace("{role}", label)
        return f"{prefix}{content}{turn_suffix}"

    formatted = ""
    if bos_prefix:
        formatted += bos_prefix
    if system_text and system_prompt_mode == "dedicated_turn":
        formatted += _render_turn("system", system_text)
    formatted += _render_turn("user", user_text)
    formatted += assistant_generation_prefix
    return formatted if formatted else user_text


# --- Tool-call helpers -------------------------------------------------------


def _has_tool_support(
    chat_template: str | None, chat_templates: dict[str, str] | None
) -> bool:
    # Primary signal: plural chat_templates with tool_use variant
    if isinstance(chat_templates, dict) and chat_templates:
        for key in ("tool_use", "tools", "default"):
            if isinstance(chat_templates.get(key), str) and chat_templates[key].strip():
                return True
        if any(isinstance(v, str) and v.strip() for v in chat_templates.values()):
            return True
    # Fallback: single chat_template string that handles tools (Qwen3 {% if tools %})
    if isinstance(chat_template, str) and chat_template.strip():
        # Heuristic: template that branches on tools and emits tool_call
        lower = chat_template.lower()
        if "tools" in lower and ("tool_call" in lower or "tool" in lower):
            return True
        # If template mentions tools at all, assume it supports them via jinja
        if "tools" in lower:
            return True
    return False


def _render_with_chat_templates(
    chat_template: str | None,
    chat_templates: dict[str, str] | None,
    messages: list[dict[str, Any]],
    body: Any,
    chat_contract: dict[str, Any] | None = None,
    effective_thinking: str = "suppressed",
) -> str | None:
    """Try to render a jinja chat template with tools. Returns None on failure or absence."""
    if not _JINJA_AVAILABLE:
        return None
    # Prefer tool_use variant when tools are present
    tmpl_str: str | None = None
    if (
        body is not None
        and getattr(body, "tools", None)
        and isinstance(chat_templates, dict)
    ):
        for key in ("tool_use", "tools", "default"):
            candidate = chat_templates.get(key)
            if isinstance(candidate, str) and candidate.strip():
                tmpl_str = candidate
                break
    if tmpl_str is None and isinstance(chat_template, str) and chat_template.strip():
        tmpl_str = chat_template
    if not tmpl_str:
        return None
    try:
        instructions = getattr(body, "instructions", None) if body is not None else None
        if isinstance(instructions, str) and instructions.strip():
            messages = [{"role": "system", "content": instructions}, *messages]
        tools = None
        if body is not None and getattr(body, "tools", None):
            tools = [t.model_dump() for t in body.tools]  # type: ignore
        env = jinja2.sandbox.SandboxedEnvironment(
            undefined=jinja2.StrictUndefined, autoescape=False
        )  # type: ignore
        tmpl = env.from_string(tmpl_str)
        # Qwen-like templates expect enable_thinking flag
        rendered = tmpl.render(
            messages=messages,
            tools=tools,
            tool_choice=getattr(body, "tool_choice", None)
            if body is not None
            else None,
            enable_thinking=(effective_thinking == "visible"),
            add_generation_prompt=True,
        )
        return str(rendered)
    except Exception as exc:
        try:
            log_error(
                f"jinja render failed (fallback to contract): {exc} template_len={len(tmpl_str)}"
            )
        except Exception:
            pass
        return None


def _extract_tool_calls_from_text(
    text: str,
    allowed_names: set[str] | None,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Parse tool calls from generated text.

    Returns (tool_calls, error_code, error_message).
    - Malformed JSON -> error_code="malformed"
    - Unknown tool name -> error_code="unknown"
    - Otherwise tool_calls is list of {name, arguments:str, call_id}
    """
    if not text or not text.strip():
        return [], None, None
    stripped = text.strip()
    # 1) Try <tool_call>...</tool_call> blocks (Qwen style) — still support if present in text
    tool_blocks: list[str] = []
    import re as _re

    for m in _re.finditer(
        r"<tool_call>(.*?)</tool_call>", text, flags=_re.DOTALL | _re.IGNORECASE
    ):
        tool_blocks.append(m.group(1).strip())
    # also handle bare JSON objects
    candidates: list[str] = []
    if tool_blocks:
        candidates = tool_blocks
    else:
        # If whole text is a JSON object or array, use it directly
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            candidates = [stripped]
        else:
            # scan for balanced JSON objects containing "name"
            # naive brace matching
            stack = []
            start = -1
            for idx, ch in enumerate(text):
                if ch == "{":
                    if not stack:
                        start = idx
                    stack.append(ch)
                elif ch == "}":
                    if stack:
                        stack.pop()
                        if not stack and start != -1:
                            snippet = text[start : idx + 1]
                            if '"name"' in snippet:
                                candidates.append(snippet)
                            start = -1
            # trailing incomplete JSON -> malformed
            if stack and start != -1:
                snippet = text[start:]
                if '"name"' in snippet or '"arguments"' in snippet:
                    try:
                        json.loads(snippet)
                    except json.JSONDecodeError as exc:
                        return [], "malformed", f"malformed tool call: {exc}"
                    # if it parses but incomplete, still consider malformed
                    return [], "malformed", "malformed tool call: incomplete JSON"
    tool_calls: list[dict[str, Any]] = []
    for snippet in candidates:
        try:
            obj = json.loads(snippet)
        except json.JSONDecodeError as exc:
            return [], "malformed", f"malformed tool call: {exc}"
        # obj may be dict or list
        items = obj if isinstance(obj, list) else [obj]
        for item in items:
            if not isinstance(item, dict):
                return [], "malformed", "malformed tool call: expected object"
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                # try alternative keys like function.name
                func = item.get("function")
                if isinstance(func, dict):
                    name = func.get("name")
                if not isinstance(name, str) or not name.strip():
                    return [], "malformed", "malformed tool call: missing name"
            name = str(name).strip()
            if allowed_names is not None and name not in allowed_names:
                return [], "unknown", f"unknown tool {name!r}"
            args = item.get("arguments")
            if args is None and isinstance(item.get("function"), dict):
                args = item["function"].get("arguments")
            # arguments may be dict or string
            if isinstance(args, dict):
                args_str = json.dumps(args, separators=(",", ":"))
            elif isinstance(args, str):
                # validate that string is valid JSON if non-empty
                if args.strip():
                    try:
                        json.loads(args)
                    except json.JSONDecodeError as exc:
                        return [], "malformed", f"malformed tool call arguments: {exc}"
                args_str = args
            elif args is None:
                args_str = "{}"
            else:
                return [], "malformed", "malformed tool call: invalid arguments"
            tool_calls.append(
                {
                    "name": name,
                    "arguments": args_str,
                    "call_id": f"call_{uuid.uuid4().hex[:24]}",
                }
            )
    return tool_calls, None, None


def _strip_tool_json_from_text(
    text: str, tool_calls: list[dict[str, Any]] | None
) -> str:
    """Remove tool JSON snippets from text to get residual message text."""
    if not text or not tool_calls:
        return text
    stripped = text.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            json.loads(stripped)
            return ""
        except json.JSONDecodeError:
            pass
    remaining = text
    # remove <tool_call> blocks
    import re as _re2

    remaining = _re2.sub(
        r"<tool_call>.*?</tool_call>",
        "",
        remaining,
        flags=_re2.DOTALL | _re2.IGNORECASE,
    )
    # remove json snippets that were parsed — naive: remove each json dump if present
    for tc in tool_calls:
        # try to find json containing name
        pattern = _re2.escape(tc["name"])
        # just strip any json object containing the name on a best-effort basis
        # fallback: remove first json object containing the name
        m = _re2.search(r"\{[^}]*" + pattern + r"[^}]*\}", remaining)
        if m:
            remaining = remaining[: m.start()] + remaining[m.end() :]
    return remaining.strip()


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
    chat_contract: dict[str, Any] | None = None,
    chat_template: str | None = None,
    chat_templates: dict[str, str] | None = None,
    cancel_wait_seconds: float = 10.0,
):
    """Build the FastAPI app around a live session (real or injected fake).

    This server is single-flight: only one generate request is processed at a
    time.  Concurrent requests receive HTTP 429 (Too Many Requests).
    """
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
    response_store: OrderedDict[str, dict[str, Any]] = OrderedDict()
    response_history_store: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    response_store_lock = threading.Lock()
    response_store_limit = 256
    _flight_lock = threading.Lock()
    # Per-response cancellation registry: response_id -> {cancelled, finished, thread}
    active_streams: dict[str, dict[str, Any]] = {}
    active_streams_lock = threading.Lock()

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

    def _validate_request(body) -> None:
        if body.model != model:
            raise HTTPException(
                status_code=404,
                detail=f"Model {body.model!r} is not loaded; available model: {model!r}",
            )
        if body.tools:
            if not _has_tool_support(chat_template, chat_templates):
                raise HTTPException(
                    status_code=501,
                    detail="Model doesn't support tool calling",
                )

    def _store_response(
        response_id: str,
        response: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> None:
        with response_store_lock:
            response_store[response_id] = response
            response_history_store[response_id] = history
            response_store.move_to_end(response_id)
            response_history_store.move_to_end(response_id)
            while len(response_store) > response_store_limit:
                evicted_id, _ = response_store.popitem(last=False)
                response_history_store.pop(evicted_id, None)

    def _request_messages(body) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if body.previous_response_id:
            with response_store_lock:
                previous = response_history_store.get(body.previous_response_id)
            if previous is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Previous response {body.previous_response_id!r} not found",
                )
            messages.extend(dict(message) for message in previous)

        try:
            current = _input_chat_messages(body.input)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        known_call_ids = {
            call.get("id")
            for message in [*messages, *current]
            for call in message.get("tool_calls", [])
            if isinstance(call, dict) and call.get("id")
        }
        for message in current:
            if message.get("role") != "tool":
                continue
            call_id = message.get("tool_call_id")
            if call_id not in known_call_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Function output references unknown call_id {call_id!r}",
                )
        messages.extend(current)
        return messages

    def _prepare_request(body):
        _validate_request(body)
        messages = _request_messages(body)
        prompt = _extract_prompt(body)
        tok_limit = (
            body.max_output_tokens if body.max_output_tokens is not None else max_tokens
        )
        temperature_eff = (
            body.temperature if body.temperature is not None else temperature
        )
        top_p_eff = body.top_p if body.top_p is not None else top_p
        effective_flags = flags

        # Thinking: absent -> suppressed, present -> visible (ignore contract default)
        effective_thinking = "visible" if body.reasoning is not None else "suppressed"

        # Try Jinja rendering when we have a template (tool_use variant preferred)
        jinja_rendered: str | None = None
        if (chat_template is not None or chat_templates is not None) and (
            chat_contract is not None or body.tools
        ):
            jinja_rendered = _render_with_chat_templates(
                chat_template,
                chat_templates,
                messages,
                body,
                chat_contract,
                effective_thinking,
            )

        requires_role_rendering = any(
            message.get("role") != "user" or message.get("tool_calls")
            for message in messages
        )
        if requires_role_rendering and jinja_rendered is None:
            raise HTTPException(
                status_code=422,
                detail="The selected model template cannot render role-aware tool history",
            )

        if jinja_rendered is not None and jinja_rendered.strip():
            prompt = jinja_rendered
            effective_flags |= CK_SESSION_REQUEST_RAW_PROMPT
        elif chat_contract is not None:
            prompt = _format_prompt_with_chat_contract(
                prompt,
                chat_contract,
                thinking_mode=effective_thinking,
                system_prompt=body.instructions,
            )
            effective_flags |= CK_SESSION_REQUEST_RAW_PROMPT
        elif isinstance(body.instructions, str):
            prompt = f"{body.instructions}\n{prompt}".strip()

        # Guard: never send empty prompt to native (causes embed failed n<=0)
        if not prompt or not prompt.strip():
            prompt = _extract_prompt(body) or "Hello"
            if isinstance(body.instructions, str) and body.instructions.strip():
                prompt = f"{body.instructions}\n{prompt}".strip()
            if not prompt.strip():
                prompt = "Hello"

        return prompt, tok_limit, temperature_eff, top_p_eff, effective_flags

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
        incomplete_details: dict[str, Any] | None = None,
        completed_at: int | None = None,
        item_status: str = "completed",
        reasoning_item_id: str | None = None,
        include_empty_reasoning: bool = False,
        tool_calls: list[dict[str, Any]] | None = None,
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
            if tool_calls:
                for tc in tool_calls:
                    output.append(
                        FunctionCall(
                            id=tc.get("id") or f"fc_{uuid.uuid4().hex[:24]}",
                            call_id=tc.get("call_id")
                            or f"call_{uuid.uuid4().hex[:24]}",
                            name=tc.get("name") or "",
                            arguments=tc.get("arguments") or "{}",
                            status=item_status_value,  # type: ignore
                        ).model_dump()
                    )
                # Also keep message if there is leftover text after tool extraction
                if text and text.strip():
                    message = ResponseOutputMessage(
                        id=message_id,
                        content=[ResponseOutputText(text=text)],
                        role="assistant",
                        status=item_status_value,
                    )
                    output.append(message.model_dump())
            else:
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
            "incomplete_details": incomplete_details,
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
        # Respect store:false → ephemeral (OpenAI spec) — do not persist
        should_store = True
        try:
            if getattr(body, "store", None) is False:
                should_store = False
        except Exception:
            pass
        if should_store:
            history = _request_messages(body)
            assistant: dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": call.get("name"),
                            "arguments": json.loads(call.get("arguments") or "{}"),
                        },
                    }
                    for call in tool_calls
                ]
            _store_response(response_id, resp, [*history, assistant])
        return resp

    def stream_events(
        body,
        prompt,
        *,
        max_tokens,
        temperature,
        top_p,
        effective_flags=None,
        request=None,
    ):
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
        events: queue.Queue = queue.Queue()
        cancelled = threading.Event()
        worker_finished = threading.Event()
        splitter = _StreamThinkSplitter() if think_enabled else None
        # Will register after worker_thread is created (needs thread ref)
        worker_thread_ref: list[threading.Thread | None] = [None]

        def on_token(_tid, text):
            if text:
                complete.append(text)
                if not body.tools:
                    if splitter is not None:
                        for state, delta in splitter.feed(text):
                            events.put(
                                (
                                    "reasoning_text"
                                    if state == "thinking"
                                    else "text",
                                    delta,
                                )
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
                    flags=effective_flags if effective_flags is not None else flags,
                    stop_on_text=stop_markers,
                    stop_at_eos=stop_at_eos,
                )
                if splitter is not None and not body.tools:
                    for state, delta in splitter.flush():
                        events.put(
                            ("reasoning_text" if state == "thinking" else "text", delta)
                        )
                terminal_event = ("done", result)
            except SessionBusyError:
                terminal_event = (
                    "busy",
                    "Session busy: another request is in progress.",
                )
            except Exception as e:
                terminal_event = ("error", str(e))
            finally:
                worker_finished.set()
            events.put(terminal_event)

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread_ref[0] = worker_thread
        # Register per-response cancellation state before starting
        with active_streams_lock:
            active_streams[response_id] = {
                "cancelled": cancelled,
                "finished": worker_finished,
                "thread": worker_thread,
            }
        worker_thread.start()

        reasoning_started = False
        message_started = False
        message_content_part_added = False

        def emit(kind, data):
            nonlocal seq
            data["sequence_number"] = seq
            seq += 1
            yield _sse(kind, data)

        try:
            while True:
                try:
                    kind, payload = events.get(timeout=0.2)
                except queue.Empty:
                    # Poll for client disconnect / cancellation
                    if request is not None:
                        try:
                            # Starlette Request.is_disconnected is async; try sync poll
                            import asyncio as _asyncio

                            try:
                                is_disc = _asyncio.run(request.is_disconnected())  # type: ignore
                            except RuntimeError:
                                # Already in event loop — schedule check via loop
                                loop = _asyncio.get_event_loop()
                                if loop.is_running():
                                    # Cannot block; fallback to GeneratorExit handling
                                    is_disc = False
                                else:
                                    is_disc = loop.run_until_complete(
                                        request.is_disconnected()
                                    )  # type: ignore
                            if is_disc:
                                if not cancelled.is_set():
                                    cancelled.set()
                                    try:
                                        session.cancel()
                                    except Exception:
                                        pass
                                # Keep looping to drain terminal event
                        except Exception:
                            pass
                    if cancelled.is_set() and worker_finished.is_set():
                        # Worker cancelled, terminal event should be next
                        continue
                    if worker_finished.is_set():
                        # No more events expected but queue empty
                        continue
                    continue
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
                    # B parity: message at 1 when reasoning requested (even if empty)
                    msg_idx = 1 if think_enabled else 0
                    if not message_started:
                        yield from emit(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": msg_idx,
                                "item": ResponseOutputMessage(
                                    id=message_id,
                                    content=[],
                                    role="assistant",
                                    status="in_progress",
                                ).model_dump(),
                            },
                        )
                        message_started = True
                        # OpenAI streams content_part lifecycle for text
                        yield from emit(
                            "response.content_part.added",
                            {
                                "type": "response.content_part.added",
                                "item_id": message_id,
                                "output_index": msg_idx,
                                "content_index": 0,
                                "part": {
                                    "type": "output_text",
                                    "text": "",
                                    "annotations": [],
                                },
                            },
                        )
                        message_content_part_added = True
                    yield from emit(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": message_id,
                            "output_index": msg_idx,
                            "content_index": 0,
                            "delta": payload,
                            "logprobs": None,
                        },
                    )
                elif kind == "done":
                    result = payload or {}
                    # If cancellation was requested, prefer cancelled status
                    stop_reason_val = int(result.get("stop_reason") or 0)
                    is_cancelled = cancelled.is_set() or stop_reason_val == 3
                    text = _truncate_stop_markers("".join(complete), all_stop_markers)
                    thinking = None
                    reasoning_tokens = 0
                    if think_enabled:
                        thinking, text = split_thinking(text)
                        thinking = thinking or None
                        reasoning_tokens = 0
                    input_tokens = int(result.get("prompt_tokens") or 0)
                    output_tokens = int(result.get("generated_tokens") or len(complete))
                    message_index = 1 if think_enabled else 0

                    # --- Tool-call detection (manifest chat_templates-gated) ---
                    tool_calls: list[dict[str, Any]] | None = None
                    tool_error_code: str | None = None
                    tool_error_msg: str | None = None
                    has_tools = bool(getattr(body, "tools", None))
                    if has_tools:
                        allowed = {
                            t.name
                            for t in body.tools  # type: ignore
                            if getattr(t, "type", None) == "function"
                            and getattr(t, "name", None)
                        }
                        tool_calls, tool_error_code, tool_error_msg = (
                            _extract_tool_calls_from_text(text, allowed)
                        )
                        if tool_error_code is None and not tool_calls:
                            tool_calls = None  # no tool JSON => plain text
                        elif tool_calls == [] and tool_error_code is None:
                            tool_calls = None

                    # Handle malformed / unknown as failed (unless cancelled)
                    if has_tools and tool_error_code is not None and not is_cancelled:
                        # First emit reasoning lifecycle if needed (still needed for parity)
                        if think_enabled:
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
                                item_status_cancel = (
                                    "incomplete" if is_cancelled else "completed"
                                )
                                yield from emit(
                                    "response.output_item.done",
                                    {
                                        "type": "response.output_item.done",
                                        "output_index": 0,
                                        "item": ReasoningItem(
                                            id=reasoning_item_id,
                                            status=item_status_cancel,
                                            content=(
                                                [ReasoningTextContent(text=thinking)]
                                                if thinking is not None
                                                else []
                                            ),
                                            summary=[],
                                        ).model_dump(),
                                    },
                                )
                            else:
                                item_status_cancel = (
                                    "incomplete" if is_cancelled else "completed"
                                )
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
                                yield from emit(
                                    "response.output_item.done",
                                    {
                                        "type": "response.output_item.done",
                                        "output_index": 0,
                                        "item": ReasoningItem(
                                            id=reasoning_item_id,
                                            status=item_status_cancel,
                                            content=[],
                                            summary=[],
                                        ).model_dump(),
                                    },
                                )
                        final = build_response(
                            body,
                            response_id=response_id,
                            message_id=message_id,
                            created_at=created_at,
                            completed_at=int(time.time()),
                            status=ResponseStatus.failed,
                            text=text,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            thinking=thinking,
                            reasoning_tokens=max(0, len(thinking or "") // 4)
                            if thinking
                            else 0,
                            result=result,
                            error={
                                "code": "server_error",
                                "message": tool_error_msg or "tool call failed",
                            },
                            reasoning_item_id=reasoning_item_id,
                            include_empty_reasoning=think_enabled,
                            item_status="incomplete",
                            tool_calls=None,
                        )
                        yield from emit(
                            "error",
                            {
                                "type": "error",
                                "code": "server_error",
                                "message": tool_error_msg or "tool call failed",
                                "param": None,
                            },
                        )
                        yield from emit(
                            "response.failed",
                            {
                                "type": "response.failed",
                                "response": final,
                                "error": {
                                    "code": "server_error",
                                    "message": tool_error_msg or "tool call failed",
                                },
                            },
                        )
                        return

                    # Prepare final status helpers for tool / non-tool
                    incomplete_details = None
                    tool_incomplete = False
                    if has_tools and tool_calls:
                        if (
                            getattr(body, "parallel_tool_calls", True) is False
                            and len(tool_calls) > 1
                        ):
                            tool_calls = tool_calls[:1]
                            tool_incomplete = True
                            incomplete_details = {"reason": "max_tool_calls"}
                    remaining_text = (
                        _strip_tool_json_from_text(text, tool_calls)
                        if tool_calls
                        else text
                    )
                    # Determine final status: cancelled > incomplete (tool count / token_limit) > completed
                    if is_cancelled:
                        final_status = ResponseStatus.cancelled
                        msg_status = "incomplete"
                    elif tool_incomplete:
                        final_status = ResponseStatus.incomplete
                        msg_status = "incomplete"
                    elif stop_reason_val == 2:  # token_limit
                        final_status = ResponseStatus.incomplete
                        msg_status = "incomplete"
                        incomplete_details = {"reason": "max_output_tokens"}
                    else:
                        final_status = ResponseStatus.completed
                        msg_status = "completed"

                    if think_enabled:
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
                            item_status_cancel = (
                                "incomplete" if is_cancelled else "completed"
                            )
                            yield from emit(
                                "response.output_item.done",
                                {
                                    "type": "response.output_item.done",
                                    "output_index": 0,
                                    "item": ReasoningItem(
                                        id=reasoning_item_id,
                                        status=item_status_cancel,
                                        content=(
                                            [ReasoningTextContent(text=thinking)]
                                            if thinking is not None
                                            else []
                                        ),
                                        summary=[],
                                    ).model_dump(),
                                },
                            )
                        else:
                            # B: empty reasoning when no markers but reasoning requested
                            item_status_cancel = (
                                "incomplete" if is_cancelled else "completed"
                            )
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
                            yield from emit(
                                "response.output_item.done",
                                {
                                    "type": "response.output_item.done",
                                    "output_index": 0,
                                    "item": ReasoningItem(
                                        id=reasoning_item_id,
                                        status=item_status_cancel,
                                        content=[],
                                        summary=[],
                                    ).model_dump(),
                                },
                            )
                    # Emit tool or text events
                    if tool_calls:
                        # For tool calls, emit function_call lifecycle for each
                        base_idx = message_index
                        for idx, tc in enumerate(tool_calls):
                            out_idx = base_idx + idx
                            func_id = tc.get("id") or f"fc_{uuid.uuid4().hex[:24]}"
                            tc["id"] = func_id
                            yield from emit(
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "output_index": out_idx,
                                    "item": FunctionCall(
                                        id=func_id,
                                        call_id=tc.get("call_id")
                                        or f"call_{uuid.uuid4().hex[:24]}",
                                        name=tc["name"],
                                        arguments="",
                                        status="in_progress",  # type: ignore
                                    ).model_dump(),
                                },
                            )
                            args_str = tc.get("arguments") or "{}"
                            # emit arguments delta chunked (simple: whole at once)
                            if args_str:
                                yield from emit(
                                    "response.function_call_arguments.delta",
                                    {
                                        "type": "response.function_call_arguments.delta",
                                        "item_id": func_id,
                                        "output_index": out_idx,
                                        "delta": args_str,
                                    },
                                )
                            yield from emit(
                                "response.function_call_arguments.done",
                                {
                                    "type": "response.function_call_arguments.done",
                                    "item_id": func_id,
                                    "output_index": out_idx,
                                    "arguments": args_str,
                                },
                            )
                            yield from emit(
                                "response.output_item.done",
                                {
                                    "type": "response.output_item.done",
                                    "output_index": out_idx,
                                    "item": FunctionCall(
                                        id=func_id,
                                        call_id=tc.get("call_id") or func_id,
                                        name=tc["name"],
                                        arguments=args_str,
                                        status=msg_status,  # type: ignore
                                    ).model_dump(),
                                },
                            )
                        # If there is residual text after stripping tool json, also emit message
                        if remaining_text and remaining_text.strip():
                            msg_out_idx = base_idx + len(tool_calls)
                            if not message_started:
                                yield from emit(
                                    "response.output_item.added",
                                    {
                                        "type": "response.output_item.added",
                                        "output_index": msg_out_idx,
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
                                    "response.content_part.added",
                                    {
                                        "type": "response.content_part.added",
                                        "item_id": message_id,
                                        "output_index": msg_out_idx,
                                        "content_index": 0,
                                        "part": {
                                            "type": "output_text",
                                            "text": "",
                                            "annotations": [],
                                        },
                                    },
                                )
                                message_content_part_added = True
                            if message_content_part_added:
                                yield from emit(
                                    "response.output_text.delta",
                                    {
                                        "type": "response.output_text.delta",
                                        "item_id": message_id,
                                        "output_index": msg_out_idx,
                                        "content_index": 0,
                                        "delta": remaining_text,
                                        "logprobs": None,
                                    },
                                )
                                yield from emit(
                                    "response.content_part.done",
                                    {
                                        "type": "response.content_part.done",
                                        "item_id": message_id,
                                        "output_index": msg_out_idx,
                                        "content_index": 0,
                                        "part": {
                                            "type": "output_text",
                                            "text": remaining_text,
                                            "annotations": [],
                                        },
                                    },
                                )
                            yield from emit(
                                "response.output_text.done",
                                {
                                    "type": "response.output_text.done",
                                    "item_id": message_id,
                                    "output_index": msg_out_idx,
                                    "content_index": 0,
                                    "text": remaining_text,
                                },
                            )
                            yield from emit(
                                "response.output_item.done",
                                {
                                    "type": "response.output_item.done",
                                    "output_index": msg_out_idx,
                                    "item": ResponseOutputMessage(
                                        id=message_id,
                                        content=[
                                            ResponseOutputText(text=remaining_text)
                                        ],
                                        role="assistant",
                                        status=msg_status,
                                    ).model_dump(),
                                },
                            )
                    else:
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
                            message_started = True
                            yield from emit(
                                "response.content_part.added",
                                {
                                    "type": "response.content_part.added",
                                    "item_id": message_id,
                                    "output_index": message_index,
                                    "content_index": 0,
                                    "part": {
                                        "type": "output_text",
                                        "text": "",
                                        "annotations": [],
                                    },
                                },
                            )
                            message_content_part_added = True
                        if has_tools and remaining_text:
                            yield from emit(
                                "response.output_text.delta",
                                {
                                    "type": "response.output_text.delta",
                                    "item_id": message_id,
                                    "output_index": message_index,
                                    "content_index": 0,
                                    "delta": remaining_text,
                                    "logprobs": None,
                                },
                            )
                        # Content part done before output_text.done (OpenAI order)
                        if message_content_part_added:
                            yield from emit(
                                "response.content_part.done",
                                {
                                    "type": "response.content_part.done",
                                    "item_id": message_id,
                                    "output_index": message_index,
                                    "content_index": 0,
                                    "part": {
                                        "type": "output_text",
                                        "text": remaining_text,
                                        "annotations": [],
                                    },
                                },
                            )
                        yield from emit(
                            "response.output_text.done",
                            {
                                "type": "response.output_text.done",
                                "item_id": message_id,
                                "output_index": message_index,
                                "content_index": 0,
                                "text": remaining_text,
                            },
                        )
                        yield from emit(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": message_index,
                                "item": ResponseOutputMessage(
                                    id=message_id,
                                    content=[ResponseOutputText(text=remaining_text)],
                                    role="assistant",
                                    status=msg_status,
                                ).model_dump(),
                            },
                        )
                    # Reasoning tokens approximated as len(thinking)//4 when thinking present
                    if thinking is not None:
                        reasoning_tokens = max(0, len(thinking) // 4)
                    final = build_response(
                        body,
                        response_id=response_id,
                        message_id=message_id,
                        created_at=created_at,
                        completed_at=int(time.time()),
                        status=final_status,
                        text=remaining_text,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        thinking=thinking,
                        reasoning_tokens=reasoning_tokens,
                        result=result,
                        incomplete_details=incomplete_details,
                        reasoning_item_id=reasoning_item_id,
                        include_empty_reasoning=think_enabled,
                        item_status=msg_status,
                        tool_calls=tool_calls,
                    )
                    if stats:
                        _log_performance(model, final.get("performance"))
                    if is_cancelled:
                        yield from emit(
                            "response.cancelled",
                            {"type": "response.cancelled", "response": final},
                        )
                    elif final_status == ResponseStatus.incomplete:
                        yield from emit(
                            "response.incomplete",
                            {"type": "response.incomplete", "response": final},
                        )
                    elif final_status == ResponseStatus.failed:
                        yield from emit(
                            "error",
                            {
                                "type": "error",
                                "code": "server_error",
                                "message": tool_error_msg or "tool call failed",
                                "param": None,
                            },
                        )
                        yield from emit(
                            "response.failed",
                            {"type": "response.failed", "response": final},
                        )
                    else:
                        yield from emit(
                            "response.completed",
                            {"type": "response.completed", "response": final},
                        )
                    return
                elif kind == "busy":
                    final = build_response(
                        body,
                        response_id=response_id,
                        message_id=message_id,
                        created_at=created_at,
                        completed_at=int(time.time()),
                        status=ResponseStatus.failed,
                        text="".join(complete),
                        input_tokens=0,
                        output_tokens=0,
                        error={"code": "session_busy", "message": str(payload)},
                    )
                    # Emit top-level error event for SDK compatibility (rate_limit_exceeded is OpenAI standard for busy)
                    yield from emit(
                        "error",
                        {
                            "type": "error",
                            "code": "rate_limit_exceeded",
                            "message": str(payload),
                            "param": None,
                        },
                    )
                    yield from emit(
                        "response.failed",
                        {
                            "type": "response.failed",
                            "response": final,
                            "error": {"code": "session_busy", "message": str(payload)},
                        },
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
                        "error",
                        {
                            "type": "error",
                            "code": "server_error",
                            "message": str(payload),
                            "param": None,
                        },
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
        finally:
            # Client disconnect or generator close -> cancel if still running
            if not worker_finished.is_set():
                cancelled.set()
                try:
                    session.cancel()
                except Exception:
                    pass
                # Give native loop a chance to notice cancel_requested / callback -1
                worker_thread.join(timeout=10.0)
            # Now unregister (allow cancel endpoint to find entry while worker was still running)
            with active_streams_lock:
                active_streams.pop(response_id, None)
            if worker_finished.is_set():
                try:
                    _flight_lock.release()
                except RuntimeError:
                    pass
            else:
                log_error(
                    "generation worker did not stop after cancellation (10s); "
                    "keeping the single-flight guard locked to protect the native session"
                )
                # Still pop and release after timeout to avoid permanent 429;
                # native mutex will still be held until worker exits, next request will get native busy.
                # Release python lock so cancel endpoint's wait can proceed and next request gets native 429 instead of deadlock.
                try:
                    _flight_lock.release()
                except RuntimeError:
                    pass

    def _acquire_flight_or_429(timeout: float = 2.0) -> None:
        """Try to acquire the single-flight lock with a short wait to absorb
        concurrent harness requests (e.g., title + main) instead of immediate 429.
        """
        deadline = time.monotonic() + timeout
        while True:
            if _flight_lock.acquire(blocking=False):
                return
            if time.monotonic() >= deadline:
                raise HTTPException(
                    status_code=429,
                    detail="Session busy: another request is in progress. Retry later.",
                )
            time.sleep(0.05)

    @router.post("/responses", response_model=None)
    def create_response(body: CreateResponseRequest, request: Request):
        (
            prompt,
            tok_limit,
            temperature_eff,
            top_p_eff,
            effective_flags,
        ) = _prepare_request(body)
        think_enabled = body.reasoning is not None

        _acquire_flight_or_429(timeout=2.0)

        if body.stream:
            return StreamingResponse(
                stream_events(
                    body,
                    prompt,
                    max_tokens=tok_limit,
                    temperature=temperature_eff,
                    top_p=top_p_eff,
                    effective_flags=effective_flags,
                    request=request,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        chunks: list[str] = []
        result: dict[str, Any] | None = None
        # Non-stream reasoning id mirrors streaming stable id for parity (option B)
        non_stream_reasoning_id = (
            f"rsn_{uuid.uuid4().hex[:24]}" if think_enabled else None
        )
        try:
            try:

                def _collect(_tid, text):
                    if text:
                        chunks.append(text)
                    return 0

                result = session.generate(
                    None,
                    prompt,
                    max_tokens=tok_limit,
                    temperature=temperature_eff,
                    top_p=top_p_eff,
                    on_token=_collect,
                    flags=effective_flags,
                    stop_on_text=stop_markers,
                    stop_at_eos=stop_at_eos,
                )
            except SessionBusyError:
                raise HTTPException(
                    status_code=429,
                    detail="Session busy: another request is in progress. Retry later.",
                )
            except Exception as e:
                try:
                    log_error(
                        f"non-stream native failed: prompt_len={len(prompt)} err={e} preview={(prompt[:600] if len(prompt) > 600 else prompt)!r}"
                    )
                except Exception:
                    pass
                # Map generic failures to OpenAI-style Response with status:failed (200, not 500)
                err_text = _truncate_stop_markers("".join(chunks), all_stop_markers)
                err_thinking: str | None = None
                err_reasoning_tokens = 0
                if think_enabled:
                    err_thinking, err_text = split_thinking(err_text)
                    err_thinking = err_thinking or None
                    if err_thinking is not None:
                        err_reasoning_tokens = max(0, len(err_thinking) // 4)
                err_input = int(result.get("prompt_tokens") or 0) if result else 0
                err_output = (
                    int(result.get("generated_tokens") or len(chunks))
                    if result
                    else len(chunks)
                )
                resp = build_response(
                    body,
                    response_id=f"resp_{uuid.uuid4().hex[:24]}",
                    message_id=f"msg_{uuid.uuid4().hex[:24]}",
                    created_at=int(time.time()),
                    completed_at=int(time.time()),
                    status=ResponseStatus.failed,
                    text=err_text,
                    input_tokens=err_input,
                    output_tokens=err_output,
                    thinking=err_thinking,
                    reasoning_tokens=err_reasoning_tokens,
                    result=result,
                    error={"code": "server_error", "message": str(e)},
                    reasoning_item_id=non_stream_reasoning_id,
                    include_empty_reasoning=think_enabled,  # B: parity with streaming empty reasoning
                )
                if stats:
                    _log_performance(model, resp.get("performance"))
                return resp
        finally:
            _flight_lock.release()
        text = _truncate_stop_markers("".join(chunks), all_stop_markers)
        thinking = None
        reasoning_tokens = 0
        if think_enabled:
            thinking, text = split_thinking(text)
            thinking = thinking or None
            if thinking is not None:
                reasoning_tokens = max(0, len(thinking) // 4)
        # Tool parsing for non-stream
        tool_calls: list[dict[str, Any]] | None = None
        tool_error_code: str | None = None
        tool_error_msg: str | None = None
        if body.tools:
            allowed = {
                t.name
                for t in body.tools
                if getattr(t, "type", None) == "function" and getattr(t, "name", None)  # type: ignore
            }
            tool_calls, tool_error_code, tool_error_msg = _extract_tool_calls_from_text(
                text, allowed
            )
            if tool_error_code is None and not tool_calls:
                tool_calls = None
            elif tool_calls == [] and tool_error_code is None:
                tool_calls = None
        remaining_text = (
            _strip_tool_json_from_text(text, tool_calls) if tool_calls else text
        )
        input_tokens = int(result.get("prompt_tokens") or 0) if result else 0
        output_tokens = (
            int(result.get("generated_tokens") or len(chunks))
            if result
            else len(chunks)
        )
        stop_reason_val = int(result.get("stop_reason") or 0) if result else 0
        incomplete_details = None
        final_status = ResponseStatus.completed
        item_status = "completed"
        error: dict[str, Any] | None = None
        if tool_error_code is not None:
            final_status = ResponseStatus.failed
            item_status = "incomplete"
            error = {
                "code": "server_error",
                "message": tool_error_msg or "tool call failed",
            }
        elif (
            tool_calls
            and getattr(body, "parallel_tool_calls", True) is False
            and len(tool_calls) > 1
        ):
            tool_calls = tool_calls[:1]
            incomplete_details = {"reason": "max_tool_calls"}
            final_status = ResponseStatus.incomplete
            item_status = "incomplete"
        elif stop_reason_val == 3:  # cancelled
            final_status = ResponseStatus.cancelled
            item_status = "incomplete"
        elif stop_reason_val == 2:  # token_limit → incomplete
            final_status = ResponseStatus.incomplete
            incomplete_details = {"reason": "max_output_tokens"}
            item_status = "incomplete"
        resp = build_response(
            body,
            response_id=f"resp_{uuid.uuid4().hex[:24]}",
            message_id=f"msg_{uuid.uuid4().hex[:24]}",
            created_at=int(time.time()),
            completed_at=int(time.time()),
            status=final_status,
            text=remaining_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking=thinking,
            reasoning_tokens=reasoning_tokens,
            result=result,
            incomplete_details=incomplete_details,
            item_status=item_status,
            reasoning_item_id=non_stream_reasoning_id,
            include_empty_reasoning=think_enabled,  # B: empty reasoning when reasoning requested but no markers
            tool_calls=tool_calls,
            error=error,
        )
        if stats:
            _log_performance(model, resp.get("performance"))
        return resp

    @router.get("/responses/{response_id}")
    def get_response(response_id: str):
        with response_store_lock:
            resp = response_store.get(response_id)
        if not resp:
            raise HTTPException(status_code=404, detail="Response not found")
        return resp

    @router.post("/responses/{response_id}/cancel")
    def cancel_response(response_id: str):
        with response_store_lock:
            response = response_store.get(response_id)
            if response is None:
                raise HTTPException(status_code=404, detail="Response not found")
            if response["status"] != ResponseStatus.in_progress:
                raise HTTPException(
                    status_code=409, detail="Response is not in progress"
                )
        # Per-response cancellation: only cancel if id is active
        entry = None
        with active_streams_lock:
            entry = active_streams.get(response_id)
        if entry is not None:
            entry["cancelled"].set()
            try:
                session.cancel()
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Native session cancellation failed: {exc}",
                ) from exc
            finished: threading.Event = entry["finished"]
            if not finished.wait(timeout=max(0.0, cancel_wait_seconds)):
                raise HTTPException(
                    status_code=504,
                    detail="Generation did not stop before the cancellation deadline",
                )
        else:
            # Stale: store says in_progress but no active stream entry.
            # Do not call process-wide session.cancel() - it would cancel the
            # wrong (currently active) stream. Aligns with mock/live/OpenAI 409.
            raise HTTPException(status_code=409, detail="Response is not in progress")
        with response_store_lock:
            # Re-check status under lock to avoid flipping a just-completed response
            cur = response_store.get(response_id)
            if cur is not None and cur["status"] == ResponseStatus.in_progress:
                cur["status"] = ResponseStatus.cancelled
                return cur
            # If status changed concurrently to completed/failed/cancelled, honour that
            if cur is not None:
                return cur
            return response

    model_created_at = int(time.time())

    def _model_obj(mid: str) -> dict[str, Any]:
        return {
            "id": mid,
            "object": "model",
            "created": model_created_at,
            "owned_by": "cke",
        }

    @router.get("/models")
    def list_models():
        return {"object": "list", "data": [_model_obj(model)]}

    @router.get("/models/{model_id}")
    def retrieve_model(model_id: str):
        if model_id != model:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id!r} not found; available model: {model!r}",
            )
        return _model_obj(model_id)

    @router.get("/health")
    def health():
        return {"status": "ok", "mode": "live", "inference": True}

    app = FastAPI(title="CKE v3 live inference server", version="0.3.0")
    app.include_router(router, prefix="/v1")
    app.include_router(conversations_router, prefix="/v1")

    if viz_html is not None:

        @app.get("/viz", response_class=HTMLResponse)
        def viz_page():
            return HTMLResponse(viz_html)

    # Expose internal state for testing / debugging
    app.state.response_store = response_store
    app.state.response_history_store = response_history_store
    app.state.response_store_lock = response_store_lock
    app.state.active_streams = active_streams
    app.state.active_streams_lock = active_streams_lock
    app.state.flight_lock = _flight_lock
    app.state.session = session

    return app


# -----------------------------------------------------------------------------
# CLI — runtime helpers re-exported from ck_serve_runtime_v8 (model-ready side)
# -----------------------------------------------------------------------------

# Server owns the HTTP/session lifecycle; artifact preparation (download →
# convert → build IR → codegen → compile libmodel.so) lives in
# ck_serve_runtime_v8. Re-export here for backward compat so
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

    session = SessionV8.open(
        run_dir,
        context_length=args.context_len,
    )

    chat_template, chat_templates, chat_contract = _load_runtime_templates(run_dir)
    if chat_contract is not None:
        contract_name = str(chat_contract.get("name") or "runtime")
        log(
            f"Loaded chat contract {contract_name!r} from weights_manifest.json; "
            "Python-side prompt formatting enabled"
        )
        if chat_template:
            log(
                f"Loaded chat_template ({len(chat_template)} chars) from manifest",
                C_GRAY,
            )
        if chat_templates:
            log(
                f"Loaded chat_templates variants {list(chat_templates.keys())} from manifest",
                C_GRAY,
            )
    else:
        log("Runtime chat contract not found; using C-side format_chat", C_ORANGE)
        # still try to load templates even without contract
        if chat_template or chat_templates:
            log(
                f"Loaded templates without contract: chat_template={bool(chat_template)} chat_templates={list(chat_templates.keys()) if chat_templates else None}",
                C_GRAY,
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
