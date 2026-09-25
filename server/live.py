"""Live generic OpenAI Responses + tools factory for ``server/``.

Ports the Responses lifecycle from ``version/v8/scripts/ck_serve_v8.py`` into
the server boundary so a generic harness can use ``POST /v1/responses`` with
``function``/``mcp`` tools without importing the scripts tree:

* single-flight native session (HTTP 429 when busy)
* ``previous_response_id`` + ``function_call_output`` multi-turn history
* tool parsing driven by the loaded ``chat_template.jinja`` (Qwen3-native
  ``<tool_call>`` JSON blocks + bare JSON); ``malformed``/``unknown``
  → ``failed``, ``parallel_tool_calls:false`` + >1 → ``incomplete``
* non-stream + SSE streaming with OpenAI event order and ``sequence_number``
* real ``usage`` + ``performance``, capacity preflight, cancel (404/409/500/504)

The server never executes tools server-side. Only function-like tools
(``function``/``mcp``) are parsed back into ``function_call`` items for the
client to execute. ``file_search`` / ``computer`` / ``web_search`` /
``code_interpreter`` / ``image_generation`` are accepted + prompt-visible but
never emitted as output items (plain text fallback).
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

try:
    import jinja2
    import jinja2.sandbox

    _JINJA_AVAILABLE = True
except ImportError:
    jinja2 = None
    _JINJA_AVAILABLE = False

from .routes.conversations import router as conversations_router
from .runtime import load_manifest_templates
from .schemas.common import ResponseStatus
from .schemas.content import ResponseOutputText
from .schemas.output_items import (
    FunctionCall,
    ReasoningItem,
    ReasoningTextContent,
    ResponseOutputMessage,
)
from .schemas.response import CreateResponseRequest
from .session_v8 import (
    CK_SESSION_REQUEST_RAW_PROMPT,
    SessionBusyError,
    stop_reason_name,
    truncate_stop_markers,
)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

C_GRAY = "\033[38;5;242m"
C_ORANGE = "\033[38;5;208m"
C_RESET = "\033[0m"

# How long a second request waits for the single-flight lock before the
# server answers 429 (harness retry storm mitigation).
_FLIGHT_WAIT_SECONDS = 30.0

_IGNORED_TOOL_TYPES = frozenset(
    {
        "file_search",
        "computer",
        "computer_use_preview",
        "web_search",
        "code_interpreter",
        "image_generation",
    }
)
_FUNCTION_LIKE_TYPES = frozenset({"function", "mcp"})

# Legacy name (mirrors the ck_serve_v8.py helper); canonical is
# server.session_v8.truncate_stop_markers.
_truncate_stop_markers = truncate_stop_markers


def _harness_error(
    status_code: int,
    message: str,
    *,
    err_type: str,
    code: str,
    retry_after: float | None = None,
) -> HTTPException:
    """HTTP error carrying an OpenAI-shaped ``error`` object.

    The app-level exception handler renders this as
    ``{"error": {...}, "detail": message}`` so harnesses can parse ``error``
    while existing ``detail`` readers keep working.
    """
    headers = (
        {"Retry-After": str(int(retry_after))} if retry_after is not None else None
    )
    return HTTPException(
        status_code=status_code,
        detail={"error": {"message": message, "type": err_type, "code": code}},
        headers=headers,
    )


def _log_rejection(server_model: str, body: Any, exc: HTTPException) -> None:
    """One warn line per rejected request (rejections otherwise stay silent)."""
    try:
        tools_n: Any = len(getattr(body, "tools", None) or [])
    except Exception:
        tools_n = "?"
    if isinstance(exc.detail, dict) and isinstance(exc.detail.get("error"), dict):
        err = exc.detail["error"]
        code: Any = err.get("code", "?")
        message: Any = err.get("message", "")
    else:
        code, message = "?", exc.detail
    print(
        f"{C_ORANGE}{server_model} - rejected {exc.status_code} "
        f"({code}): {message} "
        f"[model={getattr(body, 'model', '?')} "
        f"max_output_tokens={getattr(body, 'max_output_tokens', '?')} "
        f"tools={tools_n} "
        f"previous_response_id={getattr(body, 'previous_response_id', None)}]{C_RESET}",
        flush=True,
    )


# --- prompt helpers -----------------------------------------------------------


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
                    continue
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
            message: dict[str, Any] = {
                "role": str(role),
                "content": _content_text(item.content),
            }
            if str(role) == "assistant":
                # Native templates (e.g. Qwen3 line 48
                # `{%- if message.tool_calls %}`) read this key unguarded;
                # under StrictUndefined a missing key aborts the whole
                # render, surfacing as 422 downstream.
                message["tool_calls"] = []
            messages.append(message)
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
                            "function": {"name": item.name, "arguments": arguments},
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
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


def _sse(event_type: str, data: Any) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


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
        "stop_reason": stop_reason_name(result.get("stop_reason")),
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


def _marker_index(lowered: str, marker: str) -> int:
    anchored = lowered.find("\n" + marker)
    if anchored != -1:
        return anchored + 1
    if lowered.startswith(marker):
        return 0
    return lowered.find(marker)


def split_thinking(text: str) -> tuple[str, str]:
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
        if self._mode == "answer" or not self._look:
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


# --- chat contract ------------------------------------------------------------


def _resolve_contract_thinking_overrides(
    contract: dict[str, Any], thinking_mode: str | None
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
        str(m).lower()
        for m in list(contract.get("last_user_prefix_suppression_markers") or [])
        if str(m or "").strip()
    ]
    assistant_generation_prefix, last_user_prefix = (
        _resolve_contract_thinking_overrides(contract, thinking_mode)
    )
    user_text = str(prompt or "")
    if last_user_prefix:
        lowered = user_text.lower()
        if last_user_prefix.lower() not in lowered and not any(
            m in lowered for m in suppression_markers
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
        return f"{turn_prefix.replace('{role}', label)}{content}{turn_suffix}"

    formatted = ""
    if bos_prefix:
        formatted += bos_prefix
    if system_text and system_prompt_mode == "dedicated_turn":
        formatted += _render_turn("system", system_text)
    formatted += _render_turn("user", user_text)
    formatted += assistant_generation_prefix
    return formatted if formatted else user_text


def _load_runtime_chat_contract(run_dir: str | Path) -> dict[str, Any] | None:
    """No chat contract is loaded from disk (pure-Jinja runtime)."""
    return None


def _load_runtime_templates(
    run_dir: str | Path,
) -> tuple[str | None, dict[str, str] | None, dict[str, Any] | None]:
    """Return (chat_template, chat_templates, None) from the sidecar files."""
    return load_manifest_templates(Path(run_dir))


def _load_builtin_chat_contract(
    template_name: str | None,
    *,
    _seen: set[str] | None = None,
) -> dict[str, Any] | None:
    """Backward-compat stub: the manifest is the single source of truth."""
    return None


# --- tools --------------------------------------------------------------------


def _effective_tools(body: Any) -> list[Any]:
    tools = getattr(body, "tools", None) if body is not None else None
    if not tools:
        return []
    return [t for t in tools if getattr(t, "type", None) in _FUNCTION_LIKE_TYPES]


def _has_function_tools(body: Any) -> bool:
    return bool(_effective_tools(body))


def _resolve_thinking_mode(body: Any) -> str:
    """Single thinking on/off authority: ``"visible"`` or ``"suppressed"``.

    ``"none"`` means reasoning off even when a reasoning object is present;
    ``"default"`` (or any graded effort, or an absent effort) means on. The
    engine has no effort gradations — on/off is the only distinction. Both
    prompt formatting and response handling key off this value.
    """
    reasoning = getattr(body, "reasoning", None) if body is not None else None
    if reasoning is None:
        return "suppressed"
    if getattr(reasoning, "effort", None) == "none":
        return "suppressed"
    return "visible"


def _has_tool_support(
    chat_template: str | None, chat_templates: dict[str, str] | None
) -> bool:
    if isinstance(chat_templates, dict) and chat_templates:
        for key in ("tool_use", "tools", "default"):
            if isinstance(chat_templates.get(key), str) and chat_templates[key].strip():
                return True
        if any(isinstance(v, str) and v.strip() for v in chat_templates.values()):
            return True
    if isinstance(chat_template, str) and chat_template.strip():
        lower = chat_template.lower()
        if "tools" in lower:
            return True
    return False


#: Tool-call syntax emitted by the model-native Jinja template: Qwen3-style
#: ``<tool_call>{"name": ..., "arguments": ...}</tool_call>`` blocks.
_TOOL_SYNTAX_TOOL_CALL_JSON = "tool_call_json"
#: Plain JSON tool calls with no template-declared tag wrapper.
_TOOL_SYNTAX_JSON = "json"


def _detect_template_tool_syntax(
    chat_template: str | None, chat_templates: dict[str, str] | None
) -> str:
    """Detect the tool-call syntax declared by the loaded chat template.

    The template is the single source of truth for the tool-call wire
    format. Returns ``"tool_call_json"`` when any loaded template renders
    ``<tool_call>`` blocks (Qwen3 native: JSON inside the tags), else
    ``"json"`` (bare JSON tool calls only).
    """
    texts: list[str] = []
    if isinstance(chat_template, str) and chat_template.strip():
        texts.append(chat_template)
    if isinstance(chat_templates, dict):
        for value in chat_templates.values():
            if isinstance(value, str) and value.strip():
                texts.append(value)
    for text in texts:
        if "<tool_call" in text.lower():
            return _TOOL_SYNTAX_TOOL_CALL_JSON
    return _TOOL_SYNTAX_JSON


def _render_with_chat_templates(
    chat_template: str | None,
    chat_templates: dict[str, str] | None,
    messages: list[dict[str, Any]],
    body: Any,
    chat_contract: dict[str, Any] | None = None,
    effective_thinking: str = "suppressed",
) -> str | None:
    if not _JINJA_AVAILABLE:
        return None
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
            tools = [t.model_dump() for t in body.tools]
        env = jinja2.sandbox.SandboxedEnvironment(
            undefined=jinja2.StrictUndefined, autoescape=False
        )
        tmpl = env.from_string(tmpl_str)
        return str(
            tmpl.render(
                messages=messages,
                tools=tools,
                tool_choice=getattr(body, "tool_choice", None)
                if body is not None
                else None,
                enable_thinking=(effective_thinking == "visible"),
                add_generation_prompt=True,
            )
        )
    except Exception as exc:
        # Never fail silently: a swallowed render error surfaces later as
        # 422 ("cannot render role-aware tool history") with no clue.
        # Keep returning None (caller decides fallback vs 422) but log the
        # root cause with a traceback.
        import traceback as _traceback

        print(
            f"{C_ORANGE}chat template render failed "
            f"({type(exc).__name__}: {exc}); "
            f"roles={[m.get('role') for m in messages]}{C_RESET}",
            flush=True,
        )
        print(_traceback.format_exc(), flush=True)
        return None


def _extract_tool_calls_from_text(
    text: str,
    allowed_names: set[str] | None,
    *,
    tool_syntax: str = _TOOL_SYNTAX_TOOL_CALL_JSON,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Parse model-native tool calls from generated text.

    The wire format follows the loaded ``chat_template.jinja`` (see
    :func:`_detect_template_tool_syntax`): ``<tool_call>{"name": ...,
    "arguments": ...}</tool_call>`` blocks when the template declares
    ``<tool_call>`` tags, otherwise bare JSON. Hardcoded Qwen
    ``<function=>`` / ``<parameter=>`` tags are not recognized.
    """
    if not text or not text.strip():
        return [], None, None
    stripped = text.strip()
    if tool_syntax == _TOOL_SYNTAX_TOOL_CALL_JSON:
        tool_blocks: list[str] = []
        for m in re.finditer(
            r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL | re.IGNORECASE
        ):
            tool_blocks.append(m.group(1).strip())
    else:
        tool_blocks = []
    candidates: list[str | dict[str, Any]] = []
    if tool_blocks:
        # Native template syntax: each block must hold a JSON tool-call
        # object (or list of them). Anything else is malformed output.
        candidates = list(tool_blocks)
    else:
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            candidates = [stripped]
        else:
            stack: list[str] = []
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
            if stack and start != -1:
                snippet = text[start:]
                if '"name"' in snippet or '"arguments"' in snippet:
                    try:
                        json.loads(snippet)
                    except json.JSONDecodeError as exc:
                        return [], "malformed", f"malformed tool call: {exc}"
                    return [], "malformed", "malformed tool call: incomplete JSON"
    tool_calls: list[dict[str, Any]] = []
    for snippet in candidates:
        if isinstance(snippet, dict):
            obj = snippet
        else:
            try:
                obj = json.loads(snippet)
            except json.JSONDecodeError as exc:
                return [], "malformed", f"malformed tool call: {exc}"
        items = obj if isinstance(obj, list) else [obj]
        for item in items:
            if not isinstance(item, dict):
                return [], "malformed", "malformed tool call: expected object"
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
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
            if isinstance(args, dict):
                args_str = json.dumps(args, separators=(",", ":"))
            elif isinstance(args, str):
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
    text: str,
    tool_calls: list[dict[str, Any]] | None,
    *,
    tool_syntax: str = _TOOL_SYNTAX_TOOL_CALL_JSON,
) -> str:
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
    if tool_syntax == _TOOL_SYNTAX_TOOL_CALL_JSON:
        remaining = re.sub(
            r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
    for tc in tool_calls:
        m = re.search(r"\{[^}]*" + re.escape(tc["name"]) + r"[^}]*\}", remaining)
        if m:
            remaining = remaining[: m.start()] + remaining[m.end() :]
    return remaining.strip()


def _classify_stream_mode(
    buffer: str, *, tool_syntax: str = _TOOL_SYNTAX_TOOL_CALL_JSON
) -> str | None:
    """Classify buffered raw output as ``"tool"``, ``"text"``, or ``None``.

    Returns ``None`` when more input is needed (whitespace only so far).
    Tool-shaped prefixes (``{``, ``[``, and ``<tool_call`` when the loaded
    template declares ``<tool_call>`` tags) commit to streaming raw chunks
    as ``function_call_arguments.delta``; anything else streams as
    ``output_text.delta``. Terminals are still decided by the full-text parse
    at ``done`` — this only controls what streams live, so a wrong commit
    can never change the final outcome.
    """
    stripped = buffer.lstrip()
    if not stripped:
        return None
    if stripped.startswith("{") or stripped.startswith("["):
        return "tool"
    if stripped.startswith("<"):
        if stripped.lower().startswith("<tool_call"):
            return "tool" if tool_syntax == _TOOL_SYNTAX_TOOL_CALL_JSON else "text"
        return "text"
    return "text"


# --- app factory --------------------------------------------------------------


def create_app(
    session,
    *,
    model: str = "ck-v8",
    context_length: int | None = None,
    stats: bool = True,
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
    viz_html: str | None = None,
    extra_route_registrar: Callable[[APIRouter, Callable[..., Any]], None] | None = None,
):
    """Build the live Responses FastAPI app around a session (real or fake).

    ``viz_html`` serves a ``GET /viz`` page when provided (scripts-side file
    loading stays in the caller). ``extra_route_registrar`` is called with
    ``(router, create_response)`` before the app is built so hosts can attach
    compatibility routes (e.g. Chat Completions) against the canonical handler.
    """

    router = APIRouter()
    response_store: OrderedDict[str, dict[str, Any]] = OrderedDict()
    response_history_store: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    response_store_lock = threading.Lock()
    response_store_limit = 256
    _flight_lock = threading.Lock()
    active_streams: dict[str, dict[str, Any]] = {}
    active_streams_lock = threading.Lock()

    stop_markers = [str(m) for m in (stop_on_text or ()) if str(m)]
    all_stop_markers = list(stop_markers)
    if stop_at_eos:
        all_stop_markers.append("<eos>")

    # Tool-call wire format is declared by the loaded chat template, not by
    # hardcoded tags (see _detect_template_tool_syntax).
    tool_syntax = _detect_template_tool_syntax(chat_template, chat_templates)

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
        if (
            body.tools
            and _effective_tools(body)
            and not _has_tool_support(chat_template, chat_templates)
        ):
            raise HTTPException(
                status_code=501, detail="Model doesn't support tool calling"
            )

    def _store_response(
        response_id: str, response: dict[str, Any], history: list[dict[str, Any]]
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
            messages.extend(dict(m) for m in previous)
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
            if message.get("tool_call_id") not in known_call_ids:
                raise _harness_error(
                    400,
                    f"Function output references unknown call_id {message.get('tool_call_id')!r}",
                    err_type="invalid_request_error",
                    code="unknown_call_id",
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
        effective_thinking = _resolve_thinking_mode(body)
        jinja_rendered: str | None = None
        if chat_template is not None or chat_templates is not None:
            jinja_rendered = _render_with_chat_templates(
                chat_template,
                chat_templates,
                messages,
                body,
                chat_contract,
                effective_thinking,
            )
        requires_role_rendering = any(
            m.get("role") != "user" or m.get("tool_calls") for m in messages
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
        if not prompt or not prompt.strip():
            prompt = _extract_prompt(body) or "Hello"
            if isinstance(body.instructions, str) and body.instructions.strip():
                prompt = f"{body.instructions}\n{prompt}".strip()
            if not prompt.strip():
                prompt = "Hello"
        if context_length is not None:
            if tok_limit >= context_length:
                raise _harness_error(
                    400,
                    f"max_output_tokens {tok_limit} leaves no room in the loaded "
                    f"context capacity {context_length}; request a smaller output "
                    "budget or load a larger generated runtime",
                    err_type="invalid_request_error",
                    code="context_length_exceeded",
                )
            count_tokens = getattr(session, "count_tokens", None)
            if callable(count_tokens):
                prompt_tokens = count_tokens(prompt)
                if prompt_tokens + tok_limit > context_length:
                    available = max(0, context_length - prompt_tokens)
                    raise _harness_error(
                        400,
                        f"rendered prompt has {prompt_tokens} tokens and the request "
                        f"reserves {tok_limit} output tokens, exceeding loaded context "
                        f"capacity {context_length}; at most {available} output tokens remain",
                        err_type="invalid_request_error",
                        code="context_length_exceeded",
                    )
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
                if text and text.strip():
                    output.append(
                        ResponseOutputMessage(
                            id=message_id,
                            content=[ResponseOutputText(text=text)],
                            role="assistant",
                            status=item_status_value,
                        ).model_dump()
                    )
            else:
                output.append(
                    ResponseOutputMessage(
                        id=message_id,
                        content=[ResponseOutputText(text=text)],
                        role="assistant",
                        status=item_status_value,
                    ).model_dump()
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
            "parallel_tool_calls": body.parallel_tool_calls
            if body.parallel_tool_calls is not None
            else True,
            "temperature": body.temperature
            if body.temperature is not None
            else temperature,
            "top_p": body.top_p if body.top_p is not None else top_p,
            "top_logprobs": body.top_logprobs,
            "tool_choice": body.tool_choice,
            "tools": [t.model_dump() for t in body.tools] if body.tools else [],
            "truncation": body.truncation,
            "text": body.text.model_dump() if body.text is not None else None,
            "user": body.user,
            "background": body.background,
            "conversation": _conversation_echo(body),
            "max_output_tokens": body.max_output_tokens
            if body.max_output_tokens is not None
            else max_tokens,
            "max_tool_calls": body.max_tool_calls,
            "moderation": body.moderation.model_dump()
            if body.moderation is not None
            else None,
            "previous_response_id": body.previous_response_id,
            "prompt": body.prompt.model_dump() if body.prompt is not None else None,
            "prompt_cache_key": body.prompt_cache_key,
            "prompt_cache_options": (
                body.prompt_cache_options.model_dump()
                if body.prompt_cache_options is not None
                else None
            ),
            "prompt_cache_retention": body.prompt_cache_retention,
            "reasoning": body.reasoning.model_dump()
            if body.reasoning is not None
            else None,
            "safety_identifier": body.safety_identifier,
            "service_tier": body.service_tier,
            "usage": _usage(input_tokens, output_tokens, reasoning_tokens),
        }
        if result is not None:
            resp["performance"] = _performance_profile(result)
        should_store = getattr(body, "store", None) is not False
        if should_store:
            history = _request_messages(body)
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": text,
                # Always present: the next turn renders this history entry
                # through the native template, which reads
                # `message.tool_calls` unguarded (StrictUndefined raises on
                # a missing key).
                "tool_calls": [],
            }
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

    def _parse_tool_result(
        text: str, body
    ) -> tuple[list[dict[str, Any]] | None, str | None, str | None]:
        if not _has_function_tools(body):
            return None, None, None
        allowed = {t.name for t in _effective_tools(body) if getattr(t, "name", None)}
        tool_calls, code, msg = _extract_tool_calls_from_text(
            text, allowed, tool_syntax=tool_syntax
        )
        if code is None and not tool_calls:
            return None, None, None
        if tool_calls == [] and code is None:
            return None, None, None
        return tool_calls, code, msg

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
        think_enabled = _resolve_thinking_mode(body) == "visible"
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
        # Live-stream classifier state (tools attached, no reasoning): raw
        # chunks accumulate in classify_parts until _classify_stream_mode can
        # commit to "tool" (raw → function_call_arguments.delta) or "text"
        # (raw → output_text.delta). Before commit nothing streams.
        classify_parts: list[str] = []
        stream_mode: str | None = None
        tool_provisional: dict[str, Any] | None = None

        def on_token(_tid, text):
            nonlocal stream_mode, tool_provisional
            if text:
                complete.append(text)
                if not _has_function_tools(body):
                    if splitter is not None:
                        for state, delta in splitter.feed(text):
                            events.put(
                                (
                                    "reasoning_text" if state == "thinking" else "text",
                                    delta,
                                )
                            )
                    else:
                        events.put(("text", text))
                elif splitter is not None:
                    # Reasoning requested with tools: keep today's buffered
                    # behavior (thinking/tool interplay streams at done).
                    pass
                else:
                    classify_parts.append(text)
                    if stream_mode is None:
                        stream_mode = _classify_stream_mode(
                            "".join(classify_parts), tool_syntax=tool_syntax
                        )
                        if stream_mode == "tool":
                            tool_provisional = {
                                "id": f"fc_{uuid.uuid4().hex[:24]}",
                                "call_id": f"call_{uuid.uuid4().hex[:24]}",
                            }
                            events.put(("tool_added", dict(tool_provisional)))
                            events.put(("tool_delta", "".join(classify_parts)))
                            classify_parts.clear()
                        elif stream_mode == "text":
                            events.put(("text", "".join(classify_parts)))
                            classify_parts.clear()
                    elif stream_mode == "tool":
                        events.put(("tool_delta", text))
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
                if splitter is not None and not _has_function_tools(body):
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

        def emit_reasoning_lifecycle(thinking, is_cancelled):
            nonlocal reasoning_started
            status_value = "incomplete" if is_cancelled else "completed"
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
                            status=status_value,
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
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": ReasoningItem(
                            id=reasoning_item_id,
                            status=status_value,
                            content=[],
                            summary=[],
                        ).model_dump(),
                    },
                )

        try:
            while True:
                try:
                    kind, payload = events.get(timeout=0.2)
                except queue.Empty:
                    if cancelled.is_set() and worker_finished.is_set():
                        continue
                    if worker_finished.is_set():
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
                elif kind == "tool_added":
                    # Provisional function_call item: name is unknown until the
                    # done-parse; ids are reconciled with the parsed call then.
                    # think_enabled is False whenever this branch runs.
                    yield from emit(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": FunctionCall(
                                id=payload["id"],
                                call_id=payload["call_id"],
                                name="",
                                arguments="",
                                status="in_progress",  # type: ignore
                            ).model_dump(),
                        },
                    )
                elif kind == "tool_delta":
                    yield from emit(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": (
                                tool_provisional["id"] if tool_provisional else ""
                            ),
                            "output_index": 0,
                            "delta": payload,
                        },
                    )
                elif kind == "done":
                    result = payload or {}
                    stop_reason_val = int(result.get("stop_reason") or 0)
                    is_cancelled = cancelled.is_set() or stop_reason_val == 3
                    text = truncate_stop_markers("".join(complete), all_stop_markers)
                    thinking = None
                    if think_enabled:
                        thinking, text = split_thinking(text)
                        thinking = thinking or None
                    input_tokens = int(result.get("prompt_tokens") or 0)
                    output_tokens = int(result.get("generated_tokens") or len(complete))
                    message_index = 1 if think_enabled else 0
                    tool_calls, tool_error_code, tool_error_msg = _parse_tool_result(
                        text, body
                    )
                    if tool_error_code is not None and not is_cancelled:
                        if think_enabled:
                            yield from emit_reasoning_lifecycle(thinking, is_cancelled)
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
                    incomplete_details = None
                    tool_incomplete = False
                    if (
                        tool_calls
                        and getattr(body, "parallel_tool_calls", True) is False
                        and len(tool_calls) > 1
                    ):
                        tool_calls = tool_calls[:1]
                        tool_incomplete = True
                        incomplete_details = {"reason": "max_tool_calls"}
                    remaining_text = (
                        _strip_tool_json_from_text(text, tool_calls, tool_syntax=tool_syntax)
                        if tool_calls
                        else text
                    )
                    if is_cancelled:
                        final_status = ResponseStatus.cancelled
                        msg_status = "incomplete"
                    elif tool_incomplete:
                        final_status = ResponseStatus.incomplete
                        msg_status = "incomplete"
                    elif stop_reason_val == 2:
                        final_status = ResponseStatus.incomplete
                        msg_status = "incomplete"
                        incomplete_details = {"reason": "max_output_tokens"}
                    else:
                        final_status = ResponseStatus.completed
                        msg_status = "completed"
                    if think_enabled:
                        yield from emit_reasoning_lifecycle(thinking, is_cancelled)
                    if tool_calls:
                        # If tool-mode deltas already streamed, the first call
                        # reuses the provisional ids so added→delta→done link
                        # up; its added/delta events are not re-emitted.
                        streamed_tool = (
                            tool_provisional is not None and stream_mode == "tool"
                        )
                        if streamed_tool and tool_provisional is not None:
                            tool_calls[0]["id"] = tool_provisional["id"]
                            tool_calls[0]["call_id"] = tool_provisional["call_id"]
                        base_idx = message_index
                        for idx, tc in enumerate(tool_calls):
                            out_idx = base_idx + idx
                            func_id = tc.get("id") or f"fc_{uuid.uuid4().hex[:24]}"
                            tc["id"] = func_id
                            if idx == 0 and streamed_tool:
                                args_str = tc.get("arguments") or "{}"
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
                                continue
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
                        if remaining_text and remaining_text.strip():
                            msg_out_idx = base_idx + len(tool_calls)
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
                        # Catch-up delta for flows that buffered (tool-mode false
                        # positive, or no live streaming). Text-mode already
                        # streamed these tokens; re-sending would duplicate.
                        if (
                            _has_function_tools(body)
                            and remaining_text
                            and stream_mode != "text"
                        ):
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
                    reasoning_tokens = (
                        max(0, len(thinking) // 4) if thinking is not None else 0
                    )
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
            if not worker_finished.is_set():
                cancelled.set()
                try:
                    session.cancel()
                except Exception:
                    pass
                worker_thread.join(timeout=10.0)
            with active_streams_lock:
                active_streams.pop(response_id, None)
            try:
                _flight_lock.release()
            except RuntimeError:
                pass

    def _acquire_flight_or_429(timeout: float | None = None) -> None:
        """Take the single-flight lock, waiting briefly for harness bursts.

        Concurrent harness requests (e.g. title + main, client retries) wait
        up to ``_FLIGHT_WAIT_SECONDS`` instead of failing instantly; only a
        genuinely stuck generation still answers 429 with ``Retry-After``.
        """
        if timeout is None:
            timeout = _FLIGHT_WAIT_SECONDS
        deadline = time.monotonic() + timeout
        while True:
            if _flight_lock.acquire(blocking=False):
                return
            if time.monotonic() >= deadline:
                raise _harness_error(
                    429,
                    "Session busy: another request is in progress. Retry later.",
                    err_type="rate_limit_error",
                    code="rate_limit_exceeded",
                    retry_after=timeout,
                )
            time.sleep(0.05)

    @router.post("/responses", response_model=None)
    def create_response(body: CreateResponseRequest, request: Request):
        acquired = False
        try:
            _validate_request(body)
            _acquire_flight_or_429()
            acquired = True
            prompt, tok_limit, temperature_eff, top_p_eff, effective_flags = (
                _prepare_request(body)
            )
        except HTTPException as exc:
            _log_rejection(model, body, exc)
            if acquired:
                _flight_lock.release()
            raise
        except Exception:
            if acquired:
                _flight_lock.release()
            raise
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
        think_enabled = _resolve_thinking_mode(body) == "visible"
        chunks: list[str] = []
        result: dict[str, Any] | None = None
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
                busy_exc = _harness_error(
                    429,
                    "Session busy: another request is in progress. Retry later.",
                    err_type="rate_limit_error",
                    code="rate_limit_exceeded",
                    retry_after=_FLIGHT_WAIT_SECONDS,
                )
                _log_rejection(model, body, busy_exc)
                raise busy_exc
            except Exception as e:
                err_text = truncate_stop_markers("".join(chunks), all_stop_markers)
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
                err_resp = build_response(
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
                    include_empty_reasoning=think_enabled,
                )
                if stats:
                    _log_performance(model, err_resp.get("performance"))
                return err_resp
        finally:
            _flight_lock.release()
        text = truncate_stop_markers("".join(chunks), all_stop_markers)
        thinking = None
        reasoning_tokens = 0
        if think_enabled:
            thinking, text = split_thinking(text)
            thinking = thinking or None
            if thinking is not None:
                reasoning_tokens = max(0, len(thinking) // 4)
        tool_calls, tool_error_code, tool_error_msg = _parse_tool_result(text, body)
        remaining_text = (
            _strip_tool_json_from_text(text, tool_calls, tool_syntax=tool_syntax)
            if tool_calls
            else text
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
        elif stop_reason_val == 3:
            final_status = ResponseStatus.cancelled
            item_status = "incomplete"
        elif stop_reason_val == 2:
            final_status = ResponseStatus.incomplete
            incomplete_details = {"reason": "max_output_tokens"}
            item_status = "incomplete"
        final_resp = build_response(
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
            include_empty_reasoning=think_enabled,
            tool_calls=tool_calls,
            error=error,
        )
        if stats:
            _log_performance(model, final_resp.get("performance"))
        return final_resp

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
        with active_streams_lock:
            entry = active_streams.get(response_id)
        if entry is not None:
            entry["cancelled"].set()
            try:
                session.cancel()
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Native session cancellation failed: {exc}"
                ) from exc
            finished: threading.Event = entry["finished"]
            if not finished.wait(timeout=max(0.0, cancel_wait_seconds)):
                raise HTTPException(
                    status_code=504,
                    detail="Generation did not stop before the cancellation deadline",
                )
        else:
            raise HTTPException(status_code=409, detail="Response is not in progress")
        with response_store_lock:
            cur = response_store.get(response_id)
            if cur is not None and cur["status"] == ResponseStatus.in_progress:
                cur["status"] = ResponseStatus.cancelled
                return cur
            if cur is not None:
                return cur
            return response

    model_created_at = int(time.time())

    def _model_obj(mid: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": mid,
            "object": "model",
            "created": model_created_at,
            "owned_by": "cke",
            # Reasoning capability block (OpenRouter-shaped): harnesses gate
            # their Effort picker on its presence ("none" = off,
            # "default" = on). No custom capability fields — strict
            # discovery parsers must see only standard shapes here.
            "reasoning": {
                "supported_efforts": ["none", "default"],
                "default_effort": "default",
                "default_enabled": False,
            },
            "supported_parameters": ["reasoning", "reasoning_effort"],
        }
        if context_length is not None:
            result["cke_context_length"] = context_length
            result["cke_default_max_output_tokens"] = max_tokens
        return result

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

    if extra_route_registrar is not None:
        extra_route_registrar(router, create_response)

    app = FastAPI(title="CKE v8 live Responses server", version="0.2.0")
    app.include_router(router, prefix="/v1")
    app.include_router(conversations_router, prefix="/v1")

    @app.exception_handler(HTTPException)
    async def _harness_error_handler(request: Request, exc: HTTPException):
        """Render HTTP errors as ``{"error": {...}, "detail": message}``.

        Harness SDKs parse ``error`` (OpenAI shape); existing ``detail``
        readers keep working unchanged.
        """
        detail = exc.detail
        if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
            error = dict(detail["error"])
            message = str(error.get("message", ""))
        else:
            message = str(detail)
            error = {
                "message": message,
                "type": "invalid_request_error",
                "code": "invalid_request",
            }
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": error, "detail": message},
            headers=dict(exc.headers or {}),
        )

    if viz_html is not None:

        @app.get("/viz", response_class=HTMLResponse)
        def viz_page():
            return HTMLResponse(viz_html)

    app.state.response_store = response_store
    app.state.response_history_store = response_history_store
    app.state.response_store_lock = response_store_lock
    app.state.active_streams = active_streams
    app.state.active_streams_lock = active_streams_lock
    app.state.flight_lock = _flight_lock
    app.state.session = session

    return app


def open_live_session(
    run_dir: str | Path,
    *,
    context_length: int | None = None,
    num_threads: int | None = None,
):
    """Open a native session for ``run_dir`` (imports native binding lazily)."""
    from .runtime import load_manifest_templates, resolve_runtime_context_length
    from .session_v8 import SessionV8

    run_dir = Path(run_dir).expanduser().resolve()
    capacity = resolve_runtime_context_length(run_dir, context_length)
    session = SessionV8.open(run_dir, context_length=capacity, num_threads=num_threads)
    _, _, contract = load_manifest_templates(run_dir)
    return session, capacity, contract


def create_live_app_from_run_dir(
    run_dir: str | Path,
    *,
    model: str = "ck-v8",
    context_length: int | None = None,
    num_threads: int | None = None,
    **kwargs: Any,
):
    """Build a live app directly from a compiled runtime directory."""
    from .runtime import load_manifest_templates

    run_dir = Path(run_dir).expanduser().resolve()
    session, capacity, _ = open_live_session(
        run_dir, context_length=context_length, num_threads=num_threads
    )
    chat_template, chat_templates, contract = load_manifest_templates(run_dir)
    return create_app(
        session,
        model=model,
        context_length=capacity,
        chat_contract=contract,
        chat_template=chat_template,
        chat_templates=chat_templates,
        **kwargs,
    )
