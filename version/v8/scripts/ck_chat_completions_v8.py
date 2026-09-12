"""Chat Completions compatibility adapter over CKE's Responses endpoint."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from server.schemas.chat import CreateChatCompletionRequest
from server.schemas.response import CreateResponseRequest, Reasoning


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content)


def _responses_input(body: CreateChatCompletionRequest) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    known_calls: set[str] = set()
    for message in body.messages:
        if message.name:
            raise HTTPException(
                status_code=422, detail="Named Chat Completions messages are unsupported"
            )
        if message.reasoning or message.reasoning_content:
            raise HTTPException(
                status_code=422,
                detail="Reasoning history is unsupported; configure thinking off",
            )
        text = _message_text(message.content)
        if message.role == "assistant" and message.tool_calls:
            if text:
                items.append({"type": "message", "role": "assistant", "content": text})
            for call in message.tool_calls:
                if call.id in known_calls:
                    raise HTTPException(
                        status_code=400, detail=f"Duplicate tool call id {call.id!r}"
                    )
                known_calls.add(call.id)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                )
            continue
        if message.role == "tool":
            if not message.tool_call_id:
                raise HTTPException(
                    status_code=400, detail="Tool messages require tool_call_id"
                )
            if message.tool_call_id not in known_calls:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Tool message references unknown tool_call_id "
                        f"{message.tool_call_id!r}"
                    ),
                )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": text,
                }
            )
            continue
        items.append({"type": "message", "role": message.role, "content": text})
    return items


def _responses_tools(body: CreateChatCompletionRequest) -> list[dict[str, Any]] | None:
    if not body.tools:
        return None
    return [
        {
            "type": "function",
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": tool.function.parameters,
            "strict": tool.function.strict,
        }
        for tool in body.tools
    ]


def _validate_supported(body: CreateChatCompletionRequest) -> None:
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    if body.n != 1:
        raise HTTPException(status_code=422, detail="CKE supports only n=1")
    unsupported = {
        "stop": body.stop,
        "frequency_penalty": body.frequency_penalty,
        "presence_penalty": body.presence_penalty,
        "logprobs": body.logprobs,
        "top_logprobs": body.top_logprobs,
        "response_format": body.response_format,
        "seed": body.seed,
    }
    active = [name for name, value in unsupported.items() if value is not None]
    if active:
        raise HTTPException(
            status_code=422,
            detail="Unsupported Chat Completions options: " + ", ".join(active),
        )
    template_keys = set((body.chat_template_kwargs or {}).keys())
    if template_keys - {"enable_thinking"}:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported chat_template_kwargs: "
                + ", ".join(sorted(template_keys - {"enable_thinking"}))
            ),
        )
    if (
        body.max_tokens is not None
        and body.max_completion_tokens is not None
        and body.max_tokens != body.max_completion_tokens
    ):
        raise HTTPException(
            status_code=422,
            detail="max_tokens and max_completion_tokens disagree",
        )


def _to_responses_request(body: CreateChatCompletionRequest) -> CreateResponseRequest:
    _validate_supported(body)
    template_kwargs = body.chat_template_kwargs or {}
    include_thinking = bool(template_kwargs.get("enable_thinking"))
    return CreateResponseRequest(
        model=body.model,
        input=_responses_input(body),
        tools=_responses_tools(body),
        tool_choice=body.tool_choice,
        parallel_tool_calls=body.parallel_tool_calls,
        max_output_tokens=body.max_completion_tokens or body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        stream=body.stream,
        reasoning=Reasoning(effort=body.reasoning_effort)
        if include_thinking or body.reasoning_effort
        else None,
        user=body.user,
    )


def _finish_reason(response: dict[str, Any]) -> str:
    if any(item.get("type") == "function_call" for item in response.get("output", [])):
        return "tool_calls"
    details = response.get("incomplete_details") or {}
    if details.get("reason") == "max_output_tokens":
        return "length"
    return "stop"


def _chat_completion(response: dict[str, Any]) -> dict[str, Any]:
    status = str(response.get("status") or "")
    if status in {"failed", "cancelled"}:
        error = response.get("error") or {}
        raise HTTPException(
            status_code=500,
            detail=error.get("message") or f"CKE generation {status}",
        )
    tool_calls = [
        {
            "id": item["call_id"],
            "type": "function",
            "function": {"name": item["name"], "arguments": item["arguments"]},
        }
        for item in response.get("output", [])
        if item.get("type") == "function_call"
    ]
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.get("output_text") or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    reasoning = "".join(
        part.get("text", "")
        for item in response.get("output", [])
        if item.get("type") == "reasoning"
        for part in item.get("content") or []
    )
    if reasoning:
        message["reasoning_content"] = reasoning
    usage = response.get("usage") or {}
    return {
        "id": response["id"].replace("resp_", "chatcmpl_", 1),
        "object": "chat.completion",
        "created": response["created_at"],
        "model": response["model"],
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _finish_reason(response),
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def _chat_chunk(
    completion_id: str,
    model: str,
    created: int,
    *,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": (
            []
            if usage is not None
            else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        ),
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload)}\n\n"


async def _iter_response_events(response: StreamingResponse) -> AsyncIterator[dict[str, Any]]:
    pending = ""
    async for chunk in response.body_iterator:
        pending += chunk.decode() if isinstance(chunk, bytes) else str(chunk)
        while "\n\n" in pending:
            block, pending = pending.split("\n\n", 1)
            data = "\n".join(
                line[6:] for line in block.splitlines() if line.startswith("data: ")
            )
            if data:
                yield json.loads(data)
    if pending.strip():
        raise RuntimeError("Responses stream ended with an incomplete SSE event")


async def _chat_stream(
    response: StreamingResponse,
    body: CreateChatCompletionRequest,
) -> AsyncIterator[str]:
    created = int(time.time())
    completion_id = "chatcmpl_pending"
    tool_indexes: dict[int, int] = {}
    role_sent = False
    terminal_sent = False
    async for event in _iter_response_events(response):
        event_type = event.get("type")
        if event_type == "response.created":
            current = event.get("response") or {}
            completion_id = str(current.get("id") or completion_id).replace(
                "resp_", "chatcmpl_", 1
            )
            created = int(current.get("created_at") or created)
            yield _chat_chunk(
                completion_id, body.model, created, delta={"role": "assistant"}
            )
            role_sent = True
        elif event_type == "response.output_text.delta":
            if not role_sent:
                yield _chat_chunk(
                    completion_id, body.model, created, delta={"role": "assistant"}
                )
                role_sent = True
            yield _chat_chunk(
                completion_id,
                body.model,
                created,
                delta={"content": event.get("delta") or ""},
            )
        elif event_type == "response.reasoning_text.delta":
            yield _chat_chunk(
                completion_id,
                body.model,
                created,
                delta={"reasoning_content": event.get("delta") or ""},
            )
        elif event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                output_index = int(event.get("output_index") or 0)
                tool_index = len(tool_indexes)
                tool_indexes[output_index] = tool_index
                yield _chat_chunk(
                    completion_id,
                    body.model,
                    created,
                    delta={
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "id": item.get("call_id"),
                                "type": "function",
                                "function": {
                                    "name": item.get("name"),
                                    "arguments": "",
                                },
                            }
                        ]
                    },
                )
        elif event_type == "response.function_call_arguments.delta":
            output_index = int(event.get("output_index") or 0)
            yield _chat_chunk(
                completion_id,
                body.model,
                created,
                delta={
                    "tool_calls": [
                        {
                            "index": tool_indexes[output_index],
                            "function": {"arguments": event.get("delta") or ""},
                        }
                    ]
                },
            )
        elif event_type in {"response.completed", "response.incomplete"}:
            current = event.get("response") or {}
            yield _chat_chunk(
                completion_id,
                body.model,
                created,
                delta={},
                finish_reason=_finish_reason(current),
            )
            if body.stream_options and body.stream_options.include_usage:
                usage = current.get("usage") or {}
                yield _chat_chunk(
                    completion_id,
                    body.model,
                    created,
                    delta={},
                    usage={
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                )
            terminal_sent = True
        elif event_type in {"response.failed", "response.cancelled", "error"}:
            if terminal_sent:
                continue
            message = event.get("message")
            if not message:
                current = event.get("response") or {}
                message = (current.get("error") or {}).get("message")
            yield f"data: {json.dumps({'error': {'message': message or event_type, 'type': 'server_error'}})}\n\n"
            terminal_sent = True
    if not terminal_sent:
        yield f"data: {json.dumps({'error': {'message': 'CKE stream ended without a terminal event', 'type': 'server_error'}})}\n\n"
    yield "data: [DONE]\n\n"


def add_chat_completions_route(
    router: APIRouter,
    create_response: Callable[[CreateResponseRequest, Request], Any],
) -> None:
    """Register the compatibility route against the canonical Responses handler."""

    @router.post("/chat/completions", response_model=None)
    async def create_chat_completion(
        body: CreateChatCompletionRequest, request: Request
    ):
        responses_body = _to_responses_request(body)
        response = create_response(responses_body, request)
        if body.stream:
            if not isinstance(response, StreamingResponse):
                raise HTTPException(
                    status_code=500, detail="CKE did not return a streaming response"
                )
            return StreamingResponse(
                _chat_stream(response, body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        if not isinstance(response, dict):
            raise HTTPException(status_code=500, detail="Invalid CKE response envelope")
        return _chat_completion(response)
