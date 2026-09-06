"""Mock Responses API routes for schema and streaming-event development.

No CKE runtime is loaded. The production server must replace this module's
deterministic mock generation and in-memory store with a bounded native runtime
queue, cancellation, and one model load per server process.
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..schemas.common import ItemStatus, ResponseStatus, Role, Usage
from ..schemas.content import ResponseOutputText
from ..schemas.output_items import ReasoningItem, ResponseOutputMessage
from ..schemas.response import CreateResponseRequest, Response

router = APIRouter()

_response_store: dict[str, Response] = {}


MOCK_TEXT = "CKE schema scaffold: inference is not connected."


def _make_mock_response(model: str, body: CreateResponseRequest | None = None) -> Response:
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    rsn_id = f"rsn_{uuid.uuid4().hex[:24]}"
    now = int(time.time())
    # B: include empty reasoning when reasoning requested (even if MOCK_TEXT has no markers)
    output_items: list = []
    if body is not None and body.reasoning is not None:
        output_items.append(
            ReasoningItem(
                id=rsn_id,
                status=ItemStatus.completed,
                content=[],
                summary=[],
            )
        )
    output_items.append(
        ResponseOutputMessage(
            id=msg_id,
            content=[ResponseOutputText(text=MOCK_TEXT)],
            role=Role.assistant,
            status=ItemStatus.completed,
        )
    )
    resp = Response(
        id=resp_id,
        created_at=now,
        completed_at=now,
        status=ResponseStatus.completed,
        model=model,
        output_text=MOCK_TEXT,
        output=output_items,  # type: ignore[arg-type]
        usage=Usage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_tokens_details={"cache_write_tokens": 0, "cached_tokens": 0},
            output_tokens_details={"reasoning_tokens": 0},
        ),
    )
    # Echo standard request fields where applicable (so scaffold mirrors live echo)
    if body is not None:
        if body.instructions is not None:
            resp.instructions = body.instructions
        if body.reasoning is not None:
            resp.reasoning = body.reasoning
        if body.temperature is not None:
            resp.temperature = body.temperature
        if body.top_p is not None:
            resp.top_p = body.top_p
        if body.top_logprobs is not None:
            resp.top_logprobs = body.top_logprobs
        if body.max_output_tokens is not None:
            resp.max_output_tokens = body.max_output_tokens
        if body.background is not None:
            resp.background = body.background
        if body.conversation is not None:
            if isinstance(body.conversation, str):
                from server.schemas.response import ResponseConversationParam as _RCP

                resp.conversation = _RCP(id=body.conversation)
            else:
                resp.conversation = body.conversation  # type: ignore[assignment]
    # Persist unless store==False (OpenAI: store:false → ephemeral)
    should_store = True
    if body is not None and body.store is False:
        should_store = False
    if should_store:
        _response_store[resp_id] = resp
    return resp


def _build_mock_pending_and_final(resp: Response) -> tuple[dict, dict]:
    """Return (pending, final) dicts for streaming — pending is in_progress with empty output."""
    pending = resp.model_dump(mode="json")
    # pending state: mimic live pending (seq 0/1)
    pending["status"] = ResponseStatus.in_progress.value if isinstance(ResponseStatus.in_progress, str) else str(ResponseStatus.in_progress)
    pending["status"] = "in_progress"
    pending["output"] = []
    pending["output_text"] = ""
    pending["completed_at"] = None
    pending["usage"] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    final = resp.model_dump(mode="json")
    # final stays completed as built by _make_mock_response
    return pending, final


def _stream_response(resp: Response) -> StreamingResponse:
    """Return the mocked response through the standard SSE shape, fully OpenAI-compatible.

    Emits: created → in_progress → [reasoning added/done] → output_item.added → content_part.added → delta → content_part.done → output_text.done → output_item.done → completed
    with incrementing sequence_number and item_id (not response_id), plus proper SSE headers.
    For option B, when reasoning is requested we emit an empty ReasoningItem (no deltas) to match live non-stream parity.
    """
    pending, final = _build_mock_pending_and_final(resp)
    # Detect reasoning in final output (option B)
    has_reasoning = False
    rsn_id = None
    msg_id = None
    msg_item = None
    for it in final.get("output") or []:
        if it.get("type") == "reasoning":
            has_reasoning = True
            rsn_id = it.get("id")
        elif it.get("type") == "message":
            msg_id = it.get("id")
            msg_item = it
    if msg_id is None:
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        msg_item = {"id": msg_id, "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": MOCK_TEXT, "annotations": []}]}
    if has_reasoning and rsn_id is None:
        rsn_id = f"rsn_{uuid.uuid4().hex[:24]}"
    # Indices differ when reasoning present
    msg_output_index = 1 if has_reasoning else 0
    content_index = 0
    part_added = {"type": "output_text", "text": "", "annotations": []}
    part_done = {"type": "output_text", "text": MOCK_TEXT, "annotations": []}

    seq = 0

    def _sse(event_type: str, data: dict) -> dict:
        nonlocal seq
        data["sequence_number"] = seq
        seq += 1
        # ensure type inside data matches event line
        if "type" not in data:
            data["type"] = event_type
        return {"event": event_type, "data": data}

    events: list[dict] = []
    events.append(_sse("response.created", {"type": "response.created", "response": pending}))
    events.append(_sse("response.in_progress", {"type": "response.in_progress", "response": pending}))
    if has_reasoning:
        # Empty reasoning item lifecycle (no thinking deltas for MOCK_TEXT)
        events.append(
            _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "id": rsn_id,
                        "type": "reasoning",
                        "status": "in_progress",
                        "content": [],
                        "summary": [],
                    },
                },
            )
        )
        events.append(
            _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "id": rsn_id,
                        "type": "reasoning",
                        "status": "completed",
                        "content": [],
                        "summary": [],
                    },
                },
            )
        )
    # output_item.added for message (empty content, in_progress)
    events.append(
        _sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": msg_output_index,
                "item": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            },
        )
    )
    events.append(
        _sse(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": msg_id,
                "output_index": msg_output_index,
                "content_index": content_index,
                "part": part_added,
            },
        )
    )
    events.append(
        _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": msg_id,
                "output_index": msg_output_index,
                "content_index": content_index,
                "delta": MOCK_TEXT,
                "logprobs": None,
            },
        )
    )
    events.append(
        _sse(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": msg_id,
                "output_index": msg_output_index,
                "content_index": content_index,
                "part": part_done,
            },
        )
    )
    events.append(
        _sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": msg_id,
                "output_index": msg_output_index,
                "content_index": content_index,
                "text": MOCK_TEXT,
            },
        )
    )
    events.append(
        _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": msg_output_index,
                "item": msg_item if msg_item else {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": MOCK_TEXT, "annotations": []}],
                },
            },
        )
    )
    events.append(_sse("response.completed", {"type": "response.completed", "response": final}))

    async def event_generator():
        for ev in events:
            et = ev["event"]
            data = ev["data"]
            yield f"event: {et}\n"
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/responses", response_model=None)
def create_response(body: CreateResponseRequest):
    resp = _make_mock_response(body.model, body)
    if body.stream:
        return _stream_response(resp)
    return resp


@router.get("/responses/{response_id}", response_model=Response)
def get_response(response_id: str):
    resp = _response_store.get(response_id)
    if not resp:
        raise HTTPException(status_code=404, detail="Response not found")
    return resp


@router.post("/responses/{response_id}/cancel", response_model=Response)
def cancel_response(response_id: str):
    resp = _response_store.get(response_id)
    if not resp:
        raise HTTPException(status_code=404, detail="Response not found")
    if resp.status != ResponseStatus.in_progress:
        raise HTTPException(status_code=409, detail="Response is not in progress")
    resp.status = ResponseStatus.cancelled
    return resp
