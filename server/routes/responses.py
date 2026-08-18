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
from ..schemas.output_items import ResponseOutputMessage
from ..schemas.response import CreateResponseRequest, Response

router = APIRouter()

_response_store: dict[str, Response] = {}


MOCK_TEXT = "CKE schema scaffold: inference is not connected."


def _make_mock_response(model: str) -> Response:
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    now = int(time.time())
    resp = Response(
        id=resp_id,
        created_at=now,
        completed_at=now,
        status=ResponseStatus.completed,
        model=model,
        output_text=MOCK_TEXT,
        output=[
            ResponseOutputMessage(
                id=f"msg_{uuid.uuid4().hex[:24]}",
                content=[ResponseOutputText(text=MOCK_TEXT)],
                role=Role.assistant,
                status=ItemStatus.completed,
            )
        ],
        usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
    )
    _response_store[resp_id] = resp
    return resp


def _stream_response(resp: Response) -> StreamingResponse:
    """Return the mocked response through the standard SSE request shape."""
    serialized = resp.model_dump(mode="json")
    events = [
        {"type": "response.created", "response": serialized},
        {"type": "response.in_progress", "response": serialized},
        {
            "type": "response.output_text.delta",
            "response_id": resp.id,
            "output_index": 0,
            "content_index": 0,
            "delta": MOCK_TEXT,
        },
        {"type": "response.completed", "response": serialized},
    ]

    async def event_generator():
        for event in events:
            yield f"event: {event['type']}\n"
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/responses", response_model=None)
def create_response(body: CreateResponseRequest):
    resp = _make_mock_response(body.model)
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
    resp.status = ResponseStatus.cancelled
    return resp
