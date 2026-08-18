"""Streaming event shapes; generation remains mocked in this scaffold.

The event union mirrors the documented Responses stream events that this host
emits: lifecycle events carry ``sequence_number``, content events reference the
item they belong to via ``item_id``, and reasoning streams through the
``response.reasoning_*`` events.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from .output_items import ResponseOutputItem


class ResponseCreatedEvent(BaseModel):
    type: Literal["response.created"] = "response.created"
    response: dict[str, Any]
    sequence_number: int = 0


class ResponseInProgressEvent(BaseModel):
    type: Literal["response.in_progress"] = "response.in_progress"
    response: dict[str, Any]
    sequence_number: int = 0


class ResponseQueuedEvent(BaseModel):
    type: Literal["response.queued"] = "response.queued"
    response: dict[str, Any]
    sequence_number: int = 0


class ResponseCompletedEvent(BaseModel):
    type: Literal["response.completed"] = "response.completed"
    response: dict[str, Any]
    sequence_number: int = 0


class ResponseFailedEvent(BaseModel):
    type: Literal["response.failed"] = "response.failed"
    response: dict[str, Any]
    sequence_number: int = 0


class ResponseIncompleteEvent(BaseModel):
    type: Literal["response.incomplete"] = "response.incomplete"
    response: dict[str, Any]
    sequence_number: int = 0


class ResponseErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str | None = None
    message: str
    param: str | None = None
    sequence_number: int = 0


class ResponseOutputItemAddedEvent(BaseModel):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int
    item: ResponseOutputItem
    sequence_number: int = 0


class ResponseOutputItemDoneEvent(BaseModel):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int
    item: ResponseOutputItem
    sequence_number: int = 0


class ResponseContentPartAddedEvent(BaseModel):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    item_id: str
    output_index: int
    content_index: int
    part: dict[str, Any]
    sequence_number: int = 0


class ResponseContentPartDoneEvent(BaseModel):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: str
    output_index: int
    content_index: int
    part: dict[str, Any]
    sequence_number: int = 0


class ResponseTextDeltaEvent(BaseModel):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    item_id: str
    output_index: int
    content_index: int
    delta: str
    logprobs: list[Any] | None = None
    sequence_number: int = 0


class ResponseTextDoneEvent(BaseModel):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    item_id: str
    output_index: int
    content_index: int
    text: str
    sequence_number: int = 0


class ResponseRefusalDeltaEvent(BaseModel):
    type: Literal["response.refusal.delta"] = "response.refusal.delta"
    item_id: str
    output_index: int
    content_index: int
    delta: str
    sequence_number: int = 0


class ResponseRefusalDoneEvent(BaseModel):
    type: Literal["response.refusal.done"] = "response.refusal.done"
    item_id: str
    output_index: int
    content_index: int
    refusal: str
    sequence_number: int = 0


class ResponseFunctionCallArgumentsDeltaEvent(BaseModel):
    type: Literal[
        "response.function_call_arguments.delta"
    ] = "response.function_call_arguments.delta"
    item_id: str
    output_index: int
    delta: str
    sequence_number: int = 0


class ResponseFunctionCallArgumentsDoneEvent(BaseModel):
    type: Literal[
        "response.function_call_arguments.done"
    ] = "response.function_call_arguments.done"
    item_id: str
    output_index: int
    arguments: str
    sequence_number: int = 0


class ResponseReasoningTextDeltaEvent(BaseModel):
    type: Literal["response.reasoning_text.delta"] = "response.reasoning_text.delta"
    item_id: str
    output_index: int
    content_index: int
    delta: str
    sequence_number: int = 0


class ResponseReasoningTextDoneEvent(BaseModel):
    type: Literal["response.reasoning_text.done"] = "response.reasoning_text.done"
    item_id: str
    output_index: int
    content_index: int
    text: str
    sequence_number: int = 0


class ResponseReasoningSummaryTextDeltaEvent(BaseModel):
    type: Literal[
        "response.reasoning_summary_text.delta"
    ] = "response.reasoning_summary_text.delta"
    item_id: str
    output_index: int
    summary_index: int
    delta: str
    sequence_number: int = 0


class ResponseReasoningSummaryTextDoneEvent(BaseModel):
    type: Literal[
        "response.reasoning_summary_text.done"
    ] = "response.reasoning_summary_text.done"
    item_id: str
    output_index: int
    summary_index: int
    text: str
    sequence_number: int = 0


class ResponseReasoningSummaryPartAddedEvent(BaseModel):
    type: Literal[
        "response.reasoning_summary_part.added"
    ] = "response.reasoning_summary_part.added"
    item_id: str
    output_index: int
    summary_index: int
    part: dict[str, Any]
    sequence_number: int = 0


class ResponseReasoningSummaryPartDoneEvent(BaseModel):
    type: Literal[
        "response.reasoning_summary_part.done"
    ] = "response.reasoning_summary_part.done"
    item_id: str
    output_index: int
    summary_index: int
    part: dict[str, Any]
    sequence_number: int = 0


ResponseStreamEvent = (
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseQueuedEvent
    | ResponseCompletedEvent
    | ResponseFailedEvent
    | ResponseIncompleteEvent
    | ResponseErrorEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseRefusalDeltaEvent
    | ResponseRefusalDoneEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ResponseReasoningTextDeltaEvent
    | ResponseReasoningTextDoneEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseReasoningSummaryTextDoneEvent
    | ResponseReasoningSummaryPartAddedEvent
    | ResponseReasoningSummaryPartDoneEvent
)