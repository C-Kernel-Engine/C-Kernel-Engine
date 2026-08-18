"""Responses output-item shapes emitted as deterministic mock data."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .common import FileSearchStatus, ItemStatus, Phase, WebSearchStatus
from .computer_actions import ComputerAction
from .content import ResponseInputContent, ResponseOutputContent


class ResponseOutputMessage(BaseModel):
    id: str
    content: list[ResponseOutputContent]
    role: Literal["assistant"] = "assistant"
    status: ItemStatus
    type: Literal["message"] = "message"
    phase: Phase | None = None


class FileSearchResult(BaseModel):
    attributes: dict[str, str | int | float | bool] | None = None
    file_id: str | None = None
    filename: str | None = None
    score: float | None = None
    text: str | None = None


class FileSearchCall(BaseModel):
    id: str
    queries: list[str]
    status: FileSearchStatus
    type: Literal["file_search_call"] = "file_search_call"
    results: list[FileSearchResult] | None = None


class ComputerCall(BaseModel):
    id: str
    call_id: str
    pending_safety_checks: list[dict[str, Any]] = []
    status: ItemStatus
    type: Literal["computer_call"] = "computer_call"
    action: ComputerAction | None = None
    actions: list[ComputerAction] | None = None


class ComputerCallOutputScreenshot(BaseModel):
    type: Literal["computer_screenshot"] = "computer_screenshot"
    file_id: str | None = None
    image_url: str | None = None


class SafetyCheck(BaseModel):
    id: str
    code: str | None = None
    message: str | None = None


class ComputerCallOutput(BaseModel):
    call_id: str
    output: ComputerCallOutputScreenshot
    type: Literal["computer_call_output"] = "computer_call_output"
    id: str | None = None
    acknowledged_safety_checks: list[SafetyCheck] | None = None
    status: ItemStatus | None = None


class WebSearchActionSearch(BaseModel):
    type: Literal["search"] = "search"
    queries: list[str] | None = None
    query: str | None = None
    sources: list[dict[str, str]] | None = None


class WebSearchActionOpenPage(BaseModel):
    type: Literal["open_page"] = "open_page"
    url: str | None = None


class WebSearchActionFindInPage(BaseModel):
    pattern: str
    type: Literal["find_in_page"] = "find_in_page"
    url: str


WebSearchAction = Annotated[
    WebSearchActionSearch | WebSearchActionOpenPage | WebSearchActionFindInPage,
    Field(discriminator="type"),
]


class WebSearchCall(BaseModel):
    id: str
    action: WebSearchAction
    status: WebSearchStatus
    type: Literal["web_search_call"] = "web_search_call"


class CallerDirect(BaseModel):
    type: Literal["direct"] = "direct"


class CallerProgram(BaseModel):
    caller_id: str
    type: Literal["program"] = "program"


Caller = Annotated[CallerDirect | CallerProgram, Field(discriminator="type")]


class FunctionCall(BaseModel):
    arguments: str
    call_id: str
    name: str
    type: Literal["function_call"] = "function_call"
    id: str | None = None
    caller: Caller | None = None
    namespace: str | None = None
    status: ItemStatus | None = None


class FunctionCallOutput(BaseModel):
    call_id: str
    output: str | list[ResponseInputContent]
    type: Literal["function_call_output"] = "function_call_output"
    id: str | None = None
    caller: Caller | None = None
    status: ItemStatus | None = None


class ToolSearchCall(BaseModel):
    arguments: Any = None
    type: Literal["tool_search_call"] = "tool_search_call"
    id: str | None = None
    call_id: str | None = None
    execution: Literal["server", "client"] | None = None
    status: ItemStatus | None = None


class ToolSearchOutput(BaseModel):
    tools: list[Any] = []
    type: Literal["tool_search_output"] = "tool_search_output"
    id: str | None = None
    call_id: str | None = None
    status: ItemStatus | None = None


class ReasoningTextContent(BaseModel):
    text: str
    type: Literal["reasoning_text"] = "reasoning_text"


class SummaryTextContent(BaseModel):
    text: str
    type: Literal["summary_text"] = "summary_text"


class ReasoningItem(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    id: str | None = None
    status: ItemStatus | None = None
    content: list[ReasoningTextContent] | None = None
    summary: list[SummaryTextContent] = Field(default_factory=list)
    encrypted_content: str | None = None


ResponseOutputItem = Annotated[
    ResponseOutputMessage
    | FileSearchCall
    | ComputerCall
    | ComputerCallOutput
    | WebSearchCall
    | FunctionCall
    | FunctionCallOutput
    | ToolSearchCall
    | ToolSearchOutput
    | ReasoningItem,
    Field(discriminator="type"),
]
