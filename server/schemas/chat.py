"""OpenAI Chat Completions shapes used by local coding-agent clients."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ChatFunctionCall(BaseModel):
    name: str
    arguments: str


class ChatToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ChatFunctionCall


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[ChatContentPart] | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ChatToolCall] | None = None


class ChatFunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool = False


class ChatTool(BaseModel):
    type: Literal["function"] = "function"
    function: ChatFunctionDefinition


class ChatStreamOptions(BaseModel):
    include_usage: bool = False


class CreateChatCompletionRequest(BaseModel):
    """Supported Chat Completions subset, with unsupported options explicit."""

    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: ChatStreamOptions | None = None
    tools: list[ChatTool] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = None
    top_p: float | None = None
    n: int = Field(default=1, gt=0)
    stop: str | list[str] | None = None
    user: str | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    response_format: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None
