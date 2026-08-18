"""Request and response models for the experimental schema subset."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import (
    Includable,
    ReasoningEffort,
    ReasoningMode,
    ResponseError,
    ResponseIncompleteDetails,
    ResponseStatus,
    ServiceTier,
    Truncation,
    Usage,
)
from .input_items import EasyInputMessage, Message
from .output_items import (
    ComputerCall,
    ComputerCallOutput,
    FileSearchCall,
    FunctionCall,
    FunctionCallOutput,
    ReasoningItem,
    ResponseOutputItem,
    ResponseOutputMessage,
    ToolSearchCall,
    ToolSearchOutput,
    WebSearchCall,
)
from .tool_definitions import ToolDefinition


class ContextManagementEntry(BaseModel):
    type: str = "compaction"
    compact_threshold: int | None = None


class ResponseConversationParam(BaseModel):
    id: str


class Reasoning(BaseModel):
    context: Literal["auto", "current_turn", "all_turns"] | None = None
    effort: ReasoningEffort | None = None
    generate_summary: Literal["auto", "concise", "detailed"] | None = None
    mode: ReasoningMode | str | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None


class ModerationPolicy(BaseModel):
    mode: Literal["score", "block"] | None = None


class ModerationPolicyConfig(BaseModel):
    input: ModerationPolicy | None = None
    output: ModerationPolicy | None = None


class Moderation(BaseModel):
    model: str | None = None
    policy: ModerationPolicyConfig | None = None


class ResponsePrompt(BaseModel):
    id: str
    variables: dict[str, Any] | None = None
    version: str | None = None


class PromptCacheOptions(BaseModel):
    mode: Literal["implicit", "explicit"] | None = None
    ttl: Literal["30m"] | None = None


class StreamOptions(BaseModel):
    include_obfuscation: bool | None = None


class TextConfig(BaseModel):
    format: dict[str, Any] | None = None
    verbosity: Literal["low", "medium", "high"] | None = None


CreateResponseInput = str | list[
    EasyInputMessage
    | Message
    | ResponseOutputMessage
    | FileSearchCall
    | ComputerCall
    | ComputerCallOutput
    | WebSearchCall
    | FunctionCall
    | FunctionCallOutput
    | ToolSearchCall
    | ToolSearchOutput
    | ReasoningItem
]


class CreateResponseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    input: CreateResponseInput | None = None
    instructions: str | None = None
    conversation: str | ResponseConversationParam | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    include: list[Includable] | None = None
    metadata: dict[str, str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_logprobs: int | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    stream: bool | None = None
    stream_options: StreamOptions | None = None
    reasoning: Reasoning | None = None
    truncation: Truncation | str | None = None
    text: TextConfig | None = None
    user: str | None = None
    background: bool | None = None
    context_management: list[ContextManagementEntry] | None = None
    prompt_cache_options: PromptCacheOptions | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    moderation: Moderation | None = None
    prompt: ResponsePrompt | None = None
    safety_identifier: str | None = None
    service_tier: ServiceTier | str | None = None


class Response(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    status: ResponseStatus
    error: ResponseError | None = None
    incomplete_details: ResponseIncompleteDetails | None = None
    instructions: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    model: str
    output: list[ResponseOutputItem] = Field(default_factory=list)
    parallel_tool_calls: bool = True
    temperature: float | None = None
    tool_choice: str | dict[str, Any] | None = None
    tools: list[ToolDefinition] = Field(default_factory=list)
    top_p: float | None = None
    top_logprobs: int | None = None
    background: bool | None = None
    completed_at: int | None = None
    conversation: ResponseConversationParam | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    moderation: Moderation | None = None
    output_text: str | None = None
    previous_response_id: str | None = None
    prompt: ResponsePrompt | None = None
    prompt_cache_key: str | None = None
    prompt_cache_options: PromptCacheOptions | None = None
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    reasoning: Reasoning | None = None
    safety_identifier: str | None = None
    service_tier: ServiceTier | str | None = None
    text: TextConfig | None = None
    truncation: Truncation | str | None = None
    usage: Usage | None = None
    user: str | None = None


class ResponseList(BaseModel):
    object: str = "list"
    data: list[Response]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False