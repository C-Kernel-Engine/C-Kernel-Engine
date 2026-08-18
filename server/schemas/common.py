"""Shared enums and metadata shapes for the schema scaffold."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Role(StrEnum):
    user = "user"
    assistant = "assistant"
    system = "system"
    developer = "developer"


class ItemStatus(StrEnum):
    in_progress = "in_progress"
    completed = "completed"
    incomplete = "incomplete"


class ItemType(StrEnum):
    message = "message"
    file_search_call = "file_search_call"
    computer_call = "computer_call"
    computer_call_output = "computer_call_output"
    web_search_call = "web_search_call"
    function_call = "function_call"
    function_call_output = "function_call_output"
    tool_search_call = "tool_search_call"
    tool_search_output = "tool_search_output"
    reasoning = "reasoning"
    code_interpreter_call = "code_interpreter_call"


class ContentType(StrEnum):
    input_text = "input_text"
    input_image = "input_image"
    input_file = "input_file"
    output_text = "output_text"
    refusal = "refusal"


class ResponseStatus(StrEnum):
    completed = "completed"
    failed = "failed"
    in_progress = "in_progress"
    cancelled = "cancelled"
    queued = "queued"
    incomplete = "incomplete"


class ReasoningEffort(StrEnum):
    none = "none"
    minimal = "minimal"
    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"
    max = "max"


class ReasoningMode(StrEnum):
    standard = "standard"
    pro = "pro"


class ServiceTier(StrEnum):
    auto = "auto"
    default = "default"
    flex = "flex"
    scale = "scale"
    priority = "priority"
    fast = "fast"
    ultrafast = "ultrafast"


class Truncation(StrEnum):
    auto = "auto"
    disabled = "disabled"


class ErrorCode(StrEnum):
    server_error = "server_error"
    rate_limit_exceeded = "rate_limit_exceeded"
    invalid_prompt = "invalid_prompt"
    data_residency_mismatch = "data_residency_mismatch"
    bio_policy = "bio_policy"
    vector_store_timeout = "vector_store_timeout"
    invalid_image = "invalid_image"
    invalid_image_format = "invalid_image_format"
    invalid_base64_image = "invalid_base64_image"
    invalid_image_url = "invalid_image_url"
    image_too_large = "image_too_large"
    image_too_small = "image_too_small"
    image_parse_error = "image_parse_error"
    image_content_policy_violation = "image_content_policy_violation"
    invalid_image_mode = "invalid_image_mode"
    image_file_too_large = "image_file_too_large"
    unsupported_image_media_type = "unsupported_image_media_type"
    empty_image_file = "empty_image_file"
    failed_to_download_image = "failed_to_download_image"
    image_file_not_found = "image_file_not_found"


class IncompleteReason(StrEnum):
    max_output_tokens = "max_output_tokens"
    content_filter = "content_filter"


class FileSearchStatus(StrEnum):
    in_progress = "in_progress"
    searching = "searching"
    completed = "completed"
    incomplete = "incomplete"
    failed = "failed"


class WebSearchStatus(StrEnum):
    in_progress = "in_progress"
    searching = "searching"
    completed = "completed"
    failed = "failed"


class Phase(StrEnum):
    commentary = "commentary"
    final_answer = "final_answer"


class ComputerActionType(StrEnum):
    click = "click"
    double_click = "double_click"
    drag = "drag"
    keypress = "keypress"
    move = "move"
    screenshot = "screenshot"
    scroll = "scroll"
    type = "type"
    wait = "wait"


class ImageDetail(StrEnum):
    low = "low"
    high = "high"
    auto = "auto"
    original = "original"


class FileDetail(StrEnum):
    auto = "auto"
    low = "low"
    high = "high"


class FilterOperator(StrEnum):
    eq = "eq"
    ne = "ne"
    gt = "gt"
    gte = "gte"
    lt = "lt"
    lte = "lte"
    inn = "in"
    nin = "nin"


class FilterType(StrEnum):
    and_ = "and"
    or_ = "or"


class Includable(StrEnum):
    file_search_call_results = "file_search_call.results"
    web_search_call_results = "web_search_call.results"
    web_search_call_action_sources = "web_search_call.action.sources"
    message_input_image_image_url = "message.input_image.image_url"
    computer_call_output_output_image_url = "computer_call_output.output.image_url"
    code_interpreter_call_outputs = "code_interpreter_call.outputs"
    reasoning_encrypted_content = "reasoning.encrypted_content"
    message_output_text_logprobs = "message.output_text.logprobs"


class ToolType(StrEnum):
    function = "function"
    file_search = "file_search"
    computer = "computer"
    computer_use_preview = "computer_use_preview"
    web_search = "web_search"
    web_search_2025_08_26 = "web_search_2025_08_26"
    mcp = "mcp"
    code_interpreter = "code_interpreter"
    programmatic_tool_calling = "programmatic_tool_calling"
    image_generation = "image_generation"


class ResponseIncompleteDetails(BaseModel):
    reason: IncompleteReason | None = None


class ResponseError(BaseModel):
    code: ErrorCode
    message: str


class UsageInputTokensDetails(BaseModel):
    cache_write_tokens: int = 0
    cached_tokens: int = 0


class UsageOutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    input_tokens_details: UsageInputTokensDetails | None = None
    output_tokens_details: UsageOutputTokensDetails | None = None
