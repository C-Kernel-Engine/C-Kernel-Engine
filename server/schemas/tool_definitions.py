"""Tool-definition shapes; no tool execution is connected."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .filters import Filter, RankingOptions


class FunctionTool(BaseModel):
    name: str
    parameters: dict[str, Any]
    strict: bool = False
    type: Literal["function"] = "function"
    allowed_callers: list[Literal["direct", "programmatic"]] | None = None
    defer_loading: bool | None = None
    description: str | None = None
    output_schema: dict[str, Any] | None = None


class FileSearchTool(BaseModel):
    type: Literal["file_search"] = "file_search"
    vector_store_ids: list[str]
    filters: Filter | None = None
    max_num_results: int | None = None
    ranking_options: RankingOptions | None = None


class ComputerTool(BaseModel):
    type: Literal["computer"] = "computer"


class ComputerUsePreviewTool(BaseModel):
    display_height: int
    display_width: int
    environment: Literal["windows", "mac", "linux", "ubuntu", "browser"]
    type: Literal["computer_use_preview"] = "computer_use_preview"


class WebSearchUserLocation(BaseModel):
    city: str | None = None
    country: str | None = None
    region: str | None = None
    timezone: str | None = None
    type: Literal["approximate"] | None = "approximate"


class WebSearchTool(BaseModel):
    type: Literal["web_search", "web_search_2025_08_26"]
    external_web_access: bool | None = None
    filters: dict[str, list[str]] | None = None
    search_context_size: Literal["low", "medium", "high"] | None = None
    user_location: WebSearchUserLocation | None = None


class CodeInterpreterToolAuto(BaseModel):
    type: Literal["auto"] = "auto"
    file_ids: list[str] | None = None
    memory_limit: Literal["1g", "4g", "16g", "64g"] | None = None
    network_policy: dict[str, Any] | None = None


class CodeInterpreterTool(BaseModel):
    container: str | CodeInterpreterToolAuto
    type: Literal["code_interpreter"] = "code_interpreter"
    allowed_callers: list[Literal["direct", "programmatic"]] | None = None


class ProgrammaticToolCallingTool(BaseModel):
    type: Literal["programmatic_tool_calling"] = "programmatic_tool_calling"


class ImageGenerationTool(BaseModel):
    type: Literal["image_generation"] = "image_generation"
    action: Literal["generate", "edit", "auto"] | None = None
    background: Literal["transparent", "opaque", "auto"] | None = None
    input_fidelity: Literal["high", "low"] | None = None
    input_image_mask: dict[str, Any] | None = None
    model: str | None = None
    moderation: Literal["auto", "low"] | None = None
    output_compression: int | None = None
    output_format: Literal["png", "webp", "jpeg"] | None = None
    partial_images: int | None = None
    quality: Literal["low", "medium", "high", "auto"] | None = None
    size: str | None = None


class McpToolFilter(BaseModel):
    read_only: bool | None = None
    tool_names: list[str] | None = None


class McpToolApprovalFilter(BaseModel):
    always: McpToolFilter | None = None
    never: McpToolFilter | None = None


class McpTool(BaseModel):
    server_label: str
    type: Literal["mcp"] = "mcp"
    allowed_callers: list[Literal["direct", "programmatic"]] | None = None
    allowed_tools: list[str] | McpToolFilter | None = None
    authorization: str | None = None
    connector_id: Literal[
        "connector_dropbox",
        "connector_gmail",
        "connector_googlecalendar",
        "connector_googledrive",
        "connector_microsoftteams",
        "connector_outlookcalendar",
        "connector_outlookemail",
        "connector_sharepoint",
    ] | None = None
    defer_loading: bool | None = None
    headers: dict[str, str] | None = None
    require_approval: McpToolApprovalFilter | Literal["always", "never"] | None = None
    server_description: str | None = None
    server_url: str | None = None
    tunnel_id: str | None = None


ToolDefinition = Annotated[
    FunctionTool
    | FileSearchTool
    | ComputerTool
    | ComputerUsePreviewTool
    | WebSearchTool
    | McpTool
    | CodeInterpreterTool
    | ProgrammaticToolCallingTool
    | ImageGenerationTool,
    Field(discriminator="type"),
]
