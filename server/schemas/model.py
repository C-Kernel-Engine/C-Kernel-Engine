"""Model listing schemas for GET /v1/models (OpenAI-compatible)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "cke"
    cke_context_length: int | None = None
    cke_default_max_output_tokens: int | None = None


class ModelList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object: Literal["list"] = "list"
    data: list[Model]
