"""Conversation request shapes backed only by process-local mock storage."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Conversation(BaseModel):
    id: str
    object: str = "conversation"
    created_at: int
    items: list[Any] = Field(default_factory=list)


class CreateConversationRequest(BaseModel):
    items: list[Any] | None = None
    metadata: dict[str, str] | None = None


class AddConversationItemsRequest(BaseModel):
    items: list[Any]
