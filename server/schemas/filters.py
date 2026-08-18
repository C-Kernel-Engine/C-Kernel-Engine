"""Filter schema shapes; the scaffold does not execute searches."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ComparisonFilter(BaseModel):
    key: str
    type: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"]
    value: str | int | float | bool | list[str | int | float]


class CompoundFilter(BaseModel):
    filters: list["ComparisonFilter | CompoundFilter"]
    type: Literal["and", "or"]


Filter = Annotated[
    ComparisonFilter | CompoundFilter,
    Field(discriminator="type"),
]


class RankingOptions(BaseModel):
    hybrid_search: dict[str, float] | None = None
    ranker: Literal["auto", "default-2024-11-15"] | None = None
    score_threshold: float | None = None
