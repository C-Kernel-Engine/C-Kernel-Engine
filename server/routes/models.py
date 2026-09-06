"""Model listing routes for schema scaffold (mock, no inference).

Exposes OpenAI-compatible:
  GET /v1/models       -> ModelList with single ck-v8 entry
  GET /v1/models/{id}  -> Model or 404
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from ..schemas.model import Model, ModelList

router = APIRouter()

_CREATED = int(time.time())
_MODEL_ID = "ck-v8"
_OWNED_BY = "cke"


@router.get("/models", response_model=ModelList)
def list_models() -> ModelList:
    return ModelList(
        data=[Model(id=_MODEL_ID, created=_CREATED, owned_by=_OWNED_BY)],
    )


@router.get("/models/{model_id}", response_model=Model)
def retrieve_model(model_id: str) -> Model:
    if model_id != _MODEL_ID:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found")
    return Model(id=_MODEL_ID, created=_CREATED, owned_by=_OWNED_BY)
