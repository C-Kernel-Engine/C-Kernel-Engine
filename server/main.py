"""Experimental schema server with mocked responses and no model execution.

FastAPI is used only to validate the initial HTTP and Pydantic contract. For
generic OpenAI Responses + tools with real inference, build the live app via
``server.live.create_app(session, ...)`` or
``server.live.create_live_app_from_run_dir(run_dir, ...)`` backed by
``server.session_v8.SessionV8``; that factory ports the single-flight native
lifecycle (history, function/mcp tool calls, SSE streaming, cancel) into the
server boundary.
"""

from __future__ import annotations

from fastapi import FastAPI

from .routes.conversations import router as conversations_router
from .routes.models import router as models_router
from .routes.responses import router as responses_router

app = FastAPI(
    title="CKE Experimental Responses Schema Scaffold",
    description=(
        "Schema-only development scaffold. Responses are deterministic mock "
        "data; no CKE model is loaded or executed."
    ),
    version="0.1.0",
)

app.include_router(responses_router, prefix="/v1")
app.include_router(conversations_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "schema_scaffold",
        "inference": False,
    }
