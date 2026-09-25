"""Runtime artifact helpers for ``server/`` (no FastAPI here).

The server resolves the generated capacity (``layout_decode.json``) and the
canonical ``chat_template.jinja`` sidecar without importing the scripts tree.
``weights_manifest.json`` is never consulted for prompt rendering. Path
constants are shared with ``server.session_v8`` (single source of truth).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .session_v8 import BUILD_DIR, PROJECT_ROOT, SESSION_LIB_PATH


def ensure_native_session_lib() -> None:
    """Build ``libck_session_v8.so`` if it is not already present."""
    if SESSION_LIB_PATH.is_file():
        return
    subprocess.run(["make", "ck-session-v8"], cwd=str(PROJECT_ROOT), check=True)


def resolve_runtime_context_length(run_dir: Path, requested: int | None) -> int | None:
    """Return the generated plan's capacity, guarding oversized requests."""
    layout_path = Path(run_dir) / "layout_decode.json"
    try:
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
        value = payload.get("config", {}).get("context_length")
    except (OSError, UnicodeDecodeError, ValueError, AttributeError):
        value = None
    planned = value if isinstance(value, int) and value > 0 else None
    if requested is not None:
        if planned is not None and requested > planned:
            raise ValueError(
                f"requested context length {requested} exceeds generated runtime "
                f"capacity {planned}; rebuild with --context-len {requested}"
            )
        return requested
    return planned


def load_manifest_templates(
    run_dir: Path,
) -> tuple[str | None, dict[str, str] | None, dict[str, Any] | None]:
    """Load (chat_template, chat_templates, chat_contract) for a run dir.

    The native ``chat_template`` comes exclusively from the canonical
    ``chat_template.jinja`` sidecar emitted by GGUF conversion.
    ``weights_manifest.json`` / ``config.json`` are never read here, and no
    chat contract is loaded from disk (always ``None``) — prompt rendering
    is pure Jinja from the sidecar.
    """
    run_dir = Path(run_dir)
    chat_template: str | None = None
    chat_templates: dict[str, str] | None = None
    chat_contract: dict[str, Any] | None = None
    sidecar = run_dir / "chat_template.jinja"
    if sidecar.is_file():
        try:
            txt = sidecar.read_text(encoding="utf-8").strip()
            if txt:
                chat_template = txt
        except OSError:
            pass
    additional_dir = run_dir / "additional_chat_templates"
    if additional_dir.is_dir():
        collected: dict[str, str] = {}
        for jinja_file in additional_dir.glob("*.jinja"):
            try:
                txt = jinja_file.read_text(encoding="utf-8").strip()
                if txt:
                    collected[jinja_file.stem] = txt
            except OSError:
                continue
        if collected:
            chat_templates = collected
    return chat_template, chat_templates, chat_contract
