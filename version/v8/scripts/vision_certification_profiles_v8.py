#!/usr/bin/env python3
"""Load fail-closed model contracts for multimodal corpus certification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROFILE_SCHEMA = "cke.multimodal_certification_profile"
PROFILE_SCHEMA_VERSION = 1
PROFILE_DIR = Path(__file__).resolve().parents[1] / "parity_profiles" / "vision"
BUILTIN_PROFILES = {
    "gemma4": PROFILE_DIR / "gemma4_llamacpp_v1.json",
    "qwen3vl": PROFILE_DIR / "qwen3vl_llamacpp_v1.json",
    "qwen36vl": PROFILE_DIR / "qwen36vl_llamacpp_v1.json",
}
_REQUIRED_KEYS = {
    "schema",
    "schema_version",
    "id",
    "model_label",
    "chat_template",
    "composition_circuit",
    "encoder_source",
    "oracle",
}
_ALLOWED_KEYS = _REQUIRED_KEYS | {"notes"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_profile(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"vision profile must be a JSON object: {resolved}")
    missing = sorted(_REQUIRED_KEYS - payload.keys())
    unknown = sorted(payload.keys() - _ALLOWED_KEYS)
    if missing:
        raise ValueError(f"vision profile is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"vision profile has unknown fields: {', '.join(unknown)}")
    if payload["schema"] != PROFILE_SCHEMA or payload["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported vision profile schema: "
            f"{payload['schema']} v{payload['schema_version']}"
        )
    for key in ("id", "model_label", "chat_template"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"vision profile field {key!r} must be a non-empty string")
    composition = payload["composition_circuit"]
    if composition is not None and (not isinstance(composition, str) or not composition.strip()):
        raise ValueError("vision profile composition_circuit must be null or a non-empty string")
    if payload["encoder_source"] not in {"mmproj_gguf", "prebuilt_runtime"}:
        raise ValueError("vision profile encoder_source must be mmproj_gguf or prebuilt_runtime")
    oracle = payload["oracle"]
    if not isinstance(oracle, dict) or set(oracle) != {
        "backend",
        "decode_mode",
        "flash_attention",
        "prefix_source",
    }:
        raise ValueError(
            "vision profile oracle must contain only backend, decode_mode, "
            "flash_attention, and prefix_source"
        )
    if oracle["backend"] != "llama.cpp" or oracle["decode_mode"] != "batched":
        raise ValueError("this certification runner requires the llama.cpp batched oracle")
    if oracle["prefix_source"] != "cke":
        raise ValueError(
            "this runner compares decoders from a CKE-produced prefix; independent encoder "
            "parity requires a separate adapter"
        )
    if oracle["flash_attention"] not in {"disabled", "enabled", "auto"}:
        raise ValueError(
            "vision profile oracle flash_attention must be disabled, enabled, or auto"
        )
    result = dict(payload)
    result["path"] = str(resolved)
    result["sha256"] = _sha256_file(resolved)
    return result


def resolve_profile(profile_id: str, profile_file: Path | None = None) -> dict[str, Any]:
    path = profile_file if profile_file is not None else BUILTIN_PROFILES.get(profile_id)
    if path is None:
        available = ", ".join(sorted(BUILTIN_PROFILES))
        raise ValueError(f"unknown vision profile {profile_id!r}; built-ins: {available}")
    profile = load_profile(path)
    if profile_file is None and profile["id"] != profile_id:
        raise ValueError(
            f"built-in vision profile id mismatch: requested={profile_id} actual={profile['id']}"
        )
    if profile_file is not None and profile_id != "qwen3vl" and profile["id"] != profile_id:
        raise ValueError(
            f"vision profile id mismatch: requested={profile_id} actual={profile['id']}"
        )
    return profile


def apply_profile(args: Any) -> dict[str, Any]:
    profile = resolve_profile(
        str(getattr(args, "model_profile", "qwen3vl")),
        getattr(args, "profile_file", None),
    )
    source = "prebuilt_runtime" if getattr(args, "encoder_runtime", None) is not None else "mmproj_gguf"
    if source != profile["encoder_source"]:
        raise ValueError(
            f"vision profile {profile['id']} requires encoder_source={profile['encoder_source']}; "
            f"received {source}"
        )
    if getattr(args, "model_label", None) is None:
        args.model_label = profile["model_label"]
    if getattr(args, "chat_template", None) is None:
        args.chat_template = profile["chat_template"]
    if getattr(args, "composition_circuit", None) is None:
        args.composition_circuit = profile["composition_circuit"]
    args.model_profile = profile["id"]
    args.llama_flash_attention = profile["oracle"]["flash_attention"]
    args.profile_contract = profile
    return profile
