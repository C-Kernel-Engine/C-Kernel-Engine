#!/usr/bin/env python3
"""Derive a serving-declaration inventory from circuits and referenced profiles.

This is a coverage ledger, not a registry or a claim that an artifact was executed.
Artifact and task evidence remain absent until a separately identified run supplies it.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V8 = ROOT / "version/v8"
REPORT = ROOT / "docs/site/site/serving-coverage.json"
PAGE = ROOT / "docs/site/_pages/v8-serving-coverage.html"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(circuit: dict, *, v8: Path) -> tuple[dict | None, str | None]:
    declaration = circuit.get("serving")
    if declaration is None:
        return None, None
    if not isinstance(declaration, dict) or declaration.get("schema") != "cke.circuit_serving.v1":
        raise ValueError(f"invalid serving declaration: {circuit.get('name')}")
    reference = declaration.get("profile_ref")
    if not isinstance(reference, str) or not reference.startswith("serving_profiles/"):
        raise ValueError(f"invalid serving profile reference: {circuit.get('name')}")
    path = (v8 / reference).resolve()
    if not path.is_relative_to((v8 / "serving_profiles").resolve()) or not path.is_file():
        raise ValueError(f"missing or escaping serving profile: {reference}")
    profile = json.loads(path.read_bytes())
    if profile.get("schema") != "cke.serving_profile.v1":
        raise ValueError(f"invalid serving profile: {reference}")
    variant = declaration.get("default_variant")
    if not isinstance(variant, str) or variant not in profile.get("variants", {}):
        raise ValueError(f"missing serving variant: {circuit.get('name')}")
    return profile, _digest(path)


def _chat_scope(circuit: dict, *, linked: bool) -> tuple[str, str]:
    """Classify only what the circuit structure establishes, never model names."""
    if linked:
        return "linked_chat_declaration", "circuit references a resolved serving profile"
    sequence = circuit.get("sequence") or []
    operations = {
        step if isinstance(step, str) else step.get("op")
        for step in sequence if isinstance(step, (str, dict))
    }
    contract = circuit.get("contract") or {}
    if ("audio_decoder" in contract or (operations and "decoder" not in operations)
            or (not operations and set(contract) <= {"vision_contract", "audio_frontend"}
                and bool(contract))):
        return "component", "no independent text-decoder serving path declared"
    if (operations == {"decoder"} and "tokenizer_contract" in contract
            and "logits_contract" in contract):
        return "text_decoder_candidate", "text decoder and token/logit contracts; serving unlinked"
    return "unassessed", "independent chat-serving path not established by circuit metadata"


def inventory(*, v8: Path = V8) -> dict:
    rows = []
    for path in sorted((v8 / "circuits").glob("*.json")):
        circuit = json.loads(path.read_bytes())
        if not isinstance(circuit, dict) or not isinstance(circuit.get("name"), str):
            raise ValueError(f"invalid circuit: {path.name}")
        profile, profile_hash = _profile(circuit, v8=v8)
        declaration = circuit.get("serving") or {}
        variant_name = declaration.get("default_variant")
        variant = profile["variants"][variant_name] if profile else None
        chat_scope, scope_reason = _chat_scope(circuit, linked=profile is not None)
        rows.append({
            "circuit": circuit["name"],
            "family": circuit.get("family"),
            "circuit_path": f"version/v8/circuits/{path.name}",
            "circuit_sha256": _digest(path),
            "serving_declaration": "circuit_linked" if profile else "missing",
            "chat_scope": chat_scope,
            "chat_scope_reason": scope_reason,
            "profile_ref": declaration.get("profile_ref"),
            "profile_sha256": profile_hash,
            "profile_id": profile.get("id") if profile else None,
            "variant": variant_name,
            "renderer": profile.get("renderer") if profile else None,
            "publisher_template": variant.get("chat") if variant else None,
            "tool_template": variant.get("tools") if variant else None,
            "output_protocol": variant.get("output_protocol") if variant else None,
            "reasoning_protocol": None,
            "stop_policy": None,
            "input_modalities": profile.get("input_modalities") if profile else None,
            "artifact": {"repository": None, "revision": None, "quantization": None,
                         "conversion_to_circuit": None,
                         "tokenizer_sha256": None, "resolved_bundle_identity": None},
            "evidence": {key: "not_assessed" for key in (
                "generated_runtime", "rendered_prompt", "exact_token_ids",
                "tool_protocol", "tool_round_trip", "harness_task",
                "cancellation_recovery", "loaded_artifact_identity")},
        })
    if not rows:
        raise ValueError("no v8 circuit templates found")
    return {
        "schema": "cke.serving_coverage_inventory.v1",
        "source": "version/v8/circuits/*.json and referenced serving profiles",
        "scope": "declarations_only; artifact and executed evidence are not inferred",
        "circuit_count": len(rows),
        "circuit_linked_count": sum(row["serving_declaration"] == "circuit_linked" for row in rows),
        "chat_scope_counts": {scope: sum(row["chat_scope"] == scope for row in rows)
                              for scope in ("linked_chat_declaration", "text_decoder_candidate",
                                            "component", "unassessed")},
        "rows": rows,
    }


def page(report: dict) -> str:
    body = []
    for row in report["rows"]:
        source = row["publisher_template"] or {}
        template = source.get("source", "unresolved")
        protocol = row["output_protocol"] or "unresolved"
        profile = row["profile_ref"] or "missing"
        body.append("<tr>" + "".join(
            f"<td><code>{html.escape(str(value))}</code></td>" for value in (
                row["circuit"], row["family"] or "unknown", row["chat_scope"], profile, template,
                protocol, ", ".join(row["input_modalities"] or []) or "unresolved",
            )) + "</tr>")
    return (f"""<h1>v8 Serving Coverage Inventory</h1>
<p>This page is derived from {report['circuit_count']} circuit templates and their referenced
serving profiles. {report['circuit_linked_count']} circuits declare a profile. A declaration
does not certify a model artifact, tokenizer, protocol, generated runtime, or harness task.</p>
<p>Chat scope is derived from circuit structure: a linked declaration, an unlinked text-decoder
candidate, a component without an independent text-decoder path, or unassessed. Component and
unassessed rows are not counted as chat-serving models. A text-decoder candidate still needs a
resolved bundle and executed serving evidence.</p>
<p>The <a href="site/serving-coverage.json">machine-readable inventory</a> records source hashes
and separate evidence fields. Repository revision, quantization, tokenizer identity, stop policy,
reasoning behavior, and task results remain unassessed until an exact resolved bundle and run
report supply them. Missing declarations are visible; legacy imported sidecars may still allow
chat but are not circuit-linked certification.</p>
<p>For the execution model behind these declarations — admission, lifecycle, the experimental
two-slot batch decode, and the continuous-batching roadmap — see
<a href="serving.html">Serving &amp; Batching</a>.</p>
<table class="table"><thead><tr><th>Circuit</th><th>Family</th><th>Chat scope</th><th>Profile</th>
<th>Chat asset source</th><th>Declared output protocol</th><th>Input modalities</th></tr></thead>
<tbody>
{chr(10).join(body)}
</tbody></table>
""")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Reject stale derived site assets")
    args = parser.parse_args()
    report = inventory()
    outputs = {REPORT: json.dumps(report, indent=2, sort_keys=True) + "\n", PAGE: page(report)}
    for path, content in outputs.items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"stale serving coverage inventory: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print(f"serving coverage: {report['circuit_linked_count']}/{report['circuit_count']} circuit profiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
