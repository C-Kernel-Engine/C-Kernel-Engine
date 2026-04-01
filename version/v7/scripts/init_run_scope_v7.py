#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPEC_RUNG_RE = re.compile(r"(?P<spec>spec\d+[a-z]?).*?(?P<rung>r\d+)\b", re.IGNORECASE)
SPEC_ONLY_RE = re.compile(r"(?P<spec>spec\d+[a-z]?)", re.IGNORECASE)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _read_text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _infer_spec_rung(run_name: str) -> tuple[str | None, str | None]:
    m = SPEC_RUNG_RE.search(run_name)
    if m:
        return m.group("spec").lower(), m.group("rung").lower()
    m = SPEC_ONLY_RE.search(run_name)
    if m:
        return m.group("spec").lower(), None
    return None, None


def _merge_list(existing: Any, incoming: list[str]) -> list[str]:
    if incoming:
        return incoming
    if isinstance(existing, list):
        return [str(item) for item in existing if str(item).strip()]
    return []


def _select_list(*candidates: Any) -> list[str]:
    for candidate in candidates:
        if isinstance(candidate, list):
            values = [str(item).strip() for item in candidate if str(item).strip()]
            if values:
                return values
    return []


def _normalize_path_list(*candidates: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        for item in candidate:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


def _normalize_text_list(*candidates: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        for item in candidate:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value.strip()
            continue
        return value
    return None


def _normalize_scope_payload(
    *,
    run_dir: Path,
    existing_scope: dict[str, Any] | None,
    template_scope: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    run_name = run_dir.name
    inferred_spec, inferred_rung = _infer_spec_rung(run_name)

    source_scope = existing_scope if isinstance(existing_scope, dict) else {}
    template_scope = template_scope if isinstance(template_scope, dict) else {}

    created_at = _coalesce(
        source_scope.get("created_at"),
        template_scope.get("created_at"),
        now_iso,
    )

    notes_text = _coalesce(
        _read_text(Path(args.notes_file).expanduser().resolve()) if args.notes_file else None,
        args.notes,
        template_scope.get("notes"),
        source_scope.get("notes"),
    )

    title = _coalesce(
        args.title,
        template_scope.get("title"),
        source_scope.get("title"),
    )
    spec = _coalesce(args.spec, template_scope.get("spec"), source_scope.get("spec"), inferred_spec)
    rung = _coalesce(args.rung, template_scope.get("rung"), source_scope.get("rung"), inferred_rung)
    family = _coalesce(args.family, template_scope.get("family"), source_scope.get("family"))
    objective = _coalesce(args.objective, template_scope.get("objective"), source_scope.get("objective"))
    hypothesis = _coalesce(args.hypothesis, template_scope.get("hypothesis"), source_scope.get("hypothesis"))
    prompt_contract = _coalesce(
        args.prompt_contract,
        template_scope.get("prompt_contract"),
        source_scope.get("prompt_contract"),
    )
    output_contract = _coalesce(
        args.output_contract,
        template_scope.get("output_contract"),
        source_scope.get("output_contract"),
    )

    if not title:
        parts = [part for part in (spec, rung) if part]
        title = " ".join(parts) if parts else run_name

    payload = {
        "schema": "ck.run_scope.v1",
        "created_at": created_at,
        "updated_at": now_iso,
        "run_dir": str(run_dir),
        "run_name": run_name,
        "spec": spec,
        "rung": rung,
        "family": family,
        "title": title,
        "objective": objective,
        "hypothesis": hypothesis,
        "prompt_contract": prompt_contract,
        "output_contract": output_contract,
        "in_scope": _select_list(args.in_scope, template_scope.get("in_scope"), source_scope.get("in_scope")),
        "out_of_scope": _select_list(args.out_of_scope, template_scope.get("out_of_scope"), source_scope.get("out_of_scope")),
        "success_gates": _select_list(args.success_gate, template_scope.get("success_gates"), source_scope.get("success_gates")),
        "guardrails": _select_list(args.guardrail, template_scope.get("guardrails"), source_scope.get("guardrails")),
        "follow_ups": _select_list(args.follow_up, template_scope.get("follow_ups"), source_scope.get("follow_ups")),
        "research_priors": _normalize_text_list(
            args.research_prior,
            template_scope.get("research_priors"),
            source_scope.get("research_priors"),
        ),
        "lessons_learned": _normalize_text_list(
            args.lesson_learned,
            template_scope.get("lessons_learned"),
            source_scope.get("lessons_learned"),
        ),
        "read_first": _normalize_path_list(args.read_first, template_scope.get("read_first"), source_scope.get("read_first")),
        "context_files": _normalize_path_list(args.context_file, template_scope.get("context_files"), source_scope.get("context_files")),
        "notes": notes_text,
    }
    return payload


def _render_scope_markdown(scope: dict[str, Any]) -> str:
    def _section(title: str, values: list[str]) -> str:
        if not values:
            return ""
        body = "\n".join(f"- {item}" for item in values)
        return f"## {title}\n{body}\n\n"

    lines = [
        "# Run Scope",
        "",
        f"- title: {scope.get('title') or '-'}",
        f"- spec: {scope.get('spec') or '-'}",
        f"- rung: {scope.get('rung') or '-'}",
        f"- family: {scope.get('family') or '-'}",
        f"- run_name: {scope.get('run_name') or '-'}",
        "",
    ]
    if scope.get("objective"):
        lines.extend(["## Objective", str(scope["objective"]), ""])
    if scope.get("hypothesis"):
        lines.extend(["## Hypothesis", str(scope["hypothesis"]), ""])
    if scope.get("prompt_contract"):
        lines.extend(["## Prompt Contract", str(scope["prompt_contract"]), ""])
    if scope.get("output_contract"):
        lines.extend(["## Output Contract", str(scope["output_contract"]), ""])
    if scope.get("read_first"):
        lines.extend(["## Read First", *[f"- {item}" for item in scope["read_first"]], ""])
    if scope.get("local_context_files"):
        lines.extend(["## Local Context Copies", *[f"- {item}" for item in scope["local_context_files"]], ""])
    lines.append(_section("In Scope", scope.get("in_scope") or []))
    lines.append(_section("Out Of Scope", scope.get("out_of_scope") or []))
    lines.append(_section("Success Gates", scope.get("success_gates") or []))
    lines.append(_section("Guardrails", scope.get("guardrails") or []))
    lines.append(_section("Research Priors", scope.get("research_priors") or []))
    lines.append(_section("Lessons Learned", scope.get("lessons_learned") or []))
    lines.append(_section("Follow Ups", scope.get("follow_ups") or []))
    if scope.get("notes"):
        lines.extend(["## Notes", str(scope["notes"]), ""])
    return "\n".join(part for part in lines if part is not None).rstrip() + "\n"


def _render_agent_markdown(scope: dict[str, Any]) -> str:
    lines = [
        "# Agent Brief",
        "",
        "Read this before interpreting the run artifacts.",
        "",
        f"- run_name: {scope.get('run_name') or '-'}",
        f"- spec: {scope.get('spec') or '-'}",
        f"- rung: {scope.get('rung') or '-'}",
        f"- family: {scope.get('family') or '-'}",
        "",
    ]
    if scope.get("objective"):
        lines.extend(["## Objective", str(scope["objective"]), ""])
    if scope.get("hypothesis"):
        lines.extend(["## Hypothesis", str(scope["hypothesis"]), ""])
    if scope.get("prompt_contract"):
        lines.extend(["## Model-Facing Prompt Contract", str(scope["prompt_contract"]), ""])
    if scope.get("output_contract"):
        lines.extend(["## Output Contract", str(scope["output_contract"]), ""])
    if scope.get("read_first"):
        lines.extend(["## Read First", *[f"- {item}" for item in scope["read_first"]], ""])
    if scope.get("local_context_files"):
        lines.extend(["## Local Context Copies", *[f"- {item}" for item in scope["local_context_files"]], ""])
    if scope.get("in_scope"):
        lines.extend(["## In Scope", *[f"- {item}" for item in scope["in_scope"]], ""])
    if scope.get("out_of_scope"):
        lines.extend(["## Out Of Scope", *[f"- {item}" for item in scope["out_of_scope"]], ""])
    if scope.get("guardrails"):
        lines.extend(["## Guardrails", *[f"- {item}" for item in scope["guardrails"]], ""])
    if scope.get("success_gates"):
        lines.extend(["## Success Gates", *[f"- {item}" for item in scope["success_gates"]], ""])
    if scope.get("research_priors"):
        lines.extend(["## Research Priors", *[f"- {item}" for item in scope["research_priors"]], ""])
    if scope.get("lessons_learned"):
        lines.extend(["## Lessons Learned", *[f"- {item}" for item in scope["lessons_learned"]], ""])
    return "\n".join(lines).rstrip() + "\n"


def _render_training_markdown(scope: dict[str, Any]) -> str:
    lines = [
        "# Training Brief",
        "",
        f"- title: {scope.get('title') or scope.get('run_name') or '-'}",
        f"- run_name: {scope.get('run_name') or '-'}",
        f"- spec: {scope.get('spec') or '-'}",
        f"- rung: {scope.get('rung') or '-'}",
        "",
        "This file is the operator-facing summary for this rung.",
        "",
    ]
    if scope.get("objective"):
        lines.extend(["## Goal", str(scope["objective"]), ""])
    if scope.get("hypothesis"):
        lines.extend(["## Working Hypothesis", str(scope["hypothesis"]), ""])
    if scope.get("success_gates"):
        lines.extend(["## Success Gates", *[f"- {item}" for item in scope["success_gates"]], ""])
    if scope.get("research_priors"):
        lines.extend(["## Research Priors", *[f"- {item}" for item in scope["research_priors"]], ""])
    if scope.get("lessons_learned"):
        lines.extend(["## Lessons Learned", *[f"- {item}" for item in scope["lessons_learned"]], ""])
    if scope.get("read_first"):
        lines.extend(["## Read First", *[f"- {item}" for item in scope["read_first"]], ""])
    if scope.get("local_context_files"):
        lines.extend(["## Local Context Copies", *[f"- {item}" for item in scope["local_context_files"]], ""])
    if scope.get("follow_ups"):
        lines.extend(["## Follow Ups", *[f"- {item}" for item in scope["follow_ups"]], ""])
    if scope.get("notes"):
        lines.extend(["## Notes", str(scope["notes"]), ""])
    return "\n".join(lines).rstrip() + "\n"


def _copy_context_files(run_dir: Path, scope: dict[str, Any]) -> list[str]:
    requested = scope.get("context_files")
    if not isinstance(requested, list) or not requested:
        return []
    context_dir = run_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for raw in requested:
        src = Path(str(raw)).expanduser().resolve()
        if not src.exists() or not src.is_file():
            continue
        dest = context_dir / src.name
        if dest.resolve() != src:
            shutil.copy2(src, dest)
        copied.append(str(dest))
    return copied


def _update_training_plan(run_dir: Path, scope: dict[str, Any]) -> None:
    plan_path = run_dir / "training_plan.json"
    plan = _load_json(plan_path) or {"schema": "ck.training_plan.v1", "run_dir": str(run_dir)}
    plan["run_scope"] = scope
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Create or update run_scope artifacts for a training run.")
    ap.add_argument("--run", required=True, help="Run directory")
    ap.add_argument("--from-json", help="Optional JSON template to merge before flag overrides")
    ap.add_argument("--spec")
    ap.add_argument("--rung")
    ap.add_argument("--family")
    ap.add_argument("--title")
    ap.add_argument("--objective")
    ap.add_argument("--hypothesis")
    ap.add_argument("--prompt-contract")
    ap.add_argument("--output-contract")
    ap.add_argument("--notes")
    ap.add_argument("--notes-file")
    ap.add_argument("--in-scope", action="append", default=[])
    ap.add_argument("--out-of-scope", action="append", default=[])
    ap.add_argument("--success-gate", action="append", default=[])
    ap.add_argument("--guardrail", action="append", default=[])
    ap.add_argument("--follow-up", action="append", default=[])
    ap.add_argument("--research-prior", action="append", default=[])
    ap.add_argument("--lesson-learned", action="append", default=[])
    ap.add_argument("--read-first", action="append", default=[])
    ap.add_argument("--context-file", action="append", default=[])
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    template_scope = _load_json(Path(args.from_json).expanduser().resolve()) if args.from_json else None
    existing_scope = _load_json(run_dir / "run_scope.json")
    if existing_scope is None:
        existing_plan = _load_json(run_dir / "training_plan.json")
        if isinstance(existing_plan, dict) and isinstance(existing_plan.get("run_scope"), dict):
            existing_scope = existing_plan.get("run_scope")

    scope = _normalize_scope_payload(
        run_dir=run_dir,
        existing_scope=existing_scope,
        template_scope=template_scope,
        args=args,
    )
    scope["local_context_files"] = _copy_context_files(run_dir, scope)

    (run_dir / "run_scope.json").write_text(json.dumps(scope, indent=2), encoding="utf-8")
    (run_dir / "run_scope.md").write_text(_render_scope_markdown(scope), encoding="utf-8")
    (run_dir / "agent.md").write_text(_render_agent_markdown(scope), encoding="utf-8")
    (run_dir / "training.md").write_text(_render_training_markdown(scope), encoding="utf-8")
    _update_training_plan(run_dir, scope)

    print(f"[run-scope] wrote {run_dir / 'run_scope.json'}")
    print(f"[run-scope] wrote {run_dir / 'run_scope.md'}")
    print(f"[run-scope] wrote {run_dir / 'agent.md'}")
    print(f"[run-scope] wrote {run_dir / 'training.md'}")
    if scope.get("local_context_files"):
        print(f"[run-scope] copied {len(scope['local_context_files'])} context file(s) into {run_dir / 'context'}")
    print(f"[run-scope] synced {run_dir / 'training_plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
