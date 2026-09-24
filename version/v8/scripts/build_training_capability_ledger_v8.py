#!/usr/bin/env python3
"""Derive a dtype/backward inventory from v8 kernel maps; never infer certification."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MAPS = ROOT / "version/v8/kernel_maps"
DEFAULT_OUT = ROOT / "docs/site/_pages/v8-training-capability-ledger.html"
QUANTIZED = re.compile(r"(?:^|[_-])(?:q[2-8](?:[_-]|$)|int[248](?:[_-]|$)|nf4(?:[_-]|$)|nvfp4(?:[_-]|$)|fp8(?:[_-]|$))")


def _family(document: dict[str, Any], path: Path) -> str:
    quant = document.get("quant") if isinstance(document.get("quant"), dict) else {}
    terms = "_".join(str(value).lower() for value in (
        path.stem, document.get("id", ""), document.get("variant", ""),
        quant.get("weight", ""), quant.get("activation", ""), quant.get("output", ""),
    ))
    if QUANTIZED.search(terms):
        return "quantized"
    if "bf16" in terms:
        return "bf16-tagged"
    if "fp32" in terms or "f32" in terms:
        return "fp32-tagged"
    return "unspecified"


def inventory() -> dict[str, Any]:
    providers: list[dict[str, Any]] = []
    source_hash = hashlib.sha256()
    for path in sorted(MAPS.glob("*.json")):
        if path.name in {"KERNEL_REGISTRY.json", "kernel_bindings.json"}:
            continue
        try:
            raw = path.read_bytes()
            document = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict) or not document.get("id") or not document.get("op"):
            continue
        source_hash.update(path.name.encode() + b"\0" + raw + b"\0")
        modes = document.get("modes") if isinstance(document.get("modes"), dict) else {}
        tests = document.get("tests") if isinstance(document.get("tests"), dict) else {}
        declared_backward = modes.get("backward") is True
        named_backward = "backward" in path.stem or "backward" in str(document["id"])
        providers.append({
            "id": str(document["id"]), "op": str(document["op"]),
            "map": str(path.relative_to(ROOT)), "family": _family(document, path),
            "quant": document.get("quant", {}),
            "declared_backward": declared_backward,
            "backward_named_without_mode": named_backward and not declared_backward,
            "declared_training_forward": modes.get("training") is True and not declared_backward,
            "registered_tests": bool(tests.get("parity") or tests.get("unit")),
        })
    return {"schema": "cke.v8.training_capability_ledger.v1",
            "source_sha256": source_hash.hexdigest(), "provider_count": len(providers),
            "execution_evidence": "NOT_MEASURED_BY_MAP_INVENTORY", "providers": providers}


def render_html(ledger: dict[str, Any]) -> str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ledger["providers"]:
        groups[(row["family"], row["op"])].append(row)
    lines = ["<!-- TITLE: v8 Training Kernel Capability Ledger -->",
             '<h1>v8 training kernel capability ledger</h1>',
             '<p>Generated from <code>version/v8/kernel_maps/*.json</code> by '
             '<code>version/v8/scripts/build_training_capability_ledger_v8.py</code>. '
             f'Source digest: <code>{ledger["source_sha256"]}</code>; provider maps: {ledger["provider_count"]}.</p>',
             '<p>This is a <strong>map inventory, not a training PASS</strong>. A backward declaration means a '
             'backward provider is registered; it does not prove matching geometry, selection by a generated '
             'circuit, or executed PyTorch parity. BF16-tagged includes storage or arithmetic tags. Quantized '
             'forward maps do not imply quantized-base gradients or QLoRA support. Direct executed evidence: '
             '<code>NOT_MEASURED_BY_MAP_INVENTORY</code>.</p>',
             '<table class="table"><thead><tr><th>Dtype family</th><th>Provider maps</th>'
             '<th>Forward or unspecified</th><th>Declared training forward</th><th>Declared backward</th>'
             '<th>Backward-named without mode</th><th>Maps with registered tests</th></tr></thead><tbody>']
    for family in ("fp32-tagged", "bf16-tagged", "quantized", "unspecified"):
        rows = [row for row in ledger["providers"] if row["family"] == family]
        values = (family, len(rows), sum(not r['declared_backward'] for r in rows),
                  sum(r['declared_training_forward'] for r in rows), sum(r['declared_backward'] for r in rows),
                  sum(r['backward_named_without_mode'] for r in rows), sum(r['registered_tests'] for r in rows))
        lines.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in values) + '</tr>')
    lines += ['</tbody></table>', '<h2>Operation inventory</h2>',
              '<table class="table"><thead><tr><th>Dtype family</th><th>Exact map operation</th><th>Maps</th>'
              '<th>Declared backward providers</th><th>Backward-named without mode</th></tr></thead><tbody>']
    for (family, op), rows in sorted(groups.items()):
        backward = ", ".join(row['id'] for row in rows if row["declared_backward"]) or "—"
        missing_mode = ", ".join(row['id'] for row in rows if row["backward_named_without_mode"]) or "—"
        values = (family, op, len(rows), backward, missing_mode)
        lines.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in values) + '</tr>')
    lines += ['</tbody></table>',
              f'<details><summary>All {ledger["provider_count"]} provider maps</summary>',
              '<table class="table"><thead><tr><th>Map</th><th>Provider ID</th><th>Operation</th>'
              '<th>Dtype family</th><th>Backward mode</th><th>Registered test</th></tr></thead><tbody>']
    for row in ledger['providers']:
        map_path = html.escape(row['map'])
        map_link = f'<a href="https://github.com/C-Kernel-Engine/C-Kernel-Engine/blob/main/{map_path}">{map_path}</a>'
        values = (row['id'], row['op'], row['family'],
                  'declared' if row['declared_backward'] else 'missing mode' if row['backward_named_without_mode'] else 'not declared',
                  'registered' if row['registered_tests'] else 'not registered')
        lines.append('<tr><td>' + map_link + '</td>' +
                     ''.join(f'<td>{html.escape(str(value))}</td>' for value in values) + '</tr>')
    lines += ['</tbody></table></details>', '<p>Regenerate with '
              '<code>python3 version/v8/scripts/build_training_capability_ledger_v8.py</code>; use '
              '<code>--check</code> to detect a stale snapshot. Exact provider maps retain dtype, layout, '
              'scratch, implementation, and registered-test details.</p>', '']
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    ledger = inventory()
    rendered = render_html(ledger)
    if args.check:
        return 0 if args.out.read_text(encoding="utf-8") == rendered else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} from {ledger['provider_count']} kernel maps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
