#!/usr/bin/env python3
from __future__ import annotations

"""Audit a safetensors index without downloading or opening weight shards."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAFETENSORS_CK_MAP = SCRIPT_DIR.parent / "model_maps" / "safetensors_ck_map.json"


# `layers.{L}` is the generic HF block name; RWKV-7 checkpoints instead use
# `rwkv7.blocks.{L}`, so the alternation keeps those tensors attributable to a
# layer. The leading `(?:^|\.)` boundary still prevents accidental matches
# inside other names (for example `model.visual.blocks.{L}`, which is a vision
# block rather than a decoder layer and is intentionally left unmatched).
LAYER_RE = re.compile(r"(?:^|\.)(?:layers|rwkv7\.blocks)\.(\d+)\.")
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


def _load_index(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "model.safetensors.index.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if "weight_map" not in data:
        raise SystemExit(f"{path} is not a safetensors index with weight_map")
    return data


def _family(name: str) -> str:
    if name in {
        "lm_head.weight",
        "head.weight",
        "language_model.lm_head.weight",
        "backbone.embeddings.weight",
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
    }:
        return name
    if name.startswith("rwkv7."):
        if name == "rwkv7.emb.weight":
            return "rwkv7_embedding"
        if name.startswith("rwkv7.ln_out."):
            return "rwkv7_final_norm"
        if ".att." in name:
            return "rwkv7_attention"
        if ".ffn." in name:
            return "rwkv7_channelmix"
        if ".ln0." in name or ".ln1." in name or ".ln2." in name:
            return "rwkv7_norm"
        return "rwkv7_other"
    if name.startswith("model.visual.patch_embed."):
        return "vision_patch_embed"
    if name.startswith("model.visual.pos_embed."):
        return "vision_position"
    if name.startswith("model.visual.blocks."):
        if ".attn." in name:
            return "vision_attention"
        if ".mlp." in name:
            return "vision_mlp"
        if ".norm" in name:
            return "vision_norm"
        return "vision_block_other"
    if name.startswith("model.visual.merger.") or name.startswith("model.visual.deepstack_merger_list."):
        return "multimodal_projector"
    if name.startswith("vision_tower."):
        return "vision_tower"
    if name.startswith("multi_modal_projector."):
        return "multimodal_projector"
    if ".mixer." in name:
        tail = name.split(".mixer.", 1)[1]
        if tail.startswith("self_attn.") or tail in {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"}:
            return "attention"
        if tail.startswith("experts."):
            return "moe_expert"
        if tail.startswith("shared_experts."):
            return "moe_shared_expert"
        if tail.startswith("gate."):
            return "moe_router"
        if tail in {"up_proj.weight", "down_proj.weight"}:
            return "dense_mlp"
        if tail.startswith(("in_proj.", "out_proj.", "conv1d.", "dt_bias", "A_log", "D", "norm.")):
            return "mamba"
        return "mixer_other"
    if ".self_attn." in name:
        tail = name.split(".self_attn.", 1)[1]
        if tail in {"q_proj.weight", "o_proj.weight", "rotary_emb.inv_freq"}:
            return "attention"
        if tail.startswith(("kv_a_proj_with_mqa.", "kv_a_layernorm.", "kv_b_proj.")):
            return "mla_attention"
        return "attention"
    if ".mlp.experts." in name:
        return "moe_expert"
    if ".mlp.shared_experts." in name:
        return "moe_shared_expert"
    if ".mlp.gate." in name:
        return "moe_router"
    if ".mlp." in name:
        return "dense_mlp"
    if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "model.norm.weight")):
        return "norm"
    return "other"



def _pattern_regex(pattern: str) -> re.Pattern[str]:
    sentinel_star = "<<<STAR>>>"
    text = re.escape(pattern.replace("*", sentinel_star))
    text = text.replace(r"\{L\}", r"(?P<L>\d+)")
    text = text.replace(r"\{E\}", r"(?P<E>\d+)")
    text = text.replace(sentinel_star, r".*")
    return re.compile(r"^" + text + r"$")


def _load_model_map(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"architectures": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {"architectures": {}}


def _arch_contract(model_map: dict[str, Any], arch: str | None) -> dict[str, Any]:
    if not arch:
        return {}
    contracts = model_map.get("architectures") if isinstance(model_map.get("architectures"), dict) else {}
    key = str(arch).lower()
    row = contracts.get(key)
    if isinstance(row, dict):
        return row
    for candidate, contract in contracts.items():
        if not isinstance(contract, dict):
            continue
        aliases = [str(candidate).lower()]
        aliases.extend(str(x).lower() for x in contract.get("aliases", []) if isinstance(x, str))
        if key in aliases:
            return contract
    return {}


def _validate_required_patterns(weight_names: list[str], contract: dict[str, Any]) -> dict[str, Any]:
    patterns = contract.get("required_tensor_patterns")
    if patterns is None:
        patterns = contract.get("required_tensor_families")
    if not isinstance(patterns, list):
        patterns = []
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw in patterns:
        pattern = str(raw)
        rx = _pattern_regex(pattern)
        matched = []
        layers: set[int] = set()
        experts: dict[int, set[int]] = defaultdict(set)
        for name in weight_names:
            m = rx.match(name)
            if not m:
                continue
            matched.append(name)
            gd = m.groupdict()
            if gd.get("L") is not None:
                layer = int(gd["L"])
                layers.add(layer)
                if gd.get("E") is not None:
                    experts[layer].add(int(gd["E"]))
        if not matched:
            missing.append(pattern)
        row: dict[str, Any] = {
            "pattern": pattern,
            "count": len(matched),
            "layers": sorted(layers),
        }
        if experts:
            row["expert_counts"] = {str(layer): len(ids) for layer, ids in sorted(experts.items())}
        rows.append(row)
    return {
        "status": "pass" if not missing else "fail",
        "required_count": len(rows),
        "missing_count": len(missing),
        "missing": missing,
        "patterns": rows,
    }


def _validate_declared_index_contract(
    *,
    tensor_count: int,
    shard_count: int,
    layers: dict[int, Counter[str]],
    expert_ids: dict[int, set[int]],
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    declared = contract.get("index_contract")
    if not isinstance(declared, dict):
        return None
    failures: list[str] = []

    def require_equal(key: str, actual: int) -> None:
        if key not in declared:
            return
        expected = int(declared[key])
        if actual != expected:
            failures.append(f"{key}: expected {expected}, observed {actual}")

    require_equal("tensor_count", tensor_count)
    require_equal("shard_count", shard_count)
    require_equal("layer_count", len(layers))

    dense_prefix = int(declared.get("dense_prefix_layers") or 0)
    experts_per_layer = int(declared.get("moe_experts_per_layer") or 0)
    if experts_per_layer > 0:
        for layer in sorted(layers):
            observed = expert_ids.get(layer, set())
            if layer < dense_prefix:
                if observed:
                    failures.append(
                        f"layer {layer}: dense prefix unexpectedly has {len(observed)} experts"
                    )
                continue
            expected_ids = set(range(experts_per_layer))
            if observed != expected_ids:
                missing = sorted(expected_ids - observed)
                extra = sorted(observed - expected_ids)
                failures.append(
                    f"layer {layer}: expected expert ids 0..{experts_per_layer - 1}; "
                    f"missing={missing} extra={extra}"
                )
    return {
        "status": "pass" if not failures else "fail",
        "declared": declared,
        "failures": failures,
    }

def audit_index(path: Path, arch: str | None = None, model_map_path: Path = DEFAULT_SAFETENSORS_CK_MAP) -> dict[str, Any]:
    data = _load_index(path)
    wm = data["weight_map"]
    weight_names = sorted(str(name) for name in wm)
    families = Counter(_family(name) for name in wm)
    layers: dict[int, Counter[str]] = defaultdict(Counter)
    expert_ids: dict[int, set[int]] = defaultdict(set)
    shards = Counter(wm.values())
    for name in wm:
        m = LAYER_RE.search(name)
        if not m:
            continue
        layer = int(m.group(1))
        fam = _family(name)
        layers[layer][fam] += 1
        em = EXPERT_RE.search(name)
        if em:
            expert_ids[layer].add(int(em.group(1)))
    layer_summary = {
        str(layer): {
            "families": dict(sorted(counter.items())),
            "expert_count": len(expert_ids.get(layer, set())),
        }
        for layer, counter in sorted(layers.items())
    }
    model_map = _load_model_map(model_map_path)
    contract = _arch_contract(model_map, arch)
    required = _validate_required_patterns(weight_names, contract) if contract else None
    declared_index = _validate_declared_index_contract(
        tensor_count=len(wm),
        shard_count=len(shards),
        layers=layers,
        expert_ids=expert_ids,
        contract=contract,
    ) if contract else None
    report = {
        "metadata": data.get("metadata", {}),
        "tensor_count": len(wm),
        "shard_count": len(shards),
        "shards": dict(sorted(shards.items())),
        "families": dict(sorted(families.items())),
        "layer_count": len(layers),
        "layers": layer_summary,
    }
    if arch:
        report["arch"] = str(arch)
        report["model_map_status"] = "known" if contract else "unknown"
        if required is not None:
            report["required_tensor_patterns"] = required
            report["status"] = required["status"]
            if declared_index is not None:
                report["declared_index_contract"] = declared_index
                if declared_index["status"] == "fail":
                    report["status"] = "fail"
        elif contract:
            report["status"] = "pass"
        else:
            report["status"] = "fail"
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit safetensors index tensor families without loading shards")
    ap.add_argument("index", type=Path, help="model.safetensors.index.json or directory")
    ap.add_argument("--arch", help="optional CK architecture key or alias to validate against model_maps/safetensors_ck_map.json")
    ap.add_argument("--model-map", type=Path, default=DEFAULT_SAFETENSORS_CK_MAP, help="safetensors CK model map JSON")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    report = audit_index(args.index, arch=args.arch, model_map_path=args.model_map)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report.get("status", "pass") != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
