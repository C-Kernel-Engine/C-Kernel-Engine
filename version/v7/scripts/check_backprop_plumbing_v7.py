#!/usr/bin/env python3
"""
check_backprop_plumbing_v7.py

Static backprop-plumbing audit for v7 training artifacts.

Checks:
- IR2 graph/dataflow integrity (op refs, tensor refs, producer refs)
- trainable weight -> grad.weight coverage + writer coverage
- per-layer forward/backward/gradient flow coverage
- layout coverage + optimizer state coverage + overlap/bounds sanity
- manifest/IR dim agreement
- weight tying status (token_emb vs output.weight)
- optional runtime stitch smoke summary from train_e2e report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object at {path}, got {type(obj)}")
    return obj


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _shape_numel(shape: Any) -> Optional[int]:
    if not isinstance(shape, list) or not shape:
        return None
    n = 1
    for d in shape:
        if not isinstance(d, int) or d <= 0:
            return None
        n *= d
    return int(n)


def _tensor_numel(meta: Dict[str, Any]) -> Optional[int]:
    n = meta.get("numel")
    if isinstance(n, int) and n > 0:
        return int(n)
    return _shape_numel(meta.get("shape"))


def _cfg_int(cfg: Dict[str, Any], keys: Iterable[str]) -> Optional[int]:
    for key in keys:
        if key not in cfg:
            continue
        try:
            v = int(cfg.get(key))
        except Exception:
            continue
        if v > 0:
            return v
    return None


def _extract_layer(name: str) -> Optional[int]:
    # Accept IDs like:
    # - weight.layer.22.w1
    # - grad.weight.layer.22.w1
    # - act.L22.residual_add.1.out
    if ".layer." in name:
        marker = ".layer."
        s = name.split(marker, 1)[1]
        num = s.split(".", 1)[0]
        if num.isdigit():
            return int(num)
    if ".L" in name:
        # Example: act.L22.foo
        idx = name.find(".L")
        rest = name[idx + 2 :]
        num = ""
        for ch in rest:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            return int(num)
    return None


def _iter_op_slots(op: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    dataflow = op.get("dataflow")
    if not isinstance(dataflow, dict):
        return
    for side in ("inputs", "outputs"):
        slots = dataflow.get(side)
        if not isinstance(slots, dict):
            continue
        for slot_name, slot in slots.items():
            if isinstance(slot_name, str) and isinstance(slot, dict):
                yield side, slot_name, slot


def _kind_compatible(slot_kind: Any, tensor_kind: Any) -> bool:
    if slot_kind == tensor_kind:
        return True
    if not isinstance(slot_kind, str) or not isinstance(tensor_kind, str):
        return False
    sk = slot_kind.strip().lower()
    tk = tensor_kind.strip().lower()
    if sk == tk:
        return True
    # grad_accumulate op uses abstract slot kinds while tensor registry is concrete.
    if sk == "grad_accumulator" and tk.startswith("grad_"):
        return True
    if sk == "tmp_grad" and tk.startswith("tmp_grad"):
        return True
    if sk == "saved" and tk.startswith("saved"):
        return True
    if sk == "activation" and tk in {"activation", "saved_activation"}:
        return True
    return False


def _collect_region_bytes(layout: Dict[str, Any]) -> Dict[str, int]:
    regions = layout.get("regions")
    if isinstance(regions, list) and regions:
        out: Dict[str, int] = {}
        for row in regions:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", ""))
            b = row.get("bytes")
            if not name:
                continue
            if isinstance(b, (int, float)):
                out[name] = int(b)
        if out:
            return out
    # Fallback: aggregate by tensor.region.
    out: Dict[str, int] = {}
    for row in layout.get("tensors", []) or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("region", "unknown"))
        b = row.get("bytes")
        if isinstance(b, (int, float)):
            out[name] = out.get(name, 0) + int(b)
    return out


def _layer_layout_bytes(layout_tensors: List[Dict[str, Any]]) -> Dict[int, Dict[str, int]]:
    out: Dict[int, Dict[str, int]] = {}
    for row in layout_tensors:
        tid = str(row.get("id", ""))
        if not tid:
            continue
        layer = _extract_layer(tid)
        if layer is None:
            continue
        region = str(row.get("region", "unknown"))
        b = int(row.get("bytes", 0) or 0)
        if layer not in out:
            out[layer] = {}
        out[layer][region] = out[layer].get(region, 0) + b
    return out


def _first_checked_row(train_report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rows = train_report.get("parity_steps")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and bool(row.get("checked")):
            return row
    return None


def _manifest_tying(manifest: Dict[str, Any]) -> Dict[str, Any]:
    cfg = manifest.get("config")
    cfg = cfg if isinstance(cfg, dict) else {}

    entries = manifest.get("entries")
    if not isinstance(entries, list):
        entries = []
    by_name: Dict[str, Dict[str, Any]] = {}
    for row in entries:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if isinstance(name, str) and name:
            by_name[name] = row

    tok = by_name.get("token_emb")
    out = by_name.get("output.weight")
    tie_cfg = cfg.get("tie_word_embeddings")

    actual_tied = False
    reason = "indeterminate"
    if tok is not None and out is None:
        actual_tied = True
        reason = "output.weight_missing"
    elif tok is not None and out is not None:
        tok_off_raw = tok.get("offset")
        out_off_raw = out.get("offset")
        tok_sz_raw = tok.get("size")
        out_sz_raw = out.get("size")
        tok_off = int(tok_off_raw) if isinstance(tok_off_raw, (int, float)) else -1
        out_off = int(out_off_raw) if isinstance(out_off_raw, (int, float)) else -2
        tok_sz = int(tok_sz_raw) if isinstance(tok_sz_raw, (int, float)) else -1
        out_sz = int(out_sz_raw) if isinstance(out_sz_raw, (int, float)) else -2
        same_off = (tok_off == out_off)
        same_sz = (tok_sz == out_sz)
        actual_tied = bool(same_off and same_sz)
        reason = "same_offset_and_size" if actual_tied else "distinct_manifest_entries"

    tie_cfg_bool = bool(tie_cfg) if isinstance(tie_cfg, bool) else None
    config_match = True
    if tie_cfg_bool is not None:
        config_match = (tie_cfg_bool == actual_tied)

    return {
        "config_tie_word_embeddings": tie_cfg_bool,
        "actual_tied": bool(actual_tied),
        "reason": reason,
        "config_matches_manifest": bool(config_match),
        "has_token_emb": tok is not None,
        "has_output_weight": out is not None,
        "token_emb_offset": (
            int(tok.get("offset")) if (tok is not None and isinstance(tok.get("offset"), (int, float))) else None
        ),
        "output_weight_offset": (
            int(out.get("offset")) if (out is not None and isinstance(out.get("offset"), (int, float))) else None
        ),
        "token_emb_size": (
            int(tok.get("size")) if (tok is not None and isinstance(tok.get("size"), (int, float))) else None
        ),
        "output_weight_size": (
            int(out.get("size")) if (out is not None and isinstance(out.get("size"), (int, float))) else None
        ),
    }


def audit_backprop_plumbing(
    *,
    ir2: Dict[str, Any],
    layout: Dict[str, Any],
    manifest: Dict[str, Any],
    runtime_report: Optional[Dict[str, Any]],
    runtime_summary: Optional[Dict[str, Any]],
    strict: bool,
    max_samples: int,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    failures: List[str] = []
    warnings: List[str] = []

    def add_check(name: str, passed: bool, detail: str, **extra: Any) -> None:
        row = {"name": name, "passed": bool(passed), "detail": detail}
        row.update(extra)
        checks.append(row)
        if not passed:
            failures.append(f"{name}: {detail}")

    tensors = ir2.get("tensors")
    if not isinstance(tensors, dict):
        raise RuntimeError("IR2 missing `tensors` dict")

    forward = [op for op in (ir2.get("forward") or []) if isinstance(op, dict)]
    backward = [op for op in (ir2.get("backward") or []) if isinstance(op, dict)]
    all_ops = forward + backward

    ir2_stats = ir2.get("stats")
    ir2_stats = ir2_stats if isinstance(ir2_stats, dict) else {}
    unresolved_count = int(ir2_stats.get("unresolved", len(ir2.get("unresolved") or [])) or 0)
    warnings_count = int(ir2_stats.get("warnings", len(ir2.get("warnings") or [])) or 0)
    issues_count = int(ir2_stats.get("issues", len(ir2.get("issues") or [])) or 0)
    hard_clean = (unresolved_count == 0 and issues_count == 0)
    if strict:
        hard_clean = hard_clean and (warnings_count == 0)
    add_check(
        "ir2_clean",
        hard_clean,
        f"unresolved={unresolved_count} warnings={warnings_count} issues={issues_count} strict={strict}",
    )

    # op_id integrity
    op_ids: Dict[int, Dict[str, Any]] = {}
    duplicate_ids: List[int] = []
    missing_op_id = 0
    for op in all_ops:
        oid = op.get("op_id")
        if not isinstance(oid, int):
            missing_op_id += 1
            continue
        if oid in op_ids:
            duplicate_ids.append(oid)
            continue
        op_ids[oid] = op
    add_check(
        "op_id_integrity",
        (missing_op_id == 0 and len(duplicate_ids) == 0),
        f"missing={missing_op_id} duplicate={len(duplicate_ids)}",
        duplicate_ids=sorted(set(duplicate_ids))[:max_samples],
    )

    fwd_ids: Set[int] = {int(op["op_id"]) for op in forward if isinstance(op.get("op_id"), int)}
    bad_forward_ref: List[Dict[str, Any]] = []
    for op in backward:
        ref = op.get("forward_ref")
        if ref is None:
            continue
        if not isinstance(ref, int) or ref not in fwd_ids:
            bad_forward_ref.append({"op_id": op.get("op_id"), "forward_ref": ref, "op": op.get("op")})
    add_check(
        "backward_forward_ref_integrity",
        len(bad_forward_ref) == 0,
        f"bad_refs={len(bad_forward_ref)}",
        samples=bad_forward_ref[:max_samples],
    )

    # dataflow tensor + metadata consistency
    missing_tensor_refs: List[Dict[str, Any]] = []
    slot_meta_mismatch: List[Dict[str, Any]] = []
    for op in all_ops:
        oid = op.get("op_id")
        opname = op.get("op")
        for side, slot_name, slot in _iter_op_slots(op):
            tid = slot.get("tensor")
            if not isinstance(tid, str) or not tid:
                continue
            meta = tensors.get(tid)
            if not isinstance(meta, dict):
                missing_tensor_refs.append(
                    {"op_id": oid, "op": opname, "side": side, "slot": slot_name, "tensor": tid}
                )
                continue
            for field in ("dtype", "kind", "shape", "numel"):
                sval = slot.get(field)
                mval = meta.get(field)
                if sval is None or mval is None:
                    continue
                if field == "kind" and _kind_compatible(sval, mval):
                    continue
                if sval != mval:
                    slot_meta_mismatch.append(
                        {
                            "op_id": oid,
                            "op": opname,
                            "side": side,
                            "slot": slot_name,
                            "tensor": tid,
                            "field": field,
                            "slot_value": sval,
                            "tensor_value": mval,
                        }
                    )
    add_check(
        "dataflow_tensor_registry_coverage",
        len(missing_tensor_refs) == 0,
        f"missing_refs={len(missing_tensor_refs)}",
        samples=missing_tensor_refs[:max_samples],
    )
    add_check(
        "dataflow_slot_metadata_consistency",
        len(slot_meta_mismatch) == 0,
        f"mismatches={len(slot_meta_mismatch)}",
        samples=slot_meta_mismatch[:max_samples],
    )

    # save_for_backward contract integrity
    forward_by_id: Dict[int, Dict[str, Any]] = {
        int(op["op_id"]): op for op in forward if isinstance(op.get("op_id"), int)
    }
    bad_sfb_refs: List[Dict[str, Any]] = []
    bad_sfb_saved_producer: List[Dict[str, Any]] = []
    sfb_saved_by_forward: Dict[int, Set[str]] = {}
    all_saved_tensors: Set[str] = set()
    for op in forward:
        oid_raw = op.get("op_id")
        sfb = op.get("save_for_backward")
        if not isinstance(oid_raw, int) or not isinstance(sfb, dict):
            continue
        oid = int(oid_raw)
        for key, ref in sfb.items():
            if not isinstance(ref, dict):
                bad_sfb_refs.append(
                    {"op_id": oid, "op": op.get("op"), "save_key": key, "reason": "non_dict_ref"}
                )
                continue
            tid = ref.get("tensor")
            if not isinstance(tid, str) or not tid:
                bad_sfb_refs.append(
                    {"op_id": oid, "op": op.get("op"), "save_key": key, "reason": "missing_tensor"}
                )
                continue
            tmeta = tensors.get(tid)
            if not isinstance(tmeta, dict):
                bad_sfb_refs.append(
                    {"op_id": oid, "op": op.get("op"), "save_key": key, "tensor": tid, "reason": "tensor_not_in_registry"}
                )
                continue
            if not _kind_compatible(ref.get("kind"), tmeta.get("kind")):
                bad_sfb_refs.append(
                    {
                        "op_id": oid,
                        "op": op.get("op"),
                        "save_key": key,
                        "tensor": tid,
                        "reason": "kind_mismatch",
                        "save_kind": ref.get("kind"),
                        "tensor_kind": tmeta.get("kind"),
                    }
                )
            is_saved = str(ref.get("kind", "")).startswith("saved") or tid.startswith("saved.")
            if not is_saved:
                continue
            all_saved_tensors.add(tid)
            sfb_saved_by_forward.setdefault(oid, set()).add(tid)
            prod = tmeta.get("producer")
            if not isinstance(prod, dict) or int(prod.get("op_id", -1)) != oid:
                bad_sfb_saved_producer.append(
                    {
                        "op_id": oid,
                        "op": op.get("op"),
                        "save_key": key,
                        "tensor": tid,
                        "producer": prod,
                        "reason": "saved_tensor_not_produced_by_forward_op",
                    }
                )

    add_check(
        "save_for_backward_tensor_integrity",
        len(bad_sfb_refs) == 0,
        f"bad_refs={len(bad_sfb_refs)}",
        samples=bad_sfb_refs[:max_samples],
    )
    add_check(
        "save_for_backward_saved_tensor_producer_integrity",
        len(bad_sfb_saved_producer) == 0,
        f"bad_saved_producers={len(bad_sfb_saved_producer)}",
        samples=bad_sfb_saved_producer[:max_samples],
    )

    # producer references
    bad_producers: List[Dict[str, Any]] = []
    for tid, meta in tensors.items():
        if not isinstance(tid, str) or not isinstance(meta, dict):
            continue
        prod = meta.get("producer")
        if not isinstance(prod, dict):
            continue
        pid = prod.get("op_id")
        out_name = prod.get("output_name")
        if not isinstance(pid, int) or pid not in op_ids:
            bad_producers.append({"tensor": tid, "producer": prod, "reason": "missing_op"})
            continue
        op = op_ids[pid]
        kind = str(meta.get("kind", ""))
        if kind.startswith("saved"):
            # Saved tensors may be materialized by kernel internals and not appear
            # in forward dataflow outputs.
            continue
        outputs = op.get("dataflow", {}).get("outputs")
        outputs = outputs if isinstance(outputs, dict) else {}
        if not isinstance(out_name, str) or out_name not in outputs:
            bad_producers.append({"tensor": tid, "producer": prod, "reason": "missing_output_name"})
            continue
        slot = outputs.get(out_name)
        stid = slot.get("tensor") if isinstance(slot, dict) else None
        if stid != tid:
            bad_producers.append(
                {"tensor": tid, "producer": prod, "reason": "producer_output_tensor_mismatch", "slot_tensor": stid}
            )
    add_check(
        "tensor_producer_integrity",
        len(bad_producers) == 0,
        f"bad_producers={len(bad_producers)}",
        samples=bad_producers[:max_samples],
    )

    # weight/grad coverage
    trainable_weights = sorted(
        tid
        for tid, meta in tensors.items()
        if isinstance(tid, str)
        and tid.startswith("weight.")
        and isinstance(meta, dict)
        and bool(meta.get("requires_grad"))
    )
    grad_weight_tensors = {
        tid for tid, meta in tensors.items() if isinstance(tid, str) and tid.startswith("grad.weight.") and isinstance(meta, dict)
    }

    grad_summary = ir2.get("gradient_summary")
    grad_summary = grad_summary if isinstance(grad_summary, dict) else {}
    grad_summary_tensors = set(
        t for t in (grad_summary.get("grad_weight_tensors") or []) if isinstance(t, str) and t.startswith("grad.weight.")
    )
    grad_writers = grad_summary.get("grad_weight_writers")
    grad_writers = grad_writers if isinstance(grad_writers, dict) else {}

    missing_grad_tensor: List[str] = []
    missing_summary_tensor: List[str] = []
    missing_writer: List[str] = []
    for wtid in trainable_weights:
        pname = wtid[len("weight.") :]
        gtid = f"grad.weight.{pname}"
        if gtid not in grad_weight_tensors:
            missing_grad_tensor.append(gtid)
        if gtid not in grad_summary_tensors:
            missing_summary_tensor.append(gtid)
        writers_for = grad_writers.get(gtid)
        if not isinstance(writers_for, list) or len(writers_for) == 0:
            missing_writer.append(gtid)

    add_check(
        "trainable_weight_grad_tensor_coverage",
        len(missing_grad_tensor) == 0,
        f"trainable={len(trainable_weights)} missing_grad_tensors={len(missing_grad_tensor)}",
        samples=missing_grad_tensor[:max_samples],
    )
    add_check(
        "gradient_summary_grad_weight_coverage",
        len(missing_summary_tensor) == 0,
        f"missing_grad_weight_summary={len(missing_summary_tensor)}",
        samples=missing_summary_tensor[:max_samples],
    )
    add_check(
        "gradient_writer_coverage",
        len(missing_writer) == 0,
        f"missing_writers={len(missing_writer)}",
        samples=missing_writer[:max_samples],
    )

    # all trainable weights should be used by at least one op weight slot
    used_weights: Set[str] = set()
    for op in forward:
        wmap = op.get("weights")
        if not isinstance(wmap, dict):
            continue
        for _, w in wmap.items():
            if not isinstance(w, dict):
                continue
            tid = w.get("tensor")
            if isinstance(tid, str) and tid.startswith("weight."):
                used_weights.add(tid)
    unused_trainable = sorted(set(trainable_weights) - used_weights)
    add_check(
        "trainable_weight_usage_in_forward",
        len(unused_trainable) == 0,
        f"unused_trainable={len(unused_trainable)}",
        samples=unused_trainable[:max_samples],
    )

    # layer flow coverage
    manifest_cfg = manifest.get("config")
    manifest_cfg = manifest_cfg if isinstance(manifest_cfg, dict) else {}
    num_layers = _cfg_int(manifest_cfg, ["num_layers"]) or _cfg_int(ir2.get("config", {}) if isinstance(ir2.get("config"), dict) else {}, ["num_layers"]) or int(ir2.get("num_layers", 0) or 0)
    if num_layers <= 0:
        # fallback: infer from tensors/op layer ids
        inferred = -1
        for tid in tensors.keys():
            if isinstance(tid, str):
                li = _extract_layer(tid)
                if li is not None:
                    inferred = max(inferred, li)
        for op in all_ops:
            li = op.get("layer")
            if isinstance(li, int):
                inferred = max(inferred, li)
        num_layers = inferred + 1 if inferred >= 0 else 0

    layer_rows: List[Dict[str, Any]] = []
    bad_layers: List[int] = []
    for layer in range(max(0, num_layers)):
        fwd_count = sum(
            1
            for op in forward
            if isinstance(op.get("layer"), int) and int(op.get("layer")) == layer
        )
        bwd_count = sum(
            1
            for op in backward
            if isinstance(op.get("layer"), int) and int(op.get("layer")) == layer
        )
        layer_weights = [w for w in trainable_weights if w.startswith(f"weight.layer.{layer}.")]
        layer_grads = [g for g in grad_weight_tensors if g.startswith(f"grad.weight.layer.{layer}.")]
        layer_writer_cov = 0
        for g in layer_grads:
            wr = grad_writers.get(g)
            if isinstance(wr, list) and len(wr) > 0:
                layer_writer_cov += 1
        ok = (fwd_count > 0 and bwd_count > 0 and len(layer_weights) == len(layer_grads) == layer_writer_cov)
        layer_rows.append(
            {
                "layer": layer,
                "forward_ops": fwd_count,
                "backward_ops": bwd_count,
                "trainable_weights": len(layer_weights),
                "grad_weights": len(layer_grads),
                "writer_covered_grad_weights": layer_writer_cov,
                "passed": bool(ok),
            }
        )
        if not ok:
            bad_layers.append(layer)
    add_check(
        "multilayer_gradient_flow",
        len(bad_layers) == 0 and num_layers > 0,
        f"num_layers={num_layers} bad_layers={len(bad_layers)}",
        samples=bad_layers[:max_samples],
    )

    # layout coverage and optimizer slots
    layout_tensors = layout.get("tensors")
    if not isinstance(layout_tensors, list):
        raise RuntimeError("layout missing `tensors` list")
    layout_by_id: Dict[str, Dict[str, Any]] = {}
    for row in layout_tensors:
        if not isinstance(row, dict):
            continue
        tid = row.get("id")
        if isinstance(tid, str) and tid:
            layout_by_id[tid] = row

    expected_layout_tensors = [
        tid for tid, meta in tensors.items() if isinstance(tid, str) and isinstance(meta, dict) and _tensor_numel(meta)
    ]
    missing_layout = sorted(tid for tid in expected_layout_tensors if tid not in layout_by_id)
    add_check(
        "layout_tensor_coverage",
        len(missing_layout) == 0,
        f"expected={len(expected_layout_tensors)} missing={len(missing_layout)}",
        samples=missing_layout[:max_samples],
    )

    # saved tensor placement + backward consumer wiring
    saved_region = None
    for row in (layout.get("regions") or []):
        if isinstance(row, dict) and str(row.get("name", "")) == "saved":
            saved_region = row
            break

    bad_saved_layout: List[Dict[str, Any]] = []
    for tid in sorted(all_saved_tensors):
        lrow = layout_by_id.get(tid)
        if not isinstance(lrow, dict):
            bad_saved_layout.append({"tensor": tid, "reason": "missing_layout_row"})
            continue
        region = str(lrow.get("region", ""))
        if region != "saved":
            bad_saved_layout.append({"tensor": tid, "reason": "wrong_region", "region": region})
            continue
        if isinstance(saved_region, dict):
            s_off = int(saved_region.get("offset", 0) or 0)
            s_end = s_off + int(saved_region.get("bytes", 0) or 0)
            off = int(lrow.get("offset", -1) or -1)
            end = int(lrow.get("end", -1) or -1)
            if off < s_off or end > s_end or off < 0 or end < off:
                bad_saved_layout.append(
                    {
                        "tensor": tid,
                        "reason": "saved_region_bounds",
                        "offset": off,
                        "end": end,
                        "saved_region_offset": s_off,
                        "saved_region_end": s_end,
                    }
                )
    add_check(
        "saved_tensor_layout_region_integrity",
        len(bad_saved_layout) == 0,
        f"saved_tensors={len(all_saved_tensors)} bad_layout={len(bad_saved_layout)}",
        samples=bad_saved_layout[:max_samples],
    )

    bad_saved_bindings: List[Dict[str, Any]] = []
    saved_use_count: Dict[str, int] = {tid: 0 for tid in all_saved_tensors}
    for op in backward:
        ref = op.get("forward_ref")
        if not isinstance(ref, int):
            continue
        expected = sfb_saved_by_forward.get(int(ref), set())
        ins = op.get("dataflow", {}).get("inputs")
        ins = ins if isinstance(ins, dict) else {}
        for slot_name, slot in ins.items():
            if not isinstance(slot, dict):
                continue
            tid = slot.get("tensor")
            kind = str(slot.get("kind", ""))
            if not isinstance(tid, str):
                continue
            if not (tid.startswith("saved.") or kind.startswith("saved")):
                continue
            if tid not in expected:
                fwd = forward_by_id.get(int(ref), {})
                bad_saved_bindings.append(
                    {
                        "op_id": op.get("op_id"),
                        "op": op.get("op"),
                        "forward_ref": ref,
                        "slot": slot_name,
                        "tensor": tid,
                        "reason": "saved_input_not_in_forward_save_for_backward",
                        "forward_op": fwd.get("op"),
                    }
                )
            if tid in saved_use_count:
                saved_use_count[tid] += 1
    missing_saved_consumers = sorted(tid for tid, c in saved_use_count.items() if c <= 0)
    add_check(
        "saved_tensor_backward_binding_integrity",
        len(bad_saved_bindings) == 0,
        f"bad_bindings={len(bad_saved_bindings)}",
        samples=bad_saved_bindings[:max_samples],
    )
    add_check(
        "saved_tensor_backward_consumption_coverage",
        len(missing_saved_consumers) == 0,
        f"saved_tensors={len(saved_use_count)} missing_consumers={len(missing_saved_consumers)}",
        samples=missing_saved_consumers[:max_samples],
    )

    # Optional runtime generated summary check for saved slots.
    runtime_saved_summary: Dict[str, Any] = {"available": runtime_summary is not None}
    if runtime_summary is not None:
        slot_rows_raw = runtime_summary.get("tensor_slots")
        slot_rows_raw = slot_rows_raw if isinstance(slot_rows_raw, list) else []
        slots_by_name: Dict[str, Dict[str, Any]] = {}
        for row in slot_rows_raw:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            if isinstance(name, str) and name:
                slots_by_name[name] = row

        missing_saved_slots: List[str] = []
        bad_saved_slot_meta: List[Dict[str, Any]] = []
        for tid in sorted(all_saved_tensors):
            row = slots_by_name.get(tid)
            if not isinstance(row, dict):
                missing_saved_slots.append(tid)
                continue
            section = str(row.get("section", ""))
            wf = int(row.get("writable_fwd", 0) or 0)
            if section != "saved":
                bad_saved_slot_meta.append({"tensor": tid, "section": section, "reason": "wrong_section"})
            if wf <= 0:
                bad_saved_slot_meta.append({"tensor": tid, "writable_fwd": wf, "reason": "not_fwd_writable"})

        add_check(
            "runtime_saved_slot_coverage",
            len(missing_saved_slots) == 0,
            f"saved_tensors={len(all_saved_tensors)} missing_runtime_slots={len(missing_saved_slots)}",
            samples=missing_saved_slots[:max_samples],
        )
        add_check(
            "runtime_saved_slot_metadata_integrity",
            len(bad_saved_slot_meta) == 0,
            f"bad_runtime_saved_slots={len(bad_saved_slot_meta)}",
            samples=bad_saved_slot_meta[:max_samples],
        )
        runtime_saved_summary["saved_slot_count"] = len(
            [row for row in slot_rows_raw if isinstance(row, dict) and str(row.get("section", "")) == "saved"]
        )
    else:
        runtime_saved_summary["note"] = "runtime summary not provided; skipped runtime saved-slot checks"

    missing_opt_m: List[str] = []
    missing_opt_v: List[str] = []
    for gtid in sorted(grad_weight_tensors):
        pname = gtid[len("grad.weight.") :]
        mtid = f"optimizer.m.{pname}"
        vtid = f"optimizer.v.{pname}"
        if mtid not in layout_by_id:
            missing_opt_m.append(mtid)
        if vtid not in layout_by_id:
            missing_opt_v.append(vtid)
    add_check(
        "optimizer_state_coverage",
        len(missing_opt_m) == 0 and len(missing_opt_v) == 0,
        f"missing_m={len(missing_opt_m)} missing_v={len(missing_opt_v)}",
        missing_m=missing_opt_m[:max_samples],
        missing_v=missing_opt_v[:max_samples],
    )

    # overlap/bounds quick sanity
    total_bytes = int(layout.get("total_bytes", 0) or 0)
    intervals: List[Tuple[int, int, str]] = []
    bad_bounds = 0
    for tid, row in layout_by_id.items():
        off_raw = row.get("offset")
        end_raw = row.get("end")
        b_raw = row.get("bytes")
        off = int(off_raw) if isinstance(off_raw, (int, float)) else -1
        end = int(end_raw) if isinstance(end_raw, (int, float)) else -1
        b = int(b_raw) if isinstance(b_raw, (int, float)) else 0
        if off < 0 or end < 0 or end < off or b <= 0 or (end - off) != b or end > total_bytes:
            bad_bounds += 1
            continue
        intervals.append((off, end, tid))
    intervals.sort(key=lambda x: (x[0], x[1], x[2]))
    overlaps = 0
    prev_end = -1
    for off, end, _ in intervals:
        if prev_end > off:
            overlaps += 1
        if end > prev_end:
            prev_end = end
    add_check(
        "layout_bounds_and_overlap",
        bad_bounds == 0 and overlaps == 0,
        f"bad_bounds={bad_bounds} overlaps={overlaps} total_bytes={total_bytes}",
    )

    # Dim agreement (manifest vs ir2 config)
    ir2_cfg = ir2.get("config")
    ir2_cfg = ir2_cfg if isinstance(ir2_cfg, dict) else {}
    manifest_dims = {
        "vocab": _cfg_int(manifest_cfg, ["vocab_size"]),
        "d_model": _cfg_int(manifest_cfg, ["embed_dim", "hidden_size", "d_model"]),
        "hidden": _cfg_int(manifest_cfg, ["hidden_size", "intermediate_size", "hidden_dim"]),
        "num_layers": _cfg_int(manifest_cfg, ["num_layers"]),
    }
    ir2_dims = {
        "vocab": _cfg_int(ir2_cfg, ["vocab_size"]),
        "d_model": _cfg_int(ir2_cfg, ["embed_dim", "hidden_size", "d_model"]),
        "hidden": _cfg_int(ir2_cfg, ["hidden_size", "intermediate_size", "hidden_dim"]),
        "num_layers": _cfg_int(ir2_cfg, ["num_layers"]),
    }
    dim_mismatch: Dict[str, Dict[str, Optional[int]]] = {}
    for k in ("vocab", "d_model", "hidden", "num_layers"):
        mv = manifest_dims.get(k)
        iv = ir2_dims.get(k)
        if mv is not None and iv is not None and mv != iv:
            dim_mismatch[k] = {"manifest": mv, "ir2": iv}
    add_check(
        "manifest_ir2_dim_agreement",
        len(dim_mismatch) == 0,
        f"mismatch_fields={len(dim_mismatch)}",
        manifest_dims=manifest_dims,
        ir2_dims=ir2_dims,
        mismatches=dim_mismatch,
    )

    tie = _manifest_tying(manifest)
    add_check(
        "weight_tying_consistency",
        bool(tie.get("config_matches_manifest", True)),
        (
            "tie_cfg=%s actual_tied=%s reason=%s"
            % (tie.get("config_tie_word_embeddings"), tie.get("actual_tied"), tie.get("reason"))
        ),
    )

    # Optional runtime stitch check
    runtime_summary: Dict[str, Any] = {"available": runtime_report is not None}
    if runtime_report is not None:
        train_dims = runtime_report.get("train_dims")
        train_dims = train_dims if isinstance(train_dims, dict) else {}
        eff = train_dims.get("effective")
        eff = eff if isinstance(eff, dict) else {}
        runtime_dim_mismatch: Dict[str, Dict[str, Optional[int]]] = {}
        for k_manifest, k_eff in (("vocab", "vocab"), ("d_model", "d_model"), ("hidden", "hidden"), ("num_layers", "num_layers")):
            mv = manifest_dims.get(k_manifest)
            ev_raw = eff.get(k_eff)
            ev = int(ev_raw) if isinstance(ev_raw, int) else None
            if mv is not None and ev is not None and mv != ev:
                runtime_dim_mismatch[k_manifest] = {"manifest": mv, "runtime_effective": ev}
        add_check(
            "runtime_manifest_dim_wiring",
            len(runtime_dim_mismatch) == 0,
            f"runtime_dim_mismatch={len(runtime_dim_mismatch)}",
            mismatches=runtime_dim_mismatch,
            train_dims_source=train_dims.get("source"),
            requested=train_dims.get("requested"),
            effective=eff,
        )

        first_checked = _first_checked_row(runtime_report)
        if first_checked is None:
            add_check("runtime_first_parity_check_present", False, "no checked parity step in runtime report")
        else:
            first_bad_tensor = first_checked.get("first_bad_tensor")
            oracle_error = first_checked.get("oracle_error")
            slots_compared = int(first_checked.get("slots_compared", 0) or 0)
            loss_diff = first_checked.get("loss_diff")
            logits_diff = first_checked.get("logits_max_abs_diff")
            first_ok = (not first_bad_tensor) and (not oracle_error) and (slots_compared > 0)
            add_check(
                "runtime_first_step_stitch",
                first_ok,
                (
                    "step=%s first_bad_tensor=%s oracle_error=%s slots_compared=%d loss_diff=%s logits_diff=%s"
                    % (
                        first_checked.get("step"),
                        first_bad_tensor,
                        oracle_error,
                        slots_compared,
                        loss_diff,
                        logits_diff,
                    )
                ),
            )
            runtime_summary["first_checked"] = first_checked
    else:
        msg = "runtime report not provided; skipped runtime stitch checks"
        runtime_summary["note"] = msg

    if strict and warnings:
        # In strict mode, warnings are promoted to failures.
        for w in warnings:
            failures.append(f"strict_warning: {w}")

    passed = len(failures) == 0

    region_bytes = _collect_region_bytes(layout)
    layer_region_bytes = _layer_layout_bytes(layout_tensors)

    return {
        "format": "v7-backprop-plumbing-audit",
        "generated_at": _utc_now_iso(),
        "passed": bool(passed),
        "strict": bool(strict),
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "stats": {
            "forward_ops": len(forward),
            "backward_ops": len(backward),
            "tensor_count": len(tensors),
            "trainable_weights": len(trainable_weights),
            "grad_weight_tensors": len(grad_weight_tensors),
            "num_layers": int(num_layers),
            "layout_tensor_count": len(layout_by_id),
            "layout_total_bytes": total_bytes,
        },
        "weight_tying": tie,
        "runtime_stitch": runtime_summary,
        "runtime_saved": runtime_saved_summary,
        "layer_flow": layer_rows,
        "layout_map": {
            "total_bytes": total_bytes,
            "region_bytes": region_bytes,
            "region_order": sorted(region_bytes.keys()),
            "layer_region_bytes": layer_region_bytes,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v7 backprop plumbing from IR2/layout/manifest artifacts.")
    ap.add_argument("--ir2", type=Path, required=True, help="Path to ir2_train_backward.json")
    ap.add_argument("--layout", type=Path, required=True, help="Path to layout_train.json")
    ap.add_argument("--manifest", type=Path, required=True, help="Path to weights_manifest.json")
    ap.add_argument("--runtime-report", type=Path, default=None, help="Optional train_e2e_latest.json for runtime stitch checks")
    ap.add_argument("--runtime-summary", type=Path, default=None, help="Optional generated_train_runtime_summary_v7.json for saved-slot checks")
    ap.add_argument("--strict", action="store_true", help="Promote warnings to failures")
    ap.add_argument("--max-samples", type=int, default=16, help="Max samples to include for each failure category")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional output JSON path")
    args = ap.parse_args()

    ir2 = _load_json(args.ir2)
    layout = _load_json(args.layout)
    manifest = _load_json(args.manifest)
    runtime_report = _load_json(args.runtime_report) if args.runtime_report and args.runtime_report.exists() else None
    runtime_summary = _load_json(args.runtime_summary) if args.runtime_summary and args.runtime_summary.exists() else None

    report = audit_backprop_plumbing(
        ir2=ir2,
        layout=layout,
        manifest=manifest,
        runtime_report=runtime_report,
        runtime_summary=runtime_summary,
        strict=bool(args.strict),
        max_samples=max(1, int(args.max_samples)),
    )

    print("=" * 96)
    print("v7 BACKPROP PLUMBING AUDIT")
    print("=" * 96)
    print(f"- passed: {report.get('passed')}")
    stats = report.get("stats", {})
    print(
        "- ops/tensors: forward=%d backward=%d tensors=%d trainable_weights=%d grad_weights=%d"
        % (
            int(stats.get("forward_ops", 0) or 0),
            int(stats.get("backward_ops", 0) or 0),
            int(stats.get("tensor_count", 0) or 0),
            int(stats.get("trainable_weights", 0) or 0),
            int(stats.get("grad_weight_tensors", 0) or 0),
        )
    )
    print(f"- checks: {len(report.get('checks', []))} failures: {len(report.get('failures', []))} warnings: {len(report.get('warnings', []))}")
    tie = report.get("weight_tying", {})
    print(
        "- tying: config=%s actual=%s reason=%s"
        % (tie.get("config_tie_word_embeddings"), tie.get("actual_tied"), tie.get("reason"))
    )
    if report.get("failures"):
        print("- first_failure:", report["failures"][0])
    print("=" * 96)

    if args.json_out is not None:
        _save_json(args.json_out, report)
        print("JSON:", args.json_out)

    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
