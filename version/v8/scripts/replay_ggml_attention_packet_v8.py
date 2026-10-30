#!/usr/bin/env python3
"""Replay retained attention operands through pinned GGML attention graphs."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

if __package__:
    from .replay_attention_from_dumps_v8 import (
        _load_resolved_bundle, _metrics, _sha256, replay_resolved_segmented_prefill,
    )
    from .xray_operation_bundle_v8 import BundleError, load_bundle
else:
    from replay_attention_from_dumps_v8 import (
        _load_resolved_bundle, _metrics, _sha256, replay_resolved_segmented_prefill,
    )
    from xray_operation_bundle_v8 import BundleError, load_bundle


SOURCE = Path(__file__).with_name("ggml_nonflash_attention_packet_v8.c")
OBSERVED_SOURCE = Path(__file__).with_name("ggml_nonflash_attention_observed_v8.c")
FLASH_SOURCE = Path(__file__).with_name("ggml_flash_attention_rows_v8.c")
REPO = Path(__file__).resolve().parents[3]


def _observed_capture(dump_dir: Path, name: str) -> tuple[dict, dict]:
    if not name.startswith("kqv_out-") or "/" in name or "\\" in name or ".." in name:
        raise BundleError("observed replay requires a kqv_out capture name")
    specs = {}
    for role, dtype, width in (("q", 0, 4), ("k", 1, 2), ("v", 1, 2), ("mask", 0, 4)):
        path = dump_dir / f"{name}.nonflash_{role}"
        try:
            meta = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
            size = Path(str(path) + ".bin").stat().st_size
        except (OSError, json.JSONDecodeError) as exc:
            raise BundleError(f"missing or invalid observed {role} metadata") from exc
        ne, nb = meta.get("ne"), meta.get("nb")
        if (meta.get("role") != role or meta.get("type") != dtype or
            not isinstance(ne, list) or not isinstance(nb, list) or len(ne) != 4 or len(nb) != 4 or
            any(type(x) is not int or x <= 0 for x in ne) or
            any(type(x) is not int or x <= 0 for x in nb) or nb[0] != width or
            ne[3] != 1 or meta.get("nbytes") != size):
            raise BundleError(f"invalid observed {role} storage contract")
        extent = (ne[0] - 1) * nb[0] + (ne[1] - 1) * nb[1] + (ne[2] - 1) * nb[2] + width
        if extent != meta["nbytes"] or extent > 64 * 1024 * 1024:
            raise BundleError(f"invalid observed {role} physical extent")
        specs[role] = meta
    q, k, v, mask = (specs[role] for role in ("q", "k", "v", "mask"))
    dim, queries, heads = q["ne"][:3]
    keys, kv_heads = k["ne"][1:3]
    if (not (1 <= queries <= 4096 and 1 <= keys <= 4096 and 1 <= heads <= 128 and
             1 <= kv_heads <= heads and heads % kv_heads == 0 and 1 <= dim <= 1024) or
        k["ne"][0] != dim or v["ne"][:3] != [keys, dim, kv_heads] or
        mask["ne"][:3] != [keys, queries, 1] or
        q["nb"][1] < dim * 4 or k["nb"][1] < dim * 2 or
        v["nb"][1] < keys * 2 or mask["nb"][1] < keys * 4):
        raise BundleError("observed Q/K/V/mask geometry or strides disagree")
    if (sum(spec["nbytes"] for spec in specs.values()) > 64 * 1024 * 1024 or
        queries * keys * heads > 4 * 1024 * 1024 or
        queries * heads * dim * 4 > 64 * 1024 * 1024):
        raise BundleError("observed attention replay exceeds bounded workspace")
    output = dump_dir / f"{name}.bin"
    expected_bytes = queries * heads * dim * 4
    sidecar = dump_dir / f"{name}.json"
    try:
        output_meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError("observed attention output metadata is missing") from exc
    if (not output.is_file() or output.stat().st_size != expected_bytes or
        output_meta.get("type") != 0 or
        output_meta.get("ne") != [heads * dim, queries, 1, 1] or
        output_meta.get("nb", [])[:2] != [4, heads * dim * 4]):
        raise BundleError("observed attention output is missing or truncated")
    return specs, {"queries": queries, "keys": keys, "heads": heads,
                   "kv_heads": kv_heads, "dim": dim, "output": output}


def replay_observed_oracle(
    tree: Path, dump_dir: Path, name: str, *, scale: float, alibi: float, threads: int,
    parity_report: Path | None = None,
) -> dict:
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(alibi) or not 1 <= threads <= 256:
        raise BundleError("observed replay requires declared scale, alibi, and thread count")
    specs, geometry = _observed_capture(dump_dir, name)
    if parity_report is not None:
        try:
            capture_report = json.loads(parity_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BundleError("observed replay parity report is missing or invalid") from exc
        comparison = capture_report.get("dump_compare", {})
        if (not isinstance(comparison, dict) or
            comparison.get("llama_flash_attention_mode") != "disabled" or
            not isinstance(comparison.get("llama_dump_dir"), str) or
            Path(comparison["llama_dump_dir"]).resolve() != dump_dir.resolve() or
            type(comparison.get("llama_nonflash_input_captures")) is not int or
            comparison["llama_nonflash_input_captures"] <= 0):
            raise BundleError("parity report does not bind this non-flash oracle capture")
    identity = pinned_tree_identity(tree)
    build_bin = tree / "build" / "bin"
    with tempfile.TemporaryDirectory(prefix="cke-ggml-observed-attn-") as temp:
        executable = Path(temp) / "replay"
        output = Path(temp) / "out.f32"
        resolved = _compile_ggml_source(tree, OBSERVED_SOURCE, executable)
        args = [str(executable)]
        args.extend(str(dump_dir / f"{name}.nonflash_{role}.bin")
                    for role in ("q", "k", "v", "mask"))
        args.extend([str(output), *(str(geometry[key]) for key in
                    ("queries", "keys", "heads", "kv_heads", "dim"))])
        args.extend(str(specs[role]["nb"][axis]) for role, axis in (
            ("q", 1), ("q", 2), ("k", 1), ("k", 2),
            ("v", 1), ("v", 2), ("mask", 1)))
        args.extend((str(scale), str(alibi), str(threads)))
        subprocess.run(args, env=dict(os.environ, LD_LIBRARY_PATH=str(build_bin)),
                       check=True, capture_output=True, text=True)
        actual = np.fromfile(output, dtype="<f4")
    expected = np.fromfile(geometry["output"], dtype="<f4")
    if actual.size != expected.size or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise BundleError("observed replay output is incomplete or nonfinite")
    comparison = _metrics(actual, expected)
    identity.update({"replay_source_sha256": _sha256(OBSERVED_SOURCE),
                     "resolved_libraries": resolved,
                     "library_build_revision_verified": False})
    return {"status": "diagnostic_only",
            "oracle_replay_fidelity": "pass" if np.array_equal(actual, expected) else "fail",
            "comparison": comparison, "geometry": {k: v for k, v in geometry.items() if k != "output"},
            "capture_name": name,
            "capture_sha256": {
                **{role: _sha256(dump_dir / f"{name}.nonflash_{role}.bin")
                   for role in ("q", "k", "v", "mask")},
                "output": _sha256(geometry["output"]),
                **({"parity_report": _sha256(parity_report)} if parity_report else {}),
            },
            "oracle_graph": identity}


def validate_oracle_contract(manifest: dict, call: dict, segments: list[int]) -> None:
    context = manifest["context"]
    tokens = sum(segments)
    if context.get("oracle_mode") != "disabled":
        raise BundleError("GGML replay requires the retained non-flash oracle mode")
    if call["sliding_window"] < tokens:
        raise BundleError("this adapter cannot replay a wrapped sliding attention window")
    if context.get("valid_cache_positions") != [0, tokens]:
        raise BundleError("invalid post-store cache position range")
    if call["head_dim"] != call["aligned_head_dim"]:
        raise BundleError("GGML replay does not support padded head dimensions")
    if call["selector"] not in (
        "CK_ATTN_REDUCTION_F16_REGULAR_NOSCALE",
        "CK_ATTN_REDUCTION_F16_FLASH_AUTO_QTILE64_REGULAR_TAIL_NOSCALE",
    ):
        raise BundleError("GGML replay does not support this attention scaling contract")


def pinned_tree_identity(tree: Path) -> dict[str, object]:
    expected = subprocess.check_output(
        ["git", "ls-tree", "HEAD", "llama.cpp"], cwd=REPO, text=True,
    ).split()[2]
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tree, text=True,
    ).strip()
    if expected != actual:
        raise BundleError(f"llama.cpp checkout {actual} does not match CKE gitlink {expected}")
    lib = tree / "build" / "bin" / "libggml-cpu.so"
    base = tree / "build" / "bin" / "libggml-base.so"
    if not lib.is_file() or not base.is_file():
        raise BundleError("pinned llama.cpp GGML libraries are not built")
    return {"source_revision": actual, "ggml_cpu_sha256": _sha256(lib),
            "ggml_base_sha256": _sha256(base)}


def _compile_ggml_source(tree: Path, source: Path, executable: Path) -> dict:
    build_bin = tree / "build" / "bin"
    compile_cmd = [
        "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        "-I", str(tree / "ggml" / "include"), str(source),
        "-L", str(build_bin), "-Wl,-rpath," + str(build_bin),
        "-lggml-cpu", "-lggml-base", "-lggml", "-lm", "-o", str(executable),
    ]
    subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
    linked = subprocess.check_output(["ldd", str(executable)], text=True)
    resolved = {}
    for line in linked.splitlines():
        fields = line.strip().split("=>", 1)
        if len(fields) != 2 or not fields[0].startswith("libggml"):
            continue
        path = Path(fields[1].strip().split(" ", 1)[0]).resolve()
        if path.parent != build_bin.resolve() or not path.is_file():
            raise BundleError(f"GGML replay linked an unexpected library: {line.strip()}")
        resolved[fields[0].strip()] = {"path": str(path), "sha256": _sha256(path)}
    if not any(name.startswith("libggml-cpu") for name in resolved) or not any(
        name.startswith("libggml-base") for name in resolved
    ):
        raise BundleError("GGML replay did not link the expected CPU/base libraries")
    return resolved


def run_ggml(tree: Path, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, dict]:
    tokens, heads, dim = q.shape
    kv_heads = k.shape[1]
    identity = pinned_tree_identity(tree)
    build_bin = tree / "build" / "bin"
    with tempfile.TemporaryDirectory(prefix="cke-ggml-attn-replay-") as temp:
        directory = Path(temp)
        executable = directory / "replay"
        resolved = _compile_ggml_source(tree, SOURCE, executable)
        paths = [directory / name for name in ("q.f32", "k.f16", "v.f16", "out.f32")]
        np.asarray(q, dtype="<f4").tofile(paths[0])
        np.asarray(k, dtype="<f2").tofile(paths[1])
        np.asarray(v, dtype="<f2").tofile(paths[2])
        env = dict(os.environ, LD_LIBRARY_PATH=str(build_bin))
        subprocess.run(
            [str(executable), *(str(path) for path in paths),
             str(tokens), str(heads), str(kv_heads), str(dim)],
            env=env, check=True, capture_output=True, text=True,
        )
        output = np.fromfile(paths[3], dtype="<f4")
        if output.size != q.size or not np.isfinite(output).all():
            raise BundleError("GGML replay produced incomplete or nonfinite output")
        identity["replay_binary_sha256"] = _sha256(executable)
        identity["replay_source_sha256"] = _sha256(SOURCE)
        identity["resolved_libraries"] = resolved
        identity["library_build_revision_verified"] = False
        return output.reshape(q.shape), identity


def run_ggml_flash_rows(
    tree: Path, q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
    scale: float, threads: int = 1, allow_subset_queries: bool = False,
) -> tuple[np.ndarray, dict]:
    """Replay full, unmasked flash self-attention; subsets are diagnostic only."""
    q, k, v = np.asarray(q), np.asarray(k), np.asarray(v)
    if q.dtype != np.float32 or k.dtype != np.float16 or v.dtype != np.float16:
        raise BundleError("flash replay requires observed FP32 Q and post-store FP16 K/V")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise BundleError("flash replay requires Q[queries,heads,dim] and matching K/V[keys,kv_heads,dim]")
    queries, heads, dim = q.shape
    keys, kv_heads, key_dim = k.shape
    if not (1 <= queries <= 32768 and 1 <= keys <= 32768 and 1 <= heads <= 128 and
            1 <= kv_heads <= heads and heads % kv_heads == 0 and
            1 <= dim <= 256 and key_dim == dim and 1 <= threads <= 256):
        raise BundleError("unsupported full-flash attention geometry")
    if queries != keys and not allow_subset_queries:
        raise BundleError("full-query geometry is required for connected flash replay fidelity")
    if not math.isfinite(scale) or scale <= 0 or not all(np.isfinite(x).all() for x in (q, k, v)):
        raise BundleError("flash replay requires finite operands and positive finite scale")
    if q.nbytes + k.nbytes + v.nbytes > 256 * 1024 * 1024:
        raise BundleError("flash replay operands exceed capacity")

    identity = pinned_tree_identity(tree)
    build_bin = tree / "build" / "bin"
    with tempfile.TemporaryDirectory(prefix="cke-ggml-flash-replay-") as temp:
        directory = Path(temp)
        executable = directory / "replay"
        resolved = _compile_ggml_source(tree, FLASH_SOURCE, executable)
        paths = [directory / name for name in ("q.f32", "k.f16", "v.f16", "out.f32")]
        np.asarray(q, dtype="<f4").tofile(paths[0])
        np.asarray(k, dtype="<f2").tofile(paths[1])
        np.asarray(v, dtype="<f2").tofile(paths[2])
        env = dict(os.environ, LD_LIBRARY_PATH=str(build_bin))
        subprocess.run(
            [str(executable), *(str(path) for path in paths),
             str(queries), str(keys), str(heads), str(kv_heads), str(dim),
             repr(float(np.float32(scale))), str(threads)],
            env=env, check=True, capture_output=True, text=True, timeout=300,
        )
        output = np.fromfile(paths[3], dtype="<f4")
        if output.size != q.size or not np.isfinite(output).all():
            raise BundleError("GGML flash replay produced incomplete or nonfinite output")
        identity["replay_binary_sha256"] = _sha256(executable)
        identity["replay_source_sha256"] = _sha256(FLASH_SOURCE)
        identity["resolved_libraries"] = resolved
        identity["library_build_revision_verified"] = False
        identity["contract"] = {
            "mode": "full_noncausal_flash", "q_dtype": "fp32", "kv_dtype": "fp16",
            "accumulation": "ggml_prec_f32", "mask": "none", "scale": float(np.float32(scale)),
            "queries": queries, "keys": keys, "heads": heads, "kv_heads": kv_heads,
            "head_dim": dim, "threads": threads,
            "query_extent": "complete" if queries == keys else "subset_not_connected_fidelity",
        }
        return output.reshape(q.shape), identity


def fp64_full_flash_reference_rows(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, rows: list[int], *, scale: float,
) -> np.ndarray:
    """Evaluate selected noncausal rows with FP16-rounded K/V and FP64 arithmetic."""
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float16)
    v = np.asarray(v, dtype=np.float16)
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise BundleError("FP64 flash reference requires rank-three Q and matching K/V")
    queries, heads, dim = q.shape
    keys, kv_heads, key_dim = k.shape
    if (not queries or not keys or not heads or not kv_heads or key_dim != dim or
            heads % kv_heads or not math.isfinite(scale) or scale <= 0 or
            not all(np.isfinite(x).all() for x in (q, k, v))):
        raise BundleError("invalid FP64 flash reference contract")
    if not rows or len(set(rows)) != len(rows) or any(row < 0 or row >= queries for row in rows):
        raise BundleError("FP64 flash reference requires distinct in-range query rows")
    result = np.empty((len(rows), heads, dim), dtype=np.float64)
    k64 = k.astype(np.float64)
    v64 = v.astype(np.float64)
    for index, row in enumerate(rows):
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            scores = (k64[:, kv_head] @ q[row, head].astype(np.float64)) * scale
            weights = np.exp(scores - scores.max())
            result[index, head] = (weights / weights.sum()) @ v64[:, kv_head]
    return result


def fp64_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    tokens, heads, dim = q.shape
    kv_heads = k.shape[1]
    result = np.empty_like(q)
    for t in range(tokens):
        for h in range(heads):
            kh = h // (heads // kv_heads)
            scores = k[:t + 1, kh].astype(np.float64) @ q[t, h].astype(np.float64)
            weights = np.exp(scores - np.max(scores))
            weights /= np.sum(weights)
            result[t, h] = weights @ v[:t + 1, kh].astype(np.float64)
    return result


def compare_packet(bundle: Path, tree: Path) -> dict:
    dumps, layer, call, segments, engine = _load_resolved_bundle(bundle)
    manifest, tensors = load_bundle(bundle)
    validate_oracle_contract(manifest, call, segments)
    fidelity = replay_resolved_segmented_prefill(dumps, layer, call, segments, engine,
                                                  independent_fp64=True)
    if fidelity["status"] != "pass":
        return {"status": "fail_replay_fidelity", "cke_replay": fidelity,
                "oracle_graph": "not_run"}
    q = tensors["Qcur_rope"]
    k = tensors["Kcache_post_store"]
    v = tensors["Vcache_post_store"]
    oracle, identity = run_ggml(tree, q, k, v)
    reference = fp64_reference(q, k, v)
    cke = tensors["kqv_out"]
    worst = np.unravel_index(np.argmax(np.abs(cke - oracle)), cke.shape)
    return {
        "status": "incomplete_oracle_runtime_fidelity",
        "interpretation": "Compact valid-key GGML diagnostic only; the executed llama.cpp graph may use a larger masked physical cache and cannot be inferred from this result",
        "first_proven_local_divergence": None,
        "missing_oracle_capability": "oracle post-RoPE Q, complete post-store K/V and resolved mask for replay fidelity; CKE and oracle score/softmax captures for first-stage attribution",
        "bundle_sha256": _sha256(bundle),
        "cke_replay": {"status": fidelity["status"], "bitwise_equal": fidelity["bitwise_equal"],
                       "cache_reconstruction": fidelity["cache_reconstruction"]},
        "oracle_graph": identity,
        "contract": {"mode": "non_flash", "effective_mask_assumption": "causal prefix, valid keys 0..query_position; oracle mask not captured",
                     "scale": 1.0, "cache_dtype": "fp16", "heads": q.shape[1],
                     "kv_heads": k.shape[1], "head_dim": q.shape[2], "segments": segments,
                     "key_extent": "compact valid keys; not the oracle's physical cache extent"},
        "cke_vs_ggml": _metrics(cke, oracle),
        "cke_vs_fp64": _metrics(cke, reference),
        "ggml_vs_fp64": _metrics(oracle, reference),
        "worst_cke_ggml_index": list(map(int, worst)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundle", type=Path)
    mode.add_argument("--oracle-dump-dir", type=Path)
    parser.add_argument("--oracle-dump-name")
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--alibi", type=float)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--llama-tree", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.oracle_dump_dir and (not args.oracle_dump_name or not args.parity_report or
                                 args.scale is None or
                                 args.alibi is None or args.threads is None):
        parser.error("observed oracle replay requires capture report, dump name, scale, alibi, and threads")
    if args.bundle and (args.oracle_dump_name or args.parity_report or args.scale is not None or
                        args.alibi is not None or args.threads is not None):
        parser.error("observed oracle options cannot be combined with a CKE bundle")
    try:
        if args.oracle_dump_dir:
            report = replay_observed_oracle(
                args.llama_tree, args.oracle_dump_dir, args.oracle_dump_name,
                scale=args.scale, alibi=args.alibi, threads=args.threads,
                parity_report=args.parity_report,
            )
        else:
            report = compare_packet(args.bundle, args.llama_tree)
    except (BundleError, ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        report = {"status": "incomplete", "error": str(exc)}
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if report.get("oracle_replay_fidelity") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
