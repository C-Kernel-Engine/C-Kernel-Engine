#!/usr/bin/env python3
"""
bump_inspect.py
==============

Inspect a BUMPWGT4/5 weights file. For BUMPWGT5, read metadata footer and
validate metadata hash. Optionally compare with a sidecar manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from typing import Any, Dict, Optional, Tuple


BUMP_MAGIC_V4 = b"BUMPWGT4"
BUMP_MAGIC_V5 = b"BUMPWGT5"
META_FOOTER_MAGIC = b"BUMPV5MD"
HEADER_SIZE = 128
META_FOOTER_SIZE = 48  # 8 + 8 + 32


def _canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_header(fp) -> Dict[str, Any]:
    fp.seek(0, os.SEEK_SET)
    raw = fp.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise ValueError("File too small for BUMP header")

    # Header layout matches BUMPWGT4 (120 bytes used, 8 bytes reserved)
    fmt = "<8sIIIIIIIIIIQQQQII32s"
    sz = struct.calcsize(fmt)
    if sz > len(raw):
        raise ValueError("Header parse size exceeds buffer")

    unpacked = struct.unpack(fmt, raw[:sz])
    return {
        "magic": unpacked[0],
        "version": unpacked[1],
        "model_type": unpacked[2],
        "num_layers": unpacked[3],
        "vocab_size": unpacked[4],
        "embed_dim": unpacked[5],
        "intermediate_size": unpacked[6],
        "context_length": unpacked[7],
        "num_heads": unpacked[8],
        "num_kv_heads": unpacked[9],
        "head_dim": unpacked[10],
        "aligned_embed_dim": unpacked[11],
        "aligned_head_dim": unpacked[12],
        "aligned_intermediate": unpacked[13],
        "aligned_context": unpacked[14],
        "num_merges": unpacked[15],
        "total_vocab_bytes": unpacked[16],
        "payload_sha256": unpacked[17],
    }


def _read_meta_footer(fp) -> Tuple[int, bytes]:
    fp.seek(0, os.SEEK_END)
    file_size = fp.tell()
    if file_size < META_FOOTER_SIZE:
        raise ValueError("File too small for metadata footer")
    fp.seek(file_size - META_FOOTER_SIZE, os.SEEK_SET)
    raw = fp.read(META_FOOTER_SIZE)
    magic, meta_size, meta_sha256 = struct.unpack("<8sQ32s", raw)
    if magic != META_FOOTER_MAGIC:
        raise ValueError(f"Missing metadata footer magic: {magic}")
    if meta_size > file_size - META_FOOTER_SIZE:
        raise ValueError("Invalid meta_size (beyond file bounds)")
    return meta_size, meta_sha256


def _read_metadata(fp, meta_size: int) -> Dict[str, Any]:
    fp.seek(0, os.SEEK_END)
    file_size = fp.tell()
    meta_offset = file_size - META_FOOTER_SIZE - meta_size
    if meta_offset < HEADER_SIZE:
        raise ValueError("Metadata offset overlaps header/payload")
    fp.seek(meta_offset, os.SEEK_SET)
    raw = fp.read(meta_size)
    return json.loads(raw.decode("utf-8"))


def _read_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_header(h: Dict[str, Any]) -> None:
    print("[header]")
    print(f"  magic: {h['magic']!r}")
    print(f"  version: {h['version']}")
    print(f"  layers: {h['num_layers']}")
    print(f"  vocab_size: {h['vocab_size']}")
    print(f"  embed_dim: {h['embed_dim']}")
    print(f"  heads: {h['num_heads']}/{h['num_kv_heads']} (head_dim={h['head_dim']})")
    print(f"  intermediate: {h['intermediate_size']}")
    print(f"  context: {h['context_length']}")
    print(f"  aligned: embed={h['aligned_embed_dim']} head={h['aligned_head_dim']} inter={h['aligned_intermediate']} ctx={h['aligned_context']}")
    print(f"  num_merges: {h['num_merges']} total_vocab_bytes: {h['total_vocab_bytes']}")
    print(f"  payload_sha256: {h['payload_sha256'].hex()}")


def _print_meta(meta: Dict[str, Any], meta_hash_hex: str, stored_hash_hex: str) -> None:
    print("[metadata]")
    print(f"  schema_version: {meta.get('schema_version')}")
    print(f"  format: {meta.get('format')}")
    print(f"  created_by: {meta.get('created_by')}")
    print(f"  created_at: {meta.get('created_at')}")
    tmpl = meta.get("template", {})
    print(f"  template_name: {tmpl.get('name', tmpl.get('id', 'unknown'))}")
    print(f"  template_hash: {meta.get('template_hash')}")
    cfg = meta.get("config", {})
    if isinstance(cfg, dict):
        print(
            "  config: "
            f"model={cfg.get('model')} layers={cfg.get('num_layers')} "
            f"embed={cfg.get('embed_dim')} heads={cfg.get('num_heads')}/{cfg.get('num_kv_heads')} "
            f"head_dim={cfg.get('head_dim')} inter={cfg.get('intermediate_size')} ctx={cfg.get('context_length')}"
        )
    quant = meta.get("quant_summary", {})
    print(f"  quant_summary_layers: {len(quant) if isinstance(quant, dict) else 0}")
    print(f"  manifest_hash: {meta.get('manifest_hash')}")
    print(f"  meta_sha256: {stored_hash_hex}")
    print(f"  meta_sha256_check: {'ok' if meta_hash_hex == stored_hash_hex else 'MISMATCH'}")


def _print_manifest(manifest: Dict[str, Any], manifest_hash_hex: str, meta_hash: Optional[str]) -> None:
    print("[manifest]")
    print(f"  version: {manifest.get('version')}")
    print(f"  model: {manifest.get('model')}")
    print(f"  layers: {manifest.get('num_layers')}")
    print(f"  embed_dim: {manifest.get('embed_dim')}")
    print(f"  context_length: {manifest.get('context_length')}")
    entries = manifest.get("entries", [])
    print(f"  entries: {len(entries) if isinstance(entries, list) else 0}")
    print(f"  manifest_sha256: {manifest_hash_hex}")
    if meta_hash:
        print(f"  manifest_hash_match: {'ok' if manifest_hash_hex == meta_hash else 'MISMATCH'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect BUMPWGT4/5 weights file")
    ap.add_argument("bump", help="Path to weights.bump")
    ap.add_argument("--manifest", help="Optional weights_manifest.json")
    ap.add_argument("--dump-meta", action="store_true", help="Print full metadata JSON")
    ap.add_argument("--dump-manifest", action="store_true", help="Print full manifest JSON")
    args = ap.parse_args()

    if not os.path.exists(args.bump):
        print(f"[error] bump file not found: {args.bump}", file=sys.stderr)
        return 2

    with open(args.bump, "rb") as fp:
        header = _read_header(fp)
        _print_header(header)

        magic = header["magic"]
        if magic == BUMP_MAGIC_V5:
            meta_size, meta_sha256 = _read_meta_footer(fp)
            meta = _read_metadata(fp, meta_size)
            meta_bytes = _canonical_json_bytes(meta)
            meta_hash_hex = _sha256_hex(meta_bytes)
            _print_meta(meta, meta_hash_hex, meta_sha256.hex())
            if args.dump_meta:
                print("\n[metadata_json]")
                print(json.dumps(meta, indent=2))
        elif magic == BUMP_MAGIC_V4:
            print("[metadata] none (BUMPWGT4)")
        else:
            print(f"[error] unknown magic: {magic!r}", file=sys.stderr)
            return 1

    if args.manifest:
        if not os.path.exists(args.manifest):
            print(f"[error] manifest not found: {args.manifest}", file=sys.stderr)
            return 2
        manifest = _read_manifest(args.manifest)
        manifest_bytes = _canonical_json_bytes(manifest)
        manifest_hash_hex = _sha256_hex(manifest_bytes)
        meta_manifest_hash = None
        if magic == BUMP_MAGIC_V5:
            meta_manifest_hash = meta.get("manifest_hash")
        _print_manifest(manifest, manifest_hash_hex, meta_manifest_hash)
        if args.dump_manifest:
            print("\n[manifest_json]")
            print(json.dumps(manifest, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
