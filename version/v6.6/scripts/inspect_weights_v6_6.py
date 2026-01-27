#!/usr/bin/env python3
"""
inspect_weights_v6.py
=====================

Lightweight weights inspection for v6:
  - GGUF: parse header and emit a manifest with per-weight dtypes
  - HF (safetensors): read headers and emit a manifest from WEIGHT_MAP_V4

This does NOT convert weights or write bump files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_V4_DIR = _SCRIPT_DIR / "v4"
for path in (_SCRIPT_DIR, _V4_DIR):
    if path.is_dir():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

import ir_core_v6_6 as v4
import convert_gguf_to_bump_v6_6 as gguf


def pick(cfg: Dict, keys: List[str], default=None):
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def map_safetensors_dtype(dtype: str) -> str:
    if not dtype:
        return "fp32"
    if dtype.upper() in ("F16", "BF16", "F32"):
        return "fp32"
    return "fp32"


def load_safetensors_entries(model_dir: Path) -> Dict[str, Dict]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = v4.read_weights_index(str(index_path))
        weight_map = index.get("weight_map", {})
        headers: Dict[str, Dict] = {}
        for filename in sorted(set(weight_map.values())):
            header_path = model_dir / filename
            headers[filename] = v4.read_safetensors_header(str(header_path))
        entries: Dict[str, Dict] = {}
        for name, filename in weight_map.items():
            header = headers.get(filename, {})
            if name in header:
                entries[name] = header[name]
        return entries

    files = sorted(model_dir.glob("*.safetensors"))
    if len(files) != 1:
        raise SystemExit("Expected exactly one safetensors file or an index JSON")
    header = v4.read_safetensors_header(str(files[0]))
    return {k: v for k, v in header.items() if k != "__metadata__"}


def build_hf_manifest(config: Dict, st_entries: Dict[str, Dict], max_layers: Optional[int]) -> List[Dict]:
    num_layers = pick(config, ["num_hidden_layers", "num_layers"])
    if num_layers is None:
        raise SystemExit("Config missing num_layers")
    if max_layers:
        num_layers = min(int(max_layers), int(num_layers))

    manifest_entries: List[Dict] = []
    pack_groups: Dict[str, List[str]] = {}

    def add_entry(ck_name: str, hf_name: str, optional: bool = False) -> None:
        entry = st_entries.get(hf_name)
        if entry is None:
            if optional:
                return
            raise SystemExit(f"Missing weight: {hf_name}")
        dtype = map_safetensors_dtype(entry.get("dtype", ""))
        manifest_entries.append({
            "name": ck_name,
            "dtype": dtype,
            "shape": entry.get("shape"),
            "source_dtype": entry.get("dtype"),
        })

    for mapping in v4.WEIGHT_MAP_V4:
        hf_tpl = mapping["hf"]
        ck_tpl = mapping["ck"]
        optional = bool(mapping.get("optional", False))
        pack = mapping.get("pack")

        if "{layer}" in hf_tpl:
            for layer in range(num_layers):
                hf_name = hf_tpl.replace("{layer}", str(layer))
                ck_name = ck_tpl.replace("{L}", str(layer))
                if pack:
                    pack_groups.setdefault(ck_name, []).append(hf_name)
                else:
                    add_entry(ck_name, hf_name, optional=optional)
        else:
            if pack:
                pack_groups.setdefault(ck_tpl, []).append(hf_tpl)
            else:
                add_entry(ck_tpl, hf_tpl, optional=optional)

    for ck_name, hf_names in pack_groups.items():
        dtypes = []
        shapes = []
        source_dtypes = []
        for hf_name in hf_names:
            entry = st_entries.get(hf_name)
            if entry is None:
                raise SystemExit(f"Missing packed weight: {hf_name}")
            dtypes.append(map_safetensors_dtype(entry.get("dtype", "")))
            source_dtypes.append(entry.get("dtype"))
            shapes.append(entry.get("shape"))
        if len(set(dtypes)) != 1:
            raise SystemExit(f"Packed weights for {ck_name} have mismatched dtypes: {dtypes}")
        manifest_entries.append({
            "name": ck_name,
            "dtype": dtypes[0],
            "shape": shapes[0],
            "source_dtype": ",".join([d for d in source_dtypes if d]),
        })

    return manifest_entries


def dtype_name(dt: int) -> str:
    if dt == gguf.CK_DT_Q4_K:
        return "q4_k"
    if dt == gguf.CK_DT_Q6_K:
        return "q6_k"
    return "fp32"


def weight_dtype(info: "gguf.TensorInfo", label: str) -> int:
    supported_types = (
        gguf.GGML_TYPE_Q4_K,
        gguf.GGML_TYPE_Q6_K,
        gguf.GGML_TYPE_Q5_0,
        gguf.GGML_TYPE_Q5_1,
        gguf.GGML_TYPE_Q8_0,
        gguf.GGML_TYPE_F32,
    )
    if info.ggml_type not in supported_types:
        raise gguf.GGUFError(
            f"{info.name}: expected Q4_K/Q6_K/Q5_0/Q5_1/Q8_0/F32 for {label}, got "
            f"{gguf.ggml_type_name(info.ggml_type)}"
        )
    return gguf.ck_dtype_from_ggml_type(info.ggml_type)


def inspect_gguf(gguf_path: Path, max_layers: Optional[int]) -> Tuple[Dict, List[Dict]]:
    with open(gguf_path, "rb") as f:
        r = gguf.GGUFReader(f)
        magic = r._read_exact(4)
        if magic != b"GGUF":
            raise gguf.GGUFError("Invalid GGUF magic")
        r.u32()  # version
        n_tensors = r.u64()
        n_kv = r.u64()

        meta: Dict[str, object] = {}
        arch_prefixes = ("general.", "llama.", "qwen2.", "qwen.")
        for _ in range(n_kv):
            key = r.key_str()
            vtype = r.u32()
            if key.startswith(arch_prefixes):
                meta[key] = gguf._gguf_read_value(r, vtype)
            else:
                gguf._gguf_skip_value(r, vtype)

        tensors: Dict[str, gguf.TensorInfo] = {}
        for _ in range(n_tensors):
            name = r.key_str()
            n_dims = r.u32()
            dims = tuple(int(r.u64()) for _ in range(n_dims))
            ggml_type = r.u32()
            offset = r.u64()
            tensors[name] = gguf.TensorInfo(
                name=name,
                dims=dims,
                ggml_type=int(ggml_type),
                offset=int(offset),
            )

        arch = str(meta.get("general.architecture", "llama")).lower()

        def meta_int(key: str) -> Optional[int]:
            v = meta.get(key)
            if v is None:
                return None
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, (int,)):
                return int(v)
            return None

        def meta_float(key: str) -> Optional[float]:
            v = meta.get(key)
            if v is None:
                return None
            if isinstance(v, (float,)):
                return float(v)
            if isinstance(v, (int,)):
                return float(v)
            return None

        def meta_int_arch(suffix: str) -> Optional[int]:
            prefixes = (arch, "llama", "qwen2", "qwen")
            seen = set()
            for prefix in prefixes:
                if not prefix or prefix in seen:
                    continue
                seen.add(prefix)
                value = meta_int(f"{prefix}.{suffix}")
                if value is not None:
                    return value
            return None

        def meta_float_arch(suffix: str) -> Optional[float]:
            prefixes = (arch, "llama", "qwen2", "qwen")
            seen = set()
            for prefix in prefixes:
                if not prefix or prefix in seen:
                    continue
                seen.add(prefix)
                value = meta_float(f"{prefix}.{suffix}")
                if value is not None:
                    return value
            return None

        tok_name = "token_embd.weight"
        tok = tensors.get(tok_name)
        if tok is None:
            raise gguf.GGUFError(f"Missing required tensor: {tok_name}")

        embed_dim = meta_int_arch("embedding_length") or tok.ne0
        vocab_size = tok.ne1
        num_layers = meta_int_arch("block_count")
        if num_layers is None:
            layer_ids = []
            for name in tensors:
                if name.startswith("blk.") and ".attn_norm.weight" in name:
                    try:
                        layer_ids.append(int(name.split(".")[1]))
                    except Exception:
                        pass
            if not layer_ids:
                raise gguf.GGUFError("Could not infer num_layers")
            num_layers = max(layer_ids) + 1

        intermediate = meta_int_arch("feed_forward_length")
        if intermediate is None:
            gate0 = tensors.get("blk.0.ffn_gate.weight")
            if gate0 and len(gate0.dims) == 2:
                intermediate = gate0.ne1
        if intermediate is None:
            raise gguf.GGUFError("Could not determine intermediate_size")

        num_heads = meta_int_arch("attention.head_count")
        if num_heads is None:
            raise gguf.GGUFError("Missing attention.head_count (num_heads)")
        num_kv_heads = meta_int_arch("attention.head_count_kv") or num_heads
        head_dim = embed_dim // num_heads

        context_len = meta_int_arch("context_length") or 0
        if context_len <= 0:
            raise gguf.GGUFError("Could not determine context length")

        rope_theta = meta_float_arch("rope.freq_base") or 10000.0
        rms_eps = meta_float_arch("norm_rms_eps") or 1e-5

        if max_layers:
            num_layers = min(int(max_layers), int(num_layers))

        config = {
            "model_type": arch,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "intermediate_dim": intermediate,
            "num_layers": num_layers,
            "vocab_size": vocab_size,
            "max_seq_len": context_len,
            "rope_theta": rope_theta,
            "rms_norm_eps": rms_eps,
            "tie_word_embeddings": "output.weight" not in tensors,
            "dtype": "fp32",
        }

        manifest_entries: List[Dict] = []

        tok_dt = weight_dtype(tok, "token_emb")
        manifest_entries.append({
            "name": "token_emb",
            "dtype": dtype_name(tok_dt),
            "shape": [tok.ne1, tok.ne0],
            "source_dtype": gguf.ggml_type_name(tok.ggml_type),
        })

        for layer in range(num_layers):
            attn_norm = tensors.get(f"blk.{layer}.attn_norm.weight")
            ffn_norm = tensors.get(f"blk.{layer}.ffn_norm.weight")
            if not attn_norm or not ffn_norm:
                raise gguf.GGUFError(f"Layer {layer}: missing norms")

            wq = tensors.get(f"blk.{layer}.attn_q.weight")
            wk = tensors.get(f"blk.{layer}.attn_k.weight")
            wv = tensors.get(f"blk.{layer}.attn_v.weight")
            wo = tensors.get(f"blk.{layer}.attn_output.weight")
            gate = tensors.get(f"blk.{layer}.ffn_gate.weight")
            up = tensors.get(f"blk.{layer}.ffn_up.weight")
            down = tensors.get(f"blk.{layer}.ffn_down.weight")
            if not wq or not wk or not wv or not wo:
                raise gguf.GGUFError(f"Layer {layer}: missing attention tensors")
            if not gate or not up or not down:
                raise gguf.GGUFError(f"Layer {layer}: missing ffn tensors")

            gate_dt = weight_dtype(gate, "ffn_gate")
            up_dt = weight_dtype(up, "ffn_up")
            if gate_dt != up_dt:
                raise gguf.GGUFError(
                    f"Layer {layer}: ffn_gate and ffn_up dtype mismatch"
                )

            manifest_entries.extend([
                {"name": f"layer.{layer}.ln1_gamma", "dtype": "fp32", "shape": [embed_dim]},
                {"name": f"layer.{layer}.wq", "dtype": dtype_name(weight_dtype(wq, "attn_q")),
                 "shape": [wq.ne1, wq.ne0], "source_dtype": gguf.ggml_type_name(wq.ggml_type)},
                {"name": f"layer.{layer}.wk", "dtype": dtype_name(weight_dtype(wk, "attn_k")),
                 "shape": [wk.ne1, wk.ne0], "source_dtype": gguf.ggml_type_name(wk.ggml_type)},
                {"name": f"layer.{layer}.wv", "dtype": dtype_name(weight_dtype(wv, "attn_v")),
                 "shape": [wv.ne1, wv.ne0], "source_dtype": gguf.ggml_type_name(wv.ggml_type)},
                {"name": f"layer.{layer}.wo", "dtype": dtype_name(weight_dtype(wo, "attn_output")),
                 "shape": [wo.ne1, wo.ne0], "source_dtype": gguf.ggml_type_name(wo.ggml_type)},
                {"name": f"layer.{layer}.ln2_gamma", "dtype": "fp32", "shape": [embed_dim]},
                {"name": f"layer.{layer}.w1", "dtype": dtype_name(gate_dt),
                 "shape": [2 * intermediate, embed_dim], "source_dtype": gguf.ggml_type_name(gate.ggml_type)},
                {"name": f"layer.{layer}.w2", "dtype": dtype_name(weight_dtype(down, "ffn_down")),
                 "shape": [embed_dim, intermediate], "source_dtype": gguf.ggml_type_name(down.ggml_type)},
            ])

        manifest_entries.append({"name": "final_ln_weight", "dtype": "fp32", "shape": [embed_dim]})
        out_weight = tensors.get("output.weight")
        if out_weight is not None:
            out_dt = weight_dtype(out_weight, "lm_head")
            manifest_entries.append({
                "name": "lm_head_weight",
                "dtype": dtype_name(out_dt),
                "shape": [out_weight.ne1, out_weight.ne0],
                "source_dtype": gguf.ggml_type_name(out_weight.ggml_type),
            })

        return config, manifest_entries


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect weights and emit v6 manifest (no conversion).")
    ap.add_argument("--gguf", help="GGUF file to inspect")
    ap.add_argument("--checkpoint", help="HF checkpoint directory (with safetensors)")
    ap.add_argument("--config-out", help="Optional config JSON output path")
    ap.add_argument("--manifest-out", required=True, help="Manifest output JSON path")
    ap.add_argument("--max-layers", type=int, help="Limit number of layers for manifest")
    args = ap.parse_args()

    if bool(args.gguf) == bool(args.checkpoint):
        raise SystemExit("Specify exactly one of --gguf or --checkpoint.")

    if args.gguf:
        config, entries = inspect_gguf(Path(args.gguf), args.max_layers)
    else:
        model_dir = Path(args.checkpoint)
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise SystemExit(f"config.json not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        st_entries = load_safetensors_entries(model_dir)
        entries = build_hf_manifest(config, st_entries, args.max_layers)

    manifest = {
        "version": 5,
        "source": "gguf" if args.gguf else "safetensors",
        "entries": entries,
    }

    out_path = Path(args.manifest_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if args.config_out:
        config_path = Path(args.config_out)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
