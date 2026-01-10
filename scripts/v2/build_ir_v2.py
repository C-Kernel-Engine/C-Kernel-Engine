#!/usr/bin/env python3
"""
build_ir_v2.py
==============

Generate IR v2 JSON from config.json + (optional) weights headers.
Supports safetensors (local or HTTP range) and GGUF (local or HTTP range).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_DIR = os.path.join(ROOT, "kernel_maps", "kernels")
PLAN_PATH = os.path.join(ROOT, "kernel_maps", "decoder_layer_plan.json")
PLAN_BWD_PATH = os.path.join(ROOT, "kernel_maps", "decoder_layer_plan_backward.json")
GLOBAL_PATH = os.path.join(ROOT, "kernel_maps", "global_buffers.json")

CK_DIM = {
    "tokens": 0,
    "embed": 1,
    "aligned_embed": 2,
    "head_dim": 3,
    "aligned_head": 4,
    "num_heads": 5,
    "num_kv_heads": 6,
    "aligned_ctx": 7,
    "intermediate": 8,
    "aligned_intermediate": 9,
    "vocab": 10,
    "end": 11,
}
CK_DIM_ID_TO_NAME = {v: k for k, v in CK_DIM.items()}
DEFAULT_ALIGN_BYTES = 64


GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14
GGML_TYPE_BF16 = 16

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


@dataclass
class KernelSpec:
    name: str
    forward: Optional[str]
    backward: Optional[str]
    variants: Dict[str, Dict[str, str]]
    buffers: List[Dict]
    buffers_backward: List[Dict]


def is_url(path: str) -> bool:
    if not path:
        return False
    return urllib.parse.urlparse(path).scheme in ("http", "https")


def read_json(path: str) -> Dict:
    if is_url(path):
        with urllib.request.urlopen(path) as resp:
            data = resp.read()
        return json.loads(data.decode("utf-8"))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_hf_repo(value: str) -> str:
    if not value:
        return ""
    if is_url(value):
        parsed = urllib.parse.urlparse(value)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return ""
    return value


def hf_base_url(repo_id: str, revision: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/"


def hf_list_siblings(repo_id: str) -> List[str]:
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    with urllib.request.urlopen(api_url) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    siblings = data.get("siblings", [])
    return [s.get("rfilename") for s in siblings if isinstance(s, dict) and s.get("rfilename")]


def pick_weights_file(siblings: List[str]) -> Optional[str]:
    if not siblings:
        return None
    index_files = [s for s in siblings if s.endswith(".safetensors.index.json")]
    if "model.safetensors.index.json" in index_files:
        return "model.safetensors.index.json"
    if index_files:
        return sorted(index_files, key=lambda s: (len(s), s))[0]

    tensor_files = [s for s in siblings if s.endswith(".safetensors")]
    if "model.safetensors" in tensor_files:
        return "model.safetensors"
    if tensor_files:
        return sorted(tensor_files, key=lambda s: (len(s), s))[0]

    gguf_files = [s for s in siblings if s.endswith(".gguf")]
    if gguf_files:
        preferred = [s for s in gguf_files if "q4_k_m" in s.lower()]
        if preferred:
            return sorted(preferred, key=lambda s: (len(s), s))[0]
        return sorted(gguf_files, key=lambda s: (len(s), s))[0]
    return None


def resolve_hf_path(base: str, path: str) -> str:
    if not path:
        return path
    if is_url(path) or os.path.exists(path):
        return path
    return urllib.parse.urljoin(base, path)


def find_first(obj, keys: Iterable[str]):
    sentinel = object()

    def walk(node):
        if isinstance(node, dict):
            for key in keys:
                if key in node:
                    return node[key]
            for value in node.values():
                got = walk(value)
                if got is not sentinel:
                    return got
        elif isinstance(node, list):
            for value in node:
                got = walk(value)
                if got is not sentinel:
                    return got
        return sentinel

    result = walk(obj)
    return None if result is sentinel else result


def normalize_dtype_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = str(value).lower()
    if key in ("fp32", "float32", "f32"):
        return "fp32"
    if key in ("fp16", "float16", "f16"):
        return "fp16"
    if key in ("bf16", "bfloat16"):
        return "bf16"
    if key in ("q4_0", "q4_0_ggml"):
        return "q4_0"
    if key in ("q4_k", "q4_k_m", "q4_k_s"):
        return "q4_k"
    if key in ("q6_k",):
        return "q6_k"
    if key in ("q8_0", "q8_0_ggml"):
        return "q8_0"
    return None


def load_config(path: str) -> Dict:
    raw = read_json(path)
    text_cfg = raw.get("text_config") or raw
    cfg = {
        "num_layers": int(find_first(text_cfg, ["num_hidden_layers", "n_layer"]) or 0),
        "hidden_size": int(find_first(text_cfg, ["hidden_size", "n_embd", "d_model"]) or 0),
        "intermediate_size": int(find_first(text_cfg, ["intermediate_size", "n_inner", "ffn_dim", "mlp_dim"]) or 0),
        "num_attention_heads": int(find_first(text_cfg, ["num_attention_heads", "n_head", "num_heads"]) or 0),
        "num_kv_heads": int(find_first(text_cfg, ["num_key_value_heads", "num_kv_heads"]) or 0),
        "vocab_size": int(find_first(text_cfg, ["vocab_size", "n_vocab"]) or 0),
        "context_window": int(find_first(text_cfg, ["max_position_embeddings", "n_positions", "context_length", "seq_len"]) or 0),
        "rms_norm_eps": float(find_first(text_cfg, ["rms_norm_eps", "layer_norm_eps"]) or 1e-5),
        "rope_theta": float(find_first(text_cfg, ["rope_theta", "rope_base"]) or 0.0),
    }
    if cfg["num_kv_heads"] <= 0:
        cfg["num_kv_heads"] = cfg["num_attention_heads"]

    dtype = normalize_dtype_name(find_first(raw, ["dtype", "torch_dtype"]))
    cfg["dtype"] = dtype

    tie = find_first(raw, ["tie_word_embeddings"])
    cfg["tie_word_embeddings"] = bool(tie) if tie is not None else None

    hidden_act = find_first(text_cfg, ["hidden_act", "activation"])
    cfg["hidden_act"] = str(hidden_act) if hidden_act is not None else None

    return cfg


def norm_shape(shape: List[Dict]) -> List[Dict]:
    tokens = []
    for item in shape:
        if isinstance(item, str):
            dim_name = item
            mult = 1
            div = 1
        else:
            dim = item.get("dim")
            if isinstance(dim, int):
                dim_name = CK_DIM_ID_TO_NAME.get(dim)
            else:
                dim_name = dim
            mult = int(item.get("mult", 1))
            div = int(item.get("div", 1))
        if dim_name not in CK_DIM:
            raise ValueError(f"unknown dim '{dim_name}' in shape")
        if mult <= 0 or div <= 0:
            raise ValueError(f"invalid mult/div for dim '{dim_name}': {mult}/{div}")
        tokens.append({
            "dim": dim_name,
            "dim_id": CK_DIM[dim_name],
            "mult": mult,
            "div": div,
        })
    return tokens


def align_up_bytes(n: int, align: int) -> int:
    if align <= 0:
        return n
    return (n + align - 1) & ~(align - 1)


def align_up_elems(elems: int, elem_bytes: int, align_bytes: int) -> int:
    if elems <= 0:
        return 0
    bytes_total = elems * elem_bytes
    bytes_total = align_up_bytes(bytes_total, align_bytes)
    return bytes_total // elem_bytes


def compute_align(cfg: Dict, elem_bytes: int = 4, align_bytes: int = DEFAULT_ALIGN_BYTES) -> Dict:
    hidden = int(cfg.get("hidden_size") or 0)
    heads = int(cfg.get("num_attention_heads") or 0)
    head_dim = hidden // heads if heads else 0
    return {
        "aligned_embed": align_up_elems(hidden, elem_bytes, align_bytes),
        "aligned_head": align_up_elems(head_dim, elem_bytes, align_bytes),
        "aligned_intermediate": align_up_elems(int(cfg.get("intermediate_size") or 0), elem_bytes, align_bytes),
        "aligned_ctx": align_up_elems(int(cfg.get("context_window") or 0), elem_bytes, align_bytes),
    }


def resolve_dim(name: str, cfg: Dict, align: Dict) -> int:
    if name == "tokens":
        return int(cfg.get("context_window") or 0)
    if name == "embed":
        return int(cfg.get("hidden_size") or 0)
    if name == "aligned_embed":
        return int(align.get("aligned_embed") or 0)
    if name == "head_dim":
        hidden = int(cfg.get("hidden_size") or 0)
        heads = int(cfg.get("num_attention_heads") or 0)
        return hidden // heads if heads else 0
    if name == "aligned_head":
        return int(align.get("aligned_head") or 0)
    if name == "num_heads":
        return int(cfg.get("num_attention_heads") or 0)
    if name == "num_kv_heads":
        return int(cfg.get("num_kv_heads") or 0)
    if name == "aligned_ctx":
        return int(align.get("aligned_ctx") or 0)
    if name == "intermediate":
        return int(cfg.get("intermediate_size") or 0)
    if name == "aligned_intermediate":
        return int(align.get("aligned_intermediate") or 0)
    if name == "vocab":
        return int(cfg.get("vocab_size") or 0)
    return 0


def resolve_shape(shape: List[Dict], cfg: Dict, align: Dict) -> List[int]:
    resolved = []
    for token in shape:
        dim_name = token.get("dim")
        mult = int(token.get("mult", 1))
        div = int(token.get("div", 1))
        base = resolve_dim(dim_name, cfg, align)
        if div <= 0:
            div = 1
        resolved.append(int(base * mult // div))
    return resolved


def build_dimensions_table(cfg: Dict, align: Dict) -> List[Dict]:
    dims = []
    for name, dim_id in sorted(CK_DIM.items(), key=lambda kv: kv[1]):
        if name == "end":
            continue
        dims.append({
            "id": dim_id,
            "name": name,
            "value": resolve_dim(name, cfg, align),
        })
    return dims


def merge_role(prev_role: str, new_role: str) -> str:
    if prev_role == new_role:
        return prev_role
    role_rank = {
        "input": 0,
        "output": 1,
        "activation": 2,
        "scratch": 2,
        "grad": 3,
        "weight": 4,
    }
    if prev_role not in role_rank or new_role not in role_rank:
        raise ValueError(f"unknown role merge: {prev_role} vs {new_role}")
    return prev_role if role_rank[prev_role] >= role_rank[new_role] else new_role


def add_buffer(buffers: Dict, order: List[str], name: str, spec: Dict) -> None:
    if name in buffers:
        prev = buffers[name]
        if prev["scope"] != spec["scope"]:
            raise ValueError(f"buffer scope mismatch for '{name}'")
        if prev["shape"] != spec["shape"]:
            raise ValueError(f"buffer shape mismatch for '{name}'")
        prev["role"] = merge_role(prev["role"], spec["role"])
        prev["optional"] = bool(prev["optional"] or spec["optional"])
        if not prev["alias_of"]:
            prev["alias_of"] = spec["alias_of"]
        if not prev["condition"]:
            prev["condition"] = spec["condition"]
        return
    buffers[name] = spec
    order.append(name)


def load_kernel_specs() -> Dict[str, KernelSpec]:
    kernels = {}
    for name in sorted(os.listdir(KERNEL_DIR)):
        if not name.endswith(".json"):
            continue
        data = read_json(os.path.join(KERNEL_DIR, name))
        impl = data.get("impl", {}) or {}
        kernels[data["name"]] = KernelSpec(
            name=data["name"],
            forward=impl.get("forward"),
            backward=impl.get("backward"),
            variants=impl.get("variants", {}) or {},
            buffers=data.get("buffers") or [],
            buffers_backward=data.get("buffers_backward") or [],
        )
    return kernels


def select_kernel_impl(spec: KernelSpec, dtype: str, backward: bool) -> str:
    impl = spec.backward if backward else spec.forward
    variants = spec.variants or {}
    variant = variants.get(dtype, {})
    name = variant.get("backward" if backward else "forward") or impl
    return name or spec.name


def build_buffers(kernels: Dict[str, KernelSpec]) -> Tuple[List[Dict], Dict[str, Dict]]:
    plan = read_json(PLAN_PATH)
    plan_bwd = read_json(PLAN_BWD_PATH)
    globals_spec = read_json(GLOBAL_PATH)

    buffers: Dict[str, Dict] = {}
    order: List[str] = []

    for buf in globals_spec.get("buffers", []):
        name = buf.get("name")
        if not name:
            raise ValueError("global buffer missing name")
        spec = {
            "name": name,
            "scope": buf.get("scope", "global"),
            "role": buf.get("role", "activation"),
            "shape": norm_shape(buf.get("shape", [])),
            "optional": bool(buf.get("optional", False)),
            "alias_of": buf.get("alias_of"),
            "condition": buf.get("condition"),
        }
        add_buffer(buffers, order, name, spec)

    def merge_plan_buffers(plan_obj: Dict, plan_kind: str) -> None:
        for step in plan_obj.get("steps", []):
            kernel_name = step.get("kernel")
            if kernel_name not in kernels:
                raise ValueError(f"unknown kernel in plan: {kernel_name}")
            kernel = kernels[kernel_name]
            bindings = step.get("bind", {})
            buf_list = kernel.buffers_backward if plan_kind == "backward" else kernel.buffers
            for buf in buf_list:
                buf_id = buf.get("id")
                if not buf_id:
                    raise ValueError(f"kernel '{kernel_name}' buffer missing id")
                if buf_id not in bindings and not buf.get("optional"):
                    raise ValueError(f"kernel '{kernel_name}' missing bind for '{buf_id}'")
                name = bindings.get(buf_id)
                if not name:
                    continue
                spec = {
                    "name": name,
                    "scope": buf.get("scope", "layer"),
                    "role": buf.get("role", "activation"),
                    "shape": norm_shape(buf.get("shape", [])),
                    "optional": bool(buf.get("optional", False)),
                    "alias_of": buf.get("alias_of"),
                    "condition": buf.get("condition"),
                }
                add_buffer(buffers, order, name, spec)

    merge_plan_buffers(plan, "forward")
    merge_plan_buffers(plan_bwd, "backward")

    buffer_list = [buffers[name] for name in order]
    return buffer_list, buffers


def plan_nodes(kernels: Dict[str, KernelSpec], config: Dict, dtype: str, backward: bool) -> List[Dict]:
    plan = read_json(PLAN_BWD_PATH if backward else PLAN_PATH)
    steps = plan.get("steps", [])
    nodes = []
    for layer in range(int(config["num_layers"] or 0)):
        for step in steps:
            kernel_name = step.get("kernel")
            kernel = kernels[kernel_name]
            variant = select_kernel_impl(kernel, dtype, backward)
            node = {
                "layer": layer,
                "op": kernel_name,
                "kernel": variant,
                "kernel_variant": variant,
                "kernel_dtype": dtype,
                "flags": 0,
                "condition": step.get("when") or step.get("condition"),
                "bindings": [],
            }
            binds = step.get("bind", {})
            buf_list = kernel.buffers_backward if backward else kernel.buffers
            for buf in buf_list:
                arg = buf.get("id")
                if not arg:
                    continue
                if arg not in binds:
                    if buf.get("optional"):
                        continue
                    continue
                node["bindings"].append({"arg": arg, "buffer": binds[arg]})
            nodes.append(node)
    return nodes


def http_range_read(url: str, start: int, size: int) -> bytes:
    end = start + size - 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def read_safetensors_header(path: str) -> Dict[str, Dict]:
    if is_url(path):
        head = http_range_read(path, 0, 8)
        header_len = struct.unpack("<Q", head)[0]
        header = http_range_read(path, 8, int(header_len))
    else:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = f.read(int(header_len))
    data = json.loads(header.decode("utf-8"))
    tensors = {}
    for name, info in data.items():
        if name == "__metadata__":
            continue
        tensors[name] = info
    return tensors


def read_safetensors_index(path: str, fetch_shards: bool) -> Dict[str, Dict]:
    index = read_json(path)
    weight_map = index.get("weight_map", {})
    tensors = {name: {"dtype": None, "shape": None} for name in weight_map.keys()}
    if not fetch_shards:
        return tensors
    shard_files = sorted(set(weight_map.values()))
    base = path
    if is_url(path):
        base = urllib.parse.urljoin(path, ".")
    else:
        base = os.path.dirname(path)
    for shard in shard_files:
        shard_path = urllib.parse.urljoin(base + "/", shard) if is_url(path) else os.path.join(base, shard)
        try:
            header = read_safetensors_header(shard_path)
        except Exception:
            continue
        for name, info in header.items():
            tensors[name] = info
    return tensors


class RangeReader:
    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self.size = None

    def read(self, n: int) -> bytes:
        if n <= 0:
            return b""
        start = self.pos
        end = start + n - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            crange = resp.headers.get("Content-Range")
            if crange and "/" in crange:
                try:
                    self.size = int(crange.split("/")[-1])
                except Exception:
                    pass
        if len(data) != n:
            raise IOError(f"short read: wanted {n} bytes, got {len(data)}")
        self.pos += len(data)
        return data

    def seek(self, pos: int, whence: int = 0) -> None:
        if whence == 0:
            self.pos = pos
        elif whence == 1:
            self.pos += pos
        elif whence == 2:
            if self.size is None:
                raise IOError("cannot seek from end without size")
            self.pos = self.size + pos
        else:
            raise ValueError("invalid whence")

    def tell(self) -> int:
        return self.pos


class GGUFReader:
    def __init__(self, path: str):
        if is_url(path):
            self._f = RangeReader(path)
            self._close = None
        else:
            self._f = open(path, "rb")
            self._close = self._f.close

    def close(self) -> None:
        if self._close:
            self._close()

    def read(self, n: int) -> bytes:
        return self._f.read(n)

    def tell(self) -> int:
        return self._f.tell()

    def seek(self, pos: int) -> None:
        self._f.seek(pos)

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def key_str(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8")


def gguf_scalar_size(vtype: int) -> Optional[int]:
    return {
        GGUF_TYPE_UINT8: 1,
        GGUF_TYPE_INT8: 1,
        GGUF_TYPE_UINT16: 2,
        GGUF_TYPE_INT16: 2,
        GGUF_TYPE_UINT32: 4,
        GGUF_TYPE_INT32: 4,
        GGUF_TYPE_FLOAT32: 4,
        GGUF_TYPE_BOOL: 1,
        GGUF_TYPE_UINT64: 8,
        GGUF_TYPE_INT64: 8,
        GGUF_TYPE_FLOAT64: 8,
    }.get(vtype)


def gguf_skip_value(r: GGUFReader, vtype: int) -> None:
    if vtype == GGUF_TYPE_STRING:
        n = r.u64()
        r.read(n)
        return
    if vtype == GGUF_TYPE_ARRAY:
        elem_type = r.u32()
        count = r.u64()
        if elem_type == GGUF_TYPE_STRING:
            for _ in range(count):
                n = r.u64()
                r.read(n)
            return
        size = gguf_scalar_size(elem_type)
        if size is None:
            raise ValueError(f"unknown GGUF array element type {elem_type}")
        r.read(int(count) * size)
        return
    size = gguf_scalar_size(vtype)
    if size is None:
        raise ValueError(f"unknown GGUF value type {vtype}")
    r.read(size)


def read_gguf_header(path: str) -> Dict[str, int]:
    r = GGUFReader(path)
    try:
        magic = r.read(4)
        if magic != b"GGUF":
            raise ValueError("invalid GGUF magic")
        _version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()

        for _ in range(n_kv):
            _key = r.key_str()
            vtype = r.u32()
            gguf_skip_value(r, vtype)

        tensors = {}
        for _ in range(n_tensors):
            name = r.key_str()
            n_dims = r.u32()
            for _ in range(n_dims):
                r.u64()
            ggml_type = r.u32()
            r.u64()  # offset
            tensors[name] = int(ggml_type)
        return tensors
    finally:
        r.close()


def infer_family(names: Iterable[str]) -> str:
    names = list(names)
    if any(n.startswith("blk.") for n in names) and "token_embd.weight" in names:
        return "gguf_llama"
    if any("model.layers.0.self_attn.q_proj.weight" in n for n in names):
        return "hf_llama"
    if any("transformer.h.0.attn.c_attn.weight" in n for n in names):
        return "gpt2"
    return "unknown"


def map_dtype(value, fmt: str) -> Optional[str]:
    if value is None:
        return None
    if fmt == "gguf":
        if value == GGML_TYPE_F32:
            return "fp32"
        if value == GGML_TYPE_F16:
            return "fp16"
        if value == GGML_TYPE_BF16:
            return "bf16"
        if value == GGML_TYPE_Q4_0:
            return "q4_0"
        if value == GGML_TYPE_Q4_K:
            return "q4_k"
        if value == GGML_TYPE_Q6_K:
            return "q6_k"
        if value == GGML_TYPE_Q8_0:
            return "q8_0"
        return None
    return normalize_dtype_name(value)


def find_weight_dtype(names_to_dtype: Dict[str, object], fmt: str, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in names_to_dtype:
            return map_dtype(names_to_dtype[name], fmt)
    return None


def infer_weight_dtypes(names_to_dtype: Dict[str, object], fmt: str) -> Tuple[Dict[str, str], Dict[str, bool]]:
    names = set(names_to_dtype.keys())
    family = infer_family(names)

    gguf_map = {
        "token_emb": ["token_embd.weight"],
        "lm_head_weight": ["output.weight"],
        "final_ln_weight": ["output_norm.weight"],
        "ln1_gamma": ["blk.0.attn_norm.weight"],
        "ln2_gamma": ["blk.0.ffn_norm.weight"],
        "wq": ["blk.0.attn_q.weight"],
        "wk": ["blk.0.attn_k.weight"],
        "wv": ["blk.0.attn_v.weight"],
        "wo": ["blk.0.attn_output.weight"],
        "w1": ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"],
        "w2": ["blk.0.ffn_down.weight"],
        "bq": ["blk.0.attn_q.bias"],
        "bk": ["blk.0.attn_k.bias"],
        "bv": ["blk.0.attn_v.bias"],
        "bo": ["blk.0.attn_output.bias"],
        "b1": ["blk.0.ffn_gate.bias", "blk.0.ffn_up.bias"],
        "b2": ["blk.0.ffn_down.bias"],
    }

    hf_map = {
        "token_emb": ["model.embed_tokens.weight", "model.tok_embeddings.weight"],
        "lm_head_weight": ["lm_head.weight", "model.lm_head.weight"],
        "final_ln_weight": ["model.norm.weight"],
        "ln1_gamma": ["model.layers.0.input_layernorm.weight"],
        "ln2_gamma": ["model.layers.0.post_attention_layernorm.weight"],
        "wq": ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.attention.q_proj.weight"],
        "wk": ["model.layers.0.self_attn.k_proj.weight", "model.layers.0.attention.k_proj.weight"],
        "wv": ["model.layers.0.self_attn.v_proj.weight", "model.layers.0.attention.v_proj.weight"],
        "wo": ["model.layers.0.self_attn.o_proj.weight", "model.layers.0.attention.o_proj.weight"],
        "w1": ["model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.w1.weight"],
        "w2": ["model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.w2.weight"],
        "bq": ["model.layers.0.self_attn.q_proj.bias", "model.layers.0.attention.q_proj.bias"],
        "bk": ["model.layers.0.self_attn.k_proj.bias", "model.layers.0.attention.k_proj.bias"],
        "bv": ["model.layers.0.self_attn.v_proj.bias", "model.layers.0.attention.v_proj.bias"],
        "bo": ["model.layers.0.self_attn.o_proj.bias", "model.layers.0.attention.o_proj.bias"],
        "b1": ["model.layers.0.mlp.gate_proj.bias", "model.layers.0.mlp.w1.bias"],
        "b2": ["model.layers.0.mlp.down_proj.bias", "model.layers.0.mlp.w2.bias"],
    }

    gpt2_map = {
        "token_emb": ["transformer.wte.weight"],
        "lm_head_weight": ["lm_head.weight"],
        "final_ln_weight": ["transformer.ln_f.weight"],
        "ln1_gamma": ["transformer.h.0.ln_1.weight"],
        "ln2_gamma": ["transformer.h.0.ln_2.weight"],
        "wq": ["transformer.h.0.attn.c_attn.weight"],
        "wk": ["transformer.h.0.attn.c_attn.weight"],
        "wv": ["transformer.h.0.attn.c_attn.weight"],
        "wo": ["transformer.h.0.attn.c_proj.weight"],
        "w1": ["transformer.h.0.mlp.c_fc.weight"],
        "w2": ["transformer.h.0.mlp.c_proj.weight"],
        "bq": ["transformer.h.0.attn.c_attn.bias"],
        "bk": ["transformer.h.0.attn.c_attn.bias"],
        "bv": ["transformer.h.0.attn.c_attn.bias"],
        "bo": ["transformer.h.0.attn.c_proj.bias"],
        "b1": ["transformer.h.0.mlp.c_fc.bias"],
        "b2": ["transformer.h.0.mlp.c_proj.bias"],
    }

    if family == "gguf_llama":
        mapping = gguf_map
    elif family == "hf_llama":
        mapping = hf_map
    else:
        mapping = gpt2_map

    dtype_map = {}
    for buf, candidates in mapping.items():
        dtype = find_weight_dtype(names_to_dtype, fmt, candidates)
        if dtype:
            dtype_map[buf] = dtype

    fused_qkv = any(name in names for name in [
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.attention.qkv_proj.weight",
        "transformer.h.0.attn.c_attn.weight",
    ])
    gated_mlp = any(name in names for name in [
        "model.layers.0.mlp.gate_proj.weight",
        "blk.0.ffn_gate.weight",
    ])
    has_pos_emb = any(name in names for name in [
        "model.embed_positions.weight",
        "transformer.wpe.weight",
        "position_embeddings.weight",
    ])
    has_lm_head = any(name in names for name in mapping.get("lm_head_weight", []))

    meta = {
        "family": family,
        "fused_qkv": fused_qkv,
        "gated_mlp": gated_mlp,
        "has_pos_emb": has_pos_emb,
        "has_lm_head": has_lm_head,
    }
    return dtype_map, meta


def build_ir(config: Dict,
             buffers: List[Dict],
             kernels: Dict[str, KernelSpec],
             kernel_dtype: str,
             activation_dtype: str,
             weight_dtypes: Dict[str, str],
             tie_word_embeddings: Optional[bool],
             backward: bool,
             extra_meta: Dict) -> Dict:
    config_out = {
        "num_layers": int(config.get("num_layers") or 0),
        "hidden_size": int(config.get("hidden_size") or 0),
        "intermediate_size": int(config.get("intermediate_size") or 0),
        "num_attention_heads": int(config.get("num_attention_heads") or 0),
        "num_kv_heads": int(config.get("num_kv_heads") or 0),
        "vocab_size": int(config.get("vocab_size") or 0),
        "context_window": int(config.get("context_window") or 0),
        "rms_norm_eps": float(config.get("rms_norm_eps") or 1e-5),
        "rope_theta": float(config.get("rope_theta") or 0.0),
    }

    if config_out["num_kv_heads"] <= 0:
        config_out["num_kv_heads"] = config_out["num_attention_heads"]

    align = compute_align(config_out)
    dimensions = build_dimensions_table(config_out, align)
    notes = [
        "=== IR V2 FORMAT GUIDE ===",
        "",
        "DIMENSIONS: The 'dimensions' array maps numeric IDs to named values from config.json:",
        "  - To resolve shape.dim, look up dimensions[dim_id].value",
        "  - Example: dim:10 -> dimensions[10] -> 'vocab' -> 151936",
        "",
        "BUFFERS: Each buffer has:",
        "  - shape: symbolic dimensions [{dim, mult, div}, ...] where size = dim_value * mult / div",
        "  - resolved_shape: concrete byte sizes after applying alignment",
        "  - role: 'weight' (model params), 'activation' (runtime), or 'grad' (backward pass)",
        "  - scope: 'global' (shared) or 'layer' (per-layer)",
        "",
        "NODES: Each node is one kernel invocation:",
        "  - layer: which transformer layer (0-indexed), or -1 for global ops",
        "  - kernel: the C function to call",
        "  - bindings: maps kernel args to buffer names",
        "",
        f"ALIGNMENT: resolved_shape uses alignment_bytes={DEFAULT_ALIGN_BYTES}, elem_bytes=4 (fp32)",
    ]

    ir_buffers = []
    for buf in buffers:
        dtype = activation_dtype
        if buf["role"] == "weight":
            dtype = weight_dtypes.get(buf["name"], dtype)
        if buf["role"] == "grad":
            dtype = "fp32"
        condition = buf.get("condition")
        alias_of = buf.get("alias_of")
        if buf["name"] == "lm_head_weight" and tie_word_embeddings is not None:
            alias_of = "token_emb" if tie_word_embeddings else None
        if buf["name"] == "pos_emb":
            condition = "rope_disabled"
        ir_buffers.append({
            "name": buf["name"],
            "scope": buf["scope"],
            "role": buf["role"],
            "dtype": dtype or "fp32",
            "optional": 1 if buf.get("optional") else 0,
            "shape": buf["shape"],
            "resolved_shape": resolve_shape(buf["shape"], config_out, align),
            "alias_of": alias_of,
            "condition": condition,
        })

    nodes = plan_nodes(kernels, config_out, kernel_dtype, backward)
    ir = {
        "version": 2,
        "notes": notes,
        "config": config_out,
        "dimensions": dimensions,
        "buffers": ir_buffers,
        "nodes": nodes,
    }
    if extra_meta:
        ir["meta"] = extra_meta
    return ir


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate IR v2 JSON from config + weights headers.")
    ap.add_argument("--config", help="Path/URL to config.json")
    ap.add_argument("--hf", help="Hugging Face repo id or URL (auto-find config/weights)")
    ap.add_argument("--revision", default="main", help="Hugging Face revision (default: main)")
    ap.add_argument("--weights", help="Path/URL to safetensors/gguf or safetensors index json")
    ap.add_argument("--out", default=os.path.join("build", "ir_v2.json"), help="Output IR JSON path")
    ap.add_argument("--meta-out", help="Write weights meta JSON to path")
    ap.add_argument("--meta-only", action="store_true", help="Only emit weights meta JSON (skip IR)")
    ap.add_argument("--cache-dir", help="Cache remote config/index to this directory")
    ap.add_argument("--ctx", type=int, default=0, help="Override context_window")
    ap.add_argument("--kernel-dtype", default="fp32", help="Kernel dtype for variant selection (fp32/bf16/fp16)")
    ap.add_argument("--activation-dtype", default="fp32", help="Activation dtype (default fp32)")
    ap.add_argument("--weight-dtype", default=None, help="Fallback weight dtype when header data is missing")
    ap.add_argument("--no-shard-headers", action="store_true", help="Skip reading safetensors shard headers")
    ap.add_argument("--backward", action="store_true", help="Emit backward plan nodes instead of forward")
    args = ap.parse_args()

    config_path = args.config
    weights_path = args.weights
    config_url = None
    weights_url = None

    if args.hf:
        repo_id = parse_hf_repo(args.hf)
        if not repo_id:
            raise SystemExit("invalid --hf repo or URL")
        base_url = hf_base_url(repo_id, args.revision)
        if not config_path:
            config_path = urllib.parse.urljoin(base_url, "config.json")
            config_url = config_path
        if not weights_path:
            try:
                siblings = hf_list_siblings(repo_id)
                pick = pick_weights_file(siblings)
            except Exception:
                pick = None
            if pick:
                weights_path = urllib.parse.urljoin(base_url, pick)
                weights_url = weights_path
        else:
            weights_path = resolve_hf_path(base_url, weights_path)
            if is_url(weights_path):
                weights_url = weights_path

    if not config_path:
        raise SystemExit("--config or --hf is required")

    if is_url(config_path) and args.cache_dir:
        data = urllib.request.urlopen(config_path).read()
        os.makedirs(args.cache_dir, exist_ok=True)
        local_name = os.path.basename(urllib.parse.urlparse(config_path).path) or "config.json"
        local_path = os.path.join(args.cache_dir, local_name)
        with open(local_path, "wb") as f:
            f.write(data)
        config_path = local_path
    if not config_url and is_url(args.config or ""):
        config_url = args.config
    if not weights_url and is_url(weights_path or ""):
        weights_url = weights_path

    cfg = load_config(config_path)
    if args.ctx and args.ctx > 0:
        cfg["context_window"] = int(args.ctx)

    kernels = load_kernel_specs()
    buffers, _ = build_buffers(kernels)

    weight_dtypes = {}
    meta = {}
    weights_format = None
    if weights_path and is_url(weights_path) and args.cache_dir and weights_path.endswith(".json"):
        data = urllib.request.urlopen(weights_path).read()
        os.makedirs(args.cache_dir, exist_ok=True)
        local_name = os.path.basename(urllib.parse.urlparse(weights_path).path) or "model.safetensors.index.json"
        local_path = os.path.join(args.cache_dir, local_name)
        with open(local_path, "wb") as f:
            f.write(data)
        # Keep the URL for shard header reads; we just cache a copy for convenience.

    if weights_path:
        fmt = None
        names_to_dtype: Dict[str, object] = {}
        if weights_path.endswith(".gguf"):
            fmt = "gguf"
            names_to_dtype = read_gguf_header(weights_path)
        elif weights_path.endswith(".safetensors"):
            fmt = "safetensors"
            header = read_safetensors_header(weights_path)
            names_to_dtype = {name: info.get("dtype") for name, info in header.items()}
        elif weights_path.endswith(".json"):
            fmt = "safetensors"
            header = read_safetensors_index(weights_path, not args.no_shard_headers)
            names_to_dtype = {name: info.get("dtype") for name, info in header.items()}
        if fmt:
            weights_format = fmt
            weight_dtypes, meta = infer_weight_dtypes(names_to_dtype, fmt)

    kernel_dtype = normalize_dtype_name(args.kernel_dtype) or "fp32"
    activation_dtype = normalize_dtype_name(args.activation_dtype) or "fp32"
    fallback_weight_dtype = normalize_dtype_name(args.weight_dtype) or cfg.get("dtype") or "fp32"

    for buf in buffers:
        if buf["role"] == "weight" and buf["name"] not in weight_dtypes:
            weight_dtypes[buf["name"]] = fallback_weight_dtype

    tie_word_embeddings = cfg.get("tie_word_embeddings")
    if meta.get("has_lm_head") is False and meta.get("family") != "unknown":
        tie_word_embeddings = True

    meta_out = {
        "format": weights_format,
        "family": meta.get("family"),
        "fused_qkv": bool(meta.get("fused_qkv")) if "fused_qkv" in meta else None,
        "gated_mlp": bool(meta.get("gated_mlp")) if "gated_mlp" in meta else None,
        "has_pos_emb": bool(meta.get("has_pos_emb")) if "has_pos_emb" in meta else None,
        "has_lm_head": bool(meta.get("has_lm_head")) if "has_lm_head" in meta else None,
        "tie_word_embeddings": tie_word_embeddings,
        "weight_dtypes": weight_dtypes,
        "config_path": config_path,
        "config_url": config_url,
        "weights_path": weights_path,
        "weights_url": weights_url,
    }

    if args.meta_out:
        out_dir = os.path.dirname(args.meta_out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(args.meta_out, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)
        print(f"[build_ir_v2] wrote meta {args.meta_out}")

    if args.meta_only:
        return 0

    ir = build_ir(cfg,
                  buffers,
                  kernels,
                  kernel_dtype,
                  activation_dtype,
                  weight_dtypes,
                  tie_word_embeddings,
                  args.backward,
                  meta)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ir, f, indent=2)
    print(f"[build_ir_v2] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
