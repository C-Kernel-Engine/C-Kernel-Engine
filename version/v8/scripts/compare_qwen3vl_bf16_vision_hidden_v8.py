#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from array import array
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
NUMERIC_PARITY = SCRIPT_DIR / "numeric_parity_qwen3vl_mmproj_v8.py"


def _import_numeric_parity() -> Any:
    sys.path.insert(0, str(SCRIPT_DIR))
    import numeric_parity_qwen3vl_mmproj_v8 as numeric  # type: ignore

    return numeric


def _metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, float | bool]:
    if ref.shape != got.shape:
        raise RuntimeError(f"shape mismatch: ref={ref.shape} got={got.shape}")
    byte_exact = bool(np.array_equal(ref.view(np.uint8), got.view(np.uint8)))
    diff = got.astype(np.float32, copy=False) - ref.astype(np.float32, copy=False)
    denom = float(np.linalg.norm(ref) * np.linalg.norm(got))
    rmse = float(math.sqrt(float(np.mean(diff * diff)))) if diff.size else 0.0
    ref_rms = float(math.sqrt(float(np.mean(ref.astype(np.float32, copy=False) ** 2)))) if ref.size else 0.0
    return {
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": rmse,
        "ref_rms": ref_rms,
        "relative_rmse": rmse / ref_rms if ref_rms > 0.0 else (0.0 if rmse == 0.0 else float("inf")),
        # A float32 norm/dot can round an exact high-dimensional tensor below
        # one. Exact identity is stronger evidence than that derived metric.
        "cosine": 1.0 if byte_exact else (float(np.dot(ref.reshape(-1), got.reshape(-1)) / denom) if denom > 0.0 else 0.0),
        "byte_exact": byte_exact,
    }


def _child_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def _load_visual_model(checkpoint: Path, attn_implementation: str, architecture: str):
    import torch
    from safetensors.torch import load_file

    if architecture == "cohere_compass":
        from transformers.models.cohere_compass.configuration_cohere_compass import CohereCompassVisionConfig
        from transformers.models.cohere_compass.modeling_cohere_compass import CohereCompassVisionModel

        config_type = CohereCompassVisionConfig
        model_type = CohereCompassVisionModel
    else:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        config_type = Qwen3VLVisionConfig
        model_type = Qwen3VLVisionModel

    cfg = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    vision_cfg = config_type(**cfg["vision_config"])
    if attn_implementation != "auto":
        vision_cfg._attn_implementation = attn_implementation
    model = model_type(vision_cfg)
    inv_freq_fp32 = model.rotary_pos_emb.inv_freq.detach().clone().float()
    model.to(dtype=torch.bfloat16)
    # Hugging Face's full Qwen3-VL BF16 loader keeps the rotary frequency
    # buffer in FP32. Converting this buffer to BF16 changes the vision
    # prefix by enough to produce false CK-vs-PyTorch attribution failures.
    model.rotary_pos_emb.register_buffer("inv_freq", inv_freq_fp32, persistent=False)

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        needed_files = sorted({fname for key, fname in weight_map.items() if key.startswith("model.visual.")})
    else:
        needed_files = ["model.safetensors"]
    state: dict[str, torch.Tensor] = {}
    for fname in needed_files:
        tensors = load_file(str(checkpoint / fname), device="cpu")
        for key, value in tensors.items():
            if key.startswith("model.visual."):
                state[key.removeprefix("model.visual.")] = value
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"visual state mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    return model


def _parse_selector(selector: str) -> tuple[str, int | None]:
    if "@" not in selector:
        return selector, None
    name, layer_s = selector.rsplit("@", 1)
    return name, int(layer_s)


def _ck_semantic_selector(selector: str) -> tuple[str, int | None]:
    name, layer = _parse_selector(selector)
    if name != "layer_input" or layer is None:
        return name, layer
    if layer == 0:
        return "vision_position_embeddings", None
    return "layer_out", layer - 1


def _torch_captures(
    checkpoint: Path,
    image: Path,
    torch_prefix: Path | None,
    out_dir: Path,
    attn_implementation: str,
    selectors: list[str],
    architecture: str,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    if architecture == "cohere_compass":
        from transformers.models.cohere_compass.modeling_cohere_compass import (
            ALL_ATTENTION_FUNCTIONS,
            apply_rotary_pos_emb_vision,
            eager_attention_forward,
        )
    else:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            ALL_ATTENTION_FUNCTIONS,
            apply_rotary_pos_emb_vision,
            eager_attention_forward,
        )

    model = _load_visual_model(checkpoint, attn_implementation, architecture)
    captures: dict[str, torch.Tensor] = {}
    parsed = [_parse_selector(sel) for sel in selectors]
    frontend_wanted = {name for name, layer in parsed if layer is None}
    wanted_by_layer: dict[int, set[str]] = {}
    for name, layer in parsed:
        if layer is not None:
            wanted_by_layer.setdefault(layer, set()).add(name)

    handles: list[Any] = []
    original_mlp_forwards: list[tuple[Any, Any]] = []
    original_attn_forwards: list[tuple[Any, Any]] = []

    class _CaptureComplete(RuntimeError):
        pass

    if "vision_spatial_merge" in frontend_wanted:
        def capture_spatial_merge(_module, _inputs, output):
            merged_width = int(model.merger.hidden_size)
            captures["vision_spatial_merge"] = output.reshape(-1, merged_width).detach().cpu().float()

        handles.append(model.merger.norm.register_forward_hook(capture_spatial_merge))

    if "vision_projector_out" in frontend_wanted:
        def capture_projector_fc2(_module, _inputs, output):
            captures["vision_projector_out"] = output.detach().cpu().float()

        handles.append(model.merger.linear_fc2.register_forward_hook(capture_projector_fc2))

    if "vision_projector_fc1" in frontend_wanted:
        def capture_projector_fc1(_module, _inputs, output):
            captures["vision_projector_fc1"] = output.detach().cpu().float()

        handles.append(model.merger.linear_fc1.register_forward_hook(capture_projector_fc1))

    if "vision_projector_gelu" in frontend_wanted:
        def capture_projector_gelu(_module, inputs):
            captures["vision_projector_gelu"] = inputs[0].detach().cpu().float()

        handles.append(model.merger.linear_fc2.register_forward_pre_hook(capture_projector_gelu))

    def make_norm1_hook(layer: int):
        def hook(_module, _inputs, output):
            captures[f"ln1@{layer}"] = output.detach().cpu().float()
        return hook

    def make_layer_input_hook(layer: int):
        def hook(_module, inputs):
            captures[f"layer_input@{layer}"] = inputs[0].detach().cpu().float()
        return hook

    def make_attn_forward(attn: Any, layer: int, original_forward: Any):
        def attn_forward(
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: Any = None,
            position_embeddings: Any = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            wanted = wanted_by_layer.get(layer, set())
            internal = {
                "qkv_packed",
                "q_proj",
                "k_proj",
                "v_proj",
                "rope_q",
                "rope_k",
                "attn_out_head_major",
                "out_proj",
            }
            if not (internal & wanted):
                return original_forward(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            seq_length = hidden_states.shape[0]
            packed_qkv = attn.qkv(hidden_states)
            if "qkv_packed" in wanted:
                captures[f"qkv_packed@{layer}"] = packed_qkv.contiguous().detach().cpu().float()
            query_states, key_states, value_states = (
                packed_qkv.reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            )
            if "q_proj" in wanted:
                captures[f"q_proj@{layer}"] = query_states.permute(1, 0, 2).contiguous().detach().cpu().float()
            if "k_proj" in wanted:
                captures[f"k_proj@{layer}"] = key_states.permute(1, 0, 2).contiguous().detach().cpu().float()
            if "v_proj" in wanted:
                captures[f"v_proj@{layer}"] = value_states.permute(1, 0, 2).contiguous().detach().cpu().float()

            if position_embeddings is None:
                return original_forward(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
            if "rope_q" in wanted:
                captures[f"rope_q@{layer}"] = query_states.permute(1, 0, 2).contiguous().detach().cpu().float()
            if "rope_k" in wanted:
                captures[f"rope_k@{layer}"] = key_states.permute(1, 0, 2).contiguous().detach().cpu().float()

            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)

            attention_interface = eager_attention_forward
            if attn.config._attn_implementation != "eager":
                if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
                    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                        attn.config._attn_implementation, eager_attention_forward
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]

            if attn.config._attn_implementation == "flash_attention_2":
                max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
                attn_output, _ = attention_interface(
                    attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=None,
                    scaling=attn.scaling,
                    dropout=0.0 if not attn.training else attn.attention_dropout,
                    cu_seq_lens_q=cu_seqlens,
                    cu_seq_lens_k=cu_seqlens,
                    max_length_q=max_seqlen,
                    max_length_k=max_seqlen,
                    is_causal=False,
                    **kwargs,
                )
            else:
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                splits = [
                    torch.split(tensor, lengths.tolist(), dim=2)
                    for tensor in (query_states, key_states, value_states)
                ]
                attn_outputs = [
                    attention_interface(
                        attn,
                        q,
                        k,
                        v,
                        attention_mask=None,
                        scaling=attn.scaling,
                        dropout=0.0 if not attn.training else attn.attention_dropout,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for q, k, v in zip(*splits)
                ]
                attn_output = torch.cat(attn_outputs, dim=1)

            if "attn_out_head_major" in wanted:
                squeezed = attn_output.squeeze(0)
                if squeezed.shape[0] == seq_length:
                    head_major = squeezed.permute(1, 0, 2).contiguous()
                else:
                    head_major = squeezed.contiguous()
                captures[f"attn_out_head_major@{layer}"] = head_major.detach().cpu().float()

            attn_output = attn_output.reshape(seq_length, -1).contiguous()
            attn_output = attn.proj(attn_output)
            if "out_proj" in wanted:
                captures[f"out_proj@{layer}"] = attn_output.detach().cpu().float()
            return attn_output

        return attn_forward

    def make_layer_out_hook(layer: int):
        def hook(_module, _inputs, output):
            captures[f"layer_out@{layer}"] = output.detach().cpu().float()
        return hook

    def make_stop_hook():
        def hook(_module, _inputs, _output):
            raise _CaptureComplete
        return hook

    def make_norm2_pre_hook(layer: int):
        def hook(_module, inputs):
            captures[f"after_attn@{layer}"] = inputs[0].detach().cpu().float()
        return hook

    def make_norm2_hook(layer: int):
        def hook(_module, _inputs, output):
            captures[f"ffn_inp_normed@{layer}"] = output.detach().cpu().float()
        return hook

    def make_mlp_forward(block: Any, layer: int, original_forward: Any):
        def mlp_forward(hidden_state):
            wanted = wanted_by_layer.get(layer, set())
            if {"mlp_up", "ffn_gelu", "mlp_down"} & wanted:
                up = block.mlp.linear_fc1(hidden_state)
                if "mlp_up" in wanted:
                    captures[f"mlp_up@{layer}"] = up.detach().cpu().float()
                gelu = block.mlp.act_fn(up)
                if "ffn_gelu" in wanted:
                    captures[f"ffn_gelu@{layer}"] = gelu.detach().cpu().float()
                down = block.mlp.linear_fc2(gelu)
                if "mlp_down" in wanted:
                    captures[f"mlp_down@{layer}"] = down.detach().cpu().float()
                return down
            return original_forward(hidden_state)
        return mlp_forward

    downstream_frontend = {
        "vision_spatial_merge",
        "vision_projector_fc1",
        "vision_projector_gelu",
        "vision_projector_out",
        "vision_output",
    }
    stop_after_layer = None
    if wanted_by_layer and not (frontend_wanted & downstream_frontend) and torch_prefix is None:
        stop_after_layer = max(wanted_by_layer)

    for layer, wanted in wanted_by_layer.items():
        if layer < 0 or layer >= len(model.blocks):
            raise ValueError(f"selector layer {layer} out of range for {len(model.blocks)} vision blocks")
        block = model.blocks[layer]
        if "layer_input" in wanted:
            handles.append(block.register_forward_pre_hook(make_layer_input_hook(layer)))
        if "ln1" in wanted:
            handles.append(block.norm1.register_forward_hook(make_norm1_hook(layer)))
        if {"qkv_packed", "q_proj", "k_proj", "v_proj", "rope_q", "rope_k", "attn_out_head_major", "out_proj"} & wanted:
            original_forward = block.attn.forward
            original_attn_forwards.append((block.attn, original_forward))
            block.attn.forward = make_attn_forward(block.attn, layer, original_forward)  # type: ignore[method-assign]
        if "layer_out" in wanted:
            handles.append(block.register_forward_hook(make_layer_out_hook(layer)))
        if "after_attn" in wanted:
            handles.append(block.norm2.register_forward_pre_hook(make_norm2_pre_hook(layer)))
        if "ffn_inp_normed" in wanted:
            handles.append(block.norm2.register_forward_hook(make_norm2_hook(layer)))
        if {"mlp_up", "ffn_gelu", "mlp_down"} & wanted:
            original_forward = block.mlp.forward
            original_mlp_forwards.append((block.mlp, original_forward))
            block.mlp.forward = make_mlp_forward(block, layer, original_forward)  # type: ignore[method-assign]
        if layer == stop_after_layer:
            handles.append(block.register_forward_hook(make_stop_hook()))

    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    image_obj = Image.open(image).convert("RGB")
    processor_kwargs: dict[str, Any] = {}
    if architecture == "qwen3vl":
        processor_kwargs.update(min_pixels=1, max_pixels=1048576)
    proc = processor.image_processor(images=image_obj, return_tensors="pt", **processor_kwargs)
    pixel_values = proc["pixel_values"].to(dtype=torch.bfloat16)
    grid = proc["image_grid_thw"]
    frontend_only = {
        "vision_patch_sum",
        "vision_patch_bias",
        "vision_patch_projection",
        "vision_position_embeddings",
    }
    needs_model_forward = any(layer is not None for _name, layer in parsed) or bool(
        frontend_wanted - frontend_only
    )
    final = None
    deepstack: list[torch.Tensor] = []

    try:
        with torch.no_grad():
            if {
                "vision_patch_sum",
                "vision_patch_bias",
                "vision_patch_projection",
                "vision_position_embeddings",
            } & frontend_wanted:
                patch_sum = model.patch_embed(pixel_values)
                if "vision_patch_sum" in frontend_wanted:
                    captures["vision_patch_sum"] = patch_sum.detach().cpu().float()
                if "vision_patch_bias" in frontend_wanted:
                    captures["vision_patch_bias"] = patch_sum.detach().cpu().float()
                if "vision_patch_projection" in frontend_wanted:
                    captures["vision_patch_projection"] = patch_sum.detach().cpu().float()
                if "vision_position_embeddings" in frontend_wanted:
                    pos_embeds = model.fast_pos_embed_interpolate(grid)
                    captures["vision_position_embeddings"] = (
                        patch_sum + pos_embeds.to(patch_sum.dtype)
                    ).detach().cpu().float()
            if needs_model_forward:
                try:
                    model_output = model(pixel_values, grid_thw=grid)
                except _CaptureComplete:
                    model_output = None
                if model_output is not None:
                    if architecture == "cohere_compass":
                        final = model_output.pooler_output
                        deepstack = model_output.deepstack_features
                    else:
                        final, deepstack = model_output
    finally:
        for mlp, original_forward in original_mlp_forwards:
            mlp.forward = original_forward  # type: ignore[method-assign]
        for attn, original_forward in original_attn_forwards:
            attn.forward = original_forward  # type: ignore[method-assign]
        for handle in handles:
            handle.remove()

    if "vision_output" in frontend_wanted:
        if final is None:
            raise RuntimeError("vision_output requested without running the vision model")
        captures["vision_output"] = torch.cat([final, *deepstack], dim=-1).detach().cpu().float()

    prefix_orders: dict[str, dict[str, float]] = {}
    if torch_prefix is not None and torch_prefix.exists():
        if final is None:
            raise RuntimeError("--torch-prefix requires a full vision-model capture")
        ref_prefix = np.fromfile(torch_prefix, dtype=np.float32)
        candidates = {
            "final_then_deep": torch.cat([final, *deepstack], dim=-1),
            "deep_then_final": torch.cat([*deepstack, final], dim=-1),
        }
        for name, tensor in candidates.items():
            arr = tensor.detach().cpu().float().numpy().reshape(-1)
            if arr.shape == ref_prefix.shape:
                prefix_orders[name] = _metrics(ref_prefix, arr)

    missing = [selector for selector in selectors if selector not in captures]
    if missing:
        raise RuntimeError(f"requested PyTorch selectors were not captured: {missing}")

    torch_dir = out_dir / "torch"
    torch_dir.mkdir(parents=True, exist_ok=True)
    tensor_meta: dict[str, Any] = {}
    for name, tensor in captures.items():
        arr = tensor.numpy().astype(np.float32, copy=False)
        path = torch_dir / f"{name.replace('@', '_layer_')}.f32"
        arr.reshape(-1).tofile(path)
        tensor_meta[name] = {"path": str(path), "shape": list(arr.shape)}

    return {
        "architecture": architecture,
        "pixel_values_shape": list(pixel_values.shape),
        "grid_thw": grid.tolist(),
        "prefix_order_metrics": prefix_orders,
        "tensors": tensor_meta,
    }

def _array_to_np(data: array) -> np.ndarray:
    return np.frombuffer(data.tobytes(), dtype=np.float32).copy()


def _qwen3vl_processor_pixels_to_planar(
    pixel_values: np.ndarray,
    grid_thw: list[int] | tuple[int, int, int],
    *,
    patch_size: int,
    temporal_patch_size: int,
    height: int,
    width: int,
    merge_size: int = 2,
    temporal_atol: float = 1.0e-6,
) -> list[float]:
    """Reconstruct CK's planar image input from Qwen3-VL processor patches.

    Hugging Face feeds Qwen3-VL vision patchified, normalized ``pixel_values``
    shaped ``[grid_h * grid_w, 3 * temporal_patch * patch * patch]``. CK's
    generated BF16 vision encoder still accepts a single normalized CHW image
    plane and internally applies the two temporal patch slices. That contract is
    exact for image inputs because Qwen3-VL duplicates the still image across
    temporal slices. If a future processor stops doing that, the generated input
    ABI must change instead of silently comparing different tensors.
    """
    if len(grid_thw) != 3:
        raise ValueError(f"expected grid_thw with 3 entries, got {grid_thw!r}")
    grid_t, grid_h, grid_w = [int(x) for x in grid_thw]
    if grid_t != 1:
        raise ValueError(f"single-image CK planar input expects grid_t=1, got {grid_t}")
    if grid_h * int(patch_size) != int(height) or grid_w * int(patch_size) != int(width):
        raise ValueError(
            "processor grid does not match CK runtime image geometry: "
            f"grid={grid_thw} patch={patch_size} runtime={height}x{width}"
        )

    arr = np.asarray(pixel_values, dtype=np.float32)
    expected_shape = (grid_h * grid_w, 3 * int(temporal_patch_size) * int(patch_size) * int(patch_size))
    if arr.shape != expected_shape:
        raise ValueError(f"pixel_values shape mismatch: got {arr.shape}, expected {expected_shape}")

    merge = int(merge_size)
    if merge <= 0:
        raise ValueError(f"merge_size must be positive, got {merge_size}")
    if grid_h % merge != 0 or grid_w % merge != 0:
        raise ValueError(f"Qwen3-VL grid must be divisible by merge_size: grid={grid_thw} merge={merge}")

    # HF flattens patches after this logical transpose:
    #   (t, gh//m, gw//m, mh, mw, c, temporal, py, px)
    # CK's image ABI wants a planar CHW image whose im2patch pass sees row-major
    # patches. Undo HF's merge-tiled order before reconstructing the plane.
    tiled = arr.reshape(
        grid_t,
        grid_h // merge,
        grid_w // merge,
        merge,
        merge,
        3,
        int(temporal_patch_size),
        int(patch_size),
        int(patch_size),
    )
    patches = tiled[0].transpose(0, 2, 1, 3, 4, 5, 6, 7).reshape(
        grid_h, grid_w, 3, int(temporal_patch_size), int(patch_size), int(patch_size)
    )
    if int(temporal_patch_size) > 1:
        temporal_ref = patches[:, :, :, 0, :, :]
        for t in range(1, int(temporal_patch_size)):
            max_diff = float(np.max(np.abs(temporal_ref - patches[:, :, :, t, :, :])))
            if max_diff > temporal_atol:
                raise ValueError(
                    "processor temporal patch slices differ; CK's single-plane "
                    f"vision input ABI is not exact for this sample (slice={t}, max_diff={max_diff})"
                )

    planar = patches[:, :, :, 0, :, :].transpose(2, 0, 3, 1, 4).reshape(3, int(height), int(width))
    return planar.reshape(-1).astype(np.float32, copy=False).tolist()


def _load_processor_planar(
    checkpoint: Path,
    image: Path,
    *,
    height: int,
    width: int,
    architecture: str,
) -> list[float]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    cfg = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    vision_cfg = cfg.get("vision_config", {})
    patch_size = int(vision_cfg.get("patch_size", 16))
    temporal_patch_size = int(vision_cfg.get("temporal_patch_size", 2))
    merge_size = int(vision_cfg.get("spatial_merge_size") or vision_cfg.get("merge_size") or 2)

    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    image_obj = Image.open(image).convert("RGB")
    processor_kwargs: dict[str, Any] = {}
    if architecture == "qwen3vl":
        processor_kwargs.update(min_pixels=1, max_pixels=1048576)
    proc = processor.image_processor(images=image_obj, return_tensors="pt", **processor_kwargs)
    pixel_values = proc["pixel_values"].detach().to(dtype=torch.float32).cpu().numpy()
    grid = [int(x) for x in proc["image_grid_thw"][0].tolist()]
    return _qwen3vl_processor_pixels_to_planar(
        pixel_values,
        grid,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        height=int(height),
        width=int(width),
    )


def _literal_call_arg(operation: dict[str, Any], name: str) -> int | None:
    for argument in operation.get("args", []):
        if str(argument.get("name", "")) != name:
            continue
        expression = str(argument.get("expr", "")).strip()
        try:
            value = int(expression, 0)
        except ValueError:
            return None
        return value if value >= 0 else None
    return None


def _operation_output_elements(operation: dict[str, Any]) -> int | None:
    """Return the logical output extent declared by a supported call ABI."""
    rows = _literal_call_arg(operation, "M")
    columns = _literal_call_arg(operation, "N")
    if rows is not None and columns is not None:
        return rows * columns
    for name in ("n", "num_elements", "elements"):
        elements = _literal_call_arg(operation, name)
        if elements is not None:
            return elements
    return None


def _run_ck_selector(
    args: argparse.Namespace,
    selector: str,
    numeric: Any,
    *,
    expected_elements: int | None = None,
) -> np.ndarray:
    runtime_dir = args.runtime_dir.resolve()
    cfg = json.loads((runtime_dir / "config.json").read_text(encoding="utf-8"))
    planar_image = _load_processor_planar(
        args.checkpoint.resolve(),
        args.image.resolve(),
        height=int(cfg["image_height"]),
        width=int(cfg["image_width"]),
        architecture=args.architecture,
    )
    restore_import_path = os.environ.get("CK_DEBUG_IMPORT_HIDDEN")
    restore_import_checkpoint = os.environ.get("CK_DEBUG_IMPORT_CHECKPOINT")
    restore_import_layer = os.environ.get("CK_DEBUG_IMPORT_LAYER")
    restore_stop_op = os.environ.get("CK_STOP_OP")
    requested_name, _ = _parse_selector(selector)
    semantic_name, semantic_layer = _ck_semantic_selector(selector)
    call_ir = json.loads((runtime_dir / "call.json").read_text(encoding="utf-8"))
    stop_op = None
    stop_buffer_ref = None
    stop_operation = None
    for operation in call_ir.get("operations", []):
        for checkpoint in operation.get("semantic_checkpoints", []):
            checkpoint_layer = int(checkpoint.get("layer", -1))
            if str(checkpoint.get("tensor")) == semantic_name and (
                semantic_layer is None or checkpoint_layer == semantic_layer
            ):
                output_refs = {
                    str(argument["buffer_ref"])
                    for argument in operation.get("args", [])
                    if str(argument.get("source", "")).startswith("output:")
                    and argument.get("buffer_ref")
                }
                if len(output_refs) == 1:
                    stop_op = int(operation["idx"])
                    stop_buffer_ref = next(iter(output_refs))
                    stop_operation = operation
                break
        if stop_op is not None:
            break
    if stop_op is not None:
        os.environ["CK_STOP_OP"] = str(stop_op)
    numeric_output_name = stop_buffer_ref or selector
    if args.ck_import_layer_input is not None:
        os.environ["CK_DEBUG_IMPORT_HIDDEN"] = str(args.ck_import_layer_input.resolve())
        os.environ["CK_DEBUG_IMPORT_CHECKPOINT"] = str(args.ck_import_checkpoint)
        os.environ["CK_DEBUG_IMPORT_LAYER"] = str(args.ck_import_layer)
    try:
        data = numeric._run_generated_encoder(
            model_so=runtime_dir / args.model_so_name,
            weights_bump=args.weights_bump.resolve(),
            manifest_map=runtime_dir / "weights_manifest.map",
            layout_path=runtime_dir / "layout.json",
            planar_image=planar_image,
            output_name=numeric_output_name,
        )
    finally:
        if restore_stop_op is None:
            os.environ.pop("CK_STOP_OP", None)
        else:
            os.environ["CK_STOP_OP"] = restore_stop_op
        if args.ck_import_layer_input is not None:
            if restore_import_path is None:
                os.environ.pop("CK_DEBUG_IMPORT_HIDDEN", None)
            else:
                os.environ["CK_DEBUG_IMPORT_HIDDEN"] = restore_import_path
            if restore_import_layer is None:
                os.environ.pop("CK_DEBUG_IMPORT_LAYER", None)
            else:
                os.environ["CK_DEBUG_IMPORT_LAYER"] = restore_import_layer
            if restore_import_checkpoint is None:
                os.environ.pop("CK_DEBUG_IMPORT_CHECKPOINT", None)
            else:
                os.environ["CK_DEBUG_IMPORT_CHECKPOINT"] = restore_import_checkpoint
    result = _array_to_np(data)
    if expected_elements is not None and result.size != expected_elements:
        if result.size < expected_elements:
            raise RuntimeError(
                f"CK checkpoint {selector!r} contains {result.size} elements; "
                f"the oracle declares {expected_elements}"
            )
        logical_elements = _operation_output_elements(stop_operation or {})
        if logical_elements != expected_elements:
            raise RuntimeError(
                f"CK checkpoint {selector!r} exposes allocation capacity ({result.size} elements), "
                f"but its logical extent cannot be proven: call_ir={logical_elements!r}, "
                f"oracle={expected_elements}"
            )
        result = result[:expected_elements].copy()
    base_name, _ = _parse_selector(selector)
    # The lower-level encoder reader canonicalizes every exported head-major
    # tensor to the flattened order consumed by the next operation. Projection
    # and RoPE oracles are explicitly captured as [head, token, channel], so
    # restore those here. The PyTorch attention interface is captured in its
    # projection-consumption order; restoring it again would undo the required
    # CK attention transpose and create a false layout divergence.
    head_major_names = {"q_proj", "k_proj", "v_proj", "rope_q", "rope_k"}
    if base_name in head_major_names:
        head_dim = int(cfg.get("head_dim") or cfg.get("aligned_head_dim") or 0)
        if base_name in {"q_proj", "rope_q", "attn_out_head_major"}:
            heads = int(cfg.get("num_heads") or cfg.get("vision_num_heads") or 0)
        else:
            heads = int(cfg.get("num_kv_heads") or cfg.get("vision_num_kv_heads") or cfg.get("num_heads") or 0)
        row_width = heads * head_dim
        if row_width > 0 and result.size % row_width == 0:
            tokens = result.size // row_width
            result = result.reshape(tokens, heads, head_dim).transpose(1, 0, 2).copy().reshape(-1)
    return result


def _run_ck_selector_isolated(
    args: argparse.Namespace,
    selector: str,
    output: Path,
    *,
    expected_elements: int | None = None,
) -> np.ndarray:
    """Capture one model-sized CK checkpoint in a fresh process."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint", str(args.checkpoint),
        "--runtime-dir", str(args.runtime_dir),
        "--weights-bump", str(args.weights_bump),
        "--image", str(args.image),
        "--out-dir", str(args.out_dir),
        "--threads", str(args.threads),
        "--attn-implementation", str(args.attn_implementation),
        "--architecture", str(args.architecture),
        "--model-so-name", str(getattr(args, "model_so_name", "libqwen3vl_bf16_encoder_v8.so")),
        "--ck-worker-selector", selector,
        "--ck-worker-output", str(output),
    ]
    if expected_elements is not None:
        cmd.extend(["--ck-worker-elements", str(expected_elements)])
    if args.ck_import_layer_input is not None:
        cmd.extend([
            "--ck-import-layer-input", str(args.ck_import_layer_input),
            "--ck-import-layer", str(args.ck_import_layer),
            "--ck-import-checkpoint", str(args.ck_import_checkpoint),
        ])
    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(args.threads)
    env["OMP_NUM_THREADS"] = str(args.threads)
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"isolated CK capture failed for {selector!r}: rc={completed.returncode}")
    if not output.is_file():
        raise RuntimeError(f"isolated CK capture did not produce {output}")
    return np.fromfile(output, dtype=np.float32)


def _run_torch_captures_isolated(args: argparse.Namespace, selectors: list[str]) -> dict[str, Any]:
    """Capture PyTorch checkpoints in a process that exits before CK loads."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint", str(args.checkpoint),
        "--runtime-dir", str(args.runtime_dir),
        "--weights-bump", str(args.weights_bump),
        "--image", str(args.image),
        "--out-dir", str(args.out_dir),
        "--threads", str(args.threads),
        "--attn-implementation", str(args.attn_implementation),
        "--architecture", str(args.architecture),
        "--model-so-name", str(args.model_so_name),
        "--skip-ck",
    ]
    if args.torch_prefix is not None:
        cmd.extend(["--torch-prefix", str(args.torch_prefix)])
    for selector in selectors:
        cmd.extend(["--selector", selector])
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=os.environ.copy())
    if completed.returncode != 0:
        raise RuntimeError(f"isolated PyTorch capture failed: rc={completed.returncode}")
    report_path = args.out_dir / "report.json"
    if not report_path.is_file():
        raise RuntimeError(f"isolated PyTorch capture did not produce {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tensors = ((report.get("torch") or {}).get("tensors") or {})
    missing = [selector for selector in selectors if selector not in tensors]
    if missing:
        raise RuntimeError(f"isolated PyTorch capture omitted selectors: {missing}")
    return report["torch"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare BF16 vision hidden tensors against CK hidden exports.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--runtime-dir", type=Path, required=True)
    ap.add_argument("--weights-bump", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--torch-prefix", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("build/qwen3vl_bf16_hidden_compare"))
    ap.add_argument(
        "--selector",
        action="append",
        default=[],
        help="Tensor selector such as ffn_inp_normed@9. May be repeated.",
    )
    ap.add_argument("--threads", type=int, default=int(os.environ.get("CK_NUM_THREADS", "20") or "20"))
    ap.add_argument("--attn-implementation", choices=("auto", "eager", "sdpa"), default="auto")
    ap.add_argument("--architecture", choices=("qwen3vl", "cohere_compass"), default="qwen3vl")
    ap.add_argument("--model-so-name", default="libqwen3vl_bf16_encoder_v8.so")
    ap.add_argument("--ck-import-layer-input", type=Path, help="Inject an exact FP32 tensor before a CK layer's first residual save")
    ap.add_argument("--ck-import-layer", type=int, help="Layer index for --ck-import-layer-input")
    ap.add_argument("--ck-import-checkpoint", choices=("layer_input", "after_attn"), default="layer_input")
    ap.add_argument("--skip-ck", action="store_true", help="Only run the PyTorch hook/reference side")
    ap.add_argument("--min-cosine", type=float, default=None)
    ap.add_argument("--max-rmse", type=float, default=None)
    ap.add_argument("--max-abs", type=float, default=None)
    ap.add_argument("--max-relative-rmse", type=float, default=None)
    ap.add_argument("--final-max-rmse", type=float, default=None)
    ap.add_argument("--final-max-abs", type=float, default=None)
    ap.add_argument("--ck-worker-selector", help=argparse.SUPPRESS)
    ap.add_argument("--ck-worker-output", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--ck-worker-elements", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if (args.ck_import_layer_input is None) != (args.ck_import_layer is None):
        ap.error("--ck-import-layer-input and --ck-import-layer must be provided together")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CK_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    if args.ck_worker_selector:
        if args.ck_worker_output is None:
            ap.error("--ck-worker-output is required with --ck-worker-selector")
        numeric = _import_numeric_parity()
        captured = _run_ck_selector(
            args,
            args.ck_worker_selector,
            numeric,
            expected_elements=args.ck_worker_elements,
        )
        args.ck_worker_output.parent.mkdir(parents=True, exist_ok=True)
        captured.tofile(args.ck_worker_output)
        return 0
    selectors = args.selector or ["ffn_inp_normed@9", "mlp_up@9", "ffn_gelu@9", "mlp_down@9", "layer_out@9"]

    t0 = time.perf_counter()
    if args.skip_ck:
        torch_report = _torch_captures(
            args.checkpoint.resolve(),
            args.image.resolve(),
            args.torch_prefix,
            args.out_dir,
            args.attn_implementation,
            selectors,
            args.architecture,
        )
    else:
        torch_report = _run_torch_captures_isolated(args, selectors)
    t_torch = time.perf_counter()

    rows: dict[str, Any] = {}
    if not args.skip_ck:
        ck_dir = args.out_dir / "ck"
        ck_dir.mkdir(parents=True, exist_ok=True)
        for selector in selectors:
            print(f"[ck] {selector}", flush=True)
            ck_path = ck_dir / f"{selector.replace('@', '_layer_')}.f32"
            torch_tensor = torch_report["tensors"][selector]
            torch_shape = [int(value) for value in torch_tensor["shape"]]
            expected_elements = math.prod(torch_shape)
            cpu_before = _child_cpu_seconds()
            wall_before = time.perf_counter()
            ck = _run_ck_selector_isolated(
                args,
                selector,
                ck_path,
                expected_elements=expected_elements,
            )
            ck_wall_sec = time.perf_counter() - wall_before
            ck_cpu_sec = _child_cpu_seconds() - cpu_before
            average_cores = ck_cpu_sec / ck_wall_sec if ck_wall_sec > 0.0 else 0.0
            torch_path = Path(torch_tensor["path"])
            ref = np.fromfile(torch_path, dtype=np.float32)
            rows[selector] = {
                "ck_path": str(ck_path),
                "torch_path": str(torch_path),
                "shape": torch_shape,
                "performance": {
                    "scope": "isolated_process_through_checkpoint",
                    "wall_sec": ck_wall_sec,
                    "cpu_sec": ck_cpu_sec,
                    "average_core_equivalents": average_cores,
                    "configured_threads": int(args.threads),
                    "thread_utilization_ratio": average_cores / float(args.threads),
                    "idle_core_seconds": max(0.0, float(args.threads) * ck_wall_sec - ck_cpu_sec),
                },
                **_metrics(ref, ck),
            }

    failures: list[str] = []
    for selector, metrics in rows.items():
        if args.min_cosine is not None and float(metrics["cosine"]) < args.min_cosine:
            failures.append(f"{selector}: cosine {metrics['cosine']:.9f} < {args.min_cosine:.9f}")
        if args.max_rmse is not None and float(metrics["rmse"]) > args.max_rmse:
            failures.append(f"{selector}: rmse {metrics['rmse']:.9f} > {args.max_rmse:.9f}")
        if args.max_abs is not None and float(metrics["max_abs"]) > args.max_abs:
            failures.append(f"{selector}: max_abs {metrics['max_abs']:.9f} > {args.max_abs:.9f}")
        if args.max_relative_rmse is not None and float(metrics["relative_rmse"]) > args.max_relative_rmse:
            failures.append(f"{selector}: relative_rmse {metrics['relative_rmse']:.9f} > {args.max_relative_rmse:.9f}")
        if selector == "vision_output" and args.final_max_rmse is not None and float(metrics["rmse"]) > args.final_max_rmse:
            failures.append(f"{selector}: final rmse {metrics['rmse']:.9f} > {args.final_max_rmse:.9f}")
        if selector == "vision_output" and args.final_max_abs is not None and float(metrics["max_abs"]) > args.final_max_abs:
            failures.append(f"{selector}: final max_abs {metrics['max_abs']:.9f} > {args.final_max_abs:.9f}")

    performance_debt = sorted(
        (
            {
                "selector": selector,
                **metrics["performance"],
            }
            for selector, metrics in rows.items()
        ),
        key=lambda row: float(row["idle_core_seconds"]),
        reverse=True,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "runtime_dir": str(args.runtime_dir),
        "weights_bump": str(args.weights_bump),
        "image": str(args.image),
        "selectors": selectors,
        "attn_implementation": args.attn_implementation,
        "architecture": args.architecture,
        "timings_sec": {
            "torch": t_torch - t0,
            "total": time.perf_counter() - t0,
        },
        "torch": torch_report,
        "comparisons": rows,
        "performance_debt": performance_debt,
        "thresholds": {
            "min_cosine": args.min_cosine,
            "max_rmse": args.max_rmse,
            "max_abs": args.max_abs,
            "max_relative_rmse": args.max_relative_rmse,
            "final_max_rmse": args.final_max_rmse,
            "final_max_abs": args.final_max_abs,
        },
        "failures": failures,
        "status": "fail" if failures else "pass",
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "comparisons": rows, "failures": failures}, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
