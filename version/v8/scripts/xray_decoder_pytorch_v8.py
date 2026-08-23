#!/usr/bin/env python3
"""Attribute decoder drift between a generated CK runtime and PyTorch."""

from __future__ import annotations

import argparse
import ast
import ctypes
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import xray_numerical_parity_v8 as xray
from compare_first_token_logits_v8 import compare_logits
from compare_multitoken_logits_v8 import load_ck_greedy_trajectory


def parse_token_ids(value: str) -> list[int]:
    path = Path(value)
    if "," not in value and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("token_ids") or payload.get("tokens")
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a token list")
        return [int(token) for token in payload]
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def split_teacher_forced_tokens(
    token_ids: list[int],
    prompt_token_count: int | None,
) -> tuple[list[int], list[int]] | None:
    if prompt_token_count is None:
        return None
    if prompt_token_count <= 0 or prompt_token_count >= len(token_ids):
        raise ValueError(
            "--prompt-token-count must leave at least one prompt token and "
            "one teacher-forced token"
        )
    return token_ids[:prompt_token_count], token_ids[prompt_token_count:]


def decoder_layers(model: Any) -> list[Any]:
    candidates = [
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("transformer", "h"),
    ]
    for path in candidates:
        current = model
        for component in path:
            current = getattr(current, component, None)
            if current is None:
                break
        if current is not None:
            layers = list(current)
            if layers:
                return layers
    raise RuntimeError("unable to locate an ordered decoder-layer collection")


def _tensor_from_output(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def _last_feature_row(tensor: Any) -> Any:
    """Select the final logical row from a projection with arbitrary leading axes."""
    return tensor.reshape(-1, tensor.shape[-1])[-1]


def _split_mla_kv_projection(
    output: Any,
    heads: int,
    k_width: int,
    v_width: int,
) -> tuple[Any, Any]:
    expanded = _last_feature_row(output).reshape(heads, k_width + v_width)
    k_nope, value = expanded.split((k_width, v_width), dim=-1)
    return k_nope.reshape(-1), value.reshape(-1)


def _split_mla_kv_projection_full(
    output: Any,
    heads: int,
    k_width: int,
    v_width: int,
) -> tuple[Any, Any]:
    expanded = output.reshape(-1, heads, k_width + v_width)
    return expanded.split((k_width, v_width), dim=-1)


def _logical_ck_capture(
    values: np.ndarray,
    boundary: str,
    token_count: int,
) -> tuple[str, np.ndarray]:
    """Return the final logical row from either full or explicit-last exports."""
    if boundary.endswith("_last"):
        return boundary.removesuffix("_last"), values
    if values.size % token_count != 0:
        raise RuntimeError(
            f"capture has {values.size} values for {token_count} tokens"
        )
    return boundary, values.reshape(token_count, -1)[-1]


def _instrument_attention_resolver(
    resolver: Any,
    captured: dict[int, dict[str, Any]],
    full_capture_layers: frozenset[int] = frozenset(),
) -> Any:
    """Wrap a Transformers attention interface with semantic MLA captures."""
    def instrumented_resolver(config: Any) -> Any:
        interface = resolver(config)

        def instrumented_interface(
            module: Any,
            query: Any,
            key: Any,
            value: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            layer = int(module.layer_idx)
            captured[layer]["mla_query"] = (
                query[0, :, -1, :].reshape(-1).float().cpu().contiguous()
            )
            captured[layer]["mla_key"] = (
                key[0, :, -1, :].reshape(-1).float().cpu().contiguous()
            )
            if layer in full_capture_layers:
                captured[layer]["mla_query_full"] = (
                    query[0].transpose(0, 1).float().cpu().contiguous()
                )
                captured[layer]["mla_key_full"] = (
                    key[0].transpose(0, 1).float().cpu().contiguous()
                )
                captured[layer]["mla_value_full"] = (
                    value[0].transpose(0, 1).float().cpu().contiguous()
                )
            result = interface(module, query, key, value, *args, **kwargs)
            context = _tensor_from_output(result)
            captured[layer]["mla_context"] = (
                context[0, -1].reshape(-1).float().cpu().contiguous()
            )
            if layer in full_capture_layers:
                captured[layer]["mla_context_full"] = (
                    context[0].float().cpu().contiguous()
                )
            return result

        return instrumented_interface

    return instrumented_resolver


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_tensor(path: Path, values: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.ascontiguousarray(values, dtype=np.float32)
    values.tofile(path)
    return {"path": str(path.resolve()), "shape": list(values.shape), "sha256": _sha256(path)}


def capture_pytorch(
    checkpoint: Path,
    token_ids: list[int],
    output_dir: Path,
    threads: int,
    mla_replay_layer: int | None = None,
) -> tuple[dict[int, dict[str, dict[str, Any]]], np.ndarray]:
    import torch
    import transformers.activations as activations
    from transformers import AutoModelForCausalLM

    if not hasattr(activations, "PytorchGELUTanh"):
        activations.PytorchGELUTanh = activations.GELUTanh
    torch.set_num_threads(int(threads))
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    captured: dict[int, dict[str, Any]] = {}
    layer_residuals: dict[int, Any] = {}
    layer_residuals_full: dict[int, Any] = {}
    handles = []
    attention_interface_module = None
    attention_interface_original = None
    routed_function_originals: dict[Any, Any] = {}
    moe_layers: dict[int, int] = {}
    for layer, module in enumerate(decoder_layers(model)):
        captured[layer] = {}

        def layer_pre_hook(_module: Any, inputs: Any, layer: int = layer) -> None:
            hidden = _tensor_from_output(inputs[0])
            layer_residuals[layer] = (
                _last_feature_row(hidden).detach().clone()
            )
            if layer == mla_replay_layer:
                layer_residuals_full[layer] = hidden[0].detach().clone()

        def layer_hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
            captured[layer]["layer_out"] = _tensor_from_output(output)[0, -1].float().cpu().contiguous()
            if mla_replay_layer is not None and layer <= mla_replay_layer:
                captured[layer]["layer_out_full"] = (
                    _tensor_from_output(output)[0].float().cpu().contiguous()
                )
            if isinstance(output, (tuple, list)) and len(output) > 1:
                captured[layer]["routed_free_out"] = (
                    output[1][0, -1].float().cpu().contiguous()
                )
                if mla_replay_layer is not None and layer <= mla_replay_layer:
                    captured[layer]["routed_free_out_full"] = (
                        output[1][0].float().cpu().contiguous()
                    )
        handles.append(module.register_forward_pre_hook(layer_pre_hook))
        handles.append(module.register_forward_hook(layer_hook))

        input_norm = getattr(module, "input_layernorm", None)
        if input_norm is not None:
            def input_norm_hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                captured[layer]["block_rmsnorm"] = output[0, -1].float().cpu().contiguous()
                if layer == mla_replay_layer:
                    captured[layer]["block_rmsnorm_full"] = (
                        output[0].float().cpu().contiguous()
                    )
            handles.append(input_norm.register_forward_hook(input_norm_hook))

        attention = getattr(module, "self_attn", None)
        if attention is not None:
            if attention_interface_module is None:
                candidate_module = sys.modules.get(attention.__class__.__module__)
                candidate_resolver = getattr(
                    candidate_module, "get_attention_interface", None
                )
                if callable(candidate_resolver):
                    attention_interface_module = candidate_module
                    attention_interface_original = candidate_resolver
                    candidate_module.get_attention_interface = (
                        _instrument_attention_resolver(
                            candidate_resolver,
                            captured,
                            frozenset({mla_replay_layer})
                            if mla_replay_layer is not None
                            else frozenset(),
                        )
                    )
            q_proj = getattr(attention, "q_proj", None)
            if q_proj is not None:
                def q_proj_hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                    captured[layer]["q_proj"] = output[0, -1].float().cpu().contiguous()
                    if layer == mla_replay_layer:
                        captured[layer]["q_proj_full"] = (
                            output[0].float().cpu().contiguous()
                        )
                handles.append(q_proj.register_forward_hook(q_proj_hook))
            kv_a = getattr(attention, "kv_a_proj_with_mqa", None)
            if kv_a is not None:
                def kv_a_hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                    captured[layer]["mla_kv_a"] = output[0, -1].float().cpu().contiguous()
                    if layer == mla_replay_layer:
                        captured[layer]["mla_kv_a_full"] = (
                            output[0].float().cpu().contiguous()
                        )
                handles.append(kv_a.register_forward_hook(kv_a_hook))
            kv_norm = getattr(attention, "kv_a_layernorm", None)
            if kv_norm is not None:
                def kv_norm_hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                    captured[layer]["mla_kv_norm"] = output[0, -1].float().cpu().contiguous()
                    if layer == mla_replay_layer:
                        captured[layer]["mla_kv_norm_full"] = (
                            output[0].float().cpu().contiguous()
                        )
                handles.append(kv_norm.register_forward_hook(kv_norm_hook))
            kv_b = getattr(attention, "kv_b_proj", None)
            if kv_b is not None:
                heads = int(getattr(attention, "num_heads"))
                k_width = int(getattr(attention, "qk_nope_head_dim"))
                v_width = int(getattr(attention, "v_head_dim"))

                def kv_b_hook(
                    _module: Any,
                    _inputs: Any,
                    output: Any,
                    layer: int = layer,
                    heads: int = heads,
                    k_width: int = k_width,
                    v_width: int = v_width,
                ) -> None:
                    k_nope, value = _split_mla_kv_projection(
                        output, heads, k_width, v_width
                    )
                    captured[layer]["mla_k_nope"] = (
                        k_nope.float().cpu().contiguous()
                    )
                    captured[layer]["mla_value"] = (
                        value.float().cpu().contiguous()
                    )
                    if layer == mla_replay_layer:
                        k_nope_full, value_full = _split_mla_kv_projection_full(
                            output, heads, k_width, v_width
                        )
                        captured[layer]["mla_k_nope_full"] = (
                            k_nope_full.float().cpu().contiguous()
                        )
                        captured[layer]["mla_value_projection_full"] = (
                            value_full.float().cpu().contiguous()
                        )
                handles.append(kv_b.register_forward_hook(kv_b_hook))
            gate_proj = getattr(attention, "gate_proj", None)
            if gate_proj is not None:
                def gate_proj_hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                    captured[layer]["attn_gate"] = output[0, -1].float().cpu().contiguous()
                    if layer == mla_replay_layer:
                        captured[layer]["attn_gate_full"] = (
                            output[0].float().cpu().contiguous()
                        )
                handles.append(gate_proj.register_forward_hook(gate_proj_hook))
            out_proj = getattr(attention, "o_proj", None)
            if out_proj is not None:
                def out_proj_pre_hook(_module: Any, inputs: Any, layer: int = layer) -> None:
                    captured[layer]["attn_out"] = inputs[0][0, -1].float().cpu().contiguous()
                    if layer == mla_replay_layer:
                        captured[layer]["attn_out_full"] = (
                            inputs[0][0].float().cpu().contiguous()
                        )
                def out_proj_hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                    projected = _last_feature_row(output)
                    captured[layer]["out_proj"] = projected.float().cpu().contiguous()
                    residual = layer_residuals.get(layer)
                    if residual is not None:
                        captured[layer]["after_attn"] = (
                            residual + projected
                        ).float().cpu().contiguous()
                    residual_full = layer_residuals_full.get(layer)
                    if residual_full is not None:
                        captured[layer]["out_proj_full"] = (
                            output[0].float().cpu().contiguous()
                        )
                        captured[layer]["after_attn_full"] = (
                            residual_full + output[0]
                        ).float().cpu().contiguous()
                handles.append(out_proj.register_forward_pre_hook(out_proj_pre_hook))
                handles.append(out_proj.register_forward_hook(out_proj_hook))

        ffn_norm = getattr(module, "post_attention_layernorm", None)
        if ffn_norm is not None:
            def norm_pre_hook(_module: Any, inputs: Any, layer: int = layer) -> None:
                captured[layer]["ffn_input"] = inputs[0][0, -1].float().cpu().contiguous()
                if layer == mla_replay_layer:
                    captured[layer]["ffn_input_full"] = (
                        inputs[0][0].float().cpu().contiguous()
                    )
            def norm_hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                captured[layer]["ffn_norm"] = output[0, -1].float().cpu().contiguous()
                if layer == mla_replay_layer:
                    captured[layer]["ffn_norm_full"] = (
                        output[0].float().cpu().contiguous()
                    )
            handles.append(ffn_norm.register_forward_pre_hook(norm_pre_hook))
            handles.append(ffn_norm.register_forward_hook(norm_hook))

        mlp = getattr(module, "mlp", None)
        if mlp is not None:
            moe_layers[id(mlp)] = layer
            candidate_module = sys.modules.get(mlp.__class__.__module__)
            routed_function = getattr(
                candidate_module, "routed_expert_output", None
            )
            if callable(routed_function) and candidate_module not in routed_function_originals:
                routed_function_originals[candidate_module] = routed_function

                def instrumented_routed_expert_output(
                    moe: Any,
                    hidden: Any,
                    *args: Any,
                    _original: Any = routed_function,
                    **kwargs: Any,
                ) -> Any:
                    output = _original(moe, hidden, *args, **kwargs)
                    routed_layer = moe_layers.get(id(moe))
                    if routed_layer is not None:
                        captured[routed_layer]["moe_routed_output"] = (
                            _last_feature_row(output).float().cpu().contiguous()
                        )
                        if routed_layer == mla_replay_layer:
                            captured[routed_layer]["moe_routed_output_full"] = (
                                output.reshape(-1, output.shape[-1])
                                .float().cpu().contiguous()
                            )
                    return output

                candidate_module.routed_expert_output = instrumented_routed_expert_output
        dense_gate = getattr(mlp, "gate_proj", None)
        if dense_gate is not None:
            def dense_gate_hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                captured[layer]["mlp_gate"] = _last_feature_row(output).float().cpu().contiguous()
            handles.append(dense_gate.register_forward_hook(dense_gate_hook))
        dense_up = getattr(mlp, "up_proj", None)
        if dense_up is not None:
            def dense_up_hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                captured[layer]["mlp_up"] = _last_feature_row(output).float().cpu().contiguous()
            handles.append(dense_up.register_forward_hook(dense_up_hook))
        dense_down = getattr(mlp, "down_proj", None)
        if dense_down is not None:
            def dense_down_pre_hook(_module: Any, inputs: Any, layer: int = layer) -> None:
                captured[layer]["mlp_swiglu"] = _last_feature_row(inputs[0]).float().cpu().contiguous()
            def dense_down_hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                captured[layer]["mlp_down"] = _last_feature_row(output).float().cpu().contiguous()
            handles.append(dense_down.register_forward_pre_hook(dense_down_pre_hook))
            handles.append(dense_down.register_forward_hook(dense_down_hook))
        gate = getattr(mlp, "gate", None)
        if gate is not None:
            def gate_hook(gate_module: Any, inputs: Any, output: Any, layer: int = layer) -> None:
                import torch.nn.functional as functional

                hidden = inputs[0]
                logits = functional.linear(
                    hidden.reshape(-1, hidden.shape[-1]).float(),
                    gate_module.weight.float(),
                ).reshape(*hidden.shape[:-1], -1)
                captured[layer]["moe_router_logits"] = logits[0, -1].cpu().contiguous()
                captured[layer]["moe_routing_weights"] = output[1][-1].float().cpu().contiguous()
                if layer == mla_replay_layer:
                    captured[layer]["moe_router_logits_full"] = (
                        logits[0].cpu().contiguous()
                    )
                    captured[layer]["moe_routing_weights_full"] = (
                        output[1].reshape(hidden.shape[1], -1)
                        .float().cpu().contiguous()
                    )
            handles.append(gate.register_forward_hook(gate_hook))

        shared = getattr(mlp, "shared_experts", None)
        if shared is not None:
            def shared_hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                captured[layer]["moe_shared_output"] = (
                    _tensor_from_output(output)[0, -1].float().cpu().contiguous()
                )
                if layer == mla_replay_layer:
                    captured[layer]["moe_shared_output_full"] = (
                        _tensor_from_output(output)[0].float().cpu().contiguous()
                    )
            handles.append(shared.register_forward_hook(shared_hook))
        if mlp is not None:
            def mlp_hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                captured[layer]["moe_combined_output"] = (
                    _tensor_from_output(output)[0, -1].float().cpu().contiguous()
                )
                if layer == mla_replay_layer:
                    captured[layer]["moe_combined_output_full"] = (
                        _tensor_from_output(output)[0].float().cpu().contiguous()
                    )
            handles.append(mlp.register_forward_hook(mlp_hook))

    ids = torch.tensor([token_ids], dtype=torch.long)
    try:
        with torch.inference_mode():
            result = model(input_ids=ids, use_cache=False, return_dict=True)
        logits = result.logits[0, -1].float().cpu().numpy()
        tensors = {}
        for layer, boundaries in sorted(captured.items()):
            tensors[layer] = {
                boundary: _write_tensor(
                    output_dir / f"layer_{layer:03d}_{boundary}.f32",
                    tensor.numpy(),
                )
                for boundary, tensor in boundaries.items()
            }
    finally:
        for handle in handles:
            handle.remove()
        if attention_interface_module is not None:
            attention_interface_module.get_attention_interface = (
                attention_interface_original
            )
        for module, original in routed_function_originals.items():
            module.routed_expert_output = original
        del model
        gc.collect()
    return tensors, np.ascontiguousarray(logits, dtype=np.float32)


def capture_ck(
    runtime: Path,
    token_ids: list[int],
    output_dir: Path,
    mla_replay_layer: int | None = None,
) -> tuple[dict[int, dict[str, dict[str, Any]]], np.ndarray]:
    dump_dir = output_dir / "raw"
    dump_dir.mkdir(parents=True, exist_ok=True)
    old = {
        name: os.environ.get(name)
        for name in (
            "CK_DEBUG_EXPORT_HIDDEN",
            "CK_DEBUG_EXPORT_HIDDEN_NAME",
            "CK_DEBUG_EXPORT_HIDDEN_NAMES",
        )
    }
    os.environ["CK_DEBUG_EXPORT_HIDDEN"] = str(dump_dir.resolve())
    os.environ.pop("CK_DEBUG_EXPORT_HIDDEN_NAME", None)
    os.environ["CK_DEBUG_EXPORT_HIDDEN_NAMES"] = ",".join(
        (
            "layer_out",
            "block_rmsnorm",
            "q_proj",
            "mla_kv_a",
            "mla_kv_norm",
            "mla_k_nope",
            "mla_value",
            "mla_query",
            "mla_key",
            "mla_context",
            "attn_gate",
            "attn_out",
            "out_proj",
            "after_attn",
            "ffn_input",
            "ffn_norm",
            "mlp_gate_last",
            "mlp_up_last",
            "mlp_swiglu",
            "mlp_down",
            "routed_free_out",
            "moe_router_logits",
            "moe_routing_weights",
            "moe_routed_output",
            "moe_combined_output",
        )
    )
    try:
        trajectory = load_ck_greedy_trajectory(
            model_dir=runtime,
            prompt_tokens=token_ids,
            max_new_tokens=1,
        )
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    tensors: dict[int, dict[str, dict[str, Any]]] = {}
    for path in sorted(dump_dir.glob("tok_*_layer_*.f32")):
        layer_tail = path.name.split("_layer_", 1)[1]
        layer = int(layer_tail.split("_", 1)[0])
        boundary = layer_tail.split("_", 1)[1].removesuffix(".f32")
        values = np.fromfile(path, dtype=np.float32)
        retain_full = (
            mla_replay_layer is not None
            and (
                (
                    layer <= mla_replay_layer
                    and boundary in {"layer_out", "routed_free_out"}
                )
                or (
                    layer == mla_replay_layer
                    and boundary
                    in {
                        "block_rmsnorm",
                        "q_proj",
                        "mla_kv_a",
                        "mla_kv_norm",
                        "mla_k_nope",
                        "mla_value",
                        "mla_query",
                        "mla_key",
                        "mla_context",
                        "attn_gate",
                        "attn_out",
                        "out_proj",
                        "after_attn",
                        "ffn_input",
                        "ffn_norm",
                        "moe_router_logits",
                        "moe_routing_weights",
                        "moe_routed_output",
                        "moe_combined_output",
                    }
                )
            )
        )
        if retain_full:
            if values.size % len(token_ids) != 0:
                raise RuntimeError(
                    f"{path} has {values.size} values for {len(token_ids)} tokens"
                )
            tensors.setdefault(layer, {})[f"{boundary}_full"] = _write_tensor(
                output_dir / f"layer_{layer:03d}_{boundary}_full.f32",
                values.reshape(len(token_ids), -1),
            )
        try:
            boundary, last = _logical_ck_capture(
                values, boundary, len(token_ids)
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{path} {exc}") from exc
        tensors.setdefault(layer, {})[boundary] = _write_tensor(
            output_dir / f"layer_{layer:03d}_{boundary}.f32",
            last,
        )
    return tensors, trajectory["logits"][0]


def _literal_call_arg(operation: dict[str, Any], name: str) -> int | float:
    argument = next(
        (item for item in operation.get("args", []) if item.get("name") == name),
        None,
    )
    if argument is None:
        raise RuntimeError(f"MLA operation is missing {name!r}")
    try:
        value = ast.literal_eval(str(argument["expr"]))
    except (KeyError, SyntaxError, ValueError) as exc:
        raise RuntimeError(f"MLA {name!r} is not a numeric literal") from exc
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"MLA {name!r} is not numeric")
    return value


def _mla_operation(call_ir: dict[str, Any], layer: int) -> dict[str, Any]:
    matches = [
        operation
        for operation in call_ir.get("operations", [])
        if operation.get("op") == "mla_attention"
        and int(operation.get("layer", -1)) == layer
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one MLA attention operation for layer {layer}, got {len(matches)}"
        )
    return matches[0]


def _load_full_mla_tensor(
    tensors: dict[int, dict[str, dict[str, Any]]],
    layer: int,
    boundary: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    entry = tensors.get(layer, {}).get(f"{boundary}_full")
    if entry is None:
        raise RuntimeError(f"missing full MLA capture for layer {layer} {boundary}")
    values = np.fromfile(entry["path"], dtype=np.float32)
    if values.size != int(np.prod(shape)):
        raise RuntimeError(
            f"layer {layer} {boundary} has {values.size} values, expected {shape}"
        )
    return np.ascontiguousarray(values.reshape(shape))


def replay_mla_same_input(
    runtime: Path,
    call_ir: dict[str, Any],
    tensors: dict[int, dict[str, dict[str, Any]]],
    layer: int,
    token_count: int,
) -> dict[str, Any]:
    """Compare the resolved C and PyTorch MLA reductions from identical CKE inputs."""
    import torch

    operation = _mla_operation(call_ir, layer)
    heads = int(_literal_call_arg(operation, "num_heads"))
    kv_heads = int(_literal_call_arg(operation, "num_kv_heads"))
    qk_width = int(_literal_call_arg(operation, "qk_head_dim"))
    value_width = int(_literal_call_arg(operation, "v_head_dim"))
    scale = float(_literal_call_arg(operation, "scale"))
    q = _load_full_mla_tensor(
        tensors, layer, "mla_query", (token_count, heads, qk_width)
    )
    k = _load_full_mla_tensor(
        tensors, layer, "mla_key", (token_count, kv_heads, qk_width)
    )
    v = _load_full_mla_tensor(
        tensors, layer, "mla_value", (token_count, kv_heads, value_width)
    )
    observed = _load_full_mla_tensor(
        tensors, layer, "mla_context", (token_count, heads, value_width)
    )

    library = ctypes.CDLL(str((runtime / "libckernel_engine.so").resolve()))
    function = library.deepseek_mla_attention_f32_workspace
    float_pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [
        float_pointer,
        float_pointer,
        float_pointer,
        float_pointer,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        float_pointer,
        ctypes.c_size_t,
    ]
    function.restype = None
    c_output = np.empty((token_count, heads, value_width), dtype=np.float32)
    scores = np.empty(token_count, dtype=np.float32)
    function(
        q.ctypes.data_as(float_pointer),
        k.ctypes.data_as(float_pointer),
        v.ctypes.data_as(float_pointer),
        c_output.ctypes.data_as(float_pointer),
        heads,
        kv_heads,
        token_count,
        qk_width,
        value_width,
        scale,
        scores.ctypes.data_as(float_pointer),
        scores.nbytes,
    )

    if heads % kv_heads != 0:
        raise RuntimeError("MLA same-input replay requires integral GQA groups")
    q_t = torch.from_numpy(q).to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0)
    k_t = torch.from_numpy(k).to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0)
    v_t = torch.from_numpy(v).to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0)
    repeats = heads // kv_heads
    if repeats > 1:
        k_t = k_t.repeat_interleave(repeats, dim=1)
        v_t = v_t.repeat_interleave(repeats, dim=1)
    weights = torch.matmul(q_t, k_t.transpose(2, 3)) * scale
    causal = torch.triu(
        torch.full(
            (token_count, token_count),
            float("-inf"),
            dtype=torch.float32,
        ),
        diagonal=1,
    )
    weights = torch.softmax(weights.float() + causal, dim=-1).to(torch.bfloat16)
    torch_output = (
        torch.matmul(weights, v_t)
        .transpose(1, 2)
        .contiguous()[0]
        .float()
        .cpu()
        .numpy()
    )
    axes = ["token", "head", "feature"]
    return {
        "layer": layer,
        "kernel_id": operation.get("call_abi", {}).get("kernel_id"),
        "function": operation.get("function"),
        "attention_scale": scale,
        "shape": [token_count, heads, value_width],
        "captured_vs_c_replay": xray._metrics(observed, c_output, axes),
        "c_vs_pytorch_same_input": xray._metrics(c_output, torch_output, axes),
    }


def capture_ck_persistent(
    runtime: Path,
    prompt_ids: list[int],
    forced_ids: list[int],
    output_dir: Path,
) -> tuple[dict[int, dict[str, dict[str, Any]]], np.ndarray]:
    dump_dir = output_dir / "raw"
    dump_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "layer_out", "block_rmsnorm", "q_proj", "mla_kv_a", "mla_kv_norm",
        "mla_k_nope", "mla_value", "mla_query", "mla_key", "mla_context",
        "attn_gate", "attn_out", "out_proj",
        "after_attn", "ffn_input", "ffn_norm", "moe_router_logits", "moe_routing_weights",
        "mlp_gate", "mlp_gate_last", "mlp_up", "mlp_up_last", "mlp_swiglu", "mlp_down",
        "routed_free_out",
        "moe_routed_output", "moe_combined_output",
    )
    old = {
        name: os.environ.get(name)
        for name in (
            "CK_DEBUG_EXPORT_HIDDEN",
            "CK_DEBUG_EXPORT_HIDDEN_NAME",
            "CK_DEBUG_EXPORT_HIDDEN_NAMES",
        )
    }
    os.environ["CK_DEBUG_EXPORT_HIDDEN"] = str(dump_dir.resolve())
    os.environ.pop("CK_DEBUG_EXPORT_HIDDEN_NAME", None)
    os.environ["CK_DEBUG_EXPORT_HIDDEN_NAMES"] = ",".join(names)
    try:
        trajectory = load_ck_greedy_trajectory(
            model_dir=runtime,
            prompt_tokens=prompt_ids,
            max_new_tokens=len(forced_ids) + 1,
        )
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    generated_prefix = trajectory["generated_tokens"][: len(forced_ids)]
    if generated_prefix != forced_ids:
        raise RuntimeError(
            "persistent trajectory left the teacher-forced prefix before the "
            f"target: generated={generated_prefix}, required={forced_ids}"
        )
    candidates: dict[tuple[int, str], tuple[int, Path]] = {}
    for path in dump_dir.glob("tok_*_layer_*.f32"):
        token_pos = int(path.name.split("tok_", 1)[1].split("_", 1)[0])
        layer_tail = path.name.split("_layer_", 1)[1]
        layer = int(layer_tail.split("_", 1)[0])
        boundary = layer_tail.split("_", 1)[1].removesuffix(".f32")
        key = (layer, boundary)
        if key not in candidates or token_pos > candidates[key][0]:
            candidates[key] = (token_pos, path)
    tensors: dict[int, dict[str, dict[str, Any]]] = {}
    for (layer, boundary), (_token_pos, path) in candidates.items():
        values = np.fromfile(path, dtype=np.float32)
        tensors.setdefault(layer, {})[boundary] = _write_tensor(
            output_dir / f"layer_{layer:03d}_{boundary}.f32", values
        )
    return tensors, trajectory["logits"][len(forced_ids)]


def operation_metadata(call_ir: dict[str, Any], layer: int, boundary: str) -> dict[str, str]:
    producer_by_boundary = {
        "block_rmsnorm": "block_rmsnorm",
        "q_proj": "q_proj",
        "mla_kv_a": "kv_a_proj",
        "mla_kv_norm": "kv_a_layernorm",
        "mla_k_nope": "kv_lora_decompress",
        "mla_value": "kv_lora_decompress",
        "mla_query": "partial_rope_concat",
        "mla_key": "partial_rope_concat",
        "mla_context": "mla_attention",
        "attn_gate": "attention_gate_projection",
        "attn_out": "attn_gate_sigmoid_mul",
        "out_proj": "out_proj",
        "after_attn": "residual_add",
        "ffn_input": "block_rmsnorm",
        "ffn_norm": "block_rmsnorm",
        "mlp_gate": "mlp_gate_up",
        "mlp_up": "mlp_gate_up",
        "mlp_swiglu": "silu_mul",
        "mlp_down": "mlp_down",
        "routed_free_out": "farskip_routed_shared_combine",
        "moe_router_logits": "moe_router",
        "moe_routing_weights": "group_limited_topk_router",
        "moe_routed_output": "moe_swiglu_expert_mlp",
        "moe_combined_output": "shared_swiglu_expert_mlp",
        "layer_out": "residual_add",
    }
    producers = (
        {"residual_add", "farskip_routed_shared_combine"}
        if boundary == "layer_out"
        else {producer_by_boundary.get(boundary, boundary)}
    )
    matches = [
        operation
        for operation in call_ir.get("operations", [])
        if int(operation.get("layer", -1)) == layer
        and operation.get("op") in producers
    ]
    operation = matches[-1] if matches else {}
    resolved = operation.get("resolved_numerical_execution") or {}
    call_abi = operation.get("call_abi") or {}
    kernel_id = (
        operation.get("kernel_id")
        or resolved.get("kernel_id")
        or call_abi.get("kernel_id")
        or operation.get("function")
        or "unresolved"
    )
    contract_id = (
        operation.get("resolved_contract_id")
        or resolved.get("contract_id")
        or f"{call_abi.get('owner', 'unresolved')}:{kernel_id}"
    )
    return {
        "producer": str(operation.get("op") or sorted(producers)[0]),
        "resolved_contract_id": str(contract_id),
        "kernel_id": str(kernel_id),
        "function": str(
            operation.get("function")
            or resolved.get("function")
            or "unresolved"
        ),
    }


def manifest(
    backend: str,
    model_name: str,
    source: Path,
    tensors: dict[int, dict[str, dict[str, Any]]],
    call_ir: dict[str, Any],
    selected: list[tuple[int, str]],
) -> dict[str, Any]:
    checkpoints = []
    for layer, boundary in selected:
        tensor = tensors.get(layer, {}).get(boundary)
        if tensor is None:
            continue
        metadata = operation_metadata(call_ir, layer, boundary)
        checkpoints.append(
            {
                "checkpoint_id": f"decoder.layer.{layer}.{boundary}",
                "producer": metadata["producer"],
                "phase": "teacher_forced",
                "layer": layer,
                "tensor_path": tensor["path"],
                "storage_dtype": "bf16",
                "exported_dtype": "fp32",
                "logical_shape": tensor["shape"],
                "physical_shape": tensor["shape"],
                "logical_layout": "channel",
                "axis_names": ["channel"],
                "physical_axis_names": ["channel"],
                "resolved_contract_id": metadata["resolved_contract_id"],
                "kernel_id": metadata["kernel_id"],
                "function": metadata["function"],
                "sha256": tensor["sha256"],
            }
        )
    return {
        "schema": "cke.checkpoint_manifest",
        "schema_version": 1,
        "backend": backend,
        "run": {
            "model": model_name,
            "phase": "teacher_forced",
            "source": str(source.resolve()),
        },
        "checkpoints": checkpoints,
    }


def profile(order: list[str], intervals: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "schema": "cke.parity_profile",
        "schema_version": 1,
        "name": "pytorch_bf16_decoder_xray",
        "backend": "pytorch",
        "contract_schema_version": 1,
        "required_match_fields": [
            "checkpoint_id", "producer", "logical_layout", "axis_names",
            "resolved_contract_id", "kernel_id", "function",
        ],
        "observed_storage": {"default": "bf16", "checkpoints": {}},
        "dtype_thresholds": {
            "bf16": {
                "cosine_min": 0.999,
                "rmse_max": 0.02,
                "relative_rmse_max": 0.02,
                "max_abs_max": 0.25,
                "finite_required": True,
            }
        },
        "checkpoint_order": order,
        "interval_expansions": intervals,
        "backend_mappings": {},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    token_ids = parse_token_ids(args.token_ids)
    if not token_ids:
        raise ValueError("at least one teacher-forced token is required")
    persistent_split = split_teacher_forced_tokens(
        token_ids, getattr(args, "prompt_token_count", None)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    call_ir = json.loads(args.call_ir.read_text(encoding="utf-8"))
    mla_replay_layer = getattr(args, "mla_replay_layer", None)
    torch_tensors, torch_logits = capture_pytorch(
        args.checkpoint,
        token_ids,
        args.output_dir / "pytorch",
        args.threads,
        mla_replay_layer,
    )
    ck_tensors, ck_logits = capture_ck(
        args.runtime,
        token_ids,
        args.output_dir / "ck",
        mla_replay_layer,
    )
    layer_count = min(len(torch_tensors), len(ck_tensors))
    all_layers = list(range(layer_count))
    sparse_layers = sorted({0, layer_count // 3, (2 * layer_count) // 3, layer_count - 1})
    all_selected = [(layer, "layer_out") for layer in all_layers]
    sparse_selected = [(layer, "layer_out") for layer in sparse_layers]
    all_order = [f"decoder.layer.{layer}.layer_out" for layer in all_layers]
    sparse_order = [f"decoder.layer.{layer}.layer_out" for layer in sparse_layers]
    intervals = {}
    for lower, upper in zip(sparse_layers, sparse_layers[1:]):
        intervals[f"decoder.layer.{lower}.layer_out->decoder.layer.{upper}.layer_out"] = [
            f"decoder.layer.{layer}.layer_out" for layer in range(lower + 1, upper)
        ] or [f"decoder.layer.{upper}.layer_out"]

    manifests = {
        "ck_all": manifest("ck", args.model_name, args.runtime, ck_tensors, call_ir, all_selected),
        "pytorch_all": manifest(
            "pytorch", args.model_name, args.checkpoint, torch_tensors, call_ir, all_selected
        ),
        "ck_sparse": manifest(
            "ck", args.model_name, args.runtime, ck_tensors, call_ir, sparse_selected
        ),
        "pytorch_sparse": manifest(
            "pytorch", args.model_name, args.checkpoint, torch_tensors, call_ir, sparse_selected
        ),
    }
    sparse_report = xray.compare_manifests(
        manifests["ck_sparse"],
        manifests["pytorch_sparse"],
        profile(sparse_order, intervals),
        checkpoint_order=sparse_order,
    )
    all_report = xray.compare_manifests(
        manifests["ck_all"],
        manifests["pytorch_all"],
        profile(all_order, {}),
        checkpoint_order=all_order,
    )
    first_material = all_report.get("first_divergence") or {}
    material_id = str(first_material.get("checkpoint_id") or "")
    target_layer = (
        int(material_id.split(".")[2])
        if material_id.startswith("decoder.layer.")
        else 0
    )
    granular_boundaries = [
        "block_rmsnorm",
        "q_proj",
        "mla_kv_a",
        "mla_kv_norm",
        "mla_k_nope",
        "mla_value",
        "mla_query",
        "mla_key",
        "mla_context",
        "attn_gate",
        "attn_out",
        "out_proj",
        "after_attn",
        "ffn_input",
        "ffn_norm",
        "mlp_gate",
        "mlp_up",
        "mlp_swiglu",
        "mlp_down",
        "moe_router_logits",
        "moe_routing_weights",
        "moe_routed_output",
        "moe_combined_output",
        "routed_free_out",
        "layer_out",
    ]
    granular_selected = [
        (target_layer, boundary)
        for boundary in granular_boundaries
        if boundary in ck_tensors.get(target_layer, {})
        and boundary in torch_tensors.get(target_layer, {})
    ]
    granular_order = [
        f"decoder.layer.{target_layer}.{boundary}"
        for _layer, boundary in granular_selected
    ]
    granular_report = None
    if len(granular_selected) >= 2:
        manifests["ck_granular"] = manifest(
            "ck", args.model_name, args.runtime, ck_tensors, call_ir, granular_selected
        )
        manifests["pytorch_granular"] = manifest(
            "pytorch", args.model_name, args.checkpoint, torch_tensors, call_ir,
            granular_selected,
        )
        granular_report = xray.compare_manifests(
            manifests["ck_granular"],
            manifests["pytorch_granular"],
            profile(granular_order, {}),
            checkpoint_order=granular_order,
        )
    ranking = compare_logits(ck_logits, torch_logits, args.top_k)
    ranking["ck_top1"] = int(np.argmax(ck_logits))
    ranking["pytorch_top1"] = int(np.argmax(torch_logits))
    ranking["top1_match"] = ranking["ck_top1"] == ranking["pytorch_top1"]

    mla_same_input = None
    if mla_replay_layer is not None:
        mla_same_input = replay_mla_same_input(
            args.runtime,
            call_ir,
            ck_tensors,
            mla_replay_layer,
            len(token_ids),
        )
        (args.output_dir / "mla_same_input_replay.json").write_text(
            json.dumps(mla_same_input, indent=2) + "\n",
            encoding="utf-8",
        )

    state_report = None
    state_granular_report = None
    state_ranking = None
    state_layer = None
    if persistent_split is not None:
        prompt_ids, forced_ids = persistent_split
        persistent_tensors, persistent_logits = capture_ck_persistent(
            args.runtime,
            prompt_ids,
            forced_ids,
            args.output_dir / "ck_persistent",
        )
        state_layers = sorted(set(ck_tensors) & set(persistent_tensors))
        state_selected = [
            (layer, "layer_out")
            for layer in state_layers
            if "layer_out" in ck_tensors[layer]
            and "layer_out" in persistent_tensors[layer]
        ]
        state_order = [
            f"decoder.layer.{layer}.layer_out"
            for layer, _boundary in state_selected
        ]
        manifests["ck_replay_state"] = manifest(
            "ck_replay",
            args.model_name,
            args.runtime,
            ck_tensors,
            call_ir,
            state_selected,
        )
        manifests["ck_persistent_state"] = manifest(
            "ck_persistent",
            args.model_name,
            args.runtime,
            persistent_tensors,
            call_ir,
            state_selected,
        )
        if state_order:
            state_report = xray.compare_manifests(
                manifests["ck_persistent_state"],
                manifests["ck_replay_state"],
                profile(state_order, {}),
                checkpoint_order=state_order,
            )
            first_state_edge = (
                state_report.get("first_divergence")
                or state_report.get("first_non_exact_checkpoint")
                or {}
            )
            state_checkpoint = str(first_state_edge.get("checkpoint_id") or "")
            if state_checkpoint.startswith("decoder.layer."):
                state_layer = int(state_checkpoint.split(".")[2])

        if state_layer is not None:
            state_granular_selected = [
                (state_layer, boundary)
                for boundary in granular_boundaries
                if boundary in ck_tensors.get(state_layer, {})
                and boundary in persistent_tensors.get(state_layer, {})
            ]
            state_granular_order = [
                f"decoder.layer.{state_layer}.{boundary}"
                for _layer, boundary in state_granular_selected
            ]
            if len(state_granular_order) >= 2:
                manifests["ck_replay_state_granular"] = manifest(
                    "ck_replay",
                    args.model_name,
                    args.runtime,
                    ck_tensors,
                    call_ir,
                    state_granular_selected,
                )
                manifests["ck_persistent_state_granular"] = manifest(
                    "ck_persistent",
                    args.model_name,
                    args.runtime,
                    persistent_tensors,
                    call_ir,
                    state_granular_selected,
                )
                state_granular_report = xray.compare_manifests(
                    manifests["ck_persistent_state_granular"],
                    manifests["ck_replay_state_granular"],
                    profile(state_granular_order, {}),
                    checkpoint_order=state_granular_order,
                )

        state_ranking = compare_logits(persistent_logits, ck_logits, args.top_k)
        state_ranking["persistent_top1"] = int(np.argmax(persistent_logits))
        state_ranking["replay_top1"] = int(np.argmax(ck_logits))
        state_ranking["top1_match"] = (
            state_ranking["persistent_top1"] == state_ranking["replay_top1"]
        )

    for name, value in manifests.items():
        (args.output_dir / f"{name}.checkpoints.json").write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
    for name, value in (
        ("sparse_report.json", sparse_report),
        ("all_layers_report.json", all_report),
        ("granular_report.json", granular_report),
        ("ranking.json", ranking),
        ("state_report.json", state_report),
        ("state_granular_report.json", state_granular_report),
        ("state_ranking.json", state_ranking),
    ):
        if value is None:
            continue
        (args.output_dir / name).write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
    ranking_pass = bool(ranking["top1_match"])
    numerical_pass = all_report.get("status") == "pass"
    state_pass = (
        persistent_split is None
        or (
            state_report is not None
            and state_report.get("status") == "pass"
            and state_ranking is not None
            and bool(state_ranking["top1_match"])
        )
    )
    overall_pass = ranking_pass and numerical_pass and state_pass
    summary = {
        "schema": "cke.xray.decoder_pytorch",
        "schema_version": 1,
        "status": "pass" if overall_pass else "diverged",
        "ranking_status": "pass" if ranking_pass else "diverged",
        "numerical_status": "pass" if numerical_pass else "diverged",
        "persistent_state_status": (
            "not_run"
            if persistent_split is None
            else ("pass" if state_pass else "diverged")
        ),
        "token_count": len(token_ids),
        "ranking": ranking,
        "mla_same_input_replay": mla_same_input,
        "sparse_first_non_exact": sparse_report.get("first_non_exact_checkpoint"),
        "sparse_first_material": sparse_report.get("first_divergence"),
        "all_layers_first_non_exact": all_report.get("first_non_exact_checkpoint"),
        "all_layers_first_material": all_report.get("first_divergence"),
        "granular_layer": target_layer,
        "granular_first_non_exact": (
            granular_report.get("first_non_exact_checkpoint")
            if granular_report else None
        ),
        "granular_first_material": (
            granular_report.get("first_divergence")
            if granular_report else None
        ),
        "persistent_vs_replay": (
            {
                "prompt_token_count": len(persistent_split[0]),
                "forced_token_count": len(persistent_split[1]),
                "ranking": state_ranking,
                "first_non_exact": (
                    state_report.get("first_non_exact_checkpoint")
                    if state_report else None
                ),
                "first_material": (
                    state_report.get("first_divergence")
                    if state_report else None
                ),
                "granular_layer": state_layer,
                "granular_first_non_exact": (
                    state_granular_report.get("first_non_exact_checkpoint")
                    if state_granular_report else None
                ),
                "granular_first_material": (
                    state_granular_report.get("first_divergence")
                    if state_granular_report else None
                ),
            }
            if persistent_split is not None
            else None
        ),
        "attribution_scope": (
            "persistent_vs_replay"
            if persistent_split is not None and not state_pass
            else "layer_interval"
        ),
        "next_action": (
            "Fix the first persistent-versus-replay semantic edge using its "
            "resolved execution contract."
            if persistent_split is not None and not state_pass
            else (
                "Expand the first material PyTorch comparison layer into "
                "semantic sub-edges before attributing the failure."
                if not numerical_pass
                else "Advance the teacher-forced parity window."
            )
        ),
    }
    (args.output_dir / "xray_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--call-ir", type=Path, required=True)
    parser.add_argument("--token-ids", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="decoder")
    parser.add_argument(
        "--prompt-token-count",
        type=int,
        help=(
            "Split --token-ids into the original prompt and teacher-forced "
            "suffix, then compare persistent decode against full replay."
        ),
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument(
        "--mla-replay-layer",
        type=int,
        help=(
            "Retain complete MLA Q/K/V/context tensors for one layer and compare "
            "the resolved C and PyTorch reductions from identical CKE inputs."
        ),
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
