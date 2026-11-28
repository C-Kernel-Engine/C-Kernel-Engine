"""Generated single-tile RADIO v2.5-H encoder for Nemotron Nano 12B v2 VL.

This is the FP32 execution lane: imported BF16 weights are widened exactly,
but intermediate activations are not rounded to BF16 between providers.
"""

from pathlib import Path

import numpy as np

from version.v8.scripts.instella_vl_checked_v8 import CompiledClipStage, Graph, patch_projection
from version.v8.scripts.nemotron_vl_checked_v8 import CHANNELS, GRID, validate_publisher_config


REGISTERS = 16
LAYERS = 32
PATCH = 16
RESOLUTION = GRID * PATCH
RADIO = "vision_model.radio_model.model."


def radio_block(rows, width, heads, inner):
    """RADIO ViT-H: full attention and exact-erf GELU, with no final norm."""
    if any(type(v) is not int or v <= 0 for v in (rows, width, heads, inner)) or width % heads:
        raise ValueError("invalid RADIO block geometry")
    g = Graph("radio_v2_5_block", rows, width, heads, inner)
    g.norm("norm1", "external:input", width, "norm1", epsilon=1e-6)
    for name in ("q", "k", "v"):
        g.project(name, "norm1", width, width, name)
    count, scratch = rows * width, rows * (width + 1)
    g.op("context", "attention_cross_token_major_checked", "attention_cross_token_major_checked_f32",
         {"query": "q", "key": "k", "value": "v"}, width, {
             "query_elements": count, "key_elements": count, "value_elements": count,
             "output_elements": count, "scratch_elements": scratch, "query_tokens": rows,
             "memory_tokens": rows, "heads": heads, "head_dim": width // heads,
         }, Q=rows, S=rows, H=heads, D=width // heads, X=scratch)
    g.project("attention_out", "context", width, width, "attn.proj")
    g.add("attention_residual", "external:input", "attention_out", width)
    g.norm("norm2", "attention_residual", width, "norm2", epsilon=1e-6)
    g.project("expanded", "norm2", width, inner, "mlp.fc1")
    count = rows * inner
    g.op("activated", "gelu", "gelu_rows_erf_checked_f32", {"input": "expanded"}, inner, {
        "gelu_input_elements": count, "gelu_input_stride": inner,
        "gelu_output_elements": count, "gelu_output_stride": inner,
        "gelu_scratch_elements": count, "gelu_rows": rows, "gelu_channels": inner,
    }, M=rows, C=inner, Q=count)
    g.project("down", "activated", inner, width, "mlp.fc2")
    g.add("result", "attention_residual", "down", width)
    return g.finish()


def positioned_patches():
    g = Graph("radio_v2_5_positioned_patches", GRID**2, CHANNELS, 1, CHANNELS)
    g.buffers["position"] = {"shape": [GRID**2, CHANNELS]}
    g.add("result", "external:input", "external:position", CHANNELS)
    return g.finish()


def bf16_bits(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def widen(stored):
    if not isinstance(stored, np.ndarray) or stored.dtype != np.uint16:
        raise ValueError("expected stored BF16 weight bits")
    out = (stored.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("nonfinite RADIO tensor")
    return out


def import_positions(stored):
    if stored.shape != (1, 128**2, CHANNELS):
        raise ValueError("unexpected RADIO position grid")
    grid = widen(stored).reshape(128, 128, CHANNELS)
    at = np.arange(GRID, dtype=np.float32) * np.float32(127 / (GRID - 1))
    left = at.astype(np.int32)
    right = np.minimum(left + 1, 127)
    frac = at - left.astype(np.float32)
    lo, hi = (1 - frac)[None, :, None], frac[None, :, None]
    horizontal = grid[:, left] * lo + grid[:, right] * hi
    lo, hi = (1 - frac)[:, None, None], frac[:, None, None]
    return widen(bf16_bits(horizontal[left] * lo + horizontal[right] * hi)).reshape(-1, CHANNELS)


def tile_patches(image, mean, std):
    """Pack the normalized fixed RGB tile in channel, local y, local x order."""
    if image.shape != (RESOLUTION, RESOLUTION, 3) or image.dtype != np.uint8:
        raise ValueError("expected one fixed 512 x 512 RGB tile")
    if (mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError("invalid RADIO normalization constants")
    pixel = (image.astype(np.float32) / 255 - mean.astype(np.float32)) / std.astype(np.float32)
    return widen(bf16_bits(pixel.reshape(GRID, PATCH, GRID, PATCH, 3)
                           .transpose(0, 2, 4, 1, 3).reshape(GRID**2, 3 * PATCH**2)))


class CompiledRadioEncoder:
    """Own a fixed tile graph and reload the weights of its 32 identical-topology blocks."""

    def __init__(self, directory, config, read_weight, mean, std):
        validate_publisher_config(config)
        vision = config["vision_config"]
        args = vision.get("args", {})
        if (not isinstance(args, dict) or args.get("model") != "vit_huge_patch16_224"
                or args.get("model_norm") is not False
                or args.get("cpe_max_size") != 2048
                or "feature_normalizer_config" not in vision or vision["feature_normalizer_config"] is not None):
            raise ValueError("unsupported RADIO block or final normalization")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.read_weight = read_weight
        self.mean, self.std = np.asarray(mean).copy(), np.asarray(std).copy()
        self.positions = import_positions(read_weight(RADIO + "patch_generator.pos_embed"))
        self.registers = widen(read_weight(RADIO + "patch_generator.cls_token.token"))
        if self.registers.shape != (REGISTERS, CHANNELS):
            raise ValueError("unexpected RADIO register extent")
        self.patch = CompiledClipStage(
            directory / "patch", patch_projection(GRID**2, 3 * PATCH**2, CHANNELS),
            {"patch.weight": widen(read_weight(RADIO + "patch_generator.embedder.weight"))},
            openmp=True,
        )
        self.position = CompiledClipStage(directory / "position", positioned_patches(), {})
        graph = radio_block(GRID**2 + REGISTERS, CHANNELS, 16, 4 * CHANNELS)
        self.block = CompiledClipStage(directory / "block", graph, self._weights(0, graph.weights),
                                       openmp=True)

    def _weights(self, layer, keys):
        prefix = RADIO + f"blocks.{layer}."
        values = {}
        for suffix in ("weight", "bias"):
            qkv = widen(self.read_weight(prefix + "attn.qkv." + suffix))
            if qkv.shape[0] != 3 * CHANNELS:
                raise ValueError("unexpected RADIO fused QKV extent")
            for name, value in zip(("q", "k", "v"), np.split(qkv, 3)):
                values[f"{name}.{suffix}"] = value
        for key in keys.keys() - values.keys():
            values[key] = widen(self.read_weight(prefix + key))
        return values

    def __call__(self, rgb, *, checkpoint=None):
        patches = self.patch(tile_patches(rgb, self.mean, self.std))
        hidden = np.concatenate((self.registers, self.position(patches, position=self.positions)), axis=0)
        for layer in range(LAYERS):
            self.block.load(self._weights(layer, self.block.weights))
            hidden = self.block(hidden)
            if checkpoint is not None:
                checkpoint(layer, hidden)
        return bf16_bits(hidden[REGISTERS:])
