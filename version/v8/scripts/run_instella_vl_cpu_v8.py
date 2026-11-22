#!/usr/bin/env python3
"""Bounded Instella-VL-1B image-to-answer with CKE and a Python host.

Use the publisher's FP16 SafeTensors and tokenizer.json. The standalone host
uses NumPy, Pillow, SafeTensors and tokenizers, but neither PyTorch nor AMD
model code. The checked encoder is opt-in; the decoder uses the CKE
kernel ABI directly. This is not a serving runtime.
"""

from __future__ import annotations

import argparse
import ctypes as C
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

from replay_f16_patch_projection_v8 import replay
from xray_operation_bundle_v8 import sha256


TILE = 336
GRID = 24
WIDTH = 2048
VISION = "model.vision_tower.vision_tower.vision_model."
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def image_tiles(image, candidates):
    width, height = image.size

    def score(target):
        w, h = target
        scale = min(w / width, h / height)
        effective = min(int(width * scale) * int(height * scale), width * height)
        return effective, effective - w * h

    w, h = max(candidates, key=score)
    if w % TILE or h % TILE or not (0 < w <= 5 * TILE and 0 < h <= 5 * TILE):
        raise ValueError("unsupported Instella tile grid")
    scale = min(w / width, h / height)
    size = min(math.ceil(width * scale), w), min(math.ceil(height * scale), h)
    padded = Image.new("RGB", (w, h))
    padded.paste(image.resize(size, Image.Resampling.BICUBIC),
                 ((w - size[0]) // 2, (h - size[1]) // 2))
    tiles = [image.resize((TILE, TILE), Image.Resampling.BICUBIC)]
    tiles.extend(padded.crop((x, y, x + TILE, y + TILE))
                 for y in range(0, h, TILE) for x in range(0, w, TILE))
    pixels = (np.stack([np.asarray(tile) for tile in tiles]).astype(np.float64) / 255).astype(np.float32)
    return np.ascontiguousarray(((pixels - MEAN) / STD).transpose(0, 3, 1, 2)), (w // TILE, h // TILE)


def prompt_ids(tokenizer, prompt):
    if not prompt.strip() or "<image>" in prompt:
        raise ValueError("expected one nonempty text prompt without image markers")
    before = tokenizer.encode("|||IP_ADDRESS|||<|user|>\n").ids
    after = tokenizer.encode("\n" + prompt.strip() + "\n<|assistant|>\n").ids
    return before, after


def spatial_rows(source_size, tile_grid):
    gx, gy = tile_grid
    h, w = gy * GRID, gx * GRID
    width, height = source_size
    pad_y = (h - int(height * (w / width))) // 2 if width * h > height * w else 0
    pad_x = (w - int(width * (h / height))) // 2 if width * h <= height * w else 0
    return slice(pad_y, h - pad_y), slice(pad_x, w - pad_x)


def bridge(projected, newline, source_size, tile_grid):
    gx, gy = tile_grid
    expected = (1 + gx * gy, GRID * GRID, WIDTH)
    if projected.shape != expected or newline.shape != (WIDTH,):
        raise ValueError("invalid projected tile/newline shape")
    rows = projected[1:].reshape(gy, gx, GRID, GRID, WIDTH)
    rows = rows.transpose(0, 2, 1, 3, 4).reshape(gy * GRID, gx * GRID, WIDTH)
    ys, xs = spatial_rows(source_size, tile_grid)
    rows = rows[ys, xs]
    separators = np.broadcast_to(newline, (rows.shape[0], 1, WIDTH))
    spatial = np.concatenate((rows, separators), axis=1).reshape(-1, WIDTH)
    return np.concatenate((projected[0], spatial))


class Kernels:
    def __init__(self, engine, weights, threads):
        self.engine, self.weights = engine, weights
        self.lib = C.CDLL(str(engine.resolve()))
        p, i, f = C.c_void_p, C.c_int, C.c_float

        def bind(name, types):
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = types, None
            return fn

        bind("ck_set_num_threads", [i])(threads)
        bind("ck_set_strict_parity", [i])(0)
        self.gemm = bind("gemm_blocked_serial", [p] * 4 + [i] * 3)
        self.layernorm = bind("layernorm_fp32_exact_parallel_dispatch", [p] * 6 + [i, i, f])
        self.full = bind("attention_forward_full_head_major_gqa_exact_strided", [p] * 4 + [i] * 6)
        self.causal = bind("attention_forward_causal_head_major_gqa_flash", [p] * 4 + [i] * 5)
        self.decode = bind("attention_forward_decode_head_major_gqa_flash", [p] * 4 + [i] * 6)
        self.rope = bind("rope_forward_qk_split_direct_f32", [p] * 3 + [i] * 8 + [f])
        self.residual = bind("ck_residual_add_token_major", [p] * 3 + [i, i])
        self.sigmoid = bind("sigmoid_forward", [p, p, C.c_size_t])
        self.gelu = bind("gelu_pytorch_erf_f32_inplace", [p, C.c_size_t])
        self.swiglu = bind("swiglu_forward_ggml_split", [p] * 3 + [i, i])

    def weight(self, name, shape=None, widen=True):
        w = self.weights.get_tensor(name)
        if w.dtype != np.float16 or (shape is not None and w.shape != shape):
            raise ValueError(f"unexpected weight storage or shape: {name}: {w.dtype} {w.shape}")
        return w.astype(np.float32) if widen else w

    def linear(self, x, key, channels, bias=False):
        w = self.weight(key + ".weight", (channels, x.shape[-1]))
        b = self.weight(key + ".bias", (channels,)) if bias else None
        return self.project(x, w, b)

    def project(self, x, w, b=None):
        out = np.empty((*x.shape[:-1], len(w)), np.float32)
        width = x.shape[-1]
        self.gemm(x.ctypes.data, w.ctypes.data, None if b is None else b.ctypes.data,
                  out.ctypes.data, x.size // width, len(w), width)
        return out

    def norm(self, x, key=None):
        width = x.shape[-1]
        a = self.weight(key + ".weight", (width,)) if key else None
        b = self.weight(key + ".bias", (width,)) if key else None
        y = np.empty_like(x)
        self.layernorm(x.ctypes.data, None if a is None else a.ctypes.data,
                       None if b is None else b.ctypes.data, y.ctypes.data, None, None,
                       x.size // width, width, 1e-5)
        return y

    def add(self, x, y):
        out = np.empty_like(x)
        self.residual(x.ctypes.data, y.ctypes.data, out.ctypes.data,
                      x.size // x.shape[-1], x.shape[-1])
        return out

    def qkv(self, x, key, bias):
        batch, tokens, width = x.shape
        return [np.ascontiguousarray(self.linear(x, key + name + "_proj", width, bias)
                                     .reshape(batch, tokens, 16, width // 16).transpose(0, 2, 1, 3))
                for name in ("q", "k", "v")]

    def encode_generated(self, tiles, directory):
        from instella_vl_checked_v8 import CompiledClipStage, clip_block, input_norm, patch_projection, projector
        from replay_f16_patch_projection_v8 import patch_rows

        def load(graph, stem):
            return {
                name: np.full(shape, 1.702, np.float32) if name == "quick_gelu_scale"
                else self.weight(stem + name, tuple(shape))
                for name, (shape, _) in graph.weights.items()
            }

        base = VISION + "encoder.layers."
        block = clip_block()
        proj = projector()
        norm = input_norm()
        patch = patch_projection()
        directory.mkdir(parents=True, exist_ok=False)
        patch_key = VISION + "embeddings.patch_embedding.weight"
        patch_run = CompiledClipStage(directory / "patch", patch, {
            "patch.weight": self.weight(patch_key, (1024, 3, 14, 14)).reshape(1024, 588),
        }, openmp=True)
        norm_run = CompiledClipStage(directory / "input_norm", norm, load(norm, VISION), openmp=True)
        block_run = CompiledClipStage(directory / "block", block, load(block, base + "0."), openmp=True)
        project_run = CompiledClipStage(directory / "projector", proj, load(proj, "model.mm_projector."),
                                       openmp=True)
        positions = self.weight(VISION + "embeddings.position_embedding.weight", (577, 1024))
        states = np.empty((len(tiles), 577, 1024), np.float32)
        states[:, 0] = self.weight(VISION + "embeddings.class_embedding", (1024,))
        for i, tile in enumerate(tiles):
            states[i, 1:] = patch_run(patch_rows(tile[None], 14, 14))
            states[i] = norm_run(states[i], position=positions)
        for layer in range(23):
            if layer:
                block_run.load(load(block, base + f"{layer}."))
            for i in range(len(tiles)):
                states[i] = block_run(states[i])
            print(f"generated CLIP block {layer} complete", flush=True)
        return np.stack([project_run(state[1:]) for state in states])

    def encode(self, tiles):
        patch_w = self.weight(VISION + "embeddings.patch_embedding.weight", (1024, 3, 14, 14), False)
        patches = replay(tiles, patch_w, self.engine)
        x = np.empty((len(tiles), 577, 1024), np.float32)
        x[:, 0] = self.weight(VISION + "embeddings.class_embedding", (1024,))
        x[:, 1:] = patches.reshape(len(tiles), 1024, 576).transpose(0, 2, 1)
        x += self.weight(VISION + "embeddings.position_embedding.weight", (577, 1024))
        x = self.norm(x, VISION + "pre_layrnorm")
        for i in range(23):
            key = VISION + f"encoder.layers.{i}."
            qkv = self.qkv(self.norm(x, key + "layer_norm1"), key + "self_attn.", True)
            out = np.empty_like(qkv[0])
            for q, k, v, y in zip(*qkv, out):
                self.full(q.ctypes.data, k.ctypes.data, v.ctypes.data, y.ctypes.data,
                          16, 16, 577, 64, 64, 577)
            a = np.ascontiguousarray(out.transpose(0, 2, 1, 3)).reshape(x.shape)
            x = self.add(x, self.linear(a, key + "self_attn.out_proj", 1024, True))
            h = self.linear(self.norm(x, key + "layer_norm2"), key + "mlp.fc1", 4096, True)
            scaled = h * np.float32(1.702)
            gate = np.empty_like(h)
            self.sigmoid(scaled.ctypes.data, gate.ctypes.data, h.size)
            h *= gate
            x = self.add(x, self.linear(h, key + "mlp.fc2", 1024, True))
        y = self.linear(np.ascontiguousarray(x[:, 1:]), "model.mm_projector.0", WIDTH, True)
        self.gelu(y.ctypes.data, y.size)
        return self.linear(y, "model.mm_projector.2", WIDTH, True)

    def decoder(self, x, embed, caches, position):
        rows = x.shape[1]
        for i, (kc, vc) in enumerate(caches):
            key = f"model.layers.{i}."
            q, k, v = [a[0] for a in self.qkv(self.norm(x), key + "self_attn.", False)]
            self.rope(q.ctypes.data, k.ctypes.data, None, 0, 16, 16, rows, 128, 128, position, 128, 10000)
            end = position + rows
            kc[:, position:end], vc[:, position:end] = k, v
            y = np.empty_like(q)
            if position == 0:
                self.causal(q.ctypes.data, k.ctypes.data, v.ctypes.data, y.ctypes.data,
                            16, 16, rows, 128, 128)
            else:
                if rows != 1:
                    raise ValueError("only singleton continuation is supported")
                self.decode(q.ctypes.data, kc.ctypes.data, vc.ctypes.data, y.ctypes.data,
                            16, 16, end, kc.shape[1], 128, 128)
            a = np.ascontiguousarray(y.transpose(1, 0, 2)).reshape(x.shape)
            x = self.add(x, self.linear(a, key + "self_attn.o_proj", WIDTH))
            n = self.norm(x)
            g = self.linear(n, key + "mlp.gate_proj", 8192)
            u = self.linear(n, key + "mlp.up_proj", 8192)
            h = np.empty_like(g)
            self.swiglu(g.ctypes.data, u.ctypes.data, h.ctypes.data, rows, 8192)
            x = self.add(x, self.linear(h, key + "mlp.down_proj", WIDTH))
        logits = self.project(self.norm(np.ascontiguousarray(x[:, -1:])), embed)[0, 0]
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite native logits")
        return logits


def main():
    from safetensors import safe_open
    from tokenizers import Tokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-tokens", default=16, type=int)
    p.add_argument("--threads", default=4, type=int)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--prefix-out", type=Path, help="Optional local NPY with CKE's visual prefix")
    p.add_argument("--generated-vision-dir", type=Path, help="Build and run the checked generated encoder stages")
    p.add_argument("--expect-answer", help="Require an exact case-insensitive answer and a normal stop")
    args = p.parse_args()
    if not 1 <= args.max_tokens <= 64 or not 1 <= args.threads <= 32:
        p.error("expected 1..64 generated tokens and 1..32 threads")
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    contracts = {"model_type": "instellavl", "hidden_size": WIDTH, "num_hidden_layers": 16,
                 "num_attention_heads": 16, "num_key_value_heads": 16, "intermediate_size": 8192,
                 "mm_hidden_size": 1024, "mm_vision_select_layer": -2, "rope_theta": 10000.0,
                 "mm_vision_tower": "openai/clip-vit-large-patch14-336", "vocab_size": 50282,
                 "image_aspect_ratio": "anyres", "mm_patch_merge_type": "spatial_unpad",
                 "mm_projector_type": "mlp2x_gelu", "mm_vision_select_feature": "patch",
                 "tie_word_embeddings": True, "hidden_act": "silu", "attention_bias": False,
                 "clip_qkv": None, "rope_scaling": None, "max_position_embeddings": 2048}
    if any(cfg.get(key) != value for key, value in contracts.items()):
        p.error("checkpoint does not match bounded Instella-VL-1B configuration")
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    before, after = prompt_ids(tokenizer, args.prompt)
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    tiles, geometry = image_tiles(image, cfg["image_grid_pinpoints"])
    ys, xs = spatial_rows(image.size, geometry)
    visual_tokens = GRID * GRID + (ys.stop - ys.start) * (xs.stop - xs.start + 1)
    capacity = len(before) + visual_tokens + len(after) + args.max_tokens
    if capacity > cfg["max_position_embeddings"]:
        p.error("visual prefix, prompt and output budget exceed decoder capacity")
    started = perf_counter()
    with safe_open(args.checkpoint / "model.safetensors", framework="np") as weights:
        kernels = Kernels(args.engine, weights, args.threads)
        projected = (kernels.encode_generated(tiles, args.generated_vision_dir)
                     if args.generated_vision_dir else kernels.encode(tiles))
        prefix = bridge(projected, kernels.weight("model.image_newline", (WIDTH,)),
                        image.size, geometry)
        if prefix.shape != (visual_tokens, WIDTH) or not np.isfinite(prefix).all():
            raise ValueError("invalid native prefix extent or values")
        if args.prefix_out:
            np.save(args.prefix_out, prefix)
        encoder_seconds = perf_counter() - started
        print(f"encoder: {len(prefix)} visual tokens in {encoder_seconds:.2f}s", flush=True)
        embed = kernels.weight("model.embed_tokens.weight", (cfg["vocab_size"], WIDTH))
        x = np.concatenate((embed[before], prefix, embed[after]))[None]
        caches = [(np.empty((16, capacity, 128), np.float32),
                   np.empty((16, capacity, 128), np.float32)) for _ in range(16)]
        pos = 0
        generated = []
        stopped = False
        decoder_started = perf_counter()
        terminators = {cfg["eos_token_id"], tokenizer.token_to_id("|||IP_ADDRESS|||")}
        for _ in range(args.max_tokens):
            token = int(kernels.decoder(x, embed, caches, pos).argmax())
            generated.append(token)
            if token in terminators:
                stopped = True
                break
            pos += x.shape[1]
            x = embed[token:token + 1][None]
        decoder_seconds = perf_counter() - decoder_started
    report = {"scope": "bounded CKE image-to-answer with Python host; decoder not generated",
              "vision_execution": "checked_generated_c" if args.generated_vision_dir else "native_kernel_host",
              "decoder_execution": "native_kernel_host",
              "engine_sha256": sha256(args.engine), "image_sha256": sha256(args.image),
              "checkpoint_sha256": sha256(args.checkpoint / "model.safetensors"),
              "tiles": list(tiles.shape), "visual_tokens": len(prefix), "stopped": stopped,
              "tiles_grid_xy": list(geometry), "text_ids_before": before, "text_ids_after": after,
              "tokens": generated, "answer": tokenizer.decode(generated).strip(),
              "encoder_seconds": encoder_seconds, "decoder_seconds": decoder_seconds}
    if args.generated_vision_dir:
        report["vision_build_identities"] = {}
        for name in ("patch", "input_norm", "block", "projector"):
            identity_path = args.generated_vision_dir / name / "build-identity.json"
            identity = json.loads(identity_path.read_text())
            report["vision_build_identities"][name] = {
                key: identity[key] for key in ("library_sha256", "build_identity_sha256",
                                                "embedded_runtime_build_identity")
            }
    if args.expect_answer is not None:
        report["task_pass"] = stopped and report["answer"].casefold() == args.expect_answer.strip().casefold()
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report.get("task_pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
