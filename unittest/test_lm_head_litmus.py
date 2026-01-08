import argparse
import json
import math
import os
import shutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F


def detect_cc() -> str:
    """Detect C compiler (portable default: gcc; opt-in icx via CK_USE_ICX=1)."""
    cc_env = os.environ.get("CC")
    if cc_env and shutil.which(cc_env):
        return cc_env
    if os.environ.get("CK_USE_ICX") == "1":
        if shutil.which("icx"):
            return "icx"
        if shutil.which("icc"):
            return "icc"
    return "gcc"


def openmp_flag(cc: str) -> str:
    cc_lower = cc.lower()
    if "icx" in cc_lower or "icc" in cc_lower:
        return "-qopenmp"
    return "-fopenmp"


def align_up(n, a=16):
    return (n + a - 1) // a * a


def write_f32(path, arr):
    arr.astype(np.float32).tofile(path)


def write_i32(path, arr):
    arr.astype(np.int32).tofile(path)


def read_f32(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def read_scalar_f32(path):
    return float(np.fromfile(path, dtype=np.float32)[0])


def assert_close(name, got, ref, atol=1e-4, rtol=1e-4):
    diff = np.abs(got - ref)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    ok = np.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"{name}: max_abs={max_diff:.3e} mean_abs={mean_diff:.3e}")
    if not ok:
        raise AssertionError(f"{name} mismatch (max_abs={max_diff:.3e})")


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def lerp(a, b, t):
    return a + (b - a) * t


def rgb_to_hex(rgb):
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def lerp_color(c1, c2, t):
    return (
        int(lerp(c1[0], c2[0], t) + 0.5),
        int(lerp(c1[1], c2[1], t) + 0.5),
        int(lerp(c1[2], c2[2], t) + 0.5),
    )


def softmax_rows(logits):
    maxv = np.max(logits, axis=1, keepdims=True)
    expv = np.exp(logits - maxv)
    denom = np.sum(expv, axis=1, keepdims=True)
    return expv / denom


def make_svg_report(path, logits, targets, cfg):
    T, V = logits.shape
    D = int(cfg["hidden_size"])
    aligned_D = align_up(D, 16)

    logits_min = float(np.min(logits))
    logits_max = float(np.max(logits))
    probs = softmax_rows(logits)
    probs_min = float(np.min(probs))
    probs_max = float(np.max(probs))

    rows = np.arange(T)
    p_minus = probs.copy()
    p_minus[rows, targets] -= 1.0
    p_minus_min = float(np.min(p_minus))
    p_minus_max = float(np.max(p_minus))

    probs_target = probs[rows, targets]
    loss_per = -np.log(np.clip(probs_target, 1e-12, 1.0))
    loss_mean = float(np.mean(loss_per))
    loss_min = float(np.min(loss_per))
    loss_max = float(np.max(loss_per))

    config_lines = [
        f"layers: {cfg['num_hidden_layers']}",
        f"embed: {D} (aligned {aligned_D})",
        f"intermediate: {cfg['intermediate_size']}",
        f"heads: {cfg['num_attention_heads']}  kv: {cfg['num_key_value_heads']}",
        f"vocab: {V}",
        f"context: {T}",
    ]
    stats_lines = [
        f"logits min/max: {logits_min:.3f} / {logits_max:.3f}",
        f"softmax min/max: {probs_min:.3e} / {probs_max:.3e}",
        f"p-1hot min/max: {p_minus_min:.3f} / {p_minus_max:.3f}",
        f"loss mean: {loss_mean:.6f}",
        f"loss min/max: {loss_min:.6f} / {loss_max:.6f}",
    ]

    max_dim = max(T, V)
    cell = int(max(2, min(7, 560 / max_dim)))
    heat_w = V * cell
    heat_h = T * cell

    margin = 36
    gap = 36
    header_h = 48
    line_h = 16
    config_block = 44 + len(config_lines) * line_h
    stats_block = 42 + max(0, len(stats_lines) - 1) * line_h
    summary_h = config_block + 6 + stats_block + 28
    width = max(heat_w * 2 + gap + margin * 2, 980)
    summary_w = int((width - margin * 2 - gap) / 2)
    summary_x = margin
    summary_y = margin + header_h
    weight_x = summary_x + summary_w + gap
    weight_y = summary_y
    heat_y = summary_y + summary_h + gap
    loss_y = heat_y + heat_h + 80
    loss_h = 200
    height = loss_y + loss_h + margin

    def softmax_color(v):
        v = clamp(v, 0.0, 1.0)
        v = math.sqrt(v)
        c1 = (10, 20, 30)
        c2 = (24, 110, 122)
        c3 = (242, 188, 60)
        if v < 0.5:
            return rgb_to_hex(lerp_color(c1, c2, v / 0.5))
        return rgb_to_hex(lerp_color(c2, c3, (v - 0.5) / 0.5))

    def delta_color(v):
        v = clamp(v, -1.0, 1.0)
        neg = (176, 53, 79)
        mid = (244, 238, 226)
        pos = (28, 138, 148)
        if v < 0.0:
            return rgb_to_hex(lerp_color(neg, mid, (v + 1.0)))
        return rgb_to_hex(lerp_color(mid, pos, v))

    def append_heatmap(x0, y0, data, title, legend_min, legend_max, color_fn):
        h, w = data.shape
        parts.append(f'<text x="{x0}" y="{y0 - 24}" class="section">{title}</text>')
        legend_w = min(160, w * cell)
        legend_h = 8
        legend_x = x0 + w * cell - legend_w
        legend_y = y0 - 14
        steps = 20
        for i in range(steps):
            t = i / (steps - 1)
            color = color_fn(legend_min + (legend_max - legend_min) * t)
            lx = legend_x + i * (legend_w / steps)
            parts.append(
                f'<rect x="{lx:.2f}" y="{legend_y}" width="{legend_w / steps:.2f}" height="{legend_h}" fill="{color}"/>'
            )
        parts.append(
            f'<text x="{legend_x}" y="{legend_y - 2}" class="mono" text-anchor="start">{legend_min:.2f}</text>'
        )
        parts.append(
            f'<text x="{legend_x + legend_w}" y="{legend_y - 2}" class="mono" text-anchor="end">{legend_max:.2f}</text>'
        )

        for i in range(h):
            row = data[i]
            y = y0 + i * cell
            for j in range(w):
                color = color_fn(float(row[j]))
                x = x0 + j * cell
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}"/>'
                )
        parts.append(f'<rect x="{x0}" y="{y0}" width="{w * cell}" height="{h * cell}" class="heat-border"/>')

        tick_step_x = max(1, w // 5)
        tick_step_y = max(1, h // 5)
        for j in range(0, w, tick_step_x):
            tx = x0 + j * cell
            parts.append(f'<text x="{tx}" y="{y0 + h * cell + 18}" class="mono" text-anchor="middle">{j}</text>')
        if w - 1 not in range(0, w, tick_step_x):
            tx = x0 + (w - 1) * cell
            parts.append(f'<text x="{tx}" y="{y0 + h * cell + 18}" class="mono" text-anchor="middle">{w - 1}</text>')

        for i in range(0, h, tick_step_y):
            ty = y0 + i * cell
            parts.append(f'<text x="{x0 - 6}" y="{ty + 4}" class="mono" text-anchor="end">{i}</text>')
        if h - 1 not in range(0, h, tick_step_y):
            ty = y0 + (h - 1) * cell
            parts.append(f'<text x="{x0 - 6}" y="{ty + 4}" class="mono" text-anchor="end">{h - 1}</text>')

        parts.append(
            f'<text x="{x0 + (w * cell) / 2}" y="{y0 + h * cell + 34}" class="label" text-anchor="middle">vocab index</text>'
        )
        label_x = x0 - 24
        label_y = y0 + (h * cell) / 2
        parts.append(
            f'<text x="{label_x}" y="{label_y}" class="label" text-anchor="middle" transform="rotate(-90 {label_x} {label_y})">token index</text>'
        )

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append(
        "<defs>"
        '<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#F7F3ED"/>'
        '<stop offset="100%" stop-color="#E6DEC8"/>'
        "</linearGradient>"
        '<pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">'
        '<circle cx="3" cy="3" r="1.2" fill="#D5CCBC" opacity="0.5"/>'
        "</pattern>"
        "</defs>"
    )
    parts.append('<rect width="100%" height="100%" fill="url(#bg)"/>')
    parts.append('<rect width="100%" height="100%" fill="url(#dots)" opacity="0.5"/>')
    parts.append(
        "<style>"
        ".title{font-family:'Space Grotesk','IBM Plex Sans','Noto Sans',sans-serif;font-size:24px;font-weight:600;fill:#151A20;}"
        ".subtitle{font-family:'IBM Plex Sans','Noto Sans',sans-serif;font-size:12px;fill:#4B5156;}"
        ".label{font-family:'IBM Plex Sans','Noto Sans',sans-serif;font-size:12px;fill:#1B1F23;}"
        ".mono{font-family:'IBM Plex Mono','SFMono-Regular',monospace;font-size:11px;fill:#1B1F23;}"
        ".panel{fill:#FDF9F3;stroke:#2E2E2E;stroke-width:1;}"
        ".box{fill:#F4EFE6;stroke:#2E2E2E;stroke-width:1;}"
        ".heat-border{fill:none;stroke:#2E2E2E;stroke-width:1;}"
        ".section{font-family:'Space Grotesk','IBM Plex Sans',sans-serif;font-size:14px;font-weight:600;fill:#151A20;}"
        ".grid{stroke:#C7BFB0;stroke-width:1;opacity:0.6;}"
        ".link{stroke:#2E2E2E;stroke-width:2;}"
        "</style>"
    )

    title_x = margin
    title_y = margin
    subtitle_y = margin + 22
    subtitle = (
        f"layers={cfg['num_hidden_layers']} embed={D} (aligned {aligned_D}) "
        f"intermediate={cfg['intermediate_size']} heads={cfg['num_attention_heads']} "
        f"kv_heads={cfg['num_key_value_heads']} vocab={V} ctx={T}"
    )
    parts.append(f'<text x="{title_x}" y="{title_y}" class="title">LM Head + CE Litmus Report</text>')
    parts.append(f'<text x="{title_x}" y="{subtitle_y}" class="subtitle">{subtitle}</text>')

    parts.append(f'<rect x="{summary_x}" y="{summary_y}" width="{summary_w}" height="{summary_h}" class="panel"/>')
    parts.append(f'<text x="{summary_x + 16}" y="{summary_y + 22}" class="section">Config</text>')
    y = summary_y + 44
    for line in config_lines:
        parts.append(f'<text x="{summary_x + 18}" y="{y}" class="mono">{line}</text>')
        y += line_h

    sep_y = y + 6
    parts.append(
        f'<line x1="{summary_x + 12}" y1="{sep_y}" x2="{summary_x + summary_w - 12}" y2="{sep_y}" class="grid"/>'
    )
    parts.append(f'<text x="{summary_x + 16}" y="{sep_y + 20}" class="section">Stats</text>')
    y = sep_y + 42
    for line in stats_lines:
        parts.append(f'<text x="{summary_x + 18}" y="{y}" class="mono">{line}</text>')
        y += line_h

    parts.append(f'<rect x="{weight_x}" y="{weight_y}" width="{summary_w}" height="{summary_h}" class="panel"/>')
    parts.append(f'<text x="{weight_x + 16}" y="{weight_y + 22}" class="section">Weight tying</text>')
    box_w = summary_w - 80
    box_h = 34
    box_x = weight_x + (summary_w - box_w) / 2
    box1_y = weight_y + 50
    box2_y = box1_y + box_h + 40
    parts.append(f'<rect x="{box_x}" y="{box1_y}" width="{box_w}" height="{box_h}" class="box"/>')
    parts.append(f'<rect x="{box_x}" y="{box2_y}" width="{box_w}" height="{box_h}" class="box"/>')
    parts.append(
        f'<text x="{box_x + box_w / 2}" y="{box1_y + 14}" class="label" text-anchor="middle">token_embedding W</text>'
    )
    parts.append(
        f'<text x="{box_x + box_w / 2}" y="{box1_y + 28}" class="mono" text-anchor="middle">({V} x {D})</text>'
    )
    parts.append(
        f'<text x="{box_x + box_w / 2}" y="{box2_y + 14}" class="label" text-anchor="middle">lm_head W</text>'
    )
    parts.append(
        f'<text x="{box_x + box_w / 2}" y="{box2_y + 28}" class="mono" text-anchor="middle">({V} x {D})</text>'
    )
    line_x = box_x + box_w / 2
    parts.append(f'<line x1="{line_x}" y1="{box1_y + box_h}" x2="{line_x}" y2="{box2_y}" class="link"/>')
    parts.append(
        f'<text x="{line_x}" y="{box1_y + box_h + 18}" class="mono" text-anchor="middle">shared weights</text>'
    )
    parts.append(
        f'<text x="{weight_x + 18}" y="{weight_y + summary_h - 28}" class="mono">logits[t] = hidden[t] @ W^T</text>'
    )

    append_heatmap(
        summary_x,
        heat_y,
        probs,
        "Softmax p[t, v]",
        0.0,
        1.0,
        softmax_color,
    )
    append_heatmap(
        summary_x + heat_w + gap,
        heat_y,
        p_minus,
        "p[t, v] - one_hot[t]",
        -1.0,
        1.0,
        delta_color,
    )

    loss_x = margin
    loss_w = width - margin * 2
    parts.append(f'<rect x="{loss_x}" y="{loss_y}" width="{loss_w}" height="{loss_h}" class="panel"/>')
    parts.append(f'<text x="{loss_x + 16}" y="{loss_y + 22}" class="section">Loss per token</text>')
    parts.append(
        f'<text x="{loss_x + 16}" y="{loss_y + 40}" class="mono">loss[t] = -log(p[t, y_t])  mean={loss_mean:.6f}</text>'
    )

    chart_pad_x = 50
    chart_pad_y = 56
    chart_x = loss_x + chart_pad_x
    chart_y = loss_y + chart_pad_y
    chart_w = loss_w - chart_pad_x - 28
    chart_h = loss_h - chart_pad_y - 26

    grid_lines = 4
    for i in range(grid_lines + 1):
        gy = chart_y + i * (chart_h / grid_lines)
        parts.append(f'<line x1="{chart_x}" y1="{gy}" x2="{chart_x + chart_w}" y2="{gy}" class="grid"/>')

    loss_range = loss_max - loss_min
    if loss_range < 1e-9:
        loss_range = 1.0

    path_cmds = []
    for i in range(T):
        px = chart_x + (i / max(1, T - 1)) * chart_w
        py = chart_y + (1.0 - (loss_per[i] - loss_min) / loss_range) * chart_h
        cmd = "M" if i == 0 else "L"
        path_cmds.append(f"{cmd} {px:.2f} {py:.2f}")
    parts.append(f'<path d="{" ".join(path_cmds)}" fill="none" stroke="#1C7F86" stroke-width="2"/>')

    mean_y = chart_y + (1.0 - (loss_mean - loss_min) / loss_range) * chart_h
    parts.append(f'<line x1="{chart_x}" y1="{mean_y}" x2="{chart_x + chart_w}" y2="{mean_y}" stroke="#B85A3F" stroke-width="1.5"/>')
    parts.append(
        f'<text x="{chart_x + chart_w}" y="{mean_y - 4}" class="mono" text-anchor="end">mean</text>'
    )

    parts.append(
        f'<text x="{chart_x + chart_w / 2}" y="{chart_y + chart_h + 24}" class="label" text-anchor="middle">token index</text>'
    )
    parts.append(
        f'<text x="{chart_x - 32}" y="{chart_y + chart_h / 2}" class="label" text-anchor="middle" transform="rotate(-90 {chart_x - 32} {chart_y + chart_h / 2})">loss</text>'
    )

    parts.append(f'<text x="{chart_x}" y="{chart_y + chart_h + 10}" class="mono">0</text>')
    parts.append(
        f'<text x="{chart_x + chart_w}" y="{chart_y + chart_h + 10}" class="mono" text-anchor="end">{T - 1}</text>'
    )
    parts.append(
        f'<text x="{chart_x - 6}" y="{chart_y + 4}" class="mono" text-anchor="end">{loss_max:.3f}</text>'
    )
    parts.append(
        f'<text x="{chart_x - 6}" y="{chart_y + chart_h}" class="mono" text-anchor="end">{loss_min:.3f}</text>'
    )

    parts.append("</svg>")

    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(parts))


def parse_args():
    parser = argparse.ArgumentParser(description="LM head + CE backward parity litmus")
    parser.add_argument("--config", default=None, help="Path to HF-style config.json")
    parser.add_argument("--layers", type=int, default=None, help="Override num_hidden_layers")
    parser.add_argument("--embed", type=int, default=None, help="Override hidden_size")
    parser.add_argument("--intermediate", type=int, default=None, help="Override intermediate_size")
    parser.add_argument("--heads", type=int, default=None, help="Override num_attention_heads")
    parser.add_argument("--kv-heads", type=int, default=None, help="Override num_key_value_heads")
    parser.add_argument("--vocab", type=int, default=None, help="Override vocab_size")
    parser.add_argument("--ctx", type=int, default=None, help="Override max_position_embeddings")
    parser.add_argument("--build-dir", default=None, help="Override build directory")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--skip-emit", action="store_true", help="Skip make emit step")
    parser.add_argument("--skip-compile", action="store_true", help="Skip gcc compile step")
    parser.add_argument("--svg", default=None, help="Write SVG report to this path")
    return parser.parse_args()


def build_config(args):
    cfg = {
        "num_hidden_layers": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 50,
        "max_position_embeddings": 8,
    }
    if args.config:
        with open(args.config, "r", encoding="ascii") as f:
            on_disk = json.load(f)
        cfg.update(on_disk)

    if args.layers is not None:
        cfg["num_hidden_layers"] = args.layers
    if args.embed is not None:
        cfg["hidden_size"] = args.embed
    if args.intermediate is not None:
        cfg["intermediate_size"] = args.intermediate
    if args.heads is not None:
        cfg["num_attention_heads"] = args.heads
    if args.kv_heads is not None:
        cfg["num_key_value_heads"] = args.kv_heads
    if args.vocab is not None:
        cfg["vocab_size"] = args.vocab
    if args.ctx is not None:
        cfg["max_position_embeddings"] = args.ctx

    if "num_key_value_heads" not in cfg:
        cfg["num_key_value_heads"] = cfg["num_attention_heads"]

    return cfg


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    build_dir = args.build_dir or os.path.join(root, "build")
    os.makedirs(build_dir, exist_ok=True)

    cfg = build_config(args)
    cfg_path = os.path.join(build_dir, "litmus.config.json")
    with open(cfg_path, "w", encoding="ascii") as f:
        json.dump(cfg, f)

    gen_c = os.path.join(build_dir, "litmus_generated.c")
    gen_bin = os.path.join(build_dir, "litmus_generated")
    if not args.skip_emit:
        subprocess.run(
            ["make", "emit", f"CONFIG={cfg_path}", f"OUT={gen_c}"],
            cwd=root,
            check=True,
        )
    if not args.skip_compile:
        manifest = gen_c + ".kernels"
        cc = os.environ.get("CC", detect_cc())
        cmd = [cc, "-O2", "-march=native", "-Iinclude", gen_c, "-lm", "-o", gen_bin]
        if os.path.exists(os.path.join(root, manifest)):
            with open(os.path.join(root, manifest), "r", encoding="utf-8") as f:
                extra_sources = [line.strip() for line in f if line.strip()]
            cmd.extend(extra_sources)
            cmd.insert(1, openmp_flag(cc))
        subprocess.run(cmd, cwd=root, check=True)

    T = int(cfg["max_position_embeddings"])
    D = int(cfg["hidden_size"])
    V = int(cfg["vocab_size"])
    aligned_D = align_up(D, 16)

    torch.manual_seed(args.seed)
    hidden = torch.randn(T, D, dtype=torch.float32)
    weights = torch.randn(V, D, dtype=torch.float32)
    targets = torch.randint(0, V, (T,), dtype=torch.int64)
    targets_np = targets.numpy()

    hidden_padded = np.zeros((T, aligned_D), dtype=np.float32)
    hidden_padded[:, :D] = hidden.numpy()
    weights_padded = np.zeros((V, aligned_D), dtype=np.float32)
    weights_padded[:, :D] = weights.numpy()

    hidden_path = os.path.join(build_dir, "litmus_hidden.bin")
    weights_path = os.path.join(build_dir, "litmus_weights.bin")
    targets_path = os.path.join(build_dir, "litmus_targets.bin")
    out_logits = os.path.join(build_dir, "litmus_logits.bin")
    out_dlogits = os.path.join(build_dir, "litmus_dlogits.bin")
    out_dhidden = os.path.join(build_dir, "litmus_dhidden.bin")
    out_dweights = os.path.join(build_dir, "litmus_dweights.bin")
    out_loss = os.path.join(build_dir, "litmus_loss.bin")

    write_f32(hidden_path, hidden_padded)
    write_f32(weights_path, weights_padded)
    write_i32(targets_path, targets_np)

    subprocess.run(
        [
            gen_bin,
            "--litmus",
            "--hidden",
            hidden_path,
            "--weights",
            weights_path,
            "--targets",
            targets_path,
            "--out-logits",
            out_logits,
            "--out-dlogits",
            out_dlogits,
            "--out-dhidden",
            out_dhidden,
            "--out-dweights",
            out_dweights,
            "--out-loss",
            out_loss,
            "--no-forward",
        ],
        cwd=root,
        check=True,
    )

    logits_c = read_f32(out_logits, (T, V))
    d_logits_c = read_f32(out_dlogits, (T, V))
    d_hidden_c = read_f32(out_dhidden, (T, aligned_D))[:, :D]
    d_weights_c = read_f32(out_dweights, (V, aligned_D))[:, :D]
    loss_c = read_scalar_f32(out_loss)

    hidden_t = hidden.clone().detach().requires_grad_(True)
    weights_t = weights.clone().detach().requires_grad_(True)
    logits_t = hidden_t @ weights_t.t()
    logits_t.retain_grad()
    loss_t = F.cross_entropy(logits_t, targets, reduction="mean")
    loss_t.backward()

    assert_close("loss", np.array(loss_c, dtype=np.float32), np.array(loss_t.item(), dtype=np.float32), atol=1e-3, rtol=1e-3)
    assert_close("logits", logits_c, logits_t.detach().numpy())
    assert_close("d_logits", d_logits_c, logits_t.grad.detach().numpy())
    assert_close("d_hidden", d_hidden_c, hidden_t.grad.detach().numpy())
    assert_close("d_weights", d_weights_c, weights_t.grad.detach().numpy())

    pad_hidden = read_f32(out_dhidden, (T, aligned_D))[:, D:]
    pad_weights = read_f32(out_dweights, (V, aligned_D))[:, D:]
    if pad_hidden.size and np.max(np.abs(pad_hidden)) > 1e-6:
        raise AssertionError("d_hidden padding not zero")
    if pad_weights.size and np.max(np.abs(pad_weights)) > 1e-6:
        raise AssertionError("d_weights padding not zero")

    if args.svg:
        make_svg_report(args.svg, logits_c, targets_np, cfg)
        print(f"SVG report written to {args.svg}")

    print("LM head + CE litmus PASSED")


if __name__ == "__main__":
    main()
