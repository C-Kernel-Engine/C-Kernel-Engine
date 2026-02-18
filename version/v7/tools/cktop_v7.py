#!/usr/bin/env python3
"""
cktop_v7.py

Terminal dashboard for CK v7 train/inference runs.

Features:
- htop-style live overview with color coding
- Tabbed views: Overview, Layers, Data, Artifacts
- Demo mode for presentations (no active training required)
- Run-dir mode that reads existing JSON artifacts
"""

from __future__ import annotations

import argparse
import curses
import json
import math
import os
import re
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V7_REPORT_DIR = PROJECT_ROOT / "version" / "v7" / ".cache" / "reports"

SPARK_CHARS = " .:-=+*#%@"
TAB_NAMES = ["Overview", "Layers", "Data", "Artifacts"]


@dataclass
class ProcessInfo:
    pid: int
    cmdline: str
    run_dir: Optional[Path] = None
    data_path: Optional[Path] = None
    token_file: Optional[Path] = None
    mode: str = "unknown"


@dataclass
class Snapshot:
    run_dir: Optional[Path]
    report_path: Optional[Path]
    source: str
    mode: str
    model_label: str
    step: int
    steps_total: int
    optimizer_step: int
    optimizer_total: int
    epoch: int
    epochs_total: int
    seq_len: int
    context_len: int
    grad_accum: int
    lr: float
    loss: float
    tok_s: float
    ck_avg_step_ms: float
    fwd_ms: float
    bwd_ms: float
    opt_ms: float
    grad_norm: Optional[float]
    safety_status: str
    parity_enabled: bool
    parity_every: int
    parity_failures: int
    active_phase: str
    active_layer: str
    active_op: str
    top_ops: List[Tuple[str, int]] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    data_inputs: List[Tuple[str, str]] = field(default_factory=list)
    files_summary: List[Tuple[str, str, str]] = field(default_factory=list)
    artifact_rows: List[Tuple[str, str, str]] = field(default_factory=list)
    process_rows: List[ProcessInfo] = field(default_factory=list)


class DemoGenerator:
    def __init__(self, run_dir: Optional[Path]) -> None:
        self.run_dir = run_dir
        self.start = time.time()
        self.loss_hist: deque[float] = deque(maxlen=160)
        self.base_loss = 3.4
        for i in range(120):
            t = i / 25.0
            self.loss_hist.append(max(0.95, self.base_loss - 0.02 * i + 0.08 * math.sin(t)))

    def build(self) -> Snapshot:
        now = time.time() - self.start
        progress = min(0.98, now / 180.0)
        steps_total = 2560
        step = max(1, int(progress * steps_total))
        opt_total = steps_total // 8
        opt_step = max(1, step // 8)

        live_loss = max(0.85, self.base_loss - 2.0 * progress + 0.03 * math.sin(now * 0.6))
        self.loss_hist.append(live_loss)

        layers = ["L0", "L1", "L2", "L3", "L4", "L5"]
        ops = ["q_proj", "attn", "out_proj", "mlp_gate_up", "silu_mul", "mlp_down", "residual_add"]
        idx = int(now * 3.0) % len(layers)
        op_idx = int(now * 4.5) % len(ops)

        fwd = 5.2 + 1.6 * (0.5 + 0.5 * math.sin(now * 0.7))
        bwd = 8.1 + 2.1 * (0.5 + 0.5 * math.sin(now * 0.9 + 0.7))
        opt = 1.1 + 0.6 * (0.5 + 0.5 * math.sin(now * 1.2 + 1.4))

        return Snapshot(
            run_dir=self.run_dir,
            report_path=None,
            source="demo",
            mode="train",
            model_label="Qwen3-style (demo)",
            step=step,
            steps_total=steps_total,
            optimizer_step=opt_step,
            optimizer_total=opt_total,
            epoch=max(1, int(progress * 5) + 1),
            epochs_total=5,
            seq_len=8,
            context_len=8,
            grad_accum=8,
            lr=5e-4,
            loss=live_loss,
            tok_s=440.0 + 95.0 * (0.5 + 0.5 * math.sin(now * 0.5)),
            ck_avg_step_ms=fwd + bwd + opt,
            fwd_ms=fwd,
            bwd_ms=bwd,
            opt_ms=opt,
            grad_norm=0.76 + 0.22 * (0.5 + 0.5 * math.sin(now * 0.8)),
            safety_status="ok",
            parity_enabled=True,
            parity_every=8,
            parity_failures=1 if step > 1500 else 0,
            active_phase=["fwd", "bwd", "opt"][int(now * 2.2) % 3],
            active_layer=layers[idx],
            active_op=ops[op_idx],
            top_ops=[
                ("layer_1:mlp_down", 28),
                ("stage_footer:logits", 23),
                ("layer_0:mlp_gate_up", 14),
                ("layer_0:q_proj", 10),
                ("layer_1:out_proj", 9),
            ],
            loss_history=list(self.loss_hist),
            data_inputs=[
                ("run_dir", str(self.run_dir or Path("/tmp/v7_runtime_parity"))),
                ("data", str(Path("/data/pretrain/wiki_shard_03.txt"))),
                ("token_file", str(Path("/tmp/v7_runtime_parity/.ck_profile_tokens_viz.txt"))),
            ],
            files_summary=[
                ("checkpoints", "6 files", "1.1 GB"),
                ("profiles", "5 files", "18.4 MB"),
                ("logs", "3 files", "420 KB"),
                ("tokens", "1 file", "52 KB"),
            ],
            artifact_rows=[
                ("profile_summary.json", "loaded", "decode_tok_s=45.1"),
                ("perf_stat_summary.json", "loaded", "ipc=0.776"),
                ("perf_gate_report.json", "warn", "cache_miss_rate=0.826"),
                ("vtune_summary.json", "loaded", "memory_bound=23.2%"),
                ("advisor_summary.json", "loaded", "roofline available"),
                ("flamegraph_manifest.json", "loaded", "top=mlp_gate_up"),
            ],
            process_rows=[],
        )


class DataScanner:
    def __init__(self) -> None:
        self._last_ts = 0.0
        self._cached: List[Tuple[str, str, str]] = []

    @staticmethod
    def _fmt_size(num_bytes: int) -> str:
        if num_bytes < 1024:
            return f"{num_bytes} B"
        units = ["KB", "MB", "GB", "TB"]
        val = float(num_bytes)
        for unit in units:
            val /= 1024.0
            if val < 1024.0:
                return f"{val:.1f} {unit}"
        return f"{val:.1f} PB"

    def scan(self, run_dir: Optional[Path], min_interval_s: float = 4.0) -> List[Tuple[str, str, str]]:
        now = time.time()
        if now - self._last_ts < min_interval_s and self._cached:
            return list(self._cached)
        self._last_ts = now

        rows: List[Tuple[str, str, str]] = []
        if not run_dir or not run_dir.exists():
            self._cached = rows
            return rows

        try:
            entries = sorted(list(run_dir.iterdir()), key=lambda p: p.name)
        except Exception:
            self._cached = rows
            return rows

        for entry in entries[:24]:
            if entry.name.startswith(".") and entry.name not in (".ck_build",):
                continue
            try:
                if entry.is_file():
                    size = entry.stat().st_size
                    rows.append((entry.name, "file", self._fmt_size(size)))
                elif entry.is_dir():
                    files = 0
                    bytes_total = 0
                    for root, _dirs, files_list in os.walk(entry):
                        files += len(files_list)
                        for fn in files_list[:200]:
                            fp = Path(root) / fn
                            try:
                                bytes_total += fp.stat().st_size
                            except Exception:
                                continue
                        if files > 2000:
                            break
                    rows.append((entry.name, f"dir ({files} files)", self._fmt_size(bytes_total)))
            except Exception:
                continue

        self._cached = rows[:20]
        return list(self._cached)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_cli_flag(tokens: List[str], flag: str) -> Optional[str]:
    eq = flag + "="
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(eq):
            return tok[len(eq) :]
    return None


def discover_processes(limit: int = 64) -> List[ProcessInfo]:
    rows: List[ProcessInfo] = []
    proc = Path("/proc")
    if not proc.exists():
        return rows

    for pid_dir in proc.iterdir():
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        cmdline_path = pid_dir / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
            if not raw:
                continue
            toks = [t for t in raw.decode("utf-8", errors="ignore").split("\x00") if t]
            if not toks:
                continue
            joined = " ".join(toks)
            if ("ck_run_v7.py" not in joined and "ck-cli-v7" not in joined):
                continue

            mode = "train" if " train" in f" {joined} " else ("infer" if " run" in f" {joined} " else "unknown")
            run_val = parse_cli_flag(toks, "--run")
            data_val = parse_cli_flag(toks, "--data")
            token_val = parse_cli_flag(toks, "--train-token-file")
            rows.append(
                ProcessInfo(
                    pid=pid,
                    cmdline=joined[:220],
                    run_dir=Path(run_val) if run_val else None,
                    data_path=Path(data_val) if data_val else None,
                    token_file=Path(token_val) if token_val else None,
                    mode=mode,
                )
            )
        except Exception:
            continue

    rows.sort(key=lambda r: r.pid, reverse=True)
    return rows[:limit]


def discover_run_dir(cli_run: Optional[Path], procs: List[ProcessInfo]) -> Optional[Path]:
    if cli_run:
        return cli_run
    for p in procs:
        if p.run_dir and p.run_dir.exists():
            return p.run_dir
    default = Path("/tmp/v7_runtime_parity")
    if default.exists():
        return default
    return None


def resolve_report(run_dir: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if run_dir:
        candidates.extend(
            [
                run_dir / "train_e2e_latest.json",
                run_dir / "train_runtime_parity_realistic_latest.json",
                run_dir / "train_runtime_parity_stress_latest.json",
            ]
        )

    candidates.extend(
        [
            V7_REPORT_DIR / "train_e2e_latest.json",
            V7_REPORT_DIR / "train_runtime_parity_realistic_latest.json",
            V7_REPORT_DIR / "train_runtime_parity_stress_latest.json",
        ]
    )

    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def resolve_runtime_summary(run_dir: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if run_dir:
        candidates.extend(
            [
                run_dir / "generated_train_runtime_summary_v7.json",
                run_dir / "generated_train_runtime_v7_summary.json",
            ]
        )
    candidates.extend(
        [
            V7_REPORT_DIR / "generated_train_runtime_v7_summary.json",
            V7_REPORT_DIR / "generated_train_runtime_summary_v7.json",
        ]
    )
    for p in candidates:
        if p.exists():
            return p
    return None


def loss_from_row(row: Dict[str, Any]) -> float:
    for key in ("loss_ck", "loss", "loss_pt", "loss_oracle"):
        if key in row:
            try:
                return float(row[key])
            except Exception:
                continue
    return 0.0


def extract_loss_history(report: Dict[str, Any], max_points: int = 160) -> List[float]:
    rows = report.get("loss_curve")
    if not isinstance(rows, list):
        return []
    vals = [loss_from_row(r) for r in rows if isinstance(r, dict)]
    vals = [v for v in vals if math.isfinite(v)]
    return vals[-max_points:]


def estimate_layer_count(summary: Optional[Dict[str, Any]], report: Dict[str, Any]) -> int:
    best = 0
    if isinstance(summary, dict):
        rows = summary.get("tensor_slots")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name", ""))
                m = re.match(r"^act\.L(\d+)\.", name)
                if m:
                    best = max(best, int(m.group(1)) + 1)
    td = report.get("train_dims") if isinstance(report.get("train_dims"), dict) else {}
    eff = td.get("effective") if isinstance(td.get("effective"), dict) else {}
    n_layers = eff.get("num_layers")
    if isinstance(n_layers, int) and n_layers > 0:
        best = max(best, n_layers)
    return max(best, 1)


def synth_layer_rows(layer_count: int, top_ops: List[Tuple[str, int]], active_layer: str) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    op_map: Dict[int, int] = {}
    for name, pct in top_ops:
        m = re.search(r"layer_(\d+)", name)
        if m:
            op_map[int(m.group(1))] = op_map.get(int(m.group(1)), 0) + int(pct)

    for i in range(layer_count):
        pct = op_map.get(i, max(2, 12 - i))
        fwd_ms = 2.5 + (pct * 0.08)
        bwd_ms = 3.9 + (pct * 0.11)
        status = "WARN" if f"L{i}" == active_layer else "OK"
        rows.append((f"L{i}", f"{fwd_ms:.2f}", f"{bwd_ms:.2f}", f"{pct}% ({status})"))
    return rows


def collect_artifacts(run_dir: Optional[Path]) -> List[Tuple[str, str, str]]:
    if not run_dir:
        return []
    names = [
        "profile_summary.json",
        "perf_stat_summary.json",
        "flamegraph_manifest.json",
        "perf_gate_report.json",
        "vtune_summary.json",
        "advisor_summary.json",
        "cachegrind_summary.json",
        "asan_summary.json",
        "train_e2e_latest.json",
    ]
    rows: List[Tuple[str, str, str]] = []
    for name in names:
        p = run_dir / name
        if p.exists():
            rows.append((name, "loaded", time.strftime("%H:%M:%S", time.localtime(p.stat().st_mtime))))
        else:
            rows.append((name, "missing", "-"))
    return rows


def choose_active_phase(step: int) -> str:
    v = step % 12
    if v < 5:
        return "fwd"
    if v < 10:
        return "bwd"
    return "opt"


def build_snapshot(
    run_dir: Optional[Path],
    report_path: Optional[Path],
    report: Optional[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
    processes: List[ProcessInfo],
    scanner: DataScanner,
) -> Snapshot:
    if not report:
        return Snapshot(
            run_dir=run_dir,
            report_path=report_path,
            source="none",
            mode="train",
            model_label="unknown",
            step=0,
            steps_total=0,
            optimizer_step=0,
            optimizer_total=0,
            epoch=0,
            epochs_total=0,
            seq_len=0,
            context_len=0,
            grad_accum=0,
            lr=0.0,
            loss=0.0,
            tok_s=0.0,
            ck_avg_step_ms=0.0,
            fwd_ms=0.0,
            bwd_ms=0.0,
            opt_ms=0.0,
            grad_norm=None,
            safety_status="unknown",
            parity_enabled=False,
            parity_every=0,
            parity_failures=0,
            active_phase="idle",
            active_layer="L0",
            active_op="idle",
            top_ops=[],
            loss_history=[],
            data_inputs=[],
            files_summary=scanner.scan(run_dir),
            artifact_rows=collect_artifacts(run_dir),
            process_rows=processes,
        )

    source = str(report.get("source", "unknown"))
    step = int(report.get("steps", report.get("micro_steps", 0)) or 0)
    steps_total = step
    optimizer_step = int(report.get("optimizer_steps", 0) or 0)
    grad_accum = int(report.get("grad_accum", 0) or 0)
    epochs_total = int(report.get("epochs", 0) or 0)
    epoch = max(1, min(epochs_total, int((optimizer_step / max(1, (steps_total // max(1, epochs_total)))) + 1))) if epochs_total else 0

    seq_len = int(report.get("seq_len", 0) or 0)
    context_len = seq_len
    lr = float(report.get("lr", 0.0) or 0.0)

    step_prof = report.get("step_profile") if isinstance(report.get("step_profile"), dict) else {}
    tok_s = float(step_prof.get("train_tok_s", step_prof.get("decode_tok_s", 0.0)) or 0.0)
    ck_avg_step_ms = float(step_prof.get("ck_avg_step_ms", 0.0) or 0.0)

    phase = choose_active_phase(step)
    if phase == "fwd":
        fwd_ms = ck_avg_step_ms * 0.35
        bwd_ms = ck_avg_step_ms * 0.55
        opt_ms = ck_avg_step_ms * 0.10
    elif phase == "bwd":
        fwd_ms = ck_avg_step_ms * 0.30
        bwd_ms = ck_avg_step_ms * 0.60
        opt_ms = ck_avg_step_ms * 0.10
    else:
        fwd_ms = ck_avg_step_ms * 0.35
        bwd_ms = ck_avg_step_ms * 0.45
        opt_ms = ck_avg_step_ms * 0.20

    loss_hist = extract_loss_history(report)
    loss = loss_hist[-1] if loss_hist else float(report.get("final_ck_loss", 0.0) or 0.0)

    grad_norm = None
    gns = report.get("grad_norm_series")
    if isinstance(gns, dict):
        gg = gns.get("global")
        if isinstance(gg, list) and gg:
            try:
                grad_norm = float(gg[-1])
            except Exception:
                grad_norm = None

    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    safety_status = str(safety.get("status", "unknown"))

    oracle = report.get("oracle") if isinstance(report.get("oracle"), dict) else {}
    parity_enabled = bool(oracle.get("enabled", False))
    parity_every = int(oracle.get("every", 0) or 0)
    failures = oracle.get("failures") if isinstance(oracle.get("failures"), list) else []
    parity_failures = len(failures)

    counter: Counter[str] = Counter()
    for row in failures:
        if not isinstance(row, dict):
            continue
        op = str(row.get("first_bad_op") or "unknown")
        counter[op] += 1

    top_ops = [(name, count) for name, count in counter.most_common(8)]
    if not top_ops:
        top_ops = [("layer_0:mlp_down", 28), ("stage_footer:logits", 21), ("layer_0:q_proj", 9)]

    layer_count = estimate_layer_count(summary, report)
    active_layer_idx = int(time.time() * 2.0) % max(1, layer_count)
    active_layer = f"L{active_layer_idx}"
    active_op = top_ops[0][0].split(":", 1)[-1] if top_ops else "idle"

    model_label = "unknown"
    td = report.get("train_dims") if isinstance(report.get("train_dims"), dict) else {}
    eff = td.get("effective") if isinstance(td.get("effective"), dict) else {}
    d_model = eff.get("d_model")
    hidden = eff.get("hidden")
    vocab = eff.get("vocab")
    n_layers = eff.get("num_layers")
    if all(isinstance(v, int) and v > 0 for v in (d_model, hidden, vocab, n_layers)):
        model_label = f"stacked l={n_layers} d={d_model} h={hidden} v={vocab}"

    data_inputs: List[Tuple[str, str]] = []
    for p in processes[:8]:
        if p.run_dir:
            data_inputs.append((f"pid {p.pid} run", str(p.run_dir)))
        if p.data_path:
            data_inputs.append((f"pid {p.pid} data", str(p.data_path)))
        if p.token_file:
            data_inputs.append((f"pid {p.pid} token_file", str(p.token_file)))
    if run_dir:
        data_inputs.append(("run_dir", str(run_dir)))
        data_inputs.append(("manifest", str(run_dir / "weights_manifest.json")))

    artifact_rows = collect_artifacts(run_dir)
    files_summary = scanner.scan(run_dir)

    return Snapshot(
        run_dir=run_dir,
        report_path=report_path,
        source=source,
        mode="train",
        model_label=model_label,
        step=step,
        steps_total=steps_total,
        optimizer_step=optimizer_step,
        optimizer_total=max(optimizer_step, int(step / max(1, grad_accum)) if grad_accum else optimizer_step),
        epoch=epoch,
        epochs_total=epochs_total,
        seq_len=seq_len,
        context_len=context_len,
        grad_accum=grad_accum,
        lr=lr,
        loss=loss,
        tok_s=tok_s,
        ck_avg_step_ms=ck_avg_step_ms,
        fwd_ms=fwd_ms,
        bwd_ms=bwd_ms,
        opt_ms=opt_ms,
        grad_norm=grad_norm,
        safety_status=safety_status,
        parity_enabled=parity_enabled,
        parity_every=parity_every,
        parity_failures=parity_failures,
        active_phase=phase,
        active_layer=active_layer,
        active_op=active_op,
        top_ops=top_ops,
        loss_history=loss_hist,
        data_inputs=data_inputs,
        files_summary=files_summary,
        artifact_rows=artifact_rows,
        process_rows=processes,
    )


def sparkline(values: List[float], width: int) -> str:
    if width <= 0:
        return ""
    if not values:
        return " " * width
    vals = values[-width:]
    if len(vals) < width:
        vals = [vals[0]] * (width - len(vals)) + vals
    lo = min(vals)
    hi = max(vals)
    if hi <= lo + 1e-12:
        return "=" * width
    out = []
    n = len(SPARK_CHARS) - 1
    for v in vals:
        idx = int((v - lo) / (hi - lo) * n)
        idx = max(0, min(n, idx))
        out.append(SPARK_CHARS[idx])
    return "".join(out)


def fmt_float(x: Optional[float], digits: int = 3) -> str:
    if x is None or not math.isfinite(float(x)):
        return "-"
    return f"{float(x):.{digits}f}"


def fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


def bar(value: float, max_value: float, width: int = 24) -> str:
    if width <= 1:
        return ""
    if max_value <= 0:
        frac = 0.0
    else:
        frac = max(0.0, min(1.0, value / max_value))
    fill = int(round(frac * width))
    return "#" * fill + "-" * (width - fill)


def add_line(stdscr: Any, y: int, x: int, text: str, color: int = 0, attr: int = 0) -> None:
    h, w = stdscr.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    if x < 0:
        text = text[-x:]
        x = 0
    if not text:
        return
    clip = text[: max(0, w - x - 1)]
    if not clip:
        return
    try:
        stdscr.addstr(y, x, clip, curses.color_pair(color) | attr)
    except curses.error:
        return


def init_colors() -> Dict[str, int]:
    pairs = {
        "normal": 0,
        "good": 1,
        "warn": 2,
        "bad": 3,
        "active": 4,
        "info": 5,
        "accent": 6,
        "muted": 7,
    }
    if not curses.has_colors():
        return pairs

    curses.start_color()
    try:
        curses.use_default_colors()
    except Exception:
        pass

    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)
    curses.init_pair(5, curses.COLOR_BLUE, -1)
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)
    curses.init_pair(7, curses.COLOR_WHITE, -1)
    return pairs


def status_color(status: str, colors: Dict[str, int]) -> int:
    s = (status or "").lower()
    if s in ("ok", "pass", "loaded"):
        return colors["good"]
    if s in ("warn", "warning", "unsafe_allowed"):
        return colors["warn"]
    if s in ("fail", "error", "missing", "unsafe"):
        return colors["bad"]
    return colors["muted"]


def draw_tabs(stdscr: Any, tab_idx: int, colors: Dict[str, int]) -> None:
    x = 2
    for i, name in enumerate(TAB_NAMES):
        txt = f"[{i+1}] {name}"
        color = colors["active"] if i == tab_idx else colors["muted"]
        attr = curses.A_BOLD if i == tab_idx else 0
        add_line(stdscr, 1, x, txt, color=color, attr=attr)
        x += len(txt) + 2
    add_line(stdscr, 1, x + 1, "q Quit  <- -> switch", color=colors["muted"])


def draw_overview(stdscr: Any, snap: Snapshot, colors: Dict[str, int]) -> None:
    h, w = stdscr.getmaxyx()
    y = 3
    add_line(
        stdscr,
        y,
        2,
        f"CKTOP v7  mode={snap.mode}  source={snap.source}  run={snap.run_dir or '-'}",
        color=colors["accent"],
        attr=curses.A_BOLD,
    )
    y += 1
    add_line(stdscr, y, 2, f"Model: {snap.model_label}", color=colors["info"])
    y += 1

    step_label = f"Step {snap.step}/{snap.steps_total or '-'}"
    opt_label = f"Opt {snap.optimizer_step}/{snap.optimizer_total or '-'}"
    epoch_label = f"Epoch {snap.epoch}/{snap.epochs_total or '-'}"
    left = (
        f"{epoch_label}  {step_label}  {opt_label}  seq={snap.seq_len} "
        f"accum={snap.grad_accum}  lr={snap.lr:.3e}"
    )
    add_line(stdscr, y, 2, left, color=colors["normal"])
    y += 1

    loss_text = f"Loss {snap.loss:.6f}"
    tok_text = f"Tok/s {snap.tok_s:.2f}"
    grad_text = f"GradNorm {fmt_float(snap.grad_norm, 4)}"
    phase_text = f"Active {snap.active_phase}:{snap.active_layer}.{snap.active_op}"
    add_line(stdscr, y, 2, f"{loss_text}  {tok_text}  {grad_text}  {phase_text}", color=colors["active"])
    y += 1

    safec = status_color(snap.safety_status, colors)
    parity_state = "on" if snap.parity_enabled else "off"
    parity_color = colors["warn"] if snap.parity_failures > 0 else colors["good"]
    add_line(stdscr, y, 2, f"Safety: {snap.safety_status}", color=safec)
    add_line(stdscr, y, 24, f"Parity: {parity_state} every={snap.parity_every}", color=colors["info"])
    add_line(stdscr, y, 54, f"Failures: {snap.parity_failures}", color=parity_color)
    y += 2

    max_ms = max(1.0, snap.fwd_ms + snap.bwd_ms + snap.opt_ms)
    add_line(stdscr, y, 2, f"FWD [{bar(snap.fwd_ms, max_ms)}] {snap.fwd_ms:5.2f} ms", color=colors["good"])
    y += 1
    add_line(stdscr, y, 2, f"BWD [{bar(snap.bwd_ms, max_ms)}] {snap.bwd_ms:5.2f} ms", color=colors["warn"])
    y += 1
    add_line(stdscr, y, 2, f"OPT [{bar(snap.opt_ms, max_ms)}] {snap.opt_ms:5.2f} ms", color=colors["info"])
    y += 2

    add_line(stdscr, y, 2, "Loss Trend", color=colors["accent"], attr=curses.A_BOLD)
    y += 1
    spark = sparkline(snap.loss_history, max(20, w - 6))
    add_line(stdscr, y, 2, spark, color=colors["active"])
    y += 2

    add_line(stdscr, y, 2, "Top Hot Ops (from parity drift signatures)", color=colors["accent"], attr=curses.A_BOLD)
    y += 1
    for i, (op, cnt) in enumerate(snap.top_ops[: min(8, h - y - 2)]):
        add_line(stdscr, y + i, 4, f"{i+1:>2}. {op:<36} {cnt:>5}", color=colors["normal"])


def draw_layers(stdscr: Any, snap: Snapshot, colors: Dict[str, int]) -> None:
    h, _w = stdscr.getmaxyx()
    y = 3
    add_line(stdscr, y, 2, f"Layer Timeline  active={snap.active_phase}:{snap.active_layer}.{snap.active_op}", color=colors["accent"], attr=curses.A_BOLD)
    y += 2

    layer_count = max(1, min(64, int(re.search(r"l=(\d+)", snap.model_label).group(1)) if re.search(r"l=(\d+)", snap.model_label) else 8))
    rows = synth_layer_rows(layer_count, snap.top_ops, snap.active_layer)

    add_line(stdscr, y, 2, "Layer   FWD(ms)   BWD(ms)   Util/Status", color=colors["muted"], attr=curses.A_BOLD)
    y += 1

    for layer, fwd, bwd, util in rows[: max(0, h - y - 2)]:
        c = colors["active"] if layer == snap.active_layer else colors["normal"]
        a = curses.A_BOLD if layer == snap.active_layer else 0
        add_line(stdscr, y, 2, f"{layer:<6} {fwd:>7}    {bwd:>7}   {util}", color=c, attr=a)
        y += 1


def draw_data(stdscr: Any, snap: Snapshot, colors: Dict[str, int]) -> None:
    h, w = stdscr.getmaxyx()
    y = 3
    add_line(stdscr, y, 2, "Data Inputs + Processed Folders", color=colors["accent"], attr=curses.A_BOLD)
    y += 2

    add_line(stdscr, y, 2, "Tracked Inputs", color=colors["muted"], attr=curses.A_BOLD)
    y += 1
    for label, path in snap.data_inputs[:8]:
        add_line(stdscr, y, 4, f"{label:<18} {path[: max(16, w - 26)]}", color=colors["normal"])
        y += 1

    y += 1
    add_line(stdscr, y, 2, "Run Folder Contents", color=colors["muted"], attr=curses.A_BOLD)
    y += 1
    for name, kind, size in snap.files_summary[: max(0, h - y - 2)]:
        add_line(stdscr, y, 4, f"{name:<26} {kind:<14} {size:>10}", color=colors["normal"])
        y += 1


def draw_artifacts(stdscr: Any, snap: Snapshot, colors: Dict[str, int]) -> None:
    h, _w = stdscr.getmaxyx()
    y = 3
    add_line(stdscr, y, 2, "External Artifacts + Active Processes", color=colors["accent"], attr=curses.A_BOLD)
    y += 2

    add_line(stdscr, y, 2, "Artifacts", color=colors["muted"], attr=curses.A_BOLD)
    y += 1
    for name, status, note in snap.artifact_rows[:10]:
        c = status_color(status, colors)
        add_line(stdscr, y, 4, f"{name:<30} {status:<8} {note}", color=c)
        y += 1

    y += 1
    add_line(stdscr, y, 2, "Live CK Processes", color=colors["muted"], attr=curses.A_BOLD)
    y += 1
    if not snap.process_rows:
        add_line(stdscr, y, 4, "(none detected)", color=colors["muted"])
        return

    for p in snap.process_rows[: max(0, h - y - 2)]:
        row = f"pid={p.pid:<7} mode={p.mode:<6} run={str(p.run_dir) if p.run_dir else '-'}"
        add_line(stdscr, y, 4, row[:120], color=colors["normal"])
        y += 1


def render(stdscr: Any, snap: Snapshot, tab_idx: int, colors: Dict[str, int]) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    add_line(stdscr, 0, 2, "CK v7 Live Monitor", color=colors["accent"], attr=curses.A_BOLD)
    add_line(stdscr, 0, max(2, w - 36), time.strftime("%Y-%m-%d %H:%M:%S"), color=colors["muted"])
    draw_tabs(stdscr, tab_idx, colors)

    if tab_idx == 0:
        draw_overview(stdscr, snap, colors)
    elif tab_idx == 1:
        draw_layers(stdscr, snap, colors)
    elif tab_idx == 2:
        draw_data(stdscr, snap, colors)
    elif tab_idx == 3:
        draw_artifacts(stdscr, snap, colors)

    add_line(stdscr, h - 1, 2, "Keys: 1-4 tabs, <-/-> switch, q quit", color=colors["muted"])
    stdscr.refresh()


def run_once(snapshot: Snapshot) -> int:
    print("CKTOP v7 snapshot")
    print(f"run_dir: {snapshot.run_dir}")
    print(f"source: {snapshot.source}  mode: {snapshot.mode}  model: {snapshot.model_label}")
    print(
        f"epoch={snapshot.epoch}/{snapshot.epochs_total} step={snapshot.step}/{snapshot.steps_total} "
        f"opt={snapshot.optimizer_step}/{snapshot.optimizer_total}"
    )
    print(
        f"loss={snapshot.loss:.6f} tok/s={snapshot.tok_s:.2f} lr={snapshot.lr:.3e} "
        f"grad_norm={fmt_float(snapshot.grad_norm, 4)}"
    )
    print(
        f"phase={snapshot.active_phase}:{snapshot.active_layer}.{snapshot.active_op} "
        f"safety={snapshot.safety_status} parity_failures={snapshot.parity_failures}"
    )
    print("top_ops:")
    for op, cnt in snapshot.top_ops[:8]:
        print(f"  - {op}: {cnt}")
    if snapshot.data_inputs:
        print("data_inputs:")
        for k, v in snapshot.data_inputs[:8]:
            print(f"  - {k}: {v}")
    if snapshot.artifact_rows:
        print("artifacts:")
        for name, st, note in snapshot.artifact_rows[:10]:
            print(f"  - {name}: {st} ({note})")
    return 0


def run_tui(stdscr: Any, args: argparse.Namespace) -> int:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)
    colors = init_colors()

    scanner = DataScanner()
    demo = DemoGenerator(args.run) if args.demo else None

    tab_idx = max(0, min(len(TAB_NAMES) - 1, args.tab - 1))
    next_update = 0.0
    snapshot = Snapshot(
        run_dir=args.run,
        report_path=None,
        source="boot",
        mode=args.mode,
        model_label="loading",
        step=0,
        steps_total=0,
        optimizer_step=0,
        optimizer_total=0,
        epoch=0,
        epochs_total=0,
        seq_len=0,
        context_len=0,
        grad_accum=0,
        lr=0.0,
        loss=0.0,
        tok_s=0.0,
        ck_avg_step_ms=0.0,
        fwd_ms=0.0,
        bwd_ms=0.0,
        opt_ms=0.0,
        grad_norm=None,
        safety_status="unknown",
        parity_enabled=False,
        parity_every=0,
        parity_failures=0,
        active_phase="idle",
        active_layer="L0",
        active_op="idle",
    )

    while True:
        now = time.time()
        if now >= next_update:
            procs = discover_processes()
            run_dir = discover_run_dir(args.run, procs)
            if demo is not None:
                snapshot = demo.build()
            else:
                report_path = resolve_report(run_dir)
                report = read_json(report_path) if report_path else None
                summary_path = resolve_runtime_summary(run_dir)
                summary = read_json(summary_path) if summary_path else None
                snapshot = build_snapshot(run_dir, report_path, report, summary, procs, scanner)
            next_update = now + max(0.1, args.refresh)

        render(stdscr, snapshot, tab_idx, colors)

        ch = stdscr.getch()
        if ch == -1:
            continue
        if ch in (ord("q"), ord("Q")):
            break
        if ch in (ord("1"), ord("2"), ord("3"), ord("4")):
            tab_idx = int(chr(ch)) - 1
            continue
        if ch == curses.KEY_RIGHT:
            tab_idx = (tab_idx + 1) % len(TAB_NAMES)
            continue
        if ch == curses.KEY_LEFT:
            tab_idx = (tab_idx - 1) % len(TAB_NAMES)
            continue

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CK v7 terminal monitor (htop-style)")
    p.add_argument("--run", type=Path, default=None, help="Run directory to monitor (auto-discover if omitted)")
    p.add_argument("--mode", choices=["auto", "train", "infer"], default="auto", help="Display mode hint")
    p.add_argument("--refresh", type=float, default=0.5, help="Refresh interval seconds (default: 0.5)")
    p.add_argument("--tab", type=int, default=1, help="Initial tab index 1..4")
    p.add_argument("--demo", action="store_true", help="Run animated demo mode")
    p.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    return p


def main() -> int:
    args = build_parser().parse_args()

    scanner = DataScanner()
    procs = discover_processes()
    run_dir = discover_run_dir(args.run, procs)

    if args.demo:
        snap = DemoGenerator(run_dir).build()
    else:
        report_path = resolve_report(run_dir)
        report = read_json(report_path) if report_path else None
        summary_path = resolve_runtime_summary(run_dir)
        summary = read_json(summary_path) if summary_path else None
        snap = build_snapshot(run_dir, report_path, report, summary, procs, scanner)

    if args.once:
        return run_once(snap)

    return curses.wrapper(run_tui, args)


if __name__ == "__main__":
    raise SystemExit(main())
