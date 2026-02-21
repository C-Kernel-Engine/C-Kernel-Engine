#!/usr/bin/env python3
"""
IR Visualizer Launcher for C-Kernel-Engine v7

Usage:
    python3 version/v7/tools/open_ir_visualizer.py              # Open visualizer
    python3 version/v7/tools/open_ir_visualizer.py --list       # List available models
    python3 version/v7/tools/open_ir_visualizer.py <model>      # Generate and open report
    python3 version/v7/tools/open_ir_visualizer.py --generate <model>  # Generate full report (profile+probes)
    python3 version/v7/tools/open_ir_visualizer.py --generate <model> --interactive
    python3 version/v7/tools/open_ir_visualizer.py --generate <model> --html-only  # Generate HTML only
    python3 version/v7/tools/open_ir_visualizer.py --generate <model> --with-profile
    python3 version/v7/tools/open_ir_visualizer.py --generate <model> --with-probes
    python3 version/v7/tools/open_ir_visualizer.py --generate <model> --with-probes --weight-dtype float32
    python3 version/v7/tools/open_ir_visualizer.py --generate <model> --with-probes --perf-runtime cli
    python3 version/v7/tools/open_ir_visualizer.py --generate <model> --no-vtune
    python3 version/v7/tools/open_ir_visualizer.py --generate --run ./runs/exp1_baseline --with-probes --advisor
    python3 version/v7/tools/open_ir_visualizer.py --generate <model> --with-probes --run-model hf://... --chat-template none
    python3 version/v7/tools/open_ir_visualizer.py --generate --run ./runs/exp1_baseline
"""
import os
import sys
import json
import base64
import re
import shutil
import webbrowser
import argparse
import subprocess
from pathlib import Path

# Path construction:
# Script is at: version/v7/tools/open_ir_visualizer.py
SCRIPT_DIR = Path(__file__).parent              # .../version/v7/tools
V7_ROOT = SCRIPT_DIR.parent                    # .../version/v7
PROJECT_ROOT = V7_ROOT.parent.parent           # .../Workspace/C-Kernel-Engine

sys.path.insert(0, str(V7_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_PATH = Path.home() / ".cache" / "ck-engine-v7" / "models"
CACHE_PATH_FALLBACK = Path.home() / ".cache" / "ck-engine-v6.6" / "models"
VISUALIZER = SCRIPT_DIR / "ir_visualizer.html"
CK_RUN_SCRIPT = V7_ROOT / "scripts" / "ck_run_v7.py"
MEMORY_SIGNOFF_SCRIPT = V7_ROOT / "scripts" / "memory_signoff_v7.py"
V7_REPORT_PATH = Path(os.environ.get("CK_V7_REPORT_DIR", str(V7_ROOT / ".cache" / "reports"))).expanduser()
V7_REPORT_PATH_LEGACY = V7_ROOT / "reports"


def run_cmd(cmd: list[str], cwd: Path, extra_env: dict | None = None):
    print(f"[run] {' '.join(cmd)}")
    env = None
    if extra_env:
        env = dict(os.environ)
        env.update(extra_env)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def try_run_cmd(step: str, cmd: list[str], cwd: Path, extra_env: dict | None = None) -> bool:
    try:
        run_cmd(cmd, cwd=cwd, extra_env=extra_env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: {step} failed (continuing): {e}")
        return False


def resolve_model_target(model_arg: str) -> tuple[Path, Path]:
    """Resolve a model argument to ck_build/model dir and model root directory."""
    candidate = Path(model_arg)
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"Model path is not a directory: {candidate}")
        if candidate.name in {"ck_build", ".ck_build"}:
            ck_build = candidate
            model_root = candidate.parent
        else:
            if (candidate / "ck_build").exists():
                ck_build = candidate / "ck_build"
            elif (candidate / ".ck_build").exists():
                ck_build = candidate / ".ck_build"
            else:
                ck_build = candidate
            model_root = candidate
        return ck_build, model_root

    ck_build = CACHE_PATH / model_arg / "ck_build"
    if ck_build.exists():
        return ck_build, CACHE_PATH / model_arg

    dot_ck_build = CACHE_PATH / model_arg / ".ck_build"
    if dot_ck_build.exists():
        return dot_ck_build, CACHE_PATH / model_arg

    model_dir = CACHE_PATH / model_arg
    if model_dir.exists():
        return model_dir, model_dir

    ck_build_fallback = CACHE_PATH_FALLBACK / model_arg / "ck_build"
    if ck_build_fallback.exists():
        return ck_build_fallback, CACHE_PATH_FALLBACK / model_arg

    dot_ck_build_fallback = CACHE_PATH_FALLBACK / model_arg / ".ck_build"
    if dot_ck_build_fallback.exists():
        return dot_ck_build_fallback, CACHE_PATH_FALLBACK / model_arg

    model_dir_fallback = CACHE_PATH_FALLBACK / model_arg
    if model_dir_fallback.exists():
        return model_dir_fallback, model_dir_fallback

    raise ValueError(f"Model not found: {model_arg}")


def resolve_run_target(run_arg: Path) -> tuple[Path, Path]:
    """Resolve a run directory to (preferred_build_dir, run_root)."""
    run_root = run_arg.expanduser().resolve()
    if not run_root.exists():
        raise ValueError(f"Run directory not found: {run_root}")
    if not run_root.is_dir():
        raise ValueError(f"Run path is not a directory: {run_root}")

    if (run_root / "ck_build").exists():
        return run_root / "ck_build", run_root
    if (run_root / ".ck_build").exists():
        return run_root / ".ck_build", run_root
    return run_root, run_root


def has_local_runnable_source(path: Path) -> bool:
    """Whether path can be used as ck_run_v7 local input source."""
    if not path.exists() or not path.is_dir():
        return False
    if any(path.glob("*.gguf")):
        return True
    if any(path.glob("*.safetensors")):
        return True
    if any(path.glob("*.safetensors.index.json")):
        return True
    if (path / "model.safetensors.index.json").exists():
        return True
    return False


def has_local_compiled_runtime(path: Path) -> bool:
    """Whether path already has compiled runtime outputs usable by ck-cli."""
    if not (
        path.exists()
        and path.is_dir()
        and (path / "libmodel.so").exists()
        and (path / "weights.bump").exists()
    ):
        return False
    # Reject runtimes with unresolved shared-lib deps (e.g., libimf.so missing).
    lib_path = path / "libmodel.so"
    try:
        result = subprocess.run(
            ["ldd", str(lib_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        text = f"{result.stdout}\n{result.stderr}"
        if "=> not found" in text:
            return False
    except Exception:
        # If ldd is unavailable, keep best-effort behavior.
        pass
    return True


def resolve_local_runnable_input(path: Path) -> str:
    """
    Prefer concrete local file inputs where possible.
    - If directory contains a GGUF, return that file path (ck_run handles gguf file directly).
    - Otherwise return directory path.
    """
    if path.exists() and path.is_dir():
        ggufs = sorted(path.glob("*.gguf"))
        if ggufs:
            return str(ggufs[0])
    return str(path)


def infer_run_model_input(model_root: Path, weight_dtype: str | None = None) -> str | None:
    """
    Best-effort mapping from cache model folder names to hf:// model inputs.
    Keeps launcher ergonomic for common v7 model aliases.
    For training-oriented dtypes (float32/bf16), prefer source checkpoints.
    """
    name = model_root.name.lower()
    wants_source_ckpt = weight_dtype in {"float32", "bf16"}
    if "gemma-3-270m-it" in name:
        return "hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf"
    if "qwen2-0.5b-instruct" in name or "qwen2-0_5b-instruct" in name:
        if wants_source_ckpt:
            return "Qwen/Qwen2-0.5B-Instruct"
        return "hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf"
    if "qwen3-0.6b" in name:
        if wants_source_ckpt:
            return "Qwen/Qwen3-0.6B"
        return "hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
    return None


def infer_chat_template(run_model_input: str, model_root: Path) -> str:
    probe = f"{run_model_input} {model_root.name}".lower()
    # Gemma GGUF chat templates are often not compatible with our runtime path;
    # prefer raw prompt mode unless the operator overrides explicitly.
    if "gemma" in probe:
        return "none"
    return "auto"


def prompt_with_default(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    return raw


def maybe_interactive_configure(args: argparse.Namespace, model_root: Path) -> None:
    if not getattr(args, "interactive", False):
        return
    if not sys.stdin.isatty():
        print("Warning: --interactive requested, but stdin is not a TTY. Skipping prompts.")
        return

    print("\nInteractive report options (press Enter to keep default):")

    suggested_model = args.run_model or ""
    if not suggested_model:
        if has_local_runnable_source(model_root):
            suggested_model = resolve_local_runnable_input(model_root)
        else:
            suggested_model = infer_run_model_input(model_root, args.weight_dtype) or ""
    if suggested_model:
        args.run_model = prompt_with_default("Runtime source (--run-model)", suggested_model)
    else:
        manual = input("Runtime source (--run-model) [leave blank for auto-infer]: ").strip()
        if manual:
            args.run_model = manual

    weight_default = args.weight_dtype if args.weight_dtype else "auto"
    allowed_weight = {"auto", "float32", "bf16", "q4_0", "q4_1", "q4_k", "q4_k_m", "q5_0", "q5_1", "q6_k", "q8_0"}
    while True:
        chosen = prompt_with_default("Weight dtype", weight_default)
        if chosen in allowed_weight:
            args.weight_dtype = None if chosen == "auto" else chosen
            break
        print(f"Invalid weight dtype: {chosen}")

    chat_default = args.chat_template if args.chat_template else "auto"
    allowed_chat = {"auto", "none", "qwen", "gemma"}
    while True:
        chosen = prompt_with_default("Chat template", chat_default)
        if chosen in allowed_chat:
            args.chat_template = chosen
            break
        print(f"Invalid chat template: {chosen}")

    perf_default = args.perf_runtime if args.perf_runtime else "cli"
    allowed_perf = {"cli", "python"}
    while True:
        chosen = prompt_with_default("Perf runtime", perf_default)
        if chosen in allowed_perf:
            args.perf_runtime = chosen
            break
        print(f"Invalid perf runtime: {chosen}")

    print(
        "Selected: "
        f"run-model={args.run_model or '<auto>'}, "
        f"weight-dtype={args.weight_dtype or 'auto'}, "
        f"chat-template={args.chat_template}, "
        f"perf-runtime={args.perf_runtime}"
    )


def detect_model_dir_from_input(model_input: str) -> Path | None:
    try:
        from ck_run_v7 import CACHE_DIR, detect_input_type  # type: ignore
    except Exception:
        return None

    input_type, info = detect_input_type(model_input)
    if input_type == "hf_gguf":
        return CACHE_DIR / info["repo_id"].replace("/", "--")
    if input_type == "hf_id":
        return CACHE_DIR / info["model_id"].replace("/", "--")
    if input_type == "gguf":
        return CACHE_DIR / info["path"].stem
    if input_type == "local_dir":
        local = Path(info["path"]).resolve()
        return local.parent if local.name == ".ck_build" else local
    if input_type == "local_config":
        cfg_parent = Path(info["path"]).resolve().parent
        return cfg_parent.parent if cfg_parent.name == ".ck_build" else cfg_parent
    return None


def maybe_localize_hf_gguf(model_input: str) -> str:
    """
    If model_input is hf://repo/file.gguf and a local .ck_cache copy exists,
    use the local file path to avoid network dependence.
    """
    if not model_input.startswith("hf://"):
        return model_input
    try:
        body = model_input[len("hf://"):]
        repo_id, filename = body.rsplit("/", 1)
        if not filename.lower().endswith(".gguf"):
            return model_input
        local = PROJECT_ROOT / ".ck_cache" / repo_id.replace("/", "--") / filename
        if local.exists():
            print(f"Using local cached GGUF: {local}")
            return str(local)
    except Exception:
        return model_input
    return model_input


def copy_artifacts_if_needed(src_model_dir: Path, dst_model_dir: Path) -> None:
    if src_model_dir.resolve() == dst_model_dir.resolve():
        return
    artifact_names = [
        "profile_summary.json",
        "perf_stat_summary.json",
        "flamegraph_manifest.json",
        "cachegrind_summary.json",
        "asan_summary.json",
        "vtune_summary.json",
        "advisor_summary.json",
        "memory_signoff.json",
        "memory_verification_latest.json",
        "perf_gate_report.json",
        "ir1_train_forward.json",
        "ir2_train_backward.json",
        "ir2_train_summary.json",
        "ir_train_invariants.json",
        "contract_report_latest.json",
        "parity_1token_latest.json",
        "qk_norm_backward_parity_latest.json",
        "fd_gradients_latest.json",
        "train_parity_epochs_3_latest.json",
        "train_parity_epochs_5_latest.json",
        "replay_determinism_latest.json",
        "backprop_stitch_runtime_latest.json",
        "regression_ledger.json",
    ]
    copied = 0
    dst_model_dir.mkdir(parents=True, exist_ok=True)
    src_roots = [src_model_dir, src_model_dir / ".ck_build"]
    for name in artifact_names:
        src = None
        for root in src_roots:
            candidate = root / name
            if candidate.exists():
                src = candidate
                break
        if src is None:
            continue
        shutil.copy2(src, dst_model_dir / name)
        copied += 1
    if copied:
        print(f"Copied {copied} artifact(s) from {src_model_dir} -> {dst_model_dir}")


def has_model_artifact(model_root: Path, ck_build: Path, name: str) -> bool:
    return (
        (model_root / name).exists()
        or (model_root / ".ck_build" / name).exists()
        or (ck_build / name).exists()
    )


def validate_artifact_set(
    model_root: Path,
    ck_build: Path,
    expect_perf: bool,
    expect_vtune: bool,
    expect_advisor: bool = False,
    expect_cachegrind: bool = False,
    expect_asan: bool = False,
) -> list[str]:
    missing: list[str] = []
    base_required = ["memory_signoff.json", "profile_summary.json"]
    for name in base_required:
        if not has_model_artifact(model_root, ck_build, name):
            missing.append(name)
    if expect_perf:
        for name in ["perf_stat_summary.json", "flamegraph_manifest.json", "perf_gate_report.json"]:
            if not has_model_artifact(model_root, ck_build, name):
                missing.append(name)
    if expect_vtune and not has_model_artifact(model_root, ck_build, "vtune_summary.json"):
        missing.append("vtune_summary.json")
    if expect_advisor and not has_model_artifact(model_root, ck_build, "advisor_summary.json"):
        missing.append("advisor_summary.json")
    if expect_cachegrind and not has_model_artifact(model_root, ck_build, "cachegrind_summary.json"):
        missing.append("cachegrind_summary.json")
    if expect_asan and not has_model_artifact(model_root, ck_build, "asan_summary.json"):
        missing.append("asan_summary.json")
    return missing


def has_train_runtime_artifacts(run_dir: Path) -> bool:
    if not run_dir.exists() or not run_dir.is_dir():
        return False
    has_lib = (run_dir / "libtrain.so").exists() or (run_dir / ".ck_build" / "libtrain.so").exists()
    has_weights = (run_dir / "weights.bump").exists()
    has_manifest = (run_dir / "weights_manifest.json").exists()
    has_summary = (run_dir / "generated_train_runtime_summary_v7.json").exists()
    return bool(has_lib and has_weights and has_manifest and has_summary)


def infer_train_vocab_size(run_dir: Path, default_vocab: int = 1024) -> int:
    candidates = [
        run_dir / "train_init_config.json",
        run_dir / "config.json",
    ]
    keys = ("train_vocab", "vocab_size", "vocab", "n_vocab")
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        # Direct keys.
        for key in keys:
            value = payload.get(key)
            if isinstance(value, int) and value > 0:
                return value
        # Nested config dictionary.
        cfg = payload.get("config")
        if isinstance(cfg, dict):
            for key in keys:
                value = cfg.get(key)
                if isinstance(value, int) and value > 0:
                    return value
    return default_vocab


def ensure_train_token_file(run_dir: Path, token_count: int = 8192) -> Path:
    vocab = max(2, infer_train_vocab_size(run_dir))
    token_path = run_dir / ".ck_profile_tokens_viz.txt"
    if token_path.exists():
        return token_path
    seq = " ".join(str(i % vocab) for i in range(token_count))
    token_path.write_text(seq)
    return token_path


def _resolve_asset_path(raw_path: str, ck_build_path: Path, model_root: Path) -> Path | None:
    if not raw_path:
        return None
    p = Path(raw_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(model_root / p)
        candidates.append(ck_build_path / p)
        candidates.append(PROJECT_ROOT / p)
    for c in candidates:
        if c.exists():
            return c
    return None


def _encode_image_data_uri(path: Path) -> str | None:
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix)
    if mime is None:
        return None
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _trim_memory_signoff_payload(payload: dict, max_items: int = 120) -> dict:
    """Trim very large warning/error arrays to keep standalone report size manageable."""
    out = dict(payload)
    # Preserve original totals (if present) for drill-down/debug.
    if "warning_count" in out:
        out["warning_count_raw"] = out.get("warning_count", 0)
    if "error_count" in out:
        out["error_count_raw"] = out.get("error_count", 0)
    checks = out.get("checks")
    if not isinstance(checks, dict):
        return out
    trimmed_checks = {}
    total_errors = 0
    total_warnings = 0
    for key, check in checks.items():
        if not isinstance(check, dict):
            trimmed_checks[key] = check
            continue
        c2 = dict(check)
        for list_key in ("warnings", "errors", "info"):
            arr = c2.get(list_key)
            if isinstance(arr, list) and len(arr) > max_items:
                c2[f"{list_key}_truncated"] = len(arr) - max_items
                c2[list_key] = arr[:max_items]
        errs = c2.get("errors")
        warns = c2.get("warnings")
        if isinstance(errs, list):
            total_errors += len(errs)
        if isinstance(warns, list):
            total_warnings += len(warns)
        trimmed_checks[key] = c2
    out["checks"] = trimmed_checks
    # Keep headline counts consistent with what is rendered in each row.
    out["error_count"] = total_errors
    out["warning_count"] = total_warnings
    return out


def _piece_display(piece: str, limit: int = 64) -> str:
    disp = piece.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if len(disp) > limit:
        disp = disp[:limit] + "..."
    return disp


def _bytes_hex_preview(text: str, limit: int = 12) -> str:
    raw = text.encode("utf-8", errors="replace")
    chunk = raw[:limit]
    hx = " ".join(f"{b:02X}" for b in chunk)
    if len(raw) > limit:
        hx += " ..."
    return hx


def _parse_merge_row(row) -> tuple[str, str]:
    if isinstance(row, list) and len(row) >= 2:
        return str(row[0]), str(row[1])
    if isinstance(row, str):
        parts = row.split(" ", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return "", ""


def _build_tokenizer_preview(
    files: dict,
    ck_build_path: Path,
    model_root: Path,
    run_dir: Path | None = None,
) -> dict | None:
    pipeline = files.get("training_pipeline") if isinstance(files.get("training_pipeline"), dict) else {}
    data_lab = pipeline.get("data_lab") if isinstance(pipeline, dict) and isinstance(pipeline.get("data_lab"), dict) else {}
    tokenizer_lineage = pipeline.get("tokenizer_lineage") if isinstance(pipeline, dict) and isinstance(pipeline.get("tokenizer_lineage"), dict) else {}

    candidate_values: list[str] = []
    for raw in (
        data_lab.get("tokenizer_json_path"),
        tokenizer_lineage.get("tokenizer_path"),
        str((run_dir / "tokenizer.json")) if run_dir is not None else None,
        str(model_root / "tokenizer.json"),
        str(ck_build_path / "tokenizer.json"),
    ):
        if isinstance(raw, str) and raw.strip():
            candidate_values.append(raw.strip())

    tokenizer_json_path: Path | None = None
    for raw in candidate_values:
        resolved = _resolve_asset_path(raw, ck_build_path, model_root)
        if resolved and resolved.exists():
            tokenizer_json_path = resolved
            break
    if tokenizer_json_path is None:
        return None

    try:
        payload = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "status": "error",
            "path": str(tokenizer_json_path),
            "error": f"failed_to_parse_tokenizer_json: {e}",
        }

    model = payload.get("model") if isinstance(payload, dict) else {}
    if not isinstance(model, dict):
        model = {}
    model_type = str(model.get("type") or "unknown")
    vocab_raw = model.get("vocab")
    vocab_items: list[tuple[int, str]] = []
    id_to_piece: dict[int, str] = {}
    if isinstance(vocab_raw, dict):
        for piece, idx in vocab_raw.items():
            if isinstance(piece, str) and isinstance(idx, int) and idx >= 0:
                vocab_items.append((int(idx), piece))
                id_to_piece[int(idx)] = piece
    vocab_items.sort(key=lambda kv: kv[0])

    merges_raw = model.get("merges")
    merge_samples: list[dict] = []
    merge_count = 0
    if isinstance(merges_raw, list):
        merge_count = len(merges_raw)
        for i, row in enumerate(merges_raw[:48]):
            left, right = _parse_merge_row(row)
            if not left and not right:
                continue
            merge_samples.append(
                {
                    "rank": int(i),
                    "left": _piece_display(left),
                    "right": _piece_display(right),
                    "merged_hint": _piece_display(left + right),
                }
            )

    non_ascii_piece_count = 0
    for _, piece in vocab_items:
        if any(ord(ch) > 127 for ch in piece):
            non_ascii_piece_count += 1

    vocab_samples = [
        {
            "id": int(idx),
            "piece": _piece_display(piece),
            "bytes_hex": _bytes_hex_preview(piece),
        }
        for idx, piece in vocab_items[:48]
    ]

    added_tokens = payload.get("added_tokens")
    special_tokens: list[dict] = []
    if isinstance(added_tokens, list):
        for row in added_tokens[:32]:
            if not isinstance(row, dict):
                continue
            content = row.get("content")
            token_id = row.get("id")
            if isinstance(content, str):
                special_tokens.append(
                    {
                        "id": int(token_id) if isinstance(token_id, int) else None,
                        "content": _piece_display(content),
                        "special": bool(row.get("special")),
                    }
                )

    pre_tokenizer = payload.get("pre_tokenizer")
    decoder = payload.get("decoder")
    pre_tokenizer_type = pre_tokenizer.get("type") if isinstance(pre_tokenizer, dict) else None
    decoder_type = decoder.get("type") if isinstance(decoder, dict) else None

    roundtrip = files.get("tokenizer_roundtrip") if isinstance(files.get("tokenizer_roundtrip"), dict) else {}
    sample_rows = roundtrip.get("sample_rows") if isinstance(roundtrip.get("sample_rows"), list) else []
    example = None
    for row in sample_rows:
        if not isinstance(row, dict):
            continue
        ids = row.get("token_ids")
        if not isinstance(ids, list) or not ids:
            continue
        flow = []
        for tid in ids[:48]:
            if not isinstance(tid, int):
                continue
            piece = id_to_piece.get(int(tid))
            flow.append(
                {
                    "id": int(tid),
                    "piece": _piece_display(piece) if isinstance(piece, str) else "<missing>",
                }
            )
        example = {
            "line_no": row.get("line_no"),
            "source": row.get("source"),
            "decoded": row.get("decoded"),
            "exact_match": row.get("exact_match"),
            "token_flow": flow,
        }
        break

    return {
        "status": "ok",
        "path": str(tokenizer_json_path),
        "model_type": model_type,
        "vocab_size": int(len(vocab_items)),
        "merge_count": int(merge_count),
        "pre_tokenizer_type": pre_tokenizer_type,
        "decoder_type": decoder_type,
        "bytelevel_mode": bool(pre_tokenizer_type == "ByteLevel" or decoder_type == "ByteLevel"),
        "ascii_piece_count": int(len(vocab_items) - non_ascii_piece_count),
        "non_ascii_piece_count": int(non_ascii_piece_count),
        "vocab_samples": vocab_samples,
        "merge_samples": merge_samples,
        "special_tokens": special_tokens,
        "encode_decode_example": example,
    }


def list_available_models():
    """List all models in cache."""
    models = []
    for cache_root in [CACHE_PATH, CACHE_PATH_FALLBACK]:
        if not cache_root.exists():
            continue
        for model_dir in cache_root.iterdir():
            if not model_dir.is_dir():
                continue

            ck_build = model_dir / "ck_build"
            has_data = (
                (ck_build.exists() and (ck_build / "ir1_decode.json").exists()) or
                (model_dir / "ir1_decode.json").exists()
            )

            models.append({
                "name": model_dir.name,
                "path": str(ck_build),
                "has_data": has_data
            })

    return sorted(models, key=lambda m: m["name"])


def collect_analysis_checkpoints(search_roots: list[Path]) -> dict | None:
    """Load and merge analysis_checkpoint_step_*.json files by step."""
    step_to_path: dict[int, Path] = {}
    # Preserve root priority: earlier roots win for the same step.
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for candidate in sorted(root.glob("analysis_checkpoint_step_*.json")):
            m = re.search(r"_step_(\d+)\.json$", candidate.name)
            if not m:
                continue
            step = int(m.group(1))
            if step not in step_to_path:
                step_to_path[step] = candidate

    checkpoints: list[dict] = []
    for step in sorted(step_to_path.keys()):
        p = step_to_path[step]
        try:
            with open(p, "r") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                if "step" not in payload:
                    payload["step"] = step
                payload.setdefault("_source", str(p))
                checkpoints.append(payload)
        except Exception as e:
            print(f"  ! analysis checkpoint {p.name}: {e}")

    if not checkpoints:
        return None

    return {
        "schema_version": "ck.analysis.v1",
        "count": len(checkpoints),
        "checkpoints": checkpoints,
    }


def _has_renderable_profile_payload(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    entries = payload.get("entries")
    if isinstance(entries, list) and len(entries) > 0:
        return True
    by_op = payload.get("by_op")
    if isinstance(by_op, dict) and len(by_op) > 0:
        return True
    by_mode = payload.get("by_mode")
    if isinstance(by_mode, dict) and len(by_mode) > 0:
        return True
    return False


def _is_profile_summary_stub(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    if _has_renderable_profile_payload(payload):
        return False
    return str(payload.get("schema") or "") == "ck.profile.summary.v1"


def _derive_profile_alias_roots(run_dir: Path | None, model_root: Path) -> list[Path]:
    """
    Derive sibling cache roots from local GGUF filenames.
    Example:
      run dir: .../Qwen--Qwen3-0.6B-GGUF
      gguf:    Qwen3-0.6B-Q8_0.gguf
      alias:   .../Qwen3-0.6B-Q8_0
    """
    bases: list[Path] = []
    if run_dir is not None:
        bases.append(run_dir)
    bases.append(model_root)

    roots: list[Path] = []
    seen: set[str] = set()

    def push(p: Path) -> None:
        key = str(p)
        if key in seen:
            return
        seen.add(key)
        roots.append(p)

    for base in bases:
        if not base.exists() or not base.is_dir():
            continue
        parent = base.parent
        for gguf in sorted(base.glob("*.gguf")):
            stem = gguf.stem
            if not stem:
                continue
            alias_root = parent / stem
            if alias_root.exists() and alias_root.is_dir():
                push(alias_root)
                push(alias_root / "ck_build")
                push(alias_root / ".ck_build")
    return roots


def _resolve_path_loose(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except TypeError:
        # Python <3.10 compatibility (strict kw may be unavailable in some envs)
        return path.resolve()
    except Exception:
        return path


def _path_under_any_root(path: Path, roots: list[Path]) -> bool:
    target = _resolve_path_loose(path)
    for root in roots:
        base = _resolve_path_loose(root)
        if target == base or base in target.parents:
            return True
    return False


def load_model_data(
    ck_build_path: Path,
    run_dir: Path | None = None,
    strict_run_artifacts: bool = False,
) -> dict:
    """Load all IR/profile/training data for a model or run directory."""
    strict_run_scope = bool(strict_run_artifacts and run_dir is not None)
    if run_dir is not None:
        model_root = run_dir
        model_name = run_dir.name
    else:
        model_name = ck_build_path.parent.name if ck_build_path.name in {"ck_build", ".ck_build"} else ck_build_path.name
        model_root = ck_build_path.parent if ck_build_path.name in {"ck_build", ".ck_build"} else ck_build_path
    train_runtime_available = has_train_runtime_artifacts(run_dir if run_dir is not None else model_root)

    search_roots: list[Path] = []
    if run_dir is not None:
        search_roots.extend([run_dir, run_dir / "ck_build", run_dir / ".ck_build"])
    search_roots.extend([ck_build_path, model_root, model_root / ".ck_build"])

    deduped_roots: list[Path] = []
    seen_roots: set[str] = set()
    for root in search_roots:
        key = str(root)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        deduped_roots.append(root)
    search_roots = deduped_roots

    # Include canonical sibling cache roots derived from local GGUF stems.
    # This handles cases like:
    #   run_dir = .../Qwen--Qwen3-0.6B-GGUF
    #   profile outputs in .../Qwen3-0.6B-Q8_0
    profile_alias_roots = _derive_profile_alias_roots(run_dir, model_root)
    for root in profile_alias_roots:
        key = str(root)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        search_roots.append(root)

    # Define required vs optional files
    REQUIRED_FILES = [
        "ir1_decode",
        "layout_decode",
        "lowered_decode_call",
    ]
    # Run-directory workflows can be training-only; avoid false missing-required warnings.
    if run_dir is not None:
        REQUIRED_FILES = []
    OPTIONAL_FILES = [
        "ir1_prefill",
        "layout_prefill",
        "lowered_prefill_call",
        "lowered_decode",
        "lowered_prefill",
        "ir1_train",
        "ir2_train",
        "ir_train_invariants",
        "ir2_train_summary",
        "layout_train",
        "layout_train_audit",
        "train_exec_plan",
        "memory_diagnostic",
        "memory_verification",
        "generated_train_runtime_summary",
        "training_loss_curve",
        "training_grad_norms",
        "training_parity",
        "training_step_profile",
        "training_checkpoint_policy",
        "training_pipeline",
        "dataset_qc",
        "dataset_profile",
        "tokenizer_roundtrip",
        "post_train_eval",
        "training_epoch_sweep",
        "analysis_checkpoints",
        "train_e2e",
        "run_config",
        "sanity_overfit",
        "parity_report",
        "profile_latest",
        "contract_report",
        "parity_1token",
        "qk_norm_backward_parity",
        "fd_gradients",
        "train_parity_epochs_3",
        "train_parity_epochs_5",
        "train_runtime_parity_realistic",
        "train_runtime_parity_stress",
        "replay_determinism",
        "backprop_stitch_runtime",
        "grad_rules",
        "manifest",
        "kernel_registry",
        "profile_summary",
        "perf_stat_summary",
        "flamegraph_manifest",
        "cachegrind_summary",
        "asan_summary",
        "vtune_summary",
        "advisor_summary",
        "memory_signoff",
        "perf_gate_report",
        "regression_ledger",
        "embedding_dump",
    ]

    def model_candidates(name: str) -> list[Path]:
        return [root / name for root in search_roots]

    data_files = {
        "ir1_decode": model_candidates("ir1_decode.json"),
        "ir1_prefill": model_candidates("ir1_prefill.json"),
        "layout_decode": model_candidates("layout_decode.json"),
        "layout_prefill": model_candidates("layout_prefill.json"),
        "lowered_decode": model_candidates("lowered_decode.json"),
        "lowered_prefill": model_candidates("lowered_prefill.json"),
        "lowered_decode_call": model_candidates("lowered_decode_call.json") + model_candidates("lowered_decode.json"),
        "lowered_prefill_call": model_candidates("lowered_prefill_call.json") + model_candidates("lowered_prefill.json"),
        "ir1_train": model_candidates("ir1_train_forward.json") + model_candidates("ir1_train.json") + [V7_REPORT_PATH / "ir1_train_forward_latest.json", V7_REPORT_PATH_LEGACY / "ir1_train_forward_latest.json"],
        "ir2_train": model_candidates("ir2_train_backward.json") + model_candidates("ir2_train.json") + [V7_REPORT_PATH / "ir2_train_backward_latest.json", V7_REPORT_PATH_LEGACY / "ir2_train_backward_latest.json"],
        "ir_train_invariants": model_candidates("ir_train_invariants.json") + [V7_REPORT_PATH / "ir_train_invariants_latest.json", V7_REPORT_PATH_LEGACY / "ir_train_invariants_latest.json"],
        "ir2_train_summary": model_candidates("ir2_train_summary.json") + [V7_REPORT_PATH / "ir2_train_summary_latest.json", V7_REPORT_PATH_LEGACY / "ir2_train_summary_latest.json"],
        "layout_train": model_candidates("layout_train.json") + model_candidates("layout_train_latest.json") + [V7_REPORT_PATH / "layout_train_latest.json", V7_REPORT_PATH_LEGACY / "layout_train_latest.json"],
        "layout_train_audit": model_candidates("layout_train_audit.json") + model_candidates("layout_train_audit_latest.json") + [V7_REPORT_PATH / "layout_train_audit_latest.json", V7_REPORT_PATH_LEGACY / "layout_train_audit_latest.json"],
        "train_exec_plan": model_candidates("train_exec_plan.json") + model_candidates("train_exec_plan_latest.json") + [V7_REPORT_PATH / "train_exec_plan_latest.json", V7_REPORT_PATH_LEGACY / "train_exec_plan_latest.json"],
        "memory_diagnostic": model_candidates("memory_diagnostic_latest.json") + model_candidates("memory_diagnostic.json") + [V7_REPORT_PATH / "memory_diagnostic_latest.json", V7_REPORT_PATH_LEGACY / "memory_diagnostic_latest.json"],
        "memory_verification": model_candidates("memory_verification_latest.json") + model_candidates("memory_verification.json") + [V7_REPORT_PATH / "memory_verification_latest.json", V7_REPORT_PATH_LEGACY / "memory_verification_latest.json"],
        "generated_train_runtime_summary": model_candidates("generated_train_runtime_summary_v7.json") + model_candidates("generated_train_runtime_summary.json") + [V7_REPORT_PATH / "generated_train_runtime_summary_v7.json", V7_REPORT_PATH_LEGACY / "generated_train_runtime_summary_v7.json"],
        "training_loss_curve": model_candidates("training_loss_curve.json") + model_candidates("training_loss_curve_latest.json") + [V7_REPORT_PATH / "training_loss_curve_latest.json", V7_REPORT_PATH_LEGACY / "training_loss_curve_latest.json"],
        "training_grad_norms": model_candidates("training_grad_norms.json") + model_candidates("training_grad_norms_latest.json") + [V7_REPORT_PATH / "training_grad_norms_latest.json", V7_REPORT_PATH_LEGACY / "training_grad_norms_latest.json"],
        "training_parity": model_candidates("training_parity.json") + model_candidates("training_parity_latest.json") + [V7_REPORT_PATH / "training_parity_latest.json", V7_REPORT_PATH_LEGACY / "training_parity_latest.json"],
        "training_step_profile": model_candidates("training_step_profile.json") + model_candidates("training_step_profile_latest.json") + [V7_REPORT_PATH / "training_step_profile_latest.json", V7_REPORT_PATH_LEGACY / "training_step_profile_latest.json"],
        "training_checkpoint_policy": model_candidates("training_checkpoint_policy.json") + model_candidates("training_checkpoint_policy_latest.json") + [V7_REPORT_PATH / "training_checkpoint_policy_latest.json", V7_REPORT_PATH_LEGACY / "training_checkpoint_policy_latest.json"],
        "training_pipeline": model_candidates("training_pipeline.json") + model_candidates("training_pipeline_latest.json") + [V7_REPORT_PATH / "training_pipeline_latest.json", V7_REPORT_PATH_LEGACY / "training_pipeline_latest.json"],
        "dataset_qc": model_candidates("dataset_qc.json"),
        "dataset_profile": model_candidates("dataset_profile.json"),
        "tokenizer_roundtrip": model_candidates("tokenizer_roundtrip.json"),
        "post_train_eval": model_candidates("post_train_eval.json"),
        "training_epoch_sweep": model_candidates("training_epoch_sweep.json") + model_candidates("training_epoch_sweep_latest.json") + [V7_REPORT_PATH / "training_epoch_sweep_latest.json", V7_REPORT_PATH_LEGACY / "training_epoch_sweep_latest.json"],
        "train_e2e": model_candidates("train_e2e.json") + model_candidates("train_e2e_latest.json") + [V7_REPORT_PATH / "train_e2e_latest.json", V7_REPORT_PATH_LEGACY / "train_e2e_latest.json"],
        "run_config": model_candidates("config.json"),
        "sanity_overfit": model_candidates("sanity_overfit.json"),
        "parity_report": model_candidates("parity_report.json"),
        "profile_latest": model_candidates("profile_latest.json"),
        "contract_report": model_candidates("contract_report_latest.json") + [V7_REPORT_PATH / "contract_report_latest.json", V7_REPORT_PATH_LEGACY / "contract_report_latest.json"],
        "parity_1token": model_candidates("parity_1token_latest.json") + [V7_REPORT_PATH / "parity_1token_latest.json", V7_REPORT_PATH_LEGACY / "parity_1token_latest.json"],
        "qk_norm_backward_parity": model_candidates("qk_norm_backward_parity_latest.json") + [V7_REPORT_PATH / "qk_norm_backward_parity_latest.json", V7_REPORT_PATH_LEGACY / "qk_norm_backward_parity_latest.json"],
        "fd_gradients": model_candidates("fd_gradients_latest.json") + [V7_REPORT_PATH / "fd_gradients_latest.json", V7_REPORT_PATH_LEGACY / "fd_gradients_latest.json"],
        "train_parity_epochs_3": model_candidates("train_parity_epochs_3_latest.json") + [V7_REPORT_PATH / "train_parity_epochs_3_latest.json", V7_REPORT_PATH_LEGACY / "train_parity_epochs_3_latest.json"],
        "train_parity_epochs_5": model_candidates("train_parity_epochs_5_latest.json") + [V7_REPORT_PATH / "train_parity_epochs_5_latest.json", V7_REPORT_PATH_LEGACY / "train_parity_epochs_5_latest.json"],
        "train_runtime_parity_realistic": model_candidates("train_runtime_parity_realistic_latest.json") + [V7_REPORT_PATH / "train_runtime_parity_realistic_latest.json", V7_REPORT_PATH_LEGACY / "train_runtime_parity_realistic_latest.json"],
        "train_runtime_parity_stress": model_candidates("train_runtime_parity_stress_latest.json") + [V7_REPORT_PATH / "train_runtime_parity_stress_latest.json", V7_REPORT_PATH_LEGACY / "train_runtime_parity_stress_latest.json"],
        "replay_determinism": model_candidates("replay_determinism_latest.json") + [V7_REPORT_PATH / "replay_determinism_latest.json", V7_REPORT_PATH_LEGACY / "replay_determinism_latest.json"],
        "backprop_stitch_runtime": model_candidates("backprop_stitch_runtime_latest.json") + model_candidates("backprop_stitch_runtime.json") + [V7_REPORT_PATH / "backprop_stitch_runtime_latest.json", V7_REPORT_PATH_LEGACY / "backprop_stitch_runtime_latest.json"],
        "grad_rules": [V7_ROOT / "scripts" / "grad_rules_v7.json"],
        "manifest": model_candidates("weights_manifest.json"),
        "profile_summary": model_candidates("profile_summary.json"),
        "perf_stat_summary": model_candidates("perf_stat_summary.json") + [V7_REPORT_PATH / "perf_stat_summary.json", V7_REPORT_PATH_LEGACY / "perf_stat_summary.json"],
        "flamegraph_manifest": model_candidates("flamegraph_manifest.json") + [V7_REPORT_PATH / "flamegraph_manifest.json", V7_REPORT_PATH_LEGACY / "flamegraph_manifest.json"],
        "cachegrind_summary": model_candidates("cachegrind_summary.json") + [V7_REPORT_PATH / "cachegrind_summary.json", V7_REPORT_PATH_LEGACY / "cachegrind_summary.json"],
        "asan_summary": model_candidates("asan_summary.json") + [V7_REPORT_PATH / "asan_summary.json", V7_REPORT_PATH_LEGACY / "asan_summary.json"],
        "vtune_summary": model_candidates("vtune_summary.json") + [V7_REPORT_PATH / "vtune_summary.json", V7_REPORT_PATH_LEGACY / "vtune_summary.json"],
        "advisor_summary": model_candidates("advisor_summary.json") + [V7_REPORT_PATH / "advisor_summary.json", V7_REPORT_PATH_LEGACY / "advisor_summary.json"],
        "memory_signoff": model_candidates("memory_signoff.json") + [V7_REPORT_PATH / "memory_signoff.json", V7_REPORT_PATH_LEGACY / "memory_signoff.json"],
        "perf_gate_report": model_candidates("perf_gate_report.json") + [V7_REPORT_PATH / "perf_gate_report.json", V7_REPORT_PATH_LEGACY / "perf_gate_report.json"],
        "embedding_dump": model_candidates("embedding_dump.json") + model_candidates("embedding_dump_latest.json"),
        "regression_ledger": (
            model_candidates("regression_ledger.json")
            + [V7_REPORT_PATH / "regression_ledger_latest.json", V7_REPORT_PATH_LEGACY / "regression_ledger_latest.json"]
            + [V7_ROOT / "reports" / "REGRESSION_LEDGER.json"]
        ),
        "kernel_registry": [V7_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"],
    }

    if strict_run_scope:
        # Keep run reports self-consistent: load artifacts only from the explicit
        # run/model roots (plus derived alias roots), not global latest caches.
        strict_exempt = {"grad_rules", "kernel_registry", "regression_ledger"}
        for key, candidates in list(data_files.items()):
            if key in strict_exempt:
                continue
            data_files[key] = [p for p in candidates if _path_under_any_root(p, search_roots)]

    data = {
        "meta": {
            "model": model_name,
            "path": str(ck_build_path),
            "run_dir": str(run_dir) if run_dir is not None else None,
            "project_root": str(PROJECT_ROOT),
            "has_train_runtime": bool(train_runtime_available),
            "strict_run_artifacts": strict_run_scope,
            "warnings": [],
        },
        "files": {}
    }

    loaded = []
    loaded_paths: dict[str, str] = {}
    missing_required = []
    missing_optional = []

    for key, candidates in data_files.items():
        picked_path = None
        picked_payload = None
        fallback_stub_path = None
        fallback_stub_payload = None
        first_error = None

        for path in candidates:
            if not path.exists():
                continue
            try:
                with open(path, "r") as f:
                    payload = json.load(f)
            except Exception as e:
                if first_error is None:
                    first_error = (path, e)
                continue

            # Prefer renderable profile payloads over marker stubs.
            if key == "profile_summary" and _is_profile_summary_stub(payload):
                if fallback_stub_payload is None:
                    fallback_stub_path = path
                    fallback_stub_payload = payload
                continue

            picked_path = path
            picked_payload = payload
            break

        if picked_payload is None and fallback_stub_payload is not None:
            picked_path = fallback_stub_path
            picked_payload = fallback_stub_payload

        if picked_payload is not None:
            data["files"][key] = picked_payload
            loaded.append(key)
            loaded_paths[key] = str(picked_path)
        else:
            if first_error is not None:
                err_path, err = first_error
                print(f"  ! {key} ({err_path}): {err}")
            if key in REQUIRED_FILES:
                missing_required.append(key)
            else:
                missing_optional.append(key)

    analysis_roots = list(search_roots)
    if not strict_run_scope:
        analysis_roots.extend([V7_REPORT_PATH, V7_REPORT_PATH_LEGACY])
    analysis_payload = collect_analysis_checkpoints(analysis_roots)
    if analysis_payload is not None:
        data["files"]["analysis_checkpoints"] = analysis_payload
        loaded.append("analysis_checkpoints")
    else:
        missing_optional.append("analysis_checkpoints")

    # If only runtime parity reports are present, derive dashboard-friendly
    # training_* aliases so the viewer renders without manual file renaming.
    runtime_payload = None
    for key in ("train_runtime_parity_realistic", "train_runtime_parity_stress"):
        candidate = data["files"].get(key)
        if isinstance(candidate, dict):
            runtime_payload = candidate
            break
    if isinstance(runtime_payload, dict):
        files = data["files"]
        if "train_e2e" not in files:
            files["train_e2e"] = runtime_payload
            loaded.append("train_e2e(runtime)")
        if "training_loss_curve" not in files and isinstance(runtime_payload.get("loss_curve"), list):
            files["training_loss_curve"] = {"steps": runtime_payload.get("loss_curve")}
            loaded.append("training_loss_curve(runtime)")
        if "training_parity" not in files and isinstance(runtime_payload.get("parity_steps"), list):
            files["training_parity"] = {"steps": runtime_payload.get("parity_steps")}
            loaded.append("training_parity(runtime)")
        if "training_step_profile" not in files and isinstance(runtime_payload.get("step_profile"), dict):
            files["training_step_profile"] = runtime_payload.get("step_profile")
            loaded.append("training_step_profile(runtime)")
        if "training_grad_norms" not in files and isinstance(runtime_payload.get("grad_norm_series"), dict):
            files["training_grad_norms"] = runtime_payload.get("grad_norm_series")
            loaded.append("training_grad_norms(runtime)")
        if "training_pipeline" not in files:
            active_stage = str(runtime_payload.get("train_mode") or runtime_payload.get("mode") or "pretrain").strip().lower()
            if not active_stage:
                active_stage = "pretrain"
            stages = ["pretrain", "sft", "dpo", "grpo", "ppo"]
            if active_stage not in stages:
                stages = [active_stage] + stages
            active_idx = stages.index(active_stage)
            stage_timeline = []
            for idx, stage in enumerate(stages):
                if idx < active_idx:
                    status = "completed"
                elif idx == active_idx:
                    status = "active"
                else:
                    status = "planned"
                stage_timeline.append({"stage": stage, "order": idx, "status": status, "active": stage == active_stage})
            files["training_pipeline"] = {
                "schema": "ck.training_pipeline.v1",
                "generated_at": runtime_payload.get("generated_at"),
                "active_stage": active_stage,
                "stage_timeline": stage_timeline,
                "backend": runtime_payload.get("backend"),
                "optimizer": {
                    "name": runtime_payload.get("optimizer"),
                    "lr": runtime_payload.get("lr"),
                    "hparams": runtime_payload.get("optimizer_hparams"),
                },
                "execution": {
                    "epochs": runtime_payload.get("epochs"),
                    "steps": runtime_payload.get("steps"),
                    "micro_steps": runtime_payload.get("micro_steps"),
                    "optimizer_steps": runtime_payload.get("optimizer_steps"),
                    "seq_len": runtime_payload.get("seq_len"),
                    "grad_accum": runtime_payload.get("grad_accum"),
                    "tokens_total": runtime_payload.get("total_tokens"),
                    "tokens_per_update": runtime_payload.get("tokens_per_update"),
                },
                "train_dims": runtime_payload.get("train_dims"),
                "data_provenance": runtime_payload.get("data_provenance")
                if isinstance(runtime_payload.get("data_provenance"), list)
                else [],
                "tokenizer_lineage": runtime_payload.get("tokenizer_lineage")
                if isinstance(runtime_payload.get("tokenizer_lineage"), dict)
                else {},
                "sources": {"summary": "runtime_parity", "run_dir": str(run_dir) if run_dir is not None else None},
            }
            loaded.append("training_pipeline(runtime)")

    # Training-only run directories often lack decode/prefill filenames.
    # Add decode/prefill aliases so shared viewer tabs (Memory/Kernel/Stats)
    # still render from training IR/layout/exec-plan artifacts.
    files = data["files"]
    aliased_optional: list[str] = []
    if "ir1_decode" not in files and isinstance(files.get("ir1_train"), dict):
        files["ir1_decode"] = files["ir1_train"]
        loaded.append("ir1_decode(train_alias)")
        aliased_optional.append("ir1_decode")
    if "layout_decode" not in files and isinstance(files.get("layout_train"), dict):
        files["layout_decode"] = files["layout_train"]
        loaded.append("layout_decode(train_alias)")
        aliased_optional.append("layout_decode")
    if "lowered_decode_call" not in files and isinstance(files.get("train_exec_plan"), dict):
        files["lowered_decode_call"] = files["train_exec_plan"]
        loaded.append("lowered_decode_call(train_alias)")
        aliased_optional.append("lowered_decode_call")
    if "lowered_decode" not in files and isinstance(files.get("train_exec_plan"), dict):
        files["lowered_decode"] = files["train_exec_plan"]
        loaded.append("lowered_decode(train_alias)")
        aliased_optional.append("lowered_decode")
    if "ir1_prefill" not in files and isinstance(files.get("ir1_train"), dict):
        files["ir1_prefill"] = files["ir1_train"]
        loaded.append("ir1_prefill(train_alias)")
        aliased_optional.append("ir1_prefill")
    if "layout_prefill" not in files and isinstance(files.get("layout_train"), dict):
        files["layout_prefill"] = files["layout_train"]
        loaded.append("layout_prefill(train_alias)")
        aliased_optional.append("layout_prefill")
    if "lowered_prefill_call" not in files and isinstance(files.get("train_exec_plan"), dict):
        files["lowered_prefill_call"] = files["train_exec_plan"]
        loaded.append("lowered_prefill_call(train_alias)")
        aliased_optional.append("lowered_prefill_call")
    if "lowered_prefill" not in files and isinstance(files.get("train_exec_plan"), dict):
        files["lowered_prefill"] = files["train_exec_plan"]
        loaded.append("lowered_prefill(train_alias)")
        aliased_optional.append("lowered_prefill")
    if aliased_optional:
        missing_optional = [k for k in missing_optional if k not in aliased_optional]

    tokenizer_preview = _build_tokenizer_preview(data["files"], ck_build_path, model_root, run_dir=run_dir)
    if isinstance(tokenizer_preview, dict):
        data["files"]["tokenizer_preview"] = tokenizer_preview
        loaded.append("tokenizer_preview(derived)")

    # Merge stand-alone data-lab artifacts into training_pipeline so the UI can
    # render a single structured view regardless of file-loading path.
    pipeline = files.get("training_pipeline")
    if isinstance(pipeline, dict):
        data_lab = pipeline.get("data_lab")
        if not isinstance(data_lab, dict):
            data_lab = {}
        for key in ("dataset_qc", "dataset_profile", "tokenizer_roundtrip", "post_train_eval", "embedding_dump"):
            payload = files.get(key)
            if isinstance(payload, dict):
                data_lab[key] = payload
        artifacts = data_lab.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
        for src_key, dst_key in (
            ("dataset_qc", "dataset_qc_json"),
            ("dataset_profile", "dataset_profile_json"),
            ("tokenizer_roundtrip", "tokenizer_roundtrip_json"),
            ("post_train_eval", "post_train_eval_json"),
            ("embedding_dump", "embedding_dump_json"),
        ):
            src_path = loaded_paths.get(src_key)
            if isinstance(src_path, str) and src_path:
                artifacts[dst_key] = src_path
        if artifacts:
            data_lab["artifacts"] = artifacts

        if "dataset_path" not in data_lab:
            qc = data_lab.get("dataset_qc")
            if isinstance(qc, dict) and isinstance(qc.get("path"), str):
                data_lab["dataset_path"] = qc.get("path")
        if "dataset_dir" not in data_lab:
            ds_path = data_lab.get("dataset_path")
            if isinstance(ds_path, str) and ds_path:
                data_lab["dataset_dir"] = str(Path(ds_path).parent)
        if "tokenizer_json_path" not in data_lab:
            tok = pipeline.get("tokenizer_lineage")
            if isinstance(tok, dict) and isinstance(tok.get("tokenizer_path"), str):
                data_lab["tokenizer_json_path"] = tok.get("tokenizer_path")
        if isinstance(tokenizer_preview, dict) and tokenizer_preview:
            data_lab["tokenizer_preview"] = tokenizer_preview

        pipeline["data_lab"] = data_lab

    # Enrich artifacts for standalone report portability.
    flame = data["files"].get("flamegraph_manifest")
    if isinstance(flame, dict):
        svg_path = flame.get("svg_path")
        if isinstance(svg_path, str):
            resolved = _resolve_asset_path(svg_path, ck_build_path, model_root)
            if resolved and resolved.suffix.lower() == ".svg":
                try:
                    flame["svg_inline"] = resolved.read_text(errors="ignore")
                    flame["svg_resolved_path"] = str(resolved)
                except Exception as e:
                    data["meta"]["warnings"].append(f"Failed to inline flamegraph SVG: {e}")

    vtune = data["files"].get("vtune_summary")
    if isinstance(vtune, dict):
        path_keys = [
            "result_dir",
            "svg_path",
            "png_path",
            "image_path",
            "report_path",
            "csv_path",
            "hotspots_svg",
            "hotspots_png",
        ]
        for key in path_keys:
            raw = vtune.get(key)
            if not isinstance(raw, str):
                continue
            resolved = _resolve_asset_path(raw, ck_build_path, model_root)
            if not resolved:
                continue
            vtune[f"{key}_resolved"] = str(resolved)
            if resolved.suffix.lower() == ".svg":
                try:
                    vtune[f"{key}_inline"] = resolved.read_text(errors="ignore")
                except Exception:
                    pass
            else:
                data_uri = _encode_image_data_uri(resolved)
                if data_uri:
                    vtune[f"{key}_data_uri"] = data_uri

        artifacts = vtune.get("artifacts")
        if isinstance(artifacts, list):
            enriched = []
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                item2 = dict(item)
                raw = item2.get("path")
                if isinstance(raw, str):
                    resolved = _resolve_asset_path(raw, ck_build_path, model_root)
                    if resolved:
                        item2["resolved_path"] = str(resolved)
                        if resolved.suffix.lower() == ".svg":
                            try:
                                item2["inline"] = resolved.read_text(errors="ignore")
                            except Exception:
                                pass
                        else:
                            data_uri = _encode_image_data_uri(resolved)
                            if data_uri:
                                item2["data_uri"] = data_uri
                enriched.append(item2)
            vtune["artifacts"] = enriched

        analyses = vtune.get("analyses")
        if isinstance(analyses, list):
            enriched_analyses = []
            for entry in analyses:
                if not isinstance(entry, dict):
                    continue
                entry2 = dict(entry)
                for key in ("result_dir", "report_text", "report_csv"):
                    raw = entry2.get(key)
                    if not isinstance(raw, str):
                        continue
                    resolved = _resolve_asset_path(raw, ck_build_path, model_root)
                    if resolved:
                        entry2[f"{key}_resolved"] = str(resolved)
                enriched_analyses.append(entry2)
            vtune["analyses"] = enriched_analyses

    advisor = data["files"].get("advisor_summary")
    if isinstance(advisor, dict):
        for key in ("project_dir", "project_path", "report_path", "csv_path", "html_path"):
            raw = advisor.get(key)
            if not isinstance(raw, str):
                continue
            resolved = _resolve_asset_path(raw, ck_build_path, model_root)
            if resolved:
                advisor[f"{key}_resolved"] = str(resolved)
        artifacts = advisor.get("artifacts")
        if isinstance(artifacts, list):
            enriched = []
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                item2 = dict(item)
                raw = item2.get("path")
                if isinstance(raw, str):
                    resolved = _resolve_asset_path(raw, ck_build_path, model_root)
                    if resolved:
                        item2["resolved_path"] = str(resolved)
                enriched.append(item2)
            advisor["artifacts"] = enriched

    cachegrind = data["files"].get("cachegrind_summary")
    if isinstance(cachegrind, dict):
        for key in ("cachegrind_out", "annotate_path"):
            raw = cachegrind.get(key)
            if not isinstance(raw, str):
                continue
            resolved = _resolve_asset_path(raw, ck_build_path, model_root)
            if resolved:
                cachegrind[f"{key}_resolved"] = str(resolved)

    asan = data["files"].get("asan_summary")
    if isinstance(asan, dict):
        for key in ("verify_report_path", "memory_diagnostic_path"):
            raw = asan.get(key)
            if not isinstance(raw, str):
                continue
            resolved = _resolve_asset_path(raw, ck_build_path, model_root)
            if resolved:
                asan[f"{key}_resolved"] = str(resolved)

    mem = data["files"].get("memory_signoff")
    if isinstance(mem, dict):
        data["files"]["memory_signoff"] = _trim_memory_signoff_payload(mem)

    # Report missing files
    if missing_required:
        print(f"  ! Missing required files: {missing_required}")
        data["meta"]["warnings"].append(f"Missing required: {missing_required}")

    if missing_optional:
        print(f"  - Missing optional files: {missing_optional}")

    if loaded_paths:
        data["meta"]["loaded_paths"] = loaded_paths
    if profile_alias_roots:
        data["meta"]["profile_roots"] = [str(p) for p in profile_alias_roots]
    if strict_run_scope:
        data["meta"]["artifact_roots"] = [str(p) for p in search_roots]

    print(f"  Loaded {len(loaded)} files")
    return data


def generate_html_report(
    ck_build_path: Path,
    output_path: Path = None,
    run_dir: Path | None = None,
    strict_run_artifacts: bool = False,
):
    """Generate standalone HTML report."""
    from datetime import datetime

    model_name = ck_build_path.parent.name if ck_build_path.name in {"ck_build", ".ck_build"} else ck_build_path.name
    print(f"Generating report for: {model_name}")

    # Load data
    data = load_model_data(
        ck_build_path,
        run_dir=run_dir,
        strict_run_artifacts=strict_run_artifacts,
    )
    data["meta"]["generated_at"] = datetime.now().isoformat()
    data["meta"]["engine_version"] = "v7"

    # Read visualizer template
    if not VISUALIZER.exists():
        raise FileNotFoundError(f"Visualizer not found: {VISUALIZER}")

    with open(VISUALIZER, "r") as f:
        html = f.read()

    # Embed data safely for inline <script> context.
    # Prevent accidental </script> termination from embedded payloads (e.g., inline SVG/JS).
    data_json = json.dumps(data)
    data_json = data_json.replace("</", "<\\/")
    data_json = data_json.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

    # Embed data
    data_js = (
        f"window.EMBEDDED_IR_DATA = {data_json};"
        "window.dispatchEvent(new Event('ckEmbeddedDataLoaded'));"
        "if (window.bootstrapFromEmbeddedData) { window.bootstrapFromEmbeddedData(); }"
    )
    html = html.replace('</body>', f'<script>{data_js}</script></body>')

    # Update title
    html = html.replace(
        '<title>IR Visualizer | C-Kernel-Engine</title>',
        f'<title>IR Visualizer | {model_name} | C-Kernel-Engine</title>'
    )

    # Write output
    if output_path is None:
        output_path = ck_build_path / "ir_report.html"

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="IR Visualizer Launcher for C-Kernel-Engine v7",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 version/v7/tools/open_ir_visualizer.py              # Open visualizer
    python3 version/v7/tools/open_ir_visualizer.py --list       # List available models
    python3 version/v7/tools/open_ir_visualizer.py gemma3       # Generate and open report
    python3 version/v7/tools/open_ir_visualizer.py --generate gemma3  # Generate full report (profile+probes)
    python3 version/v7/tools/open_ir_visualizer.py --generate gemma3 --interactive
    python3 version/v7/tools/open_ir_visualizer.py --generate gemma3 --html-only  # Generate HTML only
    python3 version/v7/tools/open_ir_visualizer.py --generate gemma3 --with-profile
    python3 version/v7/tools/open_ir_visualizer.py --generate gemma3 --with-probes --force-compile
    python3 version/v7/tools/open_ir_visualizer.py --generate Qwen--Qwen3-0.6B --with-probes --run-model Qwen/Qwen3-0.6B --weight-dtype float32
    python3 version/v7/tools/open_ir_visualizer.py --generate gemma3 --with-probes --perf-runtime cli
    python3 version/v7/tools/open_ir_visualizer.py --generate gemma3 --no-vtune
    python3 version/v7/tools/open_ir_visualizer.py --generate gemma3 --with-probes --run-model hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf --chat-template none
        """
    )

    parser.add_argument(
        "model",
        nargs="?",
        help="Model name (e.g., gemma3) or path to ck_build directory"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models in cache"
    )
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Generate HTML report without opening browser"
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Generate HTML only (skip profile/probe capture)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for generated HTML (default: ck_build/ir_report.html)"
    )
    parser.add_argument(
        "--run",
        type=Path,
        help="Run directory produced by cks-v7-run (single source of training/profile artifacts)"
    )
    parser.add_argument(
        "--run-model",
        type=str,
        help="Explicit model input for runtime probes/profile (hf://..., .gguf, or source checkpoint dir)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Prompt for runtime/probe options after model selection"
    )
    parser.add_argument(
        "--with-profile", "--profile",
        dest="with_profile",
        action="store_true",
        help="Run ck_run_v7.py --profile before generating report"
    )
    parser.add_argument(
        "--with-probes", "--probes",
        dest="with_probes",
        action="store_true",
        help="Run memory sign-off + perf probes (perf stat/flamegraph + perf budget gate) before report generation"
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=1024,
        help="Context length for --with-profile/--with-probes runs (default: 1024)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Max tokens for --with-profile/--with-probes runs (default: 16)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Prompt text for --with-profile/--with-probes runs"
    )
    parser.add_argument(
        "--force-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force compile runtime artifacts for --with-profile/--with-probes (default: enabled, use --no-force-compile to reuse existing binaries)"
    )
    parser.add_argument(
        "--chat-template",
        choices=["auto", "none", "qwen", "gemma"],
        default="auto",
        help="Chat template mode for runtime probe/profile commands"
    )
    parser.add_argument(
        "--perf-runtime",
        choices=["cli", "python"],
        default="cli",
        help="Runtime used by --with-probes perf targets (default: cli for pure C runtime)"
    )
    parser.add_argument(
        "--weight-dtype",
        choices=["float32", "bf16", "q4_0", "q4_1", "q4_k", "q4_k_m", "q5_0", "q5_1", "q6_k", "q8_0"],
        default=None,
        help="Weight dtype override forwarded to ck_run_v7.py for profile/probe prep (e.g., float32 for training-focused builds)"
    )
    parser.add_argument(
        "--vtune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture VTune artifacts in probe flow (default: enabled, use --no-vtune to skip)"
    )
    parser.add_argument(
        "--cachegrind",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture cachegrind artifacts for run-dir native train probes when available (default: enabled)"
    )
    parser.add_argument(
        "--asan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture ASan verification artifacts for run-dir native train probes when available (default: enabled)"
    )
    parser.add_argument(
        "--advisor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture Advisor roofline artifacts for run-dir native train probes when available (default: enabled, use --no-advisor to skip)"
    )
    parser.add_argument(
        "--strict-run-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="When using --run, load artifacts only from run/model roots (default: enabled for --run)"
    )

    args = parser.parse_args()

    if args.strict_run_artifacts is None:
        args.strict_run_artifacts = bool(args.run)

    # Operator-first default:
    # `--generate <model>` should produce a fully populated report unless explicitly disabled.
    if args.html_only:
        if args.with_profile or args.with_probes:
            print("Note: --html-only ignores --with-profile/--with-probes and only renders existing artifacts.")
        args.with_profile = False
        args.with_probes = False
    elif args.generate and (args.model or args.run) and not args.with_profile and not args.with_probes:
        if args.run and not args.model:
            print("Defaulting to artifact-only generation for --run (use --with-probes/--with-profile for active capture).")
        else:
            args.with_probes = True
            if not args.force_compile:
                args.force_compile = True
            print("Defaulting to full generation: enabling --with-probes --force-compile (use --html-only for quick HTML-only output).")

    if args.list:
        print("Available models in cache:")
        for m in list_available_models():
            status = "OK" if m["has_data"] else "no data"
            print(f"  - {m['name']} ({status})")
        return

    if (args.with_profile or args.with_probes) and not (args.model or args.run):
        parser.error("--with-profile/--with-probes require --run or a model argument")

    run_dir: Path | None = None

    if args.model or args.run:

        if args.run:
            try:
                ck_build, run_dir = resolve_run_target(args.run)
                model_root = run_dir
            except ValueError as e:
                print(f"Error: {e}")
                return
            if args.model:
                print("Note: --run provided; positional model is ignored for artifact loading.")
        else:
            try:
                ck_build, model_root = resolve_model_target(args.model)
            except ValueError as e:
                print(f"Error: {e}")
                print("\nAvailable models:")
                for m in list_available_models():
                    print(f"  - {m['name']}")
                return

        maybe_interactive_configure(args, model_root)

        if args.with_profile or args.with_probes:
            run_model_input = args.run_model
            run_model_compiled_only = False
            if not run_model_input:
                # Prefer source/HF inputs so probe flow can always prep via ck_run_v7.py first.
                if has_local_runnable_source(model_root):
                    run_model_input = resolve_local_runnable_input(model_root)
                    if run_model_input != str(model_root):
                        print(f"Using local runnable file input: {run_model_input}")
                else:
                    inferred = infer_run_model_input(model_root, args.weight_dtype)
                    if inferred:
                        run_model_input = inferred
                        print(f"Inferred runtime model input: {run_model_input}")
                    elif args.perf_runtime == "cli" and has_local_compiled_runtime(model_root):
                        run_model_input = str(model_root)
                        run_model_compiled_only = True
                        print(
                            f"Using local compiled runtime for CLI probes (no source detected): "
                            f"{run_model_input}"
                        )
                    else:
                        raise SystemExit(
                            "Error: report model directory is not a runnable source checkpoint.\n"
                            "Use --run-model with hf://... (or a .gguf/source checkpoint path)."
                        )
            else:
                run_model_path = Path(run_model_input)
                if not run_model_input.startswith("hf://"):
                    # If caller explicitly provided a local path-like run-model, fail fast when it doesn't exist.
                    looks_like_local = (
                        "/" in run_model_input
                        or run_model_input.startswith(".")
                        or run_model_path.suffix.lower() in {".gguf", ".json", ".safetensors"}
                    )
                    if looks_like_local and not run_model_path.exists():
                        raise SystemExit(
                            f"Error: --run-model local path not found: {run_model_input}\n"
                            "Tip: use a single-line absolute path, or quote it if needed."
                        )
                if run_model_path.exists() and run_model_path.is_dir() and not has_local_runnable_source(run_model_path):
                    if args.perf_runtime == "cli" and has_local_compiled_runtime(run_model_path):
                        run_model_compiled_only = True
                    else:
                        raise SystemExit(
                            "Error: --run-model points to a local directory without .gguf/.safetensors files.\n"
                            "Use an hf://... GGUF source, a .gguf file path, or a source checkpoint directory."
                        )
            run_model_input = maybe_localize_hf_gguf(run_model_input)

            if args.weight_dtype and run_model_input.lower().endswith(".gguf"):
                fallback_source = infer_run_model_input(model_root, args.weight_dtype)
                if fallback_source and not fallback_source.lower().endswith(".gguf"):
                    print(
                        "Switching runtime model input to source checkpoint "
                        f"for --weight-dtype={args.weight_dtype}: {fallback_source}"
                    )
                    run_model_input = fallback_source
                else:
                    raise SystemExit(
                        "Error: --weight-dtype is not compatible with GGUF inputs.\n"
                        "Use --run-model <HF model id or local safetensors dir> when requesting float32/bf16 conversion."
                    )

            effective_chat_template = args.chat_template
            if effective_chat_template == "auto":
                inferred_chat = infer_chat_template(run_model_input, model_root)
                if inferred_chat != "auto":
                    effective_chat_template = inferred_chat
                    print(f"Inferred chat template for probes/profile: {effective_chat_template}")

            chat_args: list[str] = []
            if effective_chat_template and effective_chat_template != "auto":
                chat_args = ["--chat-template", effective_chat_template]

            make_overrides = [
                f"V7_MODEL={run_model_input}",
                f"V7_PERF_RUNTIME={args.perf_runtime}",
                f"V7_WITH_VTUNE={1 if args.vtune else 0}",
                f"V7_FORCE_COMPILE={1 if args.force_compile else 0}",
                f"V7_PREP_WITH_PYTHON={0 if run_model_compiled_only else 1}",
            ]
            if args.weight_dtype:
                make_overrides.append(f"V7_WEIGHT_DTYPE={args.weight_dtype}")
            if effective_chat_template and effective_chat_template != "auto":
                make_overrides.append(f"V7_CHAT_TEMPLATE={effective_chat_template}")
            if effective_chat_template == "none":
                make_overrides.append("V7_CLI_ARGS=--no-chat-template")

            profile_cmd = [
                sys.executable,
                str(CK_RUN_SCRIPT),
                "run",
                run_model_input,
                "--context-len",
                str(args.context_len),
                "--max-tokens",
                str(args.max_tokens),
                "--prompt",
                args.prompt,
                "--profile",
            ]
            profile_cmd.extend(chat_args)
            if args.weight_dtype:
                profile_cmd.extend(["--weight-dtype", args.weight_dtype])
            if args.force_compile:
                profile_cmd.append("--force-compile")

            if args.with_profile:
                if args.perf_runtime == "cli":
                    print("Preparing runtime via ck_run_v7.py...")
                    try_run_cmd(
                        "runtime prep",
                        ["make", "--no-print-directory", "profile-v7-prepare-runtime", *make_overrides],
                        PROJECT_ROOT,
                    )
                    print("Running CLI profile capture...")
                    try_run_cmd(
                        "cli profile capture",
                        ["make", "--no-print-directory", "profile-v7-decode", *make_overrides],
                        PROJECT_ROOT,
                    )
                else:
                    print("Running profile capture...")
                    try_run_cmd("profile capture", profile_cmd, PROJECT_ROOT)

            if args.with_probes:
                report_model_dir = model_root
                has_layout = (report_model_dir / "layout_decode.json").exists()
                has_lowered = (report_model_dir / "lowered_decode_call.json").exists() or (report_model_dir / "lowered_decode.json").exists()
                need_probe_prep = args.force_compile or not (has_layout and has_lowered)
                if need_probe_prep and run_model_compiled_only:
                    print(
                        "Skipping probe prep via ck_run_v7.py: runtime-only input has no source checkpoint "
                        "(use --run-model hf://... or local .gguf/.safetensors to enable prep)."
                    )
                elif need_probe_prep:
                    print("Running probe prep (generate-only)...")
                    prep_cmd = [
                        sys.executable,
                        str(CK_RUN_SCRIPT),
                        "run",
                        run_model_input,
                        "--generate-only",
                        "--profile",
                        "--context-len",
                        str(max(args.context_len, 128)),
                        "--max-tokens",
                        "1",
                        "--prompt",
                        "Hello",
                    ]
                    prep_cmd.extend(chat_args)
                    if args.weight_dtype:
                        prep_cmd.extend(["--weight-dtype", args.weight_dtype])
                    if args.force_compile:
                        prep_cmd.append("--force-compile")
                    try_run_cmd("probe prep (generate-only)", prep_cmd, PROJECT_ROOT)
                else:
                    print("Skipping probe prep (layout/lowered artifacts already present).")

                if not args.with_profile:
                    # Ensure profile_summary.json exists for report profile charts.
                    if args.perf_runtime == "cli":
                        print("Preparing runtime via ck_run_v7.py (required for probes)...")
                        try_run_cmd(
                            "runtime prep (required for probes)",
                            ["make", "--no-print-directory", "profile-v7-prepare-runtime", *make_overrides],
                            PROJECT_ROOT,
                        )
                        print("Running CLI profile capture (required for probes)...")
                        try_run_cmd(
                            "cli profile capture (required for probes)",
                            ["make", "--no-print-directory", "profile-v7-decode", *make_overrides],
                            PROJECT_ROOT,
                        )
                    else:
                        print("Running profile capture (required for probes)...")
                        try_run_cmd("profile capture (required for probes)", profile_cmd, PROJECT_ROOT)

                print("Running memory sign-off...")
                try_run_cmd(
                    "memory sign-off",
                    [
                        sys.executable,
                        str(MEMORY_SIGNOFF_SCRIPT),
                        "--model-dir",
                        str(report_model_dir),
                    ],
                    PROJECT_ROOT,
                )

                print("Running perf probes + budget gate...")
                perf_available = shutil.which("perf") is not None
                vtune_available = shutil.which("vtune") is not None
                advisor_available = shutil.which("advisor") is not None
                valgrind_available = shutil.which("valgrind") is not None
                flamegraph_ok = (
                    (PROJECT_ROOT / "FlameGraph" / "stackcollapse-perf.pl").exists()
                    and (PROJECT_ROOT / "FlameGraph" / "flamegraph.pl").exists()
                )
                if not perf_available:
                    print("Skipping perf probes (perf not installed).")
                elif not flamegraph_ok:
                    print("Skipping perf probes (FlameGraph tools missing).")
                else:
                    try_run_cmd(
                        "perf probe gate",
                        ["make", "--no-print-directory", "v7-perf-gate", *make_overrides],
                        PROJECT_ROOT,
                    )

                expect_advisor = False
                expect_cachegrind = False
                expect_asan = False
                if run_dir is not None and has_train_runtime_artifacts(run_dir):
                    ck_cli = PROJECT_ROOT / "build" / "ck-cli-v7"
                    thread_hint = os.environ.get("CK_NUM_THREADS", "8")
                    token_file = ensure_train_token_file(run_dir)
                    try_run_cmd("build ck-cli-v7 (native profile probes)", ["make", "--no-print-directory", "ck-cli-v7"], PROJECT_ROOT)

                    native_probe_prefix = [
                        str(ck_cli),
                        "profile",
                        "--run",
                        str(run_dir),
                        "--train-token-file",
                        str(token_file),
                        "--train-epochs",
                        "1",
                        "--train-seq-len",
                        "8",
                        "--train-total-tokens",
                        "2048",
                        "--train-grad-accum",
                        "8",
                        "--threads",
                        thread_hint,
                    ]

                    if args.vtune:
                        if vtune_available:
                            print("Running native train VTune probe (ck-cli-v7 profile)...")
                            try_run_cmd(
                                "native train VTune probe",
                                [*native_probe_prefix, "--tool", "vtune"],
                                PROJECT_ROOT,
                                extra_env={"CK_NUM_THREADS": thread_hint},
                            )
                        else:
                            print("Skipping native train VTune probe (vtune not installed).")

                    if args.cachegrind:
                        if valgrind_available:
                            expect_cachegrind = True
                            print("Running native train cachegrind probe (ck-cli-v7 profile)...")
                            try_run_cmd(
                                "native train cachegrind probe",
                                [*native_probe_prefix, "--tool", "cachegrind"],
                                PROJECT_ROOT,
                                extra_env={"CK_NUM_THREADS": thread_hint},
                            )
                        else:
                            print("Skipping native train cachegrind probe (valgrind not installed).")

                    if args.asan:
                        expect_asan = True
                        print("Running native train ASan verification probe (ck-cli-v7 profile)...")
                        try_run_cmd(
                            "native train ASan probe",
                            [*native_probe_prefix, "--tool", "asan"],
                            PROJECT_ROOT,
                            extra_env={"CK_NUM_THREADS": thread_hint},
                        )

                    if args.advisor:
                        if advisor_available:
                            expect_advisor = True
                            print("Running native train Advisor probe (ck-cli-v7 profile)...")
                            try_run_cmd(
                                "native train Advisor probe",
                                [*native_probe_prefix, "--tool", "advisor"],
                                PROJECT_ROOT,
                                extra_env={"CK_NUM_THREADS": thread_hint},
                            )
                        else:
                            print("Skipping native train Advisor probe (advisor not installed).")
                elif args.advisor and run_dir is not None:
                    print("Skipping native train Advisor probe (run directory does not contain compiled train runtime artifacts).")
                if run_dir is None and (args.advisor or args.cachegrind or args.asan):
                    print("Skipping native train probes (no --run directory).")

                runtime_model_dir = detect_model_dir_from_input(run_model_input)
                if runtime_model_dir and runtime_model_dir.exists():
                    copy_artifacts_if_needed(runtime_model_dir, report_model_dir)
                elif runtime_model_dir and not runtime_model_dir.exists():
                    print(f"Warning: runtime artifact directory not found, skip copy: {runtime_model_dir}")

                missing = validate_artifact_set(
                    model_root=report_model_dir,
                    ck_build=ck_build,
                    expect_perf=perf_available and flamegraph_ok,
                    expect_vtune=bool(args.vtune and vtune_available),
                    expect_advisor=expect_advisor,
                    expect_cachegrind=expect_cachegrind,
                    expect_asan=expect_asan,
                )
                if missing:
                    print(f"Warning: missing expected artifacts after probe run: {missing}")
                    print("Tip: rerun with explicit model source or disable optional stages (e.g., --no-vtune/--no-advisor).")

        # Generate report
        output = args.output or ((run_dir or model_root) / "ir_report.html")
        report_path = generate_html_report(
            ck_build,
            output,
            run_dir=run_dir,
            strict_run_artifacts=bool(args.strict_run_artifacts),
        )
        print(f"\nGenerated: {report_path}")

        if not args.generate:
            webbrowser.open(f"file://{report_path}")
    else:
        # Open visualizer
        if VISUALIZER.exists():
            webbrowser.open(f"file://{VISUALIZER}")
        else:
            print(f"Error: Visualizer not found: {VISUALIZER}")


if __name__ == "__main__":
    main()
