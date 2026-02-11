#!/usr/bin/env python3
"""
IR Visualizer Launcher for C-Kernel-Engine v6.6

Usage:
    python version/v6.6/tools/open_ir_visualizer.py              # Open visualizer
    python version/v6.6/tools/open_ir_visualizer.py --list       # List available models
    python version/v6.6/tools/open_ir_visualizer.py <model>      # Generate and open report
    python version/v6.6/tools/open_ir_visualizer.py --generate <model>  # Generate full report (profile+probes)
    python version/v6.6/tools/open_ir_visualizer.py --generate <model> --html-only  # Generate HTML only
    python version/v6.6/tools/open_ir_visualizer.py --generate <model> --with-profile
    python version/v6.6/tools/open_ir_visualizer.py --generate <model> --with-probes
    python version/v6.6/tools/open_ir_visualizer.py --generate <model> --with-probes --perf-runtime cli
    python version/v6.6/tools/open_ir_visualizer.py --generate <model> --no-vtune
    python version/v6.6/tools/open_ir_visualizer.py --generate <model> --with-probes --run-model hf://... --chat-template none
"""
import os
import sys
import json
import base64
import shutil
import webbrowser
import argparse
import subprocess
from pathlib import Path

# Path construction:
# Script is at: version/v6.6/tools/open_ir_visualizer.py
SCRIPT_DIR = Path(__file__).parent              # .../version/v6.6/tools
V66_ROOT = SCRIPT_DIR.parent                    # .../version/v6.6
PROJECT_ROOT = V66_ROOT.parent.parent           # .../Workspace/C-Kernel-Engine

sys.path.insert(0, str(V66_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_PATH = Path.home() / ".cache" / "ck-engine-v6.6" / "models"
VISUALIZER = SCRIPT_DIR / "ir_visualizer.html"
CK_RUN_SCRIPT = V66_ROOT / "scripts" / "ck_run_v6_6.py"
MEMORY_SIGNOFF_SCRIPT = V66_ROOT / "scripts" / "memory_signoff_v6_6.py"


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
        if candidate.name == "ck_build":
            ck_build = candidate
            model_root = candidate.parent
        else:
            ck_build = candidate / "ck_build" if (candidate / "ck_build").exists() else candidate
            model_root = candidate
        return ck_build, model_root

    ck_build = CACHE_PATH / model_arg / "ck_build"
    if ck_build.exists():
        return ck_build, CACHE_PATH / model_arg

    model_dir = CACHE_PATH / model_arg
    if model_dir.exists():
        return model_dir, model_dir

    raise ValueError(f"Model not found: {model_arg}")


def has_local_runnable_source(path: Path) -> bool:
    """Whether path can be used as ck_run_v6_6 local input source."""
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


def infer_run_model_input(model_root: Path) -> str | None:
    """
    Best-effort mapping from cache model folder names to hf:// model inputs.
    Keeps launcher ergonomic for common v6.6 model aliases.
    """
    name = model_root.name.lower()
    if "gemma-3-270m-it" in name:
        return "hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf"
    if "qwen2-0.5b-instruct" in name or "qwen2-0_5b-instruct" in name:
        return "hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf"
    if "qwen3-0.6b" in name:
        return "hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
    return None


def infer_chat_template(run_model_input: str, model_root: Path) -> str:
    probe = f"{run_model_input} {model_root.name}".lower()
    # Gemma GGUF chat templates are often not compatible with our runtime path;
    # prefer raw prompt mode unless the operator overrides explicitly.
    if "gemma" in probe:
        return "none"
    return "auto"


def detect_model_dir_from_input(model_input: str) -> Path | None:
    try:
        from ck_run_v6_6 import CACHE_DIR, detect_input_type  # type: ignore
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
        if (local / "libmodel.so").exists() and (local / "weights.bump").exists():
            return local
        if (local / ".ck_build").exists():
            return local / ".ck_build"
        return local
    if input_type == "local_config":
        return Path(info["path"]).parent / ".ck_build"
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
        "vtune_summary.json",
        "memory_signoff.json",
        "perf_gate_report.json",
    ]
    copied = 0
    dst_model_dir.mkdir(parents=True, exist_ok=True)
    for name in artifact_names:
        src = src_model_dir / name
        if not src.exists():
            continue
        shutil.copy2(src, dst_model_dir / name)
        copied += 1
    if copied:
        print(f"Copied {copied} artifact(s) from {src_model_dir} -> {dst_model_dir}")


def has_model_artifact(model_root: Path, ck_build: Path, name: str) -> bool:
    return (model_root / name).exists() or (ck_build / name).exists()


def validate_artifact_set(
    model_root: Path,
    ck_build: Path,
    expect_perf: bool,
    expect_vtune: bool,
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
    return missing


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
    checks = out.get("checks")
    if not isinstance(checks, dict):
        return out
    trimmed_checks = {}
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
        trimmed_checks[key] = c2
    out["checks"] = trimmed_checks
    return out


def list_available_models():
    """List all models in cache."""
    if not CACHE_PATH.exists():
        return []

    models = []
    for model_dir in CACHE_PATH.iterdir():
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


def load_model_data(ck_build_path: Path) -> dict:
    """Load all IR data for a model."""
    model_name = ck_build_path.parent.name if ck_build_path.name == "ck_build" else ck_build_path.name
    model_root = ck_build_path.parent if ck_build_path.name == "ck_build" else ck_build_path

    # Define required vs optional files
    REQUIRED_FILES = [
        "ir1_decode",
        "layout_decode",
        "lowered_decode_call",
    ]
    OPTIONAL_FILES = [
        "ir1_prefill",
        "layout_prefill",
        "lowered_prefill_call",
        "lowered_decode",
        "lowered_prefill",
        "manifest",
        "kernel_registry",
        "profile_summary",
        "perf_stat_summary",
        "flamegraph_manifest",
        "vtune_summary",
        "memory_signoff",
        "perf_gate_report",
    ]

    def model_candidates(name: str) -> list[Path]:
        # Prefer ck_build, then model root (some runs write directly to model dir).
        return [ck_build_path / name, model_root / name]

    data_files = {
        "ir1_decode": model_candidates("ir1_decode.json"),
        "ir1_prefill": model_candidates("ir1_prefill.json"),
        "layout_decode": model_candidates("layout_decode.json"),
        "layout_prefill": model_candidates("layout_prefill.json"),
        "lowered_decode": model_candidates("lowered_decode.json"),
        "lowered_prefill": model_candidates("lowered_prefill.json"),
        "lowered_decode_call": model_candidates("lowered_decode_call.json") + model_candidates("lowered_decode.json"),
        "lowered_prefill_call": model_candidates("lowered_prefill_call.json") + model_candidates("lowered_prefill.json"),
        "manifest": model_candidates("weights_manifest.json"),
        "profile_summary": model_candidates("profile_summary.json"),
        "perf_stat_summary": model_candidates("perf_stat_summary.json"),
        "flamegraph_manifest": model_candidates("flamegraph_manifest.json"),
        "vtune_summary": model_candidates("vtune_summary.json"),
        "memory_signoff": model_candidates("memory_signoff.json"),
        "perf_gate_report": model_candidates("perf_gate_report.json"),
        "kernel_registry": [V66_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"],
    }

    data = {
        "meta": {
            "model": model_name,
            "path": str(ck_build_path),
            "warnings": [],
        },
        "files": {}
    }

    loaded = []
    missing_required = []
    missing_optional = []

    for key, candidates in data_files.items():
        picked = None
        for path in candidates:
            if path.exists():
                picked = path
                break
        if picked is not None:
            try:
                with open(picked, "r") as f:
                    data["files"][key] = json.load(f)
                loaded.append(key)
            except Exception as e:
                print(f"  ! {key}: {e}")
        else:
            if key in REQUIRED_FILES:
                missing_required.append(key)
            else:
                missing_optional.append(key)

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
            "svg_path",
            "png_path",
            "image_path",
            "report_path",
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

    mem = data["files"].get("memory_signoff")
    if isinstance(mem, dict):
        data["files"]["memory_signoff"] = _trim_memory_signoff_payload(mem)

    # Report missing files
    if missing_required:
        print(f"  ! Missing required files: {missing_required}")
        data["meta"]["warnings"].append(f"Missing required: {missing_required}")

    if missing_optional:
        print(f"  - Missing optional files: {missing_optional}")

    print(f"  Loaded {len(loaded)} files")
    return data


def generate_html_report(ck_build_path: Path, output_path: Path = None):
    """Generate standalone HTML report."""
    from datetime import datetime

    model_name = ck_build_path.parent.name if ck_build_path.name == "ck_build" else ck_build_path.name
    print(f"Generating report for: {model_name}")

    # Load data
    data = load_model_data(ck_build_path)
    data["meta"]["generated_at"] = datetime.now().isoformat()
    data["meta"]["engine_version"] = "v6.6"

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
        description="IR Visualizer Launcher for C-Kernel-Engine v6.6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python version/v6.6/tools/open_ir_visualizer.py              # Open visualizer
    python version/v6.6/tools/open_ir_visualizer.py --list       # List available models
    python version/v6.6/tools/open_ir_visualizer.py gemma3       # Generate and open report
    python version/v6.6/tools/open_ir_visualizer.py --generate gemma3  # Generate full report (profile+probes)
    python version/v6.6/tools/open_ir_visualizer.py --generate gemma3 --html-only  # Generate HTML only
    python version/v6.6/tools/open_ir_visualizer.py --generate gemma3 --with-profile
    python version/v6.6/tools/open_ir_visualizer.py --generate gemma3 --with-probes --force-compile
    python version/v6.6/tools/open_ir_visualizer.py --generate gemma3 --with-probes --perf-runtime cli
    python version/v6.6/tools/open_ir_visualizer.py --generate gemma3 --no-vtune
    python version/v6.6/tools/open_ir_visualizer.py --generate gemma3 --with-probes --run-model hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf --chat-template none
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
        "--run-model",
        type=str,
        help="Explicit model input for runtime probes/profile (hf://..., .gguf, or source checkpoint dir)"
    )
    parser.add_argument(
        "--with-profile", "--profile",
        dest="with_profile",
        action="store_true",
        help="Run ck_run_v6_6.py --profile before generating report"
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
        "--vtune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture VTune artifacts in probe flow (default: enabled, use --no-vtune to skip)"
    )

    args = parser.parse_args()

    # Operator-first default:
    # `--generate <model>` should produce a fully populated report unless explicitly disabled.
    if args.html_only:
        args.with_profile = False
        args.with_probes = False
    elif args.generate and args.model and not args.with_profile and not args.with_probes:
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

    if (args.with_profile or args.with_probes) and not args.model:
        parser.error("--with-profile/--with-probes require a model argument")

    if args.model:
        try:
            ck_build, model_root = resolve_model_target(args.model)
        except ValueError as e:
            print(f"Error: {e}")
            print("\nAvailable models:")
            for m in list_available_models():
                print(f"  - {m['name']}")
            return

        if args.with_profile or args.with_probes:
            run_model_input = args.run_model
            run_model_compiled_only = False
            if not run_model_input:
                # Prefer source/HF inputs so probe flow can always prep via ck_run_v6_6.py first.
                if has_local_runnable_source(model_root):
                    run_model_input = resolve_local_runnable_input(model_root)
                    if run_model_input != str(model_root):
                        print(f"Using local runnable file input: {run_model_input}")
                else:
                    inferred = infer_run_model_input(model_root)
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
                f"V66_MODEL={run_model_input}",
                f"V66_PERF_RUNTIME={args.perf_runtime}",
                f"V66_WITH_VTUNE={1 if args.vtune else 0}",
                f"V66_FORCE_COMPILE={1 if args.force_compile else 0}",
                f"V66_PREP_WITH_PYTHON={0 if run_model_compiled_only else 1}",
            ]
            if effective_chat_template and effective_chat_template != "auto":
                make_overrides.append(f"V66_CHAT_TEMPLATE={effective_chat_template}")
            if effective_chat_template == "none":
                make_overrides.append("V66_CLI_ARGS=--no-chat-template")

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
            if args.force_compile:
                profile_cmd.append("--force-compile")

            if args.with_profile:
                if args.perf_runtime == "cli":
                    print("Preparing runtime via ck_run_v6_6.py...")
                    try_run_cmd(
                        "runtime prep",
                        ["make", "--no-print-directory", "profile-v6-prepare-runtime", *make_overrides],
                        PROJECT_ROOT,
                    )
                    print("Running CLI profile capture...")
                    try_run_cmd(
                        "cli profile capture",
                        ["make", "--no-print-directory", "profile-v6-decode", *make_overrides],
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
                        "Skipping probe prep via ck_run_v6_6.py: runtime-only input has no source checkpoint "
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
                    if args.force_compile:
                        prep_cmd.append("--force-compile")
                    try_run_cmd("probe prep (generate-only)", prep_cmd, PROJECT_ROOT)
                else:
                    print("Skipping probe prep (layout/lowered artifacts already present).")

                if not args.with_profile:
                    # Ensure profile_summary.json exists for report profile charts.
                    if args.perf_runtime == "cli":
                        print("Preparing runtime via ck_run_v6_6.py (required for probes)...")
                        try_run_cmd(
                            "runtime prep (required for probes)",
                            ["make", "--no-print-directory", "profile-v6-prepare-runtime", *make_overrides],
                            PROJECT_ROOT,
                        )
                        print("Running CLI profile capture (required for probes)...")
                        try_run_cmd(
                            "cli profile capture (required for probes)",
                            ["make", "--no-print-directory", "profile-v6-decode", *make_overrides],
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
                        ["make", "--no-print-directory", "v6.6-perf-gate", *make_overrides],
                        PROJECT_ROOT,
                    )

                runtime_model_dir = detect_model_dir_from_input(run_model_input)
                if runtime_model_dir and runtime_model_dir.exists():
                    copy_artifacts_if_needed(runtime_model_dir, report_model_dir)
                elif runtime_model_dir and not runtime_model_dir.exists():
                    print(f"Warning: runtime artifact directory not found, skip copy: {runtime_model_dir}")

                vtune_available = shutil.which("vtune") is not None
                missing = validate_artifact_set(
                    model_root=report_model_dir,
                    ck_build=ck_build,
                    expect_perf=perf_available and flamegraph_ok,
                    expect_vtune=bool(args.vtune and vtune_available),
                )
                if missing:
                    print(f"Warning: missing expected artifacts after probe run: {missing}")
                    print("Tip: rerun with explicit model source or disable optional stages (e.g., --no-vtune).")

        # Generate report
        output = args.output or ck_build / "ir_report.html"
        report_path = generate_html_report(ck_build, output)
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
