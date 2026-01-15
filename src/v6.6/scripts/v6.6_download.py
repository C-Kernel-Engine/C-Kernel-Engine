#!/usr/bin/env python3
"""
v6.6_download.py - Download models from HuggingFace

Usage:
    python v6.6_download.py --repo Qwen/Qwen2-0.5B-Instruct --output ~/.cache/ck-engine-v6.6/models/qwen2-0.5b
    python v6.6_download.py --preset qwen2-0.5b

Supports:
    - GGUF models (direct download)
    - Safetensors models (convert to BUMP)
    - HF .bin models (convert to BUMP)
"""

import argparse
import os
import subprocess
import sys
import urllib.request
import json
from pathlib import Path
from typing import Optional, Dict, List

# Cache directory
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/ck-engine-v6.6")

# Model presets
PRESETS = {
    "qwen2-0.5b": {
        "repo": "Qwen/Qwen2-0.5B-Instruct",
        "files": ["qwen2-0_5b-instruct-q4_k_m.gguf"],
        "convert": False,  # GGUF, no conversion needed
    },
    "qwen2-1.5b": {
        "repo": "Qwen/Qwen2-1.5B-Instruct",
        "files": ["qwen2-1_5b-instruct-q4_k_m.gguf"],
        "convert": False,
    },
    "smollm-135": {
        "repo": "HuggingFaceTB/SmolLM-135M",
        "files": ["smollm-135m-ggml.bin"],
        "convert": True,
    },
    "smollm-360": {
        "repo": "HuggingFaceTB/SmolLM-360M",
        "files": ["smollm-360m-ggml.bin"],
        "convert": True,
    },
}


def get_huggingface_token() -> Optional[str]:
    """Get HuggingFace token from environment or file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    token_file = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_file):
        with open(token_file) as f:
            return f.read().strip()

    return None


def get_repo_info(repo: str) -> Dict:
    """Get repository information from HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi()
    return api.repo_info(repo)


def download_file_hf(repo: str, filename: str, output_path: str,
                     token: Optional[str] = None, quiet: bool = False) -> bool:
    """Download a single file from HuggingFace."""
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="model",
            local_dir=output_path,
            token=token,
            force_download=True,
        )
        return True
    except Exception as e:
        if not quiet:
            print(f"[ERROR] Failed to download {filename}: {e}")
        return False


def download_with_wget(url: str, output_path: str, quiet: bool = False) -> bool:
    """Download file using wget (fallback)."""
    try:
        cmd = ["wget", "-q", "-O", output_path, url]
        if quiet:
            cmd.insert(2, "--quiet")
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_with_curl(url: str, output_path: str, quiet: bool = False) -> bool:
    """Download file using curl (fallback)."""
    try:
        cmd = ["curl", "-sSL", "-o", output_path, url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def guess_hf_url(repo: str, filename: str) -> str:
    """Guess the HuggingFace CDN URL for a file."""
    # HF uses cdn.jsdelivr.net for raw file access
    repo_id = repo.replace("/", "--")
    return f"https://cdn.jsdelivr.net/gh/{repo_id}@main/{filename}"


def convert_to_bump(gguf_path: str, output_dir: str) -> bool:
    """Convert GGUF model to BUMP format."""
    convert_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "scripts", "convert_gguf_to_bump.py"
    )

    if not os.path.exists(convert_script):
        print("[WARN] GGUF to BUMP conversion script not found")
        return False

    try:
        cmd = [sys.executable, convert_script, "--input", gguf_path, "--output", output_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("[INFO] Converted to BUMP format")
            return True
        else:
            print(f"[WARN] Conversion failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[WARN] Conversion error: {e}")
        return False


def download_preset(preset_name: str, output_dir: str, token: Optional[str] = None,
                    force: bool = False) -> bool:
    """Download a preset model."""
    if preset_name not in PRESETS:
        print(f"[ERROR] Unknown preset: {preset_name}")
        return False

    preset = PRESETS[preset_name]
    repo = preset["repo"]

    print(f"[INFO] Downloading preset: {preset_name}")
    print(f"[INFO] Repository: {repo}")
    print(f"[INFO] Output: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Try huggingface_hub first
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        # Get repository info
        repo_info = api.repo_info(repo, repo_type="model")

        # Download each file
        for filename in preset["files"]:
            print(f"[INFO] Downloading {filename}...")
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path) and not force:
                print(f"[INFO] File exists, skipping: {filename}")
                continue

            try:
                api.hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    local_dir=output_dir,
                    force_download=force,
                )
                print(f"[INFO] Downloaded: {filename}")
            except Exception as e:
                print(f"[ERROR] Failed to download {filename}: {e}")
                return False

    except ImportError:
        print("[WARN] huggingface_hub not installed, using wget/curl")
        url = guess_hf_url(repo, preset["files"][0])
        output_path = os.path.join(output_dir, preset["files"][0])

        if os.path.exists(output_path) and not force:
            print(f"[INFO] File exists, skipping")
        else:
            print(f"[INFO] Downloading from {url}...")
            if not download_with_curl(url, output_path):
                download_with_wget(url, output_path)

    # Convert if needed
    if preset.get("convert", False):
        gguf_path = os.path.join(output_dir, preset["files"][0])
        if os.path.exists(gguf_path):
            convert_to_bump(gguf_path, output_dir)

    # Generate weights manifest
    generate_manifest(output_dir, repo)

    print(f"[INFO] Download complete: {output_dir}")
    return True


def generate_manifest(output_dir: str, repo: str) -> None:
    """Generate weights manifest for downloaded model."""
    manifest = {
        "version": 1,
        "model": repo,
        "generated": str(__import__("datetime").datetime.utcnow().isoformat()) + "Z",
        "entries": [],
    }

    manifest_path = os.path.join(output_dir, "weights_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Generated manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download models from HuggingFace")
    parser.add_argument("--repo", help="HuggingFace repository (e.g., Qwen/Qwen2-0.5B-Instruct)")
    parser.add_argument("--files", nargs="+", help="Files to download")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--preset", help="Use preset model (e.g., qwen2-0.5b, smollm-135)")
    parser.add_argument("--token", help="HuggingFace token")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-download")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")

    args = parser.parse_args()

    if args.list_presets:
        print("Available presets:")
        for name, preset in PRESETS.items():
            print(f"  {name}: {preset['repo']}")
        return 0

    if not args.repo and not args.preset:
        parser.print_help()
        print("\n[ERROR] Must specify --repo or --preset")
        return 1

    # Determine output directory
    if args.output:
        output_dir = args.output
    elif args.preset:
        output_dir = os.path.join(DEFAULT_CACHE_DIR, "models", args.preset)
    else:
        repo_name = args.repo.split("/")[-1].lower()
        output_dir = os.path.join(DEFAULT_CACHE_DIR, "models", repo_name)

    # Get token
    token = args.token or get_huggingface_token()

    # Download
    if args.preset:
        success = download_preset(args.preset, output_dir, token, args.force)
    else:
        # Download from repo
        print(f"[INFO] Downloading from {args.repo}")
        os.makedirs(output_dir, exist_ok=True)

        files = args.files or ["config.json", "tokenizer.json"]
        for filename in files:
            print(f"[INFO] Downloading {filename}...")
            success = download_file_hf(args.repo, filename, output_dir, token)
            if not success:
                print(f"[ERROR] Failed to download {filename}")
                return 1

        print(f"[INFO] Download complete: {output_dir}")
        success = True

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
