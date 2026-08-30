#!/usr/bin/env python3
"""Certify Cohere Compass OCR quality with the official PyTorch runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import certify_multimodal_ocr_corpus_v8 as corpus


def _tensor_sha256(tensor: Any) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _model_identity(checkpoint: Path) -> dict[str, Any]:
    config = checkpoint / "config.json"
    weights = sorted(checkpoint.glob("*.safetensors"))
    return {
        "checkpoint": str(checkpoint.resolve()),
        "config_sha256": corpus._sha256_file(config),
        "weight_files": [
            {"name": path.name, "size": path.stat().st_size, "sha256": corpus._sha256_file(path)}
            for path in weights
        ],
    }


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _oracle_identity() -> dict[str, Any]:
    import tokenizers
    import torch
    import torchvision
    import transformers

    source = Path(transformers.__file__).resolve()
    source_root = source.parents[1]
    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
        "transformers_source": str(source_root),
        "transformers_commit": _git_commit(source_root),
    }


def _stop_reason(token_ids: list[int], max_new_tokens: int, eos_token_ids: set[int]) -> str:
    if token_ids and token_ids[-1] in eos_token_ids:
        return "stop_token"
    if len(token_ids) >= max_new_tokens:
        return "max_tokens"
    return "generation_complete"


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    checkpoint = args.checkpoint.resolve()
    samples = corpus._load_samples(args.manifest)
    selected = samples[args.start_index - 1 :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if args.require_images is not None and len(samples) < args.require_images:
        raise RuntimeError(f"manifest has {len(samples)} images; {args.require_images} required")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).eval()
    load_sec = time.perf_counter() - load_started

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    config = {
        "schema": "cke.cohere_compass_pytorch_ocr_certification",
        "schema_version": 1,
        "model": _model_identity(checkpoint),
        "oracle": _oracle_identity(),
        "manifest_sha256": corpus._sha256_file(args.manifest),
        "prompt_template_sha256": corpus._sha256_bytes(args.prompt_template.encode("utf-8")),
        "threads": args.threads,
        "attn_implementation": args.attn_implementation,
        "max_new_tokens": args.max_new_tokens,
        "repetitions": args.repetitions,
    }
    config_hash = corpus._sha256_json(config)
    corpus._write_json(output_dir / "config.json", {**config, "config_sha256": config_hash})

    rows: list[dict[str, Any]] = []
    for sample in selected:
        case_dir = output_dir / f"image{sample['index']:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        case_dir.chmod(0o700)
        prompt = corpus._build_prompt(args.prompt_template, sample["truth"], sample["prompt"])
        max_new_tokens = corpus._max_new_tokens(args.max_new_tokens, sample)
        case_config = corpus._case_config(config_hash, sample, prompt, max_new_tokens)
        case_path = case_dir / "case_result.json"
        resumed = corpus._load_resumed(case_path, case_config)
        if resumed is not None:
            rows.append(resumed)
            print(f"[{len(rows)}/{len(selected)}] image {sample['index']:02d}: resumed", flush=True)
            continue

        started = time.perf_counter()
        try:
            image = Image.open(sample["image"]).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            prepared = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            prompt_tokens = int(prepared["input_ids"].shape[-1])
            token_traces: list[list[int]] = []
            generation_wall_sec: list[float] = []
            for _ in range(args.repetitions):
                generation_started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **prepared,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                generation_wall_sec.append(time.perf_counter() - generation_started)
                token_traces.append(generated[0, prompt_tokens:].detach().cpu().tolist())
            token_ids = token_traces[0]
            repeatable = all(trace == token_ids for trace in token_traces[1:])
            generated_text = processor.decode(token_ids, skip_special_tokens=True)
            parsed = corpus._extract_json_object(generated_text)
            metrics = corpus._score(sample["truth"], parsed)
            row = {
                "case_config": case_config,
                "image_index": sample["index"],
                "image_id": sample["id"],
                "image_path": str(sample["image"]),
                "image_sha256": sample["image_sha256"],
                "truth_path": str(sample["truth_path"]),
                "truth_sha256": sample["truth_sha256"],
                "status": "complete",
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "input_ids_sha256": _tensor_sha256(prepared["input_ids"]),
                "generated_text": generated_text,
                "generated_tokens": len(token_ids),
                "generated_token_ids": token_ids,
                "output_sha256": corpus._sha256_bytes(generated_text.encode("utf-8")),
                "token_trace_sha256": corpus._sha256_json(token_ids),
                "stop_reason": _stop_reason(
                    token_ids,
                    max_new_tokens,
                    {
                        int(value)
                        for value in (
                            model.generation_config.eos_token_id
                            if isinstance(model.generation_config.eos_token_id, list)
                            else [model.generation_config.eos_token_id]
                        )
                        if value is not None
                    },
                ),
                "repeatability": {
                    "repetitions": args.repetitions,
                    "exact": repeatable,
                    "trace_sha256": [corpus._sha256_json(trace) for trace in token_traces],
                },
                "timings": {
                    "wall_sec": time.perf_counter() - started,
                    "generation_wall_sec": generation_wall_sec,
                },
                "metrics": metrics,
            }
        except Exception as exc:
            row = {
                "case_config": case_config,
                "image_index": sample["index"],
                "image_sha256": sample["image_sha256"],
                "truth_sha256": sample["truth_sha256"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "timings": {"wall_sec": time.perf_counter() - started},
            }
        corpus._write_json(case_path, row)
        rows.append(row)
        if row["status"] == "complete":
            metrics = row["metrics"]
            print(
                f"[{len(rows)}/{len(selected)}] image {sample['index']:02d}: "
                f"json={'yes' if metrics['json_valid'] else 'no'} "
                f"fields={metrics['exact_fields']}/{metrics['expected_fields']}",
                flush=True,
            )
        else:
            print(f"[{len(rows)}/{len(selected)}] image {sample['index']:02d}: {row['error']}", flush=True)

    public_rows = [corpus._public_row(row) for row in rows if row.get("status") == "complete"]
    summary = {
        **config,
        "config_sha256": config_hash,
        "model_load_sec": load_sec,
        "aggregate": corpus._aggregate(rows, len(selected)),
        "cases": public_rows,
    }
    corpus._write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-template", default=corpus.DEFAULT_PROMPT)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=896)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-images", type=int)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args(argv)
    if (
        args.start_index <= 0
        or args.threads <= 0
        or args.max_new_tokens <= 0
        or args.repetitions <= 0
    ):
        parser.error("start index, thread count, and max token count must be positive")
    summary = run(args)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    return 1 if summary["aggregate"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
