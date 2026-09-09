#!/usr/bin/env python3
"""Certify greedy Muse-Glimmer text trajectories against PyTorch eager BF16."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


CASES = {
    "c_code": "Write a complete C function that returns the larger of two integers.",
    "svg": "Create a standalone SVG of a red circle on a white background.",
    "cke_analysis": (
        "Analyze this C-Kernel-Engine v8 design: JSON circuits declare operations; "
        "kernel maps declare call ABIs and scratch contracts; build_ir_v8.py lowers "
        "the graph and the memory planner assigns arena offsets. Explain one strength "
        "and one risk."
    ),
}


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_ids(tokenizer, prompt: str, date_string: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        current_date=date_string,
    )
    values = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return [int(value) for value in values]


def _resolved_symbol_library(lib: ctypes.CDLL, symbol: str) -> Path:
    function = getattr(lib, symbol)
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    dladdr.restype = ctypes.c_int
    info = _DlInfo()
    if (
        dladdr(ctypes.cast(function, ctypes.c_void_p), ctypes.byref(info)) == 0
        or not info.dli_fname
    ):
        raise RuntimeError(f"dladdr could not resolve runtime symbol {symbol}")
    return Path(os.fsdecode(info.dli_fname)).resolve()


def _verify_loaded_engine(lib: ctypes.CDLL, requested_engine: Path) -> Path:
    requested_engine = requested_engine.resolve()
    loaded_engine = _resolved_symbol_library(lib, "ck_set_num_threads")
    if loaded_engine != requested_engine:
        raise RuntimeError(
            "Muse certification resolved a different CK engine than requested: "
            f"requested={requested_engine} loaded={loaded_engine}"
        )
    return loaded_engine


def _validate_reference_manifest(reference: object) -> list[dict]:
    if not isinstance(reference, dict):
        raise ValueError("reference manifest must be a JSON object")
    if reference.get("schema") != "cke.v8.muse_glimmer_text_reference":
        raise ValueError("reference manifest has the wrong schema")
    max_tokens = reference.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError("reference max_tokens must be a positive integer")
    cases = reference.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("reference manifest must contain nonempty cases")
    if not all(isinstance(case, dict) for case in cases):
        raise ValueError("every reference case must be an object")
    names = [case.get("name") for case in cases]
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("every reference case must have a nonempty string name")
    if len(names) != len(set(names)):
        raise ValueError("reference case names must be unique")
    expected_names = set(CASES)
    actual_names = set(names)
    if actual_names != expected_names:
        raise ValueError(
            "reference cases do not match the certification suite: "
            f"missing={sorted(expected_names - actual_names)} "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    for case in cases:
        prompt_tokens = case.get("prompt_tokens")
        generated_tokens = case.get("generated_tokens")
        if (
            not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens <= 0
        ):
            raise ValueError(
                f"reference {case['name']} prompt_tokens must be positive"
            )
        if (
            not isinstance(generated_tokens, int)
            or isinstance(generated_tokens, bool)
            or generated_tokens <= 0
            or generated_tokens > max_tokens
        ):
            raise ValueError(
                f"reference {case['name']} generated_tokens must be within "
                f"1..{max_tokens}"
            )
        if not isinstance(case.get("output"), str):
            raise ValueError(f"reference {case['name']} output must be a string")
    return cases


def _validate_reference_arrays(
    case: dict, arrays: object, vocab: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"prompt_ids", "generated_ids", "logits"}
    files = set(getattr(arrays, "files", ()))
    if files != required:
        raise ValueError(
            f"reference {case['name']} arrays mismatch: "
            f"missing={sorted(required - files)} unexpected={sorted(files - required)}"
        )
    raw_prompt_ids = arrays["prompt_ids"]
    raw_expected_ids = arrays["generated_ids"]
    raw_expected_logits = arrays["logits"]
    if raw_prompt_ids.ndim != 1 or raw_prompt_ids.size == 0:
        raise ValueError(f"reference {case['name']} prompt_ids must be nonempty rank 1")
    if raw_expected_ids.ndim != 1 or raw_expected_ids.size == 0:
        raise ValueError(
            f"reference {case['name']} generated_ids must be nonempty rank 1"
        )
    if not np.issubdtype(raw_prompt_ids.dtype, np.integer) or not np.issubdtype(
        raw_expected_ids.dtype, np.integer
    ):
        raise ValueError(f"reference {case['name']} token IDs must be integers")
    if raw_expected_logits.ndim != 2:
        raise ValueError(f"reference {case['name']} logits must have rank 2")
    if raw_expected_logits.shape != (raw_expected_ids.size, vocab):
        raise ValueError(
            f"reference {case['name']} logits shape {raw_expected_logits.shape} "
            f"does not match trajectory/vocab {(raw_expected_ids.size, vocab)}"
        )
    if raw_expected_logits.dtype != np.float32:
        raise ValueError(f"reference {case['name']} logits must use float32 storage")
    if not np.isfinite(raw_expected_logits).all():
        raise ValueError(f"reference {case['name']} logits contain non-finite values")
    prompt_ids = raw_prompt_ids.astype(np.int32)
    expected_ids = raw_expected_ids.astype(np.int32)
    expected_logits = raw_expected_logits.astype(np.float32)
    if np.any(prompt_ids < 0) or np.any(prompt_ids >= vocab):
        raise ValueError(f"reference {case['name']} prompt IDs exceed the vocabulary")
    if np.any(expected_ids < 0) or np.any(expected_ids >= vocab):
        raise ValueError(f"reference {case['name']} generated IDs exceed the vocabulary")
    if case.get("prompt_tokens") != prompt_ids.size:
        raise ValueError(f"reference {case['name']} prompt token count is inconsistent")
    if case.get("generated_tokens") != expected_ids.size:
        raise ValueError(f"reference {case['name']} generated token count is inconsistent")
    return prompt_ids, expected_ids, expected_logits


def _case_passes(free: dict, forced: dict, expected_steps: int) -> bool:
    return bool(
        expected_steps > 0
        and forced.get("first_logit_divergence") is None
        and forced.get("finite_logit_rows") == expected_steps
        and free.get("first_token_divergence") is None
        and free.get("finite_logit_rows") == expected_steps
    )


def create_reference(
    model_dir: Path,
    output_dir: Path,
    max_tokens: int,
    date_string: str,
) -> None:
    import torch
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    raw_eos = model.generation_config.eos_token_id
    eos_ids = {
        int(value) for value in (raw_eos if isinstance(raw_eos, list) else [raw_eos])
        if value is not None
    }
    manifest = {
        "schema": "cke.v8.muse_glimmer_text_reference",
        "model_dir": str(model_dir.resolve()),
        "dtype": "bfloat16",
        "attention_backend": "eager",
        "date_string": date_string,
        "max_tokens": max_tokens,
        "provenance": {
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "config_sha256": _sha256(model_dir / "config.json"),
            "weight_index_sha256": _sha256(
                model_dir / "model.safetensors.index.json"
            ),
            "tokenizer_sha256": _sha256(model_dir / "tokenizer.json"),
            "chat_template_sha256": _sha256(model_dir / "chat_template.jinja"),
        },
        "cases": [],
    }
    with torch.inference_mode():
        for name, prompt in CASES.items():
            prompt_ids = _prompt_ids(tokenizer, prompt, date_string)
            result = model(
                input_ids=torch.tensor([prompt_ids], dtype=torch.long),
                use_cache=True,
                logits_to_keep=1,
            )
            cache = result.past_key_values
            generated: list[int] = []
            logits_rows: list[np.ndarray] = []
            for _ in range(max_tokens):
                logits = result.logits[0, -1].float().cpu().numpy()
                logits_rows.append(logits)
                token = int(logits.argmax())
                generated.append(token)
                if token in eos_ids:
                    break
                result = model(
                    input_ids=torch.tensor([[token]], dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = result.past_key_values
            np.savez(
                output_dir / f"{name}.npz",
                prompt_ids=np.asarray(prompt_ids, dtype=np.int32),
                generated_ids=np.asarray(generated, dtype=np.int32),
                logits=np.stack(logits_rows),
            )
            manifest["cases"].append(
                {
                    "name": name,
                    "prompt": prompt,
                    "prompt_tokens": len(prompt_ids),
                    "generated_tokens": len(generated),
                    "output": tokenizer.decode(generated, skip_special_tokens=True),
                }
            )
    (output_dir / "reference.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def compare_cke(runtime_dir: Path, reference_dir: Path, report_path: Path) -> int:
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from compare_ck_prefill_decode_logits_v8 import _extract_logits, _init_model

    from transformers import AutoTokenizer

    report = {
        "schema": "cke.v8.muse_glimmer_text_parity",
        "runtime_dir": str(runtime_dir.resolve()),
        "reference_dir": str(reference_dir.resolve()),
        "comparison": {
            "numeric": "forced-reference-token history",
            "trajectory": "free-running greedy history",
            "exactness": "float32 IEEE-754 bit-pattern equality",
        },
        "provenance": {
            "libmodel_sha256": _sha256(runtime_dir / "libmodel.so"),
            "engine_sha256": _sha256(runtime_dir / "libckernel_engine.so"),
            "generated_c_sha256": _sha256(runtime_dir / "model_v8.c"),
            "runtime_bundle_sha256": _sha256(runtime_dir / ".ck_runtime_bundle.json"),
            "reference_manifest_sha256": _sha256(reference_dir / "reference.json"),
            "runner_checkout_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=scripts_dir, text=True
            ).strip(),
            "runner_checkout_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=scripts_dir, text=True
                ).strip()
            ),
        },
        "cases": [],
        "status": "error",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def publish() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    publish()
    lib = None
    tokenizer = None
    vocab = 0

    def prefill(prompt_ids: np.ndarray, name: str) -> None:
        assert lib is not None
        lib.ck_model_kv_cache_reset()
        token_array = (ctypes.c_int32 * len(prompt_ids))(*prompt_ids.tolist())
        if lib.ck_model_embed_tokens(token_array, len(prompt_ids)) != 0:
            raise RuntimeError(f"CK prefill embedding failed for {name}")
        if lib.ck_model_forward(None) != 0:
            raise RuntimeError(f"CK prefill failed for {name}")

    def run_path(
        prompt_ids: np.ndarray,
        expected_ids: np.ndarray,
        expected_logits: np.ndarray,
        name: str,
        forced: bool,
    ) -> dict:
        assert lib is not None and tokenizer is not None
        prefill(prompt_ids, name)
        actual_ids: list[int] = []
        first_logit_divergence = None
        first_token_divergence = None
        exact_rows = 0
        maximum_error = 0.0
        finite_rows = 0
        first_nonfinite_step = None
        for step, expected_token in enumerate(expected_ids):
            actual_logits, _, _ = _extract_logits(
                lib, vocab, len(prompt_ids) if step == 0 else 1
            )
            expected_row = expected_logits[step]
            finite = bool(np.isfinite(actual_logits).all())
            finite_rows += int(finite)
            if not finite and first_nonfinite_step is None:
                first_nonfinite_step = step
            bits_equal = bool(
                np.array_equal(
                    actual_logits.view(np.uint32), expected_row.view(np.uint32)
                )
            )
            exact_rows += int(bits_equal)
            delta = np.abs(actual_logits - expected_row)
            row_error = float(delta.max()) if finite else float("inf")
            maximum_error = max(maximum_error, row_error)
            actual_token = int(actual_logits.argmax())
            actual_ids.append(actual_token)
            if first_logit_divergence is None and not bits_equal:
                first_logit_divergence = {
                    "step": step,
                    "finite": finite,
                    "max_abs_error": row_error,
                }
            if first_token_divergence is None and actual_token != int(expected_token):
                first_token_divergence = {
                    "step": step,
                    "expected_token": int(expected_token),
                    "actual_token": actual_token,
                }
            if step + 1 < len(expected_ids):
                next_token = int(expected_token) if forced else actual_token
                if lib.ck_model_decode(ctypes.c_int32(next_token), None) != 0:
                    raise RuntimeError(f"CK decode failed for {name} at step {step}")
        return {
            "actual_ids": actual_ids,
            "actual_output": tokenizer.decode(actual_ids, skip_special_tokens=True),
            "bit_exact_logit_rows": exact_rows,
            "finite_logit_rows": finite_rows,
            "first_nonfinite_step": first_nonfinite_step,
            "max_abs_error": maximum_error,
            "first_logit_divergence": first_logit_divergence,
            "first_token_divergence": first_token_divergence,
        }

    passed = False
    try:
        reference = json.loads((reference_dir / "reference.json").read_text())
        cases = _validate_reference_manifest(reference)
        engine_path = (runtime_dir / "libckernel_engine.so").resolve()
        model_path = (runtime_dir / "libmodel.so").resolve()
        if not engine_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(
                f"runtime requires {engine_path} and {model_path}"
            )
        ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(str(model_path), mode=ctypes.RTLD_GLOBAL)
        loaded_engine = _verify_loaded_engine(lib, engine_path)
        report["provenance"]["loaded_engine_path"] = str(loaded_engine)
        report["provenance"]["loaded_engine_sha256"] = _sha256(loaded_engine)
        report["status"] = "running"
        publish()
        tokenizer = AutoTokenizer.from_pretrained(
            runtime_dir, fix_mistral_regex=True
        )
        _init_model(lib, runtime_dir)
        lib.ck_model_get_vocab_size.argtypes = []
        lib.ck_model_get_vocab_size.restype = ctypes.c_int
        vocab = int(lib.ck_model_get_vocab_size())
        if vocab <= 0:
            raise RuntimeError(f"runtime reported invalid vocabulary size {vocab}")
        lib.ck_model_kv_cache_reset.argtypes = []
        lib.ck_model_kv_cache_reset.restype = None
        lib.ck_model_embed_tokens.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
        ]
        lib.ck_model_embed_tokens.restype = ctypes.c_int
        lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
        lib.ck_model_forward.restype = ctypes.c_int
        lib.ck_model_decode.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.ck_model_decode.restype = ctypes.c_int
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None

        passed = True
        for case in cases:
            result = {"name": case["name"], "status": "error"}
            report["cases"].append(result)
            try:
                with np.load(
                    reference_dir / f"{case['name']}.npz", allow_pickle=False
                ) as arrays:
                    prompt_ids, expected_ids, expected_logits = (
                        _validate_reference_arrays(case, arrays, vocab)
                    )
                free = run_path(
                    prompt_ids, expected_ids, expected_logits, case["name"], False
                )
                if free["first_token_divergence"] is None:
                    forced = dict(free)
                    forced["execution_reused"] = True
                    forced["reuse_reason"] = (
                        "Free-running tokens equal the reference, so both histories are identical."
                    )
                else:
                    forced = run_path(
                        prompt_ids, expected_ids, expected_logits, case["name"], True
                    )
                    forced["execution_reused"] = False
                case_passed = _case_passes(free, forced, len(expected_ids))
                result.update(
                    {
                        "status": "pass" if case_passed else "fail",
                        "prompt_tokens": len(prompt_ids),
                        "generated_tokens": len(expected_ids),
                        "reference_ids": expected_ids.tolist(),
                        "reference_output": case["output"],
                        "forced_reference_history": forced,
                        "free_running_history": free,
                    }
                )
                passed = passed and case_passed
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                passed = False
            report["status"] = "running" if passed else "fail"
            publish()
        passed = passed and len(report["cases"]) == len(cases) and all(
            case.get("status") == "pass" for case in report["cases"]
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        passed = False
    finally:
        if lib is not None and hasattr(lib, "ck_model_free"):
            try:
                lib.ck_model_free()
            except Exception as exc:
                report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                passed = False
    report["status"] = "pass" if passed else "fail"
    publish()
    print(json.dumps(report, indent=2))
    return 0 if passed else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reference = subparsers.add_parser("reference")
    reference.add_argument("--model-dir", required=True, type=Path)
    reference.add_argument("--output-dir", required=True, type=Path)
    reference.add_argument("--max-tokens", type=int, default=128)
    reference.add_argument("--date-string", default="2026-09-09")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--runtime-dir", required=True, type=Path)
    compare.add_argument("--reference-dir", required=True, type=Path)
    compare.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "reference":
        create_reference(args.model_dir, args.output_dir, args.max_tokens, args.date_string)
        return 0
    return compare_cke(args.runtime_dir, args.reference_dir, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
