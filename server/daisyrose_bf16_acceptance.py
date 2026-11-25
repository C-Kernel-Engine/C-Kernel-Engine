#!/usr/bin/env python3
"""Compare an attested DaisyRose BF16 HTTP service against streamed PyTorch."""
from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys

import httpx
import numpy as np

from server import http_lifecycle_acceptance as lifecycle
from server.serving_bundle import (
    _loaded_model_token_decoder,
    _loaded_model_token_lookup,
    bundle_path,
    load_resolved_serving,
    resolved_renderer_tokens,
    runtime_auto_bos_token,
    sha256_file,
    verified_loaded_symbol_backing,
    verify_runtime_dependencies,
)
from version.v8.scripts.compare_ck_prefill_decode_logits_v8 import _extract_logits, _init_model


PIN = Path(__file__).with_name("tests") / "fixtures" / "daisyrose_checkpoint.json"
CASES = (
    ("short", [{"role": "user", "content": "Reply with exactly: hello"}], "hello"),
    ("long", [{"role": "user", "content": (
        "These are examples for a transcription exercise. Red becomes red, blue becomes blue, "
        "and green becomes green. Copy just the final word below, with no punctuation.\n\namber"
    )}], "amber"),
    ("multi-turn", [
        {"role": "user", "content": "The codeword is cobalt. Acknowledge with okay."},
        {"role": "assistant", "content": "okay"},
        {"role": "user", "content": "What was the codeword? Reply with only the word."},
    ], "cobalt"),
)


def compare_logits(native, oracle, *, max_abs_limit, rmse_limit):
    if native.shape != oracle.shape or native.ndim != 2 or not native.size:
        raise ValueError("native and oracle must have identical, nonempty logits geometry")
    if not np.isfinite(native).all() or not np.isfinite(oracle).all():
        raise ValueError("nonfinite native or oracle logits")
    rows = []
    for i, (left, right) in enumerate(zip(native, oracle, strict=True)):
        delta = left.astype(np.float64) - right
        max_abs = float(np.max(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(delta**2)))
        rows.append({
            "phase": "prefill" if i == 0 else "teacher_forced_decode",
            "step": i, "max_abs": max_abs, "rmse": rmse,
            "native_argmax": int(np.argmax(left)), "oracle_argmax": int(np.argmax(right)),
            "numeric_pass": max_abs <= max_abs_limit and rmse <= rmse_limit,
        })
    return rows


def verify_checkpoint(checkpoint):
    pin = json.loads(PIN.read_text())
    for name, digest in pin["files"].items():
        if sha256_file(checkpoint / name) != digest:
            raise ValueError(f"publisher checkpoint digest mismatch: {name}")
    return pin


def publisher_logits(checkpoint, cases, threads):
    """Use publisher modules and cache with at most one layer mapped at a time."""
    import torch
    import transformers
    from safetensors import safe_open
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
    )

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model."
    config = Qwen3_5TextConfig.from_dict(json.loads((checkpoint / "config.json").read_text())["text_config"])
    config._attn_implementation = "eager"
    torch.set_num_threads(threads)

    def weights(names):
        by_shard = defaultdict(list)
        for name in names:
            by_shard[index[name]].append(name)
        tensors = {}
        for shard, keys in by_shard.items():
            with safe_open(checkpoint / shard, framework="pt", device="cpu") as source:
                for key in keys:
                    tensor = source.get_tensor(key)
                    if tensor.dtype != torch.bfloat16:
                        raise ValueError(f"unexpected publisher weight dtype: {key}")
                    tensors[key] = tensor
        return tensors

    with torch.inference_mode():
        embed = weights([prefix + "embed_tokens.weight"])[prefix + "embed_tokens.weight"]
        rope = Qwen3_5TextRotaryEmbedding(config)
        sequences = []
        for case in cases:
            offset, frames = 0, []
            for tokens in [case["prompt_ids"], *([token] for token in case["answer_ids"])]:
                hidden = embed[torch.tensor(tokens)[None]]
                positions = torch.arange(offset, offset + len(tokens))[None, :]
                offset += len(tokens)
                visible = torch.arange(offset)[None, :] <= positions[0, :, None]
                mask = torch.zeros(visible.shape, dtype=torch.bfloat16).masked_fill(
                    ~visible, torch.finfo(torch.bfloat16).min
                )[None, None]
                frames.append((hidden, rope(hidden, positions[None].expand(3, -1, -1)), mask, positions))
            sequences.append((frames, DynamicCache(config=config)))
        del embed
        for index_in_model in range(config.num_hidden_layers):
            layer_prefix = f"{prefix}layers.{index_in_model}."
            with torch.device("meta"):
                layer = Qwen3_5DecoderLayer(config, index_in_model)
            tensors = weights(layer_prefix + key for key in layer.state_dict())
            layer.load_state_dict({key.removeprefix(layer_prefix): val for key, val in tensors.items()},
                                  strict=True, assign=True)
            del tensors
            layer.eval()
            for frames, cache in sequences:
                for i, (hidden, rotary, mask, positions) in enumerate(frames):
                    output = layer(hidden, rotary, mask if layer.block_type == "full_attention" else None,
                                   positions, past_key_values=cache, use_cache=True)
                    frames[i] = (output, rotary, mask, positions)
            del layer

        norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.load_state_dict({"weight": weights([prefix + "norm.weight"])[prefix + "norm.weight"]}, assign=True)
        head = weights(["lm_head.weight"])["lm_head.weight"]
        results = [
            torch.nn.functional.linear(norm(torch.cat([frame[0][:, -1, :] for frame in frames])), head)
            .float().numpy()
            for frames, _ in sequences
        ]
    return results, {"torch": torch.__version__, "transformers": transformers.__version__, "threads": threads}


def native_logits(bundle, contract, checkpoint):
    from transformers import AutoTokenizer
    from server.live import _render_with_chat_templates

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    verify_runtime_dependencies(contract, native_library=bundle / "libmodel.so")
    lib = C.CDLL(str(bundle / "libmodel.so"), mode=C.RTLD_GLOBAL)
    verified_loaded_symbol_backing(lib, "ck_model_init", bundle / "libmodel.so")
    bindings = {
        "ck_model_kv_cache_reset": ([], None),
        "ck_model_free": ([], None),
        "ck_model_get_vocab_size": ([], C.c_int),
        "ck_model_embed_tokens": ([C.POINTER(C.c_int32), C.c_int], C.c_int),
        "ck_model_forward": ([C.POINTER(C.c_float)], C.c_int),
        "ck_model_decode": ([C.c_int32, C.POINTER(C.c_float)], C.c_int),
        "ck_model_encode_text": ([C.c_char_p, C.c_int], C.c_int),
        "ck_model_get_token_buffer": ([], C.POINTER(C.c_int32)),
    }
    for name, (args, result) in bindings.items():
        symbol = getattr(lib, name)
        symbol.argtypes, symbol.restype = args, result
    _init_model(lib, bundle)
    try:
        vocab = lib.ck_model_get_vocab_size()
        renderer_tokens = resolved_renderer_tokens(
            bundle, contract, decode_ids=_loaded_model_token_decoder(bundle),
            lookup_token=_loaded_model_token_lookup(bundle),
        )
        template = bundle_path(bundle, contract["assets"]["chat"]["path"]).read_text()
        bos = runtime_auto_bos_token(bundle)

        def execute(ids, followup):
            lib.ck_model_kv_cache_reset()
            array = (C.c_int32 * len(ids))(*ids)
            if lib.ck_model_embed_tokens(array, len(ids)) or lib.ck_model_forward(None):
                raise RuntimeError("native prefill failed")
            logits = [_extract_logits(lib, vocab, len(ids))[0]]
            for token in followup:
                if lib.ck_model_decode(token, None):
                    raise RuntimeError("native continuation failed")
                logits.append(_extract_logits(lib, vocab, 1)[0])
            return np.stack(logits)

        cases, results = [], []
        for name, messages, answer in CASES:
            rendered = _render_with_chat_templates(
                template, None, messages, None, contract, renderer_tokens=renderer_tokens,
                tool_protocol=contract["output_protocol"],
            )
            expected = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
            if rendered != expected:
                raise ValueError(f"{name}: CKE and publisher rendered prompts differ")
            text = ((bos if bos and not rendered.startswith(bos) else "") + rendered).encode()
            reference_ids = tokenizer.encode(expected, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            count = lib.ck_model_encode_text(text, len(text))
            if count <= 0 or count != len(reference_ids) or count + len(answer_ids) >= contract["context_capacity"]:
                raise ValueError(f"{name}: invalid native token count or context capacity")
            native_ids = list(lib.ck_model_get_token_buffer()[:count])
            if native_ids != reference_ids:
                raise ValueError(f"{name}: native and publisher prompt token IDs differ")
            cases.append({"name": name, "prompt_ids": reference_ids, "answer_ids": answer_ids})
            results.append(execute(native_ids, answer_ids))
        if not np.array_equal(results[0], execute(cases[0]["prompt_ids"], cases[0]["answer_ids"])):
            raise RuntimeError("first native request drifted after intervening requests")
    finally:
        lib.ck_model_free()
    return cases, results


def check_http(args, contract, report):
    session_hash = sha256_file(args.session_library)
    endpoint = args.endpoint.rstrip("/")
    with httpx.Client(timeout=60.0) as client:
        response = client.get(endpoint + "/cke/loaded-identity")
        response.raise_for_status()
        before = response.json()
        lifecycle._check_identity(before, expected_serving=contract["identity"], expected_session=session_hash,
                                  model=args.model)
        report["loaded_identity"] = before
        report["http"] = {}
        for name, messages, answer in CASES:
            response = client.post(endpoint + "/chat/completions", json={
                "model": args.model, "messages": messages, "temperature": 0, "max_tokens": 16,
            })
            response.raise_for_status()
            choice, = response.json()["choices"]
            report["http"][name] = {
                "answer": choice["message"]["content"], "finish_reason": choice["finish_reason"],
                "pass": choice["message"]["content"] == answer and choice["finish_reason"] == "stop",
            }
        for suite in ("plain-chat", "lifecycle"):
            process = subprocess.run([
                sys.executable, "-m", "server.http_lifecycle_acceptance", "--suite", suite,
                "--endpoint", endpoint, "--model", args.model, "--prefill-lines", "8",
                "--lifecycle-max-tokens", "24",
                "--expected-serving-identity", contract["identity"],
                "--expected-session-library-sha256", session_hash,
                "--output", str(args.output_dir / f"{suite}.json"),
            ], check=False)
            report["http"][suite] = {"pass": process.returncode == 0}
        after = client.get(endpoint + "/cke/loaded-identity")
        after.raise_for_status()
        if after.json() != before:
            raise ValueError("loaded serving identity changed during HTTP acceptance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="daisyrose-27b-bf16")
    parser.add_argument("--session-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oracle-threads", type=int, default=4)
    parser.add_argument("--max-abs", type=float, required=True, help="Predeclared maximum BF16 logit error")
    parser.add_argument("--max-rmse", type=float, required=True, help="Predeclared BF16 logit RMSE bound")
    args = parser.parse_args()
    if args.oracle_threads <= 0 or any(not np.isfinite(x) or x < 0 for x in (args.max_abs, args.max_rmse)):
        parser.error("threads must be positive and logit bounds finite and nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "fail", "limits": {"max_abs": args.max_abs, "rmse": args.max_rmse}}
    try:
        pin = verify_checkpoint(args.checkpoint)
        contract = load_resolved_serving(args.bundle)
        if not contract:
            raise ValueError("the generated bundle needs a resolved serving contract")
        runtime = contract["assets"][".ck_runtime_bundle.json"]
        build = json.loads(bundle_path(args.bundle, runtime["path"]).read_text())["inputs"]
        report["checkpoint"] = pin
        report["build"] = {
            key: build.get(key) for key in (
                "machine", "compiler", "kernel_build_args", "arm_engine_args",
                "extra_cflags", "extra_ldflags", "openmp_flag",
            )
        }
        report["bundle_identity"] = contract["identity"]
        try:
            check_http(args, contract, report)
        except Exception as exc:
            report["http_error"] = f"{type(exc).__name__}: {exc}"

        cases, actual = native_logits(args.bundle, contract, args.checkpoint)
        expected, oracle = publisher_logits(args.checkpoint, cases, args.oracle_threads)
        report.update(oracle=oracle, native_reset_repeat="bit_exact", cases=cases)
        eos = json.loads((args.checkpoint / "generation_config.json").read_text())["eos_token_id"]
        eos = eos if isinstance(eos, list) else [eos]
        for case, native, reference in zip(cases, actual, expected, strict=True):
            rows = compare_logits(native, reference, max_abs_limit=args.max_abs, rmse_limit=args.max_rmse)
            case["logits"] = rows
            case["numerical_pass"] = all(row["numeric_pass"] for row in rows)
            for kind in ("native", "oracle"):
                ids = [row[f"{kind}_argmax"] for row in rows]
                case[f"{kind}_answer_eos_pass"] = ids[:-1] == case["answer_ids"] and ids[-1] in eos
            case["greedy_ids_match"] = all(row["native_argmax"] == row["oracle_argmax"] for row in rows)
            np.savez(args.output_dir / f"{case['name']}-logits.npz", native=native, oracle=reference)
        report["numerical_pass"] = all(c["numerical_pass"] for c in cases)
        report["greedy_answer_eos_pass"] = all(
            c["native_answer_eos_pass"] and c["oracle_answer_eos_pass"] and c["greedy_ids_match"] for c in cases
        )
        report["http_pass"] = not report.get("http_error") and all(row["pass"] for row in report["http"].values())
        if report["numerical_pass"] and report["greedy_answer_eos_pass"] and report["http_pass"]:
            report["status"] = "pass"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "error": report.get("error"),
                      "report": str(args.output_dir / "summary.json")}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
