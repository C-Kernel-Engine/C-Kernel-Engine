"""Run a persistent, pinned Sigma-3B BF16 text-completion harness."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

import numpy as np

from version.v8.scripts.build_sigma_native_session_v8 import build_circuit
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.scripts.sigma_checkpoint_v8 import pinned_weights
from version.v8.scripts.sigma_session_v8 import CANCEL, PTR, SIZE, Seeded, Session, bind_session


GEOMETRY = dict(layers=26, vocabulary=131072, width=3072, heads=32, kv_heads=8,
                dim=128, inner=9216, latent_width=16, bands=32)
TOKENIZER_CONFIG_SHA256 = "282733babeb4f94dd3cbf4a680b7498f971d36e1cbb335339cc161c052f0448a"
TOKENIZER_SHA256 = "3277c00fe5fb3963b3cb7c07b7f183722d2af4d775a4aea7cfb3684d7cccbc2f"


class SigmaTextSession:
    def __init__(self, folder, arena, library, geometry, tokenizer, schedule, guidance, *, chat_template=None):
        self.arena, self.library = arena, library
        self.state, self.identity = bind_session(folder, arena, library, geometry)
        self.tokenizer, self.chat_template = tokenizer, chat_template
        self.schedule = np.ascontiguousarray(schedule, dtype=np.float64)
        self.guidance = guidance
        self.native = library.ck_generate_seeded_diffusion
        self.native.argtypes = [C.POINTER(Session), C.POINTER(Seeded)]
        self.native.restype = C.c_int

    def generate(self, text, *, seed, budget, stop_strings=(), should_cancel=None, system=None):
        if not text.strip():
            raise ValueError("Sigma requires a nonempty prompt")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("Sigma seed must be an unsigned 64-bit integer")
        if any(not delimiter for delimiter in stop_strings):
            raise ValueError("Sigma stop strings must be nonempty")
        if self.chat_template is None:
            ids = self.tokenizer.encode(text).ids
        else:
            messages = [{"role": "system", "content": system}] if system is not None else []
            messages.append({"role": "user", "content": text})
            rendered = self.chat_template.render(messages=messages)
            ids = self.tokenizer.encode(rendered, add_special_tokens=False).ids
        n = len(ids)
        block = self.state.block_rows
        if not 0 < n <= self.state.prompt_rows:
            raise ValueError(f"Prompt uses {n} tokens, capacity is {self.state.prompt_rows}")
        if type(budget) is not int or budget <= 0 or budget % block or budget > self.state.capacity - n:
            raise ValueError(f"Budget must be a positive multiple of {block} within the remaining cache")

        prompt = np.array(ids, dtype=np.int32)
        noise_rows = 1 if self.state.step_bindings.noise == SIZE(-1).value else self.schedule.size
        workspace = np.empty(noise_rows * self.state.latent_elements, np.float64)
        staging, output = np.empty(block, np.int32), np.empty(budget, np.int32)
        request = Seeded(prompt.ctypes.data, n, self.schedule.ctypes.data, self.schedule.size,
                         seed, budget, self.guidance, workspace.ctypes.data, workspace.size,
                         staging.ctypes.data, output.ctypes.data, staging.size, output.size, 0, None, None)
        inspected = 0
        finish = "budget"
        callback_error = None

        def decode():
            return self.tokenizer.decode(output[:request.generated_rows].tolist(), skip_special_tokens=True)

        def delimiter_index(text):
            return min((index for marker in stop_strings if (index := text.find(marker)) >= 0), default=-1)

        @CANCEL
        def stop(_):
            nonlocal inspected, finish, callback_error
            try:
                if should_cancel is not None and should_cancel():
                    finish = "cancelled"
                    return 1
                if stop_strings and request.generated_rows > inspected:
                    inspected = request.generated_rows
                    if delimiter_index(decode()) >= 0:
                        finish = "stop_string"
                        return 1
            except Exception as exc:
                callback_error = exc
                return 1
            return 0

        if stop_strings or should_cancel is not None:
            request.should_cancel = C.cast(stop, PTR)
        start = time.perf_counter()
        status = self.native(C.byref(self.state), C.byref(request))
        seconds = time.perf_counter() - start
        if callback_error is not None:
            raise callback_error
        if status and (status != -7 or finish == "budget"):
            raise RuntimeError(f"Sigma native generation failed with status {status}")
        completion = decode()
        index = delimiter_index(completion)
        if index >= 0 and finish != "cancelled":
            completion, finish = completion[:index], "stop_string"
        return {
            "text": completion, "prompt_ids": ids, "generated_ids": output[:request.generated_rows].tolist(),
            "finish_reason": finish, "seconds": seconds, "seed": seed, "build_identity": self.identity,
            "block_width": block, "steps": int(self.schedule.size - 1),
            "gamma_endpoints": [float(self.schedule[0]), float(self.schedule[-1])],
            "guidance": self.guidance,
            "sampler": "ddim" if self.state.step_bindings.noise == SIZE(-1).value else "ddpm",
        }


def pinned_chat_template(checkpoint):
    from jinja2.sandbox import SandboxedEnvironment

    config = (checkpoint / "tokenizer_config.json").read_bytes()
    if hashlib.sha256(config).hexdigest() != TOKENIZER_CONFIG_SHA256:
        raise ValueError("Sigma tokenizer configuration hash mismatch")
    return SandboxedEnvironment().from_string(json.loads(config)["chat_template"])


def build_session(args, folder, *, chat=False):
    import safetensors
    from tokenizers import Tokenizer
    import torch

    if not os.environ.get("CK_BLAS_LIBRARY"):
        raise ValueError("Set CK_BLAS_LIBRARY to an LP64 BLAS library exporting dgemm_")
    torch.set_num_threads(args.threads)
    tokenizer_path = args.checkpoint / "tokenizer.json"
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Sigma tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    template = pinned_chat_template(args.checkpoint) if chat else None
    graph, cfg, schema = build_circuit(args.prompt_capacity, args.block, cache_capacity=args.cache,
                                      score_temperature_milli=args.score_temperature_milli if args.sampler == "ddim" else None,
                                      **GEOMETRY, blas=True)
    weights = pinned_weights(args.checkpoint, schema, torch, safetensors)
    payloads = {name: (value.view(torch.uint16) if schema[name][1] == "BF16" else value).contiguous().numpy()
                for name, value in weights.items()}
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    name = "gamma_bounds.endpoint_values"
    with safetensors.safe_open(args.checkpoint / index["weight_map"][name], framework="pt", device="cpu") as src:
        clean, noisy = src.get_tensor(name).tolist()
    schedule = np.linspace(noisy, clean, args.steps + 1, dtype=np.float64)

    _, arena, _, library = compile_checked_weight_graph(
        folder, graph, payloads, schema, source_config=cfg, target_isa=args.isa,
        runtime_sources=("src/runtime/checked_diffusion_session.c", "src/runtime/checked_diffusion_schedule.c"),
        extra_exports=("ck_generate_seeded_diffusion",),
    )
    return SigmaTextSession(folder, arena, library, GEOMETRY, tokenizer, schedule, args.guidance,
                            chat_template=template)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", action="append", help="repeat for several requests; omit for line-oriented REPL")
    parser.add_argument("--prompt-capacity", type=int, default=32)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--cache", type=int, default=64)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--score-temperature-milli", type=int, default=300, help="DDIM score temperature in thousandths")
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--isa", choices=("portable", "avx2"), default="avx2")
    parser.add_argument("--chat", action="store_true", help="render one user turn using the pinned chat template")
    parser.add_argument("--system", help="system message for --chat")
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--json", action="store_true", help="emit JSON lines with token IDs and timings")
    parser.add_argument("--build-root", type=Path, help="retain generated artifacts under a new directory here")
    args = parser.parse_args()
    if not 1 <= args.steps <= 128 or args.threads < 1 or not math.isfinite(args.guidance):
        parser.error("expected 1–128 steps, positive threads and finite guidance")

    if args.system is not None and not args.chat:
        parser.error("--system requires --chat")

    interrupted = busy = False

    def interrupt(_signal, _frame):
        nonlocal interrupted
        if not busy:
            raise KeyboardInterrupt
        interrupted = True

    def run(folder):
        nonlocal interrupted, busy
        start = time.perf_counter()
        session = build_session(args, Path(folder), chat=args.chat)
        print(f"Sigma loaded in {time.perf_counter() - start:.2f}s", file=sys.stderr, flush=True)
        prior_handler = signal.signal(signal.SIGINT, interrupt)
        try:
            def repl():
                while True:
                    if sys.stdin.isatty():
                        print("sigma> ", end="", file=sys.stderr, flush=True)
                    line = sys.stdin.readline()
                    if not line or line.rstrip("\n") == ":quit":
                        return
                    yield line.rstrip("\n")

            for prompt in args.prompt if args.prompt is not None else repl():
                interrupted = False
                busy = True
                try:
                    result = session.generate(prompt, seed=args.seed, budget=args.budget,
                                              stop_strings=args.stop, should_cancel=lambda: interrupted,
                                              system=args.system)
                    if args.sampler == "ddim":
                        result["score_temperature_milli"] = args.score_temperature_milli
                    print(json.dumps(result, ensure_ascii=False) if args.json else result["text"], flush=True)
                except (ValueError, RuntimeError) as exc:
                    if args.prompt is not None:
                        raise
                    print(str(exc), file=sys.stderr, flush=True)
                finally:
                    busy = False
        finally:
            signal.signal(signal.SIGINT, prior_handler)

    try:
        if args.build_root is None:
            with tempfile.TemporaryDirectory(prefix="cke-sigma-") as folder:
                run(folder)
        else:
            args.build_root.mkdir(parents=True, exist_ok=True)
            folder = tempfile.mkdtemp(prefix="cke-sigma-", dir=args.build_root)
            print(f"Sigma artifacts at {folder}", file=sys.stderr, flush=True)
            run(folder)
    except KeyboardInterrupt:
        parser.exit(130)
    except (OSError, ImportError, ValueError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
