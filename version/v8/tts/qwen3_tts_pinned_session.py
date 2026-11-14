#!/usr/bin/env python3
"""Persistent, bounded Qwen3-TTS generated execution for checked text lengths.

The host imports weights, prepares text IDs and writes PCM. The native checked
frame-codec shell invokes the circuit-generated prefill, frame and codec entries.
The fixed seven-token path and optional 1..16-token path retain separate
reference evidence. Unsupported or budget-exhausting requests publish no PCM.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import numbers
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "version/v8/scripts"))

from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph
from version.v8.tts.build_qwen3_tts_loop_codec_circuit import build_circuit
from version.v8.tts.build_qwen3_tts_runtime_loop_codec_circuit import (
    build_circuit as build_runtime_circuit,
)
from version.v8.tts.build_qwen3_tts_prompt_first_block_circuit import prompt_plan
from version.v8.tts.build_qwen3_tts_talker_predictor_frame_circuit import bridge_plan
from version.v8.tts.capture_qwen3_tts_customvoice_oracle import (
    DEFAULT_TEXT, MODEL_REVISION, verify_model_dir,
)

CODE_FRAMES = 46
CODEBOOKS = 16
MAX_FRAMES = 64
SAMPLES = 88320
SAMPLE_RATE = 24000
CONTEXT_TOKENS = 7
PROMPT_ROWS = 18
RUNTIME_CONTEXT_CAPACITY = 16

# Native execution shells are not neural providers. Arithmetic source closure
# is obtained from the resolved kernel maps by the shared bundle compiler.
RUNTIME_SOURCES = ("src/runtime/checked_frame_loop.c", "src/runtime/checked_frame_codec.c")


class FrameLoop(ctypes.Structure):
    _fields_ = [
        ("prefill", ctypes.c_void_p), ("frame", ctypes.c_void_p),
        ("arena", ctypes.POINTER(ctypes.c_uint8)),
        ("arena_bytes", ctypes.c_size_t),
        ("step_index_offset", ctypes.c_size_t),
        ("prefill_valid_code_count_offset", ctypes.c_size_t),
        ("frame_valid_code_count_offset", ctypes.c_size_t),
        ("prefill_code_offsets", ctypes.POINTER(ctypes.c_size_t)),
        ("frame_code_offsets", ctypes.POINTER(ctypes.c_size_t)),
        ("codebooks", ctypes.c_size_t), ("max_frames", ctypes.c_size_t),
        ("staging_codes", ctypes.POINTER(ctypes.c_int32)),
        ("staging_elements", ctypes.c_size_t),
        ("output_codes", ctypes.POINTER(ctypes.c_int32)),
        ("output_elements", ctypes.c_size_t),
        ("valid_frames", ctypes.POINTER(ctypes.c_size_t)),
    ]


class FrameCodec(ctypes.Structure):
    _fields_ = [
        ("generation", ctypes.POINTER(FrameLoop)), ("codec", ctypes.c_void_p),
        ("codec_frame_major_offset", ctypes.c_size_t),
        ("expected_codebooks", ctypes.c_size_t),
        ("expected_code_frames", ctypes.c_size_t),
        ("waveform_offset", ctypes.c_size_t),
        ("waveform_valid_samples_offset", ctypes.c_size_t),
        ("expected_samples", ctypes.c_size_t),
        ("output_samples", ctypes.POINTER(ctypes.c_float)),
        ("output_capacity", ctypes.c_size_t),
        ("valid_samples", ctypes.POINTER(ctypes.c_size_t)),
    ]


class FrameCodecPrefix(ctypes.Structure):
    _fields_ = [("base", FrameCodec), ("samples_per_code_frame", ctypes.c_size_t)]


CancelCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class FrameCodecPrefixControl(ctypes.Structure):
    _fields_ = [("base", FrameCodecPrefix),
                ("should_cancel", CancelCallback),
                ("cancel_context", ctypes.c_void_p)]


class SynthesisCancelled(RuntimeError):
    """The current generated request stopped at a checked boundary."""


def _weight_shapes(graph, main, codec):
    refs = {name for block in graph["sequence"]
            for op in graph["block_types"][block].get("body", {}).get("ops", [])
            for name in op.get("weight_refs", {}).values()}
    shapes = {}
    for name in refs:
        if name.startswith("codec."):
            if name.endswith("effective_embedding"):
                shape = (2048, 256)
            elif name.endswith("output_projection.weight"):
                shape = (512, 256)
            else:
                shape = codec.get_slice(name.removeprefix("codec.")).get_shape()
        else:
            shape = main.get_slice(name).get_shape()
        shapes[name] = tuple(shape)
    return shapes


def _codec_value(source, name):
    if name == "codec.semantic.effective_embedding":
        group, index = "rvq_first", 0
    elif name.startswith("codec.acoustic.") and name.endswith(".effective_embedding"):
        group, index = "rvq_rest", int(name.split(".")[2])
    else:
        group = None
    if group is not None:
        prefix = f"decoder.quantizer.{group}.vq.layers.{index}._codebook."
        usage = source.get_tensor(prefix + "cluster_usage").float().numpy()
        summed = source.get_tensor(prefix + "embedding_sum").float().numpy()
        return np.ascontiguousarray(
            (summed / np.maximum(usage, 1e-5)[:, None]).astype(np.float32))
    if name == "codec.semantic.output_projection.weight":
        source_name = "decoder.quantizer.rvq_first.output_proj.weight"
    elif name == "codec.acoustic.output_projection.weight":
        source_name = "decoder.quantizer.rvq_rest.output_proj.weight"
    else:
        source_name = name.removeprefix("codec.")
    return np.ascontiguousarray(source.get_tensor(source_name).float().numpy().reshape(
        (512, 256) if name.endswith("output_projection.weight") else
        source.get_slice(source_name).get_shape()))


def _compile(folder: Path, graph: dict, shapes: dict):
    folder.mkdir(parents=True, exist_ok=True)
    circuit = folder / "circuit.json"
    circuit.write_text(json.dumps(graph), encoding="utf-8")
    entries = []
    offset = 4096
    for name, shape in shapes.items():
        offset = (offset + 63) & ~63
        size = int(np.prod(shape)) * 4
        entries.append({"name": name, "shape": list(shape), "dtype": "F32",
                        "size": size, "file_offset": offset})
        offset += size
    request_rows = graph["contract"]["runtime_invariants"].get("prompt_rows", 1)
    source = {"config": {
        "model": graph["name"], "arch": graph["name"],
        "num_layers": 1, "embed_dim": 1024, "num_heads": 16,
        "num_kv_heads": 8, "head_dim": 128, "intermediate_size": 3072,
        "context_length": request_rows, "max_seq_len": request_rows,
        "vocab_size": 151936,
        "activation_buffer_dtypes": {
            name: "i32" for name in (
                "text_ids", "codec_ids", "left_indices", "right_indices",
                "step_index_input", "step_index_checked", "position_checked",
                "request_context_tokens", "valid_context_tokens",
                "valid_text_rows", "valid_prompt_rows",
                "valid_q_norm_rows", "valid_k_norm_rows",
                "valid_hidden_elements", "valid_mlp_elements",
                "published_prompt_rows", "frame_context_tokens",
                "frame_prompt_rows", "frame_prompt_rows_value",
                "prompt_rows_verified")},
    }, "entries": entries, "quant_summary": {}, "template": graph}
    layout, call_ir, library, _, _ = compile_native_graph(
        folder, source, graph,
        extra_exports=("ck_run_checked_frame_codec_prefix_control",
                       "audio_conv1d_checked_avx2_fma_available"),
        runtime_sources=RUNTIME_SOURCES, openmp=True, fp_contract="fast",
        provider_warnings="report", c_standard="gnu11")
    acts = {entry["name"]: entry for entry in
            layout["memory"]["activations"]["buffers"]}
    weights = {entry["name"]: entry for entry in
               layout["memory"]["weights"]["entries"]}
    return acts, weights, layout, call_ir, library


def _view(arena, acts, name, dtype, shape):
    return np.ndarray(shape, dtype=dtype, buffer=arena,
                      offset=acts[name]["abs_offset"])


def _prompt_ids(model_dir: Path, text: str, tokenizer=None,
                context_tokens: int | None = CONTEXT_TOKENS):
    if not isinstance(text, str) or not text.strip() or any(ord(c) < 32 for c in text):
        raise ValueError("text must be nonempty and contain no control characters")
    if "<|" in text or "|>" in text:
        raise ValueError("special-token syntax is unsupported in spoken text")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    context = tokenizer.encode(text, add_special_tokens=False)
    prefix = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    if len(prefix) != 3:
        raise ValueError("pinned role prefix must encode to exactly 3 tokens")
    if context_tokens is None:
        if not 1 <= len(context) <= RUNTIME_CONTEXT_CAPACITY:
            raise ValueError(f"text must encode to 1..{RUNTIME_CONTEXT_CAPACITY} tokens")
    elif len(context) != context_tokens:
        raise ValueError(f"text must encode to exactly {context_tokens} tokens")
    config = json.loads((model_dir / "config.json").read_text())
    talker = config["talker_config"]
    text_values = [
        *prefix, config["tts_pad_token_id"], config["tts_bos_token_id"],
        config["tts_eos_token_id"], *context,
    ]
    codec_values = [
        talker["codec_think_id"], talker["codec_think_bos_id"],
        talker["codec_language_id"]["english"],
        talker["codec_think_eos_id"], talker["spk_id"]["ryan"],
        talker["codec_pad_id"], talker["codec_bos_id"],
    ]
    for label, values in (("text", text_values), ("codec", codec_values)):
        if any(isinstance(value, (bool, np.bool_)) or
               not isinstance(value, numbers.Integral) or
               not 0 <= value <= np.iinfo(np.int32).max for value in values):
            raise ValueError(f"{label} token IDs must be nonnegative int32 values")
    text_ids = np.asarray(text_values, dtype=np.int32)
    codec_ids = np.asarray(codec_values, dtype=np.int32)
    return text_ids, codec_ids


def _set_inputs(arena, acts, text_ids, codec_ids):
    context_tokens = len(text_ids) - 6
    prompt_rows = context_tokens + 11
    _view(arena, acts, "text_ids", np.int32, (len(text_ids),))[:] = text_ids
    _view(arena, acts, "codec_ids", np.int32, (7,))[:] = codec_ids
    for name, indices in prompt_plan(context_tokens).items():
        _view(arena, acts, name, np.int32, (len(indices),))[:] = indices
    for name, indices in bridge_plan(prompt_rows).items():
        _view(arena, acts, "bridge_" + name, np.int32, (2,))[:] = indices
    if "request_context_tokens" in acts:
        _view(arena, acts, "request_context_tokens", np.int32, (1,))[0] = context_tokens
    for name in ("pred_feedback_step_index_input", "feedback_left_indices",
                 "talker_step_index_input", "second_feedback_step_index_input"):
        _view(arena, acts, name, np.int32, (1,))[0] = 0
    _view(arena, acts, "feedback_right_indices", np.int32, (1,))[0] = 3
    _view(arena, acts, "second_bridge_left_indices", np.int32, (2,))[:] = [0, -1]
    _view(arena, acts, "second_bridge_right_indices", np.int32, (2,))[:] = [-1, 0]


def write_wav(path: Path, samples: np.ndarray):
    if (samples.ndim != 1 or not 0 < samples.size <= SAMPLES or
            samples.size % (SAMPLES // CODE_FRAMES) or
            not np.isfinite(samples).all()):
        raise ValueError("invalid generated PCM extent or values")
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(pcm.tobytes())


class PinnedQwenSession:
    """Keep one generated library and one weight arena loaded across calls."""

    def __init__(self, model_dir: Path, build_dir: Path,
                 provider: str = "avx2_fma", *, runtime_prompts: bool = False):
        from safetensors import safe_open
        from transformers import AutoTokenizer

        self.model_dir = Path(model_dir)
        self.model_hashes = verify_model_dir(self.model_dir)
        self._runtime_prompts = runtime_prompts
        circuit_builder = build_runtime_circuit if runtime_prompts else build_circuit
        options = {"frames": CODE_FRAMES, "max_generated_frames": MAX_FRAMES,
                   "codec_conv_provider": provider}
        if runtime_prompts:
            options["context_capacity"] = RUNTIME_CONTEXT_CAPACITY
        self.graph = circuit_builder(**options)
        started = time.monotonic()
        with safe_open(self.model_dir / "model.safetensors", framework="pt", device="cpu") as main, \
             safe_open(self.model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as codec:
            shapes = _weight_shapes(self.graph, main, codec)
            compile_started = time.monotonic()
            self.acts, weights, self.layout, calls, library = _compile(
                Path(build_dir), self.graph, shapes)
            self.compile_seconds = time.monotonic() - compile_started
            if len(calls["entry"]["segments"]) != 3:
                raise ValueError("generated Qwen entry must have three segments")
            self.arena_bytes = self.layout["memory"]["arena"]["total_size"]
            self.activation_base = self.layout["memory"]["arena"]["activations_base"]
            if max(item["abs_offset"] + item["size"] for item in weights.values()) > \
                    self.activation_base:
                raise ValueError("planned activations overlap weights")
            raw = (ctypes.c_uint8 * (self.arena_bytes + 63))()
            self._raw = raw
            self.arena = (ctypes.c_uint8 * self.arena_bytes).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for name, item in weights.items():
                value = (_codec_value(codec, name) if name.startswith("codec.")
                         else np.ascontiguousarray(main.get_tensor(name).float().numpy()))
                if value.shape != shapes[name] or value.nbytes != item["size"]:
                    raise ValueError(f"model weight disagrees with planned storage: {name}")
                self.arena[item["abs_offset"]:item["abs_offset"] + value.nbytes] = value.tobytes()
        self.library_path = library
        self.library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
        self.build_identity_sha256 = hashlib.sha256((library.parent / "build-identity.json").read_bytes()).hexdigest()
        from version.v8.scripts.checked_bundle_evidence_v8 import load_verified_checked_bundle
        self._native = load_verified_checked_bundle(
            library, library.parent / "build-identity.json", self.build_identity_sha256)
        if provider == "avx2_fma":
            available = self._native.audio_conv1d_checked_avx2_fma_available
            available.argtypes, available.restype = [], ctypes.c_int
            if not available():
                raise RuntimeError("AVX2/FMA codec provider is unavailable on this CPU")
        self._segments = []
        for segment in self.graph["native_entry"]["segments"]:
            fn = getattr(self._native, segment["function"])
            fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
            fn.restype = ctypes.c_int
            self._segments.append(fn)
        self._run = self._native.ck_run_checked_frame_codec_prefix_control
        self._run.argtypes = [ctypes.POINTER(FrameCodecPrefixControl)]
        self._run.restype = ctypes.c_int
        self._cancel_requested = threading.Event()
        self._cancel_callback = CancelCallback(
            lambda _context: int(self._cancel_requested.is_set()))
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        self.text_ids, self.codec_ids = _prompt_ids(
            self.model_dir, DEFAULT_TEXT, self._tokenizer)
        self.load_seconds = time.monotonic() - started
        self.load_count = 1
        self._lock = threading.Lock()

    def synthesize(self, text: str) -> tuple[np.ndarray, dict]:
        with self._lock:
            self._cancel_requested.clear()
            text_ids, codec_ids = _prompt_ids(
                self.model_dir, text, self._tokenizer,
                context_tokens=None if self._runtime_prompts else CONTEXT_TOKENS)
            context_tokens = len(text_ids) - 6
            samples, metric = self._synthesize_checked(text_ids, codec_ids)
            metric["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            metric["text_tokens"] = context_tokens
            metric["prompt_rows"] = context_tokens + 11
            return samples, metric

    def stop(self) -> None:
        """Stop the active request at its next checked native boundary."""
        self._cancel_requested.set()

    def _synthesize_checked(self, text_ids, codec_ids) -> tuple[np.ndarray, dict]:
        # The planner places all mutable activations after immutable weights.
        # This also clears recurrent/KV state before each request.
        ctypes.memset(ctypes.addressof(self.arena) + self.activation_base, 0,
                      self.arena_bytes - self.activation_base)
        _set_inputs(self.arena, self.acts, text_ids, codec_ids)
        first = ["talker_first_code_id", "pred_predictor_codebook0_selected_id"] + [
            f"pred_step{i}_selected_id" for i in range(1, 15)]
        later = ["second_talker_first_code_id",
                 "second_predictor_codebook0_selected_id"] + [
            f"second_step{i}_selected_id" for i in range(1, 15)]
        first_offsets = (ctypes.c_size_t * CODEBOOKS)(
            *(self.acts[name]["abs_offset"] for name in first))
        later_offsets = (ctypes.c_size_t * CODEBOOKS)(
            *(self.acts[name]["abs_offset"] for name in later))
        staging = np.full(MAX_FRAMES * CODEBOOKS, -901, np.int32)
        codes = np.full(MAX_FRAMES * CODEBOOKS, -901, np.int32)
        samples = np.full(SAMPLES + 64, -901, np.float32)
        frames = ctypes.c_size_t(0)
        valid = ctypes.c_size_t(0)
        loop = FrameLoop(
            ctypes.cast(self._segments[0], ctypes.c_void_p),
            ctypes.cast(self._segments[1], ctypes.c_void_p),
            self.arena, self.arena_bytes,
            self.acts["talker_step_index_input"]["abs_offset"],
            self.acts["pred_predictor_valid_code_count"]["abs_offset"],
            self.acts["second_predictor_valid_code_count"]["abs_offset"],
            first_offsets, later_offsets, CODEBOOKS, MAX_FRAMES,
            staging.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(staging),
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(codes),
            ctypes.pointer(frames))
        request = FrameCodecPrefixControl(FrameCodecPrefix(FrameCodec(
            ctypes.pointer(loop), ctypes.cast(self._segments[2], ctypes.c_void_p),
            self.acts["loop_frame_major_codes"]["abs_offset"],
            CODEBOOKS, CODE_FRAMES,
            self.acts["codec_waveform_output"]["abs_offset"],
            self.acts["waveform_valid_samples"]["abs_offset"],
            SAMPLES,
            samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(samples), ctypes.pointer(valid)), SAMPLES // CODE_FRAMES),
            self._cancel_callback, None)
        started = time.monotonic()
        status = self._run(ctypes.byref(request))
        seconds = time.monotonic() - started
        if status == -7:
            if valid.value != 0:
                raise RuntimeError("cancelled synthesis published an output extent")
            raise SynthesisCancelled("native Qwen synthesis was cancelled")
        if status != 0 or not 0 < frames.value <= CODE_FRAMES or \
                valid.value != frames.value * (SAMPLES // CODE_FRAMES):
            if valid.value != 0:
                raise RuntimeError("failed synthesis published an output extent")
            raise RuntimeError(f"native Qwen synthesis failed: status={status}, frames={frames.value}")
        if not np.isfinite(samples[:valid.value]).all() or \
                np.any(samples[valid.value:] != -901):
            raise RuntimeError("native Qwen output or padding is invalid")
        return samples[:valid.value].copy(), {
            "frames": frames.value, "samples": valid.value,
            "code_sha256": hashlib.sha256(codes[:frames.value * CODEBOOKS].tobytes()).hexdigest(),
            "waveform_f32_sha256": hashlib.sha256(samples[:valid.value].tobytes()).hexdigest(),
            "library_sha256": self.library_sha256,
            "build_identity_sha256": self.build_identity_sha256,
            "load_count": self.load_count,
            "run_seconds": round(seconds, 3),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", required=True,
                        help="English text within the selected token-length contract")
    parser.add_argument("--runtime-prompts", action="store_true",
                        help="accept checked 1..16-token prompts through one generated library")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec-conv-provider", choices=("scalar", "avx2_fma"),
                        default="avx2_fma")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run the same request repeatedly on one loaded model")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 8:
        parser.error("--repeat must be in 1..8")
    try:
        if args.runtime_prompts:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir),
                                                       local_files_only=True)
            _prompt_ids(args.model_dir, args.text, tokenizer, context_tokens=None)
        else:
            _prompt_ids(args.model_dir, args.text)
    except ValueError as exc:
        parser.error(str(exc))
    with tempfile.TemporaryDirectory(prefix="cke-qwen3tts-") as temp:
        session = PinnedQwenSession(args.model_dir, Path(temp),
                                    args.codec_conv_provider,
                                    runtime_prompts=args.runtime_prompts)
        print("QWEN3_TTS_PINNED_LOAD " + json.dumps({
            "model_revision": MODEL_REVISION,
            "model_sha256": session.model_hashes["model.safetensors"],
            "library_sha256": session.library_sha256,
            "codec_conv_provider": args.codec_conv_provider,
            "runtime_prompts": args.runtime_prompts,
            "arena_bytes": session.arena_bytes,
            "compile_seconds": round(session.compile_seconds, 3),
            "cold_load_seconds": round(session.load_seconds, 3),
            "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        }, sort_keys=True), flush=True)
        hashes = []
        for _ in range(args.repeat):
            samples, metric = session.synthesize(args.text)
            hashes.append(metric["waveform_f32_sha256"])
            metric["peak_rss_mib"] = round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
            print("QWEN3_TTS_PINNED_SESSION " + json.dumps(metric, sort_keys=True),
                  flush=True)
        if len(set(hashes)) != 1:
            raise RuntimeError("repeated requests produced different PCM")
        write_wav(args.output, samples)
        print("QWEN3_TTS_PINNED_WAV " + json.dumps({
            "path": str(args.output), "samples": len(samples),
            "sample_rate_hz": SAMPLE_RATE,
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
