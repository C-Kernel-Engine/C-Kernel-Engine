"""Run pinned Drama 3 through generated C on a persistent text backbone."""

import argparse
import ctypes
import json
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
import wave

import numpy as np

from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.tts.build_drama3_bf16_codebook_frame import build_circuit, build_prefill_circuit
from version.v8.tts.build_drama3_codec_waveform_circuit import build_circuit as codec_circuit
from version.v8.tts.build_drama3_feedback_circuit import build_circuit as feedback_circuit
from version.v8.tts.capture_drama3_two_frame_reference import CHECKPOINT_REVISION
from version.v8.tts.drama3_native_assets import TOWER_SETTINGS, load_codec, load_speech_weights

FP = ctypes.POINTER(ctypes.c_float)
IP = ctypes.POINTER(ctypes.c_int32)
WIDTH, VOCAB, CODES = 2048, 252288, 4096
SAMPLE_RATE, SAMPLES_PER_FRAME = 44100, 2048
STRUCTURAL = ("<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>",
              "<|semantic:", "<|speaker")


class Cancelled(RuntimeError):
    """A request stopped at a generated-entry boundary."""


def prompt_ids(tokenizer, text):
    if not isinstance(text, str) or not text.strip() or any(mark in text for mark in STRUCTURAL):
        raise ValueError("provide nonempty plain text without Drama prompt-control tokens")
    segments = ("<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n",
                "<|im_start|>user\n", f"<|speaker:0|> {text}", "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>")
    return np.asarray([token for segment in segments for token in tokenizer.encode(segment).ids], dtype=np.int32)


def speech_token(logits, begin, end, eos, history):
    scores = logits[begin:end + 1].copy()
    if not np.isfinite(scores).all() or not np.isfinite(logits[eos]):
        raise RuntimeError("nonfinite speech candidate scores")
    if begin + int(scores.argmax()) in history:
        scores[np.asarray(history, dtype=np.int32) - begin] = -np.inf
    index = int(scores.argmax())
    return eos if logits[eos] >= scores[index] else begin + index


def bf16_bits(values):
    bits = values.view(np.uint32)
    if np.any(bits & np.uint32(0xffff)):
        raise RuntimeError("expected widened BF16 activation")
    return (bits >> 16).astype(np.uint16)


class Backbone:
    def __init__(self, bundle, weights):
        self.model = ctypes.CDLL(str(bundle / "libmodel.so"))
        api = {
            "init_with_manifest": ([ctypes.c_char_p, ctypes.c_char_p], ctypes.c_int),
            "embed_tokens": ([IP, ctypes.c_int], ctypes.c_int),
            "forward": ([FP], ctypes.c_int),
            "decode_prepared_embedding": ([FP, ctypes.c_size_t, FP, ctypes.c_size_t], ctypes.c_int),
            "get_named_activation_nbytes": ([ctypes.c_char_p], ctypes.c_ssize_t),
            "get_named_activation_ptr": ([ctypes.c_char_p], ctypes.c_size_t),
            "get_context_window": ([], ctypes.c_int),
            "get_active_tokens": ([], ctypes.c_int),
            "get_vocab_size": ([], ctypes.c_int),
            "kv_cache_reset": ([], None),
            "free": ([], None),
        }
        for name, (args, result) in api.items():
            fn = getattr(self.model, "ck_model_" + name)
            fn.argtypes, fn.restype = args, result
        if self.model.ck_model_get_vocab_size() != VOCAB:
            raise ValueError("bundle vocabulary does not match the pinned Drama backbone")
        status = self.model.ck_model_init_with_manifest(
            os.fsencode(weights / "weights.bump"), os.fsencode(bundle / "weights_manifest.map"))
        if status:
            self.model.ck_model_free()
            raise RuntimeError(f"native backbone initialization failed ({status})")
        self.capacity = self.model.ck_model_get_context_window()
        self.logits = np.empty(VOCAB, dtype=np.float32)

    def hidden(self, rows):
        name = b"embedded_input"
        count = rows * WIDTH
        pointer = self.model.ck_model_get_named_activation_ptr(name)
        if not pointer or self.model.ck_model_get_named_activation_nbytes(name) < count * 4:
            raise RuntimeError("native backbone hidden output is too small")
        return np.ctypeslib.as_array(ctypes.cast(pointer, FP), shape=(count,)).reshape(rows, WIDTH).copy()

    def prefill(self, tokens):
        self.model.ck_model_kv_cache_reset()
        if (self.model.ck_model_embed_tokens(tokens.ctypes.data_as(IP), len(tokens)) or
                self.model.ck_model_get_active_tokens() != len(tokens) or
                self.model.ck_model_forward(self.logits.ctypes.data_as(FP))):
            raise RuntimeError("native backbone prefill failed")
        return self.hidden(len(tokens))

    def decode(self, embedding):
        previous = self.model.ck_model_get_active_tokens()
        status = self.model.ck_model_decode_prepared_embedding(
            embedding.ctypes.data_as(FP), embedding.size, self.logits.ctypes.data_as(FP), self.logits.size)
        if status or self.model.ck_model_get_active_tokens() != previous + 1:
            raise RuntimeError(f"native backbone continuation failed ({status})")
        return self.hidden(1)


class Session:
    def __init__(self, checkpoint, bundle, weights, build_dir):
        from tokenizers import Tokenizer

        self.build_dir = build_dir
        self.weights, self.tables, feedback = load_speech_weights(checkpoint)
        self.codec_weights, self.rotary = load_codec(checkpoint)
        self.tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
        config = json.loads((checkpoint / "config.json").read_text())
        self.begin, self.end, self.eos = (
            config[name] for name in ("semantic_start_token_id", "semantic_end_token_id", "eos_token_id"))
        manifest = json.loads((weights / "weights_manifest.json").read_text())
        entry, = (entry for entry in manifest["entries"] if entry["name"] == "token_emb")
        if (manifest["config"].get("artifact_scope") != "fish_omni_text_backbone" or
                entry["dtype"] != "bf16" or entry["shape"] != [VOCAB, WIDTH]):
            raise ValueError("converted text weights do not match pinned Drama geometry")
        self.token_embedding = np.memmap(weights / "weights.bump", dtype=np.uint16, mode="r",
                                         offset=entry["file_offset"], shape=(VOCAB, WIDTH))
        self.backbone = Backbone(bundle, weights)
        try:
            self.feedback = self.compile("feedback", feedback_circuit(), {})
            self.feedback[2]("codebook_embeddings", np.uint16)[:] = feedback
            self.tower_capacity = min(self.backbone.capacity, 256)
            graph = build_circuit(prefix_rows=self.tower_capacity, **TOWER_SETTINGS)
            self.tower = self.compile("tower", graph, self.weights, target_isa="avx2")
            self.fill_tables(self.tower[2], prefill=False)
        except BaseException:
            self.backbone.model.ck_model_free()
            raise
        self.lock = Lock()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        with self.lock:
            if not self.closed:
                self.backbone.model.ck_model_free()
                self.closed = True

    def compile(self, name, circuit, weights, **policy):
        graph, config, *ports = circuit
        declarations = ports[0] if ports else {
            key: (list(value.shape), "F32") for key, value in weights.items()
        }
        folder = self.build_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        return compile_checked_weight_graph(folder, graph, {key: weights[key] for key in declarations},
                                            declarations, source_config=config, **policy)

    def fill_tables(self, view, *, prefill):
        for label, table in self.tables.items():
            dest = view(f"{label}_embedding_table", np.uint16)
            if prefill:
                dest[:-1] = table
                dest[-1].fill(0)
            else:
                dest[:] = table

    def prefill_tower(self, hidden):
        rows = len(hidden) - 1
        settings = {key: value for key, value in TOWER_SETTINGS.items() if key != "residual_vocab"}
        run, arena, view, _ = self.compile(
            f"prefill-{rows}", build_prefill_circuit(rows, **settings), self.weights, target_isa="avx2")
        view("hidden", np.uint16)[:] = bf16_bits(hidden[:-1])
        view("unseen_sum", np.uint16).fill(0)
        view("positions", np.int32)[:] = np.arange(rows)
        positions = np.arange(1, rows + 1, dtype=np.int32)
        view("ranges", np.int32)[:] = np.column_stack((np.zeros_like(positions), positions, positions, positions))
        view("code_rows", np.int32).fill(CODES)
        for index in range(1, TOWER_SETTINGS["residual_layers"]):
            view(f"residual{index}_code_rows", np.int32).fill(CODES)
        self.fill_tables(view, prefill=True)
        if run(arena, len(arena)):
            raise RuntimeError("generated tower prefill failed")
        for label in self.tables:
            for plane, name in (("key", "k_rotary"), ("value", "v")):
                target = self.tower[2](f"{label}_prefix_{plane}", np.uint16)
                target[:rows] = view(f"{label}_{name}", np.uint16)
                target[rows:].fill(0)

    def frame(self, hidden, unseen, semantic, position):
        run, arena, view, _ = self.tower
        view("hidden", np.uint16)[:] = bf16_bits(hidden)
        view("unseen_sum", np.uint16)[:] = unseen
        view("positions", np.int32)[0] = position
        view("code_rows", np.int32)[0] = semantic - self.begin
        capacity = self.tower_capacity
        view("ranges", np.int32)[0] = (0, position, capacity, capacity + 1)
        if run(arena, len(arena)):
            raise RuntimeError("generated tower frame failed")
        for label in self.tables:
            for plane, name in (("key", "k_rotary"), ("value", "v")):
                view(f"{label}_prefix_{plane}", np.uint16)[position] = view(f"{label}_{name}", np.uint16)[0]
        codes = view("frame_codes", np.int32).copy()
        if codes.shape != (10,) or np.any((codes < 0) | (codes >= CODES)):
            raise RuntimeError("generated tower returned invalid codes")
        return codes

    def prepared_row(self, token, codes):
        run, arena, view, _ = self.feedback
        view("text_embedding", np.uint16)[:] = self.token_embedding[token]
        for index in range(1, 10):
            view(f"row{index}", np.int32)[0] = index * CODES + codes[index]
        if run(arena, len(arena)):
            raise RuntimeError("generated acoustic feedback failed")
        return view("backbone_embedding"), view("tower_enrichment", np.uint16)

    def waveform(self, frames, check_cancel):
        count = len(frames)
        if not count:
            return np.empty(0, dtype=np.float32), 0.0
        run, arena, view, _ = self.compile(f"codec-{count}", codec_circuit(count), self.codec_weights,
                                           openmp=True, provider_warnings="report", c_standard="gnu11")
        check_cancel()
        view("rotary_cos")[:] = self.rotary[:count, :, 0]
        view("rotary_sin")[:] = self.rotary[:count, :, 1]
        for index, codes in enumerate(np.asarray(frames, dtype=np.int32).T):
            view(f"ids{index}", np.int32)[:] = np.minimum(codes, 1023) if index else codes
        started = perf_counter()
        if run(arena, len(arena)):
            raise RuntimeError("generated codec failed")
        elapsed = perf_counter() - started
        check_cancel()
        samples = view("pcm")[0].copy()
        if len(samples) != count * SAMPLES_PER_FRAME or not np.isfinite(samples).all():
            raise RuntimeError("generated codec returned invalid PCM")
        return samples, elapsed

    def synthesize(self, text, max_frames, *, stop_requested=lambda: False):
        def check_cancel():
            if stop_requested():
                raise Cancelled("Drama request cancelled")

        tokens = prompt_ids(self.tokenizer, text)
        if (type(max_frames) is not int or max_frames < 1 or len(tokens) < 2 or
                len(tokens) + max_frames > self.tower_capacity or max_frames > len(self.rotary)):
            raise ValueError("prompt and frame budget exceed the built session capacity")
        with self.lock:
            if self.closed:
                raise RuntimeError("Drama session is closed")
            check_cancel()
            started = perf_counter()
            hidden = self.backbone.prefill(tokens)
            check_cancel()
            self.prefill_tower(hidden)
            prefilled = perf_counter()
            unseen = np.zeros((1, WIDTH), dtype=np.uint16)
            hidden = hidden[-1:]
            history, frames = [], []
            finish = "length"
            for index in range(max_frames):
                check_cancel()
                token = speech_token(self.backbone.logits, self.begin, self.end, self.eos, history[-16:])
                if token == self.eos:
                    finish = "eos"
                    break
                codes = self.frame(hidden, unseen, token, len(tokens) - 1 + index)
                history.append(token)
                frames.append(codes)
                if index + 1 < max_frames:
                    check_cancel()
                    embedding, unseen = self.prepared_row(token, codes)
                    hidden = self.backbone.decode(embedding)
            check_cancel()
            generated = perf_counter()
            samples, codec_seconds = self.waveform(frames, check_cancel)
            check_cancel()
            return samples, {
                "checkpoint_revision": CHECKPOINT_REVISION, "text": text, "text_tokens": len(tokens),
                "finish_reason": finish, "sampling": "greedy_with_nonrepeating_retry",
                "semantic_ids": history, "code_frames": [row.tolist() for row in frames],
                "sample_rate": SAMPLE_RATE, "samples": len(samples),
                "prefill_seconds": prefilled - started, "generation_seconds": generated - prefilled,
                "codec_seconds": codec_seconds, "waveform_stage_seconds": perf_counter() - generated,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "bundle", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--text", required=True, action="append")
    parser.add_argument("--max-frames", type=int, default=96)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with Session(args.checkpoint, args.bundle, args.weights, args.output / "build") as session:
        for index, text in enumerate(args.text, 1):
            samples, report = session.synthesize(text, args.max_frames)
            stem = args.output / f"request-{index}"
            temporary = stem.with_suffix(".wav.tmp")
            try:
                with wave.open(str(temporary), "wb") as writer:
                    writer.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
                    writer.writeframes(np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
                temporary.replace(stem.with_suffix(".wav"))
            finally:
                temporary.unlink(missing_ok=True)
            stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{stem}.wav ({report['samples']} samples, {report['finish_reason']})", flush=True)


if __name__ == "__main__":
    main()
