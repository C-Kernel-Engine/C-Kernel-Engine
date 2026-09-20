from __future__ import annotations

import shutil
import subprocess
import importlib.util
import sys
import tempfile
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "version" / "v8" / "src" / "ck_audio_encoder_decoder_transcribe_v8.c"


def _load_certifier():
    path = ROOT / "version" / "v8" / "scripts" / "certify_cohere_generated_standalone_v8.py"
    spec = importlib.util.spec_from_file_location("cohere_standalone_certifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_native_host_builds_without_python_runtime() -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        return
    source = HOST.read_text(encoding="utf-8")
    assert "Python" not in source
    assert "system(" not in source
    assert "popen(" not in source
    assert "ck_model_run_audio_encoder" in source
    assert "ck_model_set_encoder_memory" in source
    assert "ck_model_kv_cache_reset" in source
    assert "ck_model_audio_prompt_token_id" in source
    assert "<|startoftranscript|>" not in source
    assert "tokens[i] ==" not in source
    assert "cke_audio_segments_v1" in source
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(HOST),
                "-ldl",
                "-o",
                str(Path(directory) / "cohere-transcribe"),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


def test_native_host_owns_multi_segment_schedule(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        return
    host = tmp_path / "cohere-transcribe"
    encoder_source = tmp_path / "encoder.c"
    decoder_source = tmp_path / "decoder.c"
    encoder_source.write_text(r'''
#include <stddef.h>
#include <stdint.h>
int ck_model_init_with_manifest(const char *w, const char *m) { (void)w; (void)m; return 0; }
void ck_model_free(void) {}
int ck_model_audio_sample_rate(void) { return 16000; }
int ck_model_audio_max_source_frames(void) { return 160; }
int ck_model_audio_hop_length(void) { return 16; }
int ck_model_audio_feature_channels(void) { return 1; }
int ck_model_audio_subsampling_factor(void) { return 1; }
int ck_model_audio_encoder_output_dim(void) { return 1; }
int ck_model_audio_encoder_frame_capacity(void) { return 11; }
int ck_model_prepare_audio_wav_features(const uint8_t *wav, size_t n, float *out,
    int cap, int *frames, void *scratch) {
    (void)wav; (void)n; (void)scratch;
    if (cap < 2) return -1;
    out[0] = 1.0f; out[1] = 2.0f; *frames = 2;
    return 0;
}
size_t ck_model_audio_encoder_workspace_bytes(int in, int out) {
    (void)in; (void)out; return 4;
}
int ck_model_run_audio_encoder(const float *in, int in_frames, int source_frames,
    float *out, int cap, int *frames, void *scratch, size_t bytes) {
    (void)in; (void)in_frames; (void)source_frames; (void)scratch; (void)bytes;
    if (cap < 1) return -1;
    out[0] = 3.0f; *frames = 1;
    return 0;
}
''')
    decoder_source.write_text(r'''
#include <stdint.h>
#include <stdio.h>
static int dirty = 1;
int ck_model_init_with_manifest(const char *w, const char *m) { (void)w; (void)m; return 0; }
void ck_model_free(void) {}
void ck_model_kv_cache_reset(void) { dirty = 0; }
int ck_model_set_encoder_memory(const float *x, int n, int d) {
    return (!x || n != 1 || d != 1 || dirty) ? -1 : 0;
}
int ck_model_get_encoder_memory_capacity(void) { return 11; }
int ck_model_get_encoder_memory_dim(void) { return 1; }
int ck_model_get_vocab_size(void) { return 4; }
int ck_model_audio_prompt_token_count(void) { return 1; }
int32_t ck_model_audio_prompt_token_id(int i) { return i == 0 ? 3 : -1; }
int ck_model_embed_tokens(const int32_t *x, int n) { return (!x || n != 1) ? -1 : 0; }
int ck_model_forward(float *x) {
    if (dirty) return -1;
    dirty = 1; x[0]=0; x[1]=2; x[2]=1; x[3]=0; return 0;
}
int ck_model_decode(int32_t token, float *x) {
    if (token != 1) return -1;
    x[0]=0; x[1]=0; x[2]=2; x[3]=0;
    return 0;
}
int ck_model_decode_tokens(const int32_t *x, int n, char *out, int cap) {
    return (!x || n != 1 || cap < 4) ? -1 : snprintf(out, (size_t)cap, " ok");
}
int ck_model_is_stop_token(int32_t token) { return token == 2; }
''')
    encoder = tmp_path / "encoder.so"
    decoder = tmp_path / "decoder.so"
    for command in (
        [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", str(HOST), "-ldl", "-o", str(host)],
        [compiler, "-shared", "-fPIC", "-std=c11", "-Wall", "-Wextra", "-Werror", str(encoder_source), "-o", str(encoder)],
        [compiler, "-shared", "-fPIC", "-std=c11", "-Wall", "-Wextra", "-Werror", str(decoder_source), "-o", str(decoder)],
    ):
        subprocess.run(command, check=True, capture_output=True, text=True)
    audio = tmp_path / "input.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 240)
    plan = tmp_path / "segments.txt"
    plan.write_text("cke_audio_segments_v1 16000 2\n0 120\n120 240\n")
    placeholder = tmp_path / "unused"
    completed = subprocess.run(
        [str(host), str(encoder), str(placeholder), str(placeholder),
         str(decoder), str(placeholder), str(placeholder), str(audio), str(plan)],
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == " ok ok\n"
    assert completed.stderr.count("window_token_ids[") == 2
    assert "completed_windows=2 source_frames=240 consumed_frames=240" in completed.stderr


def test_native_host_rejects_overlapping_segment_plan(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        return
    host = tmp_path / "cohere-transcribe"
    subprocess.run(
        [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", str(HOST),
         "-ldl", "-o", str(host)],
        check=True, capture_output=True, text=True,
    )
    audio = tmp_path / "input.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 240)
    plan = tmp_path / "segments.txt"
    plan.write_text("cke_audio_segments_v1 16000 2\n0 160\n120 240\n")
    completed = subprocess.run(
        [str(host), "missing", "missing", "missing", "missing", "missing",
         "missing", str(audio), str(plan)],
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode != 0
    assert "invalid native audio segment plan" in completed.stderr


def test_certifier_requires_explicit_native_token_trajectory() -> None:
    certifier = _load_certifier()
    assert certifier._tokens("noise\ntoken_ids=2,17,3\n") == [2, 17, 3]
    try:
        certifier._tokens("tokens=2,17,3\n")
    except ValueError as error:
        assert "token trajectory" in str(error)
    else:
        raise AssertionError("missing native token evidence was accepted")
