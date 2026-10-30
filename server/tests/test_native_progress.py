"""The native progress snapshot changes only after real generated work."""

import ctypes
import subprocess
import threading
import time

from fastapi.testclient import TestClient

from server.live import create_app
from server.session_v8 import SessionError, SessionV8


MODEL = r'''
#include "ck_model_abi_v8.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static int32_t ids[] = {1, 2};
static float logits[4] = {0, 1, 0, 0};
static atomic_int hold, entered;
static const atomic_int *cancel_flag;
static void (*progress)(void *, int32_t, int32_t, int32_t, int32_t);
static void *progress_user;
static int active;
static const uint64_t caps = CK_MODEL_CAP_INIT | CK_MODEL_CAP_AUTOREGRESSIVE_DECODE |
    CK_MODEL_CAP_TEXT_ENCODE | CK_MODEL_CAP_TOKEN_DECODE;
uint32_t ck_model_get_abi_version(void) { return CK_MODEL_ABI_V8_VERSION; }
uint64_t ck_model_get_capabilities(void) { return caps; }
int ck_model_get_runtime_descriptor(CKModelRuntimeDescriptorV8 *out, size_t size) {
    if (!out || size < sizeof(*out)) return -1;
    *out = (CKModelRuntimeDescriptorV8){.struct_size=sizeof(*out),
        .abi_version=CK_MODEL_ABI_V8_VERSION, .capabilities=caps,
        .artifact_role=CK_MODEL_ROLE_DECODER, .context_length=16, .vocab_size=4};
    return 0;
}
int ck_model_init(const char *weights) { (void)weights; return 0; }
void ck_model_free(void) {}
void ck_model_set_cancel_flag(const atomic_int *flag) { cancel_flag = flag; }
#ifndef CK_NO_PROGRESS
void ck_model_set_progress_callback(
    void (*callback)(void *, int32_t, int32_t, int32_t, int32_t), void *user) {
    progress = callback; progress_user = user;
}
#endif
int ck_model_can_encode_text(void) { return 1; }
int ck_model_encode_text(const char *text, int length) {
    (void)text; (void)length; return 2;
}
const int32_t *ck_model_get_token_buffer(void) { return ids; }
int ck_model_decode_tokens(const int32_t *tokens, int count, char *text, int capacity) {
    (void)tokens; (void)count;
    if (capacity < 2) return -1;
    text[0] = 'x'; text[1] = 0; return 1;
}
int ck_model_get_context_window(void) { return 16; }
int ck_model_get_vocab_size(void) { return 4; }
int ck_model_get_logits_stride(void) { return 0; }
float *ck_model_get_logits(void) { return logits; }
int ck_model_get_active_tokens(void) { return active; }
int ck_model_kv_cache_enable(int count) { (void)count; return 0; }
void ck_model_kv_cache_reset(void) { active = 0; }
int ck_model_embed_tokens(const int32_t *tokens, int count) {
    if (!tokens || count != 2) return -1;
    if (progress) progress(progress_user, 1, 2, count, 0);
    atomic_store(&entered, 1);
    while (atomic_load(&hold)) {}
    if (cancel_flag && atomic_load(cancel_flag)) return -2;
    if (progress) progress(progress_user, 2, 2, count, 0);
    active = 1; return 0;
}
int ck_model_forward(float *output) { (void)output; return 0; }
int ck_model_decode(int32_t token, float *output) {
    (void)token; (void)output; return 0;
}
void ck_fake_hold(int value) { atomic_store(&entered, 0); atomic_store(&hold, value); }
int ck_fake_entered(void) { return atomic_load(&entered); }
'''


def test_progress_tracks_completed_layers_cancellation_and_reuse(tmp_path):
    source = tmp_path / "fake.c"
    source.write_text(MODEL)
    model_path = tmp_path / "libmodel.so"
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    subprocess.run(["cc", "-shared", "-fPIC", "-std=c11", "-I", str(root / "include"),
                    str(source), "-o", str(model_path)], check=True)
    (tmp_path / "weights.bump").write_bytes(b"")
    (tmp_path / "weights_manifest.map").write_text("")
    model = ctypes.CDLL(str(model_path))
    model.ck_fake_hold.argtypes = [ctypes.c_int]
    session = SessionV8.open(tmp_path, context_length=16, num_threads=1)
    try:
        assert session.progress() == {
            "request_serial": 0, "phase": "idle", "completed_layers": 0,
            "total_layers": 0, "chunk_tokens": 0,
            "chunk_start_position": 0,
        }
        model.ck_fake_hold(1)
        results = []
        worker = threading.Thread(target=lambda: results.append(session.generate(
            None, "hello", max_tokens=1, temperature=0, top_p=1,
            on_token=lambda *_: 0, flags=1)))
        worker.start()
        try:
            deadline = time.monotonic() + 3
            while not model.ck_fake_entered() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert model.ck_fake_entered()
            assert session.progress() == {
                "request_serial": 1, "phase": "prefill", "completed_layers": 1,
                "total_layers": 2, "chunk_tokens": 2,
                "chunk_start_position": 0,
            }
            session.cancel()
        finally:
            model.ck_fake_hold(0)
            worker.join(timeout=3)
        assert not worker.is_alive()
        assert results[0]["stop_reason"] == 3
        assert session.progress()["phase"] == "complete"

        results.clear()
        session.generate(None, "hello", max_tokens=1, temperature=0, top_p=1,
                         on_token=lambda *_: 0, flags=1)
        assert session.progress() == {
            "request_serial": 2, "phase": "complete", "completed_layers": 2,
            "total_layers": 2, "chunk_tokens": 2,
            "chunk_start_position": 0,
        }
        app = create_app(session, model="m", chat_template="{{ messages[0].content }}")
        response = TestClient(app).get("/v1/cke/native-progress")
        assert response.status_code == 200
        assert response.json()["completed_layers"] == 2
    finally:
        session.close()


def test_older_generated_model_reports_unavailable_layer_progress(tmp_path):
    source = tmp_path / "fake.c"
    source.write_text(MODEL)
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    model_path = tmp_path / "libmodel.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-std=c11", "-DCK_NO_PROGRESS",
                    "-I", str(root / "include"), str(source), "-o", str(model_path)],
                   check=True)
    (tmp_path / "weights.bump").write_bytes(b"")
    (tmp_path / "weights_manifest.map").write_text("")
    model = ctypes.CDLL(str(model_path))
    model.ck_fake_hold.argtypes = [ctypes.c_int]
    session = SessionV8.open(tmp_path, context_length=16, num_threads=1)
    result = []
    model.ck_fake_hold(1)
    worker = threading.Thread(target=lambda: result.append(session.generate(
        None, "hello", max_tokens=1, temperature=0, top_p=1,
        on_token=lambda *_: 0, flags=1)))
    worker.start()
    try:
        deadline = time.monotonic() + 3
        while not model.ck_fake_entered() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert model.ck_fake_entered()
        progress = session.progress()
        assert progress["phase"] == "prefill"
        assert progress["total_layers"] == 0
        assert progress["completed_layers"] == 0
        session.cancel()
    finally:
        model.ck_fake_hold(0)
        worker.join(timeout=3)
        session.close()
    assert not worker.is_alive()
    assert result[0]["stop_reason"] == 3


def test_progress_reader_holds_native_session_alive_until_close(tmp_path):
    source = tmp_path / "fake.c"
    source.write_text(MODEL)
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    model_path = tmp_path / "libmodel.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-std=c11", "-I", str(root / "include"),
                    str(source), "-o", str(model_path)], check=True)
    (tmp_path / "weights.bump").write_bytes(b"")
    (tmp_path / "weights_manifest.map").write_text("")
    session = SessionV8.open(tmp_path, context_length=16, num_threads=1)
    entered_native = threading.Event()
    release_native = threading.Event()
    close_waiting = threading.Event()
    native_closed = threading.Event()
    original_get = session.lib.ck_session_v8_get_progress
    original_close = session.lib.ck_session_v8_close

    class SignallingLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if threading.current_thread().name == "close-session":
                close_waiting.set()
            self.lock.acquire()
            return self

        def __exit__(self, *_):
            self.lock.release()

    def held_get(*args):
        entered_native.set()
        assert release_native.wait(3), "progress test did not release native getter"
        return original_get(*args)

    def observed_close(*args):
        native_closed.set()
        return original_close(*args)

    session._lifecycle_lock = SignallingLock()
    session.lib.ck_session_v8_get_progress = held_get
    session.lib.ck_session_v8_close = observed_close
    progress_result = []
    reader = threading.Thread(target=lambda: progress_result.append(session.progress()))
    closer = threading.Thread(target=session.close, name="close-session")
    try:
        reader.start()
        assert entered_native.wait(3)
        closer.start()
        assert close_waiting.wait(3)
        assert not native_closed.is_set(), "close freed a session held by progress"
    finally:
        release_native.set()
        reader.join(timeout=3)
        if closer.ident is not None:
            closer.join(timeout=3)
        session.close()
    assert not reader.is_alive() and not closer.is_alive()
    assert native_closed.is_set()
    assert progress_result[0]["phase"] == "idle"
    try:
        session.progress()
    except SessionError as exc:
        assert exc.status == -6
    else:
        raise AssertionError("progress accepted a closed native session")
