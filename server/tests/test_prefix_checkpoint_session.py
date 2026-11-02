"""Python binding checks for caller-owned, aligned prefix snapshots."""

import ctypes as C

import pytest

from server.session_v8 import SequenceCheckpointV8, SessionError, SessionV8


class NativeFunction:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def test_checkpoint_prefill_binding_preserves_ids_alignment_and_native_errors():
    calls = []

    def requirements(_session, size, alignment):
        C.cast(size, C.POINTER(C.c_size_t))[0] = 96
        C.cast(alignment, C.POINTER(C.c_size_t))[0] = 64
        return 0

    def prefill(_session, ids, count, reserved, flags):
        calls.append((tuple(ids[:count]), reserved, flags))
        return -1 if ids[0] == 9999 else 0

    def contract(_session, minimum, alignment, group):
        C.cast(minimum, C.POINTER(C.c_int32))[0] = 64
        C.cast(alignment, C.POINTER(C.c_int32))[0] = 64
        C.cast(group, C.POINTER(C.c_int32))[0] = 1
        return 0

    class Library:
        ck_session_v8_prefill_contract = NativeFunction(contract)
        ck_session_v8_checkpoint_requirements = NativeFunction(requirements)
        ck_session_v8_checkpoint_save = NativeFunction(
            lambda _session, address, size: 0 if address.value % 64 == 0 and size == 96 else -1)
        ck_session_v8_checkpoint_restore = NativeFunction(
            lambda _session, address, size: 0 if address.value % 64 == 0 and size == 96 else -1)
        ck_session_v8_prefill_ids = NativeFunction(prefill)
        ck_session_v8_last_error = NativeFunction(lambda _session: b"invalid token")

    session = SessionV8()
    session.lib = Library()
    session.session = C.c_void_p(1)
    size, alignment = session.checkpoint_requirements()
    assert session.prefill_contract() == (64, 64, 1)
    snapshot = SequenceCheckpointV8(size, alignment)
    assert snapshot.address % alignment == 0
    assert snapshot.nbytes == 96
    session.checkpoint_save(snapshot)
    session.checkpoint_restore(snapshot)
    session.prefill_ids([1, 2, 3, 4], reserved_tokens=5)
    session.prefill_ids([5, 6, 7, 8], reserved_tokens=2, continue_state=True)
    assert calls == [((1, 2, 3, 4), 5, 0), ((5, 6, 7, 8), 2, 1)]
    with pytest.raises(SessionError, match="invalid token"):
        session.prefill_ids([9999, 2], reserved_tokens=1)
    with pytest.raises(ValueError, match="invalid checkpoint prefill"):
        session.prefill_ids([1, 2], reserved_tokens=1 << 40)
    with pytest.raises(ValueError, match="invalid checkpoint prefill"):
        session.prefill_ids([1 << 40], reserved_tokens=1)
    assert len(calls) == 3
    with pytest.raises(ValueError, match="invalid checkpoint"):
        SequenceCheckpointV8(96, 3)
