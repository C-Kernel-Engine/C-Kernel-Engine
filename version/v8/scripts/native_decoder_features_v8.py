"""Read final token features from a generated decoder's declared native ABI.

The host submits tokens, never layers. Export selection comes from resolved call
IR; no model-name dispatch or oracle tensor input is accepted here. One owner per
loaded library is required because the current decoder ABI owns global state.
"""

import ctypes as C
from contextlib import contextmanager
import json
import threading
from pathlib import Path
import numpy as np

_Decode = C.CFUNCTYPE(C.c_int, C.c_int32, C.POINTER(C.c_float))
_Reset = C.CFUNCTYPE(None)
_GetInt = C.CFUNCTYPE(C.c_int)
_Cancel = C.CFUNCTYPE(C.c_int, C.c_void_p)


class NativeDecoderFeatures(C.Structure):
    """Borrowed ABI 1 descriptor matching ck_decoder_features.h.

    Function pointers refer directly to the loaded generated library, never
    Python callbacks. The owner must remain locked and loaded while C uses it.
    """

    _fields_ = [
        ("abi_version", C.c_uint32),
        ("channels", C.c_size_t),
        ("features", C.POINTER(C.c_float)),
        ("feature_elements", C.c_size_t),
        ("decode", _Decode),
        ("reset", _Reset),
        ("active_tokens", _GetInt),
        ("context_window", _GetInt),
        ("vocabulary_size", _GetInt),
        ("should_cancel", _Cancel),
        ("cancel_context", C.c_void_p),
    ]


def final_feature_contract(call):
    if call.get("errors") or call.get("mode") != "decode":
        raise ValueError("resolved decode call IR required")
    ops = call["operations"]
    matches = [
        (i, o)
        for i, o in enumerate(ops)
        if o.get("section") == "footer" and o.get("op") == "final_rmsnorm"
    ]
    if len(matches) != 1:
        raise ValueError("one declared final normalization required")
    index, op = matches[0]
    outputs = [
        a
        for a in op["args"]
        if a.get("source", "").startswith("output:") and a.get("buffer_ref")
    ]
    if len(outputs) != 1:
        raise ValueError("ambiguous feature output")
    name = outputs[0]["buffer_ref"]
    buffers = {b["name"]: b for b in call["memory"]["activations"]["buffers"]}
    b = buffers[name]
    if b.get("dtype") != "fp32":
        raise ValueError("feature export requires declared FP32 storage")
    widths = [int(a["expr"]) for a in op["args"] if a["name"] == "d_model"]
    strides = [int(a["expr"]) for a in op["args"] if a["name"] == "aligned_embed_dim"]
    if (
        len(widths) != 1
        or widths != strides
        or widths[0] <= 0
        or b["size"] < 4 * widths[0]
    ):
        raise ValueError("unsupported feature row geometry")
    # Later writes must not overwrite the selected row before host publication.
    later = []
    for later_op in ops[index + 1 :]:
        for arg in later_op.get("args", []):
            if arg.get("source", "").startswith("output:") or (
                arg.get("buffer_ref")
                and not arg.get("expr", "").lstrip().startswith("(const ")
            ):
                if not arg.get("buffer_ref"):
                    raise ValueError("unresolved later output")
                later.append(arg["buffer_ref"])
    return dict(
        name=name,
        channels=widths[0],
        later_outputs=sorted(set(later)),
        provider=op.get("call_abi", {}).get("kernel_id"),
    )


class DecoderFeatures:
    _owners = set()
    _owners_lock = threading.Lock()

    def __init__(self, folder):
        self.lock = threading.RLock()
        self.owner_key = None
        self.native_borrowed = False
        self.folder = Path(folder)
        self.closed = True
        call = json.loads((self.folder / "lowered_decode_call.json").read_text())
        self.contract = final_feature_contract(call)
        self.lib = C.CDLL(str(self.folder / "libmodel.so"))
        self.lib.ck_model_init_with_manifest.argtypes = [C.c_char_p, C.c_char_p]
        self.lib.ck_model_init_with_manifest.restype = C.c_int
        self.lib.ck_model_decode.argtypes = [C.c_int32, C.POINTER(C.c_float)]
        self.lib.ck_model_decode.restype = C.c_int
        self.lib.ck_model_kv_cache_reset.argtypes = []
        self.lib.ck_model_kv_cache_reset.restype = None
        self.lib.ck_model_free.argtypes = []
        self.lib.ck_model_free.restype = None
        for name in (
            "ck_model_get_context_window",
            "ck_model_get_vocab_size",
            "ck_model_get_active_tokens",
        ):
            getattr(self.lib, name).argtypes = []
            getattr(self.lib, name).restype = C.c_int
        for name, dtype in (
            ("ck_model_get_named_activation_ptr", C.c_size_t),
            ("ck_model_get_named_activation_nbytes", C.c_ssize_t),
        ):
            fn = getattr(self.lib, name)
            fn.argtypes = [C.c_char_p]
            fn.restype = dtype
        key = C.cast(self.lib.ck_model_decode, C.c_void_p).value
        with self._owners_lock:
            if key in self._owners:
                raise RuntimeError("generated decoder already has an owner")
            self._owners.add(key)
            self.owner_key = key
        if (
            self.lib.ck_model_init_with_manifest(
                str(self.folder / "weights.bump").encode(),
                str(self.folder / "weights_manifest.map").encode(),
            )
            != 0
        ):
            # Init may have allocated only part of its native state. The owner
            # is still marked closed, so release explicitly before unregistering.
            try:
                self.lib.ck_model_free()
            finally:
                self.close()
            raise RuntimeError("native backbone initialization failed")
        self.closed = False
        try:
            self.capacity = self.lib.ck_model_get_context_window()
            self.vocabulary = self.lib.ck_model_get_vocab_size()
            name = self.contract["name"].encode()
            self.address = self.lib.ck_model_get_named_activation_ptr(name)
            self.channels = self.contract["channels"]
            if (
                not self.address
                or self.lib.ck_model_get_named_activation_nbytes(name)
                < 4 * self.channels
            ):
                raise ValueError("native feature extent unavailable")
            for output in self.contract["later_outputs"]:
                ptr = self.lib.ck_model_get_named_activation_ptr(output.encode())
                size = self.lib.ck_model_get_named_activation_nbytes(output.encode())
                if (
                    not ptr
                    or size <= 0
                    or (
                        ptr < self.address + 4 * self.channels
                        and self.address < ptr + size
                    )
                ):
                    raise ValueError("feature output lifetime is not preserved")
        except BaseException:
            self.close()
            raise

    @contextmanager
    def borrow_native_descriptor(self):
        """Hold exclusive ownership for one generated component invocation.

        The descriptor cannot be retained, serialized or used after this scope.
        It exposes existing native entry points; it runs no token/layer loop.
        """
        with self.lock:
            if self.closed or self.native_borrowed:
                raise RuntimeError("decoder closed or already borrowed")
            descriptor = NativeDecoderFeatures(
                1,
                self.channels,
                C.cast(self.address, C.POINTER(C.c_float)),
                self.channels,
                C.cast(self.lib.ck_model_decode, _Decode),
                C.cast(self.lib.ck_model_kv_cache_reset, _Reset),
                C.cast(self.lib.ck_model_get_active_tokens, _GetInt),
                C.cast(self.lib.ck_model_get_context_window, _GetInt),
                C.cast(self.lib.ck_model_get_vocab_size, _GetInt),
                _Cancel(),
                None,
            )
            self.native_borrowed = True
            try:
                yield descriptor
            finally:
                self.native_borrowed = False

    def encode(self, ids):
        with self.lock:
            if self.native_borrowed:
                raise RuntimeError("decoder already borrowed")
            return self._encode(ids)

    def _encode(self, ids):
        if self.closed:
            raise RuntimeError("decoder closed")
        if (
            not isinstance(ids, (list, tuple))
            or not 1 <= len(ids) <= self.capacity
            or any(type(x) is not int or not 0 <= x < self.vocabulary for x in ids)
        ):
            raise ValueError("invalid bounded token request")
        result = np.empty((len(ids), self.channels), dtype="float32")
        self.lib.ck_model_kv_cache_reset()
        try:
            if self.lib.ck_model_get_active_tokens() != 0:
                raise RuntimeError("reset did not clear position")
            for i, token in enumerate(ids):
                if (
                    self.lib.ck_model_decode(token, None) != 0
                    or self.lib.ck_model_get_active_tokens() != i + 1
                ):
                    raise RuntimeError("native token execution failed")
                result[i] = np.ctypeslib.as_array(
                    (C.c_float * self.channels).from_address(self.address)
                )
                if not np.isfinite(result[i]).all():
                    raise RuntimeError("nonfinite native features")
            return result
        finally:
            self.lib.ck_model_kv_cache_reset()

    def close(self):
        with self.lock:
            if self.native_borrowed:
                raise RuntimeError("decoder already borrowed")
            if not self.closed:
                self.lib.ck_model_free()
                self.closed = True
            if self.owner_key is not None:
                with self._owners_lock:
                    self._owners.discard(self.owner_key)
                self.owner_key = None
