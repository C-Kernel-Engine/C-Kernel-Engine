"""Fresh-process native checkpoint replay; test worker, not a model scheduler."""

import ctypes
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
import run_training_workflow_v8 as workflow


def run(request_path):
    request = json.loads(Path(request_path).read_text())
    library = Path(request["library"])
    if hashlib.sha256(library.read_bytes()).hexdigest() != request["library_sha256"]:
        raise ValueError("generated library identity mismatch")
    lib = ctypes.CDLL(str(library))
    workflow._configure(lib)
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int32)
    sizes = np.asarray(request["sizes"], dtype="i4")
    initial = np.zeros(int(sizes.sum()), dtype="f4")
    lib.ck_train_init.argtypes = [fp, ip, ctypes.c_int]
    if lib.ck_train_init(
        initial.ctypes.data_as(fp), sizes.ctypes.data_as(ip), len(sizes)
    ) != len(sizes):
        raise RuntimeError("native initialization failed")
    del initial
    try:
        lib.ck_set_strict_parity(1)
        workflow._checkpoint_load(Path(request["checkpoint"]), lib, request["metadata"])
        ids = np.zeros(request["tokens"], dtype="i4")
        metadata = np.asarray(request["integer_inputs"], dtype="i4")
        lib.ck_train_step_with_i32_ex.argtypes = [
            ip,
            ip,
            ctypes.c_int,
            ip,
            ctypes.c_size_t,
            fp,
            ctypes.c_float,
        ]
        losses = []
        for _ in range(request["remaining_steps"]):
            loss = ctypes.c_float()
            rc = lib.ck_train_step_with_i32_ex(
                ids.ctypes.data_as(ip),
                ids.ctypes.data_as(ip),
                len(ids),
                metadata.ctypes.data_as(ip),
                len(metadata),
                ctypes.byref(loss),
                request["learning_rate"],
            )
            if rc <= 0:
                raise RuntimeError(f"native resumed step failed: {rc}")
            losses.append(loss.value)
        result = {"losses": losses, "optimizer_step": lib.ck_train_get_opt_step()}
        for name in ("weight", "optimizer_state"):
            values = workflow._array_export(lib, name)
            result[name + "_sha256"] = hashlib.sha256(
                values.astype("<f4", copy=False).tobytes()
            ).hexdigest()
        Path(request["result"]).write_text(json.dumps(result, sort_keys=True))
    finally:
        lib.ck_train_free()


if __name__ == "__main__":
    run(sys.argv[1])
