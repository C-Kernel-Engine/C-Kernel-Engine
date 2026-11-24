"""RWKV-7 channelmix: row-parallel dispatch must be bitwise identical.

Both GEMVs are split only by output row. Each row still runs the same
``ck_rwkv7_channelmix_{hidden,out}_rows`` primitive the serial provider uses,
so the parallel result is exactly equal -- no tolerance, no rounding caveat.

The generated model reaches the parallel provider through the
``CK_PARALLEL_DECODE`` macro override, which is checked by preprocessing an
actual call site.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "version" / "v8" / "src" / "ck_parallel_decode_v8.c"
THREADPOOL = ROOT / "src" / "ck_threadpool.c"
KERNEL = ROOT / "src" / "kernels" / "rwkv7_decode.c"

F32 = ctypes.c_float
P = ctypes.POINTER(F32)

CHANNELMIX_ARGS = [
    P, P, P, P, P, P, P,
    ctypes.c_int, ctypes.c_int, P, ctypes.c_size_t,
]

# Real 1.5B shape: 2048 x 8192 channelmix. The dispatcher's safety and
# threshold decisions are size-sensitive, so test the production size.
DIM = 2048
HIDDEN = 8192
THREADS = 4

# Symbols the wrapper references for kernels this test never executes. They
# must resolve for ctypes (RTLD_NOW) to load the object.
UNUSED_KERNEL_PREFIXES = ("gemv_", "quantize_row_", "gated_deltanet_")


def _compiler() -> str:
    for candidate in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if candidate and shutil.which(candidate):
            return candidate
    pytest.skip("no C compiler available")


def _cc(compiler: str, args: list[str]) -> None:
    result = subprocess.run(
        [compiler, *args], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _undefined_symbols(compiler: str, obj: Path) -> set[str]:
    nm = shutil.which("nm")
    if nm is None:
        pytest.skip("nm is required to discover unused kernel symbols")
    result = subprocess.run(
        [nm, "-u", str(obj)], capture_output=True, text=True, check=True
    )
    symbols = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts:
            symbols.add(parts[-1])
    return symbols


def _build(tmp_path: Path) -> Path:
    compiler = _compiler()
    common = [
        "-std=c11", "-O3", "-march=native", "-fPIC",
        "-I", str(ROOT / "include"),
        "-I", str(ROOT / "version" / "v8" / "src"),
    ]
    objects = []
    for source in (KERNEL, THREADPOOL, WRAPPER):
        obj = tmp_path / (source.stem + ".o")
        _cc(compiler, [*common, "-c", str(source), "-o", str(obj)])
        objects.append(obj)

    unused = sorted(
        symbol
        for symbol in _undefined_symbols(compiler, tmp_path / "ck_parallel_decode_v8.o")
        if symbol.startswith(UNUSED_KERNEL_PREFIXES)
    )
    assert unused, "expected the wrapper to reference other GEMV kernels"

    stubs = tmp_path / "stubs.c"
    stubs.write_text(
        "#include <stddef.h>\n"
        f"int ck_get_num_threads(void) {{ return {THREADS}; }}\n"
        f"int ck_get_physical_cores(void) {{ return {THREADS}; }}\n"
        "int ck_topology_heterogeneous(void) { return 0; }\n"
        "int ck_topology_big_cpus(int *cpus, int max_cpus) {\n"
        "    (void)cpus; (void)max_cpus; return 0;\n"
        "}\n"
        "/* Never called: this test only exercises the channelmix provider. */\n"
        + "".join(f"void {symbol}(void) {{}}\n" for symbol in unused),
        encoding="utf-8",
    )
    stub_obj = tmp_path / "stubs.o"
    _cc(compiler, [*common, "-c", str(stubs), "-o", str(stub_obj)])

    library = tmp_path / "librwkv7_channelmix_parallel.so"
    _cc(
        compiler,
        [
            "-shared",
            "-o",
            str(library),
            *[str(obj) for obj in objects],
            str(stub_obj),
            "-lm",
            "-lpthread",
        ],
    )
    return library


@pytest.fixture(scope="module")
def lib(tmp_path_factory: pytest.TempPathFactory):
    os.environ["CK_NUM_THREADS"] = str(THREADS)
    path = _build(tmp_path_factory.mktemp("channelmix_parallel"))
    module = ctypes.CDLL(str(path))
    for name in (
        "ck_rwkv7_channelmix_decode",
        "ck_rwkv7_channelmix_decode_parallel",
    ):
        fn = getattr(module, name)
        fn.argtypes = CHANNELMIX_ARGS
        fn.restype = ctypes.c_int
    binding_ok = module.ck_rwkv7_channelmix_parallel_binding_ok
    binding_ok.argtypes = [P] * 8 + [ctypes.c_int, ctypes.c_int]
    binding_ok.restype = ctypes.c_int
    global_pool = module.ck_threadpool_global
    global_pool.restype = ctypes.c_void_p
    n_threads = module.ck_threadpool_n_threads
    n_threads.argtypes = [ctypes.c_void_p]
    n_threads.restype = ctypes.c_int
    handle = global_pool()
    if not handle or n_threads(handle) <= 1:
        pytest.skip("thread pool did not start with more than one worker")
    assert n_threads(handle) == THREADS
    yield module
    module.ck_threadpool_global_destroy()


def _array(values) -> ctypes.Array:
    return values.ctypes.data_as(P)


def _inputs(np, *, dim: int, hidden: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(dim, dtype=np.float32)
    shift = rng.standard_normal(dim, dtype=np.float32)
    mu = rng.random(dim, dtype=np.float32)
    wk = rng.standard_normal((hidden, dim), dtype=np.float32) * 0.2
    wv = rng.standard_normal((dim, hidden), dtype=np.float32) * 0.05
    return x, shift, mu, wk, wv


def _run(np, fn, x, shift, mu, wk, wv, *, shift_out=None, out=None):
    dim = x.shape[0]
    hidden = wk.shape[0]
    if shift_out is None:
        shift_out = np.zeros(dim, dtype=np.float32)
    if out is None:
        out = np.zeros(dim, dtype=np.float32)
    scratch = np.zeros(dim + hidden + 8, dtype=np.float32)
    rc = fn(
        _array(x), _array(shift), _array(mu),
        _array(wk), _array(wv),
        _array(shift_out), _array(out),
        dim, hidden, _array(scratch), scratch.size,
    )
    assert rc == 0
    return shift_out, out


def _bits(array):
    return array.view("uint32")


def _preprocess(tmp_path: Path, *, parallel_define: bool) -> str:
    compiler = _compiler()
    source = tmp_path / ("call_parallel.c" if parallel_define else "call_plain.c")
    source.write_text(
        ("#define CK_PARALLEL_DECODE 1\n" if parallel_define else "")
        + '#include "ck_parallel_decode_v8.h"\n'
        "int probe(const float *x, const float *s, const float *mu,\n"
        "          const float *wk, const float *wv, float *so, float *o,\n"
        "          int dim, int hidden, float *scratch, unsigned long n) {\n"
        "    return ck_rwkv7_channelmix_decode(\n"
        "        x, s, mu, wk, wv, so, o, dim, hidden, scratch, n);\n"
        "}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            compiler, "-E", "-P", "-std=c11",
            "-I", str(ROOT / "include"),
            "-I", str(ROOT / "version" / "v8" / "src"),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_macro_override_reaches_the_parallel_provider(tmp_path: Path) -> None:
    rewritten = _preprocess(tmp_path, parallel_define=True)
    assert "return ck_rwkv7_channelmix_decode_parallel(" in rewritten
    assert not re.search(r"return ck_rwkv7_channelmix_decode\(", rewritten)

    # Without the define, the generated call keeps using the serial provider.
    plain = _preprocess(tmp_path, parallel_define=False)
    assert "return ck_rwkv7_channelmix_decode(" in plain


def test_parallel_channelmix_is_bitwise_identical(lib) -> None:
    np = pytest.importorskip("numpy")
    x, shift, mu, wk, wv = _inputs(np, dim=DIM, hidden=HIDDEN)

    serial_shift, serial_out = _run(
        np, lib.ck_rwkv7_channelmix_decode, x, shift, mu, wk, wv
    )
    parallel_shift, parallel_out = _run(
        np, lib.ck_rwkv7_channelmix_decode_parallel, x, shift, mu, wk, wv
    )

    np.testing.assert_array_equal(_bits(parallel_shift), _bits(serial_shift))
    np.testing.assert_array_equal(_bits(parallel_out), _bits(serial_out))
    assert np.isfinite(parallel_out).all()


def test_parallel_binding_predicate_accepts_production_bindings(lib) -> None:
    np = pytest.importorskip("numpy")
    x, shift, mu, wk, wv = _inputs(np, dim=DIM, hidden=HIDDEN, seed=3)
    out = np.zeros(DIM, dtype=np.float32)
    scratch = np.zeros(DIM + HIDDEN + 8, dtype=np.float32)
    ok = lib.ck_rwkv7_channelmix_parallel_binding_ok

    def check(shift_out, out_ptr, scratch_ptr) -> bool:
        return bool(
            ok(
                _array(x), _array(shift), _array(mu), _array(wk), _array(wv),
                shift_out, out_ptr, scratch_ptr, DIM, HIDDEN,
            )
        )

    # The circuit's real binding: ffn_shift is read and written in place.
    assert check(_array(shift), _array(out), _array(scratch))
    # Both carries the serial provider documents.
    assert check(_array(x), _array(out), _array(scratch))
    assert check(_array(shift), _array(x), _array(scratch))
    # Scratch may not overlap anything, and the two outputs stay disjoint.
    assert not check(_array(shift), _array(out), _array(x))
    assert not check(_array(out), _array(out), _array(scratch))
    assert not check(_array(shift[1:]), _array(out), _array(scratch))


def test_parallel_channelmix_preserves_in_place_carry(lib) -> None:
    np = pytest.importorskip("numpy")
    x, shift, mu, wk, wv = _inputs(np, dim=DIM, hidden=HIDDEN, seed=11)

    # The circuit binds shift_out == shift_in in place. That carry is safe for
    # row splitting (shift_in is consumed by the serial prologue), so the
    # parallel provider runs and must still match the serial result exactly.
    serial_shift, serial_out = _run(
        np, lib.ck_rwkv7_channelmix_decode, x, shift, mu, wk, wv
    )
    inplace = shift.copy()
    parallel_shift, parallel_out = _run(
        np, lib.ck_rwkv7_channelmix_decode_parallel,
        x, inplace, mu, wk, wv, shift_out=inplace,
    )
    np.testing.assert_array_equal(_bits(parallel_shift), _bits(serial_shift))
    np.testing.assert_array_equal(_bits(parallel_out), _bits(serial_out))


def test_parallel_channelmix_rejects_bad_scratch_like_serial(lib) -> None:
    np = pytest.importorskip("numpy")
    x, shift, mu, wk, wv = _inputs(np, dim=64, hidden=128)
    scratch = np.zeros(8, dtype=np.float32)
    for name in (
        "ck_rwkv7_channelmix_decode",
        "ck_rwkv7_channelmix_decode_parallel",
    ):
        fn = getattr(lib, name)
        rc = fn(
            _array(x), _array(shift), _array(mu), _array(wk), _array(wv),
            _array(x.copy()), _array(np.zeros(64, dtype=np.float32)),
            64, 128, _array(scratch), scratch.size,
        )
        assert rc == -1, name
