"""Execute table GELU with disjoint workers, preserving serial stored bits."""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class GGMLGeluParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("gcc"):
            print("TEST SKIPPED: requires GCC")
            raise unittest.SkipTest("requires GCC")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.threads = min(4, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                          else (os.cpu_count() or 1))
        cls.output = Path(cls.tmp.name) / "gelu.so"
        source = '#include "' + str(ROOT / 'src/kernels/gelu_kernels.c') + '"\n' + r'''
#include <stdatomic.h>
#define EXPORT __attribute__((visibility("default")))
static int width = 1;
int ck_get_num_threads(void) { return width; }
int ck_get_physical_cores(void) { return 4; }
EXPORT void configure(int n) {
    ck_threadpool_global_destroy();
    width = n;
    setenv("CK_NUM_THREADS", "4", 1);
    setenv("CK_THREADPOOL_CAPACITY", "4", 1);
}
EXPORT void reference(float *data, size_t n) { gelu_ggml_serial(data, n); }
EXPORT void backend(int external) {
    pthread_once(&ck_gelu_ggml_runtime_once, ck_gelu_ggml_runtime_init);
    pthread_once(&ck_gelu_ggml_table_once, ck_gelu_ggml_table_init);
    /* Exercise the bound-table branch without depending on an installed oracle. */
    ck_gelu_runtime_table_f16 = ck_gelu_ggml_table_f16;
    ck_gelu_runtime_fp32_to_fp16 = ggml_fp32_to_fp16;
    ck_gelu_runtime_fp16_to_fp32 = ggml_fp16_to_fp32;
    ck_gelu_runtime_ready = external;
}
EXPORT uint64_t execute(float *data, size_t n) {
    ck_threadpool_t *pool = ck_threadpool_global();
    ck_threadpool_profile_reset(pool);
    gelu_ggml_inplace(data, n);
    ck_threadpool_profile_t profile;
    ck_threadpool_profile_snapshot(pool, &profile);
    return profile.dispatch_count;
}
typedef struct { float *data; size_t n; atomic_int bad; } outer_args;
static void outer(int ith, int nth, void *opaque) {
    outer_args *args = opaque;
    if (!ck_threadpool_in_dispatch(ck_threadpool_global()))
        atomic_store(&args->bad, 1);
    gelu_ggml_inplace(args->data + (size_t)ith * args->n, args->n);
}
EXPORT int nested(float *data, size_t n) {
    ck_threadpool_t *pool = ck_threadpool_global();
    if (ck_threadpool_in_dispatch(pool)) return 2;
    outer_args args = {.data = data, .n = n};
    atomic_init(&args.bad, 0);
    ck_threadpool_profile_reset(pool);
    ck_threadpool_dispatch_n(pool, width, outer, &args);
    ck_threadpool_profile_t p;
    ck_threadpool_profile_snapshot(pool, &p);
    return atomic_load(&args.bad) || p.dispatch_count != 1 || ck_threadpool_in_dispatch(pool);
}
'''
        subprocess.run([
            "gcc", "-x", "c", "-", str(ROOT / "src/ck_threadpool.c"),
            "-shared", "-fPIC", "-O3", "-fno-fast-math", "-ffp-contract=off",
            "-ffunction-sections", "-fdata-sections", "-fvisibility=hidden",
            "-Wl,--gc-sections", "-I" + str(ROOT / "include"),
            "-pthread", "-lm", "-ldl", "-o", str(cls.output),
        ], input=source, text=True, capture_output=True, check=True, timeout=120)
        cls.lib = ctypes.CDLL(str(cls.output))
        cls.lib.configure.argtypes = [ctypes.c_int]
        cls.lib.backend.argtypes = [ctypes.c_int]
        cls.lib.reference.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        cls.lib.execute.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        cls.lib.execute.restype = ctypes.c_uint64
        cls.addClassCleanup(cls.lib.configure, 1)

    def test_bits_guards_tails_and_dispatch(self):
        self.lib.backend(0)
        rng = np.random.default_rng(1127)
        special = np.array([-np.inf, -10, -0., 0., 10, np.inf, np.nan,
                            -1.00048828125, 1.00048828125], dtype=np.float32)
        for threads in range(1, self.threads + 1):
            self.lib.configure(threads)
            for n in (0, 1, 19, 65535, 65536, 65537, 131079):
                with self.subTest(threads=threads, n=n):
                    storage = np.full(n + 32, -9182.25, dtype=np.float32)
                    data = storage[16:-16]
                    data[:] = rng.uniform(-11, 11, n).astype(np.float32)
                    data[:min(n, len(special))] = special[:min(n, len(special))]
                    expected = data.copy()
                    self.lib.reference(expected.ctypes.data, n)
                    calls = self.lib.execute(data.ctypes.data, n)
                    np.testing.assert_array_equal(data.view(np.uint32), expected.view(np.uint32))
                    self.assertEqual(calls, int(threads > 1 and n >= 65536))
                    self.assertTrue(np.all(storage[:16] == -9182.25))
                    self.assertTrue(np.all(storage[-16:] == -9182.25))
        self.assertEqual(self.lib.execute(None, 65536), 0)

    def test_all_fp16_inputs_and_repeated_calls(self):
        self.lib.backend(0)
        original = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
        expected = original.copy()
        self.lib.reference(expected.ctypes.data, expected.size)
        self.lib.configure(self.threads)
        for _ in range(3):
            actual = original.copy()
            self.assertEqual(self.lib.execute(actual.ctypes.data, actual.size), int(self.threads > 1))
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_bound_table_backend_uses_identical_bytes(self):
        self.lib.backend(1)
        self.lib.configure(self.threads)
        try:
            data = np.linspace(-12, 12, 131079, dtype=np.float32)
            expected = data.copy()
            self.lib.reference(expected.ctypes.data, expected.size)
            self.assertEqual(self.lib.execute(data.ctypes.data, data.size), int(self.threads > 1))
            np.testing.assert_array_equal(data.view(np.uint32), expected.view(np.uint32))
        finally:
            self.lib.backend(0)

    def test_nested_main_and_worker_callbacks_do_not_redispatch(self):
        code = r'''
import ctypes, sys, os
import numpy as np
lib = ctypes.CDLL(sys.argv[1])
threads = min(4, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1))
lib.configure(threads)
for name in ('nested', 'reference'):
    getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t]
n = 65539
x = np.linspace(-11, 11, threads*n, dtype=np.float32)
expected = x.copy()
lib.reference(expected.ctypes.data, expected.size)
assert lib.nested(x.ctypes.data, n) == 0
np.testing.assert_array_equal(x.view(np.uint32), expected.view(np.uint32))
'''
        import sys
        subprocess.run([sys.executable, "-c", code, str(self.output)], check=True,
                       capture_output=True, timeout=30, cwd=self.tmp.name)


if __name__ == "__main__":
    unittest.main()
