"""Preserve the pre-optimization attention softmax's stored values and sum."""
import ctypes
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class AttentionSoftmaxExactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if platform.machine() not in ("x86_64", "AMD64") or not shutil.which("gcc"):
            print("TEST SKIPPED: requires x86 GCC")
            raise unittest.SkipTest("requires x86 GCC")
        flags = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
        if not all(flag in flags for flag in ("avx512f", "avx512dq")):
            print("TEST SKIPPED: requires AVX-512F/DQ execution")
            raise unittest.SkipTest("requires AVX-512F/DQ execution")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        output = Path(cls.tmp.name) / "attention_softmax.so"
        source = '#include "' + str(ROOT / "src/kernels/attention_kernels.c") + '"\n' + r'''
#include <sys/mman.h>
#include <unistd.h>
/* Frozen old GCC AVX-512 path: same tree, cast locations and scalar tail. */
__attribute__((visibility("default"), noinline, optimize("O0")))
double reference(int n, float *y, const float *x, float maximum) {
    int i = 0;
    double sum = 0.0;
    for (; i + 15 < n; i += 16) {
        const __m512 val = ck_ggml_v_expf512(
            _mm512_sub_ps(_mm512_loadu_ps(x + i), _mm512_set1_ps(maximum)));
        _mm512_storeu_ps(y + i, val);
        sum += (double)_mm512_reduce_add_ps(val);
    }
    for (; i < n; ++i) {
        const float val = expf(x[i] - maximum);
        y[i] = val;
        sum += (double)val;
    }
    return sum;
}
__attribute__((visibility("default")))
double replay(int n, float *y, const float *x, float maximum) {
    return ck_ggml_vec_soft_max_row(n, y, x, maximum);
}
__attribute__((visibility("default")))
double repeated(int count, int candidate, int n, float *y, const float *x, float maximum) {
    double sum = 0;
    for (int i = 0; i < count; ++i)
        sum += candidate ? replay(n, y, x, maximum) : reference(n, y, x, maximum);
    return sum;
}
__attribute__((visibility("default")))
int guarded(void) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    char *mapping = mmap(NULL, 4*page, PROT_READ|PROT_WRITE,
                         MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) return 2;
    if (mprotect(mapping+page, page, PROT_NONE) ||
        mprotect(mapping+3*page, page, PROT_NONE)) { munmap(mapping, 4*page); return 3; }
    int result = 0;
    for (int n = 1; n <= 65; ++n) {
        float *x = (float *)(mapping+page) - n;
        float *y = (float *)(mapping+3*page) - n;
        float expected[65];
        for (int i = 0; i < n; ++i) x[i] = (float)(i%17 - 8)*0.125f;
        double old = reference(n, expected, x, 1.0f);
        double now = replay(n, y, x, 1.0f);
        if (memcmp(&old, &now, sizeof(old)) || memcmp(y, expected, n*sizeof(float))) result = 1;
    }
    munmap(mapping, 4*page);
    return result;
}
'''
        subprocess.run([
            "gcc", "-x", "c", "-", "-shared", "-fPIC", "-O3", "-march=native",
            "-ffp-contract=fast", "-fno-fast-math", "-ffunction-sections", "-fdata-sections",
            "-fvisibility=hidden", "-I" + str(ROOT / "include"), "-Wl,--gc-sections",
            "-lm", "-o", str(output),
        ], input=source, text=True, check=True, capture_output=True, timeout=120)
        cls.lib = ctypes.CDLL(str(output))
        for name in ("reference", "replay"):
            fn = getattr(cls.lib, name)
            fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float]
            fn.restype = ctypes.c_double
        cls.lib.guarded.argtypes = []
        cls.lib.guarded.restype = ctypes.c_int
        cls.lib.repeated.argtypes = [ctypes.c_int]*3 + [ctypes.c_void_p]*2 + [ctypes.c_float]
        cls.lib.repeated.restype = ctypes.c_double

    def check_row(self, x, maximum, inplace=False):
        before = x.tobytes()
        expected = np.empty_like(x)
        old = self.lib.reference(x.size, expected.ctypes.data, x.ctypes.data, maximum)
        storage = np.full(x.size + 2, 12345.25, dtype="f4")
        actual = storage[1:-1]
        actual[:] = x
        now = self.lib.replay(x.size, actual.ctypes.data,
                              actual.ctypes.data if inplace else x.ctypes.data, maximum)
        np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
        self.assertEqual(struct.pack("d", now), struct.pack("d", old))
        self.assertEqual(x.tobytes(), before)
        np.testing.assert_array_equal(storage[[0, -1]], np.float32(12345.25))
        return now, actual.copy()

    def test_lengths_tails_aliasing_and_masked_values(self):
        rng = np.random.default_rng(20261114)
        for n in (0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 72, 127, 336, 1025, 4096, 32769):
            for spread in (0.01, 1.0, 20.0, 100.0):
                x = rng.normal(0, spread, n).astype("f4")
                maximum = float(x.max()) if n else 0.0
                if n > 2:
                    x[1] = -np.inf
                for inplace in (False, True):
                    with self.subTest(n=n, spread=spread, inplace=inplace):
                        self.check_row(x, maximum, inplace)

    def test_rounding_extremes_and_reduction_negative_control(self):
        x = np.array([0, -0.0, -1e-7, -0.001, -0.1, -1, -2, -10,
                      -20, -80, -87, -88, -100, -103, -104, -np.inf], dtype="f4")
        for maximum in (0.0, 0.125):
            for inplace in (False, True):
                self.check_row(x, maximum, inplace)
        total, values = self.check_row(x, 0.0)
        # Summing every lane in FP64 is mathematically plausible but a different contract.
        self.assertNotEqual(total, values.astype("f8").sum())
        total, values = self.check_row(np.full(64, -np.inf, dtype="f4"), 0.0)
        self.assertEqual(total, 0.0)
        self.assertTrue(np.all(values == 0))

    def test_guard_pages(self):
        self.assertEqual(self.lib.guarded(), 0)


if __name__ == "__main__":
    unittest.main()
