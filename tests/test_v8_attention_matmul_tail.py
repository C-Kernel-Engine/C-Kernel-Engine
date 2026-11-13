"""Exercise the actual AVX-512 attention matrix tail and its rounding contract."""
import ctypes
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class AttentionMatmulTailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if platform.machine() not in ("x86_64", "AMD64") or not shutil.which("gcc"):
            print("TEST SKIPPED: requires x86 GCC")
            raise unittest.SkipTest("requires x86 GCC")
        if not Path("/proc/cpuinfo").exists() or "avx512f" not in Path("/proc/cpuinfo").read_text():
            print("TEST SKIPPED: requires AVX-512 execution")
            raise unittest.SkipTest("requires AVX-512 execution")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        output = Path(cls.tmp.name) / "attention_tail.so"
        source = '#include "' + str(ROOT / "src/kernels/attention_kernels.c") + '"\n' + r'''
#include <sys/mman.h>
#include <unistd.h>
__attribute__((visibility("default")))
void replay(float *c, const float *a, const float *b, int m, int k, int n) {
    ck_attention_matmul_f32_accum(c, a, b, m, k, n);
}
__attribute__((visibility("default")))
void reference(float *c, const float *a, const float *b, int m, int k, int n) {
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
        float sum = c[i*n+j];
        for (int q = 0; q < k; ++q) {
            if (j < n / 16 * 16) sum = fmaf(a[i*k+q], b[q*n+j], sum);
            else {
                volatile float product = a[i*k+q] * b[q*n+j];
                sum = sum + product;
            }
        }
        c[i*n+j] = sum;
    }
}
__attribute__((visibility("default")))
int guarded_tail(void) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    void *mapping[3] = {NULL, NULL, NULL};
    float *values[3];
    const size_t count[3] = {5*7, 7*72, 5*72};
    for (int i=0; i<3; ++i) {
        mapping[i] = mmap(NULL, 2*page, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (mapping[i] == MAP_FAILED) return 2;
        if (mprotect((char *)mapping[i]+page, page, PROT_NONE)) return 3;
        values[i] = (float *)((char *)mapping[i]+page-count[i]*sizeof(float));
        for(size_t j=0; j<count[i]; ++j) values[i][j] = (float)((int)(j%17)-8)*0.125f;
    }
    float expected[5*72];
    memcpy(expected, values[2], sizeof(expected));
    reference(expected, values[0], values[1], 5, 7, 72);
    replay(values[2], values[0], values[1], 5, 7, 72);
    int result = memcmp(expected, values[2], sizeof(expected)) != 0;
    for (int i=0; i<3; ++i) munmap(mapping[i], 2*page);
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
        for name in ("replay", "reference"):
            fn = getattr(cls.lib, name)
            fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
            fn.restype = None
        cls.lib.guarded_tail.argtypes = []
        cls.lib.guarded_tail.restype = ctypes.c_int

    def run_op(self, name, a, b, c):
        getattr(self.lib, name)(c.ctypes.data, a.ctypes.data, b.ctypes.data,
                                a.shape[0], a.shape[1], b.shape[1])

    def test_shapes_guards_and_identical_input_reference(self):
        rng = np.random.default_rng(20261113)
        for m in (1, 3, 4, 5, 7):
            for k in (1, 3, 17, 73):
                for n in (1, 7, 8, 15, 16, 17, 63, 64, 65, 71, 72, 73, 79, 80):
                    with self.subTest(m=m, k=k, n=n):
                        a = rng.uniform(-2, 2, (m, k)).astype("f4")
                        b = rng.uniform(-2, 2, (k, n)).astype("f4")
                        initial = rng.uniform(-1, 1, (m, n)).astype("f4")
                        expected = initial.copy()
                        self.run_op("reference", a, b, expected)
                        storage = np.full(m*n + 2, 12345.25, dtype="f4")
                        actual = storage[1:-1].reshape(m, n)
                        actual[:] = initial
                        a_before, b_before = a.tobytes(), b.tobytes()
                        self.run_op("replay", a, b, actual)
                        np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
                        self.assertEqual(storage[0], 12345.25)
                        self.assertEqual(storage[-1], 12345.25)
                        self.assertEqual(a.tobytes(), a_before)
                        self.assertEqual(b.tobytes(), b_before)

    def test_tail_is_nonfused_even_with_compiler_contraction_enabled(self):
        a = np.full((5, 1), np.float32(1 + 2**-23))
        b = np.full((1, 72), np.float32(1 - 2**-23))
        c = np.full((5, 72), -1, dtype="f4")
        self.run_op("replay", a, b, c)
        self.assertTrue(np.all(c[:, 64:] == 0))
        self.assertTrue(np.all(c[:, :64] != 0), "fixture must detect fused arithmetic")

    def test_tail_does_not_access_guard_pages(self):
        self.assertEqual(self.lib.guarded_tail(), 0)


if __name__ == "__main__":
    unittest.main()
