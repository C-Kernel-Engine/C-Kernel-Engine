"""Check output-lane SIMD against an independent sequential FP32 reduction."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class BF16ExactOutputLanesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cpuinfo = Path('/proc/cpuinfo')
        if not shutil.which('gcc') or not cpuinfo.exists() or 'avx512f' not in cpuinfo.read_text():
            print('TEST SKIPPED: requires GCC and AVX-512')
            raise unittest.SkipTest('requires GCC and AVX-512')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        output = Path(cls.tmp.name) / 'replay.so'
        source = '#include "' + str(ROOT / 'src/kernels/gemm_kernels_bf16.c') + '"\n' + r'''
#include <sys/mman.h>
#include <unistd.h>
__attribute__((visibility("default")))
void replay(const float *a, const void *b, const float *bias, float *c,
            int m, int n, int k, int begin, int end) {
    gemm_nt_bf16_row_range(a, b, bias, c, m, n, k, begin, end);
}
__attribute__((visibility("default")))
int guarded(void) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    void *maps[4];
    void *ptrs[4];
    const size_t sizes[4] = {17*33*4, 17*33*2, 17*4, 17*17*4};
    for (int i = 0; i < 4; ++i) {
        maps[i] = mmap(NULL, 2*page, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (maps[i] == MAP_FAILED) return 1;
        if (mprotect((char *)maps[i]+page, page, PROT_NONE)) return 2;
        ptrs[i] = (char *)maps[i]+page-sizes[i];
    }
    float *a = ptrs[0], *bias = ptrs[2], *c = ptrs[3];
    uint16_t *b = ptrs[1];
    for (int i=0; i<17*33; ++i) a[i] = 0.25f;
    for (int i=0; i<17*33; ++i) b[i] = 0x3f00;
    for (int i=0; i<17; ++i) bias[i] = 0.5f;
    gemm_nt_bf16_row_range(a, b, bias, c, 17, 17, 33, 0, 17);
    int bad = 0;
    for (int i=0; i<17*17; ++i) if (c[i] != 4.625f) bad = 3;
    for (int i=0; i<4; ++i) munmap(maps[i], 2*page);
    return bad;
}
'''
        subprocess.run(['gcc', '-x', 'c', '-', '-shared', '-fPIC', '-O3', '-march=native',
                        '-ffp-contract=fast', '-fno-fast-math', '-ffunction-sections',
                        '-fdata-sections', '-fvisibility=hidden', '-Wl,--gc-sections',
                        '-I' + str(ROOT / 'include'), '-lm', '-o', str(output)],
                       input=source, text=True, capture_output=True, check=True, timeout=120)
        cls.lib = ctypes.CDLL(str(output))
        cls.fn = cls.lib.replay
        cls.fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
        cls.fn.restype = None
        cls.lib.guarded.argtypes = []
        cls.lib.guarded.restype = ctypes.c_int

    def check_case(self, m, n, k, bias_enabled=True, begin=0, end=None):
        end = m if end is None else end
        rng = np.random.default_rng(m * 10000 + n * 100 + k)
        a = rng.standard_normal((m, k), dtype=np.float32)
        # Include BF16 ties, signed zero, and both signs.
        a.flat[:min(a.size, 4)] = np.array([0x3f808000, 0xbf818000, 0, 0x80000000],
                                         dtype=np.uint32).view(np.float32)[:min(a.size, 4)]
        w = rng.standard_normal((n, k), dtype=np.float32).view(np.uint32)
        b = (w >> 16).astype(np.uint16)
        bias = rng.standard_normal(n, dtype=np.float32)
        guards = np.full(m * n + 32, -9182.25, dtype=np.float32)
        c = guards[16:-16].reshape(m, n)
        expected = c.copy()
        bits = a.view(np.uint32)
        rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
        weights = (b.astype(np.uint32) << 16).view(np.float32)
        if 0 <= begin < end <= m:
            acc = np.broadcast_to(bias if bias_enabled else np.zeros(n, np.float32), (end-begin, n)).copy()
            for q in range(k):
                product = rounded[begin:end, q, None] * weights[None, :, q]
                acc = acc + product
            expected[begin:end] = acc
        before = (a.tobytes(), b.tobytes(), bias.tobytes())
        self.fn(a.ctypes.data, b.ctypes.data, bias.ctypes.data if bias_enabled else None,
                c.ctypes.data, m, n, k, begin, end)
        np.testing.assert_array_equal(c.view(np.uint32), expected.view(np.uint32))
        self.assertTrue(np.all(guards[:16] == -9182.25) and np.all(guards[-16:] == -9182.25))
        self.assertEqual(before, (a.tobytes(), b.tobytes(), bias.tobytes()))

    def test_shapes_and_bias(self):
        for m in (1, 3, 4, 5, 9, 15, 16, 17, 31, 32, 33):
            for n in (1, 7, 15, 16, 17, 31, 32, 65):
                for k in (1, 7, 31, 32, 33, 129):
                    with self.subTest(m=m, n=n, k=k):
                        self.check_case(m, n, k)
                        self.check_case(m, n, k, False)

    def test_row_ranges_and_rejection(self):
        for begin, end in ((1, 8), (3, 5), (0, 0), (-1, 5), (0, 10)):
            with self.subTest(begin=begin, end=end):
                self.check_case(9, 19, 65, begin=begin, end=end)
        for begin, end in ((1, 32), (7, 24), (16, 33)):
            with self.subTest(begin=begin, end=end):
                self.check_case(33, 19, 65, begin=begin, end=end)

    def test_guard_pages_for_input_and_output_tails(self):
        self.assertEqual(self.lib.guarded(), 0)


if __name__ == '__main__':
    unittest.main()
