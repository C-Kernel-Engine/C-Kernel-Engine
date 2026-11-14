"""Execute selected encoder dispatchers and preserve serial stored bytes."""
import ctypes
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class EncoderRowParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('gcc'):
            raise unittest.SkipTest('GCC required')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.threads = min(4, len(os.sched_getaffinity(0)))
        output = Path(cls.tmp.name)/'rows.so'
        cls.output = output
        source = r'''
#include "ckernel_engine.h"
#include "ck_threadpool.h"
#define EXPORT __attribute__((visibility("default")))
static int width = 1, strict = 0;
int ck_get_num_threads(void) { return width; }
int ck_get_physical_cores(void) { return 4; }
int ck_strict_parity_enabled(void) { return strict; }
EXPORT void configure(int n) { ck_threadpool_global_destroy(); width = n; }
EXPORT void set_strict(int n) { strict = n; }
EXPORT uint64_t rope(int parallel, int storage, float *q, float *k,
    const int32_t *positions, int heads, int kvheads, int tokens) {
    ck_threadpool_t *pool = ck_threadpool_global();
    ck_threadpool_profile_reset(pool);
#define RUN(F) F(q,k,positions,heads,kvheads,tokens,72,80,72,12,12,12,0,4096,10000.f,1.f,0.f,1.f,32.f,1.f)
    if (storage == 1) {
        if (parallel) RUN(mrope_qk_vision_fp16_storage_parallel_dispatch);
        else RUN(mrope_qk_vision_fp16_storage);
    } else if (storage == 2) {
        if (parallel) RUN(mrope_qk_vision_bf16_storage_parallel_dispatch);
        else RUN(mrope_qk_vision_bf16_storage);
    } else {
        if (parallel) RUN(mrope_qk_vision_parallel_dispatch);
        else RUN(mrope_qk_vision);
    }
#undef RUN
    ck_threadpool_profile_t p;
    ck_threadpool_profile_snapshot(pool, &p);
    return p.dispatch_count;
}
EXPORT uint64_t norm(int parallel, const float *x, const float *g,
    const float *b, float *y, float *mean, float *rstd, int tokens, int dim) {
    ck_threadpool_t *pool = ck_threadpool_global();
    ck_threadpool_profile_reset(pool);
    if (parallel) layernorm_fp32_exact_parallel_dispatch(x,g,b,y,mean,rstd,tokens,dim,1e-5f);
    else layernorm_naive_serial_matched_precision(x,g,b,y,mean,rstd,tokens,dim,1e-5f);
    ck_threadpool_profile_t p;
    ck_threadpool_profile_snapshot(pool, &p);
    return p.dispatch_count;
}
typedef struct { float *x, *y; } nested_args;
static void nested_work(int ith, int nth, void *opaque) {
    nested_args *a = opaque;
    const size_t offset = (size_t)ith * 257 * 73;
    layernorm_fp32_exact_parallel_dispatch(a->x+offset,0,0,a->y+offset,0,0,257,73,1e-5f);
}
EXPORT uint64_t nested(float *x, float *y) {
    ck_threadpool_t *pool = ck_threadpool_global();
    ck_threadpool_profile_reset(pool);
    nested_args a = {x,y};
    ck_threadpool_dispatch_n(pool,width,nested_work,&a);
    ck_threadpool_profile_t p;
    ck_threadpool_profile_snapshot(pool,&p);
    return p.dispatch_count;
}
typedef struct { float *q, *k; const int32_t *pos; } nested_rope_args;
static void nested_rope_work(int ith, int nth, void *opaque) {
    nested_rope_args *a=opaque;
    size_t offset=(size_t)ith*2*257*80;
    mrope_qk_vision_fp16_storage_parallel_dispatch(a->q+offset,a->k+offset,a->pos,
        2,2,257,72,80,72,12,12,12,0,4096,10000.f,1.f,0.f,1.f,32.f,1.f);
}
EXPORT uint64_t nested_rope(float *q, float *k, const int32_t *pos) {
    ck_threadpool_t *pool=ck_threadpool_global();
    ck_threadpool_profile_reset(pool);
    nested_rope_args a={q,k,pos};
    ck_threadpool_dispatch_n(pool,width,nested_rope_work,&a);
    ck_threadpool_profile_t p;
    ck_threadpool_profile_snapshot(pool,&p);
    return p.dispatch_count;
}
'''
        # Compile the actual selected entry points, so a map reverting to the
        # serial provider fails the executed dispatch-count assertions.
        for name in ('layernorm_fp32_exact', 'mrope_qk_vision',
                     'mrope_qk_vision_fp16_storage', 'mrope_qk_vision_bf16_storage'):
            provider=json.loads((ROOT/'version/v8/kernel_maps'/f'{name}.json').read_text())
            source=source.replace(name+'_parallel_dispatch', provider['impl']['function'])
        subprocess.run(['gcc', *shlex.split(os.environ.get('CK_ENCODER_TEST_CFLAGS', '')),
                        '-x', 'c', '-', 'src/ck_threadpool.c',
                        'src/kernels/rope_kernels.c', 'src/kernels/layernorm_kernels.c',
                        'version/v8/src/ck_parallel_prefill_v8.c',
                        '-Iinclude', '-Iversion/v8/src', '-O3', '-fno-fast-math',
                        '-fPIC', '-shared', '-fvisibility=hidden',
                        '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections',
                        '-pthread', '-lm', '-ldl', '-o', str(output)],
                       cwd=ROOT, input=source, text=True, capture_output=True,
                       check=True, timeout=120)
        cls.lib = ctypes.CDLL(str(output))
        cls.lib.rope.argtypes = [ctypes.c_int]*2 + [ctypes.c_void_p]*3 + [ctypes.c_int]*3
        cls.lib.rope.restype = ctypes.c_uint64
        cls.lib.norm.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*6 + [ctypes.c_int]*2
        cls.lib.norm.restype = ctypes.c_uint64
        cls.addClassCleanup(cls.lib.configure, 1)

    def test_rope_storage_tails_alias_and_dispatch(self):
        rng = np.random.default_rng(1128)
        for threads in (1, self.threads):
            self.lib.configure(threads)
            for storage in (0, 1, 2):
                for heads, kvheads, tokens, alias in ((3,3,257,False), (4,4,513,False),
                        (3,2,257,False), (3,3,31,False), (3,3,257,True)):
                    with self.subTest(threads=threads, storage=storage, heads=heads,
                                      kvheads=kvheads, tokens=tokens, alias=alias):
                        pos = rng.integers(0, 30, (4,tokens), dtype=np.int32)
                        q = rng.normal(size=heads*tokens*80+32).astype(np.float32)
                        k = q if alias else rng.normal(size=kvheads*tokens*80+32).astype(np.float32)
                        qr, kr = q.copy(), k.copy()
                        if alias:
                            kr = qr
                        self.lib.rope(0, storage, qr[16:-16].ctypes.data, kr[16:-16].ctypes.data,
                                      pos.ctypes.data, heads, kvheads, tokens)
                        calls = self.lib.rope(1, storage, q[16:-16].ctypes.data, k[16:-16].ctypes.data,
                                              pos.ctypes.data, heads, kvheads, tokens)
                        self.assertEqual(calls, int(threads > 1 and heads == kvheads and tokens >=256 and not alias))
                        np.testing.assert_array_equal(q.view(np.uint32), qr.view(np.uint32))
                        np.testing.assert_array_equal(k.view(np.uint32), kr.view(np.uint32))

    def test_norm_tails_caches_inplace_and_dispatch(self):
        rng = np.random.default_rng(7)
        for threads in (1, self.threads):
            self.lib.configure(threads)
            for tokens, dim in ((31,72), (257,73), (513,72)):
                for inplace in (False, True):
                    with self.subTest(threads=threads,tokens=tokens,dim=dim,inplace=inplace):
                        x = rng.normal(size=(tokens,dim)).astype(np.float32)
                        g, b = rng.normal(size=(2,dim)).astype(np.float32)
                        expected, actual = x.copy(), x.copy()
                        means = [np.zeros(tokens, np.float32) for _ in range(2)]
                        rstds = [np.zeros(tokens, np.float32) for _ in range(2)]
                        for parallel, y in enumerate((expected, actual)):
                            calls = self.lib.norm(parallel, y.ctypes.data if inplace else x.ctypes.data,
                                g.ctypes.data,b.ctypes.data,y.ctypes.data,means[parallel].ctypes.data,
                                rstds[parallel].ctypes.data,tokens,dim)
                            self.assertEqual(calls, int(parallel and threads > 1 and tokens >=256))
                        for a,b in ((actual,expected),(means[0],means[1]),(rstds[0],rstds[1])):
                            np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))

    def test_nested_dispatch_does_not_reenter_pool(self):
        code = r'''
import ctypes, numpy as np, sys
lib=ctypes.CDLL(sys.argv[1]); n=int(sys.argv[2]); lib.configure(n)
lib.nested.argtypes=[ctypes.c_void_p]*2; lib.nested.restype=ctypes.c_uint64
lib.norm.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*6+[ctypes.c_int]*2
x=np.linspace(-5,5,n*257*73,dtype=np.float32); y=np.zeros_like(x); ref=np.zeros_like(x)
assert lib.nested(x.ctypes.data,y.ctypes.data)==1
lib.norm(0,x.ctypes.data,None,None,ref.ctypes.data,None,None,n*257,73)
np.testing.assert_array_equal(y.view(np.uint32),ref.view(np.uint32))
lib.nested_rope.argtypes=[ctypes.c_void_p]*3; lib.nested_rope.restype=ctypes.c_uint64
lib.rope.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*3+[ctypes.c_int]*3
q=np.linspace(-2,2,n*2*257*80,dtype=np.float32); k=q.copy()
qr=q.copy(); kr=k.copy(); pos=np.tile(np.arange(257,dtype=np.int32),4)
assert lib.nested_rope(q.ctypes.data,k.ctypes.data,pos.ctypes.data)==1
lib.rope(0,1,qr.ctypes.data,kr.ctypes.data,pos.ctypes.data,n*2,n*2,257)
np.testing.assert_array_equal(q.view(np.uint32),qr.view(np.uint32))
np.testing.assert_array_equal(k.view(np.uint32),kr.view(np.uint32))
'''
        import sys
        subprocess.run([sys.executable,'-c',code,str(self.output),str(self.threads)],
                       check=True,capture_output=True,timeout=30)

    def test_shifted_overlap_keeps_serial_order(self):
        self.lib.configure(self.threads)
        for cache_alias in (False, True):
            with self.subTest(cache_alias=cache_alias):
                original=np.linspace(-2,2,257*73+33,dtype=np.float32)
                expected,actual=original.copy(),original.copy()
                for parallel,data in enumerate((expected,actual)):
                    x=data[16:16+257*73]
                    y=data[17:17+257*73]
                    calls=self.lib.norm(parallel,x.ctypes.data,None,None,y.ctypes.data,
                        y.ctypes.data if cache_alias else None,None,257,73)
                    self.assertEqual(calls,0)
                np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

    def test_maps_select_executed_dispatchers(self):
        for name in ('layernorm_fp32_exact','mrope_qk_vision',
                     'mrope_qk_vision_fp16_storage','mrope_qk_vision_bf16_storage'):
            value=json.loads((ROOT/'version/v8/kernel_maps'/f'{name}.json').read_text())
            self.assertTrue(value['impl']['function'].endswith('_parallel_dispatch'))
            self.assertIn('version/v8/src/ck_parallel_prefill_v8.c',value['impl']['sources'])


if __name__ == '__main__':
    unittest.main()
