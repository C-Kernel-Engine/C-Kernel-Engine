"""Asset-free checks of production FP32 row reuse and partial output ownership."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ckernel_engine.h"
void gemm_nt_f32_llama_production_output_range(const float*, const float*, const float*, float*, int, int, int, int, int);
int main(void) {
    const int shapes[][3] = {{1,1,1},{3,9,32},{4,9,32},{5,9,32},
        {8,17,64},{9,17,65},{11,5,32},{12,4,32},{13,7,32},
        {17,33,128},{29,33,128},{30,33,128},{31,33,128},{32,33,128},
        {33,33,128},{47,33,128},{48,33,128},{49,33,128},
        {65,33,128},{95,33,128},{96,33,128},{97,33,128},
        {9,1280,1280},{65,1280,1280}};
    for (unsigned s=0; s<sizeof(shapes)/sizeof(shapes[0]); ++s) {
        int M=shapes[s][0], N=shapes[s][1], K=shapes[s][2], total=M*N;
        float *a=malloc(M*K*sizeof(float)), *b=malloc(N*K*sizeof(float));
        float *bias=malloc(N*sizeof(float)), *ref=malloc(total*sizeof(float));
        float *out=malloc((total+2)*sizeof(float));
        if (!a || !b || !bias || !ref || !out) return 2;
        for(int i=0;i<M*K;++i) a[i]=(float)((i*37)%101-50)/17.0f;
        for(int i=0;i<N*K;++i) b[i]=(float)((i*13)%79-39)/23.0f;
        for(int i=0;i<N;++i) bias[i]=(float)(i%7-3)/31.0f;
        for(int use_bias=0;use_bias<2;++use_bias) {
            const float *bp=use_bias?bias:NULL;
            gemm_nt_f32_llama_production(a,b,bp,ref,M,N,K);
            for(int partial=0;partial<2;++partial) {
                int begin=partial && total>5?2:0, end=partial && total>5?total-3:total;
                for(int i=0;i<total+2;++i) out[i]=-12345.0f;
                gemm_nt_f32_llama_production_output_range(a,b,bp,out+1,M,N,K,begin,end);
                for(int i=0;i<total+2;++i) {
                    float expected=i-1>=begin && i-1<end?ref[i-1]:-12345.0f;
                    if(memcmp(&expected,out+i,sizeof(float))) {
                        fprintf(stderr,"shape=%u bias=%d partial=%d index=%d\n",s,use_bias,partial,i);
                        return 1;
                    }
                }
            }
        }
        free(a);free(b);free(bias);free(ref);free(out);
    }
    return 0;
}
"""


class F32RowReuseTests(unittest.TestCase):
    def run_build(self, flags, *, corrupt_m3n4=False):
        with tempfile.TemporaryDirectory(prefix="cke-f32-row-reuse-") as name:
            path = Path(name)
            (path / "probe.c").write_text(SOURCE)
            kernel = ROOT / "src/kernels/gemm_kernels.c"
            if corrupt_m3n4:
                source = kernel.read_text()
                original = "__m256 acc00 = _mm256_setzero_ps();"
                self.assertEqual(source.count(original), 1)
                kernel = path / "corrupted_gemm.c"
                kernel.write_text(source.replace(
                    original, "__m256 acc00 = _mm256_set1_ps(1.0f);", 1))
            subprocess.run([
                "cc", "-O3", "-ffp-contract=off", *flags, "-ffunction-sections",
                "-fdata-sections", "-I", str(ROOT / "include"),
                str(kernel), str(path / "probe.c"),
                "-Wl,--gc-sections", "-lm", "-o", str(path / "probe"),
            ], check=True, capture_output=True, timeout=120)
            result = subprocess.run([str(path / "probe")], check=False,
                                    capture_output=True, text=True, timeout=60)
            if corrupt_m3n4:
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("shape=", result.stderr)
                self.assertIn("index=", result.stderr)
            else:
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_avx2_m3n4_corruption_fails_numerical_gate(self):
        cpuinfo = Path("/proc/cpuinfo")
        flags = cpuinfo.read_text() if cpuinfo.exists() else ""
        if not all(flag in flags for flag in ("avx2", "fma")):
            self.skipTest("AVX2/FMA execution unavailable")
        self.run_build(["-mavx2", "-mfma"], corrupt_m3n4=True)

    def test_portable(self):
        self.run_build([])

    def test_avx2(self):
        cpuinfo = Path("/proc/cpuinfo")
        flags = cpuinfo.read_text() if cpuinfo.exists() else ""
        if not all(flag in flags for flag in ("avx2", "fma")):
            self.skipTest("AVX2/FMA execution unavailable")
        for build in (["-mavx2", "-mno-fma"], ["-mavx2", "-mfma"]):
            with self.subTest(flags=build):
                self.run_build(build)


if __name__ == "__main__":
    unittest.main()
