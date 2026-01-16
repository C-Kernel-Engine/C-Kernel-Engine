# Q4_K Packing Format Fix

## 🔍 Analysis Results

The test clearly shows that **C-Kernel-Engine's unpacking is incorrect**. The formats are completely different:

### llama.cpp Format (Correct)
```
Scales: ['0x3f', '0x15', '0x2a', '0x0c', '0x03', '0x15', '0x2a', '0x3c']
Mins:   ['0x33', '0x15', '0x2a', '0x0c', '0x03', '0x15', '0x2a', '0x3c']
```

### Current C-Kernel-Engine Format (Incorrect)
```
Scales: ['0x3f', '0x14', '0x25', '0x2a', '0x0c', '0x0f', '0x13', '0x15']
Mins:   ['0x2a', '0x32', '0x3c', '0x0c', '0x15', '0x29', '0x0a', '0x33']
```

## 📊 Understanding the Correct Format

From llama.cpp's `get_scale_min_k4()`:

```c
static inline void get_scale_min_k4(int j, const uint8_t * q, 
                                   uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;        // Direct 6-bit value
        *m = q[j + 4] & 63;    // Direct 6-bit value
    } else {
        // Complex packing for j >= 4
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
```

### Byte Layout (12 bytes total)
```
Bytes 0-3:   [sc0][sc1][sc2][sc3]  (6 bits each, 2 bits unused)
Bytes 4-7:   [m0][m1][m2][m3]     (6 bits each, 2 bits unused)
Bytes 8-11:  [sc4][sc5][sc6][sc7]  (4 bits) + [m4][m5][m6][m7] (4 bits)
             | high 2 bits from bytes 0-3 and 4-7 |
```

## 🚨 The Bug

**Current Implementation**: Assumes linear packing where all 8 scales come first, then all 8 mins.

**Correct Format**: Uses interleaved packing where:
- First 4 scales and mins use direct 6-bit packing
- Last 4 scales and mins use complex 4+2 bit packing

## ✅ The Fix

Here's the corrected implementation:

```c
static inline void unpack_q4_k_scales(const uint8_t *scales,
                                       uint8_t *sc, uint8_t *m) {
    // First 4 scales and mins: direct 6-bit values
    sc[0] = scales[0] & 0x3F;
    sc[1] = scales[1] & 0x3F;
    sc[2] = scales[2] & 0x3F;
    sc[3] = scales[3] & 0x3F;

    m[0] = scales[4] & 0x3F;
    m[1] = scales[5] & 0x3F;
    m[2] = scales[6] & 0x3F;
    m[3] = scales[7] & 0x3F;

    // Last 4 scales and mins: complex 4+2 bit packing
    sc[4] = (scales[8]  & 0x0F) | ((scales[0] >> 6) << 4);
    sc[5] = (scales[9]  & 0x0F) | ((scales[1] >> 6) << 4);
    sc[6] = (scales[10] & 0x0F) | ((scales[2] >> 6) << 4);
    sc[7] = (scales[11] & 0x0F) | ((scales[3] >> 6) << 4);

    m[4] = (scales[8]  >> 4) | ((scales[4] >> 6) << 4);
    m[5] = (scales[9]  >> 4) | ((scales[5] >> 6) << 4);
    m[6] = (scales[10] >> 4) | ((scales[6] >> 6) << 4);
    m[7] = (scales[11] >> 4) | ((scales[7] >> 6) << 4);
}
```

## 🧪 Verification

Run the test again to confirm:
```bash
python test_q4k_packing.py
```

Expected output: **✅ Packing formats MATCH!**

## 📝 Implementation Plan

### 1. Fix the Header File
Update `/home/antshiv/Workspace/C-Kernel-Engine/include/ckernel_quant.h` with the corrected implementation.

### 2. Update All Kernel Files
The function is used in multiple places:
- `gemm_kernels_q4k.c` (6 uses)
- `gemm_kernels_q4k_q8k.c` (2 uses)
- `gemm_kernels_q4k_q8k_vnni.c` (1 use)
- `gemm_kernels_q4k_q8k_avx2.c` (1 use)
- `dequant_kernels.c` (2 uses)

### 3. Test with Real Models
Load a Q4_K model and verify:
```bash
python scripts/convert_gguf_to_bump_v4.py --gguf model-q4_k.gguf --inspect
```

### 4. Performance Testing
Ensure the fix doesn't regress performance:
```bash
# Run existing unit tests
python unittest/test_q4_k_dequant_match.py
python unittest/test_quant_kernels.py
```

## 🎯 Impact

**Before Fix**: ❌ Incompatible with llama.cpp Q4_K models
**After Fix**: ✅ Full compatibility with all Q4_K models

This fix ensures your C-Kernel-Engine can correctly load and run **all Q4_K quantized models** from the llama.cpp ecosystem.