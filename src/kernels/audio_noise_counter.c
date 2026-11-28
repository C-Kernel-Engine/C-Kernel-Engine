#include "ckernel_tts.h"
#include "checked_float_buffer.h"

#include "ckernel_counter_gaussian.h"

/* Counter-based excitation. Each element depends only on its index, seed and
 * stream, so requests cannot inherit RNG state from previous calls. The
 * Box-Muller transform uses FP64 libm and stores FP32 samples. It does not
 * promise the same random stream as PyTorch. */
int audio_noise_counter_gaussian_f32(float *output, size_t output_capacity,
    size_t count, uint64_t seed, uint64_t stream) {
    if (!count) return CK_AUDIO_EXTENT_INVALID;
    if (count > (1u << 24) || output_capacity < count)
        return CK_AUDIO_EXTENT_LIMIT;
    if (!ck_checked_region(output, count)) return CK_AUDIO_EXTENT_INVALID;
    const uint64_t domain = ck_rng_gaussian_domain(seed, stream);
    for (size_t index = 0; index < count; ++index)
        output[index] = (float)ck_rng_gaussian_f64(domain, index);
    return CK_AUDIO_EXTENT_OK;
}
