#ifndef CKERNEL_COUNTER_GAUSSIAN_H
#define CKERNEL_COUNTER_GAUSSIAN_H

#include <math.h>
#include <stdint.h>

/* SplitMix64 counter domain and FP64 Box-Muller. Reproducibility is for a
 * given libm/build; the stream does not depend on any prior request. */
static inline uint64_t ck_rng_mix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static inline uint64_t ck_rng_gaussian_domain(uint64_t seed, uint64_t stream)
{
    return ck_rng_mix64(seed ^ ck_rng_mix64(stream));
}

static inline double ck_rng_gaussian_f64(uint64_t domain, uint64_t index)
{
    const uint64_t counter = index * UINT64_C(2);
    const uint64_t first = ck_rng_mix64(domain + counter);
    const uint64_t second = ck_rng_mix64(domain + counter + UINT64_C(1));
    const double uniform_positive = ((double)(first >> 11) + 0.5) * 0x1.0p-53;
    const double uniform_angle = (double)(second >> 11) * 0x1.0p-53;
    const double radius = sqrt(-2.0 * log(uniform_positive));
    return radius * cos(6.283185307179586476925286766559 * uniform_angle);
}

#endif
