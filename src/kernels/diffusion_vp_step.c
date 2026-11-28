#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "diffusion_vp_common.h"

static int vp_region_valid(const double *data, size_t count)
{
    return data && (uintptr_t)data % _Alignof(double) == 0 &&
           count <= SIZE_MAX / sizeof(double) &&
           (uintptr_t)data <= UINTPTR_MAX - count * sizeof(double);
}

static int vp_overlaps(const double *a, const double *b, size_t bytes)
{
    return (uintptr_t)a < (uintptr_t)b + bytes &&
           (uintptr_t)b < (uintptr_t)a + bytes;
}

static double vp_blend(double z, double estimate, double noise,
                       double residual_gain, double estimate_gain, double noise_gain)
{
    volatile double residual = residual_gain * z;
    volatile double denoised = estimate_gain * estimate;
    volatile double innovation = noise_gain * noise;
    return (residual + denoised) + innovation;
}

/* One variance-preserving DDPM step with externally supplied Gaussian noise.
 * All spans have count elements. Identical output/input spans are forbidden. */
int ck_diffusion_vp_ddpm_step_f64(const double *latent, const double *estimate,
                                  const double *noise, double *output, size_t count,
                                  double gamma_from, double gamma_to)
{
    if (!count || !isfinite(gamma_from) || !isfinite(gamma_to) ||
        gamma_from < gamma_to || !vp_region_valid(latent, count) ||
        !vp_region_valid(estimate, count) || !vp_region_valid(noise, count) ||
        !vp_region_valid(output, count)) return -1;
    const size_t bytes = count * sizeof(double);
    if (vp_overlaps(output, latent, bytes) || vp_overlaps(output, estimate, bytes) ||
        vp_overlaps(output, noise, bytes)) return -1;

    const double from = vp_sigmoid(-gamma_from);
    const double to = vp_sigmoid(-gamma_to);
    if (from <= 0.0 || to <= 0.0) return -1;
    const double blend = -expm1(gamma_to - gamma_from);
    const double residual_gain = (1.0 - blend) * sqrt(to / from);
    const double estimate_gain = blend * sqrt(to);
    const double noise_gain = sqrt(blend * (1.0 - to));
    if (!isfinite(residual_gain) || !isfinite(estimate_gain) || !isfinite(noise_gain)) return -1;

    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(latent[i]) || !isfinite(estimate[i]) || !isfinite(noise[i]) ||
            !isfinite(vp_blend(latent[i], estimate[i], noise[i],
                               residual_gain, estimate_gain, noise_gain))) return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        output[i] = vp_blend(latent[i], estimate[i], noise[i],
                             residual_gain, estimate_gain, noise_gain);
    }
    return 0;
}

/* Identical transition with caller-bound schedule endpoints. */
int ck_diffusion_vp_ddpm_step_schedule_f64(const double *latent, const double *estimate,
                                          const double *noise, const double *gamma, double *output,
                                          size_t count, size_t gamma_elements)
{
    if (gamma_elements != 2 || !vp_region_valid(gamma, 2) || !vp_region_valid(output, count)) return -1;
    const uintptr_t begin = (uintptr_t)gamma, out = (uintptr_t)output;
    if (begin < out + count * sizeof(double) && out < begin + 2 * sizeof(double)) return -1;
    return ck_diffusion_vp_ddpm_step_f64(latent, estimate, noise, output, count, gamma[0], gamma[1]);
}
