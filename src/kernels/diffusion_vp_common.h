#ifndef CK_DIFFUSION_VP_COMMON_H
#define CK_DIFFUSION_VP_COMMON_H

#include <math.h>

static inline double vp_sigmoid(double x)
{
    if (x >= 0.0) return 1.0 / (1.0 + exp(-x));
    const double e = exp(x);
    return e / (1.0 + e);
}

#endif
