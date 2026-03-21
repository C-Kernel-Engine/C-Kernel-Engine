#include "ckernel_engine.h"

#include <math.h>

static inline float recurrent_softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return expf(x);
    }
    return logf(1.0f + expf(x));
}

static inline float recurrent_sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

void recurrent_dt_gate_forward(const float *alpha,
                               const float *dt_bias,
                               const float *a,
                               float *gate,
                               int rows,
                               int dim) {
    for (int row = 0; row < rows; ++row) {
        const float *alpha_row = alpha + (size_t) row * (size_t) dim;
        float *gate_row = gate + (size_t) row * (size_t) dim;
        for (int col = 0; col < dim; ++col) {
            const float x = alpha_row[col] + dt_bias[col];
            gate_row[col] = recurrent_softplus(x) * a[col];
        }
    }
}

void recurrent_dt_gate_backward(const float *d_gate,
                                const float *alpha,
                                const float *dt_bias,
                                const float *a,
                                float *d_alpha,
                                float *d_dt_bias,
                                float *d_a,
                                int rows,
                                int dim) {
    for (int col = 0; col < dim; ++col) {
        d_dt_bias[col] = 0.0f;
        d_a[col] = 0.0f;
    }

    for (int row = 0; row < rows; ++row) {
        const float *d_gate_row = d_gate + (size_t) row * (size_t) dim;
        const float *alpha_row = alpha + (size_t) row * (size_t) dim;
        float *d_alpha_row = d_alpha + (size_t) row * (size_t) dim;
        for (int col = 0; col < dim; ++col) {
            const float x = alpha_row[col] + dt_bias[col];
            const float sp = recurrent_softplus(x);
            const float sig = recurrent_sigmoid(x);
            const float d_out = d_gate_row[col];
            d_a[col] += d_out * sp;
            {
                const float d_x = d_out * a[col] * sig;
                d_alpha_row[col] = d_x;
                d_dt_bias[col] += d_x;
            }
        }
    }
}

void recurrent_silu_forward(const float *x,
                            float *out,
                            int rows,
                            int dim) {
    for (int row = 0; row < rows; ++row) {
        const float *x_row = x + (size_t) row * (size_t) dim;
        float *out_row = out + (size_t) row * (size_t) dim;
        for (int col = 0; col < dim; ++col) {
            const float xv = x_row[col];
            out_row[col] = xv * recurrent_sigmoid(xv);
        }
    }
}

void recurrent_silu_backward(const float *d_out,
                             const float *x,
                             float *d_x,
                             int rows,
                             int dim) {
    for (int row = 0; row < rows; ++row) {
        const float *d_out_row = d_out + (size_t) row * (size_t) dim;
        const float *x_row = x + (size_t) row * (size_t) dim;
        float *d_x_row = d_x + (size_t) row * (size_t) dim;
        for (int col = 0; col < dim; ++col) {
            const float xv = x_row[col];
            const float sig = recurrent_sigmoid(xv);
            d_x_row[col] = d_out_row[col] * (sig + xv * sig * (1.0f - sig));
        }
    }
}
