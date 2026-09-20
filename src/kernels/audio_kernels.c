/**
 * @file audio_kernels.c
 * @brief Numerically explicit audio frontend reference kernels.
 */

#include "ckernel_audio.h"
#include "ck_threadpool.h"

#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#define CK_AUDIO_PI_F 3.14159265358979323846f
#define CK_AUDIO_PI_D 3.14159265358979323846264338327950288

/* Shared production GEMM; provided by the engine in generated runtimes. */
void gemm_nt_f32_llama_production_parallel_dispatch(
    const float *A,
    const float *B,
    const float *bias,
    float *C,
    int M,
    int N,
    int K);

void layernorm_naive_serial_matched_precision(
    const float *input,
    const float *gamma,
    const float *beta,
    float *output,
    float *mean_cache,
    float *rstd_cache,
    int tokens,
    int d_model,
    float eps);

void recurrent_silu_forward(
    const float *input,
    float *output,
    int rows,
    int dim);

static int checked_mul_size(size_t left, size_t right, size_t *result)
{
    if (result == NULL || (right != 0 && left > SIZE_MAX / right)) {
        return -1;
    }
    *result = left * right;
    return 0;
}

static int checked_add_size(size_t left, size_t right, size_t *result)
{
    if (result == NULL || left > SIZE_MAX - right) {
        return -1;
    }
    *result = left + right;
    return 0;
}

static int conv_output_extent(int input, int kernel, int stride, int padding)
{
    if (input <= 0 || kernel <= 0 || stride <= 0 || padding < 0 ||
        input > INT_MAX - 2 * padding || input + 2 * padding < kernel) {
        return -1;
    }
    return (input + 2 * padding - kernel) / stride + 1;
}

static int reflect_index(int index, int length)
{
    while (index < 0 || index >= length) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * length - index - 2;
        }
    }
    return index;
}

static uint16_t read_u16_le(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_u32_le(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
        ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int audio_wav_parse_memory(
    const uint8_t *bytes,
    size_t byte_count,
    CKAudioWavInfo *info)
{
    if (bytes == NULL || info == NULL) {
        return -1;
    }
    if (byte_count < 12 || memcmp(bytes, "RIFF", 4) != 0 ||
        memcmp(bytes + 8, "WAVE", 4) != 0) {
        return -2;
    }
    const size_t riff_end = (size_t)read_u32_le(bytes + 4) + 8u;
    if (riff_end < 12 || riff_end > byte_count) {
        return -3;
    }
    memset(info, 0, sizeof(*info));
    int found_format = 0;
    int found_data = 0;
    size_t offset = 12;
    while (offset + 8 <= riff_end) {
        const uint8_t *chunk = bytes + offset;
        const uint32_t chunk_bytes = read_u32_le(chunk + 4);
        const size_t payload = offset + 8;
        if ((size_t)chunk_bytes > riff_end - payload) {
            return -3;
        }
        if (!found_format && memcmp(chunk, "fmt ", 4) == 0) {
            if (chunk_bytes < 16) {
                return -4;
            }
            info->format_tag = (int)read_u16_le(bytes + payload);
            info->channels = (int)read_u16_le(bytes + payload + 2);
            info->sample_rate = (int)read_u32_le(bytes + payload + 4);
            info->bits_per_sample = (int)read_u16_le(bytes + payload + 14);
            found_format = 1;
        } else if (!found_data && memcmp(chunk, "data", 4) == 0) {
            info->data_offset = payload;
            info->data_bytes = chunk_bytes;
            found_data = 1;
        }
        const size_t padded = (size_t)chunk_bytes + ((size_t)chunk_bytes & 1u);
        if (padded > SIZE_MAX - payload) {
            return -3;
        }
        offset = payload + padded;
    }
    if (!found_format || !found_data || info->format_tag != 1 ||
        info->channels <= 0 || info->sample_rate <= 0 ||
        info->bits_per_sample != 16) {
        return -5;
    }
    const size_t bytes_per_frame = (size_t)info->channels * 2u;
    if (bytes_per_frame == 0 || info->data_bytes % bytes_per_frame != 0 ||
        info->data_bytes / bytes_per_frame > (size_t)INT_MAX) {
        return -6;
    }
    info->frames = (int)(info->data_bytes / bytes_per_frame);
    return info->frames > 0 ? 0 : -6;
}

int audio_wav_decode_pcm16_mono_f32(
    const uint8_t *bytes,
    size_t byte_count,
    const CKAudioWavInfo *info,
    float *mono,
    int mono_capacity)
{
    if (bytes == NULL || info == NULL || mono == NULL) {
        return -1;
    }
    if (info->format_tag != 1 || info->bits_per_sample != 16 ||
        info->channels <= 0 || info->frames <= 0 || mono_capacity < info->frames ||
        info->data_offset > byte_count || info->data_bytes > byte_count - info->data_offset) {
        return -2;
    }
    const uint8_t *pcm = bytes + info->data_offset;
    const float scale = 1.0f / 32768.0f;
    for (int frame = 0; frame < info->frames; ++frame) {
        float sum = 0.0f;
        for (int channel = 0; channel < info->channels; ++channel) {
            const size_t index = ((size_t)frame * info->channels + channel) * 2u;
            sum += (float)(int16_t)read_u16_le(pcm + index);
        }
        mono[frame] = (sum / (float)info->channels) * scale;
    }
    return info->frames;
}

int audio_wav_decode_memory_pcm16_mono_f32(
    const uint8_t *bytes,
    size_t byte_count,
    float *mono,
    int mono_capacity,
    CKAudioWavInfo *info)
{
    return audio_wav_decode_memory_pcm16_mono_window_f32(
        bytes, byte_count, 0, mono, mono_capacity, info);
}

int audio_wav_decode_memory_pcm16_mono_window_f32(
    const uint8_t *bytes,
    size_t byte_count,
    int start_frame,
    float *mono,
    int mono_capacity,
    CKAudioWavInfo *info)
{
    const int status = audio_wav_parse_memory(bytes, byte_count, info);
    if (status != 0) {
        return status;
    }
    if (start_frame < 0 || start_frame >= info->frames || mono_capacity <= 0) {
        return -7;
    }
    const int available = info->frames - start_frame;
    const int decoded = available < mono_capacity ? available : mono_capacity;
    const uint8_t *pcm = bytes + info->data_offset;
    const float scale = 1.0f / 32768.0f;
    for (int frame = 0; frame < decoded; ++frame) {
        float sum = 0.0f;
        const size_t source_frame = (size_t)start_frame + (size_t)frame;
        for (int channel = 0; channel < info->channels; ++channel) {
            const size_t index =
                (source_frame * (size_t)info->channels + (size_t)channel) * 2u;
            sum += (float)(int16_t)read_u16_le(pcm + index);
        }
        mono[frame] = (sum / (float)info->channels) * scale;
    }
    return decoded;
}

int audio_pcm_s16_to_mono_f32(
    const int16_t *interleaved,
    int n_frames,
    int n_channels,
    float *mono)
{
    if (interleaved == NULL || mono == NULL) {
        return -1;
    }
    if (n_frames <= 0 || n_channels <= 0) {
        return -2;
    }
    const float scale = 1.0f / 32768.0f;
    for (int frame = 0; frame < n_frames; ++frame) {
        float sum = 0.0f;
        for (int channel = 0; channel < n_channels; ++channel) {
            sum += (float)interleaved[(size_t)frame * n_channels + channel];
        }
        mono[frame] = (sum / (float)n_channels) * scale;
    }
    return 0;
}

int audio_resampled_frame_count(
    int input_frames,
    int input_rate,
    int output_rate)
{
    if (input_frames <= 0 || input_rate <= 0 || output_rate <= 0) {
        return -1;
    }
    return 1 + (int)(((long long)(input_frames - 1) * output_rate) / input_rate);
}

int audio_resample_linear_f32(
    const float *input,
    int input_frames,
    int input_rate,
    float *output,
    int output_frames,
    int output_rate)
{
    if (input == NULL || output == NULL) {
        return -1;
    }
    const int expected = audio_resampled_frame_count(input_frames, input_rate, output_rate);
    if (expected <= 0 || output_frames != expected) {
        return -2;
    }
    for (int frame = 0; frame < output_frames; ++frame) {
        const long long numerator = (long long)frame * input_rate;
        const int left = (int)(numerator / output_rate);
        const int right = left + 1 < input_frames ? left + 1 : left;
        const float fraction = (float)(numerator % output_rate) / (float)output_rate;
        output[frame] = fmaf(input[right] - input[left], fraction, input[left]);
    }
    return 0;
}

int audio_resample_windowed_sinc_f32(
    const float *input,
    int input_frames,
    int input_rate,
    float *output,
    int output_frames,
    int output_rate,
    int radius)
{
    if (input == NULL || output == NULL) {
        return -1;
    }
    const int expected = audio_resampled_frame_count(input_frames, input_rate, output_rate);
    if (expected <= 0 || output_frames != expected || radius < 2 || radius > 128) {
        return -2;
    }
    const double ratio = (double)output_rate / (double)input_rate;
    const double cutoff = ratio < 1.0 ? ratio : 1.0;
    for (int frame = 0; frame < output_frames; ++frame) {
        const double source = (double)frame * (double)input_rate / (double)output_rate;
        const int center = (int)floor(source);
        double weighted = 0.0;
        double weight_sum = 0.0;
        for (int tap = center - radius + 1; tap <= center + radius; ++tap) {
            if (tap < 0 || tap >= input_frames) {
                continue;
            }
            const double distance = source - (double)tap;
            const double scaled = cutoff * distance;
            const double sinc = fabs(scaled) < 1.0e-12 ? 1.0 :
                sin(CK_AUDIO_PI_D * scaled) / (CK_AUDIO_PI_D * scaled);
            const double window_x = distance / (double)radius;
            if (fabs(window_x) >= 1.0) {
                continue;
            }
            const double window = 0.5 * (1.0 + cos(CK_AUDIO_PI_D * window_x));
            const double weight = cutoff * sinc * window;
            weighted += (double)input[tap] * weight;
            weight_sum += weight;
        }
        output[frame] = weight_sum != 0.0 ? (float)(weighted / weight_sum) : 0.0f;
    }
    return 0;
}

int audio_pad_or_truncate_f32(
    const float *input,
    int input_frames,
    float *output,
    int output_frames)
{
    if (input == NULL || output == NULL) {
        return -1;
    }
    if (input_frames <= 0 || output_frames <= 0) {
        return -2;
    }
    const int copied = input_frames < output_frames ? input_frames : output_frames;
    memmove(output, input, (size_t)copied * sizeof(float));
    if (copied < output_frames) {
        memset(
            output + copied,
            0,
            (size_t)(output_frames - copied) * sizeof(float));
    }
    return copied;
}

int audio_hann_window_f32(float *output, int frames, int periodic)
{
    if (output == NULL) {
        return -1;
    }
    if (frames <= 1 || (periodic != 0 && periodic != 1)) {
        return -2;
    }
    const double denominator = (double)(periodic ? frames : frames - 1);
    for (int frame = 0; frame < frames; ++frame) {
        const double phase = 2.0 * CK_AUDIO_PI_D * (double)frame / denominator;
        output[frame] = (float)(0.5 - 0.5 * cos(phase));
    }
    return 0;
}

int audio_preemphasis_f32(
    const float *input,
    float *output,
    int frames,
    float coefficient)
{
    if (input == NULL || output == NULL) {
        return -1;
    }
    if (frames <= 0 || !isfinite(coefficient)) {
        return -2;
    }
    if (input == output) {
        for (int frame = frames - 1; frame > 0; --frame) {
            output[frame] = input[frame] - coefficient * input[frame - 1];
        }
    } else {
        output[0] = input[0];
        for (int frame = 1; frame < frames; ++frame) {
            output[frame] = input[frame] - coefficient * input[frame - 1];
        }
    }
    return 0;
}

int audio_feature_normalize_per_feature_f32(
    const float *input,
    float *output,
    int channels,
    int frames,
    float epsilon)
{
    if (input == NULL || output == NULL) {
        return -1;
    }
    if (channels <= 0 || frames <= 0 || !isfinite(epsilon) || epsilon < 0.0f) {
        return -2;
    }
    const int denominator = frames > 1 ? frames - 1 : 1;
    for (int channel = 0; channel < channels; ++channel) {
        double sum = 0.0;
        double sum_squared_difference = 0.0;
        for (int frame = 0; frame < frames; ++frame) {
            sum += (double)input[(size_t)frame * channels + channel];
        }
        const double mean = sum / (double)frames;
        for (int frame = 0; frame < frames; ++frame) {
            const double difference =
                (double)input[(size_t)frame * channels + channel] - mean;
            sum_squared_difference += difference * difference;
        }
        float standard_deviation = sqrtf(
            (float)(sum_squared_difference / (double)denominator));
        if (isnan(standard_deviation)) {
            standard_deviation = 0.0f;
        }
        const float inverse_standard_deviation =
            1.0f / (standard_deviation + epsilon);
        for (int frame = 0; frame < frames; ++frame) {
            const size_t index = (size_t)frame * channels + channel;
            output[index] = (float)((double)input[index] - mean) *
                inverse_standard_deviation;
        }
    }
    return 0;
}

int audio_relative_sinusoidal_position_f32(
    float *output,
    int frames,
    int channels)
{
    if (output == NULL) {
        return -1;
    }
    if (frames <= 0 || frames > (INT_MAX / 2) + 1 ||
        channels <= 0 || (channels & 1) != 0) {
        return -2;
    }
    if ((size_t)(2 * frames - 1) > SIZE_MAX / (size_t)channels) {
        return -3;
    }
    const int positions = 2 * frames - 1;
    for (int row = 0; row < positions; ++row) {
        const float position = (float)(frames - 1 - row);
        for (int channel = 0; channel < channels; channel += 2) {
            const float exponent = (float)channel / (float)channels;
            const float inverse_frequency = 1.0f / powf(10000.0f, exponent);
            const float frequency = inverse_frequency * position;
            output[(size_t)row * channels + channel] = sinf(frequency);
            output[(size_t)row * channels + channel + 1] = cosf(frequency);
        }
    }
    return 0;
}

int audio_batch_norm_inference_channel_major_f32(
    const float *input,
    const float *running_mean,
    const float *running_variance,
    const float *weight,
    const float *bias,
    float *output,
    int channels,
    int frames,
    float epsilon)
{
    if (input == NULL || running_mean == NULL || running_variance == NULL ||
        weight == NULL || bias == NULL || output == NULL) {
        return -1;
    }
    if (channels <= 0 || frames <= 0 || !isfinite(epsilon) || epsilon < 0.0f) {
        return -2;
    }
    for (int channel = 0; channel < channels; ++channel) {
        const float variance = running_variance[channel];
        if (!isfinite(variance) || variance + epsilon <= 0.0f) {
            return -3;
        }
    }
    for (int channel = 0; channel < channels; ++channel) {
        const float variance = running_variance[channel];
        const float scale = weight[channel] / sqrtf(variance + epsilon);
        const float offset = bias[channel] - running_mean[channel] * scale;
        for (int frame = 0; frame < frames; ++frame) {
            const size_t index = (size_t)channel * frames + frame;
            output[index] = input[index] * scale + offset;
        }
    }
    return 0;
}

int audio_lstm_step_f32(
    const float *input,
    const float *weight_ih,
    const float *weight_hh,
    const float *bias_ih,
    const float *bias_hh,
    float *hidden_state,
    float *cell_state,
    float *output,
    float *gates_scratch,
    size_t gates_scratch_bytes,
    int input_size,
    int hidden_size)
{
    if (input == NULL || weight_ih == NULL || weight_hh == NULL ||
        bias_ih == NULL || bias_hh == NULL || hidden_state == NULL ||
        cell_state == NULL || output == NULL || gates_scratch == NULL) {
        return -1;
    }
    if (input_size <= 0 || hidden_size <= 0 ||
        (size_t)hidden_size > SIZE_MAX / 4u) {
        return -2;
    }
    const size_t gate_count = (size_t)hidden_size * 4u;
    if (gate_count > SIZE_MAX / sizeof(float) ||
        gates_scratch_bytes < gate_count * sizeof(float)) {
        return -3;
    }

    for (size_t gate = 0; gate < gate_count; ++gate) {
        float sum = bias_ih[gate] + bias_hh[gate];
        const float *input_weight = weight_ih + gate * (size_t)input_size;
        const float *hidden_weight = weight_hh + gate * (size_t)hidden_size;
        for (int index = 0; index < input_size; ++index) {
            sum += input_weight[index] * input[index];
        }
        for (int index = 0; index < hidden_size; ++index) {
            sum += hidden_weight[index] * hidden_state[index];
        }
        gates_scratch[gate] = sum;
    }

    for (int index = 0; index < hidden_size; ++index) {
        const float input_gate = 1.0f /
            (1.0f + expf(-gates_scratch[index]));
        const float forget_gate = 1.0f /
            (1.0f + expf(-gates_scratch[hidden_size + index]));
        const float cell_gate = tanhf(gates_scratch[2 * hidden_size + index]);
        const float output_gate = 1.0f /
            (1.0f + expf(-gates_scratch[3 * hidden_size + index]));
        const float cell = forget_gate * cell_state[index] +
            input_gate * cell_gate;
        const float hidden = output_gate * tanhf(cell);
        cell_state[index] = cell;
        hidden_state[index] = hidden;
        output[index] = hidden;
    }
    return 0;
}

int audio_stft_precompute_tables_f32(
    int n_fft,
    float *window,
    float *cos_table,
    float *sin_table)
{
    if (window == NULL || cos_table == NULL || sin_table == NULL) {
        return -1;
    }
    if (n_fft <= 0 || (n_fft & 1) != 0) {
        return -2;
    }
    const int bins = n_fft / 2 + 1;
    for (int sample = 0; sample < n_fft; ++sample) {
        window[sample] = 0.5f - 0.5f * cosf(
            2.0f * CK_AUDIO_PI_F * (float)sample / (float)n_fft);
    }
    for (int bin = 0; bin < bins; ++bin) {
        for (int sample = 0; sample < n_fft; ++sample) {
            const float angle = -2.0f * CK_AUDIO_PI_F *
                (float)(bin * sample) / (float)n_fft;
            const size_t index = (size_t)bin * n_fft + sample;
            cos_table[index] = cosf(angle);
            sin_table[index] = sinf(angle);
        }
    }
    return 0;
}

static double audio_hz_to_mel_slaney(double hz)
{
    if (hz < 1000.0) {
        return hz / (200.0 / 3.0);
    }
    return 15.0 + log(hz / 1000.0) / (log(6.4) / 27.0);
}

static double audio_mel_to_hz_slaney(double mel)
{
    if (mel < 15.0) {
        return (200.0 / 3.0) * mel;
    }
    return 1000.0 * exp((log(6.4) / 27.0) * (mel - 15.0));
}

int audio_whisper_mel_filters_slaney_f32(
    int sample_rate,
    int n_fft,
    int n_mels,
    float *mel_filters)
{
    if (mel_filters == NULL) {
        return -1;
    }
    if (sample_rate <= 0 || n_fft <= 0 || (n_fft & 1) != 0 || n_mels <= 0) {
        return -2;
    }
    const int bins = n_fft / 2 + 1;
    const double mel_min = audio_hz_to_mel_slaney(0.0);
    const double mel_max = audio_hz_to_mel_slaney((double)sample_rate / 2.0);
    for (int mel = 0; mel < n_mels; ++mel) {
        const double left_mel =
            mel_min + (mel_max - mel_min) * (double)mel / (double)(n_mels + 1);
        const double center_mel =
            mel_min + (mel_max - mel_min) * (double)(mel + 1) / (double)(n_mels + 1);
        const double right_mel =
            mel_min + (mel_max - mel_min) * (double)(mel + 2) / (double)(n_mels + 1);
        const double left = audio_mel_to_hz_slaney(left_mel);
        const double center = audio_mel_to_hz_slaney(center_mel);
        const double right = audio_mel_to_hz_slaney(right_mel);
        const double normalization = 2.0 / (right - left);
        for (int bin = 0; bin < bins; ++bin) {
            const double hz =
                ((double)sample_rate / 2.0) * (double)bin / (double)(bins - 1);
            const double lower = (hz - left) / (center - left);
            const double upper = (right - hz) / (right - center);
            const double triangle = fmax(0.0, fmin(lower, upper));
            mel_filters[(size_t)mel * bins + bin] =
                (float)(triangle * normalization);
        }
    }
    return 0;
}

int audio_stft_power_precomputed_f32(
    const float *samples,
    int n_samples,
    const float *window,
    const float *cos_table,
    const float *sin_table,
    int n_fft,
    int hop_length,
    float *power,
    int n_frames)
{
    if (samples == NULL || window == NULL || cos_table == NULL ||
        sin_table == NULL || power == NULL) {
        return -1;
    }
    if (n_fft <= 0 || hop_length <= 0 || n_samples <= n_fft / 2 ||
        (n_fft & 1) != 0 || n_frames <= 0) {
        return -2;
    }
    if (n_frames != n_samples / hop_length) {
        return -3;
    }
    const int bins = n_fft / 2 + 1;
    const int center = n_fft / 2;
    for (int frame = 0; frame < n_frames; ++frame) {
        for (int bin = 0; bin < bins; ++bin) {
            const float *cos_row = cos_table + (size_t)bin * n_fft;
            const float *sin_row = sin_table + (size_t)bin * n_fft;
            float real = 0.0f;
            float imag = 0.0f;
            for (int sample = 0; sample < n_fft; ++sample) {
                const int source = reflect_index(
                    frame * hop_length + sample - center, n_samples);
                const float value = samples[source] * window[sample];
                real = fmaf(value, cos_row[sample], real);
                imag = fmaf(value, sin_row[sample], imag);
            }
            power[(size_t)frame * bins + bin] =
                fmaf(real, real, imag * imag);
        }
    }
    return 0;
}

int audio_stft_power_centered_window_f32(
    const float *samples,
    int n_samples,
    const float *window,
    int window_length,
    const float *cos_table,
    const float *sin_table,
    int n_fft,
    int hop_length,
    int reflect_padding,
    float *power,
    int n_frames)
{
    if (samples == NULL || window == NULL || cos_table == NULL ||
        sin_table == NULL || power == NULL) {
        return -1;
    }
    if (n_samples <= 0 || window_length <= 0 || n_fft <= 0 ||
        window_length > n_fft || (n_fft & 1) != 0 || hop_length <= 0 ||
        n_frames <= 0 || (reflect_padding != 0 && reflect_padding != 1)) {
        return -2;
    }
    if (n_frames != n_samples / hop_length + 1) {
        return -3;
    }

    const int bins = n_fft / 2 + 1;
    const int center = n_fft / 2;
    const int window_start = (n_fft - window_length) / 2;
    for (int frame = 0; frame < n_frames; ++frame) {
        for (int bin = 0; bin < bins; ++bin) {
            const float *cos_row = cos_table + (size_t)bin * n_fft;
            const float *sin_row = sin_table + (size_t)bin * n_fft;
            float real = 0.0f;
            float imag = 0.0f;
            for (int sample = 0; sample < window_length; ++sample) {
                const int fft_sample = window_start + sample;
                int source = frame * hop_length + fft_sample - center;
                if (source < 0 || source >= n_samples) {
                    if (!reflect_padding) {
                        continue;
                    }
                    source = reflect_index(source, n_samples);
                }
                const float value = samples[source] * window[sample];
                real = fmaf(value, cos_row[fft_sample], real);
                imag = fmaf(value, sin_row[fft_sample], imag);
            }
            power[(size_t)frame * bins + bin] =
                fmaf(real, real, imag * imag);
        }
    }
    return 0;
}

int audio_log_mel_time_major_f32(
    const float *power,
    const float *mel_filters,
    float *log_mel,
    int frames,
    int bins,
    int channels,
    float epsilon)
{
    if (power == NULL || mel_filters == NULL || log_mel == NULL) {
        return -1;
    }
    if (frames <= 0 || bins <= 0 || channels <= 0 || epsilon <= 0.0f) {
        return -2;
    }
    for (int frame = 0; frame < frames; ++frame) {
        const float *spectrum = power + (size_t)frame * bins;
        float *output = log_mel + (size_t)frame * channels;
        for (int channel = 0; channel < channels; ++channel) {
            const float *filter = mel_filters + (size_t)channel * bins;
            float sum = 0.0f;
            for (int bin = 0; bin < bins; ++bin) {
                sum = fmaf(spectrum[bin], filter[bin], sum);
            }
            output[channel] = logf(sum + epsilon);
        }
    }
    return 0;
}

static void audio_stft_power_fft400_frame_f32(
    const float *samples,
    int n_samples,
    int frame,
    const float *window,
    const float *cos_table,
    const float *sin_table,
    float *power,
    float *fft_scratch);

int audio_stft_power_fft400_f32(
    const float *samples,
    int n_samples,
    const float *window,
    const float *cos_table,
    const float *sin_table,
    int hop_length,
    float *power,
    int n_frames,
    float *fft_scratch)
{
    const int n_fft = CK_AUDIO_WHISPER_N_FFT;
    if (samples == NULL || window == NULL || cos_table == NULL ||
        sin_table == NULL || power == NULL || fft_scratch == NULL) {
        return -1;
    }
    if (hop_length != CK_AUDIO_WHISPER_HOP_LENGTH ||
        n_samples <= n_fft / 2 || n_frames <= 0 ||
        n_frames != n_samples / hop_length) {
        return -2;
    }
    for (int frame = 0; frame < n_frames; ++frame) {
        audio_stft_power_fft400_frame_f32(
            samples,
            n_samples,
            frame,
            window,
            cos_table,
            sin_table,
            power + (size_t)frame * CK_AUDIO_WHISPER_POWER_BINS,
            fft_scratch);
    }
    return 0;
}

static void audio_stft_power_fft400_frame_f32(
    const float *samples,
    int n_samples,
    int frame,
    const float *window,
    const float *cos_table,
    const float *sin_table,
    float *power,
    float *fft_scratch)
{
    const int radix = 20;
    const int center = CK_AUDIO_WHISPER_N_FFT / 2;
    float *stage_real = fft_scratch;
    float *stage_imag = fft_scratch + CK_AUDIO_WHISPER_N_FFT;
    for (int p = 0; p < radix; ++p) {
        for (int k = 0; k < radix; ++k) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int q = 0; q < radix; ++q) {
                const int sample = p + radix * q;
                const int source = reflect_index(
                    frame * CK_AUDIO_WHISPER_HOP_LENGTH + sample - center,
                    n_samples);
                const float value = samples[source] * window[sample];
                const size_t twiddle =
                    (size_t)k * CK_AUDIO_WHISPER_N_FFT + radix * q;
                real = fmaf(value, cos_table[twiddle], real);
                imag = fmaf(value, sin_table[twiddle], imag);
            }
            stage_real[p * radix + k] = real;
            stage_imag[p * radix + k] = imag;
        }
    }
    for (int frequency = 0;
         frequency < CK_AUDIO_WHISPER_POWER_BINS;
         ++frequency) {
        const int k = frequency % radix;
        float real = 0.0f;
        float imag = 0.0f;
        for (int p = 0; p < radix; ++p) {
            const float a = stage_real[p * radix + k];
            const float b = stage_imag[p * radix + k];
            const size_t twiddle =
                (size_t)frequency * CK_AUDIO_WHISPER_N_FFT + p;
            const float c = cos_table[twiddle];
            const float s = sin_table[twiddle];
            real = fmaf(a, c, fmaf(-b, s, real));
            imag = fmaf(a, s, fmaf(b, c, imag));
        }
        power[frequency] = fmaf(real, real, imag * imag);
    }
}

int audio_whisper_log_mel_window_wav_pcm16_f32(
    const uint8_t *bytes,
    size_t byte_count,
    int start_frame,
    int target_sample_rate,
    const float *window,
    const float *cos_table,
    const float *sin_table,
    const float *mel_filters,
    int n_mels,
    int output_frames,
    float *log_mel)
{
    if (bytes == NULL || window == NULL || cos_table == NULL ||
        sin_table == NULL || mel_filters == NULL || log_mel == NULL) {
        return -1;
    }
    CKAudioWavInfo info;
    if (audio_wav_parse_memory(bytes, byte_count, &info) != 0 ||
        info.sample_rate != target_sample_rate ||
        target_sample_rate != CK_AUDIO_WHISPER_SAMPLE_RATE ||
        start_frame < 0 ||
        start_frame % CK_AUDIO_WHISPER_HOP_LENGTH != 0 ||
        n_mels <= 0 || output_frames <= 0) {
        return -2;
    }
    float *samples = (float *)malloc((size_t)info.frames * sizeof(float));
    if (samples == NULL) {
        return -3;
    }
    const int decoded = audio_wav_decode_pcm16_mono_f32(
        bytes, byte_count, &info, samples, info.frames);
    if (decoded != info.frames) {
        free(samples);
        return -4;
    }

    memset(
        log_mel,
        0,
        (size_t)n_mels * (size_t)output_frames * sizeof(float));
    const int global_frames = info.frames / CK_AUDIO_WHISPER_HOP_LENGTH;
    const int start_feature = start_frame / CK_AUDIO_WHISPER_HOP_LENGTH;
    float maximum = -INFINITY;
    float power[CK_AUDIO_WHISPER_POWER_BINS];
    float fft_scratch[2 * CK_AUDIO_WHISPER_N_FFT];
    for (int frame = 0; frame < global_frames; ++frame) {
        audio_stft_power_fft400_frame_f32(
            samples,
            info.frames,
            frame,
            window,
            cos_table,
            sin_table,
            power,
            fft_scratch);
        for (int mel = 0; mel < n_mels; ++mel) {
            const float *filter =
                mel_filters + (size_t)mel * CK_AUDIO_WHISPER_POWER_BINS;
            float sum = 0.0f;
            for (int bin = 0; bin < CK_AUDIO_WHISPER_POWER_BINS; ++bin) {
                sum = fmaf(filter[bin], power[bin], sum);
            }
            const float value = log10f(fmaxf(sum, 1.0e-10f));
            maximum = fmaxf(maximum, value);
            const int output_frame = frame - start_feature;
            if (output_frame >= 0 && output_frame < output_frames) {
                log_mel[(size_t)mel * output_frames + output_frame] = value;
            }
        }
    }
    free(samples);
    if (!isfinite(maximum)) {
        return -5;
    }

    const int available = global_frames - start_feature;
    const int valid_frames =
        available < output_frames ? (available > 0 ? available : 0) : output_frames;
    const float floor = maximum - 8.0f;
    for (int mel = 0; mel < n_mels; ++mel) {
        float *output = log_mel + (size_t)mel * output_frames;
        for (int frame = 0; frame < valid_frames; ++frame) {
            output[frame] = (fmaxf(output[frame], floor) + 4.0f) / 4.0f;
        }
    }
    return valid_frames;
}

size_t audio_fastconformer_subsampling_workspace_bytes(
    int feature_frames,
    int feature_channels,
    int conv_channels,
    int kernel_size,
    int stride)
{
    if (feature_frames <= 0 || feature_channels <= 0 || conv_channels <= 0 ||
        kernel_size != 3 || stride != 2) {
        return 0;
    }
    const int padding = kernel_size / 2;
    const int stage_frames = conv_output_extent(
        feature_frames, kernel_size, stride, padding);
    const int stage_width = conv_output_extent(
        feature_channels, kernel_size, stride, padding);
    if (stage_frames <= 0 || stage_width <= 0) {
        return 0;
    }
    size_t elements = 0;
    size_t bytes = 0;
    if (checked_mul_size((size_t)stage_frames, (size_t)stage_width, &elements) != 0 ||
        checked_mul_size(elements, (size_t)conv_channels, &elements) != 0 ||
        checked_mul_size(elements, 2u * sizeof(float), &bytes) != 0) {
        return 0;
    }
    return bytes;
}

static void zero_subsampling_padding(
    float *value,
    int channels,
    int frames,
    int width,
    int live_frames)
{
    if (live_frames >= frames) {
        return;
    }
    const size_t row_bytes = (size_t)width * sizeof(float);
    for (int channel = 0; channel < channels; ++channel) {
        float *channel_data = value + (size_t)channel * (size_t)frames * (size_t)width;
        for (int frame = live_frames; frame < frames; ++frame) {
            memset(channel_data + (size_t)frame * (size_t)width, 0, row_bytes);
        }
    }
}

static void relu_subsampling_inplace(float *value, size_t elements)
{
    for (size_t index = 0; index < elements; ++index) {
        if (value[index] < 0.0f) {
            value[index] = 0.0f;
        }
    }
}

int audio_fastconformer_subsampling_f32(
    const float *features,
    const float *conv0_weight,
    const float *conv0_bias,
    const float *depthwise1_weight,
    const float *depthwise1_bias,
    const float *pointwise1_weight,
    const float *pointwise1_bias,
    const float *depthwise2_weight,
    const float *depthwise2_bias,
    const float *pointwise2_weight,
    const float *pointwise2_bias,
    const float *linear_weight,
    const float *linear_bias,
    float *output,
    void *workspace,
    size_t workspace_bytes,
    int feature_frames,
    int live_frames,
    int feature_channels,
    int conv_channels,
    int hidden_size,
    int kernel_size,
    int stride,
    int output_capacity_frames,
    int *output_frames)
{
    if (features == NULL || conv0_weight == NULL || conv0_bias == NULL ||
        depthwise1_weight == NULL || depthwise1_bias == NULL ||
        pointwise1_weight == NULL || pointwise1_bias == NULL ||
        depthwise2_weight == NULL || depthwise2_bias == NULL ||
        pointwise2_weight == NULL || pointwise2_bias == NULL ||
        linear_weight == NULL || linear_bias == NULL || output == NULL ||
        workspace == NULL || output_frames == NULL || feature_frames <= 0 ||
        live_frames <= 0 || live_frames > feature_frames || feature_channels <= 0 ||
        conv_channels <= 0 || hidden_size <= 0 || kernel_size != 3 || stride != 2 ||
        output_capacity_frames <= 0) {
        return -1;
    }

    const size_t required_workspace = audio_fastconformer_subsampling_workspace_bytes(
        feature_frames, feature_channels, conv_channels, kernel_size, stride);
    if (required_workspace == 0 || workspace_bytes < required_workspace) {
        return -2;
    }

    const int padding = kernel_size / 2;
    int heights[3];
    int widths[3];
    int live_heights[3];
    heights[0] = conv_output_extent(feature_frames, kernel_size, stride, padding);
    widths[0] = conv_output_extent(feature_channels, kernel_size, stride, padding);
    live_heights[0] = conv_output_extent(live_frames, kernel_size, stride, padding);
    for (int stage = 1; stage < 3; ++stage) {
        heights[stage] = conv_output_extent(heights[stage - 1], kernel_size, stride, padding);
        widths[stage] = conv_output_extent(widths[stage - 1], kernel_size, stride, padding);
        live_heights[stage] = conv_output_extent(
            live_heights[stage - 1], kernel_size, stride, padding);
    }
    if (heights[2] <= 0 || widths[2] <= 0 || live_heights[2] <= 0 ||
        output_capacity_frames < heights[2]) {
        return -3;
    }

    size_t stage0_elements = 0;
    if (checked_mul_size((size_t)conv_channels, (size_t)heights[0], &stage0_elements) != 0 ||
        checked_mul_size(stage0_elements, (size_t)widths[0], &stage0_elements) != 0) {
        return -4;
    }
    float *ping = (float *)workspace;
    float *pong = ping + stage0_elements;

    int status = audio_conv2d_whc_grouped_f32(
        features, conv0_weight, conv0_bias, ping,
        feature_channels, feature_frames, 1, conv_channels,
        kernel_size, kernel_size, stride, stride, padding, padding, 1,
        widths[0], heights[0]);
    if (status != 0) return -10;
    zero_subsampling_padding(ping, conv_channels, heights[0], widths[0], live_heights[0]);
    relu_subsampling_inplace(ping, stage0_elements);

    status = audio_conv2d_whc_grouped_f32(
        ping, depthwise1_weight, depthwise1_bias, pong,
        widths[0], heights[0], conv_channels, conv_channels,
        kernel_size, kernel_size, stride, stride, padding, padding, conv_channels,
        widths[1], heights[1]);
    if (status != 0) return -11;
    zero_subsampling_padding(pong, conv_channels, heights[1], widths[1], live_heights[1]);

    status = audio_conv2d_whc_grouped_f32(
        pong, pointwise1_weight, pointwise1_bias, ping,
        widths[1], heights[1], conv_channels, conv_channels,
        1, 1, 1, 1, 0, 0, 1, widths[1], heights[1]);
    if (status != 0) return -12;
    size_t stage1_elements = 0;
    if (checked_mul_size((size_t)conv_channels, (size_t)heights[1], &stage1_elements) != 0 ||
        checked_mul_size(stage1_elements, (size_t)widths[1], &stage1_elements) != 0) {
        return -4;
    }
    zero_subsampling_padding(ping, conv_channels, heights[1], widths[1], live_heights[1]);
    relu_subsampling_inplace(ping, stage1_elements);

    status = audio_conv2d_whc_grouped_f32(
        ping, depthwise2_weight, depthwise2_bias, pong,
        widths[1], heights[1], conv_channels, conv_channels,
        kernel_size, kernel_size, stride, stride, padding, padding, conv_channels,
        widths[2], heights[2]);
    if (status != 0) return -13;
    zero_subsampling_padding(pong, conv_channels, heights[2], widths[2], live_heights[2]);

    status = audio_conv2d_whc_grouped_f32(
        pong, pointwise2_weight, pointwise2_bias, ping,
        widths[2], heights[2], conv_channels, conv_channels,
        1, 1, 1, 1, 0, 0, 1, widths[2], heights[2]);
    if (status != 0) return -14;
    size_t final_elements = 0;
    if (checked_mul_size((size_t)conv_channels, (size_t)heights[2], &final_elements) != 0 ||
        checked_mul_size(final_elements, (size_t)widths[2], &final_elements) != 0) {
        return -4;
    }
    zero_subsampling_padding(ping, conv_channels, heights[2], widths[2], live_heights[2]);
    relu_subsampling_inplace(ping, final_elements);

    if (widths[2] > INT_MAX / conv_channels) return -4;
    const int flattened_width = conv_channels * widths[2];
    for (int frame = 0; frame < heights[2]; ++frame) {
        float *row = pong + (size_t)frame * (size_t)flattened_width;
        for (int channel = 0; channel < conv_channels; ++channel) {
            const float *source = ping +
                ((size_t)channel * (size_t)heights[2] + (size_t)frame) * (size_t)widths[2];
            memcpy(row + (size_t)channel * (size_t)widths[2], source,
                   (size_t)widths[2] * sizeof(float));
        }
    }

    gemm_nt_f32_llama_production_parallel_dispatch(
        pong, linear_weight, linear_bias, output,
        heights[2], hidden_size, flattened_width);
    *output_frames = heights[2];
    return 0;
}

typedef struct {
    const float *input;
    const float *weight;
    const float *bias;
    float *output;
    int input_channels;
    int output_channels;
    int input_frames;
    int kernel_size;
    int stride;
    int padding;
    int output_frames;
    int groups;
    int input_channels_per_group;
    int output_channels_per_group;
    int use_stride2_contiguous;
} ck_audio_conv1d_f32_args_t;

#if defined(__AVX2__) && defined(__FMA__)
static inline __m256 ck_audio_load_stride2_8(const float *input)
{
    const __m256i select_even = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    const __m256 lo = _mm256_permutevar8x32_ps(
        _mm256_loadu_ps(input), select_even);
    const __m256 hi = _mm256_permutevar8x32_ps(
        _mm256_loadu_ps(input + 8), select_even);
    return _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm256_castps256_ps128(lo)),
        _mm256_castps256_ps128(hi), 1);
}

#endif

static void ck_audio_conv1d_channel_major_f32_work(
    int ith,
    int nth,
    void *opaque)
{
    const ck_audio_conv1d_f32_args_t *args =
        (const ck_audio_conv1d_f32_args_t *)opaque;
    for (int out_channel = ith; out_channel < args->output_channels;
         out_channel += nth) {
        const int group = out_channel / args->output_channels_per_group;
        const int input_channel_offset = group * args->input_channels_per_group;
        const float *weight_channel = args->weight +
            (size_t)out_channel * args->input_channels_per_group * args->kernel_size;
        float *output_channel = args->output +
            (size_t)out_channel * args->output_frames;
        int out_frame = 0;
        const int interior_begin =
            (args->padding + args->stride - 1) / args->stride;
        for (; out_frame < interior_begin && out_frame < args->output_frames;
             ++out_frame) {
            float sum = args->bias != NULL ? args->bias[out_channel] : 0.0f;
            for (int local_channel = 0;
                 local_channel < args->input_channels_per_group;
                 ++local_channel) {
                const int in_channel = input_channel_offset + local_channel;
                const float *input_channel = args->input +
                    (size_t)in_channel * args->input_frames;
                const float *weight_row = weight_channel +
                    (size_t)local_channel * args->kernel_size;
                for (int kernel = 0; kernel < args->kernel_size; ++kernel) {
                    const int in_frame =
                        out_frame * args->stride + kernel - args->padding;
                    if (in_frame >= 0 && in_frame < args->input_frames) {
                        sum = fmaf(input_channel[in_frame], weight_row[kernel], sum);
                    }
                }
            }
            output_channel[out_frame] = sum;
        }
#if defined(__AVX2__) && defined(__FMA__)
        for (; out_frame + 7 < args->output_frames &&
               (out_frame + 7) * args->stride + args->kernel_size - 1 -
                   args->padding < args->input_frames;
             out_frame += 8) {
            __m256 sums = _mm256_set1_ps(
                args->bias != NULL ? args->bias[out_channel] : 0.0f);
            for (int local_channel = 0;
                 local_channel < args->input_channels_per_group;
                 ++local_channel) {
                const int in_channel = input_channel_offset + local_channel;
                const float *input_channel = args->input +
                    (size_t)in_channel * args->input_frames;
                const float *weight_row = weight_channel +
                    (size_t)local_channel * args->kernel_size;
                for (int kernel = 0; kernel < args->kernel_size; ++kernel) {
                    const int base =
                        out_frame * args->stride + kernel - args->padding;
                    __m256 samples;
                    if (args->stride == 1) {
                        samples = _mm256_loadu_ps(input_channel + base);
                    } else if (args->stride == 2 &&
                               args->use_stride2_contiguous &&
                               base + 15 < args->input_frames) {
                        samples = ck_audio_load_stride2_8(input_channel + base);
                    } else if (args->stride == 2) {
                        const __m256i indices = _mm256_setr_epi32(
                            base, base + 2, base + 4, base + 6,
                            base + 8, base + 10, base + 12, base + 14);
                        samples = _mm256_i32gather_ps(input_channel, indices, 4);
                    } else {
                        samples = _mm256_setr_ps(
                            input_channel[base],
                            input_channel[base + args->stride],
                            input_channel[base + 2 * args->stride],
                            input_channel[base + 3 * args->stride],
                            input_channel[base + 4 * args->stride],
                            input_channel[base + 5 * args->stride],
                            input_channel[base + 6 * args->stride],
                            input_channel[base + 7 * args->stride]);
                    }
                    sums = _mm256_fmadd_ps(
                        samples, _mm256_set1_ps(weight_row[kernel]), sums);
                }
            }
            _mm256_storeu_ps(output_channel + out_frame, sums);
        }
#endif
        for (; out_frame < args->output_frames; ++out_frame) {
            float sum = args->bias != NULL ? args->bias[out_channel] : 0.0f;
            for (int local_channel = 0;
                 local_channel < args->input_channels_per_group;
                 ++local_channel) {
                const int in_channel = input_channel_offset + local_channel;
                const float *input_channel = args->input +
                    (size_t)in_channel * args->input_frames;
                const float *weight_row = weight_channel +
                    (size_t)local_channel * args->kernel_size;
                for (int kernel = 0; kernel < args->kernel_size; ++kernel) {
                    const int in_frame =
                        out_frame * args->stride + kernel - args->padding;
                    if (in_frame >= 0 && in_frame < args->input_frames) {
                        sum = fmaf(input_channel[in_frame], weight_row[kernel], sum);
                    }
                }
            }
            output_channel[out_frame] = sum;
        }
    }
}

int audio_conv1d_channel_major_f32(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    int input_channels,
    int output_channels,
    int input_frames,
    int kernel_size,
    int stride,
    int padding,
    int output_frames)
{
    return audio_conv1d_channel_major_grouped_f32(
        input, weight, bias, output, input_channels, output_channels,
        input_frames, kernel_size, stride, padding, 1, output_frames);
}

int audio_conv1d_channel_major_grouped_f32(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    int input_channels,
    int output_channels,
    int input_frames,
    int kernel_size,
    int stride,
    int padding,
    int groups,
    int output_frames)
{
    if (input == NULL || weight == NULL || output == NULL) {
        return -1;
    }
    if (input_channels <= 0 || output_channels <= 0 || input_frames <= 0 ||
        kernel_size <= 0 || stride <= 0 || padding < 0 || groups <= 0 ||
        output_frames <= 0 || input_channels % groups != 0 ||
        output_channels % groups != 0) {
        return -2;
    }
    const int64_t padded_frames =
        (int64_t)input_frames + 2 * (int64_t)padding - kernel_size;
    if (padded_frames < 0 || padded_frames / stride + 1 > INT_MAX ||
        output_frames != (int)(padded_frames / stride + 1)) {
        return -3;
    }
    const char *disable_stride2 =
        getenv("CK_DISABLE_AUDIO_CONV_STRIDE2_CONTIGUOUS");
    ck_audio_conv1d_f32_args_t args = {
        .input = input,
        .weight = weight,
        .bias = bias,
        .output = output,
        .input_channels = input_channels,
        .output_channels = output_channels,
        .input_frames = input_frames,
        .kernel_size = kernel_size,
        .stride = stride,
        .padding = padding,
        .output_frames = output_frames,
        .groups = groups,
        .input_channels_per_group = input_channels / groups,
        .output_channels_per_group = output_channels / groups,
        .use_stride2_contiguous = !(
            disable_stride2 && disable_stride2[0] &&
            strcmp(disable_stride2, "0") != 0),
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active > output_channels) active = output_channels;
    if (pool != NULL && active > 1) {
        ck_threadpool_dispatch_n(
            pool, active, ck_audio_conv1d_channel_major_f32_work, &args);
    } else {
        ck_audio_conv1d_channel_major_f32_work(0, 1, &args);
    }
    return 0;
}

typedef struct {
    const float *input;
    const float *weight;
    const float *bias;
    float *output;
    int input_width;
    int input_height;
    int input_channels;
    int output_channels;
    int kernel_width;
    int kernel_height;
    int stride_width;
    int stride_height;
    int padding_width;
    int padding_height;
    int groups;
    int output_width;
    int output_height;
} ck_audio_conv2d_whc_f32_args_t;

static void ck_audio_conv2d_whc_grouped_f32_range(
    int begin,
    int end,
    void *opaque)
{
    const ck_audio_conv2d_whc_f32_args_t *args =
        (const ck_audio_conv2d_whc_f32_args_t *)opaque;
    const int outputs_per_channel = args->output_width * args->output_height;
    const int input_channels_per_group = args->input_channels / args->groups;
    const int output_channels_per_group = args->output_channels / args->groups;
    for (int index = begin; index < end; ++index) {
        const int output_channel = index / outputs_per_channel;
        const int spatial = index - output_channel * outputs_per_channel;
        const int output_y = spatial / args->output_width;
        const int output_x = spatial - output_y * args->output_width;
        const int group = output_channel / output_channels_per_group;
        const int input_channel_begin = group * input_channels_per_group;
        float sum = args->bias != NULL ? args->bias[output_channel] : 0.0f;
        for (int input_channel_offset = 0;
             input_channel_offset < input_channels_per_group;
             ++input_channel_offset) {
            const int input_channel = input_channel_begin + input_channel_offset;
            for (int kernel_y = 0; kernel_y < args->kernel_height; ++kernel_y) {
                const int input_y = output_y * args->stride_height + kernel_y -
                    args->padding_height;
                if (input_y < 0 || input_y >= args->input_height) {
                    continue;
                }
                for (int kernel_x = 0; kernel_x < args->kernel_width; ++kernel_x) {
                    const int input_x = output_x * args->stride_width + kernel_x -
                        args->padding_width;
                    if (input_x < 0 || input_x >= args->input_width) {
                        continue;
                    }
                    const size_t input_index =
                        ((size_t)input_channel * args->input_height + input_y) *
                        args->input_width + input_x;
                    const size_t weight_index =
                        (((size_t)output_channel * input_channels_per_group +
                          input_channel_offset) * args->kernel_height + kernel_y) *
                        args->kernel_width + kernel_x;
                    sum = fmaf(args->input[input_index], args->weight[weight_index], sum);
                }
            }
        }
        args->output[index] = sum;
    }
}

int audio_conv2d_whc_grouped_f32(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    int input_width,
    int input_height,
    int input_channels,
    int output_channels,
    int kernel_width,
    int kernel_height,
    int stride_width,
    int stride_height,
    int padding_width,
    int padding_height,
    int groups,
    int output_width,
    int output_height)
{
    if (input == NULL || weight == NULL || output == NULL) {
        return -1;
    }
    if (input_width <= 0 || input_height <= 0 || input_channels <= 0 ||
        output_channels <= 0 || kernel_width <= 0 || kernel_height <= 0 ||
        stride_width <= 0 || stride_height <= 0 || padding_width < 0 ||
        padding_height < 0 || groups <= 0 || output_width <= 0 ||
        output_height <= 0 || input_channels % groups != 0 ||
        output_channels % groups != 0) {
        return -2;
    }
    const int expected_width =
        (input_width + 2 * padding_width - kernel_width) / stride_width + 1;
    const int expected_height =
        (input_height + 2 * padding_height - kernel_height) / stride_height + 1;
    if (output_width != expected_width || output_height != expected_height) {
        return -3;
    }
    ck_audio_conv2d_whc_f32_args_t args = {
        .input = input,
        .weight = weight,
        .bias = bias,
        .output = output,
        .input_width = input_width,
        .input_height = input_height,
        .input_channels = input_channels,
        .output_channels = output_channels,
        .kernel_width = kernel_width,
        .kernel_height = kernel_height,
        .stride_width = stride_width,
        .stride_height = stride_height,
        .padding_width = padding_width,
        .padding_height = padding_height,
        .groups = groups,
        .output_width = output_width,
        .output_height = output_height,
    };
    const int output_elements = output_channels * output_height * output_width;
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active > output_elements) {
        active = output_elements;
    }
    if (pool != NULL && active > 1) {
        const int grain = output_width > 0 ? output_width : 1;
        ck_threadpool_parallel_for_n(
            pool, active, 0, output_elements, grain,
            ck_audio_conv2d_whc_grouped_f32_range, &args);
    } else {
        ck_audio_conv2d_whc_grouped_f32_range(0, output_elements, &args);
    }
    return 0;
}

typedef struct {
    const float *value;
    const float *gate;
    float *output;
} ck_audio_glu_split_f32_args_t;

static void ck_audio_glu_split_f32_range(int begin, int end, void *opaque)
{
    const ck_audio_glu_split_f32_args_t *args =
        (const ck_audio_glu_split_f32_args_t *)opaque;
    for (int index = begin; index < end; ++index) {
        const float gate = args->gate[index];
        const float sigmoid = 1.0f / (1.0f + expf(-gate));
        args->output[index] = args->value[index] * sigmoid;
    }
}

int audio_glu_split_channel_major_f32(
    const float *input,
    float *output,
    int channels,
    int frames)
{
    if (input == NULL || output == NULL) {
        return -1;
    }
    if (channels <= 0 || frames <= 0) {
        return -2;
    }
    const int elements = channels * frames;
    ck_audio_glu_split_f32_args_t args = {
        .value = input,
        .gate = input + elements,
        .output = output,
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active > elements) {
        active = elements;
    }
    if (pool != NULL && active > 1) {
        ck_threadpool_parallel_for_n(
            pool, active, 0, elements, 256,
            ck_audio_glu_split_f32_range, &args);
    } else {
        ck_audio_glu_split_f32_range(0, elements, &args);
    }
    return 0;
}

typedef struct {
    const float *raw_scores;
    float *scores;
    int query_frames;
    int raw_key_frames;
} ck_audio_relative_shift_f32_args_t;

static void ck_audio_relative_shift_f32_range(
    int begin, int end, void *opaque)
{
    const ck_audio_relative_shift_f32_args_t *args =
        (const ck_audio_relative_shift_f32_args_t *)opaque;
    const int frames = args->query_frames;
    const int raw_frames = args->raw_key_frames;
    for (int row = begin; row < end; ++row) {
        const int query = row % frames;
        const float *raw = args->raw_scores + (size_t)row * raw_frames;
        float *output = args->scores + (size_t)row * frames;
        const int origin = frames - 1 - query;
        for (int key = 0; key < frames; ++key) {
            output[key] = raw[origin + key];
        }
    }
}

int audio_relative_shift_f32(
    const float *raw_scores,
    float *scores,
    int heads,
    int query_frames)
{
    if (raw_scores == NULL || scores == NULL) {
        return -1;
    }
    if (heads <= 0 || query_frames <= 0) {
        return -2;
    }
    const int rows = heads * query_frames;
    ck_audio_relative_shift_f32_args_t args = {
        .raw_scores = raw_scores,
        .scores = scores,
        .query_frames = query_frames,
        .raw_key_frames = 2 * query_frames - 1,
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active > rows) {
        active = rows;
    }
    if (pool != NULL && active > 1) {
        ck_threadpool_parallel_for_n(
            pool, active, 0, rows, 1,
            ck_audio_relative_shift_f32_range, &args);
    } else {
        ck_audio_relative_shift_f32_range(0, rows, &args);
    }
    return 0;
}

typedef struct {
    const float *query;
    const float *key;
    const float *value;
    const float *relative_key;
    const float *bias_u;
    const float *bias_v;
    float *output;
    float *scores_scratch;
    int frames;
    int heads;
    int head_dim;
    int channels;
    float scale;
} ck_audio_conformer_relative_attention_f32_args_t;

static void ck_audio_conformer_relative_attention_f32_head_range(
    int begin,
    int end,
    void *opaque)
{
    const ck_audio_conformer_relative_attention_f32_args_t *args =
        (const ck_audio_conformer_relative_attention_f32_args_t *)opaque;
    for (int head = begin; head < end; ++head) {
        const int head_offset = head * args->head_dim;
        const float *u = args->bias_u + head_offset;
        const float *v = args->bias_v + head_offset;
        float *scores = args->scores_scratch + (size_t)head * args->frames;
        for (int query_frame = 0; query_frame < args->frames; ++query_frame) {
            const float *q = args->query +
                (size_t)query_frame * args->channels + head_offset;
            float maximum = -INFINITY;
            for (int key_frame = 0; key_frame < args->frames; ++key_frame) {
                const float *k = args->key +
                    (size_t)key_frame * args->channels + head_offset;
                const int relative_frame =
                    args->frames - 1 + key_frame - query_frame;
                const float *r = args->relative_key +
                    (size_t)relative_frame * args->channels + head_offset;
                float content = 0.0f;
                float position = 0.0f;
                for (int dim = 0; dim < args->head_dim; ++dim) {
                    content += (q[dim] + u[dim]) * k[dim];
                    position += (q[dim] + v[dim]) * r[dim];
                }
                const float score = (content + position) * args->scale;
                scores[key_frame] = score;
                if (score > maximum) {
                    maximum = score;
                }
            }
            float denominator = 0.0f;
            for (int key_frame = 0; key_frame < args->frames; ++key_frame) {
                const float probability = expf(scores[key_frame] - maximum);
                scores[key_frame] = probability;
                denominator += probability;
            }
            const float inverse_denominator = 1.0f / denominator;
            float *out = args->output +
                (size_t)query_frame * args->channels + head_offset;
            for (int dim = 0; dim < args->head_dim; ++dim) {
                float sum = 0.0f;
                for (int key_frame = 0; key_frame < args->frames; ++key_frame) {
                    const float *value_row = args->value +
                        (size_t)key_frame * args->channels + head_offset;
                    sum += scores[key_frame] * inverse_denominator * value_row[dim];
                }
                out[dim] = sum;
            }
        }
    }
}

int audio_conformer_relative_attention_f32(
    const float *query,
    const float *key,
    const float *value,
    const float *relative_key,
    const float *bias_u,
    const float *bias_v,
    float *output,
    int frames,
    int heads,
    int head_dim,
    float scale,
    float *scores_scratch,
    size_t scores_scratch_bytes)
{
    if (query == NULL || key == NULL || value == NULL || relative_key == NULL ||
        bias_u == NULL || bias_v == NULL || output == NULL ||
        scores_scratch == NULL) {
        return -1;
    }
    if (frames <= 0 || frames > (INT_MAX / 2) + 1 || heads <= 0 ||
        head_dim <= 0 || heads > INT_MAX / head_dim || !isfinite(scale) ||
        scale <= 0.0f) {
        return -2;
    }
    if ((size_t)frames > SIZE_MAX / (size_t)heads ||
        (size_t)frames * (size_t)heads > SIZE_MAX / sizeof(float) ||
        scores_scratch_bytes <
            (size_t)frames * (size_t)heads * sizeof(float)) {
        return -3;
    }
    const int channels = heads * head_dim;
    ck_audio_conformer_relative_attention_f32_args_t args = {
        .query = query,
        .key = key,
        .value = value,
        .relative_key = relative_key,
        .bias_u = bias_u,
        .bias_v = bias_v,
        .output = output,
        .scores_scratch = scores_scratch,
        .frames = frames,
        .heads = heads,
        .head_dim = head_dim,
        .channels = channels,
        .scale = scale,
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active > heads) {
        active = heads;
    }
    if (pool != NULL && active > 1) {
        ck_threadpool_parallel_for_n(
            pool, active, 0, heads, 1,
            ck_audio_conformer_relative_attention_f32_head_range, &args);
    } else {
        ck_audio_conformer_relative_attention_f32_head_range(0, heads, &args);
    }
    return 0;
}

int audio_transpose_channel_to_token_f32(
    const float *input,
    float *output,
    int channels,
    int frames)
{
    if (input == NULL || output == NULL) {
        return -1;
    }
    if (channels <= 0 || frames <= 0) {
        return -2;
    }
    for (int frame = 0; frame < frames; ++frame) {
        for (int channel = 0; channel < channels; ++channel) {
            output[(size_t)frame * channels + channel] =
                input[(size_t)channel * frames + frame];
        }
    }
    return 0;
}

int audio_scaled_residual_add_f32(
    const float *residual,
    const float *branch,
    float scale,
    float *output,
    size_t elements)
{
    if (residual == NULL || branch == NULL || output == NULL) {
        return -1;
    }
    if (elements == 0 || !isfinite(scale)) {
        return -2;
    }
    for (size_t index = 0; index < elements; ++index) {
        /* Preserve the pinned elementwise multiply then add cast boundary. */
        volatile float scaled = branch[index] * scale;
        output[index] = residual[index] + scaled;
    }
    return 0;
}

size_t audio_fastconformer_block_workspace_bytes(
    int frames,
    int hidden_size,
    int intermediate_size,
    int heads)
{
    if (frames <= 0 || frames > (INT_MAX / 2) + 1 || hidden_size <= 0 ||
        intermediate_size <= 0 || heads <= 0 || hidden_size % heads != 0) {
        return 0;
    }
    size_t token_elements = 0;
    size_t ff_elements = 0;
    size_t relative_elements = 0;
    size_t doubled_token_elements = 0;
    size_t score_elements = 0;
    if (checked_mul_size((size_t)frames, (size_t)hidden_size, &token_elements) != 0 ||
        checked_mul_size((size_t)frames, (size_t)intermediate_size, &ff_elements) != 0 ||
        checked_mul_size((size_t)(2 * frames - 1), (size_t)hidden_size,
                         &relative_elements) != 0 ||
        checked_mul_size(token_elements, 2u, &doubled_token_elements) != 0 ||
        checked_mul_size((size_t)heads, (size_t)frames, &score_elements) != 0) {
        return 0;
    }
    size_t large_elements = ff_elements;
    if (relative_elements > large_elements) large_elements = relative_elements;
    if (doubled_token_elements > large_elements) {
        large_elements = doubled_token_elements;
    }
    size_t total_elements = 0;
    size_t token_buffers = 0;
    if (checked_mul_size(token_elements, 5u, &token_buffers) != 0 ||
        checked_add_size(token_buffers, large_elements, &total_elements) != 0 ||
        checked_add_size(total_elements, score_elements, &total_elements) != 0 ||
        checked_mul_size(total_elements, sizeof(float), &total_elements) != 0) {
        return 0;
    }
    return total_elements;
}

int audio_fastconformer_block_f32(
    const float *input,
    const float *relative_positions,
    const float *ff1_norm_weight,
    const float *ff1_norm_bias,
    const float *ff1_up_weight,
    const float *ff1_up_bias,
    const float *ff1_down_weight,
    const float *ff1_down_bias,
    const float *attn_norm_weight,
    const float *attn_norm_bias,
    const float *q_weight,
    const float *q_bias,
    const float *k_weight,
    const float *k_bias,
    const float *v_weight,
    const float *v_bias,
    const float *relative_weight,
    const float *attn_bias_u,
    const float *attn_bias_v,
    const float *attn_out_weight,
    const float *attn_out_bias,
    const float *conv_norm_weight,
    const float *conv_norm_bias,
    const float *conv_pw1_weight,
    const float *conv_pw1_bias,
    const float *conv_dw_weight,
    const float *conv_dw_bias,
    const float *conv_bn_mean,
    const float *conv_bn_variance,
    const float *conv_bn_weight,
    const float *conv_bn_bias,
    const float *conv_pw2_weight,
    const float *conv_pw2_bias,
    const float *ff2_norm_weight,
    const float *ff2_norm_bias,
    const float *ff2_up_weight,
    const float *ff2_up_bias,
    const float *ff2_down_weight,
    const float *ff2_down_bias,
    const float *out_norm_weight,
    const float *out_norm_bias,
    float *output,
    void *workspace,
    size_t workspace_bytes,
    int frames,
    int hidden_size,
    int intermediate_size,
    int heads,
    int head_dim,
    int conv_kernel_size,
    float layer_norm_epsilon,
    float batch_norm_epsilon)
{
    if (input == NULL || relative_positions == NULL ||
        ff1_norm_weight == NULL || ff1_norm_bias == NULL ||
        ff1_up_weight == NULL || ff1_down_weight == NULL ||
        attn_norm_weight == NULL || attn_norm_bias == NULL ||
        q_weight == NULL || k_weight == NULL || v_weight == NULL ||
        relative_weight == NULL || attn_bias_u == NULL || attn_bias_v == NULL ||
        attn_out_weight == NULL || conv_norm_weight == NULL ||
        conv_norm_bias == NULL || conv_pw1_weight == NULL ||
        conv_dw_weight == NULL || conv_bn_mean == NULL ||
        conv_bn_variance == NULL || conv_bn_weight == NULL ||
        conv_bn_bias == NULL || conv_pw2_weight == NULL ||
        ff2_norm_weight == NULL || ff2_norm_bias == NULL ||
        ff2_up_weight == NULL || ff2_down_weight == NULL ||
        out_norm_weight == NULL || out_norm_bias == NULL || output == NULL ||
        workspace == NULL) {
        return -1;
    }
    if (frames <= 0 || hidden_size <= 0 || intermediate_size <= 0 ||
        heads <= 0 || head_dim <= 0 || heads > INT_MAX / head_dim ||
        heads * head_dim != hidden_size || conv_kernel_size <= 0 ||
        (conv_kernel_size & 1) == 0 || !isfinite(layer_norm_epsilon) ||
        layer_norm_epsilon <= 0.0f || !isfinite(batch_norm_epsilon) ||
        batch_norm_epsilon <= 0.0f) {
        return -2;
    }
    const size_t required_workspace = audio_fastconformer_block_workspace_bytes(
        frames, hidden_size, intermediate_size, heads);
    if (required_workspace == 0 || workspace_bytes < required_workspace) {
        return -3;
    }

    const size_t token_elements = (size_t)frames * (size_t)hidden_size;
    const size_t ff_elements = (size_t)frames * (size_t)intermediate_size;
    const size_t relative_elements =
        (size_t)(2 * frames - 1) * (size_t)hidden_size;
    const size_t doubled_token_elements = 2u * token_elements;
    size_t large_elements = ff_elements;
    if (relative_elements > large_elements) large_elements = relative_elements;
    if (doubled_token_elements > large_elements) {
        large_elements = doubled_token_elements;
    }

    float *cursor = (float *)workspace;
    float *norm = cursor;
    cursor += token_elements;
    float *large = cursor;
    cursor += large_elements;
    float *buffer2 = cursor;
    cursor += token_elements;
    float *buffer3 = cursor;
    cursor += token_elements;
    float *buffer4 = cursor;
    cursor += token_elements;
    float *buffer5 = cursor;
    cursor += token_elements;
    float *scores = cursor;

    memcpy(output, input, token_elements * sizeof(float));

    layernorm_naive_serial_matched_precision(
        output, ff1_norm_weight, ff1_norm_bias, norm, NULL, NULL,
        frames, hidden_size, layer_norm_epsilon);
    gemm_nt_f32_llama_production_parallel_dispatch(
        norm, ff1_up_weight, ff1_up_bias, large,
        frames, intermediate_size, hidden_size);
    recurrent_silu_forward(large, large, frames, intermediate_size);
    gemm_nt_f32_llama_production_parallel_dispatch(
        large, ff1_down_weight, ff1_down_bias, buffer2,
        frames, hidden_size, intermediate_size);
    if (audio_scaled_residual_add_f32(
            output, buffer2, 0.5f, output, token_elements) != 0) {
        return -10;
    }

    layernorm_naive_serial_matched_precision(
        output, attn_norm_weight, attn_norm_bias, norm, NULL, NULL,
        frames, hidden_size, layer_norm_epsilon);
    gemm_nt_f32_llama_production_parallel_dispatch(
        norm, q_weight, q_bias, buffer2, frames, hidden_size, hidden_size);
    gemm_nt_f32_llama_production_parallel_dispatch(
        norm, k_weight, k_bias, buffer3, frames, hidden_size, hidden_size);
    gemm_nt_f32_llama_production_parallel_dispatch(
        norm, v_weight, v_bias, buffer4, frames, hidden_size, hidden_size);
    gemm_nt_f32_llama_production_parallel_dispatch(
        relative_positions, relative_weight, NULL, large,
        2 * frames - 1, hidden_size, hidden_size);
    if (audio_conformer_relative_attention_f32(
            buffer2, buffer3, buffer4, large, attn_bias_u, attn_bias_v,
            buffer5, frames, heads, head_dim, 1.0f / sqrtf((float)head_dim),
            scores, (size_t)heads * (size_t)frames * sizeof(float)) != 0) {
        return -11;
    }
    gemm_nt_f32_llama_production_parallel_dispatch(
        buffer5, attn_out_weight, attn_out_bias, norm,
        frames, hidden_size, hidden_size);
    if (audio_scaled_residual_add_f32(
            output, norm, 1.0f, output, token_elements) != 0) {
        return -12;
    }

    layernorm_naive_serial_matched_precision(
        output, conv_norm_weight, conv_norm_bias, norm, NULL, NULL,
        frames, hidden_size, layer_norm_epsilon);
    for (int frame = 0; frame < frames; ++frame) {
        for (int channel = 0; channel < hidden_size; ++channel) {
            buffer2[(size_t)channel * (size_t)frames + (size_t)frame] =
                norm[(size_t)frame * (size_t)hidden_size + (size_t)channel];
        }
    }
    if (audio_conv1d_channel_major_grouped_f32(
            buffer2, conv_pw1_weight, conv_pw1_bias, large,
            hidden_size, 2 * hidden_size, frames, 1, 1, 0, 1, frames) != 0 ||
        audio_glu_split_channel_major_f32(
            large, buffer3, hidden_size, frames) != 0 ||
        audio_conv1d_channel_major_grouped_f32(
            buffer3, conv_dw_weight, conv_dw_bias, buffer4,
            hidden_size, hidden_size, frames, conv_kernel_size, 1,
            conv_kernel_size / 2, hidden_size, frames) != 0 ||
        audio_batch_norm_inference_channel_major_f32(
            buffer4, conv_bn_mean, conv_bn_variance, conv_bn_weight,
            conv_bn_bias, buffer5, hidden_size, frames,
            batch_norm_epsilon) != 0) {
        return -13;
    }
    recurrent_silu_forward(buffer5, buffer5, hidden_size, frames);
    if (audio_conv1d_channel_major_grouped_f32(
            buffer5, conv_pw2_weight, conv_pw2_bias, buffer3,
            hidden_size, hidden_size, frames, 1, 1, 0, 1, frames) != 0) {
        return -14;
    }
    for (int frame = 0; frame < frames; ++frame) {
        for (int channel = 0; channel < hidden_size; ++channel) {
            norm[(size_t)frame * (size_t)hidden_size + (size_t)channel] =
                buffer3[(size_t)channel * (size_t)frames + (size_t)frame];
        }
    }
    if (audio_scaled_residual_add_f32(
            output, norm, 1.0f, output, token_elements) != 0) {
        return -15;
    }

    layernorm_naive_serial_matched_precision(
        output, ff2_norm_weight, ff2_norm_bias, norm, NULL, NULL,
        frames, hidden_size, layer_norm_epsilon);
    gemm_nt_f32_llama_production_parallel_dispatch(
        norm, ff2_up_weight, ff2_up_bias, large,
        frames, intermediate_size, hidden_size);
    recurrent_silu_forward(large, large, frames, intermediate_size);
    gemm_nt_f32_llama_production_parallel_dispatch(
        large, ff2_down_weight, ff2_down_bias, buffer2,
        frames, hidden_size, intermediate_size);
    if (audio_scaled_residual_add_f32(
            output, buffer2, 0.5f, output, token_elements) != 0) {
        return -16;
    }
    layernorm_naive_serial_matched_precision(
        output, out_norm_weight, out_norm_bias, norm, NULL, NULL,
        frames, hidden_size, layer_norm_epsilon);
    memcpy(output, norm, token_elements * sizeof(float));
    return 0;
}

int audio_argmax_first_f32(const float *values, int elements, int *selected)
{
    if (values == NULL || selected == NULL) {
        return -1;
    }
    if (elements <= 0) {
        return -2;
    }
    if (!isfinite(values[0])) {
        return -3;
    }
    int best = 0;
    float maximum = values[0];
    for (int index = 1; index < elements; ++index) {
        if (!isfinite(values[index])) {
            return -3;
        }
        if (values[index] > maximum) {
            maximum = values[index];
            best = index;
        }
    }
    *selected = best;
    return 0;
}

int audio_whisper_stft_power_reference_f32(
    const float *samples,
    int n_samples,
    float *power,
    int n_frames)
{
    if (samples == NULL || power == NULL) {
        return -1;
    }
    if (n_samples <= CK_AUDIO_WHISPER_N_FFT / 2 || n_frames <= 0) {
        return -2;
    }
    if (n_frames != n_samples / CK_AUDIO_WHISPER_HOP_LENGTH) {
        return -3;
    }

    const int center = CK_AUDIO_WHISPER_N_FFT / 2;
    for (int frame = 0; frame < n_frames; ++frame) {
        for (int bin = 0; bin < CK_AUDIO_WHISPER_POWER_BINS; ++bin) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int sample = 0; sample < CK_AUDIO_WHISPER_N_FFT; ++sample) {
                const int source = reflect_index(
                    frame * CK_AUDIO_WHISPER_HOP_LENGTH + sample - center,
                    n_samples);
                const float window = 0.5f - 0.5f * cosf(
                    2.0f * CK_AUDIO_PI_F * (float)sample /
                    (float)CK_AUDIO_WHISPER_N_FFT);
                const float value = samples[source] * window;
                const float angle = -2.0f * CK_AUDIO_PI_F *
                    (float)(bin * sample) / (float)CK_AUDIO_WHISPER_N_FFT;
                real = fmaf(value, cosf(angle), real);
                imag = fmaf(value, sinf(angle), imag);
            }
            power[(size_t)frame * CK_AUDIO_WHISPER_POWER_BINS + bin] =
                fmaf(real, real, imag * imag);
        }
    }
    return 0;
}

int audio_whisper_log_mel_from_power_reference_f32(
    const float *power,
    const float *mel_filters,
    int n_mels,
    int n_frames,
    float *log_mel)
{
    if (power == NULL || mel_filters == NULL || log_mel == NULL) {
        return -1;
    }
    if (n_mels <= 0 || n_frames <= 0) {
        return -2;
    }

    float maximum = -INFINITY;
    for (int mel = 0; mel < n_mels; ++mel) {
        const float *filter = mel_filters + (size_t)mel * CK_AUDIO_WHISPER_POWER_BINS;
        float *output = log_mel + (size_t)mel * n_frames;
        for (int frame = 0; frame < n_frames; ++frame) {
            const float *spectrum = power + (size_t)frame * CK_AUDIO_WHISPER_POWER_BINS;
            float sum = 0.0f;
            for (int bin = 0; bin < CK_AUDIO_WHISPER_POWER_BINS; ++bin) {
                sum = fmaf(filter[bin], spectrum[bin], sum);
            }
            const float value = log10f(fmaxf(sum, 1.0e-10f));
            output[frame] = value;
            maximum = fmaxf(maximum, value);
        }
    }

    const float floor = maximum - 8.0f;
    for (int mel = 0; mel < n_mels; ++mel) {
        float *output = log_mel + (size_t)mel * n_frames;
        for (int frame = 0; frame < n_frames; ++frame) {
            output[frame] = (fmaxf(output[frame], floor) + 4.0f) / 4.0f;
        }
    }
    return 0;
}

int audio_whisper_log_mel_reference_f32(
    const float *samples,
    int n_samples,
    const float *mel_filters,
    int n_mels,
    float *power_scratch,
    float *log_mel,
    int n_frames)
{
    const int stft_status = audio_whisper_stft_power_reference_f32(
        samples, n_samples, power_scratch, n_frames);
    if (stft_status != 0) {
        return stft_status;
    }
    return audio_whisper_log_mel_from_power_reference_f32(
        power_scratch, mel_filters, n_mels, n_frames, log_mel);
}
