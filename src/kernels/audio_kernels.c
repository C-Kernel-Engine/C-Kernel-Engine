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
        const float *weight_channel = args->weight +
            (size_t)out_channel * args->input_channels * args->kernel_size;
        float *output_channel = args->output +
            (size_t)out_channel * args->output_frames;
        int out_frame = 0;
        const int interior_begin =
            (args->padding + args->stride - 1) / args->stride;
        for (; out_frame < interior_begin && out_frame < args->output_frames;
             ++out_frame) {
            float sum = args->bias != NULL ? args->bias[out_channel] : 0.0f;
            for (int in_channel = 0; in_channel < args->input_channels;
                 ++in_channel) {
                const float *input_channel = args->input +
                    (size_t)in_channel * args->input_frames;
                const float *weight_row = weight_channel +
                    (size_t)in_channel * args->kernel_size;
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
            for (int in_channel = 0; in_channel < args->input_channels;
                 ++in_channel) {
                const float *input_channel = args->input +
                    (size_t)in_channel * args->input_frames;
                const float *weight_row = weight_channel +
                    (size_t)in_channel * args->kernel_size;
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
            for (int in_channel = 0; in_channel < args->input_channels;
                 ++in_channel) {
                const float *input_channel = args->input +
                    (size_t)in_channel * args->input_frames;
                const float *weight_row = weight_channel +
                    (size_t)in_channel * args->kernel_size;
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
    if (input == NULL || weight == NULL || output == NULL) {
        return -1;
    }
    if (input_channels <= 0 || output_channels <= 0 || input_frames <= 0 ||
        kernel_size <= 0 || stride <= 0 || padding < 0 || output_frames <= 0) {
        return -2;
    }
    const int expected = (input_frames + 2 * padding - kernel_size) / stride + 1;
    if (output_frames != expected) {
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
