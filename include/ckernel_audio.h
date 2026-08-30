#ifndef CKERNEL_AUDIO_H
#define CKERNEL_AUDIO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CK_AUDIO_WHISPER_SAMPLE_RATE 16000
#define CK_AUDIO_WHISPER_N_FFT 400
#define CK_AUDIO_WHISPER_HOP_LENGTH 160
#define CK_AUDIO_WHISPER_POWER_BINS 201

typedef struct CKAudioWavInfo {
    int format_tag;
    int channels;
    int sample_rate;
    int bits_per_sample;
    int frames;
    size_t data_offset;
    size_t data_bytes;
} CKAudioWavInfo;

int audio_wav_parse_memory(
    const uint8_t *bytes,
    size_t byte_count,
    CKAudioWavInfo *info);

int audio_wav_decode_pcm16_mono_f32(
    const uint8_t *bytes,
    size_t byte_count,
    const CKAudioWavInfo *info,
    float *mono,
    int mono_capacity);

int audio_wav_decode_memory_pcm16_mono_f32(
    const uint8_t *bytes,
    size_t byte_count,
    float *mono,
    int mono_capacity,
    CKAudioWavInfo *info);

int audio_wav_decode_memory_pcm16_mono_window_f32(
    const uint8_t *bytes,
    size_t byte_count,
    int start_frame,
    float *mono,
    int mono_capacity,
    CKAudioWavInfo *info);

int audio_pcm_s16_to_mono_f32(
    const int16_t *interleaved,
    int n_frames,
    int n_channels,
    float *mono);

int audio_resampled_frame_count(
    int input_frames,
    int input_rate,
    int output_rate);

int audio_resample_linear_f32(
    const float *input,
    int input_frames,
    int input_rate,
    float *output,
    int output_frames,
    int output_rate);

int audio_resample_windowed_sinc_f32(
    const float *input,
    int input_frames,
    int input_rate,
    float *output,
    int output_frames,
    int output_rate,
    int radius);

int audio_pad_or_truncate_f32(
    const float *input,
    int input_frames,
    float *output,
    int output_frames);

int audio_preemphasis_f32(
    const float *input,
    float *output,
    int frames,
    float coefficient);

int audio_feature_normalize_per_feature_f32(
    const float *input,
    float *output,
    int channels,
    int frames,
    float epsilon);

int audio_stft_precompute_tables_f32(
    int n_fft,
    float *window,
    float *cos_table,
    float *sin_table);

int audio_whisper_mel_filters_slaney_f32(
    int sample_rate,
    int n_fft,
    int n_mels,
    float *mel_filters);

int audio_stft_power_precomputed_f32(
    const float *samples,
    int n_samples,
    const float *window,
    const float *cos_table,
    const float *sin_table,
    int n_fft,
    int hop_length,
    float *power,
    int n_frames);

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
    int n_frames);

int audio_log_mel_time_major_f32(
    const float *power,
    const float *mel_filters,
    float *log_mel,
    int frames,
    int bins,
    int channels,
    float epsilon);

int audio_stft_power_fft400_f32(
    const float *samples,
    int n_samples,
    const float *window,
    const float *cos_table,
    const float *sin_table,
    int hop_length,
    float *power,
    int n_frames,
    float *fft_scratch);

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
    int output_frames);

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
    int output_height);

int audio_glu_split_channel_major_f32(
    const float *input,
    float *output,
    int channels,
    int frames);

int audio_relative_shift_f32(
    const float *raw_scores,
    float *scores,
    int heads,
    int query_frames);

int audio_transpose_channel_to_token_f32(
    const float *input,
    float *output,
    int channels,
    int frames);

int audio_whisper_stft_power_reference_f32(
    const float *samples,
    int n_samples,
    float *power,
    int n_frames);

int audio_whisper_log_mel_from_power_reference_f32(
    const float *power,
    const float *mel_filters,
    int n_mels,
    int n_frames,
    float *log_mel);

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
    float *log_mel);

int audio_whisper_log_mel_reference_f32(
    const float *samples,
    int n_samples,
    const float *mel_filters,
    int n_mels,
    float *power_scratch,
    float *log_mel,
    int n_frames);

#ifdef __cplusplus
}
#endif

#endif
