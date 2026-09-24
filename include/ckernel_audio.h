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

int audio_hann_window_f32(
    float *output,
    int frames,
    int periodic);

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

int audio_relative_sinusoidal_position_f32(
    float *output,
    int frames,
    int channels);

int audio_batch_norm_inference_channel_major_f32(
    const float *input,
    const float *running_mean,
    const float *running_variance,
    const float *weight,
    const float *bias,
    float *output,
    int channels,
    int frames,
    float epsilon);

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
    int hidden_size);

/* Packed direction-major weights: [2, 4*H, I] and [2, 4*H, H].
 * H must be at most INT_MAX/4 because the IFGO step uses signed gate indices.
 * Output rows are [forward H, reverse H]. State is reset at each call.
 * Returns -1 for null pointers, -2 for invalid/overflowing geometry,
 * and -3 for insufficient input, weight, output, state or scratch capacity.
 */
int audio_lstm_bidirectional_scan_f32(
    const float *input, size_t input_elements,
    const float *weight_ih, size_t weight_ih_elements,
    const float *weight_hh, size_t weight_hh_elements,
    const float *bias_ih, size_t bias_ih_elements,
    const float *bias_hh, size_t bias_hh_elements,
    float *output, size_t output_elements,
    float *hidden_state, size_t hidden_state_elements,
    float *cell_state, size_t cell_state_elements,
    float *gates_scratch, size_t gates_scratch_bytes,
    int tokens, int input_size, int hidden_size,
    size_t input_stride, size_t output_stride);

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

size_t audio_fastconformer_subsampling_workspace_bytes(
    int feature_frames,
    int feature_channels,
    int conv_channels,
    int kernel_size,
    int stride);

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
    int *output_frames);

/*
 * Returns the recommended workspace for the configured CK thread pool. The
 * block also accepts the compatible minimum allocation with one score row per
 * attention head and caps attention workers to the supplied score capacity.
 */
size_t audio_fastconformer_block_workspace_bytes(
    int frames,
    int hidden_size,
    int intermediate_size,
    int heads);

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
    float batch_norm_epsilon);

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
    size_t scores_scratch_bytes);

int audio_transpose_channel_to_token_f32(
    const float *input,
    float *output,
    int channels,
    int frames);

int audio_scaled_residual_add_f32(
    const float *residual,
    const float *branch,
    float scale,
    float *output,
    size_t elements);

int audio_argmax_first_f32(
    const float *values,
    int elements,
    int *selected);

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
