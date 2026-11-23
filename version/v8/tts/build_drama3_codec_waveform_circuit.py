"""Connect the Drama 3 codec through the causal DAC waveform decoder."""
from math import prod

from version.v8.tts.build_drama3_codec_upsampler_circuit import build_circuit as upsampler_circuit


def build_circuit(frames: int, *, heads: int = 16, head_dim: int = 64,
                  decoder_width: int = 1536, rates=(8, 8, 4, 2), **transformer):
    """Declare fixed-size PCM using checked FP32 convolution and activation providers.

    The pinned weight preparation verifies that adding the publisher's tiny
    Snake denominator epsilon leaves every FP32 alpha unchanged.
    """
    if (type(decoder_width) is not int or decoder_width < 1 or len(rates) != 4 or
            decoder_width % (2 ** len(rates)) or any(type(n) is not int or n < 1 for n in rates)):
        raise ValueError("expected four positive DAC rates and channels divisible by 16")
    graph, config = upsampler_circuit(frames, heads=heads, head_dim=head_dim, **transformer)
    length = frames * 4
    samples = length * prod(rates)
    ops = graph["block_types"]["rvq"]["body"]["ops"]
    graph["name"] = "drama3_codec_waveform"
    graph["native_entry"]["function"] = "ck_drama3_codec_waveform"
    graph["contract"]["runtime_invariants"].update(
        pcm_samples=samples, pcm_sample_rate=44100, pcm_channels=1)

    def append(name, channels, count, op, kernel, inputs, dims, scalars, *, weights=None):
        graph["activation_buffers"][name] = {"shape": [channels, count]}
        graph["activation_bindings"][name] = name
        entry = {"id": name, "op": op, "kernel": kernel, "returns_status": True,
                 "graph_slots": {"inputs": inputs, "outputs": {"output": name}},
                 "params": {**dims, "call_constants": scalars}}
        if weights is not None:
            entry["weight_refs"] = weights
        ops.append(entry)

    def conv(name, source, inputs, outputs, count, taps, dilation, stem):
        padded = count + (taps - 1) * dilation
        if taps > 1:
            pad = f"{name}_left_pad"
            append(pad, inputs, padded, "audio_zero_pad1d_left_checked",
                   "audio_zero_pad1d_left_channel_major_f32_checked", {"input": source},
                   {"C": inputs, "T": count, "U": padded}, {
                       "zero_pad_input_elements": inputs * count, "zero_pad_input_stride": count,
                       "zero_pad_output_elements": inputs * padded, "zero_pad_output_stride": padded,
                       "zero_pad_channels": inputs, "zero_pad_input_frames": count,
                       "zero_pad_left_padding": padded - count, "zero_pad_output_frames": padded})
            source = pad
        size = count * outputs
        append(name, outputs, count, "audio_conv1d_dilated_checked",
               "audio_conv1d_dilated_checked_channel_major_f32", {"input": source},
               {"I": inputs, "O": outputs, "T": padded, "K": taps, "U": count, "Q": size}, {
                   "conv_input_elements": inputs * padded, "conv_input_stride": padded,
                   "conv_weight_elements": inputs * outputs * taps, "conv_bias_elements": outputs,
                   "conv_output_elements": size, "conv_output_stride": count,
                   "conv_scratch_elements": size, "conv_input_channels": inputs,
                   "conv_output_channels": outputs, "conv_input_frames": padded,
                   "conv_kernel_size": taps, "conv_stride": 1, "conv_padding": 0,
                   "conv_dilation": dilation, "conv_output_frames": count},
               weights={"weight": f"{stem}.weight", "bias": f"{stem}.bias"})

    def snake(name, source, channels, count, stem):
        size = channels * count
        append(name, channels, count, "audio_snake_strided_checked", "audio_snake_strided_f32_checked",
               {"input": source}, {"C": channels, "T": count}, {
                   "snake_input_elements": size, "snake_input_stride": count,
                   "snake_alpha_elements": channels, "snake_output_elements": size,
                   "snake_output_stride": count, "snake_channels": channels, "snake_frames": count},
               weights={"alpha": f"{stem}.alpha"})

    def upsample(name, source, channels, outputs, count, scale, stem):
        out_frames = count * scale
        size = outputs * out_frames
        taps = 2 * scale
        append(name, outputs, out_frames, "audio_conv_transpose1d_dense_crop_right_checked",
               "audio_conv_transpose1d_dense_crop_right_channel_major_f32_checked", {"input": source},
               {"I": channels, "O": outputs, "T": count, "K": taps, "U": out_frames, "Q": size}, {
                   "cropped_deconv_input_elements": channels * count, "cropped_deconv_input_stride": count,
                   "cropped_deconv_weight_elements": channels * outputs * taps,
                   "cropped_deconv_bias_elements": outputs, "cropped_deconv_output_elements": size,
                   "cropped_deconv_output_stride": out_frames, "cropped_deconv_scratch_elements": size,
                   "cropped_deconv_input_channels": channels, "cropped_deconv_output_channels": outputs,
                   "cropped_deconv_input_frames": count, "cropped_deconv_kernel_size": taps,
                   "cropped_deconv_stride": scale, "cropped_deconv_padding": 0,
                   "cropped_deconv_output_padding": 0, "cropped_deconv_right_crop": scale,
                   "cropped_deconv_output_frames": out_frames},
               weights={"weight": f"{stem}.weight", "bias": f"{stem}.bias"})

    source = "decoder_input_conv"
    channels = decoder_width
    conv(source, "dac_input", heads * head_dim, channels, length, 7, 1, "decoder.model.0.conv")
    for layer, rate in enumerate(rates, 1):
        prefix = f"decoder.model.{layer}.block"
        snake(f"dec{layer}_pre_act", source, channels, length, f"{prefix}.0")
        source = f"dec{layer}_upsampled"
        upsample(source, f"dec{layer}_pre_act", channels, channels // 2, length, rate, f"{prefix}.1.conv")
        channels //= 2
        length *= rate
        for index, dilation in enumerate((1, 3, 9), 2):
            name, stem = f"dec{layer}_res{index}", f"{prefix}.{index}.block"
            snake(f"{name}_act1", source, channels, length, f"{stem}.0")
            conv(f"{name}_conv1", f"{name}_act1", channels, channels, length, 7, dilation, f"{stem}.1.conv")
            snake(f"{name}_act2", f"{name}_conv1", channels, length, f"{stem}.2")
            conv(f"{name}_delta", f"{name}_act2", channels, channels, length, 1, 1, f"{stem}.3.conv")
            count = channels * length
            append(name, channels, length, "add_rows_checked", "add_rows_checked_f32",
                   {"left": source, "right": f"{name}_delta"}, {"N": count}, {
                       "left_elements": count, "right_elements": count, "output_elements": count,
                       "elements": count})
            source = name

    snake("decoder_final_act", source, channels, length, "decoder.model.5")
    conv("decoder_pre_tanh", "decoder_final_act", channels, 1, length, 7, 1, "decoder.model.6.conv")
    append("pcm", 1, length, "tanh_strided_checked", "tanh_strided_f32_checked",
           {"input": "decoder_pre_tanh"}, {"R": 1, "C": length}, {
               "tanh_input_elements": length, "tanh_input_stride": length,
               "tanh_output_elements": length, "tanh_output_stride": length,
               "tanh_rows": 1, "tanh_columns": length})
    return graph, config
