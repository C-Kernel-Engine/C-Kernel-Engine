"""Connect Drama 3's two latent upsampling stages after its codec transformer."""
from version.v8.scripts.checked_circuit_authoring_v8 import linear
from version.v8.tts.build_drama3_codec_transformer_circuit import build_circuit as transformer_circuit


def build_circuit(frames: int, *, heads: int = 16, head_dim: int = 64, **transformer):
    """Declare fixed extents from speech codes to the channel-major DAC input."""
    graph, config = transformer_circuit(frames, heads=heads, head_dim=head_dim, **transformer)
    graph["name"] = "drama3_codec_upsampler"
    graph["native_entry"]["function"] = "ck_drama3_codec_upsampler"
    graph["contract"]["runtime_invariants"]["dac_input_frames"] = frames * 4
    channels = heads * head_dim
    wide = 4 * channels
    ops = graph["block_types"]["rvq"]["body"]["ops"]

    def buffer(name, shape):
        graph["activation_buffers"][name] = {"shape": shape}
        graph["activation_bindings"][name] = name

    def add_op(name, shape, op, kernel, inputs, dims, scalars, *, weights=None):
        buffer(name, shape)
        entry = {"id": name, "op": op, "kernel": kernel, "returns_status": True,
                 "graph_slots": {"inputs": inputs, "outputs": {"output": name}},
                 "params": {**dims, "call_constants": scalars}}
        if weights is not None:
            entry["weight_refs"] = weights
        ops.append(entry)

    def unary(name, shape, op, kernel, source, dims, scalars, *, weights=None):
        add_op(name, shape, op, kernel, {"input": source}, dims, scalars, weights=weights)

    def transpose(name, source, rows, cols):
        count = rows * cols
        unary(name, [cols, rows], "transpose_strided_checked", "transpose_strided_f32_checked",
              source, {"R": rows, "C": cols}, {
                  "transpose_input_elements": count, "transpose_input_stride": cols,
                  "transpose_output_elements": count, "transpose_output_stride": rows,
                  "transpose_rows": rows, "transpose_columns": cols})

    source = "upsampler_input"
    transpose(source, "codec_transformer_out", frames, channels)
    length = frames
    for i in range(2):
        stem = f"upsample{i}"
        weight = f"upsample.{i}"
        output_length = length * 2
        count = channels * output_length
        deconv = f"{stem}_deconv"
        unary(deconv, [channels, output_length], "audio_conv_transpose1d_dense_checked",
              "audio_conv_transpose1d_dense_channel_major_f32_checked", source,
              {"I": channels, "O": channels, "T": length, "K": 2, "U": output_length, "Q": count}, {
                  "dense_deconv_input_elements": channels * length, "dense_deconv_input_stride": length,
                  "dense_deconv_weight_elements": channels * channels * 2,
                  "dense_deconv_bias_elements": channels, "dense_deconv_output_elements": count,
                  "dense_deconv_output_stride": output_length, "dense_deconv_scratch_elements": count,
                  "dense_deconv_input_channels": channels, "dense_deconv_output_channels": channels,
                  "dense_deconv_input_frames": length, "dense_deconv_kernel_size": 2,
                  "dense_deconv_stride": 2, "dense_deconv_padding": 0,
                  "dense_deconv_output_padding": 0, "dense_deconv_output_frames": output_length},
              weights={"weight": f"{weight}.0.conv.weight", "bias": f"{weight}.0.conv.bias"})
        taps = 7
        padded = output_length + taps - 1
        pad = f"{stem}_left_pad"
        unary(pad, [channels, padded], "audio_zero_pad1d_left_checked",
              "audio_zero_pad1d_left_channel_major_f32_checked", deconv,
              {"C": channels, "T": output_length, "U": padded}, {
                  "zero_pad_input_elements": count, "zero_pad_input_stride": output_length,
                  "zero_pad_output_elements": channels * padded, "zero_pad_output_stride": padded,
                  "zero_pad_channels": channels, "zero_pad_input_frames": output_length,
                  "zero_pad_left_padding": taps - 1, "zero_pad_output_frames": padded})
        dwconv = f"{stem}_depthwise"
        unary(dwconv, [channels, output_length], "audio_conv1d_depthwise_checked",
              "audio_conv1d_depthwise_channel_major_f32_checked", pad,
              {"C": channels, "T": padded, "K": taps, "U": output_length, "Q": count}, {
                  "conv_input_elements": channels * padded, "conv_input_stride": padded,
                  "conv_weight_elements": channels * taps, "conv_bias_elements": channels,
                  "conv_output_elements": count, "conv_output_stride": output_length,
                  "conv_scratch_elements": count, "conv_channels": channels,
                  "conv_input_frames": padded, "conv_kernel_size": taps,
                  "conv_stride": 1, "conv_output_frames": output_length},
              weights={"weight": f"{weight}.1.dwconv.conv.weight", "bias": f"{weight}.1.dwconv.conv.bias"})
        transpose(f"{stem}_depthwise_token", dwconv, channels, output_length)
        norm = f"{stem}_norm"
        norm_scratch = output_length * (channels + 2)
        unary(norm, [output_length, channels], "layernorm_rows_param_checked",
              "layernorm_rows_checked_param_epsilon_f32", f"{stem}_depthwise_token",
              {"M": output_length, "C": channels, "Q": norm_scratch, "epsilon": 1e-6}, {
                  "norm_input_elements": count, "norm_input_stride": channels,
                  "norm_gamma_elements": channels, "norm_beta_elements": channels,
                  "norm_output_elements": count, "norm_output_stride": channels,
                  "norm_scratch_elements": norm_scratch, "norm_rows": output_length,
                  "norm_channels": channels},
              weights={"gamma": f"{weight}.1.norm.weight", "beta": f"{weight}.1.norm.bias"})
        first, act, second = f"{stem}_pw1", f"{stem}_gelu", f"{stem}_pw2"
        buffer(first, [output_length, wide])
        ops.append(linear(first, norm, first, channels, wide, output_length, f"{weight}.1.pwconv1"))
        wide_count = output_length * wide
        unary(act, [output_length, wide], "gelu", "gelu_rows_erf_checked_f32", first,
              {"M": output_length, "C": wide, "Q": wide_count}, {
                  "gelu_input_elements": wide_count, "gelu_input_stride": wide,
                  "gelu_output_elements": wide_count, "gelu_output_stride": wide,
                  "gelu_scratch_elements": wide_count, "gelu_rows": output_length, "gelu_channels": wide})
        buffer(second, [output_length, channels])
        ops.append(linear(second, act, second, wide, channels, output_length, f"{weight}.1.pwconv2"))
        scaled = f"{stem}_scaled"
        unary(scaled, [output_length, channels], "channel_scale_rows_checked", "channel_scale_rows_checked_f32",
              second, {"R": output_length, "C": channels}, {
                  "input_elements": count, "input_stride": channels, "scale_elements": channels,
                  "output_elements": count, "output_stride": channels,
                  "rows": output_length, "channels": channels}, weights={"scale": f"{weight}.1.gamma"})
        delta = f"{stem}_delta"
        transpose(delta, scaled, output_length, channels)
        source = "dac_input" if i == 1 else f"{stem}_out"
        add_op(source, [channels, output_length], "add_rows_checked", "add_rows_checked_f32",
               {"left": deconv, "right": delta}, {"N": count},
               {"left_elements": count, "right_elements": count, "output_elements": count, "elements": count})
        length = output_length
    return graph, config
