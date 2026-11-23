"""Run a complete generated Drama 3 codes-to-PCM circuit."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.test_v8_drama3_codec_rvq import invoke
from tests.test_v8_drama3_codec_transformer import set_rotary, synthetic
from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.tts.build_drama3_codec_upsampler_circuit import build_circuit as upsampler_circuit
from version.v8.tts.build_drama3_codec_waveform_circuit import build_circuit

# Half of one signed PCM16 step at full scale.
PCM_HALF_STEP = 0.5 / 32767


def compile_waveform(folder, frames, weights, **geometry):
    for name, value in weights.items():
        if name.endswith(".alpha") and (not np.isfinite(value).all() or
                (value == 0).any() or not np.array_equal(value + np.float32(1e-9), value)):
            raise ValueError(f"unsupported Snake denominator for {name}")
    graph, config = build_circuit(frames, **geometry)
    declarations = {name: (list(value.shape), "F32") for name, value in weights.items()}
    return compile_checked_weight_graph(folder, graph, weights, declarations, source_config=config,
                                        openmp=True, provider_warnings="report", c_standard="gnu11")


def pcm16(samples):
    return np.rint(np.clip(samples, -1, 1) * 32767).astype(np.int32)


def compile_upsampler(folder, frames, weights, **geometry):
    graph, config = upsampler_circuit(frames, **geometry)
    declarations = {name: (list(value.shape), "F32") for name, value in weights.items()}
    return compile_checked_weight_graph(folder, graph, weights, declarations, source_config=config,
                                        openmp=True, provider_warnings="report", c_standard="gnu11")


@pytest.fixture
def upsampler_inputs(synthetic):
    _, codes, prefix_weights, cos, sin, (frames, heads, head_dim, layers, window) = synthetic
    weights = dict(prefix_weights)
    channels = heads * head_dim
    wide = 4 * channels
    rng = np.random.default_rng(654)
    for i in range(2):
        stem = f"upsample.{i}"
        for key, shape in (
            ("0.conv.weight", (channels, channels, 2)), ("0.conv.bias", (channels,)),
            ("1.dwconv.conv.weight", (channels, 7)), ("1.dwconv.conv.bias", (channels,)),
            ("1.norm.weight", (channels,)), ("1.norm.bias", (channels,)),
            ("1.pwconv1.weight", (wide, channels)), ("1.pwconv1.bias", (wide,)),
            ("1.pwconv2.weight", (channels, wide)), ("1.pwconv2.bias", (channels,)),
            ("1.gamma", (channels,)),
        ):
            weights[f"{stem}.{key}"] = rng.normal(0, 0.2, shape).astype(np.float32)
    geometry = dict(heads=heads, head_dim=head_dim, layers=layers,
                    intermediate=10, window=window, vocabularies=(5,) * 10, book_width=3)
    return codes, weights, cos, sin, geometry


def test_connected_upsampler_shape_causality_and_reference(tmp_path, upsampler_inputs):
    codes, weights, cos, sin, geometry = upsampler_inputs
    channels = geometry["heads"] * geometry["head_dim"]
    frames = codes.shape[1]
    build = tmp_path / "upsampler"
    build.mkdir()
    runtime = compile_upsampler(build, frames, weights, **geometry)
    set_rotary(runtime, cos, sin)
    view = runtime[2]
    assert invoke(runtime, codes) == 0
    native = view("dac_input").copy()
    assert native.shape == (channels, frames * 4)
    changed = codes.copy()
    changed[0, -1] = (changed[0, -1] + 1) % 5
    assert invoke(runtime, changed) == 0
    np.testing.assert_array_equal(view("dac_input")[:, :-4], native[:, :-4])
    assert not np.array_equal(view("dac_input")[:, -4:], native[:, -4:])
    latest = view("dac_input").copy()
    changed[-1, -1] = 5
    assert invoke(runtime, changed) != 0
    np.testing.assert_array_equal(view("dac_input"), latest)
    assert invoke(runtime, codes) == 0
    np.testing.assert_array_equal(view("dac_input"), native)

    torch = pytest.importorskip("torch")
    f = torch.nn.functional
    x = torch.from_numpy(view("codec_transformer_out").T.copy())[None]
    for i in range(2):
        stem = f"upsample.{i}"
        def param(name):
            return torch.from_numpy(weights[f"{stem}.{name}"])
        up = f.conv_transpose1d(x, param("0.conv.weight"), param("0.conv.bias"), stride=2)
        np.testing.assert_allclose(view(f"upsample{i}_deconv"), up[0].numpy(), rtol=1e-6, atol=1e-6)
        y = f.conv1d(f.pad(up, (6, 0)), param("1.dwconv.conv.weight")[:, None],
                     param("1.dwconv.conv.bias"), groups=channels).transpose(1, 2)
        y = f.layer_norm(y, (channels,), param("1.norm.weight"), param("1.norm.bias"), eps=1e-6)
        y = f.gelu(f.linear(y, param("1.pwconv1.weight"), param("1.pwconv1.bias")))
        y = f.linear(y, param("1.pwconv2.weight"), param("1.pwconv2.bias")) * param("1.gamma")
        x = up + y.transpose(1, 2)
    np.testing.assert_allclose(native, x[0].numpy(), rtol=1e-6, atol=1e-6)



def test_nonzero_waveform_causal_crop_and_recovery(tmp_path, upsampler_inputs):
    torch = pytest.importorskip("torch")
    codes, prefix, cos, sin, geometry = upsampler_inputs
    weights = dict(prefix)
    rng = np.random.default_rng(522)

    def tensor(name, shape, mean=0.0):
        weights[name] = rng.normal(mean, 0.1, shape).astype(np.float32)

    def conv(name, i, o, taps):
        tensor(f"{name}.weight", (o, i, taps))
        tensor(f"{name}.bias", (o,))

    channels = 16
    conv("decoder.model.0.conv", geometry["heads"] * geometry["head_dim"], channels, 7)
    for i in range(1, 5):
        stem = f"decoder.model.{i}.block"
        tensor(f"{stem}.0.alpha", (channels,), 1.0)
        tensor(f"{stem}.1.conv.weight", (channels, channels // 2, 4))
        channels //= 2
        tensor(f"{stem}.1.conv.bias", (channels,))
        for j in range(2, 5):
            branch = f"{stem}.{j}.block"
            tensor(f"{branch}.0.alpha", (channels,), 1.0)
            conv(f"{branch}.1.conv", channels, channels, 7)
            tensor(f"{branch}.2.alpha", (channels,), 1.0)
            conv(f"{branch}.3.conv", channels, channels, 1)
    tensor("decoder.model.5.alpha", (channels,), 1.0)
    conv("decoder.model.6.conv", channels, 1, 7)

    frames = codes.shape[1]
    build = tmp_path / "waveform"
    build.mkdir()
    runtime = compile_waveform(build, frames, weights, decoder_width=16,
                               rates=(2,) * 4, **geometry)
    set_rotary(runtime, cos, sin)
    view = runtime[2]
    assert invoke(runtime, codes) == 0
    actual = view("pcm").copy()
    assert actual.shape == (1, frames * 64)
    expected_input = view("dac_input").copy()
    changed = codes.copy()
    changed[0, -1] = (changed[0, -1] + 1) % 5
    assert invoke(runtime, changed) == 0
    np.testing.assert_array_equal(view("pcm")[:, :-64], actual[:, :-64])
    assert not np.array_equal(view("pcm")[:, -64:], actual[:, -64:])
    latest = view("pcm").copy()
    changed[-1, -1] = -1
    assert invoke(runtime, changed) != 0
    np.testing.assert_array_equal(view("pcm"), latest)
    assert invoke(runtime, codes) == 0
    np.testing.assert_array_equal(view("pcm"), actual)

    f = torch.nn.functional

    def params(stem):
        return tuple(torch.from_numpy(weights[f"{stem}.{name}"]) for name in ("weight", "bias"))

    def convolve(x, name, taps, dilation=1):
        return f.conv1d(f.pad(x, ((taps - 1) * dilation, 0)), *params(name), dilation=dilation)

    def activate(x, name):
        alpha = torch.from_numpy(weights[f"{name}.alpha"])[None, :, None]
        return x + (alpha + 1e-9).reciprocal() * (x * alpha).sin().square()

    x = convolve(torch.from_numpy(expected_input)[None], "decoder.model.0.conv", 7)
    for i in range(1, 5):
        stem = f"decoder.model.{i}.block"
        x = f.conv_transpose1d(activate(x, f"{stem}.0"), *params(f"{stem}.1.conv"), stride=2)[..., :-2]
        for j, dilation in enumerate((1, 3, 9), 2):
            branch = f"{stem}.{j}.block"
            y = convolve(activate(x, f"{branch}.0"), f"{branch}.1.conv", 7, dilation)
            x = x + convolve(activate(y, f"{branch}.2"), f"{branch}.3.conv", 1)
    x = convolve(activate(x, "decoder.model.5"), "decoder.model.6.conv", 7).tanh()
    np.testing.assert_allclose(actual, x[0].numpy(), rtol=2e-6, atol=2e-6)


def check_latent_roundoff(view, weights, record_property):
    # With stride == kernel size == 2, each sample is a dot product over
    # input channels with a single initial bias. Bound ordered FP32 FMA
    # rounding by gamma_n times the sum of absolute terms, using FP64 math.
    for i, source in enumerate(("upsampler_input", "upsample0_out")):
        x = view(source).astype(np.float64)
        w = weights[f"upsample.{i}.0.conv.weight"].astype(np.float64)
        bias = weights[f"upsample.{i}.0.conv.bias"].astype(np.float64)[:, None]
        output = view(f"upsample{i}_deconv")
        n = x.shape[0]
        nu = n * 2 ** -24
        gamma_n = nu / (1 - nu)
        for tap in range(2):
            weight = w[:, :, tap].T
            expected = weight @ x + bias
            envelope = gamma_n * (np.abs(weight) @ np.abs(x) + np.abs(bias))
            delta = np.abs(output[:, tap::2] - expected)
            assert np.all(delta <= envelope + 1e-12)
            record_property(f"upsample{i}_tap{tap}_fp64_same_input_max_abs", float(delta.max()))


@pytest.mark.parametrize("frames", [1, 3, 8])
def test_pinned_codes_to_pcm(tmp_path, frames, record_property):
    location = os.environ.get("CK_DRAMA3_CODEC_WAVEFORM_REFERENCE_DIR")
    if location is None:
        pytest.skip("CK_DRAMA3_CODEC_WAVEFORM_REFERENCE_DIR is not configured")
    folder = Path(location)
    report = json.loads((folder / "reference-result.json").read_text())
    assert report["model_revision"] == "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216"
    assert report["implementation"] == "dfb6b43b360b04e53daa7c36aed0bc448789d3a4"
    assert report["codec_sha256"] == "74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f"
    assert report["waveform"] == {"strides": [8, 8, 4, 2], "snake_epsilon_identity": True}
    assert report["sample_rate"] == 44100 and report["samples_per_frame"] == 2048
    names = tuple(f"{kind}_weights.npz" for kind in ("rvq", "transformer", "upsampler", "decoder"))
    weights = {}
    fixture = f"reference-codes-{frames}.npz"
    for name in (*names, fixture):
        with (folder / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == report["replay_sha256"][name]
        if name in names:
            with np.load(folder / name) as archive:
                weights.update(archive)
    runtime = compile_waveform(tmp_path, frames, weights)
    with np.load(folder / fixture) as oracle:
        set_rotary(runtime, oracle["rotary_cos"], oracle["rotary_sin"])
        assert invoke(runtime, oracle["codes"][0]) == 0
        view = runtime[2]
        assert view("dac_input").shape == (1024, frames * 4)
        assert view("pcm").shape == (1, frames * 2048)
        native, expected = view("pcm")[0], oracle["samples"]
        assert np.isfinite(native).all()
        np.testing.assert_allclose(native, expected, rtol=0, atol=PCM_HALF_STEP)
        assert np.abs(pcm16(native) - pcm16(expected)).max() <= 1
        check_latent_roundoff(view, weights, record_property)
        record_property("pcm_max_abs", float(np.abs(native - expected).max()))
        for name in ("upsample0_deconv", "upsample0_out", "upsample1_deconv", "dac_input",
                     "decoder_input_conv", "dec1_res4", "dec2_res4", "dec3_res4", "dec4_res4",
                     "decoder_pre_tanh"):
            actual, reference = view(name), oracle[name][0]
            delta = np.abs(actual - reference)
            record_property(f"{name}_max_abs", float(delta.max()))
            record_property(f"{name}_over_1e5_atol_1e5_rtol",
                            int(np.count_nonzero(delta > 1e-5 + 1e-5 * np.abs(reference))))
