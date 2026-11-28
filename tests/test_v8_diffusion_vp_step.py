"""Same-input property checks for native FP64 VP transitions."""
import ctypes
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    lib = tmp_path_factory.mktemp("ddpm") / "ddpm.so"
    subprocess.run([
        "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
        "-shared", "-fPIC", str(ROOT / "src/kernels/diffusion_vp_step.c"), "-lm", "-o", str(lib),
    ], check=True)
    return ctypes.CDLL(str(lib))


@pytest.fixture(scope="module")
def step(library):
    native = library.ck_diffusion_vp_ddpm_step_f64
    native.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t] + [ctypes.c_double] * 2
    native.restype = ctypes.c_int

    def invoke(latent, estimate, noise, output, begin, end):
        return native(latent.ctypes.data, estimate.ctypes.data, noise.ctypes.data,
                      output.ctypes.data, latent.size, begin, end)
    return invoke


def test_ddpm_transition_matches_pytorch_at_same_noise(step):
    torch = pytest.importorskip("torch")
    z, x, noise = np.random.default_rng(34).normal(size=(3, 1027))
    result = np.full(z.size + 1, -91.0)
    for a, b in ((8.0, 4.25), (4.25, -4.2), (-4.2, -5.25), (-5.25, -5.25)):
        assert step(z, x, noise, result, a, b) == 0
        begin, end = torch.tensor([a, b], dtype=torch.float64)
        src, dst = torch.sigmoid(-begin), torch.sigmoid(-end)
        blend = -torch.expm1(end - begin)
        expected = ((1 - blend) * (dst / src).sqrt() * torch.from_numpy(z) +
                    blend * dst.sqrt() * torch.from_numpy(x) +
                    (blend * (1 - dst)).sqrt() * torch.from_numpy(noise))
        np.testing.assert_allclose(result[:-1], expected.numpy(), atol=2e-15, rtol=2e-15)
        assert result[-1] == -91.0


def test_rejects_invalid_state_before_writing_and_recovers(step):
    z = np.array([0.5, -0.2, -0.1])
    x, noise, output = np.array([0.2, 0.4, 0.8]), np.ones_like(z), np.full_like(z, -91)
    assert step(z, x, noise, output, -2, 4) != 0
    noise[-1] = np.nan
    assert step(z, x, noise, output, 4, -2) != 0
    np.testing.assert_array_equal(output, -91)
    noise[-1] = 1.0
    before = z.copy()
    assert step(z, x, noise, z, 4, -2) != 0
    np.testing.assert_array_equal(z, before)
    assert step(z, x, noise, output, 4, -2) == 0


def test_ddim_checked_failure_preserves_output(library):
    native = library.ck_diffusion_vp_ddim_step_schedule_f64
    native.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t] * 3
    native.restype = ctypes.c_int
    latent, estimate = np.array([1.2, -.7, .4]), np.array([.1, .6, -.3])
    gamma, out = np.array([4.0, -3.0]), np.full_like(latent, -81)

    def invoke(destination, temperature=300, gamma_count=2):
        return native(latent.ctypes.data, estimate.ctypes.data, gamma.ctypes.data,
                      destination.ctypes.data, latent.size, gamma_count, temperature)

    assert invoke(out, temperature=0) == -1
    assert invoke(out, gamma_count=1) == -1
    gamma[:] = (-3.0, 4.0)
    assert invoke(out) == -1
    gamma[:] = (4.0, -3.0)
    prior = latent.copy()
    assert invoke(latent) == -1
    np.testing.assert_array_equal(latent, prior)
    estimate[-1] = np.nan
    assert invoke(out) == -1
    np.testing.assert_array_equal(out, -81)
    estimate[-1] = -.3
    assert invoke(out) == 0
    assert np.isfinite(out).all()
    gamma[1] = gamma[0]
    assert invoke(out) == 0
    np.testing.assert_array_equal(out, latent)


def declared_ddpm(count, gamma_from, gamma_to):
    names = ("latent", "estimate", "noise", "output")
    graph = {
        "version": 3, "name": "bounded_ddpm", "family": "bounded_graph",
        "checked_native_entry": True,
        "activation_buffers": {name: {"shape": [count]} for name in names},
        "activation_bindings": {name: name for name in names},
        "contract": {"runtime_invariants": {"inference_only": True}},
        "native_entry": {
            "function": "ck_test_ddpm",
            "params": [{"c_type": "uint8_t *", "name": "arena"},
                       {"c_type": "size_t", "name": "arena_bytes"}],
            "arena": {"pointer": "arena", "bytes": "arena_bytes"},
        },
        "sequence": ["block"],
        "block_types": {"block": {
            "sequence": ["header", "body", "footer"], "header": [], "footer": [],
            "body": {"type": "dense", "ops": [{
                "id": "denoise_step", "op": "diffusion_vp_ddpm_step_checked",
                "kernel": "diffusion_vp_ddpm_step_checked_f64", "returns_status": True,
                "graph_slots": {"inputs": {name: f"external:{name}" for name in names[:-1]},
                                "outputs": {"output": "output"}},
                "params": {"N": count, "gamma_from": gamma_from, "gamma_to": gamma_to,
                           "call_constants": {"count": count}},
            }]},
        }},
    }
    config = {"num_layers": 1, "embed_dim": 16, "num_heads": 1, "num_kv_heads": 1,
              "head_dim": 16, "intermediate_size": 16, "context_length": 1, "max_seq_len": 1,
              "activation_buffer_dtypes": dict.fromkeys(names, "fp64")}
    return names, graph, config


def test_generated_ddpm_uses_typed_fp64_storage(tmp_path, step):
    from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph

    count = 17
    names, graph, config = declared_ddpm(count, 4.25, -4.2)
    run, arena, view, _ = compile_checked_weight_graph(tmp_path, graph, {}, {}, source_config=config)
    z, x, noise = np.random.default_rng(359).normal(size=(3, count))
    for name, data in zip(names[:-1], (z, x, noise), strict=True):
        view(name, np.float64)[:] = data
    expected = np.empty_like(z)
    assert step(z, x, noise, expected, 4.25, -4.2) == 0
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("output", np.float64), expected)

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    with pytest.raises(RuntimeError, match="capacity exceeds planned storage|storage dtype disagrees"):
        compile_checked_weight_graph(
            invalid, graph, {}, {}, source_config={**config, "activation_buffer_dtypes": {}}
        )


def test_generated_ddpm_replays_sigma_reference_transitions(tmp_path):
    from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph

    fixture = ROOT / "tests/fixtures/diffusion/sigma3b_ddpm_bf16_reference.json"
    trace = json.loads(fixture.read_text())
    latent = np.array(trace["initial"], dtype=np.float64)
    assert latent.size == np.prod(trace["shape"])
    for index, transition in enumerate(trace["transitions"]):
        names, graph, config = declared_ddpm(
            latent.size, transition["gamma_from"], transition["gamma_to"]
        )
        folder = tmp_path / str(index)
        folder.mkdir()
        run, arena, view, _ = compile_checked_weight_graph(
            folder, graph, {}, {}, source_config=config
        )
        operands = (latent, transition["estimate"], transition["noise"])
        for name, values in zip(names[:-1], operands, strict=True):
            view(name, np.float64)[:] = values
        assert run(arena, len(arena)) == 0
        latent = view("output", np.float64).copy()
        np.testing.assert_allclose(latent, transition["expected"], atol=2e-15, rtol=2e-15)
