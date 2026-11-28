"""Independent small reference for routed MoE BF16 storage boundaries."""

import ctypes
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FPTR = ctypes.POINTER(ctypes.c_float)
IPTR = ctypes.POINTER(ctypes.c_int)
U16PTR = ctypes.POINTER(ctypes.c_uint16)


def bf16(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def bits(values):
    return (bf16(values).view(np.uint32) >> 16).astype(np.uint16)


def reference(hidden, indices, routing, gate, up, down, *, fp32_accum=False):
    rows, width = hidden.shape
    intermediate = gate.shape[1]
    out = np.zeros((rows, width), dtype=np.float32)
    for row in range(rows):
        # Publisher expert-major index_add order, independent of top-k order.
        for expert in range(gate.shape[0]):
            for slot in range(indices.shape[1]):
                if indices[row, slot] != expert:
                    continue
                x = bf16(hidden[row])
                gate_value = bf16(gate[expert] @ x)
                up_value = bf16(up[expert] @ x)
                silu = bf16(gate_value / (1.0 + np.exp(-gate_value)))
                activated = bf16(silu * up_value)
                assert activated.shape == (intermediate,)
                projected = bf16(down[expert] @ activated)
                weighted = projected * routing[row, slot]
                if fp32_accum:
                    out[row] += weighted
                else:
                    out[row] = bf16(out[row] + bf16(weighted))
    return bf16(out)


class MoEBf16StageStorageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "avx512f" in Path("/proc/cpuinfo").read_text() and not os.environ.get("CK_SLEEF_LIBRARY"):
            import torch
            lib = Path(torch.__file__).resolve().parent / "lib/libtorch_cpu.so"
            if not lib.is_file():
                raise RuntimeError("AVX-512 BF16 parity requires CK_SLEEF_LIBRARY")
            os.environ["CK_SLEEF_LIBRARY"] = str(lib)
        subprocess.run(
            ["make", "--no-print-directory", "build/libckernel_engine.so"],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
        )
        cls.engine = ctypes.CDLL(str(ROOT / "build/libckernel_engine.so"))

    def test_stages_and_expert_order(self):
        rng = np.random.default_rng(745)
        rows, width, intermediate, experts, top_k = 2, 7, 5, 3, 2
        hidden = np.ascontiguousarray(rng.normal(0.0, 0.7, (rows, width)), dtype=np.float32)
        gate = bf16(rng.normal(0.0, 0.5, (experts, intermediate, width)))
        up = bf16(rng.normal(0.0, 0.45, (experts, intermediate, width)))
        down = bf16(rng.normal(0.0, 0.4, (experts, width, intermediate)))
        indices = np.ascontiguousarray([[2, 0], [2, 1]], dtype=np.int32)
        routing = np.ascontiguousarray([[0.72413, 0.27587], [0.63891, 0.36109]], dtype=np.float32)
        expected = reference(hidden, indices, routing, gate, up, down)
        gate_bits, up_bits, down_bits = bits(gate), bits(up), bits(down)

        def call(name):
            kernel = getattr(self.engine, name)
            kernel.argtypes = [FPTR, IPTR, FPTR, U16PTR, U16PTR, U16PTR, FPTR] + [ctypes.c_int] * 5
            actual = np.empty_like(expected)
            kernel(
                hidden.ctypes.data_as(FPTR), indices.ctypes.data_as(IPTR), routing.ctypes.data_as(FPTR),
                gate_bits.ctypes.data_as(U16PTR), up_bits.ctypes.data_as(U16PTR),
                down_bits.ctypes.data_as(U16PTR), actual.ctypes.data_as(FPTR),
                rows, width, intermediate, experts, top_k,
            )
            return actual

        actual = call("moe_swiglu_expert_forward_pytorch_bf16_storage")
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, bf16(actual))

        fp32_accum = call("moe_swiglu_expert_forward_bf16_stages_f32_accum")
        np.testing.assert_array_equal(
            fp32_accum, reference(hidden, indices, routing, gate, up, down, fp32_accum=True),
        )
        self.assertFalse(np.array_equal(fp32_accum, expected))

        # The old provider intentionally keeps FP32 intermediates. This fixture
        # must detect selecting that different numerical contract.
        old = call("moe_swiglu_expert_forward_bf16")
        self.assertGreater(float(np.max(np.abs(old - expected))), 0.004)

    def test_circuit_selects_declared_provider(self):
        circuit = json.loads((ROOT / "version/v8/circuits/instella_moe.json").read_text())
        selected = circuit["kernels"]["moe_swiglu_expert_mlp"]
        self.assertEqual(selected, "moe_swiglu_expert_forward_pytorch_bf16_storage")
        provider = json.loads((ROOT / "version/v8/kernel_maps" / (selected + ".json")).read_text())
        self.assertEqual(provider["numerical_contract"],
                         "moe_expert_pytorch_bf16_stages_fp32_route_bf16_accum")

    def test_composite_resolver_honors_compatible_circuit_binding(self):
        sys.path.insert(0, str(ROOT / "version/v8/scripts"))
        from build_ir_v8 import resolve_storage_aware_circuit_provider

        registry = json.loads((ROOT / "version/v8/kernel_maps/KERNEL_REGISTRY.json").read_text())
        kwargs = dict(
            registry=registry, template_op="moe_swiglu_expert_mlp",
            kernel_op="moe_swiglu_expert_mlp",
            layer_quant={
                "moe_expert_gate": "bf16", "moe_expert_up": "bf16",
                "moe_expert_down": "bf16",
            },
            header_quant={}, weight_keys=["moe_expert_gate", "moe_expert_up", "moe_expert_down"],
            mode="decode", prefer_q8_activation=False,
            bf16_dense_matmul=False,
        )
        selected = "moe_swiglu_expert_forward_pytorch_bf16_storage"
        self.assertEqual(
            resolve_storage_aware_circuit_provider(**kwargs, explicit_kernel=selected),
            selected,
        )
        # Existing legacy bindings can name an FP32 default even when one
        # converted layer carries BF16; the storage-aware fallback remains.
        self.assertEqual(
            resolve_storage_aware_circuit_provider(
                **kwargs, explicit_kernel="moe_swiglu_expert_forward_f32",
            ),
            "moe_swiglu_expert_forward_bf16",
        )
        with self.assertRaisesRegex(RuntimeError, "unknown provider"):
            resolve_storage_aware_circuit_provider(
                **kwargs, explicit_kernel="not_a_real_provider",
            )


if __name__ == "__main__":
    unittest.main()
