"""Declare Sigma's compact BF16 normalization and Q/K/V projection boundary."""
from version.v8.scripts.checked_bf16_qkv_authoring_v8 import bf16_qkv_ingress


def build_circuit(rows: int, *, width: int = 3072, q_width: int = 4096, kv_width: int = 1024):
    return bf16_qkv_ingress(
        name="sigma_bf16_attention_ingress", entry="ck_sigma_bf16_attention_ingress",
        rows=rows, width=width, q_width=q_width, kv_width=kv_width, epsilon=1e-5,
        norm="rmsnorm_rows_bf16_checked",
    )
