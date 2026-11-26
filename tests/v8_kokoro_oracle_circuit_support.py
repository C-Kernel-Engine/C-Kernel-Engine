"""Recreate test-only Kokoro circuit JSON without checked-in copies."""

from hashlib import sha256
from pathlib import Path
import json

from tests.v8_checked_graph_test_support import compile_native_graph


# SHA-256 of the previously checked-in snapshots. Preserve the independent
# regeneration check and the circuit bytes used by build identity.
SNAPSHOT_SHA256 = {
    "kokoro_generator_main_resblock0_oracle_input_bounded":
        "d9e80af7121752b6bf5994b0e7a879d378fb0d547eaebef2f9867907b06fe5cf",
    "kokoro_generator_stage0_pool_oracle_input_bounded":
        "1df1f7ff329b7c413bba0c99e0f642cddc1a31d0cacd000dd3354fc728d5f110",
    "kokoro_generator_stage1_ingress_oracle_input_bounded":
        "cef48fc8e0b744641df9b05de3322732fd4930954a37c1b36dcc9c1e661f753f",
    "kokoro_generator_stage1_join_oracle_input_bounded":
        "d8c1b9fad2a67e08af2e765709a8a5b5bd295ecb7f013ca68cae1bacf6934a96",
    "kokoro_generator_stage1_pool_oracle_input_bounded":
        "9d93349f3d96c35bc22cab0fce6666230d251bf0958e475b5c625fa2c65988b8",
    "kokoro_generator_stage1_source_conv_oracle_input_bounded":
        "917adb5f289d62efd548a12d90aed5c570bea0ed7d8ac37d3e5e3ad10541e751",
    "kokoro_source_residual_block0_oracle_input_bounded":
        "12421b99071ab5d14a25f6c3e33fbd0426fa535f748c4485d61f4b68eb7d3760",
    "kokoro_source_residual_first_pair_oracle_input_bounded":
        "dc6fac36652ed7dc7e6950132f4058b3ec603acf7e5b834fd1f973a2d95fc22c",
    "kokoro_source_residual_oracle_input_bounded":
        "467c82c6efdc44eb6e3754e765e722978edb072d4ca64e2c6de4d4508d99f74f",
    "kokoro_waveform_oracle_input_bounded":
        "431550dbaf7903b39a41f1ac55a99a6ca3cf9cf1f1741c022e37cc9bfdba4144",
}


def oracle_circuit_bytes(graph):
    name = graph["name"]
    content = (json.dumps(graph, indent=2) + "\n").encode("utf-8")
    if sha256(content).hexdigest() != SNAPSHOT_SHA256[name]:
        raise AssertionError(f"Kokoro oracle circuit differs from pinned snapshot: {name}")
    return content


def compile_oracle_graph(directory, source):
    graph = source["template"]
    circuit = Path(directory) / (graph["name"] + ".json")
    circuit.write_bytes(oracle_circuit_bytes(graph))
    return compile_native_graph(directory, source, circuit)
