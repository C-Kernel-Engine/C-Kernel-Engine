"""Generated BF16 storage at input, weights and output of a checked projection."""
import numpy as np
import pytest

from version.v8.scripts.compile_checked_bundle_v8 import compile_checked_weight_graph
from version.v8.scripts.replay_bf16_projection_bundle_v8 import CONTRACT, PROVIDER, projection_graph, replay
from version.v8.scripts.xray_operation_bundle_v8 import BundleError, write_bundle


def test_generated_bf16_projection_and_rejection(tmp_path):
    torch = pytest.importorskip("torch")
    t, width, channels = 3, 5, 7
    graph = projection_graph(t, width, channels)
    generator = torch.Generator().manual_seed(6)
    x = torch.randn(t, width, generator=generator).bfloat16()
    w = torch.randn(channels, width, generator=generator).bfloat16()
    params = {"projection.weight": w.view(torch.uint16).numpy()}
    declarations = {"projection.weight": ([channels, width], "BF16")}
    config = {"num_layers": 1, "embed_dim": width, "num_heads": 1, "num_kv_heads": 1,
              "head_dim": width, "intermediate_size": channels, "context_length": t, "max_seq_len": t,
              "activation_buffer_dtypes": {"input": "bf16", "output": "bf16"}}
    run, arena, view, _ = compile_checked_weight_graph(
        tmp_path, graph, params, declarations, source_config=config)
    view("input", np.uint16)[:] = x.view(torch.uint16).numpy()
    expected = torch.nn.functional.linear(x, w).view(torch.uint16).numpy()
    assert run(arena, len(arena)) == 0
    np.testing.assert_array_equal(view("output", np.uint16), expected)
    tensors = {
        "activation": (x.view(torch.uint16).numpy(), "bf16_bits", ("token", "feature")),
        "weight": (w.view(torch.uint16).numpy(), "bf16_bits", ("channel", "feature")),
        "cke": ((expected.astype(np.uint32) << 16).view(np.float32), "fp32", ("token", "channel")),
        "oracle": (torch.nn.functional.linear(x, w).float().numpy(), "fp32", ("token", "channel")),
    }
    packet = write_bundle(tmp_path / "packet", operation=PROVIDER, provider=PROVIDER,
                          contract=CONTRACT, runtime=tmp_path / "generated.so",
                          context={}, tensors=tensors)
    result = replay(packet, tmp_path / "replayed")
    assert result["capture_reproduced"]
    assert result["oracle_exact_count"] == result["output_count"]
    wrong = {**tensors, "activation": (tensors["activation"][0], "bf16_bits", ("feature", "token"))}
    packet = write_bundle(tmp_path / "wrong", operation=PROVIDER, provider=PROVIDER,
                          contract=CONTRACT, runtime=tmp_path / "generated.so",
                          context={}, tensors=wrong)
    with pytest.raises(BundleError, match="storage or layout"):
        replay(packet, tmp_path / "rejected")
    view("input", np.uint16)[-1, -1] = 0x7FC0
    view("output", np.uint16).fill(11)
    assert run(arena, len(arena)) != 0
    np.testing.assert_array_equal(view("output", np.uint16), 11)

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    with pytest.raises(RuntimeError, match="storage dtype disagrees with planned layout"):
        compile_checked_weight_graph(
            invalid, graph, params, declarations,
            source_config={**config, "activation_buffer_dtypes": {}},
        )
