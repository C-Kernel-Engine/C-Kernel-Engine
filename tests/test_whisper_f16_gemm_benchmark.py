import importlib.util
from pathlib import Path
from unittest import mock

import numpy as np


def test_invocation_records_cpu_and_wall_time():
    path = Path(__file__).resolve().parents[1] / "benchmarks/bench_whisper_f16_gemm.py"
    spec = importlib.util.spec_from_file_location("whisper_gemm_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = []
    with mock.patch.object(module.time, "perf_counter", side_effect=[10.0, 10.5]), \
         mock.patch.object(module.time, "process_time", side_effect=[20.0, 22.0]):
        elapsed = module._invoke(
            mock.Mock(), np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float16),
            np.zeros((1, 1), dtype=np.float32),
            baseline=False, cpu_samples=samples,
        )
    assert elapsed == 0.5
    assert samples == [2.0]
