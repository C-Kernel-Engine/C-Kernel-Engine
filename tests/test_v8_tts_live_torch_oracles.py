"""Nightly-only live PyTorch oracle gate with an explicit missing-dependency status."""

import unittest


if __name__ == "__main__":
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"TEST SKIPPED: live PyTorch TTS oracle unavailable: {exc}")
    else:
        from test_v8_audio_duration_expand_oracle import AudioDurationExpandOracleTest
        from test_v8_audio_istft_oracle import AudioIstftOracleTest

        suite = unittest.TestSuite([
            AudioDurationExpandOracleTest("test_live_torch_oracle"),
            AudioIstftOracleTest("test_centered_hann_against_torch"),
        ])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
