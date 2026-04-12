#!/usr/bin/env python3
"""
V8 test entry points

Use these compatibility entry points from the current operator surface:

  python3 version/v8/test/ck_test_runner.py --quick
  python3 version/v8/test/ck_test_runner.py --all
  python3 version/v8/test/trace_divergence.py --model <run_dir> --token 25
  make v8-visualizer-health
  make v8-visualizer-generated-e2e
"""


if __name__ == "__main__":
    print(__doc__.strip())
