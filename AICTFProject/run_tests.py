#!/usr/bin/env python3
"""Run the test suite with TensorFlow/oneDNN stderr quiet (before any heavy imports)."""

from __future__ import annotations

import os
import sys
import unittest


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    suite = unittest.defaultTestLoader.discover(
        start_dir=os.path.join(here, "tests"),
        pattern="test*.py",
        top_level_dir=here,
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
