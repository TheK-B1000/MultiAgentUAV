"""Test package: reduce third-party log noise before heavy imports (TensorBoard / TensorFlow)."""

from __future__ import annotations

import os

# TensorBoard can pull TensorFlow; keep stderr quiet for local and CI test runs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
