"""Compatibility module retained after GPU env wrapper extraction."""
from __future__ import annotations

from ._adapter import GPUEnvAdapter, _FakeGM
from ._envs import GPUCTFSingleEnv, GPUCTFVecEnv

__all__ = ["GPUEnvAdapter", "_FakeGM", "GPUCTFVecEnv", "GPUCTFSingleEnv"]
