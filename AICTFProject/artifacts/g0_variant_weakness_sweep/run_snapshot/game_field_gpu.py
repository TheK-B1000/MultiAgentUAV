"""Compatibility facade for the GPU CTF environment package."""

from __future__ import annotations

from gpu_env import *  # noqa: F403
from gpu_env import __all__ as _GPU_ENV_ALL

__all__ = list(_GPU_ENV_ALL)
