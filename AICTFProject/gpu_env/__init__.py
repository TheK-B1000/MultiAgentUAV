"""GPU CTF environment package.

Torch-free constants and config are always available at import time.
Heavy torch-dependent classes (BatchedCTFCore, GPUCTFVecEnv, etc.) are
lazy-loaded on first attribute access via PEP 562 ``__getattr__`` so that
``import gpu_env`` does not force a torch import — e.g. tests that only
need ``GPUFieldConfig`` or the ``gpu_env.state`` sub-package work in
environments where torch is absent.
"""
from __future__ import annotations

from ._constants import (
    CNN_COLS,
    CNN_ROWS,
    GLOBAL_STATE_CHANNELS,
    MAP_SET_SEED_OFFSETS,
    METRIC_ZONE_COLS,
    METRIC_ZONE_ROWS,
    NUM_CNN_CHANNELS,
    VEC_OBS_DIM,
)
from ._config import GPUFieldConfig, RewardConfig, RewardProfile
from ._maps import MAP_A_OPEN, MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2, MAP_LAYOUTS

# Torch-dependent classes are lazy-loaded on first access (PEP 562).
_LAZY_TORCH: dict[str, str] = {
    "BatchedCTFCore": "._core_class",
    "GPUEnvAdapter": "._adapter",
    "GPUCTFSingleEnv": "._envs",
    "GPUCTFVecEnv": "._envs",
}


def __getattr__(name: str):
    if name in _LAZY_TORCH:
        import importlib
        mod = importlib.import_module(_LAZY_TORCH[name], __package__)
        obj = getattr(mod, name)
        globals()[name] = obj  # cache so subsequent access skips __getattr__
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GPUCTFVecEnv",
    "GPUCTFSingleEnv",
    "GPUFieldConfig",
    "BatchedCTFCore",
    "RewardConfig",
    "RewardProfile",
    "VEC_OBS_DIM",
    "CNN_COLS",
    "CNN_ROWS",
    "NUM_CNN_CHANNELS",
    "GLOBAL_STATE_CHANNELS",
    "MAP_SET_SEED_OFFSETS",
    "METRIC_ZONE_ROWS",
    "METRIC_ZONE_COLS",
    "GPUEnvAdapter",
    "MAP_A_OPEN",
    "MAP_B_SPLIT_LANE",
    "MAP_B_SPLIT_LANE_V2",
    "MAP_LAYOUTS",
]
