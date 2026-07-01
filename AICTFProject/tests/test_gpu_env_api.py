from __future__ import annotations

import unittest

import game_field_gpu


EXPECTED = {
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
}


class GPUEnvApiTests(unittest.TestCase):
    def test_public_api_intact(self) -> None:
        missing = EXPECTED - set(dir(game_field_gpu))
        self.assertFalse(missing, f"public API regression: {missing}")
