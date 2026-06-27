from __future__ import annotations

from rl.custom_ppo.checkpoints.archive import _torch_load_checkpoint
from rl.custom_ppo.checkpoints.loader import (
    ACTION_GENERATOR_SEED_OFFSET,
    STRATEGY_GENERATOR_SEED_OFFSET,
    _model_kwargs_from_cfg,
    apply_deterministic_sampling_generators,
    load_custom_ppo_checkpoint,
    load_custom_ppo_policy,
)
from rl.custom_ppo.checkpoints.metadata import (
    CUSTOM_PPO_ACTOR_ARCH,
    CUSTOM_PPO_FORMAT,
    CUSTOM_PPO_LATENT_FORMAT,
    CUSTOM_PPO_VEC_SCHEMA_VERSION,
    assert_compatible_global_state_dim as _assert_compatible_global_state_dim,
    canonicalize_latent_strategy_cfg,
    read_custom_ppo_metadata,
)
from rl.custom_ppo.checkpoints.state_dict import (
    _expand_cnn_obs_channels,
    _load_model_state_dict_compat,
    _remap_legacy_strategy_aux_head_state_dict,
)
from rl.custom_ppo.checkpoints.validation import run_behavioral_equivalence_probe
from rl.custom_ppo.inference_policy import CustomPPOInferencePolicy

FORCED_Z_PROFILE_MAX_ROWS = 4096
FORCED_Z_MACRO_ACTIONS: tuple[tuple[int, str], ...] = (
    (0, "go_to"),
    (1, "grab_mine"),
    (2, "get_flag"),
    (3, "place_mine"),
    (4, "go_home"),
)

__all__ = [
    "ACTION_GENERATOR_SEED_OFFSET",
    "CUSTOM_PPO_ACTOR_ARCH",
    "CUSTOM_PPO_FORMAT",
    "CUSTOM_PPO_LATENT_FORMAT",
    "CUSTOM_PPO_VEC_SCHEMA_VERSION",
    "CustomPPOInferencePolicy",
    "FORCED_Z_MACRO_ACTIONS",
    "FORCED_Z_PROFILE_MAX_ROWS",
    "STRATEGY_GENERATOR_SEED_OFFSET",
    "_assert_compatible_global_state_dim",
    "_expand_cnn_obs_channels",
    "_load_model_state_dict_compat",
    "_model_kwargs_from_cfg",
    "_remap_legacy_strategy_aux_head_state_dict",
    "_torch_load_checkpoint",
    "apply_deterministic_sampling_generators",
    "canonicalize_latent_strategy_cfg",
    "load_custom_ppo_checkpoint",
    "load_custom_ppo_policy",
    "read_custom_ppo_metadata",
    "run_behavioral_equivalence_probe",
]
