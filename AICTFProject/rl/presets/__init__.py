"""Registry and resolution entry points for training presets."""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl.train_ppo import PPOConfig

from rl.presets.plan_faithful import (
    apply_plan_faithful_latent,
    apply_plan_faithful_latent_no_persistence,
    apply_plan_faithful_latent_strategic,
    apply_plan_faithful_latent_step6,
    apply_plan_faithful_latent_episode_strategic,
    apply_plan_faithful_latent_v3b_marginal,
    apply_plan_faithful_latent_no_entropy,
    apply_plan_faithful_latent_phase1_coupling,
    apply_plan_faithful_latent_phase2_credit,
    apply_plan_faithful_latent_phase3_reward_geometry,
    apply_plan_faithful_latent_phase3b_outcome_clean,
    apply_plan_faithful_latent_phase3b_ablate_k1,
    apply_plan_faithful_latent_phase3b_ablate_no_persistence,
    apply_plan_faithful_latent_phase4a_rescue,
    apply_plan_faithful_latent_phase4a_rescue_hardpool,
    apply_plan_faithful_latent_episode_z_clean,
    apply_plan_faithful_latent_option_a_episode_credit,
    apply_plan_faithful_latent_option_a,
    apply_plan_faithful_latent_k1,
    apply_plan_faithful_no_latent,
    apply_plan_option_a,
    apply_plan_option_b_lamp,
    apply_latent_a1_plan_faithful,
)
from rl.presets.hypothesis import (
    apply_hypothesis_flat_opprand,
    apply_hypothesis_latent_opprand_optiona,
    apply_hypothesis_latent_opprand_optionb_lamp_coef05,
    apply_hypothesis_latent_opprand_optionb_no_lamp,
    apply_hypothesis_latent_opprand_optionb_coef03,
    apply_hypothesis_flat_opprand_op35,
    apply_hypothesis_latent_opprand_optionb_lamp_coef05_op35,
)
from rl.presets.other import (
    apply_latent_op3_push80_1m,
    apply_latent_train80_op3_1m,
    apply_latent_op3_wrmax_1m,
    apply_latent_op3_wrmax_train_2m,
)

PRESET_REGISTRY = {
    # Plan-faithful family
    "plan_faithful_latent": apply_plan_faithful_latent,
    "plan_faithful_latent_persist_entropy": apply_plan_faithful_latent,
    "latent_plan_faithful": apply_plan_faithful_latent,
    "latent_plan_faithful_persist_entropy": apply_plan_faithful_latent,
    "latent_recommended": apply_plan_faithful_latent,
    "plan_faithful_latent_no_persistence": apply_plan_faithful_latent_no_persistence,
    "latent_plan_faithful_no_persistence": apply_plan_faithful_latent_no_persistence,
    "latent_recommended_no_persistence": apply_plan_faithful_latent_no_persistence,
    "plan_faithful_latent_strategic": apply_plan_faithful_latent_strategic,
    "latent_strategic": apply_plan_faithful_latent_strategic,
    "plan_faithful_latent_option_credit": apply_plan_faithful_latent_strategic,
    "plan_faithful_latent_step6": apply_plan_faithful_latent_step6,
    "latent_step6": apply_plan_faithful_latent_step6,
    "plan_faithful_latent_episode_strategic": apply_plan_faithful_latent_episode_strategic,
    "latent_episode_strategic": apply_plan_faithful_latent_episode_strategic,
    "plan_faithful_latent_v3b_marginal": apply_plan_faithful_latent_v3b_marginal,
    "latent_v3b_marginal": apply_plan_faithful_latent_v3b_marginal,
    "plan_faithful_latent_intent_credit": apply_plan_faithful_latent_episode_strategic,
    "plan_faithful_latent_no_entropy": apply_plan_faithful_latent_no_entropy,
    "latent_plan_faithful_no_entropy": apply_plan_faithful_latent_no_entropy,
    "latent_recommended_no_entropy": apply_plan_faithful_latent_no_entropy,
    "plan_faithful_latent_phase1_coupling": apply_plan_faithful_latent_phase1_coupling,
    "latent_phase1_coupling": apply_plan_faithful_latent_phase1_coupling,
    "plan_faithful_latent_phase2_credit": apply_plan_faithful_latent_phase2_credit,
    "latent_phase2_credit": apply_plan_faithful_latent_phase2_credit,
    "plan_faithful_latent_phase3_reward_geometry": apply_plan_faithful_latent_phase3_reward_geometry,
    "latent_phase3_reward_geometry": apply_plan_faithful_latent_phase3_reward_geometry,
    "plan_faithful_latent_phase3b_outcome_clean": apply_plan_faithful_latent_phase3b_outcome_clean,
    "latent_phase3b_outcome_clean": apply_plan_faithful_latent_phase3b_outcome_clean,
    "plan_faithful_latent_phase3b_ablate_k1": apply_plan_faithful_latent_phase3b_ablate_k1,
    "latent_phase3b_ablate_k1": apply_plan_faithful_latent_phase3b_ablate_k1,
    "plan_faithful_latent_phase3b_ablate_no_persistence": apply_plan_faithful_latent_phase3b_ablate_no_persistence,
    "latent_phase3b_ablate_no_persistence": apply_plan_faithful_latent_phase3b_ablate_no_persistence,
    "plan_faithful_latent_phase4a_rescue": apply_plan_faithful_latent_phase4a_rescue,
    "latent_phase4a_rescue": apply_plan_faithful_latent_phase4a_rescue,
    "plan_faithful_latent_phase4a_rescue_hardpool": apply_plan_faithful_latent_phase4a_rescue_hardpool,
    "latent_phase4a_rescue_hardpool": apply_plan_faithful_latent_phase4a_rescue_hardpool,
    "plan_faithful_latent_episode_z_clean": apply_plan_faithful_latent_episode_z_clean,
    "latent_episode_z_clean": apply_plan_faithful_latent_episode_z_clean,
    "plan_faithful_latent_option_a_episode_credit": apply_plan_faithful_latent_option_a_episode_credit,
    "latent_option_a_episode_credit": apply_plan_faithful_latent_option_a_episode_credit,
    "plan_faithful_latent_episode_credit": apply_plan_faithful_latent_option_a_episode_credit,
    "plan_faithful_latent_option_a": apply_plan_faithful_latent_option_a,
    "latent_option_a": apply_plan_faithful_latent_option_a,
    "plan_faithful_latent_fix_d": apply_plan_faithful_latent_option_a,
    "latent_fix_d": apply_plan_faithful_latent_option_a,
    "plan_faithful_latent_k1": apply_plan_faithful_latent_k1,
    "latent_plan_faithful_k1": apply_plan_faithful_latent_k1,
    "plan_faithful_collapsed_latent": apply_plan_faithful_latent_k1,
    "latent_recommended_collapsed_k1": apply_plan_faithful_latent_k1,
    "plan_faithful_no_latent": apply_plan_faithful_no_latent,
    "no_latent_plan_faithful": apply_plan_faithful_no_latent,
    "no_latent_baseline": apply_plan_faithful_no_latent,
    "plan_option_a": apply_plan_option_a,
    "plan_option_b_lamp": apply_plan_option_b_lamp,
    "plan_option_b": apply_plan_option_b_lamp,

    # Hypothesis family
    "hypothesis_flat_opprand": apply_hypothesis_flat_opprand,
    "hypothesis_latent_opprand_optiona": apply_hypothesis_latent_opprand_optiona,
    "hypothesis_latent_opprand_optionb_lamp_coef05": apply_hypothesis_latent_opprand_optionb_lamp_coef05,
    "hypothesis_latent_opprand_optionb_no_lamp": apply_hypothesis_latent_opprand_optionb_no_lamp,
    "hypothesis_latent_opprand_optionb_coef03": apply_hypothesis_latent_opprand_optionb_coef03,
    "hypothesis_flat_opprand_op35": apply_hypothesis_flat_opprand_op35,
    "hypothesis_latent_opprand_optionb_lamp_coef05_op35": apply_hypothesis_latent_opprand_optionb_lamp_coef05_op35,

    # Other family
    "latent_op3_push80_1m": apply_latent_op3_push80_1m,
    "latent_push80_1m": apply_latent_op3_push80_1m,
    "latent_train80_op3_1m": apply_latent_train80_op3_1m,
    "latent_op3_train80_1m": apply_latent_train80_op3_1m,
    "latent_op3_wrmax_1m": apply_latent_op3_wrmax_1m,
    "latent_wrmax_op3_1m": apply_latent_op3_wrmax_1m,
    "latent_op3_wrmax_2m": apply_latent_op3_wrmax_1m,
    "latent_wrmax_op3_2m": apply_latent_op3_wrmax_1m,
    "latent_a1_plan_faithful": apply_latent_a1_plan_faithful,
    "latent_op3_a1_plan_faithful": apply_latent_a1_plan_faithful,
    "latent_op3_wrmax_train_2m": apply_latent_op3_wrmax_train_2m,
    "latent_wrmax_op3_train_2m": apply_latent_op3_wrmax_train_2m,
}


def apply_preset(cfg: PPOConfig, preset: str) -> PPOConfig:
    """Apply named high-level presets for repeatable training recipes."""
    key = str(preset).strip().lower()
    if not key:
        return cfg
    fn = PRESET_REGISTRY.get(key)
    if fn is None:
        raise ValueError(
            f"Unknown preset {preset!r}. Supported presets: "
            "'plan_option_a', 'plan_option_b_lamp', 'plan_option_b', "
            "'hypothesis_flat_opprand', 'hypothesis_latent_opprand_optiona', "
            "'hypothesis_latent_opprand_optionb_lamp_coef05', 'hypothesis_latent_opprand_optionb_no_lamp', "
            "'hypothesis_latent_opprand_optionb_coef03', "
            "'hypothesis_flat_opprand_op35', 'hypothesis_latent_opprand_optionb_lamp_coef05_op35', "
            "'latent_op3_push80_1m', 'latent_push80_1m', "
            "'latent_train80_op3_1m', 'latent_op3_train80_1m', "
            "'latent_op3_wrmax_1m', 'latent_wrmax_op3_1m', "
            "'latent_op3_wrmax_2m', 'latent_wrmax_op3_2m' (aliases for wrmax 1M), "
            "'latent_op3_wrmax_train_2m', 'latent_wrmax_op3_train_2m', "
            "'latent_a1_plan_faithful', 'latent_op3_a1_plan_faithful', "
            "'plan_faithful_latent_persist_entropy', 'plan_faithful_latent_no_persistence', "
            "'plan_faithful_latent_no_entropy', 'plan_faithful_latent_k1', 'plan_faithful_no_latent', "
            "'plan_faithful_latent_phase1_coupling', 'plan_faithful_latent_phase2_credit', "
            "'plan_faithful_latent_phase3_reward_geometry', "
            "'plan_faithful_latent_phase3b_outcome_clean', "
            "'plan_faithful_latent_phase3b_ablate_k1', "
            "'plan_faithful_latent_phase3b_ablate_no_persistence', "
            "'plan_faithful_latent_phase4a_rescue', "
            "'plan_faithful_latent_phase4a_rescue_hardpool', "
            "'plan_faithful_latent_episode_z_clean', "
            "'plan_faithful_latent_option_a' (a.k.a. 'plan_faithful_latent_fix_d'), "
            "'plan_faithful_latent_option_a_episode_credit'."
        )
    return fn(cfg)
