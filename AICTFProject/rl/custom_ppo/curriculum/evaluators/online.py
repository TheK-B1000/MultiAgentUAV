"""Online (training-telemetry) gate evaluators."""

from __future__ import annotations

from typing import Any

import numpy as np

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum.context import GateContext
from rl.custom_ppo.curriculum.types import (
    GATE_STATUS_NOT_RUN,
    LATENT_PAIR_INDEX,
    GateResult,
    gate_family_result_from_bool,
)
from rl.custom_ppo.gate_protocol import evaluate_actor_intervention


def _gate_eval_to_result(result: Any) -> GateResult:
    return GateResult(
        status=result.status,
        reason=result.reason,
        details=dict(result.details),
    )


def evaluate_coverage(context: GateContext) -> GateResult:
    latent_state = context.trainer.latent_state
    min_eps = int(context.cfg.latent_cf_min_episodes_per_z)
    ep_counts = latent_state.cf_episode_counts.tolist()
    coverage_passed = all(int(c) >= min_eps for c in ep_counts)

    rolling_occ = [0.0] * int(context.latent_k)
    if len(latent_state.recent_z_history) > 0:
        hist = list(latent_state.recent_z_history)
        for z in hist:
            rolling_occ[int(z)] += 1.0
        rolling_occ = [c / len(hist) for c in rolling_occ]
    occupancy_passed = all(0.20 <= o <= 0.30 for o in rolling_occ)

    return gate_family_result_from_bool(
        coverage_passed and occupancy_passed,
        details={
            "cf_episode_counts": ep_counts,
            "recent_z_occupancy": rolling_occ,
            "occupancy": rolling_occ,
            "latent_cf_min_episodes_per_z": min_eps,
        },
    )


def evaluate_competence(context: GateContext) -> GateResult:
    latent_state = context.trainer.latent_state
    comp_scores, competence_ready = latent_state.compute_competence_scores()
    competence_passed = bool(competence_ready) and all(float(s) >= 0.50 for s in comp_scores)
    j_returns = latent_state.cf_J.tolist()
    ret_std = float(np.sqrt(max(1e-8, latent_state.cf_return_var)))
    return gate_family_result_from_bool(
        competence_passed,
        details={
            "cf_competence_ready": bool(competence_ready),
            "competence_scores": [float(s) for s in comp_scores.tolist()],
            "return_ema_z": {f"z{i}": float(j_returns[i]) for i in range(len(j_returns))},
            "competence_z": {f"z{i}": float(comp_scores[i]) for i in range(len(comp_scores))},
            "best_minus_worst_return": float(np.max(j_returns) - np.min(j_returns)) if j_returns else 0.0,
            "return_standard_deviation": ret_std,
        },
    )


def evaluate_training_integrity(context: GateContext) -> GateResult:
    stats = dict(getattr(context.trainer, "last_stats", {}) or {})
    k = int(context.latent_k)
    forced_frac = float(stats.get("latent_forced_z_step_fraction", 0.0))
    router_samples = sum(
        float(stats.get(f"router_sample_count_by_z_{z}", 0.0) or 0.0) for z in range(k)
    )
    switch_count = float(stats.get("strategy_switch_count", 0.0))
    qphi_grad = float(
        stats.get(
            "q_phi_grad_norm",
            stats.get("main_loop_q_phi_grad_norm", stats.get("strategy_grad_norm", 0.0)),
        )
        or 0.0
    )
    router_opt_steps = float(
        getattr(context.trainer.latent_state, "router_optimizer_step_count", 0.0)
    )

    passed = (
        abs(forced_frac - 1.0) < 1e-5
        and router_samples == 0.0
        and router_opt_steps == 0.0
        and qphi_grad < 1e-7
        and switch_count == 0.0
    )
    return gate_family_result_from_bool(
        passed,
        details={
            "forced_z_fraction": forced_frac,
            "router_sample_count": router_samples,
            "router_optimizer_step_count": router_opt_steps,
            "q_phi_grad_norm": qphi_grad,
            "strategy_switch_count": switch_count,
        },
    )


def v6i1_intervention(context: GateContext) -> GateResult:
    latent_state = context.trainer.latent_state
    margin = float(context.cfg.latent_cf_jsd_margin)
    valid_updates = int(getattr(latent_state, "pairwise_ema_valid_updates", 0) or 0)
    if valid_updates <= 0:
        return GateResult(
            status=GATE_STATUS_NOT_RUN,
            reason="no_pairwise_profile_ema_updates",
            details={
                "pairwise_ema_valid_updates": valid_updates,
                "jsd_margin": margin,
            },
        )
    pair_jsd = latent_state.pair_jsd_ema.tolist()
    num_valid = sum(1 for jsd in pair_jsd if float(jsd) >= margin)
    min_jsd = float(min(pair_jsd)) if pair_jsd else 0.0
    required_consecutive = int(getattr(context.cfg, "latent_cf_gate_consecutive_updates", 5))
    update_ok = num_valid >= 5 and min_jsd >= 0.5 * margin
    passed = update_ok and int(latent_state.jsd_gate_consecutive_updates) >= required_consecutive
    pair_details = {
        f"pair_jsd_ema_{idx}": float(pair_jsd[idx]) if idx < len(pair_jsd) else 0.0
        for idx, _pair in enumerate(LATENT_PAIR_INDEX)
    }
    pair_identity = {
        f"pair_{idx}_z{pair[0]}_z{pair[1]}": pair_details[f"pair_jsd_ema_{idx}"]
        for idx, pair in enumerate(LATENT_PAIR_INDEX)
    }
    return gate_family_result_from_bool(
        passed,
        details={
            "pair_jsd_ema": pair_jsd,
            "pair_order": [list(pair) for pair in LATENT_PAIR_INDEX],
            **pair_details,
            **pair_identity,
            "pairwise_ema_valid_updates": valid_updates,
            "jsd_margin": margin,
            "num_pairs_above_margin": int(num_valid),
            "min_pair_jsd_ema": min_jsd,
            "min_pair_floor": 0.5 * margin,
            "single_update_ok": bool(update_ok),
            "jsd_consecutive_updates": int(latent_state.jsd_gate_consecutive_updates),
            "latent_cf_gate_consecutive_updates": required_consecutive,
        },
    )


def v6i2_actor_intervention(context: GateContext) -> GateResult:
    return _gate_eval_to_result(
        evaluate_actor_intervention(context.cfg, context.trainer.latent_state)
    )


__all__ = [
    "evaluate_competence",
    "evaluate_coverage",
    "evaluate_training_integrity",
    "v6i1_intervention",
    "v6i2_actor_intervention",
]
