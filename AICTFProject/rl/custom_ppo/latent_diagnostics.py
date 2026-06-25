from __future__ import annotations

import csv
import math
import os
from typing import Any, Callable

import numpy as np
import torch

from rl.behavior_telemetry import (
    BEHAVIOR_TELEMETRY_NAMES,
    N_ATTACK_DEFENSE_RATIO_BUCKET,
    N_ROLE_BUCKET_MI,
    N_TELEMETRY,
)
from rl.custom_ppo.csv_writers import (
    SCRIPTED_OPPONENT_MI_COUNT,
    V3I3_REFRESH_REASON_LABELS,
    _ensure_additive_csv_header,
    _strategy_experience_fieldnames,
    _v3i3_refresh_log_fieldnames,
)
from rl.custom_ppo.inference import FORCED_Z_MACRO_ACTIONS, FORCED_Z_PROFILE_MAX_ROWS
from rl.forced_z_behavior_vectors import (
    build_behavior_distance_profile,
    behavior_vector_from_macro_probs,
)
from rl.discrete_mi import discrete_mi_plugin
from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_phase_labels import TEAM_PHASES


# ---------------------------------------------------------------------------
# Internal helpers shared by the rollout diagnostics below.
#
# These exist because the same three patterns appear ~8-10 times each inside
# `_latent_opponent_rollout_diag`:
#   1. pull an optional integer/float field out of the rollout buffer,
#   2. compute plug-in MI(z; x) for some categorical x,
#   3. write P(z=k | bucket=b) for every (b, k).
# Centralising them keeps the diagnostic readable as a checklist and replaces
# the Python-level joint-histogram loops with a single vectorised bincount.
# ---------------------------------------------------------------------------


def _flat_long_np(buffer: Any, name: str, length: int) -> np.ndarray | None:
    """Return ``buffer.fields[name][:length]`` flattened to int64 numpy, or None."""
    if name not in buffer.fields:
        return None
    return buffer.fields[name][:length].reshape(-1).long().cpu().numpy()


def _flat_float_np(buffer: Any, name: str, length: int) -> np.ndarray | None:
    """Return ``buffer.fields[name][:length]`` flattened to float32 numpy, or None."""
    if name not in buffer.fields:
        return None
    return buffer.fields[name][:length].reshape(-1).float().cpu().numpy()


def _mi_z_vs(z: np.ndarray, K: int, x: np.ndarray | None, n_x: int) -> float:
    """Plug-in MI(z; x) in nats. Returns 0.0 when ``x`` is missing or empty."""
    if x is None:
        return 0.0
    valid = (z >= 0) & (z < K) & (x >= 0) & (x < n_x)
    if not bool(valid.any()):
        return 0.0
    idx = z[valid].astype(np.int64) * n_x + x[valid].astype(np.int64)
    joint = np.bincount(idx, minlength=K * n_x).reshape(K, n_x).astype(np.float64)
    return float(discrete_mi_plugin(joint))


def _bucket_z_fracs(
    out: dict[str, float],
    z: np.ndarray,
    K: int,
    bucket: np.ndarray,
    n_buckets: int,
    key: Callable[[int, int], str],
) -> None:
    """Write ``out[key(b, k)] = P(z=k | bucket=b)`` for every (b, k); zeros when empty."""
    for b in range(n_buckets):
        mask = bucket == b
        if bool(mask.any()):
            z_sub = np.clip(z[mask], 0, K - 1)
            for k in range(K):
                out[key(b, k)] = float((z_sub == k).mean())
        else:
            for k in range(K):
                out[key(b, k)] = 0.0


def _fill_zero_z_fracs(
    out: dict[str, float], K: int, n_buckets: int, key: Callable[[int, int], str]
) -> None:
    """Default branch for ``_bucket_z_fracs`` when the bucket field is absent."""
    for b in range(n_buckets):
        for k in range(K):
            out[key(b, k)] = 0.0


def _shannon_entropy_nats(arr: np.ndarray | None, num_categories: int) -> float:
    """Plug-in Shannon entropy in nats. Returns 0.0 when ``arr`` is missing/empty."""
    if arr is None or arr.size == 0:
        return 0.0
    counts = np.bincount(arr, minlength=num_categories).astype(np.float64)
    total = counts.sum()
    if total <= 0.0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


# ---------------------------------------------------------------------------
# Public diagnostics. The function names below are imported by the trainer
# (``rl.custom_ppo.ppo_updater``) and by ``tests/test_latent_strategy_alignment``;
# their signatures, return-dict keys, and numeric outputs are part of the
# trainer's CSV contract and must not change.
# ---------------------------------------------------------------------------


def _latent_rollout_stats(trainer: Any, buffer: Any) -> dict[str, float]:
    """Summarize strategy occupancy and switching for the latest rollout.

    Existing keys (CSV-pinned, do not rename or remove):

    * ``strategy_unique_count``, ``strategy_dominant``,
      ``strategy_switch_count``, ``strategy_switch_fraction``,
      ``strategy_resample_count``, ``strategy_resample_fraction_rollout``,
      ``strategy_occupancy_{0..K-1}``.

    Added for v5i5 (occupancy-collapse diagnostics requested with the
    entropy-floor experiment; no new gradient channel, no new objective):

    * ``effective_num_latents`` -- ``exp(H)`` of the **sampled-z**
      rollout-marginal distribution (one categorical sample per state,
      ``argmax``-style). Equals ``K`` for uniform sampled occupancy;
      equals 1 for full collapse.
    * ``latent_marginal_entropy_nats`` -- the underlying ``H`` of the
      **sampled-z empirical histogram** (NOT ``H(E_s[q_phi(z|s)])``;
      see ``router_rollout_soft_marginal_entropy_nats`` for the soft
      analogue computed from differentiable router probabilities).
    * ``latent_occupancy_min`` / ``latent_occupancy_max`` --
      **sampled-z** per-z population fractions; one categorical sample
      per state. Distinct from ``router_rollout_soft_argmax_occupancy_*``
      which is computed from ``argmax_z q_phi(z|s)`` over the same
      states (no sampling noise but still a hard decision; both labels
      are kept for cross-checking).
    * ``latent_occupancy_ratio`` -- ``max / max(min, 1e-8)`` of the
      sampled-z occupancy. Large = severe sampled imbalance; ``1.0`` =
      perfect uniform sampling.
    * ``mean_strategy_duration`` -- mean dwell length in decision steps
      between latent switches. Computed as
      ``num_decision_steps / max(1, num_arc_boundaries)`` where
      ``num_arc_boundaries = strategy_resample_count``. Stays a
      diagnostic; not used in any loss.

    All five v5i5 diagnostics above are derived from one-sample-per-state
    empirical histograms over the categorical samples ``z_t`` written to
    the rollout buffer. The soft-router analogues (``H(\bar q)``,
    ``mean_i H(q_i)``, MI proxy, soft-argmax occupancy, per-z ``\bar q``)
    live in the ``router_rollout_soft_*`` columns and are emitted by
    ``rl/latent_losses.py::rollout_router_soft_diagnostics`` from the
    same population the v5i6 marginal-entropy loss is taken over.
    """
    if not trainer.use_latent_strategy or "z" not in buffer.fields:
        return {}
    length = int(buffer.pos)
    z = buffer.fields["z"][:length].reshape(-1).long()
    prev_z = buffer.fields["prev_z"][:length].reshape(-1).long()
    if z.numel() == 0:
        return {}
    counts = torch.bincount(z.clamp(min=0, max=trainer.latent_k - 1), minlength=trainer.latent_k).float()
    occupancy = counts / counts.sum().clamp_min(1.0)
    persist_mask = buffer.fields["z_persist_mask"][:length].reshape(-1).bool()
    resample_field = "z_resampled_actual" if "z_resampled_actual" in buffer.fields else "z_resampled"
    resampled = buffer.fields[resample_field][:length].reshape(-1).bool()
    switched = persist_mask & (z != prev_z)
    resample_count = float(resampled.sum().detach().cpu().item())
    persistence_valid = (
        buffer.fields["persistence_valid"][:length].reshape(-1).bool()
        if "persistence_valid" in buffer.fields
        else persist_mask
    )
    out = {
        "strategy_unique_count": float((counts > 0).sum().detach().cpu().item()),
        "strategy_dominant": float(torch.argmax(counts).detach().cpu().item()),
        "strategy_switch_count": float(switched.sum().detach().cpu().item()),
        "strategy_switch_fraction": float(
            (switched.float().sum() / persist_mask.float().sum().clamp_min(1.0)).detach().cpu().item()
        ),
        "strategy_resample_count": resample_count,
        "z_resampled_actual": resample_count,
        "router_opportunity_count": resample_count,
        "persistence_valid_pair_count": float(persistence_valid.sum().detach().cpu().item()),
        "strategy_resample_fraction_rollout": float(resampled.float().mean().detach().cpu().item()),
    }
    for idx, value in enumerate(occupancy.detach().cpu().tolist()):
        out[f"strategy_occupancy_{idx}"] = float(value)

    # v5i5 occupancy diagnostics. Pure functions of the existing per-z
    # counts, no new tensors / autograd edges. Values use natural log so
    # ``effective_num_latents`` is in [1, K].
    occ = occupancy.detach().cpu()
    occ_clamped = occ.clamp_min(1e-12)
    marginal_entropy = float((-(occ_clamped * occ_clamped.log()).sum()).item())
    occ_list = [float(v) for v in occ.tolist()]
    occ_min = float(min(occ_list)) if occ_list else 0.0
    occ_max = float(max(occ_list)) if occ_list else 0.0
    out["latent_marginal_entropy_nats"] = marginal_entropy
    out["effective_num_latents"] = float(math.exp(marginal_entropy))
    out["latent_occupancy_min"] = occ_min
    out["latent_occupancy_max"] = occ_max
    out["latent_occupancy_ratio"] = float(occ_max / max(occ_min, 1e-8))
    # Mean decision-steps per latent arc. Dividing by ``resample_count``
    # treats every resample (episode start or sparse refresh) as an arc
    # boundary, so this is a stable stand-in for "mean strategy
    # duration" without needing a separate accumulator.
    total_decisions = float(z.numel())
    arc_boundaries = max(1.0, resample_count)
    out["mean_strategy_duration"] = total_decisions / arc_boundaries
    return out


def _q_phi_probs_and_entropy(
    buffer: Any, length: int, K: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Softmax probs and per-step entropy of stored q_phi logits, or (None, None)."""
    if "z_logits" not in buffer.fields:
        return None, None
    logits = buffer.fields["z_logits"][:length].reshape(-1, K).float()
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return probs.detach().cpu().numpy(), entropy.detach().cpu().numpy()


def _flag_state_per_step(buffer: Any, length: int) -> np.ndarray:
    """Encode (blue_has_red_flag, red_has_blue_flag) as a 4-valued bucket per step."""
    gs_all = buffer.fields["global_state"][:length].cpu().numpy()
    gs_flat = gs_all.reshape(-1, gs_all.shape[-1])
    blue_cap = (gs_flat[:, 10] > 0.5).astype(np.int64)
    red_cap = (gs_flat[:, 11] > 0.5).astype(np.int64)
    return blue_cap + 2 * red_cap


def _phase_block(
    out: dict[str, float],
    z: np.ndarray,
    K: int,
    sw: np.ndarray,
    ba: np.ndarray | None,
    rsp_bin: np.ndarray | None,
    pid: np.ndarray,
    q_probs: np.ndarray | None,
    q_entropy: np.ndarray | None,
) -> None:
    """Per-phase z fractions, switch / blue-ahead / capture means, and q_phi means."""
    for p in range(len(TEAM_PHASES)):
        mask = pid == p
        if not bool(mask.any()):
            for k in range(K):
                out[f"latent_phase{p}_z{k}_frac"] = 0.0
            out[f"latent_phase{p}_switch_mean"] = 0.0
            out[f"latent_phase{p}_blue_ahead_mean"] = 0.0
            out[f"latent_phase{p}_capture_step_mean"] = 0.0
            out[f"q_phi_phase{p}_entropy_mean"] = 0.0
            for k in range(K):
                out[f"q_phi_phase{p}_z{k}_prob_mean"] = 0.0
            continue

        z_sub = np.clip(z[mask], 0, K - 1)
        for k in range(K):
            out[f"latent_phase{p}_z{k}_frac"] = float((z_sub == k).mean())
        out[f"latent_phase{p}_switch_mean"] = float(sw[mask].mean())
        out[f"latent_phase{p}_blue_ahead_mean"] = float(ba[mask].mean()) if ba is not None else 0.0
        out[f"latent_phase{p}_capture_step_mean"] = (
            float(rsp_bin[mask].mean()) if rsp_bin is not None else 0.0
        )

        if q_probs is None or q_entropy is None:
            out[f"q_phi_phase{p}_entropy_mean"] = 0.0
            for k in range(K):
                out[f"q_phi_phase{p}_z{k}_prob_mean"] = 0.0
        else:
            out[f"q_phi_phase{p}_entropy_mean"] = float(q_entropy[mask].mean())
            q_phase = q_probs[mask]
            for k in range(K):
                out[f"q_phi_phase{p}_z{k}_prob_mean"] = float(q_phase[:, k].mean())


def _reward_sum_after_switch_5(out: dict[str, float], buffer: Any, length: int) -> None:
    """Mean of the next-5-step reward sum following a z-switch (excludes terminal switches)."""
    rewards = buffer.fields["rewards"][:length].detach().cpu().numpy()
    z_tb = buffer.fields["z"][:length].detach().cpu().numpy()
    pz_tb = buffer.fields["prev_z"][:length].detach().cpu().numpy()
    Tn, Bn = int(z_tb.shape[0]), int(z_tb.shape[1])
    sums: list[float] = []
    for t in range(Tn):
        h = min(5, Tn - 1 - t)
        if h <= 0:
            continue
        for b in range(Bn):
            if int(z_tb[t, b]) != int(pz_tb[t, b]):
                sums.append(float(rewards[t + 1 : t + 1 + h, b].sum()))
    out["latent_reward_sum_5_after_z_switch_mean"] = float(np.mean(sums)) if sums else 0.0


def _flag_return_indices(
    blue_cap_col: np.ndarray, red_cap_col: np.ndarray, abs_rsp_col: np.ndarray
) -> np.ndarray:
    """Time indices where a flag was returned (carrier dropped it) without scoring."""
    if blue_cap_col.shape[0] < 2:
        return np.empty(0, dtype=np.int64)
    # blue_cap is bool; for the time-stepped diff we need integer logic against ~.
    blue_ret = blue_cap_col[:-1] & ~blue_cap_col[1:]
    red_ret = red_cap_col[:-1] & ~red_cap_col[1:]
    no_score = abs_rsp_col[1:] < 1.0
    hit = (blue_ret | red_ret) & no_score
    return np.where(hit)[0] + 1  # +1 because index t in original referred to the second step.


def _switch_proximity_fracs(out: dict[str, float], buffer: Any, length: int) -> None:
    """Fraction of z-switches within 3 steps of a capture / kill / flag-return event.

    Also surfaces the raw counts so a zero fraction is interpretable:

    * ``latent_switch_near_eligible_count``: # of mid-episode z-switches in
      the rollout (the fraction's denominator). Zero under presets that
      only resample at episode start (e.g. v5_strict_summer, v5i1, v5i2,
      v5i3) AND have event-refresh / sparse-tactical-refresh disabled,
      in which case the cap/kill/ret fractions are not meaningful.
    * ``latent_capture_event_count`` / ``..kill_event_count`` /
      ``..return_event_count``: # of qualifying events in the rollout.
      A zero fraction with eligible_count > 0 and event_count > 0 is a
      real null result; a zero fraction with either count == 0 is a
      missing-data artefact, not evidence of "switches do not align".
    * ``latent_switch_near_capture_count`` / ``..kill_count`` /
      ``..return_count``: the numerators (number of switches that
      landed within 3 steps of the corresponding event).
    """
    rsp = buffer.fields["reward_sparse_points"][:length].cpu().numpy()
    z_env = buffer.fields["z"][:length].cpu().numpy()
    pz_env = buffer.fields["prev_z"][:length].cpu().numpy()
    persist_env = buffer.fields["z_persist_mask"][:length].cpu().numpy()
    sw_env = persist_env & (z_env != pz_env)
    total = float(sw_env.sum())

    gs_env = buffer.fields["global_state"][:length].cpu().numpy()
    blue_cap_env = gs_env[:, :, 10] > 0.5
    red_cap_env = gs_env[:, :, 11] > 0.5

    capture_event_count = 0
    kill_event_count = 0
    return_event_count = 0
    near_capture = near_kill = near_return = 0.0

    for b in range(int(buffer.n_envs)):
        abs_rsp = np.abs(rsp[:, b])
        capture_idx = np.where(abs_rsp > 50.0)[0]
        kill_idx = np.where((abs_rsp > 1.0) & (abs_rsp < 40.0))[0]
        return_idx = _flag_return_indices(blue_cap_env[:, b], red_cap_env[:, b], abs_rsp)
        capture_event_count += int(capture_idx.size)
        kill_event_count += int(kill_idx.size)
        return_event_count += int(return_idx.size)

        switch_idx = np.where(sw_env[:, b])[0]
        if switch_idx.size == 0:
            continue
        for idx in switch_idx:
            if capture_idx.size and int(np.min(np.abs(capture_idx - idx))) <= 3:
                near_capture += 1.0
            if kill_idx.size and int(np.min(np.abs(kill_idx - idx))) <= 3:
                near_kill += 1.0
            if return_idx.size and int(np.min(np.abs(return_idx - idx))) <= 3:
                near_return += 1.0

    if total > 0.0:
        out["latent_switch_near_capture_frac"] = near_capture / total
        out["latent_switch_near_kill_frac"] = near_kill / total
        out["latent_switch_near_return_frac"] = near_return / total
    else:
        out["latent_switch_near_capture_frac"] = 0.0
        out["latent_switch_near_kill_frac"] = 0.0
        out["latent_switch_near_return_frac"] = 0.0

    out["latent_switch_near_eligible_count"] = total
    out["latent_switch_near_capture_count"] = near_capture
    out["latent_switch_near_kill_count"] = near_kill
    out["latent_switch_near_return_count"] = near_return
    out["latent_capture_event_count"] = float(capture_event_count)
    out["latent_kill_event_count"] = float(kill_event_count)
    out["latent_return_event_count"] = float(return_event_count)


def _latent_opponent_rollout_diag(trainer: Any, buffer: Any) -> dict[str, float]:
    """Per-opponent z occupancy plus MI(z; opponent/phase/outcome) and phase / behavior bucket rollups."""
    if not trainer.use_latent_strategy or "z" not in buffer.fields:
        return {}
    length = int(buffer.pos)
    if length <= 0:
        return {}

    K = int(trainer.latent_k)
    z = _flat_long_np(buffer, "z", length)
    prev_z = _flat_long_np(buffer, "prev_z", length)
    assert z is not None and prev_z is not None  # guarded by the "z in fields" check above
    sw = (z != prev_z).astype(np.float64)
    out: dict[str, float] = {}

    q_probs, q_entropy = _q_phi_probs_and_entropy(buffer, length, K)

    # --- MI(z; categorical context fields) plus per-opponent occupancy --------
    oid = _flat_long_np(buffer, "opponent_id", length)
    if oid is not None:
        out["latent_mi_z_opponent_nats"] = _mi_z_vs(z, K, oid, SCRIPTED_OPPONENT_MI_COUNT)
        _bucket_z_fracs(out, z, K, oid, SCRIPTED_OPPONENT_MI_COUNT,
                        lambda o, k: f"strategy_occupancy_op{o}_z{k}")
    else:
        out["latent_mi_z_opponent_nats"] = 0.0

    pid = _flat_long_np(buffer, "phase_id", length)
    out["latent_mi_z_phase_nats"] = _mi_z_vs(z, K, pid, len(TEAM_PHASES))
    out["MI_executed_z_phase"] = out["latent_mi_z_phase_nats"]

    yid = _flat_long_np(buffer, "outcome_id", length)
    out["latent_mi_z_outcome_nats"] = _mi_z_vs(z, K, yid, 3)
    out["MI_executed_z_outcome"] = out["latent_mi_z_outcome_nats"]

    # --- Phase-conditioned rollups + ahead/trail switch rates ----------------
    ba = _flat_float_np(buffer, "blue_ahead", length)
    rsp_flat = _flat_float_np(buffer, "reward_sparse_points", length)
    rsp_bin = (np.abs(rsp_flat) > 1e-5).astype(np.float64) if rsp_flat is not None else None

    if pid is not None:
        _phase_block(out, z, K, sw, ba, rsp_bin, pid, q_probs, q_entropy)

    if ba is not None:
        ahead = ba > 0.5
        trail = ~ahead
        out["latent_switch_rate_blue_ahead"] = float(sw[ahead].mean()) if bool(ahead.any()) else 0.0
        out["latent_switch_rate_blue_trail"] = float(sw[trail].mean()) if bool(trail.any()) else 0.0
    else:
        out["latent_switch_rate_blue_ahead"] = 0.0
        out["latent_switch_rate_blue_trail"] = 0.0

    _reward_sum_after_switch_5(out, buffer, length)

    # --- MI(z; situation bucket) for spread / role / pressure / ADR ----------
    n_sb, n_rb, n_pb = 3, int(N_ROLE_BUCKET_MI), 3
    n_adr = int(N_ATTACK_DEFENSE_RATIO_BUCKET)

    sb = _flat_long_np(buffer, "spread_bucket_id", length)
    out["latent_mi_z_spread_bucket_nats"] = _mi_z_vs(z, K, sb, n_sb)

    rb = _flat_long_np(buffer, "role_bucket_id", length)
    if rb is not None:
        out["latent_mi_z_role_bucket_nats"] = _mi_z_vs(z, K, rb, n_rb)
        _bucket_z_fracs(out, z, K, rb, n_rb, lambda r, k: f"latent_role{r}_z{k}_frac")
        for r in range(n_rb):
            mask = rb == r
            out[f"latent_role{r}_switch_mean"] = float(sw[mask].mean()) if bool(mask.any()) else 0.0
    else:
        out["latent_mi_z_role_bucket_nats"] = 0.0
        _fill_zero_z_fracs(out, K, n_rb, lambda r, k: f"latent_role{r}_z{k}_frac")
        for r in range(n_rb):
            out[f"latent_role{r}_switch_mean"] = 0.0

    pb = _flat_long_np(buffer, "pressure_bucket_id", length)
    out["latent_mi_z_pressure_bucket_nats"] = _mi_z_vs(z, K, pb, n_pb)

    adb = _flat_long_np(buffer, "attack_defense_ratio_bucket_id", length)
    out["latent_mi_z_attack_defense_ratio_bucket_nats"] = _mi_z_vs(z, K, adb, n_adr)

    # --- Flag-state derived from global_state --------------------------------
    flag_state = _flag_state_per_step(buffer, length)
    out["latent_mi_z_flag_state_nats"] = _mi_z_vs(z, K, flag_state, 4)
    out["MI_executed_z_flag"] = out["latent_mi_z_flag_state_nats"]
    _bucket_z_fracs(out, z, K, flag_state, 4, lambda f, k: f"latent_flag_state{f}_z{k}_frac")

    # --- Per-bucket occupancy distributions (spread, ADR) --------------------
    if sb is not None:
        _bucket_z_fracs(out, z, K, sb, 3, lambda s, k: f"latent_spread{s}_z{k}_frac")
    else:
        _fill_zero_z_fracs(out, K, 3, lambda s, k: f"latent_spread{s}_z{k}_frac")

    if adb is not None:
        _bucket_z_fracs(out, z, K, adb, 3, lambda a, k: f"latent_adr{a}_z{k}_frac")
    else:
        _fill_zero_z_fracs(out, K, 3, lambda a, k: f"latent_adr{a}_z{k}_frac")

    # --- Per-phase Shannon entropy over z ------------------------------------
    for p in range(len(TEAM_PHASES)):
        if pid is None:
            out[f"latent_phase{p}_entropy"] = 0.0
            continue
        mask = pid == p
        if bool(mask.any()):
            out[f"latent_phase{p}_entropy"] = _shannon_entropy_nats(
                np.clip(z[mask], 0, K - 1), K
            )
        else:
            out[f"latent_phase{p}_entropy"] = 0.0

    # --- Overall diversity (Shannon entropy of the bucket distribution) ------
    out["latent_role_diversity"] = _shannon_entropy_nats(rb, n_rb)
    out["latent_spread_diversity"] = _shannon_entropy_nats(sb, n_sb)
    out["latent_pressure_diversity"] = _shannon_entropy_nats(pb, n_pb)
    out["latent_adr_diversity"] = _shannon_entropy_nats(adb, n_adr)

    # Normalized MI = I(z; x) / H(z). Stays plan-faithful: just a ratio of
    # already-computed plug-in quantities, no training-time supervision. H(z) is
    # computed from valid z entries (z in [0, K)); the same filter used by MI.
    z_valid = z[(z >= 0) & (z < K)] if z is not None else None
    h_z = _shannon_entropy_nats(z_valid.astype(np.int64), K) if z_valid is not None else 0.0
    if h_z > 1e-12:
        out["latent_normalized_mi_z_opponent"] = float(out["latent_mi_z_opponent_nats"] / h_z)
        out["latent_normalized_mi_z_phase"] = float(out["latent_mi_z_phase_nats"] / h_z)
        out["latent_normalized_mi_z_outcome"] = float(out["latent_mi_z_outcome_nats"] / h_z)
        out["latent_normalized_mi_z_flag_state"] = float(out["latent_mi_z_flag_state_nats"] / h_z)
    else:
        out["latent_normalized_mi_z_opponent"] = 0.0
        out["latent_normalized_mi_z_phase"] = 0.0
        out["latent_normalized_mi_z_outcome"] = 0.0
        out["latent_normalized_mi_z_flag_state"] = 0.0
    out["latent_z_marginal_entropy_nats"] = float(h_z)

    _switch_proximity_fracs(out, buffer, length)

    return out


def _behavior_diversity_stats(trainer: Any, buffer: Any) -> dict[str, float]:
    """Post-hoc behavior spread by sampled z; diagnostics only, no labels or losses."""
    if not trainer.use_latent_strategy or "z" not in buffer.fields or "behavior_telemetry" not in buffer.fields:
        return {}
    length = int(buffer.pos)
    if length <= 0:
        return {}
    z = buffer.fields["z"][:length].reshape(-1).long()
    beh = buffer.fields["behavior_telemetry"][:length].reshape(-1, N_TELEMETRY).float()
    out: dict[str, float] = {}
    means: list[torch.Tensor] = []
    for k in range(int(trainer.latent_k)):
        mask = z == k
        if bool(mask.any().item()):
            mean_k = beh[mask].mean(dim=0)
            means.append(mean_k)
        else:
            mean_k = torch.zeros((N_TELEMETRY,), dtype=torch.float32, device=beh.device)
        for j, name in enumerate(BEHAVIOR_TELEMETRY_NAMES):
            out[f"latent_z{k}_behavior_{name}_mean"] = float(mean_k[j].detach().cpu().item())

    if len(means) >= 2:
        pairwise: list[float] = []
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                pairwise.append(float(torch.linalg.vector_norm(means[i] - means[j]).detach().cpu().item()))
        out["latent_behavior_diversity_l2_mean"] = float(np.mean(pairwise)) if pairwise else 0.0
    else:
        out["latent_behavior_diversity_l2_mean"] = 0.0
    return out


def _macro_probs_from_logits(trainer: Any, logits: torch.Tensor) -> torch.Tensor:
    """Return macro-action probabilities with shape (B, n_agents, macro_dim)."""
    macro_chunks: list[torch.Tensor] = []
    offset = 0
    for _agent_idx in range(int(trainer.model.n_agents)):
        for head_idx in range(int(trainer.model.heads_per_agent)):
            dim = int(trainer.model.per_agent_action_dims[head_idx])
            chunk = logits[:, offset : offset + dim]
            if head_idx == 0:
                macro_chunks.append(torch.softmax(chunk, dim=-1))
            offset += dim
    if not macro_chunks:
        raise AssertionError("could not find macro-action heads for forced-z profiling")
    return torch.stack(macro_chunks, dim=1)


def _batched_policy_trunk_features(
    trainer: Any, obs_batch: dict[str, torch.Tensor], z_idx: torch.Tensor
) -> torch.Tensor:
    """Batched ``policy_trunk_features`` with the same chunking as policy logits."""
    total = z_idx.shape[0]
    batch_size = min(1024, int(getattr(trainer.cfg, "batch_size", 1024)))
    if total <= batch_size:
        return trainer.model.policy_trunk_features(obs_batch, z_idx=z_idx)
    chunks: list[torch.Tensor] = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        slice_obs = {k: v[start:end] for k, v in obs_batch.items()}
        slice_z = z_idx[start:end]
        chunks.append(trainer.model.policy_trunk_features(slice_obs, z_idx=slice_z))
    return torch.cat(chunks, dim=0)


def _batched_policy_logits(
    trainer: Any, obs_batch: dict[str, torch.Tensor], z_idx: torch.Tensor
) -> torch.Tensor:
    """Run model.policy_logits in smaller mini-batches to prevent CUDA OOM on large state spaces (e.g. 4v4)."""
    total = z_idx.shape[0]
    batch_size = min(1024, int(getattr(trainer.cfg, "batch_size", 1024)))
    if total <= batch_size:
        return trainer.model.policy_logits(obs_batch, z_idx=z_idx)
    logits_list = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        slice_obs = {k: v[start:end] for k, v in obs_batch.items()}
        slice_z = z_idx[start:end]
        slice_logits = trainer.model.policy_logits(slice_obs, z_idx=slice_z)
        logits_list.append(slice_logits)
    return torch.cat(logits_list, dim=0)


def _forced_z_behavior_profile(trainer: Any, buffer: Any) -> dict[str, float]:
    """Profile actor macro preferences under every forced z on the same rollout observations."""
    if not trainer.use_latent_strategy:
        return {}
    length = int(buffer.pos)
    if length <= 0:
        return {}
    total = length * int(buffer.n_envs)
    if total <= 0:
        return {}
    if total > FORCED_Z_PROFILE_MAX_ROWS:
        row_idx = torch.linspace(
            0,
            total - 1,
            steps=FORCED_Z_PROFILE_MAX_ROWS,
            device=trainer.device,
        ).long()
    else:
        row_idx = torch.arange(total, device=trainer.device)
    row_idx = torch.clamp(row_idx, 0, total - 1)
    obs_batch = {
        "grid": buffer.fields["obs_grid"][:length].reshape(total, *buffer.fields["obs_grid"].shape[2:]).index_select(0, row_idx),
        "vec": buffer.fields["obs_vec"][:length].reshape(total, *buffer.fields["obs_vec"].shape[2:]).index_select(0, row_idx),
        "agent_mask": buffer.fields["obs_agent_mask"][:length].reshape(total, *buffer.fields["obs_agent_mask"].shape[2:]).index_select(0, row_idx),
        "mask": buffer.fields["obs_mask"][:length].reshape(total, *buffer.fields["obs_mask"].shape[2:]).index_select(0, row_idx),
    }
    out: dict[str, float] = {}
    mean_macros: list[torch.Tensor] = []
    behavior_vectors: list[np.ndarray] = []
    with torch.no_grad():
        for z_id in range(int(trainer.latent_k)):
            z_idx = torch.full((int(row_idx.numel()),), z_id, dtype=torch.long, device=trainer.device)
            logits = _batched_policy_logits(trainer, obs_batch, z_idx=z_idx)
            logits = trainer.model._mask_logits(logits, obs_batch.get("mask"))
            macro_probs = _macro_probs_from_logits(trainer, logits)
            mean_macro = macro_probs.mean(dim=(0, 1))
            mean_macros.append(mean_macro)
            macro_entropy = -(
                macro_probs.clamp_min(1e-8) * macro_probs.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
            for action_id, action_name in FORCED_Z_MACRO_ACTIONS:
                if action_id < int(mean_macro.numel()):
                    out[f"forced_z{z_id}_macro_{action_name}_prob"] = float(mean_macro[action_id].detach().cpu().item())
                else:
                    out[f"forced_z{z_id}_macro_{action_name}_prob"] = 0.0
            out[f"forced_z{z_id}_macro_entropy"] = float(macro_entropy.detach().cpu().item())
            behavior_vectors.append(behavior_vector_from_macro_probs(mean_macro))

    out.update(
        build_behavior_distance_profile(
            behavior_vectors,
            source="macro",
            pair_count=int(trainer.latent_k) * (int(trainer.latent_k) - 1) // 2,
            latent_k=int(trainer.latent_k),
        )
    )

    if len(mean_macros) >= 2:
        js_vals: list[float] = []
        pair_idx = 0
        for i in range(len(mean_macros)):
            for j in range(i + 1, len(mean_macros)):
                p = mean_macros[i].clamp_min(1e-8)
                q = mean_macros[j].clamp_min(1e-8)
                m = 0.5 * (p + q)
                js = 0.5 * (p * (p.log() - m.log())).sum() + 0.5 * (q * (q.log() - m.log())).sum()
                js_val = float(js.detach().cpu().item())
                js_vals.append(js_val)
                out[f"forced_z_pair_jsd_{pair_idx}"] = js_val
                pair_idx += 1
        out["forced_z_macro_jsd_mean"] = float(np.mean(js_vals)) if js_vals else 0.0
    else:
        out["forced_z_macro_jsd_mean"] = 0.0
    out["forced_z_macro_jsd"] = out["forced_z_macro_jsd_mean"]
    return out


def _strategy_resample_advantage_stats(trainer: Any, buffer: Any) -> dict[str, float]:
    """Per-z mean/std of raw GAE advantages at z-resample steps (pre-minibatch normalization)."""
    if not trainer.use_latent_strategy or trainer.fixed_latent_strategy:
        return {}
    length = int(buffer.pos)
    if length <= 0 or "advantages" not in buffer.fields or "z" not in buffer.fields:
        return {}
    adv = buffer.fields["advantages"][:length]
    z = buffer.fields["z"][:length].long()
    rs = buffer.fields["z_resampled"][:length].bool()
    flat_adv = adv.reshape(-1).float()
    flat_z = z.reshape(-1)
    flat_rs = rs.reshape(-1)
    out: dict[str, float] = {}
    K = int(trainer.latent_k)
    for k in range(K):
        m = flat_rs & (flat_z == k)
        n = int(m.sum().item())
        out[f"strategy_resample_adv_n_z{k}"] = float(n)
        if n > 0:
            vals = flat_adv[m]
            out[f"strategy_resample_adv_mean_z{k}"] = float(vals.mean().item())
            out[f"strategy_resample_adv_std_z{k}"] = (
                float(vals.std(unbiased=False).item()) if n > 1 else 0.0
            )
        else:
            out[f"strategy_resample_adv_mean_z{k}"] = 0.0
            out[f"strategy_resample_adv_std_z{k}"] = 0.0
    return out


def _rollout_advantage_diagnostics(trainer: Any, buffer: Any) -> dict[str, float]:
    """Raw GAE advantage scale and split at latent z-segment starts (t>0, z[t]!=z[t-1])."""
    length = int(buffer.pos)
    if length <= 0 or "advantages" not in buffer.fields:
        return {}
    adv = buffer.fields["advantages"][:length].detach().float()
    flat = adv.reshape(-1)
    out: dict[str, float] = {
        "rollout_adv_std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
    }
    if (
        not trainer.use_latent_strategy
        or trainer.fixed_latent_strategy
        or "z" not in buffer.fields
        or length < 2
    ):
        return out
    z = buffer.fields["z"][:length].long()
    z_switch = torch.zeros((length, z.shape[1]), dtype=torch.bool, device=z.device)
    z_switch[1:] = z[1:] != z[:-1]
    flat_sw = z_switch.reshape(-1)
    if flat_sw.any() and (~flat_sw).any():
        out["rollout_adv_std_at_z_switch"] = float(flat[flat_sw].std(unbiased=False).item())
        out["rollout_adv_std_not_z_switch"] = float(flat[~flat_sw].std(unbiased=False).item())
    else:
        out["rollout_adv_std_at_z_switch"] = float(out["rollout_adv_std"])
        out["rollout_adv_std_not_z_switch"] = float(out["rollout_adv_std"])
    return out


def _write_strategy_experience_table(trainer: Any) -> dict[str, float]:
    if not trainer.strategy_experience_csv_path or not trainer.use_latent_strategy or trainer.latent_k <= 0:
        return {"strategy_bucket_best_match_frac": 0.0, "strategy_experience_records": 0.0, "strategy_experience_buckets": 0.0}
    records = list(trainer.latent_state.rollout_strategy_episode_records)
    if not records:
        return {"strategy_bucket_best_match_frac": 0.0, "strategy_experience_records": 0.0, "strategy_experience_buckets": 0.0}

    by_bucket: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        by_bucket.setdefault(int(r["bucket_id"]), []).append(r)

    rows: list[dict[str, Any]] = []
    best_match = 0
    total = 0
    for bucket_id, bucket_records in sorted(by_bucket.items()):
        bucket_count = len(bucket_records)
        total += bucket_count
        returns_by_z: dict[int, list[float]] = {z: [] for z in range(trainer.latent_k)}
        wins_by_z: dict[int, list[int]] = {z: [] for z in range(trainer.latent_k)}
        for r in bucket_records:
            z = int(r["z"])
            if 0 <= z < trainer.latent_k:
                returns_by_z[z].append(float(r["episode_return"]))
                wins_by_z[z].append(int(r["episode_win"]))
        best_candidates = [
            (float(np.mean(vals)), z) for z, vals in returns_by_z.items() if vals
        ]
        best_z = max(best_candidates)[1] if best_candidates else -1
        if best_z >= 0:
            best_match += len(returns_by_z[best_z])
        best_z_match_frac = float(len(returns_by_z.get(best_z, []))) / float(max(1, bucket_count))
        for z in range(trainer.latent_k):
            z_returns = returns_by_z[z]
            z_wins = wins_by_z[z]
            prob_vals = [
                float(r["q_phi_probs"][z])
                for r in bucket_records
                if z < len(r.get("q_phi_probs", []))
            ]
            count = len(z_returns)
            rows.append(
                {
                    "update": int(trainer._updates_completed),
                    "run_id": trainer.run_id,
                    "run_pid": trainer.run_pid,
                    "timesteps": int(trainer.global_step),
                    "bucket_id": int(bucket_id),
                    "z": int(z),
                    "count": int(count),
                    "bucket_count": int(bucket_count),
                    "mean_return": "" if count <= 0 else float(np.mean(z_returns)),
                    "win_rate": "" if count <= 0 else float(np.mean(z_wins)),
                    "q_phi_prob_mean": float(np.mean(prob_vals)) if prob_vals else "",
                    "chosen_freq": float(count) / float(max(1, bucket_count)),
                    "best_z": int(best_z),
                    "best_z_match_frac": best_z_match_frac,
                }
            )

    fieldnames = _strategy_experience_fieldnames()
    path = trainer.strategy_experience_csv_path
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    _ensure_additive_csv_header(path, fieldnames)
    nonempty = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not nonempty:
            writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)
    return {
        "strategy_bucket_best_match_frac": float(best_match) / float(max(1, total)),
        "strategy_experience_records": float(total),
        "strategy_experience_buckets": float(len(by_bucket)),
    }


def _write_refresh_log_table(trainer: Any) -> dict[str, float]:
    """Write one CSV row per finalized v3i3 refresh event for this rollout.

    Schema (see ``csv_writers._v3i3_refresh_log_fieldnames``):
        update, run_id, run_pid, timesteps,
        env_id, episode_id, decision_step, reason_id, reason,
        prev_z, next_z, opponent_id, flag_state_bucket,
        return_at_refresh, return_from_now_to_end

    No-op when v3i3 logging is disabled, the path is empty, or no refresh
    records were finalized this rollout. Returns a stats dict the caller
    can merge into ``last_stats`` for surfacing in the per-update metrics.
    """
    if not bool(getattr(trainer, "latent_v3i3_refresh_log_enabled", False)):
        return {"latent_v3i3_refresh_log_rows": 0.0}
    path = str(getattr(trainer, "latent_v3i3_refresh_log_path", "") or "")
    if not path:
        return {"latent_v3i3_refresh_log_rows": 0.0}
    if not getattr(trainer, "use_latent_strategy", False):
        return {"latent_v3i3_refresh_log_rows": 0.0}
    latent_state = getattr(trainer, "latent_state", None)
    if latent_state is None:
        return {"latent_v3i3_refresh_log_rows": 0.0}
    records = list(getattr(latent_state, "rollout_refresh_records", []) or [])
    if not records:
        return {"latent_v3i3_refresh_log_rows": 0.0}
    fieldnames = _v3i3_refresh_log_fieldnames()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    _ensure_additive_csv_header(path, fieldnames)
    nonempty = os.path.isfile(path) and os.path.getsize(path) > 0
    rows: list[dict[str, Any]] = []
    update_idx = int(getattr(trainer, "_updates_completed", 0))
    timesteps = int(getattr(trainer, "global_step", 0))
    for rec in records:
        reason_id = int(rec.get("reason_id", -1))
        reason_label = (
            V3I3_REFRESH_REASON_LABELS[reason_id]
            if 0 <= reason_id < len(V3I3_REFRESH_REASON_LABELS)
            else "unknown"
        )
        rows.append(
            {
                "update": update_idx,
                "run_id": getattr(trainer, "run_id", ""),
                "run_pid": getattr(trainer, "run_pid", 0),
                "timesteps": timesteps,
                "env_id": int(rec.get("env_id", -1)),
                "episode_id": int(rec.get("episode_id", -1)),
                "decision_step": int(rec.get("decision_step", -1)),
                "reason_id": reason_id,
                "reason": reason_label,
                "prev_z": int(rec.get("prev_z", -1)),
                "next_z": int(rec.get("next_z", -1)),
                "opponent_id": int(rec.get("opponent_id", -1)),
                "flag_state_bucket": int(rec.get("flag_state_bucket", -1)),
                "carrier_progress_bucket": int(rec.get("carrier_progress_bucket", -1)),
                "return_at_refresh": float(rec.get("return_at_refresh", 0.0)),
                "return_from_now_to_end": float(
                    rec.get("return_from_now_to_end", rec.get("future_return", 0.0))
                ),
            }
        )
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not nonempty:
            writer.writeheader()
        writer.writerows({key: r.get(key, "") for key in fieldnames} for r in rows)
    return {"latent_v3i3_refresh_log_rows": float(len(rows))}


_ZERO_OPT_ADV: dict[str, float] = {
    "latent_q_phi_option_advantage_mean": 0.0,
    "latent_q_phi_option_advantage_std": 0.0,
    "latent_q_phi_option_advantage_count": 0.0,
}


def _latent_option_advantage_stats(trainer: Any, buffer: Any) -> dict[str, float]:
    """Calculate mean, std, and count of option advantages at resampled steps."""
    if not trainer.use_latent_strategy or trainer.fixed_latent_strategy:
        return dict(_ZERO_OPT_ADV)
    length = int(buffer.pos)
    if length <= 0 or "option_advantages" not in buffer.fields or "z_resampled" not in buffer.fields:
        return dict(_ZERO_OPT_ADV)

    opt_adv = buffer.fields["option_advantages"][:length].reshape(-1).float()
    rs = buffer.fields["z_resampled"][:length].reshape(-1).bool()
    vals = opt_adv[rs]
    count = int(vals.numel())
    if count == 0:
        return dict(_ZERO_OPT_ADV)
    return {
        "latent_q_phi_option_advantage_mean": float(vals.mean().item()),
        "latent_q_phi_option_advantage_std": (
            float(vals.std(unbiased=False).item()) if count > 1 else 0.0
        ),
        "latent_q_phi_option_advantage_count": float(count),
    }


def _policy_z_sensitivity_kl(trainer: Any, buffer: Any) -> dict[str, Any]:
    """Probe actor behavior under every forced z on the same observations."""
    zero_stats: dict[str, Any] = {
        "policy_z_sensitivity_KL": 0.0,
        "actor_z_jsd_mean": 0.0,
        "actor_z_jsd_min": 0.0,
        "actor_z_jsd_max": 0.0,
        "actor_z_pairs_total": 0.0,
        "actor_z_pairs_above_margin": 0.0,
        "actor_z_pairs_above_margin_fraction": 0.0,
        "actor_z_eval_state_count": 0.0,
        "actor_z_eval_pair_count": 0.0,
        "actor_z_jsd_per_head": "",
        "actor_z_argmax_disagree": 0.0,
        "actor_z_logit_l2": 0.0,
        "actor_z_entropy_by_z": "",
        "actor_z_trunk_l2": 0.0,
        "actor_z_film_mod_l2": 0.0,
    }
    if not trainer.use_latent_strategy or trainer.latent_k <= 1:
        return zero_stats
    length = int(buffer.pos)
    if length <= 0:
        return zero_stats
    total = length * int(buffer.n_envs)
    if total <= 0:
        return zero_stats

    from rl.custom_ppo.inference import FORCED_Z_PROFILE_MAX_ROWS
    if total > FORCED_Z_PROFILE_MAX_ROWS:
        row_idx = torch.linspace(
            0,
            total - 1,
            steps=FORCED_Z_PROFILE_MAX_ROWS,
            device=trainer.device,
        ).long()
    else:
        row_idx = torch.arange(total, device=trainer.device)
    row_idx = torch.clamp(row_idx, 0, total - 1)

    obs_batch = {
        "grid": buffer.fields["obs_grid"][:length].reshape(total, *buffer.fields["obs_grid"].shape[2:]).index_select(0, row_idx),
        "vec": buffer.fields["obs_vec"][:length].reshape(total, *buffer.fields["obs_vec"].shape[2:]).index_select(0, row_idx),
        "agent_mask": buffer.fields["obs_agent_mask"][:length].reshape(total, *buffer.fields["obs_agent_mask"].shape[2:]).index_select(0, row_idx),
        "mask": buffer.fields["obs_mask"][:length].reshape(total, *buffer.fields["obs_mask"].shape[2:]).index_select(0, row_idx),
    }

    logits_by_z: list[torch.Tensor] = []
    trunk_by_z: list[torch.Tensor] = []
    dists_by_z: list[list[torch.distributions.Categorical]] = []
    with torch.no_grad():
        for z_id in range(int(trainer.latent_k)):
            z_idx = torch.full((int(row_idx.numel()),), z_id, dtype=torch.long, device=trainer.device)
            logits = _batched_policy_logits(trainer, obs_batch, z_idx=z_idx)
            logits = trainer.model._mask_logits(logits, obs_batch.get("mask"))
            logits_by_z.append(logits.float())
            trunk_by_z.append(_batched_policy_trunk_features(trainer, obs_batch, z_idx=z_idx).float())
            dists_by_z.append(list(trainer.model._categoricals(logits)))

    kl_values: list[float] = []
    latent_k = int(trainer.latent_k)
    for i in range(latent_k):
        for j in range(latent_k):
            if i == j:
                continue
            dists_i = dists_by_z[i]
            dists_j = dists_by_z[j]
            kl_sum = torch.zeros((int(row_idx.numel()),), device=trainer.device)
            for di, dj in zip(dists_i, dists_j):
                kl_sum += torch.distributions.kl.kl_divergence(di, dj)
            kl_values.append(float(kl_sum.mean().item()))

    mean_kl = float(np.mean(kl_values)) if kl_values else 0.0
    action_dims = tuple(int(dim) for dim in trainer.model.action_dims)
    heads_per_agent = int(trainer.model.heads_per_agent)
    agent_mask = obs_batch.get("agent_mask")
    if agent_mask is None:
        agent_mask = torch.ones(
            (int(row_idx.numel()), int(trainer.model.n_agents)),
            dtype=torch.float32,
            device=trainer.device,
        )
    else:
        agent_mask = agent_mask.to(device=trainer.device).float()

    offsets: list[tuple[int, int]] = []
    offset = 0
    for dim in action_dims:
        offsets.append((offset, offset + dim))
        offset += dim

    entropy_by_z: list[float] = []
    for z_id, dists in enumerate(dists_by_z):
        entropy_values: list[torch.Tensor] = []
        for action_idx, dist in enumerate(dists):
            agent_idx = action_idx // heads_per_agent
            valid = agent_mask[:, agent_idx] > 0.5
            if bool(valid.any()):
                entropy_values.append(dist.entropy()[valid])
        entropy = (
            torch.cat(entropy_values).mean()
            if entropy_values
            else torch.zeros((), device=trainer.device)
        )
        entropy_by_z.append(float(entropy.item()))
        zero_stats[f"actor_z_entropy_z{z_id}"] = float(entropy.item())

    jsd_values: list[torch.Tensor] = []
    argmax_disagreements: list[torch.Tensor] = []
    logit_l2_values: list[torch.Tensor] = []
    jsd_by_head: list[list[torch.Tensor]] = [
        [] for _ in range(heads_per_agent)
    ]
    for i in range(latent_k):
        for j in range(i + 1, latent_k):
            logits_i = logits_by_z[i]
            logits_j = logits_by_z[j]
            for action_idx, (start, end) in enumerate(offsets):
                agent_idx = action_idx // heads_per_agent
                head_idx = action_idx % heads_per_agent
                valid = agent_mask[:, agent_idx] > 0.5
                if not bool(valid.any()):
                    continue
                head_i = logits_i[valid, start:end]
                head_j = logits_j[valid, start:end]
                jsd = _jsd_from_logits(head_i, head_j)
                jsd_values.append(jsd)
                jsd_by_head[head_idx].append(jsd)
                argmax_disagreements.append(
                    (head_i.argmax(dim=-1) != head_j.argmax(dim=-1)).float()
                )
                logit_l2_values.append(
                    torch.linalg.vector_norm(head_i - head_j, dim=-1)
                )

    all_jsd = (
        torch.cat(jsd_values)
        if jsd_values
        else torch.zeros((1,), device=trainer.device)
    )
    pair_count = max(0, latent_k * (latent_k - 1) // 2)
    actor_margin = float(getattr(trainer.cfg, "actor_jsd_margin", 0.001) or 0.001)
    pair_means: list[float] = []
    for i in range(latent_k):
        for j in range(i + 1, latent_k):
            logits_i = logits_by_z[i]
            logits_j = logits_by_z[j]
            values: list[torch.Tensor] = []
            for action_idx, (start, end) in enumerate(offsets):
                agent_idx = action_idx // heads_per_agent
                valid = agent_mask[:, agent_idx] > 0.5
                if bool(valid.any()):
                    values.append(_jsd_from_logits(logits_i[valid, start:end], logits_j[valid, start:end]))
            if values:
                pair_means.append(float(torch.cat(values).mean().detach().cpu().item()))
    pairs_above = sum(1 for value in pair_means if value >= actor_margin)
    all_disagree = (
        torch.cat(argmax_disagreements)
        if argmax_disagreements
        else torch.zeros((1,), device=trainer.device)
    )
    all_logit_l2 = (
        torch.cat(logit_l2_values)
        if logit_l2_values
        else torch.zeros((1,), device=trainer.device)
    )
    per_head_jsd = [
        float(torch.cat(values).mean().item()) if values else 0.0
        for values in jsd_by_head
    ]
    for head_idx, value in enumerate(per_head_jsd):
        zero_stats[f"actor_z_jsd_head_{head_idx}"] = value

    trunk_l2_values: list[torch.Tensor] = []
    for i in range(latent_k):
        for j in range(i + 1, latent_k):
            diff = trunk_by_z[i] - trunk_by_z[j]
            trunk_l2_values.append(torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=-1))
    trunk_l2_mean = (
        float(torch.cat(trunk_l2_values).mean().item())
        if trunk_l2_values
        else 0.0
    )
    latent_actor = getattr(trainer.model, "latent_actor", None)
    film_mod_l2 = 0.0
    if latent_actor is not None and hasattr(latent_actor, "film_modulation_l2"):
        z0 = torch.zeros((1,), dtype=torch.long, device=trainer.device)
        z1 = torch.ones((1,), dtype=torch.long, device=trainer.device)
        film_mod_l2 = float(latent_actor.film_modulation_l2(z0, z1))

    zero_stats.update(
        {
            "policy_z_sensitivity_KL": mean_kl,
            "actor_z_jsd_mean": float(all_jsd.mean().item()),
            "actor_z_jsd_min": float(all_jsd.min().item()),
            "actor_z_jsd_max": float(all_jsd.max().item()),
            "actor_z_pairs_total": float(pair_count),
            "actor_z_pairs_above_margin": float(pairs_above),
            "actor_z_pairs_above_margin_fraction": float(pairs_above) / float(max(1, pair_count)),
            "actor_z_eval_state_count": float(int(row_idx.numel())),
            "actor_z_eval_pair_count": float(pair_count),
            "actor_z_jsd_per_head": ",".join(
                f"{value:.8e}" for value in per_head_jsd
            ),
            "actor_z_argmax_disagree": float(all_disagree.mean().item()),
            "actor_z_logit_l2": float(all_logit_l2.mean().item()),
            "actor_z_entropy_by_z": ",".join(
                f"{value:.8e}" for value in entropy_by_z
            ),
            "actor_z_trunk_l2": trunk_l2_mean,
            "actor_z_film_mod_l2": film_mod_l2,
        }
    )
    return zero_stats


def _jsd_from_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return Jensen-Shannon divergence for matching categorical logits."""
    log_p = torch.log_softmax(logits_a.float(), dim=dim)
    log_q = torch.log_softmax(logits_b.float(), dim=dim)
    p = log_p.exp()
    q = log_q.exp()
    mixture = 0.5 * (p + q)
    log_mixture = torch.log(mixture.clamp_min(float(eps)))
    kl_pm = torch.sum(p * (log_p - log_mixture), dim=dim)
    kl_qm = torch.sum(q * (log_q - log_mixture), dim=dim)
    return 0.5 * (kl_pm + kl_qm)


# ---------------------------------------------------------------------------
# V6I7 Phase-1 diagnostics: pairwise JSD, adapter gradient norms, critic
# value variance across latent IDs for identical states.
# ---------------------------------------------------------------------------

def compute_pairwise_actor_jsd(
    model: "SharedActorCentralizedCritic",
    local_features: torch.Tensor,
) -> dict[str, float]:
    """Compute mean/min/max pairwise actor JSD and argmax disagreement.

    ``local_features`` must have shape ``(N, local_feature_dim)`` — a batch of
    pre-encoded local observations (CNN + vec, no z concatenated).  The
    function evaluates all K*(K-1)/2 pairs and returns a flat stats dict
    suitable for the metrics CSV.
    """
    from itertools import combinations

    K = int(model.latent_k)
    if K < 2 or not model.uses_latent_strategy:
        return {
            "actor_jsd_mean": float("nan"),
            "actor_jsd_min": float("nan"),
            "actor_jsd_max": float("nan"),
            "actor_argmax_disagree": float("nan"),
        }
    device = local_features.device
    N = local_features.shape[0]

    with torch.no_grad():
        logits_by_z = []
        for k in range(K):
            z_t = torch.full((N,), k, dtype=torch.long, device=device)
            logits_k = model.latent_actor(local_features, z_t)
            logits_by_z.append(logits_k)

    pair_jsds: list[float] = []
    pair_disagrees: list[float] = []
    for i, j in combinations(range(K), 2):
        jsd = _jsd_from_logits(logits_by_z[i], logits_by_z[j]).mean().item()
        pair_jsds.append(float(jsd))
        argmax_i = logits_by_z[i].argmax(dim=-1)
        argmax_j = logits_by_z[j].argmax(dim=-1)
        pair_disagrees.append(float((argmax_i != argmax_j).float().mean().item()))

    return {
        "actor_jsd_mean": float(sum(pair_jsds) / len(pair_jsds)),
        "actor_jsd_min": float(min(pair_jsds)),
        "actor_jsd_max": float(max(pair_jsds)),
        "actor_argmax_disagree": float(sum(pair_disagrees) / len(pair_disagrees)),
    }


def compute_adapter_grad_norms(model: "SharedActorCentralizedCritic") -> dict[str, float]:
    """Return per-latent adapter and action-bias gradient L2 norms.

    Call after ``loss.backward()`` but before ``optimizer.zero_grad()``.
    Returns an empty dict when residual adapters are not enabled.
    """
    la = getattr(model, "latent_actor", None)
    if la is None or not getattr(la, "enable_latent_z_residual", False):
        return {}
    out: dict[str, float] = {}
    adapters = getattr(la, "latent_adapters", None)
    if adapters is not None:
        for k, adapter in enumerate(adapters):
            total_sq = sum(
                p.grad.pow(2).sum().item()
                for p in adapter.parameters()
                if p.grad is not None
            )
            out[f"adapter_grad_norm_z{k}"] = float(total_sq ** 0.5)
    gates = getattr(la, "latent_adapter_gates", None)
    if gates is not None and gates.grad is not None:
        # Gate gradient = A_z(h) * upstream_grad.  At initialization A_z(h)=0
        # (zero-init weights), so gate grad is 0 for the first few updates.
        # This is expected — the gate wakes up once adapter weights move from 0.
        for k in range(int(gates.shape[0])):
            out[f"adapter_gate_grad_z{k}"] = float(gates.grad[k].abs().item())
    biases = getattr(la, "latent_action_biases", None)
    if biases is not None and biases.grad is not None:
        for k in range(int(biases.shape[0])):
            out[f"action_bias_grad_norm_z{k}"] = float(biases.grad[k].norm().item())
    return out


def compute_critic_value_variance(
    model: "SharedActorCentralizedCritic",
    global_state: torch.Tensor,
) -> dict[str, float]:
    """Return Var_z[V(s, z)] for identical global states across all K latents.

    ``global_state`` has shape ``(N, global_state_dim)``.  Returns nan when
    latent strategy is not enabled.
    """
    K = int(model.latent_k)
    if K < 2 or not model.uses_latent_strategy:
        return {"critic_value_var_z": float("nan")}
    device = global_state.device
    N = global_state.shape[0]

    with torch.no_grad():
        values_by_z = []
        for k in range(K):
            z_t = torch.full((N,), k, dtype=torch.long, device=device)
            v_k = model.values(global_state, z_idx=z_t)
            values_by_z.append(v_k)
        stacked = torch.stack(values_by_z, dim=0)  # (K, N)
        var_across_z = stacked.var(dim=0).mean().item()

    return {"critic_value_var_z": float(var_across_z)}


def _v6i8_residual_adapter_stats(runtime: Any, buffer: Any) -> dict[str, float]:
    """Post-update V6I8 adapter diagnostics: pairwise actor JSD and adapter grad norms.

    Pairwise JSD: identical random local features, all K latents — measures
    differentiation from adapters and biases only (z-embedding contribution
    held fixed because the same input tensor is reused across z values).

    Adapter grad norms: a diagnostic forward-backward on the same random
    sample, immediately zeroed after measurement.  These are diagnostic
    gradient magnitudes, not training gradients.

    Returns an empty dict when ``enable_latent_z_residual`` is False.
    """
    from itertools import combinations

    model = getattr(runtime, "model", None)
    if model is None:
        return {}
    la = getattr(model, "latent_actor", None)
    if la is None or not getattr(la, "enable_latent_z_residual", False):
        return {}
    K = int(getattr(model, "latent_k", 0))
    if K < 2:
        return {}

    try:
        device = next(model.parameters()).device
        N = 64
        full_input_dim = int(model.actor_input_dim)
        local_feats = torch.randn(N, full_input_dim, device=device)

        with torch.no_grad():
            logits_by_z = []
            for k in range(K):
                z_t = torch.full((N,), k, dtype=torch.long, device=device)
                logits_k = la(local_feats, z_t)
                logits_by_z.append(logits_k)

        pair_jsds: list[float] = []
        pair_disagrees: list[float] = []
        for i, j in combinations(range(K), 2):
            jsd = _jsd_from_logits(logits_by_z[i], logits_by_z[j]).mean().item()
            pair_jsds.append(float(jsd))
            argmax_i = logits_by_z[i].argmax(dim=-1)
            argmax_j = logits_by_z[j].argmax(dim=-1)
            pair_disagrees.append(float((argmax_i != argmax_j).float().mean().item()))

        out: dict[str, float] = {
            "actor_jsd_mean": float(sum(pair_jsds) / len(pair_jsds)),
            "actor_jsd_min": float(min(pair_jsds)),
            "actor_jsd_max": float(max(pair_jsds)),
            "actor_argmax_disagree": float(sum(pair_disagrees) / len(pair_disagrees)),
        }

        # Diagnostic grad norms: independent forward-backward on random data.
        # Called from post_update.run() AFTER all minibatch optimizer.step()
        # and zero_grad() calls, so no training gradient is read or overwritten.
        # The surrounding zero_grad() calls isolate this entirely from training.
        model.zero_grad()
        z_diag = torch.randint(K, (N,), device=device)
        logits_diag = la(local_feats.detach(), z_diag)
        logits_diag.sum().backward()
        out.update(compute_adapter_grad_norms(model))
        model.zero_grad()

        return out
    except Exception:
        return {}


# Public alias so post_update.py can import it by the stable diagnostic name.
_v6i8_adapter_stats = _v6i8_residual_adapter_stats
