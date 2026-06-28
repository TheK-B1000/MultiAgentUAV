"""Latent switching proximity diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _flag_return_indices(
    blue_cap_col: np.ndarray, red_cap_col: np.ndarray, abs_rsp_col: np.ndarray
) -> np.ndarray:
    """Time indices where a flag was returned (carrier dropped it) without scoring."""
    if blue_cap_col.shape[0] < 2:
        return np.empty(0, dtype=np.int64)
    blue_ret = blue_cap_col[:-1] & ~blue_cap_col[1:]
    red_ret = red_cap_col[:-1] & ~red_cap_col[1:]
    no_score = abs_rsp_col[1:] < 1.0
    hit = (blue_ret | red_ret) & no_score
    return np.where(hit)[0] + 1


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


flag_return_indices = _flag_return_indices
reward_sum_after_switch_5 = _reward_sum_after_switch_5
switch_proximity_fracs = _switch_proximity_fracs

__all__ = [
    "_flag_return_indices",
    "_reward_sum_after_switch_5",
    "_switch_proximity_fracs",
    "flag_return_indices",
    "reward_sum_after_switch_5",
    "switch_proximity_fracs",
]
