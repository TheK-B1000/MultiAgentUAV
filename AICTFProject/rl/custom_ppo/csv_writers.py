from __future__ import annotations

import csv
import os
from typing import Any, Optional

from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES
from rl.latent_phase_labels import TEAM_PHASES

E3_STEP_TELEMETRY_FIELDS: tuple[str, ...] = (
    "update",
    "rollout_step",
    "env_id",
    "global_step",
    "z_t",
    "q_phi_entropy",
    "q_phi_argmax",
    "switched",
    "game_phase",
    "team_phase",
    "score_outcome",
    "stalemate_frac",
    "opponent_id",
    "phase_id",
    "blue_ahead",
) + BEHAVIOR_TELEMETRY_NAMES + (
    "spread_bucket",
    "role_bucket",
    "pressure_bucket",
    "attack_defense_ratio_bucket",
)

# When renaming metrics columns, old CSV headers may still use the legacy name; see ``_write_csv_row``.
_METRICS_CSV_LEGACY_COLUMN_FILL: dict[str, str] = {"strategy_aux_return_loss": "strategy_q_loss"}

# Columns for MI(z; opponent) and episode_opp{idx}_z* (OP1 … OP5_RUSHER, OP6, OP7).
SCRIPTED_OPPONENT_MI_COUNT: int = 7


def _opponent_id_int_from_info(cfg: Any, info: dict[str, Any]) -> int:
    """Scripted opponent index for MI telemetry: OP1→0 … OP7→6; ``-1`` if unknown / non-scripted."""
    er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
    kind = str(er.get("opponent_kind", info.get("opponent_kind", "scripted")) or "scripted").lower()
    if kind != "scripted":
        return -1
    tag_raw = str(
        er.get("scripted_tag")
        or info.get("opponent_key", getattr(cfg, "fixed_opponent_tag", "OP3"))
        or ""
    ).strip().upper()
    tag = "OP5_RUSHER" if tag_raw == "OP5" else tag_raw
    if tag == "OP6_TURTLE":
        tag = "OP6"
    if tag == "OP7_SWITCHER":
        tag = "OP7"
    return {"OP1": 0, "OP2": 1, "OP3": 2, "OP4": 3, "OP5_RUSHER": 4, "OP6": 5, "OP7": 6}.get(tag, -1)


def _opponent_id_csv_from_info(cfg: Any, info: dict[str, Any]) -> str:
    oid = _opponent_id_int_from_info(cfg, info)
    return str(int(oid)) if oid >= 0 else ""


def _opponent_legend(cfg: Any, info: dict[str, Any]) -> str:
    """Compact opponent string for logging (scripted:OP3, snapshot:name, ...)."""
    er = info.get("episode_result") or {}
    kind = str(er.get("opponent_kind", info.get("opponent_kind", "scripted")) or "scripted").lower()
    if kind == "scripted":
        tag = str(er.get("scripted_tag") or info.get("opponent_key", getattr(cfg, "fixed_opponent_tag", "OP3")))
        return f"SCRIPTED:{str(tag).upper()}"
    if kind == "snapshot":
        snap = str(er.get("opponent_snapshot", "") or info.get("opponent_key", ""))
        return f"SNAPSHOT:{snap}" if snap else "SNAPSHOT:unknown"
    return f"{kind.upper()}:?"


def _episode_fieldnames() -> list[str]:
    return [
        "episode_id",
        "run_id",
        "run_pid",
        "timesteps",
        "policy_update",
        "rollout_step",
        "latent_z",
        "curriculum_phase",
        "mode",
        "map_set",
        "opponent",
        "opponent_id",
        "success",
        "blue_score",
        "red_score",
        "win_margin",
        "decision_steps",
        "zone_coverage",
        "collision_free_episode",
        "collision_events_per_episode",
        "near_misses_per_episode",
        "time_to_first_score",
        "mean_inter_robot_dist",
        "reward_terminal",
        "reward_offense",
        "reward_pbrs",
        "reward_team",
        "reward_sparse",
        "reward_sparse_points",
        "reward_failure",
        "reward_total",
    ]


def _update_fieldnames(use_latent_strategy: bool, latent_k: int) -> list[str]:
    fields = [
        "update",
        "run_id",
        "run_pid",
        "timesteps",
        "episodes_completed",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "rolling_win_rate_50ep",
        "rolling_win_rate_200ep",
        "rollout_reward_mean",
        "rollout_reward_std",
        "rollout_return_mean",
        "rollout_return_std",
        "rollout_episodes",
        "rollout_wins",
        "rollout_losses",
        "rollout_draws",
        "rollout_win_rate",
        "rollout_win_margin_mean",
        "rollout_blue_score_mean",
        "rollout_red_score_mean",
        "explained_variance",
        "reward_terminal_mean",
        "reward_offense_mean",
        "reward_pbrs_mean",
        "reward_team_mean",
        "reward_sparse_mean",
        "reward_sparse_points_mean",
        "reward_failure_mean",
        "reward_total_mean",
        "reward_outcome_mean",
        "reward_shaping_mean",
        "reward_shaping_to_outcome_abs_ratio",
        "reward_shaping_coef",
        "reward_failure_to_outcome_abs",
        "policy_loss",
        "value_loss",
        "value_loss_min",
        "value_loss_std",
        "value_loss_p10",
        "value_loss_p50",
        "value_loss_p90",
        "value_loss_max",
        "return_norm_mean",
        "return_norm_std",
        "return_norm_count",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
        "learning_rate",
        "strategy_entropy",
        "strategy_entropy_frac",
        "strategy_policy_loss",
        "strategy_approx_kl",
        "strategy_clip_fraction",
        "strategy_ratio_std",
        "strategy_aux_return_loss",
        "strategy_persist_loss",
        "strategy_grad_norm",
        "strategy_resample_count",
        "strategy_resample_fraction",
        "latent_episode_pg_loss",
        "latent_episode_v_loss",
        "latent_episode_entropy",
        "latent_episode_adv_mean",
        "latent_episode_adv_std",
        "latent_episode_return_mean",
        "latent_episode_return_std",
        "latent_episode_ratio_mean",
        "latent_episode_ratio_max",
        "latent_episode_ratio_min",
        "latent_episode_approx_kl",
        "latent_episode_clip_fraction",
        "latent_episode_count",
        "latent_episode_loss",
        "strategy_bucket_best_match_frac",
        "strategy_experience_records",
        "strategy_experience_buckets",
        "strategy_unique_count",
        "strategy_dominant",
        "strategy_switch_count",
        "strategy_switch_fraction",
        "strategy_wr_spread",
        "strategy_resample_fraction_rollout",
        "rollout_adv_std",
        "rollout_adv_std_at_z_switch",
        "rollout_adv_std_not_z_switch",
        "curriculum_phase",
        "curriculum_phase_idx",
        "curriculum_phase_episodes",
        "curriculum_phase_win_rate",
    ]
    if use_latent_strategy:
        fields.append("strategy_kl")
        fields.extend(f"strategy_occupancy_{idx}" for idx in range(latent_k))
        for idx in range(latent_k):
            fields.extend(
                [
                    f"episode_z_{idx}_count",
                    f"episode_z_{idx}_win_rate",
                    f"episode_z_{idx}_blue_score_mean",
                    f"episode_z_{idx}_red_score_mean",
                    f"episode_z_{idx}_win_margin_mean",
                ]
            )
        for idx in range(latent_k):
            fields.extend(
                [
                    f"strategy_resample_adv_mean_z{idx}",
                    f"strategy_resample_adv_std_z{idx}",
                    f"strategy_resample_adv_n_z{idx}",
                ]
            )
        fields.append("latent_mi_z_opponent_nats")
        fields.append("latent_mi_z_phase_nats")
        fields.append("latent_mi_z_outcome_nats")
        fields.append("latent_mi_z_flag_state_nats")
        fields.append("latent_mi_z_spread_bucket_nats")
        fields.append("latent_mi_z_role_bucket_nats")
        fields.append("latent_mi_z_pressure_bucket_nats")
        fields.append("latent_mi_z_attack_defense_ratio_bucket_nats")
        for r in range(5):  # N_ROLE_BUCKET_MI is 5
            for z_idx in range(latent_k):
                fields.append(f"latent_role{r}_z{z_idx}_frac")
            fields.append(f"latent_role{r}_switch_mean")
        fields.extend(
            [
                "latent_switch_rate_blue_ahead",
                "latent_switch_rate_blue_trail",
                "latent_reward_sum_5_after_z_switch_mean",
            ]
        )
        for p in range(len(TEAM_PHASES)):
            for z_idx in range(latent_k):
                fields.append(f"latent_phase{p}_z{z_idx}_frac")
            fields.extend(
                [
                    f"latent_phase{p}_switch_mean",
                    f"latent_phase{p}_blue_ahead_mean",
                    f"latent_phase{p}_capture_step_mean",
                    f"q_phi_phase{p}_entropy_mean",
                ]
            )
            for z_idx in range(latent_k):
                fields.append(f"q_phi_phase{p}_z{z_idx}_prob_mean")
        fields.append("latent_behavior_diversity_l2_mean")
        for z_idx in range(latent_k):
            for name in BEHAVIOR_TELEMETRY_NAMES:
                fields.append(f"latent_z{z_idx}_behavior_{name}_mean")
        fields.append("forced_z_macro_jsd_mean")
        from rl.custom_ppo.inference import FORCED_Z_MACRO_ACTIONS
        for z_idx in range(latent_k):
            for _action_id, action_name in FORCED_Z_MACRO_ACTIONS:
                fields.append(f"forced_z{z_idx}_macro_{action_name}_prob")
            fields.append(f"forced_z{z_idx}_macro_entropy")
        for o_idx in range(SCRIPTED_OPPONENT_MI_COUNT):
            for z_idx in range(latent_k):
                fields.append(f"strategy_occupancy_op{o_idx}_z{z_idx}")
        for o_idx in range(SCRIPTED_OPPONENT_MI_COUNT):
            for z_idx in range(latent_k):
                fields.extend(
                    [
                        f"episode_opp{o_idx}_z{z_idx}_count",
                        f"episode_opp{o_idx}_z{z_idx}_win_rate",
                    ]
                )
        # Append new diagnostic columns
        fields.append("strategy_phase_loss")
        fields.extend([
            "latent_switch_near_capture_frac",
            "latent_switch_near_kill_frac",
            "latent_switch_near_return_frac",
        ])
        for f in range(4):
            for k in range(latent_k):
                fields.append(f"latent_flag_state{f}_z{k}_frac")
        for s in range(3):
            for k in range(latent_k):
                fields.append(f"latent_spread{s}_z{k}_frac")
        for a in range(3):
            for k in range(latent_k):
                fields.append(f"latent_adr{a}_z{k}_frac")
        for p in range(len(TEAM_PHASES)):
            fields.append(f"latent_phase{p}_entropy")
        fields.extend([
            "latent_role_diversity",
            "latent_spread_diversity",
            "latent_pressure_diversity",
            "latent_adr_diversity",
        ])
    return fields


def _strategy_experience_fieldnames() -> list[str]:
    return [
        "update",
        "run_id",
        "run_pid",
        "timesteps",
        "bucket_id",
        "z",
        "count",
        "bucket_count",
        "mean_return",
        "win_rate",
        "q_phi_prob_mean",
        "chosen_freq",
        "best_z",
        "best_z_match_frac",
    ]


def _write_csv_row(
    path: str,
    fieldnames: list[str],
    row: dict[str, Any],
    *,
    legacy_column_fill: Optional[dict[str, str]] = None,
) -> None:
    """Append one row with a stable header; used for long-run audit telemetry."""
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    legacy_fill = legacy_column_fill or {}
    nonempty = os.path.isfile(path) and os.path.getsize(path) > 0
    if nonempty:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames
            if old_fields is None:
                raise ValueError(f"CSV schema mismatch for {path!r}: empty or invalid header.")
            old_list = list(old_fields)
            old_rows = list(reader)
        if old_list != fieldnames:
            dropped = [c for c in old_list if c not in fieldnames]
            if dropped:
                allowed_old = set(legacy_fill.values())
                if not (legacy_fill and set(dropped).issubset(allowed_old)):
                    raise ValueError(
                        f"CSV schema mismatch for {path!r}: existing columns dropped or renamed "
                        f"{dropped!r}; existing header {old_list!r} vs expected {fieldnames!r}. "
                        "Use a new output path or migrate manually."
                    )
            print(
                f"[PPO] Migrating CSV (additive columns): {path}\n"
                f"      was {len(old_list)} cols -> now {len(fieldnames)} cols; "
                f"rewriting {len(old_rows)} row(s)."
            )
            with open(path, "w", newline="", encoding="utf-8") as wf:
                writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in old_rows:
                    out_row: dict[str, Any] = {}
                    for k in fieldnames:
                        v = r.get(k, "")
                        if v == "" and k in legacy_fill:
                            v = r.get(legacy_fill[k], "")
                        out_row[k] = v
                    writer.writerow(out_row)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not nonempty:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _ensure_additive_csv_header(path: str, fieldnames: list[str]) -> None:
    """Rewrite CSV when new columns are appended (additive-only; never drop old columns)."""
    if not path or not (os.path.isfile(path) and os.path.getsize(path) > 0):
        return
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        old_list = list(reader.fieldnames)
        old_rows = list(reader)
    if old_list == fieldnames:
        return
    dropped = [c for c in old_list if c not in fieldnames]
    if dropped:
        raise ValueError(
            f"E3 telemetry CSV schema mismatch for {path!r}: cannot drop columns {dropped!r}. "
            f"Use a new --e3 path or migrate manually."
        )
    if len(fieldnames) <= len(old_list):
        return
    print(
        f"[PPO] Migrating E3 step CSV (additive columns): {path}\n"
        f"      was {len(old_list)} cols -> now {len(fieldnames)} cols; "
        f"rewriting {len(old_rows)} row(s).",
        flush=True,
    )
    with open(path, "w", newline="", encoding="utf-8") as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in old_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
