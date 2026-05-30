"""Telemetry sink for :class:`CustomPPOTrainer`.

This module owns the *observation* side of training — anything that reads
trainer / buffer state and writes it to disk or stdout. It does **not** make
decisions about actions, rewards, advantages, gradients, or z-selection.

The split lets us refactor the trainer's "decide" code (rollout collector,
PPO updater, latent state machine) independently from the "observe and
write" code, and lets us swap or silence telemetry in tests without
touching the training loop.

State ownership
---------------
Owned by :class:`TrainingTelemetry`:

- ``e3_step_telemetry_path`` (resolved from ``cfg`` at construction)
- the persistent e3-step CSV file handle, dict-writer, field cache and
  rows-since-flush counter

Read from the trainer (context-object pattern):

- ``trainer.cfg``, ``trainer.run_id``, ``trainer.run_pid``
- ``trainer.episode_csv_path``, ``trainer.metrics_csv_path``
- ``trainer._episodes_completed``, ``trainer._updates_completed``
- ``trainer._ep_wins / _ep_losses / _ep_draws``
- ``trainer._recent_episode_successes``, ``trainer._rollout_episode_records``
- ``trainer.use_latent_strategy``, ``trainer.latent_k``
- ``trainer.reward_dense_weight``, ``trainer.global_step``
- ``trainer.curriculum``
- ``trainer._reward_shaping_coef()``
"""

from __future__ import annotations

import csv
import math
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES
from rl.global_state import (
    coarse_game_phase_from_global_state,
    outcome_label_from_global_state,
    team_phase_id_from_global_state,
    team_phase_label_from_global_state,
)
from rl.custom_ppo.csv_writers import (
    E3_STEP_TELEMETRY_FIELDS,
    SCRIPTED_OPPONENT_MI_COUNT,
    _METRICS_CSV_LEGACY_COLUMN_FILL,
    _ensure_additive_csv_header,
    _episode_fieldnames,
    _opponent_id_csv_from_info,
    _opponent_id_int_from_info,
    _opponent_legend,
    _update_fieldnames,
    _write_csv_row,
)
from rl.ppo_core import TensorDictRolloutBuffer

if TYPE_CHECKING:
    from rl.custom_ppo.trainer import CustomPPOTrainer


class TrainingTelemetry:
    """Episode/update CSV writers + persistent e3-step CSV file handle.

    The trainer constructs one telemetry instance and holds it as
    ``self.telemetry``. All callers read/write through method calls
    (``self.telemetry.write_episode_metrics(...)``) rather than poking the
    e3 file handle directly.
    """

    def __init__(self, trainer: "CustomPPOTrainer") -> None:
        self.trainer = trainer
        cfg = trainer.cfg
        self.e3_step_telemetry_path = str(getattr(cfg, "e3_step_telemetry_path", "") or "")
        self._e3_file: Any = None
        self._e3_writer: Any = None
        self._e3_fields_cache: Optional[list[str]] = None
        self._e3_rows_since_flush = 0
        self._e3_flush_every_steps = 100

    @staticmethod
    def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
        y_pred = values.detach().float().reshape(-1)
        y_true = returns.detach().float().reshape(-1)
        if y_true.numel() <= 1:
            return 0.0
        var_y = torch.var(y_true, unbiased=False)
        if float(var_y.detach().cpu().item()) <= 1e-12:
            return 0.0
        ev = 1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y
        return float(ev.detach().cpu().item())

    def append_e3_step(
        self,
        *,
        rollout_step: int,
        global_step_at_step_end: int,
        decision_global_state_np: np.ndarray,
        z_t: torch.Tensor,
        prev_z: torch.Tensor,
        strategy_aux: dict[str, torch.Tensor],
        infos: list[Any],
        behavior_telemetry_np: np.ndarray,
        spread_bucket_np: np.ndarray,
        role_bucket_np: np.ndarray,
        pressure_bucket_np: np.ndarray,
        attack_defense_ratio_bucket_np: np.ndarray,
        blue_ahead_np: np.ndarray,
    ) -> None:
        trainer = self.trainer
        if not self.e3_step_telemetry_path or not trainer.use_latent_strategy:
            return
        path = self.e3_step_telemetry_path
        zt = z_t.detach().cpu().numpy()
        pz = prev_z.detach().cpu().numpy()
        zH = strategy_aux["z_entropy"].detach().cpu().numpy()
        zlog = strategy_aux["z_logits"].detach().cpu().numpy()
        am = zlog.argmax(axis=-1)
        n_e = int(zt.shape[0])
        assert int(decision_global_state_np.shape[0]) == n_e, (decision_global_state_np.shape, n_e)
        fields = self._e3_fields_cache or list(E3_STEP_TELEMETRY_FIELDS)
        if self._e3_writer is None:
            d = os.path.dirname(os.path.abspath(path)) or "."
            os.makedirs(d, exist_ok=True)
            _ensure_additive_csv_header(path, fields)
            needs_header = not (os.path.isfile(path) and os.path.getsize(path) > 0)
            self._e3_file = open(path, "a", newline="", encoding="utf-8")
            self._e3_writer = csv.DictWriter(self._e3_file, fieldnames=fields, extrasaction="ignore")
            if needs_header:
                self._e3_writer.writeheader()
            self._e3_fields_cache = fields
        w = self._e3_writer
        upd = int(trainer._updates_completed)
        for e in range(n_e):
            info = dict(infos[e]) if e < len(infos) else {}
            gs_e = decision_global_state_np[e]
            sf = float(info.get("stalemate_frac", 0.0) or 0.0)
            pid = int(team_phase_id_from_global_state(gs_e, stalemate_frac=sf))
            row: dict[str, Any] = {
                "update": upd,
                "rollout_step": int(rollout_step),
                "env_id": e,
                "global_step": int(global_step_at_step_end),
                "z_t": int(zt[e]),
                "q_phi_entropy": float(zH[e]),
                "q_phi_argmax": int(am[e]),
                "switched": int(bool(int(zt[e]) != int(pz[e]))),
                "game_phase": coarse_game_phase_from_global_state(gs_e),
                "team_phase": team_phase_label_from_global_state(gs_e, stalemate_frac=sf),
                "score_outcome": outcome_label_from_global_state(gs_e),
                "stalemate_frac": sf,
                "opponent_id": int(_opponent_id_int_from_info(trainer.cfg, info)),
                "phase_id": pid,
                "blue_ahead": float(blue_ahead_np[e]),
                "spread_bucket": int(spread_bucket_np[e]),
                "role_bucket": int(role_bucket_np[e]),
                "pressure_bucket": int(pressure_bucket_np[e]),
                "attack_defense_ratio_bucket": int(attack_defense_ratio_bucket_np[e]),
            }
            for j, name in enumerate(BEHAVIOR_TELEMETRY_NAMES):
                row[name] = float(behavior_telemetry_np[e, j])
            w.writerow({key: row.get(key, "") for key in fields})
        self._e3_rows_since_flush += 1
        if self._e3_rows_since_flush >= self._e3_flush_every_steps:
            if self._e3_file is not None:
                self._e3_file.flush()
            self._e3_rows_since_flush = 0

    def close_e3_step_telemetry(self) -> None:
        """Flush and close the persistent e3 step telemetry file (idempotent)."""
        f = self._e3_file
        if f is None:
            return
        try:
            f.flush()
        except Exception:
            pass
        try:
            f.close()
        except Exception:
            pass
        self._e3_file = None
        self._e3_writer = None
        self._e3_rows_since_flush = 0

    def write_episode_metrics(
        self,
        info: dict[str, Any],
        *,
        blue_score: int,
        red_score: int,
        timestep: int,
        rollout_step: Optional[int] = None,
        latent_z: Optional[int] = None,
    ) -> None:
        trainer = self.trainer
        if not trainer.episode_csv_path:
            return
        er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
        row = {
            "episode_id": trainer._episodes_completed,
            "run_id": trainer.run_id,
            "run_pid": trainer.run_pid,
            "timesteps": int(timestep),
            "policy_update": int(trainer._updates_completed),
            "rollout_step": "" if rollout_step is None else int(rollout_step),
            "latent_z": "" if latent_z is None else int(latent_z),
            "curriculum_phase": str(info.get("phase", "")),
            "mode": str(getattr(trainer.cfg, "mode", "FIXED_OPPONENT")),
            "map_set": str(info.get("map_set", getattr(trainer.cfg, "map_set", "train"))).lower(),
            "opponent": _opponent_legend(trainer.cfg, info),
            "opponent_id": _opponent_id_csv_from_info(trainer.cfg, info),
            "success": 1 if blue_score > red_score else 0,
            "blue_score": int(blue_score),
            "red_score": int(red_score),
            "win_margin": int(blue_score) - int(red_score),
            "decision_steps": int(er.get("decision_steps", info.get("decision_steps", 0)) or 0),
            "zone_coverage": float(er.get("zone_coverage", 0.0) or 0.0),
            "collision_free_episode": int(er.get("collision_free_episode", 1) or 0),
            "collision_events_per_episode": int(er.get("collision_events_per_episode", 0) or 0),
            "near_misses_per_episode": int(er.get("near_misses_per_episode", 0) or 0),
            "time_to_first_score": er.get("time_to_first_score", ""),
            "mean_inter_robot_dist": er.get("mean_inter_robot_dist", ""),
            "reward_terminal": float(er.get("reward_terminal", info.get("reward_terminal", 0.0)) or 0.0),
            "reward_offense": float(er.get("reward_offense", info.get("reward_offense", 0.0)) or 0.0),
            "reward_pbrs": float(er.get("reward_pbrs", info.get("reward_pbrs", 0.0)) or 0.0),
            "reward_team": float(er.get("reward_team", info.get("reward_team", 0.0)) or 0.0),
            "reward_sparse": float(er.get("reward_sparse", info.get("reward_sparse", 0.0)) or 0.0),
            "reward_sparse_points": float(
                er.get("reward_sparse_points", info.get("reward_sparse_points", info.get("sparse_points", 0.0))) or 0.0
            ),
            "reward_failure": float(er.get("reward_failure", info.get("reward_failure", 0.0)) or 0.0),
            "reward_total": float(er.get("reward_total", info.get("reward_total", 0.0)) or 0.0),
        }
        _write_csv_row(trainer.episode_csv_path, _episode_fieldnames(), row)

    def rollout_episode_summary(self) -> dict[str, Any]:
        trainer = self.trainer
        records = list(trainer._rollout_episode_records)
        n = len(records)
        if n <= 0:
            base: dict[str, Any] = {
                "rollout_episodes": 0,
                "rollout_wins": 0,
                "rollout_losses": 0,
                "rollout_draws": 0,
                "rollout_win_rate": 0.0,
                "rollout_win_margin_mean": 0.0,
                "rollout_blue_score_mean": 0.0,
                "rollout_red_score_mean": 0.0,
            }
        else:
            wins = sum(int(r["success"]) for r in records)
            margins = [int(r["win_margin"]) for r in records]
            losses = sum(1 for m in margins if m < 0)
            draws = sum(1 for m in margins if m == 0)
            base = {
                "rollout_episodes": n,
                "rollout_wins": wins,
                "rollout_losses": losses,
                "rollout_draws": draws,
                "rollout_win_rate": float(wins) / float(n),
                "rollout_win_margin_mean": float(np.mean(margins)),
                "rollout_blue_score_mean": float(np.mean([int(r["blue_score"]) for r in records])),
                "rollout_red_score_mean": float(np.mean([int(r["red_score"]) for r in records])),
            }
        if trainer.use_latent_strategy:
            for z_idx in range(trainer.latent_k):
                z_records = [r for r in records if r.get("latent_z") == z_idx]
                zn = len(z_records)
                base[f"episode_z_{z_idx}_count"] = zn
                if zn <= 0:
                    base[f"episode_z_{z_idx}_win_rate"] = ""
                    base[f"episode_z_{z_idx}_blue_score_mean"] = ""
                    base[f"episode_z_{z_idx}_red_score_mean"] = ""
                    base[f"episode_z_{z_idx}_win_margin_mean"] = ""
                else:
                    base[f"episode_z_{z_idx}_win_rate"] = float(sum(int(r["success"]) for r in z_records)) / float(zn)
                    base[f"episode_z_{z_idx}_blue_score_mean"] = float(np.mean([int(r["blue_score"]) for r in z_records]))
                    base[f"episode_z_{z_idx}_red_score_mean"] = float(np.mean([int(r["red_score"]) for r in z_records]))
                    base[f"episode_z_{z_idx}_win_margin_mean"] = float(np.mean([int(r["win_margin"]) for r in z_records]))
            for o_idx in range(SCRIPTED_OPPONENT_MI_COUNT):
                for z_idx in range(trainer.latent_k):
                    sub = [
                        r
                        for r in records
                        if int(r.get("opponent_id", -1)) == o_idx and r.get("latent_z") == z_idx
                    ]
                    zn = len(sub)
                    base[f"episode_opp{o_idx}_z{z_idx}_count"] = zn
                    if zn <= 0:
                        base[f"episode_opp{o_idx}_z{z_idx}_win_rate"] = ""
                    else:
                        base[f"episode_opp{o_idx}_z{z_idx}_win_rate"] = float(
                            sum(int(r["success"]) for r in sub)
                        ) / float(zn)
        return base

    def rolling_win_rate(self, window: int) -> float:
        recent = list(self.trainer._recent_episode_successes)[-max(1, int(window)):]
        if not recent:
            return 0.0
        return float(sum(recent)) / float(len(recent))

    def write_update_metrics(
        self,
        stats: dict[str, float],
        buffer: TensorDictRolloutBuffer,
    ) -> dict[str, Any]:
        trainer = self.trainer
        if not trainer.metrics_csv_path:
            return {}
        rewards = buffer.fields["rewards"][: int(buffer.pos)].detach().float().reshape(-1)
        returns = buffer.fields["returns"][: int(buffer.pos)].detach().float().reshape(-1)
        values = buffer.fields["values"][: int(buffer.pos)].detach().float().reshape(-1)
        games = trainer._ep_wins + trainer._ep_losses + trainer._ep_draws
        row: dict[str, Any] = {
            "update": trainer._updates_completed,
            "run_id": trainer.run_id,
            "run_pid": trainer.run_pid,
            "timesteps": int(trainer.global_step),
            "episodes_completed": int(trainer._episodes_completed),
            "wins": int(trainer._ep_wins),
            "losses": int(trainer._ep_losses),
            "draws": int(trainer._ep_draws),
            "win_rate": float(trainer._ep_wins) / float(max(1, games)),
            "rolling_win_rate_50ep": self.rolling_win_rate(50),
            "rolling_win_rate_200ep": self.rolling_win_rate(200),
            "rollout_reward_mean": float(rewards.mean().detach().cpu().item()) if rewards.numel() > 0 else 0.0,
            "rollout_reward_std": float(rewards.std(unbiased=False).detach().cpu().item()) if rewards.numel() > 1 else 0.0,
            "rollout_return_mean": float(returns.mean().detach().cpu().item()) if returns.numel() > 0 else 0.0,
            "rollout_return_std": float(returns.std(unbiased=False).detach().cpu().item()) if returns.numel() > 1 else 0.0,
            "explained_variance": self.explained_variance(values, returns),
        }
        row.update(self.rollout_episode_summary())
        if trainer.curriculum is not None:
            row.update(
                {
                    "curriculum_phase": str(trainer.curriculum.phase),
                    "curriculum_phase_idx": int(trainer.curriculum.phase_idx),
                    "curriculum_phase_episodes": int(trainer.curriculum.phase_episode_count),
                    "curriculum_phase_win_rate": float(trainer.curriculum.phase_winrate()),
                }
            )
        for key in (
            "reward_terminal",
            "reward_offense",
            "reward_pbrs",
            "reward_team",
            "reward_sparse",
            "reward_sparse_points",
            "reward_failure",
            "reward_total",
        ):
            vals = buffer.fields[key][: int(buffer.pos)].detach().float().reshape(-1)
            row[f"{key}_mean"] = float(vals.mean().detach().cpu().item()) if vals.numel() > 0 else 0.0
        reward_outcome = float(row.get("reward_terminal_mean", 0.0)) + float(row.get("reward_sparse_mean", 0.0))
        reward_shaping = (
            float(row.get("reward_offense_mean", 0.0))
            + trainer.reward_dense_weight
            * (float(row.get("reward_pbrs_mean", 0.0)) + float(row.get("reward_team_mean", 0.0)))
        )
        reward_failure = float(row.get("reward_failure_mean", 0.0))
        row["reward_outcome_mean"] = reward_outcome
        row["reward_shaping_mean"] = reward_shaping
        row["reward_shaping_to_outcome_abs_ratio"] = abs(reward_shaping) / (abs(reward_outcome) + 1e-6)
        row["reward_shaping_coef"] = float(trainer._reward_shaping_coef())
        row["reward_failure_to_outcome_abs"] = abs(reward_failure) / (abs(reward_outcome) + 1e-6)
        row.update(stats)
        if trainer.use_latent_strategy:
            entropy = float(row.get("strategy_entropy", 0.0) or 0.0)
            row["strategy_entropy_frac"] = entropy / max(1e-6, math.log(max(2, int(trainer.latent_k))))
            z_win_rates: list[float] = []
            for z_idx in range(trainer.latent_k):
                value = row.get(f"episode_z_{z_idx}_win_rate", "")
                if value == "":
                    continue
                z_win_rates.append(float(value))
            row["strategy_wr_spread"] = (
                float(max(z_win_rates) - min(z_win_rates)) if len(z_win_rates) >= 2 else 0.0
            )
        else:
            row["strategy_entropy_frac"] = 0.0
            row["strategy_wr_spread"] = 0.0
        _write_csv_row(
            trainer.metrics_csv_path,
            _update_fieldnames(trainer.use_latent_strategy, trainer.latent_k),
            row,
            legacy_column_fill=_METRICS_CSV_LEGACY_COLUMN_FILL,
        )
        return row

    def print_episode_progress(self, info: dict[str, Any]) -> None:
        trainer = self.trainer
        n = trainer._episodes_completed
        w, l, d = trainer._ep_wins, trainer._ep_losses, trainer._ep_draws
        wr = 100.0 * float(w) / float(max(1, w + l + d))
        mode = str(getattr(trainer.cfg, "mode", "FIXED_OPPONENT"))
        opp = _opponent_legend(trainer.cfg, info)
        print(
            f"[PPO] ep={n} mode={mode} opp={opp} "
            f"W={w} L={l} D={d} WR={wr:.1f}%"
            + (f" phase={trainer.curriculum.phase}" if trainer.curriculum is not None else "")
        )


__all__ = ["TrainingTelemetry"]
