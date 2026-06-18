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

Static collaborators (injected explicitly into the constructor):

- ``cfg`` — raw :class:`PPOConfig` for ad-hoc ``getattr`` lookups
- ``hparams`` — resolved :class:`TrainerHyperparams`
- ``curriculum`` — optional curriculum-state collaborator
- ``reward_shaping_coef`` — zero-arg callable returning the current shaping coef

Shared mutable runtime state (kept on the trainer; reached through the
``runtime`` back-reference): ``global_step``, ``_updates_completed``, and
the episode-scope sub-component ``trainer.episode_stats`` (win/loss/draw
tallies, rollout episode records, and the recent-successes deque).
"""

from __future__ import annotations

import csv
import math
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES
from rl.global_state import coarse_game_phase_from_global_state
from rl.latent_phase_labels import (
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
    _opponent_tag_from_id,
    _update_fieldnames,
    _write_csv_row,
)
from rl.ppo_core import TensorDictRolloutBuffer
from rl.custom_ppo.trainer_config import TrainerHyperparams

if TYPE_CHECKING:
    from rl.custom_ppo.trainer import CustomPPOTrainer


class TrainingTelemetry:
    """Episode/update CSV writers + persistent e3-step CSV file handle.

    The trainer constructs one telemetry instance and holds it as
    ``self.telemetry``. All callers read/write through method calls
    (``self.telemetry.write_episode_metrics(...)``) rather than poking the
    e3 file handle directly.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        hparams: TrainerHyperparams,
        curriculum: Any,
        reward_shaping_coef: Any,
        runtime: "CustomPPOTrainer",
    ) -> None:
        self.cfg = cfg
        self.hparams = hparams
        self.curriculum = curriculum
        self._reward_shaping_coef_fn = reward_shaping_coef
        self.runtime = runtime
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
        context_state: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.e3_step_telemetry_path or not self.hparams.use_latent_strategy:
            return
        path = self.e3_step_telemetry_path
        zt = z_t.detach().cpu().numpy()
        pz = prev_z.detach().cpu().numpy()
        zH = strategy_aux["z_entropy"].detach().cpu().numpy()
        zlog = strategy_aux["z_logits"].detach().cpu().numpy()
        import torch.nn.functional as F
        zprobs = F.softmax(strategy_aux["z_logits"], dim=-1).detach().cpu().numpy()
        ctx_np = context_state.detach().cpu().numpy() if context_state is not None else None
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
        upd = int(self.runtime._updates_completed)
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
                "opponent_id": int(_opponent_id_int_from_info(self.cfg, info)),
                "phase_id": pid,
                "blue_ahead": float(blue_ahead_np[e]),
                "spread_bucket": int(spread_bucket_np[e]),
                "role_bucket": int(role_bucket_np[e]),
                "pressure_bucket": int(pressure_bucket_np[e]),
                "attack_defense_ratio_bucket": int(attack_defense_ratio_bucket_np[e]),
            }
            for j, name in enumerate(BEHAVIOR_TELEMETRY_NAMES):
                row[name] = float(behavior_telemetry_np[e, j])
            
            K = zlog.shape[1]
            for i in range(4):
                if i < K:
                    row[f"qlogit_{i}"] = float(zlog[e, i])
                    row[f"qprob_{i}"] = float(zprobs[e, i])
                else:
                    row[f"qlogit_{i}"] = 0.0
                    row[f"qprob_{i}"] = 0.0
            
            row["strategy_entropy"] = float(zH[e])
            row["strategy_entropy_frac"] = float(zH[e]) / max(1e-6, math.log(max(2, int(self.hparams.latent_k))))
            
            if ctx_np is not None:
                for i in range(ctx_np.shape[1]):
                    row[f"q_phi_context_{i}"] = float(ctx_np[e, i])
                    
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
        runtime = self.runtime
        hparams = self.hparams
        cfg = self.cfg
        if not hparams.episode_csv_path:
            return
        er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
        row = {
            "episode_id": runtime.episode_stats.episodes_completed,
            "run_id": hparams.run_id,
            "run_pid": hparams.run_pid,
            "timesteps": int(timestep),
            "policy_update": int(runtime._updates_completed),
            "rollout_step": "" if rollout_step is None else int(rollout_step),
            "latent_z": "" if latent_z is None else int(latent_z),
            "curriculum_phase": str(info.get("phase", "")),
            "mode": str(getattr(cfg, "mode", "FIXED_OPPONENT")),
            "map_set": str(info.get("map_set", getattr(cfg, "map_set", "train"))).lower(),
            "map_layout": str(er.get("map_layout", info.get("map_layout", "map_a_open"))),
            "map_vertical_mirror": int(er.get("map_vertical_mirror", int(bool(info.get("map_vertical_mirror", False)))) or 0),
            "opponent": _opponent_legend(cfg, info),
            "opponent_id": _opponent_id_csv_from_info(cfg, info),
            "success": 1 if blue_score > red_score else 0,
            "blue_score": int(blue_score),
            "red_score": int(red_score),
            "win_margin": int(blue_score) - int(red_score),
            "decision_steps": int(er.get("decision_steps", info.get("decision_steps", 0)) or 0),
            "zone_coverage": float(er.get("zone_coverage", 0.0) or 0.0),
            "collision_free_episode": int(er.get("collision_free_episode", 1) or 0),
            "collision_events_per_episode": int(er.get("collision_events_per_episode", 0) or 0),
            "obstacle_collision_events_per_episode": int(er.get("obstacle_collision_events_per_episode", 0) or 0),
            "near_misses_per_episode": int(er.get("near_misses_per_episode", 0) or 0),
            "blue_route_upper_crossings": int(er.get("blue_route_upper_crossings", 0) or 0),
            "blue_route_lower_crossings": int(er.get("blue_route_lower_crossings", 0) or 0),
            "red_route_upper_crossings": int(er.get("red_route_upper_crossings", 0) or 0),
            "red_route_lower_crossings": int(er.get("red_route_lower_crossings", 0) or 0),
            "blue_attack_upper_crossings": int(er.get("blue_attack_upper_crossings", 0) or 0),
            "blue_attack_lower_crossings": int(er.get("blue_attack_lower_crossings", 0) or 0),
            "blue_return_upper_crossings": int(er.get("blue_return_upper_crossings", 0) or 0),
            "blue_return_lower_crossings": int(er.get("blue_return_lower_crossings", 0) or 0),
            "blue_intercept_upper_crossings": int(er.get("blue_intercept_upper_crossings", 0) or 0),
            "blue_intercept_lower_crossings": int(er.get("blue_intercept_lower_crossings", 0) or 0),
            "red_attack_upper_crossings": int(er.get("red_attack_upper_crossings", 0) or 0),
            "red_attack_lower_crossings": int(er.get("red_attack_lower_crossings", 0) or 0),
            "red_return_upper_crossings": int(er.get("red_return_upper_crossings", 0) or 0),
            "red_return_lower_crossings": int(er.get("red_return_lower_crossings", 0) or 0),
            "red_intercept_upper_crossings": int(er.get("red_intercept_upper_crossings", 0) or 0),
            "red_intercept_lower_crossings": int(er.get("red_intercept_lower_crossings", 0) or 0),
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
        _write_csv_row(hparams.episode_csv_path, _episode_fieldnames(), row)

    def rollout_episode_summary(self) -> dict[str, Any]:
        hparams = self.hparams
        records = list(self.runtime.episode_stats.rollout_records)
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
        if hparams.use_latent_strategy:
            for z_idx in range(hparams.latent_k):
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
                for z_idx in range(hparams.latent_k):
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
        return self.runtime.episode_stats.rolling_win_rate(window)

    def write_update_metrics(
        self,
        stats: dict[str, float],
        buffer: TensorDictRolloutBuffer,
    ) -> dict[str, Any]:
        runtime = self.runtime
        hparams = self.hparams
        if not hparams.metrics_csv_path:
            return {}
        rewards = buffer.fields["rewards"][: int(buffer.pos)].detach().float().reshape(-1)
        returns = buffer.fields["returns"][: int(buffer.pos)].detach().float().reshape(-1)
        values = buffer.fields["values"][: int(buffer.pos)].detach().float().reshape(-1)
        ep = runtime.episode_stats
        row: dict[str, Any] = {
            "update": runtime._updates_completed,
            "run_id": hparams.run_id,
            "run_pid": hparams.run_pid,
            "timesteps": int(runtime.global_step),
            "episodes_completed": int(ep.episodes_completed),
            "wins": int(ep.wins),
            "losses": int(ep.losses),
            "draws": int(ep.draws),
            "win_rate": ep.cumulative_win_rate,
            "rolling_win_rate_50ep": self.rolling_win_rate(50),
            "rolling_win_rate_200ep": self.rolling_win_rate(200),
            "rollout_reward_mean": float(rewards.mean().detach().cpu().item()) if rewards.numel() > 0 else 0.0,
            "rollout_reward_std": float(rewards.std(unbiased=False).detach().cpu().item()) if rewards.numel() > 1 else 0.0,
            "rollout_return_mean": float(returns.mean().detach().cpu().item()) if returns.numel() > 0 else 0.0,
            "rollout_return_std": float(returns.std(unbiased=False).detach().cpu().item()) if returns.numel() > 1 else 0.0,
            "explained_variance": self.explained_variance(values, returns),
        }
        row.update(self.rollout_episode_summary())
        v6i1_curriculum = getattr(runtime, "v6i1_curriculum", None)
        if v6i1_curriculum is not None:
            row.update(
                {
                    "curriculum_phase": str(v6i1_curriculum.phase),
                    "v6i1_phase_label": str(v6i1_curriculum.phase),
                }
            )
        elif self.curriculum is not None:
            row.update(
                {
                    "curriculum_phase": str(self.curriculum.phase),
                    "curriculum_phase_idx": int(self.curriculum.phase_idx),
                    "curriculum_phase_episodes": int(self.curriculum.phase_episode_count),
                    "curriculum_phase_win_rate": float(self.curriculum.phase_winrate()),
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
            "reward_behavior_contrast",
            "reward_csia",
            "reward_total",
        ):
            vals = buffer.fields[key][: int(buffer.pos)].detach().float().reshape(-1)
            row[f"{key}_mean"] = float(vals.mean().detach().cpu().item()) if vals.numel() > 0 else 0.0
        reward_outcome = float(row.get("reward_terminal_mean", 0.0)) + float(row.get("reward_sparse_mean", 0.0))
        reward_shaping = (
            float(row.get("reward_offense_mean", 0.0))
            + hparams.reward_dense_weight
            * (float(row.get("reward_pbrs_mean", 0.0)) + float(row.get("reward_team_mean", 0.0)))
        )
        reward_failure = float(row.get("reward_failure_mean", 0.0))
        row["reward_outcome_mean"] = reward_outcome
        row["reward_shaping_mean"] = reward_shaping
        row["reward_shaping_to_outcome_abs_ratio"] = abs(reward_shaping) / (abs(reward_outcome) + 1e-6)
        row["reward_shaping_coef"] = float(self._reward_shaping_coef_fn())
        row["reward_failure_to_outcome_abs"] = abs(reward_failure) / (abs(reward_outcome) + 1e-6)
        collector = getattr(runtime, "rollout_collector", None)
        csia_model = getattr(collector, "csia_reward_model", None)
        if csia_model is not None:
            row.update(csia_model.stats())
        row.update(stats)
        if hparams.use_latent_strategy:
            input_contract = runtime.model.input_dim_contract()
            row["z_sensitivity_KL"] = float(
                row.get("policy_z_sensitivity_KL", 0.0) or 0.0
            )
            row["z_sep_JSD"] = float(
                row.get("actor_z_jsd_mean", 0.0) or 0.0
            )
            row["actor_input_dim"] = int(input_contract["actor_input_dim"])
            row["z_embed_dim"] = int(input_contract["actor_z_embed_dim"])
            entropy = float(row.get("strategy_entropy", 0.0) or 0.0)
            row["strategy_entropy_frac"] = entropy / max(1e-6, math.log(max(2, int(hparams.latent_k))))
            z_win_rates: list[float] = []
            for z_idx in range(hparams.latent_k):
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
            hparams.metrics_csv_path,
            _update_fieldnames(hparams.use_latent_strategy, hparams.latent_k),
            row,
            legacy_column_fill=_METRICS_CSV_LEGACY_COLUMN_FILL,
        )
        return row

    def print_update_diagnostics(
        self,
        row: Optional[dict[str, Any]],
        stats: dict[str, Any],
    ) -> None:
        """Print the per-update stdout diagnostic block (``[PPO|diag]`` etc).

        Composes three independent diagnostic prints, each gated on its own
        condition:

        * The big ``[PPO|diag]`` line (+ optional ``[Switch Near]``
          follow-up for latent runs) — only when ``row`` is non-empty,
          i.e. when ``write_update_metrics`` actually wrote a row.
        * ``[PPO|return_norm]`` — when ``hparams.normalize_returns``.
        * ``[PPO|custom]`` verbose line — when ``cfg.verbose_training``.

        ``row`` is the dict returned by :meth:`write_update_metrics` (may
        be empty when nothing was written); ``stats`` is the return of
        ``PPOUpdater.update`` for this rollout.
        """
        runtime = self.runtime
        hparams = self.hparams
        if row:
            z_wr_parts: list[str] = []
            z_occ_parts: list[str] = []
            if hparams.use_latent_strategy:
                for i in range(hparams.latent_k):
                    wr = row.get(f"episode_z_{i}_win_rate", "")
                    occ = row.get(f"strategy_occupancy_{i}", "")
                    z_wr_parts.append("-" if wr == "" else f"{float(wr):.3f}")
                    z_occ_parts.append("-" if occ == "" else f"{float(occ):.3f}")
            z_entropy = float(row.get("strategy_entropy", 0.0) or 0.0)
            z_entropy_frac = float(row.get("strategy_entropy_frac", 0.0) or 0.0)
            z_wr_spread = float(row.get("strategy_wr_spread", 0.0) or 0.0)
            opp_suffix = ""
            if hparams.use_latent_strategy:
                mi_z_o = float(row.get("latent_mi_z_opponent_nats", 0.0) or 0.0)
                mi_z_p = float(row.get("latent_mi_z_phase_nats", 0.0) or 0.0)
                mi_z_y = float(row.get("latent_mi_z_outcome_nats", 0.0) or 0.0)
                mi_z_f = float(row.get("latent_mi_z_flag_state_nats", 0.0) or 0.0)
                opp_diag_bits: list[str] = []
                for o in range(SCRIPTED_OPPONENT_MI_COUNT):
                    occ_o = [
                        float(row.get(f"strategy_occupancy_op{o}_z{k}", 0.0) or 0.0)
                        for k in range(hparams.latent_k)
                    ]
                    wr_o = [
                        row.get(f"episode_opp{o}_z{k}_win_rate", "")
                        for k in range(hparams.latent_k)
                    ]
                    if sum(occ_o) < 1e-9 and all(w == "" for w in wr_o):
                        continue
                    occ_s = ",".join(f"{x:.2f}" for x in occ_o)
                    wr_s = ",".join("-" if w == "" else f"{float(w):.2f}" for w in wr_o)
                    opp_diag_bits.append(
                        f"{_opponent_tag_from_id(o)}:z_occ=[{occ_s}] z_wr=[{wr_s}]"
                    )
                opp_suffix = (
                    f" MI_z_o={mi_z_o:.4f} MI_z_phase={mi_z_p:.4f} "
                    f"MI_z_flag={mi_z_f:.4f} MI_z_outcome={mi_z_y:.4f} | "
                    + " ".join(opp_diag_bits)
                )
            # Active q_phi credit path selection for the printed diag line.
            #
            # There are THREE mutually-relevant code paths that can deliver
            # gradient to q_phi, and only one is "live" for any given
            # preset. Picking the wrong one for the print makes the diag
            # line print structural zeros and looks like q_phi is dead
            # when it is actually learning:
            #
            #   * Per-step strategy-PPO (legacy v1/v2): active when
            #     ``latent_strategy_ppo_coef > 0``. Writes
            #     ``strategy_grad_norm`` / ``strategy_policy_loss`` /
            #     ``strategy_ratio_std`` into the CSV.
            #
            #   * Episode-credit PPO (v3 episode_credit family): active
            #     when ``latent_episode_strategy_ppo=True``. Writes
            #     ``episode_credit_grad_norm`` /
            #     ``latent_episode_pg_loss`` /
            #     ``latent_episode_ratio_std``.
            #
            #   * Arc-credit PPO (v3i19+ / v4i1 / v4i3): active when
            #     ``latent_arc_credit_enabled=True``. The router-only
            #     grad (excluding the V(s,z) baseline head) lives at
            #     ``q_phi_strategy_encoder_grad_norm`` -- this is the
            #     "is z being trained?" signal. The combined
            #     ``q_phi_grad_norm`` is dominated by the baseline value
            #     head early in training and is NOT a good router gauge.
            #     Policy loss / clip stats live at
            #     ``latent_arc_policy_loss`` / ``latent_arc_clipfrac``.
            #
            # Priority for the print: episode_credit > arc_credit >
            # per-step strategy_ppo. If multiple were on at once (no
            # current preset does this) the print picks the highest-
            # priority one; the CSV always has every field.
            episode_credit_on = bool(getattr(self.cfg, "latent_episode_strategy_ppo", False))
            arc_credit_on = bool(getattr(self.cfg, "latent_arc_credit_enabled", False))
            if episode_credit_on:
                qphi_field_label = "qphi_grad_main"
                z_activity_field_label = "z_ratio"
                qphi_grad_val = float(row.get("episode_credit_grad_norm", 0.0) or 0.0)
                z_pi_val = float(row.get("latent_episode_pg_loss", 0.0) or 0.0)
                z_activity_val = float(row.get("latent_episode_ratio_std", 0.0) or 0.0)
            elif arc_credit_on:
                # Router-only grad -- the value-head portion is the baseline,
                # not the routing policy, so combining them masks router
                # starvation behind a noisy baseline loss.
                qphi_field_label = "qphi_grad_arc_router"
                # Arc-credit does not store a ratio_std; clip fraction is
                # the PPO-activity gauge that is actually written for this
                # path, and we relabel so the printed name matches the
                # printed value's semantics (clip fraction != ratio std).
                z_activity_field_label = "z_clipfrac"
                qphi_grad_val = float(
                    row.get("q_phi_strategy_encoder_grad_norm", 0.0) or 0.0
                )
                z_pi_val = float(row.get("latent_arc_policy_loss", 0.0) or 0.0)
                z_activity_val = float(row.get("latent_arc_clipfrac", 0.0) or 0.0)
            else:
                qphi_field_label = "qphi_grad"
                z_activity_field_label = "z_ratio"
                qphi_grad_val = float(row.get("strategy_grad_norm", 0.0) or 0.0)
                z_pi_val = float(row.get("strategy_policy_loss", 0.0) or 0.0)
                z_activity_val = float(row.get("strategy_ratio_std", 0.0) or 0.0)
            print(
                f"[PPO|diag] steps={runtime.global_step} "
                f"ev={row['explained_variance']:.3f} "
                f"v_loss={row['value_loss']:.3f} "
                f"shape/out={row['reward_shaping_mean']:.3f}/{row['reward_outcome_mean']:.3f} "
                f"{qphi_field_label}={qphi_grad_val:.8f} "
                f"lamH={row.get('latent_lam_h', 0.0):.6f} "
                f"zH={z_entropy:.3f}({z_entropy_frac:.2f}) "
                f"z_wr_spread={z_wr_spread:.3f} "
                f"z_aux_ret={float(row.get('strategy_aux_return_loss', row.get('strategy_q_loss', 0.0))):.3f} "
                f"z_pi={z_pi_val:.3f} "
                f"{z_activity_field_label}={z_activity_val:.3f} "
                f"z_occ=[{','.join(z_occ_parts)}] "
                f"z_wr=[{','.join(z_wr_parts)}]"
                f"{opp_suffix}"
            )
            if hparams.use_latent_strategy:
                sw_cap = float(row.get("latent_switch_near_capture_frac", 0.0) or 0.0)
                sw_kill = float(row.get("latent_switch_near_kill_frac", 0.0) or 0.0)
                sw_ret = float(row.get("latent_switch_near_return_frac", 0.0) or 0.0)
                sw_elig = int(float(row.get("latent_switch_near_eligible_count", 0.0) or 0.0))
                cap_n = int(float(row.get("latent_switch_near_capture_count", 0.0) or 0.0))
                kill_n = int(float(row.get("latent_switch_near_kill_count", 0.0) or 0.0))
                ret_n = int(float(row.get("latent_switch_near_return_count", 0.0) or 0.0))
                cap_ev = int(float(row.get("latent_capture_event_count", 0.0) or 0.0))
                kill_ev = int(float(row.get("latent_kill_event_count", 0.0) or 0.0))
                ret_ev = int(float(row.get("latent_return_event_count", 0.0) or 0.0))
                div_role = float(row.get("latent_role_diversity", 0.0) or 0.0)
                div_spread = float(row.get("latent_spread_diversity", 0.0) or 0.0)
                div_pres = float(row.get("latent_pressure_diversity", 0.0) or 0.0)
                div_adr = float(row.get("latent_adr_diversity", 0.0) or 0.0)
                # Tag rows where the eligible denominator is 0 so the
                # fraction is not confused with a "no alignment" signal.
                # Under v5_strict_summer / v5i1 / v5i2 / v5i3 this is the
                # expected state (sample = episode start, refresh disabled).
                elig_tag = "" if sw_elig > 0 else " [N/A: no mid-ep switches]"
                print(
                    f"      [Switch Near] cap={sw_cap:.3f}({cap_n}/{sw_elig}|ev={cap_ev}) "
                    f"kill={sw_kill:.3f}({kill_n}/{sw_elig}|ev={kill_ev}) "
                    f"ret={sw_ret:.3f}({ret_n}/{sw_elig}|ev={ret_ev}){elig_tag} | "
                    f"div_role={div_role:.3f} div_spread={div_spread:.3f} "
                    f"div_pressure={div_pres:.3f} div_adr={div_adr:.3f}"
                )
                # Per-opponent z-WR spread + top per-z behavior spread.
                #
                # The global ``z_wr_spread`` averages over all opponents and
                # tactical contexts; useful per-opponent specialization
                # (e.g. z2 wins against OP5 but loses against OP7) can
                # cancel out. ``[Z Slices]`` surfaces the SLICE max:
                #
                #   * per-opp WR spread = max_z(WR | opp) - min_z(WR | opp)
                #     for each opponent the trainer saw this update.
                #     ``op_max_spread`` is the worst-case slice -- if this
                #     is large while global ``z_wr_spread`` is small, z is
                #     specializing per opponent.
                #
                #   * behavior fingerprint spread = max over dims of
                #     (max_z(mean) - min_z(mean)) / (mean|val| + eps) using
                #     the ``latent_z{k}_behavior_{dim}_mean`` columns. If
                #     this is > ~0.05, at least one behavioral dimension
                #     diverges across z (z is doing something distinct);
                #     if it stays < 0.02, z is decorative.
                op_spreads: list[tuple[int, float]] = []
                for o in range(SCRIPTED_OPPONENT_MI_COUNT):
                    wrs: list[float] = []
                    for k in range(hparams.latent_k):
                        wr = row.get(f"episode_opp{o}_z{k}_win_rate", "")
                        cnt = row.get(f"episode_opp{o}_z{k}_count", "")
                        if wr == "" or cnt in ("", None) or float(cnt or 0) <= 0:
                            continue
                        wrs.append(float(wr))
                    if len(wrs) >= 2:
                        op_spreads.append((o, max(wrs) - min(wrs)))
                if op_spreads:
                    op_max = max(s for _, s in op_spreads)
                    op_mean = sum(s for _, s in op_spreads) / len(op_spreads)
                    per_op = " ".join(
                        f"{_opponent_tag_from_id(o)}={s:.3f}" for o, s in op_spreads
                    )
                else:
                    op_max = 0.0
                    op_mean = 0.0
                    per_op = "-"

                bhv_spreads: list[tuple[str, float, float]] = []
                bhv_keys = [
                    key for key in row.keys()
                    if key.startswith("latent_z0_behavior_") and key.endswith("_mean")
                ]
                for k0 in bhv_keys:
                    dim = k0[len("latent_z0_behavior_") : -len("_mean")]
                    vals: list[float] = []
                    for k in range(hparams.latent_k):
                        v = row.get(f"latent_z{k}_behavior_{dim}_mean", "")
                        if v == "" or v is None:
                            continue
                        try:
                            vals.append(float(v))
                        except (TypeError, ValueError):
                            continue
                    if len(vals) >= 2:
                        rng = max(vals) - min(vals)
                        mn_abs = sum(abs(v) for v in vals) / len(vals)
                        rel = rng / (mn_abs + 1e-8)
                        bhv_spreads.append((dim, rng, rel))
                bhv_spreads.sort(key=lambda t: t[2], reverse=True)
                top_b = bhv_spreads[:3] if bhv_spreads else []
                if top_b:
                    top_b_s = " ".join(
                        f"{d}=rel{r:.3f}/abs{a:.3f}" for d, a, r in top_b
                    )
                    max_rel = top_b[0][2]
                else:
                    top_b_s = "-"
                    max_rel = 0.0

                print(
                    "      [Z Slices] "
                    f"opp_wr_spread_max={op_max:.3f} mean={op_mean:.3f} "
                    f"per_opp=[{per_op}] | "
                    f"behavior_rel_spread_max={max_rel:.3f} "
                    f"top3=[{top_b_s}]"
                )
                print(
                    "      [Actor Z] "
                    f"sensitivity_KL={float(row.get('z_sensitivity_KL', 0.0) or 0.0):.8e} "
                    f"sep_JSD={float(row.get('z_sep_JSD', 0.0) or 0.0):.8e} "
                    f"max_JSD={float(row.get('actor_z_jsd_max', 0.0) or 0.0):.8e} "
                    f"argmax_disagree={float(row.get('actor_z_argmax_disagree', 0.0) or 0.0):.6f} "
                    f"logit_l2={float(row.get('actor_z_logit_l2', 0.0) or 0.0):.8e} "
                    f"actor_input_dim={int(row.get('actor_input_dim', 0) or 0)} "
                    f"z_embed_dim={int(row.get('z_embed_dim', 0) or 0)}"
                )
                print(
                    "      [Actor Z detail] "
                    f"jsd_per_head=[{row.get('actor_z_jsd_per_head', '')}] "
                    f"entropy_by_z=[{row.get('actor_z_entropy_by_z', '')}]"
                )
                v6i1_curriculum = getattr(runtime, "v6i1_curriculum", None)
                if v6i1_curriculum is not None:
                    from rl.custom_ppo.v6i1_phase_runtime import format_v6i1_rollout_stdout_line

                    required_consecutive = int(
                        getattr(self.cfg, "latent_cf_gate_consecutive_updates", 5) or 5
                    )
                    print(
                        format_v6i1_rollout_stdout_line(
                            row,
                            phase=str(v6i1_curriculum.phase),
                            required_consecutive=required_consecutive,
                            gate_protocol=getattr(
                                self.cfg, "gate_protocol_version", None
                            ),
                        ),
                        flush=True,
                    )
                if hparams.latent_sparse_tactical_refresh_enabled:
                    z_change = float(row.get("z_change_count", 0.0) or 0.0)
                    z_dwell = float(row.get("z_dwell_mean", 0.0) or 0.0)
                    refresh_attempt = float(
                        row.get("z_refresh_attempt_count", 0.0) or 0.0
                    )
                    refresh_accept = float(
                        row.get("z_refresh_accept_count", 0.0) or 0.0
                    )
                    refresh_reject = float(
                        row.get("z_refresh_reject_dwell_count", 0.0) or 0.0
                    )
                    reason_interval = float(
                        row.get("z_refresh_reason_interval", 0.0) or 0.0
                    )
                    reason_flag = float(
                        row.get("z_refresh_reason_flag", 0.0) or 0.0
                    )
                    reason_phase = float(
                        row.get("z_refresh_reason_phase", 0.0) or 0.0
                    )
                    reason_score = float(
                        row.get(
                            "z_refresh_reason_score_pressure",
                            0.0,
                        )
                        or 0.0
                    )
                    agreement = float(
                        row.get(
                            "q_phi_argmax_vs_executed_z_agreement",
                            0.0,
                        )
                        or 0.0
                    )
                    print(
                        "      [Sparse Refresh] "
                        f"change={z_change:.0f} dwell={z_dwell:.2f} "
                        f"attempt={refresh_attempt:.0f} accept={refresh_accept:.0f} "
                        f"reject_dwell={refresh_reject:.0f} "
                        f"reason=i:{reason_interval:.0f}/f:{reason_flag:.0f}/"
                        f"p:{reason_phase:.0f}/s:{reason_score:.0f} "
                        f"qarg_exec={agreement:.3f}"
                    )
            # Arc-credit q_phi telemetry: under v3i19+ / v4i1 / v4i3
            # (latent_arc_credit_enabled=True, latent_strategy_ppo_coef=0,
            # latent_episode_strategy_ppo=False) the main learning signal
            # for q_phi flows through ``apply_arc_strategy_ppo`` (PPO over
            # arc-boundary z-decisions with the V(s,z) baseline). Print
            # the dedicated arc-credit diagnostics so the diag stream can
            # answer the question "is the ROUTER actually getting updated
            # (vs the baseline value head soaking up the gradient)?".
            #
            #   * arc_v_loss             : V(s,z) MSE -- decreasing means the
            #                              baseline is converging and freeing
            #                              advantages to update the router.
            #   * qphi_grad_arc_router   : L2 grad onto strategy_encoder
            #                              (the actual policy over z).
            #   * qphi_grad_arc_value    : L2 grad onto the V(s,z) head
            #                              (the baseline, NOT the routing).
            #   * qphi_router_frac       : router / (router + value).
            #                              Tiny means the router is starved
            #                              even though q_phi_grad_norm looks
            #                              big -- watch this number rise as
            #                              the baseline converges.
            #   * arc_pi / arc_clipfrac  : PPO PG loss + clip activity
            #                              specifically for z-decisions.
            #   * arc_adv_mean/std       : arc-credit advantages going INTO
            #                              the router update. Zero std means
            #                              no usable contrast across z.
            #   * arc_n                  : number of completed arcs used in
            #                              the update window. Should be
            #                              roughly n_envs * (rollout_steps /
            #                              latent_resample_every_n).
            if arc_credit_on:
                arc_v_loss = float(row.get("latent_arc_value_loss", 0.0) or 0.0)
                arc_pi = float(row.get("latent_arc_policy_loss", 0.0) or 0.0)
                arc_clipfrac = float(row.get("latent_arc_clipfrac", 0.0) or 0.0)
                arc_kl = float(row.get("latent_arc_approx_kl", 0.0) or 0.0)
                arc_adv_mean = float(row.get("latent_arc_advantage_mean", 0.0) or 0.0)
                arc_adv_std = float(row.get("latent_arc_advantage_std", 0.0) or 0.0)
                arc_n = float(row.get("latent_arc_count", 0.0) or 0.0)
                arc_len = float(row.get("latent_arc_mean_length", 0.0) or 0.0)
                router_g = float(
                    row.get("q_phi_strategy_encoder_grad_norm", 0.0) or 0.0
                )
                value_g = float(row.get("q_phi_value_head_grad_norm", 0.0) or 0.0)
                router_frac = router_g / (router_g + value_g + 1e-8)
                print(
                    "      [Arc Credit] "
                    f"arc_v_loss={arc_v_loss:.3f} "
                    f"arc_pi={arc_pi:.6f} "
                    f"arc_clipfrac={arc_clipfrac:.3f} "
                    f"arc_kl={arc_kl:.4f} "
                    f"arc_adv_mean={arc_adv_mean:+.4f} "
                    f"arc_adv_std={arc_adv_std:.4f} "
                    f"arc_n={arc_n:.0f} "
                    f"arc_len={arc_len:.1f} | "
                    f"qphi_grad_arc_router={router_g:.6f} "
                    f"qphi_grad_arc_value={value_g:.6f} "
                    f"qphi_router_frac={router_frac:.4f}"
                )
            # Episode-credit q_phi telemetry: under v3 (latent_episode_strategy_ppo=True,
            # latent_strategy_ppo_coef=0) the qphi_grad_main / z_pi / z_ratio fields on the
            # main diag line are all structurally zero -- the per-step path is disabled by
            # design. The real q_phi learning signal lives in apply_episode_strategy_ppo and
            # is shown here so the user can verify q_phi is actually getting credit each
            # update.
            if episode_credit_on:
                ep_count = float(row.get("latent_episode_count", 0.0) or 0.0)
                ep_pg = float(row.get("latent_episode_pg_loss", 0.0) or 0.0)
                ep_v = float(row.get("latent_episode_v_loss", 0.0) or 0.0)
                ep_adv_std = float(row.get("latent_episode_adv_std", 0.0) or 0.0)
                ep_kl = float(row.get("latent_episode_approx_kl", 0.0) or 0.0)
                ep_clip = float(row.get("latent_episode_clip_fraction", 0.0) or 0.0)
                ep_ratio_mn = float(row.get("latent_episode_ratio_min", 1.0) or 1.0)
                ep_ratio_mx = float(row.get("latent_episode_ratio_max", 1.0) or 1.0)
                ep_ret_mean = float(row.get("latent_episode_return_mean", 0.0) or 0.0)
                ep_margin = float(row.get("qphi_margin_resample_mean", 0.0) or 0.0)
                ep_ent_res = float(row.get("strategy_entropy_resample_mean", 0.0) or 0.0)
                ep_g = float(row.get("episode_credit_grad_norm", 0.0) or 0.0)
                print(
                    f"      [episode-credit] n_eps={ep_count:.0f} pg={ep_pg:.4f} v={ep_v:.4f} "
                    f"adv_std={ep_adv_std:.3f} kl={ep_kl:.4f} clip={ep_clip:.3f} "
                    f"ratio=[{ep_ratio_mn:.3f},{ep_ratio_mx:.3f}] ret_mean={ep_ret_mean:.3f} "
                    f"margin={ep_margin:.4f} ent_res={ep_ent_res:.4f} grad_norm={ep_g:.8f}"
                )
                # v3d bucket-baseline telemetry. Only printed when the bucket
                # baseline is active (mode != None on the cfg). Reads the same
                # row dict so log scrapers can locate fields by name.
                bucket_mode = getattr(self.cfg, "latent_q_phi_bucket_baseline", None)
                if bucket_mode:
                    b_count = float(row.get("bucket_baseline_count", 0.0) or 0.0)
                    b_fallback = float(row.get("bucket_baseline_fallback_frac", 0.0) or 0.0)
                    b_var = float(row.get("bucket_baseline_var_reduction", 1.0) or 1.0)
                    b_gmean = float(row.get("bucket_baseline_global_mean", 0.0) or 0.0)
                    b_rstd = float(row.get("bucket_baseline_raw_return_std", 0.0) or 0.0)
                    b_astd = float(row.get("bucket_baseline_adv_std", 0.0) or 0.0)
                    # var_reduction < 1.0 means bucket baseline successfully
                    # stratified the return signal -- the lower, the more the
                    # router gradient gains from the per-bucket structure.
                    print(
                        f"      [bucket-baseline] mode={bucket_mode} n_buckets={b_count:.0f} "
                        f"fallback={b_fallback:.3f} var_reduction={b_var:.3f} "
                        f"(R_std={b_rstd:.3f} -> adv_std={b_astd:.3f}) "
                        f"global_mean={b_gmean:.3f}"
                    )
        if hparams.normalize_returns:
            print(
                "[PPO|return_norm] "
                f"update={runtime._updates_completed} "
                f"mean={stats.get('return_norm_mean', 0.0):.4f} "
                f"std={stats.get('return_norm_std', 0.0):.4f} "
                f"count={stats.get('return_norm_count', 0.0):.0f}"
            )
        if bool(getattr(self.cfg, "verbose_training", False)):
            latent_bits = ""
            if hparams.use_latent_strategy:
                latent_bits = (
                    f" z_entropy={stats.get('strategy_entropy', 0.0):.4f} "
                    f"z_persist={stats.get('strategy_persist_loss', 0.0):.4f}"
                )
            print(
                "[PPO|custom] "
                f"steps={runtime.global_step} policy_loss={stats['policy_loss']:.4f} "
                f"value_loss={stats['value_loss']:.4f} approx_kl={stats['approx_kl']:.5f}"
                f"{latent_bits}"
            )

    def print_episode_progress(self, info: dict[str, Any]) -> None:
        runtime = self.runtime
        ep = runtime.episode_stats
        n = ep.episodes_completed
        w, l, d = ep.wins, ep.losses, ep.draws
        wr = 100.0 * ep.cumulative_win_rate
        mode = str(getattr(self.cfg, "mode", "FIXED_OPPONENT"))
        opp = _opponent_legend(self.cfg, info)
        print(
            f"[PPO] ep={n} mode={mode} opp={opp} "
            f"W={w} L={l} D={d} WR={wr:.1f}%"
            + (f" phase={self.curriculum.phase}" if self.curriculum is not None else "")
        )


__all__ = ["TrainingTelemetry"]
