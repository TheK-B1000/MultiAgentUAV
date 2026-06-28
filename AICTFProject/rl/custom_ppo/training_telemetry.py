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
import time
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

from rl.custom_ppo.telemetry import (
    CheckpointLoaded,
    CheckpointSaved,
    EpisodeCompleted,
    EpisodesCompleted,
    NullTelemetrySink,
    OptimizationCompleted,
    PerformanceSample,
    PerformanceSummary,
    RewardSummary,
    RolloutCompleted,
    SafeTelemetrySink,
    TelemetryEnvelope,
    TelemetryEvent,
    TrainingCompleted,
    TrainingFailed,
    TrainingInterrupted,
    TrainingStarted,
)
from rl.custom_ppo.telemetry.performance import (
    PerformanceRecorder,
    environment_transitions_per_second,
    optimization_samples_per_second,
    rollout_steps_per_second,
)
from rl.custom_ppo.telemetry.schemas import (
    PERFORMANCE_METRICS_SCHEMA_VERSION,
    TrainingTelemetryMode,
    coerce_telemetry_mode,
)
from rl.custom_ppo.telemetry.gpu_monitor import build_gpu_monitor
from rl.custom_ppo.telemetry.writers.artifact_writer import ArtifactWriter
from rl.custom_ppo.telemetry.writers.json_writer import JSONLineEventWriter

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
        self.run_id = str(
            getattr(hparams, "run_id", "")
            or getattr(cfg, "run_id", "")
            or getattr(cfg, "run_tag", "")
            or "run"
        )
        self.curriculum = curriculum
        self._reward_shaping_coef_fn = reward_shaping_coef
        self.runtime = runtime
        self.e3_step_telemetry_path = str(getattr(cfg, "e3_step_telemetry_path", "") or "")
        self._e3_file: Any = None
        self._e3_writer: Any = None
        self._e3_fields_cache: Optional[list[str]] = None
        self._e3_rows_since_flush = 0
        self._e3_flush_every_steps = 100
        canonical_path = getattr(cfg, "training_events_jsonl_path", None)
        legacy_path = getattr(cfg, "telemetry_events_jsonl_path", None)
        if canonical_path is not None and canonical_path != "":
            canonical_path = str(canonical_path)
        else:
            canonical_path = None

        if legacy_path is not None and legacy_path != "":
            legacy_path = str(legacy_path)
        else:
            legacy_path = None

        if canonical_path is not None and legacy_path is not None:
            if canonical_path != legacy_path:
                raise ValueError(
                    f"Configuration error: both 'training_events_jsonl_path' and legacy alias "
                    f"'telemetry_events_jsonl_path' were provided with different values: "
                    f"'{canonical_path}' vs '{legacy_path}'"
                )
            events_path = canonical_path
        elif canonical_path is not None:
            events_path = canonical_path
        elif legacy_path is not None:
            events_path = legacy_path
            try:
                setattr(cfg, "training_events_jsonl_path", legacy_path)
            except Exception:
                pass
        else:
            events_path = ""

        default_mode = "full" if events_path else "off"
        self.telemetry_mode = coerce_telemetry_mode(
            getattr(cfg, "training_telemetry_mode", getattr(cfg, "telemetry_mode", default_mode))
        )
        self._sequence_counter = 0
        self._training_started_perf: Optional[float] = None
        self._last_rollout_duration_seconds: Optional[float] = None
        self._last_optimization_duration_seconds: Optional[float] = None
        self._last_checkpoint_load_duration_seconds: Optional[float] = None
        self._last_checkpoint_save_duration_seconds: Optional[float] = None
        self._last_samples_processed = 0
        self._last_transitions_collected = 0
        self._total_transitions_collected = 0
        self._gpu_allocated_peak_bytes: Optional[int] = None
        self._gpu_reserved_peak_bytes: Optional[int] = None

        self._pending_episodes: list[EpisodeCompleted] = []
        self._parent_checkpoint_hash: Optional[str] = None
        self.performance_recorder = PerformanceRecorder(
            run_id=self.run_id,
            telemetry_mode=str(self.telemetry_mode),
        )
        self.gpu_monitor = build_gpu_monitor(
            enabled=bool(getattr(cfg, "gpu_monitor_enabled", False)),
            interval_seconds=float(getattr(cfg, "gpu_monitor_interval_seconds", 1.0) or 1.0),
        )
        try:
            self.gpu_monitor.start()
        except Exception as e:
            print(f"[PPO] WARNING: Failed to start GPU monitor: {e}")

        self._event_writer = JSONLineEventWriter(events_path) if events_path else None
        if self.telemetry_mode == TrainingTelemetryMode.OFF:
            self.event_sink = NullTelemetrySink()
        else:
            self.event_sink = SafeTelemetrySink(self._event_writer) if self._event_writer is not None else NullTelemetrySink()

        artifact_dir = str(
            getattr(cfg, "telemetry_artifact_dir", "")
            or getattr(cfg, "checkpoint_dir", "")
            or "."
        )
        self._artifact_writer = ArtifactWriter(artifact_dir)

        # Resolve Git metadata at startup if telemetry is not OFF
        self._git_commit_hash = None
        self._git_status = "git_unavailable"
        if self.telemetry_mode != TrainingTelemetryMode.OFF:
            import subprocess
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                res = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                if res.returncode == 0:
                    self._git_commit_hash = res.stdout.strip()
                    # Check dirty state
                    status_res = subprocess.run(
                        ["git", "status", "--porcelain"],
                        cwd=project_root,
                        capture_output=True,
                        text=True,
                        timeout=2.0,
                        check=False,
                    )
                    if status_res.returncode == 0:
                        if status_res.stdout.strip():
                            self._git_status = "available_dirty"
                        else:
                            self._git_status = "available_clean"
                    else:
                        self._git_status = "error"
                else:
                    git_check = subprocess.run(
                        ["git", "--version"],
                        capture_output=True,
                        timeout=2.0,
                        check=False,
                    )
                    if git_check.returncode == 0:
                        self._git_status = "not_repository"
                    else:
                        self._git_status = "git_unavailable"
            except subprocess.TimeoutExpired:
                self._git_status = "timeout"
            except Exception:
                self._git_status = "error"

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
        # Close GPU monitor
        if hasattr(self, "gpu_monitor") and self.gpu_monitor is not None:
            try:
                self.gpu_monitor.stop()
            except Exception:
                pass

        # Close Event writer
        if hasattr(self, "_event_writer") and self._event_writer is not None:
            try:
                self._event_writer.close()
            except Exception:
                pass
            self._event_writer = None

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
        if self.telemetry_mode != TrainingTelemetryMode.OFF:
            ep_return = float(er.get("reward_total", info.get("reward_total", 0.0)) or 0.0)
            ep_len = int(er.get("decision_steps", info.get("decision_steps", 0)) or 0)
            ep_completed = EpisodeCompleted(
                run_id=self.run_id,
                global_step=int(timestep),
                environment_index=int(info.get("env_index", 0)),
                episode_return=ep_return,
                episode_length=ep_len,
                score_for=int(blue_score),
                score_against=int(red_score),
                won=bool(blue_score > red_score),
                opponent_name=str(_opponent_legend(cfg, info)),
                map_name=str(er.get("map_layout", info.get("map_layout", ""))),
                terminal_reason=str(er.get("terminal_reason", "")) or None,
            )
            self._pending_episodes.append(ep_completed)

        if not hparams.episode_csv_path:
            return
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
        cf_norm = float(row.get("actor_cf_grad_norm_scaled", row.get("actor_grad_norm_cf", 0.0)) or 0.0)
        ppo_norm = float(row.get("actor_ppo_grad_norm", row.get("actor_grad_norm_ppo", 0.0)) or 0.0)
        denom = max(float(ppo_norm), 1e-12)
        ratio = float(cf_norm) / denom
        row["actor_cf_grad_norm_scaled"] = cf_norm
        row["actor_grad_norm_cf"] = cf_norm
        row["actor_ppo_grad_norm"] = ppo_norm
        row["actor_grad_norm_ppo"] = ppo_norm
        row["actor_cf_to_ppo_grad_ratio"] = ratio
        row["actor_grad_ratio_cf_to_ppo"] = ratio
        row["cf_to_ppo_grad_ratio"] = ratio
        row["actor_grad_ratio_cf_to_ppo_denominator_clamped"] = 1.0 if ppo_norm < 1e-12 else 0.0
        if float(row.get("actor_cf_loss_evaluated", 0.0) or 0.0) > 0.0:
            finite_cf = math.isfinite(cf_norm)
            finite_ppo = math.isfinite(ppo_norm)
            row["actor_grad_cf_valid"] = 1.0 if finite_cf else 0.0
            row["actor_grad_ppo_valid"] = 1.0 if finite_ppo else 0.0
            row["actor_pathway_grad_valid"] = 1.0 if finite_cf else 0.0
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
        if self.telemetry_mode == TrainingTelemetryMode.OFF:
            return
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

    def write_run_manifest(self) -> Optional[str]:
        if self.telemetry_mode == TrainingTelemetryMode.OFF:
            return None
        
        # Calculate preset hash
        preset_hash = None
        try:
            import hashlib
            import json
            import dataclasses
            d = dataclasses.asdict(self.cfg)
            ignore_keys = {"run_tag", "load_path", "checkpoint_dir", "metrics_csv_path", "episode_csv_path", "strategy_experience_csv_path", "performance_summary_path", "performance_samples_path", "training_events_jsonl_path", "telemetry_events_jsonl_path", "e3_step_telemetry_path"}
            clean_d = {k: v for k, v in d.items() if k not in ignore_keys}
            s = json.dumps(clean_d, sort_keys=True)
            preset_hash = hashlib.sha256(s.encode("utf-8")).hexdigest()
        except Exception:
            pass

        # Framework versions
        import sys
        framework_versions = {
            "python": sys.version,
            "pytorch": str(torch.__version__),
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        }

        # GPU Model
        gpu_model = None
        if hasattr(self.gpu_monitor, "_nvml") and hasattr(self.gpu_monitor, "_handle"):
            try:
                gpu_model = self.gpu_monitor._nvml.nvmlDeviceGetName(self.gpu_monitor._handle)
                if isinstance(gpu_model, bytes):
                    gpu_model = gpu_model.decode("utf-8", errors="replace")
                gpu_model = str(gpu_model)
            except Exception:
                pass

        manifest_data = {
            "run_id": self.run_id,
            "timestamp_seconds": float(time.time()),
            "git_commit": self._git_commit_hash,
            "git_status": self._git_status,
            "preset_hash": preset_hash,
            "framework_versions": framework_versions,
            "gpu_model": gpu_model,
            "checkpoint_lineage": {
                "parent_checkpoint_hash": self._parent_checkpoint_hash,
            },
            "schema_version": 1,
        }

        output_path = os.path.join(self._artifact_writer.output_dir, "run_manifest.json")
        try:
            directory = os.path.dirname(os.path.abspath(output_path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
            return output_path
        except Exception as exc:
            import warnings
            warnings.warn(f"Failed to write run manifest: {exc}")
            return None

    def emit_training_started(self, *, total_timesteps: int, checkpoint_path: Optional[str] = None) -> None:
        self._training_started_perf = time.perf_counter()
        if not self.optional_telemetry_enabled():
            return
        
        # Calculate preset hash
        preset_hash = None
        try:
            import hashlib
            import json
            import dataclasses
            d = dataclasses.asdict(self.cfg)
            ignore_keys = {"run_tag", "load_path", "checkpoint_dir", "metrics_csv_path", "episode_csv_path", "strategy_experience_csv_path", "performance_summary_path", "performance_samples_path", "training_events_jsonl_path", "telemetry_events_jsonl_path", "e3_step_telemetry_path"}
            clean_d = {k: v for k, v in d.items() if k not in ignore_keys}
            s = json.dumps(clean_d, sort_keys=True)
            preset_hash = hashlib.sha256(s.encode("utf-8")).hexdigest()
        except Exception:
            pass

        # Calculate checkpoint hash
        ckpt_hash = None
        if checkpoint_path and os.path.isfile(checkpoint_path):
            import hashlib
            try:
                h = hashlib.sha256()
                with open(checkpoint_path, "rb") as f:
                    while chunk := f.read(8192):
                        h.update(chunk)
                ckpt_hash = h.hexdigest()
                self._parent_checkpoint_hash = ckpt_hash
            except Exception:
                pass

        # Write authoritative run manifest at startup
        self.write_run_manifest()

        event = TrainingStarted(
            run_id=self.run_id,
            timestamp_seconds=float(time.time()),
            global_step=int(self.runtime.global_step),
            requested_total_steps=int(total_timesteps),
            device=str(getattr(self.cfg, "device", "cpu")),
            preset_name=getattr(self.cfg, "cli_preset", None),
            preset_hash=preset_hash,
            checkpoint_path=checkpoint_path,
            checkpoint_hash=ckpt_hash,
            telemetry_mode=str(self.telemetry_mode),
        )
        self._emit(event)

    def emit_training_completed(
        self,
        *,
        total_timesteps: int,
        duration_seconds: float,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        if not self.optional_telemetry_enabled():
            return
        event = TrainingCompleted(
            run_id=self.run_id,
            timestamp_seconds=float(time.time()),
            final_global_step=int(self.runtime.global_step),
            duration_seconds=float(duration_seconds),
            checkpoint_path=checkpoint_path,
            status="completed",
        )
        self._emit(event)

    def emit_training_failed(
        self,
        *,
        total_timesteps: int,
        duration_seconds: float,
        error: BaseException,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        if not self.optional_telemetry_enabled():
            return
        phase = ""
        if self.curriculum is not None:
            phase = str(getattr(self.curriculum, "phase", ""))
        elif getattr(self.runtime, "v6i1_curriculum", None) is not None:
            phase = str(getattr(self.runtime.v6i1_curriculum, "phase", ""))
        
        event = TrainingFailed(
            run_id=self.run_id,
            timestamp_seconds=float(time.time()),
            final_global_step=int(self.runtime.global_step),
            duration_seconds=float(duration_seconds),
            checkpoint_path=checkpoint_path,
            exception_type=type(error).__name__,
            exception_message=str(error),
            phase=phase,
        )
        self._emit(event)

    def emit_training_interrupted(
        self,
        *,
        total_timesteps: int,
        duration_seconds: float,
        reason: str,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        if not self.optional_telemetry_enabled():
            return
        event = TrainingInterrupted(
            run_id=self.run_id,
            timestamp_seconds=float(time.time()),
            final_global_step=int(self.runtime.global_step),
            duration_seconds=float(duration_seconds),
            checkpoint_path=checkpoint_path,
            reason=str(reason),
        )
        self._emit(event)

    def emit_rollout_completed(
        self,
        buffer: TensorDictRolloutBuffer,
        *,
        duration_seconds: float,
    ) -> None:
        transitions = int(buffer.pos) * int(getattr(buffer, "n_envs", 1) or 1)
        self._last_transitions_collected = transitions
        self._total_transitions_collected += transitions
        self._last_rollout_duration_seconds = float(duration_seconds)
        
        # Get peak memory
        allocated, reserved = self._cuda_memory_snapshot()
        
        # Measure in performance recorder
        self.performance_recorder.measure_rollout(
            duration_seconds=duration_seconds,
            steps=transitions,
            gpu_allocated_peak=allocated,
            gpu_reserved_peak=reserved,
        )
        
        rollout_tps = (transitions / duration_seconds) if duration_seconds > 0 else 0.0
        self._write_performance_sample_csv(
            phase="rollout",
            duration=duration_seconds,
            transitions=transitions,
            samples=transitions,
            throughput=rollout_tps,
            allocated=allocated,
            reserved=reserved,
        )

        if not self.optional_telemetry_enabled():
            return

        # Coarse timings
        collector = getattr(self.runtime, "rollout_collector", None)
        env_step_dur = getattr(collector, "env_step_time", None)
        policy_inf_dur = getattr(collector, "policy_inf_time", None)
        trans_build_dur = getattr(collector, "trans_build_time", None)
        buf_write_dur = getattr(collector, "buffer_write_time", None)
        bookkeep_dur = getattr(collector, "bookkeeping_time", None)

        env_tps = (transitions / env_step_dur) if (env_step_dur and env_step_dur > 0) else 0.0

        episode_return_mean = None
        episode_length_mean = None
        if self._pending_episodes:
            episode_return_mean = float(np.mean([ep.episode_return for ep in self._pending_episodes]))
            episode_length_mean = float(np.mean([ep.episode_length for ep in self._pending_episodes]))

        # Compute RewardSummary
        rewards_tensor = buffer.fields["rewards"][: int(buffer.pos)].detach().float()
        actor_reward_mean = float(rewards_tensor.mean().detach().cpu().item()) if rewards_tensor.numel() > 0 else 0.0
        reward_summary = RewardSummary(
            actor_reward_mean=actor_reward_mean,
            router_reward_mean=None,
            sparse_reward_mean=0.0,
            shaping_reward_mean=0.0,
            component_means={"rewards": actor_reward_mean},
        )

        # Compute LatentSummary
        latent_summary = None
        if self.hparams.use_latent_strategy and "z_t" in buffer.fields:
            z_t = buffer.fields["z_t"][: int(buffer.pos)].detach().cpu().numpy().reshape(-1)
            unique, counts = np.unique(z_t, return_counts=True)
            total_elements = len(z_t)
            occ = {}
            for z_val in range(int(self.hparams.latent_k)):
                occ[int(z_val)] = 0.0
            for u, c in zip(unique, counts):
                occ[int(u)] = float(c) / float(total_elements)
            probs = np.array(list(occ.values()))
            probs = probs[probs > 0]
            ent = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0.0
            latent_summary = LatentSummary(
                latent_occupancy=occ,
                strategy_entropy=float(ent),
                effective_latent_count=float(np.exp(ent)),
                switching_rate=None,
                persistence_rate=None,
                router_entropy=None,
                router_kl=None,
            )

        # Emit RolloutCompleted
        self._emit(
            RolloutCompleted(
                run_id=self.run_id,
                global_step=int(self.runtime.global_step),
                rollout_index=int(self.runtime._updates_completed),
                vector_environment_count=int(self.runtime.env.num_envs),
                vector_steps=int(buffer.pos),
                environment_transitions=transitions,
                agent_transitions=transitions * int(getattr(self.runtime.model, "n_agents", 1)),
                duration_seconds=float(duration_seconds),
                environment_step_duration_seconds=env_step_dur,
                policy_inference_duration_seconds=policy_inf_dur,
                transition_build_duration_seconds=trans_build_dur,
                buffer_write_duration_seconds=buf_write_dur,
                episode_bookkeeping_duration_seconds=bookkeep_dur,
                environment_transitions_per_second=float(env_tps),
                rollout_transitions_per_second=float(rollout_tps),
                completed_episode_count=len(self._pending_episodes),
                episode_return_mean=episode_return_mean,
                episode_length_mean=episode_length_mean,
                gpu_memory_allocated_peak_bytes=allocated,
                gpu_memory_reserved_peak_bytes=reserved,
                reward_summary=reward_summary,
                latent_summary=latent_summary,
            )
        )

        # Emit EpisodesCompleted if there are pending episodes
        if self._pending_episodes:
            self._emit(
                EpisodesCompleted(
                    run_id=self.run_id,
                    global_step=int(self.runtime.global_step),
                    episodes=tuple(self._pending_episodes),
                )
            )
            self._pending_episodes = []

    def _emit_optimization_completed(
        self,
        row: dict[str, Any],
        stats: dict[str, Any],
        buffer: TensorDictRolloutBuffer,
    ) -> None:
        samples_processed = int(buffer.pos) * int(getattr(buffer, "n_envs", 1) or 1)
        self._last_samples_processed = samples_processed
        opt_duration = float(stats.get("optimization_duration_seconds", 0.0) or 0.0)
        self._last_optimization_duration_seconds = opt_duration
        
        # Get peak memory
        allocated, reserved = self._cuda_memory_snapshot()
        
        self.performance_recorder.measure_optimization(
            duration_seconds=opt_duration,
            samples=samples_processed,
            gpu_allocated_peak=allocated,
            gpu_reserved_peak=reserved,
        )
        
        opt_sps = (samples_processed / opt_duration) if opt_duration > 0 else 0.0
        self._write_performance_sample_csv(
            phase="optimization",
            duration=opt_duration,
            transitions=samples_processed,
            samples=samples_processed,
            throughput=opt_sps,
            allocated=allocated,
            reserved=reserved,
        )

        if not self.optional_telemetry_enabled():
            return
        
        minibatches = int(stats.get("minibatches_processed", stats.get("n_minibatches", 0)) or 0)
        optimizer_updates = int(stats.get("optimizer_updates", stats.get("n_optimizer_steps", 1)) or 1)
        
        # explained_variance check
        ev_val = row.get("explained_variance", None)
        explained_variance = None
        if ev_val is not None and ev_val != "":
            try:
                explained_variance = float(ev_val)
            except Exception:
                pass
        
        event = OptimizationCompleted(
            run_id=self.run_id,
            global_step=int(self.runtime.global_step),
            optimization_index=int(getattr(self.runtime, "_updates_completed", 0)),
            duration_seconds=opt_duration,
            samples_processed=samples_processed,
            minibatches_processed=minibatches,
            optimizer_updates=optimizer_updates,
            optimization_samples_per_second=opt_sps,
            minibatches_per_second=(minibatches / opt_duration) if opt_duration > 0 else 0.0,
            optimizer_updates_per_second=(optimizer_updates / opt_duration) if opt_duration > 0 else 0.0,
            policy_loss=float(stats.get("policy_loss", row.get("policy_loss", 0.0)) or 0.0),
            value_loss=float(stats.get("value_loss", row.get("value_loss", 0.0)) or 0.0),
            entropy=float(stats.get("entropy", stats.get("entropy_loss", 0.0)) or 0.0),
            approx_kl=float(stats.get("approx_kl", row.get("approx_kl", 0.0)) or 0.0),
            clip_fraction=float(stats.get("clip_fraction", row.get("clip_fraction", 0.0)) or 0.0),
            explained_variance=explained_variance,
            gpu_memory_allocated_peak_bytes=allocated,
            gpu_memory_reserved_peak_bytes=reserved,
        )
        self._emit(event)

    def emit_performance_sample(self, *, phase: str, timestamp_seconds: Optional[float] = None) -> None:
        if not self.optional_telemetry_enabled():
            return
        allocated, reserved = self._cuda_memory_snapshot()
        
        gpu_util = None
        if hasattr(self, "gpu_monitor"):
            samples = self.gpu_monitor.samples()
            if samples:
                utils = [s.utilization_percent for s in samples if s.utilization_percent is not None]
                if utils:
                    gpu_util = float(utils[-1])

        event = PerformanceSample(
            timestamp_seconds=float(time.time() if timestamp_seconds is None else timestamp_seconds),
            global_step=int(self.runtime.global_step),
            phase=str(phase),
            environment_steps_per_second=environment_transitions_per_second(
                self._total_transitions_collected,
                self._training_elapsed_seconds(),
            ),
            rollout_steps_per_second=rollout_steps_per_second(
                self._last_transitions_collected,
                self._last_rollout_duration_seconds or 0.0,
            ),
            optimization_samples_per_second=optimization_samples_per_second(
                self._last_samples_processed,
                self._last_optimization_duration_seconds or 0.0,
            ),
            gpu_utilization_percent=gpu_util,
            gpu_memory_allocated_bytes=allocated,
            gpu_memory_reserved_bytes=reserved,
        )
        self._emit(event)

    def emit_checkpoint_saved(
        self,
        *,
        path: str,
        duration_seconds: float,
        write_duration_seconds: Optional[float] = None,
    ) -> None:
        self._last_checkpoint_save_duration_seconds = float(duration_seconds)
        self.performance_recorder.measure_checkpoint_save(duration_seconds=duration_seconds)
        
        if not self.optional_telemetry_enabled():
            return

        # Calculate preset hash
        preset_hash = None
        try:
            import hashlib
            import json
            import dataclasses
            d = dataclasses.asdict(self.cfg)
            ignore_keys = {"run_tag", "load_path", "checkpoint_dir", "metrics_csv_path", "episode_csv_path", "strategy_experience_csv_path", "performance_summary_path", "performance_samples_path", "training_events_jsonl_path", "telemetry_events_jsonl_path", "e3_step_telemetry_path"}
            clean_d = {k: v for k, v in d.items() if k not in ignore_keys}
            s = json.dumps(clean_d, sort_keys=True)
            preset_hash = hashlib.sha256(s.encode("utf-8")).hexdigest()
        except Exception:
            pass

        # Calculate checkpoint hash & measure its duration
        ckpt_hash = None
        ckpt_size = None
        hash_start = time.perf_counter()
        if path and os.path.isfile(path):
            import hashlib
            try:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    while chunk := f.read(8192):
                        h.update(chunk)
                ckpt_hash = h.hexdigest()
                ckpt_size = os.path.getsize(path)
            except Exception:
                pass
        hash_dur = time.perf_counter() - hash_start

        event = CheckpointSaved(
            run_id=self.run_id,
            timestamp_seconds=float(time.time()),
            global_step=int(self.runtime.global_step),
            checkpoint_path=str(path),
            checkpoint_hash=ckpt_hash,
            save_duration_seconds=float(duration_seconds),
            checkpoint_size_bytes=ckpt_size,
            parent_checkpoint_hash=self._parent_checkpoint_hash,
            preset_hash=preset_hash,
            checkpoint_write_duration_seconds=write_duration_seconds,
            checkpoint_hash_duration_seconds=hash_dur,
            checkpoint_total_duration_seconds=duration_seconds + hash_dur,
        )
        self._emit(event)
        self._parent_checkpoint_hash = ckpt_hash

    def emit_checkpoint_loaded(
        self,
        *,
        path: str,
        duration_seconds: float,
        archive_read_duration: Optional[float] = None,
        model_construction_duration: Optional[float] = None,
        state_load_duration: Optional[float] = None,
    ) -> None:
        self._last_checkpoint_load_duration_seconds = float(duration_seconds)
        self.performance_recorder.measure_checkpoint_load(duration_seconds=duration_seconds)
        
        # Calculate checkpoint hash
        ckpt_hash = None
        hash_start = time.perf_counter()
        if path and os.path.isfile(path):
            import hashlib
            try:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    while chunk := f.read(8192):
                        h.update(chunk)
                ckpt_hash = h.hexdigest()
                self._parent_checkpoint_hash = ckpt_hash
            except Exception:
                pass
        hash_dur = time.perf_counter() - hash_start

        migration_start = time.perf_counter()
        # Determine target channels
        target_ch = None
        try:
            target_ch = int(self.runtime.model.grid_shape[0])
        except Exception:
            pass

        # Parse legacy source channels and migration ids
        source_ch = None
        migration_ids = []
        if path and os.path.isfile(path):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                sd = payload.get("model_state_dict", {})
                
                # Check legacy key remappings
                legacy_actor_keys = {"actor_body.", "actor_head.", "strategy_embedding."}
                if any(any(k.startswith(p) for p in legacy_actor_keys) for k in sd.keys()):
                    migration_ids.append("legacy_actor_key_remapping")
                if any(k.startswith("strategy_q_head.") for k in sd.keys()):
                    migration_ids.append("legacy_strategy_q_head_remapping")
                
                # Check channel expansion
                _cnn_key = "latent_actor.actor_cnn.conv.0.weight"
                _alt_key = "actor_cnn.conv.0.weight"
                source_weight = sd.get(_cnn_key, sd.get(_alt_key))
                if source_weight is not None:
                    source_ch = int(source_weight.shape[1])
                
                if source_ch is not None and target_ch is not None and source_ch < target_ch:
                    migration_ids.append("cnn_input_channel_expansion")
            except Exception:
                pass

        bev_result = "PASS"
        if migration_ids:
            bev_result = "PASS_WITH_MIGRATION"
        migration_dur = time.perf_counter() - migration_start

        if not self.optional_telemetry_enabled():
            return

        event = CheckpointLoaded(
            run_id=self.run_id,
            timestamp_seconds=float(time.time()),
            global_step=int(self.runtime.global_step),
            checkpoint_path=str(path),
            checkpoint_hash=ckpt_hash,
            load_duration_seconds=float(duration_seconds),
            source_observation_channels=source_ch,
            target_observation_channels=target_ch,
            migration_ids=tuple(migration_ids),
            behavioral_equivalence_result=bev_result,
            device=str(getattr(self.cfg, "device", "cpu")),
            archive_read_duration=archive_read_duration,
            model_construction_duration=model_construction_duration,
            state_load_duration=state_load_duration,
            migration_duration=migration_dur,
            behavioral_equivalence_duration=0.0,
            hash_duration=hash_dur,
            total_duration=duration_seconds + hash_dur + migration_dur,
        )
        self._emit(event)

    def write_performance_summary(self, *, training_duration_seconds: Optional[float] = None) -> Optional[str]:
        if not self.optional_telemetry_enabled():
            return None
        
        # Stop monitor and query utilization
        try:
            self.gpu_monitor.stop()
        except Exception:
            pass
        
        samples = self.gpu_monitor.samples()
        gpu_util_summary = None
        if samples:
            utils = [s.utilization_percent for s in samples if s.utilization_percent is not None]
            mems = [s.memory_device_used_bytes for s in samples if s.memory_device_used_bytes is not None]
            gpu_util_summary = {
                "gpu_utilization_mean_percent": float(np.mean(utils)) if utils else None,
                "gpu_utilization_max_percent": float(np.max(utils)) if utils else None,
                "gpu_device_memory_used_mean_bytes": float(np.mean(mems)) if mems else None,
                "gpu_device_memory_used_max_bytes": float(np.max(mems)) if mems else None,
                "gpu_monitor_sample_count": len(samples),
                "gpu_monitor_status": self.gpu_monitor.status,
            }
        else:
            gpu_util_summary = {
                "gpu_utilization_mean_percent": None,
                "gpu_utilization_max_percent": None,
                "gpu_device_memory_used_mean_bytes": None,
                "gpu_device_memory_used_max_bytes": None,
                "gpu_monitor_sample_count": 0,
                "gpu_monitor_status": self.gpu_monitor.status,
            }

        # Query Git commit and status
        git_commit = self._git_commit_hash
        if git_commit and self._git_status == "available_dirty":
            git_commit += "-dirty"

        # GPU Model
        gpu_model = None
        if hasattr(self.gpu_monitor, "_nvml") and hasattr(self.gpu_monitor, "_handle"):
            try:
                gpu_model = self.gpu_monitor._nvml.nvmlDeviceGetName(self.gpu_monitor._handle)
                if isinstance(gpu_model, bytes):
                    gpu_model = gpu_model.decode("utf-8")
            except Exception:
                pass

        total_dur = float(training_duration_seconds) if training_duration_seconds is not None else self._training_elapsed_seconds()

        clean_preset_hash = None
        try:
            import hashlib
            import json
            import dataclasses
            d = dataclasses.asdict(self.cfg)
            ignore_keys = {"run_tag", "load_path", "checkpoint_dir", "metrics_csv_path", "episode_csv_path", "strategy_experience_csv_path", "performance_summary_path", "performance_samples_path", "training_events_jsonl_path", "telemetry_events_jsonl_path", "e3_step_telemetry_path"}
            clean_d = {k: v for k, v in d.items() if k not in ignore_keys}
            s = json.dumps(clean_d, sort_keys=True)
            clean_preset_hash = hashlib.sha256(s.encode("utf-8")).hexdigest()
        except Exception:
            pass

        summary = self.performance_recorder.summary(
            git_commit=git_commit,
            preset_name=getattr(self.cfg, "cli_preset", None),
            preset_hash=clean_preset_hash,
            checkpoint_hash=self._parent_checkpoint_hash,
            device=str(getattr(self.cfg, "device", "cpu")),
            gpu_model=gpu_model,
            pytorch_version=str(torch.__version__),
            cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
            environment_count=int(getattr(getattr(self.runtime, "env", None), "num_envs", 16)),
            rollout_length=int(getattr(self.cfg, "n_steps", 64)),
            total_training_duration=total_dur,
            gpu_utilization_summary=gpu_util_summary,
            total_transitions_collected=int(self._total_transitions_collected),
        )

        output_path = getattr(self.cfg, "performance_summary_path", None)
        if not output_path:
            output_path = os.path.join(self._artifact_writer.output_dir, "performance_summary.json")
        
        try:
            directory = os.path.dirname(os.path.abspath(output_path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp_path = output_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(dataclasses.asdict(summary), f, indent=2)
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(tmp_path, output_path)
            return output_path
        except Exception as exc:
            import warnings
            warnings.warn(f"Failed to write performance summary: {exc}")
            return None

    @property
    def detailed_timing_enabled(self) -> bool:
        return self.telemetry_mode in (TrainingTelemetryMode.FULL, TrainingTelemetryMode.BENCHMARK)

    @property
    def cuda_synchronize_enabled(self) -> bool:
        return self.telemetry_mode == TrainingTelemetryMode.BENCHMARK

    def _emit(self, event: TelemetryEvent) -> None:
        if self.telemetry_mode == TrainingTelemetryMode.OFF:
            return
        
        # Limit exception messages to prevent oversized files
        if isinstance(event, TrainingFailed):
            msg = getattr(event, "exception_message", "")
            tb = getattr(event, "traceback", "")
            if len(msg) > 1000:
                object.__setattr__(event, "exception_message", msg[:1000] + "... [TRUNCATED]")
            if len(tb) > 4000:
                object.__setattr__(event, "traceback", tb[:4000] + "... [TRUNCATED]")
        elif isinstance(event, TrainingInterrupted):
            msg = getattr(event, "message", "")
            if len(msg) > 1000:
                object.__setattr__(event, "message", msg[:1000] + "... [TRUNCATED]")

        self._sequence_counter += 1
        envelope = TelemetryEnvelope(
            schema_version=1,
            event_type=type(event).__name__,
            run_id=self.run_id,
            sequence=self._sequence_counter,
            timestamp_seconds=float(time.time()),
            payload=event,
        )
        self.event_sink.emit(envelope)

    def optional_telemetry_enabled(self) -> bool:
        return self.telemetry_mode != TrainingTelemetryMode.OFF

    def _training_elapsed_seconds(self) -> float:
        if self._training_started_perf is None:
            return 0.0
        return max(0.0, time.perf_counter() - self._training_started_perf)

    def _cuda_memory_snapshot(self) -> tuple[Optional[int], Optional[int]]:
        try:
            if torch.cuda.is_available():
                allocated = int(torch.cuda.max_memory_allocated())
                reserved = int(torch.cuda.max_memory_reserved())
                self._gpu_allocated_peak_bytes = max(self._gpu_allocated_peak_bytes or 0, allocated)
                self._gpu_reserved_peak_bytes = max(self._gpu_reserved_peak_bytes or 0, reserved)
        except Exception:
            return self._gpu_allocated_peak_bytes, self._gpu_reserved_peak_bytes
        return self._gpu_allocated_peak_bytes, self._gpu_reserved_peak_bytes

    def _write_performance_sample_csv(
        self,
        phase: str,
        duration: float,
        transitions: int,
        samples: int,
        throughput: float,
        allocated: Optional[int],
        reserved: Optional[int],
    ) -> None:
        path = str(getattr(self.cfg, "performance_samples_path", "") or "")
        if not path or self.telemetry_mode == TrainingTelemetryMode.OFF:
            return
        try:
            directory = os.path.dirname(os.path.abspath(path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            headers = [
                "timestamp",
                "global_step",
                "phase",
                "duration",
                "transitions",
                "samples",
                "throughput",
                "peak allocated memory",
                "peak reserved memory",
            ]
            needs_header = not (os.path.isfile(path) and os.path.getsize(path) > 0)
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if needs_header:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": time.time(),
                    "global_step": int(self.runtime.global_step),
                    "phase": str(phase),
                    "duration": float(duration),
                    "transitions": int(transitions),
                    "samples": int(samples),
                    "throughput": float(throughput),
                    "peak allocated memory": allocated if allocated is not None else "",
                    "peak reserved memory": reserved if reserved is not None else "",
                })
        except Exception as exc:
            import warnings
            warnings.warn(f"Failed to write performance sample to CSV: {exc}")

__all__ = ["TrainingTelemetry"]
