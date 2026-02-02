from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from game_field import make_game_field
from ctf_sb3_env import CTFGameFieldSB3Env
from rl.common import set_global_seed
from rl.curriculum import (
    CurriculumConfig,
    CurriculumController,
    CurriculumControllerConfig,
    CurriculumState,
    STRESS_BY_PHASE,
)
from rl.league import EloLeague, OpponentSpec
from rl.episode_result import parse_episode_result, EpisodeSummary
from config import MAP_NAME, MAP_PATH


class TrainMode(str, Enum):
    CURRICULUM_LEAGUE = "CURRICULUM_LEAGUE"
    CURRICULUM_NO_LEAGUE = "CURRICULUM_NO_LEAGUE"  # OLD baseline: OP1 -> OP2 -> OP3, no league
    FIXED_OPPONENT = "FIXED_OPPONENT"
    SELF_PLAY = "SELF_PLAY"


@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 2_000_000
    n_envs: int = 4
    n_steps: int = 2048
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    device: str = "cpu"

    checkpoint_dir: str = "checkpoints_sb3"
    run_tag: str = "ppo_curriculum_old_v2"
    save_every_steps: int = 50_000
    eval_every_steps: int = 25_000
    eval_episodes: int = 6
    snapshot_every_episodes: int = 100
    league_max_snapshots: int = 25  # cap league snapshots; delete oldest to save space
    enable_tensorboard: bool = False
    enable_checkpoints: bool = False
    enable_eval: bool = False

    max_decision_steps: int = 900
    op3_gate_tag: str = "OP3_HARD"

    mode: str = TrainMode.CURRICULUM_NO_LEAGUE.value
    fixed_opponent_tag: str = "OP3"
    self_play_use_latest_snapshot: bool = True
    self_play_snapshot_every_episodes: int = 25
    self_play_max_snapshots: int = 25

    # Action execution noise (reliability metrics)
    action_flip_prob: float = 0.0


def _make_env_fn(cfg: PPOConfig, *, default_opponent: Tuple[str, str], rank: int) -> Any:
    def _fn():
        np.random.seed(int(cfg.seed) + int(rank))
        torch.manual_seed(int(cfg.seed) + int(rank))
        env = CTFGameFieldSB3Env(
            make_game_field_fn=lambda: make_game_field(
                map_name=MAP_NAME or None,
                map_path=MAP_PATH or None,
            ),
            max_decision_steps=cfg.max_decision_steps,
            enforce_masks=True,
            seed=int(cfg.seed) + int(rank),
            include_mask_in_obs=True,
            default_opponent_kind=default_opponent[0],
            default_opponent_key=default_opponent[1],
            ppo_gamma=cfg.gamma,
            action_flip_prob=getattr(cfg, "action_flip_prob", 0.0),
        )
        return env
    return _fn


class MaskedMultiInputPolicy(MultiInputActorCriticPolicy):
    """
    Apply action masks to discrete macro + target logits (MultiDiscrete).
    Mask is expected in obs["mask"] as shape [2 * (n_macros + n_targets)].
    """

    def _apply_action_mask(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return logits
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask.float()

        # Determine action dimensions
        if hasattr(self.action_dist, "action_dims"):
            dims = list(self.action_dist.action_dims)
        else:
            dims = list(getattr(self.action_space, "nvec", []))
        if not dims:
            return logits

        n_macros = int(dims[0]) if len(dims) > 0 else 0
        n_targets = int(dims[1]) if len(dims) > 1 else 0
        if n_macros <= 0:
            return logits

        # Expect mask layout: [macro0, targets0, macro1, targets1]
        expected = 2 * (n_macros + n_targets)
        if mask.shape[1] < expected:
            pad = torch.ones((mask.shape[0], expected - mask.shape[1]), device=mask.device)
            mask = torch.cat([mask, pad], dim=1)

        # Build full mask for all action components (macro + target)
        full_mask = []
        offset = 0
        for i, dim in enumerate(dims):
            if i in (0, 2):  # macro for agent0 and agent1
                m = mask[:, offset: offset + n_macros]
                offset += n_macros
                full_mask.append(m)
            elif i in (1, 3):  # target for agent0 and agent1
                if n_targets > 0:
                    m = mask[:, offset: offset + n_targets]
                    offset += n_targets
                    full_mask.append(m)
                else:
                    full_mask.append(torch.ones((mask.shape[0], int(dim)), device=mask.device))
            else:
                full_mask.append(torch.ones((mask.shape[0], int(dim)), device=mask.device))

        mask_cat = torch.cat(full_mask, dim=1)
        invalid = (mask_cat <= 0.0)
        return logits.masked_fill(invalid, -1e8)

    def get_distribution(self, obs: Dict[str, torch.Tensor]):
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        if isinstance(obs, dict) and "mask" in obs:
            logits = self._apply_action_mask(logits, obs["mask"])
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        if isinstance(obs, dict) and "mask" in obs:
            logits = self._apply_action_mask(logits, obs["mask"])
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob


class LeagueCallback(BaseCallback):
    def __init__(
        self,
        *,
        cfg: PPOConfig,
        league: EloLeague,
        curriculum: CurriculumState,
        controller: CurriculumController,
    ) -> None:
        super().__init__(verbose=1)
        self.cfg = cfg
        self.league = league
        self.curriculum = curriculum
        self.controller = controller
        self.episode_idx = 0
        self.league_mode = False
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self._league_max_snapshots = max(0, int(getattr(cfg, "league_max_snapshots", 25)))

    def _enforce_league_snapshot_limit(self) -> None:
        """Delete oldest league snapshots when over cap to save disk space."""
        if self._league_max_snapshots <= 0:
            return
        while len(self.league.snapshots) > self._league_max_snapshots:
            oldest = self.league.snapshots.pop(0)
            try:
                if oldest and os.path.exists(oldest):
                    os.remove(oldest)
                    if self.verbose:
                        print(f"[League] deleted old snapshot: {oldest}")
            except Exception as exc:
                print(f"[WARN] league snapshot cleanup failed: {exc}")

    def _select_next_opponent(self) -> OpponentSpec:
        return self.controller.select_opponent(self.curriculum.phase, league_mode=self.league_mode)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            summary = parse_episode_result(info)
            if summary is None:
                continue

            self.episode_idx += 1
            blue_score = summary.blue_score
            red_score = summary.red_score
            win_by = summary.win_by

            if blue_score > red_score:
                result = "WIN"
                actual = 1.0
                win = True
                self.win_count += 1
            elif blue_score < red_score:
                result = "LOSS"
                actual = 0.0
                win = False
                self.loss_count += 1
            else:
                result = "DRAW"
                actual = 0.5
                win = False
                self.draw_count += 1

            opp_key = summary.opponent_key()
            self.league.update_elo(opp_key, actual)
            self.controller.record_result(opp_key, actual)

            phase = self.curriculum.phase
            self.curriculum.phase_episode_count += 1
            self.curriculum.record_result(phase, actual)

            is_scripted = opp_key.startswith("SCRIPTED:")
            if is_scripted:
                opp_rating = self.league.get_rating(opp_key)
                if self.curriculum.advance_if_ready(
                    learner_rating=self.league.learner_rating,
                    opponent_rating=opp_rating,
                    win_by=win_by,
                ):
                    phase = self.curriculum.phase

            if phase == "OP3" and is_scripted:
                min_eps = int(self.curriculum.config.min_episodes.get("OP3", 0))
                min_wr = float(self.curriculum.config.min_winrate.get("OP3", 0.0))
                req_win_by = int(self.curriculum.config.required_win_by.get("OP3", 0))
                meets_eps = self.curriculum.phase_episode_count >= min_eps
                meets_wr = self.curriculum.phase_winrate("OP3") >= min_wr
                meets_score = True if req_win_by <= 0 else (win_by >= req_win_by)
                gate_tag = str(self.cfg.op3_gate_tag).upper()
                gate_ok = opp_key.endswith(f":{gate_tag}")
                if self.curriculum.config.switch_to_league_after_op3_win and win and gate_ok:
                    self.league_mode = True
                elif meets_eps and meets_wr and meets_score and gate_ok:
                    self.league_mode = True

            if self.verbose:
                mode = "LEAGUE" if self.league_mode else "CURR"
                base = (
                    f"[PPO|{mode}] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} phase={phase} opp={opp_key} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )
                if self.league_mode:
                    base = f"{base} elo={self.league.learner_rating:.1f}"
                print(base)

            self.logger.record("curr/episode", self.episode_idx)
            self.logger.record("curr/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("curr/draw_rate", self.draw_count / max(1, self.episode_idx))
            self.logger.record("curr/phase_idx", float(self.curriculum.phase_idx))
            self.logger.record("curr/league_mode", float(self.league_mode))
            if self.league_mode:
                self.logger.record("league/elo", float(self.league.learner_rating))

            if (self.episode_idx % int(self.cfg.snapshot_every_episodes)) == 0:
                self._enforce_league_snapshot_limit()
                prefix = f"{self.cfg.run_tag}_league_snapshot"
                path = os.path.join(self.cfg.checkpoint_dir, f"{prefix}_ep{self.episode_idx:06d}")
                try:
                    self.model.save(path)
                except Exception as exc:
                    print(f"[WARN] snapshot save failed: {exc}")
                else:
                    self.league.add_snapshot(path + ".zip")
                    self._enforce_league_snapshot_limit()

            next_opp = self._select_next_opponent()
            env = self.model.get_env()
            if env is not None:
                env.env_method("set_next_opponent", next_opp.kind, next_opp.key)
                env.env_method("set_phase", self.curriculum.phase)
                env.env_method("set_league_mode", self.league_mode)

        return True


class CurriculumNoLeagueCallback(BaseCallback):
    """OP1 -> OP2 -> OP3 curriculum only; no league, no species, no snapshots."""

    def __init__(self, *, cfg: PPOConfig, curriculum: CurriculumState) -> None:
        super().__init__(verbose=1)
        self.cfg = cfg
        self.curriculum = curriculum
        self.episode_idx = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            summary = parse_episode_result(info)
            if summary is None:
                continue

            self.episode_idx += 1
            blue_score = summary.blue_score
            red_score = summary.red_score
            win_by = summary.win_by

            if blue_score > red_score:
                result = "WIN"
                actual = 1.0
                self.win_count += 1
            elif blue_score < red_score:
                result = "LOSS"
                actual = 0.0
                self.loss_count += 1
            else:
                result = "DRAW"
                actual = 0.5
                self.draw_count += 1

            phase = self.curriculum.phase
            self.curriculum.phase_episode_count += 1
            self.curriculum.record_result(phase, actual)

            self.curriculum.advance_if_ready(
                learner_rating=1300.0,
                opponent_rating=1200.0,
                win_by=win_by,
            )
            phase = self.curriculum.phase

            if self.verbose:
                print(
                    f"[PPO|CURR_NO_LEAGUE] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} phase={phase} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )

            self.logger.record("curr_noleague/episode", self.episode_idx)
            self.logger.record("curr_noleague/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("curr_noleague/draw_rate", self.draw_count / max(1, self.episode_idx))
            self.logger.record("curr_noleague/phase_idx", float(self.curriculum.phase_idx))

            env = self.model.get_env()
            if env is not None:
                env.env_method("set_next_opponent", "SCRIPTED", phase)
                env.env_method("set_phase", phase)

        return True


class SelfPlayCallback(BaseCallback):
    """Self-play with rolling snapshot pool: counter resets to 1 when at max and old are deleted."""

    def __init__(self, *, cfg: PPOConfig, league: EloLeague) -> None:
        super().__init__(verbose=1)
        self.cfg = cfg
        self.league = league
        self.episode_idx = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self._max_snapshots = max(0, int(getattr(cfg, "self_play_max_snapshots", 0)))
        # Rolling slot index 1..max_snapshots for snapshot filenames (resets when at max)
        self._snapshot_roll_index = 0

    def _enforce_snapshot_limit(self) -> None:
        if self._max_snapshots <= 0:
            return
        while len(self.league.snapshots) > self._max_snapshots:
            oldest = self.league.snapshots.pop(0)
            try:
                if oldest and os.path.exists(oldest):
                    os.remove(oldest)
            except Exception as exc:
                print(f"[WARN] snapshot cleanup failed: {exc}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            summary = parse_episode_result(info)
            if summary is None:
                continue

            self.episode_idx += 1
            blue_score = summary.blue_score
            red_score = summary.red_score
            if blue_score > red_score:
                result = "WIN"
                self.win_count += 1
            elif blue_score < red_score:
                result = "LOSS"
                self.loss_count += 1
            else:
                result = "DRAW"
                self.draw_count += 1

            if (self.episode_idx % int(self.cfg.self_play_snapshot_every_episodes)) == 0:
                self._enforce_snapshot_limit()
                # Rolling counter 1..max_snapshots for filenames (goes back to 1 when at max)
                max_s = max(1, self._max_snapshots)
                self._snapshot_roll_index = (self._snapshot_roll_index % max_s) + 1
                slot = self._snapshot_roll_index
                prefix = f"{self.cfg.run_tag}_selfplay_snapshot"
                path = os.path.join(self.cfg.checkpoint_dir, f"{prefix}_slot{slot:03d}")
                try:
                    self.model.save(path)
                except Exception as exc:
                    print(f"[WARN] snapshot save failed: {exc}")
                else:
                    self.league.add_snapshot(path + ".zip")
                    self._enforce_snapshot_limit()

            if self.verbose:
                print(
                    f"[PPO|SELF] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} "
                    f"snapshots={len(self.league.snapshots)} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )

            self.logger.record("self/episode", self.episode_idx)
            self.logger.record("self/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("self/draw_rate", self.draw_count / max(1, self.episode_idx))
            self.logger.record("self/snapshots", float(len(self.league.snapshots)))

            next_snapshot = None
            if bool(self.cfg.self_play_use_latest_snapshot):
                next_snapshot = self.league.latest_snapshot_key()
            else:
                spec = self.league.sample_snapshot()
                if spec.kind == "SNAPSHOT":
                    next_snapshot = spec.key

            if len(self.league.snapshots) == 0:
                fallback_path = os.path.join(
                    self.cfg.checkpoint_dir, f"{self.cfg.run_tag}_selfplay_init_fallback"
                )
                try:
                    self.model.save(fallback_path)
                except Exception as exc:
                    print(f"[WARN] self-play fallback save failed: {exc}")
                else:
                    self.league.add_snapshot(fallback_path + ".zip")
                    self._enforce_snapshot_limit()
                    next_snapshot = self.league.latest_snapshot_key()

            if next_snapshot:
                env = self.model.get_env()
                if env is not None:
                    env.env_method("set_next_opponent", "SNAPSHOT", next_snapshot)

        return True


class FixedOpponentCallback(BaseCallback):
    def __init__(self, *, cfg: PPOConfig) -> None:
        super().__init__(verbose=1)
        self.cfg = cfg
        self.episode_idx = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            summary = parse_episode_result(info)
            if summary is None:
                continue

            self.episode_idx += 1
            blue_score = summary.blue_score
            red_score = summary.red_score

            if blue_score > red_score:
                result = "WIN"
                self.win_count += 1
            elif blue_score < red_score:
                result = "LOSS"
                self.loss_count += 1
            else:
                result = "DRAW"
                self.draw_count += 1

            if self.verbose:
                opp = str(summary.scripted_tag or self.cfg.fixed_opponent_tag).upper()
                print(
                    f"[PPO|FIXED] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} opp=SCRIPTED:{opp} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )

            self.logger.record("fixed/episode", self.episode_idx)
            self.logger.record("fixed/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("fixed/draw_rate", self.draw_count / max(1, self.episode_idx))

        return True


class NoiseMetricsCSVCallback(BaseCallback):
    """Track action execution noise (flip rates, streaks) and log to CSV per episode."""

    def __init__(self, csv_path: str, eps: float, run_id: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = str(csv_path)
        self.eps = float(eps)
        self.run_id = str(run_id)
        self._reset_ep()

        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)) or ".", exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            return
        headers = [
            "run_id", "phase", "episode_idx", "steps", "agents", "eps",
            "total_actions", "flip_count", "flip_rate",
            "macro_flip_count", "macro_flip_rate",
            "target_flip_count", "target_flip_rate",
            "max_flip_streak",
            "win", "score_for", "score_against",
            "collisions", "coverage",
            "mean_inter_robot_dist", "std_inter_robot_dist",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

    def _reset_ep(self):
        self.ep_steps = 0
        self.flip_count = 0
        self.macro_flip_count = 0
        self.target_flip_count = 0
        self.curr_streak = None
        self.max_streak = 0
        self.episode_idx = 0

    def _on_training_start(self):
        n_envs = getattr(self.training_env, "num_envs", 1)
        self.curr_streak = np.zeros(n_envs, dtype=np.int32)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for env_i, info in enumerate(infos):
            self.ep_steps += 1

            flips = int(info.get("flip_count_step", 0))
            macro_flips = int(info.get("macro_flip_count_step", 0))
            target_flips = int(info.get("target_flip_count_step", 0))

            self.flip_count += flips
            self.macro_flip_count += macro_flips
            self.target_flip_count += target_flips

            # Streak tracking
            if flips > 0:
                if env_i < len(self.curr_streak):
                    self.curr_streak[env_i] += 1
                    if self.curr_streak[env_i] > self.max_streak:
                        self.max_streak = int(self.curr_streak[env_i])
            else:
                if env_i < len(self.curr_streak):
                    self.curr_streak[env_i] = 0

            # Episode end
            if env_i < len(dones) and bool(dones[env_i]):
                summary = parse_episode_result(info)
                if summary is None:
                    continue

                phase = summary.phase_name
                agents = int(info.get("num_agents", 2))
                action_components = int(info.get("action_components", 2))
                total_actions = int(self.ep_steps * agents * action_components)

                flip_rate = (self.flip_count / total_actions) if total_actions > 0 else 0.0
                macro_rate = (self.macro_flip_count / total_actions) if total_actions > 0 else 0.0
                target_rate = (self.target_flip_count / total_actions) if total_actions > 0 else 0.0

                win = summary.success
                score_for = summary.blue_score
                score_against = summary.red_score
                collisions = summary.collisions_per_episode
                coverage = summary.zone_coverage if summary.zone_coverage is not None else float("nan")
                mean_dist = summary.mean_inter_robot_dist if summary.mean_inter_robot_dist is not None else float("nan")
                std_dist = summary.std_inter_robot_dist if summary.std_inter_robot_dist is not None else float("nan")

                row = [
                    self.run_id, phase, self.episode_idx, self.ep_steps, agents, self.eps,
                    total_actions, self.flip_count, flip_rate,
                    self.macro_flip_count, macro_rate,
                    self.target_flip_count, target_rate,
                    self.max_streak,
                    win, score_for, score_against,
                    collisions, coverage,
                    mean_dist, std_dist,
                ]

                try:
                    with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(row)
                except Exception as exc:
                    if self.verbose:
                        print(f"[NoiseMetrics] CSV write failed: {exc}")

                self.episode_idx += 1
                self._reset_ep()

        return True


class MetricsCSVCallback(BaseCallback):
    """Collect Top 5 IROS-style metrics per episode and write one CSV at end of training (publish-friendly)."""

    CSV_COLUMNS = [
        "episode_id",
        "success",
        "time_to_first_score",
        "time_to_game_over",
        "collisions_per_episode",
        "near_misses_per_episode",
        "collision_free_episode",
        "mean_inter_robot_dist",
        "std_inter_robot_dist",
        "zone_coverage",
        "phase_name",
        "opponent_kind",
        "scripted_tag",
        "blue_score",
        "red_score",
    ]

    def __init__(self, *, save_path: str) -> None:
        super().__init__(verbose=0)
        self.save_path = str(save_path)
        self._rows: List[Dict[str, Any]] = []
        self._episode_id = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            summary = parse_episode_result(info)
            if summary is None:
                continue
            self._episode_id += 1
            row = {
                "episode_id": self._episode_id,
                "success": summary.success,
                "time_to_first_score": summary.time_to_first_score,
                "time_to_game_over": summary.time_to_game_over,
                "collisions_per_episode": summary.collisions_per_episode,
                "near_misses_per_episode": summary.near_misses_per_episode,
                "collision_free_episode": summary.collision_free_episode,
                "mean_inter_robot_dist": summary.mean_inter_robot_dist,
                "std_inter_robot_dist": summary.std_inter_robot_dist,
                "zone_coverage": summary.zone_coverage,
                "phase_name": summary.phase_name,
                "opponent_kind": summary.opponent_kind,
                "scripted_tag": summary.scripted_tag or "",
                "blue_score": summary.blue_score,
                "red_score": summary.red_score,
            }
            self._rows.append(row)
        return True

    def _fmt(self, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def _on_training_end(self) -> None:
        if not self._rows:
            return
        path = self.save_path
        if not path.lower().endswith(".csv"):
            path = path + ".csv"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS, extrasaction="ignore")
                w.writeheader()
                for row in self._rows:
                    out = {k: self._fmt(row.get(k)) for k in self.CSV_COLUMNS}
                    w.writerow(out)
            if self.verbose:
                print(f"[Metrics] Saved {len(self._rows)} rows to {path}")
        except Exception as exc:
            print(f"[WARN] Metrics CSV save failed: {exc}")


def train_ppo(cfg: Optional[PPOConfig] = None) -> None:
    cfg = cfg or PPOConfig()
    set_global_seed(cfg.seed)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    mode = str(cfg.mode).upper().strip()

    league = EloLeague(
        seed=cfg.seed,
        k_factor=32.0,
        matchmaking_tau=200.0,
        scripted_floor=0.50,
        species_prob=0.20,
        snapshot_prob=0.30,
    )

    curriculum: Optional[CurriculumState] = None
    controller: Optional[CurriculumController] = None
    if mode == TrainMode.CURRICULUM_LEAGUE.value:
        curriculum = CurriculumState(
            CurriculumConfig(
                phases=["OP1", "OP2", "OP3"],
                min_episodes={"OP1": 200, "OP2": 200, "OP3": 250},
                min_winrate={"OP1": 0.50, "OP2": 0.50, "OP3": 0.55},
                winrate_window=50,
                required_win_by={"OP1": 0, "OP2": 1, "OP3": 1},
                elo_margin=80.0,
                switch_to_league_after_op3_win=False,
            )
        )
        controller = CurriculumController(
            CurriculumControllerConfig(seed=cfg.seed),
            league=league,
        )
    elif mode == TrainMode.CURRICULUM_NO_LEAGUE.value:
        curriculum = CurriculumState(
            CurriculumConfig(
                phases=["OP1", "OP2", "OP3"],
                min_episodes={"OP1": 200, "OP2": 200, "OP3": 250},
                min_winrate={"OP1": 0.50, "OP2": 0.50, "OP3": 0.55},
                winrate_window=50,
                required_win_by={"OP1": 0, "OP2": 1, "OP3": 1},
                elo_margin=80.0,
                switch_to_league_after_op3_win=False,
            )
        )

    if mode == TrainMode.FIXED_OPPONENT.value:
        default_opponent = ("SCRIPTED", str(cfg.fixed_opponent_tag).upper())
        phase_name = str(cfg.fixed_opponent_tag).upper()
    elif mode == TrainMode.SELF_PLAY.value:
        default_opponent = ("SCRIPTED", "OP3")
        phase_name = "SELF_PLAY"
    elif mode == TrainMode.CURRICULUM_NO_LEAGUE.value and curriculum is not None:
        default_opponent = ("SCRIPTED", curriculum.phase)
        phase_name = curriculum.phase
    else:
        default_opponent = ("SCRIPTED", "OP1")
        phase_name = curriculum.phase if curriculum is not None else "OP1"

    env_fns = [
        _make_env_fn(cfg, default_opponent=default_opponent, rank=i)
        for i in range(max(1, int(cfg.n_envs)))
    ]
    try:
        venv = SubprocVecEnv(env_fns)
    except Exception:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv)

    # Naval realism: stress + physics-by-phase (OP1 no physics, OP2 relaxed, OP3 full)
    try:
        venv.env_method("set_stress_schedule", STRESS_BY_PHASE)
    except Exception:
        pass
    try:
        venv.env_method("set_phase", phase_name)
    except Exception:
        pass

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        policy=MaskedMultiInputPolicy,
        env=venv,
        learning_rate=float(cfg.learning_rate),
        n_steps=int(cfg.n_steps),
        batch_size=int(cfg.batch_size),
        n_epochs=int(cfg.n_epochs),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_range=float(cfg.clip_range),
        ent_coef=float(cfg.ent_coef),
        vf_coef=0.5,
        max_grad_norm=float(cfg.max_grad_norm),
        tensorboard_log=(
            os.path.join(cfg.checkpoint_dir, "tb", cfg.run_tag)
            if cfg.enable_tensorboard
            else None
        ),
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=cfg.seed,
        device=cfg.device,
    )

    if cfg.enable_tensorboard:
        model.set_logger(configure(os.path.join(cfg.checkpoint_dir, "tb", cfg.run_tag), ["tensorboard"]))
    else:
        model.set_logger(configure(None, []))

    if mode == TrainMode.SELF_PLAY.value:
        init_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_selfplay_init")
        try:
            model.save(init_path)
        except Exception as exc:
            print(f"[WARN] self-play init snapshot save failed: {exc}")
        else:
            league.add_snapshot(init_path + ".zip")
            init_key = league.latest_snapshot_key()
            max_snaps = max(0, int(getattr(cfg, "self_play_max_snapshots", 0)))
            if max_snaps > 0:
                while len(league.snapshots) > max_snaps:
                    oldest = league.snapshots.pop(0)
                    try:
                        if oldest and os.path.exists(oldest):
                            os.remove(oldest)
                    except Exception as exc:
                        print(f"[WARN] snapshot cleanup failed: {exc}")
            if init_key:
                venv.env_method("set_next_opponent", "SNAPSHOT", init_key)
                venv.reset()

    callbacks = []
    if mode == TrainMode.CURRICULUM_LEAGUE.value:
        callbacks.append(LeagueCallback(cfg=cfg, league=league, curriculum=curriculum, controller=controller))
    elif mode == TrainMode.CURRICULUM_NO_LEAGUE.value and curriculum is not None:
        callbacks.append(CurriculumNoLeagueCallback(cfg=cfg, curriculum=curriculum))
    elif mode == TrainMode.SELF_PLAY.value:
        callbacks.append(SelfPlayCallback(cfg=cfg, league=league))
    elif mode == TrainMode.FIXED_OPPONENT.value:
        callbacks.append(FixedOpponentCallback(cfg=cfg))

    # Top 5 IROS-style metrics: CSV at end of training (simple, publish-friendly)
    metrics_csv_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_metrics")
    callbacks.append(MetricsCSVCallback(save_path=metrics_csv_path))

    # Action execution noise metrics (if enabled)
    if getattr(cfg, "action_flip_prob", 0.0) > 0.0:
        noise_csv_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_noise_metrics")
        callbacks.append(
            NoiseMetricsCSVCallback(
                csv_path=noise_csv_path,
                eps=float(cfg.action_flip_prob),
                run_id=str(cfg.run_tag),
                verbose=0,
            )
        )

    if cfg.enable_checkpoints:
        callbacks.append(
            CheckpointCallback(
                save_freq=int(cfg.save_every_steps),
                save_path=cfg.checkpoint_dir,
                name_prefix=f"ckpt_{cfg.run_tag}",
            )
        )

    if cfg.enable_eval:
        eval_env = DummyVecEnv([_make_env_fn(cfg, default_opponent=("SCRIPTED", "OP3"), rank=0)])
        eval_env = VecMonitor(eval_env)
        callbacks.append(
            EvalCallback(
                eval_env,
                n_eval_episodes=int(cfg.eval_episodes),
                eval_freq=int(cfg.eval_every_steps),
                deterministic=True,
                best_model_save_path=cfg.checkpoint_dir,
            )
        )

    callbacks = CallbackList(callbacks)

    model.learn(total_timesteps=int(cfg.total_timesteps), callback=callbacks)

    final_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}")
    model.save(final_path)
    print(f"[PPO] Training complete. Final model saved to: {final_path}.zip")


if __name__ == "__main__":
    train_ppo()
