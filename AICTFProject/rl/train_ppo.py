from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

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
)
from rl.league import EloLeague, OpponentSpec
from config import MAP_NAME, MAP_PATH


class TrainMode(str, Enum):
    CURRICULUM_LEAGUE = "CURRICULUM_LEAGUE"
    CURRICULUM_NO_LEAGUE = "CURRICULUM_NO_LEAGUE"  # OLD baseline: OP1 -> OP2 -> OP3, no league
    FIXED_OPPONENT = "FIXED_OPPONENT"
    SELF_PLAY = "SELF_PLAY"


@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 1_000_000
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
    run_tag: str = "ppo_curriculum_old_v1"
    save_every_steps: int = 50_000
    eval_every_steps: int = 25_000
    eval_episodes: int = 6
    snapshot_every_episodes: int = 100
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

    def _opponent_key(self, info: Dict[str, Any]) -> str:
        kind = str(info.get("opponent_kind", "scripted")).upper()
        if kind == "SNAPSHOT":
            return str(info.get("opponent_snapshot", ""))
        if kind == "SPECIES":
            tag = str(info.get("species_tag", "BALANCED")).upper()
            return f"SPECIES:{tag}"
        tag = str(info.get("scripted_tag", "OP3")).upper()
        return f"SCRIPTED:{tag}"

    def _select_next_opponent(self) -> OpponentSpec:
        return self.controller.select_opponent(self.curriculum.phase, league_mode=self.league_mode)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            ep = info.get("episode_result", None)
            if not isinstance(ep, dict):
                continue

            self.episode_idx += 1
            blue_score = int(ep.get("blue_score", 0))
            red_score = int(ep.get("red_score", 0))
            win_by = int(blue_score - red_score)

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

            opp_key = self._opponent_key(ep)
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
                prefix = f"{self.cfg.run_tag}_league_snapshot"
                path = os.path.join(self.cfg.checkpoint_dir, f"{prefix}_ep{self.episode_idx:06d}")
                try:
                    self.model.save(path)
                except Exception as exc:
                    print(f"[WARN] snapshot save failed: {exc}")
                else:
                    self.league.add_snapshot(path + ".zip")

            next_opp = self._select_next_opponent()
            env = self.model.get_env()
            if env is not None:
                env.env_method("set_next_opponent", next_opp.kind, next_opp.key)
                env.env_method("set_phase", self.curriculum.phase)
                env.env_method("set_league_mode", self.league_mode)

        return True


class SequentialCurriculumCallback(BaseCallback):
    """
    OLD BASELINE (no league):
      OP1 -> OP2 -> OP3 progression using curriculum gates,
      but WITHOUT Elo league, matchmaking, species, or snapshots.
    """

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
            ep = info.get("episode_result", None)
            if not isinstance(ep, dict):
                continue

            self.episode_idx += 1
            blue_score = int(ep.get("blue_score", 0))
            red_score = int(ep.get("red_score", 0))
            win_by = int(blue_score - red_score)

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

            # Advance using existing gates but force Elo condition to never block
            # (so it's truly the old "OP1->OP2->OP3" gate progression).
            try:
                advanced = self.curriculum.advance_if_ready(
                    learner_rating=1e9,  # huge so elo_margin check always passes
                    opponent_rating=0.0,
                    win_by=win_by,
                )
            except Exception:
                advanced = False

            if advanced:
                phase = self.curriculum.phase  # updated phase after advancing

            if self.verbose:
                print(
                    f"[PPO|OLD_SEQ] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} phase={phase} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )

            self.logger.record("oldseq/episode", self.episode_idx)
            self.logger.record("oldseq/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("oldseq/draw_rate", self.draw_count / max(1, self.episode_idx))
            self.logger.record("oldseq/phase_idx", float(self.curriculum.phase_idx))

            # Next opponent is exactly the current phase opponent
            env = self.model.get_env()
            if env is not None:
                env.env_method("set_next_opponent", "SCRIPTED", str(self.curriculum.phase).upper())
                env.env_method("set_phase", str(self.curriculum.phase).upper())
                try:
                    env.env_method("set_league_mode", False)
                except Exception:
                    pass

        return True


class SelfPlayCallback(BaseCallback):
    def __init__(self, *, cfg: PPOConfig, league: EloLeague) -> None:
        super().__init__(verbose=1)
        self.cfg = cfg
        self.league = league
        self.episode_idx = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self._max_snapshots = max(0, int(getattr(cfg, "self_play_max_snapshots", 0)))

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
            ep = info.get("episode_result", None)
            if not isinstance(ep, dict):
                continue

            self.episode_idx += 1
            blue_score = int(ep.get("blue_score", 0))
            red_score = int(ep.get("red_score", 0))
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
                prefix = f"{self.cfg.run_tag}_selfplay_snapshot"
                path = os.path.join(self.cfg.checkpoint_dir, f"{prefix}_ep{self.episode_idx:06d}")
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
            ep = info.get("episode_result", None)
            if not isinstance(ep, dict):
                continue

            self.episode_idx += 1
            blue_score = int(ep.get("blue_score", 0))
            red_score = int(ep.get("red_score", 0))

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
                opp = str(ep.get("scripted_tag", self.cfg.fixed_opponent_tag)).upper()
                print(
                    f"[PPO|FIXED] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} opp=SCRIPTED:{opp} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )

            self.logger.record("fixed/episode", self.episode_idx)
            self.logger.record("fixed/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("fixed/draw_rate", self.draw_count / max(1, self.episode_idx))

        return True


def train_ppo(cfg: Optional[PPOConfig] = None) -> None:
    cfg = cfg or PPOConfig()
    set_global_seed(cfg.seed)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    mode = str(cfg.mode).upper().strip()

    # League exists for league/self-play; harmless for others (but not used in old baseline)
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

    # Build curriculum for BOTH curriculum modes
    if mode in (TrainMode.CURRICULUM_LEAGUE.value, TrainMode.CURRICULUM_NO_LEAGUE.value):
        curriculum = CurriculumState(
            CurriculumConfig(
                phases=["OP1", "OP2", "OP3"],
                min_episodes={"OP1": 150, "OP2": 150, "OP3": 200},
                min_winrate={"OP1": 0.55, "OP2": 0.55, "OP3": 0.60},
                winrate_window=50,
                required_win_by={"OP1": 1, "OP2": 1, "OP3": 1},
                elo_margin=100.0,
                switch_to_league_after_op3_win=False,
            )
        )

    # Only league mode needs the controller
    if mode == TrainMode.CURRICULUM_LEAGUE.value:
        controller = CurriculumController(
            CurriculumControllerConfig(seed=cfg.seed),
            league=league,
        )

    # Choose initial opponent + initial phase label
    if mode == TrainMode.FIXED_OPPONENT.value:
        default_opponent = ("SCRIPTED", str(cfg.fixed_opponent_tag).upper())
        phase_name = str(cfg.fixed_opponent_tag).upper()
    elif mode == TrainMode.SELF_PLAY.value:
        default_opponent = ("SCRIPTED", "OP3")
        phase_name = "SELF_PLAY"
    elif mode == TrainMode.CURRICULUM_NO_LEAGUE.value:
        default_opponent = ("SCRIPTED", "OP1")
        phase_name = "OP1"
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

    # Apply initial phase to all envs
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
        assert curriculum is not None
        assert controller is not None
        callbacks.append(LeagueCallback(cfg=cfg, league=league, curriculum=curriculum, controller=controller))
    elif mode == TrainMode.CURRICULUM_NO_LEAGUE.value:
        assert curriculum is not None
        callbacks.append(SequentialCurriculumCallback(cfg=cfg, curriculum=curriculum))
    elif mode == TrainMode.SELF_PLAY.value:
        callbacks.append(SelfPlayCallback(cfg=cfg, league=league))
    elif mode == TrainMode.FIXED_OPPONENT.value:
        callbacks.append(FixedOpponentCallback(cfg=cfg))

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
