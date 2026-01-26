from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from ctf_sb3_env import CTFGameFieldSB3Env
from game_field import make_game_field
from rl.common import set_global_seed
from rl.curriculum import CurriculumConfig, CurriculumController, CurriculumControllerConfig, CurriculumState
from rl.league import EloLeague
from rl.hppo_env import CTFHighLevelEnv, HighLevelModeScheduler
from rl.train_ppo import MaskedMultiInputPolicy
from config import MAP_NAME, MAP_PATH


@dataclass
class HPPOConfig:
    seed: int = 42
    low_total_timesteps: int = 600_000
    high_total_timesteps: int = 200_000
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

    max_decision_steps: int = 900
    high_horizon_steps: int = 8

    low_mode_resample_steps: int = 0
    low_attack_prob: float = 0.5

    default_opponent_tag: str = "OP3"
    checkpoint_dir: str = "checkpoints_sb3"
    run_tag: str = "hppo_attack_defend"
    enable_tensorboard: bool = False
    enable_episode_logs: bool = True
    enable_curriculum_low: bool = False
    enable_curriculum_high: bool = True

    curriculum_min_episodes: Tuple[int, int, int] = (150, 150, 200)
    curriculum_min_winrate: Tuple[float, float, float] = (0.55, 0.55, 0.60)
    curriculum_required_win_by: Tuple[int, int, int] = (1, 1, 1)

    alternating_cycles: int = 3
    low_steps_per_cycle: int = 0
    high_steps_per_cycle: int = 0


class EpisodeResultCallback(BaseCallback):
    def __init__(self, label: str) -> None:
        super().__init__(verbose=1)
        self.label = str(label).upper()
        self.episode_idx = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0

    def _opponent_key(self, ep: dict) -> str:
        kind = str(ep.get("opponent_kind", "scripted")).upper()
        if kind == "SNAPSHOT":
            return str(ep.get("opponent_snapshot", ""))
        if kind == "SPECIES":
            tag = str(ep.get("species_tag", "BALANCED")).upper()
            return f"SPECIES:{tag}"
        tag = str(ep.get("scripted_tag", "OP3")).upper()
        return f"SCRIPTED:{tag}"

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
            phase = str(ep.get("phase_name", "OP3"))

            if blue_score > red_score:
                result = "WIN"
                self.win_count += 1
            elif red_score > blue_score:
                result = "LOSS"
                self.loss_count += 1
            else:
                result = "DRAW"
                self.draw_count += 1

            opp = self._opponent_key(ep)
            print(
                f"[HPPO|{self.label}] ep={self.episode_idx} result={result} "
                f"score={blue_score}:{red_score} phase={phase} opp={opp} "
                f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
            )
        return True


class CurriculumOpponentCallback(BaseCallback):
    def __init__(
        self,
        *,
        league: EloLeague,
        curriculum: CurriculumState,
        controller: CurriculumController,
    ) -> None:
        super().__init__(verbose=1)
        self.league = league
        self.curriculum = curriculum
        self.controller = controller

    def _opponent_key(self, info: dict) -> str:
        kind = str(info.get("opponent_kind", "scripted")).upper()
        if kind == "SNAPSHOT":
            return str(info.get("opponent_snapshot", ""))
        if kind == "SPECIES":
            tag = str(info.get("species_tag", "BALANCED")).upper()
            return f"SPECIES:{tag}"
        tag = str(info.get("scripted_tag", "OP3")).upper()
        return f"SCRIPTED:{tag}"

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

            blue_score = int(ep.get("blue_score", 0))
            red_score = int(ep.get("red_score", 0))
            win_by = int(blue_score - red_score)

            if blue_score > red_score:
                actual = 1.0
            elif red_score > blue_score:
                actual = 0.0
            else:
                actual = 0.5

            opp_key = self._opponent_key(ep)
            self.league.update_elo(opp_key, actual)
            self.controller.record_result(opp_key, actual)

            phase = self.curriculum.phase
            self.curriculum.phase_episode_count += 1
            self.curriculum.record_result(phase, actual)

            if opp_key.startswith("SCRIPTED:"):
                opp_rating = self.league.get_rating(opp_key)
                self.curriculum.advance_if_ready(
                    learner_rating=self.league.learner_rating,
                    opponent_rating=opp_rating,
                    win_by=win_by,
                )

            next_opp = self.controller.select_opponent(self.curriculum.phase, league_mode=False)
            env = self.model.get_env()
            if env is not None:
                env.env_method("set_next_opponent", next_opp.kind, next_opp.key)
                env.env_method("set_phase", self.curriculum.phase)
        return True


def _make_base_env(
    cfg: HPPOConfig,
    *,
    default_opponent: Tuple[str, str],
    rank: int,
    initial_phase: str,
) -> CTFGameFieldSB3Env:
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
        include_high_level_mode=True,
        high_level_mode=0,
        high_level_mode_onehot=True,
        default_opponent_kind=default_opponent[0],
        default_opponent_key=default_opponent[1],
        ppo_gamma=cfg.gamma,
    )
    try:
        env.set_phase(initial_phase)
    except Exception:
        pass
    return env


def _make_low_env_fn(
    cfg: HPPOConfig,
    *,
    default_opponent: Tuple[str, str],
    rank: int,
    initial_phase: str,
) -> Callable[[], Any]:
    def _fn():
        env = _make_base_env(cfg, default_opponent=default_opponent, rank=rank, initial_phase=initial_phase)
        return HighLevelModeScheduler(
            env,
            resample_steps=cfg.low_mode_resample_steps,
            attack_prob=cfg.low_attack_prob,
        )
    return _fn


def train_hppo(cfg: HPPOConfig | None = None) -> None:
    cfg = cfg or HPPOConfig()
    set_global_seed(cfg.seed)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    curriculum = CurriculumState(
        CurriculumConfig(
            phases=["OP1", "OP2", "OP3"],
            min_episodes={"OP1": cfg.curriculum_min_episodes[0], "OP2": cfg.curriculum_min_episodes[1], "OP3": cfg.curriculum_min_episodes[2]},
            min_winrate={"OP1": cfg.curriculum_min_winrate[0], "OP2": cfg.curriculum_min_winrate[1], "OP3": cfg.curriculum_min_winrate[2]},
            winrate_window=50,
            required_win_by={"OP1": cfg.curriculum_required_win_by[0], "OP2": cfg.curriculum_required_win_by[1], "OP3": cfg.curriculum_required_win_by[2]},
            elo_margin=100.0,
            switch_to_league_after_op3_win=False,
        )
    )
    league = EloLeague(
        seed=cfg.seed,
        k_factor=32.0,
        matchmaking_tau=200.0,
        scripted_floor=0.50,
        species_prob=0.20,
        snapshot_prob=0.30,
    )
    controller = CurriculumController(CurriculumControllerConfig(seed=cfg.seed), league=league)

    low_model = None
    high_model = None

    if int(cfg.alternating_cycles) <= 0:
        raise ValueError("alternating_cycles must be >= 1")
    cycles = int(cfg.alternating_cycles)

    def _steps_per_cycle(total: int, per_cycle: int) -> int:
        if per_cycle and int(per_cycle) > 0:
            return int(per_cycle)
        return max(1, int(total) // max(1, cycles))

    low_steps = _steps_per_cycle(int(cfg.low_total_timesteps), int(cfg.low_steps_per_cycle))
    high_steps = _steps_per_cycle(int(cfg.high_total_timesteps), int(cfg.high_steps_per_cycle))

    for cycle in range(cycles):
        cycle_idx = cycle + 1
        phase = curriculum.phase
        opp = controller.select_opponent(phase, league_mode=False)
        if cfg.enable_curriculum_low or cfg.enable_curriculum_high:
            default_opponent = (opp.kind, opp.key)
        else:
            default_opponent = ("SCRIPTED", cfg.default_opponent_tag)

        # ----------------------
        # Low-level PPO training
        # ----------------------
        env_fns = [
            _make_low_env_fn(cfg, default_opponent=default_opponent, rank=i, initial_phase=phase)
            for i in range(cfg.n_envs)
        ]
        if cfg.n_envs > 1:
            low_env = SubprocVecEnv(env_fns)
        else:
            low_env = DummyVecEnv(env_fns)
        low_env = VecMonitor(low_env)

        if cfg.enable_tensorboard:
            tb_dir = os.path.join(cfg.checkpoint_dir, "tb_hppo_low")
            low_logger = configure(tb_dir, ["tensorboard"])
        else:
            low_logger = configure(None, [])

        if low_model is None:
            low_model = PPO(
                MaskedMultiInputPolicy,
                low_env,
                n_steps=cfg.n_steps,
                batch_size=cfg.batch_size,
                n_epochs=cfg.n_epochs,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                clip_range=cfg.clip_range,
                ent_coef=cfg.ent_coef,
                learning_rate=cfg.learning_rate,
                max_grad_norm=cfg.max_grad_norm,
                device=cfg.device,
                verbose=0,
            )
        else:
            low_model.set_env(low_env)
        low_model.set_logger(low_logger)

        low_callbacks = []
        if cfg.enable_episode_logs:
            low_callbacks.append(EpisodeResultCallback(f"LOW{cycle_idx}"))
        if cfg.enable_curriculum_low:
            low_callbacks.append(CurriculumOpponentCallback(league=league, curriculum=curriculum, controller=controller))
        low_model.learn(
            total_timesteps=int(low_steps),
            callback=CallbackList(low_callbacks),
            reset_num_timesteps=(cycle_idx == 1),
        )

        low_path = os.path.join(cfg.checkpoint_dir, f"hppo_low_{cfg.run_tag}_c{cycle_idx}")
        low_model.save(low_path)
        print(f"[HPPO] Low-level policy saved: {low_path}.zip")

        # -----------------------
        # High-level PPO training
        # -----------------------
        def _make_high_env():
            return CTFHighLevelEnv(
                make_env_fn=lambda: _make_base_env(
                    cfg,
                    default_opponent=default_opponent,
                    rank=0,
                    initial_phase=phase,
                ),
                low_level_model=low_model,
                horizon_steps=cfg.high_horizon_steps,
            )

        high_env = DummyVecEnv([_make_high_env])
        high_env = VecMonitor(high_env)

        if cfg.enable_tensorboard:
            tb_dir = os.path.join(cfg.checkpoint_dir, "tb_hppo_high")
            high_logger = configure(tb_dir, ["tensorboard"])
        else:
            high_logger = configure(None, [])

        if high_model is None:
            high_model = PPO(
                "MlpPolicy",
                high_env,
                n_steps=cfg.n_steps,
                batch_size=cfg.batch_size,
                n_epochs=cfg.n_epochs,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                clip_range=cfg.clip_range,
                ent_coef=cfg.ent_coef,
                learning_rate=cfg.learning_rate,
                max_grad_norm=cfg.max_grad_norm,
                device=cfg.device,
                verbose=0,
            )
        else:
            high_model.set_env(high_env)
        high_model.set_logger(high_logger)

        high_callbacks = []
        if cfg.enable_episode_logs:
            high_callbacks.append(EpisodeResultCallback(f"HIGH{cycle_idx}"))
        if cfg.enable_curriculum_high:
            high_callbacks.append(CurriculumOpponentCallback(league=league, curriculum=curriculum, controller=controller))
        high_model.learn(
            total_timesteps=int(high_steps),
            callback=CallbackList(high_callbacks),
            reset_num_timesteps=(cycle_idx == 1),
        )

        high_path = os.path.join(cfg.checkpoint_dir, f"hppo_high_{cfg.run_tag}_c{cycle_idx}")
        high_model.save(high_path)
        print(f"[HPPO] High-level policy saved: {high_path}.zip")

    if low_model is not None:
        final_low = os.path.join(cfg.checkpoint_dir, f"hppo_low_{cfg.run_tag}")
        low_model.save(final_low)
        print(f"[HPPO] Final low-level policy saved: {final_low}.zip")
    if high_model is not None:
        final_high = os.path.join(cfg.checkpoint_dir, f"hppo_high_{cfg.run_tag}")
        high_model.save(final_high)
        print(f"[HPPO] Final high-level policy saved: {final_high}.zip")


if __name__ == "__main__":
    train_hppo()
