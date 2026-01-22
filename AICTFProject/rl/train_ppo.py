from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from game_field import make_game_field
from ctf_sb3_env import CTFGameFieldSB3Env
from rl.common import set_global_seed
from rl.curriculum import CurriculumConfig, CurriculumState
from rl.league import EloLeague, OpponentSpec
from config import MAP_NAME, MAP_PATH


@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 1_000_000
    n_envs: int = 4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    device: str = "cpu"

    checkpoint_dir: str = "checkpoints_sb3"
    run_tag: str = "ppo_curriculum"
    save_every_steps: int = 50_000
    eval_every_steps: int = 25_000
    eval_episodes: int = 6
    snapshot_every_episodes: int = 50

    max_decision_steps: int = 600


def _make_env_fn(cfg: PPOConfig, *, default_opponent: Tuple[str, str]) -> Any:
    def _fn():
        env = CTFGameFieldSB3Env(
            make_game_field_fn=lambda: make_game_field(
                map_name=MAP_NAME or None,
                map_path=MAP_PATH or None,
            ),
            max_decision_steps=cfg.max_decision_steps,
            enforce_masks=True,
            seed=cfg.seed,
            include_mask_in_obs=False,
            default_opponent_kind=default_opponent[0],
            default_opponent_key=default_opponent[1],
            ppo_gamma=cfg.gamma,
        )
        return env
    return _fn


class LeagueCallback(BaseCallback):
    def __init__(
        self,
        *,
        cfg: PPOConfig,
        league: EloLeague,
        curriculum: CurriculumState,
    ) -> None:
        super().__init__(verbose=1)
        self.cfg = cfg
        self.league = league
        self.curriculum = curriculum
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
        if self.league_mode:
            return self.league.sample_league()
        return self.league.sample_curriculum(self.curriculum.phase)

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

            if (
                self.curriculum.config.switch_to_league_after_op3_win
                and phase == "OP3"
                and is_scripted
                and win
            ):
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

            if (self.episode_idx % int(self.cfg.snapshot_every_episodes)) == 0:
                path = os.path.join(self.cfg.checkpoint_dir, f"sp_snapshot_ep{self.episode_idx:06d}")
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

        return True


def train_ppo(cfg: Optional[PPOConfig] = None) -> None:
    cfg = cfg or PPOConfig()
    set_global_seed(cfg.seed)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    curriculum = CurriculumState(
        CurriculumConfig(
            phases=["OP1", "OP2", "OP3"],
            min_episodes={"OP1": 150, "OP2": 150, "OP3": 0},
            min_winrate={"OP1": 0.55, "OP2": 0.55, "OP3": 0.50},
            winrate_window=50,
            required_win_by={"OP1": 1, "OP2": 1, "OP3": 0},
            elo_margin=100.0,
            switch_to_league_after_op3_win=True,
        )
    )

    league = EloLeague(
        seed=cfg.seed,
        k_factor=32.0,
        matchmaking_tau=250.0,
        scripted_floor=0.25,
        species_prob=0.25,
        snapshot_prob=0.50,
    )

    env_fns = [
        _make_env_fn(cfg, default_opponent=("SCRIPTED", "OP1"))
        for _ in range(max(1, int(cfg.n_envs)))
    ]
    try:
        venv = SubprocVecEnv(env_fns)
    except Exception:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv)

    # Apply initial curriculum phase to all envs
    try:
        venv.env_method("set_phase", curriculum.phase)
    except Exception:
        pass

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        policy="MultiInputPolicy",
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
        tensorboard_log=os.path.join(cfg.checkpoint_dir, "tb", cfg.run_tag),
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=cfg.seed,
        device=cfg.device,
    )

    model.set_logger(configure(os.path.join(cfg.checkpoint_dir, "tb", cfg.run_tag), ["tensorboard"]))

    eval_env = DummyVecEnv(
        [_make_env_fn(cfg, default_opponent=("SCRIPTED", "OP3"))]
    )
    eval_env = VecMonitor(eval_env)

    callbacks = CallbackList(
        [
            LeagueCallback(cfg=cfg, league=league, curriculum=curriculum),
            CheckpointCallback(
                save_freq=int(cfg.save_every_steps),
                save_path=cfg.checkpoint_dir,
                name_prefix=f"ckpt_{cfg.run_tag}",
            ),
            EvalCallback(
                eval_env,
                n_eval_episodes=int(cfg.eval_episodes),
                eval_freq=int(cfg.eval_every_steps),
                deterministic=True,
                best_model_save_path=cfg.checkpoint_dir,
            ),
        ]
    )

    model.learn(total_timesteps=int(cfg.total_timesteps), callback=callbacks)

    final_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}")
    model.save(final_path)
    print(f"[PPO] Training complete. Final model saved to: {final_path}.zip")


if __name__ == "__main__":
    train_ppo()
