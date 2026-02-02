from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from ctf_sb3_env import CTFGameFieldSB3Env
from game_field import make_game_field
from macro_actions import MacroAction
from rl.common import env_seed, set_global_seed
from rl.curriculum import CurriculumConfig, CurriculumController, CurriculumControllerConfig, CurriculumState
from rl.league import EloLeague
from rl.train_ppo import MaskedMultiInputPolicy
from config import MAP_NAME, MAP_PATH


@dataclass
class RolePPOConfig:
    seed: int = 42
    total_timesteps: int = 600_000
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
    run_tag: str = "ppo_role_fixed"
    enable_tensorboard: bool = False
    enable_episode_logs: bool = True

    max_decision_steps: int = 900
    fixed_opponent_tag: str = "OP3"
    enable_curriculum: bool = True

    curriculum_min_episodes: Tuple[int, int, int] = (150, 150, 200)
    curriculum_min_winrate: Tuple[float, float, float] = (0.55, 0.55, 0.60)
    curriculum_required_win_by: Tuple[int, int, int] = (1, 1, 1)


class EpisodeResultCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__(verbose=1)
        self.episode_idx = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0

    def _opponent_key(self, ep: Dict[str, Any]) -> str:
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
                f"[PPO|ROLE] ep={self.episode_idx} result={result} "
                f"score={blue_score}:{red_score} phase={phase} opp={opp} "
                f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
            )

            if self.episode_idx % 50 == 0:
                macros = ep.get("macro_order", []) or []
                counts = ep.get("macro_counts", []) or []
                mine_counts = ep.get("mine_counts", []) or []
                mine_kills = ep.get("blue_mine_kills", 0)
                mines_enemy = ep.get("mines_placed_enemy_half", 0)
                mines_triggered = ep.get("mines_triggered_by_red", 0)

                def _fmt_counts(agent_idx: int) -> str:
                    if agent_idx >= len(counts):
                        return "no data"
                    pairs = []
                    for j, c in enumerate(counts[agent_idx]):
                        if c:
                            name = macros[j] if j < len(macros) else str(j)
                            pairs.append(f"{name}={int(c)}")
                    return ", ".join(pairs) if pairs else "none"

                def _fmt_mines(agent_idx: int) -> str:
                    if agent_idx >= len(mine_counts):
                        return "grab=0 place=0"
                    mc = mine_counts[agent_idx]
                    return f"grab={int(mc.get('grab', 0))} place={int(mc.get('place', 0))}"

                print(
                    f"[PPO|ROLE] ep={self.episode_idx} role_summary "
                    f"A0(def) macros[{_fmt_counts(0)}] mines[{_fmt_mines(0)}] | "
                    f"A1(att) macros[{_fmt_counts(1)}] mines[{_fmt_mines(1)}] | "
                    f"team_mines kills={int(mine_kills)} enemy_half={int(mines_enemy)} triggered={int(mines_triggered)}"
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

    def _opponent_key(self, info: Dict[str, Any]) -> str:
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


def _role_macro_indices() -> Tuple[list[int], list[int]]:
    defender = [
        int(MacroAction.GO_TO),
        int(MacroAction.GRAB_MINE),
        int(MacroAction.PLACE_MINE),
        int(MacroAction.GO_HOME),
    ]
    attacker = [
        int(MacroAction.GO_TO),
        int(MacroAction.GET_FLAG),
        int(MacroAction.GO_HOME),
    ]
    return defender, attacker


def _make_env_fn(cfg: RolePPOConfig, *, default_opponent: Tuple[str, str], rank: int) -> Any:
    """Env factory; per-env seed via env_seed so DummyVecEnv/SubprocVecEnv behave the same."""
    def _fn():
        s = env_seed(cfg.seed, rank)
        np.random.seed(s)
        torch.manual_seed(s)
        env = CTFGameFieldSB3Env(
            make_game_field_fn=lambda: make_game_field(
                map_name=MAP_NAME or None,
                map_path=MAP_PATH or None,
            ),
            max_decision_steps=cfg.max_decision_steps,
            enforce_masks=True,
            seed=s,
            include_mask_in_obs=True,
            blue_role_macros=_role_macro_indices(),
            default_opponent_kind=default_opponent[0],
            default_opponent_key=default_opponent[1],
            ppo_gamma=cfg.gamma,
        )
        return env
    return _fn


def train_role_ppo(cfg: Optional[RolePPOConfig] = None) -> None:
    cfg = cfg or RolePPOConfig()
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

    if cfg.enable_curriculum:
        phase_name = curriculum.phase
        opp = controller.select_opponent(phase_name, league_mode=False)
        default_opponent = (opp.kind, opp.key)
    else:
        default_opponent = ("SCRIPTED", str(cfg.fixed_opponent_tag).upper())
        phase_name = str(cfg.fixed_opponent_tag).upper()

    # Same env_fns work with either vec env: both call env_fns[i]() with same rank i.
    env_fns = [_make_env_fn(cfg, default_opponent=default_opponent, rank=i) for i in range(max(1, int(cfg.n_envs)))]
    if cfg.n_envs > 1:
        venv = SubprocVecEnv(env_fns)
    else:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv)

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

    callbacks = []
    if cfg.enable_episode_logs:
        callbacks.append(EpisodeResultCallback())
    if cfg.enable_curriculum:
        callbacks.append(CurriculumOpponentCallback(league=league, curriculum=curriculum, controller=controller))
    callbacks = CallbackList(callbacks)

    model.learn(total_timesteps=int(cfg.total_timesteps), callback=callbacks)

    final_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}")
    model.save(final_path)
    print(f"[PPO|ROLE] Training complete. Final model saved to: {final_path}.zip")


if __name__ == "__main__":
    train_role_ppo()
