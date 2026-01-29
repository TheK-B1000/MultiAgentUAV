from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, List
from collections import deque

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

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


# ============================================================
# Train modes (includes OLD baseline)
# ============================================================

class TrainMode(str, Enum):
    CURRICULUM_LEAGUE = "CURRICULUM_LEAGUE"
    CURRICULUM_NO_LEAGUE = "CURRICULUM_NO_LEAGUE"  # OLD baseline: OP1 -> OP2 -> OP3, no league
    FIXED_OPPONENT = "FIXED_OPPONENT"
    SELF_PLAY = "SELF_PLAY"


# ============================================================
# Config: Action-plan knobs + Metrics knobs (matching checklist)
# ============================================================

@dataclass
class PPOConfig:
    # Core PPO
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

    # Output & logging
    checkpoint_dir: str = "checkpoints_sb3"
    run_tag: str = "ppo_option3_impl"
    save_every_steps: int = 50_000
    eval_every_steps: int = 25_000
    eval_episodes: int = 6
    snapshot_every_episodes: int = 100
    enable_tensorboard: bool = False
    enable_checkpoints: bool = False
    enable_eval: bool = False

    # Environment
    max_decision_steps: int = 900
    op3_gate_tag: str = "OP3_HARD"

    # Mode
    mode: str = TrainMode.CURRICULUM_LEAGUE.value
    fixed_opponent_tag: str = "OP3"
    self_play_use_latest_snapshot: bool = True
    self_play_snapshot_every_episodes: int = 25
    self_play_max_snapshots: int = 25

    # ============================================================
    # Phase 1 (scenario lock): objective & termination semantics
    # (We don't modify env logic here, but we enforce metric extraction expectations)
    # ============================================================

    # Objective timing keys we will look for in episode_result
    # (time_to_breach / time_to_intercept is preferred; else fallback to episode length)
    objective_time_keys: Tuple[str, ...] = (
        "time_to_breach",
        "time_to_intercept",
        "breach_time",
        "intercept_time",
        "steps_to_breach",
        "steps_to_intercept",
        "breach_step",
        "intercept_step",
    )

    # Safety metric keys we will sum if present
    safety_keys: Tuple[str, ...] = (
        "safety_violations",
        "collisions",
        "collision_count",
        "out_of_bounds",
        "illegal_actions",
        "friendly_fire",
    )

    # ============================================================
    # Phase 2 (boat package realism): "physics shift" config
    # NOTE: This file passes these into the env via env_method if supported.
    # If your env doesn't implement them yet, nothing breaks.
    # ============================================================

    # Base "boat-ish" constraints (placeholder; your env must consume these to be true physics)
    boat_max_speed: float = 1.0
    boat_max_accel: float = 0.2
    boat_max_yaw_rate: float = 0.25

    # Disturbances / sensing / delays (robotics constraints)
    current_strength: float = 0.0
    drift_sigma: float = 0.0
    action_delay_steps: int = 0
    actuation_noise_sigma: float = 0.0
    sensor_range: float = 9999.0
    sensor_noise_sigma: float = 0.0
    sensor_dropout_prob: float = 0.0

    # Physics shift sweep (robustness vs dynamics)
    physics_sweep_current: Tuple[float, ...] = (0.0, 0.1, 0.2)
    physics_sweep_sensor_noise: Tuple[float, ...] = (0.0, 0.05, 0.10)
    physics_sweep_delay: Tuple[int, ...] = (0, 1, 3)

    # ============================================================
    # Phase 3 (adversary library): Held-out opponents + robustness eval
    # ============================================================

    # Unseen scripted adversaries (never sampled during training)
    unseen_opponent_tags: Tuple[str, ...] = ("OP3_HARD", "OP3_PINCER", "OP3_DECOY", "OP2_FAST")

    robustness_eval_every_steps: int = 100_000
    robustness_eval_episodes_per_opp: int = 5
    robustness_save_csv: bool = True
    robustness_save_plot: bool = True

    # General metric window
    metrics_window_episodes: int = 100


# ============================================================
# Env factory + "Option 3 realism injection"
# ============================================================

def _apply_env_option3_config(env, cfg: PPOConfig) -> None:
    """
    Attempts to apply Phase 2 realism knobs into env.
    This is 'best effort' and will not crash if your env doesn't support it yet.
    Implement these env methods to make the realism real:
      - set_dynamics_config(...)
      - set_disturbance_config(...)
      - set_robotics_constraints(...)
      - set_sensor_config(...)
      - set_physics_tag(...)
    """
    try:
        if hasattr(env, "set_dynamics_config"):
            env.set_dynamics_config(
                max_speed=float(cfg.boat_max_speed),
                max_accel=float(cfg.boat_max_accel),
                max_yaw_rate=float(cfg.boat_max_yaw_rate),
            )
    except Exception:
        pass

    try:
        if hasattr(env, "set_disturbance_config"):
            env.set_disturbance_config(
                current_strength=float(cfg.current_strength),
                drift_sigma=float(cfg.drift_sigma),
            )
    except Exception:
        pass

    try:
        if hasattr(env, "set_robotics_constraints"):
            env.set_robotics_constraints(
                action_delay_steps=int(cfg.action_delay_steps),
                actuation_noise_sigma=float(cfg.actuation_noise_sigma),
            )
    except Exception:
        pass

    try:
        if hasattr(env, "set_sensor_config"):
            env.set_sensor_config(
                sensor_range=float(cfg.sensor_range),
                sensor_noise_sigma=float(cfg.sensor_noise_sigma),
                sensor_dropout_prob=float(cfg.sensor_dropout_prob),
            )
    except Exception:
        pass

    try:
        if hasattr(env, "set_physics_tag"):
            env.set_physics_tag(
                f"CUR{cfg.current_strength}_NOISE{cfg.sensor_noise_sigma}_DLY{cfg.action_delay_steps}"
            )
    except Exception:
        pass


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

        # Phase 2 realism injection (safe no-op if env doesn’t support it)
        _apply_env_option3_config(env, cfg)

        return env
    return _fn


# ============================================================
# Policy (unchanged)
# ============================================================

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

        expected = 2 * (n_macros + n_targets)
        if mask.shape[1] < expected:
            pad = torch.ones((mask.shape[0], expected - mask.shape[1]), device=mask.device)
            mask = torch.cat([mask, pad], dim=1)

        full_mask = []
        offset = 0
        for i, dim in enumerate(dims):
            if i in (0, 2):
                m = mask[:, offset: offset + n_macros]
                offset += n_macros
                full_mask.append(m)
            elif i in (1, 3):
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


# ============================================================
# Checklist metrics (exactly what you added earlier)
# ============================================================

def _episode_win_loss_draw(ep: Dict[str, Any]) -> Tuple[str, float]:
    blue_score = int(ep.get("blue_score", 0))
    red_score = int(ep.get("red_score", 0))
    if blue_score > red_score:
        return "WIN", 1.0
    if blue_score < red_score:
        return "LOSS", 0.0
    return "DRAW", 0.5


def _extract_objective_time(cfg: PPOConfig, info: Dict[str, Any], ep: Dict[str, Any]) -> float:
    for k in cfg.objective_time_keys:
        if k in ep:
            try:
                return float(ep.get(k))
            except Exception:
                pass

    if isinstance(info.get("episode", None), dict):
        l = info["episode"].get("l", None)
        if l is not None:
            try:
                return float(l)
            except Exception:
                pass

    if "steps" in ep:
        try:
            return float(ep["steps"])
        except Exception:
            pass

    return float("nan")


def _extract_episode_return(info: Dict[str, Any], ep: Dict[str, Any]) -> float:
    if isinstance(info.get("episode", None), dict):
        r = info["episode"].get("r", None)
        if r is not None:
            try:
                return float(r)
            except Exception:
                pass

    for k in ("episode_return", "episode_reward", "return", "reward_sum", "total_reward"):
        if k in ep:
            try:
                return float(ep.get(k))
            except Exception:
                pass

    return float("nan")


def _extract_safety_events(cfg: PPOConfig, info: Dict[str, Any], ep: Dict[str, Any]) -> float:
    total = 0.0
    found_any = False

    for k in cfg.safety_keys:
        v = None
        if k in ep:
            v = ep.get(k)
        elif k in info:
            v = info.get(k)

        if v is None:
            continue

        found_any = True
        try:
            if isinstance(v, bool):
                total += 1.0 if v else 0.0
            elif isinstance(v, (int, float, np.number)):
                total += float(v)
            elif isinstance(v, (list, tuple, set, dict)):
                total += float(len(v))
            else:
                total += 1.0
        except Exception:
            total += 1.0

    return total if found_any else 0.0


class ChecklistMetricsCallback(BaseCallback):
    """
    ✅ Final Checklist (Print This) metrics:
      - Mission success rate
      - Time to breach / intercept
      - Learning curve AUC
      - Return variance
      - One safety metric
    """

    def __init__(self, *, cfg: PPOConfig, prefix: str = "checklist") -> None:
        super().__init__(verbose=0)
        self.cfg = cfg
        self.prefix = prefix

        self.episode_idx = 0
        self.win_count = 0
        self.draw_count = 0

        self._objtime_hist = deque(maxlen=int(cfg.metrics_window_episodes))
        self._return_hist = deque(maxlen=int(cfg.metrics_window_episodes))
        self._safety_hist = deque(maxlen=int(cfg.metrics_window_episodes))

        # AUC approximation: mean win_rate over episodes
        self._sum_win_rate = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue

            info = infos[i] if i < len(infos) else {}
            ep = info.get("episode_result", None)
            if not isinstance(ep, dict):
                ep = {}

            self.episode_idx += 1

            result, actual = _episode_win_loss_draw(ep)
            if actual == 1.0:
                self.win_count += 1
            if actual == 0.5:
                self.draw_count += 1

            win_rate = self.win_count / max(1, self.episode_idx)
            self._sum_win_rate += float(win_rate)
            auc = self._sum_win_rate / max(1, self.episode_idx)

            obj_time = _extract_objective_time(self.cfg, info, ep)
            if not np.isnan(obj_time):
                self._objtime_hist.append(float(obj_time))

            ep_ret = _extract_episode_return(info, ep)
            if not np.isnan(ep_ret):
                self._return_hist.append(float(ep_ret))

            safety_events = _extract_safety_events(self.cfg, info, ep)
            self._safety_hist.append(float(safety_events))

            obj_mean = float(np.mean(self._objtime_hist)) if len(self._objtime_hist) > 0 else float("nan")
            ret_var = float(np.var(self._return_hist)) if len(self._return_hist) > 1 else float("nan")
            safety_rate = float(np.mean(self._safety_hist)) if len(self._safety_hist) > 0 else 0.0

            p = self.prefix
            self.logger.record(f"{p}/episode", self.episode_idx)
            self.logger.record(f"{p}/mission_success_rate", float(win_rate))           # ✅
            self.logger.record(f"{p}/time_to_objective_mean", float(obj_mean))         # ✅
            self.logger.record(f"{p}/learning_curve_auc", float(auc))                  # ✅
            self.logger.record(f"{p}/return_variance_window", float(ret_var))          # ✅
            self.logger.record(f"{p}/safety_events_per_episode", float(safety_rate))   # ✅

            # Optional detail
            self.logger.record(f"{p}/draw_rate", float(self.draw_count / max(1, self.episode_idx)))
            self.logger.record(f"{p}/objective_time_window_n", float(len(self._objtime_hist)))
            self.logger.record(f"{p}/return_window_n", float(len(self._return_hist)))

            if self.verbose:
                _ = result  # keep for debugging if you want

        return True


# ============================================================
# Robustness evaluation:
#   (A) unseen adversaries (held-out)
#   (B) physics shifts (current/noise/delay sweeps)
# Produces:
#   - success on unseen adversaries
#   - robustness curve plot
#   - robustness CSV
# ============================================================

class RobustnessEvalCallback(BaseCallback):
    def __init__(self, *, cfg: PPOConfig) -> None:
        super().__init__(verbose=1)
        self.cfg = cfg
        self._next_eval = int(cfg.robustness_eval_every_steps)
        self._rows: List[Dict[str, Any]] = []

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self.csv_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_robustness.csv")
        self.png_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_robustness.png")

        if cfg.robustness_save_csv and not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timesteps",
                        "eval_type",
                        "opponent_tag",
                        "physics_tag",
                        "current_strength",
                        "sensor_noise_sigma",
                        "action_delay_steps",
                        "success_rate",
                        "draw_rate",
                        "avg_obj_time",
                        "episodes",
                    ],
                )
                w.writeheader()

    def _build_eval_env(self, *, opponent_tag: str, physics_cfg: Dict[str, Any]) -> Any:
        # Clone cfg + override physics fields
        # (keeps everything deterministic and isolated)
        tmp = PPOConfig(**{**self.cfg.__dict__})
        for k, v in physics_cfg.items():
            setattr(tmp, k, v)

        env = DummyVecEnv([_make_env_fn(tmp, default_opponent=("SCRIPTED", str(opponent_tag).upper()), rank=999)])
        env = VecMonitor(env)

        # Apply physics via env_method too (works if your VecEnv supports it)
        try:
            env.env_method("set_dynamics_config", dict(
                max_speed=float(tmp.boat_max_speed),
                max_accel=float(tmp.boat_max_accel),
                max_yaw_rate=float(tmp.boat_max_yaw_rate),
            ))
        except Exception:
            pass
        try:
            env.env_method("set_disturbance_config", dict(
                current_strength=float(tmp.current_strength),
                drift_sigma=float(tmp.drift_sigma),
            ))
        except Exception:
            pass
        try:
            env.env_method("set_robotics_constraints", dict(
                action_delay_steps=int(tmp.action_delay_steps),
                actuation_noise_sigma=float(tmp.actuation_noise_sigma),
            ))
        except Exception:
            pass
        try:
            env.env_method("set_sensor_config", dict(
                sensor_range=float(tmp.sensor_range),
                sensor_noise_sigma=float(tmp.sensor_noise_sigma),
                sensor_dropout_prob=float(tmp.sensor_dropout_prob),
            ))
        except Exception:
            pass
        try:
            env.env_method(
                "set_physics_tag",
                f"CUR{tmp.current_strength}_NOISE{tmp.sensor_noise_sigma}_DLY{tmp.action_delay_steps}",
            )
        except Exception:
            pass

        return env, tmp

    def _eval(self, *, opponent_tag: str, eval_type: str, physics_cfg: Dict[str, Any]) -> Dict[str, Any]:
        env, tmp = self._build_eval_env(opponent_tag=opponent_tag, physics_cfg=physics_cfg)

        wins = 0
        draws = 0
        obj_times: List[float] = []

        for _ in range(int(self.cfg.robustness_eval_episodes_per_opp)):
            obs = env.reset()
            done = [False]
            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, infos = env.step(action)

            info0 = infos[0] if infos and len(infos) > 0 else {}
            ep = info0.get("episode_result", {})
            if not isinstance(ep, dict):
                ep = {}

            _, actual = _episode_win_loss_draw(ep)
            if actual == 1.0:
                wins += 1
            if actual == 0.5:
                draws += 1

            t = _extract_objective_time(tmp, info0, ep)
            if not np.isnan(t):
                obj_times.append(float(t))

        n = int(self.cfg.robustness_eval_episodes_per_opp)
        success_rate = wins / max(1, n)
        draw_rate = draws / max(1, n)
        avg_obj_time = float(np.mean(obj_times)) if len(obj_times) > 0 else float("nan")

        physics_tag = f"CUR{tmp.current_strength}_NOISE{tmp.sensor_noise_sigma}_DLY{tmp.action_delay_steps}"

        env.close()

        return {
            "timesteps": int(self.num_timesteps),
            "eval_type": str(eval_type),
            "opponent_tag": str(opponent_tag).upper(),
            "physics_tag": physics_tag,
            "current_strength": float(tmp.current_strength),
            "sensor_noise_sigma": float(tmp.sensor_noise_sigma),
            "action_delay_steps": int(tmp.action_delay_steps),
            "success_rate": float(success_rate),
            "draw_rate": float(draw_rate),
            "avg_obj_time": float(avg_obj_time),
            "episodes": int(n),
        }

    def _maybe_plot(self) -> None:
        if not self.cfg.robustness_save_plot:
            return
        if plt is None:
            return
        if len(self._rows) == 0:
            return

        latest_ts = max(r["timesteps"] for r in self._rows)
        rows = [r for r in self._rows if r["timesteps"] == latest_ts]

        # Plot 1: Unseen adversaries (baseline physics)
        unseen_rows = [r for r in rows if r["eval_type"] == "unseen_adversary"]
        unseen_rows = sorted(unseen_rows, key=lambda r: r["opponent_tag"])

        if len(unseen_rows) > 0:
            xs = list(range(len(unseen_rows)))
            ys = [r["success_rate"] for r in unseen_rows]
            labels = [r["opponent_tag"] for r in unseen_rows]

            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.ylim(0.0, 1.0)
            plt.xticks(xs, labels, rotation=30, ha="right")
            plt.xlabel("Held-out adversary")
            plt.ylabel("Success rate")
            plt.title(f"Robustness vs unseen adversaries @ {latest_ts} timesteps")
            plt.tight_layout()
            base_png = self.png_path.replace(".png", "_unseen.png")
            plt.savefig(base_png)
            plt.close()

        # Plot 2: Physics shift curve (aggregate across opponents)
        shift_rows = [r for r in rows if r["eval_type"] == "physics_shift"]
        if len(shift_rows) > 0:
            # aggregate by physics_tag
            by_tag: Dict[str, List[float]] = {}
            for r in shift_rows:
                by_tag.setdefault(r["physics_tag"], []).append(float(r["success_rate"]))

            tags_sorted = sorted(by_tag.keys())
            xs = list(range(len(tags_sorted)))
            ys = [float(np.mean(by_tag[t])) for t in tags_sorted]

            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.ylim(0.0, 1.0)
            plt.xticks(xs, tags_sorted, rotation=30, ha="right")
            plt.xlabel("Physics shift setting")
            plt.ylabel("Mean success rate (across held-out opponents)")
            plt.title(f"Robustness vs physics shifts @ {latest_ts} timesteps")
            plt.tight_layout()
            shift_png = self.png_path.replace(".png", "_physics.png")
            plt.savefig(shift_png)
            plt.close()

    def _on_step(self) -> bool:
        if int(self.num_timesteps) < int(self._next_eval):
            return True

        # -----------------------------------
        # A) Unseen adversaries (held-out)
        # -----------------------------------
        base_physics = dict(
            current_strength=float(self.cfg.current_strength),
            sensor_noise_sigma=float(self.cfg.sensor_noise_sigma),
            action_delay_steps=int(self.cfg.action_delay_steps),
        )
        for tag in list(self.cfg.unseen_opponent_tags or ()):
            row = self._eval(opponent_tag=str(tag), eval_type="unseen_adversary", physics_cfg=base_physics)
            self._rows.append(row)

            opp = row["opponent_tag"]
            self.logger.record(f"robust/unseen/{opp}_success_rate", float(row["success_rate"]))  # ✅ (success on unseen)
            self.logger.record(f"robust/unseen/{opp}_draw_rate", float(row["draw_rate"]))
            if not np.isnan(row["avg_obj_time"]):
                self.logger.record(f"robust/unseen/{opp}_avg_time_to_objective", float(row["avg_obj_time"]))

            if self.verbose:
                print(
                    f"[ROBUST|UNSEEN] t={row['timesteps']} opp={opp} "
                    f"succ={row['success_rate']:.2f} draw={row['draw_rate']:.2f} "
                    f"avg_t={row['avg_obj_time'] if not np.isnan(row['avg_obj_time']) else 'NA'}"
                )

            if self.cfg.robustness_save_csv:
                with open(self.csv_path, "a", newline="") as f:
                    w = csv.DictWriter(
                        f,
                        fieldnames=[
                            "timesteps",
                            "eval_type",
                            "opponent_tag",
                            "physics_tag",
                            "current_strength",
                            "sensor_noise_sigma",
                            "action_delay_steps",
                            "success_rate",
                            "draw_rate",
                            "avg_obj_time",
                            "episodes",
                        ],
                    )
                    w.writerow(row)

        # -----------------------------------
        # B) Physics shifts (robustness under disturbances)
        # Sweep: current x noise x delay, evaluated on held-out opponents
        # -----------------------------------
        currents = list(self.cfg.physics_sweep_current or ())
        noises = list(self.cfg.physics_sweep_sensor_noise or ())
        delays = list(self.cfg.physics_sweep_delay or ())

        for cur in currents:
            for noise in noises:
                for dly in delays:
                    phys_cfg = dict(current_strength=float(cur), sensor_noise_sigma=float(noise), action_delay_steps=int(dly))
                    # evaluate across held-out opponents
                    for tag in list(self.cfg.unseen_opponent_tags or ()):
                        row = self._eval(opponent_tag=str(tag), eval_type="physics_shift", physics_cfg=phys_cfg)
                        self._rows.append(row)

                        # Log aggregated-friendly metrics
                        key = row["physics_tag"]
                        self.logger.record(f"robust/physics/{key}/success_rate", float(row["success_rate"]))
                        if not np.isnan(row["avg_obj_time"]):
                            self.logger.record(f"robust/physics/{key}/avg_time_to_objective", float(row["avg_obj_time"]))

                        if self.cfg.robustness_save_csv:
                            with open(self.csv_path, "a", newline="") as f:
                                w = csv.DictWriter(
                                    f,
                                    fieldnames=[
                                        "timesteps",
                                        "eval_type",
                                        "opponent_tag",
                                        "physics_tag",
                                        "current_strength",
                                        "sensor_noise_sigma",
                                        "action_delay_steps",
                                        "success_rate",
                                        "draw_rate",
                                        "avg_obj_time",
                                        "episodes",
                                    ],
                                )
                                w.writerow(row)

        # ✅ Robustness curve plots (unseen + physics)
        self._maybe_plot()

        self._next_eval += int(self.cfg.robustness_eval_every_steps)
        return True


# ============================================================
# League curriculum callback (existing, unchanged)
# ============================================================

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


# ============================================================
# Old baseline: OP1 -> OP2 -> OP3 without league (existing)
# ============================================================

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

            try:
                advanced = self.curriculum.advance_if_ready(
                    learner_rating=1e9,
                    opponent_rating=0.0,
                    win_by=win_by,
                )
            except Exception:
                advanced = False

            if advanced:
                phase = self.curriculum.phase

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

            env = self.model.get_env()
            if env is not None:
                env.env_method("set_next_opponent", "SCRIPTED", str(self.curriculum.phase).upper())
                env.env_method("set_phase", str(self.curriculum.phase).upper())
                try:
                    env.env_method("set_league_mode", False)
                except Exception:
                    pass

        return True


# ============================================================
# Self-play + Fixed (unchanged)
# ============================================================

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


# ============================================================
# Training entry: integrates Option 3 phases into code path
# ============================================================

def train_ppo(cfg: Optional[PPOConfig] = None) -> None:
    cfg = cfg or PPOConfig()
    set_global_seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    mode = str(cfg.mode).upper().strip()

    # League is required for Option 3 "threat family": scripted + species + snapshots
    league = EloLeague(
        seed=cfg.seed,
        k_factor=32.0,
        matchmaking_tau=200.0,
        scripted_floor=0.50,
        species_prob=0.20,   # Species = style diversity
        snapshot_prob=0.30,  # Snapshot pool = adaptive opponents
    )

    curriculum: Optional[CurriculumState] = None
    controller: Optional[CurriculumController] = None

    # Curriculum exists for both curriculum modes
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

    if mode == TrainMode.CURRICULUM_LEAGUE.value:
        controller = CurriculumController(
            CurriculumControllerConfig(seed=cfg.seed),
            league=league,
        )

    # Initial opponent
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

    # Apply initial scenario phase (Phase 1 lock)
    try:
        venv.env_method("set_phase", phase_name)
    except Exception:
        pass

    # Apply Option 3 realism (Phase 2) to vectorized envs if methods exist
    try:
        venv.env_method("set_dynamics_config", dict(
            max_speed=float(cfg.boat_max_speed),
            max_accel=float(cfg.boat_max_accel),
            max_yaw_rate=float(cfg.boat_max_yaw_rate),
        ))
    except Exception:
        pass
    try:
        venv.env_method("set_disturbance_config", dict(
            current_strength=float(cfg.current_strength),
            drift_sigma=float(cfg.drift_sigma),
        ))
    except Exception:
        pass
    try:
        venv.env_method("set_robotics_constraints", dict(
            action_delay_steps=int(cfg.action_delay_steps),
            actuation_noise_sigma=float(cfg.actuation_noise_sigma),
        ))
    except Exception:
        pass
    try:
        venv.env_method("set_sensor_config", dict(
            sensor_range=float(cfg.sensor_range),
            sensor_noise_sigma=float(cfg.sensor_noise_sigma),
            sensor_dropout_prob=float(cfg.sensor_dropout_prob),
        ))
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

    # Self-play init snapshot
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

    # ------------------------------------------------------------
    # Callbacks = "full implementation" of Option 3 measurement plan
    # ------------------------------------------------------------
    callbacks: List[BaseCallback] = []

    # ✅ Always-on checklist metrics (mission success, time to objective, AUC, return variance, safety)
    callbacks.append(ChecklistMetricsCallback(cfg=cfg, prefix="checklist"))

    # ✅ Robustness: unseen adversaries + physics shifts, with CSV + plots
    callbacks.append(RobustnessEvalCallback(cfg=cfg))

    # Mode-specific training
    if mode == TrainMode.CURRICULUM_LEAGUE.value:
        assert curriculum is not None and controller is not None
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
    print(f"[PPO] Robustness CSV: {os.path.join(cfg.checkpoint_dir, cfg.run_tag + '_robustness.csv')}")
    if cfg.robustness_save_plot:
        print(f"[PPO] Robustness plots: {os.path.join(cfg.checkpoint_dir, cfg.run_tag + '_robustness_unseen.png')} "
              f"and {os.path.join(cfg.checkpoint_dir, cfg.run_tag + '_robustness_physics.png')}")


if __name__ == "__main__":
    train_ppo()
