from __future__ import annotations

import os
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Gymnasium + SB3
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv

# Torch
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# DirectML (optional)
try:
    import torch_directml
    HAS_TDML = True
except Exception:
    HAS_TDML = False

# Your code
from game_field import CNN_COLS, CNN_ROWS, NUM_CNN_CHANNELS
from game_manager import GameManager
from macro_actions import MacroAction
from policies import OP1RedPolicy, OP2RedPolicy, OP3RedPolicy
from map_registry import make_game_field
from behavior_logging import BehaviorLogger, append_records_csv
from config import (
    MAP_NAME as DEFAULT_MAP_NAME,
    MAP_PATH as DEFAULT_MAP_PATH,
    LOG_DIR as DEFAULT_LOG_DIR,
    PPO_PHASE_MIN_EPISODES,
    PPO_PHASE_ELO_MARGIN,
    PPO_PHASE_REQUIRED_WIN_BY,
    PPO_CURRICULUM_SPECIES_PROB,
    PPO_CURRICULUM_SNAPSHOT_PROB,
    PPO_CURRICULUM_WINRATE_WINDOW,
    PPO_CURRICULUM_MIN_WINRATE,
    PPO_CURRICULUM_FALLBACK_PROB,
    PPO_CURRICULUM_OP3_SPECIES_SCALE,
    PPO_CURRICULUM_OP3_SNAPSHOT_SCALE,
    PPO_SWITCH_TO_ELO_AFTER_OP3,
)

_BEHAVIOR_LOGGING_DISABLED_NOTICE = False


# -------------------------
# Config
# -------------------------
SEED = 42
TOTAL_TIMESTEPS = 1_000_000

# One env.step() == one "decision window" in your sim
DECISION_WINDOW = 0.7
SIM_DT = 0.1
GAMMA = 0.995

# PPO rollout sizing
PPO_N_ENVS = 4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 256

# Schedules
LR_START = 3e-4
LR_END = 1e-5
ENT_COEF_START = 0.03
ENT_COEF_END = 0.005

# Evaluation / checkpoints
EVAL_EVERY_STEPS = 25_000
EVAL_EPISODES = 6
SAVE_EVERY_STEPS = 50_000

# Episode caps
MAX_MACRO_STEPS = 500
SCORE_LIMIT = 3
MAX_TIME = 200

# Outcome shaping
WIN_BONUS = 40.0
LOSS_PENALTY = -30.0
DRAW_PENALTY = -20.0
BLUE_SCORED_BONUS = 20.0
RED_SCORED_PENALTY = -20.0

# Stall penalty (discourage 0:0 draws)
NO_SCORE_PENALTY_PER_WINDOW = 0.03
NO_SCORE_GRACE_WINDOWS = 4

# Map selection
MAP_NAME = DEFAULT_MAP_NAME
MAP_PATH = DEFAULT_MAP_PATH

# Time penalty interpreted as "per second per agent"
TIME_PENALTY_PER_AGENT_PER_SEC = 0.002

# Invalid macro penalty (SB3 can't hard-mask logits)
INVALID_MACRO_PENALTY = 0.5
INVALID_MACRO_PENALTY_START = 3.0
INVALID_MACRO_PENALTY_END = 0.25

# Action set
USED_MACROS = [
    MacroAction.GO_TO,      # 0
    MacroAction.GRAB_MINE,  # 1
    MacroAction.GET_FLAG,   # 2
    MacroAction.PLACE_MINE, # 3
    MacroAction.GO_HOME,    # 4
]
N_MACROS = len(USED_MACROS)

CHECKPOINT_DIR = "checkpoints_sb3"
POOL_DIR = os.path.join(CHECKPOINT_DIR, "self_play_pool")
os.makedirs(POOL_DIR, exist_ok=True)

# -------------------------
# TRAIN MODE (choose before running)
# -------------------------
TRAIN_MODE_DEFAULT_LEAGUE = "league_elo_curriculum"
TRAIN_MODE_ELO_LEAGUE = "league_elo"
TRAIN_MODE_BASELINE_FIXED = "baseline_fixed_opponent"
TRAIN_MODE_BASELINE_SELFPLAY = "baseline_selfplay_uniform"

TRAIN_MODE = TRAIN_MODE_DEFAULT_LEAGUE
RUN_TAG = TRAIN_MODE

# Device selection (default CPU to avoid DML scatter crashes)
USE_DIRECTML = False
if HAS_TDML and USE_DIRECTML:
    DML_DEVICE = torch_directml.device()
    USING_DML = True
else:
    DML_DEVICE = "cpu"
    USING_DML = False


# -------------------------
# Helpers
# -------------------------
def agent_uid(agent: Any) -> str:
    uid = getattr(agent, "unique_id", None) or getattr(agent, "slot_id", None)
    if uid is None:
        side = getattr(agent, "side", "blue")
        aid = getattr(agent, "agent_id", 0)
        uid = f"{side}_{aid}"
    return str(uid)


def pop_reward_events_strict(gm: GameManager) -> List[Tuple[float, str, float]]:
    fn = getattr(gm, "pop_reward_events", None)
    if fn is None or (not callable(fn)):
        raise RuntimeError("GameManager must implement pop_reward_events() returning (t, agent_id:str, r).")
    return fn()


def clear_reward_events_best_effort(gm: GameManager) -> None:
    try:
        _ = pop_reward_events_strict(gm)
    except Exception:
        return


def outcome_bonus(blue_score: int, red_score: int) -> float:
    if blue_score > red_score:
        return float(WIN_BONUS)
    if red_score > blue_score:
        return float(LOSS_PENALTY)
    return float(DRAW_PENALTY)


# -------------------------
# Species injection
# -------------------------
def make_species_policy(species_type: str = "BALANCED") -> OP3RedPolicy:
    species_type = str(species_type).upper().strip()
    p = OP3RedPolicy("red")

    if species_type == "RUSHER":
        if hasattr(p, "mine_radius_check"):
            p.mine_radius_check = 0.1
        if hasattr(p, "defense_weight"):
            p.defense_weight = 0.0
        if hasattr(p, "flag_weight"):
            p.flag_weight = 5.0

    elif species_type == "CAMPER":
        if hasattr(p, "mine_radius_check"):
            p.mine_radius_check = 5.0
        if hasattr(p, "defense_weight"):
            p.defense_weight = 3.0

    return p


# -------------------------
# Elo + opponent specs
# -------------------------
def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(r_a - r_b) / 400.0))


@dataclass
class OpponentSpec:
    kind: str  # "SNAPSHOT" | "SPECIES" | "SCRIPTED"
    key: str   # snapshot path OR species tag OR scripted name
    rating: float


class LeagueManager:
    def __init__(
        self,
        pool_dir: str,
        mode: str,
        pool_max: int = 16,
        k_factor: float = 32.0,
        species_prob: float = 0.20,
        scripted_mix_floor: float = 0.10,
        snapshot_every_episodes: int = 50,
        min_pool_to_matchmake: int = 4,
        matchmaking_tau: float = 250.0,
        seed: int = 42,
    ) -> None:
        self.pool_dir = pool_dir
        os.makedirs(self.pool_dir, exist_ok=True)

        self.mode = str(mode)
        self.pool_max = int(pool_max)
        self.k = float(k_factor)

        self.species_prob = float(species_prob)
        self.scripted_mix_floor = float(scripted_mix_floor)

        self.snapshot_every_episodes = int(snapshot_every_episodes)
        self.min_pool_to_matchmake = int(min_pool_to_matchmake)
        self.tau = float(matchmaking_tau)

        self.rng = random.Random(int(seed))

        self.ratings: Dict[str, float] = {}
        self.snapshots: List[str] = []

        self.learner_rating: float = 1200.0
        self.learner_key: str = "__LEARNER__"

        for key in [
            "SCRIPTED:OP1",
            "SCRIPTED:OP2",
            "SCRIPTED:OP3",
            "SPECIES:RUSHER",
            "SPECIES:CAMPER",
            "SPECIES:BALANCED",
        ]:
            self.ratings[key] = 1200.0
        self.ratings[self.learner_key] = self.learner_rating

        # Curriculum: OP1 -> OP2 -> OP3 (Elo-gated)
        self.phase_names = ["OP1", "OP2", "OP3"]
        self.phase_idx = 0
        self.phase_episode_count = 0
        self.phase_min_episodes = dict(PPO_PHASE_MIN_EPISODES)
        self.phase_advance_elo_margin = float(PPO_PHASE_ELO_MARGIN)
        self.phase_required_win_by = dict(PPO_PHASE_REQUIRED_WIN_BY)
        self.curriculum_species_prob = float(PPO_CURRICULUM_SPECIES_PROB)
        self.curriculum_snapshot_prob = float(PPO_CURRICULUM_SNAPSHOT_PROB)
        self.curriculum_winrate_window = int(PPO_CURRICULUM_WINRATE_WINDOW)
        self.curriculum_min_winrate = float(PPO_CURRICULUM_MIN_WINRATE)
        self.curriculum_fallback_prob = float(PPO_CURRICULUM_FALLBACK_PROB)
        self.curriculum_op3_species_scale = float(PPO_CURRICULUM_OP3_SPECIES_SCALE)
        self.curriculum_op3_snapshot_scale = float(PPO_CURRICULUM_OP3_SNAPSHOT_SCALE)
        self.switch_to_elo_after_op3 = bool(PPO_SWITCH_TO_ELO_AFTER_OP3)
        self.last_pick_source: str = "scripted"
        self.last_pick_phase: str = self.phase_names[self.phase_idx]
        self.phase_recent_results: Dict[str, deque] = {
            name: deque(maxlen=self.curriculum_winrate_window) for name in self.phase_names
        }
        self.last_match_result: Optional[Dict[str, Any]] = None
        self.snapshot_disabled: bool = False

    def get_rating(self, key: str) -> float:
        return float(self.ratings.get(key, 1200.0))

    def set_learner_rating(self, r: float) -> None:
        self.learner_rating = float(r)
        self.ratings[self.learner_key] = float(r)

    def _snapshot_path(self, episode_idx: int) -> str:
        return os.path.join(self.pool_dir, f"sp_snapshot_ep{episode_idx:06d}")

    def add_snapshot(self, model: PPO, episode_idx: int) -> str:
        if self.snapshot_disabled:
            return ""
        path = self._snapshot_path(episode_idx)
        try:
            model.save(path)
        except Exception:
            self.snapshot_disabled = True
            raise
        zip_path = path + ".zip"

        self.snapshots.append(zip_path)
        self.ratings[zip_path] = float(self.learner_rating)

        while len(self.snapshots) > self.pool_max:
            old = self.snapshots.pop(0)
            try:
                os.remove(old)
            except Exception:
                pass
            self.ratings.pop(old, None)

        return zip_path

    def _sample_snapshot_by_elo(self) -> Optional[str]:
        if len(self.snapshots) < self.min_pool_to_matchmake:
            return self.snapshots[-1] if self.snapshots else None

        lr = float(self.learner_rating)
        weights = []
        for p in self.snapshots:
            r = self.get_rating(p)
            dist = abs(r - lr)
            w = math.exp(-dist / max(1e-6, self.tau)) + 1e-3
            weights.append(w)

        total = sum(weights)
        if total <= 0:
            return self.rng.choice(self.snapshots)

        pick = self.rng.random() * total
        acc = 0.0
        for p, w in zip(self.snapshots, weights):
            acc += w
            if acc >= pick:
                return p
        return self.snapshots[-1]

    def _sample_snapshot_uniform(self) -> Optional[str]:
        if not self.snapshots:
            return None
        return self.rng.choice(self.snapshots)

    def sample_opponent(self) -> OpponentSpec:
        # Curriculum mode: start OP1 -> OP2 -> OP3
        if self.mode == TRAIN_MODE_DEFAULT_LEAGUE:
            phase = self.phase_names[self.phase_idx]
            self.last_pick_phase = phase
            key = f"SCRIPTED:{phase}"
            self.last_pick_source = "scripted"
            return OpponentSpec(kind="SCRIPTED", key=phase, rating=self.get_rating(key))

        if self.mode == TRAIN_MODE_BASELINE_FIXED:
            key = "SCRIPTED:OP3"
            return OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.get_rating(key))

        if self.mode == TRAIN_MODE_BASELINE_SELFPLAY:
            snap = self._sample_snapshot_uniform()
            if snap is not None:
                return OpponentSpec(kind="SNAPSHOT", key=snap, rating=self.get_rating(snap))
            key = "SCRIPTED:OP3"
            return OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.get_rating(key))

        if self.rng.random() < self.species_prob:
            tag = self.rng.choice(["RUSHER", "CAMPER", "BALANCED"])
            key = f"SPECIES:{tag}"
            return OpponentSpec(kind="SPECIES", key=tag, rating=self.get_rating(key))

        if self.rng.random() < self.scripted_mix_floor:
            key = "SCRIPTED:OP3"
            return OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.get_rating(key))

        snap = self._sample_snapshot_by_elo()
        if snap is not None:
            return OpponentSpec(kind="SNAPSHOT", key=snap, rating=self.get_rating(snap))

        key = "SCRIPTED:OP3"
        return OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.get_rating(key))

    def update_elo(self, opponent_key: str, actual_score_for_learner: float) -> None:
        learner_r = float(self.learner_rating)
        opp_r = float(self.get_rating(opponent_key))

        exp = elo_expected(learner_r, opp_r)
        a = float(actual_score_for_learner)

        learner_new = learner_r + self.k * (a - exp)
        opp_new = opp_r + self.k * ((1.0 - a) - (1.0 - exp))

        self.set_learner_rating(learner_new)
        self.ratings[opponent_key] = float(opp_new)

        # Track per-phase winrate vs scripted opponents
        if opponent_key.startswith("SCRIPTED:"):
            phase = opponent_key.split(":", 1)[1]
            if phase in self.phase_recent_results:
                self.phase_recent_results[phase].append(float(actual_score_for_learner))

        # Advance curriculum phase based on Elo margin + min episodes
        if self.mode == TRAIN_MODE_DEFAULT_LEAGUE:
            self.phase_episode_count += 1
            phase = self.phase_names[self.phase_idx]
            min_eps = int(self.phase_min_episodes.get(phase, 0))
            target_key = f"SCRIPTED:{phase}"
            opp_r = float(self.get_rating(target_key))
            required_win_by = int(self.phase_required_win_by.get(phase, 0))
            meets_score_gate = True
            if required_win_by > 0:
                meets_score_gate = False
                if (
                    self.last_match_result
                    and self.last_match_result.get("opponent_key_for_elo") == target_key
                    and self.last_match_result.get("result") == "WIN"
                ):
                    blue_score = int(self.last_match_result.get("blue_score", 0))
                    red_score = int(self.last_match_result.get("red_score", 0))
                    if (blue_score - red_score) >= required_win_by:
                        meets_score_gate = True
            if (
                self.phase_idx < (len(self.phase_names) - 1)
                and self.phase_episode_count >= min_eps
                and self.learner_rating >= (opp_r + float(self.phase_advance_elo_margin))
                and meets_score_gate
            ):
                self.phase_idx += 1
                self.phase_episode_count = 0

            # Switch to Elo only after first real win vs scripted OP3.
            if self.switch_to_elo_after_op3 and self.phase_idx == (len(self.phase_names) - 1):
                if (
                    self.last_match_result
                    and self.last_match_result.get("opponent_key_for_elo") == "SCRIPTED:OP3"
                    and self.last_match_result.get("result") == "WIN"
                ):
                    self.mode = TRAIN_MODE_ELO_LEAGUE


# ==========================================================
# DirectML-safe action encoding (Discrete)
# ==========================================================
def encode_pair(macro_idx: int, tgt_idx: int, n_targets: int) -> int:
    macro_idx = int(macro_idx) % N_MACROS
    tgt_idx = int(tgt_idx) % max(1, int(n_targets))
    return macro_idx * int(n_targets) + tgt_idx


def decode_pair(a: int, n_targets: int) -> Tuple[int, int]:
    n_targets = max(1, int(n_targets))
    a = int(a) % (N_MACROS * n_targets)
    macro_idx = a // n_targets
    tgt_idx = a % n_targets
    return int(macro_idx), int(tgt_idx)


def encode_joint(b0: Tuple[int, int], b1: Tuple[int, int], n_targets: int) -> int:
    joint = N_MACROS * max(1, int(n_targets))
    a0 = encode_pair(b0[0], b0[1], n_targets)
    a1 = encode_pair(b1[0], b1[1], n_targets)
    return int(a0 + joint * a1)


def decode_joint(action: int, n_targets: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    joint = N_MACROS * max(1, int(n_targets))
    action = int(action)
    a0 = action % joint
    a1 = action // joint
    b0 = decode_pair(a0, n_targets)
    b1 = decode_pair(a1, n_targets)
    return b0, b1


# -------------------------
# Env: SB3 wrapper controlling 2 blue agents
# -------------------------
class CTFPPOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, league: LeagueManager, seed: int = 42) -> None:
        super().__init__()
        self.league = league
        self.rng = random.Random(int(seed))

        self.env = make_game_field(
            map_name=MAP_NAME or None,
            map_path=MAP_PATH or None,
            rows=CNN_ROWS,
            cols=CNN_COLS,
        )
        self.gm: GameManager = self.env.getGameManager()
        if hasattr(self.env, "set_seed"):
            try:
                self.env.set_seed(int(seed))
            except Exception:
                pass

        self.env.set_external_control("blue", True)

        self.n_targets = int(getattr(self.env, "num_macro_targets", 8) or 8)
        self._joint = N_MACROS * self.n_targets
        self._n_actions = self._joint * self._joint

        # ✅ DirectML-safe: Discrete joint-action
        self.action_space = spaces.Discrete(self._n_actions)

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(NUM_CNN_CHANNELS * 2, CNN_ROWS, CNN_COLS),
                    dtype=np.float32,
                ),
                "mask": spaces.Box(low=0.0, high=1.0, shape=(N_MACROS * 2,), dtype=np.float32),
            }
        )

        self.step_count = 0
        self.prev_blue_score = 0
        self.prev_red_score = 0
        self.windows_since_score = 0

        self.blue_uids: List[str] = []
        self.pending_reward: Dict[str, float] = {}

        self.opponent: OpponentSpec = OpponentSpec(kind="SCRIPTED", key="OP3", rating=1200.0)
        self.opponent_key_for_elo: str = "SCRIPTED:OP3"
        self.opp_model: Optional[PPO] = None

        self.last_result: Optional[Dict[str, Any]] = None
        self.current_pick_source: str = "scripted"
        self.current_pick_phase: str = "OP1"
        self.invalid_macro_penalty = float(INVALID_MACRO_PENALTY)
        self.wrapper_to_env: List[int] = list(range(N_MACROS))
        self.env_to_wrapper: Dict[int, int] = {i: i for i in range(N_MACROS)}
        self._macro_counts = np.zeros((N_MACROS,), dtype=np.int64)
        self._macro_total = 0

        # Reward breakdown (per-episode)
        self._reward_events_sum = 0.0
        self._score_shaping_sum = 0.0
        self._invalid_penalty_sum = 0.0
        self._stall_penalty_sum = 0.0
        self._time_penalty_sum = 0.0
        self._terminal_bonus = 0.0
        self._macro_counts[:] = 0
        self._macro_total = 0

        # Behavior logging
        self.behavior_logger = BehaviorLogger()
        self.behavior_csv_path = os.path.join(DEFAULT_LOG_DIR, "behavior_ppo.csv")
        self._last_written_idx = 0
        self._behavior_logging_enabled = True

        self.gm.score_limit = int(SCORE_LIMIT)
        self.gm.max_time = float(MAX_TIME)
        if hasattr(self.gm, "set_shaping_gamma"):
            try:
                self.gm.set_shaping_gamma(GAMMA)
            except Exception:
                pass

    def _rebuild_action_space(self) -> None:
        self.n_targets = int(getattr(self.env, "num_macro_targets", 8) or 8)
        self._joint = N_MACROS * max(1, self.n_targets)
        self._n_actions = self._joint * self._joint
        self.action_space = spaces.Discrete(self._n_actions)

    def _build_macro_index_maps(self) -> None:
        env_order = None
        for attr in ["macro_order", "macros", "macro_actions", "available_macros"]:
            if hasattr(self.env, attr):
                env_order = getattr(self.env, attr)
                break

        if env_order is None:
            self.wrapper_to_env = list(range(N_MACROS))
            self.env_to_wrapper = {i: i for i in range(N_MACROS)}
            return

        if isinstance(env_order, dict):
            env_order = list(env_order.keys())
        env_order = list(env_order)

        def _key(x: Any) -> str:
            name = getattr(x, "name", None)
            return str(name if name is not None else x)

        env_idx_by_key = {_key(m): i for i, m in enumerate(env_order)}
        wrapper_keys = [_key(m) for m in USED_MACROS]

        wrapper_to_env: List[int] = []
        for wi, key in enumerate(wrapper_keys):
            if key in env_idx_by_key:
                wrapper_to_env.append(int(env_idx_by_key[key]))
            else:
                wrapper_to_env.append(int(wi))

        self.wrapper_to_env = wrapper_to_env
        self.env_to_wrapper = {ei: wi for wi, ei in enumerate(wrapper_to_env)}

    def _sim_decision_window(self) -> None:
        if DECISION_WINDOW <= 0.0:
            return
        n_full = int(DECISION_WINDOW // SIM_DT)
        rem = float(DECISION_WINDOW - n_full * SIM_DT)

        for _ in range(n_full):
            if self.gm.game_over:
                return
            self.env.update(SIM_DT)

        if rem > 1e-9 and (not self.gm.game_over):
            self.env.update(rem)

    def _build_team_obs_and_mask(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        agents = self.env.blue_agents if side == "blue" else self.env.red_agents
        live = [a for a in agents if a is not None]
        while len(live) < 2:
            live.append(live[0] if live else None)

        obs_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []

        for a in live[:2]:
            if a is None:
                obs_list.append(np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32))
                mask_list.append(np.ones((N_MACROS,), dtype=np.float32))
                continue

            o = np.asarray(self.env.build_observation(a), dtype=np.float32)
            obs_list.append(o)

            mm = np.asarray(self.env.get_macro_mask(a), dtype=np.bool_).reshape(-1)
            if mm.shape != (N_MACROS,) or (not mm.any()):
                mm = np.ones((N_MACROS,), dtype=np.bool_)
            wrapper_mm = np.zeros((N_MACROS,), dtype=np.bool_)
            for wi, ei in enumerate(self.wrapper_to_env):
                if 0 <= ei < len(mm):
                    wrapper_mm[wi] = bool(mm[ei])
                else:
                    wrapper_mm[wi] = True
            if not wrapper_mm.any():
                wrapper_mm = np.ones((N_MACROS,), dtype=np.bool_)
            mask_list.append(wrapper_mm.astype(np.float32))

        grid = np.concatenate(obs_list, axis=0)
        mask = np.concatenate(mask_list, axis=0)
        return grid, mask

    def _get_obs(self) -> Dict[str, np.ndarray]:
        grid, mask = self._build_team_obs_and_mask("blue")
        return {"grid": grid, "mask": mask}

    def _init_episode_routing(self) -> None:
        self.blue_uids = [agent_uid(a) for a in self.env.blue_agents if a is not None]
        self.pending_reward = {uid: 0.0 for uid in self.blue_uids}
        clear_reward_events_best_effort(self.gm)

    def _accumulate_rewards(self) -> None:
        events = pop_reward_events_strict(self.gm)
        for _t, aid, r in events:
            if aid is None:
                continue
            k = str(aid)
            if k in self.pending_reward:
                self.pending_reward[k] += float(r)
                self._reward_events_sum += float(r)

    def _apply_score_delta_shaping(self) -> None:
        cur_b = int(getattr(self.gm, "blue_score", 0))
        cur_r = int(getattr(self.gm, "red_score", 0))
        db = cur_b - int(self.prev_blue_score)
        dr = cur_r - int(self.prev_red_score)
        self.prev_blue_score, self.prev_red_score = cur_b, cur_r

        shaped = 0.0
        if db != 0:
            shaped += float(BLUE_SCORED_BONUS) * float(db) * float(len(self.blue_uids))
        if dr != 0:
            shaped += float(RED_SCORED_PENALTY) * float(dr) * float(len(self.blue_uids))

        if db == 0 and dr == 0:
            self.windows_since_score += 1
            if self.windows_since_score > int(NO_SCORE_GRACE_WINDOWS):
                penalty = float(NO_SCORE_PENALTY_PER_WINDOW) * float(len(self.blue_uids))
                shaped -= penalty
                self._stall_penalty_sum -= penalty
        else:
            self.windows_since_score = 0

        if shaped != 0.0 and self.blue_uids:
            per = shaped / float(len(self.blue_uids))
            for uid in self.blue_uids:
                self.pending_reward[uid] += per
            self._score_shaping_sum += shaped

    def _consume_team_reward(self) -> float:
        r = 0.0
        for uid in self.blue_uids:
            r += float(self.pending_reward.get(uid, 0.0))
            self.pending_reward[uid] = 0.0

        time_penalty = float(TIME_PENALTY_PER_AGENT_PER_SEC) * float(len(self.blue_uids)) * float(DECISION_WINDOW)
        r -= time_penalty
        self._time_penalty_sum -= time_penalty
        return float(r)

    def _set_opponent(self, opp: OpponentSpec) -> None:
        self.opponent = opp
        self.current_pick_source = getattr(self.league, "last_pick_source", "scripted")
        self.current_pick_phase = getattr(self.league, "last_pick_phase", "OP1")

        if opp.kind == "SNAPSHOT":
            self.opponent_key_for_elo = opp.key
        elif opp.kind == "SPECIES":
            self.opponent_key_for_elo = f"SPECIES:{opp.key}"
        elif opp.kind == "SCRIPTED":
            self.opponent_key_for_elo = f"SCRIPTED:{opp.key}"
        else:
            self.opponent_key_for_elo = "SCRIPTED:OP3"

        if opp.kind == "SNAPSHOT":
            self.env.set_external_control("red", True)
            self.opp_model = PPO.load(opp.key, device="cpu")
            self.opp_model.policy.set_training_mode(False)

        elif opp.kind == "SPECIES":
            self.env.set_external_control("red", False)
            self.opp_model = None
            self.env.policies["red"] = make_species_policy(opp.key)

        else:
            self.env.set_external_control("red", False)
            self.opp_model = None
            if opp.key == "OP1":
                self.env.policies["red"] = OP1RedPolicy("red")
            elif opp.key == "OP2":
                self.env.policies["red"] = OP2RedPolicy("red")
            else:
                self.env.policies["red"] = OP3RedPolicy("red")

    def _build_red_external_actions_from_snapshot(self) -> Dict[str, Tuple[int, int]]:
        assert self.opp_model is not None
        grid, mask = self._build_team_obs_and_mask("red")
        obs = {"grid": grid, "mask": mask}

        # deterministic snapshots
        act, _ = self.opp_model.predict(obs, deterministic=True)
        act_int = int(np.asarray(act).reshape(-1)[0])

        (r0_macro, r0_tgt), (r1_macro, r1_tgt) = decode_joint(act_int, self.n_targets)

        red_agents = [a for a in self.env.red_agents if a is not None]
        while len(red_agents) < 2:
            red_agents.append(red_agents[0] if red_agents else None)

        actions: Dict[str, Tuple[int, int]] = {}
        if red_agents[0] is not None:
            actions[agent_uid(red_agents[0])] = (r0_macro, r0_tgt)
        if red_agents[1] is not None:
            actions[agent_uid(red_agents[1])] = (r1_macro, r1_tgt)
        return actions

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng.seed(int(seed))
            np.random.seed(int(seed))
            th.manual_seed(int(seed))
            if hasattr(self.env, "set_seed"):
                try:
                    self.env.set_seed(int(seed))
                except Exception:
                    pass

        self.last_result = None
        self.step_count = 0

        self.env.reset_default()
        self.gm = self.env.getGameManager()

        self.behavior_logger.start_episode()

        # re-assert external control (reset_default can wipe it)
        self.env.set_external_control("blue", True)

        self.gm.score_limit = int(SCORE_LIMIT)
        self.gm.max_time = float(MAX_TIME)

        self.prev_blue_score = int(getattr(self.gm, "blue_score", 0))
        self.prev_red_score = int(getattr(self.gm, "red_score", 0))
        self.windows_since_score = 0

        self._reward_events_sum = 0.0
        self._score_shaping_sum = 0.0
        self._invalid_penalty_sum = 0.0
        self._stall_penalty_sum = 0.0
        self._time_penalty_sum = 0.0
        self._terminal_bonus = 0.0

        # refresh action space if targets changed
        self._rebuild_action_space()
        self._build_macro_index_maps()

        opp = self.league.sample_opponent()
        self._set_opponent(opp)

        self._init_episode_routing()

        obs = self._get_obs()
        info = {"opponent_kind": opp.kind, "opponent_key": opp.key, "opponent_rating": opp.rating}
        return obs, info

    def step(self, action):
        self.step_count += 1

        # ✅ Discrete joint decode for blue0/blue1 (macro,target)
        act_int = int(np.asarray(action).reshape(-1)[0])
        (b0_macro, b0_tgt), (b1_macro, b1_tgt) = decode_joint(act_int, self.n_targets)

        blue_agents = [x for x in self.env.blue_agents if x is not None]
        while len(blue_agents) < 2:
            blue_agents.append(blue_agents[0] if blue_agents else None)

        invalid_count = 0
        submit: Dict[str, Tuple[int, int]] = {}

        for agent, macro_idx, tgt_idx in [
            (blue_agents[0], b0_macro, b0_tgt),
            (blue_agents[1], b1_macro, b1_tgt),
        ]:
            if agent is None or (hasattr(agent, "isEnabled") and not agent.isEnabled()):
                continue

            mm = np.asarray(self.env.get_macro_mask(agent), dtype=np.bool_).reshape(-1)
            if mm.shape != (N_MACROS,) or (not mm.any()):
                mm = np.ones((N_MACROS,), dtype=np.bool_)

            macro_idx = int(macro_idx) % N_MACROS
            tgt_idx = int(tgt_idx) % max(1, self.n_targets)

            env_macro_idx = int(self.wrapper_to_env[macro_idx]) if self.wrapper_to_env else int(macro_idx)
            if not bool(mm[env_macro_idx]):
                macro_idx = 0  # force legal fallback (GO_TO)
                env_macro_idx = int(self.wrapper_to_env[macro_idx]) if self.wrapper_to_env else int(macro_idx)
                invalid_count += 1

            submit[agent_uid(agent)] = (env_macro_idx, tgt_idx)
            self.behavior_logger.log_decision(agent, int(macro_idx))
            self._macro_counts[int(macro_idx)] += 1
            self._macro_total += 1

        if self.opponent.kind == "SNAPSHOT":
            submit.update(self._build_red_external_actions_from_snapshot())

        self.env.submit_external_actions(submit)
        self._sim_decision_window()

        terminated = bool(self.gm.game_over)
        truncated = bool(self.step_count >= MAX_MACRO_STEPS)
        done = bool(terminated or truncated)

        self._accumulate_rewards()
        self._apply_score_delta_shaping()
        reward = self._consume_team_reward()

        if done:
            terminal_bonus = outcome_bonus(int(self.gm.blue_score), int(self.gm.red_score))
            reward += terminal_bonus
            self._terminal_bonus += float(terminal_bonus)

            if self.gm.blue_score > self.gm.red_score:
                actual = 1.0
                result = "WIN"
            elif self.gm.blue_score < self.gm.red_score:
                actual = 0.0
                result = "LOSS"
            else:
                actual = 0.5
                result = "DRAW"

            self.last_result = {
                "result": result,
                "actual_score": actual,
                "opponent_key_for_elo": self.opponent_key_for_elo,
                "blue_score": int(self.gm.blue_score),
                "red_score": int(self.gm.red_score),
                "opponent_kind": self.opponent.kind,
                "opponent_key": self.opponent.key,
                "pick_source": self.current_pick_source,
                "pick_phase": self.current_pick_phase,
                "reward_events_sum": float(self._reward_events_sum),
                "score_shaping_sum": float(self._score_shaping_sum),
                "invalid_penalty_sum": float(self._invalid_penalty_sum),
                "stall_penalty_sum": float(self._stall_penalty_sum),
                "time_penalty_sum": float(self._time_penalty_sum),
                "terminal_bonus": float(self._terminal_bonus),
                "macro_choice_avg": (
                    (self._macro_counts / float(self._macro_total)).tolist()
                    if self._macro_total > 0
                    else [0.0 for _ in range(N_MACROS)]
                ),
            }

            self.behavior_logger.finalize_episode(
                self.gm,
                meta={
                    "episode_steps": int(self.step_count),
                    "opponent_kind": self.opponent.kind,
                    "opponent_key": self.opponent.key,
                    "map_name": MAP_NAME or "",
                    "map_path": MAP_PATH or "",
                    "result": result,
                },
            )
            new_eps = self.behavior_logger.episodes[self._last_written_idx :]
            if new_eps:
                records: List[Dict[str, Any]] = []
                for ep in new_eps:
                    records.extend(ep.to_flat_records())
                if self._behavior_logging_enabled:
                    try:
                        append_records_csv(self.behavior_csv_path, records)
                        self._last_written_idx = len(self.behavior_logger.episodes)
                    except Exception as exc:
                        self._behavior_logging_enabled = False
                        global _BEHAVIOR_LOGGING_DISABLED_NOTICE
                        if not _BEHAVIOR_LOGGING_DISABLED_NOTICE:
                            _BEHAVIOR_LOGGING_DISABLED_NOTICE = True
                            print(f"[WARN] behavior CSV write failed; disabling logging. err={exc}")

        obs = self._get_obs()
        info = {
            "opponent_kind": self.opponent.kind,
            "opponent_key": self.opponent.key,
            "invalid_macros": invalid_count,
            "blue_score": int(self.gm.blue_score),
            "red_score": int(self.gm.red_score),
        }
        if done and self.last_result is not None:
            info["match_result"] = dict(self.last_result)

        return obs, float(reward), terminated, truncated, info


# -------------------------
# Custom extractor (grid + mask)
# -------------------------
class CTFCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        grid_space: spaces.Box = observation_space["grid"]
        mask_space: spaces.Box = observation_space["mask"]

        c, h, w = grid_space.shape
        mask_dim = int(np.prod(mask_space.shape))

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros((1, c, h, w))
            n_flat = self.cnn(sample).shape[1]

        self.mask_mlp = nn.Sequential(
            nn.Linear(mask_dim, 64),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(n_flat + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        grid = obs["grid"]
        mask = obs["mask"]
        x1 = self.cnn(grid)
        x2 = self.mask_mlp(mask)
        return self.fc(th.cat([x1, x2], dim=1))


# -------------------------
# Callback: Elo + snapshots
# -------------------------
class LeagueSelfPlayCallback(BaseCallback):
    def __init__(self, league: LeagueManager, save_dir: str, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.league = league
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.episode_idx = 0
        self.best_eval_winrate = -1.0
        self._invalid_macro_accum = 0
        self._invalid_macro_steps = 0
        self._macro_avg_accum = np.zeros((N_MACROS,), dtype=np.float64)
        self._macro_avg_count = 0
        self._checkpoint_disabled = False

    def _log_eval(self, eval_env: DummyVecEnv) -> None:
        wins = 0
        total = 0
        total_reward = 0.0

        for _ in range(int(EVAL_EPISODES)):
            reset_out = eval_env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = eval_env.step(action)
                done = bool(dones[0]) if dones is not None else False
                ep_reward += float(reward[0]) if isinstance(reward, (list, np.ndarray)) else float(reward)
                info = infos[0] if infos else {}
                if done and "match_result" in info:
                    if info["match_result"].get("result") == "WIN":
                        wins += 1
            total += 1
            total_reward += ep_reward

        winrate = float(wins) / float(max(1, total))
        mean_reward = total_reward / float(max(1, total))
        self.logger.record("eval/winrate", winrate)
        self.logger.record("eval/ep_rew_mean", mean_reward)

        if winrate > self.best_eval_winrate:
            self.best_eval_winrate = winrate
            best_path = os.path.join(self.save_dir, f"best_model_{RUN_TAG}")
            self.model.save(best_path)
            if self.verbose:
                print(f"[EVAL] new best winrate={winrate:.3f} saved={os.path.basename(best_path)}.zip")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        timesteps = int(self.num_timesteps)

        for info in infos:
            if info and "invalid_macros" in info:
                self._invalid_macro_accum += int(info.get("invalid_macros", 0))
                self._invalid_macro_steps += 1

        for i, done in enumerate(dones):
            if not done:
                continue

            self.episode_idx += 1
            info = infos[i] if i < len(infos) else {}
            mr = info.get("match_result", None)
            if mr is None:
                continue

            opp_key = str(mr["opponent_key_for_elo"])
            actual = float(mr["actual_score"])
            self.league.last_match_result = dict(mr)
            self.league.update_elo(opp_key, actual)

            if self.verbose:
                if self.league.mode == TRAIN_MODE_DEFAULT_LEAGUE:
                    pick_phase = str(mr.get("pick_phase", self.league.phase_names[self.league.phase_idx]))
                    phase = pick_phase
                else:
                    phase = "ELO"
                    pick_phase = str(mr.get("pick_phase", "OP3"))
                pick = str(mr.get("pick_source", "scripted"))
                show_elo = (self.league.mode == TRAIN_MODE_ELO_LEAGUE)
                elo_str = (
                    f" learner_elo={self.league.learner_rating:.1f} opp_elo={self.league.get_rating(opp_key):.1f}"
                    if show_elo
                    else ""
                )
                opp_label = str(mr["opponent_kind"])
                if opp_label == "SPECIES":
                    opp_label = str(mr.get("opponent_key", opp_label))
                print(
                    f"[LEAGUE|{self.league.mode}] ep={self.episode_idx} result={mr['result']} "
                    f"score={mr['blue_score']}:{mr['red_score']} "
                    f"opp={opp_label} pick={pick} phase={phase} pick_phase={pick_phase}"
                    f"{elo_str}"
                )
                macro_avg = mr.get("macro_choice_avg")
                if isinstance(macro_avg, list) and len(macro_avg) == N_MACROS:
                    self._macro_avg_accum += np.asarray(macro_avg, dtype=np.float64)
                    self._macro_avg_count += 1
                    if (self._macro_avg_count % 50) == 0:
                        avg = self._macro_avg_accum / float(self._macro_avg_count)
                        names = [getattr(m, "name", str(m)) for m in USED_MACROS]
                        parts = [f"{n}={v:.2f}" for n, v in zip(names, avg.tolist())]
                        print(f"[MACRO_AVG|last50] " + ", ".join(parts))
                        self._macro_avg_accum[:] = 0.0
                        self._macro_avg_count = 0

            self.logger.record("league/phase", phase, exclude=("tensorboard",))
            self.logger.record("league/pick_phase", pick_phase, exclude=("tensorboard",))
            self.logger.record("league/pick_source", pick, exclude=("tensorboard",))
            if self.league.mode == TRAIN_MODE_ELO_LEAGUE:
                self.logger.record("league/learner_elo", float(self.league.learner_rating))
                self.logger.record("league/opp_elo", float(self.league.get_rating(opp_key)))
            if self._invalid_macro_steps > 0:
                self.logger.record(
                    "env/invalid_macros_mean",
                    float(self._invalid_macro_accum) / float(self._invalid_macro_steps),
                )

            if (self.episode_idx % self.league.snapshot_every_episodes) == 0 and not self.league.snapshot_disabled:
                try:
                    snap = self.league.add_snapshot(self.model, self.episode_idx)
                except Exception as exc:
                    print(f"[WARN] snapshot failed; disabling snapshots. err={exc}")
                else:
                    if self.verbose and snap:
                        print(f"[SNAPSHOT|{self.league.mode}] saved={os.path.basename(snap)} pool={len(self.league.snapshots)}")

        if (
            not self._checkpoint_disabled
            and SAVE_EVERY_STEPS > 0
            and timesteps > 0
            and (timesteps % int(SAVE_EVERY_STEPS)) == 0
        ):
            ckpt_path = os.path.join(self.save_dir, f"ckpt_{RUN_TAG}_{timesteps}")
            try:
                self.model.save(ckpt_path)
            except Exception as exc:
                self._checkpoint_disabled = True
                print(f"[WARN] checkpoint save failed; disabling checkpoints. err={exc}")
            else:
                if self.verbose:
                    print(f"[CHECKPOINT] saved={os.path.basename(ckpt_path)}.zip")

        if EVAL_EVERY_STEPS > 0 and timesteps > 0 and (timesteps % int(EVAL_EVERY_STEPS)) == 0:
            eval_league = LeagueManager(
                pool_dir=os.path.join(self.save_dir, "eval_pool"),
                mode=TRAIN_MODE_BASELINE_FIXED,
                pool_max=1,
                k_factor=0.0,
                species_prob=0.0,
                scripted_mix_floor=1.0,
                snapshot_every_episodes=999999,
                min_pool_to_matchmake=0,
                matchmaking_tau=1.0,
                seed=SEED,
            )
            eval_env = DummyVecEnv([make_env_fn(eval_league)])
            eval_env = VecMonitor(eval_env)
            try:
                self._log_eval(eval_env)
            finally:
                eval_env.close()

        return True


class EntCoefScheduleCallback(BaseCallback):
    def __init__(self, start: float, end: float):
        super().__init__()
        self._schedule = linear_schedule(start, end)

    def _on_step(self) -> bool:
        if hasattr(self.model, "_current_progress_remaining"):
            progress = float(self.model._current_progress_remaining)
            self.model.ent_coef = float(self._schedule(progress))
        return True


class InvalidPenaltyScheduleCallback(BaseCallback):
    def __init__(self, start: float, end: float):
        super().__init__()
        self._schedule = linear_schedule(start, end)

    def _on_step(self) -> bool:
        if hasattr(self.model, "_current_progress_remaining"):
            progress = float(self.model._current_progress_remaining)
            penalty = float(self._schedule(progress))
            env = self.model.get_env()
            if env is not None:
                env.set_attr("invalid_macro_penalty", penalty)
        return True


# -------------------------
# Training entry
# -------------------------
def make_env_fn(league: LeagueManager):
    def _fn():
        return CTFPPOEnv(league=league, seed=SEED)
    return _fn


def linear_schedule(start: float, end: float):
    start = float(start)
    end = float(end)

    def _fn(progress_remaining: float) -> float:
        return end + (start - end) * float(progress_remaining)

    return _fn


def train_phase1_sb3():
    random.seed(SEED)
    np.random.seed(SEED)
    th.manual_seed(SEED)

    league = LeagueManager(
        pool_dir=POOL_DIR,
        mode=TRAIN_MODE,
        pool_max=16,
        k_factor=32.0,
        species_prob=0.20,
        scripted_mix_floor=0.10,
        snapshot_every_episodes=50,
        min_pool_to_matchmake=4,
        matchmaking_tau=250.0,
        seed=SEED,
    )

    force_dummy = TRAIN_MODE in (TRAIN_MODE_DEFAULT_LEAGUE, TRAIN_MODE_ELO_LEAGUE)
    n_envs = 1 if force_dummy else max(1, int(PPO_N_ENVS))
    n_steps = max(128, int(PPO_N_STEPS))
    rollout_batch = n_envs * n_steps
    batch_size = max(32, int(PPO_BATCH_SIZE))
    if rollout_batch % batch_size != 0:
        for cand in range(batch_size, 0, -1):
            if rollout_batch % cand == 0:
                batch_size = cand
                break
        print(f"[PPO] adjusted batch_size={batch_size} to divide rollout_batch={rollout_batch}")

    env_fns = [make_env_fn(league) for _ in range(n_envs)]
    if force_dummy:
        print("[PPO] League mode uses DummyVecEnv to keep curriculum state consistent.")
        venv = DummyVecEnv(env_fns)
    else:
        try:
            venv = SubprocVecEnv(env_fns)
        except Exception:
            print("[PPO] SubprocVecEnv failed, falling back to DummyVecEnv.")
            venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv)  # ✅ no double-Monitor warning

    policy_kwargs = dict(
        features_extractor_class=CTFCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    tb_dir = os.path.join(CHECKPOINT_DIR, "tb", RUN_TAG)

    def _make_model(device):
        return PPO(
        policy="MultiInputPolicy",
        env=venv,
        learning_rate=linear_schedule(LR_START, LR_END),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.2,
        ent_coef=float(ENT_COEF_START),
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tb_dir,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=SEED,
        device=device,
        )

    model = _make_model(DML_DEVICE)
    model.set_logger(configure(tb_dir, ["tensorboard"]))

    cb = CallbackList(
        [
            LeagueSelfPlayCallback(league=league, save_dir=CHECKPOINT_DIR, verbose=1),
            EntCoefScheduleCallback(ENT_COEF_START, ENT_COEF_END),
        ]
    )
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb)
    except RuntimeError as exc:
        msg = str(exc)
        if USING_DML and "DirectML" in msg and "scatter" in msg:
            print("[WARN] DirectML scatter error; falling back to CPU for training.")
            tmp_path = os.path.join(CHECKPOINT_DIR, f"dml_fallback_{RUN_TAG}")
            model.save(tmp_path)
            model = PPO.load(tmp_path, env=venv, device="cpu")
            remaining = max(0, int(TOTAL_TIMESTEPS) - int(model.num_timesteps))
            if remaining > 0:
                model.learn(total_timesteps=remaining, callback=cb, reset_num_timesteps=False)
        else:
            raise

    final_path = os.path.join(CHECKPOINT_DIR, f"research_model_{RUN_TAG}")
    try:
        model.save(final_path)
        print(f"\nTraining complete. Final model saved to: {final_path}.zip")
    except Exception as exc:
        print(f"[WARN] final model save failed. err={exc}")


if __name__ == "__main__":
    print(f"[RUN] TRAIN_MODE={TRAIN_MODE} (default={TRAIN_MODE_DEFAULT_LEAGUE})")
    train_phase1_sb3()
