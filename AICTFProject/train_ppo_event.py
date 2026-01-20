# ==========================================================
# train_ppo_event.py  (SB3 PPO + PHASE 1 LEAGUE/ELO + 2 BASELINES)  [PATCHED]
#   Recommended fixes applied:
#     ✅ Re-assert blue external control after env.reset_default()
#     ✅ Make snapshot opponents deterministic (more stable Elo + cleaner eval)
#     ✅ Increase invalid macro penalty (SB3 can't hard-mask logits)
#     ✅ Make time penalty time-consistent (scales with DECISION_WINDOW)
#     ✅ Optional: faster control loop knobs (commented)
# ==========================================================

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Gymnasium + SB3
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Torch for custom extractor
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Your code
from game_field import GameField, CNN_COLS, CNN_ROWS, NUM_CNN_CHANNELS
from game_manager import GameManager
from macro_actions import MacroAction
from policies import OP3RedPolicy


# -------------------------
# Config
# -------------------------
SEED = 42
TOTAL_TIMESTEPS = 1_000_000

# One env.step() == one "decision window" in your sim
DECISION_WINDOW = 1.0
SIM_DT = 0.1

# (Optional recommended: snappier, more "control-loop like")
# DECISION_WINDOW = 0.25
# SIM_DT = 0.05

# Episode caps (safety)
MAX_MACRO_STEPS = 500
SCORE_LIMIT = 3
MAX_TIME = 200.0

# Reward shaping
WIN_BONUS = 25.0
LOSS_PENALTY = -25.0
DRAW_PENALTY = -5.0
BLUE_SCORED_BONUS = 15.0
RED_SCORED_PENALTY = -15.0

# Time penalty is now interpreted as "per second per agent"
TIME_PENALTY_PER_AGENT_PER_SEC = 0.001

# Invalid action handling (SB3 PPO can't hard-mask logits)
# Recommended: higher than 0.05 so PPO learns to stay legal quickly.
INVALID_MACRO_PENALTY = 0.5

# Action space matches GameField.macro_order (you already set it)
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
TRAIN_MODE_DEFAULT_LEAGUE = "league_elo_species"         # your current method
TRAIN_MODE_BASELINE_FIXED = "baseline_fixed_opponent"    # vs OP3 scripted only
TRAIN_MODE_BASELINE_SELFPLAY = "baseline_selfplay_uniform"  # vs uniform snapshots

TRAIN_MODE = TRAIN_MODE_DEFAULT_LEAGUE
RUN_TAG = TRAIN_MODE  # used to name final model + tb log dir


# -------------------------
# Helpers: stable agent id and reward routing
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


# -------------------------
# League Manager with mode switches (DEFAULT + 2 baselines)
# -------------------------
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

        for key in ["SCRIPTED:OP3", "SPECIES:RUSHER", "SPECIES:CAMPER", "SPECIES:BALANCED"]:
            self.ratings[key] = 1200.0
        self.ratings[self.learner_key] = self.learner_rating

    def get_rating(self, key: str) -> float:
        return float(self.ratings.get(key, 1200.0))

    def set_learner_rating(self, r: float) -> None:
        self.learner_rating = float(r)
        self.ratings[self.learner_key] = float(r)

    def _snapshot_path(self, episode_idx: int) -> str:
        return os.path.join(self.pool_dir, f"sp_snapshot_ep{episode_idx:06d}")

    def add_snapshot(self, model: PPO, episode_idx: int) -> str:
        path = self._snapshot_path(episode_idx)
        model.save(path)
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
        if self.mode == TRAIN_MODE_BASELINE_FIXED:
            key = "SCRIPTED:OP3"
            return OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.get_rating(key))

        if self.mode == TRAIN_MODE_BASELINE_SELFPLAY:
            snap = self._sample_snapshot_uniform()
            if snap is not None:
                return OpponentSpec(kind="SNAPSHOT", key=snap, rating=self.get_rating(snap))
            key = "SCRIPTED:OP3"
            return OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.get_rating(key))

        # DEFAULT: Elo league + species injection
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


# -------------------------
# Env: SB3 single-agent wrapper controlling 2 blue agents at once
# -------------------------
class CTFPPOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, league: LeagueManager, seed: int = 42) -> None:
        super().__init__()
        self.league = league
        self.rng = random.Random(int(seed))

        self.grid = [[0] * CNN_COLS for _ in range(CNN_ROWS)]
        self.env: GameField = GameField(self.grid)
        self.gm: GameManager = self.env.getGameManager()

        # Initial external control (also re-applied each reset)
        self.env.set_external_control("blue", True)

        self.n_targets = int(getattr(self.env, "num_macro_targets", 8) or 8)

        self.action_space = spaces.MultiDiscrete([N_MACROS, self.n_targets, N_MACROS, self.n_targets])

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

        self.blue_uids: List[str] = []
        self.pending_reward: Dict[str, float] = {}

        self.opponent: OpponentSpec = OpponentSpec(kind="SCRIPTED", key="OP3", rating=1200.0)
        self.opponent_key_for_elo: str = "SCRIPTED:OP3"
        self.opp_model: Optional[PPO] = None

        self.last_result: Optional[Dict[str, Any]] = None

        self.gm.score_limit = int(SCORE_LIMIT)
        self.gm.max_time = float(MAX_TIME)

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
            mask_list.append(mm.astype(np.float32))

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

        if shaped != 0.0 and self.blue_uids:
            per = shaped / float(len(self.blue_uids))
            for uid in self.blue_uids:
                self.pending_reward[uid] += per

    def _consume_team_reward(self) -> float:
        r = 0.0
        for uid in self.blue_uids:
            r += float(self.pending_reward.get(uid, 0.0))
            self.pending_reward[uid] = 0.0

        # ✅ time-consistent penalty
        r -= float(TIME_PENALTY_PER_AGENT_PER_SEC) * float(len(self.blue_uids)) * float(DECISION_WINDOW)
        return float(r)

    def _set_opponent(self, opp: OpponentSpec) -> None:
        self.opponent = opp

        if opp.kind == "SNAPSHOT":
            self.opponent_key_for_elo = opp.key
        elif opp.kind == "SPECIES":
            self.opponent_key_for_elo = f"SPECIES:{opp.key}"
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
            self.env.policies["red"] = OP3RedPolicy("red")

    def _build_red_external_actions_from_snapshot(self) -> Dict[str, Tuple[int, int]]:
        assert self.opp_model is not None
        grid, mask = self._build_team_obs_and_mask("red")
        obs = {"grid": grid, "mask": mask}

        # ✅ recommended: deterministic snapshot opponents
        act, _ = self.opp_model.predict(obs, deterministic=True)
        act = np.asarray(act).reshape(-1).astype(int)

        red_agents = [a for a in self.env.red_agents if a is not None]
        while len(red_agents) < 2:
            red_agents.append(red_agents[0] if red_agents else None)

        actions: Dict[str, Tuple[int, int]] = {}
        for i, agent in enumerate(red_agents[:2]):
            if agent is None:
                continue
            macro_idx = int(act[i * 2 + 0]) % N_MACROS
            target_idx = int(act[i * 2 + 1]) % self.n_targets
            actions[agent_uid(agent)] = (macro_idx, target_idx)
        return actions

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng.seed(int(seed))
            np.random.seed(int(seed))
            th.manual_seed(int(seed))

        self.last_result = None
        self.step_count = 0

        self.env.reset_default()
        self.gm = self.env.getGameManager()

        # ✅ recommended fix: re-assert external control after reset_default()
        self.env.set_external_control("blue", True)

        self.gm.score_limit = int(SCORE_LIMIT)
        self.gm.max_time = float(MAX_TIME)

        self.prev_blue_score = int(getattr(self.gm, "blue_score", 0))
        self.prev_red_score = int(getattr(self.gm, "red_score", 0))

        # Ensure action space uses current macro target count (stable in your env, but safe)
        self.n_targets = int(getattr(self.env, "num_macro_targets", 8) or 8)

        opp = self.league.sample_opponent()
        self._set_opponent(opp)

        self._init_episode_routing()

        obs = self._get_obs()
        info = {"opponent_kind": opp.kind, "opponent_key": opp.key, "opponent_rating": opp.rating}
        return obs, info

    def step(self, action):
        self.step_count += 1

        a = np.asarray(action).reshape(-1).astype(int)
        b0_macro, b0_tgt, b1_macro, b1_tgt = (int(a[0]), int(a[1]), int(a[2]), int(a[3]))

        blue_agents = [x for x in self.env.blue_agents if x is not None]
        while len(blue_agents) < 2:
            blue_agents.append(blue_agents[0] if blue_agents else None)

        invalid_count = 0
        submit: Dict[str, Tuple[int, int]] = {}

        for agent, macro_idx, tgt_idx in [(blue_agents[0], b0_macro, b0_tgt), (blue_agents[1], b1_macro, b1_tgt)]:
            if agent is None or (hasattr(agent, "isEnabled") and not agent.isEnabled()):
                continue

            mm = np.asarray(self.env.get_macro_mask(agent), dtype=np.bool_).reshape(-1)
            if mm.shape != (N_MACROS,) or (not mm.any()):
                mm = np.ones((N_MACROS,), dtype=np.bool_)

            macro_idx = int(macro_idx) % N_MACROS
            tgt_idx = int(tgt_idx) % self.n_targets

            if not bool(mm[macro_idx]):
                macro_idx = 0  # force legal fallback (GO_TO)
                invalid_count += 1

            submit[agent_uid(agent)] = (macro_idx, tgt_idx)

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

        if invalid_count > 0:
            reward -= float(INVALID_MACRO_PENALTY) * float(invalid_count)

        if done:
            reward += outcome_bonus(int(self.gm.blue_score), int(self.gm.red_score))

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
            }

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
# Custom SB3 feature extractor for Dict obs (grid+mask)
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
# Callback: Elo updates + snapshotting
# -------------------------
class LeagueSelfPlayCallback(BaseCallback):
    def __init__(self, league: LeagueManager, save_dir: str, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.league = league
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.episode_idx = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

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
            self.league.update_elo(opp_key, actual)

            if self.verbose:
                print(
                    f"[LEAGUE|{self.league.mode}] ep={self.episode_idx} result={mr['result']} "
                    f"score={mr['blue_score']}:{mr['red_score']} "
                    f"opp={mr['opponent_kind']} "
                    f"learner_elo={self.league.learner_rating:.1f} opp_elo={self.league.get_rating(opp_key):.1f}"
                )

            if (self.episode_idx % self.league.snapshot_every_episodes) == 0:
                snap = self.league.add_snapshot(self.model, self.episode_idx)
                if self.verbose:
                    print(f"[SNAPSHOT|{self.league.mode}] saved={os.path.basename(snap)} pool={len(self.league.snapshots)}")

        return True


# -------------------------
# Training entry
# -------------------------
def make_env_fn(league: LeagueManager):
    def _fn():
        env = CTFPPOEnv(league=league, seed=SEED)
        env = Monitor(env)
        return env
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

    venv = DummyVecEnv([make_env_fn(league)])
    venv = VecMonitor(venv)

    policy_kwargs = dict(
        features_extractor_class=CTFCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    tb_dir = os.path.join(CHECKPOINT_DIR, "tb", RUN_TAG)

    model = PPO(
        policy="MultiInputPolicy",
        env=venv,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tb_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=SEED,
        device="auto",
    )

    cb = LeagueSelfPlayCallback(league=league, save_dir=CHECKPOINT_DIR, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb)

    final_path = os.path.join(CHECKPOINT_DIR, f"research_model_{RUN_TAG}")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}.zip")


if __name__ == "__main__":
    print(f"[RUN] TRAIN_MODE={TRAIN_MODE} (default={TRAIN_MODE_DEFAULT_LEAGUE})")
    train_phase1_sb3()
