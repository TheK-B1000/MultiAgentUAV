from typing import Callable

import torch

from map_registry import make_game_field

from game_field import GameField, MacroAction
from rl_policy import ActorCriticNet


# =========================
# ENV / GRID CONFIG
# =========================

GRID_ROWS: int = 20
GRID_COLS: int = 20

# Map selection (registry name or file path)
MAP_NAME: str = "empty_20x20"
MAP_PATH: str = ""

# Logging
LOG_DIR: str = "logs"

# -------------------------
# PPO Curriculum (Elo-gated)
# -------------------------
PPO_PHASE_MIN_EPISODES = {"OP1": 150, "OP2": 150, "OP3": 0}
PPO_PHASE_ELO_MARGIN: float = 100.0
PPO_PHASE_REQUIRED_WIN_BY = {"OP1": 1, "OP2": 1, "OP3": 0}
PPO_CURRICULUM_SPECIES_PROB: float = 0.20
PPO_CURRICULUM_SNAPSHOT_PROB: float = 0.10
PPO_CURRICULUM_WINRATE_WINDOW: int = 50
PPO_CURRICULUM_MIN_WINRATE: float = 0.55
PPO_CURRICULUM_FALLBACK_PROB: float = 0.50
PPO_CURRICULUM_OP3_SPECIES_SCALE: float = 0.50
PPO_CURRICULUM_OP3_SNAPSHOT_SCALE: float = 0.50
PPO_SWITCH_TO_ELO_AFTER_OP3: bool = True

# OP3 Elo curriculum (best-practice opponent matching)
PPO_OP3_ELO_TARGET_MARGIN_EASY: float = 150.0
PPO_OP3_ELO_TARGET_MARGIN_HARD: float = 150.0
PPO_OP3_ELO_WINRATE_LOW: float = 0.35
PPO_OP3_ELO_WINRATE_HIGH: float = 0.65
PPO_OP3_ELO_SCRIPTED_FLOOR_BASE: float = 0.10
PPO_OP3_ELO_SCRIPTED_FLOOR_EASY: float = 0.25
PPO_OP3_ELO_SCRIPTED_FLOOR_LOSING: float = 0.02
PPO_OP3_ELO_MAX_GAP_WHEN_LOSING: float = 300.0
PPO_OP3_ELO_RESAMPLE_TRIES: int = 5

# OP3 stability control (avoid catastrophic forgetting)
PPO_STABILITY_WINRATE_TARGET: float = 0.50
PPO_STABILITY_LR_MIN_SCALE: float = 0.35
PPO_STABILITY_ENT_MIN_SCALE: float = 0.60

# Factory that creates a fresh GameField instance.
def make_env() -> GameField:
    return make_game_field(
        map_name=MAP_NAME or None,
        map_path=MAP_PATH or None,
        rows=GRID_ROWS,
        cols=GRID_COLS,
    )


# =========================
# OBS / ACTION SPACE
# =========================

# Must match GameField.build_observation() layout:
#   12 scalars + 5×5 occupancy = 37
OBS_DIM: int = 37

# Discrete macro actions (GoTo, GrabMine, GetFlag, ...)
N_ACTIONS: int = len(MacroAction)


# =========================
# DEVICE
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# PPO HYPERPARAMETERS
# =========================

TOTAL_STEPS: int = 50_000
UPDATE_EVERY: int = 2_048
PPO_EPOCHS: int = 10
MINIBATCH_SIZE: int = 256

LR: float = 3e-4
CLIP_EPS: float = 0.2
VALUE_COEF: float = 0.5
ENT_COEF: float = 0.01
MAX_GRAD_NORM: float = 0.5

GAMMA: float = 0.99
GAE_LAMBDA: float = 0.95

CHECKPOINT_EVERY: int = 5_000
CHECKPOINT_DIR: str = "checkpoints"
LOG_DIR: str = "logs"


# =========================
# OBS ENCODER / POLICY
# =========================
def make_policy(
    obs_dim: int = OBS_DIM,
    n_actions: int = N_ACTIONS,
    device: torch.device = DEVICE,
) -> ActorCriticNet:
    net = ActorCriticNet(obs_dim=obs_dim, n_actions=n_actions)
    net.to(device)
    return net

def make_obs_encoder() -> ActorCriticNet:
    return make_policy()
