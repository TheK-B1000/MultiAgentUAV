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
PPO_PHASE_MIN_EPISODES = {"OP1": 100, "OP2": 100, "OP3": 0}
PPO_PHASE_ELO_MARGIN: float = 75.0
PPO_PHASE_REQUIRED_WIN_BY = {"OP1": 3, "OP2": 0, "OP3": 0}
PPO_CURRICULUM_SPECIES_PROB: float = 0.20
PPO_CURRICULUM_SNAPSHOT_PROB: float = 0.10
PPO_CURRICULUM_WINRATE_WINDOW: int = 50
PPO_CURRICULUM_MIN_WINRATE: float = 0.25
PPO_CURRICULUM_FALLBACK_PROB: float = 0.50
PPO_CURRICULUM_OP3_SPECIES_SCALE: float = 0.50
PPO_CURRICULUM_OP3_SNAPSHOT_SCALE: float = 0.50
PPO_SWITCH_TO_ELO_AFTER_OP3: bool = True

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
