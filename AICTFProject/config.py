from typing import Callable

import torch

from game_field import GameField, MacroAction
from rl_policy import ActorCriticNet


# =========================
# ENV / GRID CONFIG
# =========================

GRID_ROWS: int = 30
GRID_COLS: int = 40

# Factory that creates a fresh GameField instance.
def make_env() -> GameField:
    grid = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    env = GameField(grid)
    return env


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
