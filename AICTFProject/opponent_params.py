# opponent_params.py
"""
OpponentParams: per-episode adversarial style (speed, deception, coordinated attack).
Each style (scripted tag, species tag, snapshot) maps to a distribution over these params.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class OpponentParams:
    """Per-episode red opponent behavior parameters (adversarial styles)."""
    speed_mult: float
    deception_prob: float
    coordinated_attack: bool
    attack_sync_window: int  # steps to hold sync; also used as feint duration k
    noise_sigma: float


def sample_opponent_params(
    kind: str,
    key: str,
    phase: str = "OP3",
    rng: Optional[random.Random] = None,
) -> OpponentParams:
    """
    Sample OpponentParams for (kind, key). Per-episode speed_mult by phase.
    kind: "SCRIPTED" | "SPECIES" | "SNAPSHOT"
    key: e.g. "OP1", "OP2", "OP3", "RUSHER", "CAMPER", "BALANCED", or snapshot path label
    phase: curriculum phase (OP1/OP2/OP3) for speed_mult distribution.
    """
    rng = rng or random.Random()
    kind = str(kind).upper()
    key = str(key).upper()
    phase = str(phase).upper()

    # Speed variation: per-episode multiplier by phase
    if phase == "OP1":
        speed_mult = rng.uniform(0.95, 1.05)
    elif phase == "OP2":
        speed_mult = rng.uniform(0.85, 1.15)
    else:
        speed_mult = rng.uniform(0.75, 1.25)

    deception_prob = 0.0
    coordinated_attack = False
    attack_sync_window = 0
    noise_sigma = 0.0

    if kind == "SCRIPTED":
        if key == "OP1":
            deception_prob = 0.0
            coordinated_attack = False
            attack_sync_window = 0
            noise_sigma = 0.0
        elif key == "OP2":
            deception_prob = rng.uniform(0.0, 0.15)
            coordinated_attack = False
            attack_sync_window = 0
            noise_sigma = rng.uniform(0.0, 0.05)
        elif key == "OP3":
            # Standard OP3
            deception_prob = rng.uniform(0.1, 0.35)
            coordinated_attack = rng.random() < 0.5
            attack_sync_window = rng.randint(3, 8) if coordinated_attack else rng.randint(3, 6)
            noise_sigma = rng.uniform(0.0, 0.08)
        else:
            deception_prob = rng.uniform(0.05, 0.25)
            coordinated_attack = rng.random() < 0.4
            attack_sync_window = rng.randint(3, 6)
            noise_sigma = rng.uniform(0.0, 0.06)

    elif kind == "SPECIES":
        # Same base logic (OP3), different parameter distributions
        if key == "RUSHER":
            speed_mult = rng.uniform(1.05, 1.25)
            deception_prob = rng.uniform(0.0, 0.15)
            coordinated_attack = rng.random() < 0.3
            attack_sync_window = rng.randint(2, 5)
            noise_sigma = rng.uniform(0.0, 0.05)
        elif key == "CAMPER":
            speed_mult = rng.uniform(0.80, 1.0)
            deception_prob = rng.uniform(0.2, 0.4)
            coordinated_attack = rng.random() < 0.4
            attack_sync_window = rng.randint(4, 8)
            noise_sigma = rng.uniform(0.02, 0.08)
        else:  # BALANCED
            speed_mult = rng.uniform(0.90, 1.10)
            deception_prob = rng.uniform(0.1, 0.3)
            coordinated_attack = rng.random() < 0.5
            attack_sync_window = rng.randint(3, 7)
            noise_sigma = rng.uniform(0.0, 0.06)

    else:  # SNAPSHOT or unknown
        speed_mult = rng.uniform(0.85, 1.15)
        deception_prob = rng.uniform(0.1, 0.3)
        coordinated_attack = rng.random() < 0.4
        attack_sync_window = rng.randint(3, 7)
        noise_sigma = rng.uniform(0.0, 0.06)

    return OpponentParams(
        speed_mult=float(speed_mult),
        deception_prob=float(deception_prob),
        coordinated_attack=bool(coordinated_attack),
        attack_sync_window=int(max(0, attack_sync_window)),
        noise_sigma=float(max(0.0, noise_sigma)),
    )
