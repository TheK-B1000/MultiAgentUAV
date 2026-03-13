"""
OpponentParams: Batched per-episode adversarial style (speed, deception, coordinated attack).
Each style maps to a distribution over these params and returns GPU tensors for BatchedCTFCore.

OP3 vs OP4 (must be clearly different for held-out eval):
  - OP3: Used in training. Medium attacker + medium defender (defender_style=1), moderate
    role switching (0.35), moderate deception and speed. Balanced play.
  - OP4: Held-out; never used in training. High-variance opponent that samples across several
    plausible scripted styles each episode. It is intentionally broader and less predictable
    than OP3 so robustness matters more than narrow specialization.
  The core uses: red_attacker_style, red_defender_style, red_deception_prob, red_speed_mult,
  red_role_switch_prob, so OP3 vs OP4 produce different red behavior.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch


def _sample_uniform(
    batch_size: int,
    low: float,
    high: float,
    *,
    device: Union[str, torch.device],
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Sample a float tensor, collapsing degenerate ranges to a constant."""
    low = float(low)
    high = float(high)
    if high <= low:
        return torch.full((batch_size,), low, dtype=torch.float32, device=device)
    return low + (high - low) * torch.rand(batch_size, device=device, generator=generator)


def _sample_int(
    batch_size: int,
    low: int,
    high: int,
    *,
    device: Union[str, torch.device],
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Sample an int tensor, collapsing degenerate ranges to a constant."""
    low = int(low)
    high = int(high)
    if high <= low:
        return torch.full((batch_size,), low, dtype=torch.int32, device=device)
    return torch.randint(low, high + 1, (batch_size,), device=device, generator=generator, dtype=torch.int32)


def _op4_profile_ranges(n_agents: int) -> Dict[int, Dict[str, Tuple[float, float] | Tuple[int, int] | float | int]]:
    """Return team-size-aware OP4 archetype ranges."""
    if n_agents >= 8:
        return {
            0: {"speed": (0.88, 1.04), "deception": (0.02, 0.10), "role": (0.34, 0.52), "coord": 0.10, "sync_coord": (1, 3), "sync_noncoord": (1, 2), "noise": (0.00, 0.02), "attacker": 1, "defender": 0},
            1: {"speed": (0.70, 0.84), "deception": (0.18, 0.34), "role": (0.02, 0.10), "coord": 0.08, "sync_coord": (1, 2), "sync_noncoord": (1, 2), "noise": (0.00, 0.02), "attacker": 0, "defender": 1},
            2: {"speed": (0.78, 0.92), "deception": (0.10, 0.22), "role": (0.14, 0.28), "coord": 0.14, "sync_coord": (1, 3), "sync_noncoord": (1, 2), "noise": (0.00, 0.025), "attacker": 1, "defender": 1},
            3: {"speed": (0.80, 0.96), "deception": (0.16, 0.30), "role": (0.42, 0.64), "coord": 0.12, "sync_coord": (1, 3), "sync_noncoord": (1, 2), "noise": (0.00, 0.03), "attacker": -1, "defender": -1},
        }
    if n_agents >= 4:
        return {
            0: {"speed": (0.94, 1.10), "deception": (0.02, 0.14), "role": (0.40, 0.60), "coord": 0.16, "sync_coord": (1, 4), "sync_noncoord": (1, 3), "noise": (0.00, 0.03), "attacker": 1, "defender": 0},
            1: {"speed": (0.76, 0.90), "deception": (0.24, 0.40), "role": (0.03, 0.12), "coord": 0.10, "sync_coord": (1, 3), "sync_noncoord": (1, 3), "noise": (0.00, 0.03), "attacker": 0, "defender": 1},
            2: {"speed": (0.84, 0.98), "deception": (0.12, 0.28), "role": (0.16, 0.32), "coord": 0.18, "sync_coord": (1, 4), "sync_noncoord": (1, 3), "noise": (0.00, 0.04), "attacker": 1, "defender": 1},
            3: {"speed": (0.86, 1.02), "deception": (0.20, 0.36), "role": (0.48, 0.72), "coord": 0.16, "sync_coord": (1, 4), "sync_noncoord": (1, 3), "noise": (0.00, 0.04), "attacker": -1, "defender": -1},
        }
    return {
        0: {"speed": (1.00, 1.16), "deception": (0.03, 0.18), "role": (0.48, 0.70), "coord": 0.24, "sync_coord": (2, 5), "sync_noncoord": (1, 4), "noise": (0.00, 0.04), "attacker": 1, "defender": 0},
        1: {"speed": (0.76, 0.92), "deception": (0.28, 0.46), "role": (0.03, 0.14), "coord": 0.12, "sync_coord": (1, 4), "sync_noncoord": (1, 3), "noise": (0.00, 0.04), "attacker": 0, "defender": 1},
        2: {"speed": (0.88, 1.04), "deception": (0.14, 0.32), "role": (0.18, 0.36), "coord": 0.22, "sync_coord": (2, 5), "sync_noncoord": (1, 4), "noise": (0.00, 0.05), "attacker": 1, "defender": 1},
        3: {"speed": (0.90, 1.08), "deception": (0.22, 0.42), "role": (0.58, 0.82), "coord": 0.20, "sync_coord": (2, 5), "sync_noncoord": (1, 4), "noise": (0.00, 0.05), "attacker": -1, "defender": -1},
    }


def sample_batched_opponent_params(
    kind: str,
    key: str,
    phase: str = "OP3",
    n_agents: int = 2,
    batch_size: int = 1,
    device: Union[str, torch.device] = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """
    Sample adversarial parameters for a full batch of environments simultaneously.
    Returns a dictionary of tensors sized [batch_size] residing on the target device.
    """
    kind = str(kind).upper()
    key = str(key).upper()
    phase = str(phase).upper()
    n_agents = max(1, int(n_agents))
    batch_size = max(0, int(batch_size))

    if batch_size == 0:
        empty_bool = torch.empty((0,), dtype=torch.bool, device=device)
        empty_float = torch.empty((0,), dtype=torch.float32, device=device)
        empty_int = torch.empty((0,), dtype=torch.int32, device=device)
        return {
            "speed_mult": empty_float,
            "deception_prob": empty_float,
            "coordinated_attack": empty_bool,
            "attack_sync_window": empty_int,
            "noise_sigma": empty_float,
            "attacker_style": empty_int,
            "defender_style": empty_int,
            "role_switch_prob": empty_float,
        }

    # 1. Determine base speed bounds by phase
    if phase == "OP1":
        s_low, s_high = 0.95, 1.05
    elif phase == "OP2":
        s_low, s_high = 0.85, 1.15
    else:
        s_low, s_high = 0.75, 1.25

    # Base defaults
    d_low, d_high = 0.0, 0.0
    c_prob = 0.0
    sync_c_low, sync_c_high = 0, 0
    sync_nc_low, sync_nc_high = 0, 0
    n_low, n_high = 0.0, 0.0
    # MOOS-IvP style role profile defaults:
    # 0 = easy, 1 = medium
    attacker_style = 0
    defender_style = 0
    role_switch_prob = 0.0

    op3_easy = n_agents > 2

    # 2. Evaluate logic tree for bounds
    if kind == "SCRIPTED":
        if key == "OP1":
            # Pav01-like baseline: Easy Attacker + Easy Defender
            attacker_style = 0
            defender_style = 0
            role_switch_prob = 0.05
            if n_agents >= 8:
                s_low, s_high = 0.82, 0.92
            elif n_agents >= 4:
                s_low, s_high = 0.90, 1.00
        elif key == "OP2":
            # Strategy 2-like: Easy Attacker + Medium Defender
            attacker_style = 0
            defender_style = 1
            role_switch_prob = 0.15
            d_low, d_high = 0.0, 0.15
            n_low, n_high = 0.0, 0.05
            if n_agents >= 8:
                s_low, s_high = 0.76, 0.88
                d_low, d_high = 0.0, 0.04
                n_low, n_high = 0.0, 0.0
            elif n_agents >= 4:
                s_low, s_high = 0.84, 1.00
                d_low, d_high = 0.0, 0.06
                n_low, n_high = 0.0, 0.0
        elif key == "OP3":
            # Strategy 3/4-like: Medium Attacker + Medium Defender + dynamic switching
            attacker_style = 1
            defender_style = 1
            role_switch_prob = 0.35
            if op3_easy:
                if n_agents >= 8:
                    s_low, s_high = 0.70, 0.82
                elif n_agents >= 4:
                    s_low, s_high = 0.66, 0.78
                else:
                    s_low, s_high = 0.88, 1.08
                    d_low, d_high = 0.05, 0.18
                    c_prob = 0.25
                    sync_c_low, sync_c_high = 2, 5
                    sync_nc_low, sync_nc_high = 2, 4
                    n_low, n_high = 0.0, 0.04
            else:
                d_low, d_high = 0.1, 0.35
                c_prob = 0.5
                sync_c_low, sync_c_high = 3, 8
                sync_nc_low, sync_nc_high = 3, 6
                n_low, n_high = 0.0, 0.08
        elif key == "OP4":
            # Held-out eval opponent: never used in training. Make it deliberately broad and
            # stochastic so robustness matters more than memorizing one scripted style.
            role_switch_prob = 0.35
            if op3_easy:
                if n_agents >= 8:
                    s_low, s_high = 0.72, 0.96
                    d_low, d_high = 0.04, 0.24
                    c_prob = 0.14
                    sync_c_low, sync_c_high = 1, 4
                    sync_nc_low, sync_nc_high = 1, 3
                    n_low, n_high = 0.0, 0.03
                elif n_agents >= 4:
                    s_low, s_high = 0.78, 1.04
                    d_low, d_high = 0.06, 0.30
                    c_prob = 0.20
                    sync_c_low, sync_c_high = 1, 5
                    sync_nc_low, sync_nc_high = 1, 4
                    n_low, n_high = 0.0, 0.04
                else:
                    s_low, s_high = 0.82, 1.12
                    d_low, d_high = 0.08, 0.38
                    c_prob = 0.24
                    sync_c_low, sync_c_high = 1, 5
                    sync_nc_low, sync_nc_high = 1, 4
                    n_low, n_high = 0.0, 0.05
            else:
                s_low, s_high = 0.84, 1.20
                d_low, d_high = 0.12, 0.46
                c_prob = 0.30
                sync_c_low, sync_c_high = 1, 6
                sync_nc_low, sync_nc_high = 1, 5
                n_low, n_high = 0.0, 0.06
        else:
            attacker_style = 1
            defender_style = 1
            role_switch_prob = 0.25
            d_low, d_high = 0.05, 0.25
            c_prob = 0.4
            sync_c_low, sync_c_high = 3, 6
            sync_nc_low, sync_nc_high = 3, 6
            n_low, n_high = 0.0, 0.06

    elif kind == "SPECIES":
        if key == "RUSHER":
            attacker_style = 1
            defender_style = 0
            role_switch_prob = 0.20
            s_low, s_high = 1.05, 1.25
            d_low, d_high = 0.0, 0.15
            c_prob = 0.3
            sync_c_low, sync_c_high = 2, 5
            sync_nc_low, sync_nc_high = 2, 5
            n_low, n_high = 0.0, 0.05
        elif key == "CAMPER":
            attacker_style = 0
            defender_style = 1
            role_switch_prob = 0.10
            s_low, s_high = 0.80, 1.0
            d_low, d_high = 0.2, 0.4
            c_prob = 0.4
            sync_c_low, sync_c_high = 4, 8
            sync_nc_low, sync_nc_high = 4, 8
            n_low, n_high = 0.02, 0.08
        else:  # BALANCED
            attacker_style = 1
            defender_style = 1
            role_switch_prob = 0.25
            s_low, s_high = 0.90, 1.10
            d_low, d_high = 0.1, 0.3
            c_prob = 0.5
            sync_c_low, sync_c_high = 3, 7
            sync_nc_low, sync_nc_high = 3, 7
            n_low, n_high = 0.0, 0.06
            
        # Species scaling for larger teams
        if n_agents >= 8:
            s_low, s_high = 0.72, 0.84 if key != "CAMPER" else (0.70, 0.82)
            d_low, d_high, c_prob = 0.0, 0.0, 0.0
            sync_c_low, sync_c_high = 0, 0
            sync_nc_low, sync_nc_high = 0, 0
            n_low, n_high = 0.0, 0.0
        elif n_agents >= 4:
            if key == "RUSHER":
                s_low, s_high = 0.80, 0.90
                d_low, d_high = 0.0, 0.03
            elif key == "CAMPER":
                s_low, s_high = 0.78, 0.88
                d_low, d_high = 0.0, 0.05
            else:
                s_low, s_high = 0.78, 0.88
                d_low, d_high = 0.0, 0.04
            c_prob = 0.0
            sync_c_low, sync_c_high = 0, 0
            sync_nc_low, sync_nc_high = 0, 0
            n_low, n_high = 0.0, 0.0

    else:  # SNAPSHOT
        attacker_style = 1
        defender_style = 1
        role_switch_prob = 0.25
        s_low, s_high = 0.85, 1.15
        d_low, d_high = 0.1, 0.3
        c_prob = 0.4
        sync_c_low, sync_c_high = 3, 7
        sync_nc_low, sync_nc_high = 3, 7
        n_low, n_high = 0.0, 0.06

    # 3. Batch Tensor Generation
    # We generate all parameters at once directly on the target GPU/CPU
    
    speed_mult = _sample_uniform(batch_size, s_low, s_high, device=device, generator=generator)
    deception_prob = _sample_uniform(batch_size, d_low, d_high, device=device, generator=generator)
    
    coordinated_attack = torch.rand(batch_size, device=device, generator=generator) < c_prob

    sync_c = _sample_int(batch_size, sync_c_low, sync_c_high, device=device, generator=generator)
    sync_nc = _sample_int(batch_size, sync_nc_low, sync_nc_high, device=device, generator=generator)
    
    # Select the correct sync window based on the coordinated_attack boolean mask
    attack_sync_window = torch.where(coordinated_attack, sync_c, sync_nc).to(torch.int32)
    
    noise_sigma = _sample_uniform(batch_size, n_low, n_high, device=device, generator=generator)

    if kind == "SCRIPTED" and key == "OP4":
        # OP4 is a held-out "random beast": each episode samples one of several scripted
        # archetypes so robustness matters more than specializing to a single unseen style.
        mode = torch.randint(0, 4, (batch_size,), device=device, generator=generator, dtype=torch.int32)
        profiles = _op4_profile_ranges(n_agents)
        role_switch_prob_t = torch.empty((batch_size,), dtype=torch.float32, device=device)
        attacker_style_t = torch.empty((batch_size,), dtype=torch.int32, device=device)
        defender_style_t = torch.empty((batch_size,), dtype=torch.int32, device=device)

        for profile_id, cfg in profiles.items():
            mask = mode == profile_id
            if not torch.any(mask):
                continue

            speed_lo, speed_hi = cfg["speed"]  # type: ignore[index]
            deception_lo, deception_hi = cfg["deception"]  # type: ignore[index]
            role_lo, role_hi = cfg["role"]  # type: ignore[index]
            noise_lo, noise_hi = cfg["noise"]  # type: ignore[index]
            sync_c_lo, sync_c_hi = cfg["sync_coord"]  # type: ignore[index]
            sync_nc_lo, sync_nc_hi = cfg["sync_noncoord"]  # type: ignore[index]

            count = int(mask.sum().item())
            speed_mult[mask] = _sample_uniform(count, speed_lo, speed_hi, device=device, generator=generator)
            deception_prob[mask] = _sample_uniform(count, deception_lo, deception_hi, device=device, generator=generator)
            role_switch_prob_t[mask] = _sample_uniform(count, role_lo, role_hi, device=device, generator=generator)
            coordinated_attack[mask] = torch.rand(count, device=device, generator=generator) < float(cfg["coord"])  # type: ignore[arg-type]
            noise_sigma[mask] = _sample_uniform(count, noise_lo, noise_hi, device=device, generator=generator)

            coord_mask = coordinated_attack[mask]
            sync_values = _sample_int(count, sync_nc_lo, sync_nc_hi, device=device, generator=generator)
            if bool(torch.any(coord_mask)):
                sync_values[coord_mask] = _sample_int(int(coord_mask.sum().item()), sync_c_lo, sync_c_hi, device=device, generator=generator)
            attack_sync_window[mask] = sync_values

            attacker_style = int(cfg["attacker"])
            defender_style = int(cfg["defender"])
            if attacker_style >= 0:
                attacker_style_t[mask] = attacker_style
            else:
                attacker_style_t[mask] = _sample_int(count, 0, 1, device=device, generator=generator)
            if defender_style >= 0:
                defender_style_t[mask] = defender_style
            else:
                defender_style_t[mask] = _sample_int(count, 0, 1, device=device, generator=generator)
    else:
        attacker_style_t = torch.full((batch_size,), int(attacker_style), dtype=torch.int32, device=device)
        defender_style_t = torch.full((batch_size,), int(defender_style), dtype=torch.int32, device=device)
        role_switch_prob_t = torch.full((batch_size,), float(role_switch_prob), dtype=torch.float32, device=device)

    speed_mult = torch.clamp(speed_mult, 0.60, 1.30)
    deception_prob = torch.clamp(deception_prob, 0.0, 0.60)
    noise_sigma = torch.clamp(noise_sigma, 0.0, 0.10)
    attack_sync_window = torch.clamp(attack_sync_window, 0, 8)
    attacker_style_t = torch.clamp(attacker_style_t, 0, 1)
    defender_style_t = torch.clamp(defender_style_t, 0, 1)
    role_switch_prob_t = torch.clamp(role_switch_prob_t, 0.0, 0.90)

    return {
        "speed_mult": speed_mult,
        "deception_prob": deception_prob,
        "coordinated_attack": coordinated_attack,
        "attack_sync_window": attack_sync_window,
        "noise_sigma": noise_sigma,
        "attacker_style": attacker_style_t,
        "defender_style": defender_style_t,
        "role_switch_prob": role_switch_prob_t,
    }
