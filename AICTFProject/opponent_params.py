"""
OpponentParams: Batched per-episode adversarial style (speed, deception, coordinated attack).
Each style maps to a distribution over these params and returns GPU tensors for BatchedCTFCore.

OP3 vs OP4 (must be clearly different for held-out eval):
  - OP3: Used in training. Medium attacker + medium defender (defender_style=1), moderate
    role switching (0.35), moderate deception and speed. Balanced play.
  - OP4: Held-out; never used in training. Attack-leaning pressure team with easier defense,
    more role switching, higher deception, and faster routing. It should feel different from
    OP3, not just slightly stronger.
  The core uses: red_attacker_style, red_defender_style, red_deception_prob, red_speed_mult,
  red_role_switch_prob, so OP3 vs OP4 produce different red behavior.
"""
from __future__ import annotations

from typing import Dict, Optional, Union

import torch


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
            # Held-out eval opponent: never used in training. Make it behaviorally distinct from
            # OP3 by keeping attack pressure high even for larger teams.
            attacker_style = 1
            defender_style = 0
            role_switch_prob = 0.50
            if op3_easy:
                if n_agents >= 8:
                    s_low, s_high = 0.82, 0.96
                    d_low, d_high = 0.06, 0.14
                    c_prob = 0.18
                    sync_c_low, sync_c_high = 2, 4
                    sync_nc_low, sync_nc_high = 1, 3
                    n_low, n_high = 0.0, 0.03
                elif n_agents >= 4:
                    s_low, s_high = 0.86, 1.02
                    d_low, d_high = 0.08, 0.18
                    c_prob = 0.24
                    sync_c_low, sync_c_high = 2, 5
                    sync_nc_low, sync_nc_high = 2, 4
                    n_low, n_high = 0.0, 0.04
                else:
                    s_low, s_high = 0.98, 1.24
                    d_low, d_high = 0.18, 0.34
                    c_prob = 0.35
                    sync_c_low, sync_c_high = 2, 6
                    sync_nc_low, sync_nc_high = 2, 5
                    n_low, n_high = 0.0, 0.05
            else:
                s_low, s_high = 0.96, 1.30
                d_low, d_high = 0.22, 0.48
                c_prob = 0.48
                sync_c_low, sync_c_high = 3, 7
                sync_nc_low, sync_nc_high = 3, 6
                n_low, n_high = 0.01, 0.08
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
    
    speed_mult = s_low + (s_high - s_low) * torch.rand(batch_size, device=device, generator=generator)
    deception_prob = d_low + (d_high - d_low) * torch.rand(batch_size, device=device, generator=generator)
    
    coordinated_attack = torch.rand(batch_size, device=device, generator=generator) < c_prob

    # Note: torch.randint upper bound is exclusive, so we add 1 to the high bounds
    sync_c = torch.randint(sync_c_low, sync_c_high + 1, (batch_size,), device=device, generator=generator)
    sync_nc = torch.randint(sync_nc_low, sync_nc_high + 1, (batch_size,), device=device, generator=generator)
    
    # Select the correct sync window based on the coordinated_attack boolean mask
    attack_sync_window = torch.where(coordinated_attack, sync_c, sync_nc).to(torch.int32)
    
    noise_sigma = n_low + (n_high - n_low) * torch.rand(batch_size, device=device, generator=generator)

    return {
        "speed_mult": speed_mult,
        "deception_prob": deception_prob,
        "coordinated_attack": coordinated_attack,
        "attack_sync_window": attack_sync_window,
        "noise_sigma": noise_sigma,
        "attacker_style": torch.full((batch_size,), int(attacker_style), dtype=torch.int32, device=device),
        "defender_style": torch.full((batch_size,), int(defender_style), dtype=torch.int32, device=device),
        "role_switch_prob": torch.full((batch_size,), float(role_switch_prob), dtype=torch.float32, device=device),
    }
