"""
OpponentParams: Batched per-episode adversarial style (speed, deception, coordinated attack).
Each style maps to a distribution over these params and returns GPU tensors for BatchedCTFCore.

**Environment wiring (read before tuning OP5 / OP3):**

- The GPU core applies from ``sample_batched_opponent_params``:
  ``deception_prob``, ``speed_mult``, ``attacker_style``, ``defender_style``, ``role_switch_prob``,
  plus ``coordinated_attack`` and ``attack_sync_window`` (see
  ``gpu_env/_core/_dynamics.py::_apply_opponent_params_for_mask`` and
  ``rl/train_ppo.py::_apply_initial_opponent_params``).
- **Not applied in sim today:** ``noise_sigma`` (sampled but unused in scripted red / dynamics).
- **``speed_mult`` headroom:** red speed cap is ``cfg.max_speed_cps * speed_mult`` per step, but
  ``_integrate_side`` sets chase ``desired_speed = min(min(2.2, max_speed_cps), cap)``. With default
  ``max_speed_cps == 2.2``, any ``speed_mult >= 1`` yields the **same** top chase speed as ``1.0``;
  multipliers **above 1** mostly do **not** make red faster. Tuning OP5 by pushing ``s_high`` past
  ~``2.2 / max_speed_cps`` is largely ineffective until integrator / cap semantics change.

OP3 vs OP4 (must be clearly different for held-out eval):
  - OP3: Training workhorse. Fixed medium attacker + medium defender, moderate role switching (~0.35),
    moderate deception / coordination on 2v2. Larger teams use the normal speed-only band so OP3
    remains a trainable baseline instead of the hard test opponent.
  - OP4: Held-out **style mixture**, not ``OP3 + noise``. Each episode draws one of four archetypes
    and is intentionally tougher than OP3 eval on larger teams:
    committed blitz (striker-heavy, low role churn), slow anchor with heavy deception,
    volatile ``(medium, medium)`` with **much** higher role + deception than OP3's fixed pivot rate,
    or high-entropy yolo / randomized styles. Goal: different *behavior regime* than OP3, not only strength.
  - OP5_RUSHER: Trainable stress-test. High sustained speed, striker-heavy red (attacker=1,
    defender=0), **high coordination** and **very low** role churn so red commits to flag
    pressure. 4v4 ``bite_v4`` adds extended sync windows (3-7 / 3-6) and small deception
    (0.04-0.12) so OP5 can't be one-strategy solved (target ~30-55% WR on flat 4v4 policies;
    2v2 retains the ``bite_v2`` tuning that ``test_op5_rusher_bounded_2v2`` pins).
  - OP6 / OP6_TURTLE: Trainable **defensive turtle**. 4v4 is intentionally the **hardest**
    training opponent — chase-speed band 1.00-1.20 (so red can actually intercept), c=0.70
    coordination, sync 3-7 / 3-6, heavy mid-field deception 0.25-0.45 (defenders feint, blue
    can't predict the intercept), role switch 0.18 (shell can shape-shift mid-push). The
    ``attacker_style=0 / defender_style=1`` identity is preserved (turtle = home shell, not
    counter-attacker); difficulty comes from the defense *working*, not from offensive pressure.
  - OP7 / OP7_SWITCHER: Trainable **deceptive switcher**: each episode samples one of several
    archetypes (slow shell / feint-intercept / volatile dual / coordinated rush). High
    within-episode variability is expressed through **stochastic role pivots** (``role_switch_prob``)
    and deception; true flag-triggered FSM logic is not in this module yet.

  The core uses: red_attacker_style, red_defender_style, red_deception_prob, red_speed_mult,
  red_role_switch_prob, red_coordinated_attack, red_attack_sync_window, so OP3 vs OP4 vs
  OP5_RUSHER vs OP6 vs OP7 produce different red behavior.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

# Bump this string whenever OP5_RUSHER / OP5 scripted tuning changes so eval CSV filenames
# match the code path (see ``plot/eval_checkpoint.py``).
OP5_RUSHER_TUNING_TAG = "bite_v4"


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
    """Return team-size-aware OP4 archetype ranges (contrasts vs fixed OP3, not ``OP3 + ε``).

    Archetypes (indices 0..3):
      0 — Committed blitz: striker-heavy, high speed, low role-switching (different from OP3's 0.35 pivot).
      1 — Anchor + mindgames: slow line, **high deception**, sticky roles (OP3 rarely sits this slow/deep).
      2 — Volatile pivot: still (1,1) like OP3's labels but **much** higher stochastic role + deception
          (OP3 uses a single moderate pivot rate; this mode churns commitments).
      3 — Yolo / max entropy: aggressive speed + very high role volatility; optional random style bits.
    """
    if n_agents >= 8:
        return {
            0: {"speed": (0.90, 1.06), "deception": (0.04, 0.14), "role": (0.07, 0.22), "coord": 0.38, "sync_coord": (1, 4), "sync_noncoord": (1, 3), "noise": (0.00, 0.03), "attacker": 1, "defender": 0},
            1: {"speed": (0.66, 0.80), "deception": (0.20, 0.36), "role": (0.04, 0.12), "coord": 0.10, "sync_coord": (1, 3), "sync_noncoord": (1, 2), "noise": (0.02, 0.05), "attacker": 0, "defender": 1},
            2: {"speed": (0.76, 0.90), "deception": (0.16, 0.30), "role": (0.46, 0.74), "coord": 0.20, "sync_coord": (1, 4), "sync_noncoord": (1, 3), "noise": (0.01, 0.04), "attacker": 1, "defender": 1},
            3: {"speed": (0.86, 1.02), "deception": (0.12, 0.28), "role": (0.56, 0.82), "coord": 0.40, "sync_coord": (2, 5), "sync_noncoord": (1, 4), "noise": (0.01, 0.04), "attacker": -1, "defender": -1},
        }
    if n_agents >= 4:
        return {
            0: {"speed": (0.96, 1.12), "deception": (0.05, 0.18), "role": (0.08, 0.24), "coord": 0.40, "sync_coord": (1, 4), "sync_noncoord": (1, 3), "noise": (0.00, 0.04), "attacker": 1, "defender": 0},
            1: {"speed": (0.62, 0.78), "deception": (0.22, 0.38), "role": (0.04, 0.12), "coord": 0.10, "sync_coord": (1, 3), "sync_noncoord": (1, 2), "noise": (0.02, 0.05), "attacker": 0, "defender": 1},
            2: {"speed": (0.78, 0.94), "deception": (0.18, 0.34), "role": (0.48, 0.78), "coord": 0.22, "sync_coord": (1, 4), "sync_noncoord": (1, 3), "noise": (0.01, 0.04), "attacker": 1, "defender": 1},
            3: {"speed": (0.88, 1.06), "deception": (0.14, 0.30), "role": (0.58, 0.85), "coord": 0.42, "sync_coord": (2, 5), "sync_noncoord": (1, 4), "noise": (0.01, 0.04), "attacker": -1, "defender": -1},
        }
    return {
        0: {"speed": (1.06, 1.22), "deception": (0.03, 0.14), "role": (0.06, 0.22), "coord": 0.44, "sync_coord": (2, 4), "sync_noncoord": (1, 3), "noise": (0.00, 0.04), "attacker": 1, "defender": 0},
        1: {"speed": (0.68, 0.86), "deception": (0.28, 0.48), "role": (0.04, 0.14), "coord": 0.12, "sync_coord": (1, 3), "sync_noncoord": (1, 3), "noise": (0.02, 0.06), "attacker": 0, "defender": 1},
        2: {"speed": (0.92, 1.10), "deception": (0.22, 0.40), "role": (0.52, 0.82), "coord": 0.26, "sync_coord": (2, 5), "sync_noncoord": (1, 4), "noise": (0.01, 0.05), "attacker": 1, "defender": 1},
        3: {"speed": (1.02, 1.20), "deception": (0.18, 0.36), "role": (0.62, 0.88), "coord": 0.50, "sync_coord": (2, 5), "sync_noncoord": (1, 4), "noise": (0.01, 0.05), "attacker": -1, "defender": -1},
    }


def _op7_profile_ranges(n_agents: int) -> Dict[int, Dict[str, Tuple[float, float] | Tuple[int, int] | float | int]]:
    """Trainable deceptive scripted opponent: episode-start mixture (distinct from OP4 held-out).

    Archetypes (indices 0..3):
      0 — Shell: defender-leaning, slow, sticky roles, moderate deception.
      1 — Feint ladder: striker-leaning with **high** role + deception (looks committed, pivots).
      2 — Volatile dual: (1,1) with very high role + deception (mid-episode commitment churn).
      3 — Coordinated surge: fast striker band, high coordination, medium role (orderly blitz).
    """
    if n_agents >= 8:
        return {
            0: {"speed": (0.64, 0.78), "deception": (0.12, 0.24), "role": (0.10, 0.24), "coord": 0.16, "sync_coord": (1, 4), "sync_noncoord": (1, 3), "noise": (0.00, 0.03), "attacker": 0, "defender": 1},
            1: {"speed": (0.80, 0.94), "deception": (0.22, 0.36), "role": (0.42, 0.66), "coord": 0.28, "sync_coord": (2, 5), "sync_noncoord": (1, 4), "noise": (0.01, 0.04), "attacker": 1, "defender": 0},
            2: {"speed": (0.74, 0.88), "deception": (0.20, 0.34), "role": (0.52, 0.78), "coord": 0.20, "sync_coord": (2, 5), "sync_noncoord": (1, 4), "noise": (0.01, 0.04), "attacker": 1, "defender": 1},
            3: {"speed": (0.86, 1.00), "deception": (0.08, 0.20), "role": (0.18, 0.36), "coord": 0.58, "sync_coord": (2, 6), "sync_noncoord": (2, 5), "noise": (0.01, 0.03), "attacker": 1, "defender": 0},
        }
    if n_agents >= 4:
        return {
            0: {"speed": (0.68, 0.82), "deception": (0.14, 0.28), "role": (0.08, 0.22), "coord": 0.22, "sync_coord": (2, 5), "sync_noncoord": (2, 4), "noise": (0.01, 0.04), "attacker": 0, "defender": 1},
            1: {"speed": (0.86, 1.02), "deception": (0.22, 0.36), "role": (0.40, 0.64), "coord": 0.32, "sync_coord": (2, 5), "sync_noncoord": (1, 4), "noise": (0.01, 0.04), "attacker": 1, "defender": 0},
            2: {"speed": (0.78, 0.94), "deception": (0.18, 0.32), "role": (0.48, 0.74), "coord": 0.24, "sync_coord": (2, 5), "sync_noncoord": (2, 4), "noise": (0.01, 0.04), "attacker": 1, "defender": 1},
            3: {"speed": (0.92, 1.10), "deception": (0.10, 0.22), "role": (0.20, 0.40), "coord": 0.55, "sync_coord": (2, 6), "sync_noncoord": (1, 4), "noise": (0.01, 0.03), "attacker": 1, "defender": 0},
        }
    return {
        0: {"speed": (0.70, 0.86), "deception": (0.16, 0.30), "role": (0.10, 0.26), "coord": 0.24, "sync_coord": (2, 5), "sync_noncoord": (2, 4), "noise": (0.01, 0.04), "attacker": 0, "defender": 1},
        1: {"speed": (0.90, 1.08), "deception": (0.24, 0.38), "role": (0.42, 0.68), "coord": 0.34, "sync_coord": (2, 5), "sync_noncoord": (1, 4), "noise": (0.01, 0.04), "attacker": 1, "defender": 0},
        2: {"speed": (0.82, 0.98), "deception": (0.20, 0.34), "role": (0.50, 0.76), "coord": 0.26, "sync_coord": (2, 6), "sync_noncoord": (2, 5), "noise": (0.01, 0.04), "attacker": 1, "defender": 1},
        3: {"speed": (0.96, 1.14), "deception": (0.12, 0.24), "role": (0.22, 0.44), "coord": 0.58, "sync_coord": (2, 6), "sync_noncoord": (2, 5), "noise": (0.01, 0.03), "attacker": 1, "defender": 0},
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
            # Strategy 3/4-like: Medium Attacker + Medium Defender + dynamic switching.
            # OP3 is the normal training baseline. Keep larger teams in the legacy
            # speed-only band; OP4 owns the harder held-out eval pressure.
            attacker_style = 1
            defender_style = 1
            role_switch_prob = 0.35
            if n_agents >= 8:
                s_low, s_high = 0.70, 0.82
            elif n_agents >= 4:
                s_low, s_high = 0.66, 0.78
            elif op3_easy:
                # n_agents == 3 (rare): use the legacy 2v2-ish band so behavior is sane.
                s_low, s_high = 0.88, 1.08
                d_low, d_high = 0.05, 0.18
                c_prob = 0.25
                sync_c_low, sync_c_high = 2, 5
                sync_nc_low, sync_nc_high = 2, 4
                n_low, n_high = 0.0, 0.04
            else:
                # 2v2 (and 1v1): unchanged from prior tuning — already tough.
                d_low, d_high = 0.1, 0.35
                c_prob = 0.5
                sync_c_low, sync_c_high = 3, 8
                sync_nc_low, sync_nc_high = 3, 6
                n_low, n_high = 0.0, 0.08
        elif key in ("OP5_RUSHER", "OP5"):
            # Fast coordinated flag pressure (2v2: bite_v2 — stronger stress axis vs bite_v1, still orderly).
            attacker_style = 1
            defender_style = 0
            role_switch_prob = 0.03
            d_low, d_high = 0.0, 0.06
            n_low, n_high = 0.0, 0.03
            c_prob = 0.90
            sync_c_low, sync_c_high = 2, 6
            sync_nc_low, sync_nc_high = 2, 5
            if n_agents >= 8:
                s_low, s_high = 0.88, 1.06
                c_prob = 0.52
                sync_c_low, sync_c_high = 1, 5
                sync_nc_low, sync_nc_high = 1, 4
            elif n_agents >= 4:
                # 4v4 bite_v4: full coordination + extended sync windows + small deception
                # so OP5 can't be one-strategy solved by a flat policy.
                s_low, s_high = 1.05, 1.28
                d_low, d_high = 0.04, 0.12
                c_prob = 0.95
                sync_c_low, sync_c_high = 3, 7
                sync_nc_low, sync_nc_high = 3, 6
                n_low, n_high = 0.0, 0.04
            else:
                # 2v2 bite_v1: 1.15–1.35, c=0.78, role=0.04 → ~95% flat WR vs OP5; bite_v2 pushes harder.
                s_low, s_high = 1.20, 1.43
        elif key in ("OP6_TURTLE", "OP6"):
            # Defensive turtle: home-anchored shell, low commitment churn, moderate midfield deception.
            attacker_style = 0
            defender_style = 1
            role_switch_prob = 0.09
            d_low, d_high = 0.10, 0.24
            c_prob = 0.34
            sync_c_low, sync_c_high = 2, 5
            sync_nc_low, sync_nc_high = 2, 4
            n_low, n_high = 0.0, 0.03
            if n_agents >= 8:
                s_low, s_high = 0.64, 0.78
                c_prob = 0.20
            elif n_agents >= 4:
                # 4v4 turtle: the *hardest* training opponent at 4v4. Real chase-speed band
                # (red can actually intercept blue raiders), heavy coordination, extended
                # sync windows, and heavy mid-field deception so blue can't predict the
                # intercept. Style identity (attacker=0/defender=1) preserved — the
                # turtle is hard because the defense *works*, not because it counter-attacks.
                s_low, s_high = 1.00, 1.20
                d_low, d_high = 0.25, 0.45
                c_prob = 0.70
                sync_c_low, sync_c_high = 3, 7
                sync_nc_low, sync_nc_high = 3, 6
                n_low, n_high = 0.0, 0.05
                role_switch_prob = 0.18
            else:
                s_low, s_high = 0.72, 0.90
                d_low, d_high = 0.08, 0.22
                c_prob = 0.28
        elif key in ("OP7_SWITCHER", "OP7"):
            # Placeholder scalars; per-episode mixture overwrites tensors in the OP7 block below.
            attacker_style = 1
            defender_style = 1
            role_switch_prob = 0.40
            s_low, s_high = 0.82, 1.05
            d_low, d_high = 0.12, 0.28
            c_prob = 0.35
            sync_c_low, sync_c_high = 2, 6
            sync_nc_low, sync_nc_high = 2, 5
            n_low, n_high = 0.0, 0.04
        elif key in ("OP8", "OP8_INTERCEPTOR"):
            # Coordinated pressure/interception: one agent pursues blue carrier,
            # one intercepts the carrier's path home. Very low role churn so agents
            # stay committed. High coordination, low deception.
            attacker_style = 1
            defender_style = 1
            role_switch_prob = 0.02
            d_low, d_high = 0.02, 0.08
            c_prob = 0.92
            sync_c_low, sync_c_high = 2, 5
            sync_nc_low, sync_nc_high = 1, 3
            n_low, n_high = 0.0, 0.03
            if n_agents >= 8:
                s_low, s_high = 0.80, 0.96
                c_prob = 0.80
            elif n_agents >= 4:
                s_low, s_high = 0.94, 1.10
                d_low, d_high = 0.03, 0.10
                c_prob = 0.90
            else:
                s_low, s_high = 0.88, 1.06
        elif key in ("OP9", "OP9_FORTRESS"):
            # Fortress + counterattack: tight guardian orbits own flag; after blue grabs
            # red flag (enemy_carrier_exists), all agents surge to intercept enemy carrier.
            # Heavy coordination on counterattack. Moderate deception from defenders.
            attacker_style = 0
            defender_style = 1
            role_switch_prob = 0.05
            d_low, d_high = 0.06, 0.18
            c_prob = 0.88
            sync_c_low, sync_c_high = 2, 5
            sync_nc_low, sync_nc_high = 1, 3
            n_low, n_high = 0.0, 0.03
            if n_agents >= 8:
                s_low, s_high = 0.72, 0.88
                c_prob = 0.75
            elif n_agents >= 4:
                s_low, s_high = 0.82, 0.98
                d_low, d_high = 0.10, 0.22
                c_prob = 0.85
            else:
                s_low, s_high = 0.76, 0.94
        elif key in ("OP10", "OP10_ESCORT"):
            # Coordinated carrier + active escort: escort agent interposes between
            # carrier and nearest enemy instead of sitting perpendicular. Focus on
            # offense with screening. Low deception; escort stays committed.
            attacker_style = 1
            defender_style = 0
            role_switch_prob = 0.04
            d_low, d_high = 0.02, 0.08
            c_prob = 0.90
            sync_c_low, sync_c_high = 2, 5
            sync_nc_low, sync_nc_high = 1, 3
            n_low, n_high = 0.0, 0.03
            if n_agents >= 8:
                s_low, s_high = 0.82, 0.98
                c_prob = 0.78
            elif n_agents >= 4:
                s_low, s_high = 0.96, 1.12
                d_low, d_high = 0.03, 0.10
                c_prob = 0.88
            else:
                s_low, s_high = 0.90, 1.08
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
    elif kind == "SCRIPTED" and key in ("OP7", "OP7_SWITCHER"):
        # Trainable deceptive switcher: episode-start archetype (distinct profile table from OP4).
        mode = torch.randint(0, 4, (batch_size,), device=device, generator=generator, dtype=torch.int32)
        profiles = _op7_profile_ranges(n_agents)
        role_switch_prob_t = torch.empty((batch_size,), dtype=torch.float32, device=device)
        attacker_style_t = torch.empty((batch_size,), dtype=torch.int32, device=device)
        defender_style_t = torch.empty((batch_size,), dtype=torch.int32, device=device)

        for profile_id, prof in profiles.items():
            mask = mode == profile_id
            if not torch.any(mask):
                continue

            speed_lo, speed_hi = prof["speed"]  # type: ignore[index]
            deception_lo, deception_hi = prof["deception"]  # type: ignore[index]
            role_lo, role_hi = prof["role"]  # type: ignore[index]
            noise_lo, noise_hi = prof["noise"]  # type: ignore[index]
            sync_c_lo, sync_c_hi = prof["sync_coord"]  # type: ignore[index]
            sync_nc_lo, sync_nc_hi = prof["sync_noncoord"]  # type: ignore[index]

            count = int(mask.sum().item())
            speed_mult[mask] = _sample_uniform(count, speed_lo, speed_hi, device=device, generator=generator)
            deception_prob[mask] = _sample_uniform(count, deception_lo, deception_hi, device=device, generator=generator)
            role_switch_prob_t[mask] = _sample_uniform(count, role_lo, role_hi, device=device, generator=generator)
            coordinated_attack[mask] = torch.rand(count, device=device, generator=generator) < float(prof["coord"])  # type: ignore[arg-type]
            noise_sigma[mask] = _sample_uniform(count, noise_lo, noise_hi, device=device, generator=generator)

            coord_mask = coordinated_attack[mask]
            sync_values = _sample_int(count, sync_nc_lo, sync_nc_hi, device=device, generator=generator)
            if bool(torch.any(coord_mask)):
                sync_values[coord_mask] = _sample_int(int(coord_mask.sum().item()), sync_c_lo, sync_c_hi, device=device, generator=generator)
            attack_sync_window[mask] = sync_values

            att = int(prof["attacker"])
            dfs = int(prof["defender"])
            if att >= 0:
                attacker_style_t[mask] = att
            else:
                attacker_style_t[mask] = _sample_int(count, 0, 1, device=device, generator=generator)
            if dfs >= 0:
                defender_style_t[mask] = dfs
            else:
                defender_style_t[mask] = _sample_int(count, 0, 1, device=device, generator=generator)
    else:
        attacker_style_t = torch.full((batch_size,), int(attacker_style), dtype=torch.int32, device=device)
        defender_style_t = torch.full((batch_size,), int(defender_style), dtype=torch.int32, device=device)
        role_switch_prob_t = torch.full((batch_size,), float(role_switch_prob), dtype=torch.float32, device=device)

    speed_mult = torch.clamp(speed_mult, 0.60, 1.45)
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
