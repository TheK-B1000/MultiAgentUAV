"""C1 -- the confirmed weakness context, as a predicate and as a scenario.

C1 was confirmed on fresh held-out data in
``artifacts/c1_confirmation/C1_CONFIRMATION.json`` (3/3 policies, deltas
0.650-0.687, all CIs excluding zero). This module makes that context usable at
runtime in two ways:

    c1_active_from_context   the frozen predicate, on a legal_context() dict
    c1_active_mask           the same predicate, batched over envs
    apply_c1_scenario        reset-state injection into the C1 region

THE PREDICATE IS DEFINED ONCE
-----------------------------
``c1_active_from_context`` reads the dict produced by
``run_g0_v2_evaluation.legal_context``, which is the function whose
``home_threatened`` definition C1 was confirmed against. ``c1_active_mask``
exists only because training runs many envs at once and cannot afford a
per-env numpy round trip; it is a batched *replication* of the same arithmetic,
and ``tests/test_c1_context.py`` asserts the two agree on randomized states. If
that test ever fails, the batched copy has drifted and is wrong -- the
evaluation-side definition wins.

WHY THIS IS NOT ``POD_DEFEND_LEAD``
-----------------------------------
``experiments/v6i26_phase_pods.py`` already contains a "defend_lead" injector.
It is not reused here, for two independent reasons.

1. It places both blue agents adjacent to their own flag
   (``v6i26_phase_pods.py``, POD_DEFEND_LEAD). C1's construction, frozen in
   C1_PROPOSAL.json before any confirmation data existed, requires both blue
   agents PAST THE MIDLINE. The confirmed context is blue leading and *out of
   position*; the pod is blue leading and *already home*. Training on the pod
   would train on a context C1 does not describe.

2. Its clock line is dead code. It guards on ``hasattr(core, "decision_step")``,
   and ``BatchedCTFCore`` has no such attribute -- the counter is
   ``step_count`` (``gpu_env/state/episode_state.py``). The pod therefore never
   set the late clock it documents. Its ``max_decision_steps`` lookup is
   likewise a miss that always returned the 240 default rather than reading the
   configured horizon (``core.max_steps``).

Those pods belong to the V6I26 phase-pod birth attempt, which produced distinct
z indices with indistinguishable behavior. Reusing that scenario family here
would repeat it.
"""
from __future__ import annotations

from typing import Sequence

import torch

# --- geometry constants, copied from legal_context so the predicate matches ---
# run_g0_v2_evaluation.legal_context uses these exact values; they are repeated
# rather than imported because that module pulls in the whole training stack.
HOME_THREAT_MARGIN = 0.05   # nearest_red < nearest_blue - margin  => threatened
FLAG_AWAY_FRAC = 0.02       # ||flag - home|| / cols  >  this      => flag away


def c1_active_from_context(ctx: dict) -> bool:
    """The frozen C1 predicate: BLUE is ahead on score AND its home is threatened.

    ``ctx`` is one dict from ``run_g0_v2_evaluation.legal_context``. Both terms
    are read straight off that dict rather than recomputed, so this cannot drift
    from the definition C1 was confirmed under.
    """
    return bool(ctx["score_diff"] > 0 and ctx["home_threatened"])


def c1_active_mask(core) -> torch.Tensor:
    """Batched ``c1_active_from_context`` -> bool tensor of shape (B,).

    Replicates ``legal_context``'s ``home_threatened`` arithmetic exactly,
    including its use of ALL agent rows for the nearest-distance terms (it does
    not mask by ``alive``). Tested against the scalar path.
    """
    cols = max(float(core.cols), 1e-6)

    blue_score = core.blue_score.to(torch.float32)
    red_score = core.red_score.to(torch.float32)
    leading = (blue_score - red_score) > 0

    flag = core.blue_flag_pos                                    # (B, 2)
    home = core.blue_flag_home                                   # (B, 2)
    flag_away = (torch.linalg.norm(flag - home, dim=1) / cols) > FLAG_AWAY_FRAC

    def _nearest(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d = torch.sqrt(
            (x - flag[:, 0:1]) ** 2 + (y - flag[:, 1:2]) ** 2
        ) / cols
        return d.min(dim=1).values

    nearest_red = _nearest(core.red_x, core.red_y)
    nearest_blue = _nearest(core.blue_x, core.blue_y)

    red_carrying_any = core.red_carrying.any(dim=1)
    home_threatened = (
        red_carrying_any
        | flag_away
        | (nearest_red < nearest_blue - HOME_THREAT_MARGIN)
    )
    return leading & home_threatened


# --- scenario injection -----------------------------------------------------

# The construction below is the one frozen in C1_PROPOSAL.json under
# "6_recreatable_from_valid_states", stated there before any confirmation data
# existed: BLUE ahead on score, RED carrying the blue flag, both BLUE agents
# past the midline, RED defender off cooldown.
C1_BLUE_SCORE = 1        # ahead, and two clear of score_limit=3 so nothing ends
C1_RED_SCORE = 0
C1_CARRIER_PROGRESS = 0.30   # carrier this fraction of the way blue home -> red home
C1_CARRIER_JITTER = 0.08     # +/- on that fraction
C1_BLUE_FORWARD_MIN = 1.5    # cells past the midline
C1_BLUE_FORWARD_MAX = 4.0
C1_LATERAL_JITTER = 2.5      # +/- cells about the vertical centre


def apply_c1_scenario(core, env_indices: Sequence[int] | None = None) -> None:
    """Mutate freshly reset envs so the episode STARTS inside C1.

    Geometry, score and carry state only -- no reward shaping and no z label,
    exactly as the LRO protocol requires. Randomised within declared ranges so
    O1 sees the C1 region rather than one memorised state.

    Two properties are deliberately NOT set:

    ``step_count``  left at 0. C1 fires mid-episode in real play, so an injected
        C1 start hands O1 a full horizon that a natural C1 would not have. The
        frozen construction says nothing about the clock, and inventing one
        would add a free parameter to a preregistered scenario. This is a
        declared injection artifact, and is the reason the O1 gates are scored
        on natural episodes rather than injected ones.

    ``blue_carrying``  cleared. The frozen construction does not mention blue
        possession; clearing it is the choice that adds nothing.
    """
    idxs = list(range(int(core.B))) if env_indices is None else [int(i) for i in env_indices]
    if not idxs:
        return

    n = len(idxs)
    dev = core.device
    sel = torch.as_tensor(idxs, dtype=torch.long, device=dev)

    cols = float(core.cols)
    rows = float(core.rows)
    mid_x = 0.5 * cols
    mid_y = 0.5 * rows

    def _u(lo: float, hi: float, shape) -> torch.Tensor:
        return core._rand_uniform(shape, lo, hi)

    # --- clean slate --------------------------------------------------------
    core.blue_carrying[sel] = False
    core.red_carrying[sel] = False
    core.blue_alive[sel] = True
    core.red_alive[sel] = True
    core.blue_tagged[sel] = False
    core.red_tagged[sel] = False
    core.blue_speed[sel] = 0.0
    core.red_speed[sel] = 0.0
    core.blue_respawn[sel] = 0
    core.red_respawn[sel] = 0

    # --- BLUE ahead on score ------------------------------------------------
    core.blue_score[sel] = C1_BLUE_SCORE
    core.red_score[sel] = C1_RED_SCORE

    # --- RED defender off cooldown -----------------------------------------
    core.blue_tag_cooldown[sel] = 0.0
    core.red_tag_cooldown[sel] = 0.0

    # --- RED carrying the blue flag, heading home ---------------------------
    bh_x = core.blue_flag_home[sel, 0]
    rh_x = core.red_flag_home[sel, 0]
    progress = _u(
        C1_CARRIER_PROGRESS - C1_CARRIER_JITTER,
        C1_CARRIER_PROGRESS + C1_CARRIER_JITTER,
        (n,),
    )
    carrier_x = bh_x + progress * (rh_x - bh_x)
    carrier_y = mid_y + _u(-C1_LATERAL_JITTER, C1_LATERAL_JITTER, (n,))

    core.red_carrying[sel, 0] = True
    core.red_x[sel, 0] = carrier_x
    core.red_y[sel, 0] = carrier_y
    core.red_heading[sel, 0] = 0.0  # toward +x, i.e. toward red home

    n_red = int(core.red_x.shape[1])
    if n_red > 1:
        # Second red between the carrier and the midline: the escort/defender
        # whose availability C1's supporting features tracked.
        for a in range(1, n_red):
            core.red_x[sel, a] = carrier_x + 1.5 + 0.6 * (a - 1)
            core.red_y[sel, a] = mid_y + _u(-C1_LATERAL_JITTER, C1_LATERAL_JITTER, (n,))
            core.red_heading[sel, a] = 0.0

    # The flag travels with its carrier; set it now so step 0 is coherent
    # rather than waiting for _apply_flag_rules on the first step.
    core.blue_flag_pos[sel, 0] = carrier_x
    core.blue_flag_pos[sel, 1] = carrier_y
    core.red_flag_pos[sel] = core.red_flag_home[sel]

    # --- both BLUE agents past the midline ----------------------------------
    n_blue = int(core.blue_x.shape[1])
    for a in range(n_blue):
        core.blue_x[sel, a] = mid_x + _u(C1_BLUE_FORWARD_MIN, C1_BLUE_FORWARD_MAX, (n,))
        core.blue_y[sel, a] = mid_y + _u(-C1_LATERAL_JITTER, C1_LATERAL_JITTER, (n,))
        core.blue_heading[sel, a] = 0.0

    # --- keep everything on the field ---------------------------------------
    for t in (core.blue_x, core.red_x):
        t[sel] = t[sel].clamp(0.5, cols - 0.5)
    for t in (core.blue_y, core.red_y):
        t[sel] = t[sel].clamp(0.5, rows - 0.5)
    core.blue_flag_pos[sel, 0] = core.blue_flag_pos[sel, 0].clamp(0.5, cols - 0.5)
    core.blue_flag_pos[sel, 1] = core.blue_flag_pos[sel, 1].clamp(0.5, rows - 0.5)


def attach_c1_injector(env) -> None:
    """Install ``apply_c1_scenario`` on an env's post-reset hook.

    Chains onto any existing hook rather than replacing it, matching
    ``rl/custom_ppo/phase_pod_runtime.py``.
    """
    import numpy as np

    prev = getattr(env, "_after_reset_indices_hook", None)

    def _after(done, infos) -> None:
        if callable(prev):
            prev(done, infos)
        idxs = [int(i) for i in np.where(np.asarray(done, dtype=bool))[0]]
        if idxs:
            apply_c1_scenario(env.core, env_indices=idxs)

    env._after_reset_indices_hook = _after


__all__ = [
    "C1_BLUE_FORWARD_MAX",
    "C1_BLUE_FORWARD_MIN",
    "C1_BLUE_SCORE",
    "C1_CARRIER_PROGRESS",
    "C1_RED_SCORE",
    "FLAG_AWAY_FRAC",
    "HOME_THREAT_MARGIN",
    "apply_c1_scenario",
    "attach_c1_injector",
    "c1_active_from_context",
    "c1_active_mask",
]
