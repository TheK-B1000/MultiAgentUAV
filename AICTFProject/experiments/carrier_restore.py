r"""CARRIER-EVASION-RESTORED pathway + its known-answer anchors.

The student action interface hard-overrides a flag carrier to EXACT own home:
`gpu_env/_core/_rules.py::_build_targets_from_action` ends with

    tx = torch.where(own_carrying, own_flag_home[:, None, 0], tx)

so a carrying agent steers straight home regardless of macro or target, while
the scripted teacher can issue a tangent/evasive carrier target. The 4v4 pole-A
trace measured that gap at up to ~14 cells on 10-20% of agent-ticks.

This module builds the counterfactual arm: identical to CURRENT PROJECTED in
every respect EXCEPT that a carrier's teacher target is preserved through the
student-compatible pathway instead of being collapsed to home.

HOW THE OVERRIDE IS SUPPRESSED (rule 11: use the real code path)
    We do NOT reimplement `_build_targets_from_action`. We temporarily zero
    `core.blue_carrying` around the call, so the real function runs unchanged
    and only its two `own_carrying` `torch.where` lines become no-ops. Reading
    the function confirms `own_carrying` is used nowhere else in it. The flag is
    restored immediately, so scoring/mechanics elsewhere are untouched.
"""

from __future__ import annotations

import numpy as np
import torch

from macro_actions import MacroAction


class CarrierRestore:
    """Context manager suppressing ONLY the carrying->home override."""

    def __init__(self, core):
        self.core = core
        self._saved = None

    def __enter__(self):
        self._saved = self.core.blue_carrying.clone()
        self.core.blue_carrying = torch.zeros_like(self.core.blue_carrying)
        return self

    def __exit__(self, *exc):
        self.core.blue_carrying = self._saved
        self._saved = None


def adapt_no_carrier_override(core, tx: float, ty: float, agent: int, W: np.ndarray):
    """`teacher_action_adapter.adapt` but WITHOUT the carrier short-circuit, so a
    carrier is routed through the ordinary semantic/waypoint choice."""
    from experiments.teacher_action_adapter import Adapted, executed_target, EXACT_TOL

    if bool(core.blue_tagged[0, agent]):
        with CarrierRestore(core):
            ex, ey = executed_target(core, MacroAction.GO_TO, 0, agent)
        return Adapted("FORCED_HOME_TAGGED", None, None,
                       float(np.hypot(tx - ex, ty - ey)), False, "tagged override kept")

    best = None
    with CarrierRestore(core):
        for macro, cat in ((MacroAction.GET_FLAG, "GET_FLAG"),
                           (MacroAction.GO_HOME, "GO_HOME")):
            ex, ey = executed_target(core, macro, 0, agent)
            r = float(np.hypot(tx - ex, ty - ey))
            if best is None or r < best.residual:
                best = Adapted(cat, int(macro), None, r, False, "semantic")
        j = int(np.argmin((W[:, 0] - tx) ** 2 + (W[:, 1] - ty) ** 2))
        ex, ey = executed_target(core, MacroAction.GO_TO, j, agent)
        r = float(np.hypot(tx - ex, ty - ey))
        if best is None or r < best.residual:
            best = Adapted("WAYPOINT", int(MacroAction.GO_TO), j, r, True, "waypoint")
    return best


def carrier_anchors(core, W: np.ndarray, agent: int = 0) -> list[str]:
    """The three known-answer anchors. Non-empty return == do not launch.

    1. NON-CARRIER: restored arm must be BIT-IDENTICAL to current projected.
    2. CARRIER, teacher target == home: restored arm must equal current behaviour.
    3. CARRIER, teacher target != home: restored arm must PRESERVE that target
       instead of substituting home.  <-- the one that matters
    """
    from experiments.teacher_action_adapter import adapt, executed_target

    fails: list[str] = []
    hx = float(core.blue_flag_home[0, 0]); hy = float(core.blue_flag_home[0, 1])
    saved = core.blue_carrying.clone()
    try:
        # ---- anchor 1: non-carrier equivalence -------------------------------
        core.blue_carrying[0, agent] = False
        probe = (float(W[11, 0]), float(W[11, 1]))
        a_cur = adapt(core, probe[0], probe[1], agent, W)
        a_new = adapt_no_carrier_override(core, probe[0], probe[1], agent, W)
        if (a_cur.category, a_cur.macro, a_cur.target_idx) != \
           (a_new.category, a_new.macro, a_new.target_idx) or \
           abs(a_cur.residual - a_new.residual) > 1e-9:
            fails.append(f"anchor1 NON-CARRIER differs: current="
                         f"({a_cur.category},{a_cur.macro},{a_cur.target_idx},{a_cur.residual:.6f}) "
                         f"new=({a_new.category},{a_new.macro},{a_new.target_idx},{a_new.residual:.6f})")

        # ---- anchor 2: carrier whose teacher target IS home ------------------
        core.blue_carrying[0, agent] = True
        a_cur = adapt(core, hx, hy, agent, W)
        a_new = adapt_no_carrier_override(core, hx, hy, agent, W)
        if a_cur.residual > 1e-6 or a_new.residual > 1e-6:
            fails.append(f"anchor2 CARRIER@home should be exactly reachable in both: "
                         f"current residual={a_cur.residual:.6f} new={a_new.residual:.6f}")

        # ---- anchor 3: carrier whose teacher target is NOT home --------------
        # pick a waypoint far from home to stand in for an evasive target
        d2 = (W[:, 0] - hx) ** 2 + (W[:, 1] - hy) ** 2
        far = int(np.argmax(d2))
        ev = (float(W[far, 0]), float(W[far, 1]))
        dist_home = float(np.hypot(ev[0] - hx, ev[1] - hy))
        a_cur = adapt(core, ev[0], ev[1], agent, W)
        a_new = adapt_no_carrier_override(core, ev[0], ev[1], agent, W)
        if a_cur.category != "FORCED_HOME_CARRYING":
            fails.append(f"anchor3 precondition: current arm should force home for a "
                         f"carrier, got {a_cur.category}")
        if abs(a_cur.residual - dist_home) > 1e-3:
            fails.append(f"anchor3 current-arm residual should equal distance-to-home "
                         f"({dist_home:.3f}), got {a_cur.residual:.3f}")
        if a_new.residual > 1e-6:
            fails.append(f"anchor3 RESTORED arm must PRESERVE the evasive target "
                         f"(residual ~0), got {a_new.residual:.6f} "
                         f"category={a_new.category}")
        # and the executed target under the restored arm must be the evasive point
        if a_new.category == "WAYPOINT":
            with CarrierRestore(core):
                ex, ey = executed_target(core, a_new.macro, a_new.target_idx, agent)
            if np.hypot(ex - ev[0], ey - ev[1]) > 1e-6:
                fails.append(f"anchor3 executed target ({ex:.3f},{ey:.3f}) != evasive "
                             f"target ({ev[0]:.3f},{ev[1]:.3f})")
            if np.hypot(ex - hx, ey - hy) < 1e-6:
                fails.append("anchor3 restored arm still executed HOME")
    finally:
        core.blue_carrying = saved
    return fails
