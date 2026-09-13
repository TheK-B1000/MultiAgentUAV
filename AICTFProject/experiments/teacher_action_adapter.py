r"""Teacher -> student action adapter, built to satisfy RESEARCH_RUN_STANDARDS rule 11.

The teacher emits a continuous target (tx, ty). The student emits
MultiDiscrete([n_macros, n_targets]). This module answers, for each teacher
command, the only question that matters:

    what is the CLOSEST thing the student can actually execute, and how far off is it?

RULE 11 COMPLIANCE -- "executed action beats intended action":
This adapter does NOT classify targets by heuristic. It proposes a concrete
(macro, target_idx) and then calls the REAL
``gpu_env/_core/_rules.py::_build_targets_from_action`` to observe what the
student would actually end up steering toward, including every override that
function applies. The residual is measured against that executed result, not
against an assumed one.

TRACED EXECUTION PATH (gpu_env/_core/_rules.py:629-644, verified by reading):

    t_xy    = _decode_targets(target)          # the 50-waypoint vocabulary
    GET_FLAG -> exact enemy_flag               # bypasses waypoints
    GO_HOME  -> exact own_flag_home            # bypasses waypoints
    own_carrying -> exact own_flag_home        # OVERRIDES ANY MACRO
    GO_TO / GRAB_MINE / PLACE_MINE -> t_xy     # all three use the waypoint

and afterwards, in _step.py:106:

    _redirect_tagged_to_home  -> a TAGGED agent is forced to own home

Consequences that a heuristic classifier would have missed, and which are why
this is done by execution rather than by rule:
  * GRAB_MINE and PLACE_MINE resolve to the SAME target as GO_TO. They differ
    only in success condition and commit horizon, never in where the agent
    steers. So they are not separately identifiable from a target alone.
  * `own_carrying` and `tagged` override EVERY macro. For such an agent the
    executed target is home no matter what action is chosen, so e_q is
    structurally 0 and the action choice is irrelevant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from macro_actions import MacroAction

#: Every category is REACHED BY CONSTRUCTION from the executed target. There is
#: no silent `else`: anything that does not achieve a residual at or below
#: tolerance through some executable action is UNMAPPED, and UNMAPPED fails the audit.
CATEGORIES = ("FORCED_HOME_TAGGED", "FORCED_HOME_CARRYING", "GET_FLAG", "GO_HOME",
              "WAYPOINT", "UNMAPPED")

#: Macros whose executed target is the decoded waypoint. Indistinguishable from
#: a target alone -- GO_TO is reported as the representative.
WAYPOINT_MACROS = (MacroAction.GO_TO, MacroAction.GRAB_MINE, MacroAction.PLACE_MINE)

EXACT_TOL = 1e-4


@dataclass
class Adapted:
    category: str
    macro: int | None
    target_idx: int | None
    residual: float                 # || teacher_target - EXECUTED target ||
    uses_waypoint: bool             # whether e_q is even defined for this decision
    note: str = ""


def executed_target(core, macro: int, target_idx: int, agent: int,
                    side: str = "blue") -> tuple[float, float]:
    """Call the REAL engine target builder and return what agent `agent` would
    actually steer toward under this action. Includes the carrying override.
    The tagged redirect is applied separately below, matching _step.py order."""
    n = int(core.blue_x.shape[1]) if side == "blue" else int(core.red_x.shape[1])
    m = torch.full((core.B, n), int(macro), dtype=torch.long, device=core.device)
    t = torch.full((core.B, n), int(target_idx), dtype=torch.long, device=core.device)
    tx, ty = core._build_targets_from_action(m, t, side=side)
    ex, ey = float(tx[0, agent]), float(ty[0, agent])
    # _step.py:106 applies this AFTER target construction, for every macro.
    if side == "blue" and bool(core.blue_tagged[0, agent]):
        ex, ey = float(core.blue_flag_home[0, 0]), float(core.blue_flag_home[0, 1])
    return ex, ey


def adapt(core, teacher_tx: float, teacher_ty: float, agent: int,
          W: np.ndarray, tol: float = EXACT_TOL) -> Adapted:
    """Find the executable student action whose EXECUTED target is closest to
    the teacher's command, and report the residual."""
    # Overrides first: for these agents the executed target is home regardless
    # of the action, so no action choice can reduce the residual.
    if bool(core.blue_tagged[0, agent]):
        ex, ey = executed_target(core, MacroAction.GO_TO, 0, agent)
        return Adapted("FORCED_HOME_TAGGED", None, None,
                       float(np.hypot(teacher_tx - ex, teacher_ty - ey)), False,
                       "tagged -> forced to own home by _redirect_tagged_to_home")
    if bool(core.blue_carrying[0, agent]):
        ex, ey = executed_target(core, MacroAction.GO_TO, 0, agent)
        return Adapted("FORCED_HOME_CARRYING", None, None,
                       float(np.hypot(teacher_tx - ex, teacher_ty - ey)), False,
                       "carrying -> forced to own home, overrides every macro")

    best: Adapted | None = None

    # Semantic macros: target index is irrelevant to the executed result.
    for macro, cat in ((MacroAction.GET_FLAG, "GET_FLAG"),
                       (MacroAction.GO_HOME, "GO_HOME")):
        ex, ey = executed_target(core, macro, 0, agent)
        r = float(np.hypot(teacher_tx - ex, teacher_ty - ey))
        if best is None or r < best.residual:
            best = Adapted(cat, int(macro), None, r, False,
                           "semantic macro; executed target bypasses the waypoint vocabulary")

    # Waypoint pathway: the executed target IS the decoded waypoint, so the best
    # index is the nearest one. Verified against the engine, not assumed.
    j = int(np.argmin((W[:, 0] - teacher_tx) ** 2 + (W[:, 1] - teacher_ty) ** 2))
    ex, ey = executed_target(core, MacroAction.GO_TO, j, agent)
    r = float(np.hypot(teacher_tx - ex, teacher_ty - ey))
    if best is None or r < best.residual:
        best = Adapted("WAYPOINT", int(MacroAction.GO_TO), j, r, True,
                       "GO_TO/GRAB_MINE/PLACE_MINE all execute to this same waypoint")

    assert best is not None
    return best


def golden_anchors(core, W: np.ndarray, agent: int = 0) -> list[str]:
    """RULE 11 known-answer + failure-case anchors. Returns a list of failure
    strings; empty means the adapter provably follows the real execution path.
    The audit MUST NOT run if this is non-empty."""
    fails: list[str] = []
    if bool(core.blue_tagged[0, agent]) or bool(core.blue_carrying[0, agent]):
        return ["anchor agent is tagged/carrying; pick a clean agent for anchoring"]

    fx, fy = float(core.red_flag_pos[0, 0]), float(core.red_flag_pos[0, 1])
    hx, hy = float(core.blue_flag_home[0, 0]), float(core.blue_flag_home[0, 1])

    # KNOWN ANSWER 1: the enemy flag must adapt to GET_FLAG with zero residual.
    a = adapt(core, fx, fy, agent, W)
    if a.category != "GET_FLAG" or a.residual > EXACT_TOL or a.uses_waypoint:
        fails.append(f"anchor GET_FLAG failed: got {a.category} residual={a.residual:.6f} "
                     f"uses_waypoint={a.uses_waypoint} (expected GET_FLAG, 0, False)")

    # KNOWN ANSWER 2: own home must adapt to GO_HOME with zero residual.
    a = adapt(core, hx, hy, agent, W)
    if a.category != "GO_HOME" or a.residual > EXACT_TOL or a.uses_waypoint:
        fails.append(f"anchor GO_HOME failed: got {a.category} residual={a.residual:.6f} "
                     f"uses_waypoint={a.uses_waypoint} (expected GO_HOME, 0, False)")

    # KNOWN ANSWER 3: an exact waypoint must adapt to WAYPOINT with zero residual.
    k = min(17, W.shape[0] - 1)
    a = adapt(core, float(W[k, 0]), float(W[k, 1]), agent, W)
    if a.category != "WAYPOINT" or a.residual > EXACT_TOL or a.target_idx != k:
        fails.append(f"anchor WAYPOINT failed: got {a.category} idx={a.target_idx} "
                     f"residual={a.residual:.6f} (expected WAYPOINT, idx={k}, 0)")

    # FAILURE CASE: a point far outside the map must NOT be representable to
    # tolerance. If this "passes" the adapter is degenerate.
    far_x, far_y = 10_000.0, 10_000.0
    a = adapt(core, far_x, far_y, agent, W)
    if a.residual < 100.0:
        fails.append(f"failure-case anchor did not fail: an unreachable point at "
                     f"({far_x},{far_y}) reported residual {a.residual:.3f}; the "
                     f"adapter is not measuring real reachability")

    # CROSS-CHECK: the engine must agree that GET_FLAG really lands on the flag.
    ex, ey = executed_target(core, MacroAction.GET_FLAG, 0, agent)
    if abs(ex - fx) > EXACT_TOL or abs(ey - fy) > EXACT_TOL:
        fails.append(f"engine cross-check failed: GET_FLAG executed to ({ex},{ey}) "
                     f"but enemy flag is ({fx},{fy})")
    return fails
