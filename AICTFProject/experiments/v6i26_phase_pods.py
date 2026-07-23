"""V6I26 phase-pod definitions: exclusive strategic niches for Path-C birth.

Pods are environment *scenario filters*, not z-role reward labels. Teachers
never receive a "you are z1" bonus. After birth, z indices are distilled slots.

Classification: DIAGNOSTIC (Claim B / population-guided; not PAPER-FAITHFUL).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

POD_OPEN_PRESSURE = "open_pressure"
POD_INTERCEPT = "intercept"
POD_ESCORT = "escort"
POD_DEFEND_LEAD = "defend_lead"

PHASE_POD_IDS: tuple[str, ...] = (
    POD_OPEN_PRESSURE,
    POD_INTERCEPT,
    POD_ESCORT,
    POD_DEFEND_LEAD,
)

POD_TO_Z: dict[str, int] = {
    POD_OPEN_PRESSURE: 0,
    POD_INTERCEPT: 1,
    POD_ESCORT: 2,
    POD_DEFEND_LEAD: 3,
}

Z_TO_POD: dict[int, str] = {v: k for k, v in POD_TO_Z.items()}


@dataclass(frozen=True)
class PhasePodSpec:
    pod_id: str
    member_id: int
    label: str
    description: str
    seed_offset: int

    @property
    def latent_z(self) -> int:
        return int(POD_TO_Z[self.pod_id])


PHASE_POD_SPECS: tuple[PhasePodSpec, ...] = (
    PhasePodSpec(
        pod_id=POD_OPEN_PRESSURE,
        member_id=0,
        label="open_pressure",
        description="No carriers; early scramble — early pressure / race",
        seed_offset=0,
    ),
    PhasePodSpec(
        pod_id=POD_INTERCEPT,
        member_id=1,
        label="intercept",
        description="Red has flag carrier mid-field — intercept and deny",
        seed_offset=100,
    ),
    PhasePodSpec(
        pod_id=POD_ESCORT,
        member_id=2,
        label="escort",
        description="Blue has flag carrier mid-field — protect and convert",
        seed_offset=200,
    ),
    PhasePodSpec(
        pod_id=POD_DEFEND_LEAD,
        member_id=3,
        label="defend_lead",
        description="Blue score lead + late clock — stall / home defense",
        seed_offset=300,
    ),
)


def get_phase_pod_specs(member_ids: Sequence[int] | None = None) -> list[PhasePodSpec]:
    specs = list(PHASE_POD_SPECS)
    if member_ids is None:
        return specs
    keep = {int(m) for m in member_ids}
    return [s for s in specs if s.member_id in keep]


def phase_pods_manifest() -> dict[str, Any]:
    return {
        "protocol": "v6i26_phase_pod_exclusive_birth",
        "classification": "DIAGNOSTIC",
        "path": "C_phase_pod_psro",
        "claim": "B_population_guided",
        "pods": [asdict(s) for s in PHASE_POD_SPECS],
        "pod_to_z": dict(POD_TO_Z),
        "note": (
            "Exclusive scenario injection at episode start; no z-role rewards; "
            "contract specialist OFF; router OFF during birth."
        ),
    }


def classify_phase_from_core(core: Any, env_index: int = 0) -> str:
    """Label the current strategic phase from live core tensors (eval / audit)."""
    i = int(env_index)
    blue_carry = bool(core.blue_carrying[i].any().item())
    red_carry = bool(core.red_carrying[i].any().item())
    score_limit = max(1.0, float(getattr(core, "score_limit", 1) or 1))
    blue_score = float(core.blue_score[i].item())
    red_score = float(core.red_score[i].item())
    max_steps = max(1, int(getattr(core, "max_decision_steps", 240) or 240))
    step = float(core.decision_step[i].item()) if hasattr(core, "decision_step") else 0.0
    frac = step / float(max_steps)
    score_diff = (blue_score - red_score) / score_limit

    if red_carry and not blue_carry:
        return POD_INTERCEPT
    if blue_carry and not red_carry:
        return POD_ESCORT
    if score_diff > 0.05 and frac >= 0.55:
        return POD_DEFEND_LEAD
    return POD_OPEN_PRESSURE


def classify_phase_from_global_state(gs: Any) -> str:
    """Classify from a GLOBAL_STATE_DIM vector (numpy or tensor)."""
    from rl.global_state import GLOBAL_STATE_FIELD_NAMES

    arr = gs.detach().cpu().numpy() if hasattr(gs, "detach") else gs
    flat = arr.reshape(-1)
    names = list(GLOBAL_STATE_FIELD_NAMES)
    idx = {n: i for i, n in enumerate(names)}

    def _f(name: str, default: float = 0.0) -> float:
        j = idx.get(name)
        if j is None or j >= flat.shape[0]:
            return default
        return float(flat[j])

    # Capture bits are 10/11; carrier geometry collapses when no carrier.
    blue_flag_cap = _f("blue_flag_captured")
    red_flag_cap = _f("red_flag_captured")
    carrier_home = _f("carrier_dist_home", 1.0)
    score_diff = _f("score_diff_norm")
    decision_frac = _f("decision_frac")

    # Heuristic: captured bit + non-default carrier home ⇒ active carrier side.
    # blue_flag_captured means blue took red's flag (blue is carrying).
    if blue_flag_cap > 0.5 and carrier_home < 0.95:
        return POD_ESCORT
    if red_flag_cap > 0.5 and carrier_home < 0.95:
        return POD_INTERCEPT
    if score_diff > 0.05 and decision_frac >= 0.55:
        return POD_DEFEND_LEAD
    return POD_OPEN_PRESSURE


def apply_phase_pod_scenario(core: Any, pod_id: str, env_indices: Sequence[int] | None = None) -> None:
    """Mutate core state so the episode *starts* inside the requested strategic niche.

    No reward labels — geometry / score / clock only.
    """
    pod = str(pod_id).strip().lower()
    if pod not in POD_TO_Z:
        raise ValueError(f"Unknown phase pod_id={pod_id!r}; expected one of {PHASE_POD_IDS}")

    if env_indices is None:
        idxs = list(range(int(core.B)))
    else:
        idxs = [int(i) for i in env_indices]

    W = float(getattr(core, "cols", 20) or 20)
    H = float(getattr(core, "rows", 20) or 20)
    mid_x = 0.5 * W
    mid_y = 0.5 * H
    max_steps = int(getattr(core, "max_decision_steps", 240) or 240)
    score_limit = float(getattr(core, "score_limit", 3) or 3)

    for i in idxs:
        # Clear carriers first.
        core.blue_carrying[i] = False
        core.red_carrying[i] = False
        core.blue_alive[i] = True
        core.red_alive[i] = True
        if hasattr(core, "blue_tagged"):
            core.blue_tagged[i] = False
        if hasattr(core, "red_tagged"):
            core.red_tagged[i] = False

        bh = core.blue_flag_home[i]
        rh = core.red_flag_home[i]

        if pod == POD_OPEN_PRESSURE:
            core.blue_score[i] = 0
            core.red_score[i] = 0
            if hasattr(core, "decision_step"):
                core.decision_step[i] = 0
            # Blue near mid-left (attacking), red near mid-right.
            n_b = int(core.blue_x.shape[1])
            n_r = int(core.red_x.shape[1])
            for a in range(n_b):
                core.blue_x[i, a] = mid_x - 2.0 - 0.8 * a
                core.blue_y[i, a] = mid_y + (a - 0.5) * 1.5
            for a in range(n_r):
                core.red_x[i, a] = mid_x + 2.0 + 0.8 * a
                core.red_y[i, a] = mid_y + (a - 0.5) * 1.5

        elif pod == POD_INTERCEPT:
            # Red carries toward blue home; blue must intercept.
            core.blue_score[i] = 0
            core.red_score[i] = 0
            if hasattr(core, "decision_step"):
                core.decision_step[i] = max(1, max_steps // 5)
            carrier = 0
            core.red_carrying[i, carrier] = True
            # Midway between red home and blue home.
            core.red_x[i, carrier] = 0.55 * float(bh[0].item()) + 0.45 * float(rh[0].item())
            core.red_y[i, carrier] = mid_y
            if int(core.red_x.shape[1]) > 1:
                core.red_x[i, 1] = float(core.red_x[i, carrier].item()) - 1.5
                core.red_y[i, 1] = mid_y + 1.2
            n_b = int(core.blue_x.shape[1])
            for a in range(n_b):
                core.blue_x[i, a] = float(bh[0].item()) + 2.0 + 0.5 * a
                core.blue_y[i, a] = mid_y + (a - 0.5) * 2.0

        elif pod == POD_ESCORT:
            # Blue carries toward blue home; protect conversion.
            core.blue_score[i] = 0
            core.red_score[i] = 0
            if hasattr(core, "decision_step"):
                core.decision_step[i] = max(1, max_steps // 5)
            carrier = 0
            core.blue_carrying[i, carrier] = True
            core.blue_x[i, carrier] = 0.45 * float(bh[0].item()) + 0.55 * float(rh[0].item())
            core.blue_y[i, carrier] = mid_y
            if int(core.blue_x.shape[1]) > 1:
                core.blue_x[i, 1] = float(core.blue_x[i, carrier].item()) + 1.2
                core.blue_y[i, 1] = mid_y - 1.0
            n_r = int(core.red_x.shape[1])
            for a in range(n_r):
                core.red_x[i, a] = float(core.blue_x[i, carrier].item()) - 2.0
                core.red_y[i, a] = mid_y + (a - 0.5) * 2.0

        elif pod == POD_DEFEND_LEAD:
            core.blue_score[i] = max(1, int(score_limit) - 1)
            core.red_score[i] = 0
            if hasattr(core, "decision_step"):
                core.decision_step[i] = max(1, int(0.7 * max_steps))
            # Blue near home; red pressuring blue flag.
            n_b = int(core.blue_x.shape[1])
            n_r = int(core.red_x.shape[1])
            for a in range(n_b):
                core.blue_x[i, a] = float(bh[0].item()) + 1.5 + 0.4 * a
                core.blue_y[i, a] = float(bh[1].item()) + (a - 0.5) * 1.5
            for a in range(n_r):
                core.red_x[i, a] = float(bh[0].item()) + 4.0 + 0.5 * a
                core.red_y[i, a] = float(bh[1].item()) + (a - 0.5) * 1.8

        # Keep positions inside field if clamps exist.
        if hasattr(core, "blue_x"):
            core.blue_x[i].clamp_(0.5, W - 0.5)
            core.blue_y[i].clamp_(0.5, H - 0.5)
            core.red_x[i].clamp_(0.5, W - 0.5)
            core.red_y[i].clamp_(0.5, H - 0.5)


def apply_phase_pod_scenario_env_method(
    core: Any,
    pod_id: str,
    *,
    env_indices: Sequence[int] | None = None,
) -> None:
    """Core method surface for ``env.env_method('apply_phase_pod_scenario', ...)``."""
    apply_phase_pod_scenario(core, pod_id, env_indices=env_indices)


__all__ = [
    "PHASE_POD_IDS",
    "PHASE_POD_SPECS",
    "POD_DEFEND_LEAD",
    "POD_ESCORT",
    "POD_INTERCEPT",
    "POD_OPEN_PRESSURE",
    "POD_TO_Z",
    "PhasePodSpec",
    "Z_TO_POD",
    "apply_phase_pod_scenario",
    "apply_phase_pod_scenario_env_method",
    "classify_phase_from_core",
    "classify_phase_from_global_state",
    "get_phase_pod_specs",
    "phase_pods_manifest",
]
