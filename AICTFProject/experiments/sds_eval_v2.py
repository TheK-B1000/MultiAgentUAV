"""Candidate evaluation under SEARCH_OBJECTIVE_V2 + OBSERVABILITY_V2.

Replaces the V1 evaluation path for FUTURE search only. Nothing in the frozen
V1 modules is edited: `run_episode` (the treatment semantics) and
`degeneracy_penalty` are imported unchanged, exactly as the confirmation runner
does, so the search cannot silently drift from the environment it is searching.

What changed from V1
--------------------
V1 objective:   J = min(delta_G, delta_B_anchor) - degeneracy_penalty
                eligibility used precommitment_uncertain =
                mean(t_intent - t_commit) > 0 with sentinel 241

V2 objective:   payoff is a CONSTRAINT, not a term to trade away.
                J_v2 = p_C - degeneracy_penalty, admissible only while
                delta_G stays above the stage's payoff floor.

The reason for the split is the confirmed result: the two-way payoff reversal
already exists and is held-out confirmed, so the searcher no longer has to
discover it. What is missing is concealment, so concealment is what J rewards.
Letting a candidate buy p_C by giving up delta_G would destroy the asset we
already have, which is why the payoff floor is a gate rather than a summand.

Threshold status
----------------
The payoff floor and the p_C screens below are DEVELOPMENT SCREENS. They steer
search and may be tuned without a protocol amendment. They are NOT the
confirmation gate. The confirmation gate is frozen and is not reachable from
this file: delta_G >= 0.15 with LCB95 > 0, and LCB95(p_C) > 0.50 at n=64.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.observability_v2 import assay  # noqa: E402
from experiments.sds_genome import SDSGenome, degeneracy_penalty  # noqa: E402
from experiments.strategic_demand_searcher import (  # noqa: E402
    BREACH, GUARD, MAX_STEPS, run_episode,
)

# ---- development screens (tunable, NOT the confirmation gate) -------------
# Progressive payoff floors by stage: the deeper the stage, the more payoff
# pressure a candidate must still be carrying to justify further spend.
PAYOFF_FLOOR_BY_STAGE = {16: 0.00, 32: 0.10}
# p_C screens chosen so a survivor is plausibly able to clear LCB95(p_C) > 0.5
# at the frozen confirmation size of n=64. At n=64 a true p_C of 0.70 clears
# with 0.89 power, so a candidate screening below ~0.65 at n=32 is not a
# credible confirmation candidate.
P_C_SCREEN_BY_STAGE = {16: 0.50, 32: 0.65}

# ---- frozen confirmation gate, for reference and assertion only ----------
CONFIRM_N = 64
CONFIRM_DELTA_G_FLOOR = 0.15
CONFIRM_P_C_LCB_FLOOR = 0.50


def paired_eval_v2(genome: SDSGenome, *, n: int, seed_base: int, device: str,
                   stage: Optional[int] = None) -> dict:
    """Evaluate one candidate. Payoff is a constraint; p_C is the target.

    Observability is measured on the BREACH episodes, matching the convention
    used throughout the campaign: commitment is a BREACH behaviour, so the
    question 'did BLUE have to commit before intent was readable' is only
    meaningful in the arm where BLUE actually commits.
    """
    guard_w, breach_w, totals, zz, breach_eps = [], [], [], [], []
    for i in range(n):
        seed = seed_base + i
        g = run_episode(style=GUARD, genome=genome, seed=seed, device=device)
        b = run_episode(style=BREACH, genome=genome, seed=seed, device=device)
        guard_w.append(g["win"])
        breach_w.append(b["win"])
        totals.extend([g["total_score"], b["total_score"]])
        zz.extend([g["zero_zero"], b["zero_zero"]])
        breach_eps.append({"t_intent": b["t_intent"], "t_commit": b["t_commit"]})

    gw, bw = float(np.mean(guard_w)), float(np.mean(breach_w))
    delta_g = gw - bw
    frac00, mean_tot = float(np.mean(zz)), float(np.mean(totals))
    pen = degeneracy_penalty(frac00, mean_tot)

    obs = assay(breach_eps, horizon=MAX_STEPS)
    p_c = obs["p_C"]

    payoff_floor = PAYOFF_FLOOR_BY_STAGE.get(stage or n, 0.0)
    p_c_screen = P_C_SCREEN_BY_STAGE.get(stage or n, 0.50)
    payoff_ok = bool(delta_g > payoff_floor)

    return {
        "genome": genome.to_dict(),
        "objective": "SEARCH_OBJECTIVE_V2",
        "observability": "OBSERVABILITY_V2",
        "n": n, "seed_base": seed_base, "stage": stage or n,

        # payoff -- a constraint
        "guard_wr": gw, "breach_wr": bw, "delta_G": delta_g,
        "payoff_floor_this_stage": payoff_floor,
        "payoff_constraint_ok": payoff_ok,

        # observability -- the search target
        "p_C": p_c,
        "p_C_lcb95": obs["lcb95"],
        "p_C_counts": obs["counts"],
        "p_C_screen_this_stage": p_c_screen,
        "p_C_screen_ok": bool(p_c >= p_c_screen),

        # degeneracy -- penalised, never a confirmation gate
        "frac_0_0": frac00, "mean_total_score": mean_tot,
        "degeneracy_penalty": pen,

        # objective
        "J_v2": float(p_c - pen) if payoff_ok else float("-inf"),
        "J_v2_definition": "p_C - degeneracy_penalty, admissible only while "
                           "delta_G clears the stage payoff floor",

        # V1 quantities retained as telemetry so old and new runs stay legible
        "telemetry_v1_mean_intent_minus_commit":
            obs["telemetry"]["complete_case_mean_gap"],
        "telemetry": obs["telemetry"],

        "search_not_gate_B": True,
    }


def development_eligible_v2(rec: dict) -> bool:
    """A candidate worth freezing and spending an audited block on.

    Deliberately stricter than V1: V1 promoted SDS_G1_4 on a statistic that
    reversed sign between development and confirmation. Here the payoff
    constraint and the observability screen must BOTH hold, and J_v2 must be
    positive, which requires p_C to exceed the degeneracy penalty rather than
    merely being non-negative.
    """
    return bool(
        rec.get("payoff_constraint_ok")
        and rec.get("p_C_screen_ok")
        and float(rec.get("J_v2", float("-inf"))) > 0.0
    )


def weakest_dimension_v2(rec: dict) -> str:
    """Which gene group to bias mutation toward, under the V2 objective."""
    payoff_gap = max(0.0, CONFIRM_DELTA_G_FLOOR - float(rec.get("delta_G", -9.0)))
    conceal_gap = max(0.0, 0.75 - float(rec.get("p_C", 0.0)))
    degen_gap = max(0.0, float(rec.get("frac_0_0", 1.0)) - 0.25)
    worst = max((payoff_gap, "GUARD_pressure"),
                (conceal_gap, "concealment"),
                (degen_gap, "non_degeneracy"))
    return worst[1]
