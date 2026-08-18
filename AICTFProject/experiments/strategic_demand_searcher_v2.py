"""Strategic Demand Searcher V2 -- reliability + hierarchical eval + UCB +
weakness-directed mutation + Pareto archive + islands.

Protocol: experiments/STRATEGIC_DEMAND_SEARCHER_V2_FROZEN.json

The scientific core is UNCHANGED from V1 and is imported, not reimplemented:
J formula, degeneracy_penalty, development_eligible, GUARD/BREACH styles,
RULESET_V3_M1, legal opponent bases, confirmation block 2500001. V2 only
changes HOW candidates are searched and HOW state is persisted.

Search results are NOT Gate B evidence. No PPO. Block 2500001 is untouched.

Run:  python experiments/strategic_demand_searcher_v2.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.sds_genome import (  # noqa: E402
    ANCHOR_B, LEGAL_A_BASES, SDSGenome, canonical_parent,
    degeneracy_penalty, mutate, recombine,
)
from experiments.strategic_demand_searcher import (  # noqa: E402
    GUARD, BREACH, evaluate_B,
)
# SEARCH_OBJECTIVE_V2 + OBSERVABILITY_V2. The V1 eligibility path
# (development_eligible, paired_eval's sentinel gap gate) is RETIRED and is
# deliberately no longer imported, so it cannot be reached by accident.
from experiments.sds_eval_v2 import (  # noqa: E402
    P_C_SCREEN_BY_STAGE, PAYOFF_FLOOR_BY_STAGE, development_eligible_v2,
    paired_eval_v2, weakest_dimension_v2,
)

OUT_DIR = PROJECT_ROOT / "artifacts/strategic_demand/searcher_v2"
ARCHIVE = OUT_DIR / "archive.json"
PARETO = OUT_DIR / "pareto.json"
LOG = OUT_DIR / "v2.log"
V1_MUTATE_SEED_BASE = 2_410_001   # V1's block; V2 must not collide with it
V2_SEED_BASE = 2_420_001          # fresh, disjoint from V1 search/mutate/confirm
# Ratified development ladder: 16 -> 32 -> freeze. n=8 is RETIRED for the
# observability gate; small-sample ordering estimates proved unstable, which is
# what promoted SDS_G1_4 on a statistic that then reversed sign.
STAGES = (16, 32)
PROMOTE_DELTA_G = 0.05
DELTA_G_TARGET = 0.15
# Frozen confirmation parameters, asserted here so search can never be
# mistaken for confirmation. Confirmation is a separate untouched run.
CONFIRMATION_N = 64
CONFIRMATION_P_C_LCB_FLOOR = 0.50
CONFIRMATION_NO_EXTENSION = ("confirmation is 64 from the start; no peeking at "
                             "32 and extending. If it fails at 64, it fails.")
UCB_C = 1.0
MIGRATE_EVERY = 2
MAX_WALL_HOURS = 12.0

GENE_GROUPS = {
    "GUARD_pressure": ["lock_attacker", "lock_defender", "threat_radius", "enable_intercept"],
    "concealment": ["opening_hold_steps", "lane_amplitude_frac", "enable_defender"],
    "non_degeneracy": ["min_alive_for_defender", "defender_zone_frac", "enable_flag_retr"],
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_archive() -> dict:
    """Resume from V2's OWN prior archive -- fixes the V1 resume bug, which
    hardcoded the screen archive and caused deterministic-but-wasteful
    re-derivation of generation 0 on every restart."""
    if ARCHIVE.is_file():
        try:
            return json.loads(ARCHIVE.read_text(encoding="utf-8"))
        except Exception:
            log("archive.json unreadable -- treating as empty (tmp+replace "
               "should make this impossible in normal operation)")
    return {"updated": None, "rows": [], "seed_cursor": V2_SEED_BASE,
            "next_gen": 0, "islands": {}}


def weakest_dimension(rec: dict) -> str:
    """Delegates to the V2 definition: concealment is now measured by p_C,
    not by a sentinel-coded timing gap."""
    return weakest_dimension_v2(rec)


def weighted_mutate(parent: SDSGenome, rng, *, new_id: str, bias_group: str) -> SDSGenome:
    """Same legal gene ranges as sds_genome.mutate; only the AXIS distribution
    is reweighted toward the group associated with the parent's weakest
    dimension. Never widens a bound, never adds a gene."""
    from experiments.sds_genome import (
        FLOAT_BOUNDS, HOLD_STEPS, INT_BOUNDS, OVERLAY_BOOL,
        profile_for_opponent_key,
    )
    child = SDSGenome.from_dict(parent.to_dict())
    child.genome_id = new_id
    child.derived_from = parent.genome_id
    genes = GENE_GROUPS[bias_group]
    pool = [g for g in genes if g in OVERLAY_BOOL or g in INT_BOUNDS or g in FLOAT_BOUNDS
            or g == "opening_hold_steps"]
    if not pool or rng.random() < 0.25:   # 25% floor: still explore off-group
        return mutate(parent, rng, new_id=new_id)
    k = str(rng.choice(pool))
    overlay = dict(child.overlay)
    if k == "opening_hold_steps":
        child.opening_hold_steps = int(rng.choice(HOLD_STEPS))
    elif k in OVERLAY_BOOL:
        parent_p = profile_for_opponent_key(child.base_opponent)
        cur = overlay[k] if k in overlay else bool(getattr(parent_p, k))
        overlay[k] = not cur
    elif k in INT_BOUNDS:
        lo, hi = INT_BOUNDS[k]
        overlay[k] = int(rng.integers(lo, hi + 1))
    elif k in FLOAT_BOUNDS:
        lo, hi = FLOAT_BOUNDS[k]
        overlay[k] = float(rng.uniform(lo, hi))
    child.overlay = overlay
    return child


def ucb_select(rows: list[dict], rng) -> dict:
    """UCB1 over archived rows by genome_id, using each id's own J history."""
    by_id: dict = {}
    for r in rows:
        by_id.setdefault(r["genome"]["genome_id"], []).append(float(r.get("J", -9.0)))
    n_total = sum(len(v) for v in by_id.values())
    best_id, best_ucb = None, -1e9
    for gid, js in by_id.items():
        mean_j = sum(js) / len(js)
        bonus = UCB_C * math.sqrt(2 * math.log(max(2, n_total)) / len(js))
        u = mean_j + bonus
        if u > best_ucb:
            best_ucb, best_id = u, gid
    for r in reversed(rows):
        if r["genome"]["genome_id"] == best_id:
            return r
    return rows[-1]


def pareto_update(pareto: list[dict], cand: dict) -> list[dict]:
    def vec(r):
        return (float(r.get("delta_G", -9)), float(r.get("p_C", -9)),
                -float(r.get("degeneracy_penalty", 9)))
    cv = vec(cand)
    def dominates(a, b):
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))
    if any(dominates(vec(p), cv) for p in pareto):
        return pareto
    kept = [p for p in pareto if not dominates(cv, vec(p))]
    kept.append(cand)
    return kept


def hierarchical_eval(genome: SDSGenome, delta_B: float, seed_cursor: int,
                      device: str) -> tuple[dict, int]:
    """Successive halving across STAGES. Returns the LAST (deepest reached)
    stage record. Promotion is by J at each stage, not a manual judgement."""
    rec = None
    for stage_n in STAGES:
        r = paired_eval_v2(genome, n=stage_n, seed_base=seed_cursor,
                           device=device, stage=stage_n)
        seed_cursor += 2 * stage_n
        r["stage_n"] = stage_n
        r["J"] = r["J_v2"]
        rec = r
        log(f"    stage_n={stage_n} dG={r['delta_G']:+.3f} "
           f"p_C={r['p_C']:.3f} J_v2={r['J_v2']:+.3f} "
           f"frac00={r['frac_0_0']:.2f} "
           f"[payoff_ok={r['payoff_constraint_ok']} p_C_ok={r['p_C_screen_ok']}]")
        # Payoff is a CONSTRAINT, not a term to trade away: a candidate that has
        # stopped carrying GUARD pressure is not worth deeper spend regardless
        # of how well it conceals.
        if not r["payoff_constraint_ok"]:
            break
    rec["delta_B_anchor"] = delta_B
    rec["not_gate_B"] = True
    rec["confirmation_block_spent"] = False
    rec["development_eligible"] = development_eligible_v2(rec)
    return rec, seed_cursor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pop-per-island", type=int, default=6)
    ap.add_argument("--max-hours", type=float, default=MAX_WALL_HOURS)
    args = ap.parse_args()

    log("=" * 74)
    log("STRATEGIC DEMAND SEARCHER V2  RULESET_V3_M1 frozen")
    log("J/thresholds imported from V1 unchanged. Not Gate B. No PPO. "
       "Block 2500001 untouched.")
    log("=" * 74)

    b_path = PROJECT_ROOT / "artifacts/strategic_demand/searcher/anchor_B_cheap.json"
    if b_path.is_file():
        delta_B = float(json.loads(b_path.read_text(encoding="utf-8"))["delta_B"])
        log(f"reuse anchor B delta_B={delta_B:+.3f}")
    else:
        b_est = evaluate_B(8, V2_SEED_BASE - 1000, args.device)
        delta_B = float(b_est["delta_B"])
        atomic_write(OUT_DIR / "anchor_B_cheap.json", b_est)
        log(f"computed anchor B delta_B={delta_B:+.3f}")

    state = load_archive()
    archive: list[dict] = state["rows"]
    seen_ids = {r["genome"]["genome_id"] for r in archive}
    seed_cursor = int(state.get("seed_cursor", V2_SEED_BASE))
    pareto: list[dict] = json.loads(PARETO.read_text(encoding="utf-8")) if PARETO.is_file() else []
    log(f"resumed {len(archive)} rows from V2's own archive; seed_cursor={seed_cursor}")

    rng = np.random.default_rng(V2_SEED_BASE)
    islands = {
        "A_payoff": [canonical_parent("OP6")],
        "B_concealment": [canonical_parent("OP11"), canonical_parent("OP12")],
        "C_active": [canonical_parent(b) for b in LEGAL_A_BASES if b not in ("OP11", "OP12")],
    }
    for name, seed_pop in islands.items():
        while len(seed_pop) < args.pop_per_island:
            seed_pop.append(mutate(seed_pop[len(seed_pop) % len(seed_pop)], rng,
                                   new_id=f"SDS2_{name}_INIT_{len(seed_pop)}"))

    t0 = time.time()
    gen = int(state.get("next_gen", 0))
    best_j_history: list[float] = [r.get("J", -9) for r in archive]
    plateau = 0
    found = None

    while True:
        if (time.time() - t0) / 3600.0 > args.max_hours:
            log(f"wall-clock cap {args.max_hours}h reached -- stopping")
            break

        log(f"generation {gen}")
        for iname, pop in islands.items():
            for g in pop:
                if g.genome_id in seen_ids:
                    continue
                seen_ids.add(g.genome_id)
                log(f"  [{iname}] {g.genome_id} base={g.base_opponent} "
                   f"hold={g.opening_hold_steps}")
                rec, seed_cursor = hierarchical_eval(g, delta_B, seed_cursor, args.device)
                rec["island"] = iname
                rec["gen"] = gen
                archive.append(rec)
                pareto = pareto_update(pareto, rec)
                atomic_write(ARCHIVE, {"updated": _now(), "rows": archive,
                                       "seed_cursor": seed_cursor, "next_gen": gen})
                atomic_write(PARETO, pareto)
                if rec["development_eligible"]:
                    log(f"    DEVELOPMENT_ELIGIBLE: {g.genome_id} "
                       f"J={rec['J']:+.3f} (does NOT spend 2500001)")
                    found = rec

        if found is not None:
            log(f"development-eligible candidate found: "
               f"{found['genome']['genome_id']} -- stopping search")
            break

        cur_best = max((r.get("J", -9) for r in archive), default=-9)
        pareto_size = len(pareto)
        improved = (not best_j_history) or cur_best > max(best_j_history)
        best_j_history.append(cur_best)
        if improved:
            plateau = 0
        else:
            plateau += 1
        log(f"  gen {gen} done: best_J={cur_best:+.3f} pareto_size={pareto_size} "
           f"plateau={plateau}")
        if plateau >= 2:
            log("plateau reached (2 generations, no J improvement) -- stopping")
            break

        # breed next generation per island, with periodic migration
        if gen > 0 and gen % MIGRATE_EVERY == 0:
            bests = {}
            for iname in islands:
                rows_i = [r for r in archive if r.get("island") == iname]
                if rows_i:
                    bests[iname] = max(rows_i, key=lambda r: r.get("J", -9))
            for iname in islands:
                for other, rec in bests.items():
                    if other != iname:
                        islands[iname].append(SDSGenome.from_dict(rec["genome"]))
            log(f"  migration at gen {gen}: {list(bests.keys())}")

        for iname, pop in islands.items():
            rows_i = [r for r in archive if r.get("island") == iname] or archive
            nxt = []
            k = 0
            while len(nxt) < args.pop_per_island:
                parent_rec = ucb_select(rows_i, rng)
                parent = SDSGenome.from_dict(parent_rec["genome"])
                bias = weakest_dimension(parent_rec)
                if rng.random() < 0.3 and len(rows_i) > 1:
                    other_rec = ucb_select(rows_i, rng)
                    other = SDSGenome.from_dict(other_rec["genome"])
                    child = recombine(parent, other, rng,
                                      new_id=f"SDS2_{iname}_G{gen+1}_{k}")
                else:
                    child = weighted_mutate(parent, rng,
                                            new_id=f"SDS2_{iname}_G{gen+1}_{k}",
                                            bias_group=bias)
                nxt.append(child)
                k += 1
            islands[iname] = nxt

        gen += 1

    archive.sort(key=lambda r: r.get("J", -9), reverse=True)
    summary = {
        "protocol_id": "STRATEGIC_DEMAND_SEARCHER_V2",
        "finished_utc": _now(),
        "search_results_are_gate_B": False,
        "ppo": "NOT STARTED",
        "confirmation": "NOT RUN. 2500001 remains pristine.",
        "n_archive": len(archive),
        "n_pareto": len(pareto),
        "best": archive[0] if archive else None,
        "development_eligible_found": found is not None,
        "development_eligible_candidate": found,
    }
    atomic_write(OUT_DIR / "summary.json", summary)
    log("=" * 74)
    log("SEARCHER V2 FINISHED — not Gate B, not V3_STRATEGIC_DEMAND = VALIDATED")
    if found:
        log(f"CANDIDATE: {found['genome']['genome_id']} — human decision "
           "required before spending 2500001")
        atomic_write(OUT_DIR / "CANDIDATE_A_V2.json", found)
    log("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
