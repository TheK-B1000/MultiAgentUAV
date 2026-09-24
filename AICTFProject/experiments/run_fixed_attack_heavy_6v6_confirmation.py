"""FIXED_ATTACK_HEAVY_6V6_CONFIRMATORY_V1_SPEC.json -- contracts, then the confirmation.

One fixed composition, chosen mechanically from the sealed 6v6 sweep BEFORE any
fresh seed, against the balanced 3A/3D baseline, on both certified 6v6 poles,
paired fresh seeds, PPO off. Gates, unchanged from the 4v4 confirmation:

    B: LCB95( WR_B(selected) - WR_B(3A/3D) ) > 0
    A: UCB95( WR_A(3A/3D) - WR_A(selected) ) <= 0.10

The selected composition is not typed in. derive_selection() re-derives it from
the sealed sweep rows, and contract C1 requires it to equal the spec's frozen
value. Exactly two arms run: the selected composition and the 3A/3D baseline.

No router. No PPO. Sealing goes through run_state.seal with AuditPlan.experiment_id.

CLAIM BOUNDARY, frozen: the 6v6 Pole B is canonical SDS_PARENT_OP7, not the 4v4
B3-3 SDS2_B3_LOCKDEF10_2V1 construction. Cross-scale differences cannot be
attributed solely to team size.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pyquaticus_6v6_role_composition_sweep import (  # noqa: E402
    BASELINE,
    CERT_6V6,
    COMPOSITIONS as SWEEP_COMPOSITIONS,
    EPISODE_CSV as SWEEP_EPISODES_CSV,
    N_AGENTS,
    POLES,
    RESULT_PATH as SWEEP_RESULT_PATH,
    composition_roles_n,
    make_env_n,
    run_episode,
)
from experiments.run_routed_composition_outcome import (  # noqa: E402
    ALPHA,
    N_BOOT,
    PPO_FORBIDDEN_PREFIXES,
    RNG_SEED,
    TAU_A_HARM,
    _bootstrap,
)
from experiments.tqdm_loop import tqdm_iter  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "FIXED_ATTACK_HEAVY_6V6_CONFIRMATORY_V1_SPEC.json"
SWEEP_AUDIT_PATH = SD / "PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP_AUDIT.json"
CONTRACT_PATH = SD / "FIXED_ATTACK_HEAVY_6V6_CONTRACT_RESULT.json"
RESULT_PATH = SD / "FIXED_ATTACK_HEAVY_6V6_OUTCOME_RESULT.json"
EPISODE_CSV = SD / "FIXED_ATTACK_HEAVY_6V6_OUTCOME_EPISODES.csv"
PARTIAL = SD / "FIXED_ATTACK_HEAVY_6V6_OUTCOME_PARTIAL.jsonl"
LABEL = "FIXED_ATTACK_HEAVY_6V6_OUTCOME"
EXPERIMENT_ID = "FIXED_ATTACK_HEAVY_6V6_CONFIRMATION_V1"

# Frozen by the spec. C1 re-derives all three from the sealed sweep rows; C8 checks
# every constant here against the spec. A_ARGMAX and B_ARGMAX are the exploratory
# argmaxes, kept as the frozen DERIVATION RECORD only -- they are not run as arms.
SELECTED = "5A_1D"
A_ARGMAX = "4A_2D"
B_ARGMAX = "6A_0D"
ARMS = (BASELINE, SELECTED)
SEED_BASE, SEED_N = 20_600_001, 192
CONFIRM_SEEDS = list(range(SEED_BASE, SEED_BASE + SEED_N))
SEED_CLASS = "sealed_confirmatory"
PARITY_SEEDS = (20_500_001, 20_500_002, 20_500_003)      # spent sweep seeds, deterministic parity only
COMPARE_FIELDS = ("blue_score", "red_score", "blue_win", "draw", "steps", "attack_ticks", "defend_ticks",
                  "attack_enemy_flag_branch_count", "carrier_home_branch_count", "defend_inward_count",
                  "defend_outward_count", "tagged_ticks_by_role")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------- the selection rule

def _maximin(wr: dict[str, dict[str, float]], candidates: set[str]) -> str:
    """Highest worst-pole win rate among the candidates. No silent tie-break: an
    exact tie at the worst-pole win rate is a decision for the PI, not for code."""
    if not candidates:
        raise SystemExit("ABORT: no candidate composition to choose among")
    worst = {c: min(wr[p][c] for p in POLES) for c in candidates}
    top = max(worst.values())
    winners = sorted(c for c in candidates if abs(worst[c] - top) < 1e-12)
    if len(winners) != 1:
        raise SystemExit(f"ABORT: tie at the maximin worst-pole win rate {top:.6f} among {winners}; "
                         "the frozen rule does not break it, so a PI decision is required")
    return winners[0]


def derive_selection(sweep_rows_csv: Path = SWEEP_EPISODES_CSV) -> dict[str, Any]:
    """Re-derive the confirmed composition from the sealed sweep rows.

    1. per pole, the argmax composition by win rate (a tie at the argmax aborts)
    2. per pole, the set of compositions whose paired win-rate contrast against the
       argmax has a 95% interval that CONTAINS zero (the indistinguishable set)
    3. intersect the two poles' {argmax} + indistinguishable sets; empty aborts
    4. choose the member with the highest WORST-POLE win rate (maximin)

    Uses the same paired percentile bootstrap (20000, seed 7) as the sealed sweep and
    the pre-declared addendum, so it reproduces those numbers.
    """
    rows: dict[tuple[str, str, int], int] = {}
    with Path(sweep_rows_csv).open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[(r["pole"], r["composition"], int(r["seed"]))] = int(r["blue_win"])
    if not rows:
        raise SystemExit("ABORT: sweep rows missing -- absence is an error state")
    seeds = sorted({k[2] for k in rows})
    comps = sorted({k[1] for k in rows}, key=lambda c: -int(c.split("_")[0][:-1]))
    wr = {p: {c: float(np.mean([rows[(p, c, s)] for s in seeds])) for c in comps} for p in POLES}

    def paired(p: str, left: str, right: str) -> np.ndarray:
        return np.asarray([rows[(p, left, s)] - rows[(p, right, s)] for s in seeds], dtype=float)

    best_set: dict[str, set[str]] = {}
    argmax: dict[str, str] = {}
    indist: dict[str, list[str]] = {}
    for p in POLES:
        top = max(wr[p].values())
        tied = [c for c in comps if abs(wr[p][c] - top) < 1e-12]
        if len(tied) != 1:
            raise SystemExit(f"ABORT: tie at the pole-{p} argmax among {tied}")
        argmax[p] = tied[0]
        indist[p] = [c for c in comps if c != argmax[p]
                     and not (lambda b: b["lcb95"] > 0 or b["ucb95"] < 0)(_bootstrap(paired(p, argmax[p], c)))]
        best_set[p] = {argmax[p], *indist[p]}
    inter = best_set["A"] & best_set["B"]
    if not inter:
        raise SystemExit("ABORT: the two poles' best sets do not intersect; no single fixed "
                         "composition is compatible with both, so this confirmation has no candidate")
    selected = _maximin(wr, inter)
    mean_across = {c: float(np.mean([wr[p][c] for p in POLES])) for c in inter}
    return {
        "selected": selected,
        "A_argmax": argmax["A"], "B_argmax": argmax["B"],
        "indistinguishable": {p: sorted(indist[p]) for p in POLES},
        "intersection": sorted(inter),
        "worst_pole_win_rate": {c: min(wr[p][c] for p in POLES) for c in sorted(inter)},
        "mean_across_poles_win_rate": {c: mean_across[c] for c in sorted(inter)},
        "win_rates": {p: {c: round(wr[p][c], 6) for c in comps} for p in POLES},
        "n_seeds": len(seeds),
    }


# ------------------------------------------------- the frozen interpretation

def decide(b_gain: dict, a_harm: dict, *, integrity_ok: bool = True,
           tau: float = TAU_A_HARM) -> dict[str, Any]:
    """The frozen terminal interpretation as a pure function, so it is tested and not
    re-argued after the result.

    Integrity takes precedence over every scientific label. Otherwise:
      B fails                         -> B_NOT_REPLICATED; no advantage demonstrated, regardless of A
      B passes, UCB95(H_A) <= tau     -> CONFIRMED (A safety demonstrated under the preregistered criterion)
      B passes, LCB95(H_A) > 0        -> A_HARM_DEMONSTRATED; the only branch that reopens routing
      B passes, else                  -> A_SAFETY_NOT_DEMONSTRATED; inconclusive, NOT evidence the plateau
                                         was fake, and does not reopen routing on its own
    """
    base = {"promotion": False, "routing_reopen_licensed": False, "note": None}
    if not integrity_ok:
        return {**base, "label": "FIXED_ATTACK_HEAVY_6V6_ARM_IDENTITY_VIOLATED",
                "note": "An arm did not run the composition it claims. Invalid; no scientific label is issued."}
    pass_b = b_gain["lcb95"] > 0
    pass_a = a_harm["ucb95"] <= tau
    if not pass_b:
        return {**base, "label": "FIXED_ATTACK_HEAVY_6V6_B_NOT_REPLICATED",
                "note": "No demonstrated advantage from the selected composition, regardless of A."}
    if pass_a:
        note = ("A harm is positive with its lower bound above zero but inside the tolerance; A safety is "
                "demonstrated under the preregistered criterion." if a_harm["lcb95"] > 0 else None)
        return {**base, "label": "FIXED_ATTACK_HEAVY_6V6_CONFIRMED", "promotion": True, "note": note}
    if a_harm["lcb95"] > 0:
        return {**base, "label": "FIXED_ATTACK_HEAVY_6V6_A_HARM_DEMONSTRATED", "routing_reopen_licensed": True,
                "note": "Demonstrated A harm is evidence against the fixed replacement and can reopen the "
                        "routed-composition question."}
    return {**base, "label": "FIXED_ATTACK_HEAVY_6V6_A_SAFETY_NOT_DEMONSTRATED",
            "note": "Promising but inconclusive on A. Not evidence the plateau was fake; do not reopen routing on "
                    "this basis alone. Any larger follow-up needs a NEW freeze and FRESH seeds. No automatic "
                    "top-up of this block."}


def arm_identity_violations(rows: dict[tuple[str, str, int], dict]) -> list[str]:
    """Every episode must have run the composition its row claims. The role telemetry is
    incremented once per agent per tick, so defend_ticks == defenders * steps and
    attack_ticks == attackers * steps, and a fixed composition never switches roles."""
    bad = []
    for (pole, comp, seed), r in sorted(rows.items()):
        d = int(comp.split("_")[1][:-1])
        a = N_AGENTS - d
        steps = int(r["steps"])
        if (int(r["defend_ticks"]) != d * steps or int(r["attack_ticks"]) != a * steps
                or int(r["role_switch_count"]) != 0):
            bad.append(f"{pole}/{comp}/{seed}")
    return bad


# ------------------------------------------------------------------ audit plan

def _paired(rows: dict, pole: str, left: str, right: str, seeds: list[int]) -> np.ndarray:
    return np.asarray([float(rows[(pole, left, s)]["blue_win"]) - float(rows[(pole, right, s)]["blue_win"])
                       for s in seeds])


def build_audit_plan(rows_csv: Path, recorded: dict[str, dict]):
    """The plan the real run seals with, factored out so contract C7 exercises the
    SAME object. recorded maps claim name -> {mean, lcb95, ucb95}."""
    from experiments import run_state as rs
    total = len(POLES) * len(ARMS) * SEED_N
    claims = (
        rs.Claim(name="B_gain_selected_minus_baseline_pole_B", recorded=recorded["B_gain"],
                 minuend={"pole": "B", "composition": SELECTED},
                 subtrahend={"pole": "B", "composition": BASELINE}, value_field="blue_win"),
        rs.Claim(name="A_harm_baseline_minus_selected_pole_A", recorded=recorded["A_harm"],
                 minuend={"pole": "A", "composition": BASELINE},
                 subtrahend={"pole": "A", "composition": SELECTED}, value_field="blue_win"),
    )
    return rs.AuditPlan(
        rows_csv=rows_csv, expected_rows=total, expected_seeds=CONFIRM_SEEDS,
        group_by=("pole", "composition"), seed_field="seed",
        int_fields=("seed", "blue_score", "red_score", "blue_win", "draw", "steps"),
        binary_fields=("blue_win", "draw"),
        derived={"blue_win": rs.Derived("blue_win == int(blue_score > red_score)",
                                        lambda r: int(int(r["blue_score"]) > int(r["red_score"]))),
                 "draw": rs.Derived("draw == int(blue_score == red_score)",
                                    lambda r: int(int(r["blue_score"]) == int(r["red_score"])))},
        spec_path=SPEC_PATH, seed_class=SEED_CLASS,
        # load-bearing: without the owner the audit reads this run's own Rule-9
        # reservation as foreign reuse (ROUTED_COMPOSITION_OUTCOME_AUDIT_CORRECTION.json)
        experiment_id=EXPERIMENT_ID,
        n_boot=N_BOOT, alpha=ALPHA, rng_seed=RNG_SEED, claims=claims,
    )


# ------------------------------------------------------------------- contracts

def run_contracts() -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str, **data: Any) -> None:
        checks.append({"name": name, "gating": True, "passed": bool(passed), "detail": detail, **data})

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    record("C0_SPEC_FROZEN", str(spec.get("status", "")).startswith("FROZEN"),
           f"spec status = {spec.get('status')!r}")

    # C1 -- the selection is a function of the sealed sweep, and equals the frozen value
    sweep = json.loads(SWEEP_RESULT_PATH.read_text(encoding="utf-8"))
    sweep_audit = json.loads(SWEEP_AUDIT_PATH.read_text(encoding="utf-8"))
    sealed_ok = sweep.get("status") == "SEALED" and bool(sweep_audit.get("passed"))
    sel = derive_selection()
    frozen = spec["THE_FROZEN_SELECTION_RULE"]["derived_and_frozen"]
    same = (sel["selected"] == SELECTED == frozen["selected"]
            and sel["A_argmax"] == A_ARGMAX == frozen["A_argmax"]
            and sel["B_argmax"] == B_ARGMAX == frozen["B_argmax"]
            and sel["intersection"] == sorted(frozen["intersection"]))
    record("C1_SELECTION_REPRODUCES_FROM_THE_SEALED_SWEEP", sealed_ok and same,
           f"sweep status={sweep.get('status')!r}, audit passed={sweep_audit.get('passed')}; re-derived "
           f"selected={sel['selected']!r} A_argmax={sel['A_argmax']!r} B_argmax={sel['B_argmax']!r} "
           f"intersection={sel['intersection']} worst-pole={sel['worst_pole_win_rate']} "
           f"mean-across={sel['mean_across_poles_win_rate']}; frozen values agree: {same}")

    # C4 -- role mapping for the four arms at n=6
    exp = {"3A_3D": (1, 1, 1, 0, 0, 0), "5A_1D": (1, 0, 0, 0, 0, 0)}
    bad = [c for c in ARMS if composition_roles_n(c, N_AGENTS) != exp[c]]
    record("C4_ROLE_MAPPING", not bad and set(ARMS) == set(exp),
           f"two arms mapped under identity-prefix at n=6; mismatches: {bad or 'none'}")

    # C2 -- both poles resolve as certified at team size six
    cert = json.loads(CERT_6V6.read_text(encoding="utf-8"))
    pole_bad, shapes = [], {}
    for pole in POLES:
        env, core, _g, live = make_env_n(pole, 99_900_613)
        try:
            shapes[pole] = [int(core.blue_x.shape[1]), int(core.red_x.shape[1])]
            want = cert["poles"][pole]
            for f in ("defender_zone_frac", "threat_radius"):
                if abs(float(live.get(f, float("nan"))) - float(want[f])) > 1e-6:
                    pole_bad.append(f"{pole}.{f}")
            if int(round(float(live.get("min_alive_for_defender", -1)))) != int(want["overlay"]["min_alive_for_defender"]):
                pole_bad.append(f"{pole}.min_alive_for_defender")
        finally:
            env.close()
    record("C2_POLES_RESOLVE_AS_CERTIFIED",
           not pole_bad and all(v == [N_AGENTS, N_AGENTS] for v in shapes.values()),
           f"live agent counts {shapes}; disagreements with {CERT_6V6.name}: {pole_bad or 'none'}")

    # C3 -- parity anchor: this executor's path reproduces sealed sweep rows exactly
    sealed: dict[tuple[str, str, int], dict] = {}
    with SWEEP_EPISODES_CSV.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sealed[(r["pole"], r["composition"], int(r["seed"]))] = r
    diffs, n_cmp, wins, parity_rows = [], 0, [], {}
    for pole in POLES:
        for comp in ARMS:
            for seed in PARITY_SEEDS:
                got = run_episode(pole, comp, seed)
                parity_rows[(pole, comp, seed)] = got
                ref = sealed[(pole, comp, seed)]
                n_cmp += 1
                wins.append(int(got["blue_win"]))
                diffs += [f"{pole}/{comp}/{seed}/{f}" for f in COMPARE_FIELDS
                          if str(got[f]) != str(ref[f])]
    identity_bad = arm_identity_violations(parity_rows)
    record("C3_PARITY_ANCHOR_AGAINST_THE_SEALED_SWEEP",
           n_cmp == len(POLES) * len(ARMS) * len(PARITY_SEEDS) and not diffs and 0 < sum(wins) < len(wins)
           and not identity_bad,
           f"{n_cmp} spent-sweep-seed episodes re-run through this executor's runner and compared on "
           f"{len(COMPARE_FIELDS)} fields; disagreements: {diffs or 'none'}; win rate on the subset "
           f"{sum(wins)}/{len(wins)} (non-degenerate, so terminal scoring is not reading reset state); "
           f"arm-identity invariant violations on those rows: {identity_bad or 'none'}")

    # C5 -- PPO unreachable
    live_ppo = sorted(m for m in sys.modules if m.startswith(PPO_FORBIDDEN_PREFIXES))
    record("C5_NO_PPO_REACHABLE", not live_ppo, f"imported PPO/trainer modules: {live_ppo or 'none'}")

    # C6 -- the block is registered to THIS experiment, exact range, unspent
    from experiments import seed_registry as sr
    ok, msg = sr.check_block(SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS, experiment_id=EXPERIMENT_ID)
    entry = next((b for b in sr.load()["blocks"] if b["experiment_id"] == EXPERIMENT_ID), None)
    record("C6_SEED_BLOCK_OWNED_AND_UNSPENT",
           ok and entry is not None and entry["status"] == "RESERVED"
           and entry["lo"] == SEED_BASE and entry["hi"] == SEED_BASE + SEED_N - 1,
           f"{msg}; registry status = {entry['status'] if entry else None}")

    # C7 -- the REAL audit plan seals a block already registered to its owner, run AFTER
    # registration, with a negative control (owner omitted must fail exactly seed_class).
    import tempfile
    from experiments import run_state as rs
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp()) / "rows.csv"
    rows, byk = [], {}
    for pole in POLES:
        for comp in ARMS:
            for s in CONFIRM_SEEDS:
                bs, rd = int(rng.integers(0, 4)), int(rng.integers(0, 4))
                r = {"seed": s, "pole": pole, "composition": comp, "blue_score": bs, "red_score": rd,
                     "blue_win": int(bs > rd), "draw": int(bs == rd), "steps": 240}
                rows.append(r)
                byk[(pole, comp, s)] = r
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rec = {"B_gain": _bootstrap(_paired(byk, "B", SELECTED, BASELINE, CONFIRM_SEEDS)),
           "A_harm": _bootstrap(_paired(byk, "A", BASELINE, SELECTED, CONFIRM_SEEDS))}
    plan = build_audit_plan(tmp, rec)
    with_owner = rs.run_audit(plan)
    no_owner = rs.run_audit(rs.AuditPlan(**{**plan.__dict__, "experiment_id": None}))
    failed = sorted(c["name"] for c in no_owner["checks"] if c["result"] == "FAIL")
    record("C7_AUDIT_PLAN_SEALS_A_REGISTERED_BLOCK", with_owner["passed"] and failed == ["seed_class"],
           f"real AuditPlan with owner on {len(rows)} synthetic rows: passed={with_owner['passed']} "
           f"({with_owner['n_gating']} gating, {with_owner['n_failed']} failed); negative control without "
           f"the owner fails exactly {failed}")

    # C8 -- executor constants equal the spec
    g = spec["PRIMARY_JOINT_GATE_UNCHANGED_FROM_4V4_STYLE"]
    d = spec["ARMS_AND_DESIGN"]
    ok8 = (float(g["tau_A_harm"]) == float(TAU_A_HARM)
           and int(d["seed_block"]["base"]) == SEED_BASE and int(d["seed_block"]["n"]) == SEED_N
           and list(d["arms"]) == list(ARMS) and d["baseline"] == BASELINE
           and int(g["bootstrap"]["n_boot"]) == N_BOOT and int(g["bootstrap"]["rng_seed"]) == RNG_SEED
           and float(g["bootstrap"]["alpha"]) == float(ALPHA))
    record("C8_EXECUTOR_CONSTANTS_EQUAL_THE_SPEC", ok8,
           f"tau {TAU_A_HARM} block {SEED_BASE}+{SEED_N} arms {list(ARMS)} baseline {BASELINE} "
           f"bootstrap {N_BOOT}/{RNG_SEED}/{ALPHA} vs spec")

    n_failed = sum(1 for c in checks if not c["passed"])
    report = {
        "record_id": "FIXED_ATTACK_HEAVY_6V6_CONTRACT_RESULT", "utc": _now(), "implements": SPEC_PATH.name,
        "selection": {k: sel[k] for k in ("selected", "A_argmax", "B_argmax", "intersection",
                                          "worst_pole_win_rate", "mean_across_poles_win_rate")},
        "n_checks": len(checks), "n_gating": len(checks), "n_failed": n_failed, "checks": checks,
        "DECISION": "CONTRACTS_PASS" if n_failed == 0 else "CONTRACT_FAILURE",
        "claim_boundary": "Contracts only. No confirmatory episode was run and no seed was spent. C3 reuses "
                          "already-spent sweep seeds for deterministic parity only.",
    }
    CONTRACT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'=' * 66}\n  CONTRACTS: {report['DECISION']}  ({n_failed}/{len(checks)} gating failed)\n{'=' * 66}")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"  -> {CONTRACT_PATH}")
    return report


# --------------------------------------------------------------------- outcome

def run_outcome() -> int:
    from experiments import run_state as rs
    from experiments import seed_registry as sr

    contracts = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contracts.get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit(f"REFUSING: contracts did not pass ({contracts.get('DECISION')!r})")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")
    entry = next((b for b in sr.load()["blocks"] if b["experiment_id"] == EXPERIMENT_ID), None)
    if entry is None or entry["status"] != "RESERVED" or entry["lo"] != SEED_BASE \
            or entry["hi"] != SEED_BASE + SEED_N - 1:
        raise SystemExit(f"REFUSING: seed block not reserved to {EXPERIMENT_ID} as frozen: {entry}")

    total = len(POLES) * len(ARMS) * SEED_N
    state = rs.RunState(SD, LABEL).begin(spec=SPEC_PATH.name, n_episodes=total)
    rows: dict[tuple[str, str, int], dict] = {}
    if PARTIAL.is_file():
        for line in PARTIAL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["pole"], r["composition"], int(r["seed"]))] = r
        print(f"  resuming: {len(rows)}/{total} episodes already recorded", flush=True)
    pending = [(p, c, s) for p in POLES for c in ARMS for s in CONFIRM_SEEDS if (p, c, s) not in rows]
    with PARTIAL.open("a", encoding="utf-8") as fh:
        for pole, comp, seed in tqdm_iter(pending, desc="6v6 fixed-composition confirmation",
                                          total=len(pending), unit="ep"):
            row = run_episode(pole, comp, seed)
            rows[(pole, comp, seed)] = row
            fh.write(json.dumps(row) + "\n")
            fh.flush()

    fields = list(next(iter(rows.values())).keys())
    with EPISODE_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])

    seeds = CONFIRM_SEEDS
    b_gain = _bootstrap(_paired(rows, "B", SELECTED, BASELINE, seeds))
    a_harm = _bootstrap(_paired(rows, "A", BASELINE, SELECTED, seeds))
    pass_b = b_gain["lcb95"] > 0
    pass_a = a_harm["ucb95"] <= TAU_A_HARM
    identity_bad = arm_identity_violations(rows)
    verdict = decide(b_gain, a_harm, integrity_ok=not identity_bad)
    decision = verdict["label"]

    def flips(pole: str, left: str, right: str) -> dict:
        d = _paired(rows, pole, left, right, seeds)
        return {"left_win_right_not": int((d > 0).sum()), "right_win_left_not": int((d < 0).sum()),
                "discordant_fraction": round(float(((d != 0).sum()) / len(d)), 4)}

    win_rates = {p: {c: round(float(np.mean([rows[(p, c, s)]["blue_win"] for s in seeds])), 6) for c in ARMS}
                 for p in POLES}
    margins = {p: {c: round(float(np.mean([rows[(p, c, s)]["blue_score"] - rows[(p, c, s)]["red_score"]
                                           for s in seeds])), 4) for c in ARMS} for p in POLES}
    sweep = json.loads(SWEEP_RESULT_PATH.read_text(encoding="utf-8"))["composition_contrasts_vs_baseline"]

    payload = {
        "record_id": "FIXED_ATTACK_HEAVY_6V6_OUTCOME_RESULT", "implements": SPEC_PATH.name, "utc": _now(),
        "poles": list(POLES), "arms": list(ARMS), "baseline": BASELINE, "selected": SELECTED,
        "seed_block": {"base": SEED_BASE, "last": SEED_BASE + SEED_N - 1, "n": SEED_N, "class": SEED_CLASS},
        "contracts": {"path": CONTRACT_PATH.name, "decision": contracts["DECISION"],
                      "n_gating": contracts["n_gating"], "n_failed": contracts["n_failed"]},
        "primary_gates": {
            "B_gain_selected_minus_baseline_on_pole_B": b_gain, "B_gate_LCB95>0": pass_b,
            "A_harm_baseline_minus_selected_on_pole_A": a_harm, "tau_A_harm": TAU_A_HARM,
            "A_gate_UCB95<=tau": pass_a,
        },
        "interpretation": {**verdict, "arm_identity_violations": identity_bad,
                           "label_is_quotable_only_if_the_record_status_is_SEALED": True},
        "discordant_pairs": {
            "B_selected_vs_baseline": flips("B", SELECTED, BASELINE),
            "A_baseline_vs_selected": flips("A", BASELINE, SELECTED),
            "note": "The A gate's power depends on discordance. At about 40% discordant seeds, UCB95 <= 0.10 "
                    "requires the true harm to be slightly negative, so a failure is weak evidence of harm.",
        },
        "win_rates": win_rates, "score_margin_blue_minus_red": margins,
        "descriptive_not_gating": {
            "not_measured_here": ("Whether per-pole routing could add value beyond the selected composition. "
                                  "Only the two arms were run; the exploratory sweep is the only evidence on that."),
            "comparison_to_exploratory_sweep_not_a_test": {
                "sweep_B_selected_minus_baseline": sweep["B"][SELECTED],
                "sweep_A_selected_minus_baseline": sweep["A"][SELECTED],
                "confirmatory_B": b_gain["mean"], "confirmatory_A_selected_minus_baseline": -a_harm["mean"],
            },
        },
        "DECISION": decision,
        "FROZEN_CLAIM_BOUNDARY_POLE_B": (
            "The 6v6 composition runs use the certified 6v6 pole pair. Pole B is canonical SDS_PARENT_OP7, "
            "not the 4v4 B3-3 SDS2_B3_LOCKDEF10_2V1 construction. Therefore cross-scale differences cannot be "
            "attributed solely to team size."),
        "claim_boundary": ("Two certified 6v6 poles, fixed scripted composition, PPO off, n=128 paired seeds. "
                           "Does not authorize a router, PPO, or any statement about other opponents."),
    }
    plan = build_audit_plan(EPISODE_CSV, {"B_gain": b_gain, "A_harm": a_harm})
    rs.seal(out_path=RESULT_PATH, payload=payload, plan=plan, state=state, strict=False)
    sealed = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    sr.set_status(EXPERIMENT_ID, "SPENT", note=f"sealed {sealed.get('status')}; {decision}")
    print(json.dumps({"DECISION": decision, "status": sealed.get("status"), "B_gain": b_gain, "A_harm": a_harm,
                      "interpretation": verdict, "win_rates": win_rates}, indent=2))
    return 0 if decision == "FIXED_ATTACK_HEAVY_6V6_CONFIRMED" else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts", "outcome"), required=True)
    args = ap.parse_args()
    if args.stage == "contracts":
        return 0 if run_contracts()["DECISION"] == "CONTRACTS_PASS" else 2
    return run_outcome()


if __name__ == "__main__":
    raise SystemExit(main())
