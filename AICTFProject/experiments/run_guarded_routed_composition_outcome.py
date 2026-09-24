"""GUARDED_ROUTED_COMPOSITION_CONFIRMATORY_V1_SPEC.json -- contracts, then the outcome arms.

Three arms on 128 fresh paired seeds, both poles: STATE_GUARDED (the frozen V2
router plus the frozen full-window startup guard), FIXED_2A2D (A-safe baseline
and the control for both primary gates) and FIXED_4A0D (descriptive reference).

Nothing about the router or the guard is re-implemented here. OnlineRouter comes
from run_routed_composition_outcome and GuardedRouter from size_routed_startup_guard
(the class that was sized). The FIXED arms use the same validated sweep runner as
every earlier run.

PPO is off and unreachable. Sealing goes through experiments/run_state.py::seal with
AuditPlan.experiment_id supplied. The seed block is reserved at freeze time; this
module verifies ownership and never allocates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pyquaticus_4v4_role_composition_sweep import (  # noqa: E402
    _action_for_roles,
    _ids,
    composition_roles,
    run_episode as run_fixed_episode,
)
from experiments.run_pyquaticus_4v4_team_evaluation import (  # noqa: E402
    HORIZON,
    _make_env,
    _telemetry_for_tick,
)
from experiments.run_routed_composition_outcome import (  # noqa: E402
    DEFAULT_COMPOSITION,
    HYSTERESIS,
    N_BOOT,
    ALPHA,
    ORACLE_PARITY_SEEDS,
    ORACLE_ROWS,
    POLES,
    PPO_FORBIDDEN_PREFIXES,
    RNG_SEED,
    TAU_A_HARM,
    THRESHOLD,
    TRIGGERED_COMPOSITION,
    V2_RESULT,
    V2_ROWS,
    WINDOW,
    DWELL,
    OnlineRouter,
    _bootstrap,
    _offline_sequence,
    _online_sequence,
    observe_d,
)
from experiments.localize_routed_a_harm import _runs_of_ones  # noqa: E402
from experiments.size_routed_startup_guard import GuardedRouter  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "GUARDED_ROUTED_COMPOSITION_CONFIRMATORY_V1_SPEC.json"
SIZING_ROWS = SD / "routed_composition_startup_guard_sizing_episodes.csv"
UNGUARDED_ROWS = SD / "ROUTED_COMPOSITION_OUTCOME_EPISODES.csv"
SIZING_READING = SD / "ROUTED_COMPOSITION_STARTUP_GUARD_SIZING_READING.json"
CONTRACT_PATH = SD / "GUARDED_ROUTED_COMPOSITION_OUTCOME_CONTRACT_RESULT.json"
RESULT_PATH = SD / "GUARDED_ROUTED_COMPOSITION_OUTCOME_RESULT.json"
EPISODE_CSV = SD / "GUARDED_ROUTED_COMPOSITION_OUTCOME_EPISODES.csv"
DIAG_CSV = SD / "GUARDED_ROUTED_COMPOSITION_OUTCOME_GUARD_DIAGNOSTICS.csv"
PARTIAL = SD / "GUARDED_ROUTED_COMPOSITION_OUTCOME_PARTIAL.jsonl"
LABEL = "GUARDED_ROUTED_COMPOSITION_OUTCOME"
EXPERIMENT_ID = "GUARDED_ROUTED_COMPOSITION_OUTCOME_V1"

ARMS = ("STATE_GUARDED", "FIXED_2A2D", "FIXED_4A0D")
FIXED_COMPOSITION = {"FIXED_2A2D": "2A_2D", "FIXED_4A0D": "4A_0D"}
SEED_BASE, SEED_N = 20_300_001, 128
OUTCOME_SEEDS = list(range(SEED_BASE, SEED_BASE + SEED_N))
SEED_CLASS = "sealed_confirmatory"
GUARD_TICK = WINDOW - 1          # first 0-indexed tick at which the buffer holds W samples
OUTCOME_FIELDS = ("blue_score", "red_score", "blue_win", "steps", "role_switch_count")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------ guarded episode

def run_guarded_routed_episode(pole: str, seed: int, *, guard: bool = True,
                               pinned: str | None = None,
                               label: str = "ROUTED_GUARDED") -> tuple[dict, dict]:
    """Mirrors run_routed_composition_outcome.run_routed_episode with the router
    swapped for GuardedRouter. With pinned set it must reproduce the sweep runner
    bit-for-bit (C5a); with guard=False it must reproduce the unguarded rows (C5b);
    with guard=True it must reproduce the sizing rows (C5c)."""
    env, core, genome, live = _make_env(pole, seed)
    try:
        router = GuardedRouter(guard=guard, pinned=pinned)
        counters = {
            "attack_ticks": 0, "defend_ticks": 0,
            "attack_enemy_flag_branch_count": 0, "carrier_home_branch_count": 0,
            "defend_inward_count": 0, "defend_outward_count": 0,
            "tagged_ticks_by_role": 0,
        }
        flags: list[int] = []
        first_roles: tuple[int, ...] | None = None
        terminal_info, steps = None, 0
        for _ in range(HORIZON):
            composition = router.update(observe_d(core))
            roles = composition_roles(composition)
            if first_roles is None:
                first_roles = roles
            flags.append(int(composition == TRIGGERED_COMPOSITION))
            _telemetry_for_tick(core, roles, counters)
            env.step_async(_action_for_roles(core, roles))
            _obs, _rew, done, infos = env.step_wait()
            steps += 1
            if bool(np.asarray(done).any()):
                terminal_info = dict(infos[0])
                break
        if terminal_info is None:
            terminal_info = {"episode_result": {"blue_score": int(core.blue_score[0].item()),
                                                "red_score": int(core.red_score[0].item())},
                             "terminal_observation": {}}
        result = dict(terminal_info.get("episode_result") or {})
        blue_score = int(result.get("blue_score", 0))
        red_score = int(result.get("red_score", 0))
        agent_mask = (terminal_info.get("terminal_observation") or {}).get("agent_mask")
        blue_alive_end = int(np.asarray(agent_mask).sum()) if agent_mask is not None else None
        attackers, defenders = _ids(first_roles or composition_roles(DEFAULT_COMPOSITION))
        bursts = _runs_of_ones(flags)
        row = {
            "seed": int(seed), "pole": pole, "composition": label,
            "defender_ids": ",".join(map(str, defenders)),
            "attacker_ids": ",".join(map(str, attackers)),
            "blue_score": blue_score, "red_score": red_score,
            "blue_win": int(blue_score > red_score), "draw": int(blue_score == red_score),
            "steps": int(steps), "blue_alive_end": blue_alive_end, "red_alive_end": None,
            "genome_id": str(genome.genome_id),
            "pole_config_hash": str(live.get("live_config_hash", "")),
            **counters, "role_switch_count": int(router.switches),
        }
        diag = {
            "pole": pole, "seed": int(seed),
            "blocked_departures": int(router.blocked_departures),
            "first_block_tick": router.first_block_tick,
            "n_bursts": len(bursts),
            "first_onset": (bursts[0][0] if bursts else None),
            "ticks_in_4A0D": int(sum(flags)),
            "fraction_in_4A0D": float(sum(flags)) / max(steps, 1),
        }
        return row, diag
    finally:
        env.close()


def run_job(pole: str, arm: str, seed: int) -> tuple[dict, dict]:
    if arm == "STATE_GUARDED":
        row, diag = run_guarded_routed_episode(pole, seed, guard=True, label="ROUTED_GUARDED")
    else:
        composition = FIXED_COMPOSITION[arm]
        row, _ = run_fixed_episode(pole, composition, seed)
        diag = {"pole": pole, "seed": int(seed), "blocked_departures": 0,
                "first_block_tick": None, "n_bursts": 0, "first_onset": None,
                "ticks_in_4A0D": (0 if composition == DEFAULT_COMPOSITION else int(row["steps"])),
                "fraction_in_4A0D": (0.0 if composition == DEFAULT_COMPOSITION else 1.0)}
    row["arm"] = arm
    diag["arm"] = arm
    return row, diag


# ------------------------------------------------------------------ audit plan

def _paired(rows: dict, pole: str, left: str, right: str, seeds: list[int],
            field: str = "blue_win") -> np.ndarray:
    return np.asarray([float(rows[(pole, left, s)][field]) - float(rows[(pole, right, s)][field])
                       for s in seeds])


def build_audit_plan(rows_csv: Path, recorded_b: dict, recorded_a: dict):
    """The plan the real run seals with. Factored out so contract C12 exercises
    the SAME object rather than a lookalike."""
    from experiments import run_state as rs
    total = len(POLES) * len(ARMS) * SEED_N
    return rs.AuditPlan(
        rows_csv=rows_csv, expected_rows=total, expected_seeds=OUTCOME_SEEDS,
        # one audit CELL is (pole, arm); each must carry the whole frozen seed block once
        group_by=("pole", "arm"), seed_field="seed",
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
        n_boot=N_BOOT, alpha=ALPHA, rng_seed=RNG_SEED,
        claims=(
            rs.Claim(name="B_improvement_STATE_GUARDED_minus_FIXED_2A2D_pole_B", recorded=recorded_b,
                     minuend={"pole": "B", "arm": "STATE_GUARDED"},
                     subtrahend={"pole": "B", "arm": "FIXED_2A2D"}, value_field="blue_win"),
            rs.Claim(name="A_harm_FIXED_2A2D_minus_STATE_GUARDED_pole_A", recorded=recorded_a,
                     minuend={"pole": "A", "arm": "FIXED_2A2D"},
                     subtrahend={"pole": "A", "arm": "STATE_GUARDED"}, value_field="blue_win"),
        ),
    )


# ------------------------------------------------------------------- contracts

def _load_csv(path: Path, key) -> dict:
    out: dict = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[key(row)] = row
    return out


def run_contracts() -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str, **data: Any) -> None:
        checks.append({"name": name, "gating": True, "passed": bool(passed), "detail": detail, **data})

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    record("C0_SPEC_FROZEN", str(spec.get("status", "")).startswith("FROZEN"),
           f"spec status = {spec.get('status')!r}")

    # C6 -- the operating point is the sealed one, byte for byte
    v2 = json.loads(V2_RESULT.read_text(encoding="utf-8"))
    sel = v2.get("selected_config") or {}
    ok = (v2.get("DECISION") == "CONSERVATIVE_B_TRIGGER_CALIBRATED_V2"
          and sel.get("window") == WINDOW and sel.get("hysteresis") == HYSTERESIS
          and sel.get("dwell") == DWELL and sel.get("threshold") == THRESHOLD)
    record("C6_CALIBRATION_PASS_SEALED", ok,
           f"V2 DECISION={v2.get('DECISION')!r}; selected=W{sel.get('window')} m{sel.get('hysteresis')} "
           f"d{sel.get('dwell')} thr{sel.get('threshold')!r}; executor=W{WINDOW} m{HYSTERESIS} "
           f"d{DWELL} thr{THRESHOLD!r}")

    # C1 -- no regime leak
    import inspect
    rng = np.random.default_rng(11)
    probe = list(rng.normal(-0.5, 1.5, 240))

    def guarded_seq(vals: list[float], guard: bool = True) -> tuple[list[int], GuardedRouter]:
        r = GuardedRouter(guard=guard)
        return [1 if r.update(v) == TRIGGERED_COMPOSITION else 0 for v in vals], r

    a, _ = guarded_seq(probe)
    b, _ = guarded_seq(probe)
    params = set(inspect.signature(GuardedRouter.update).parameters) - {"self"}
    leaky = {"pole", "genome", "opponent", "phase", "win", "score", "reward", "return"}
    record("C1_NO_REGIME_LEAK", a == b and params == {"d_value"} and not (params & leaky),
           f"identical features -> identical compositions ({a == b}); update() parameters = {sorted(params)}")

    # C2 -- default at tick 0 and under above-trigger evidence
    r0 = GuardedRouter(guard=True)
    first = r0.update(0.0)
    high, _ = guarded_seq([2.0] * 240)
    record("C2_DEFAULT_IS_2A2D", first == DEFAULT_COMPOSITION and sum(high) == 0,
           f"tick0 composition = {first}; never departs on high-D evidence (sum={sum(high)})")

    # C3 -- the guard holds, and is inert once the window fills
    strong, rs_strong = guarded_seq([-6.0] * 240)
    first_dep = strong.index(1) if 1 in strong else None
    rngc = np.random.default_rng(23)
    early = late = holds_viol = inert_viol = block_viol = 0
    for _ in range(400):
        vals = list(rngc.normal(-0.6, 1.4, 240))
        gs, gr = guarded_seq(vals, True)
        us, _ = guarded_seq(vals, False)
        g_first = gs.index(1) if 1 in gs else None
        u_first = us.index(1) if 1 in us else None
        if g_first is not None and g_first < GUARD_TICK:
            holds_viol += 1
        if gr.blocked_departures > 0 and (gr.first_block_tick is None or gr.first_block_tick >= GUARD_TICK):
            block_viol += 1
        if u_first is not None and u_first >= GUARD_TICK:
            late += 1
            if gs != us:
                inert_viol += 1
        elif u_first is not None:
            early += 1
    by_ep: dict[tuple[str, int], list[tuple[int, float]]] = {}
    with V2_ROWS.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == "holdout":
                by_ep.setdefault((row["pole"], int(row["seed"])), []).append(
                    (int(row["tick"]), float(row["p_blue"]) - float(row["p_red_r4"])))
    real_viol = real_inert_viol = real_n = 0
    for key, pairs in sorted(by_ep.items()):
        vals = [v for _t, v in sorted(pairs)]
        gs, _ = guarded_seq(vals, True)
        us, _ = guarded_seq(vals, False)
        real_n += 1
        g_first = gs.index(1) if 1 in gs else None
        u_first = us.index(1) if 1 in us else None
        if g_first is not None and g_first < GUARD_TICK:
            real_viol += 1
        if u_first is not None and u_first >= GUARD_TICK and gs != us:
            real_inert_viol += 1
    nonvacuous = early > 0 and late > 0 and real_n > 0
    record("C3_GUARD_HOLDS_AND_IS_INERT_AFTER_THE_WINDOW_FILLS",
           first_dep == GUARD_TICK and rs_strong.first_block_tick == 0 and holds_viol == 0
           and inert_viol == 0 and block_viol == 0 and real_viol == 0 and real_inert_viol == 0
           and nonvacuous,
           f"strong-B fixture: first departure at tick {first_dep} (guard boundary {GUARD_TICK}), "
           f"first blocked at tick {rs_strong.first_block_tick}; 400 random sequences: "
           f"{early} with an early unguarded departure, {late} with a late one (both >0 so the "
           f"inert branch is not vacuous), violations: holds={holds_viol} inert={inert_viol} "
           f"blocked-after-window={block_viol}; {real_n} real held-out traces: holds={real_viol} "
           f"inert={real_inert_viol}")

    # C4 -- release is never blocked
    fx = [-6.0] * 60 + [6.0] * 180
    gs, _ = guarded_seq(fx, True)
    us, _ = guarded_seq(fx, False)

    def release_tick(seq: list[int]) -> int | None:
        seen = False
        for t, s in enumerate(seq):
            seen = seen or bool(s)
            if seen and not s:
                return t
        return None
    g_rel, u_rel = release_tick(gs), release_tick(us)
    record("C4_RELEASE_IS_NEVER_BLOCKED",
           g_rel is not None and g_rel == u_rel and gs[GUARD_TICK:] == us[GUARD_TICK:],
           f"both routers release at tick guarded={g_rel} unguarded={u_rel}; sequences identical "
           f"from tick {GUARD_TICK} onward: {gs[GUARD_TICK:] == us[GUARD_TICK:]}")

    # C7 -- guard disabled == offline calibration state machine, tick for tick
    mism, ticks = [], 0
    for key, pairs in sorted(by_ep.items()):
        vals = [v for _t, v in sorted(pairs)]
        g_off, _ = guarded_seq(vals, False)
        ticks += len(vals)
        if g_off != _offline_sequence(vals) or g_off != _online_sequence(vals):
            mism.append(str(key))
    record("C7_ONLINE_OFFLINE_EQUIVALENCE", bool(by_ep) and not mism,
           f"{len(by_ep)} held-out episodes, {ticks} ticks; guard-disabled router == offline "
           f"calibration state machine == frozen OnlineRouter; mismatches: {mism or 'none'}")

    # C5 -- arms differ only in the composition sequence
    parity = []
    for pole in POLES:
        for comp in (DEFAULT_COMPOSITION, TRIGGERED_COMPOSITION):
            pinned, _ = run_guarded_routed_episode(pole, ORACLE_PARITY_SEEDS[0], pinned=comp, label=comp)
            fixed, _ = run_fixed_episode(pole, comp, ORACLE_PARITY_SEEDS[0])
            d = {k: [pinned.get(k), fixed.get(k)] for k in fixed if pinned.get(k) != fixed.get(k)}
            if d:
                parity.append({"pole": pole, "composition": comp, "diff": d})
    record("C5a_PINNED_GUARDED_RUNNER_EQUALS_FIXED_RUNNER", not parity,
           f"4 pinned-vs-fixed comparisons, field by field; disagreements: {parity or 'none'}")

    un = _load_csv(UNGUARDED_ROWS, lambda r: (r["pole"], r["arm"], int(r["seed"])))
    sz = _load_csv(SIZING_ROWS, lambda r: (r["pole"], int(r["seed"])))
    differing = sorted((p, s) for (p, s), z in sz.items()
                       if any(str(z[k]) != str(un[(p, "STATE_B_TRIGGER", s)][k]) for k in OUTCOME_FIELDS))
    identical = sorted((p, s) for (p, s), z in sz.items()
                       if all(str(z[k]) == str(un[(p, "STATE_B_TRIGGER", s)][k]) for k in OUTCOME_FIELDS))
    pick = ([x for x in differing if x[0] == "A"][:3] + [x for x in differing if x[0] == "B"][:3]
            + [x for x in identical if x[0] == "A"][:1] + [x for x in identical if x[0] == "B"][:1])
    off_bad, on_bad, n_differ = [], [], 0
    for pole, seed in pick:
        r_off, _ = run_guarded_routed_episode(pole, seed, guard=False)
        r_on, _ = run_guarded_routed_episode(pole, seed, guard=True)
        u, z = un[(pole, "STATE_B_TRIGGER", seed)], sz[(pole, seed)]
        n_differ += int(any(str(u[k]) != str(z[k]) for k in OUTCOME_FIELDS))
        off_bad += [f"{pole}/{seed}/{k}" for k in OUTCOME_FIELDS if str(r_off[k]) != str(u[k])]
        on_bad += [f"{pole}/{seed}/{k}" for k in OUTCOME_FIELDS if str(r_on[k]) != str(z[k])]
    record("C5b_GUARD_DISABLED_REPRODUCES_SEALED_UNGUARDED_ROWS", not off_bad and n_differ > 0,
           f"{len(pick)} spent-seed episodes; disagreements: {off_bad or 'none'}; guard changes the "
           f"outcome fields on {n_differ}/{len(pick)} of them, so the comparison is not vacuous")
    record("C5c_GUARD_ENABLED_REPRODUCES_SIZING_ROWS", not on_bad and n_differ > 0,
           f"{len(pick)} spent-seed episodes; disagreements: {on_bad or 'none'}; this ties the "
           f"confirmatory runner to the guard that was sized")

    # C8/C9 -- known-answer measurement parity against the sealed oracle rows
    oracle = {}
    with ORACLE_ROWS.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["arm"] == "FIXED_2A2D" and int(row["seed"]) in ORACLE_PARITY_SEEDS:
                oracle[(row["pole"], int(row["seed"]))] = row
    diffs, n_cmp, wins = [], 0, []
    for pole in POLES:
        for seed in ORACLE_PARITY_SEEDS:
            ref = oracle.get((pole, seed))
            if ref is None:
                diffs.append(f"{pole}/{seed}: missing")
                continue
            got, _ = run_fixed_episode(pole, DEFAULT_COMPOSITION, seed)
            n_cmp += 1
            wins.append(int(got["blue_win"]))
            diffs += [f"{pole}/{seed}/{k}" for k in ("blue_score", "red_score", "blue_win", "steps")
                      if int(got[k]) != int(ref[k])]
    record("C8_KNOWN_ANSWER_MEASUREMENT_PARITY", n_cmp == 2 * len(ORACLE_PARITY_SEEDS) and not diffs,
           f"{n_cmp} episodes re-run against sealed oracle FIXED_2A2D rows; disagreements: {diffs or 'none'}")
    record("C9_TERMINAL_SCORING_NOT_RESET_STATE", n_cmp > 0 and 0 < sum(wins) < len(wins),
           f"win rate on the parity subset = {sum(wins)}/{len(wins)} (non-degenerate), exact-matched above")

    # C10 -- PPO unreachable
    live_ppo = sorted(m for m in sys.modules if m.startswith(PPO_FORBIDDEN_PREFIXES))
    record("C10_NO_PPO_REACHABLE", not live_ppo, f"imported PPO/trainer modules: {live_ppo or 'none'}")

    # C11 -- the block is registered to THIS experiment, exact range, and unspent
    from experiments import seed_registry as sr
    ok, msg = sr.check_block(SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS, experiment_id=EXPERIMENT_ID)
    entry = next((b for b in sr.load()["blocks"] if b["experiment_id"] == EXPERIMENT_ID), None)
    record("C11_OUTCOME_SEED_BLOCK_OWNED_AND_UNSPENT",
           ok and entry is not None and entry["status"] == "RESERVED"
           and entry["lo"] == SEED_BASE and entry["hi"] == SEED_BASE + SEED_N - 1,
           f"{msg}; registry status = {entry['status'] if entry else None}")

    # C12 -- the REAL audit plan seals a block that is already registered to its owner.
    # Run AFTER registration, with a negative control: omitting the owner must fail
    # exactly seed_class, proving this dry-run can see the defect that caused the
    # previous run's AUDIT_FAILED.
    import tempfile
    from experiments import run_state as rs
    rng2 = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp()) / "rows.csv"
    rows, byk = [], {}
    for pole in POLES:
        for arm in ARMS:
            for s in OUTCOME_SEEDS:
                bs, rd = int(rng2.integers(0, 4)), int(rng2.integers(0, 4))
                r = {"seed": s, "pole": pole, "arm": arm, "blue_score": bs, "red_score": rd,
                     "blue_win": int(bs > rd), "draw": int(bs == rd), "steps": 240}
                rows.append(r)
                byk[(pole, arm, s)] = r
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rec_b = _bootstrap(_paired(byk, "B", "STATE_GUARDED", "FIXED_2A2D", OUTCOME_SEEDS))
    rec_a = _bootstrap(_paired(byk, "A", "FIXED_2A2D", "STATE_GUARDED", OUTCOME_SEEDS))
    plan = build_audit_plan(tmp, rec_b, rec_a)
    with_owner = rs.run_audit(plan)
    plan_no_owner = rs.AuditPlan(**{**plan.__dict__, "experiment_id": None})
    no_owner = rs.run_audit(plan_no_owner)
    failed_no_owner = sorted(c["name"] for c in no_owner["checks"] if c["result"] == "FAIL")
    record("C12_AUDIT_PLAN_SEALS_A_REGISTERED_BLOCK",
           with_owner["passed"] and failed_no_owner == ["seed_class"],
           f"real AuditPlan with owner on 768 synthetic rows: passed={with_owner['passed']} "
           f"({with_owner['n_gating']} gating, {with_owner['n_failed']} failed); negative control "
           f"without the owner fails exactly {failed_no_owner}",
           n_gating=with_owner["n_gating"])

    n_failed = sum(1 for c in checks if not c["passed"])
    report = {
        "record_id": "GUARDED_ROUTED_COMPOSITION_OUTCOME_CONTRACT_RESULT",
        "utc": _now(), "implements": SPEC_PATH.name,
        "router_operating_point": {"statistic": "D_t = P_blue(4.0) - P_red(4.0)", "window": WINDOW,
                                   "hysteresis": HYSTERESIS, "dwell": DWELL, "threshold": THRESHOLD},
        "guard": {"rule": "no departure from 2A/2D until a full evidence window", "first_allowed_tick": GUARD_TICK},
        "implementation_pins": {
            "size_routed_startup_guard.py_sha256": _sha256(ROOT / "experiments" / "size_routed_startup_guard.py"),
            "run_routed_composition_outcome.py_sha256": _sha256(ROOT / "experiments" / "run_routed_composition_outcome.py"),
        },
        "n_checks": len(checks), "n_gating": len(checks), "n_failed": n_failed, "checks": checks,
        "DECISION": "CONTRACTS_PASS" if n_failed == 0 else "CONTRACT_FAILURE",
        "claim_boundary": "Contracts only. No confirmatory episode was run and no seed was spent by this stage. "
                          "C5/C8 reuse already-spent seeds for deterministic parity only.",
    }
    CONTRACT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'=' * 66}\n  CONTRACTS: {report['DECISION']}  ({n_failed}/{len(checks)} gating failed)\n{'=' * 66}")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"  -> {CONTRACT_PATH}")
    return report


# --------------------------------------------------------------------- outcome

def _load_partial() -> dict[tuple[str, str, int], tuple[dict, dict]]:
    done: dict[tuple[str, str, int], tuple[dict, dict]] = {}
    if PARTIAL.is_file():
        for line in PARTIAL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                done[(obj["row"]["pole"], obj["row"]["arm"], int(obj["row"]["seed"]))] = (obj["row"], obj["diag"])
    return done


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
    done = _load_partial()
    if done:
        print(f"  resuming: {len(done)}/{total} episodes already in {PARTIAL.name}", flush=True)
    rows = {k: v[0] for k, v in done.items()}
    diags = {k: v[1] for k, v in done.items()}
    n_done = len(rows)
    with PARTIAL.open("a", encoding="utf-8") as fh:
        for pole in POLES:
            for arm in ARMS:
                for seed in OUTCOME_SEEDS:
                    if (pole, arm, seed) in rows:
                        continue
                    row, diag = run_job(pole, arm, seed)
                    rows[(pole, arm, seed)], diags[(pole, arm, seed)] = row, diag
                    fh.write(json.dumps({"row": row, "diag": diag}) + "\n")
                    fh.flush()
                    n_done += 1
                    if n_done % 32 == 0 or n_done == total:
                        print(f"  {n_done}/{total} episodes", flush=True)

    fields = list(next(iter(rows.values())).keys())
    with EPISODE_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])
    dfields = ["pole", "arm", "seed", "blocked_departures", "first_block_tick", "n_bursts",
               "first_onset", "ticks_in_4A0D", "fraction_in_4A0D"]
    with DIAG_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=dfields)
        w.writeheader()
        for key in sorted(diags):
            w.writerow({k: diags[key].get(k) for k in dfields})

    seeds = OUTCOME_SEEDS
    b_gain = _bootstrap(_paired(rows, "B", "STATE_GUARDED", "FIXED_2A2D", seeds))
    a_harm = _bootstrap(_paired(rows, "A", "FIXED_2A2D", "STATE_GUARDED", seeds))
    win_rates = {f"{p}_{a}": round(float(np.mean([rows[(p, a, s)]["blue_win"] for s in seeds])), 6)
                 for p in POLES for a in ARMS}
    margins = {f"{p}_{a}": round(float(np.mean([rows[(p, a, s)]["blue_score"] - rows[(p, a, s)]["red_score"]
                                                for s in seeds])), 4) for p in POLES for a in ARMS}

    # the guard's own invariant: no STATE_GUARDED episode may depart before the window fills
    violators = [(p, s) for p in POLES for s in seeds
                 if diags[(p, "STATE_GUARDED", s)]["first_onset"] is not None
                 and diags[(p, "STATE_GUARDED", s)]["first_onset"] < GUARD_TICK]

    pass_b = b_gain["lcb95"] is not None and b_gain["lcb95"] > 0
    pass_a = a_harm["ucb95"] is not None and a_harm["ucb95"] <= TAU_A_HARM
    if violators:
        decision = "GUARD_INVARIANT_VIOLATED"
    elif pass_b and pass_a:
        decision = "GUARDED_ROUTED_COMPOSITION_CONFIRMED"
    elif pass_b:
        decision = "GUARDED_B_GAIN_WITH_EXCESS_A_HARM"
    elif pass_a:
        decision = "GUARDED_A_PRESERVED_B_GAIN_LOST"
    else:
        decision = "GUARDED_NEITHER_GATE_PASSED"

    def flips(pole: str, left: str, right: str) -> dict:
        d = _paired(rows, pole, left, right, seeds)
        return {"left_win_right_not": int((d > 0).sum()), "right_win_left_not": int((d < 0).sum()),
                "net": int(d.sum())}

    def telemetry(pole: str) -> dict:
        g = [diags[(pole, "STATE_GUARDED", s)] for s in seeds]
        onsets = [d["first_onset"] for d in g if d["first_onset"] is not None]
        return {
            "tick_fraction_in_4A0D": round(float(np.mean([d["fraction_in_4A0D"] for d in g])), 6),
            "episodes_with_any_trigger": int(sum(1 for d in g if d["n_bursts"] > 0)),
            "switches_per_episode": round(float(np.mean([rows[(pole, "STATE_GUARDED", s)]["role_switch_count"]
                                                         for s in seeds])), 6),
            "first_onset_mean": (round(float(np.mean(onsets)), 2) if onsets else None),
            "first_onset_min": (int(min(onsets)) if onsets else None),
            "episodes_with_a_blocked_departure": int(sum(1 for d in g if d["blocked_departures"] > 0)),
            "blocked_then_triggered_later": int(sum(1 for d in g if d["blocked_departures"] > 0 and d["n_bursts"] > 0)),
            "blocked_and_never_triggered": int(sum(1 for d in g if d["blocked_departures"] > 0 and d["n_bursts"] == 0)),
        }

    num_b = _paired(rows, "B", "STATE_GUARDED", "FIXED_2A2D", seeds)
    den_b = _paired(rows, "B", "FIXED_4A0D", "FIXED_2A2D", seeds)
    num_a = _paired(rows, "A", "STATE_GUARDED", "FIXED_4A0D", seeds)
    den_a = _paired(rows, "A", "FIXED_2A2D", "FIXED_4A0D", seeds)
    den_b_ci, den_a_ci = _bootstrap(den_b), _bootstrap(den_a)
    capture = {
        "B_capture": {"numerator": _bootstrap(num_b), "denominator": den_b_ci,
                      "ratio_of_means": (round(float(num_b.mean() / den_b.mean()), 4)
                                         if den_b_ci["lcb95"] > 0 or den_b_ci["ucb95"] < 0 else None)},
        "A_damage_avoided": {"numerator": _bootstrap(num_a), "denominator": den_a_ci,
                             "ratio_of_means": (round(float(num_a.mean() / den_a.mean()), 4)
                                                if den_a_ci["lcb95"] > 0 or den_a_ci["ucb95"] < 0 else None)},
        "note": "A ratio is reported only when its denominator interval excludes zero.",
    }
    sizing = json.loads(SIZING_READING.read_text(encoding="utf-8"))["RESULTS"]
    payload = {
        "record_id": "GUARDED_ROUTED_COMPOSITION_OUTCOME_RESULT",
        "implements": SPEC_PATH.name, "utc": _now(),
        "arms": list(ARMS), "poles": list(POLES),
        "seed_block": {"base": SEED_BASE, "last": SEED_BASE + SEED_N - 1, "n": SEED_N, "class": SEED_CLASS},
        "router_operating_point": {"statistic": "D_t = P_blue(4.0) - P_red(4.0)", "window": WINDOW,
                                   "hysteresis": HYSTERESIS, "dwell": DWELL, "threshold": THRESHOLD, "retuned": False},
        "guard": {"rule": "no departure from 2A/2D until a full evidence window", "first_allowed_tick": GUARD_TICK,
                  "invariant_violations": [f"{p}/{s}" for p, s in violators]},
        "contracts": {"path": CONTRACT_PATH.name, "decision": contracts["DECISION"],
                      "n_gating": contracts["n_gating"], "n_failed": contracts["n_failed"]},
        "primary_gates": {
            "B_improvement_paired_STATE_GUARDED_minus_FIXED_2A2D_on_pole_B": b_gain, "B_gate_LCB95>0": pass_b,
            "A_harm_paired_FIXED_2A2D_minus_STATE_GUARDED_on_pole_A": a_harm,
            "tau_A_harm": TAU_A_HARM, "A_gate_UCB95<=tau": pass_a,
        },
        "discordant_pairs": {
            "B_STATE_GUARDED_vs_FIXED_2A2D": flips("B", "STATE_GUARDED", "FIXED_2A2D"),
            "A_FIXED_2A2D_vs_STATE_GUARDED": flips("A", "FIXED_2A2D", "STATE_GUARDED"),
            "note": "left_win_right_not counts seeds won by the left arm and not the right. The percentile bootstrap "
                    "is lenient when discordant pairs are very few; read the interval alongside these counts.",
        },
        "win_rates": win_rates, "score_margin_blue_minus_red": margins,
        "descriptive_not_gating": {
            "FIXED_4A0D_is_reference_only": True, "capture": capture,
            "router_telemetry": {p: telemetry(p) for p in POLES},
            "comparison_to_sizing_not_a_test": {
                "sizing_A_harm_guarded": sizing["S1_pole_A_harm"]["guarded"]["mean"],
                "sizing_B_gain_guarded": sizing["S2_pole_B_gain"]["guarded"]["mean"],
                "confirmatory_A_harm": a_harm["mean"], "confirmatory_B_gain": b_gain["mean"],
                "note": "Sizing was descriptive planning evidence on the block that revealed the mechanism. "
                        "This comparison changes no label."},
        },
        "DECISION": decision,
        "claim_boundary": ("Two frozen opponents, 4v4, PPO off, n=128 paired seeds. Does not authorize PPO, "
                           "learned coordination, or 6v6."),
    }
    rs.seal(out_path=RESULT_PATH, payload=payload, plan=build_audit_plan(EPISODE_CSV, b_gain, a_harm),
            state=state, strict=False)
    sealed = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    sr.set_status(EXPERIMENT_ID, "SPENT", note=f"sealed {sealed.get('status')}; {decision}")
    print(json.dumps({"DECISION": decision, "status": sealed.get("status"), "B_gain": b_gain,
                      "A_harm": a_harm, "win_rates": win_rates}, indent=2))
    return 0 if decision == "GUARDED_ROUTED_COMPOSITION_CONFIRMED" else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts", "outcome"), required=True)
    args = ap.parse_args()
    if args.stage == "contracts":
        return 0 if run_contracts()["DECISION"] == "CONTRACTS_PASS" else 2
    return run_outcome()


if __name__ == "__main__":
    raise SystemExit(main())
