"""ONE_DEFENDER_GOAL_VOLUME_CONFIRMATION_V1_SPEC.json.

CONFIRMATORY test of a post-hoc hypothesis on FRESH seeds: forcing exactly one policy-external
defender raises Blue goal production against Pole A and lowers it against Pole B, for both frozen
learned checkpoints. Primary criterion: J_p = dG_{p,A} - dG_{p,B} has LCB95 > 0 for both policies.
The four cell directions are a secondary replication requirement (point estimates only). Margin is
reported as a consistency check and never counts as a second vote. No PPO, no checkpoint write.

  contracts   Mechanism/registry/decision-rule contracts. Spends no seed from the fresh block.
  run         One shard of the 768 full episodes (resumable, append-only partial file per shard).
              Prints progress COUNTS only -- no outcomes -- per the frozen no-interim-look rule.
  analyze     Refuses to run unless all 768 cells are present; computes the frozen endpoints,
              seals through the formal audit pipeline, marks the seed block SPENT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.probe_learned_composition as P  # noqa: E402
import experiments.run_defender_injection_causal_bridge as C  # noqa: E402
import experiments.run_learned_composition_probe as L  # noqa: E402
from experiments.run_routed_composition_outcome import ALPHA, N_BOOT, RNG_SEED, _bootstrap  # noqa: E402
from experiments.tqdm_loop import tqdm_iter  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "ONE_DEFENDER_GOAL_VOLUME_CONFIRMATION_V1_SPEC.json"
CONTRACT_PATH = SD / "ONE_DEFENDER_GOAL_VOLUME_CONFIRMATION_CONTRACT_RESULT.json"
RESULT_PATH = SD / "ONE_DEFENDER_GOAL_VOLUME_CONFIRMATION_RESULT.json"
ROWS_PATH = SD / "ONE_DEFENDER_GOAL_VOLUME_CONFIRMATION_ROWS.csv"
SHARD_GLOB = "ONE_DEFENDER_GOAL_VOLUME_CONFIRMATION_SHARD*_PARTIAL.jsonl"
MECHANISM_FILE = ROOT / "experiments" / "probe_learned_composition.py"

DEVICE = "cuda"
EXP_ID = "ONE_DEFENDER_GOAL_VOLUME_CONFIRMATION"
SEED_BASE, SEED_N, SEED_CLASS = 21_000_001, 96, "sealed_confirmatory"
SEEDS = list(range(SEED_BASE, SEED_BASE + SEED_N))
POLICIES = ("pi_A", "pi_B")
ARMS = ("native", "plus_one_defender")
LABEL_CONFIRMED = "OPPONENT_CONDITIONED_GOAL_INTERACTION_CONFIRMED"
LABEL_PARTIAL = "INTERACTION_CONFIRMED_PATTERN_PARTIAL"
LABEL_NOT = "OPPONENT_CONDITIONED_GOAL_INTERACTION_NOT_CONFIRMED"
INTERPRETATION_SENTENCE = ("A positive J establishes differential intervention response between the two certified "
                           "opponents. It does not distinguish opponent-specific strategic sensitivity from "
                           "regression/compression toward a common goal-production level.")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _cells() -> list[tuple[str, str, str, int]]:
    return [(policy, pole, arm, s) for s in SEEDS for policy in POLICIES for pole in P.POLES for arm in ARMS]


def _registered() -> tuple[bool, str]:
    """Bind to the real registry entry: reserved to THIS experiment, exact range, and the frozen class."""
    from experiments import seed_registry as sr
    ok, msg = sr.check_block(SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS, experiment_id=EXP_ID)
    entry = next((b for b in sr.load()["blocks"] if b["experiment_id"] == EXP_ID), None)
    good = bool(ok and entry is not None and entry["status"] == "RESERVED" and entry["lo"] == SEED_BASE
                and entry["hi"] == SEED_BASE + SEED_N - 1 and entry["seed_class"] == SEED_CLASS)
    return good, f"{msg}; registry status={entry and entry['status']} class={entry and entry['seed_class']}"


# ---------------------------------------------------------------- frozen decision rule

def terminal_label(j_lcb: dict[str, float], cell_mean: dict[tuple[str, str], float]) -> str:
    """The frozen terminal outcome (audit failure is handled by the sealed status, not here).
    NOT_CONFIRMED if either J lower bound is not above zero; else CONFIRMED if all four cell point
    estimates reproduce the pre-registered signs (A strictly positive, B strictly negative); else PARTIAL."""
    if not all(j_lcb[p] > 0 for p in POLICIES):
        return LABEL_NOT
    signs_ok = all(cell_mean[(p, "A")] > 0 and cell_mean[(p, "B")] < 0 for p in POLICIES)
    return LABEL_CONFIRMED if signs_ok else LABEL_PARTIAL


def _label_selftest() -> tuple[bool, str]:
    good = {("pi_A", "A"): 0.8, ("pi_A", "B"): -0.5, ("pi_B", "A"): 0.9, ("pi_B", "B"): -0.4}
    cases = [
        ("all criteria met", {"pi_A": 0.5, "pi_B": 0.3}, good, LABEL_CONFIRMED),
        ("one cell has the wrong sign", {"pi_A": 0.5, "pi_B": 0.3}, {**good, ("pi_B", "B"): 0.1}, LABEL_PARTIAL),
        ("a cell estimate of exactly zero fails its sign", {"pi_A": 0.5, "pi_B": 0.3}, {**good, ("pi_A", "A"): 0.0}, LABEL_PARTIAL),
        ("J lower bound exactly zero is not above zero", {"pi_A": 0.5, "pi_B": 0.0}, good, LABEL_NOT),
        ("J lower bound negative", {"pi_A": -0.2, "pi_B": 0.3}, good, LABEL_NOT),
        ("J fails AND a sign fails -> NOT_CONFIRMED takes precedence", {"pi_A": 0.5, "pi_B": -0.1}, {**good, ("pi_A", "B"): 0.2}, LABEL_NOT),
        ("no requirement that cell intervals exclude zero (only point estimates enter)", {"pi_A": 0.01, "pi_B": 0.01}, good, LABEL_CONFIRMED),
    ]
    bad = [name for name, j, m, want in cases if terminal_label(j, m) != want]
    return not bad, (f"{len(cases)}/{len(cases)} synthetic cases return the frozen label" if not bad else f"WRONG on: {bad}")


# ---------------------------------------------------------------------------- contracts

def contracts() -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "gating": True, "passed": bool(passed), "detail": detail})

    spec = _load_json(SPEC_PATH)
    record("C0_SPEC_FROZEN", str(spec.get("status", "")).startswith("FROZEN"), f"spec status = {spec.get('status')!r}")

    want_pin = spec["MECHANISM_UNCHANGED"]["source_pin"]["sha256"]
    got_pin = _sha_file(MECHANISM_FILE)
    record("C1_MECHANISM_SOURCE_EQUALS_THE_PIN", got_pin == want_pin, f"{got_pin} vs pinned {want_pin}")

    g = P.pole_genomes()
    mismatches = []
    for (pole, did, seed) in (("A", 1, 99_900_971), ("B", 5, 99_900_972)):
        ok, detail = C._equivalence_cell(pole, "5A_1D", did, seed, g)
        if not ok:
            mismatches.append(detail)
    record("C2_INJECTION_STILL_MATCHES_SCRIPTED_DEFEND_EXACTLY", not mismatches,
           "; ".join(mismatches) if mismatches else "2/2 fresh (pole, id) cells exact at n_macros 5 and 8 "
                                                   "(pos/alive/tagged/carrying/intent/true_defend/outcome)")

    ids = [P.defender_id_for_seed(s) for s in SEEDS]
    counts = {i: ids.count(i) for i in range(P.N_AGENTS)}
    record("C3_ROTATION_EXACTLY_UNIFORM_OVER_THE_BLOCK", len(set(counts.values())) == 1 and SEED_N % P.N_AGENTS == 0,
           f"counts per agent id: {counts}")

    pins, got = L._checkpoint_pins(), {n: L._sha(p) for n, p in P.CHECKPOINTS.items()}
    record("C4_CHECKPOINTS_MATCH_THE_PINNED_HASHES", got == pins, f"{got} vs pinned {pins}")

    ok5, msg5 = _registered()
    prior = [(13_680_001, 13_680_128), (20_700_001, 20_700_016), (20_700_101, 20_700_228),
             (20_800_001, 20_800_016), (20_900_001, 20_900_096)]
    disjoint = all(SEED_BASE + SEED_N - 1 < lo or SEED_BASE > hi for lo, hi in prior)
    record("C5_BLOCK_RESERVED_AS_SEALED_CONFIRMATORY_UNSPENT_AND_DISJOINT", ok5 and disjoint,
           f"{msg5}; disjoint from {prior}: {disjoint}")

    src = inspect.getsource(P.run_learned_episode_with_forced_defender)
    record("C6_REAL_RUNNER_NEVER_WRITES_A_RAW_MACRO_OR_COMMIT_TENSOR", "action[" not in src and "commit_macro" not in src,
           "the runner only calls install_forced_defend_target(); it touches no action array or commit_* tensor")

    ok7, msg7 = _label_selftest()
    record("C7_TERMINAL_RULE_RETURNS_THE_FROZEN_LABEL_ON_SYNTHETIC_CASES", ok7, msg7)

    n_failed = sum(1 for c in checks if not c["passed"])
    report = {"record_id": "ONE_DEFENDER_GOAL_VOLUME_CONFIRMATION_CONTRACT_RESULT", "utc": _now(),
              "implements": SPEC_PATH.name, "n_checks": len(checks), "n_gating": len(checks), "n_failed": n_failed,
              "checks": checks, "DECISION": "CONTRACTS_PASS" if n_failed == 0 else "CONTRACT_FAILURE",
              "claim_boundary": f"Contracts only. No seed from {SEED_BASE}-{SEED_BASE + SEED_N - 1} was spent."}
    CONTRACT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'=' * 66}\n  CONFIRMATION CONTRACTS: {report['DECISION']}  ({n_failed}/{len(checks)} gating failed)\n{'=' * 66}")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    return report


# -------------------------------------------------------------------------------- run

def _shard_path(i: int, k: int) -> Path:
    return SD / f"ONE_DEFENDER_GOAL_VOLUME_CONFIRMATION_SHARD{i}OF{k}_PARTIAL.jsonl"


def _load_partial(path: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                out[tuple(o["key"])] = o["row"]
    return out


def run_shard(shard: int, n_shards: int) -> int:
    if _load_json(CONTRACT_PATH).get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit("REFUSING: contracts did not pass")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")
    ok, msg = _registered()
    if not ok:
        raise SystemExit(f"REFUSING: seed block not reserved as frozen: {msg}")

    mine = [c for i, c in enumerate(_cells()) if i % n_shards == shard]
    path = _shard_path(shard, n_shards)
    done = _load_partial(path)
    pending = [c for c in mine if c not in done]
    print(f"  shard {shard}/{n_shards}: {len(mine)} cells, {len(done)} already recorded, {len(pending)} to run", flush=True)
    if not pending:
        return 0
    import torch
    g = P.pole_genomes()
    pols = P.load_policies(DEVICE)
    before = {n: L._param_digest(p) for n, p in pols.items()}
    with path.open("a", encoding="utf-8") as fh, torch.no_grad():
        for key in tqdm_iter(pending, desc=f"confirmation shard {shard}/{n_shards} (cuda)", total=len(pending), unit="ep"):
            policy, pole, arm, seed = key
            defender_id = P.defender_id_for_seed(seed)
            if arm == "native":
                tr = P.run_learned_episode(pols[policy], pole, seed, g, DEVICE)
            else:
                tr = P.run_learned_episode_with_forced_defender(pols[policy], pole, seed, g, DEVICE, defender_id)
            row = {"policy": policy, "pole": pole, "arm": arm, "seed": seed, "defender_id": defender_id,
                   "steps": int(tr["steps"]), "blue": int(tr["blue"]), "red": int(tr["red"]),
                   "win": int(tr["win"]), "margin": int(tr["margin"])}
            fh.write(json.dumps({"key": list(key), "row": row}) + "\n"); fh.flush()
    if before != {n: L._param_digest(p) for n, p in pols.items()}:
        raise SystemExit("ABORT: policy parameters changed during a read-only run")
    print(f"  shard {shard}/{n_shards} complete (parameter digests unchanged)", flush=True)
    return 0


# ----------------------------------------------------------------------------- analyze

def analyze() -> int:
    from experiments import run_state as rs
    from experiments import seed_registry as sr

    if _load_json(CONTRACT_PATH).get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit("REFUSING: contracts did not pass")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")
    ok, msg = _registered()
    if not ok:
        raise SystemExit(f"REFUSING: seed block not reserved as frozen: {msg}")

    # Absence is an error state: every one of the 768 cells, exactly once, or nothing is reported.
    merged: dict[tuple, dict] = {}
    for path in sorted(SD.glob(SHARD_GLOB)):
        for key, row in _load_partial(path).items():
            if key in merged and merged[key] != row:
                raise SystemExit(f"ABORT: conflicting rows for {key} across shards")
            merged[key] = row
    want = set(_cells())
    missing, extra = sorted(want - set(merged)), sorted(set(merged) - want)
    if missing or extra:
        raise SystemExit(f"ABORT: {len(missing)} cell(s) missing (e.g. {missing[:2]}), {len(extra)} unexpected "
                         f"(e.g. {extra[:2]}). Run/resume the shards first.")
    bad_rot = [k for k, r in merged.items() if int(r["defender_id"]) != P.defender_id_for_seed(k[3])]
    if bad_rot:
        raise SystemExit(f"ABORT: {len(bad_rot)} row(s) whose defender_id is not seed % 6, e.g. {bad_rot[0]}")
    rows = [merged[c] for c in _cells()]
    with ROWS_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    state = rs.RunState(SD, "ONE_DEFENDER_GOAL_VOLUME_CONFIRMATION").begin(spec=SPEC_PATH.name, n_episodes=len(rows))

    val = {f: {(r["policy"], r["pole"], r["arm"], int(r["seed"])): float(r[f]) for r in rows} for f in ("blue", "margin")}

    def cell_diff(f: str, policy: str, pole: str) -> np.ndarray:
        return np.asarray([val[f][(policy, pole, "plus_one_defender", s)] - val[f][(policy, pole, "native", s)]
                           for s in SEEDS], dtype=np.float64)

    def native_mean(f: str, policy: str, pole: str) -> float:
        return float(np.mean([val[f][(policy, pole, "native", s)] for s in SEEDS]))

    def plus_mean(f: str, policy: str, pole: str) -> float:
        return float(np.mean([val[f][(policy, pole, "plus_one_defender", s)] for s in SEEDS]))

    per_cell: dict[str, dict] = {"blue_goals": {}, "margin": {}}
    j_arr: dict[str, dict[str, np.ndarray]] = {"blue": {}, "margin": {}}
    dG_mean: dict[tuple[str, str], float] = {}
    for f, name in (("blue", "blue_goals"), ("margin", "margin")):
        for policy in POLICIES:
            for pole in P.POLES:
                d = cell_diff(f, policy, pole)
                per_cell[name][f"{policy}_pole{pole}"] = {
                    "native_mean": round(native_mean(f, policy, pole), 6),
                    "plus_one_defender_mean": round(plus_mean(f, policy, pole), 6), **_bootstrap(d)}
                if f == "blue":
                    dG_mean[(policy, pole)] = float(d.mean())
            j_arr[f][policy] = cell_diff(f, policy, "A") - cell_diff(f, policy, "B")

    # Independent re-derivation of every J: plain per-seed dict lookups sharing no code path with cell_diff.
    worst = 0.0
    for f in ("blue", "margin"):
        for policy in POLICIES:
            ref = np.asarray([(val[f][(policy, "A", "plus_one_defender", s)] - val[f][(policy, "A", "native", s)])
                              - (val[f][(policy, "B", "plus_one_defender", s)] - val[f][(policy, "B", "native", s)])
                              for s in SEEDS])
            worst = max(worst, float(np.max(np.abs(ref - j_arr[f][policy]))))
    if worst > 1e-9:
        raise SystemExit(f"ABORT: independent re-derivation of J disagrees by {worst:.3e}")
    print(f"  independent re-derivation of J (Blue goals and margin) PASS (max |diff| {worst:.2e})", flush=True)

    J = {policy: _bootstrap(j_arr["blue"][policy]) for policy in POLICIES}
    J_margin = {policy: _bootstrap(j_arr["margin"][policy]) for policy in POLICIES}
    label = terminal_label({p: J[p]["lcb95"] for p in POLICIES}, dG_mean)
    signs = {f"{p}_pole{r}": bool(dG_mean[(p, r)] > 0 if r == "A" else dG_mean[(p, r)] < 0)
             for p in POLICIES for r in P.POLES}

    claims = []
    for f, name in (("blue", "dG"), ("margin", "dM")):
        key = "blue_goals" if f == "blue" else "margin"
        for policy in POLICIES:
            for pole in P.POLES:
                b = per_cell[key][f"{policy}_pole{pole}"]
                claims.append(rs.Claim(
                    name=f"{name}_{policy}_pole{pole}", recorded={"mean": b["mean"], "lcb95": b["lcb95"], "ucb95": b["ucb95"]},
                    minuend={"policy": policy, "pole": pole, "arm": "plus_one_defender"},
                    subtrahend={"policy": policy, "pole": pole, "arm": "native"}, value_field=f))
    pins = L._checkpoint_pins()

    payload = {
        "record_id": "ONE_DEFENDER_GOAL_VOLUME_CONFIRMATION_RESULT", "implements": SPEC_PATH.name, "utc": _now(),
        "device": DEVICE, "seed_block": {"base": SEED_BASE, "n": SEED_N, "class": SEED_CLASS},
        "PRIMARY_J_PER_POLICY_BLUE_GOALS": {p: {**J[p], "lcb95_above_zero": bool(J[p]["lcb95"] > 0)} for p in POLICIES},
        "SECONDARY_FOUR_SIGNS_REPRODUCED_POINT_ESTIMATES": signs,
        "PER_CELL_DELTA_G": per_cell["blue_goals"],
        "MARGIN_CONSISTENCY_CHECK_NOT_A_SECOND_VOTE": {"per_cell": per_cell["margin"], "J_per_policy": J_margin},
        "TERMINAL_OUTCOME": label,
        "independent_rederivation_J": {"max_abs_diff": worst, "seeds": len(SEEDS)},
        "interpretation_guard": INTERPRETATION_SENTENCE,
        "wording": "opponent-conditioned across the two certified 6v6 poles",
        "provenance": {"spec_sha256": _sha_file(SPEC_PATH), "script_sha256": _sha_file(Path(__file__)),
                       "mechanism_source_sha256": _sha_file(MECHANISM_FILE)},
        "claim_boundary": "Two frozen learned checkpoints, two certified 6v6 opponents, rollout-time intervention, fresh "
                          "sealed_confirmatory seeds. TERMINAL_OUTCOME is certified ONLY if this record's own status is "
                          "SEALED; integrity/audit failure takes precedence over every label. " + INTERPRETATION_SENTENCE,
    }
    plan = rs.AuditPlan(
        rows_csv=ROWS_PATH, expected_rows=len(rows), expected_seeds=SEEDS, group_by=("policy", "pole", "arm"),
        seed_field="seed", int_fields=("seed", "steps", "defender_id", "blue", "red", "margin"), binary_fields=("win",),
        derived={"margin": rs.Derived("margin == blue - red", lambda r: r["blue"] - r["red"]),
                 "win": rs.Derived("win == (blue > red)", lambda r: int(r["blue"] > r["red"]))},
        checkpoints={n: (p, pins[n]) for n, p in P.CHECKPOINTS.items()}, spec_path=SPEC_PATH,
        seed_class=SEED_CLASS, experiment_id=EXP_ID, n_boot=N_BOOT, alpha=ALPHA, rng_seed=RNG_SEED, claims=claims)
    rs.seal(out_path=RESULT_PATH, payload=payload, plan=plan, state=state, strict=False)
    sealed = _load_json(RESULT_PATH)
    sr.set_status(EXP_ID, "SPENT", note=f"sealed {sealed.get('status')}; {label}")

    print(json.dumps({"status": sealed.get("status"), "TERMINAL_OUTCOME": label,
                      "J": {p: {k: J[p][k] for k in ("mean", "lcb95", "ucb95")} for p in POLICIES},
                      "signs_reproduced": signs}, indent=2))
    return 0 if sealed.get("status") == "SEALED" else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts", "run", "analyze"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1, dest="n_shards")
    a = ap.parse_args()
    if a.stage == "contracts":
        return 0 if contracts()["DECISION"] == "CONTRACTS_PASS" else 2
    if a.stage == "run":
        if not 0 <= a.shard < a.n_shards:
            raise SystemExit(f"--shard must be in [0, {a.n_shards})")
        return run_shard(a.shard, a.n_shards)
    return analyze()


if __name__ == "__main__":
    raise SystemExit(main())
