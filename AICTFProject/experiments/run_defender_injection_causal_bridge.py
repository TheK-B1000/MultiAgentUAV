"""DEFENDER_INJECTION_CAUSAL_BRIDGE_V1_SPEC.json.

One causal question on the FROZEN pi_A/pi_B checkpoints: does forcing exactly one
agent's resolved movement target to MacroAction.DEFEND change episode outcomes,
relative to the native (0-defender) rollout? No PPO, no checkpoint write.

  contracts   Verifies the injection mechanism against the scripted DEFEND
              reference (bit-for-bit, both n_macros=5 and 8), the seed-to-agent-id
              rotation, and the checkpoint pins. Spends no seed from the
              20900001-20900096 block.
  run         PI-authorized 2026-09-20 after DECISION_RULES_FROZEN was added to the
              spec. 768 FULL episodes (pi_A/pi_B x Pole A/B x {native, plus_one_defender}
              x 96 paired seeds), deterministic, cuda, torch.no_grad(). Computes the
              four primary per-cell deltas, the per-pole interaction, the four secondary
              crossover-under-condition diagnostics, and seals one of five terminal
              scientific labels (see spec: TERMINAL_SCIENTIFIC_OUTCOMES_FROZEN).
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

import experiments.probe_learned_composition as P  # noqa: E402
import experiments.run_learned_composition_probe as L  # noqa: E402
from experiments.run_routed_composition_outcome import ALPHA, N_BOOT, RNG_SEED, _bootstrap  # noqa: E402
from experiments.tqdm_loop import tqdm_iter  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "DEFENDER_INJECTION_CAUSAL_BRIDGE_V1_SPEC.json"
CONTRACT_PATH = SD / "DEFENDER_INJECTION_CAUSAL_BRIDGE_CONTRACT_RESULT.json"
RESULT_PATH = SD / "DEFENDER_INJECTION_CAUSAL_BRIDGE_RESULT.json"
ROWS_PATH = SD / "DEFENDER_INJECTION_CAUSAL_BRIDGE_ROWS.csv"
PARTIAL_PATH = SD / "DEFENDER_INJECTION_CAUSAL_BRIDGE_PARTIAL.jsonl"

DEVICE = "cuda"
EXP_ID = "DEFENDER_INJECTION_CAUSAL_BRIDGE"
SEED_BASE, SEED_N, SEED_CLASS = 20_900_001, 96, "exploratory"
SEEDS = list(range(SEED_BASE, SEED_BASE + SEED_N))
POLICIES = ("pi_A", "pi_B")
ARMS = ("native", "plus_one_defender")
TRACE_KEYS = ("pos", "alive", "tagged", "carrying", "intent", "true_defend")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------- equivalence matrix

def _equivalence_cell(pole: str, comp: str, override_id: int, seed: int, g: dict) -> tuple[bool, str]:
    """One (pole, defender-id) cell of the contract matrix: a full-length scripted
    episode via the real n_macros=8 raw-DEFEND path, compared against the
    injection at n_macros=8 and again at n_macros=5, with everything but the
    defender's own raw macro held identical."""
    roles = tuple(1 if i == override_id else 0 for i in range(P.N_AGENTS))
    ref = P.run_scripted_episode(comp, pole, seed, g, DEVICE, roles=roles)
    problems: list[str] = []
    for n_macros in (8, 5):
        inj = P.run_scripted_episode_with_forced_override(
            comp, pole, seed, g, DEVICE, override_id, n_macros=n_macros,
            override_raw_macro=int(P.MacroAction.GO_TO), roles=roles)
        for k in TRACE_KEYS:
            a, b = np.asarray(ref[k]), np.asarray(inj[k])
            if a.shape != b.shape or not np.array_equal(a, b):
                problems.append(f"pole={pole} id={override_id} seed={seed} n_macros={n_macros} field={k} MISMATCH")
        want = (ref["blue"], ref["red"], ref["win"], ref["margin"])
        got = (inj["blue"], inj["red"], inj["win"], inj["margin"])
        if want != got:
            problems.append(f"pole={pole} id={override_id} seed={seed} n_macros={n_macros} outcome {got} != {want}")
    return (not problems, "; ".join(problems) if problems else f"pole={pole} id={override_id}: exact at n_macros=5,8")


def contracts() -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str, **data: Any) -> None:
        checks.append({"name": name, "gating": True, "passed": bool(passed), "detail": detail, **data})

    spec = _load_json(SPEC_PATH)
    record("C0_SPEC_MECHANISM_FROZEN", str(spec.get("status", "")).startswith("FROZEN"),
           f"spec status = {spec.get('status')!r}")
    record("C0b_THRESHOLDS_EXPLICITLY_NOT_YET_CONFIRMED",
           spec.get("DECISION_THRESHOLDS_PROPOSED_PENDING_PI_CONFIRMATION") is not None,
           "decision thresholds are proposed, not frozen -- contracts below verify mechanism only, "
           "and this check exists so that fact cannot silently disappear from the record")

    # C1: the equivalence matrix -- both poles, two non-trivial defender ids, full episodes
    g = P.pole_genomes()
    cells = [(pole, did) for pole in P.POLES for did in (2, 4)]
    seed_iter = iter(range(99_900_951, 99_900_951 + len(cells)))  # disposable smoke seeds, outside every registered block
    mismatches = []
    for pole, did in cells:
        ok, detail = _equivalence_cell(pole, "5A_1D", did, next(seed_iter), g)
        if not ok:
            mismatches.append(detail)
    record("C1_INJECTION_MATCHES_SCRIPTED_DEFEND_REFERENCE_EXACTLY",
           not mismatches, "; ".join(mismatches) if mismatches else f"{len(cells)}/{len(cells)} cells exact "
                                                                     f"(pos/alive/tagged/carrying/intent/true_defend/outcome)")

    # C2: rotation -- exact uniform coverage of ids 0..5 over the reserved block
    ids = [P.defender_id_for_seed(s) for s in range(SEED_BASE, SEED_BASE + SEED_N)]
    counts = {i: ids.count(i) for i in range(P.N_AGENTS)}
    uniform = len(set(counts.values())) == 1 and SEED_N % P.N_AGENTS == 0
    record("C2_ROTATION_IS_EXACTLY_UNIFORM_OVER_THE_RESERVED_BLOCK", uniform, f"counts per agent id: {counts}")

    # C3: checkpoints are the pinned ones (same pins the rest of this research line uses)
    pins = L._checkpoint_pins()
    got = {n: L._sha(p) for n, p in P.CHECKPOINTS.items()}
    record("C3_CHECKPOINTS_MATCH_THE_PINNED_HASHES", got == pins, f"{got} vs pinned {pins}")

    # C4: the fresh seed block is reserved to this experiment and unspent, and disjoint from every prior block
    ok4, msg4 = L._registered(EXP_ID, SEED_BASE, SEED_N)
    prior = [(20_700_001, 20_700_016), (20_700_101, 20_700_228), (20_800_001, 20_800_016), (13_680_001, 13_680_128)]
    disjoint = all(SEED_BASE + SEED_N - 1 < lo or SEED_BASE > hi for lo, hi in prior)
    record("C4_SEED_BLOCK_OWNED_UNSPENT_AND_DISJOINT_FROM_EVERY_PRIOR_BLOCK", ok4 and disjoint,
           f"{msg4}; disjoint from {prior}: {disjoint}")

    # C5: no macro id 7 ever reaches the discrete action array at n_macros=5 in the real runner
    #     (mechanical proof: install_forced_defend_target never writes commit_macro; the runner's own
    #     action array for the overridden agent is whatever the POLICY chose, never touched or replaced)
    import inspect
    src = inspect.getsource(P.run_learned_episode_with_forced_defender)
    no_raw_seven = "action[" not in src and "commit_macro" not in src
    record("C5_REAL_RUNNER_NEVER_WRITES_A_RAW_MACRO_OR_COMMIT_TENSOR", no_raw_seven,
           "run_learned_episode_with_forced_defender only calls install_forced_defend_target(); "
           "it does not touch the action array or any commit_* tensor directly")

    n_failed = sum(1 for c in checks if not c["passed"])
    report = {"record_id": "DEFENDER_INJECTION_CAUSAL_BRIDGE_CONTRACT_RESULT", "utc": _now(),
              "implements": SPEC_PATH.name, "n_checks": len(checks), "n_gating": len(checks),
              "n_failed": n_failed, "checks": checks,
              "DECISION": "CONTRACTS_PASS" if n_failed == 0 else "CONTRACT_FAILURE",
              "claim_boundary": "Mechanism contracts only. No seed from 20900001-20900096 was spent. "
                                "The decision thresholds are not verified here (see spec: proposed, not frozen)."}
    CONTRACT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'=' * 66}\n  CAUSAL BRIDGE CONTRACTS: {report['DECISION']}  ({n_failed}/{len(checks)} gating failed)\n{'=' * 66}")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    return report


# --------------------------------------------------------------------- run stage

def _load_partial(path: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                out[tuple(o["key"])] = o["row"]
    return out


def _win_by(rows: list[dict], policy: str, pole: str, arm: str) -> dict[int, float]:
    return {r["seed"]: float(r["win"]) for r in rows if r["policy"] == policy and r["pole"] == pole and r["arm"] == arm}


def _paired(a: dict[int, float], b: dict[int, float]) -> np.ndarray:
    return np.asarray([a[s] - b[s] for s in SEEDS], dtype=np.float64)


def run_causal_bridge() -> int:
    from experiments import run_state as rs
    from experiments import seed_registry as sr
    import torch

    contracts = _load_json(CONTRACT_PATH)
    if contracts.get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit(f"REFUSING: mechanism contracts did not pass ({contracts.get('DECISION')!r})")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")
    ok, msg = L._registered(EXP_ID, SEED_BASE, SEED_N)
    if not ok:
        raise SystemExit(f"REFUSING: seed block not reserved as frozen: {msg}")

    g = P.pole_genomes()
    pols = P.load_policies(DEVICE)
    before = {n: L._param_digest(p) for n, p in pols.items()}

    keys = [(policy, pole, arm, s) for s in SEEDS for policy in POLICIES for pole in P.POLES for arm in ARMS]
    state = rs.RunState(SD, "DEFENDER_INJECTION_CAUSAL_BRIDGE").begin(spec=SPEC_PATH.name, n_episodes=len(keys))
    done = _load_partial(PARTIAL_PATH)
    if done:
        print(f"  resuming: {len(done)}/{len(keys)} episodes recorded", flush=True)
    pending = [k for k in keys if k not in done]
    with PARTIAL_PATH.open("a", encoding="utf-8") as fh, torch.no_grad():
        for key in tqdm_iter(pending, desc="defender-injection causal bridge (cuda, full episodes)",
                             total=len(pending), unit="ep"):
            policy, pole, arm, seed = key
            defender_id = P.defender_id_for_seed(seed)
            if arm == "native":
                tr = P.run_learned_episode(pols[policy], pole, seed, g, DEVICE)
            else:
                tr = P.run_learned_episode_with_forced_defender(pols[policy], pole, seed, g, DEVICE, defender_id)
            row = {"policy": policy, "pole": pole, "arm": arm, "seed": seed, "defender_id": defender_id,
                   "steps": int(tr["steps"]), "blue": int(tr["blue"]), "red": int(tr["red"]),
                   "win": int(tr["win"]), "margin": int(tr["margin"])}
            done[key] = row
            fh.write(json.dumps({"key": list(key), "row": row}) + "\n"); fh.flush()

    after = {n: L._param_digest(p) for n, p in pols.items()}
    if before != after:
        raise SystemExit("ABORT: policy parameters changed during the read-only run")

    rows = [done[k] for k in keys]
    fields = list(rows[0])
    with ROWS_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

    # ---- primary: four per-cell deltas, paired by seed ----
    per_cell = {}
    for policy in POLICIES:
        for pole in P.POLES:
            d = _paired(_win_by(rows, policy, pole, "plus_one_defender"), _win_by(rows, policy, pole, "native"))
            b = _bootstrap(d)
            per_cell[(policy, pole)] = {
                **b, "resolved_improvement": bool(b["lcb95"] > 0), "resolved_harm": bool(b["ucb95"] < 0)}

    # ---- interaction per pole: I_r = Delta_{piB,r} - Delta_{piA,r}, per-seed then bootstrapped ----
    vec_I = {}
    for pole in P.POLES:
        dB = _paired(_win_by(rows, "pi_B", pole, "plus_one_defender"), _win_by(rows, "pi_B", pole, "native"))
        dA = _paired(_win_by(rows, "pi_A", pole, "plus_one_defender"), _win_by(rows, "pi_A", pole, "native"))
        vec_I[pole] = dB - dA
    interaction = {pole: _bootstrap(vec_I[pole]) for pole in P.POLES}

    # ---- secondary diagnostics: crossover under each condition, and the paired change ----
    cA_native = _paired(_win_by(rows, "pi_A", "A", "native"), _win_by(rows, "pi_B", "A", "native"))
    cA_plus1d = _paired(_win_by(rows, "pi_A", "A", "plus_one_defender"), _win_by(rows, "pi_B", "A", "plus_one_defender"))
    cB_native = _paired(_win_by(rows, "pi_B", "B", "native"), _win_by(rows, "pi_A", "B", "native"))
    cB_plus1d = _paired(_win_by(rows, "pi_B", "B", "plus_one_defender"), _win_by(rows, "pi_A", "B", "plus_one_defender"))
    crossover = {
        "Delta_A_native": _bootstrap(cA_native),
        "Delta_A_plus1D": _bootstrap(cA_plus1d),
        "Delta_B_native": _bootstrap(cB_native),
        "Delta_B_plus1D": _bootstrap(cB_plus1d),
    }
    C_A = _bootstrap(cA_plus1d - cA_native)
    C_B = _bootstrap(cB_plus1d - cB_native)

    # ---- independent re-derivation of I_A, I_B, C_A, C_B: plain per-seed dict lookups, no shared
    #      vectorised code path with _win_by/_paired above ----
    by_key = {(r["policy"], r["pole"], r["arm"], r["seed"]): float(r["win"]) for r in rows}

    def loop_diff(pA: str, poleA: str, armA: str, pB: str, poleB: str, armB: str) -> np.ndarray:
        return np.asarray([by_key[(pA, poleA, armA, s)] - by_key[(pB, poleB, armB, s)] for s in SEEDS])

    worst = 0.0
    for pole in P.POLES:
        ref = (loop_diff("pi_B", pole, "plus_one_defender", "pi_B", pole, "native")
               - loop_diff("pi_A", pole, "plus_one_defender", "pi_A", pole, "native"))
        worst = max(worst, float(np.max(np.abs(ref - vec_I[pole]))))
    ref_CA = (loop_diff("pi_A", "A", "plus_one_defender", "pi_B", "A", "plus_one_defender")
             - loop_diff("pi_A", "A", "native", "pi_B", "A", "native"))
    ref_CB = (loop_diff("pi_B", "B", "plus_one_defender", "pi_A", "B", "plus_one_defender")
             - loop_diff("pi_B", "B", "native", "pi_A", "B", "native"))
    worst = max(worst, float(np.max(np.abs(ref_CA - (cA_plus1d - cA_native)))))
    worst = max(worst, float(np.max(np.abs(ref_CB - (cB_plus1d - cB_native)))))
    if worst > 1e-9:
        raise SystemExit(f"ABORT: independent re-derivation of I_A/I_B/C_A/C_B disagrees by {worst:.3e}")
    print(f"  independent re-derivation of I_A/I_B/C_A/C_B PASS (max |diff| {worst:.2e})", flush=True)

    # ---- terminal scientific label, per DECISION_RULES_FROZEN / TERMINAL_SCIENTIFIC_OUTCOMES_FROZEN ----
    transferable = all(per_cell[(p, r)]["resolved_improvement"] for p in POLICIES for r in P.POLES)
    b_support_by_pole = {r: bool(per_cell[("pi_B", r)]["resolved_improvement"] and interaction[r]["lcb95"] > 0)
                         for r in P.POLES}
    b_disproportionate = any(b_support_by_pole.values())
    has_improvement = any(per_cell[k]["resolved_improvement"] for k in per_cell)
    has_harm = any(per_cell[k]["resolved_harm"] for k in per_cell)
    if transferable:
        verdict = "TRANSFERABLE_ONE_DEFENDER_SCAFFOLD"
    elif b_disproportionate:
        verdict = "B_DISPROPORTIONATE_ONE_DEFENDER_SUPPORT"
    elif has_improvement and has_harm:
        verdict = "MIXED_ONE_DEFENDER_EFFECT"
    elif not has_improvement and not has_harm:
        verdict = "NO_RESOLVED_ONE_DEFENDER_BENEFIT"
    elif has_harm and not has_improvement:
        verdict = "ONE_DEFENDER_HARM_ONLY"
    else:                                             # has_improvement, no harm, neither stronger label -- policy/pole-specific
        verdict = "MIXED_ONE_DEFENDER_EFFECT"

    def _pc(p, r):
        b = per_cell[(p, r)]
        return {"mean": b["mean"], "lcb95": b["lcb95"], "ucb95": b["ucb95"], "n": b["n"],
               "resolved_improvement": b["resolved_improvement"], "resolved_harm": b["resolved_harm"]}

    claims = [
        rs.Claim(name=f"delta_{p}_{r}", recorded={"mean": per_cell[(p, r)]["mean"], "lcb95": per_cell[(p, r)]["lcb95"], "ucb95": per_cell[(p, r)]["ucb95"]},
                minuend={"policy": p, "pole": r, "arm": "plus_one_defender"}, subtrahend={"policy": p, "pole": r, "arm": "native"}, value_field="win")
        for p in POLICIES for r in P.POLES
    ] + [
        rs.Claim(name="crossover_A_native", recorded={"mean": crossover["Delta_A_native"]["mean"], "lcb95": crossover["Delta_A_native"]["lcb95"], "ucb95": crossover["Delta_A_native"]["ucb95"]},
                minuend={"policy": "pi_A", "pole": "A", "arm": "native"}, subtrahend={"policy": "pi_B", "pole": "A", "arm": "native"}, value_field="win"),
        rs.Claim(name="crossover_A_plus1D", recorded={"mean": crossover["Delta_A_plus1D"]["mean"], "lcb95": crossover["Delta_A_plus1D"]["lcb95"], "ucb95": crossover["Delta_A_plus1D"]["ucb95"]},
                minuend={"policy": "pi_A", "pole": "A", "arm": "plus_one_defender"}, subtrahend={"policy": "pi_B", "pole": "A", "arm": "plus_one_defender"}, value_field="win"),
        rs.Claim(name="crossover_B_native", recorded={"mean": crossover["Delta_B_native"]["mean"], "lcb95": crossover["Delta_B_native"]["lcb95"], "ucb95": crossover["Delta_B_native"]["ucb95"]},
                minuend={"policy": "pi_B", "pole": "B", "arm": "native"}, subtrahend={"policy": "pi_A", "pole": "B", "arm": "native"}, value_field="win"),
        rs.Claim(name="crossover_B_plus1D", recorded={"mean": crossover["Delta_B_plus1D"]["mean"], "lcb95": crossover["Delta_B_plus1D"]["lcb95"], "ucb95": crossover["Delta_B_plus1D"]["ucb95"]},
                minuend={"policy": "pi_B", "pole": "B", "arm": "plus_one_defender"}, subtrahend={"policy": "pi_A", "pole": "B", "arm": "plus_one_defender"}, value_field="win"),
    ]
    pins = L._checkpoint_pins()

    payload = {
        "record_id": "DEFENDER_INJECTION_CAUSAL_BRIDGE_RESULT", "implements": SPEC_PATH.name, "utc": _now(),
        "device": DEVICE, "seed_block": {"base": SEED_BASE, "n": SEED_N, "class": SEED_CLASS},
        "PER_CELL_DELTA": {f"{p}_{r}": _pc(p, r) for p in POLICIES for r in P.POLES},
        "INTERACTION_PER_POLE_I_r": interaction,
        "TRANSFERABLE_ONE_DEFENDER_SCAFFOLD": transferable,
        "B_DISPROPORTIONATE_SUPPORT_BY_POLE": b_support_by_pole,
        "B_DISPROPORTIONATE_ONE_DEFENDER_SUPPORT": b_disproportionate,
        "SECONDARY_DIAGNOSTIC_NOT_GATING": {
            "crossover_under_condition": crossover,
            "paired_change_from_native_crossover": {"C_A": C_A, "C_B": C_B},
        },
        "SCIENTIFIC_VERDICT": verdict,
        "independent_rederivation_I_C": {"max_abs_diff": worst, "seeds": len(SEEDS)},
        "read_only_guarantee": {"param_digest_equal_before_after": before == after,
                                "checkpoint_sha256_matches_pinned": {n: L._sha(p) for n, p in P.CHECKPOINTS.items()} == pins},
        "claim_boundary": "768 full episodes, frozen checkpoints, rollout-time intervention only, fresh seeds "
                          "20900001-20900096 spent for the first time by this spec. SCIENTIFIC_VERDICT is certified "
                          "ONLY if this record's own status below is SEALED -- integrity/audit failure takes "
                          "precedence over every scientific label, per the frozen spec.",
    }
    plan = rs.AuditPlan(rows_csv=ROWS_PATH, expected_rows=len(rows), expected_seeds=SEEDS,
                        group_by=("policy", "pole", "arm"), seed_field="seed", int_fields=("seed", "steps", "defender_id"),
                        binary_fields=("win",), derived={},
                        checkpoints={n: (p, pins[n]) for n, p in P.CHECKPOINTS.items()},
                        spec_path=SPEC_PATH, seed_class=SEED_CLASS, experiment_id=EXP_ID,
                        n_boot=N_BOOT, alpha=ALPHA, rng_seed=RNG_SEED, claims=claims)
    rs.seal(out_path=RESULT_PATH, payload=payload, plan=plan, state=state, strict=False)
    sealed = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    sr.set_status(EXP_ID, "SPENT", note=f"sealed {sealed.get('status')}; verdict={verdict}")
    print(json.dumps({"status": sealed.get("status"), "verdict": verdict, "transferable": transferable,
                      "b_disproportionate_by_pole": b_support_by_pole}, indent=2))
    return 0 if sealed.get("status") == "SEALED" else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts", "run"), required=True)
    args = ap.parse_args()
    if args.stage == "contracts":
        return 0 if contracts()["DECISION"] == "CONTRACTS_PASS" else 2
    return run_causal_bridge()


if __name__ == "__main__":
    raise SystemExit(main())
