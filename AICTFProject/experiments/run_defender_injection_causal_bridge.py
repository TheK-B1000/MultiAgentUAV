"""DEFENDER_INJECTION_CAUSAL_BRIDGE_V1_SPEC.json.

One causal question on the FROZEN pi_A/pi_B checkpoints: does forcing exactly one
agent's resolved movement target to MacroAction.DEFEND change episode outcomes,
relative to the native (0-defender) rollout? No PPO, no checkpoint write.

  contracts   Verifies the injection mechanism against the scripted DEFEND
              reference (bit-for-bit, both n_macros=5 and 8), the seed-to-agent-id
              rotation, and the checkpoint pins. Spends no seed from the
              20900001-20900096 block. Does NOT verify the decision thresholds,
              which are proposed-not-frozen in the spec pending PI confirmation.

The `run` stage is deliberately NOT implemented in this file: per the frozen
spec's own NOT_AUTHORIZED list, no seed from the reserved block may be spent
until contracts pass AND the PI confirms the decision thresholds.
"""
from __future__ import annotations

import argparse
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

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "DEFENDER_INJECTION_CAUSAL_BRIDGE_V1_SPEC.json"
CONTRACT_PATH = SD / "DEFENDER_INJECTION_CAUSAL_BRIDGE_CONTRACT_RESULT.json"

DEVICE = "cuda"
EXP_ID = "DEFENDER_INJECTION_CAUSAL_BRIDGE"
SEED_BASE, SEED_N, SEED_CLASS = 20_900_001, 96, "exploratory"
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts",), required=True)
    args = ap.parse_args()
    if args.stage == "contracts":
        return 0 if contracts()["DECISION"] == "CONTRACTS_PASS" else 2
    raise SystemExit("the run stage is intentionally unwritten -- see the module docstring")


if __name__ == "__main__":
    raise SystemExit(main())
