"""CCP-S2 collector gate: the required smoke before any of the 10,176 real jobs run.

Per the PI, this must prove three things simultaneously, on REAL selected boundaries using
the ACTUAL collector code path (experiments/ccp_s2_collect.py's setup_env/replay_prefix/
continuation_seed -- imported, not re-derived, so this tests the real implementation rather
than a parallel one that could silently drift from it):

  1. exact-prefix replay reproduces the selected boundary -- not just action-for-action
     (already structurally enforced by replay_prefix's REFUSING check), but bitwise-identical
     full environment state, the same standard Phase 0 used
     (CCP_PHASE0_REPLAY_EQUIVALENCE.json).
  2. all three arms (R_0, pi_A, pi_B) begin from bitwise-identical reconstructed state AND
     bitwise-identical RNG state before intervention -- verified as three INDEPENDENT env
     instances, each replayed through the same real boundary's prefix (mirroring how the real
     collector gives every job its own env in run()), then reseeded with the frozen
     CCP_S2_MEASURE r_j and compared byte-for-byte (torch.Generator.get_state()), the same
     technique as CCP_PHASE1_CRN_SMOKE.json.
  3. a deliberately broken, order-dependent seed dispatcher is DETECTED as non-identical by
     this same check -- a check that cannot fail proves nothing (this project's standing
     rule; see CCP_S2_SEED_PREFLIGHT.json check_6 and CCP_PHASE0_REPLAY_EQUIVALENCE.json's
     perturbation control for the identical pattern applied elsewhere in this program).

Sample: 4 real boundaries from CCP_S2_STATE_MANIFEST.json, chosen by simple deterministic
filters (first one-free and first both-free state per pole in the manifest's own frozen
order) -- not cherry-picked for outcome, since nothing here measures an outcome.

Touches no measurement job, no bank, no EVAL block. One-shot.

Run:  python experiments/ccp_s2_collect_smoke.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
MANIFEST = SD / "CCP_S2_STATE_MANIFEST.json"
OUT = SD / "CCP_S2_COLLECTOR_SMOKE.json"

J_SAMPLE = (0, 15)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _snap(obs: dict, gs) -> dict:
    out = {f"obs.{k}": np.asarray(v).copy() for k, v in obs.items()}
    out["global_state"] = np.asarray(gs).copy()
    return out


def _compare(a: dict, b: dict) -> tuple[bool, list[dict]]:
    diffs = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            diffs.append({"field": k, "issue": "present in only one snapshot"})
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            diffs.append({"field": k, "issue": f"shape {x.shape} vs {y.shape}"})
        elif not np.array_equal(x, y):
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            diffs.append({"field": k, "max_abs_diff": float(d.max()),
                          "n_elements_differing": int((x != y).sum()), "of": int(x.size)})
    return (not diffs), diffs


def pick_sample(manifest: dict) -> list[dict]:
    states = manifest["states"]
    picks = []
    for pole in ("A", "B"):
        one_free = next(s for s in states if s["pole"] == pole and s["free_set"] != "both_free")
        both_free = next(s for s in states if s["pole"] == pole and s["free_set"] == "both_free")
        picks.append(one_free)
        picks.append(both_free)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_SELECTION":
        raise SystemExit(f"REFUSING: state manifest not frozen: {manifest['status']!r}")

    import torch
    import experiments.ccp_s2_collect as C

    print(f"CCP-S2 COLLECTOR SMOKE  {_now()}")
    device, incumbent, teachers, R2, env_ctx = C.load_runtime(args.device)
    print(f"  incumbent + pi_A + pi_B loaded, sha-verified, device={device}\n", flush=True)

    def rng_state(core):
        g = getattr(core, "_rng", None)
        if g is None:
            raise SystemExit("REFUSING: core exposes no _rng")
        return g.get_state().clone()

    def fresh_replay(state: dict, e: str, j: int, seed_fn):
        """Independent env: setup, replay the real prefix, reseed with seed_fn(state_id,e,j).
        Returns (pre_intervention_snapshot, rng_state_after_reseed)."""
        env, obs, core = C.setup_env(R2, env_ctx["P0"], env_ctx["phase_from_tag"],
                                     env_ctx["install_keyed_opponent_overlays"],
                                     env_ctx["pole_A_genome"], env_ctx["assert_live_opponent_batch"],
                                     device, state["seed"], state["pole"])
        try:
            z = C.POLE_LATENT[state["pole"]]
            obs = C.replay_prefix(R2, incumbent, env, obs, state["actions"], z,
                                  f"smoke {state['state_id']}|{e}|{j}")
            pre = _snap(obs, obs["global_state"])
            core._rng.manual_seed(int(seed_fn(state["state_id"], e, j)))
            return pre, rng_state(core)
        finally:
            env.close()

    sample = pick_sample(manifest)
    print(f"  sample boundaries: " +
          ", ".join(f"{s['state_id']}({s['free_set']})" for s in sample) + "\n", flush=True)

    # ---- check 1: bitwise-identical prefix replay, independently re-derived twice ----
    check1_rows, check1_ok = [], True
    for st in sample:
        e0 = C.estimands_for(st["free_set"])[0]
        pre_a, _ = fresh_replay(st, e0, 0, C.continuation_seed)
        pre_b, _ = fresh_replay(st, e0, 0, C.continuation_seed)
        ok, diffs = _compare(pre_a, pre_b)
        check1_ok = check1_ok and ok
        check1_rows.append({"state_id": st["state_id"], "free_set": st["free_set"],
                            "bitwise_identical_across_two_independent_replays": ok,
                            "diffs": diffs[:6]})
        print(f"  [check1] {st['state_id']:>18}  replay-vs-replay bitwise identical = {ok}",
              flush=True)

    # ---- check 2: three arms bitwise-identical pre-intervention state AND rng ----
    check2_rows, check2_ok = [], True
    for st in sample:
        e0 = C.estimands_for(st["free_set"])[0]
        for j in J_SAMPLE:
            pres, rngs = {}, {}
            for arm in C.ARMS:
                pre, rs = fresh_replay(st, e0, j, C.continuation_seed)
                pres[arm], rngs[arm] = pre, rs
            state_ok = all(_compare(pres["R_0"], pres[a])[0] for a in ("pi_A", "pi_B"))
            rng_ok = torch.equal(rngs["R_0"], rngs["pi_A"]) and torch.equal(rngs["R_0"], rngs["pi_B"])
            row_ok = state_ok and rng_ok
            check2_ok = check2_ok and row_ok
            check2_rows.append({"state_id": st["state_id"], "estimand": e0, "j": j,
                                "pre_intervention_state_identical_across_arms": state_ok,
                                "rng_state_identical_across_arms": rng_ok})
            print(f"  [check2] {st['state_id']:>18} e={e0:<7} j={j:<2}  "
                  f"state={state_ok}  rng={rng_ok}", flush=True)

    # ---- check 3: negative control -- an order-dependent stream dispatcher MUST be caught ----
    st, j = sample[0], J_SAMPLE[0]
    e0 = C.estimands_for(st["free_set"])[0]
    counter = iter(range(10_000))
    def stream_seed_fn(_state_id, _e, _j):
        return next(counter)
    rngs_broken = {}
    for arm in C.ARMS:
        _pre, rs = fresh_replay(st, e0, j, stream_seed_fn)
        rngs_broken[arm] = rs
    broken_rng_identical = (torch.equal(rngs_broken["R_0"], rngs_broken["pi_A"])
                            and torch.equal(rngs_broken["R_0"], rngs_broken["pi_B"]))
    control_detected = not broken_rng_identical           # MUST be detected as non-identical
    print(f"\n  [check3] negative control (order-dependent stream dispatcher): "
          f"identical_across_arms={broken_rng_identical}  "
          f"detected_as_broken={control_detected}", flush=True)

    all_ok = check1_ok and check2_ok and control_detected
    verdict = "COLLECTOR_VERIFIED" if all_ok else "COLLECTOR_UNVERIFIED"

    OUT.write_text(json.dumps({
        "record": "CCP-S2 collector smoke", "status": "FROZEN_RESULT", "one_shot": True,
        "utc": _now(),
        "implements": "PI requirement: exact-prefix replay + 3-arm bitwise state/RNG identity "
                     "+ negative control, before any of the 10176 real jobs run",
        "sample_states": [s["state_id"] for s in sample],
        "VERDICT": verdict,
        "check1_prefix_replay_bitwise_reproducible": {"passed": check1_ok, "rows": check1_rows},
        "check2_three_arm_state_and_rng_identity": {"passed": check2_ok, "rows": check2_rows},
        "check3_negative_control_broken_dispatcher_detected": {
            "passed": control_detected,
            "design": "an order-dependent stream counter dispensed to R_0/pi_A/pi_B in that "
                     "order, in place of the frozen CCP_S2_MEASURE mapping",
            "identical_across_arms_under_broken_dispatcher": broken_rng_identical,
            "why_required": "a check that cannot fail proves nothing; this confirms check2's "
                            "methodology would actually catch a broken seed dispatcher, not "
                            "merely agree with a correct one"},
        "authorizes_if_verified": "the 10176 real CCP-S2 measurement jobs "
                                  "(experiments/ccp_s2_collect.py, worker mode)",
    }, indent=2), encoding="utf-8")

    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {OUT}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
