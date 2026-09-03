"""RSCFT sealed EVAL: the one-shot held-out payoff test for both retention arms.

Implements RSCFT_SPEC.json#EVAL_PROTOCOL against both terminal checkpoints frozen in
RSCFT_MODELS_FROZEN.json. Reuses _mean_ci from eval_hog_psp_v3 UNCHANGED -- the same paired
percentile bootstrap (seed as resampling unit, n_boot=20000, alpha=0.05, rng_seed=7) every
EVAL in this program has used.

PRIMARY GATE, per arm (RSCFT_SPEC.json: "same primary gate. No mercy, no reinterpretation."):
    delta_A = V(z0,A) - V(z1,A)        delta_B = V(z1,B) - V(z0,B)
    pass iff delta > 0 AND LCB95(delta) > 0, on BOTH poles

ATTRIBUTION:
    Gamma_A = delta_A_retention - delta_A_control
    Gamma_B = delta_B_retention - delta_B_control
Both arms are scored on the SAME 64 seeds, so each Gamma's per-seed paired difference is
itself a seed-indexed array and _mean_ci's existing bootstrap over that dimension IS the
preregistered paired bootstrap. No new statistical machinery.

INTERPRETATION is applied from the frozen list, and -- unlike CCP-S2's six-row matrix --
RSCFT's four rows do NOT partition every outcome: there is no row for
(retention FAIL, control PASS). That combination is reported explicitly as UNMATCHED rather
than coerced into the nearest row, because inventing a reading after the fact is exactly what
preregistration exists to prevent.

exact_tie_or_reversal_on_either_arm: if any of the four deltas is <= 0, this run REFUSES to
write a verdict-bearing record and writes only the raw rows plus a flag requiring the
row-level integrity audit first -- the same standing rule CCP-S2 was held to.

One-shot: runs exactly once, no rerun under any outcome.

Run:  python experiments/eval_rscft.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval_hog_psp_v3 import _mean_ci

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "RSCFT_SPEC.json"
FROZEN = SD / "RSCFT_MODELS_FROZEN.json"
OUT = SD / "RSCFT_EVAL_RESULT.json"
ROWS_CSV = SD / "rscft_eval_rows.csv"
PREAUDIT_FLAG = SD / "RSCFT_EVAL_INTEGRITY_REQUIRED.json"

EVAL_SEEDS = list(range(11_704_001, 11_704_065))
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7
ARMS = ("CONTROL", "TREATMENT")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _preflight():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: RSCFT spec not frozen: {spec['status']!r}")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if not frozen["status"].startswith("FROZEN_MODELS"):
        raise SystemExit(f"REFUSING: models not frozen: {frozen['status']!r}")
    if frozen["EVAL_STATE_AT_FREEZE"]["touched"]:
        raise SystemExit("REFUSING: EVAL block was marked touched at model freeze")
    for arm in ARMS:
        rec = frozen[arm]
        ck = ROOT / rec["TERMINAL_CHECKPOINT"]["path"]
        if not ck.is_file():
            raise SystemExit(f"REFUSING: {arm} terminal checkpoint missing: {ck}")
        if _sha(ck) != rec["TERMINAL_CHECKPOINT"]["sha256"]:
            raise SystemExit(f"REFUSING: {arm} terminal checkpoint sha mismatch")
        if rec["TERMINAL_RECORD_VALIDITY"]["verdict"] != "VALID":
            raise SystemExit(f"REFUSING: {arm} training run not VALID")
    if OUT.is_file() or ROWS_CSV.is_file() or PREAUDIT_FLAG.is_file():
        raise SystemExit("REFUSING: an RSCFT EVAL output already exists; EVAL is one-shot")
    block = spec["SEEDS"]["sealed_eval_block"]
    lo, hi = (int(x) for x in block.split(".."))
    if [EVAL_SEEDS[0], EVAL_SEEDS[-1]] != [lo, hi] or len(EVAL_SEEDS) != 64:
        raise SystemExit(f"REFUSING: evaluator seeds do not match the frozen block {block}")
    return spec, frozen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    spec, frozen = _preflight()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    print(f"RSCFT SEALED EVAL  {_now()}")
    print("  both terminal checkpoints sha256 VERIFIED against RSCFT_MODELS_FROZEN.json")
    print(f"  seeds {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]} (n={len(EVAL_SEEDS)}), SHARED across arms")
    print(f"  gate: delta > 0 AND lcb95 > 0 on BOTH poles, per arm")
    print(f"  bootstrap n={N_BOOT}, alpha={ALPHA}, rng_seed={BOOTSTRAP_SEED}\n", flush=True)

    probe = R2.build_env(device, EVAL_SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    policies = {}
    for arm in ARMS:
        ck = ROOT / frozen[arm]["TERMINAL_CHECKPOINT"]["path"]
        pol = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
        if pol.model.latent_k != 2 or not pol.model.uses_latent_strategy:
            raise SystemExit(f"REFUSING: {arm} checkpoint is not a latent K=2 policy")
        policies[arm] = pol

    def run_cell(policy, pole: str, z: int, seed: int) -> dict:
        env = R2.build_env(device, seed)
        core = env.core
        try:
            policy.fixed_latent_strategy = True
            policy.fixed_latent_strategy_id = int(z)
            policy.reset_strategy()
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            genomes = {"OP6": pole_A_genome()} if pole == "A" else {}
            install_keyed_opponent_overlays(core, genomes)
            key = P0.POLES[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                                       context=f"rscft eval {pole} z{z} seed {seed}")
            terminal = None
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    res = (i0 or {}).get("episode_result") or {}
                    terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                    break
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            blue, red = terminal
            return {"blue": blue, "red": red, "win": int(blue > red), "margin": blue - red}
        finally:
            env.close()

    rows = []
    for arm in ARMS:
        for z, pole in ((0, "A"), (1, "A"), (0, "B"), (1, "B")):
            for seed in EVAL_SEEDS:
                rows.append({"arm": arm, "z": f"z{z}", "pole": pole, "seed": seed,
                             **run_cell(policies[arm], pole, z, seed)})
            wr = np.mean([r["win"] for r in rows
                          if r["arm"] == arm and r["z"] == f"z{z}" and r["pole"] == pole])
            print(f"  {arm:9s} z{z} on Pole {pole}: win rate {wr:.4f}", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def wins(arm, z, pole):
        by = {r["seed"]: r["win"] for r in rows
              if r["arm"] == arm and r["z"] == z and r["pole"] == pole}
        return np.array([by[s] for s in EVAL_SEEDS], dtype=np.float64)

    raw, delta = {}, {}
    for arm in ARMS:
        raw[f"A_{arm[0]}"] = wins(arm, "z0", "A") - wins(arm, "z1", "A")
        raw[f"B_{arm[0]}"] = wins(arm, "z1", "B") - wins(arm, "z0", "B")
    for k, v in raw.items():
        d = _mean_ci(v)
        d["passes"] = bool(d["mean"] > 0 and d["lcb95"] > 0)
        delta[k] = d

    gamma_a = _mean_ci(raw["A_T"] - raw["A_C"]); gamma_a["passes"] = gamma_a["lcb95"] > 0
    gamma_b = _mean_ci(raw["B_T"] - raw["B_C"]); gamma_b["passes"] = gamma_b["lcb95"] > 0

    tie_or_reversal = [k for k, d in delta.items() if d["mean"] <= 0.0]
    if tie_or_reversal:
        PREAUDIT_FLAG.write_text(json.dumps({
            "record": "RSCFT EVAL integrity audit REQUIRED", "status": "FLAGGED", "utc": _now(),
            "implements": "RSCFT_SPEC.json#EVAL_PROTOCOL.tie_or_reversal",
            "triggered_by": tie_or_reversal,
            "point_estimates": {k: delta[k]["mean"] for k in delta},
            "rule": "requires a row-level integrity audit, written before its rows are read. "
                    f"Raw rows are written ({ROWS_CSV.name}) but no verdict-bearing "
                    "RSCFT_EVAL_RESULT.json is written until that audit exists.",
        }, indent=2), encoding="utf-8")
        print(f"\n  TIE/REVERSAL on {tie_or_reversal} -- integrity audit REQUIRED before any "
              f"verdict. Raw rows written to {ROWS_CSV.name}.")
        print(f"  -> {PREAUDIT_FLAG}")
        return 0

    retention_pass = bool(delta["A_T"]["passes"] and delta["B_T"]["passes"])
    control_pass = bool(delta["A_C"]["passes"] and delta["B_C"]["passes"])
    gamma_pass = bool(gamma_a["passes"] and gamma_b["passes"])

    table = {(True, False): "crossover recovered under retention stabilization",
             (True, True): "crossover recovered, but retention is not necessary under this run",
             (False, False): "retention stabilization insufficient"}
    reading = table.get((retention_pass, control_pass))
    unmatched = reading is None
    if unmatched:
        reading = (f"UNMATCHED COMBINATION (retention={'PASS' if retention_pass else 'FAIL'}, "
                   f"control={'PASS' if control_pass else 'FAIL'}). RSCFT_SPEC.json's frozen "
                   "interpretation list contains no row for this outcome. Recorded as such "
                   "rather than coerced into the nearest row; it requires an explicit PI "
                   "reading.")
    qualifier = None
    if retention_pass and not gamma_pass:
        qualifier = ("claim recovery under the complete treatment, not statistically "
                     "established retention attribution")

    print("\n  PRIMARY GATE")
    for arm in ARMS:
        a, b = delta[f"A_{arm[0]}"], delta[f"B_{arm[0]}"]
        print(f"    {arm:9s} delta_A {a['mean']:+.4f} [{a['lcb95']:+.4f}, {a['ucb95']:+.4f}] "
              f"{'PASS' if a['passes'] else 'FAIL'}   "
              f"delta_B {b['mean']:+.4f} [{b['lcb95']:+.4f}, {b['ucb95']:+.4f}] "
              f"{'PASS' if b['passes'] else 'FAIL'}")
    print(f"\n  ATTRIBUTION")
    print(f"    Gamma_A {gamma_a['mean']:+.4f} [{gamma_a['lcb95']:+.4f}, {gamma_a['ucb95']:+.4f}]"
          f" {'PASS' if gamma_a['passes'] else 'FAIL'}")
    print(f"    Gamma_B {gamma_b['mean']:+.4f} [{gamma_b['lcb95']:+.4f}, {gamma_b['ucb95']:+.4f}]"
          f" {'PASS' if gamma_b['passes'] else 'FAIL'}")
    print(f"\n  READING: {reading}")
    if qualifier:
        print(f"  QUALIFIER: {qualifier}")

    OUT.write_text(json.dumps({
        "record": "RSCFT sealed EVAL", "status": "FROZEN_RESULT", "one_shot": True,
        "utc": _now(), "implements": "RSCFT_SPEC.json#EVAL_PROTOCOL",
        "seeds": {"block": [EVAL_SEEDS[0], EVAL_SEEDS[-1]], "n": len(EVAL_SEEDS),
                  "shared_across_arms": True},
        "PRIMARY_GATE": {
            "TREATMENT_retention": {"delta_A": delta["A_T"], "delta_B": delta["B_T"],
                                    "passes": retention_pass},
            "CONTROL_causal_only": {"delta_A": delta["A_C"], "delta_B": delta["B_C"],
                                    "passes": control_pass}},
        "ATTRIBUTION": {"Gamma_A": gamma_a, "Gamma_B": gamma_b, "passes": gamma_pass},
        "READING": reading, "QUALIFIER": qualifier,
        "unmatched_interpretation_combination": unmatched,
        "checkpoints": {a: frozen[a]["TERMINAL_CHECKPOINT"]["sha256"] for a in ARMS},
        "bootstrap": {"procedure": "paired percentile bootstrap over evaluation seeds",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
        "prior_caveats_still_apply": "CCP_S2_PRELAUNCH_INTERPRETATION_CAVEATS.json -- the "
            "z0/z1 supervision imbalance (177 vs 51) is unchanged, since RSCFT reuses the "
            "exact same frozen causal bank",
        "no_model_selection_occurred": True, "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
