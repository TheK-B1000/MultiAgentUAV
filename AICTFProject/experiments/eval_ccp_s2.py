"""CCP-S2 sealed EVAL: the one-shot held-out payoff test for both matched arms.

Implements CCP_S2_SPEC.json#EVAL_PROTOCOL against both terminal checkpoints frozen in
CCP_S2_MODELS_FROZEN.json. Reuses _mean_ci from eval_hog_psp_v3 UNCHANGED -- the same paired
percentile bootstrap (seed as the resampling unit, n_boot=20000, alpha=0.05, rng_seed=7) every
EVAL in this program has used.

PRIMARY_CROSSOVER_GATE, per arm:
    delta_A = V(z0,A) - V(z1,A)      criterion: LCB95(delta_A) > 0
    delta_B = V(z1,B) - V(z0,B)      criterion: LCB95(delta_B) > 0

ATTRIBUTION_REFINEMENT (why it exists, verbatim from the spec: "'treatment passes, control
fails' is not by itself proof the causal loss beat continued PPO"):
    Gamma_A = delta_A_T - delta_A_C          Gamma_B = delta_B_T - delta_B_C
Both arms are scored on the SAME 64 seeds (CCP_S2_SPEC.json#EVAL_PROTOCOL.
shared_seeds_across_arms), so Gamma's per-seed paired difference
    d_A(s) = [win_T(z0,A,s) - win_T(z1,A,s)] - [win_C(z0,A,s) - win_C(z1,A,s)]
is itself a seed-indexed array, and _mean_ci's existing bootstrap -- resampling that same
seed dimension -- IS the preregistered paired bootstrap the spec calls for. No new bootstrap
machinery is written here.

PREREGISTERED_INTERPRETATION_MATRIX is applied verbatim, in order, and its own rule is
enforced: no row is discarded for producing a less favourable narrative.

exact_tie_or_reversal_on_either_arm: if any of the four deltas is exactly 0.0 (tie) or
negative (reversal), this run REFUSES to write a verdict-bearing final record and instead
writes ONLY the raw per-episode rows plus a flagged pre-audit note -- CCP_S2_SPEC.json
requires "a row-level integrity audit, written before its rows are read" in that case, and an
audit written after inventing a specific hypothesis from having already read the rows would
not satisfy that rule.

One-shot: this EVAL runs exactly once. Re-running after any outcome is not permitted by the
frozen spec, regardless of which arm produced it.

Run:  python experiments/eval_ccp_s2.py --device cuda
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
SPEC = SD / "CCP_S2_SPEC.json"
FROZEN = SD / "CCP_S2_MODELS_FROZEN.json"
OUT = SD / "CCP_S2_EVAL_RESULT.json"
ROWS_CSV = SD / "ccp_s2_eval_rows.csv"
PREAUDIT_FLAG = SD / "CCP_S2_EVAL_INTEGRITY_REQUIRED.json"

EVAL_SEEDS = list(range(11_701_001, 11_701_065))
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7
POLES = ("A", "B")
ARMS = ("CONTROL", "TREATMENT")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _preflight() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: S2 spec not frozen: {spec['status']!r}")
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
        actual = _sha(ck)
        if actual != rec["TERMINAL_CHECKPOINT"]["sha256"]:
            raise SystemExit(f"REFUSING: {arm} terminal checkpoint sha mismatch")
        if rec["TERMINAL_RECORD_VALIDITY"]["verdict"] != "VALID":
            raise SystemExit(f"REFUSING: {arm} training run not VALID")

    if OUT.is_file() or ROWS_CSV.is_file() or PREAUDIT_FLAG.is_file():
        raise SystemExit("REFUSING: a CCP-S2 EVAL output already exists; EVAL is one-shot")
    ep = spec["EVAL_PROTOCOL"]
    if ep["seeds_per_cell"] != 64:
        raise SystemExit("REFUSING: seeds_per_cell drifted from the frozen spec")
    if [EVAL_SEEDS[0], EVAL_SEEDS[-1]] != [11_701_001, 11_701_064] or len(EVAL_SEEDS) != 64:
        raise SystemExit("REFUSING: evaluator seed range does not match the frozen block")
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

    print(f"CCP-S2 SEALED EVAL  {_now()}")
    print("  both terminal checkpoints sha256 VERIFIED against CCP_S2_MODELS_FROZEN.json")
    print(f"  seeds {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]}  (n={len(EVAL_SEEDS)}), SHARED across arms")
    print(f"  gate: BOTH delta lcb95 > 0 per arm; Gamma: BOTH lcb95 > 0")
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
        if not getattr(pol.model.critic, "private_z_heads", False):
            raise SystemExit(f"REFUSING: {arm} critic does not have private latent heads")
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
                                       context=f"s2 eval {pole} z{z} seed {seed}")
            terminal = None
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _reward, done, info = env.step_wait()
                obs["global_state"] = env.state()
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    result = (i0 or {}).get("episode_result") or {}
                    terminal = (int(result.get("blue_score", 0)), int(result.get("red_score", 0)))
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
                res = run_cell(policies[arm], pole, z, seed)
                rows.append({"arm": arm, "z": f"z{z}", "pole": pole, "seed": seed, **res})
            wr = np.mean([r["win"] for r in rows
                          if r["arm"] == arm and r["z"] == f"z{z}" and r["pole"] == pole])
            print(f"  {arm:9s} z{z} on Pole {pole}: win rate {wr:.4f}", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def wins(arm: str, z: str, pole: str) -> np.ndarray:
        by_seed = {r["seed"]: r["win"] for r in rows
                   if r["arm"] == arm and r["z"] == z and r["pole"] == pole}
        return np.array([by_seed[s] for s in EVAL_SEEDS], dtype=np.float64)

    delta = {}
    for arm in ARMS:
        da = wins(arm, "z0", "A") - wins(arm, "z1", "A")
        db = wins(arm, "z1", "B") - wins(arm, "z0", "B")
        delta[f"A_{arm[0]}"] = {**_mean_ci(da), "_raw": da}
        delta[f"B_{arm[0]}"] = {**_mean_ci(db), "_raw": db}
    for k, d in delta.items():
        d["passes"] = d["lcb95"] > 0

    gamma_a_raw = delta["A_T"]["_raw"] - delta["A_C"]["_raw"]
    gamma_b_raw = delta["B_T"]["_raw"] - delta["B_C"]["_raw"]
    gamma_a = _mean_ci(gamma_a_raw)
    gamma_b = _mean_ci(gamma_b_raw)
    gamma_a["passes"] = gamma_a["lcb95"] > 0
    gamma_b["passes"] = gamma_b["lcb95"] > 0

    # --- exact_tie_or_reversal_on_either_arm gate, checked BEFORE any verdict is derived
    tie_or_reversal = [k for k, d in delta.items() if d["mean"] <= 0.0]
    if tie_or_reversal:
        PREAUDIT_FLAG.write_text(json.dumps({
            "record": "CCP-S2 EVAL integrity audit REQUIRED", "status": "FLAGGED",
            "utc": _now(),
            "implements": "CCP_S2_SPEC.json#EVAL_PROTOCOL.exact_tie_or_reversal_on_either_arm",
            "triggered_by": tie_or_reversal,
            "point_estimates": {k: delta[k]["mean"] for k in delta},
            "rule": "requires a row-level integrity audit, written before its rows are read, "
                    "the same standing rule as OG-PSP's GENUINE_TIE and the sealed "
                    "predecessor's GENUINE_REVERSAL. Raw rows are written "
                    f"({ROWS_CSV.name}) but no verdict-bearing CCP_S2_EVAL_RESULT.json is "
                    "written until that audit exists.",
            "next": "write experiments/verify_ccp_s2_eval_integrity.py's CHECKS before "
                    "reading ccp_s2_eval_rows.csv in detail, mirroring "
                    "verify_ccp_successor_eval_integrity.py's five-check structure",
        }, indent=2), encoding="utf-8")
        print(f"\n  TIE/REVERSAL on {tie_or_reversal} -- integrity audit REQUIRED before any "
              f"verdict. Raw rows written to {ROWS_CSV.name}.")
        print(f"  -> {PREAUDIT_FLAG}")
        return 0

    treatment_pass = bool(delta["A_T"]["passes"] and delta["B_T"]["passes"])
    control_pass = bool(delta["A_C"]["passes"] and delta["B_C"]["passes"])
    gamma_pass = bool(gamma_a["passes"] and gamma_b["passes"])

    matrix = spec["EVAL_PROTOCOL"]["PREREGISTERED_INTERPRETATION_MATRIX"]

    def _row_matches(row) -> bool:
        t_ok = (row["treatment"] == "PASS") == treatment_pass
        c_ok = (row["control"] == "PASS") == control_pass
        if row["gamma"] == "any":
            g_ok = True
        elif row["gamma"] == "both PASS":
            g_ok = gamma_pass
        else:
            g_ok = not gamma_pass
        return t_ok and c_ok and g_ok

    matched = [r for r in matrix if _row_matches(r)]
    if len(matched) != 1:
        raise SystemExit(f"REFUSING: {len(matched)} interpretation-matrix rows matched "
                         f"(expected exactly 1) for treatment={treatment_pass} "
                         f"control={control_pass} gamma={gamma_pass}")
    reading = matched[0]["reading"]

    print("\n  PRIMARY CROSSOVER GATE")
    for arm in ARMS:
        a, b = delta[f"A_{arm[0]}"], delta[f"B_{arm[0]}"]
        print(f"    {arm:9s} delta_A {a['mean']:+.4f} [{a['lcb95']:+.4f}, {a['ucb95']:+.4f}] "
              f"{'PASS' if a['passes'] else 'FAIL'}   "
              f"delta_B {b['mean']:+.4f} [{b['lcb95']:+.4f}, {b['ucb95']:+.4f}] "
              f"{'PASS' if b['passes'] else 'FAIL'}")
    print(f"\n  ATTRIBUTION REFINEMENT")
    print(f"    Gamma_A {gamma_a['mean']:+.4f} [{gamma_a['lcb95']:+.4f}, {gamma_a['ucb95']:+.4f}] "
          f"{'PASS' if gamma_a['passes'] else 'FAIL'}")
    print(f"    Gamma_B {gamma_b['mean']:+.4f} [{gamma_b['lcb95']:+.4f}, {gamma_b['ucb95']:+.4f}] "
          f"{'PASS' if gamma_b['passes'] else 'FAIL'}")
    print(f"\n  READING: {reading}")

    def _strip(d):
        return {k: v for k, v in d.items() if k != "_raw"}

    OUT.write_text(json.dumps({
        "record": "CCP-S2 sealed EVAL", "status": "FROZEN_RESULT", "one_shot": True,
        "utc": _now(), "implements": "CCP_S2_SPEC.json#EVAL_PROTOCOL",
        "seeds": {"block": [EVAL_SEEDS[0], EVAL_SEEDS[-1]], "n": len(EVAL_SEEDS),
                 "shared_across_arms": True},
        "PRIMARY_CROSSOVER_GATE": {
            "TREATMENT": {"delta_A": _strip(delta["A_T"]), "delta_B": _strip(delta["B_T"]),
                         "passes": treatment_pass},
            "CONTROL": {"delta_A": _strip(delta["A_C"]), "delta_B": _strip(delta["B_C"]),
                       "passes": control_pass},
        },
        "ATTRIBUTION_REFINEMENT": {"Gamma_A": _strip(gamma_a), "Gamma_B": _strip(gamma_b),
                                   "passes": gamma_pass},
        "READING": reading,
        "checkpoints": {arm: {"sha256": frozen[arm]["TERMINAL_CHECKPOINT"]["sha256"],
                              "path": frozen[arm]["TERMINAL_CHECKPOINT"]["path"]}
                        for arm in ARMS},
        "bootstrap": {"procedure": "paired percentile bootstrap over evaluation seeds",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED,
                      "provenance": "unchanged from V4/V3/OG-PSP/V1/successor"},
        "prelaunch_interpretation_caveats": "CCP_S2_PRELAUNCH_INTERPRETATION_CAVEATS.json, "
            "frozen before this result existed -- z0/z1 supervision imbalance (177 vs 51) and "
            "near-zero warm-start entropy both apply to reading this result",
        "no_model_selection_occurred": True,
        "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")

    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
