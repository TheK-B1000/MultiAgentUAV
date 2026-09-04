"""SAC-RFT sealed EVAL: one-shot held-out payoff test for EMA_CONTROL vs ANCHOR_TREATMENT.

Implements SAC_RFT_SPEC.json#EVAL_PROTOCOL. Same gate / bootstrap as RSCFT; fresh seed block
11804001..11804064.

Run:  python experiments/eval_sac_rft.py --device cuda
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
SPEC = SD / "SAC_RFT_SPEC.json"
FROZEN = SD / "SAC_RFT_MODELS_FROZEN.json"
OUT = SD / "SAC_RFT_EVAL_RESULT.json"
ROWS_CSV = SD / "sac_rft_eval_rows.csv"
PREAUDIT_FLAG = SD / "SAC_RFT_EVAL_INTEGRITY_REQUIRED.json"

EVAL_SEEDS = list(range(11_804_001, 11_804_065))
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7
ARMS = ("CONTROL", "TREATMENT")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _preflight():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] not in ("ACTIVATED", "FROZEN_BEFORE_IMPLEMENTATION",
                              "FROZEN_PENDING_RSCFT_FAIL_ACTIVATION"):
        raise SystemExit(f"REFUSING: SAC-RFT spec status unexpected: {spec['status']!r}")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if not frozen["status"].startswith("FROZEN_MODELS"):
        raise SystemExit(f"REFUSING: models not frozen: {frozen['status']!r}")
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
        raise SystemExit("REFUSING: a SAC-RFT EVAL output already exists; EVAL is one-shot")
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

    print(f"SAC-RFT SEALED EVAL  {_now()}")
    print("  both terminal checkpoints sha256 VERIFIED against SAC_RFT_MODELS_FROZEN.json")
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
                                       context=f"sac_rft eval {pole} z{z} seed {seed}")
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
            "record": "SAC-RFT EVAL integrity audit REQUIRED", "status": "FLAGGED",
            "utc": _now(),
            "implements": "SAC_RFT_SPEC.json#EVAL_PROTOCOL.tie_or_reversal",
            "triggered_by": tie_or_reversal,
            "point_estimates": {k: delta[k]["mean"] for k in delta},
            "rule": "requires a row-level integrity audit before any verdict-bearing "
                    f"SAC_RFT_EVAL_RESULT.json. Raw rows: {ROWS_CSV.name}.",
        }, indent=2), encoding="utf-8")
        print(f"\n  TIE/REVERSAL on {tie_or_reversal} -- integrity audit REQUIRED.")
        print(f"  -> {PREAUDIT_FLAG}")
        return 0

    retention_pass = bool(delta["A_T"]["passes"] and delta["B_T"]["passes"])
    control_pass = bool(delta["A_C"]["passes"] and delta["B_C"]["passes"])
    gamma_pass = bool(gamma_a["passes"] and gamma_b["passes"])

    table = {(True, False): "frozen strategic anchor recovered crossover where EMA retention did not",
             (True, True): "crossover recovered under both teachers; frozen anchor not necessary under this run",
             (False, False): "frozen strategic anchor also insufficient"}
    reading = table.get((retention_pass, control_pass))
    unmatched = reading is None
    if unmatched:
        reading = (f"UNMATCHED COMBINATION (anchor={'PASS' if retention_pass else 'FAIL'}, "
                   f"ema_control={'PASS' if control_pass else 'FAIL'}). Recorded as such.")
    qualifier = None
    if retention_pass and not gamma_pass:
        qualifier = ("claim recovery under the complete treatment, not statistically "
                     "established anchor attribution")

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

    OUT.write_text(json.dumps({
        "record": "SAC-RFT sealed EVAL", "status": "FROZEN_RESULT", "one_shot": True,
        "utc": _now(), "implements": "SAC_RFT_SPEC.json#EVAL_PROTOCOL",
        "seeds": {"block": [EVAL_SEEDS[0], EVAL_SEEDS[-1]], "n": len(EVAL_SEEDS),
                  "shared_across_arms": True},
        "PRIMARY_GATE": {
            "TREATMENT_frozen_anchor": {"delta_A": delta["A_T"], "delta_B": delta["B_T"],
                                        "passes": retention_pass},
            "CONTROL_ema": {"delta_A": delta["A_C"], "delta_B": delta["B_C"],
                            "passes": control_pass}},
        "ATTRIBUTION": {"Gamma_A": gamma_a, "Gamma_B": gamma_b, "passes": gamma_pass},
        "READING": reading, "QUALIFIER": qualifier,
        "unmatched_interpretation_combination": unmatched,
        "checkpoints": {a: frozen[a]["TERMINAL_CHECKPOINT"]["sha256"] for a in ARMS},
        "bootstrap": {"procedure": "paired percentile bootstrap over evaluation seeds",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
        "no_model_selection_occurred": True, "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
