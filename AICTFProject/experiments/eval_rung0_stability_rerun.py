"""Rung 0 STABILITY RERUN -- a second independent n=64 block, diagnostic, non-sealed.

Implements RUNG0_STABILITY_RERUN_SPEC.json. Same bit-exact wrapper, same four cells, same gate
arithmetic and bootstrap as the sealed Rung 0 run, on fresh block 11970001..11970064. This is
NOT a second attempt at the seal: RUNG0_CROSSOVER_EVAL_RESULT.json stands unchanged. The point
is to separate seed-block instability from insufficient power.

Primary reading is PER BLOCK. A preregistered secondary pooled (n=128) estimate over both blocks
is reported as descriptive only.

Run:  python experiments/eval_rung0_stability_rerun.py --device cuda
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
SPEC = SD / "RUNG0_STABILITY_RERUN_SPEC.json"
PREFLIGHT = SD / "RUNG0_EQUIVALENCE_PREFLIGHT.json"
SPECIALISTS = SD / "SPECIALIST_BASELINE_SPEC.json"
BLOCK1_RESULT = SD / "RUNG0_CROSSOVER_EVAL_RESULT.json"
BLOCK1_ROWS = SD / "rung0_crossover_eval_rows.csv"
OUT = SD / "RUNG0_STABILITY_RERUN_RESULT.json"
ROWS_CSV = SD / "rung0_stability_rerun_rows.csv"

EVAL_SEEDS = list(range(11_970_001, 11_970_065))
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _preflight(device: str):
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: stability spec not frozen: {spec['status']!r}")
    if device != "cuda":
        raise SystemExit("REFUSING: the spec fixes cuda to match the sealed run")
    pre = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if pre.get("VERDICT") != "PASS" or not pre["logit_and_head_equivalence"]["ALL_EXACT"]:
        raise SystemExit("REFUSING: wrapper equivalence preflight is not PASS/ALL_EXACT")
    if not BLOCK1_RESULT.is_file() or not BLOCK1_ROWS.is_file():
        raise SystemExit("REFUSING: the sealed block-1 result must exist; this rerun does not replace it")
    if OUT.is_file() or ROWS_CSV.is_file():
        raise SystemExit("REFUSING: a stability rerun output already exists; one-shot")
    lo, hi = (int(x) for x in "11970001..11970064".split(".."))
    if [EVAL_SEEDS[0], EVAL_SEEDS[-1]] != [lo, hi] or len(EVAL_SEEDS) != 64:
        raise SystemExit("REFUSING: evaluator seeds do not match the frozen block")
    b1 = json.loads(BLOCK1_RESULT.read_text(encoding="utf-8"))
    if b1["seeds"]["block"][0] == EVAL_SEEDS[0]:
        raise SystemExit("REFUSING: block 2 must not reuse block 1's seeds")
    tspec = json.loads(SPECIALISTS.read_text(encoding="utf-8"))["MODELS_UNDER_TEST"]
    for n in ("pi_A", "pi_B"):
        if _sha(ROOT / tspec[n]["path"]) != tspec[n]["sha256"]:
            raise SystemExit(f"REFUSING: {n} sha mismatch")
    return spec, tspec, b1


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
    from rl.rung0_dispatch import Rung0DispatchPolicy

    spec, tspec, b1 = _preflight(args.device)
    if not torch.cuda.is_available():
        raise SystemExit("REFUSING: cuda unavailable")
    device = args.device

    print(f"RUNG 0 STABILITY RERUN (block 2, non-sealed diagnostic)  {_now()}  device={device}")
    print(f"  seeds {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]} (n=64)   block 1 was {b1['seeds']['block']}")
    print(f"  block 1: delta_A {b1['PRIMARY_GATE']['delta_A']['mean']:+.4f} "
          f"[{b1['PRIMARY_GATE']['delta_A']['lcb95']:+.4f}, {b1['PRIMARY_GATE']['delta_A']['ucb95']:+.4f}]   "
          f"delta_B {b1['PRIMARY_GATE']['delta_B']['mean']:+.4f} "
          f"[{b1['PRIMARY_GATE']['delta_B']['lcb95']:+.4f}, {b1['PRIMARY_GATE']['delta_B']['ucb95']:+.4f}]\n", flush=True)

    probe = R2.build_env(device, EVAL_SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    specialists = {n: load_custom_ppo_policy(str(ROOT / tspec[n]["path"]), obs_space, act_space,
                                             device=device) for n in ("pi_A", "pi_B")}
    policy = Rung0DispatchPolicy(specialists["pi_A"], specialists["pi_B"])

    def run_cell(pole: str, z: int, seed: int) -> dict:
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
                                       context=f"rung0 stability {pole} z{z} seed {seed}")
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
    for z, pole in ((0, "A"), (1, "A"), (0, "B"), (1, "B")):
        for seed in EVAL_SEEDS:
            rows.append({"arm": "RUNG0_BLOCK2", "z": f"z{z}", "pole": pole, "seed": seed,
                         **run_cell(pole, z, seed)})
        wr = np.mean([r["win"] for r in rows if r["z"] == f"z{z}" and r["pole"] == pole])
        print(f"  RUNG0 z{z} ({'pi_A' if z == 0 else 'pi_B'}) on Pole {pole}: win rate {wr:.4f}", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def deltas(rowset, seeds):
        def wins(z, pole):
            by = {int(r["seed"]): int(r["win"]) for r in rowset if r["z"] == z and r["pole"] == pole}
            return np.array([by[s] for s in seeds], dtype=np.float64)
        da = _mean_ci(wins("z0", "A") - wins("z1", "A"))
        db = _mean_ci(wins("z1", "B") - wins("z0", "B"))
        for d in (da, db):
            d["passes"] = bool(d["mean"] > 0 and d["lcb95"] > 0)
        return da, db

    da2, db2 = deltas(rows, EVAL_SEEDS)

    # preregistered secondary: pooled n=128 over both blocks (descriptive only)
    b1_rows = list(csv.DictReader(BLOCK1_ROWS.open(encoding="utf-8")))
    pooled_rows = b1_rows + rows
    pooled_seeds = sorted({int(r["seed"]) for r in pooled_rows})
    da_p, db_p = deltas(pooled_rows, pooled_seeds)

    b1a, b1b = b1["PRIMARY_GATE"]["delta_A"], b1["PRIMARY_GATE"]["delta_B"]
    same_sign_a = (da2["mean"] > 0) == (b1a["mean"] > 0)
    magnitude_comparable_a = abs(da2["mean"] - b1a["mean"]) <= 0.10   # descriptive band, frozen
    if da2["mean"] <= 0.0:
        reading = ("delta_A collapsed toward zero or changed sign on block 2 -> explanation (1), "
                   "seed-block instability. Rung 1 does NOT proceed; the instability is the next "
                   "object of investigation.")
        verdict = "INSTABILITY"
    elif same_sign_a and magnitude_comparable_a:
        reading = ("delta_A positive again with magnitude comparable to block 1 and to the specialist "
                   "baseline -> explanation (2), block 1's CI miss was sampling noise at n=64. "
                   "Mechanism stable across blocks. LICENSES proceeding to Rung 1."
                   + ("" if da2["passes"] else " (block 2's own CI did not clear zero, which per the "
                                                "spec does not overturn this reading.)"))
        verdict = "STABLE_LICENSE_RUNG1"
    else:
        reading = ("delta_A positive but with materially different magnitude from block 1 -> report "
                   "as-is; partial stability, PI reading required before Rung 1.")
        verdict = "PARTIAL_STABILITY_PI_READ"

    print("\n  BLOCK 2 (this run, per-block PRIMARY reading)")
    print(f"    delta_A {da2['mean']:+.4f} [{da2['lcb95']:+.4f}, {da2['ucb95']:+.4f}] {'PASS' if da2['passes'] else 'FAIL'}")
    print(f"    delta_B {db2['mean']:+.4f} [{db2['lcb95']:+.4f}, {db2['ucb95']:+.4f}] {'PASS' if db2['passes'] else 'FAIL'}")
    print("\n  POOLED n=128 over both blocks (preregistered SECONDARY, descriptive only, NOT the gate)")
    print(f"    delta_A {da_p['mean']:+.4f} [{da_p['lcb95']:+.4f}, {da_p['ucb95']:+.4f}]")
    print(f"    delta_B {db_p['mean']:+.4f} [{db_p['lcb95']:+.4f}, {db_p['ucb95']:+.4f}]")
    print(f"\n  VERDICT: {verdict}\n  READING: {reading}")

    OUT.write_text(json.dumps({
        "record": "Rung 0 stability rerun (block 2, non-sealed diagnostic)", "status": "FROZEN_RESULT",
        "one_shot": True, "utc": _now(), "device": device, "implements": "RUNG0_STABILITY_RERUN_SPEC.json",
        "is_not_a_reseal": "RUNG0_CROSSOVER_EVAL_RESULT.json stands unchanged as the sealed block-1 result",
        "block2_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1]],
        "BLOCK2_PRIMARY": {"delta_A": da2, "delta_B": db2},
        "BLOCK1_FOR_COMPARISON": {"delta_A": b1a, "delta_B": b1b, "seeds": b1["seeds"]["block"]},
        "POOLED_N128_SECONDARY_DESCRIPTIVE_NOT_GATE": {"delta_A": da_p, "delta_B": db_p, "n_per_cell": len(pooled_seeds)},
        "stability_checks": {"delta_A_same_sign_as_block1": bool(same_sign_a),
                             "delta_A_magnitude_within_0.10_of_block1": bool(magnitude_comparable_a)},
        "VERDICT": verdict, "READING": reading,
        "bootstrap": {"procedure": "paired percentile bootstrap over evaluation seeds",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
        "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
