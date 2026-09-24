"""4v4 B3-3 checkpoint-over-time crossover trajectory (DIAGNOSTIC ONLY).

Implements 4V4_B3_CHECKPOINT_TRAJECTORY_DIAGNOSTIC_SPEC.json.

Question: did a specialization margin exist at an earlier training step and then
collapse, or was it flat and near-zero throughout?

THIS IS NOT A SELECTION PROCEDURE. Scanning checkpoints and keeping the best one
is exactly the thing this program refuses everywhere else. If an intermediate
checkpoint looks better, that is a HYPOTHESIS requiring a new prospectively
frozen experiment with its own seeds -- not a result. The sealed terminal FAIL
stands regardless of what this scan shows.

Run:
  ./.venv/Scripts/python.exe experiments/eval_checkpoint_trajectory_4v4.py --dry-run
  ./.venv/Scripts/python.exe experiments/eval_checkpoint_trajectory_4v4.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "4V4_B3_CHECKPOINT_TRAJECTORY_DIAGNOSTIC_SPEC.json"

N = 4
BASE_KEY = {"A": "OP6", "B": "OP7"}
N_BOOT, ALPHA, BOOT_SEED = 20_000, 0.05, 7


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mean_ci(v):
    v = np.asarray(v, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    b = v[idx].mean(axis=1)
    lo, hi = np.percentile(b, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": round(float(v.mean()), 6),
            "lcb95": round(float(lo), 6), "ucb95": round(float(hi), 6)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")

    steps = [int(s) for s in spec["CHECKPOINT_LADDER"]["steps"]]
    lo, hi = spec["SEEDS"]["block"]
    seeds = list(range(int(lo), int(hi) + 1))
    label = spec["OUTPUT_LABEL"]
    OUT = SD / f"{label}_RESULT.json"
    ROWS = SD / f"{label.lower()}_rows.csv"
    if not args.dry_run and (OUT.is_file() or ROWS.is_file()):
        raise SystemExit(f"REFUSING: output for {label!r} already exists; one-shot.")

    a_dir = ROOT / spec["POLICIES_AND_POLES"]["pi_A_dir"]
    b_dir = ROOT / spec["POLICIES_AND_POLES"]["pi_B_dir"]
    ck = {}
    for s in steps:
        pa = a_dir / f"ckpt_pi_A_specialist_4v4_b3_{s}.zip"
        pb = b_dir / f"ckpt_pi_B_specialist_4v4_b3_{s}.zip"
        for p in (pa, pb):
            if not p.is_file():
                raise SystemExit(f"FAIL-CLOSED: missing checkpoint {p}")
        ck[s] = (pa, pb)

    import torch
    from experiments.opponent_spec import (_with_full_team_defender_gate,
                                           assert_live_opponent_batch,
                                           install_keyed_opponent_overlays,
                                           pole_A_genome)
    from experiments.sds_genome import SDSGenome
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    R2.AGENTS = N
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    gpath = ROOT / "artifacts/strategic_demand/sppo/pole_b2_candidates/B3-3_lockdef10_2v1.json"
    pole_b = _with_full_team_defender_gate(
        SDSGenome.from_dict(json.loads(gpath.read_text(encoding="utf-8"))), N)
    genomes_by_pole = {"A": {"OP6": pole_A_genome(N)}, "B": {"OP7": pole_b}}

    print(f"B3-3 CHECKPOINT TRAJECTORY (DIAGNOSTIC)  {label}  {_now()}")
    print(f"  spec       {SPEC.name}  [{spec['status']}]  arm={spec['arm']}")
    print(f"  ladder     {steps}")
    print(f"  seeds      {seeds[0]}..{seeds[-1]} (n={len(seeds)}), shared across all points")
    print(f"  poles      A: OP6+{dict(pole_A_genome(N).overlay or {})}   "
          f"B: OP7+{dict(pole_b.overlay or {})}")
    print(f"  episodes   {len(steps)} x 4 x {len(seeds)} = {len(steps)*4*len(seeds)}")
    print("  DIAGNOSTIC ONLY -- does NOT license promoting any intermediate checkpoint.\n",
          flush=True)

    probe = R2.build_env(device, seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    if int(obs_space.spaces["grid"].shape[0]) != N:
        probe.close()
        raise SystemExit("FAIL-CLOSED: env grid agent dim != 4")
    probe.close()

    if args.dry_run:
        for pole in ("A", "B"):
            env = R2.build_env(device, 99_990_000 + N)
            try:
                core = env.core
                core._bt_profile_override = None
                install_keyed_opponent_overlays(core, genomes_by_pole[pole])
                env.env_method("set_phase", phase_from_tag(BASE_KEY[pole]))
                env.env_method("set_next_opponent", "SCRIPTED", BASE_KEY[pole])
                env.reset()
                r = core._bt_resolved_profile_tensors().get("min_alive_for_defender")
                got = int(r.flatten()[0].item()) if hasattr(r, "flatten") else int(r)
                print(f"  dry-run pole {pole}: min_alive_for_defender={got} "
                      f"(expect {N}) {'OK' if got == N else 'MISMATCH'}")
                if got != N:
                    raise SystemExit("FAIL-CLOSED: live pole mismatch")
            finally:
                env.close()
        print(f"  all {2*len(steps)} checkpoint files present.")
        print("\n  --dry-run: spec frozen, checkpoints present, poles resolve at N. "
              "NO episodes run, NOTHING written.")
        return 0

    def run_cell(policy, pole, seed):
        env = R2.build_env(device, seed)
        core = env.core
        try:
            policy.reset_strategy()
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            genomes = genomes_by_pole[pole]
            install_keyed_opponent_overlays(core, genomes)
            key = BASE_KEY[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                                       context=f"{label} {pole} seed {seed}")
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

    from experiments.tqdm_loop import set_postfix, tqdm_iter

    cells = [(s, who, pole, sd) for s in steps for who in ("pi_A", "pi_B")
             for pole in ("A", "B") for sd in seeds]
    rows = []
    loaded = {}
    bar = tqdm_iter(cells, desc=label, unit="ep")
    for s, who, pole, sd in bar:
        set_postfix(bar, f"{s//1000}k {who}@{pole} seed={sd}")
        key = (s, who)
        if key not in loaded:
            loaded.clear()
            pa, pb = ck[s]
            loaded[(s, "pi_A")] = load_custom_ppo_policy(str(pa), obs_space, act_space,
                                                         device=device)
            loaded[(s, "pi_B")] = load_custom_ppo_policy(str(pb), obs_space, act_space,
                                                         device=device)
        rows.append({"step": s, "policy": who, "pole": pole, "seed": sd,
                     **run_cell(loaded[key], pole, sd)})
        # Incremental reporting: emit each cell's win rate as it closes, and each
        # checkpoint's delta pair as soon as its four cells are complete. Without
        # this the whole run is silent until the final loop, which makes a multi-hour
        # scan impossible to monitor and yields nothing if it is interrupted.
        if sd == seeds[-1]:
            wr = np.mean([r["win"] for r in rows
                          if r["step"] == s and r["policy"] == who and r["pole"] == pole])
            print(f"    {s//1000}k {who}@Pole{pole}: win rate {wr:.4f}", flush=True)
            if who == "pi_B" and pole == "B":
                def _a(w, p):
                    d = {r["seed"]: r["win"] for r in rows
                         if r["step"] == s and r["policy"] == w and r["pole"] == p}
                    return np.array([d[x] for x in seeds], dtype=np.float64)
                dA, dB = mean_ci(_a("pi_A", "A") - _a("pi_B", "A")), \
                         mean_ci(_a("pi_B", "B") - _a("pi_A", "B"))
                print(f"  step {s:>8,}: dA={dA['mean']:+.4f} "
                      f"[{dA['lcb95']:+.4f},{dA['ucb95']:+.4f}]  "
                      f"dB={dB['mean']:+.4f} "
                      f"[{dB['lcb95']:+.4f},{dB['ucb95']:+.4f}]", flush=True)

    with ROWS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def arr(s, who, pole):
        d = {r["seed"]: r["win"] for r in rows
             if r["step"] == s and r["policy"] == who and r["pole"] == pole}
        return np.array([d[x] for x in seeds], dtype=np.float64)

    traj = {}
    for s in steps:
        aA, bA = arr(s, "pi_A", "A"), arr(s, "pi_B", "A")
        bB, aB = arr(s, "pi_B", "B"), arr(s, "pi_A", "B")
        traj[str(s)] = {
            "cells": {"pi_A@A": round(float(aA.mean()), 4),
                      "pi_B@A": round(float(bA.mean()), 4),
                      "pi_A@B": round(float(aB.mean()), 4),
                      "pi_B@B": round(float(bB.mean()), 4)},
            "delta_A": mean_ci(aA - bA),
            "delta_B": mean_ci(bB - aB),
        }
        d = traj[str(s)]
        print(f"  step {s:>8,}: dA={d['delta_A']['mean']:+.4f} "
              f"[{d['delta_A']['lcb95']:+.4f},{d['delta_A']['ucb95']:+.4f}]  "
              f"dB={d['delta_B']['mean']:+.4f} "
              f"[{d['delta_B']['lcb95']:+.4f},{d['delta_B']['ucb95']:+.4f}]", flush=True)

    OUT.write_text(json.dumps({
        "record": f"{label} checkpoint-over-time crossover trajectory",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "arm": "DIAGNOSTIC", "confirmatory": False,
        "implements": SPEC.name,
        "WHAT_THIS_IS": "diagnostic trajectory of the specialization margin across "
                        "training steps for the B3-3 vanilla pair",
        "WHAT_THIS_IS_NOT": "NOT a checkpoint-selection procedure. No intermediate "
                            "checkpoint may be promoted into a confirmatory result on "
                            "the basis of this scan. The sealed terminal FAIL "
                            "(CONFIRMATORY_B3_3_4V4_SPECIALIST_CROSSOVER_EVAL_RESULT.json) "
                            "stands unreinterpreted.",
        "terminal_point_caveat": "the 1,000,000-step point here is measured on block "
                                 "17300001-024 at n=24 and is NOT comparable to the sealed "
                                 "terminal number measured on 16700001-128 at n=128",
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds),
                  "shared_across_all_points": True},
        "bootstrap": {"n_boot": N_BOOT, "alpha": ALPHA, "rng_seed": BOOT_SEED},
        "trajectory": traj,
        "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}\n  -> {ROWS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
