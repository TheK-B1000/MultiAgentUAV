"""RUNG 0 SEALED EVAL -- the sharing ladder's structural positive control.

Implements SHARING_LADDER_SPEC.json#EVAL_PROTOCOL for rung 0, read under
RUNG0_INTERPRETATION_AMENDMENT.json's corrected three-way interpretation.

Rung 0 has ZERO shared parameters: z0 dispatches the entire forward pass to pi_A, z1 to pi_B.
It therefore cannot be a single latent checkpoint and is evaluated through
rl.rung0_dispatch.Rung0DispatchPolicy, which presents the same interface every other sealed
eval uses. The wrapper was verified bit-exact against both specialists on CUDA before this
seal was allowed to open (RUNG0_EQUIVALENCE_PREFLIGHT.json, VERDICT PASS) -- this script
refuses to run otherwise.

The question: do the exact two successful policies reproduce crossover on a fresh sealed block?

Run:  python experiments/eval_rung0_crossover.py --device cuda
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
LADDER = SD / "SHARING_LADDER_SPEC.json"
AMENDMENT = SD / "RUNG0_INTERPRETATION_AMENDMENT.json"
PREFLIGHT = SD / "RUNG0_EQUIVALENCE_PREFLIGHT.json"
SPECIALISTS = SD / "SPECIALIST_BASELINE_SPEC.json"
CEILING = SD / "SPECIALIST_BASELINE_EVAL_RESULT.json"
OUT = SD / "RUNG0_CROSSOVER_EVAL_RESULT.json"
ROWS_CSV = SD / "rung0_crossover_eval_rows.csv"
PREAUDIT_FLAG = SD / "RUNG0_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json"

EVAL_SEEDS = list(range(11_960_001, 11_960_065))
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _preflight(device: str):
    ladder = json.loads(LADDER.read_text(encoding="utf-8"))
    if ladder["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: ladder spec not frozen: {ladder['status']!r}")
    if device != "cuda":
        raise SystemExit("REFUSING: the ladder's eval protocol fixes cuda")
    pre = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if pre.get("VERDICT") != "PASS":
        raise SystemExit(f"REFUSING: equivalence preflight VERDICT is {pre.get('VERDICT')!r}; "
                         "a wrapper that is not bit-exact is not a positive control")
    if not pre["logit_and_head_equivalence"]["ALL_EXACT"]:
        raise SystemExit("REFUSING: preflight did not certify bit-exact equivalence")
    if OUT.is_file() or ROWS_CSV.is_file() or PREAUDIT_FLAG.is_file():
        raise SystemExit("REFUSING: a Rung 0 EVAL output already exists; one-shot")
    block = ladder["SEEDS"]["rung_0_eval"]
    lo, hi = (int(x) for x in block.split(".."))
    if [EVAL_SEEDS[0], EVAL_SEEDS[-1]] != [lo, hi] or len(EVAL_SEEDS) != 64:
        raise SystemExit(f"REFUSING: evaluator seeds do not match the frozen block {block}")
    tspec = json.loads(SPECIALISTS.read_text(encoding="utf-8"))["MODELS_UNDER_TEST"]
    for n in ("pi_A", "pi_B"):
        p = ROOT / tspec[n]["path"]
        if _sha(p) != tspec[n]["sha256"]:
            raise SystemExit(f"REFUSING: {n} sha mismatch")
    return ladder, tspec, pre


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

    ladder, tspec, pre = _preflight(args.device)
    if not torch.cuda.is_available():
        raise SystemExit("REFUSING: cuda unavailable")
    device = args.device

    print(f"RUNG 0 SEALED EVAL -- sharing-ladder positive control  {_now()}  device={device}")
    print("  wrapper verified BIT-EXACT vs both specialists (max|dlogit|=0, all heads)")
    print(f"  seeds {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]} (n={len(EVAL_SEEDS)})")
    print("  zero shared parameters: z0 -> pi_A entire forward pass, z1 -> pi_B")
    print(f"  gate: delta > 0 AND lcb95 > 0 on BOTH poles")
    print(f"  bootstrap n={N_BOOT}, alpha={ALPHA}, rng_seed={BOOTSTRAP_SEED}\n", flush=True)

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
                                       context=f"rung0 crossover {pole} z{z} seed {seed}")
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
            rows.append({"arm": "RUNG0", "z": f"z{z}", "pole": pole, "seed": seed,
                         **run_cell(pole, z, seed)})
        wr = np.mean([r["win"] for r in rows if r["z"] == f"z{z}" and r["pole"] == pole])
        who = "pi_A" if z == 0 else "pi_B"
        print(f"  RUNG0 z{z} ({who}) on Pole {pole}: win rate {wr:.4f}", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def wins(z, pole):
        by = {r["seed"]: r["win"] for r in rows if r["z"] == z and r["pole"] == pole}
        return np.array([by[s] for s in EVAL_SEEDS], dtype=np.float64)

    delta_a = _mean_ci(wins("z0", "A") - wins("z1", "A"))
    delta_b = _mean_ci(wins("z1", "B") - wins("z0", "B"))
    for d in (delta_a, delta_b):
        d["passes"] = bool(d["mean"] > 0 and d["lcb95"] > 0)

    tie_or_reversal = [k for k, d in (("delta_A", delta_a), ("delta_B", delta_b)) if d["mean"] <= 0.0]
    if tie_or_reversal:
        PREAUDIT_FLAG.write_text(json.dumps({
            "record": "Rung 0 EVAL integrity audit REQUIRED", "status": "FLAGGED", "utc": _now(),
            "triggered_by": tie_or_reversal,
            "point_estimates": {"delta_A": delta_a["mean"], "delta_B": delta_b["mean"]},
            "why_this_is_serious": "RUNG0_INTERPRETATION_AMENDMENT.json reading 3: a bit-exact "
                "dispatch of pi_A and pi_B cannot reverse the specialists' own direction. A tie "
                "or reversal here indicts the wrapper or evaluator, not the architecture. STOP "
                "the ladder and audit before any further rung.",
            "rule": f"row-level integrity audit required before any verdict. Raw rows: {ROWS_CSV.name}.",
        }, indent=2), encoding="utf-8")
        print(f"\n  TIE/REVERSAL on {tie_or_reversal} -- STOP THE LADDER, audit required.")
        print(f"  -> {PREAUDIT_FLAG}")
        return 0

    gate = bool(delta_a["passes"] and delta_b["passes"])
    if gate:
        outcome, reading, action = ("PASS",
            "positive control succeeds; the dispatch path and gate can express crossover",
            "proceed to Rung 1")
    else:
        outcome, reading, action = ("POSITIVE_DELTAS_CI_FAILURE",
            "a POWER / seed-stability outcome, not an instrument failure -- both deltas point "
            "the right way but the block did not resolve one at n=64. Investigate statistical "
            "stability and fresh-block sensitivity BEFORE attributing anything to architecture "
            "or instrumentation.",
            "investigate power; do NOT proceed to Rung 1 until settled, since every later rung "
            "inherits the same n")
    ceiling = json.loads(CEILING.read_text(encoding="utf-8"))["PRIMARY_GATE"] if CEILING.is_file() else None

    print("\n  PRIMARY GATE (RUNG 0 POSITIVE CONTROL)")
    print(f"    delta_A {delta_a['mean']:+.4f} [{delta_a['lcb95']:+.4f}, {delta_a['ucb95']:+.4f}] "
          f"{'PASS' if delta_a['passes'] else 'FAIL'}")
    print(f"    delta_B {delta_b['mean']:+.4f} [{delta_b['lcb95']:+.4f}, {delta_b['ucb95']:+.4f}] "
          f"{'PASS' if delta_b['passes'] else 'FAIL'}")
    if ceiling:
        print(f"\n  specialist baseline, different block (context only): "
              f"delta_A {ceiling['delta_A']['mean']:+.4f} [{ceiling['delta_A']['lcb95']:+.4f}, ...]  "
              f"delta_B {ceiling['delta_B']['mean']:+.4f} [{ceiling['delta_B']['lcb95']:+.4f}, ...]")
    print(f"\n  OUTCOME: {outcome}\n  READING: {reading}\n  ACTION: {action}")

    OUT.write_text(json.dumps({
        "record": "Rung 0 sealed crossover EVAL -- sharing-ladder positive control",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(), "device": device,
        "implements": "SHARING_LADDER_SPEC.json#EVAL_PROTOCOL, read under RUNG0_INTERPRETATION_AMENDMENT.json",
        "architecture": "zero shared parameters; z0 dispatches entire forward pass to pi_A, z1 to pi_B",
        "equivalence_preflight": {"verdict": pre["VERDICT"],
                                  "max_abs_logit_delta": 0.0,
                                  "n_states_checked": pre["n_states_checked"]},
        "seeds": {"block": [EVAL_SEEDS[0], EVAL_SEEDS[-1]], "n": len(EVAL_SEEDS)},
        "PRIMARY_GATE": {"delta_A": delta_a, "delta_B": delta_b, "passes": gate},
        "OUTCOME": outcome, "READING": reading, "ACTION": action,
        "specialist_ceiling_context_only_different_block": ceiling,
        "bootstrap": {"procedure": "paired percentile bootstrap over evaluation seeds",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
        "no_model_selection_occurred": True, "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
