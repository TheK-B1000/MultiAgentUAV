"""COMPRESSION CROSSOVER -- the one-shot sealed EVAL of the freshly distilled student.

Implements TEACHER_DISTILLATION_SPEC.json#EVAL_PROTOCOL_COMPRESSION_CROSSOVER. The student was
trained ONLY by teacher KL (no PPO, no reward); this asks whether pure compression of pi_A/pi_B
into one K=2 network already carries their crossover. Same cells, gate, bootstrap and
tie/reversal-audit-first discipline as every sealed EVAL in this program; single arm.

Run:  python experiments/eval_compression_crossover.py --device cuda
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
SPEC = SD / "TEACHER_DISTILLATION_SPEC.json"
FROZEN = SD / "TEACHER_DISTILLATION_STUDENT_FROZEN.json"
SPECIALISTS = SD / "SPECIALIST_BASELINE_EVAL_RESULT.json"
OUT = SD / "COMPRESSION_CROSSOVER_EVAL_RESULT.json"
ROWS_CSV = SD / "compression_crossover_eval_rows.csv"
PREAUDIT_FLAG = SD / "COMPRESSION_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json"

EVAL_SEEDS = list(range(11_922_001, 11_922_065))
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _preflight():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: spec not frozen: {spec['status']!r}")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen["status"] != "FROZEN_STUDENT":
        raise SystemExit(f"REFUSING: student status is {frozen['status']!r}; the eval opens only "
                         "on FROZEN_STUDENT (fit check passed)")
    if not frozen["fit_check"]["passed"]:
        raise SystemExit("REFUSING: fit check did not pass")
    if frozen["EVAL_STATE_AT_FREEZE"]["touched"]:
        raise SystemExit("REFUSING: EVAL block marked touched at freeze")
    ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    if not ck.is_file() or _sha(ck) != frozen["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: student checkpoint missing or sha mismatch")
    if OUT.is_file() or ROWS_CSV.is_file() or PREAUDIT_FLAG.is_file():
        raise SystemExit("REFUSING: a Compression Crossover EVAL output already exists; one-shot")
    block = spec["EVAL_PROTOCOL_COMPRESSION_CROSSOVER"]["block"]
    lo, hi = (int(x) for x in block.split(".."))
    if [EVAL_SEEDS[0], EVAL_SEEDS[-1]] != [lo, hi] or len(EVAL_SEEDS) != 64:
        raise SystemExit(f"REFUSING: evaluator seeds do not match the frozen block {block}")
    return spec, frozen, ck


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

    spec, frozen, ck = _preflight()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    print(f"COMPRESSION CROSSOVER SEALED EVAL  {_now()}")
    print("  student checkpoint sha256 VERIFIED against TEACHER_DISTILLATION_STUDENT_FROZEN.json")
    print(f"  seeds {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]} (n={len(EVAL_SEEDS)})")
    print(f"  gate: delta > 0 AND lcb95 > 0 on BOTH poles")
    print(f"  bootstrap n={N_BOOT}, alpha={ALPHA}, rng_seed={BOOTSTRAP_SEED}\n", flush=True)

    probe = R2.build_env(device, EVAL_SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policy = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
    if policy.model.latent_k != 2 or not policy.model.uses_latent_strategy:
        raise SystemExit("REFUSING: student is not a latent K=2 policy")

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
                                       context=f"compression crossover {pole} z{z} seed {seed}")
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
            rows.append({"arm": "STUDENT", "z": f"z{z}", "pole": pole, "seed": seed,
                         **run_cell(pole, z, seed)})
        wr = np.mean([r["win"] for r in rows if r["z"] == f"z{z}" and r["pole"] == pole])
        print(f"  STUDENT z{z} on Pole {pole}: win rate {wr:.4f}", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

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
            "record": "Compression Crossover EVAL integrity audit REQUIRED", "status": "FLAGGED",
            "utc": _now(), "implements": "TEACHER_DISTILLATION_SPEC.json#EVAL_PROTOCOL_COMPRESSION_CROSSOVER.tie_or_reversal",
            "triggered_by": tie_or_reversal,
            "point_estimates": {"delta_A": delta_a["mean"], "delta_B": delta_b["mean"]},
            "rule": "requires a row-level integrity audit, written before its rows are read, before "
                    f"any verdict. Raw rows: {ROWS_CSV.name}.",
        }, indent=2), encoding="utf-8")
        print(f"\n  TIE/REVERSAL on {tie_or_reversal} -- integrity audit REQUIRED before any verdict.")
        print(f"  -> {PREAUDIT_FLAG}")
        return 0

    gate = bool(delta_a["passes"] and delta_b["passes"])
    matrix = spec["PHASES_AND_THE_THREE_FROZEN_COMPARISONS"]["reading_matrix"]
    reading = ("compression preserved crossover -- Phase 2 (PPO adaptation) is now the question"
               if gate else matrix["phase1_FAIL"])
    ref = json.loads(SPECIALISTS.read_text(encoding="utf-8"))["PRIMARY_GATE"] if SPECIALISTS.is_file() else None

    print("\n  PRIMARY GATE (COMPRESSION CROSSOVER)")
    print(f"    delta_A {delta_a['mean']:+.4f} [{delta_a['lcb95']:+.4f}, {delta_a['ucb95']:+.4f}] "
          f"{'PASS' if delta_a['passes'] else 'FAIL'}")
    print(f"    delta_B {delta_b['mean']:+.4f} [{delta_b['lcb95']:+.4f}, {delta_b['ucb95']:+.4f}] "
          f"{'PASS' if delta_b['passes'] else 'FAIL'}")
    print(f"\n  GATE: {'PASS' if gate else 'FAIL'}\n  READING: {reading}")

    OUT.write_text(json.dumps({
        "record": "Compression Crossover sealed EVAL", "status": "FROZEN_RESULT", "one_shot": True,
        "utc": _now(), "implements": "TEACHER_DISTILLATION_SPEC.json#EVAL_PROTOCOL_COMPRESSION_CROSSOVER",
        "seeds": {"block": [EVAL_SEEDS[0], EVAL_SEEDS[-1]], "n": len(EVAL_SEEDS)},
        "PRIMARY_GATE": {"delta_A": delta_a, "delta_B": delta_b, "passes": gate},
        "READING": reading,
        "student": {"sha256": frozen["TERMINAL_CHECKPOINT"]["sha256"],
                    "final_holdout": frozen.get("final_holdout")},
        "specialist_ceiling_context_only_different_block": ref,
        "bootstrap": {"procedure": "paired percentile bootstrap over evaluation seeds",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
        "no_model_selection_occurred": True, "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
