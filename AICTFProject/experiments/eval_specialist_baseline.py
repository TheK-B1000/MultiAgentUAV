"""Separate-specialists baseline, formal sealed EVAL.

Implements SPECIALIST_BASELINE_SPEC.json. Same tie/reversal-triggers-an-audit-first discipline
as every sealed EVAL in this program, applied here for the first time to the ceiling reference
rather than a novel treatment -- a reversal here would be a serious finding about the opponent
poles or evaluator, not something to wave through because the baseline "should" pass.

Structurally: episode-running code is eval_specialist_anchor.py's (policy.reset_strategy(),
no z-forcing, since neither checkpoint has a latent mechanism). Tie/audit-gating structure is
eval_trunk_freeze.py's / eval_sac_rft.py's two-policy pattern, adapted to pi_A/pi_B in place of
CONTROL/TREATMENT.

Run:  python experiments/eval_specialist_baseline.py --device cuda
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
SPEC = SD / "SPECIALIST_BASELINE_SPEC.json"
OUT = SD / "SPECIALIST_BASELINE_EVAL_RESULT.json"
ROWS_CSV = SD / "specialist_baseline_eval_rows.csv"
PREAUDIT_FLAG = SD / "SPECIALIST_BASELINE_EVAL_INTEGRITY_REQUIRED.json"

EVAL_SEEDS = list(range(11_805_001, 11_805_065))
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7
POLICIES = ("pi_A", "pi_B")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _preflight():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_EVAL_IS_OPENED":
        raise SystemExit(f"REFUSING: spec not frozen: {spec['status']!r}")
    paths = {}
    for name in POLICIES:
        rec = spec["MODELS_UNDER_TEST"][name]
        ck = ROOT / rec["path"]
        if not ck.is_file():
            raise SystemExit(f"REFUSING: {name} checkpoint missing: {ck}")
        if _sha(ck) != rec["sha256"]:
            raise SystemExit(f"REFUSING: {name} checkpoint sha mismatch")
        paths[name] = ck
    if OUT.is_file() or ROWS_CSV.is_file() or PREAUDIT_FLAG.is_file():
        raise SystemExit("REFUSING: a specialist-baseline EVAL output already exists; one-shot")
    block = spec["PROTOCOL"]["block"]
    lo, hi = (int(x) for x in block.split(".."))
    if [EVAL_SEEDS[0], EVAL_SEEDS[-1]] != [lo, hi] or len(EVAL_SEEDS) != 64:
        raise SystemExit(f"REFUSING: evaluator seeds do not match the frozen block {block}")
    return spec, paths


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

    spec, paths = _preflight()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    print(f"SPECIALIST BASELINE SEALED EVAL  {_now()}")
    print("  both checkpoints sha256 VERIFIED against SPECIALIST_BASELINE_SPEC.json")
    print(f"  seeds {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]} (n={len(EVAL_SEEDS)}), SHARED across policies")
    print(f"  gate: delta_A = V(pi_A,A)-V(pi_B,A) > 0, LCB95 > 0; delta_B symmetric on Pole B")
    print(f"  bootstrap n={N_BOOT}, alpha={ALPHA}, rng_seed={BOOTSTRAP_SEED}\n", flush=True)

    probe = R2.build_env(device, EVAL_SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    policies = {n: load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
                for n, p in paths.items()}

    def run_cell(policy, pole: str, seed: int) -> dict:
        env = R2.build_env(device, seed)
        core = env.core
        try:
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
                                       context=f"specialist baseline {pole} seed {seed}")
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
    for name in POLICIES:
        for pole in ("A", "B"):
            for seed in EVAL_SEEDS:
                rows.append({"policy": name, "pole": pole, "seed": seed,
                             **run_cell(policies[name], pole, seed)})
            wr = np.mean([r["win"] for r in rows
                          if r["policy"] == name and r["pole"] == pole])
            print(f"  {name:5s} on Pole {pole}: win rate {wr:.4f}", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def wins(name, pole):
        by = {r["seed"]: r["win"] for r in rows if r["policy"] == name and r["pole"] == pole}
        return np.array([by[s] for s in EVAL_SEEDS], dtype=np.float64)

    delta_a = _mean_ci(wins("pi_A", "A") - wins("pi_B", "A"))
    delta_b = _mean_ci(wins("pi_B", "B") - wins("pi_A", "B"))
    delta_a["passes"] = bool(delta_a["mean"] > 0 and delta_a["lcb95"] > 0)
    delta_b["passes"] = bool(delta_b["mean"] > 0 and delta_b["lcb95"] > 0)

    tie_or_reversal = [k for k, d in (("delta_A", delta_a), ("delta_B", delta_b))
                       if d["mean"] <= 0.0]
    if tie_or_reversal:
        PREAUDIT_FLAG.write_text(json.dumps({
            "record": "Specialist-baseline EVAL integrity audit REQUIRED", "status": "FLAGGED",
            "utc": _now(),
            "implements": "SPECIALIST_BASELINE_SPEC.json#TIE_OR_REVERSAL",
            "triggered_by": tie_or_reversal,
            "point_estimates": {"delta_A": delta_a["mean"], "delta_B": delta_b["mean"]},
            "rule": "requires a row-level integrity audit before any verdict-bearing "
                    f"SPECIALIST_BASELINE_EVAL_RESULT.json. Raw rows: {ROWS_CSV.name}. No "
                    "exception for this being the baseline 'expected' to pass.",
        }, indent=2), encoding="utf-8")
        print(f"\n  TIE/REVERSAL on {tie_or_reversal} -- integrity audit REQUIRED.")
        print(f"  -> {PREAUDIT_FLAG}")
        return 0

    gate_passes = bool(delta_a["passes"] and delta_b["passes"])
    print("\n  PRIMARY GATE")
    print(f"    delta_A {delta_a['mean']:+.4f} [{delta_a['lcb95']:+.4f}, {delta_a['ucb95']:+.4f}]"
          f" {'PASS' if delta_a['passes'] else 'FAIL'}")
    print(f"    delta_B {delta_b['mean']:+.4f} [{delta_b['lcb95']:+.4f}, {delta_b['ucb95']:+.4f}]"
          f" {'PASS' if delta_b['passes'] else 'FAIL'}")
    print(f"\n  GATE: {'PASS' if gate_passes else 'FAIL'}")

    OUT.write_text(json.dumps({
        "record": "Specialist baseline sealed EVAL", "status": "FROZEN_RESULT", "one_shot": True,
        "utc": _now(), "implements": "SPECIALIST_BASELINE_SPEC.json#PRIMARY_GATE",
        "seeds": {"block": [EVAL_SEEDS[0], EVAL_SEEDS[-1]], "n": len(EVAL_SEEDS),
                  "shared_across_policies": True},
        "PRIMARY_GATE": {"delta_A": delta_a, "delta_B": delta_b, "passes": gate_passes},
        "checkpoints": {n: spec["MODELS_UNDER_TEST"][n]["sha256"] for n in POLICIES},
        "bootstrap": {"procedure": "paired percentile bootstrap over evaluation seeds",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
        "no_model_selection_occurred": True, "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
