"""Rung 1 matched evaluation: the 128 Rung-0 seeds, read through within-seed differences.

Implements LADDER_MATCHED_EVALUATION_AMENDMENT.json for Rung 1. Loads the frozen Rung-1
checkpoint, wraps each branch in CustomPPOInferencePolicy exactly as the specialists are, and
dispatches by z through the bit-exact-verified Rung 0 wrapper -- so this exercises the identical
evaluator path Rung 0 used. Runs the four cells on all 128 matched seeds (cuda), then reports:

    PRIMARY:  D_A, D_B  = mean_s [delta_R1(s) - delta_R0(s)], paired bootstrap over seeds,
              against the frozen per-seed Rung-0 values in RUNG0_LADDER_REFERENCE.json.
              'sharing damaged specialization' on a pole  iff  UCB95(D) < 0.
    ALSO:     Rung 1's own n=128 gate (delta > 0 and LCB95 > 0 on both poles).

Rung 0 is never re-run. One-shot.

Run:  python experiments/eval_ladder_rung1_matched.py --device cuda
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
AMEND = SD / "LADDER_MATCHED_EVALUATION_AMENDMENT.json"
REFERENCE = SD / "RUNG0_LADDER_REFERENCE.json"
ON_POLE = (("z0", "A"), ("z1", "B"))
OFF_POLE = (("z1", "A"), ("z0", "B"))

N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _paths(rung: int):
    return (SD / f"RUNG{rung}_STUDENT_FROZEN.json", SD / f"RUNG{rung}_LADDER_EVAL_RESULT.json",
            SD / f"rung{rung}_ladder_eval_rows.csv", SD / f"RUNG{rung}_LADDER_EVAL_INTEGRITY_REQUIRED.json")


def _preflight(device: str, rung: int):
    FROZEN, OUT, ROWS_CSV, PREAUDIT_FLAG = _paths(rung)
    if device != "cuda":
        raise SystemExit("REFUSING: the matched-evaluation rule fixes cuda")
    amend = json.loads(AMEND.read_text(encoding="utf-8"))
    if amend["status"] != "FROZEN_APPEND_ONLY_AMENDMENT":
        raise SystemExit("REFUSING: matched-evaluation amendment not frozen")
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if ref["status"] != "FROZEN_REFERENCE":
        raise SystemExit("REFUSING: Rung-0 reference not frozen")
    for src in ref["sources"].values():
        if _sha(ROOT / src["path"]) != src["sha256"]:
            raise SystemExit(f"REFUSING: reference source {src['path']} sha mismatch")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen["status"] != "FROZEN_STUDENT" or not frozen["fit_check"]["passed"]:
        raise SystemExit(f"REFUSING: Rung {rung} status {frozen['status']!r} / fit check {frozen['fit_check']}")
    ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    if not ck.is_file() or _sha(ck) != frozen["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit(f"REFUSING: Rung {rung} checkpoint missing or sha mismatch")
    if OUT.is_file() or ROWS_CSV.is_file() or PREAUDIT_FLAG.is_file():
        raise SystemExit(f"REFUSING: a Rung {rung} matched-eval output already exists; one-shot")
    seeds = [int(s) for s in ref["matched_seed_set"]["seeds"]]
    if len(seeds) != 128 or len(set(seeds)) != 128:
        raise SystemExit("REFUSING: reference does not carry 128 distinct seeds")
    return ref, frozen, ck, seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rung", type=int, default=1, choices=(1, 2))
    args = ap.parse_args()
    RUNG = int(args.rung)
    FROZEN, OUT, ROWS_CSV, PREAUDIT_FLAG = _paths(RUNG)

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl import ladder_rung1 as L1
    from rl.curriculum import phase_from_tag

    ref, frozen, ck, seeds = _preflight(args.device, RUNG)
    if not torch.cuda.is_available():
        raise SystemExit("REFUSING: cuda unavailable")
    device = args.device

    print(f"RUNG {RUNG} MATCHED EVAL  {_now()}  device={device}")
    print(f"  Rung {RUNG} checkpoint sha {frozen['TERMINAL_CHECKPOINT']['sha256'][:12]}... VERIFIED; "
          f"fit check {frozen['final_holdout']['holdout_agree_z0_vs_piA']:.3f}/{frozen['final_holdout']['holdout_agree_z1_vs_piB']:.3f}")
    print(f"  128 matched seeds ({seeds[0]}..{seeds[63]} and {seeds[64]}..{seeds[-1]}), four cells")
    print(f"  PRIMARY: within-seed D vs frozen Rung-0 per-seed deltas; damaged iff UCB95(D) < 0\n", flush=True)

    probe = R2.build_env(device, seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    model, branch_cfg, _ = L1.load_rung(RUNG, str(ck), obs_space, act_space, device=device)
    policy = L1.make_dispatch_policy(model, branch_cfg, device=device)

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
                                       context=f"rung{RUNG} matched {pole} z{z} seed {seed}")
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
        for seed in seeds:
            rows.append({"arm": f"RUNG{RUNG}", "z": f"z{z}", "pole": pole, "seed": seed, **run_cell(pole, z, seed)})
        wr = np.mean([r["win"] for r in rows if r["z"] == f"z{z}" and r["pole"] == pole])
        print(f"  RUNG{RUNG} z{z} on Pole {pole}: win rate {wr:.4f}  (Rung 0 n=128: "
              f"{ref['cell_win_rates_n128'][f'z{z}_pole{pole}']:.4f})", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def wins(z, pole):
        by = {r["seed"]: r["win"] for r in rows if r["z"] == z and r["pole"] == pole}
        return np.array([by[s] for s in seeds], dtype=np.float64)

    d1_A = wins("z0", "A") - wins("z1", "A")
    d1_B = wins("z1", "B") - wins("z0", "B")
    d0_A = np.array([ref["per_seed"]["delta_A"][str(s)] for s in seeds], dtype=np.float64)
    d0_B = np.array([ref["per_seed"]["delta_B"][str(s)] for s in seeds], dtype=np.float64)

    own_A, own_B = _mean_ci(d1_A), _mean_ci(d1_B)
    for d in (own_A, own_B):
        d["passes"] = bool(d["mean"] > 0 and d["lcb95"] > 0)
    D_A, D_B = _mean_ci(d1_A - d0_A), _mean_ci(d1_B - d0_B)

    def classify(D):
        if D["ucb95"] < 0:
            return "SHARING_DAMAGED_SPECIALIZATION"
        if D["lcb95"] > 0:
            return "SHARING_HELPED"
        return "NO_DETECTABLE_WITHIN_SEED_LOSS"
    cls_A, cls_B = classify(D_A), classify(D_B)

    tie_or_reversal = [k for k, d in (("delta_A", own_A), ("delta_B", own_B)) if d["mean"] <= 0.0]
    if tie_or_reversal:
        PREAUDIT_FLAG.write_text(json.dumps({
            "record": f"Rung {RUNG} matched EVAL integrity audit REQUIRED", "status": "FLAGGED", "utc": _now(), "rung": RUNG,
            "triggered_by": tie_or_reversal,
            "point_estimates": {"delta_A": own_A["mean"], "delta_B": own_B["mean"]},
            "note": f"a tie/reversal in Rung {RUNG}'s own deltas is a legitimate possible architectural outcome, "
                    "but the standing rule still applies: row-level audit before interpretation. Rows written.",
        }, indent=2), encoding="utf-8")
        print(f"\n  TIE/REVERSAL on {tie_or_reversal} in Rung {RUNG}'s own deltas -- audit required before interpretation.")
        print(f"  -> {PREAUDIT_FLAG}")
        return 0

    cellrates = {f"z{z}_pole{p}": float(np.mean([r["win"] for r in rows if r["z"] == f"z{z}" and r["pole"] == p]))
                 for z, p in ((0, "A"), (1, "A"), (0, "B"), (1, "B"))}
    r0c = ref["cell_win_rates_n128"]
    on_shift = float(np.mean([cellrates[f"{z}_pole{p}"] - r0c[f"{z}_pole{p}"] for z, p in ON_POLE]))
    off_shift = float(np.mean([cellrates[f"{z}_pole{p}"] - r0c[f"{z}_pole{p}"] for z, p in OFF_POLE]))
    if off_shift > 0 and (on_shift >= 0 or abs(on_shift) < abs(off_shift)):
        mech = "LEAKAGE_SIGNATURE"
    elif on_shift < 0 and abs(on_shift) >= abs(off_shift):
        mech = "INTERFERENCE_SIGNATURE"
    else:
        mech = "NEITHER_SIGNATURE_CLEANLY"
    print("")
    print("  PRIMARY -- on-pole / off-pole decomposition vs Rung 0 (preregistered)")
    for z, p in ON_POLE + OFF_POLE:
        k = f"{z}_pole{p}"
        role = "ON " if (z, p) in ON_POLE else "OFF"
        print(f"    {role} {k:10s} {r0c[k]:.4f} -> {cellrates[k]:.4f}  ({cellrates[k]-r0c[k]:+.4f})")
    print(f"    mean on-pole shift {on_shift:+.4f}   mean off-pole shift {off_shift:+.4f}   -> {mech}")
    print("\n  PRIMARY -- within-seed difference vs Rung 0 (n=128 paired)")
    print(f"    D_A {D_A['mean']:+.4f} [{D_A['lcb95']:+.4f}, {D_A['ucb95']:+.4f}]  {cls_A}")
    print(f"    D_B {D_B['mean']:+.4f} [{D_B['lcb95']:+.4f}, {D_B['ucb95']:+.4f}]  {cls_B}")
    print("\n  Rung 1's own gate (n=128)")
    print(f"    delta_A {own_A['mean']:+.4f} [{own_A['lcb95']:+.4f}, {own_A['ucb95']:+.4f}] {'PASS' if own_A['passes'] else 'FAIL'}")
    print(f"    delta_B {own_B['mean']:+.4f} [{own_B['lcb95']:+.4f}, {own_B['ucb95']:+.4f}] {'PASS' if own_B['passes'] else 'FAIL'}")
    print(f"  Rung 0 reference (n=128): delta_A {ref['POOLED_N128']['delta_A']['mean']:+.4f}  delta_B {ref['POOLED_N128']['delta_B']['mean']:+.4f}")

    OUT.write_text(json.dumps({
        "record": f"Rung {RUNG} matched evaluation", "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "device": device, "implements": "LADDER_MATCHED_EVALUATION_AMENDMENT.json",
        "rung_checkpoint": frozen["TERMINAL_CHECKPOINT"], "reference": {"record": "RUNG0_LADDER_REFERENCE.json",
                                                                        "sources": ref["sources"]},
        "matched_seeds": {"n_per_cell": len(seeds), "first": seeds[0], "last": seeds[-1]},
        "rung": RUNG, "cell_win_rates": cellrates,
        "PRIMARY_ON_OFF_POLE": {"on_pole_shift": on_shift, "off_pole_shift": off_shift,
                                "mechanism_signature": mech, "rung0_cell_win_rates": r0c,
                                "rule": "leakage iff off_pole_shift>0 and |on_pole_shift| smaller; interference iff on_pole_shift<0 and |on_pole_shift|>=|off_pole_shift|"},
        "PRIMARY_WITHIN_SEED": {"D_A": D_A, "D_B": D_B, "classification": {"A": cls_A, "B": cls_B},
                                "rule": "damaged iff UCB95(D) < 0; helped iff LCB95(D) > 0; otherwise no detectable within-seed loss"},
        "OWN_GATE_N128": {"delta_A": own_A, "delta_B": own_B, "passes": bool(own_A["passes"] and own_B["passes"])},
        "rung0_reference_n128": ref["POOLED_N128"],
        "bootstrap": {"procedure": "paired percentile bootstrap over seeds", "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
        "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
