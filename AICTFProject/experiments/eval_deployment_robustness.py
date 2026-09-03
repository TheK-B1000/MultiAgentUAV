"""Deployment-only robustness sweep: localization noise, motion error, control delay.

Implements DEPLOYMENT_ROBUSTNESS_SPEC.json. Takes an ALREADY-TRAINED checkpoint (any team
size, any method -- this script does not care which) and evaluates it across the frozen
disturbance matrix: 3 families x 3 severities, plus one nominal (zero-disturbance) baseline.
Strictly inference-only -- this file contains no optimizer, no backward(), no training-loop
import of any kind.

Mechanisms, each confirmed against the real source before use, not assumed:
  localization noise   core.rt_sensor_noise_sigma_cells[:] = sigma   (gpu_env/state/scratch.py:
                        shape (B,), read by _observations.py's observation model)
  motion error          core.rt_drift_sigma_cells[:] = sigma          (same file; read by
                        _dynamics.py, added directly to post-motion position)
  control delay         rl.control_delay.DelayBuffer(ticks), wrapping the action between
                        policy.predict() and env.step_async() -- self-tested standalone in
                        rl/control_delay.py, reset() called at every episode boundary

Every disturbance is applied by direct tensor assignment on the already-constructed core,
bypassing the phase-indexed stress-schedule mechanism entirely (that mechanism exists for
CURRICULUM difficulty during training; this is a controlled, deterministic TEST condition and
must not depend on which curriculum phase happens to be active).

Output is collision-proof by construction: one file per (checkpoint, pole, disturbance,
severity), named so two different sweeps can never collide, and the script REFUSES if a
target file already exists rather than overwriting it.

Run:  python experiments/eval_deployment_robustness.py --checkpoint <path> --checkpoint-id <name> \
          --team-label 2v2 --pole A --seeds-start 11705001 --n-seeds 32 --device cuda
      python experiments/eval_deployment_robustness.py --plan-only   (no GPU, prints the matrix)
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

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "DEPLOYMENT_ROBUSTNESS_SPEC.json"
GUARANTEE = SD / "DEPLOYMENT_ONLY_GUARANTEE_CHECK.json"
OUT_DIR = SD / "robustness_eval_rows"

FAMILIES = ("nominal", "localization_noise", "motion_error", "control_delay")
SEVERITIES = ("low", "medium", "high")  # not used for "nominal"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_spec() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: robustness spec not frozen: {spec['status']!r}")
    guarantee = json.loads(GUARANTEE.read_text(encoding="utf-8"))
    if guarantee["VERDICT"] != "PASS":
        raise SystemExit(f"REFUSING: deployment-only guarantee check is not PASS: "
                         f"{guarantee['VERDICT']!r} -- do not run a perturbation sweep "
                         "until the structural check confirms nothing leaks into training")
    return spec


def build_matrix(spec: dict) -> list[dict]:
    """Every (family, severity) cell, including exactly one nominal baseline."""
    tiers = spec["TIERS"]
    cells = [{"family": "nominal", "severity": "nominal", "sensor_noise": 0.0,
             "drift": 0.0, "delay_ticks": 0}]
    for sev in SEVERITIES:
        cells.append({"family": "localization_noise", "severity": sev,
                      "sensor_noise": tiers["localization_noise"][sev],
                      "drift": 0.0, "delay_ticks": 0})
    for sev in SEVERITIES:
        cells.append({"family": "motion_error", "severity": sev,
                      "sensor_noise": 0.0, "drift": tiers["motion_error"][sev],
                      "delay_ticks": 0})
    for sev in SEVERITIES:
        cells.append({"family": "control_delay", "severity": sev,
                      "sensor_noise": 0.0, "drift": 0.0,
                      "delay_ticks": tiers["control_delay"][sev]["ticks"]})
    return cells


def out_path(checkpoint_id: str, team_label: str, pole: str, cell: dict) -> Path:
    return OUT_DIR / f"{checkpoint_id}__{team_label}__pole{pole}__{cell['family']}__{cell['severity']}.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--checkpoint-id", default=None,
                    help="short identifier used in output filenames, e.g. rscft_treatment_2v2")
    ap.add_argument("--team-label", default=None, help="e.g. 2v2, 4v4, 6v6")
    ap.add_argument("--pole", choices=("A", "B"), default=None)
    ap.add_argument("--z", type=int, default=None, help="fixed latent id to evaluate, if the "
                    "checkpoint is latent-conditioned; omit for a no-latent policy")
    ap.add_argument("--seeds-start", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    spec = load_spec()
    matrix = build_matrix(spec)

    if args.plan_only:
        print(f"DEPLOYMENT ROBUSTNESS SWEEP -- PLAN ONLY  {_now()}\n")
        print(f"  {len(matrix)} cells (1 nominal + 3 families x 3 severities):")
        for c in matrix:
            print(f"    {c['family']:20s} {c['severity']:8s}  "
                  f"sensor_noise={c['sensor_noise']}  drift={c['drift']}  "
                  f"delay_ticks={c['delay_ticks']}")
        print(f"\n  output naming: <checkpoint_id>__<team_label>__pole<X>__<family>__<severity>.csv")
        print(f"  output dir: {OUT_DIR}")
        print(f"  collision policy: REFUSES if the target file already exists")
        return 0

    required = ("checkpoint", "checkpoint_id", "team_label", "pole", "seeds_start")
    missing = [r for r in required if getattr(args, r) is None]
    if missing:
        raise SystemExit(f"REFUSING: --plan-only not set, but missing required args: {missing}")

    ck = Path(args.checkpoint)
    if not ck.is_file():
        raise SystemExit(f"REFUSING: checkpoint missing: {ck}")
    ck_sha = _sha(ck)

    seeds = list(range(args.seeds_start, args.seeds_start + args.n_seeds))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [out_path(args.checkpoint_id, args.team_label, args.pole, c) for c in matrix]
    existing = [p for p in targets if p.is_file()]
    if existing:
        raise SystemExit(f"REFUSING: {len(existing)} output file(s) already exist, would be "
                         f"overwritten: {[p.name for p in existing[:5]]}")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.control_delay import DelayBuffer
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    device = args.device if torch.cuda.is_available() or args.device == "cuda" else "cpu"
    print(f"DEPLOYMENT ROBUSTNESS SWEEP  {_now()}")
    print(f"  checkpoint {ck.name}  sha256 {ck_sha[:16]}...")
    print(f"  team_label={args.team_label}  pole={args.pole}  z={args.z}")
    print(f"  seeds {seeds[0]}..{seeds[-1]} (n={len(seeds)})")
    print(f"  {len(matrix)} cells, {len(matrix) * len(seeds)} episodes total\n", flush=True)

    probe = R2.build_env(device, seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policy = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)

    def run_episode(seed: int, cell: dict) -> dict:
        env = R2.build_env(device, seed)
        core = env.core
        delay = DelayBuffer(cell["delay_ticks"]) if cell["delay_ticks"] else None
        try:
            if args.z is not None:
                policy.fixed_latent_strategy = True
                policy.fixed_latent_strategy_id = int(args.z)
            policy.reset_strategy()
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            genomes = {"OP6": pole_A_genome()} if args.pole == "A" else {}
            install_keyed_opponent_overlays(core, genomes)
            key = P0.POLES[args.pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                                       context=f"robustness {cell['family']}/{cell['severity']} "
                                               f"{args.pole} seed {seed}")
            # apply the disturbance AFTER reset, by direct tensor assignment -- bypasses the
            # phase-indexed stress schedule entirely, a deterministic test condition
            core.rt_sensor_noise_sigma_cells[:] = float(cell["sensor_noise"])
            core.rt_drift_sigma_cells[:] = float(cell["drift"])

            terminal = None
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                exec_action = delay.push(action) if delay is not None else action
                env.step_async(exec_action)
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
            if delay is not None:
                delay.reset()
            env.close()

    for cell, target in zip(matrix, targets):
        rows = [{"seed": s, **run_episode(s, cell)} for s in seeds]
        with target.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        wr = np.mean([r["win"] for r in rows])
        print(f"  {cell['family']:20s} {cell['severity']:8s}  win rate {wr:.4f}  "
              f"-> {target.name}", flush=True)

    print(f"\n  {len(matrix)} cells written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
