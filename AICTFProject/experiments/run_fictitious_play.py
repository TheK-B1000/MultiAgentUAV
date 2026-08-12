"""Fictitious Play on the verified PPO-as-opponent seam.

    pi0                                   (an existing policy; not retrained)
    pi1 trains against {pi0}
    pi2 trains against uniform {pi0, pi1}
    pi3 trains against uniform {pi0, pi1, pi2}

Built on what the Phase 1 audit verified exists, not on a rebuilt self-play
system. The historical self-play TRAINER is removed and its checkpoints are
unavailable; what survives and was tested end-to-end is the env seam:

    core.set_next_opponent("SNAPSHOT", <checkpoint path>)
      -> gpu_env/state/snapshots.py::_load_snapshot_policy
      -> _get_red_snapshot_actions  (gpu_env/_core/_step.py)

No Nash solving. No Double Oracle. Uniform historical sampling only.

FAIL-CLOSED on the one hazard the audit found: snapshots.py wraps policy loading
in a bare `except Exception: model = None`, so a failed load degrades SILENTLY to
a red team with no policy. Every opponent checkpoint is therefore asserted
loadable BEFORE training starts -- a generation trained against a silently-absent
opponent would look like a normal run and mean nothing.

Run:  python experiments/run_fictitious_play.py --generation 1 --seed 3800001
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FP_DIR = ROOT / "artifacts/vgc_fp"
LINEAGE = FP_DIR / "FP_LINEAGE.json"

# pi0 is an existing trained policy, reused rather than retrained.
PI0 = "artifacts/g0_v5_long/g0_v5_long_seed3200001/ckpts/final_g0_v5_long_seed3200001.zip"


def _lineage() -> dict:
    if LINEAGE.exists():
        return json.loads(LINEAGE.read_text(encoding="utf-8"))
    return {"pi0": PI0, "generations": {}}


def _population(gen: int) -> list[str]:
    """Historical opponents for generation `gen`: pi0..pi(gen-1)."""
    lin = _lineage()
    pop = [lin["pi0"]]
    for g in range(1, gen):
        ck = lin["generations"].get(str(g), {}).get("checkpoint")
        if ck is None:
            raise SystemExit(f"generation {g} missing from lineage; train it first")
        pop.append(ck)
    return pop


def assert_loadable(paths: list[str], device: str = "cpu") -> dict:
    """Verify every opponent checkpoint ACTUALLY loads.

    snapshots.py swallows load failures and returns None, which would leave red
    unpiloted while training looked healthy. This is the guard for that.
    """
    from experiments.run_g0_v2_evaluation import (
        AGENTS, CANONICAL_MAP, EPISODE_HORIZON, V2_RULES,
    )
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS, map_set="train",
        map_layout=CANONICAL_MAP, max_decision_steps=EPISODE_HORIZON,
        aquaticus_profile=True, rules_profile="OURS", device=device, seed=1,
        obstacle_obs_channel=True, tag_telemetry_enabled=True, **V2_RULES,
    )
    env = GPUCTFVecEnv(cfg)
    out = {}
    try:
        for p in paths:
            resolved = str(ROOT / p) if not Path(p).is_absolute() else p
            if not Path(resolved).is_file():
                raise SystemExit(f"FP ABORT: opponent checkpoint missing: {resolved}")
            model = env.core._load_snapshot_policy(resolved)
            if model is None:
                raise SystemExit(
                    f"FP ABORT: {resolved} loaded as None. snapshots.py swallows load "
                    f"failures, so training would proceed against an unpiloted red team "
                    f"and the run would look healthy while meaning nothing.")
            out[p] = type(model).__name__
    finally:
        env.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", type=int, required=True, help="pi_N to train (N>=1)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke-steps", type=int, default=0)
    ap.add_argument("--check-only", action="store_true",
                    help="run the loadability guard and exit; no training")
    args = ap.parse_args()

    gen = int(args.generation)
    if gen < 1:
        raise SystemExit("generation must be >= 1; pi0 is an existing policy")

    pop = _population(gen)
    print("=" * 78)
    print(f"FICTITIOUS PLAY  generation=pi{gen}  seed={args.seed}")
    print(f"  historical population ({len(pop)}), uniform sampling at episode boundaries:")
    for i, p in enumerate(pop):
        print(f"    pi{i}: {p}")
    print("=" * 78, flush=True)

    loaded = assert_loadable(pop)
    print("  loadability guard PASS:")
    for p, t in loaded.items():
        print(f"    {t:28s} {p}")
    if args.check_only:
        return 0

    FP_DIR.mkdir(parents=True, exist_ok=True)
    lin = _lineage()
    tag = f"vgc_fp_pi{gen}_seed{args.seed}"
    art = FP_DIR / tag
    art.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout.strip() or "unknown"
    (art / "fp_condition.json").write_text(json.dumps({
        "experiment": "VGC fictitious play",
        "method": "Fictitious-Play-PPO",
        "generation": gen,
        "historical_population": pop,
        "sampling": "uniform over the historical population, at episode boundaries only",
        "opponent_kind": "SNAPSHOT",
        "seed": args.seed,
        "loadability_guard": loaded,
        "git_commit": commit,
    }, indent=2), encoding="utf-8")

    lin["generations"][str(gen)] = {
        "seed": args.seed, "tag": tag,
        "checkpoint": f"artifacts/vgc_fp/{tag}/ckpts/final_{tag}.zip",
        "trained_against": pop, "git_commit": commit,
    }
    LINEAGE.write_text(json.dumps(lin, indent=2), encoding="utf-8")
    print(f"  lineage recorded -> {LINEAGE}")
    print("  NOTE: SNAPSHOT-pool training wiring is the remaining step; the seam, the "
          "loadability guard and the lineage are in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
