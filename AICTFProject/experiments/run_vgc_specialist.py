"""Train one single-opponent SPECIALIST policy (S_OP7 / S_OP8).

This is EXPERIMENT 2 in the project's naming discipline and is NOT a diversity
rung. D1/D3/D7 name fixed opponent-DIVERSITY conditions; S_OP<n> names a policy
trained against exactly one opponent. Never report an S_OP run as a D rung.

Reuses run_g0_v5_long wholesale -- identical reward, architecture, health gates,
budget, map and team size -- and rebinds only the opponent pool, exactly as
run_vgc_diversity.py does. The only manipulation is |pool| = 1.

Targets, seeds and gates are locked by
artifacts/vgc_specialists/SPECIALIST_PILOT_FROZEN.json. There is deliberately
no --opponent override for real runs: a typo cannot quietly train an unfrozen
specialist, mirroring the seed lock in run_vgc_diversity.py.

Run:  python experiments/run_vgc_specialist.py --specialist S_OP7
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FROZEN = PROJECT_ROOT / "artifacts/vgc_specialists/SPECIALIST_PILOT_FROZEN.json"


def _frozen() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def _run_tag(name: str, seed: int) -> str:
    return f"vgc_{name.lower()}_seed{int(seed)}"


def _artifact_dir(name: str, seed: int) -> Path:
    return PROJECT_ROOT / "artifacts" / "vgc_specialists" / _run_tag(name, seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specialist", required=True, choices=("S_OP7", "S_OP8"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke-steps", type=int, default=0,
                    help="short run for plumbing checks; writes to a _smoke tag")
    args = ap.parse_args()

    fz = _frozen()
    name = args.specialist
    spec = fz["training"][name]
    pool = tuple(spec["opponent_pool"])
    seed = int(spec["seed"])
    steps = int(fz["training"]["total_timesteps"])
    if len(pool) != 1:
        raise SystemExit(f"{name} is not a single-opponent specialist: {pool}")

    tag_name = name if not args.smoke_steps else f"{name}_smoke"

    import experiments.run_g0_v5_long as G

    # Rebind only what the specialist changes. Everything else is inherited.
    G.OPPONENTS = pool
    G.G0V5_SEEDS = (seed,)
    G.ABLATION_SEEDS = (seed,)
    G.run_tag_for = lambda s: _run_tag(tag_name, s)
    G.artifact_dir_for = lambda s: _artifact_dir(tag_name, s)

    _build = G.build_config

    def build_config(s: int):
        cfg = _build(s)
        cfg.opponent_pool = pool          # the manipulation
        cfg.opponent_pool_weights = ()    # uniform; degenerate at |pool| = 1
        cfg.total_timesteps = steps
        if args.smoke_steps:
            cfg.total_timesteps = int(args.smoke_steps)
            cfg.periodic_checkpoint_steps = max(2048, int(args.smoke_steps) // 2)
        return cfg

    G.build_config = build_config

    print("=" * 78)
    print(f"VGC SPECIALIST  {name}  seed={seed}")
    print(f"  opponent_pool = {pool}   (|pool| = {len(pool)})")
    print(f"  budget        = {cfg_steps(args, steps):,} steps"
          f"{' [SMOKE]' if args.smoke_steps else ''}")
    print(f"  team size     = {fz['training']['team_size']}  map = {fz['training']['map']}")
    print(f"  artifacts     -> {_artifact_dir(tag_name, seed)}")
    print("  reward, architecture, health gates: inherited from run_g0_v5_long")
    print("  NOTE: this is a SPECIALIST, not a diversity rung.")
    print("=" * 78, flush=True)

    # Sidecar written BEFORE training, same rationale as run_vgc_diversity.py:
    # the shared training_manifest does not record the specialist condition and
    # the canonical trainer is used by frozen experiments, so it is not modified.
    art = _artifact_dir(tag_name, seed)
    art.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
                            capture_output=True, text=True).stdout.strip() or "unknown"
    (art / "vgc_condition.json").write_text(json.dumps({
        "experiment": "VGC explicit specialist pilot",
        "experiment_kind": "SPECIALIST",
        "is_a_diversity_rung": False,
        "specialist": name,
        "opponent_pool": list(pool),
        "n_opponents": len(pool),
        "training_method": "single-opponent PPO (mode=OPPONENT_POOL, |pool| = 1)",
        "seed": seed,
        "team_size": fz["training"]["team_size"],
        "map": fz["training"]["map"],
        "total_timesteps": int(cfg_steps(args, steps)),
        "smoke": bool(args.smoke_steps),
        "frozen_protocol": "artifacts/vgc_specialists/SPECIALIST_PILOT_FROZEN.json",
        "triggered_by": "PATH_B (artifacts/summer_2026/gate_results.json)",
        "git_commit": commit,
    }, indent=2), encoding="utf-8")

    sys.argv = ["run_g0_v5_long.py", "--seed", str(seed), "--threads", str(args.threads)]
    return G.main()


def cfg_steps(args, steps: int) -> int:
    return int(args.smoke_steps) if args.smoke_steps else int(steps)


if __name__ == "__main__":
    raise SystemExit(main())
