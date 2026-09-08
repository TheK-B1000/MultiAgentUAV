"""Mixed-opponent PPO at a frozen diversity level (D1 / D3 / D7).

Reuses run_g0_v5_long wholesale -- same reward, architecture, budget, health
probes, validation panels and TASK_HEALTH/SYSTEM_HEALTH gates -- and rebinds
only what the diversity condition changes:

    opponent_pool    the frozen D1 / D3 / D7 set
    seed set         3600000+ (D1) / 3700000+ (D3)
    artifact paths   artifacts/vgc_diversity/<condition>_seed<N>

No new opponent-sampling code exists or is needed. The Phase 1 audit established
that the trainer already samples uniformly per COMPLETED episode over
cfg.opponent_pool, deterministic under cfg.seed, with no mid-episode switching.
D1/D3/D7 differ only by pool contents.

D7 is NOT trained here: G0-V5 seeds 3200001-3 are already Mixed-PPO D7 at 2v2
under this exact protocol and budget.

Run:  python experiments/run_vgc_diversity.py --condition D1 --seed 3600001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FROZEN = PROJECT_ROOT / "artifacts/vgc_diversity/VGC_DIVERSITY_SETS_FROZEN.json"


def _frozen() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def _condition_seeds(cond: str) -> tuple[int, ...]:
    return tuple(_frozen()["training_method"]["seeds"][cond])


def _run_tag(cond: str, seed: int) -> str:
    return f"vgc_{cond.lower()}_seed{int(seed)}"


def _artifact_dir(cond: str, seed: int) -> Path:
    return PROJECT_ROOT / "artifacts" / "vgc_diversity" / _run_tag(cond, seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=("D1", "D3"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke-steps", type=int, default=0,
                    help="smoke only: shorten total_timesteps; never used for a real run")
    args = ap.parse_args()

    fz = _frozen()
    cond = args.condition
    pool = tuple(fz["THE_SETS"][cond])
    seeds = _condition_seeds(cond)
    if args.seed not in seeds and not args.smoke_steps:
        raise SystemExit(f"seed {args.seed} is not a frozen {cond} seed {seeds}")

    import experiments.run_g0_v5_long as G

    # Rebind only what the condition changes. Everything else is inherited.
    G.OPPONENTS = pool
    # A smoke seed is admitted only in smoke mode; real runs stay locked to the
    # frozen seed set so a typo cannot quietly train an unfrozen condition.
    allowed = tuple(seeds) + ((args.seed,) if args.smoke_steps else ())
    G.G0V5_SEEDS = allowed
    G.ABLATION_SEEDS = allowed
    G.run_tag_for = lambda s: _run_tag(cond, s)
    G.artifact_dir_for = lambda s: _artifact_dir(cond, s)

    _build = G.build_config

    def build_config(seed: int):
        cfg = _build(seed)
        cfg.opponent_pool = pool          # the manipulation
        cfg.opponent_pool_weights = ()    # uniform, per frozen protocol
        if args.smoke_steps:
            cfg.total_timesteps = int(args.smoke_steps)
            # smoke must produce at least one checkpoint + health panel
            cfg.periodic_checkpoint_steps = max(2048, int(args.smoke_steps) // 2)
        return cfg

    G.build_config = build_config

    print("=" * 78)
    print(f"VGC DIVERSITY  condition={cond}  seed={args.seed}")
    print(f"  opponent_pool = {pool}   (|D| = {len(pool)})")
    print(f"  budget        = {fz['budget_control']['total_timesteps']:,} steps"
          f"{' [SMOKE ' + str(args.smoke_steps) + ']' if args.smoke_steps else ''}")
    print(f"  team size     = {fz['team_size']['choice']}")
    print(f"  artifacts     -> {_artifact_dir(cond, args.seed)}")
    print("  reward, architecture, health gates: inherited from run_g0_v5_long")
    print("=" * 78, flush=True)

    # The shared training_manifest does not record the diversity condition, and
    # the canonical trainer is used by frozen experiments so it is not modified.
    # A sidecar written BEFORE training carries the condition instead.
    import subprocess
    art = _artifact_dir(cond, args.seed)
    art.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
                            capture_output=True, text=True).stdout.strip() or "unknown"
    (art / "vgc_condition.json").write_text(json.dumps({
        "experiment": "VGC opponent-diversity scaling",
        "diversity_condition": cond,
        "opponent_pool": list(pool),
        "n_opponents": len(pool),
        "held_out_opponents": fz["held_out_opponents"][cond],
        "training_method": "Mixed-PPO (mode=OPPONENT_POOL, uniform per completed episode)",
        "seed": int(args.seed),
        "team_size": fz["team_size"]["choice"],
        "total_timesteps": fz["budget_control"]["total_timesteps"],
        "smoke": bool(args.smoke_steps),
        "frozen_sets_artifact": "artifacts/vgc_diversity/VGC_DIVERSITY_SETS_FROZEN.json",
        "git_commit": commit,
    }, indent=2), encoding="utf-8")

    sys.argv = ["run_g0_v5_long.py", "--seed", str(args.seed), "--threads", str(args.threads)]
    return G.main()


if __name__ == "__main__":
    raise SystemExit(main())
