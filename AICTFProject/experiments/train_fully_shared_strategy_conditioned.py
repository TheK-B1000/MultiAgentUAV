"""Train the Fully Shared Strategy-Conditioned 4v4/6v6 sharing-axis baseline.

Implements FULLY_SHARED_STRATEGY_CONDITIONED_4V4_V1_SPEC (and contingent 6v6):
one shared ``pi_phi(a|o,z)`` under CLOSEST_DEFENDS, forced episode-static z via
``static_env``, pole-matched opponents (z0→OP6, z1→OP7). No router. No
split_attack_defend.

Production training is DEFERRED by default: requires ``--authorize-launch``.
``--smoke`` always allowed for plumbing checks. Do not launch on this PC while
CLOSEST_SPLIT 6v6 viability owns the GPU (SPEC C11).

Run (smoke):
  python experiments/train_fully_shared_strategy_conditioned.py \\
      --team-size 4 --seed 99900001 --smoke --device cpu

Run (authorized later):
  python experiments/train_fully_shared_strategy_conditioned.py \\
      --team-size 4 --seed 22500001 --authorize-launch --device cuda \\
      --spec artifacts/strategic_demand/sppo/FULLY_SHARED_STRATEGY_CONDITIONED_4V4_V1_SPEC.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_scale import (  # noqa: E402
    SMOKE_SEED_MAX,
    SMOKE_SEED_MIN,
    _git_dirty,
    _git_sha,
    _now,
    _propagate_team_size,
)
from rl.custom_ppo.fully_shared_z import (  # noqa: E402
    WARM_START_4V4_PATH,
    WARM_START_4V4_SHA256,
    apply_fully_shared_z_config,
    assert_fully_shared_contracts,
    initial_opponent_keys_for_forced_z,
)

SD = PROJECT_ROOT / "artifacts" / "strategic_demand" / "sppo"
DEFAULT_SPEC_4V4 = SD / "FULLY_SHARED_STRATEGY_CONDITIONED_4V4_V1_SPEC.json"
DEFAULT_SPEC_6V6 = SD / "FULLY_SHARED_STRATEGY_CONDITIONED_6V6_V1_SPEC.json"
BASE_KEY = {"A": "OP6", "B": "OP7"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_spec(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"FAIL-CLOSED: spec not found: {path}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    status = str(spec.get("status", ""))
    if not status.startswith("FROZEN"):
        raise SystemExit(f"FAIL-CLOSED: spec not frozen: status={status!r}")
    return spec


def configure_fully_shared_live_environment(
    env,
    cfg,
    *,
    config_contract: dict,
    team_size: int,
    expected_steps: int | None = None,
):
    """Install BOTH certified poles and bind each env to its forced-z opponent."""
    from experiments.opponent_spec import (
        assert_live_opponent_batch,
        install_keyed_opponent_overlays,
        pole_A_genome,
        pole_B_genome,
    )
    from rl.curriculum import phase_from_tag

    n = int(team_size)
    want_steps = int(cfg.total_timesteps if expected_steps is None else expected_steps)
    if int(cfg.total_timesteps) != want_steps:
        raise RuntimeError(
            f"fully-shared budget drift: cfg.total_timesteps={cfg.total_timesteps} "
            f"expected={want_steps}"
        )
    core = env.core
    if not bool(core.cfg.own_flag_home_required_to_score):
        raise RuntimeError("fully-shared requires M1 own_flag_home_required_to_score")

    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0
    genomes = {"OP6": pole_A_genome(n), "OP7": pole_B_genome(n)}
    install_keyed_opponent_overlays(core, genomes)

    ids = tuple(int(v) for v in getattr(cfg, "forced_latent_env_ids", ()) or ())
    if len(ids) != int(core.B):
        raise RuntimeError(
            f"forced_latent_env_ids length {len(ids)} != live n_envs {int(core.B)}"
        )
    initial_keys = initial_opponent_keys_for_forced_z(ids)
    for env_i, key in enumerate(initial_keys):
        env.env_method("set_phase", phase_from_tag(key), indices=[env_i])
        env.env_method("set_next_opponent", "SCRIPTED", key, indices=[env_i])
    env.reset()
    rows = assert_live_opponent_batch(
        core,
        genomes,
        allowed_keys=("OP6", "OP7"),
        context=f"fully_shared_{n}v{n} construction",
    )
    counts = {k: initial_keys.count(k) for k in ("OP6", "OP7")}
    if counts != {"OP6": int(core.B) // 2, "OP7": int(core.B) // 2}:
        raise RuntimeError(f"fully-shared initial batch not balanced: {counts}")

    # C7 structural: each env's opponent matches its forced z.
    for env_i, (z, key) in enumerate(zip(ids, initial_keys)):
        expect = "OP6" if int(z) == 0 else "OP7"
        if key != expect:
            raise RuntimeError(
                f"C7 pole-match broken at env {env_i}: z={z} key={key} expect={expect}"
            )

    return {
        "fully_shared_protocol": {
            "classification": "DIAGNOSTIC / EXPLORATORY sharing-axis baseline",
            "team_size": n,
            "forced_latent_env_ids": list(ids),
            "initial_opponent_keys": list(initial_keys),
            "initial_live_batch_counts": counts,
            "resolved_opponent_rows": rows,
            **config_contract,
        }
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-size", type=int, required=True, choices=(4, 6))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--spec", default="", help="frozen SPEC path (default by team size)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--authorize-launch",
        action="store_true",
        help="required for non-smoke training (SPEC freezes compute until PI clears queue)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--load-path", default="", help="warm-start checkpoint (defaults to SPEC pin)")
    ap.add_argument("--load-path-sha256", default="", help="expected sha256 of --load-path")
    ap.add_argument("--run-label-suffix", default="_fully_shared_z_4v4_v1")
    args = ap.parse_args()

    n = int(args.team_size)
    seed = int(args.seed)
    is_smoke = bool(args.smoke)

    if n == 6 and not is_smoke:
        raise SystemExit(
            "FAIL-CLOSED: 6v6 fully-shared is CONTINGENT "
            "(FULLY_SHARED_STRATEGY_CONDITIONED_6V6_V1_SPEC). Amend / clear contingency "
            "gates before non-smoke launch."
        )

    if is_smoke and not (SMOKE_SEED_MIN <= seed <= SMOKE_SEED_MAX):
        raise SystemExit(
            f"FAIL-CLOSED: --smoke requires seed in [{SMOKE_SEED_MIN}, {SMOKE_SEED_MAX}]"
        )
    if not is_smoke and SMOKE_SEED_MIN <= seed <= SMOKE_SEED_MAX:
        raise SystemExit(f"FAIL-CLOSED: seed {seed} is reserved for non-scientific smokes")

    if not is_smoke and not args.authorize_launch:
        raise SystemExit(
            "FAIL-CLOSED: non-smoke fully-shared training requires --authorize-launch. "
            "SPEC is freeze-only until 6v6 viability clears this GPU (C11). "
            "Use --smoke for plumbing checks."
        )

    spec_path = Path(args.spec) if args.spec else (DEFAULT_SPEC_4V4 if n == 4 else DEFAULT_SPEC_6V6)
    spec = _load_spec(spec_path)

    try:
        import torch
        torch.set_num_threads(max(1, int(args.threads)))
    except Exception:  # noqa: BLE001
        pass

    touched = _propagate_team_size(n)
    print(f"[fully_shared_z] team size {n} propagated to {len(touched)} module(s)")

    import experiments.run_r1_repertoire_training as R

    # Borrow the R1 A recipe for PPO hyperparameters; override opponents/z below.
    policy = "A"
    if is_smoke and args.total_timesteps is None:
        steps = 5_000
    elif args.total_timesteps is not None:
        steps = int(args.total_timesteps)
    else:
        steps = int(spec.get("TRAINING_locked", {}).get("total_timesteps", 200_000))

    _prefix = "smoke_" if is_smoke else "exploratory_"
    _suffix = str(args.run_label_suffix or f"_fully_shared_z_{n}v{n}_v1")
    label = f"{_prefix}pi_FS_{n}v{n}{_suffix}"

    R.POLICIES[policy] = {
        **dict(R.POLICIES[policy]),
        "seed": seed,
        "steps": steps,
        "label": label,
        "pool": ("OP6", "OP7"),
        "weights": (0.5, 0.5),
    }

    cfg, contract = R.build_r1_config(policy)
    _root = f"exploratory_scale_{n}v{n}_fully_shared" if not is_smoke else f"smoke_scale_{n}v{n}_fully_shared"
    art = PROJECT_ROOT / "artifacts" / _root / label
    cfg.run_tag = label
    cfg.device = str(args.device)
    cfg.checkpoint_dir = str(art / "ckpts")
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")
    cfg.enable_progress_bar = True
    if is_smoke:
        cfg.formal_run = False
        cfg.periodic_checkpoint_steps = max(1, int(cfg.total_timesteps) // 2)

    # Entity repair on for 4v4 warm-start from entity-repair pi_A.
    cfg.entity_repair_enabled = True
    cfg.entity_hidden_dim = 32

    locks = apply_fully_shared_z_config(cfg, team_size=n)
    assert_fully_shared_contracts(cfg)

    # Warm-start pin.
    load_path = str(args.load_path or "").strip()
    if not load_path:
        load_path = str(PROJECT_ROOT / WARM_START_4V4_PATH)
    lp = Path(load_path)
    if not lp.is_file():
        raise SystemExit(f"FAIL-CLOSED: warm-start checkpoint missing: {lp}")
    expect_sha = str(args.load_path_sha256 or WARM_START_4V4_SHA256).lower()
    got_sha = _sha(lp)
    if got_sha != expect_sha:
        raise SystemExit(
            f"FAIL-CLOSED: C9 warm-start sha mismatch for {lp}: "
            f"got {got_sha} expect {expect_sha}"
        )
    cfg.load_path = str(lp)
    cfg.warm_start_reset_progress = True
    cfg.allow_active_actor_module_migration = True  # new strategy_encoder / z embed

    print("=" * 78)
    print(f"FULLY SHARED z-CONDITIONED  {n}v{n}  {_now()}")
    print(f"  spec     {spec_path.name}  [{spec.get('status')}]")
    print(f"  label    {label}")
    print(f"  seed     {seed}  steps={int(cfg.total_timesteps)}  device={cfg.device}")
    print(f"  warm     {lp.name}  sha {got_sha[:12]}...")
    print(f"  smoke    {is_smoke}  authorize_launch={bool(args.authorize_launch)}")
    for line in locks:
        print(f"  lock     {line}")
    print("=" * 78, flush=True)

    if args.dry_run:
        print("  --dry-run: config + contracts OK. NOT training, NOT writing artifacts.")
        return 0

    art.mkdir(parents=True, exist_ok=True)
    (art / "run_manifest.json").write_text(
        json.dumps(
            {
                "record": "train_fully_shared_strategy_conditioned run manifest",
                "utc": _now(),
                "implements": spec_path.name,
                "smoke_non_scientific": is_smoke,
                "authorize_launch": bool(args.authorize_launch),
                "team_size": n,
                "seed": seed,
                "device": cfg.device,
                "git_sha": _git_sha(),
                "git_dirty": _git_dirty(),
                "total_timesteps": int(cfg.total_timesteps),
                "warm_start_from": str(lp),
                "warm_start_sha256": got_sha,
                "fully_shared_z_conditioned_enabled": True,
                "latent_assignment_mode": cfg.latent_assignment_mode,
                "forced_latent_env_ids": list(cfg.forced_latent_env_ids),
                "role_conditioning_enabled": True,
                "role_fixed_for_episode": True,
                "split_attack_defend_enabled": False,
                "arm": "EXPLORATORY",
                "confirmatory": False,
                "locks": locks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("  starting fully-shared PPO ...", flush=True)
    R.orchestrate_training_run(
        cfg,
        pre_rollout_env_setup=partial(
            configure_fully_shared_live_environment,
            config_contract=contract,
            team_size=n,
        ),
    )
    print("\n  training returned. This script does NOT start evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
