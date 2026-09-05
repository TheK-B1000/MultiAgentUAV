"""Portable team-size launcher for nominal 4v4 / 6v6 POLE SPECIALIST PPO training.

Produces pi_A (Pole A) and pi_B (Pole B) at a given team size -- the first stage of the
route that succeeded at 2v2:

    strategic demand -> specialists -> direct latent distillation -> crossover eval

WHY THIS WRAPPER EXISTS
-----------------------
``run_r1_repertoire_training.py`` resolves the poles through ``pole_A_genome()`` /
``pole_B_genome()``. Those default to ``n_agents=2`` so that all 57 existing 2v2 callers
stay byte-for-byte unchanged. Without this wrapper a "4v4 specialist" would therefore train
in a 4-agent environment while being given the 2v2 strategic pole: the run would look
perfectly healthy and answer the wrong scientific question. That is the same class of bug as
the C7 dual-AGENTS trap, and it is why every check below FAILS CLOSED on the LIVE resolved
profile rather than on a config field.

Team size reaches the run through exactly two paths, and both are asserted:
  1. ``AGENTS`` on the G0 module chain -> ``build_g0_v5_config`` -> ``cfg.max_blue_agents``
  2. ``cfg.max_blue_agents`` -> the R1 seam -> ``pole_{A,B}_genome(N)`` -> live BT profile

PRODUCTION TRAINING IS GATED ON CERTIFICATION
---------------------------------------------
A size may only be trained for production once
``STRATEGIC_DEMAND_<N>v<N>_CERTIFICATION.json`` exists with VERDICT == CERTIFIED. Without it
this script refuses unless ``--smoke`` (which is non-scientific and writes nothing eligible).

This launcher NEVER starts distillation, evaluation or robustness testing.

Run:
  python experiments/train_specialist_scale.py --team-size 4 --policy A --seed <SEED> --device cuda
  python experiments/train_specialist_scale.py --team-size 4 --policy A --seed 99900044 --smoke --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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

SUPPORTED_TEAM_SIZES = (4, 6)
SMOKE_TIMESTEPS = 5_000
SD = PROJECT_ROOT / "artifacts" / "strategic_demand" / "sppo"
BASE_KEY = {"A": "OP6", "B": "OP7"}


def _certification_verdict(n: int) -> tuple[str, Path]:
    p = SD / f"STRATEGIC_DEMAND_{n}v{n}_CERTIFICATION.json"
    if not p.is_file():
        return "MISSING", p
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("VERDICT", "UNKNOWN")), p
    except Exception:  # noqa: BLE001
        return "UNREADABLE", p


def assert_live_pole_matches_team_size(env, policy: str, n: int) -> dict:
    """FAIL CLOSED on the LIVE resolved behaviour-tree profile, not on a config field."""
    from experiments.opponent_spec import pole_A_genome, pole_B_genome

    core = env.core
    want = pole_A_genome(n) if policy == "A" else pole_B_genome(n)
    detail = {"policy": policy, "base_key": BASE_KEY[policy],
              "expected_overlay": dict(want.overlay or {})}

    resolved = None
    for attr in ("_bt_resolved_profile_tensors",):
        fn = getattr(core, attr, None)
        if callable(fn):
            try:
                resolved = fn()
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"FAIL-CLOSED: could not read the live resolved profile: {e}")
    if resolved is None:
        raise SystemExit("FAIL-CLOSED: core exposes no _bt_resolved_profile_tensors(); the "
                         "live pole cannot be verified and this run must not proceed.")

    got = resolved.get("min_alive_for_defender")
    try:
        got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"FAIL-CLOSED: unreadable min_alive_for_defender in the live "
                         f"profile ({got!r}): {e}")
    detail["live_min_alive_for_defender"] = got_val
    if got_val != n:
        raise SystemExit(
            f"FAIL-CLOSED: the LIVE pole resolves min_alive_for_defender={got_val} but the "
            f"team size is {n}. This run would train a {n}v{n} specialist against the "
            f"{got_val}-agent pole definition. Refusing.")
    return detail


def _verify_live_pole(cfg, policy: str, n: int) -> dict:
    """Build a throwaway env, install the pole overlay, and assert the LIVE profile."""
    from experiments.opponent_spec import (install_keyed_opponent_overlays, pole_A_genome,
                                           pole_B_genome)
    from rl.training.env_factory import build_training_env

    env = None
    try:
        env = build_training_env(cfg, initial_phase="phase1",
                                 initial_opponent_tag=BASE_KEY[policy])
        core = env.core
        core._bt_profile_override = None
        core._sds_opening_hold_steps = 0
        genomes = {"OP6": pole_A_genome(n)} if policy == "A" else {}
        if n != 2:
            genomes["OP7"] = pole_B_genome(n)
        install_keyed_opponent_overlays(core, genomes)
        detail = assert_live_pole_matches_team_size(env, policy, n)
        detail["installed_overlay_keys"] = sorted(genomes)
        detail["grid_agent_dim"] = int(env.observation_space.spaces["grid"].shape[0])
        if detail["grid_agent_dim"] != n:
            raise SystemExit(f"FAIL-CLOSED: live env grid agent dim "
                             f"{detail['grid_agent_dim']} != team size {n}")
        return detail
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Nominal 4v4/6v6 pole-specialist PPO launcher.")
    ap.add_argument("--team-size", type=int, required=True, choices=SUPPORTED_TEAM_SIZES)
    ap.add_argument("--policy", required=True, choices=("A", "B"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="NON-SCIENTIFIC short run; requires a 999xxxxx seed")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    n, policy, seed = int(args.team_size), args.policy, int(args.seed)
    is_smoke = bool(args.smoke)

    if is_smoke and not (SMOKE_SEED_MIN <= seed <= SMOKE_SEED_MAX):
        raise SystemExit(f"FAIL-CLOSED: --smoke requires a seed in "
                         f"[{SMOKE_SEED_MIN}, {SMOKE_SEED_MAX}], got {seed}")
    if not is_smoke and SMOKE_SEED_MIN <= seed <= SMOKE_SEED_MAX:
        raise SystemExit(f"FAIL-CLOSED: seed {seed} is reserved for non-scientific smokes")

    verdict, cert_path = _certification_verdict(n)
    if not is_smoke and verdict != "CERTIFIED":
        raise SystemExit(
            f"FAIL-CLOSED: {n}v{n} strategic demand is {verdict} ({cert_path.name}). "
            f"Production specialist training is gated on CERTIFIED: a pole pair that does not "
            f"demand different behaviour cannot support complementary specialists, and "
            f"training them would spend GPU hours on an unanswerable question.")

    try:
        import torch
        torch.set_num_threads(max(1, int(args.threads)))
    except Exception:  # noqa: BLE001
        pass

    touched = _propagate_team_size(n)
    print(f"[specialist_scale] team size {n} propagated to {len(touched)} module(s)")

    import experiments.run_r1_repertoire_training as R

    # The R1 seam asserts seed/budget against POLICIES, so the scaled run declares its own
    # entry there rather than bypassing the guard.
    spec = dict(R.POLICIES[policy])
    spec["seed"] = seed
    if is_smoke and args.total_timesteps is None:
        spec["steps"] = SMOKE_TIMESTEPS
    if args.total_timesteps is not None:
        spec["steps"] = int(args.total_timesteps)
    spec["label"] = (f"smoke_pi_{policy}_specialist_{n}v{n}" if is_smoke
                     else f"pi_{policy}_specialist_{n}v{n}")
    R.POLICIES[policy] = spec

    cfg, contract = R.build_r1_config(policy)
    art = (PROJECT_ROOT / "artifacts" / f"scale_{n}v{n}_specialists" / spec["label"])
    cfg.run_tag = spec["label"]
    cfg.device = str(args.device)
    cfg.checkpoint_dir = str(art / "ckpts")
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")
    if is_smoke:
        cfg.formal_run = False
        cfg.periodic_checkpoint_steps = max(1, int(cfg.total_timesteps) // 2)

    if int(getattr(cfg, "max_blue_agents", -1)) != n:
        raise SystemExit(f"FAIL-CLOSED: cfg.max_blue_agents={getattr(cfg,'max_blue_agents',None)} "
                         f"!= team size {n}; AGENTS propagation did not reach build_r1_config")

    ck = Path(cfg.checkpoint_dir)
    existing = sorted(ck.glob("*.zip")) if ck.is_dir() else []
    if existing:
        raise SystemExit(f"FAIL-CLOSED: {ck} already holds {len(existing)} checkpoint(s); "
                         f"refusing to overwrite. Choose another seed or move them aside.")

    sha, dirty = _git_sha(), _git_dirty()
    print("=" * 78)
    print(f"SPECIALIST_SCALE  pi_{policy}  {n}v{n}   "
          f"{'[SMOKE - NON-SCIENTIFIC]' if is_smoke else '[PRODUCTION]'}")
    print("=" * 78)
    print(f"  utc              {_now()}")
    print(f"  git sha          {sha}{'  (DIRTY)' if dirty else ''}")
    print(f"  pole             {policy}  (base {BASE_KEY[policy]})")
    print(f"  certification    {verdict}  ({cert_path.name})")
    print(f"  team size        {n}v{n}")
    print(f"  seed             {seed}")
    print(f"  device           {cfg.device}")
    print(f"  total timesteps  {int(cfg.total_timesteps):,}")
    print(f"  opponent pool    {getattr(cfg, 'opponent_pool', None)}  "
          f"fixed={getattr(cfg, 'fixed_opponent_tag', None)}")
    print(f"  max_blue_agents  {cfg.max_blue_agents}")
    print(f"  checkpoint dir   {cfg.checkpoint_dir}")
    print("=" * 78, flush=True)

    # FAIL CLOSED on the LIVE resolved pole, before any step. Builds a throwaway env,
    # installs the same overlay the R1 seam installs, and reads the behaviour tree's own
    # resolved tensors -- the only authority that cannot be fooled by an unapplied override.
    live = _verify_live_pole(cfg, policy, n)
    print("  LIVE POLE CHECK: PASS")
    for k, v in live.items():
        print(f"    {k}: {v}")

    art.mkdir(parents=True, exist_ok=True)
    (art / "run_manifest.json").write_text(json.dumps({
        "record": "train_specialist_scale run manifest", "utc": _now(),
        "smoke_non_scientific": is_smoke, "team_size": n, "policy": policy,
        "pole_base": BASE_KEY[policy], "seed": seed, "device": cfg.device,
        "git_sha": sha, "git_dirty": dirty,
        "total_timesteps": int(cfg.total_timesteps),
        "certification_verdict": verdict,
        "checkpoint_dir": cfg.checkpoint_dir,
        "recipe_inherited_from": "experiments/run_r1_repertoire_training.py::build_r1_config",
        "distillation_started_by_this_script": False,
        "evaluation_started_by_this_script": False,
    }, indent=2), encoding="utf-8")

    if args.dry_run:
        print("  --dry-run: config resolved. NOT training.")
        print(f"  -> {art / 'run_manifest.json'}")
        return 0

    print("  starting specialist PPO ...", flush=True)
    R.run_policy(policy, cfg, contract) if hasattr(R, "run_policy") else R.orchestrate_training_run(cfg)
    print("\n  training returned. This script does NOT start distillation or evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
