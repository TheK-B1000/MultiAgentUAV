"""Portable team-size launcher for nominal 4v4 / 6v6 PPO training.

ONE SOURCE OF TRUTH FOR TEAM SIZE
---------------------------------
The historical 4v4 path (experiments/run_c7_stage0_4v4.py) rebound TWO module globals --
``run_g0_v5_long.AGENTS`` (training) and ``run_g0_v2_evaluation.AGENTS`` (validation panel).
Patching only the first trains an N-agent policy and then evaluates it in a 2-agent env,
which raises a grid-shape error at every panel and yields TASK_HEALTH=ERROR: unmeasurable
rather than failing.

This launcher removes that hazard structurally rather than by remembering to patch twice:

  * team size is set in exactly ONE place -- ``--team-size`` -> ``G.AGENTS`` -> ``build_config``
  * training is driven through ``rl.train_ppo.train_ppo(cfg)``, NOT ``run_g0_v5_long.main()``,
    so the G0 validation panel (the only consumer of the second binding) is never entered
  * ``assert_team_size_reaches_everything()`` FAILS CLOSED before a single step is taken:
    it checks the config, constructs the real training env and inspects its observation
    space, and verifies that no module carrying a conflicting AGENTS global is loaded

SCIENTIFIC RECIPE IS INHERITED, NOT RESTATED
--------------------------------------------
Reward, opponent pool, architecture, horizon, n_envs/n_steps/batch/epochs and checkpoint
cadence all come from ``run_g0_v5_long.build_config`` unchanged. This file overrides only:
team size, seed, artifact paths, device, and (in --smoke mode) the step budget.

This launcher NEVER starts nominal evaluation or deployment-robustness testing.

Run:
  python experiments/train_scale.py --team-size 4 --seed <SEED> --device cuda
  python experiments/train_scale.py --team-size 6 --seed <SEED> --device cuda
  python experiments/train_scale.py --team-size 4 --seed 99900004 --device cpu --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_TEAM_SIZES = (4, 6)
SMOKE_SEED_MIN, SMOKE_SEED_MAX = 99_900_000, 99_999_999
SMOKE_TIMESTEPS = 5_000


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT),
                             capture_output=True, text=True, timeout=15)
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


#: Every module in the G0 chain that carries its own ``AGENTS`` global. ``run_g0_v5_long``
#: does ``from run_g0_v2_seed import AGENTS``, so rebinding only the former leaves the latter
#: stale at 2 -- and ``run_g0_v2_evaluation`` holds the third binding whose omission caused
#: the C7 "trains at 4, evaluates at 2" failure. All are set together so the value cannot
#: disagree with itself, and ``assert_team_size_reaches_everything`` then verifies it.
_AGENTS_MODULES = (
    "experiments.run_g0_v2_seed",
    "experiments.run_g0_v5_long",
    "experiments.run_g0_v2_evaluation",
)


def _propagate_team_size(team_size: int) -> list[str]:
    """Set the team size on every module carrying an ``AGENTS`` global.

    The named chain is imported first so the transitive imports it pulls in (the health-probe
    and evaluation helpers, each with their own ``AGENTS``) are loaded and therefore visible
    to the sweep. The sweep then rebinds every loaded ``experiments.*`` module rather than a
    hand-maintained list, so a newly added import cannot silently reintroduce a stale binding.
    """
    import importlib

    for name in _AGENTS_MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"FAIL-CLOSED: cannot import {name} to set team size: {e}") from e
        mod = sys.modules[name]
        if not hasattr(mod, "AGENTS"):
            raise SystemExit(f"FAIL-CLOSED: {name} no longer defines AGENTS; the team-size "
                             f"propagation list in train_scale.py is stale and must be updated "
                             f"rather than silently skipped")

    touched = []
    for name, mod in list(sys.modules.items()):
        if not name.startswith("experiments.") or mod is None:
            continue
        val = getattr(mod, "AGENTS", None)
        if isinstance(val, int) and not isinstance(val, bool):
            mod.AGENTS = int(team_size)
            touched.append(name)
    return sorted(touched)


def assert_team_size_reaches_everything(cfg, team_size: int, *, device: str) -> dict:
    """FAIL CLOSED: prove the requested team size reached config, env and observation space.

    Constructs the real training env (the same call the trainer makes) and inspects what
    actually came back. A config field agreeing with itself proves nothing.
    """
    detail: dict = {"requested_team_size": int(team_size)}

    if int(getattr(cfg, "max_blue_agents", -1)) != int(team_size):
        raise SystemExit(f"FAIL-CLOSED: cfg.max_blue_agents="
                         f"{getattr(cfg, 'max_blue_agents', None)} != requested {team_size}")
    detail["cfg_max_blue_agents"] = int(cfg.max_blue_agents)

    from rl.training.env_factory import build_training_env

    env = None
    try:
        env = build_training_env(cfg, initial_phase="phase1", initial_opponent_tag="OP6")
        spaces = env.observation_space.spaces
        grid_agents = int(spaces["grid"].shape[0])
        vec_agents = int(spaces["vec"].shape[0])
        mask_agents = int(spaces["agent_mask"].shape[0])
        n_heads = int(len(env.action_space.nvec))

        problems = []
        if grid_agents != team_size:
            problems.append(f"grid agent dim {grid_agents} != {team_size}")
        if vec_agents != team_size:
            problems.append(f"vec agent dim {vec_agents} != {team_size}")
        if mask_agents != team_size:
            problems.append(f"agent_mask dim {mask_agents} != {team_size}")
        if n_heads != 2 * team_size:
            problems.append(f"action heads {n_heads} != {2 * team_size}")
        if problems:
            raise SystemExit("FAIL-CLOSED: env did not honour team size -- " + "; ".join(problems))

        detail.update({"grid_agent_dim": grid_agents, "vec_agent_dim": vec_agents,
                       "agent_mask_dim": mask_agents, "n_action_heads": n_heads,
                       "grid_shape": tuple(spaces["grid"].shape),
                       "action_space": str(env.action_space)})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass

    # No loaded module may carry an AGENTS global disagreeing with the requested size --
    # this is the check that would have caught the C7 "train at 4, evaluate at 2" failure.
    conflicts = []
    checked = []
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("experiments."):
            continue
        val = getattr(mod, "AGENTS", None)
        if val is None or not isinstance(val, int):
            continue
        checked.append(f"{mod_name}={val}")
        if int(val) != int(team_size):
            conflicts.append(f"{mod_name}.AGENTS={val}")
    if conflicts:
        raise SystemExit("FAIL-CLOSED: a loaded module carries a conflicting AGENTS global -- "
                         + "; ".join(conflicts)
                         + ". Team size must have exactly one source of truth.")
    detail["agents_globals_checked"] = sorted(checked)
    detail["conflicting_agents_globals"] = []
    return detail


def main() -> int:
    ap = argparse.ArgumentParser(description="Nominal 4v4/6v6 PPO training launcher.")
    ap.add_argument("--team-size", type=int, required=True, choices=SUPPORTED_TEAM_SIZES,
                    help="4 or 6. Other sizes are rejected.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None,
                    help="artifact root; default artifacts/scale_<N>v<N>/<run_tag>")
    ap.add_argument("--total-timesteps", type=int, default=None,
                    help="override the inherited budget (smoke use)")
    ap.add_argument("--smoke", action="store_true",
                    help="NON-SCIENTIFIC short run; requires a 999xxxxx seed")
    ap.add_argument("--resume", action="store_true",
                    help="explicitly resume from the newest checkpoint in the run's ckpt dir")
    ap.add_argument("--additional-timesteps", type=int, default=None,
                    help="with --resume: steps to run BEYOND the checkpoint's step")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve config, run fail-closed checks, print, then STOP")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    ts = int(args.team_size)
    seed = int(args.seed)
    is_smoke = bool(args.smoke)

    if is_smoke and not (SMOKE_SEED_MIN <= seed <= SMOKE_SEED_MAX):
        raise SystemExit(f"FAIL-CLOSED: --smoke requires a NON-SCIENTIFIC seed in "
                         f"[{SMOKE_SEED_MIN}, {SMOKE_SEED_MAX}], got {seed}")
    if not is_smoke and SMOKE_SEED_MIN <= seed <= SMOKE_SEED_MAX:
        raise SystemExit(f"FAIL-CLOSED: seed {seed} is in the reserved NON-SCIENTIFIC smoke "
                         f"range; a production run must not use it")

    try:
        import torch
        torch.set_num_threads(max(1, int(args.threads)))
    except Exception:  # noqa: BLE001
        pass

    # --- the ONE place team size is set -------------------------------------------------
    import experiments.run_g0_v5_long as G

    _touched = _propagate_team_size(ts)
    print(f"[train_scale] team size {ts} propagated to {len(_touched)} module(s) "
          f"carrying an AGENTS global: {', '.join(m.split('.')[-1] for m in _touched)}")

    run_tag = (f"smoke_scale_{ts}v{ts}_seed{seed}" if is_smoke
               else f"scale_{ts}v{ts}_seed{seed}")
    art = (Path(args.out_dir).resolve() if args.out_dir
           else PROJECT_ROOT / "artifacts" / f"scale_{ts}v{ts}" / run_tag)

    cfg = G.build_config(seed)
    cfg.run_tag = run_tag
    cfg.device = str(args.device)
    cfg.checkpoint_dir = str(art / "ckpts")
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")
    if is_smoke and args.total_timesteps is None:
        cfg.total_timesteps = SMOKE_TIMESTEPS
    if args.total_timesteps is not None:
        cfg.total_timesteps = int(args.total_timesteps)
    if is_smoke:
        cfg.periodic_checkpoint_steps = max(1, int(cfg.total_timesteps) // 2)
        cfg.formal_run = False

    ckpt_dir = Path(cfg.checkpoint_dir)
    existing = sorted(ckpt_dir.glob("*.zip")) if ckpt_dir.is_dir() else []
    if existing and not args.resume:
        raise SystemExit(
            f"FAIL-CLOSED: {ckpt_dir} already contains {len(existing)} checkpoint(s); "
            f"first={existing[0].name}. Refusing to risk overwriting a terminal checkpoint. "
            f"Pass --resume to continue deliberately, or choose another --out-dir.")

    resume_from = None
    if args.resume:
        if not existing:
            raise SystemExit(f"FAIL-CLOSED: --resume given but {ckpt_dir} holds no checkpoint.")
        # Never resume FROM a terminal checkpoint: it is the preserved scientific artifact.
        periodic = [p for p in existing if not p.name.startswith("final_")]
        if not periodic:
            raise SystemExit(
                f"FAIL-CLOSED: the only checkpoint(s) in {ckpt_dir} are terminal (final_*). "
                f"Resuming would extend a finished run and put its terminal artifact at risk. "
                f"Start a new run directory instead.")
        resume_from = max(periodic, key=lambda p: p.stat().st_mtime)
        cfg.load_path = str(resume_from)
        cfg.load_weights_only = False
        if args.additional_timesteps is None:
            raise SystemExit("FAIL-CLOSED: --resume requires --additional-timesteps N "
                             "(steps to run beyond the checkpoint), so the budget is explicit.")
        cfg.additional_timesteps = int(args.additional_timesteps)

    sha, dirty = _git_sha(), _git_dirty()
    gpu_name = "n/a"
    try:
        import torch
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass

    print("=" * 78)
    print(f"TRAIN_SCALE  {ts}v{ts}   {'[SMOKE - NON-SCIENTIFIC]' if is_smoke else '[PRODUCTION]'}")
    print("=" * 78)
    print(f"  utc              {_now()}")
    print(f"  git sha          {sha}{'  (DIRTY WORKING TREE)' if dirty else ''}")
    print(f"  team size        {ts}v{ts}   (single source of truth: --team-size)")
    print(f"  seed             {seed}")
    print(f"  device           {cfg.device}   gpu={gpu_name}")
    print(f"  total timesteps  {cfg.total_timesteps:,}")
    print(f"  checkpoint dir   {cfg.checkpoint_dir}")
    print(f"  metrics csv      {cfg.metrics_csv_path}")
    print(f"  resume           {bool(args.resume)}"
          + (f"  from={Path(cfg.load_path).name}  +{cfg.additional_timesteps:,} steps"
             if resume_from is not None else ""))
    print("  ---- inherited from run_g0_v5_long.build_config (UNCHANGED) ----")
    for f in ("map_layout", "max_decision_steps", "n_envs", "n_steps", "batch_size",
              "n_epochs", "mode", "opponent_pool", "train_domain_randomization",
              "use_latent_strategy", "gpu_native_env", "periodic_checkpoint_steps"):
        print(f"    {f:32s} {getattr(cfg, f, '<absent>')}")
    print("=" * 78, flush=True)

    detail = assert_team_size_reaches_everything(cfg, ts, device=cfg.device)
    print("  FAIL-CLOSED TEAM-SIZE CHECK: PASS")
    for k, v in detail.items():
        print(f"    {k}: {v}")

    # deployment perturbations must never be on during training
    for field, want in (("train_domain_randomization", False),):
        if bool(getattr(cfg, field, False)) != want:
            raise SystemExit(f"FAIL-CLOSED: {field}={getattr(cfg, field)}; training must be nominal")
    print("  NOMINAL-TRAINING CHECK: PASS (no domain randomization; "
          "no localization/motion/delay perturbation is applied by this path)")

    art.mkdir(parents=True, exist_ok=True)
    manifest = {
        "record": "train_scale run manifest", "utc": _now(), "run_tag": run_tag,
        "smoke_non_scientific": is_smoke, "team_size": ts, "seed": seed,
        "device": cfg.device, "gpu": gpu_name, "git_sha": sha, "git_dirty": dirty,
        "total_timesteps": int(cfg.total_timesteps),
        "checkpoint_dir": cfg.checkpoint_dir, "resume": bool(args.resume),
        "resume_from": (str(resume_from) if resume_from is not None else None),
        "recipe_inherited_from": "experiments/run_g0_v5_long.py::build_config",
        "team_size_check": detail,
        "hyperparameters_unchanged": {
            f: getattr(cfg, f, None) for f in
            ("map_layout", "max_decision_steps", "n_envs", "n_steps", "batch_size",
             "n_epochs", "mode", "train_domain_randomization", "use_latent_strategy")},
        "evaluation_started_by_this_script": False,
        "robustness_started_by_this_script": False,
    }
    manifest["opponent_pool"] = list(getattr(cfg, "opponent_pool", ()) or ())
    (art / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  -> {art / 'run_manifest.json'}")

    if args.dry_run:
        print("\n  --dry-run: config resolved and checks passed. NOT training.")
        return 0

    from rl.train_ppo import train_ppo

    print(f"\n  starting PPO ({cfg.total_timesteps:,} steps) ...", flush=True)
    train_ppo(cfg)
    print("\n  training returned. This script does NOT start evaluation or robustness testing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
