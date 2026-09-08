"""Central, verified team-size configuration for the school-lab pipeline.

Team size in this codebase is set through TWO independent paths, and C7's own docstring
records what happens when only one is patched:

    "Patching only the training side trains a 4-agent policy and then evaluates it in a 2v2
     env, which raises 'grid must have shape (B, 4, 7, 20, 20), got (1, 2, 7, 20, 20)' and
     yields TASK_HEALTH=ERROR at every panel. That is unmeasurable rather than failing."

  TRAINING path        PPOConfig.max_blue_agents / max_red_agents
                       (run_g0_v5_long.build_config does cfg.max_blue_agents = AGENTS)
  EVAL/COLLECTION path experiments/r2_learned_crossover.AGENTS, a module global read by
                       build_env() for BOTH max_blue_agents and max_red_agents -- used by
                       ccp_s2_collect, every eval_*.py, and the robustness sweep

This module sets both and then PROVES the result by constructing a real environment and
reading the actual observation shape back, rather than trusting that every patch site was
found. A silently-wrong team size is the single most expensive failure mode available here:
it does not crash, it trains the wrong experiment for ~7 GPU-hours.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUPPORTED = (2, 4, 6)


class TeamSizeError(RuntimeError):
    pass


def set_eval_path_team_size(n: int) -> list[str]:
    """Rebind every module global that feeds env construction on the eval/collection path.

    Returns the list of modules actually patched, so a caller can record it (and so a
    module that silently disappears from the pipeline shows up as a missing patch rather
    than a silent 2-agent default).
    """
    if n not in SUPPORTED:
        raise TeamSizeError(f"team size {n} not in supported {SUPPORTED}")
    patched = []
    import experiments.r2_learned_crossover as R2
    R2.AGENTS = int(n)
    patched.append("experiments.r2_learned_crossover")
    return patched


def apply_to_config(cfg, n: int) -> None:
    """Set the training-path config fields. Both sides, explicitly -- build_config only
    sets max_blue_agents, and leaving red at its default would produce an asymmetric game
    that is not the experiment anyone intends to run."""
    if n not in SUPPORTED:
        raise TeamSizeError(f"team size {n} not in supported {SUPPORTED}")
    cfg.max_blue_agents = int(n)
    cfg.max_red_agents = int(n)


def verify(n: int, device: str = "cpu") -> dict:
    """Construct a REAL env at the requested size and read the actual shapes back.

    This is the check that makes the whole module trustworthy: it does not confirm that
    the patches were applied, it confirms that the environment they produce actually has
    n agents per side.
    """
    import numpy as np
    import experiments.r2_learned_crossover as R2

    if int(getattr(R2, "AGENTS", -1)) != int(n):
        raise TeamSizeError(
            f"R2.AGENTS is {getattr(R2, 'AGENTS', None)}, expected {n} -- call "
            "set_eval_path_team_size() before verify()")

    env = R2.build_env(device, 11_709_001)
    try:
        obs = env.reset()
        grid = np.asarray(obs["grid"])
        core = env.core
        blue = int(core.blue_x.shape[-1])
        red = int(core.red_x.shape[-1])
        # obs grid is (B, n_agents, channels, rows, cols)
        obs_agents = int(grid.shape[1])
        report = {"requested": int(n), "obs_grid_agents": obs_agents,
                  "core_blue_agents": blue, "core_red_agents": red,
                  "grid_shape": tuple(int(x) for x in grid.shape)}
        mismatches = {k: v for k, v in
                      (("obs_grid_agents", obs_agents), ("core_blue_agents", blue),
                       ("core_red_agents", red)) if v != int(n)}
        if mismatches:
            raise TeamSizeError(
                f"team size did not take effect: requested {n}, got {mismatches}. "
                f"Full grid shape {report['grid_shape']}. Some env-construction path was "
                "not patched -- do NOT start a training run in this state.")
        return report
    finally:
        env.close()


def contract_check(n: int, cfg=None, checkpoint: str | Path | None = None,
                   device: str = "cpu", verbose: bool = True) -> dict:
    """The pre-flight guard: training agents == eval agents == policy agents, or REFUSE.

    Prints the TEAM SIZE CONTRACT block and raises TeamSizeError on ANY disagreement. Call
    this immediately before a training or eval run starts consuming GPU time -- C7's
    documented failure (a 4-agent policy silently evaluated in a 2v2 env) is exactly what
    this exists to make impossible.

    cfg        optional PPOConfig -- the training side. Omit for an eval-only check.
    checkpoint optional path -- if given, the model's OWN expected agent count is read back
               from the loaded policy and must also agree.
    """
    env_report = verify(n, device=device)
    rows = [("eval env (blue)", env_report["core_blue_agents"]),
            ("eval env (red / opponent)", env_report["core_red_agents"]),
            ("eval obs grid", env_report["obs_grid_agents"])]

    if cfg is not None:
        rows.append(("training cfg (blue)", int(getattr(cfg, "max_blue_agents", -1))))
        rows.append(("training cfg (red)", int(getattr(cfg, "max_red_agents", -1))))

    if checkpoint is not None:
        import experiments.r2_learned_crossover as R2
        from rl.custom_ppo import load_custom_ppo_policy
        probe = R2.build_env(device, 11_709_002)
        try:
            pol = load_custom_ppo_policy(str(checkpoint), probe.observation_space,
                                        probe.action_space, device=device)
            model = pol.model if hasattr(pol, "model") else pol
            rows.append(("policy (model n_agents)", int(getattr(model, "n_agents", -1))))
        finally:
            probe.close()

    bad = [(label, got) for label, got in rows if got != int(n)]
    status = "PASS" if not bad else "REFUSED"
    if verbose:
        print("TEAM SIZE CONTRACT")
        for label, got in rows:
            mark = "" if got == int(n) else "   <-- MISMATCH"
            print(f"  {label:26s} {got}v{got}{mark}")
        print(f"  STATUS: {status}")
    if bad:
        raise TeamSizeError(
            f"TEAM SIZE CONTRACT REFUSED: requested {n}v{n} but {bad}. Do not start a run "
            "in this state -- a silently-wrong team size does not crash, it trains the "
            "wrong experiment for hours (see C7's own docstring).")
    return {"requested": int(n), "rows": dict(rows), "status": status}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=list(SUPPORTED))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--contract", action="store_true",
                    help="also print the full TEAM SIZE CONTRACT block per size")
    args = ap.parse_args()

    print("TEAM SIZE VERIFICATION (constructs a real env per size, CPU by default)\n")
    for n in args.sizes:
        patched = set_eval_path_team_size(n)
        if args.contract:
            contract_check(n, device=args.device)
            print()
        else:
            rep = verify(n, device=args.device)
            print(f"  [PASS] {n}v{n}: grid {rep['grid_shape']}, "
                  f"blue={rep['core_blue_agents']} red={rep['core_red_agents']}  "
                  f"(patched: {', '.join(patched)})")
    print("ALL PASS")
