"""
Run the three training modes (4M steps each by default). Run this from the AICTFProject folder.

  cd AICTFProject
  python run_three_modes.py              # one after another
  python run_three_modes.py --parallel   # all three at once (separate processes)

Or run one or more modes:
  python run_three_modes.py --which noleague
  python run_three_modes.py --which selfplay --which fixed   # self-play and fixed only
  python run_three_modes.py --which selfplay
  python run_three_modes.py --which fixed
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Ensure we're in the project root (where rl/ and ctf_sb3_env.py live)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(SCRIPT_DIR) and "rl" in os.listdir(SCRIPT_DIR):
    os.chdir(SCRIPT_DIR)
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)


def _run_one(mode: str, run_tag: str, fixed_tag: str | None, steps: int) -> None:
    """Run a single training in this process."""
    from rl.train_ppo import train_ppo, PPOConfig
    cfg = PPOConfig(mode=mode, run_tag=run_tag, total_timesteps=steps)
    if fixed_tag:
        cfg.fixed_opponent_tag = fixed_tag
    train_ppo(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run curriculum no-league, self-play, or fixed OP3 training (4M steps default).")
    parser.add_argument("--which", type=str, nargs="*", default=["all"],
                        choices=["all", "noleague", "selfplay", "fixed"],
                        help="Which run(s): all, or one or more of noleague, selfplay, fixed (e.g. --which selfplay --which fixed)")
    parser.add_argument("--steps", type=int, default=4_000_000, help="Total timesteps per run (default 4M)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run all three modes at once in separate processes (default: run one after another)")
    args = parser.parse_args()

    which = args.which if isinstance(args.which, list) else [args.which]
    if not which or "all" in which:
        which = ["noleague", "selfplay", "fixed"]
    choices = set(which)

    runs = []
    if "noleague" in choices:
        runs.append(("CURRICULUM_NO_LEAGUE", "ppo_noleague", None))
    if "selfplay" in choices:
        runs.append(("SELF_PLAY", "ppo_selfplay", None))
    if "fixed" in choices:
        runs.append(("FIXED_OPPONENT", "ppo_fixed_op3", "OP3"))

    if args.parallel and len(runs) > 1:
        # Launch each run in a separate process; wait for all to finish
        procs = []
        for mode, run_tag, fixed_tag in runs:
            cmd = [
                sys.executable, os.path.join(SCRIPT_DIR, "rl", "train_ppo.py"),
                "--mode", mode,
                "--run_tag", run_tag,
                "--total_timesteps", str(args.steps),
            ]
            if fixed_tag:
                cmd += ["--fixed_opponent_tag", fixed_tag]
            print(f"[run_three_modes] Starting: mode={mode} run_tag={run_tag} (PID will print in child)")
            p = subprocess.Popen(cmd, cwd=SCRIPT_DIR)
            procs.append((mode, run_tag, p))
        for mode, run_tag, p in procs:
            p.wait()
            status = "OK" if p.returncode == 0 else f"exit {p.returncode}"
            print(f"[run_three_modes] Finished {run_tag}: {status}")
        failed = [r for _, r, p in procs if p.returncode != 0]
        if failed:
            sys.exit(1)
        print("\nAll runs finished successfully.")
    else:
        # Sequential (one after another)
        for i, (mode, run_tag, fixed_tag) in enumerate(runs):
            print(f"\n{'='*60}\nRun {i+1}/{len(runs)}: mode={mode}, run_tag={run_tag}, steps={args.steps}\n{'='*60}\n")
            _run_one(mode, run_tag, fixed_tag, args.steps)
        print("\nAll requested runs finished.")


if __name__ == "__main__":
    main()
