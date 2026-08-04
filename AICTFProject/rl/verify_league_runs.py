#!/usr/bin/env python3
"""Audit persisted league state so a mislabeled population-based run cannot reach the paper.

Background. Until 2026-08 nothing called ``ROAStarLeague.record_result()`` during
training: ``ROAStarLeagueCallback`` overrode opponent *selection* but no code path
fed match results back. ``win_rate()`` therefore returned None for every opponent,
``_pfsp_weight()`` returned its unplayed-opponent value of 1.0 across the board,
and ``--mode pfsp`` degenerated to a uniform draw -- i.e. it silently ran
fictitious play while being labeled PFSP. Every run finished before the fix has
``"win_rate_stats": {}`` in its ``<run_tag>_league_state.json``.

This script makes that condition detectable instead of invisible. Run it before
using any league checkpoint in a table:

  python rl/verify_league_runs.py --checkpoint-root checkpoints_sb3
  python rl/verify_league_runs.py --checkpoint-root checkpoints_sb3 --strict

``--strict`` exits non-zero when any run fails, so it can gate a results pipeline.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

_MODE_RE = re.compile(r"ppo_roastar_([a-z_]+?)_(\d+)v\d+_seed(\d+)_league_state\.json$")

# Modes whose opponent-selection rule reads back per-opponent results. Uniform
# fictitious play legitimately does not, so an empty stats dict is only a defect
# for these.
RESULT_DEPENDENT_MODES = {"pfsp", "pfsp_exploiter", "do"}


@dataclass
class LeagueRunAudit:
    path: str
    mode: str
    setting: str
    seed: Optional[int]
    n_snapshots: int
    n_opponents_with_results: int
    n_exploiter_snapshots: int
    n_payoff_entries: int

    @property
    def needs_results(self) -> bool:
        return self.mode in RESULT_DEPENDENT_MODES

    @property
    def failures(self) -> List[str]:
        out: List[str] = []
        if self.needs_results and self.n_opponents_with_results == 0:
            out.append(
                "no per-opponent results recorded -- the weighting had no data, so this ran "
                "as UNIFORM sampling (fictitious play), not as its label claims"
            )
        if self.mode == "do" and self.n_payoff_entries == 0:
            out.append("empirical payoff matrix is empty -- the meta-Nash was uninformed")
        if self.mode == "pfsp_exploiter" and self.n_exploiter_snapshots == 0:
            out.append("no exploiter checkpoints entered the pool -- the exploiter stage never fired")
        if self.n_snapshots == 0:
            out.append("no snapshots in the pool -- there was no population to sample from")
        return out

    @property
    def ok(self) -> bool:
        return not self.failures


def audit_league_state(path: str) -> LeagueRunAudit:
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)

    base = os.path.basename(path)
    match = _MODE_RE.search(base)
    if match:
        mode, agents, seed = match.group(1), match.group(2), int(match.group(3))
        setting = f"{agents}v{agents}"
    else:
        mode, setting, seed = "unknown", "?", None

    return LeagueRunAudit(
        path=os.path.abspath(path),
        mode=mode,
        setting=setting,
        seed=seed,
        n_snapshots=len(state.get("snapshots", []) or []),
        n_opponents_with_results=len(state.get("win_rate_stats", {}) or {}),
        n_exploiter_snapshots=len(state.get("exploiter_snapshots", []) or []),
        n_payoff_entries=len(state.get("payoff_stats", []) or []),
    )


def discover_league_states(checkpoint_root: str) -> List[str]:
    pattern = os.path.join(checkpoint_root, "**", "*_league_state.json")
    return sorted(p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint-root", default="checkpoints_sb3")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any run fails, so a results pipeline can gate on it",
    )
    args = parser.parse_args(argv)

    paths = discover_league_states(str(args.checkpoint_root))
    if not paths:
        print(f"[verify_league_runs] no *_league_state.json under {args.checkpoint_root}")
        return 0

    audits = [audit_league_state(p) for p in paths]
    failed = [a for a in audits if not a.ok]

    print(f"[verify_league_runs] audited {len(audits)} league run(s) under {args.checkpoint_root}\n")
    for audit in audits:
        status = "OK  " if audit.ok else "FAIL"
        seed = f"seed{audit.seed}" if audit.seed is not None else "seed?"
        print(
            f"  [{status}] {audit.setting:4s} {audit.mode:16s} {seed:8s} "
            f"snapshots={audit.n_snapshots:3d} "
            f"opponents_with_results={audit.n_opponents_with_results:3d} "
            f"exploiters={audit.n_exploiter_snapshots:2d}"
        )
        for reason in audit.failures:
            print(f"           - {reason}")

    if failed:
        print(
            f"\n[verify_league_runs] {len(failed)} of {len(audits)} run(s) FAILED. "
            "Do not report these under their current method label; retrain after the "
            "record_result fix (see rl/train_ppo_roastar.py)."
        )
        return 1 if args.strict else 0

    print("\n[verify_league_runs] all runs recorded the results their sampling rule depends on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
