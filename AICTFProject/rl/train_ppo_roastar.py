"""
Stage 3B/4 orchestrator for the adapted-ROA-Star baseline: wires ROAStarLeague's
PFSP opponent sampling (rl/roastar_league.py) and the attacker exploiter
(rl/train_exploiter.py, rl/exploiter_env.py) into rl.train_ppo's existing,
battle-tested CURRICULUM_LEAGUE training loop -- WITHOUT modifying
rl/train_ppo.py on disk.

This works by monkeypatching the module-level names (EloLeague, LeagueCallback,
CallbackList) that rl.train_ppo.train_ppo() looks up at call time, for the
duration of one training call, then restoring them. Every other piece of
train_ppo()'s setup -- env config, curriculum config, PPO hyperparameters,
reward ablation handling, checkpointing, resume-from-checkpoint -- runs
completely unchanged, which is what makes the resulting comparison scientifically
clean: only the opponent-sampling strategy (Elo-distance vs. PFSP) differs.

Modes (the empirical-game-theoretic baseline ladder; only the opponent-selection
rule differs between them, everything else is byte-identical):
    fp              Fictitious play: uniform draw over the pool
                    (rl/egt_league.FictitiousPlayLeague.sample_league_fp).
    do              Double oracle / PSRO: opponents drawn from the pool's
                    meta-Nash, solved by LP over an empirical payoff matrix
                    (rl/egt_league.DoubleOracleLeague.sample_league_do).
    pfsp            Main policy trained with ROAStarLeague.sample_league_pfsp()
                    in place of EloLeague.sample_league(); everything else
                    identical to a normal CURRICULUM_LEAGUE run.
    pfsp_exploiter  pfsp, plus: periodically (by timestep or episode threshold)
                    freeze the current main checkpoint, train one attacker
                    exploiter against it (rl/train_exploiter.py), and register
                    the exploiter's checkpoint into the PFSP snapshot pool via
                    ROAStarLeague.register_exploiter_snapshot() so subsequent
                    sampling can select it -- all within the same training run.

IMPORTANT (fixed 2026-08): match results are now fed back into the league via
ROAStarLeague.record_result() in _EGTLeagueCallbackBase._update_opponent_stats.
Before that fix nothing called record_result, so win_rate() returned None for
every opponent, _pfsp_weight() returned its unplayed-opponent value of 1.0, and
`--mode pfsp` degenerated to a uniform draw -- i.e. it silently ran fictitious
play. Runs whose <run_tag>_league_state.json has an empty "win_rate_stats" were
produced by that path and are NOT valid PFSP runs.

Recommended run tags:
    ppo_roastar_pfsp_2v2_seed42
    ppo_roastar_fp_2v2_seed42
    ppo_roastar_do_2v2_seed42
    ppo_roastar_pfsp_exploiter_2v2_seed42

Usage:
    python rl/train_ppo_roastar.py --mode pfsp --agents 2 --total-steps 1000000 --seed 42
    python rl/train_ppo_roastar.py --mode fp   --agents 2 --total-steps 1000000 --seed 42
    python rl/train_ppo_roastar.py --mode do   --agents 2 --total-steps 1000000 --seed 42
    python rl/train_ppo_roastar.py --mode pfsp_exploiter --agents 2 --total-steps 1000000 --seed 42
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import os
import re
import sys
from typing import Any, Callable, Dict, Optional

# Allow `python rl/train_ppo_roastar.py ...` (sys.path[0] is rl/, not project root).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from stable_baselines3.common.callbacks import BaseCallback

import rl.train_ppo as tp
from rl.egt_league import DoubleOracleLeague, FictitiousPlayLeague
from rl.roastar_league import ROAStarLeague
from rl.train_exploiter import train_attacker_exploiter
from rl.train_ppo import LeagueCallback, OpponentSpec, PPOConfig, TrainMode, _log_line

_RESULT_SCORE = {"WIN": 1.0, "DRAW": 0.5, "LOSS": 0.0}


class _EGTLeagueCallbackBase(LeagueCallback):
    """
    Identical orchestration to LeagueCallback -- curriculum tracking, Elo
    bookkeeping, snapshotting, logging -- only opponent selection in league mode
    differs. Subclasses override _select_league_opponent().

    Also closes the feedback loop the population-based rules need: LeagueCallback
    reports every finished episode to _update_opponent_stats(), so that is where
    the result is forwarded to league.record_result(). Without this the win-rate
    stats stay empty and PFSP/DO silently degrade to a uniform draw.
    """

    def _select_league_opponent(self) -> OpponentSpec:
        raise NotImplementedError

    def _select_next_opponent(self) -> OpponentSpec:
        if not self.league_mode:
            return super()._select_next_opponent()
        return self._select_league_opponent()

    def _update_opponent_stats(self, opp_key: str, result: str):
        # Record before delegating: LeagueCallback's own tracking can be disabled
        # via _enable_opponent_tracking, but the league's stats must not be.
        score = _RESULT_SCORE.get(str(result).upper())
        if score is not None:
            self.league.record_result(str(opp_key), float(score))
        return super()._update_opponent_stats(opp_key, result)


class ROAStarLeagueCallback(_EGTLeagueCallbackBase):
    """PFSP win-rate-weighted sampling instead of Elo-distance matchmaking.
    Requires self.league to be a ROAStarLeague (enforced by _patched_train_ppo)."""

    def _select_league_opponent(self) -> OpponentSpec:
        return self.league.sample_league_pfsp(phase="OP3", enable_snapshots=True)


class FictitiousPlayLeagueCallback(_EGTLeagueCallbackBase):
    """Uniform draw over the pool. Requires self.league to be a FictitiousPlayLeague."""

    def _select_league_opponent(self) -> OpponentSpec:
        return self.league.sample_league_fp(phase="OP3", enable_snapshots=True)


class DoubleOracleLeagueCallback(_EGTLeagueCallbackBase):
    """Meta-Nash draw over the pool. Requires self.league to be a DoubleOracleLeague."""

    def _select_league_opponent(self) -> OpponentSpec:
        return self.league.sample_league_do(phase="OP3", enable_snapshots=True)


# mode -> (league class, callback class)
LEAGUE_MODES: Dict[str, Any] = {
    "pfsp": (ROAStarLeague, ROAStarLeagueCallback),
    "pfsp_exploiter": (ROAStarLeague, ROAStarLeagueCallback),
    "fp": (FictitiousPlayLeague, FictitiousPlayLeagueCallback),
    "do": (DoubleOracleLeague, DoubleOracleLeagueCallback),
}


class ExploiterTriggerCallback(BaseCallback):
    """
    Periodically freezes the current main checkpoint, trains one attacker
    exploiter against it, and registers the resulting checkpoint into the
    shared ROAStarLeague's snapshot pool -- so subsequent PFSP sampling in the
    *same* training run can select it. Runs synchronously inside _on_step
    (main-agent training pauses for the exploiter burst); no threading or
    multiprocessing, which keeps this a single, easy-to-reason-about process.

    Trigger threshold is configurable by timestep and/or episode count (either,
    both, or neither -- pass None to disable one or both trigger types).
    """

    def __init__(
        self,
        *,
        cfg: PPOConfig,
        league: ROAStarLeague,
        every_steps: Optional[int] = None,
        every_episodes: Optional[int] = None,
        exploiter_total_steps: int = 100_000,
        exploiter_n_envs: int = 32,
        league_state_path: Optional[str] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.cfg = cfg
        self.league = league
        self.every_steps = every_steps
        self.every_episodes = every_episodes
        self.exploiter_total_steps = int(exploiter_total_steps)
        self.exploiter_n_envs = int(exploiter_n_envs)
        self.league_state_path = league_state_path
        self._episode_idx = 0
        self._next_step_trigger = every_steps
        self._next_episode_trigger = every_episodes
        self._exploiter_count = 0

    def _on_step(self) -> bool:
        for done in self.locals.get("dones", []):
            if done:
                self._episode_idx += 1

        should_trigger = False
        if (
            self.every_steps is not None
            and self._next_step_trigger is not None
            and self.num_timesteps >= self._next_step_trigger
        ):
            should_trigger = True
            self._next_step_trigger += self.every_steps
        if (
            self.every_episodes is not None
            and self._next_episode_trigger is not None
            and self._episode_idx >= self._next_episode_trigger
        ):
            should_trigger = True
            self._next_episode_trigger += self.every_episodes

        if should_trigger:
            self._run_exploiter_cycle()
        return True

    def _run_exploiter_cycle(self) -> None:
        self._exploiter_count += 1
        frozen_path = os.path.join(
            self.cfg.checkpoint_dir, f"{self.cfg.run_tag}_main_frozen_ep{self._exploiter_count:03d}"
        )
        self.model.save(frozen_path)
        frozen_zip = os.path.abspath(frozen_path + ".zip")
        _log_line(
            f"[ROAStar|Exploiter] step={self.num_timesteps} freezing main -> {frozen_zip}, "
            f"training exploiter #{self._exploiter_count}"
        )
        exploiter_path = train_attacker_exploiter(
            blue_snapshot_path=frozen_zip,
            n_agents=int(getattr(self.cfg, "max_blue_agents", 2)),
            n_envs=self.exploiter_n_envs,
            total_steps=self.exploiter_total_steps,
            device=str(self.cfg.device),
            seed=int(self.cfg.seed) + 9000 + self._exploiter_count,
            run_tag=f"{self.cfg.run_tag}_exploiter{self._exploiter_count:03d}",
            checkpoint_dir=self.cfg.checkpoint_dir,
        )
        exploiter_abspath = os.path.abspath(exploiter_path)
        self.league.register_exploiter_snapshot(exploiter_abspath)
        _log_line(f"[ROAStar|Exploiter] registered exploiter checkpoint into PFSP pool: {exploiter_abspath}")
        if self.league_state_path:
            self.league.save_state(self.league_state_path)


@contextlib.contextmanager
def _patched_train_ppo(
    *,
    mode: str = "pfsp",
    pfsp_p: float = 2.0,
    pfsp_floor: float = 0.05,
    resume_league_state_path: Optional[str] = None,
    extra_callback_factory: Optional[Callable[[ROAStarLeague], BaseCallback]] = None,
):
    """Monkeypatch rl.train_ppo's module-level EloLeague/LeagueCallback/CallbackList
    for the duration of one train_ppo() call, then restore them. See module
    docstring for why this is the chosen integration strategy."""
    if mode not in LEAGUE_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(LEAGUE_MODES)}")
    league_cls, callback_cls = LEAGUE_MODES[mode]
    captured: Dict[str, Any] = {}

    def _league_factory(**kwargs: Any) -> ROAStarLeague:
        league = league_cls(pfsp_p=pfsp_p, pfsp_floor=pfsp_floor, **kwargs)
        if resume_league_state_path:
            if league.load_state_from_file(resume_league_state_path):
                _log_line(
                    f"[ROAStar] resumed league state from {resume_league_state_path} "
                    f"({len(league.snapshots)} snapshots, {len(league.win_rate_stats)} tracked opponents)"
                )
        captured["league"] = league
        return league

    orig_league_cls = tp.EloLeague
    orig_callback_cls = tp.LeagueCallback
    orig_calllist_cls = tp.CallbackList

    class _CallbackListWithExtra(orig_calllist_cls):  # type: ignore[misc,valid-type]
        def __init__(self, callbacks_list):
            league = captured.get("league")
            cbs = list(callbacks_list)
            if extra_callback_factory is not None and league is not None:
                cbs.append(extra_callback_factory(league))
            super().__init__(cbs)

    tp.EloLeague = _league_factory
    tp.LeagueCallback = callback_cls
    tp.CallbackList = _CallbackListWithExtra
    try:
        yield captured
    finally:
        tp.EloLeague = orig_league_cls
        tp.LeagueCallback = orig_callback_cls
        tp.CallbackList = orig_calllist_cls


_SNAPSHOT_EPISODE_RE = re.compile(r"_league_snapshot_ep(\d+)\.zip$")


def find_latest_snapshot(checkpoint_dir: str, run_tag: str) -> Optional[str]:
    """Find the highest-episode league snapshot for a given run_tag, so a run
    that died externally (killed process, sleep/reboot -- not train_ppo's own
    OOM/crash handlers, which already save final_*/oom_save_*/crash_save_*) can
    resume from its most recent weights instead of restarting from scratch."""
    pattern = os.path.join(checkpoint_dir, f"{run_tag}_league_snapshot_ep*.zip")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    best_path: Optional[str] = None
    best_ep = -1
    for path in candidates:
        m = _SNAPSHOT_EPISODE_RE.search(os.path.basename(path))
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_ep:
            best_ep = ep
            best_path = path
    return best_path


def run(args: argparse.Namespace) -> str:
    """Run one PFSP (or PFSP+exploiter) training job. Returns the path to the
    persisted league-state JSON (win-rate stats, snapshot pool, exploiter
    provenance) so a later process can resume with --resume-league-state."""
    cfg = PPOConfig()
    cfg.mode = TrainMode.CURRICULUM_LEAGUE.value
    cfg.max_blue_agents = int(args.agents)
    cfg.total_timesteps = int(args.total_steps)
    cfg.seed = int(args.seed)
    cfg.device = str(args.device)
    cfg.checkpoint_dir = str(args.checkpoint_dir)
    cfg.run_tag = str(args.run_tag)
    if args.load:
        cfg.load_path = args.load

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    league_state_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_league_state.json")

    resume_league_state = bool(args.resume_league_state)
    if getattr(args, "resume_latest", False) and not cfg.load_path:
        snap = find_latest_snapshot(cfg.checkpoint_dir, cfg.run_tag)
        if snap:
            cfg.load_path = snap
            _log_line(f"[ROAStar] --resume-latest: resuming main weights from {snap}")
            if os.path.isfile(league_state_path):
                resume_league_state = True
                _log_line(f"[ROAStar] --resume-latest: will restore PFSP league state from {league_state_path}")
        else:
            _log_line("[ROAStar] --resume-latest: no league snapshot found; starting fresh")

    if getattr(args, "n_envs", None) is not None:
        cfg.n_envs = max(1, int(args.n_envs))
    if getattr(args, "n_steps", None) is not None:
        cfg.n_steps = max(64, int(args.n_steps))

    use_exploiter = (args.mode == "pfsp_exploiter") and not args.disable_exploiter
    extra_cb_factory: Optional[Callable[[ROAStarLeague], BaseCallback]] = None
    if use_exploiter:
        def extra_cb_factory(league: ROAStarLeague) -> BaseCallback:
            return ExploiterTriggerCallback(
                cfg=cfg,
                league=league,
                every_steps=args.exploiter_every_steps,
                every_episodes=args.exploiter_every_episodes,
                exploiter_total_steps=args.exploiter_total_steps,
                exploiter_n_envs=args.exploiter_n_envs,
                league_state_path=league_state_path,
            )

    with _patched_train_ppo(
        mode=str(args.mode),
        pfsp_p=args.pfsp_p,
        pfsp_floor=args.pfsp_floor,
        resume_league_state_path=league_state_path if resume_league_state else None,
        extra_callback_factory=extra_cb_factory,
    ) as captured:
        tp.train_ppo(cfg)
        league = captured.get("league")
        if league is not None:
            league.save_state(league_state_path)
            _log_line(
                f"[ROAStar] persisted league state ({len(league.snapshots)} snapshots, "
                f"{len(league.exploiter_snapshots)} exploiter-origin, "
                f"{len(league.win_rate_stats)} opponents with recorded results) -> {league_state_path}"
            )
            if not league.win_rate_stats:
                _log_line(
                    "[ROAStar] WARNING: no per-opponent results were recorded. PFSP/DO weighting "
                    "had no data to act on, so this run is not a valid population-based run."
                )

    return league_state_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=sorted(LEAGUE_MODES), required=True)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default="checkpoints_sb3/2v2")
    parser.add_argument("--run-tag", default=None, help="Default: ppo_roastar_<mode>_<N>v<N>_seed<seed>")
    parser.add_argument("--load", default=None, help="Resume main-agent PPO weights from this checkpoint")
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from highest-episode <run_tag>_league_snapshot_ep*.zip (and league state JSON if present)",
    )
    parser.add_argument("--pfsp-p", type=float, default=2.0)
    parser.add_argument("--pfsp-floor", type=float, default=0.05)
    parser.add_argument(
        "--resume-league-state",
        action="store_true",
        help="Load persisted <run_tag>_league_state.json if present (win-rate stats, snapshot pool)",
    )
    parser.add_argument(
        "--disable-exploiter",
        action="store_true",
        help="Force-disable the exploiter stage even under --mode pfsp_exploiter",
    )
    parser.add_argument("--exploiter-every-steps", type=int, default=300_000)
    parser.add_argument("--exploiter-every-episodes", type=int, default=None)
    parser.add_argument("--exploiter-total-steps", type=int, default=100_000)
    parser.add_argument("--exploiter-n-envs", type=int, default=32)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Parallel envs for main agent (default: PPOConfig 8). Use 32 to match ablation matrix.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Rollout length per env (default: PPOConfig 2048). Use 512 with --n-envs 32 to match ablations.",
    )
    args = parser.parse_args()

    if args.run_tag is None:
        args.run_tag = f"ppo_roastar_{args.mode}_{args.agents}v{args.agents}_seed{args.seed}"

    league_state_path = run(args)
    print(f"[roastar] done. league state: {league_state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
