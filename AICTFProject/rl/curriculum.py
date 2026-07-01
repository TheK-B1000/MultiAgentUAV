"""Paper-style OP1 -> OP2 -> OP3 curriculum utilities.

This is the lightweight curriculum path used for Jacob-style baselines:
scripted opponents progress from easy to hard without species or self-play
snapshots mixed into the opponent diet.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


VALID_PHASES = ("OP1", "OP2", "OP3")


def phase_from_tag(tag: str) -> str:
    """Map opponent tags onto the canonical curriculum/stress phase keys."""
    t = str(tag).upper().strip()
    if t in VALID_PHASES:
        return t
    if t in ("OP4", "OP5_RUSHER", "OP5", "OP6", "OP6_TURTLE", "OP7", "OP7_SWITCHER", "SELF_PLAY", ""):
        return "OP3"
    return "OP3"


@dataclass
class CurriculumConfig:
    phases: List[str]
    min_episodes: Dict[str, int]
    min_winrate: Dict[str, float]
    winrate_window: int
    required_win_by: Dict[str, int]
    winrate_window_by_phase: Optional[Dict[str, int]] = None


@dataclass
class CurriculumState:
    """Track phase-local rolling win rates and promotion gates."""

    config: CurriculumConfig
    phase_idx: int = 0
    phase_episode_count: int = 0
    recent_results: Dict[str, Deque[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        by_phase = self.config.winrate_window_by_phase or {}

        def window_for(phase: str) -> int:
            return max(1, int(by_phase.get(phase, self.config.winrate_window)))

        self.config.phases = [str(p).upper() for p in self.config.phases]
        self.recent_results = {
            phase: deque(maxlen=window_for(phase))
            for phase in self.config.phases
        }

    @property
    def phase(self) -> str:
        return self.config.phases[self.phase_idx]

    @property
    def is_final_phase(self) -> bool:
        return self.phase_idx >= len(self.config.phases) - 1

    def record_result(self, phase: str, win: float) -> None:
        phase = str(phase).upper()
        if phase not in self.recent_results:
            return
        val = max(0.0, min(1.0, float(win)))
        self.recent_results[phase].append(val)

    def phase_winrate(self, phase: str | None = None) -> float:
        phase_s = self.phase if phase is None else str(phase).upper()
        recent = self.recent_results.get(phase_s)
        if not recent:
            return 0.0
        return float(sum(recent)) / float(len(recent))

    def should_advance(self, phase: str, win_by: int) -> bool:
        phase_s = str(phase).upper()
        min_eps = int(self.config.min_episodes.get(phase_s, 0))
        min_wr = float(self.config.min_winrate.get(phase_s, 0.0))
        req_win_by = int(self.config.required_win_by.get(phase_s, 0))
        meets_score = True if req_win_by <= 0 else int(win_by) >= req_win_by
        return bool(
            self.phase_episode_count >= min_eps
            and self.phase_winrate(phase_s) >= min_wr
            and meets_score
        )

    def advance_if_ready(self, win_by: int) -> bool:
        if self.is_final_phase:
            return False
        phase_s = self.phase
        if not self.should_advance(phase_s, win_by):
            return False
        self.phase_idx += 1
        self.phase_episode_count = 0
        self.recent_results[self.phase].clear()
        return True


def jacob_paper_curriculum_config(n_agents: int) -> CurriculumConfig:
    """Return the restored OP1/OP2/OP3 gates from the old paper mode."""
    if int(n_agents) > 2:
        min_episodes = {"OP1": 350, "OP2": 300, "OP3": 350}
        window_by_phase = {"OP1": 80, "OP2": 80, "OP3": 120}
    else:
        min_episodes = {"OP1": 200, "OP2": 200, "OP3": 250}
        window_by_phase = {"OP1": 50, "OP2": 50, "OP3": 100}
    return CurriculumConfig(
        phases=["OP1", "OP2", "OP3"],
        min_episodes=min_episodes,
        min_winrate={"OP1": 1.00, "OP2": 0.90, "OP3": 0.80},
        winrate_window=100,
        winrate_window_by_phase=window_by_phase,
        required_win_by={"OP1": 0, "OP2": 1, "OP3": 1},
    )


def jacob_paper_curriculum_state(n_agents: int) -> CurriculumState:
    return CurriculumState(jacob_paper_curriculum_config(n_agents))
