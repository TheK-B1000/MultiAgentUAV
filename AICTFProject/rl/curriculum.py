from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, List

from collections import deque


@dataclass
class CurriculumConfig:
    phases: List[str]
    min_episodes: Dict[str, int]
    min_winrate: Dict[str, float]
    winrate_window: int
    required_win_by: Dict[str, int]
    elo_margin: float
    switch_to_league_after_op3_win: bool = True


@dataclass
class CurriculumState:
    config: CurriculumConfig
    phase_idx: int = 0
    phase_episode_count: int = 0
    recent_results: Dict[str, Deque[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.recent_results = {
            phase: deque(maxlen=int(self.config.winrate_window))
            for phase in self.config.phases
        }

    @property
    def phase(self) -> str:
        return self.config.phases[self.phase_idx]

    def record_result(self, phase: str, win: float) -> None:
        phase = str(phase).upper()
        if phase not in self.recent_results:
            return
        try:
            val = float(win)
        except Exception:
            val = 1.0 if bool(win) else 0.0
        val = max(0.0, min(1.0, val))
        self.recent_results[phase].append(val)

    def phase_winrate(self, phase: str) -> float:
        phase = str(phase).upper()
        recent = self.recent_results.get(phase, None)
        if not recent:
            return 0.0
        return float(sum(recent)) / float(len(recent))

    def should_advance(
        self,
        phase: str,
        learner_rating: float,
        opponent_rating: float,
        win_by: int,
    ) -> bool:
        phase = str(phase).upper()
        min_eps = int(self.config.min_episodes.get(phase, 0))
        min_wr = float(self.config.min_winrate.get(phase, 0.0))
        req_win_by = int(self.config.required_win_by.get(phase, 0))
        winrate = self.phase_winrate(phase)

        meets_score = True if req_win_by <= 0 else (win_by >= req_win_by)
        meets_eps = self.phase_episode_count >= min_eps
        meets_wr = winrate >= min_wr
        meets_elo = float(learner_rating) >= (float(opponent_rating) + float(self.config.elo_margin))
        return bool(meets_eps and meets_wr and meets_score and meets_elo)

    def advance_if_ready(
        self,
        learner_rating: float,
        opponent_rating: float,
        win_by: int,
    ) -> bool:
        if self.phase_idx >= (len(self.config.phases) - 1):
            return False
        phase = self.phase
        if self.should_advance(phase, learner_rating, opponent_rating, win_by):
            self.phase_idx += 1
            self.phase_episode_count = 0
            return True
        return False
