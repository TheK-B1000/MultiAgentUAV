from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from collections import deque

from rl.league import EloLeague, OpponentSpec

# Phase contract: stress schedule and set_phase() use only these.
VALID_PHASES = ("OP1", "OP2", "OP3")


def phase_from_tag(tag: str) -> str:
    """
    Map opponent/tier tags to canonical phase for stress schedule.
    Stress keys are OP1 | OP2 | OP3 only; OP3_EASY/OP3_HARD/etc. map to OP3.
    Species types (BALANCED, RUSHER, CAMPER) map to OP3.
    """
    t = str(tag).upper().strip()
    if t in ("OP1", "OP2"):
        return t
    if t in ("OP3", "OP3_EASY", "OP3_HARD", "OP3EASY", "OP3HARD", "SELF_PLAY") or t == "":
        return "OP3"
    # Species types (used in OP3 phase)
    if t in ("BALANCED", "RUSHER", "CAMPER"):
        return "OP3"
    # Unknown tag: use OP3 so stress still applies; caller can fix tag
    import warnings
    warnings.warn(
        f"Unknown phase tag {tag!r}; using OP3 for stress schedule.",
        UserWarning,
        stacklevel=2,
    )
    return "OP3"


# Curriculum Axis 2: environment stress + naval realism by phase.
# Pass to env_method("set_stress_schedule", STRESS_BY_PHASE). When set_phase(phase) is called,
# env applies: physics_enabled (ASV kinematics + maritime sensors), stress knobs, and relaxed_dynamics for OP2.
# OP1 = no physics so blue learns CTF first; OP2 = physics on with relaxed stress; OP3 = full realism.
STRESS_BY_PHASE: Dict[str, Dict[str, Any]] = {
    "OP1": {
        "physics_enabled": False,
    },
    "OP2": {
        "physics_enabled": True,
        "relaxed_dynamics": True,
        "current_strength_cps": 0.02,
        "action_delay_steps": 0,
        "sensor_noise_sigma_cells": 0.05,
        "sensor_dropout_prob": 0.0,
    },
    "OP3": {
        "physics_enabled": True,
        "relaxed_dynamics": False,
        "current_strength_cps": 0.12,
        "drift_sigma_cells": 0.03,
        "action_delay_steps": 2,
        "sensor_noise_sigma_cells": 0.2,
        "sensor_dropout_prob": 0.05,
    },
}


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


@dataclass
class CurriculumControllerConfig:
    seed: int = 42
    op3_tiers: List[str] = field(default_factory=lambda: ["OP3_EASY", "OP3", "OP3_HARD"])
    window: int = 50
    min_episodes_per_tier: int = 80
    promote_winrate: float = 0.65
    demote_winrate: float = 0.25
    easy_op3_trigger_winrate: float = 0.65
    easy_op3_prob: float = 0.60

    enable_species: bool = True
    species_prob: float = 0.05
    allow_species_after: int = 800

    enable_snapshots: bool = False
    snapshot_prob: float = 0.10
    allow_snapshots_after: int = 400


class CurriculumController:
    """
    Curriculum controller for adversarial training.

    Responsibilities:
      - Select red opponent per episode.
      - Adjust difficulty dynamically based on blue performance.
      - Track robustness/generalization metrics.
      - Optionally inject species/self-play opponents later to prevent overfitting.
    """

    def __init__(self, cfg: CurriculumControllerConfig, league: EloLeague) -> None:
        self.cfg = cfg
        self.league = league
        self.rng = random.Random(int(cfg.seed))

        self._tier_idx = 0
        self._tier_recent: Deque[float] = deque(maxlen=int(cfg.window))
        self._overall_recent: Deque[float] = deque(maxlen=int(cfg.window))
        self._results_by_key: Dict[str, Deque[float]] = {}
        self._episodes_in_tier = 0
        self._episode_count = 0

    @property
    def current_tier(self) -> str:
        return str(self.cfg.op3_tiers[self._tier_idx]).upper()

    def record_result(self, opponent_key: str, result: float) -> None:
        try:
            val = float(result)
        except Exception:
            val = 0.0
        val = max(0.0, min(1.0, val))

        self._episode_count += 1
        self._overall_recent.append(val)

        key = str(opponent_key)
        if key not in self._results_by_key:
            self._results_by_key[key] = deque(maxlen=int(self.cfg.window))
        self._results_by_key[key].append(val)

        # Track tier-specific performance only when facing current tier
        tier_key = f"SCRIPTED:{self.current_tier}"
        if key.endswith(self.current_tier) or key == tier_key:
            self._tier_recent.append(val)
            self._episodes_in_tier += 1
            self._maybe_adjust_tier()

    def _maybe_adjust_tier(self) -> None:
        if self._episodes_in_tier < int(self.cfg.min_episodes_per_tier):
            return
        if not self._tier_recent:
            return

        wr = sum(self._tier_recent) / float(len(self._tier_recent))
        if wr >= float(self.cfg.promote_winrate) and self._tier_idx < (len(self.cfg.op3_tiers) - 1):
            self._tier_idx += 1
            self._tier_recent.clear()
            self._episodes_in_tier = 0
        elif wr <= float(self.cfg.demote_winrate) and self._tier_idx > 0:
            self._tier_idx -= 1
            self._tier_recent.clear()
            self._episodes_in_tier = 0

    def select_opponent(self, phase: str, *, league_mode: bool, opponent_stats: Optional[Dict[str, Dict[str, int]]] = None) -> OpponentSpec:
        """
        Select opponent with optional bias toward opponents where win rate is low.
        opponent_stats: Dict[opp_key, {wins, losses, draws, total}] for rolling window.
        """
        if league_mode:
            return self.league.sample_league()

        phase = str(phase).upper()
        if phase != "OP3":
            return self.league.sample_curriculum(phase)
        
        # Use opponent_stats to bias matchmaking: over-sample opponents with low win rate
        opp_winrates: Dict[str, float] = {}
        if opponent_stats is not None and len(opponent_stats) > 0:
            # Compute win rates
            for opp_key, stats in opponent_stats.items():
                wins = stats.get("wins", 0)
                losses = stats.get("losses", 0)
                total = wins + losses
                if total > 0:
                    opp_winrates[opp_key] = wins / float(total)
                else:
                    opp_winrates[opp_key] = 0.5  # Unknown: neutral
        
        # Keep minimum diversity: sample from scripted + species + snapshots with bias
        candidates: List[OpponentSpec] = []
        weights: List[float] = []

        # OP3 adversarial curriculum: add to candidates with bias
        if self.current_tier == "OP3_EASY":
            if len(self._tier_recent) >= int(self.cfg.window):
                wr = sum(self._tier_recent) / float(len(self._tier_recent))
                if wr >= float(self.cfg.easy_op3_trigger_winrate):
                    spec = OpponentSpec(
                        kind="SCRIPTED",
                        key="OP3",
                        rating=self.league.get_rating("SCRIPTED:OP3"),
                    )
                    candidates.append(spec)
                    # Bias: lower win rate = higher weight
                    wr_bias = 1.0 - opp_winrates.get("SCRIPTED:OP3", 0.5) if opponent_stats and len(opp_winrates) > 0 else 1.0
                    weights.append(max(0.1, wr_bias))
        
        # Species opponents
        if (
            self.cfg.enable_species
            and self._episode_count >= int(self.cfg.allow_species_after)
        ):
            try:
                spec = self.league.sample_species()
                candidates.append(spec)
                wr_bias = 1.0 - opp_winrates.get(spec.key, 0.5) if opponent_stats and len(opp_winrates) > 0 else 1.0
                weights.append(max(0.1, wr_bias))
            except Exception:
                pass
        
        # Snapshots
        if (
            self.cfg.enable_snapshots
            and self._episode_count >= int(self.cfg.allow_snapshots_after)
        ):
            try:
                spec = self.league.sample_snapshot()
                candidates.append(spec)
                wr_bias = 1.0 - opp_winrates.get(spec.key, 0.5) if opponent_stats and len(opp_winrates) > 0 else 1.0
                weights.append(max(0.1, wr_bias))
            except Exception:
                pass
        
        # Default scripted opponent
        tag = self.current_tier
        default_spec = OpponentSpec(kind="SCRIPTED", key=tag, rating=self.league.get_rating(f"SCRIPTED:{tag}"))
        candidates.append(default_spec)
        wr_bias = 1.0 - opp_winrates.get(f"SCRIPTED:{tag}", 0.5) if opponent_stats and len(opp_winrates) > 0 else 1.0
        weights.append(max(0.1, wr_bias))
        
        # Sample with bias if we have candidates, otherwise fallback
        if candidates and len(weights) == len(candidates):
            import numpy as np
            weights_np = np.array(weights, dtype=np.float64)
            weights_np = weights_np / weights_np.sum()  # Normalize
            # Use numpy random for weighted sampling (self.rng is random.Random which doesn't support p parameter)
            np_rng = np.random.RandomState(self.rng.randint(0, 2**31))
            idx = np_rng.choice(len(candidates), p=weights_np)
            return candidates[int(idx)]
        
        return default_spec

    def robustness_metrics(self) -> Dict[str, float]:
        winrates = []
        for key, dq in self._results_by_key.items():
            if len(dq) < 5:
                continue
            winrates.append(sum(dq) / float(len(dq)))

        if not winrates:
            return {"robust_min": 0.0, "robust_mean": 0.0, "generalization_std": 0.0}

        mean = sum(winrates) / float(len(winrates))
        var = sum((w - mean) ** 2 for w in winrates) / float(len(winrates))
        std = var ** 0.5
        return {
            "robust_min": min(winrates),
            "robust_mean": mean,
            "generalization_std": std,
        }

    def summary(self) -> Dict[str, float]:
        overall_wr = sum(self._overall_recent) / float(len(self._overall_recent)) if self._overall_recent else 0.0
        metrics = self.robustness_metrics()
        metrics["overall_winrate"] = overall_wr
        metrics["tier"] = float(self._tier_idx)
        return metrics
