"""Causal Strategic Impact Advantage utilities for v5i9.

CSIA consumes forced-z evaluation evidence, not natural rollout telemetry.
The payoff matrix is:

    M[o, z] = E[Return | opponent=o, do(Z=z)]

The centered interaction term removes opponent main effects and latent main
effects, leaving only opponent-latent specificity:

    S[o, z] = M[o, z] - mean_z M[o, *] - mean_o M[*, z] + mean M

The trainer uses ``S`` as a detached, gated reward bonus. This module owns the
math and file parsing so it can be tested without constructing PPO.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch


OPPONENT_ID_TO_TAG: dict[int, str] = {
    0: "OP1",
    1: "OP2",
    2: "OP3",
    3: "OP4",
    4: "OP5",
    5: "OP6",
    6: "OP7",
}


def opponent_tag(value: Any) -> str:
    """Normalize scripted opponent labels to stable public OP tags."""
    if isinstance(value, int):
        return OPPONENT_ID_TO_TAG.get(int(value), f"OP{int(value)}")
    text = str(value if value is not None else "").strip().upper()
    if text.startswith("SCRIPTED:"):
        text = text.split(":", 1)[1].strip()
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return OPPONENT_ID_TO_TAG.get(int(text), f"OP{int(text)}")
    aliases = {
        "OP5_RUSHER": "OP5",
        "OP6_TURTLE": "OP6",
        "OP7_SWITCHER": "OP7",
    }
    return aliases.get(text, text)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def read_csv_rows(path: str | os.PathLike[str] | None) -> list[dict[str, str]]:
    """Read a CSV into dict rows; missing paths are treated as no evidence."""
    if not path:
        return []
    path_s = os.fspath(path)
    if not os.path.isfile(path_s):
        return []
    with open(path_s, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass(frozen=True)
class CSIAGates:
    behavior_spread: bool = False
    interaction_strength: bool = False
    quality_floor: bool = False

    @property
    def passed(self) -> bool:
        return self.behavior_spread and self.interaction_strength and self.quality_floor


@dataclass(frozen=True)
class CSIAAnalysis:
    payoffs: dict[str, dict[int, float]] = field(default_factory=dict)
    counts: dict[str, dict[int, int]] = field(default_factory=dict)
    centered: dict[str, dict[int, float]] = field(default_factory=dict)
    baselines: dict[str, float] = field(default_factory=dict)
    behavior_spread_by_opp: dict[str, float] = field(default_factory=dict)
    specialization_strength: float = 0.0
    router_oracle_gap: float = 0.0
    routing_gain: float = 0.0
    regret_weighted_routing_score: float = 0.0
    oracle_best_z_per_opponent: dict[str, int] = field(default_factory=dict)
    gates: CSIAGates = field(default_factory=CSIAGates)
    payoff_cells: int = 0
    total_count: int = 0

    @property
    def behavior_spread_max(self) -> float:
        values = list(self.behavior_spread_by_opp.values())
        return float(max(values)) if values else 0.0


def build_payoff_matrix(
    rows: Iterable[Mapping[str, Any]],
    *,
    mode_col: str = "mode",
    fixed_mode: str = "fixed_z",
    opponent_col: str = "opponent",
    z_col: str = "z",
    value_col: str = "blue_win_rate",
    count_col: str = "n_episodes_touched",
    min_count_per_cell: int = 1,
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, int]]]:
    """Build weighted ``M[o,z]`` from forced-z rows.

    Duplicate rows are averaged by ``count_col``. Rows with missing z,
    negative z, non-fixed mode, or insufficient counts are ignored.
    """
    sums: dict[str, dict[int, float]] = {}
    counts: dict[str, dict[int, int]] = {}
    min_count = max(1, int(min_count_per_cell))
    for row in rows:
        mode = str(row.get(mode_col, fixed_mode) or fixed_mode).strip().lower()
        if mode != fixed_mode:
            continue
        z = _as_int(row.get(z_col), -1)
        if z < 0:
            continue
        count = max(0, _as_int(row.get(count_col), 1))
        if count < min_count:
            continue
        opp = opponent_tag(row.get(opponent_col))
        if not opp:
            continue
        value = _as_float(row.get(value_col), 0.0)
        sums.setdefault(opp, {})
        counts.setdefault(opp, {})
        sums[opp][z] = sums[opp].get(z, 0.0) + value * float(count)
        counts[opp][z] = counts[opp].get(z, 0) + int(count)

    payoffs: dict[str, dict[int, float]] = {}
    for opp, by_z in sums.items():
        for z, total in by_z.items():
            count = max(1, counts.get(opp, {}).get(z, 0))
            payoffs.setdefault(opp, {})[z] = float(total) / float(count)
    return payoffs, counts


def extract_evidence_metadata(
    rows: Iterable[Mapping[str, Any]],
    *,
    opponent_col: str = "opponent",
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract natural router baseline and forced-z behavior spread rows."""
    baselines: dict[str, float] = {}
    behavior_spread: dict[str, float] = {}
    for row in rows:
        opp = opponent_tag(row.get(opponent_col))
        if not opp:
            continue
        if "natural_win_rate" in row:
            baselines[opp] = _as_float(row.get("natural_win_rate"), 0.0)
        if "forced_z_behavior_spread" in row:
            behavior_spread[opp] = _as_float(row.get("forced_z_behavior_spread"), 0.0)
    return baselines, behavior_spread


def centered_interaction(
    payoffs: Mapping[str, Mapping[int, float]],
) -> tuple[dict[str, dict[int, float]], float]:
    """Return centered interaction matrix and RMS specialization strength."""
    cells: list[tuple[str, int, float]] = [
        (opp, int(z), float(v))
        for opp, by_z in payoffs.items()
        for z, v in by_z.items()
    ]
    if not cells:
        return {}, 0.0
    row_means: dict[str, float] = {}
    col_means: dict[int, float] = {}
    for opp in {opp for opp, _, _ in cells}:
        vals = [v for o, _, v in cells if o == opp]
        row_means[opp] = float(sum(vals) / max(1, len(vals)))
    for z in {z for _, z, _ in cells}:
        vals = [v for _, zz, v in cells if zz == z]
        col_means[z] = float(sum(vals) / max(1, len(vals)))
    grand = float(sum(v for _, _, v in cells) / len(cells))
    centered: dict[str, dict[int, float]] = {}
    sq: list[float] = []
    for opp, z, value in cells:
        s = float(value - row_means[opp] - col_means[z] + grand)
        centered.setdefault(opp, {})[z] = s
        sq.append(s * s)
    strength = math.sqrt(sum(sq) / max(1, len(sq)))
    return centered, float(strength)


def analyze_csia(
    payoffs: Mapping[str, Mapping[int, float]],
    counts: Mapping[str, Mapping[int, int]] | None = None,
    *,
    baselines: Mapping[str, float] | None = None,
    behavior_spread_by_opp: Mapping[str, float] | None = None,
    min_behavior_spread: float = 0.10,
    min_interaction_strength: float = 0.05,
    quality_floor_delta: float = 0.10,
) -> CSIAAnalysis:
    """Compute CSIA gates and router-vs-random-vs-oracle metrics."""
    payoff_dict = {str(o): {int(z): float(v) for z, v in by_z.items()} for o, by_z in payoffs.items()}
    count_dict = {
        str(o): {int(z): int(c) for z, c in by_z.items()}
        for o, by_z in (counts or {}).items()
    }
    baseline_dict = {opponent_tag(o): float(v) for o, v in (baselines or {}).items()}
    spread_dict = {
        opponent_tag(o): float(v)
        for o, v in (behavior_spread_by_opp or {}).items()
    }
    centered, strength = centered_interaction(payoff_dict)
    cells = [(opp, z, value) for opp, by_z in payoff_dict.items() for z, value in by_z.items()]
    total_count = sum(count_dict.get(opp, {}).get(z, 1) for opp, z, _ in cells)

    gate_a = bool(spread_dict) and max(spread_dict.values()) >= float(min_behavior_spread)
    gate_b = strength >= float(min_interaction_strength)
    gate_c = bool(baseline_dict) and bool(cells)
    if gate_c:
        for opp, z, value in cells:
            baseline = baseline_dict.get(opp)
            if baseline is None:
                gate_c = False
                break
            if value < baseline - float(quality_floor_delta):
                gate_c = False
                break

    oracle_best: dict[str, int] = {}
    weighted_gap = 0.0
    weighted_gain = 0.0
    weighted_score = 0.0
    metric_weight = 0.0
    score_weight = 0.0
    for opp, by_z in payoff_dict.items():
        if not by_z:
            continue
        row_count = sum(count_dict.get(opp, {}).values()) or len(by_z)
        row_random = float(sum(by_z.values()) / len(by_z))
        best_z, best_wr = max(by_z.items(), key=lambda item: item[1])
        oracle_best[opp] = int(best_z)
        router_wr = baseline_dict.get(opp)
        if router_wr is None:
            continue
        weighted_gap += (float(best_wr) - float(router_wr)) * float(row_count)
        weighted_gain += (float(router_wr) - row_random) * float(row_count)
        metric_weight += float(row_count)
        denom = float(best_wr) - row_random
        if abs(denom) > 1e-9:
            weighted_score += ((float(router_wr) - row_random) / denom) * float(row_count)
            score_weight += float(row_count)

    router_oracle_gap = weighted_gap / metric_weight if metric_weight > 0.0 else 0.0
    routing_gain = weighted_gain / metric_weight if metric_weight > 0.0 else 0.0
    regret_score = weighted_score / score_weight if score_weight > 0.0 else 0.0

    return CSIAAnalysis(
        payoffs=payoff_dict,
        counts=count_dict,
        centered=centered,
        baselines=baseline_dict,
        behavior_spread_by_opp=spread_dict,
        specialization_strength=float(strength),
        router_oracle_gap=float(router_oracle_gap),
        routing_gain=float(routing_gain),
        regret_weighted_routing_score=float(regret_score),
        oracle_best_z_per_opponent=oracle_best,
        gates=CSIAGates(
            behavior_spread=gate_a,
            interaction_strength=gate_b,
            quality_floor=gate_c,
        ),
        payoff_cells=len(cells),
        total_count=int(total_count),
    )


def load_csia_analysis_from_csv(
    payoff_csv_path: str | os.PathLike[str] | None,
    *,
    strategy_evidence_csv_path: str | os.PathLike[str] | None = None,
    min_count_per_cell: int = 1,
    min_behavior_spread: float = 0.10,
    min_interaction_strength: float = 0.05,
    quality_floor_delta: float = 0.10,
) -> CSIAAnalysis:
    payoff_rows = read_csv_rows(payoff_csv_path)
    evidence_rows = read_csv_rows(strategy_evidence_csv_path)
    payoffs, counts = build_payoff_matrix(
        payoff_rows,
        min_count_per_cell=min_count_per_cell,
    )
    baselines, behavior_spread = extract_evidence_metadata(evidence_rows)
    return analyze_csia(
        payoffs,
        counts,
        baselines=baselines,
        behavior_spread_by_opp=behavior_spread,
        min_behavior_spread=min_behavior_spread,
        min_interaction_strength=min_interaction_strength,
        quality_floor_delta=quality_floor_delta,
    )


class CSIARewardModel:
    """Detached trainer-side reward bonus backed by forced-z evidence."""

    def __init__(
        self,
        *,
        enabled: bool,
        reward_coef: float,
        payoff_csv_path: str = "",
        strategy_evidence_csv_path: str = "",
        probe_interval: int = 1,
        min_behavior_spread: float = 0.10,
        min_interaction_strength: float = 0.05,
        quality_floor_delta: float = 0.10,
        require_gates: bool = True,
        min_count_per_cell: int = 1,
    ) -> None:
        self.enabled = bool(enabled)
        self.reward_coef = max(0.0, float(reward_coef))
        self.payoff_csv_path = str(payoff_csv_path or "")
        self.strategy_evidence_csv_path = str(strategy_evidence_csv_path or "")
        self.probe_interval = max(0, int(probe_interval))
        self.min_behavior_spread = float(min_behavior_spread)
        self.min_interaction_strength = float(min_interaction_strength)
        self.quality_floor_delta = float(quality_floor_delta)
        self.require_gates = bool(require_gates)
        self.min_count_per_cell = max(1, int(min_count_per_cell))
        self.analysis = CSIAAnalysis()
        self.last_refresh_update: int | None = None
        self.last_refresh_timestamp: float = 0.0
        self.last_bonus_mean: float = 0.0

    @classmethod
    def from_config(cls, cfg: Any) -> "CSIARewardModel":
        return cls(
            enabled=bool(getattr(cfg, "csia_enabled", False)),
            reward_coef=float(getattr(cfg, "csia_reward_coef", 0.0) or 0.0),
            payoff_csv_path=str(getattr(cfg, "csia_payoff_csv_path", "") or ""),
            strategy_evidence_csv_path=str(
                getattr(cfg, "csia_strategy_evidence_csv_path", "") or ""
            ),
            probe_interval=int(getattr(cfg, "csia_probe_interval", 1)),
            min_behavior_spread=float(
                getattr(cfg, "csia_min_behavior_spread", 0.10) or 0.0
            ),
            min_interaction_strength=float(
                getattr(cfg, "csia_min_interaction_strength", 0.05) or 0.0
            ),
            quality_floor_delta=float(
                getattr(cfg, "csia_quality_floor_delta", 0.10) or 0.0
            ),
            require_gates=bool(getattr(cfg, "csia_require_gates", True)),
            min_count_per_cell=int(getattr(cfg, "csia_min_count_per_cell", 1) or 1),
        )

    @property
    def bonus_active(self) -> bool:
        if not self.enabled or self.reward_coef <= 0.0 or self.analysis.payoff_cells <= 0:
            return False
        return self.analysis.gates.passed if self.require_gates else True

    def refresh_if_due(self, update: int) -> None:
        if not self.enabled:
            return
        update_i = int(update)
        if not self.payoff_csv_path:
            self.last_refresh_update = update_i
            return
        if self.last_refresh_update is not None:
            if self.probe_interval <= 0:
                return
            if update_i - int(self.last_refresh_update) < self.probe_interval:
                return
        self.analysis = load_csia_analysis_from_csv(
            self.payoff_csv_path,
            strategy_evidence_csv_path=self.strategy_evidence_csv_path,
            min_count_per_cell=self.min_count_per_cell,
            min_behavior_spread=self.min_behavior_spread,
            min_interaction_strength=self.min_interaction_strength,
            quality_floor_delta=self.quality_floor_delta,
        )
        self.last_refresh_update = update_i
        self.last_refresh_timestamp = float(time.time())

    def bonus(
        self,
        opponent_ids: torch.Tensor,
        z_ids: torch.Tensor,
        *,
        device: torch.device | str,
        update: int,
    ) -> torch.Tensor:
        self.refresh_if_due(update)
        opp_flat = opponent_ids.detach().long().reshape(-1).cpu().tolist()
        z_flat = z_ids.detach().long().reshape(-1).cpu().tolist()
        values = torch.zeros(len(z_flat), dtype=torch.float32, device=device)
        if not self.bonus_active:
            self.last_bonus_mean = 0.0
            return values.reshape(z_ids.shape).detach()
        for idx, (opp_id, z) in enumerate(zip(opp_flat, z_flat)):
            opp = opponent_tag(int(opp_id))
            centered = self.analysis.centered.get(opp, {}).get(int(z))
            if centered is not None:
                values[idx] = float(self.reward_coef) * float(centered)
        self.last_bonus_mean = (
            float(values.mean().detach().cpu().item()) if values.numel() > 0 else 0.0
        )
        return values.reshape(z_ids.shape).detach()

    def stats(self) -> dict[str, Any]:
        a = self.analysis
        return {
            "csia_interaction_strength": float(a.specialization_strength),
            "centered_advantage_matrix": json.dumps(a.centered, sort_keys=True),
            "oracle_best_z_per_opponent": json.dumps(
                a.oracle_best_z_per_opponent, sort_keys=True
            ),
            "router_oracle_gap": float(a.router_oracle_gap),
            "routing_gain": float(a.routing_gain),
            "regret_weighted_routing_score": float(a.regret_weighted_routing_score),
            "gate_A_pass": 1.0 if a.gates.behavior_spread else 0.0,
            "gate_B_pass": 1.0 if a.gates.interaction_strength else 0.0,
            "gate_C_pass": 1.0 if a.gates.quality_floor else 0.0,
            "csia_bonus_active": 1.0 if self.bonus_active else 0.0,
            "csia_payoff_cells": int(a.payoff_cells),
            "csia_total_count": int(a.total_count),
            "csia_behavior_spread_max": float(a.behavior_spread_max),
            "csia_bonus_mean": float(self.last_bonus_mean),
            "csia_last_refresh_update": (
                "" if self.last_refresh_update is None else int(self.last_refresh_update)
            ),
            "csia_reward_coef": float(self.reward_coef),
        }


__all__ = [
    "CSIAAnalysis",
    "CSIAGates",
    "CSIARewardModel",
    "analyze_csia",
    "build_payoff_matrix",
    "centered_interaction",
    "extract_evidence_metadata",
    "load_csia_analysis_from_csv",
    "opponent_tag",
    "read_csv_rows",
]
