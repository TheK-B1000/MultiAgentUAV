#!/usr/bin/env python3
"""Pool admissibility analysis for the scripted-blue × scripted-red payoff matrix.

Answers one question about an opponent pool before any latent / router / PPO
work is spent on it:

    Does this pool make a strategy REPERTOIRE economically necessary?

Headline statistic ``delta_pool`` — pool-level analogue of the V6I26 branch
``delta_oracle`` gate:

    V_selective  = sum_r w_r * max_b  Mbar[b, r]   # blue IDs red, picks best style
    V_best_fixed = max_b  sum_r w_r * Mbar[b, r]   # one style for the whole pool
    delta_pool   = V_selective - V_best_fixed

``delta_pool`` is zero when some single blue style is optimal against the whole
pool (monostyle play is not punished). A dominating blue row forces this.

Bias correction
---------------
In-sample ``max_b`` over noisy cell means is winner's-curse biased upward
(structureless pools can spuriously clear LCB > 0). The gate therefore uses
**cross-fitting**: per red column, choose best responses / best-fixed on one
episode half and score on the frozen other half; average both folds; repeat
over random splits; clustered bootstrap resamples episodes and cross-fits
inside each draw. Same frozen-selector discipline as the branch-level gate.

``V_hindsight`` (per-episode max) is reported as an unachievable upper bound
only — never a gate.

Matched-seed requirement
------------------------
Episode seeds must be ``f(red, episode_index)`` only so every blue style faces
bit-identical red behavior on matched episodes. ``validate_cells`` enforces
equal, index-aligned lengths per red column.

Usage
-----
    from experiments.payoff_matrix_analysis import analyze_pool, format_report
    report = analyze_pool(cells)
    print(format_report(report))
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

CellKey = tuple[str, str]
Cells = Mapping[CellKey, np.ndarray]

# Scripted-blue WR band. Trained PPO will sit higher — re-check saturation
# after the first learned-blue run on an admissible pool.
DEFAULT_WR_BAND = (0.35, 0.55)
DEFAULT_MAX_TIE_RATE = 0.40
DEFAULT_MIN_BR_DIVERSITY = 2
DEFAULT_CROSSFIT_REPS = 50


@dataclass
class PoolReport:
    blues: list[str]
    reds: list[str]
    n_episodes: dict[str, int]

    mean_margin: dict[str, dict[str, float]]
    win_rate: dict[str, dict[str, float]]

    # Descriptive in-sample (biased upward on V_selective — not the gate).
    v_selective_insample: float
    v_best_fixed_insample: float
    best_fixed_style: str

    # Cross-fitted gate statistic.
    v_selective: float
    v_best_fixed: float
    delta_pool: float
    delta_pool_ci95: tuple[float, float]
    delta_pool_lcb: float

    v_hindsight: float

    best_response_by_red: dict[str, str]
    br_diversity: int
    dominating_blue_style: str | None
    degenerate_red_styles: list[str]

    best_blue_overall_wr: float
    tie_rate: float

    gates: dict[str, bool] = field(default_factory=dict)
    admissible: bool = False


def validate_cells(cells: Cells) -> tuple[list[str], list[str]]:
    """Check the matrix is complete and episode-aligned. Returns (blues, reds)."""
    if not cells:
        raise ValueError("cells is empty")
    blues = sorted({b for b, _ in cells})
    reds = sorted({r for _, r in cells})

    missing = [(b, r) for b in blues for r in reds if (b, r) not in cells]
    if missing:
        raise ValueError(f"incomplete matrix, missing cells: {missing}")

    for r in reds:
        lengths = {b: int(len(np.asarray(cells[(b, r)]))) for b in blues}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"red style {r!r} has unequal episode counts across blue styles "
                f"({lengths}) — episodes must be matched by seed, with the seed "
                f"a function of (red, episode_index) only"
            )
        if next(iter(lengths.values())) == 0:
            raise ValueError(f"red style {r!r} has zero episodes")
    return blues, reds


def _stack(cells: Cells, blues: Sequence[str], reds: Sequence[str]) -> dict[str, np.ndarray]:
    """Per red style, a (n_blue, n_episodes) array of margins."""
    return {
        r: np.vstack([np.asarray(cells[(b, r)], dtype=np.float64) for b in blues])
        for r in reds
    }


def _selective_and_fixed(
    per_red: Mapping[str, np.ndarray],
    reds: Sequence[str],
    weights: np.ndarray,
) -> tuple[float, float, int]:
    """In-sample (V_selective, V_best_fixed, argmax_blue). Descriptive only."""
    mbar = np.column_stack([per_red[r].mean(axis=1) for r in reds])
    v_selective = float(np.sum(weights * mbar.max(axis=0)))
    fixed_per_blue = mbar @ weights
    b_star = int(np.argmax(fixed_per_blue))
    return v_selective, float(fixed_per_blue[b_star]), b_star


def _crossfit_delta(
    per_red: Mapping[str, np.ndarray],
    reds: Sequence[str],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Cross-fitted (V_selective, V_best_fixed, delta_pool) — gate statistic.

    Per red column, split episodes in half. Choose per-red best responses and
    the single best-fixed style on the fit half; score on the eval half; swap
    folds and average. Structureless pools yield delta ≈ 0 in expectation.
    """
    folds: list[tuple[float, float]] = []
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for r in reds:
        n = int(per_red[r].shape[1])
        perm = rng.permutation(n)
        splits[r] = (perm[: n // 2], perm[n // 2 :])

    for fit_i, eval_i in ((0, 1), (1, 0)):
        fit = {r: per_red[r][:, splits[r][fit_i]] for r in reds}
        ev = {r: per_red[r][:, splits[r][eval_i]] for r in reds}
        if any(a.shape[1] == 0 for a in fit.values()) or any(
            a.shape[1] == 0 for a in ev.values()
        ):
            continue

        mbar_fit = np.column_stack([fit[r].mean(axis=1) for r in reds])
        mbar_ev = np.column_stack([ev[r].mean(axis=1) for r in reds])

        picks = mbar_fit.argmax(axis=0)
        v_sel = float(np.sum(weights * mbar_ev[picks, np.arange(len(reds))]))

        b_star = int(np.argmax(mbar_fit @ weights))
        v_fix = float(mbar_ev[b_star] @ weights)
        folds.append((v_sel, v_fix))

    if not folds:
        return float("nan"), float("nan"), float("nan")
    v_sel = float(np.mean([f[0] for f in folds]))
    v_fix = float(np.mean([f[1] for f in folds]))
    return v_sel, v_fix, v_sel - v_fix


def analyze_pool(
    cells: Cells,
    *,
    red_weights: Mapping[str, float] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    wr_band: tuple[float, float] = DEFAULT_WR_BAND,
    max_tie_rate: float = DEFAULT_MAX_TIE_RATE,
    min_br_diversity: int = DEFAULT_MIN_BR_DIVERSITY,
    crossfit_reps: int = DEFAULT_CROSSFIT_REPS,
) -> PoolReport:
    """Compute cross-fitted ``delta_pool`` with clustered bootstrap + support gates."""
    blues, reds = validate_cells(cells)
    per_red = _stack(cells, blues, reds)

    if red_weights is None:
        weights = np.full(len(reds), 1.0 / len(reds))
    else:
        w = np.array([float(red_weights[r]) for r in reds], dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError("red_weights must sum to a positive value")
        weights = w / w.sum()

    rng = np.random.default_rng(seed)

    v_sel_insample, v_fix_insample, b_star = _selective_and_fixed(
        per_red, reds, weights
    )

    # Gate: average cross-fitted delta over repeated random splits.
    xf_rows = [_crossfit_delta(per_red, reds, weights, rng) for _ in range(int(crossfit_reps))]
    v_sel = float(np.nanmean([row[0] for row in xf_rows]))
    v_fix = float(np.nanmean([row[1] for row in xf_rows]))
    delta = float(np.nanmean([row[2] for row in xf_rows]))

    # Per-episode hindsight upper bound — reference only, never a gate.
    v_hind = float(
        np.sum(weights * np.array([per_red[r].max(axis=0).mean() for r in reds]))
    )

    # Clustered bootstrap: resample episodes within each red column (blues stay
    # paired), then cross-fit inside each draw.
    boot = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        resampled = {}
        for r in reds:
            arr = per_red[r]
            idx = rng.integers(0, arr.shape[1], size=arr.shape[1])
            resampled[r] = arr[:, idx]
        boot[i] = _crossfit_delta(resampled, reds, weights, rng)[2]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])

    mean_margin = {
        b: {r: float(per_red[r][i].mean()) for r in reds} for i, b in enumerate(blues)
    }
    win_rate = {
        b: {r: float((per_red[r][i] > 0).mean()) for r in reds}
        for i, b in enumerate(blues)
    }

    br = {r: blues[int(np.argmax(per_red[r].mean(axis=1)))] for r in reds}
    br_diversity = len(set(br.values()))

    mbar = np.column_stack([per_red[r].mean(axis=1) for r in reds])
    dominating = None
    for i, b in enumerate(blues):
        others = np.delete(mbar, i, axis=0)
        if others.size == 0:
            continue
        if np.all(mbar[i] >= others.max(axis=0)) and np.any(mbar[i] > others.max(axis=0)):
            dominating = b
            break

    degenerate = [
        r
        for r in reds
        if bool(np.all(per_red[r].mean(axis=1) < 0))
        or bool(np.all(per_red[r].mean(axis=1) > 0))
    ]

    best_blue_wr = float(
        max(
            np.sum(
                weights
                * np.array([(per_red[r][i] > 0).mean() for r in reds])
            )
            for i in range(len(blues))
        )
    )
    tie_rate = float(np.mean([(per_red[r] == 0).mean() for r in reds]))

    gates = {
        "delta_pool_lcb_positive": bool(float(lo) > 0.0),
        "no_dominating_blue_style": dominating is None,
        "best_response_diversity": br_diversity >= int(min_br_diversity),
        "best_blue_wr_in_band": bool(wr_band[0] <= best_blue_wr <= wr_band[1]),
        "tie_rate_under_threshold": bool(tie_rate <= float(max_tie_rate)),
        "no_degenerate_red_styles": len(degenerate) == 0,
    }

    return PoolReport(
        blues=list(blues),
        reds=list(reds),
        n_episodes={r: int(per_red[r].shape[1]) for r in reds},
        mean_margin=mean_margin,
        win_rate=win_rate,
        v_selective_insample=v_sel_insample,
        v_best_fixed_insample=v_fix_insample,
        best_fixed_style=blues[b_star],
        v_selective=v_sel,
        v_best_fixed=v_fix,
        delta_pool=delta,
        delta_pool_ci95=(float(lo), float(hi)),
        delta_pool_lcb=float(lo),
        v_hindsight=v_hind,
        best_response_by_red=br,
        br_diversity=int(br_diversity),
        dominating_blue_style=dominating,
        degenerate_red_styles=list(degenerate),
        best_blue_overall_wr=best_blue_wr,
        tie_rate=tie_rate,
        gates=gates,
        admissible=bool(all(gates.values())),
    )


def format_report(rep: PoolReport) -> str:
    """Human-readable summary for run logs / paper appendices."""
    w = max(len(b) for b in rep.blues) + 2
    lines: list[str] = []
    lines.append("Mean win margin  (rows = blue style, cols = red preset)")
    lines.append(" " * w + "".join(f"{r:>18}" for r in rep.reds))
    for b in rep.blues:
        row = "".join(f"{rep.mean_margin[b][r]:>18.4f}" for r in rep.reds)
        lines.append(f"{b:<{w}}{row}")
    lines.append("")
    lines.append("Win rate")
    lines.append(" " * w + "".join(f"{r:>18}" for r in rep.reds))
    for b in rep.blues:
        row = "".join(f"{rep.win_rate[b][r]:>18.3f}" for r in rep.reds)
        lines.append(f"{b:<{w}}{row}")
    lines.append("")
    lines.append(
        f"V_selective (cross-fit)  = {rep.v_selective:.4f}   "
        f"(insample={rep.v_selective_insample:.4f})"
    )
    lines.append(
        f"V_best_fixed (cross-fit) = {rep.v_best_fixed:.4f}   "
        f"(insample={rep.v_best_fixed_insample:.4f}, style={rep.best_fixed_style})"
    )
    lines.append(
        f"delta_pool              = {rep.delta_pool:.4f}   "
        f"CI95=[{rep.delta_pool_ci95[0]:.4f}, {rep.delta_pool_ci95[1]:.4f}]  "
        f"LCB={rep.delta_pool_lcb:.4f}"
    )
    lines.append(
        f"V_hindsight             = {rep.v_hindsight:.4f}   "
        "(per-episode upper bound, NOT a gate)"
    )
    lines.append("")
    lines.append("Best response by red preset:")
    for r in rep.reds:
        lines.append(f"  {r:<20} -> {rep.best_response_by_red[r]}")
    lines.append(f"  distinct best responses: {rep.br_diversity}/{len(rep.reds)}")
    if rep.dominating_blue_style:
        lines.append(f"  DOMINATING blue style: {rep.dominating_blue_style}")
    if rep.degenerate_red_styles:
        lines.append(f"  degenerate red presets: {rep.degenerate_red_styles}")
    lines.append("")
    lines.append(
        f"best blue overall WR = {rep.best_blue_overall_wr:.3f}   "
        f"tie rate = {rep.tie_rate:.3f}"
    )
    lines.append("")
    lines.append("Gates:")
    for k, v in rep.gates.items():
        lines.append(f"  [{'PASS' if v else 'FAIL'}] {k}")
    lines.append("")
    lines.append(f"POOL ADMISSIBLE: {rep.admissible}")
    if not rep.admissible:
        lines.append("  -> iterate the red presets. Do not train latents on this pool.")
    return "\n".join(lines)


def write_report(rep: PoolReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(rep), f, indent=2)


def cells_from_rows(rows: Iterable[Mapping[str, object]]) -> dict[CellKey, np.ndarray]:
    """Build ``cells`` from rows with blue_style, red_style, episode_index, win_margin."""
    buckets: dict[CellKey, list[tuple[int, float]]] = {}
    for row in rows:
        key = (str(row["blue_style"]), str(row["red_style"]))
        buckets.setdefault(key, []).append(
            (int(row["episode_index"]), float(row["win_margin"]))  # type: ignore[arg-type]
        )
    return {
        k: np.array([m for _, m in sorted(v, key=lambda t: t[0])], dtype=np.float64)
        for k, v in buckets.items()
    }


def _synthetic_saturated(n: int = 64, seed: int = 0) -> dict[CellKey, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        (b, r): np.clip(rng.normal(1.2, 0.4, n).round(), 0, None)
        for b in ("rush", "turtle", "split", "escort")
        for r in ("op7", "op9", "op11", "op12")
    }


def _synthetic_counters(n: int = 64, seed: int = 0) -> dict[CellKey, np.ndarray]:
    rng = np.random.default_rng(seed)
    payoff = {
        ("rush", "bait"): -0.8,
        ("rush", "race"): 0.9,
        ("rush", "collapse"): 0.1,
        ("rush", "flank"): -0.2,
        ("turtle", "bait"): 0.4,
        ("turtle", "race"): -0.9,
        ("turtle", "collapse"): -0.1,
        ("turtle", "flank"): 0.7,
        ("split", "bait"): 0.7,
        ("split", "race"): 0.1,
        ("split", "collapse"): -0.8,
        ("split", "flank"): -0.3,
        ("escort", "bait"): -0.1,
        ("escort", "race"): -0.2,
        ("escort", "collapse"): 0.8,
        ("escort", "flank"): -0.7,
    }
    return {
        (b, r): rng.normal(mu, 0.9, n).round() for (b, r), mu in payoff.items()
    }


if __name__ == "__main__":
    print("=== SATURATED POOL (expected: inadmissible / LCB<=0) ===")
    print(format_report(analyze_pool(_synthetic_saturated(), n_boot=500, seed=0)))
    print("\n\n=== COUNTER-STRUCTURED POOL (expected: LCB>0) ===")
    print(format_report(analyze_pool(_synthetic_counters(), n_boot=500, seed=0)))
