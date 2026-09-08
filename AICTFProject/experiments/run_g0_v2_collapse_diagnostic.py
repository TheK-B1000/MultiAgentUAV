"""G0-v2 Progressive Collapse Diagnostic.

Two of three G0-v2 seeds progressively unlearned the task and converged on a
policy that never crosses the midline. This asks WHEN that happened and WHICH
PPO signal moved first, using only signals already on disk -- no new training.

The four hypotheses under test:

    A  objective misalignment  -- reward rises while gameplay worsens
    B  entropy collapse        -- exploration dies, then behaviour dies
    C  violent updates         -- KL/clip spikes precede collapse
    D  critic destabilisation  -- explained variance falls, advantages blow up

The discriminator between A and the rest is direction: under A the optimizer is
succeeding at the objective it was given, and the objective is wrong. Under
B/C/D the optimizer is failing at a correct objective.

Run:  python experiments/run_g0_v2_collapse_diagnostic.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = (2_500_001, 2_500_002, 2_500_003)
COMPETENT = 2_500_001
COLLAPSED = (2_500_002, 2_500_003)
OUT_DIR = PROJECT_ROOT / "artifacts" / "g0_v2_collapse_diagnostic"
BIN = 25_000
N_BINS = 40

# Signals grouped by the hypothesis each one speaks to.
SIGNALS = {
    "task_proxy": ["rollout_return_mean", "rollout_reward_mean"],
    "objective_A": [
        "reward_terminal_mean", "reward_sparse_mean", "reward_failure_mean",
        "reward_offense_mean", "reward_pbrs_mean", "reward_shaping_mean",
        "reward_shaping_to_outcome_abs_ratio",
    ],
    "entropy_B": ["entropy"],
    "updates_C": ["approx_kl", "clip_fraction", "grad_norm", "learning_rate"],
    "critic_D": ["value_loss", "explained_variance", "return_norm_std"],
}
ALL_SIGNALS = [c for group in SIGNALS.values() for c in group]


def load(seed: int) -> list[dict]:
    p = PROJECT_ROOT / "artifacts" / f"g0_v2_seed{seed}" / "metrics.csv"
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def binned(rows: list[dict], col: str) -> list[float]:
    out = []
    for b in range(N_BINS):
        lo, hi = b * BIN, (b + 1) * BIN
        vals = [
            float(r[col]) for r in rows
            if col in r and r[col] not in (None, "", "nan")
            and lo <= float(r["timesteps"]) < hi
        ]
        out.append(statistics.fmean(vals) if vals else float("nan"))
    return out


def episode_success(seed: int) -> list[float]:
    """Ground-truth task competence per bin, from the stamped episode rows."""
    p = PROJECT_ROOT / "artifacts" / f"g0_v2_seed{seed}" / "episode_rows.csv"
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    # Episode rows carry no timestep column; approximate by even division over
    # the run, which is enough to locate a collapse to within a bin.
    out = []
    n = len(rows)
    for b in range(N_BINS):
        seg = rows[b * n // N_BINS:(b + 1) * n // N_BINS]
        out.append(
            statistics.fmean([float(r["success"]) for r in seg]) if seg else float("nan")
        )
    return out


def first_bin_where(series: list[float], pred) -> int | None:
    for i, v in enumerate(series):
        if not math.isnan(v) and pred(v):
            return i
    return None


def _safe(v):
    return None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {s: load(s) for s in SEEDS}
    success = {s: episode_success(s) for s in SEEDS}
    series = {s: {c: binned(data[s], c) for c in ALL_SIGNALS} for s in SEEDS}

    # --- when did competence die? -----------------------------------------
    # Collapse onset: success falls below half its own early-training level.
    onset = {}
    for s in SEEDS:
        early = statistics.fmean([v for v in success[s][:4] if not math.isnan(v)])
        idx = first_bin_where(success[s], lambda v, e=early: v < 0.5 * e)
        onset[s] = {
            "early_success": round(early, 4),
            "collapse_onset_bin": idx,
            "collapse_onset_step": None if idx is None else idx * BIN,
        }

    # --- Case A: does reward rise while the task gets worse? --------------
    case_a = {}
    for s in SEEDS:
        rew = series[s]["rollout_reward_mean"]
        term = series[s]["reward_terminal_mean"]
        fail = series[s]["reward_failure_mean"]
        sparse = series[s]["reward_sparse_mean"]

        def delta(x):
            a = statistics.fmean([v for v in x[:4] if not math.isnan(v)])
            b = statistics.fmean([v for v in x[36:40] if not math.isnan(v)])
            return round(b - a, 5), round(a, 5), round(b, 5)

        d_rew, rew0, rew1 = delta(rew)
        d_suc, suc0, suc1 = delta(success[s])
        case_a[str(s)] = {
            "reward_start": rew0, "reward_end": rew1, "reward_delta": d_rew,
            "success_start": suc0, "success_end": suc1, "success_delta": d_suc,
            "terminal_start": delta(term)[1], "terminal_end": delta(term)[2],
            "failure_penalty_start": delta(fail)[1], "failure_penalty_end": delta(fail)[2],
            "sparse_start": delta(sparse)[1], "sparse_end": delta(sparse)[2],
            # The signature of objective misalignment.
            "reward_up_while_task_down": bool(d_rew > 0 and d_suc < 0),
        }

    # Magnitude comparison: how loud is shaping next to the true outcome?
    dominance = {}
    for s in SEEDS:
        term = [abs(v) for v in series[s]["reward_terminal_mean"] if not math.isnan(v)]
        fail = [abs(v) for v in series[s]["reward_failure_mean"] if not math.isnan(v)]
        t, f = statistics.fmean(term), statistics.fmean(fail)
        dominance[str(s)] = {
            "mean_abs_terminal": round(t, 5),
            "mean_abs_failure_penalty": round(f, 5),
            "failure_over_terminal_ratio": round(f / t, 1) if t > 0 else None,
        }

    # --- Cases B/C/D: did any of them move BEFORE competence died? --------
    ordering = {}
    for s in COLLAPSED:
        col_bin = onset[s]["collapse_onset_bin"]
        ent = series[s]["entropy"]
        ev = series[s]["explained_variance"]
        kl = series[s]["approx_kl"]
        ref_ent = statistics.fmean([v for v in ent[:4] if not math.isnan(v)])
        ref_kl = statistics.fmean([v for v in kl[:4] if not math.isnan(v)])
        # Explained variance is LOW for every seed at initialisation, including
        # the competent one, so an absolute threshold fires on bin 0 and means
        # nothing. Destabilisation is a fall from an established level, and it
        # only supports case D if it happens BEFORE competence dies.
        ev_ref = statistics.fmean(
            [v for v in ev[4:8] if not math.isnan(v)] or [float("nan")]
        )
        ev_drop = first_bin_where(
            ev[4:], lambda v, r=ev_ref: not math.isnan(r) and v < 0.5 * r
        )
        ordering[str(s)] = {
            "collapse_onset_bin": col_bin,
            "entropy_halved_bin": first_bin_where(ent, lambda v, r=ref_ent: v < 0.5 * r),
            "ev_established_level": None if math.isnan(ev_ref) else round(ev_ref, 4),
            "ev_halved_from_level_bin": None if ev_drop is None else ev_drop + 4,
            "ev_final": _safe(round(statistics.fmean(
                [v for v in ev[36:40] if not math.isnan(v)] or [float("nan")]), 4)),
            "kl_spike_bin": first_bin_where(kl, lambda v, r=ref_kl: v > 3 * r),
            "failure_penalty_halved_bin": first_bin_where(
                series[s]["reward_failure_mean"],
                lambda v, r=statistics.fmean(
                    [x for x in series[s]["reward_failure_mean"][:4] if not math.isnan(x)]
                ): v > 0.5 * r,   # less negative = penalty being avoided
            ),
        }

    verdicts = {
        "A_objective_misalignment": all(
            case_a[str(s)]["reward_up_while_task_down"] for s in COLLAPSED
        ) and not case_a[str(COMPETENT)]["reward_up_while_task_down"],
        "B_entropy_collapse_first": all(
            (ordering[str(s)]["entropy_halved_bin"] is not None
             and ordering[str(s)]["collapse_onset_bin"] is not None
             and ordering[str(s)]["entropy_halved_bin"] < ordering[str(s)]["collapse_onset_bin"])
            for s in COLLAPSED
        ),
        "C_violent_updates_first": all(
            ordering[str(s)]["kl_spike_bin"] is not None for s in COLLAPSED
        ),
        "D_critic_destabilised_first": all(
            (ordering[str(s)]["ev_halved_from_level_bin"] is not None
             and ordering[str(s)]["collapse_onset_bin"] is not None
             and ordering[str(s)]["ev_halved_from_level_bin"]
             < ordering[str(s)]["collapse_onset_bin"])
            for s in COLLAPSED
        ),
    }

    report = {
        "diagnostic": "G0-v2 Progressive Collapse",
        "source": "existing metrics.csv + episode_rows.csv -- no new training",
        "bin_size_steps": BIN,
        "competent_seed": COMPETENT,
        "collapsed_seeds": list(COLLAPSED),
        "collapse_onset": {str(k): v for k, v in onset.items()},
        "case_A_objective_misalignment": case_a,
        "reward_term_dominance": dominance,
        "signal_ordering": ordering,
        "verdicts": verdicts,
        "series": {
            str(s): {c: [_safe(v) for v in series[s][c]] for c in ALL_SIGNALS}
            for s in SEEDS
        },
        "success_series": {str(s): [_safe(v) for v in success[s]] for s in SEEDS},
    }
    (OUT_DIR / "collapse_diagnostic.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    print("=" * 78)
    print("G0-v2 PROGRESSIVE COLLAPSE DIAGNOSTIC")
    print("=" * 78)
    for s in SEEDS:
        o, a = onset[s], case_a[str(s)]
        tag = "COMPETENT" if s == COMPETENT else "COLLAPSED"
        print(f"\nseed {s} [{tag}]")
        print(f"  success {a['success_start']:.3f} -> {a['success_end']:.3f} "
              f"(delta {a['success_delta']:+.3f})")
        print(f"  reward  {a['reward_start']:+.4f} -> {a['reward_end']:+.4f} "
              f"(delta {a['reward_delta']:+.4f})")
        print(f"  terminal {a['terminal_start']:+.4f} -> {a['terminal_end']:+.4f} | "
              f"failure {a['failure_penalty_start']:+.4f} -> {a['failure_penalty_end']:+.4f}")
        print(f"  REWARD UP WHILE TASK DOWN: {a['reward_up_while_task_down']}")
        print(f"  onset bin={o['collapse_onset_bin']} step={o['collapse_onset_step']}")
        d = dominance[str(s)]
        print(f"  |failure| / |terminal| = {d['failure_over_terminal_ratio']}x")
    print("\nsignal ordering (bin index; lower = earlier):")
    for s, v in ordering.items():
        print(f"  seed {s}: {v}")
    print("\nVERDICTS:")
    for k, v in verdicts.items():
        print(f"  {k}: {v}")
    print(f"\nwritten: {OUT_DIR / 'collapse_diagnostic.json'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
