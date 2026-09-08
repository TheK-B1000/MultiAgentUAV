"""Which reward channel dominates, and does any channel rise while the task falls?

Three probes in a row have each moved the failure to whichever shaping channel
still paid. This measures the thing that actually matters for that pattern:
each channel's share of total absolute reward mass, and whether any channel is
rising while task performance collapses.

Absolute mass, not signed mean, because PPO responds to the size of a term's
contribution regardless of sign, and a channel that is large-and-negative shapes
behaviour just as strongly as one that is large-and-positive.

Run:  python experiments/analyze_reward_channel_mass.py <run_dir> [<run_dir> ...]
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

CHANNELS = {
    "terminal": "reward_terminal_mean",
    "sparse": "reward_sparse_mean",
    "failure": "reward_failure_mean",
    "offense": "reward_offense_mean",
    "pbrs": "reward_pbrs_mean",
    "team": "reward_team_mean",
}
EARLY = (0, 50_000)
LATE = (250_000, 310_000)


def _mean(rows, col, lo, hi) -> float:
    vals = [
        float(r[col]) for r in rows
        if col in r and r[col] not in (None, "", "nan")
        and lo <= float(r["timesteps"]) < hi
    ]
    return statistics.fmean(vals) if vals else float("nan")


def analyze(run_dir: Path) -> dict:
    metrics = list(csv.DictReader(open(run_dir / "metrics.csv", encoding="utf-8", newline="")))
    episodes = list(csv.DictReader(open(run_dir / "episode_rows.csv", encoding="utf-8", newline="")))
    n = len(episodes)
    succ_early = statistics.fmean([float(r["success"]) for r in episodes[: n // 6]])
    succ_late = statistics.fmean([float(r["success"]) for r in episodes[-n // 6:]])
    task_down = succ_late < succ_early

    early = {k: _mean(metrics, c, *EARLY) for k, c in CHANNELS.items()}
    late = {k: _mean(metrics, c, *LATE) for k, c in CHANNELS.items()}
    mass = {k: abs(v) for k, v in late.items() if not math.isnan(v)}
    total = sum(mass.values()) or 1e-12

    channels = {}
    for k in CHANNELS:
        e, l = early.get(k, float("nan")), late.get(k, float("nan"))
        rising = (not math.isnan(e) and not math.isnan(l)) and (abs(l) > abs(e))
        channels[k] = {
            "early_mean": None if math.isnan(e) else round(e, 5),
            "late_mean": None if math.isnan(l) else round(l, 5),
            "abs_mass_share_late": round(mass.get(k, 0.0) / total, 4),
            "magnitude_rising": bool(rising),
            # The misalignment signature, per channel.
            "rises_while_task_falls": bool(rising and task_down),
        }

    dominant = max(channels, key=lambda k: channels[k]["abs_mass_share_late"])
    offenders = [k for k, v in channels.items() if v["rises_while_task_falls"]]
    terminal_share = channels["terminal"]["abs_mass_share_late"]

    return {
        "run": run_dir.name,
        "success_early": round(succ_early, 4),
        "success_late": round(succ_late, 4),
        "task_declined": task_down,
        "channels": channels,
        "dominant_channel_by_mass": dominant,
        "terminal_share_of_mass": terminal_share,
        # The structural question: is the objective itself a rounding error?
        "objective_is_minority_of_reward_mass": bool(terminal_share < 0.10),
        "channels_rising_while_task_falls": offenders,
    }


def main() -> int:
    dirs = [Path(a) for a in sys.argv[1:]]
    if not dirs:
        print(__doc__)
        return 2
    out = [analyze(d) for d in dirs if (d / "metrics.csv").is_file()]
    for r in out:
        print(f"\n=== {r['run']} ===")
        print(f"  success {r['success_early']:.3f} -> {r['success_late']:.3f} "
              f"(declined={r['task_declined']})")
        print(f"  {'channel':10s} {'early':>10s} {'late':>10s} {'mass%':>8s}  rising_while_task_falls")
        for k, v in sorted(r["channels"].items(),
                           key=lambda kv: -kv[1]["abs_mass_share_late"]):
            print(f"  {k:10s} {str(v['early_mean']):>10s} {str(v['late_mean']):>10s} "
                  f"{v['abs_mass_share_late']*100:7.2f}%  {v['rises_while_task_falls']}")
        print(f"  dominant={r['dominant_channel_by_mass']} "
              f"terminal_share={r['terminal_share_of_mass']:.4f} "
              f"objective_is_minority={r['objective_is_minority_of_reward_mass']}")
        print(f"  offenders={r['channels_rising_while_task_falls'] or 'none'}")
    Path("artifacts/reward_channel_mass.json").write_text(
        json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nwritten: artifacts/reward_channel_mass.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
