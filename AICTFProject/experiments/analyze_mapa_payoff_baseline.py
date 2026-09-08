#!/usr/bin/env python3
"""Per-red summary for canonical map_a scripted payoff baseline matrices."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AGGRESSIVE = ("BLUE_RUSH", "BLUE_SPLIT", "BLUE_ESCORT")
NEAR_MAX_MARGIN = 2.5


def _paired_gap(df: pd.DataFrame, red: str, best: str, runner: str) -> float:
    sub = df[df["red_style"] == red]
    by_ep = sub.pivot_table(
        index="episode_index", columns="blue_style", values="win_margin", aggfunc="first"
    )
    if best not in by_ep.columns or runner not in by_ep.columns:
        return float("nan")
    return float((by_ep[best] - by_ep[runner]).mean())


def analyze(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    if "map" in df.columns:
        df = df[df["map"].isin(("map_a", "map_a_open"))]
    out: dict = {"reds": {}, "map": "map_a"}
    for red in sorted(df["red_style"].unique()):
        sub = df[df["red_style"] == red]
        by_style: dict = {}
        for style in sorted(sub["blue_style"].unique()):
            s = sub[sub["blue_style"] == style]
            by_style[style] = {
                "n": int(len(s)),
                "mean_margin": float(s["win_margin"].mean()),
                "win_rate": float(s["success"].mean()),
                "mean_first_pickup": float(s.loc[s["time_to_first_score"].notna(), "time_to_first_score"].mean())
                if s["time_to_first_score"].notna().any()
                else None,
                "pickup_rate": float((s["blue_score"] > 0).mean())
                if "blue_score" in s.columns
                else None,
            }
        margins = {k: v["mean_margin"] for k, v in by_style.items()}
        best = max(margins, key=margins.get)
        runner = max((k for k in margins if k != best), key=margins.get, default=best)
        agg_margins = [margins[s] for s in AGGRESSIVE if s in margins]
        sat = (
            len(agg_margins) >= 2
            and min(agg_margins) >= NEAR_MAX_MARGIN
            and max(agg_margins) - min(agg_margins) <= 0.5
        )
        out["reds"][red] = {
            "by_blue_style": by_style,
            "uniquely_best": best,
            "runner_up": runner,
            "paired_best_minus_runner_up": _paired_gap(df, red, best, runner),
            "saturation_like": bool(sat),
            "aggressive_margin_spread": float(max(agg_margins) - min(agg_margins))
            if agg_margins
            else None,
        }
    return out


def format_text(report: dict) -> str:
    lines = ["Canonical map_a payoff baseline — per-red summary", ""]
    for red, r in report["reds"].items():
        lines.append(f"=== {red} ===")
        for style, s in r["by_blue_style"].items():
            fp = s["mean_first_pickup"]
            fp_s = f"{fp:.1f}" if fp is not None else "n/a"
            lines.append(
                f"  {style:<14} margin={s['mean_margin']:+.3f}  WR={s['win_rate']:.3f}  "
                f"first_score_step≈{fp_s}"
            )
        lines.append(
            f"  best={r['uniquely_best']}  runner_up={r['runner_up']}  "
            f"paired_gap={r['paired_best_minus_runner_up']:+.3f}  "
            f"saturation_like={r['saturation_like']}  "
            f"agg_spread={r['aggressive_margin_spread']}"
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = analyze(Path(args.csv))
    (out_dir / "mapa_per_red_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = format_text(report)
    (out_dir / "mapa_per_red_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
