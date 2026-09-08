"""Sweep OP12 opening-escort activation gates from live score traces.

This is detector-only analysis. It does not run payoff acceptance and should be
used before enabling or tuning OP12 punitive responses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_scripted_style_payoff_matrix import _make_env, _zero_action


BLUE_STYLES = ("BLUE_RUSH", "BLUE_ESCORT", "BLUE_SPLIT", "BLUE_TURTLE")
RED_PRESET = "OP12_LATE_CONVERTER"
MAP_NAME = "map_b_split_lane"


def _scalar(core, attr: str, default: float = 0.0) -> float:
    val = getattr(core, attr, None)
    if val is None:
        return float(default)
    try:
        return float(val[0].item())
    except Exception:
        return float(default)


def _run_trace(style: str, seed: int, opening_steps: int, max_steps: int, device: str) -> dict:
    env = _make_env(map_name=MAP_NAME, seed=seed, max_decision_steps=max_steps, device=device)
    try:
        env.env_method("set_phase", RED_PRESET)
        env.env_method("set_next_opponent", "SCRIPTED", RED_PRESET)
        env.reset()
        core = env.core
        env.env_method("set_phase", RED_PRESET)
        env.env_method("set_next_opponent", "SCRIPTED", RED_PRESET)
        core.blue_scripted = True
        core.set_blue_style(style)

        trace = []
        first_pickup = None
        for step in range(max_steps):
            env.step_async(_zero_action(env))
            _, _, done, _ = env.step_wait()
            core = env.core
            carrying = any(bool(core.blue_carrying[0, i].item()) for i in range(2))
            if first_pickup is None and carrying:
                first_pickup = step + 1
            if step < opening_steps and first_pickup is None:
                trace.append(
                    {
                        "step": step + 1,
                        "score": _scalar(core, "bt_adapt_opening_escort_score", 0.0),
                        "compact": _scalar(core, "bt_adapt_opening_escort_compact", 0.0),
                        "narrow": _scalar(core, "bt_adapt_opening_escort_narrow", 0.0),
                        "leader": _scalar(core, "bt_adapt_opening_escort_leader", 0.0),
                        "heading": _scalar(core, "bt_adapt_opening_escort_heading", 0.0),
                        "speed_penalty": _scalar(core, "bt_adapt_opening_escort_speed_penalty", 0.0),
                    }
                )
            if bool(done.any()):
                break
        return {
            "blue_style": style,
            "seed": seed,
            "first_pickup": first_pickup,
            "trace": trace,
        }
    finally:
        env.close()


def _longest_run(flags: list[bool]) -> int:
    best = cur = 0
    for flag in flags:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return best


def _first_gate_step(flags: list[bool], trace: list[dict], mode: str, threshold: float, window: int = 3, evidence: float = 0.0):
    if mode == "consecutive":
        cur = 0
        for row, flag in zip(trace, flags):
            cur = cur + 1 if flag else 0
            if cur >= window:
                return row["step"]
        return None
    if mode == "two_of_three":
        for i in range(len(flags)):
            lo = max(0, i - 2)
            if sum(flags[lo : i + 1]) >= 2:
                return trace[i]["step"]
        return None
    if mode == "rolling_evidence":
        scores = [max(0.0, row["score"] - threshold) for row in trace]
        for i in range(len(scores)):
            lo = max(0, i - window + 1)
            if sum(scores[lo : i + 1]) >= evidence:
                return trace[i]["step"]
        return None
    raise ValueError(mode)


def _summarize_trace(trace: list[dict], threshold: float) -> dict:
    scores = [row["score"] for row in trace]
    flags = [s >= threshold for s in scores]
    return {
        "max_score": max(scores) if scores else 0.0,
        "steps_above": sum(flags),
        "longest_run_above": _longest_run(flags),
        "cumulative_above": sum(max(0.0, s - threshold) for s in scores),
        "first_crossing_step": next((row["step"] for row, flag in zip(trace, flags) if flag), None),
    }


def _gate_specs(thresholds: list[float]) -> list[dict]:
    specs = []
    for tau in thresholds:
        specs.append({"name": f"tau{tau:.2f}_consec3", "mode": "consecutive", "threshold": tau, "window": 3})
        specs.append({"name": f"tau{tau:.2f}_consec2", "mode": "consecutive", "threshold": tau, "window": 2})
        specs.append({"name": f"tau{tau:.2f}_2of3", "mode": "two_of_three", "threshold": tau, "window": 3})
        for ev in (0.30, 0.50, 0.70):
            specs.append(
                {
                    "name": f"tau{tau:.2f}_roll5_ev{ev:.2f}",
                    "mode": "rolling_evidence",
                    "threshold": tau,
                    "window": 5,
                    "evidence": ev,
                }
            )
    return specs


def _evaluate_gate(rows: list[dict], spec: dict) -> dict:
    by_style = {}
    for style in BLUE_STYLES:
        style_rows = [row for row in rows if row["blue_style"] == style]
        triggers = []
        for row in style_rows:
            flags = [step["score"] >= spec["threshold"] for step in row["trace"]]
            trig = _first_gate_step(
                flags,
                row["trace"],
                spec["mode"],
                spec["threshold"],
                int(spec.get("window", 3)),
                float(spec.get("evidence", 0.0)),
            )
            triggers.append(trig)
        by_style[style] = {
            "trigger_count": sum(t is not None for t in triggers),
            "mean_trigger_step": mean([t for t in triggers if t is not None]) if any(t is not None for t in triggers) else None,
        }
    return {"gate": spec["name"], "spec": spec, "styles": by_style}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=551001)
    parser.add_argument("--opening-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="AICTFProject/artifacts/op12_dev11_escort_gate_sweep_8seed")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[2.7, 2.8, 2.9, 3.0, 3.1, 3.2])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    for style in BLUE_STYLES:
        for ep in range(args.episodes):
            seed = args.base_seed + ep
            row = _run_trace(style, seed, args.opening_steps, args.max_steps, args.device)
            traces.append(row)
            print(f"{style} ep={ep} seed={seed} pickup={row['first_pickup']} max_score={max([x['score'] for x in row['trace']] or [0.0]):.3f}", flush=True)

    step_rows = []
    summary_rows = []
    for row in traces:
        for step_row in row["trace"]:
            step_rows.append({"blue_style": row["blue_style"], "seed": row["seed"], **step_row})
        for tau in args.thresholds:
            summary_rows.append({"blue_style": row["blue_style"], "seed": row["seed"], "threshold": tau, **_summarize_trace(row["trace"], tau)})

    with (out_dir / "score_traces.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
        writer.writeheader()
        writer.writerows(step_rows)
    with (out_dir / "threshold_episode_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    gate_results = [_evaluate_gate(traces, spec) for spec in _gate_specs(args.thresholds)]
    viable = [
        result
        for result in gate_results
        if result["styles"]["BLUE_ESCORT"]["trigger_count"] >= 7
        and result["styles"]["BLUE_RUSH"]["trigger_count"] <= 1
        and result["styles"]["BLUE_SPLIT"]["trigger_count"] == 0
    ]
    report = {
        "red_preset": RED_PRESET,
        "map": MAP_NAME,
        "episodes": args.episodes,
        "thresholds": args.thresholds,
        "gate_results": gate_results,
        "viable_gates": viable,
        "decision": "DETECTOR_GATE_CANDIDATE_FOUND" if viable else "NO_USABLE_OPERATING_POINT",
    }
    (out_dir / "gate_sweep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "viable_gates": viable[:5]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
