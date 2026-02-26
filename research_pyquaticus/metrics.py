from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def extract_scores_from_info(info: Dict[str, Any]) -> Dict[str, Optional[float]]:
    # Pyquaticus info usually includes a dict-like global_state.
    gs = info.get("global_state", None)
    blue_score = None
    red_score = None
    if isinstance(gs, dict):
        blue_score = gs.get("blue_team_score", None)
        red_score = gs.get("red_team_score", None)
    return {"blue_score": blue_score, "red_score": red_score}


def episode_record(seed: int, episode_idx: int, steps: int, info: Dict[str, Any]) -> Dict[str, Any]:
    scores = extract_scores_from_info(info)
    bs = scores["blue_score"]
    rs = scores["red_score"]
    win = None
    if bs is not None and rs is not None:
        win = 1 if bs > rs else 0
    return {
        "seed": int(seed),
        "episode": int(episode_idx),
        "steps": int(steps),
        "blue_score": bs,
        "red_score": rs,
        "blue_win": win,
    }


def aggregate_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    if not rows:
        return {
            "episodes": 0,
            "win_rate": 0.0,
            "avg_blue_score": 0.0,
            "avg_red_score": 0.0,
            "avg_episode_len": 0.0,
        }
    wins = [r["blue_win"] for r in rows if r.get("blue_win") is not None]
    blue_scores = [r["blue_score"] for r in rows if r.get("blue_score") is not None]
    red_scores = [r["red_score"] for r in rows if r.get("red_score") is not None]
    lengths = [r["steps"] for r in rows]
    return {
        "episodes": len(rows),
        "win_rate": mean(wins) if wins else 0.0,
        "avg_blue_score": mean(blue_scores) if blue_scores else 0.0,
        "avg_red_score": mean(red_scores) if red_scores else 0.0,
        "avg_episode_len": mean(lengths) if lengths else 0.0,
    }


def write_episode_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "episode", "steps", "blue_score", "red_score", "blue_win"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)


def write_summary(path_json: Path, path_csv: Path, summary: Dict[str, Any]) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    with path_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with path_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
