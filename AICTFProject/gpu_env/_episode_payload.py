from __future__ import annotations


def _build_episode_result_payload(info: dict) -> dict:
    """Build the terminal episode summary consumed by training callbacks."""
    bs = int(info["blue_score"])
    rs = int(info["red_score"])
    okind = str(info["opponent_kind"]).lower()
    okey = str(info["opponent_key"] or "")
    osnap = okey if okind == "snapshot" else ""
    return {
        "blue_score": bs,
        "red_score": rs,
        "success": 1 if bs > rs else 0,
        "opponent_kind": okind,
        "opponent_snapshot": osnap,
        "scripted_tag": okey if okind == "scripted" else "",
        "species_tag": "BALANCED",
        "collisions_per_episode": int(info["collision_events_per_episode"]),
        "collision_events_per_episode": int(info["collision_events_per_episode"]),
        "collision_free_episode": int(info["collision_free_episode"]),
        "near_misses_per_episode": int(info["near_misses_per_episode"]),
        "zone_coverage": float(info["zone_coverage"]),
        "time_to_first_score": info["time_to_first_score"],
        "mean_inter_robot_dist": info["mean_inter_robot_dist"],
        "reward_terminal": float(info["reward_terminal"]),
        "reward_offense": float(info["reward_offense"]),
        "reward_pbrs": float(info["reward_pbrs"]),
        "reward_team": float(info["reward_team"]),
        "reward_sparse": float(info["reward_sparse"]),
        "reward_failure": float(info["reward_failure"]),
        "reward_sparse_points": float(info["reward_sparse_points"]),
        "reward_total": float(info["reward_total"]),
        "decision_steps": int(info["decision_steps"]),
        "vec_schema_version": 1,
    }


__all__ = ["_build_episode_result_payload"]
