from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def agent_uid(agent: Any) -> str:
    uid = getattr(agent, "unique_id", None) or getattr(agent, "slot_id", None)
    if uid is None or str(uid).strip() == "":
        side = getattr(agent, "side", "blue")
        aid = getattr(agent, "agent_id", 0)
        uid = f"{side}_{aid}"
    return str(uid)


def simulate_decision_window(env: Any, gm: Any, window_s: float, sim_dt: float) -> None:
    if window_s <= 0.0:
        return
    if sim_dt <= 0.0:
        raise ValueError("sim_dt must be > 0")

    n_full = int(window_s // sim_dt)
    rem = float(window_s - n_full * sim_dt)

    for _ in range(n_full):
        if getattr(gm, "game_over", False):
            return
        env.update(sim_dt)

    if rem > 1e-9 and (not getattr(gm, "game_over", False)):
        env.update(rem)


def collect_team_uids(agents: Sequence[Any]) -> List[str]:
    return [agent_uid(a) for a in agents if a is not None]


def pop_reward_events_best_effort(gm: Any) -> List[Tuple[float, str, float]]:
    fn = getattr(gm, "pop_reward_events", None)
    if fn is None or (not callable(fn)):
        return []
    try:
        return list(fn())
    except Exception:
        return []


def team_reward_from_events(gm: Any, allowed_uids: Iterable[str]) -> float:
    allowed = set(str(x) for x in allowed_uids)
    total = 0.0
    for _t, aid, r in pop_reward_events_best_effort(gm):
        if aid is None:
            continue
        if str(aid) in allowed:
            total += float(r)
    return float(total)


def batch_by_agent_id(agents: Sequence[Any]) -> List[Any]:
    out = list(agents)
    try:
        out.sort(key=lambda a: int(getattr(a, "agent_id", 0)))
    except Exception:
        return out
    return out
