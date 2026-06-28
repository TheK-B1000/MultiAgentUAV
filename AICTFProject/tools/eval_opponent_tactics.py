#!/usr/bin/env python3
"""Compare tactical behavior of scripted opponents OP5 through OP12.

Runs short matched-seed rollouts with a passive blue team (hold position) so
red scripted-BT telemetry reflects opponent tactics rather than PPO skill.

Usage::

    python tools/eval_opponent_tactics.py --seeds 0 1 2 --steps 200
    python tools/eval_opponent_tactics.py --opponents OP5 OP8 OP12 --out reports/op_tactics.md
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CURRICULUM = ("OP5", "OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12")

ROLE_NAMES = {
    0: "ATTACKER",
    1: "DEFENDER",
    2: "ESCORT",
    3: "INTERCEPTOR",
    4: "FLAG_RETR",
    5: "COUNTER",
    6: "2V1_WING",
}


def _passive_blue_actions(core) -> np.ndarray:
    """Macro/target that keeps blue agents near spawn (minimal interference)."""
    B, Nb = core.B, core.Nb
    actions = np.zeros((B, Nb, 2), dtype=np.int64)
    return actions.reshape(B * Nb * 2)


def _rollout_opponent(
    opponent: str,
    *,
    seed: int,
    steps: int,
    n_agents: int = 2,
) -> Dict[str, float]:
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # type: ignore[import]

    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=n_agents,
        max_red_agents=n_agents,
        map_layout="map_b",
        max_decision_steps=max(steps + 50, 400),
        aquaticus_profile=True,
        rules_profile="OURS",
        device="cpu",
        seed=seed,
    )
    env = GPUCTFVecEnv(cfg)
    env.seed(seed)
    env.reset()
    core = env.core
    core.set_next_opponent("SCRIPTED", opponent, env_indices=[0])

    role_hist: Dict[str, int] = defaultdict(int)
    route_diversity: List[float] = []
    prev_tx = None
    prev_ty = None

    tel_escort = 0
    tel_intercept = 0
    tel_counter = 0
    tel_obj_changes = 0
    tel_stuck = 0

    blue_wins = 0
    red_wins = 0
    red_score_sum = 0.0
    blue_score_sum = 0.0

    for _ in range(steps):
        actions = _passive_blue_actions(core)
        env.step_async(actions)
        obs, rew, done, infos = env.step_wait()

        roles = core.bt_red_role[0].detach().cpu().tolist()
        for r in roles:
            role_hist[ROLE_NAMES.get(int(r), str(r))] += 1

        tx = core._debug_red_target_x[0].detach().cpu().numpy()
        ty = core._debug_red_target_y[0].detach().cpu().numpy()
        if prev_tx is not None:
            route_diversity.append(float(np.mean(np.abs(tx - prev_tx) + np.abs(ty - prev_ty))))
        prev_tx, prev_ty = tx, ty

        tel_escort = int(core.bt_tel_escort_attempts[0].item())
        tel_intercept = int(core.bt_tel_intercept_attempts[0].item())
        tel_counter = int(core.bt_tel_counter_captures[0].item())
        tel_obj_changes = int(core.bt_tel_objective_changes[0].item())
        tel_stuck = int(core.bt_tel_stuck_steps[0].item())

        red_score_sum += float(core.red_score[0].item())
        blue_score_sum += float(core.blue_score[0].item())

        if done[0]:
            info = infos[0] if isinstance(infos[0], dict) else {}
            ep = info.get("episode_result") or info.get("episode") or {}
            winner = str(ep.get("winner", "")).lower()
            if winner == "red":
                red_wins += 1
            elif winner == "blue":
                blue_wins += 1
            env.reset()
            core.set_next_opponent("SCRIPTED", opponent, env_indices=[0])

    total_roles = max(1, sum(role_hist.values()))
    role_pct = {k: 100.0 * v / total_roles for k, v in role_hist.items()}

    return {
        "opponent": opponent,
        "seed": float(seed),
        "steps": float(steps),
        "escort_attempts": float(tel_escort),
        "intercept_attempts": float(tel_intercept),
        "counter_decisions": float(tel_counter),
        "objective_changes": float(tel_obj_changes),
        "stuck_steps": float(tel_stuck),
        "route_diversity_mean": float(statistics.mean(route_diversity)) if route_diversity else 0.0,
        "red_wins": float(red_wins),
        "blue_wins": float(blue_wins),
        "avg_red_score": red_score_sum / steps,
        "avg_blue_score": blue_score_sum / steps,
        **{f"role_pct_{k}": v for k, v in role_pct.items()},
    }


def _contested_scenario_eval(opponent: str) -> Dict[str, float]:
    """Single-step BT decisions under a contested micro-scenario."""
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # type: ignore[import]

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2,
        map_layout="map_b", max_decision_steps=400,
        aquaticus_profile=True, rules_profile="OURS", device="cpu", seed=0,
    )
    env = GPUCTFVecEnv(cfg)
    env.reset()
    core = env.core
    core.set_next_opponent("SCRIPTED", opponent, env_indices=[0])
    core.bt_role_lock_ticks[0] = 0

    # Dual-pressure: own carrier + enemy carrier.
    core.red_carrying[0, 0] = True
    core.red_x[0, 0] = 14.0
    core.red_y[0, 0] = 10.0
    core.blue_x[0, 0] = 13.0
    core.blue_y[0, 0] = 10.0
    core.blue_carrying[0, 1] = True
    core.blue_x[0, 1] = 6.0
    core.blue_y[0, 1] = 10.0
    core.red_x[0, 1] = 8.0
    core.red_y[0, 1] = 10.0

    core._get_bt_targets()
    roles = core.bt_red_role[0].tolist()
    role_set = {ROLE_NAMES.get(int(r), str(r)) for r in roles}
    return {
        "opponent": opponent,
        "has_escort": float("ESCORT" in role_set),
        "has_intercept": float("INTERCEPTOR" in role_set),
        "has_counter": float("COUNTER" in role_set),
        "has_defender": float("DEFENDER" in role_set),
        "unique_roles": float(len(role_set)),
    }


def _format_contested(agg: Dict[str, Dict[str, float]]) -> str:
    lines = [
        "# Contested micro-scenario (dual carrier pressure)",
        "",
        "| Opponent | Escort | Intercept | Counter | Defender | Unique roles |",
        "|----------|--------|-----------|---------|----------|--------------|",
    ]
    for opp in CURRICULUM:
        if opp not in agg:
            continue
        r = agg[opp]
        lines.append(
            f"| {opp} | {r.get('has_escort', 0):.0f} | {r.get('has_intercept', 0):.0f} "
            f"| {r.get('has_counter', 0):.0f} | {r.get('has_defender', 0):.0f} "
            f"| {r.get('unique_roles', 0):.0f} |"
        )
    return "\n".join(lines)


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    by_opp: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_opp[row["opponent"]].append(row)

    out: Dict[str, Dict[str, float]] = {}
    for opp, group in by_opp.items():
        keys = [k for k in group[0] if k not in ("opponent", "seed")]
        out[opp] = {}
        for k in keys:
            vals = [g[k] for g in group]
            out[opp][k] = float(statistics.mean(vals))
    return out


def _format_markdown(agg: Dict[str, Dict[str, float]]) -> str:
    lines = [
        "# OP5..OP12 tactical evaluation",
        "",
        "| Opponent | Escort | Intercept | Counter | ObjChg | Stuck | Route div | ATT% | DEF% | ESC% | INT% | CTR% |",
        "|----------|--------|-----------|---------|------|-------|-----------|------|------|------|------|------|",
    ]
    for opp in CURRICULUM:
        if opp not in agg:
            continue
        r = agg[opp]
        lines.append(
            f"| {opp} "
            f"| {r.get('escort_attempts', 0):.1f} "
            f"| {r.get('intercept_attempts', 0):.1f} "
            f"| {r.get('counter_decisions', 0):.1f} "
            f"| {r.get('objective_changes', 0):.1f} "
            f"| {r.get('stuck_steps', 0):.1f} "
            f"| {r.get('route_diversity_mean', 0):.2f} "
            f"| {r.get('role_pct_ATTACKER', 0):.0f} "
            f"| {r.get('role_pct_DEFENDER', 0):.0f} "
            f"| {r.get('role_pct_ESCORT', 0):.0f} "
            f"| {r.get('role_pct_INTERCEPTOR', 0):.0f} "
            f"| {r.get('role_pct_COUNTER', 0):.0f} |"
        )
    lines.append("")
    lines.append(
        "_Role percentages are time-averaged agent-role occupancy during passive-blue rollouts. "
        "Escort/intercept/counter columns are cumulative BT telemetry counters._"
    )
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponents", nargs="*", default=list(CURRICULUM))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--mode", choices=("rollout", "contested"), default="rollout")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    rows: List[Dict[str, float]] = []
    if args.mode == "contested":
        for opp in args.opponents:
            print(f"Contested scenario {opp}...", flush=True)
            rows.append(_contested_scenario_eval(opp))
        agg = _aggregate(rows)
        report = _format_contested(agg)
    else:
        for opp in args.opponents:
            for seed in args.seeds:
                print(f"Rolling {opp} seed={seed}...", flush=True)
                rows.append(
                    _rollout_opponent(opp, seed=seed, steps=args.steps, n_agents=args.agents)
                )
        agg = _aggregate(rows)
        report = _format_markdown(agg)
    print(report)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
