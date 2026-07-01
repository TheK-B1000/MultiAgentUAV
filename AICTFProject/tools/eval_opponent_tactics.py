#!/usr/bin/env python3
"""Compare tactical behavior of scripted opponents OP5 through OP12.

Runs matched-seed rollouts with a passive blue team plus a contested-scenario
battery (forced carrier / intercept / counter geometries).

Usage::

    python tools/eval_opponent_tactics.py --seeds 0 1 2 --steps 200
    python tools/eval_opponent_tactics.py --mode both --out reports/op_tactics.md
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List

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


def _setup_intercept_feasible(c) -> None:
    c.blue_carrying[0, 0] = True
    c.blue_x[0, 0] = 10.0
    c.blue_y[0, 0] = 10.0
    c.red_x[0, 0] = 8.0
    c.red_y[0, 0] = 9.0
    c.red_x[0, 1] = 8.0
    c.red_y[0, 1] = 11.0


def _setup_counter_infeasible(c) -> None:
    c.blue_carrying[0, 0] = True
    c.blue_x[0, 0] = 1.0
    c.blue_y[0, 0] = 10.0
    c.red_x[0, 0] = 18.0
    c.red_y[0, 0] = 5.0
    c.red_x[0, 1] = 18.0
    c.red_y[0, 1] = 15.0


def _setup_escort_pursued(c) -> None:
    c.red_carrying[0, 0] = True
    c.red_x[0, 0] = 14.0
    c.red_y[0, 0] = 10.0
    c.blue_x[0, 0] = 13.0
    c.blue_y[0, 0] = 10.0


def _setup_dual_carrier(c) -> None:
    c.red_carrying[0, 0] = True
    c.red_x[0, 0] = 14.0
    c.red_y[0, 0] = 10.0
    c.blue_carrying[0, 0] = True
    c.blue_x[0, 0] = 6.0
    c.blue_y[0, 0] = 10.0
    c.red_x[0, 1] = 8.0
    c.red_y[0, 1] = 10.0
    c.blue_x[0, 1] = 1.0
    c.blue_y[0, 1] = 1.0


def _setup_op9_mine_intent(c) -> None:
    from gpu_env._core._bt_profiles import profile_for_level

    prof = profile_for_level(9)
    c.sim_step_count[0] = int(prof.mine_approach_lead_steps - 1)
    c.red_mine_charges[0, 1] = 1
    c.red_x[0, 0] = 4.0
    c.red_y[0, 0] = 10.0
    c.red_x[0, 1] = 18.0
    c.red_y[0, 1] = 10.0
    c.blue_x[0, 0] = 12.0
    c.blue_y[0, 0] = 10.0


_CONTESTED_SCENARIOS: Dict[str, Callable[[object], None]] = {
    "intercept_feasible": _setup_intercept_feasible,
    "counter_infeasible": _setup_counter_infeasible,
    "escort_pursued": _setup_escort_pursued,
    "dual_carrier": _setup_dual_carrier,
    "op9_mine_intent": _setup_op9_mine_intent,
}


def _contested_scenario_eval(opponent: str) -> Dict[str, float]:
    """Run the full contested scenario battery for one opponent."""
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # type: ignore[import]

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2,
        map_layout="map_b", max_decision_steps=400,
        aquaticus_profile=True, rules_profile="OURS", device="cpu", seed=0,
    )
    env = GPUCTFVecEnv(cfg)
    try:
        core = env.core
        hits = {name: 0.0 for name in _CONTESTED_SCENARIOS}
        for name, setup in _CONTESTED_SCENARIOS.items():
            env.reset()
            core.set_next_opponent("SCRIPTED", opponent, env_indices=[0])
            core.bt_role_lock_ticks[0] = 0
            core.bt_mine_lock_ticks.zero_()
            core.bt_want_mine.zero_()
            setup(core)
            core._assign_scripted_targets_by_role("red")
            roles = {ROLE_NAMES.get(int(r), str(r)) for r in core.bt_red_role[0].tolist()}
            if name == "intercept_feasible":
                hits[name] = float("INTERCEPTOR" in roles)
            elif name == "counter_infeasible":
                hits[name] = float("COUNTER" in roles and "INTERCEPTOR" not in roles)
            elif name == "escort_pursued":
                hits[name] = float("ESCORT" in roles)
            elif name == "dual_carrier":
                hits[name] = float("ESCORT" in roles or "INTERCEPTOR" in roles)
            elif name == "op9_mine_intent":
                hits[name] = float(bool(core.bt_want_mine[0].any().item()))
        return {"opponent": opponent, **hits}
    finally:
        env.close()


def _format_contested_battery(agg: Dict[str, Dict[str, float]]) -> str:
    lines = [
        "# Contested scenario battery (forced geometries)",
        "",
        "| Opponent | Intercept | Counter | Escort | Dual-carrier | OP9 mine |",
        "|----------|-----------|---------|--------|--------------|----------|",
    ]
    for opp in CURRICULUM:
        if opp not in agg:
            continue
        r = agg[opp]
        lines.append(
            f"| {opp} "
            f"| {r.get('intercept_feasible', 0):.0f} "
            f"| {r.get('counter_infeasible', 0):.0f} "
            f"| {r.get('escort_pursued', 0):.0f} "
            f"| {r.get('dual_carrier', 0):.0f} "
            f"| {r.get('op9_mine_intent', 0):.0f} |"
        )
    lines.append("")
    lines.append(
        "_Each cell is 1 if the scenario fired the expected role/signal for that opponent, else 0. "
        "``counter_infeasible`` expects COUNTER without INTERCEPTOR (OP12-style geometry). "
        "``op9_mine_intent`` is only meaningful for OP9 (others may read 0)._"
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
    parser.add_argument("--mode", choices=("rollout", "contested", "both"), default="both")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report_parts: List[str] = []
    if args.mode in ("rollout", "both"):
        rollout_rows: List[Dict[str, float]] = []
        for opp in args.opponents:
            for seed in args.seeds:
                print(f"Rolling {opp} seed={seed}...", flush=True)
                rollout_rows.append(
                    _rollout_opponent(opp, seed=seed, steps=args.steps, n_agents=args.agents)
                )
        report_parts.append(_format_markdown(_aggregate(rollout_rows)))

    if args.mode in ("contested", "both"):
        contested_rows: List[Dict[str, float]] = []
        for opp in args.opponents:
            print(f"Contested battery {opp}...", flush=True)
            contested_rows.append(_contested_scenario_eval(opp))
        if args.mode == "both":
            report_parts.append("")
        report_parts.append(_format_contested_battery({r["opponent"]: r for r in contested_rows}))

    report = "\n".join(report_parts)
    print(report)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
