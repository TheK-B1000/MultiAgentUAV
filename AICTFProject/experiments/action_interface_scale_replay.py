"""Matched-context I0–I3 shadow replay for ACTION_INTERFACE_SCALE_DIAGNOSTIC."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from macro_actions import MacroAction

ARMS = ("I0_CONTINUOUS_ORACLE", "I1_W50_NO_COMMIT", "I2_W50_CURRENT_COMMIT", "I3_FULL_CURRENT_MACRO")
FORBIDDEN_COLUMNS = frozenset({
    "blue_score", "red_score", "win", "reward", "return", "value", "advantage",
})

NUM_TOL = 1e-5
G6_TOL = 1e-6


def nearest_w50_index(tx: float, ty: float, W: np.ndarray) -> int:
    d2 = (W[:, 0] - tx) ** 2 + (W[:, 1] - ty) ** 2
    return int(np.argmin(d2))


def _bearing(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.arctan2(y2 - y1, x2 - x1))


def _angle_diff(a: float, b: float) -> float:
    d = (a - b + np.pi) % (2.0 * np.pi) - np.pi
    return abs(float(d))


def velocity_xy(heading, speed, current: float = 0.0):
    h = np.asarray(heading, dtype=np.float64)
    s = np.asarray(speed, dtype=np.float64)
    vx = s * np.cos(h) + current
    vy = s * np.sin(h)
    return vx, vy


@dataclass
class TraceTick:
    tick: int
    oracle_tx: np.ndarray
    oracle_ty: np.ndarray
    blue_x: np.ndarray
    blue_y: np.ndarray
    blue_heading: np.ndarray
    blue_speed: np.ndarray
    blue_alive: np.ndarray
    blue_carrying: np.ndarray
    blue_tagged: np.ndarray
    red_x: np.ndarray
    red_y: np.ndarray
    blue_speed_cap_scale: np.ndarray
    rt_current: float


@dataclass
class SourceTrace:
    scale: int
    style: str
    seed: int
    ticks: list[TraceTick]
    trace_hash: str = ""
    init_blue_x: np.ndarray | None = None
    init_blue_y: np.ndarray | None = None
    init_blue_heading: np.ndarray | None = None
    init_blue_speed: np.ndarray | None = None

    def finalize_hash(self) -> None:
        payload = json.dumps([
            {
                "tick": t.tick,
                "oracle": np.stack([t.oracle_tx, t.oracle_ty], axis=1).round(6).tolist(),
                "blue_alive": t.blue_alive.astype(int).tolist(),
            }
            for t in self.ticks
        ], sort_keys=True, separators=(",", ":"))
        self.trace_hash = hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class ShadowState:
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    commit_macro: np.ndarray
    commit_target: np.ndarray
    commit_ticks_left: np.ndarray
    held_w50: np.ndarray
    go_to_ticks_left: np.ndarray


def _macro_horizon(core, macro: int) -> int:
    from gpu_env._core._scripted_red import macro_commit_ticks

    m = torch.tensor([[int(macro)]], device=core.device, dtype=torch.long)
    return int(macro_commit_ticks(
        m,
        go_to_ticks=int(core.cfg.macro_commit_go_to_ticks),
        grab_ticks=int(core.cfg.macro_commit_grab_ticks),
        get_flag_ticks=int(core.cfg.macro_commit_get_flag_ticks),
        place_ticks=int(core.cfg.macro_commit_place_ticks),
        go_home_ticks=int(core.cfg.macro_commit_go_home_ticks),
    )[0, 0].item())


def inject_exogenous(core, tick: TraceTick) -> None:
    n = tick.oracle_tx.shape[0]
    dev = core.device
    for i in range(n):
        core.blue_carrying[0, i] = bool(tick.blue_carrying[i])
        core.blue_tagged[0, i] = bool(tick.blue_tagged[i])
        core.blue_alive[0, i] = bool(tick.blue_alive[i])
        core.red_x[0, i] = float(tick.red_x[i])
        core.red_y[0, i] = float(tick.red_y[i])


def route_blue_targets(core, tx: np.ndarray, ty: np.ndarray,
                       sx: np.ndarray, sy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    btx = torch.tensor(tx, device=core.device, dtype=torch.float32).reshape(1, -1)
    bty = torch.tensor(ty, device=core.device, dtype=torch.float32).reshape(1, -1)
    bx = torch.tensor(sx, device=core.device, dtype=torch.float32).reshape(1, -1)
    by = torch.tensor(sy, device=core.device, dtype=torch.float32).reshape(1, -1)
    btx, bty, _, _ = core._redirect_tagged_to_home(btx, bty, core.red_x, core.red_y)
    btx, bty = core._route_targets_around_obstacles(bx, by, btx, bty)
    return (btx[0].detach().cpu().numpy(), bty[0].detach().cpu().numpy())


def integrate_blue_shadow(
    core,
    tick: TraceTick,
    shadow: ShadowState,
    target_x: np.ndarray,
    target_y: np.ndarray,
    prev_red_x: np.ndarray | None = None,
    prev_red_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = shadow.x.shape[0]
    dev = core.device
    core.blue_x[0, :n] = torch.tensor(shadow.x, device=dev, dtype=torch.float32)
    core.blue_y[0, :n] = torch.tensor(shadow.y, device=dev, dtype=torch.float32)
    core.blue_heading[0, :n] = torch.tensor(shadow.heading, device=dev, dtype=torch.float32)
    core.blue_speed[0, :n] = torch.tensor(shadow.speed, device=dev, dtype=torch.float32)
    for i in range(n):
        core.blue_alive[0, i] = bool(tick.blue_alive[i])
        core.blue_carrying[0, i] = bool(tick.blue_carrying[i])
        core.blue_tagged[0, i] = bool(tick.blue_tagged[i])
    prx = prev_red_x if prev_red_x is not None else tick.red_x
    pry = prev_red_y if prev_red_y is not None else tick.red_y
    core.red_x[0, :n] = torch.tensor(prx, device=dev, dtype=torch.float32)
    core.red_y[0, :n] = torch.tensor(pry, device=dev, dtype=torch.float32)
    snapshot = {
        "prev_blue_x": core.blue_x.clone(),
        "prev_blue_y": core.blue_y.clone(),
        "prev_red_x": core.red_x.clone(),
        "prev_red_y": core.red_y.clone(),
    }
    btx = torch.tensor(target_x, device=dev, dtype=torch.float32).reshape(1, -1)
    bty = torch.tensor(target_y, device=dev, dtype=torch.float32).reshape(1, -1)
    rtx = torch.tensor(tick.red_x, device=dev, dtype=torch.float32).reshape(1, -1)
    rty = torch.tensor(tick.red_y, device=dev, dtype=torch.float32).reshape(1, -1)
    targets = {"btx": btx, "bty": bty, "rtx": rtx, "rty": rty, "red_macro": None, "red_control_mask": None}
    core._advance_dynamics_phase(targets, snapshot)
    core.red_x[0, :n] = torch.tensor(tick.red_x, device=dev, dtype=torch.float32)
    core.red_y[0, :n] = torch.tensor(tick.red_y, device=dev, dtype=torch.float32)
    return (
        core.blue_x[0, :n].detach().cpu().numpy().copy(),
        core.blue_y[0, :n].detach().cpu().numpy().copy(),
        core.blue_heading[0, :n].detach().cpu().numpy().copy(),
        core.blue_speed[0, :n].detach().cpu().numpy().copy(),
    )


def init_shadow_from_trace(trace: SourceTrace) -> ShadowState:
    t0 = trace.ticks[0]
    n = t0.oracle_tx.shape[0]
    if trace.init_blue_x is not None:
        x, y = trace.init_blue_x.copy(), trace.init_blue_y.copy()
        h = trace.init_blue_heading.copy()
        s = trace.init_blue_speed.copy()
    else:
        x, y, h, s = t0.blue_x.copy(), t0.blue_y.copy(), t0.blue_heading.copy(), t0.blue_speed.copy()
    return ShadowState(
        x=x,
        y=y,
        heading=h,
        speed=s,
        commit_macro=np.zeros(n, dtype=np.int32),
        commit_target=np.zeros(n, dtype=np.int32),
        commit_ticks_left=np.zeros(n, dtype=np.int32),
        held_w50=np.zeros(n, dtype=np.int32),
        go_to_ticks_left=np.zeros(n, dtype=np.int32),
    )


def effective_targets_for_arm(
    arm: str,
    core,
    tick: TraceTick,
    shadow: ShadowState,
    W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from experiments.teacher_action_adapter import adapt, executed_target

    n = tick.oracle_tx.shape[0]
    meta: dict[str, Any] = {}
    inject_exogenous(core, tick)
    ox, oy = tick.oracle_tx.copy(), tick.oracle_ty.copy()
    eff_x = np.zeros(n)
    eff_y = np.zeros(n)

    if arm == "I0_CONTINUOUS_ORACLE":
        eff_x, eff_y = route_blue_targets(core, ox, oy, shadow.x, shadow.y)
        return eff_x, eff_y, meta

    if arm == "I1_W50_NO_COMMIT":
        for i in range(n):
            j = nearest_w50_index(float(ox[i]), float(oy[i]), W)
            eff_x[i], eff_y[i] = float(W[j, 0]), float(W[j, 1])
        eff_x, eff_y = route_blue_targets(core, eff_x, eff_y, shadow.x, shadow.y)
        meta["w50_idx"] = [nearest_w50_index(float(ox[i]), float(oy[i]), W) for i in range(n)]
        return eff_x, eff_y, meta

    go_h = int(core.cfg.macro_commit_go_to_ticks)

    if arm == "I2_W50_CURRENT_COMMIT":
        for i in range(n):
            at_boundary = shadow.go_to_ticks_left[i] <= 0
            meta.setdefault("at_boundary", []).append(bool(at_boundary))
            if at_boundary:
                j = nearest_w50_index(float(ox[i]), float(oy[i]), W)
                shadow.held_w50[i] = j
                shadow.go_to_ticks_left[i] = go_h
            eff_x[i], eff_y[i] = float(W[shadow.held_w50[i], 0]), float(W[shadow.held_w50[i], 1])
        eff_x, eff_y = route_blue_targets(core, eff_x, eff_y, shadow.x, shadow.y)
        return eff_x, eff_y, meta

    if arm == "I3_FULL_CURRENT_MACRO":
        action = np.zeros((n, 2), dtype=np.int64)
        for i in range(n):
            at_boundary = shadow.commit_ticks_left[i] <= 0
            if at_boundary:
                core.blue_x[0, i] = float(shadow.x[i])
                core.blue_y[0, i] = float(shadow.y[i])
                a = adapt(core, float(ox[i]), float(oy[i]), i, W)
                if a.uses_waypoint:
                    action[i] = (a.macro, a.target_idx)
                elif a.macro is not None:
                    action[i] = (int(a.macro), 0)
                else:
                    action[i] = (0, 0)
                shadow.commit_macro[i] = action[i, 0]
                shadow.commit_target[i] = action[i, 1]
                shadow.commit_ticks_left[i] = _macro_horizon(core, int(action[i, 0]))
            else:
                action[i, 0] = shadow.commit_macro[i]
                action[i, 1] = shadow.commit_target[i]
            ex, ey = executed_target(
                core, int(shadow.commit_macro[i]), int(shadow.commit_target[i]), i,
            )
            eff_x[i], eff_y[i] = ex, ey
        eff_x, eff_y = route_blue_targets(core, eff_x, eff_y, shadow.x, shadow.y)
        meta["commit_macro"] = shadow.commit_macro.copy()
        return eff_x, eff_y, meta

    raise ValueError(f"unknown arm {arm!r}")


def post_tick_arm_update(arm: str, shadow: ShadowState, tick: TraceTick) -> None:
    n = shadow.x.shape[0]
    if arm == "I2_W50_CURRENT_COMMIT":
        for i in range(n):
            if not tick.blue_alive[i] or tick.blue_tagged[i]:
                shadow.go_to_ticks_left[i] = 0
                continue
            shadow.go_to_ticks_left[i] = max(0, int(shadow.go_to_ticks_left[i]) - 1)
        return
    if arm == "I3_FULL_CURRENT_MACRO":
        for i in range(n):
            if not tick.blue_alive[i] or tick.blue_tagged[i]:
                shadow.commit_ticks_left[i] = 0
                continue
            shadow.commit_ticks_left[i] = max(0, int(shadow.commit_ticks_left[i]) - 1)


def replay_trace_arm(
    trace: SourceTrace,
    arm: str,
    core,
    W: np.ndarray,
) -> dict[str, Any]:
    shadow = init_shadow_from_trace(trace)
    i0_vx: list[np.ndarray] = []
    i0_vy: list[np.ndarray] = []
    arm_vx: list[np.ndarray] = []
    arm_vy: list[np.ndarray] = []
    paths_i0: list[np.ndarray] = []
    paths_arm: list[np.ndarray] = []

    dt = float(core.dt)
    for t_idx, tick in enumerate(trace.ticks):
        cur = float(tick.rt_current)
        if t_idx == 0:
            ivx = tick.blue_speed * np.cos(tick.blue_heading) + cur
            ivy = tick.blue_speed * np.sin(tick.blue_heading) + cur
        else:
            prev = trace.ticks[t_idx - 1]
            ivx = (tick.blue_x - prev.blue_x) / max(dt, 1e-6)
            ivy = (tick.blue_y - prev.blue_y) / max(dt, 1e-6)
        i0_vx.append(np.asarray(ivx, dtype=np.float64).copy())
        i0_vy.append(np.asarray(ivy, dtype=np.float64).copy())
        paths_i0.append(np.stack([tick.blue_x, tick.blue_y], axis=1))

        inject_exogenous(core, tick)
        eff_x, eff_y, _ = effective_targets_for_arm(arm, core, tick, shadow, W)
        prx = trace.ticks[t_idx - 1].red_x if t_idx > 0 else tick.red_x
        pry = trace.ticks[t_idx - 1].red_y if t_idx > 0 else tick.red_y
        sx, sy, sh, ss = integrate_blue_shadow(
            core, tick, shadow, eff_x, eff_y, prev_red_x=prx, prev_red_y=pry,
        )
        shadow.x, shadow.y, shadow.heading, shadow.speed = sx, sy, sh, ss
        post_tick_arm_update(arm, shadow, tick)
        avx, avy = velocity_xy(shadow.heading, shadow.speed, cur)
        arm_vx.append(avx.copy())
        arm_vy.append(avy.copy())
        paths_arm.append(np.stack([shadow.x, shadow.y], axis=1))

    max_speed = float(core.cfg.max_speed_cps)
    ev_per_agent: list[float] = []
    ex_rmse_per_agent: list[float] = []
    n = trace.ticks[0].oracle_tx.shape[0]
    for agent in range(n):
        dvs = []
        dxs = []
        for t_idx, tick in enumerate(trace.ticks):
            if not tick.blue_alive[agent]:
                continue
            dvx = arm_vx[t_idx][agent] - i0_vx[t_idx][agent]
            dvy = arm_vy[t_idx][agent] - i0_vy[t_idx][agent]
            dvs.append(float(np.hypot(dvx, dvy)) / max_speed)
            dxs.append(paths_arm[t_idx][agent] - paths_i0[t_idx][agent])
        ev_per_agent.append(float(np.mean(dvs)) if dvs else 0.0)
        if dxs:
            rmse = float(np.sqrt(np.mean([np.sum(d * d) for d in dxs])))
        else:
            rmse = 0.0
        ex_rmse_per_agent.append(rmse)

    return {
        "E_v": float(np.mean(ev_per_agent)),
        "E_x_rmse": float(np.mean(ex_rmse_per_agent)),
        "per_agent_E_v": ev_per_agent,
    }


def collect_source_trace(env, core, S, horizon: int) -> SourceTrace:
    """Outcome-blind native scripted collection (pre-step oracle + post-step kinematics)."""
    scale = int(core.blue_x.shape[1])
    orig_get = core._get_scripted_targets
    ticks: list[TraceTick] = []

    init_blue_x = core.blue_x[0].detach().cpu().numpy().copy()
    init_blue_y = core.blue_y[0].detach().cpu().numpy().copy()
    init_blue_heading = core.blue_heading[0].detach().cpu().numpy().copy()
    init_blue_speed = core.blue_speed[0].detach().cpu().numpy().copy()

    for tick in range(horizon):
        otx, oty = orig_get("blue")
        oracle_tx = otx[0].detach().cpu().numpy().copy()
        oracle_ty = oty[0].detach().cpu().numpy().copy()
        cap_scale = core.rt_blue_speed_scale[0].detach().cpu().numpy().copy()
        if cap_scale.ndim == 0:
            cap_scale = np.full(scale, float(cap_scale))
        elif cap_scale.size == 1:
            cap_scale = np.full(scale, float(cap_scale.reshape(-1)[0]))
        cur = float(core.rt_current_strength_cps[0].item())
        env.step_async(env.action_space.sample() * 0)
        env.step_wait()
        snap = {
            "blue_x": core.blue_x[0].detach().cpu().numpy().copy(),
            "blue_y": core.blue_y[0].detach().cpu().numpy().copy(),
            "blue_heading": core.blue_heading[0].detach().cpu().numpy().copy(),
            "blue_speed": core.blue_speed[0].detach().cpu().numpy().copy(),
            "blue_alive": core.blue_alive[0].detach().cpu().numpy().astype(bool),
            "blue_carrying": core.blue_carrying[0].detach().cpu().numpy().astype(bool),
            "blue_tagged": core.blue_tagged[0].detach().cpu().numpy().astype(bool),
            "red_x": core.red_x[0].detach().cpu().numpy().copy(),
            "red_y": core.red_y[0].detach().cpu().numpy().copy(),
        }
        ticks.append(TraceTick(
            tick=tick,
            oracle_tx=oracle_tx,
            oracle_ty=oracle_ty,
            rt_current=cur,
            blue_speed_cap_scale=cap_scale,
            **snap,
        ))

    tr = SourceTrace(
        scale=scale,
        style="",
        seed=0,
        ticks=ticks,
        init_blue_x=init_blue_x,
        init_blue_y=init_blue_y,
        init_blue_heading=init_blue_heading,
        init_blue_speed=init_blue_speed,
    )
    tr.finalize_hash()
    return tr


def i0_native_rmse(trace: SourceTrace, core, W: np.ndarray) -> tuple[float, float]:
    """RMSE of I0 shadow vs native post-step positions recorded in trace."""
    shadow = init_shadow_from_trace(trace)
    t0 = trace.ticks[0]
    shadow.x = t0.blue_x.copy()
    shadow.y = t0.blue_y.copy()
    shadow.heading = t0.blue_heading.copy()
    shadow.speed = t0.blue_speed.copy()
    pos_err = []
    vel_err = []
    max_speed = float(core.cfg.max_speed_cps)
    for t_idx, tick in enumerate(trace.ticks):
        inject_exogenous(core, tick)
        tx, ty, _ = effective_targets_for_arm("I0_CONTINUOUS_ORACLE", core, tick, shadow, W)
        prx = trace.ticks[t_idx - 1].red_x if t_idx > 0 else tick.red_x
        pry = trace.ticks[t_idx - 1].red_y if t_idx > 0 else tick.red_y
        nx, ny, nh, ns = integrate_blue_shadow(
            core, tick, shadow, tx, ty, prev_red_x=prx, prev_red_y=pry,
        )
        shadow.x, shadow.y, shadow.heading, shadow.speed = nx, ny, nh, ns
        d = np.hypot(nx - tick.blue_x, ny - tick.blue_y)
        pos_err.append(float(np.sqrt(np.mean(d * d))))
        cur = float(tick.rt_current)
        rvx = shadow.speed * np.cos(shadow.heading) + cur
        rvy = shadow.speed * np.sin(shadow.heading) + cur
        nvx = tick.blue_speed * np.cos(tick.blue_heading) + cur
        nvy = tick.blue_speed * np.sin(tick.blue_heading) + cur
        vel_err.append(float(np.mean(np.hypot(rvx - nvx, rvy - nvy) / max_speed)))
    return float(np.mean(pos_err)), float(np.mean(vel_err))


def assert_schema_clean(fieldnames: list[str]) -> None:
    bad = [f for f in fieldnames if f in FORBIDDEN_COLUMNS]
    if bad:
        raise ValueError(f"forbidden columns in schema: {bad}")
