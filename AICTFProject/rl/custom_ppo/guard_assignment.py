"""Privileged GUARD_DISTRIBUTED_V2 assignment → z_i (ASSIGNMENT_INFORMATION_DIAGNOSTIC_SPEC).

Faithful lift of gpu_env/_core/_scripted_blue_styles.py::_blue_one_defender_v2_targets
matching loop (sequential greedy + ``taken``). No macros, actions, rewards, or poles.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

RESP_ATTACK = 0
RESP_DEFEND_THREAT = 1
RESP_DEFEND_HOLD = 2
HOME_ENTITY = -1
ENEMY_FLAG_ENTITY = -2


def _dist(ax, ay, bx, by):
    return torch.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + 1e-8)


def assign_guard_v2_responsibilities(
    *,
    own_x: torch.Tensor,
    own_y: torch.Tensor,
    own_alive: torch.Tensor,
    home_xy: torch.Tensor,
    enemy_x: torch.Tensor,
    enemy_y: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_tagged: torch.Tensor,
    enemy_flag_xy: torch.Tensor,
    on_our_side: torch.Tensor,
    defense_radius: float,
    midline_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute per-agent responsibility without reading teacher macros/actions."""
    if own_alive.dtype != torch.bool:
        own_alive = own_alive.bool()
    B, N = own_x.shape
    Ne = enemy_x.shape[1]
    device = own_x.device
    n_def = (N + 1) // 2
    lo = N - n_def

    home = home_xy.float()
    if home.dim() == 3:
        home = home[:, 0, :]
    home_x, home_y = home[:, 0], home[:, 1]
    eflag = enemy_flag_xy.float()
    if eflag.dim() == 3:
        eflag = eflag[:, 0, :]

    # Match teacher: all agents default to enemy flag; defenders overwritten below.
    target_x = eflag[:, 0:1].expand(B, N).clone()
    target_y = eflag[:, 1:2].expand(B, N).clone()
    responsibility = torch.full((B, N), RESP_ATTACK, dtype=torch.long, device=device)
    assigned_entity = torch.full((B, N), ENEMY_FLAG_ENTITY, dtype=torch.long, device=device)

    intruder = enemy_alive.bool() & (~enemy_tagged.bool()) & on_our_side.bool()
    d_home = _dist(enemy_x, enemy_y, home_x[:, None], home_y[:, None])
    big = torch.finfo(d_home.dtype).max
    d_masked = torch.where(intruder, d_home, torch.full_like(d_home, big))
    order = torch.argsort(d_masked, dim=1, stable=True)

    def_px = own_x[:, lo:]
    def_py = own_y[:, lo:]
    def_tx = home_x[:, None].expand(B, n_def).clone()
    def_ty = home_y[:, None].expand(B, n_def).clone()
    def_ent = torch.full((B, n_def), HOME_ENTITY, dtype=torch.long, device=device)
    taken = torch.zeros((B, n_def), dtype=torch.bool, device=device)

    n_slots = min(n_def, Ne)
    for k in range(n_slots):
        tidx = order[:, k]
        valid = d_masked.gather(1, tidx[:, None]).squeeze(1) < big
        tx = enemy_x.gather(1, tidx[:, None]).squeeze(1)
        ty = enemy_y.gather(1, tidx[:, None]).squeeze(1)
        dd = _dist(def_px, def_py, tx[:, None], ty[:, None])
        dd = torch.where(taken, torch.full_like(dd, big), dd)
        pick = dd.argmin(dim=1)
        free = (~taken).any(dim=1)
        do = valid & free
        sel = torch.zeros_like(taken)
        sel.scatter_(1, pick[:, None], do[:, None])
        def_tx = torch.where(sel, tx[:, None].expand_as(def_tx), def_tx)
        def_ty = torch.where(sel, ty[:, None].expand_as(def_ty), def_ty)
        ent_exp = tidx[:, None].expand_as(def_ent)
        def_ent = torch.where(sel, ent_exp, def_ent)
        taken = taken | sel

    # Teacher: never pursue past midline → hold home.
    if midline_fn is not None:
        in_our_half = midline_fn(def_tx)
        def_tx = torch.where(in_our_half, def_tx, home_x[:, None].expand_as(def_tx))
        def_ty = torch.where(in_our_half, def_ty, home_y[:, None].expand_as(def_ty))
        def_ent = torch.where(in_our_half, def_ent, torch.full_like(def_ent, HOME_ENTITY))

    radius = float(defense_radius)
    dx = def_tx - home_x[:, None]
    dy = def_ty - home_y[:, None]
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    scale = torch.clamp(radius / dist, max=1.0)
    def_tx = home_x[:, None] + dx * scale
    def_ty = home_y[:, None] + dy * scale

    target_x[:, lo:] = def_tx
    target_y[:, lo:] = def_ty
    assigned_entity[:, lo:] = def_ent
    responsibility[:, lo:] = torch.where(
        def_ent >= 0,
        torch.full_like(def_ent, RESP_DEFEND_THREAT),
        torch.full_like(def_ent, RESP_DEFEND_HOLD),
    )

    delta_x = target_x - own_x
    delta_y = target_y - own_y
    return {
        "responsibility": responsibility,
        "assigned_entity": assigned_entity,
        "delta_x": delta_x.float(),
        "delta_y": delta_y.float(),
        "target_x": target_x.float(),
        "target_y": target_y.float(),
    }


def zi_discrete_key(responsibility: torch.Tensor, assigned_entity: torch.Tensor) -> torch.Tensor:
    return responsibility.long() * 10_000 + (assigned_entity.long() + 2)


def assert_zi_builder_no_forbidden_kwargs(**kwargs) -> None:
    forbidden = (
        "macro", "action", "btx", "bty", "waypoint", "reward", "return",
        "pole", "getflag", "opponent", "genome",
    )
    bad = [k for k in kwargs if any(f in k.lower() for f in forbidden)]
    if bad:
        raise ValueError(f"z_i builder must not accept forbidden channels: {bad}")


def encode_assignment_features(
    responsibility: torch.Tensor,
    assigned_entity: torch.Tensor,
    delta_x: torch.Tensor,
    delta_y: torch.Tensor,
    defense_radius: float,
    n_enemies: int,
) -> torch.Tensor:
    """Locked ASSIGNMENT_CONDITIONING_V1 actor encoding: (B, N, 4) float32."""
    R = max(float(defense_radius), 1e-8)
    Ne = int(n_enemies)
    resp_f = responsibility.float() / 2.0
    entity_code = (assigned_entity.float() + 2.0) / float(Ne + 2)
    tanh_dx = torch.tanh(delta_x.float() / R)
    tanh_dy = torch.tanh(delta_y.float() / R)
    return torch.stack([resp_f, entity_code, tanh_dx, tanh_dy], dim=-1)


def _assignment_geometry_from_core(core) -> Dict[str, torch.Tensor]:
    """Geometry-only tensors for assign_guard_v2_responsibilities (matches diagnostic)."""
    from gpu_env._core._scripted_blue_styles import gate2b_defender_hold_radius

    home = core.blue_flag_home
    if home.dim() == 3:
        home = home[:, 0, :]
    eflag = core.red_flag_pos
    if eflag.dim() == 3:
        eflag = eflag[:, 0, :]
    on_our = core._is_on_home_side("blue", core.red_x)
    return assign_guard_v2_responsibilities(
        own_x=core.blue_x,
        own_y=core.blue_y,
        own_alive=core.blue_alive.bool(),
        home_xy=home,
        enemy_x=core.red_x,
        enemy_y=core.red_y,
        enemy_alive=core.red_alive.bool(),
        enemy_tagged=core.red_tagged.bool(),
        enemy_flag_xy=eflag,
        on_our_side=on_our,
        defense_radius=float(gate2b_defender_hold_radius(core.cfg)),
        midline_fn=lambda tx: core._is_on_home_side("blue", tx),
    )


def _living_set_changed(prev_alive: Optional[torch.Tensor], alive: torch.Tensor) -> torch.Tensor:
    alive_b = alive.bool()
    if prev_alive is None:
        return torch.ones(alive_b.shape[0], dtype=torch.bool, device=alive_b.device)
    if prev_alive.shape != alive_b.shape:
        raise ValueError("prev_alive / alive shape mismatch")
    return (prev_alive.bool() != alive_b).any(dim=1)


class AssignmentHoldState:
    """Persistent assignment features with H_a hold and immediate death reassignment."""

    def __init__(
        self,
        n_envs: int,
        n_agents: int,
        *,
        hold_ticks: int = 8,
        n_enemies: int = 4,
        device: str | torch.device = "cpu",
    ) -> None:
        self.n_envs = int(n_envs)
        self.n_agents = int(n_agents)
        self.n_enemies = int(n_enemies)
        self.hold_ticks = int(hold_ticks)
        if self.hold_ticks < 1:
            raise ValueError(f"hold_ticks must be >= 1, got {self.hold_ticks}")
        self.device = torch.device(device)
        self.features = torch.zeros(
            (self.n_envs, self.n_agents, 4), dtype=torch.float32, device=self.device
        )
        self.discrete_key = torch.zeros(
            (self.n_envs, self.n_agents), dtype=torch.long, device=self.device
        )
        self.age = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self.prev_alive: Optional[torch.Tensor] = None
        self._defense_radius: float = 6.0

    def reset_envs(self, env_mask: torch.Tensor) -> None:
        m = env_mask.bool().to(self.device)
        self.age = self.age.masked_fill(m, self.hold_ticks)
        if self.prev_alive is not None:
            self.prev_alive = self.prev_alive.clone()
            self.prev_alive[m] = False

    def update_from_zi(
        self,
        zi: Dict[str, torch.Tensor],
        alive: torch.Tensor,
        defense_radius: float,
        *,
        force: bool = False,
        advance_age: bool = True,
    ) -> torch.Tensor:
        alive = alive.to(self.device).bool()
        if alive.shape != (self.n_envs, self.n_agents):
            raise ValueError(
                f"expected alive ({self.n_envs}, {self.n_agents}), got {tuple(alive.shape)}"
            )
        self._defense_radius = float(defense_radius)
        new_key = zi_discrete_key(zi["responsibility"], zi["assigned_entity"]).to(self.device)
        new_feat = encode_assignment_features(
            zi["responsibility"],
            zi["assigned_entity"],
            zi["delta_x"],
            zi["delta_y"],
            defense_radius,
            self.n_enemies,
        ).to(self.device)

        changed = _living_set_changed(self.prev_alive, alive)
        expired = self.age >= self.hold_ticks
        if force:
            need = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        else:
            need = changed | expired

        if bool(need.any().item()):
            self.features = torch.where(need.view(-1, 1, 1), new_feat, self.features)
            self.discrete_key = torch.where(need.unsqueeze(1), new_key, self.discrete_key)
            self.age = torch.where(need, torch.zeros_like(self.age), self.age)

        if advance_age:
            self.age = self.age + 1
        self.prev_alive = alive.clone()
        return self.features.clone()


def assignment_from_core(
    core,
    hold: AssignmentHoldState,
    *,
    force: bool = False,
    advance_age: bool = True,
) -> torch.Tensor:
    """Compute / update (B, N, 4) assignment features from a BatchedCTFCore (blue)."""
    from gpu_env._core._scripted_blue_styles import gate2b_defender_hold_radius

    zi = _assignment_geometry_from_core(core)
    radius = float(gate2b_defender_hold_radius(core.cfg))
    return hold.update_from_zi(
        zi,
        core.blue_alive.bool(),
        radius,
        force=force,
        advance_age=advance_age,
    )
