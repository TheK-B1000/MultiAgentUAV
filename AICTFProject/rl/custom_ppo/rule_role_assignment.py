"""Rule-based geometric role assignment (RULE_BASED_ROLE_CONDITIONING_SPEC).

Math decides who has the job; RL learns how to perform it.

    k = N/2
    d_i = ||p_i - p_home||_2
    closest k living agents → DEFEND (r=0)
    remaining living agents → ATTACK (r=1)

No macros, rewards, pole identity, teachers, or GETFLAG state are consulted.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

ROLE_DEFEND = 0
ROLE_ATTACK = 1


def role_k_kwargs_from_cfg(cfg) -> dict:
    """Optional RoleHoldState k override. Empty dict keeps k=N/2."""
    raw = str(getattr(cfg, "role_k_defend_choices", "") or "").strip()
    k_fixed = int(getattr(cfg, "role_k_defend", 0) or 0)
    if raw and k_fixed:
        raise ValueError("role_k_defend and role_k_defend_choices are mutually exclusive")
    if raw:
        choices = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
        g = torch.Generator()
        g.manual_seed(int(getattr(cfg, "seed", 0) or 0))
        return {"k_choices": choices, "generator": g}
    if k_fixed:
        return {"k_defend": k_fixed}
    return {}


def role_k(n_agents: int) -> int:
    n = int(n_agents)
    if n < 1:
        raise ValueError(f"n_agents must be >= 1, got {n}")
    if n % 2 != 0:
        raise ValueError(
            f"RULE_BASED_ROLE_CONDITIONING locks k=N/2 for even N; got N={n}"
        )
    return n // 2


def distances_to_home(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    home_xy: torch.Tensor,
) -> torch.Tensor:
    """Euclidean distance of each agent to own base.

    pos_x, pos_y: (B, N)
    home_xy: (B, 2) or (B, 1, 2) continuous coords matching core.blue_flag_home
    returns d: (B, N)
    """
    if pos_x.shape != pos_y.shape or pos_x.dim() != 2:
        raise ValueError(f"pos_x/pos_y must be (B, N), got {tuple(pos_x.shape)} / {tuple(pos_y.shape)}")
    home = home_xy.float()
    if home.dim() == 3 and home.shape[1] == 1:
        home = home[:, 0, :]
    if home.dim() != 2 or int(home.shape[0]) != int(pos_x.shape[0]) or int(home.shape[1]) != 2:
        raise ValueError(f"home_xy must be (B, 2), got {tuple(home_xy.shape)}")
    dx = pos_x.float() - home[:, 0:1]
    dy = pos_y.float() - home[:, 1:2]
    return torch.sqrt(dx * dx + dy * dy)


def assign_roles_from_geometry(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    home_xy: torch.Tensor,
    alive: torch.Tensor,
    k: int | torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic geometric roles for a batch of teams.

    ``k`` is the number of closest living agents assigned DEFEND.
    Default ``None`` is the locked ``role_k(N) = N/2`` rule. An explicit
    int or per-env ``(B,)`` tensor is the opt-in used by the 6v6
    5A/1D vs 3A/3D contrast (k in {1, 3}); it does not change the default.

    Returns
    -------
    roles : (B, N) float32 with {0=DEFEND, 1=ATTACK}. Dead slots are 0.
    distances : (B, N) float32 d_i (raw; dead agents still have a finite d_i
        from their last position, but they are excluded from ranking).
    """
    if alive.dtype != torch.bool:
        alive = alive.bool()
    if alive.shape != pos_x.shape:
        raise ValueError(f"alive shape {tuple(alive.shape)} != pos {tuple(pos_x.shape)}")
    B, N = pos_x.shape
    if k is None:
        k_row = torch.full((B,), role_k(N), dtype=torch.long, device=pos_x.device)
    elif isinstance(k, int):
        if k < 1 or k > N:
            raise ValueError(f"k must be in [1, {N}], got {k}")
        k_row = torch.full((B,), int(k), dtype=torch.long, device=pos_x.device)
    else:
        k_row = k.to(device=pos_x.device, dtype=torch.long).reshape(-1)
        if int(k_row.shape[0]) != B:
            raise ValueError(f"k must have shape ({B},), got {tuple(k_row.shape)}")
        if int(k_row.min()) < 1 or int(k_row.max()) > N:
            raise ValueError(f"k entries must be in [1, {N}], got {k_row.tolist()}")
    d = distances_to_home(pos_x, pos_y, home_xy)
    roles = torch.zeros((B, N), dtype=torch.float32, device=pos_x.device)

    # Dead → +inf for ranking so they never win a DEFEND slot.
    rank_d = d.clone()
    rank_d = rank_d.masked_fill(~alive, float("inf"))
    # Lexicographic (d ascending, index ascending): stable sort by index, then by d.
    idx = torch.arange(N, device=pos_x.device).expand(B, N)
    order_by_idx = torch.argsort(idx, dim=1, stable=True)
    d_reordered = torch.gather(rank_d, 1, order_by_idx)
    order_by_d = torch.argsort(d_reordered, dim=1, stable=True)
    ranked = torch.gather(order_by_idx, 1, order_by_d)  # (B, N) agent ids closest→farthest

    n_alive = alive.sum(dim=1)
    k_eff = torch.minimum(n_alive, k_row.to(dtype=n_alive.dtype))

    # Default living agents to ATTACK; promote the closest k_eff to DEFEND.
    roles = roles.masked_fill(alive, float(ROLE_ATTACK))
    for b in range(B):
        ke = int(k_eff[b].item())
        if ke <= 0:
            continue
        defend_ids = ranked[b, :ke]
        roles[b, defend_ids] = float(ROLE_DEFEND)
    # Dead forced to 0 (DEFEND sentinel; excluded from ranking).
    roles = roles.masked_fill(~alive, float(ROLE_DEFEND))
    return roles, d


def living_set_changed(prev_alive: Optional[torch.Tensor], alive: torch.Tensor) -> torch.Tensor:
    """Per-env bool: living roster differs from previous step."""
    alive_b = alive.bool()
    if prev_alive is None:
        return torch.ones(alive_b.shape[0], dtype=torch.bool, device=alive_b.device)
    if prev_alive.shape != alive_b.shape:
        raise ValueError("prev_alive / alive shape mismatch")
    return (prev_alive.bool() != alive_b).any(dim=1)


class RoleHoldState:
    """Persistent roles with fixed H_r hold and immediate death reassignment."""

    def __init__(
        self,
        n_envs: int,
        n_agents: int,
        *,
        hold_ticks: int = 8,
        fixed_for_episode: bool = False,
        device: str | torch.device = "cpu",
        k_defend: int | None = None,
        k_choices: tuple[int, ...] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.n_envs = int(n_envs)
        self.n_agents = int(n_agents)
        self.hold_ticks = int(hold_ticks)
        if self.hold_ticks < 1:
            raise ValueError(f"hold_ticks must be >= 1, got {self.hold_ticks}")
        self.fixed_for_episode = bool(fixed_for_episode)
        self.device = torch.device(device)
        if k_defend is not None and k_choices:
            raise ValueError("k_defend and k_choices are mutually exclusive")
        if k_defend is not None and not (1 <= int(k_defend) <= self.n_agents):
            raise ValueError(f"k_defend must be in [1, {self.n_agents}], got {k_defend}")
        if k_choices:
            bad = [k for k in k_choices if not (1 <= int(k) <= self.n_agents)]
            if bad:
                raise ValueError(f"k_choices entries must be in [1, {self.n_agents}], got {bad}")
        self.k_defend = None if k_defend is None else int(k_defend)
        self.k_choices = tuple(int(k) for k in k_choices) if k_choices else None
        self._gen = generator
        default_k = role_k(self.n_agents) if self.k_defend is None else self.k_defend
        self.k_per_env = torch.full((self.n_envs,), default_k, dtype=torch.long, device=self.device)
        if self.k_choices:
            self._resample_k(torch.ones(self.n_envs, dtype=torch.bool, device=self.device))
        self.roles = torch.zeros(
            (self.n_envs, self.n_agents), dtype=torch.float32, device=self.device
        )
        self.distances = torch.zeros(
            (self.n_envs, self.n_agents), dtype=torch.float32, device=self.device
        )
        self.age = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self.prev_alive: Optional[torch.Tensor] = None
        # DEFEND_TEACHER_ROLE_CONDITIONING_A_V1_SPEC ROLE_ASSIGNMENT_locked:
        # only meaningful when fixed_for_episode=True. Set exclusively by
        # reset_envs() and consumed by the very next update(), independent of
        # the `force` kwarg and of `changed`/`expired` below -- the collector
        # calls update(force=True) at the top of EVERY rollout-collection
        # cycle (collector.py's per-collect() privileged-conditioning
        # refresh), not only at genuine episode boundaries, so `force` is not
        # a safe proxy for "an episode just reset". Starts all-True so the
        # first update() of a fresh RoleHoldState performs its initial
        # assignment.
        self._pending_reassign = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)

    def _resample_k(self, env_mask: torch.Tensor) -> None:
        """Draw a fresh k for the masked envs from k_choices (episode start)."""
        if not self.k_choices:
            return
        m = env_mask.bool().to(self.device)
        n = int(m.sum().item())
        if n <= 0:
            return
        choices = torch.tensor(self.k_choices, dtype=torch.long)
        draws = torch.randint(0, len(self.k_choices), (n,), generator=self._gen)
        picked = choices[draws].to(self.device)
        self.k_per_env = self.k_per_env.clone()
        self.k_per_env[m] = picked

    def reset_envs(self, env_mask: torch.Tensor) -> None:
        """Force reassignment on next update for the selected envs (episode starts)."""
        m = env_mask.bool().to(self.device)
        self.age = self.age.masked_fill(m, self.hold_ticks)  # expire hold
        if self.prev_alive is not None:
            # Clear prev so living-set check also fires.
            self.prev_alive = self.prev_alive.clone()
            self.prev_alive[m] = False
        self._pending_reassign = self._pending_reassign.clone()
        self._pending_reassign[m] = True
        self._resample_k(m)

    def update(
        self,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        home_xy: torch.Tensor,
        alive: torch.Tensor,
        *,
        force: bool = False,
        advance_age: bool = True,
    ) -> torch.Tensor:
        """Advance role state; return current roles (B, N).

        ``advance_age=False`` syncs assignment to the current geometry (e.g. for
        bootstrap ``next_values`` after ``env.step``) without consuming an extra
        hold tick — the act-time update already advanced age for this decision.
        """
        pos_x = pos_x.to(self.device)
        pos_y = pos_y.to(self.device)
        home_xy = home_xy.to(self.device)
        alive = alive.to(self.device).bool()
        if pos_x.shape != (self.n_envs, self.n_agents):
            raise ValueError(
                f"expected pos ({self.n_envs}, {self.n_agents}), got {tuple(pos_x.shape)}"
            )

        if self.fixed_for_episode:
            # role_hold_ticks is provably inert in this mode: neither
            # death/revival (`changed`) nor tick-age expiry (`expired`) nor
            # the caller-supplied `force` kwarg may trigger reassignment --
            # only a pending reset_envs() call may. See __init__ docstring.
            need = self._pending_reassign
        else:
            changed = living_set_changed(self.prev_alive, alive)
            expired = self.age >= self.hold_ticks
            if force:
                need = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
            else:
                need = changed | expired

        if bool(need.any().item()):
            new_roles, new_d = assign_roles_from_geometry(
                pos_x, pos_y, home_xy, alive, k=self.k_per_env
            )
            self.roles = torch.where(need.unsqueeze(1), new_roles, self.roles)
            self.distances = torch.where(need.unsqueeze(1), new_d, self.distances)
            self.age = torch.where(need, torch.zeros_like(self.age), self.age)
        else:
            # Refresh distances for smoke / diagnostics without changing roles.
            _, cur_d = assign_roles_from_geometry(
                pos_x, pos_y, home_xy, alive, k=self.k_per_env
            )
            self.distances = cur_d

        if self.fixed_for_episode:
            self._pending_reassign = torch.zeros_like(self._pending_reassign)

        if advance_age:
            self.age = self.age + 1
        self.prev_alive = alive.clone()
        return self.roles.clone()


def roles_from_core(
    core,
    hold: RoleHoldState,
    *,
    force: bool = False,
    advance_age: bool = True,
) -> torch.Tensor:
    """Compute / update roles from a BatchedCTFCore (blue side)."""
    home = core.blue_flag_home
    if home.dim() == 3:
        home = home[:, 0, :]
    return hold.update(
        core.blue_x,
        core.blue_y,
        home,
        core.blue_alive.bool(),
        force=force,
        advance_age=advance_age,
    )


def assert_no_strategy_leakage_inputs(**kwargs) -> None:
    """Fail closed if forbidden channels are passed into assignment."""
    forbidden = (
        "macro",
        "teacher_macro",
        "reward",
        "pole",
        "getflag",
        "getflag_preserve",
        "action",
        "logits",
    )
    bad = [k for k in kwargs if any(f in k.lower() for f in forbidden)]
    if bad:
        raise ValueError(
            f"role assignment must not read strategy-leakage channels; got {bad}"
        )
