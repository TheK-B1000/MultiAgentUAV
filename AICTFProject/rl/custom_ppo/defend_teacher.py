"""DEFEND-only teacher-imitation loss for role-conditioned pi_A training.

Frozen protocol:
artifacts/strategic_demand/sppo/DEFEND_TEACHER_ROLE_CONDITIONING_A_V1_SPEC.json

    L_teacher = 1[role_i=DEFEND] * 1[decision-eligible_i] * 1[alive_i] *
                ( CE(macro_logits_i, GO_TO) + CE(waypoint_logits_i, w_N'_i) )

``w_N'`` is a batched-GPU port of experiments/run_goto_only_defend_substitution_4v4.py's
sealed N' controller (``controller_select``, built on
experiments/audit_scaffold_to_native_representability_4v4.py::Engine.path_oracle /
rollout_ref). It is a faithful port, not a new approximation: it calls the SAME
real ``BatchedCTFCore`` physics primitives the audit's ``Engine`` wraps
(``core._defend_outward_target``, ``core._integrate_side``) rather than
reimplementing the DEFEND target law or the marine-kinematics integrator.
See TEACHER_locked and CONTRACTS_before_training C3 in the frozen spec.

Applied on student-visited PPO minibatches, separate zero_grad/backward/step,
cadence 4, matching sibling_sep / role_pres / getflag_preserve discipline.
Requires role_conditioning_enabled=True. Disabled means structurally absent:
construct no runner when lambda<=0, and skip every forward/backward/step
whenever the resolved schedule value is exactly 0.0 (the teacher-free
consolidation phase) -- not merely scale a computed loss by zero.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from macro_actions import MacroAction
from rl.custom_ppo.exp2_teacher_compression import decision_eligible_agents
from rl.custom_ppo.rule_role_assignment import ROLE_DEFEND

__all__ = [
    "DefendTeacherRunner",
    "compute_defend_teacher_waypoints",
    "defend_teacher_loss",
    "masked_macro_and_waypoint_logits",
]

GO_TO = int(MacroAction.GO_TO)

_ENTITY_KEYS = ("teammates", "teammates_valid", "enemies", "enemies_valid")


def _entity_kwargs(obs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: obs[k] for k in _ENTITY_KEYS if k in obs}


# ---------------------------------------------------------------------------
# Physics port (contract C3): the training-time w_N' target.
# ---------------------------------------------------------------------------


def _defend_target(
    core: Any, x: torch.Tensor, y: torch.Tensor, h: torch.Tensor, flag_xy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of the audit's ``Engine.defend_t``: the DEFEND_unified
    inward/outward target selection, WITHOUT ``_build_targets_from_action``'s
    own_carrying override -- N''s controller never applies that override, so
    reproducing it here would not be faithful to the sealed controller.
    """
    from gpu_env._core._rules import _pyquaticus_defender_radius_cells

    side_t = {"own_x": x, "own_y": y, "own_heading": h}
    otx, oty = core._defend_outward_target(side_t, flag_xy)
    ax = x - flag_xy[:, None, 0]
    ay = y - flag_xy[:, None, 1]
    radius = _pyquaticus_defender_radius_cells(float(core.cfg.tag_range_cells))
    inward = torch.sqrt(ax * ax + ay * ay) > radius
    tx = torch.where(inward, flag_xy[:, None, 0].expand_as(otx), otx)
    ty = torch.where(inward, flag_xy[:, None, 1].expand_as(oty), oty)
    return tx, ty


def _integrate(
    core: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    h: torch.Tensor,
    v: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror of the audit's ``Engine.step_t``: one tick of the real
    marine-kinematics integrator, ``alive=True`` throughout (an isolated
    hypothetical rollout, not the real agent's live alive status)."""
    if float(core.rt_drift_sigma_cells.max().detach().cpu()) != 0.0:
        raise RuntimeError(
            "defend_teacher physics port assumes drift_sigma_cells == 0 "
            "(_integrate_side draws from the environment's own RNG when "
            "drift is nonzero, which would silently perturb the real "
            "training env's random stream as a side effect of computing a "
            "teacher label). Revisit this port before enabling drift."
        )
    cap = float(core.cfg.max_speed_cps) * core.rt_blue_speed_scale.reshape(-1, 1).expand_as(v)
    alive = torch.ones_like(x, dtype=torch.bool)
    x_out, y_out, h_out, v_out, _oob, _yaw = core._integrate_side(
        x, y, h, v, alive, tx, ty, speed_cap=cap
    )
    return x_out, y_out, h_out, v_out


@torch.no_grad()
def compute_defend_teacher_waypoints(core: Any) -> torch.Tensor:
    """Batched port of ``controller_select``: the first-commit-boundary
    greedy GO_TO waypoint choice for every (env, agent), as an int64 index
    into ``core._macro_targets`` (shared 1:1 with the actor's waypoint head
    class index space -- zero remapping needed).

    Computed unconditionally for every agent regardless of role or
    eligibility (cheap, fully batched); callers gate by role==DEFEND,
    decision-eligible, alive at loss time.
    """
    device = core.blue_x.device
    B, N = core.blue_x.shape
    x0, y0, h0, v0 = core.blue_x, core.blue_y, core.blue_heading, core.blue_speed
    own_flag_now = core.blue_flag_pos  # (B, 2): current, possibly-captured position

    lc = int(core.cfg.macro_commit_go_to_ticks)
    if lc < 1:
        raise ValueError(f"macro_commit_go_to_ticks must be >= 1, got {lc}")
    arrival = float(core.cfg.macro_arrival_radius_cells)
    n_targets = int(core.cfg.n_targets)
    n_macros = int(core.cfg.n_macros)

    # ---- reference rollout: the isolated, unconstrained DEFEND law -------
    x, y, h, v = x0, y0, h0, v0
    ref_x = torch.zeros((lc, B, N), dtype=torch.float32, device=device)
    ref_y = torch.zeros((lc, B, N), dtype=torch.float32, device=device)
    for k in range(lc):
        tx, ty = _defend_target(core, x, y, h, own_flag_now)
        x, y, h, v = _integrate(core, x, y, h, v, tx, ty)
        ref_x[k], ref_y[k] = x, y

    # ---- candidate rollout: every fixed waypoint slot, in index order ----
    wp = core._macro_targets  # (n_targets, 2), fixed and shared across agents
    cx = x0.unsqueeze(-1).expand(B, N, n_targets).clone()
    cy = y0.unsqueeze(-1).expand(B, N, n_targets).clone()
    ch = h0.unsqueeze(-1).expand(B, N, n_targets).clone()
    cv = v0.unsqueeze(-1).expand(B, N, n_targets).clone()
    wx = wp[:, 0].to(device=device, dtype=torch.float32).view(1, 1, n_targets).expand(B, N, n_targets)
    wy = wp[:, 1].to(device=device, dtype=torch.float32).view(1, 1, n_targets).expand(B, N, n_targets)

    errs = torch.zeros((B, N, n_targets), dtype=torch.float32, device=device)
    cnt = torch.zeros((B, N, n_targets), dtype=torch.float32, device=device)
    ended = torch.zeros((B, N, n_targets), dtype=torch.bool, device=device)
    for k in range(lc):
        flat_shape = (B, N * n_targets)
        nx, ny, nh, nv = _integrate(
            core,
            cx.reshape(flat_shape),
            cy.reshape(flat_shape),
            ch.reshape(flat_shape),
            cv.reshape(flat_shape),
            wx.reshape(flat_shape),
            wy.reshape(flat_shape),
        )
        cx, cy, ch, cv = (t.view(B, N, n_targets) for t in (nx, ny, nh, nv))

        committed = ~ended
        rx = ref_x[k].unsqueeze(-1)
        ry = ref_y[k].unsqueeze(-1)
        step_err = (cx - rx) ** 2 + (cy - ry) ** 2
        errs = torch.where(committed, errs + step_err, errs)
        cnt = torch.where(committed, cnt + 1.0, cnt)
        arrived = torch.sqrt((cx - wx) ** 2 + (cy - wy) ** 2) <= arrival
        ended = ended | (committed & arrived)

    mask = core._build_action_mask(side="blue").view(B, N, n_macros + n_targets)
    legal_wp = mask[:, :, n_macros : n_macros + n_targets] > 0.0

    mean_err = errs / torch.clamp(cnt, min=1.0)
    mean_err = torch.where(legal_wp, mean_err, torch.full_like(mean_err, float("inf")))
    # Stable sort: ties break to the lowest waypoint index, matching
    # np.lexsort((idx, mac, errs))[0] in the CPU reference (candidates are
    # already enumerated in increasing waypoint-index order along the last
    # dim, so the first entry of a stable ascending sort is the tie-break
    # winner).
    order = torch.argsort(mean_err, dim=-1, stable=True)
    return order[..., 0]


# ---------------------------------------------------------------------------
# Loss.
# ---------------------------------------------------------------------------


def masked_macro_and_waypoint_logits(
    model: Any, obs: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return legality-masked (macro, waypoint) logits, each ``(B, n_agents, .)``.

    Uses the same ``policy_logits`` + ``_mask_logits`` path PPO's actor update
    uses, mirroring ``getflag_preservation.masked_macro_logits`` but pulling
    both heads per agent instead of only the macro head.
    """
    if not hasattr(model, "policy_logits") or not hasattr(model, "_mask_logits"):
        raise TypeError("model must expose policy_logits and _mask_logits")
    if "mask" not in obs:
        raise KeyError("obs missing legality mask")
    kw = _entity_kwargs(obs)
    role_kw: dict[str, torch.Tensor] = {}
    if bool(getattr(model, "role_conditioning_enabled", False)):
        if "roles" not in obs:
            raise KeyError("role-conditioned model requires obs['roles']")
        role_kw["roles"] = obs["roles"]
    flat = model._mask_logits(model.policy_logits(dict(obs), **kw, **role_kw), obs["mask"])
    dims = tuple(int(v) for v in model.action_dims)
    n_agents = int(model.n_agents)
    if len(dims) % n_agents:
        raise ValueError(f"{len(dims)} heads do not divide across {n_agents} agents")
    hpa = len(dims) // n_agents
    if hpa != 2:
        raise ValueError(f"defend-teacher expects exactly 2 heads per agent (macro, waypoint), got {hpa}")
    heads = torch.split(flat, list(dims), dim=-1)
    macro = torch.stack([heads[i * hpa] for i in range(n_agents)], dim=1)
    waypoint = torch.stack([heads[i * hpa + 1] for i in range(n_agents)], dim=1)
    return macro, waypoint


def defend_teacher_loss(
    student: Any,
    obs: Mapping[str, torch.Tensor],
    waypoint_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Gated CE(macro, GO_TO) + CE(waypoint, w_N') on DEFEND-role,
    decision-eligible, alive agents. ``waypoint_target``: (B, n_agents) int64,
    precomputed at rollout-collection time by
    :func:`compute_defend_teacher_waypoints`.
    """
    if "roles" not in obs:
        raise KeyError("defend_teacher_loss requires obs['roles'] (role_conditioning_enabled)")
    roles = obs["roles"]
    n_agents = int(student.n_agents)
    dims = tuple(int(v) for v in student.action_dims)
    decision = decision_eligible_agents(
        obs["mask"], action_dims=dims, n_agents=n_agents, agent_mask=obs.get("agent_mask"),
    )
    alive = torch.ones_like(decision, dtype=torch.bool)
    if obs.get("agent_mask") is not None:
        am = obs["agent_mask"]
        if am.dim() == 1:
            am = am.unsqueeze(0)
        alive = am > 0.5

    is_defend = roles == float(ROLE_DEFEND)
    gate = is_defend & decision & alive

    macro_logits, waypoint_logits = masked_macro_and_waypoint_logits(student, obs)
    n_gate = int(gate.sum().item())
    n_total = int(gate.numel())
    telemetry = {
        "n_gated": float(n_gate),
        "frac_gated": float(n_gate) / max(1, n_total),
        "frac_defend_role": float(is_defend.float().mean()),
        "frac_decision": float(decision.float().mean()),
    }
    if n_gate == 0:
        return macro_logits.new_zeros(()), telemetry

    macro_target = torch.full((n_gate,), GO_TO, dtype=torch.long, device=macro_logits.device)
    macro_ce = F.cross_entropy(macro_logits[gate], macro_target)
    waypoint_ce = F.cross_entropy(waypoint_logits[gate], waypoint_target[gate].long())
    loss = macro_ce + waypoint_ce
    telemetry["macro_ce"] = float(macro_ce.detach())
    telemetry["waypoint_ce"] = float(waypoint_ce.detach())
    telemetry["loss"] = float(loss.detach())
    return loss, telemetry


class DefendTeacherRunner:
    """Interleaved stop-grad DEFEND-teacher imitation — separate optimizer step.

    Unlike ``GetflagPreserveRunner``/``RolePresRunner``/``SiblingSepRunner``,
    ``lambda_teacher`` is MUTABLE: DEFEND_TEACHER_ROLE_CONDITIONING_A_V1_SPEC's
    LAMBDA_SCHEDULE_locked decays it over training, so the orchestrator/updater
    reassigns ``runner.lambda_teacher`` once per PPO ``update()`` from
    ``rl.custom_ppo.schedules.resolve_defend_teacher_lambda`` before the
    minibatch loop runs. When the resolved value is exactly 0.0 (the
    teacher-free consolidation phase), ``_step`` skips the forward/backward/
    optimizer-step entirely rather than scaling a computed loss by zero, so
    that phase really is teacher-free, not merely zero-weighted.
    """

    def __init__(
        self,
        student: Any,
        optimizer: Any,
        *,
        lambda_teacher: float,
        cadence: int = 4,
        max_grad_norm: float | None = None,
    ):
        if float(lambda_teacher) <= 0.0:
            raise ValueError(
                "DefendTeacherRunner must not be constructed with lambda_teacher <= 0. "
                "Disabled means NOT constructing the runner."
            )
        if int(cadence) < 1:
            raise ValueError("cadence must be >= 1")
        if not bool(getattr(student, "role_conditioning_enabled", False)):
            raise ValueError(
                "DefendTeacherRunner requires student.role_conditioning_enabled=True "
                "(DEFEND_TEACHER_ROLE_CONDITIONING_A_V1_SPEC TEACHER_locked.applies_to)"
            )
        self.student = student
        self.optimizer = optimizer
        self.lambda_teacher = float(lambda_teacher)
        self.cadence = int(cadence)
        self.max_grad_norm = max_grad_norm
        self.n_ppo_actor_minibatches = 0
        self.n_teacher_updates = 0
        self.n_updates = 0
        self.n_skipped_empty_gate = 0
        self.n_skipped_zero_lambda = 0
        self.last_loss = float("nan")
        self.last_telemetry: dict[str, float] = {}

    def note_ppo_minibatch(self, batch: Mapping[str, Any] | None = None) -> bool:
        self.n_ppo_actor_minibatches += 1
        if self.n_ppo_actor_minibatches % self.cadence != 0:
            return False
        if batch is None:
            raise RuntimeError("DefendTeacherRunner requires the PPO minibatch")
        return self._step(batch)

    def _obs_from_batch(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if "obs_grid" not in batch:
            raise KeyError("PPO minibatch missing obs_grid (DEFEND-teacher)")
        if "obs_roles" not in batch:
            raise KeyError(
                "PPO minibatch missing obs_roles (DEFEND-teacher requires role_conditioning_enabled)"
            )
        if "obs_defend_teacher_waypoint" not in batch:
            raise KeyError("PPO minibatch missing obs_defend_teacher_waypoint")
        obs = {
            "grid": batch["obs_grid"],
            "vec": batch["obs_vec"],
            "agent_mask": batch["obs_agent_mask"],
            "mask": batch["obs_mask"],
            "roles": batch["obs_roles"],
        }
        for k in _ENTITY_KEYS:
            bk = f"obs_{k}"
            if bk in batch:
                obs[k] = batch[bk]
        return obs

    def _step(self, batch: Mapping[str, Any]) -> bool:
        self.n_teacher_updates += 1
        if self.lambda_teacher <= 0.0:
            self.n_skipped_zero_lambda += 1
            self.last_loss = 0.0
            self.last_telemetry = {}
            return False
        obs = self._obs_from_batch(batch)
        waypoint_target = batch["obs_defend_teacher_waypoint"]
        raw, tel = defend_teacher_loss(self.student, obs, waypoint_target)
        self.last_telemetry = tel
        if int(tel.get("n_gated", 0)) == 0:
            self.n_skipped_empty_gate += 1
            self.last_loss = 0.0
            return False
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.lambda_teacher * raw
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]],
                float(self.max_grad_norm),
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.n_updates += 1
        self.last_loss = float(loss.detach())
        return True

    def telemetry(self) -> dict[str, float]:
        out = {
            "defend_teacher_lambda": float(self.lambda_teacher),
            "defend_teacher_cadence": float(self.cadence),
            "defend_teacher_n_ppo_actor_updates": float(self.n_ppo_actor_minibatches),
            "defend_teacher_n_teacher_updates": float(self.n_teacher_updates),
            "defend_teacher_n_updates": float(self.n_updates),
            "defend_teacher_n_skipped_empty_gate": float(self.n_skipped_empty_gate),
            "defend_teacher_n_skipped_zero_lambda": float(self.n_skipped_zero_lambda),
            "defend_teacher_loss": float(self.last_loss),
        }
        for k, v in self.last_telemetry.items():
            if isinstance(v, (int, float)):
                out[f"defend_teacher_{k}"] = float(v)
        return out
