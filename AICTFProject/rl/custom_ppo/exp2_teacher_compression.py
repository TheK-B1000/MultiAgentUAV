"""EXP2 online frozen-teacher KL for supervised K=2 repertoire compression.

This module implements the single intervention frozen in
``artifacts/strategic_demand/EXP2_K2_LATENT_COMPRESSION_PROTOCOL.json``.
It is deliberately separate from SAPPO's offline hard-action rehearsal.

The student sees only ``(o, z)``. Teacher selection is the immutable mapping
``z=0 -> pi_A`` and ``z=1 -> pi_B``. Opponent IDs are neither accepted by this
API nor read from PPO minibatches. The minimized loss is exactly

    lambda * KL(pi_teacher(.|o,legal_mask) || pi_student(.|o,z,legal_mask)).

When EXP2 is disabled no runner is constructed, so checkpoint loads, teacher
forwards, gradients, optimizer steps, and counters are structurally absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

__all__ = [
    "Exp2TeacherCompressionRunner",
    "decision_eligible_agents",
    "teacher_student_kl",
]


def _obs_from_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Build the decentralized actor input only. No opponent field is read."""
    return {
        "grid": batch["obs_grid"],
        "vec": batch["obs_vec"],
        "agent_mask": batch["obs_agent_mask"],
        "mask": batch["obs_mask"],
    }


def _masked_logits(model: Any, obs: Mapping[str, torch.Tensor], *, z_idx=None) -> torch.Tensor:
    """Use the exact masking functions used by PPO ``evaluate_actions``."""
    if not hasattr(model, "policy_logits") or not hasattr(model, "_mask_logits"):
        raise TypeError("teacher and student models must expose policy_logits and _mask_logits")
    return model._mask_logits(model.policy_logits(dict(obs), z_idx=z_idx), obs["mask"])


def decision_eligible_agents(
    obs_mask: torch.Tensor,
    *,
    action_dims: tuple[int, ...],
    n_agents: int,
    agent_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``(batch,n_agents)`` where a new macro choice is available.

    Commitment is represented by the environment's legal-action mask: while an
    agent is locked, its macro head is one-hot at the latched macro. A live
    agent with more than one legal macro action is therefore at exactly the
    decision point used by the SAPPO collection protocol.
    """
    if len(action_dims) % int(n_agents):
        raise ValueError("action heads do not divide across agents")
    heads_per_agent = len(action_dims) // int(n_agents)
    if heads_per_agent < 1:
        raise ValueError("each agent requires at least one action head")
    if obs_mask.dim() != 2 or int(obs_mask.shape[1]) != int(sum(action_dims)):
        raise ValueError(
            f"obs mask shape {tuple(obs_mask.shape)} does not match action dims {action_dims}"
        )
    heads = torch.split(obs_mask, list(action_dims), dim=-1)
    eligible = torch.stack(
        [heads[agent_i * heads_per_agent].gt(0).sum(dim=-1) > 1
         for agent_i in range(int(n_agents))],
        dim=1,
    )
    if agent_mask is not None:
        eligible &= agent_mask.bool()
    return eligible


@dataclass(frozen=True)
class _KLMetrics:
    kl: torch.Tensor
    agreement: torch.Tensor
    active_heads: int


def _kl_for_rows(
    teacher: Any,
    student: Any,
    obs: Mapping[str, torch.Tensor],
    z_idx: torch.Tensor,
    decision_mask: torch.Tensor,
) -> _KLMetrics:
    if tuple(teacher.action_dims) != tuple(student.action_dims):
        raise ValueError(
            f"teacher/student action dims differ: {teacher.action_dims} vs {student.action_dims}"
        )
    with torch.no_grad():
        teacher_flat = _masked_logits(teacher, obs, z_idx=None)
    student_flat = _masked_logits(student, obs, z_idx=z_idx)
    t_heads = torch.split(teacher_flat, list(student.action_dims), dim=-1)
    s_heads = torch.split(student_flat, list(student.action_dims), dim=-1)
    n_agents = int(student.n_agents)
    heads_per_agent = len(s_heads) // n_agents
    active = decision_mask.to(dtype=student_flat.dtype).repeat_interleave(
        heads_per_agent, dim=1
    )
    denom = active.sum()
    if float(denom.detach().cpu()) <= 0.0:
        raise RuntimeError("EXP2 teacher batch contains no decision-eligible action heads")

    per_head_kl = []
    per_head_agreement = []
    for t_logits, s_logits in zip(t_heads, s_heads):
        t_logp = t_logits.log_softmax(dim=-1)
        t_prob = t_logp.exp()
        s_logp = s_logits.log_softmax(dim=-1)
        per_head_kl.append((t_prob * (t_logp - s_logp)).sum(dim=-1))
        per_head_agreement.append(
            (t_logits.argmax(dim=-1) == s_logits.argmax(dim=-1)).to(student_flat.dtype)
        )
    kl_rows = torch.stack(per_head_kl, dim=1)
    agreement_rows = torch.stack(per_head_agreement, dim=1)
    return _KLMetrics(
        kl=(kl_rows * active).sum() / denom,
        agreement=(agreement_rows * active).sum() / denom,
        active_heads=int(denom.detach().cpu().item()),
    )


def teacher_student_kl(
    student: Any,
    teachers: Mapping[int, Any],
    obs: Mapping[str, torch.Tensor],
    z_idx: torch.Tensor,
    decision_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute teacher||student KL with the fixed z-to-teacher routing."""
    if set(teachers) != {0, 1}:
        raise ValueError(f"EXP2 requires exactly teacher keys {{0,1}}, got {set(teachers)}")
    if z_idx.dim() != 1 or int(z_idx.shape[0]) != int(decision_mask.shape[0]):
        raise ValueError("z_idx and decision mask batch dimensions differ")
    if bool(((z_idx < 0) | (z_idx > 1)).any().item()):
        raise ValueError("EXP2 student batch contains a latent other than z=0 or z=1")

    weighted_kl = torch.zeros((), dtype=torch.float32, device=z_idx.device)
    total_heads = 0
    telemetry: dict[str, float] = {}
    for z in (0, 1):
        rows = torch.where(z_idx == z)[0]
        if int(rows.numel()) == 0:
            raise RuntimeError(f"EXP2 rehearsal minibatch is missing z={z}")
        obs_z = {k: v.index_select(0, rows) for k, v in obs.items()}
        mask_z = decision_mask.index_select(0, rows)
        metrics = _kl_for_rows(
            teachers[z], student, obs_z,
            z_idx=torch.full((int(rows.numel()),), z, dtype=torch.long, device=z_idx.device),
            decision_mask=mask_z,
        )
        weighted_kl = weighted_kl + metrics.kl * float(metrics.active_heads)
        total_heads += int(metrics.active_heads)
        telemetry[f"kl_z{z}"] = float(metrics.kl.detach().cpu())
        telemetry[f"agreement_z{z}"] = float(metrics.agreement.detach().cpu())
        telemetry[f"rows_z{z}"] = float(rows.numel())
        telemetry[f"active_heads_z{z}"] = float(metrics.active_heads)
    if total_heads <= 0:
        raise RuntimeError("EXP2 rehearsal has zero active heads")
    loss = weighted_kl / float(total_heads)
    telemetry["kl"] = float(loss.detach().cpu())
    telemetry["active_heads"] = float(total_heads)
    return loss, telemetry


class Exp2TeacherCompressionRunner:
    """One frozen teacher-KL update per complete PPO cadence group."""

    def __init__(
        self,
        student: Any,
        optimizer: Any,
        teachers: Mapping[int, Any],
        *,
        lambda_teacher: float,
        cadence: int,
        batch_size: int,
        max_grad_norm: float | None,
        seed: int,
        device: str | torch.device,
    ) -> None:
        if float(lambda_teacher) <= 0.0:
            raise ValueError("disabled EXP2 compression means no runner; lambda must be > 0")
        if int(cadence) < 1 or int(batch_size) < 2:
            raise ValueError("cadence must be >=1 and batch_size must be >=2")
        if set(teachers) != {0, 1}:
            raise ValueError("EXP2 requires the immutable mapping z0->teacherA, z1->teacherB")
        self.student = student
        self.optimizer = optimizer
        self.teachers = dict(teachers)
        self.lambda_teacher = float(lambda_teacher)
        self.cadence = int(cadence)
        self.batch_size = int(batch_size)
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.n_ppo_actor_minibatches = 0
        self.n_teacher_updates = 0
        self.last_teacher_loss = float("nan")
        self.last_metrics: dict[str, float] = {}
        self.realized_environment_steps = 0
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(int(seed))
        for teacher in self.teachers.values():
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)

    def _sample(self, batch: Mapping[str, torch.Tensor]):
        obs = _obs_from_batch(batch)
        z = batch["z"].long()
        decision = decision_eligible_agents(
            obs["mask"],
            action_dims=tuple(int(v) for v in self.student.action_dims),
            n_agents=int(self.student.n_agents),
            agent_mask=obs.get("agent_mask"),
        )
        # Sample the two assigned modes separately. This makes every rehearsal
        # update exactly balanced by teacher instead of relying on a globally
        # random 64-row draw from an otherwise balanced rollout.
        per_mode = []
        available = []
        for mode in (0, 1):
            candidates = torch.where(decision.any(dim=1) & (z == mode))[0]
            available.append(int(candidates.numel()))
            if int(candidates.numel()) < 1:
                raise RuntimeError(f"EXP2 rehearsal has no eligible z={mode} rows")
            n_mode = min(int(self.batch_size) // 2, int(candidates.numel()))
            order = torch.randperm(
                int(candidates.numel()), generator=self._rng, device=self.device
            )[:n_mode]
            per_mode.append(candidates.index_select(0, order))
        if available[0] != available[1] and min(available) < int(self.batch_size) // 2:
            # If a small smoke batch cannot fill 32 rows per mode, trim both to
            # the same size. Production minibatches are expected to fill 64.
            n_equal = min(int(rows.numel()) for rows in per_mode)
            per_mode = [rows[:n_equal] for rows in per_mode]
        rows = torch.cat(per_mode, dim=0)
        obs = {k: v.index_select(0, rows) for k, v in obs.items()}
        z = z.index_select(0, rows)
        decision = decision.index_select(0, rows)
        if int((z == 0).sum()) != int((z == 1).sum()):
            raise RuntimeError("EXP2 sampled rehearsal batch is not exactly balanced by z")
        return obs, z, decision

    def note_ppo_minibatch(self, batch: Mapping[str, torch.Tensor]) -> bool:
        self.n_ppo_actor_minibatches += 1
        if self.n_ppo_actor_minibatches % self.cadence:
            return False
        obs, z, decision = self._sample(batch)
        self.optimizer.zero_grad(set_to_none=True)
        kl, metrics = teacher_student_kl(self.student, self.teachers, obs, z, decision)
        loss = self.lambda_teacher * kl
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]],
                float(self.max_grad_norm),
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.n_teacher_updates += 1
        self.last_teacher_loss = float(loss.detach().cpu())
        self.last_metrics = metrics
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_ppo_actor_minibatches": int(self.n_ppo_actor_minibatches),
            "n_teacher_updates": int(self.n_teacher_updates),
            "last_teacher_loss": float(self.last_teacher_loss),
            "last_metrics": dict(self.last_metrics),
            "rng_state": self._rng.get_state().cpu(),
            "cadence": int(self.cadence),
            "batch_size": int(self.batch_size),
            "lambda_teacher": float(self.lambda_teacher),
            "realized_environment_steps": int(self.realized_environment_steps),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("cadence", self.cadence)) != self.cadence:
            raise RuntimeError("EXP2 resume cadence differs from checkpoint")
        if int(state.get("batch_size", self.batch_size)) != self.batch_size:
            raise RuntimeError("EXP2 resume batch size differs from checkpoint")
        if abs(float(state.get("lambda_teacher", self.lambda_teacher)) - self.lambda_teacher) > 1e-12:
            raise RuntimeError("EXP2 resume lambda differs from checkpoint")
        self.n_ppo_actor_minibatches = int(state.get("n_ppo_actor_minibatches", 0))
        self.n_teacher_updates = int(state.get("n_teacher_updates", 0))
        expected = self.n_ppo_actor_minibatches // self.cadence
        if self.n_teacher_updates != expected:
            raise RuntimeError(
                "EXP2 resume cadence state is inconsistent: "
                f"teacher={self.n_teacher_updates}, PPO={self.n_ppo_actor_minibatches}"
            )
        self.last_teacher_loss = float(state.get("last_teacher_loss", float("nan")))
        self.last_metrics = dict(state.get("last_metrics", {}) or {})
        self.realized_environment_steps = int(state.get("realized_environment_steps", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.set_state(rng_state.cpu() if isinstance(rng_state, torch.Tensor) else rng_state)

    def telemetry(self) -> dict[str, float]:
        per_cell_steps = float(self.realized_environment_steps) / 4.0
        out = {
            "exp2_n_ppo_actor_updates": float(self.n_ppo_actor_minibatches),
            "exp2_n_teacher_updates": float(self.n_teacher_updates),
            "exp2_teacher_to_ppo_ratio": float(
                self.n_teacher_updates / max(1, self.n_ppo_actor_minibatches)
            ),
            "exp2_teacher_loss": float(self.last_teacher_loss),
            "exp2_cell_count_z0_A": 8.0,
            "exp2_cell_count_z0_B": 8.0,
            "exp2_cell_count_z1_A": 8.0,
            "exp2_cell_count_z1_B": 8.0,
            "exp2_cell_steps_z0_A": per_cell_steps,
            "exp2_cell_steps_z0_B": per_cell_steps,
            "exp2_cell_steps_z1_A": per_cell_steps,
            "exp2_cell_steps_z1_B": per_cell_steps,
        }
        for key, value in self.last_metrics.items():
            out[f"exp2_teacher_{key}"] = float(value)
        return out
