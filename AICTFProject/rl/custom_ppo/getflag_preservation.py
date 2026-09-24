"""Non-carrying GET_FLAG macro preservation for a short B-only continuation.

Frozen protocol: artifacts/strategic_demand/sppo/B_GETFLAG_PRESERVE_SPEC.json

    L = L_PPO + λ * 1[¬carrying_i] * 1[π_{B*} favors GET_FLAG] * (−log π_θ(GET_FLAG | o_i))

Macro head only. Waypoint / mine heads never enter D. Global JSD/KL on the
full action distribution is forbidden by the spec.

Applied on student-visited PPO minibatches, separate zero_grad/backward/step,
cadence 4, matching sibling_sep / role_pres discipline. Disabled means
structurally absent: construct no runner when lambda<=0.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from macro_actions import MacroAction
from rl.custom_ppo.exp2_teacher_compression import decision_eligible_agents

__all__ = [
    "GetflagPreserveRunner",
    "VEC_OWN_CARRYING_IDX",
    "getflag_preserve_loss",
    "masked_macro_logits",
]

#: Per-agent own_carrying in gpu_env/_core/_observations.py (out[..., 10]).
VEC_OWN_CARRYING_IDX = 10
GET_FLAG = int(MacroAction.GET_FLAG)
GO_TO = int(MacroAction.GO_TO)

_ENTITY_KEYS = ("teammates", "teammates_valid", "enemies", "enemies_valid")


def _entity_kwargs(obs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: obs[k] for k in _ENTITY_KEYS if k in obs}


def masked_macro_logits(model: Any, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return legality-masked macro logits with shape ``(B, n_agents, n_macro)``.

    Uses the same ``policy_logits`` + ``_mask_logits`` path PPO's actor update
    uses. Entity tensors are forwarded when present so entity-repair checkpoints
    fail closed rather than silently dropping geometry.
    """
    if not hasattr(model, "policy_logits") or not hasattr(model, "_mask_logits"):
        raise TypeError("model must expose policy_logits and _mask_logits")
    if "mask" not in obs:
        raise KeyError("obs missing legality mask")
    kw = _entity_kwargs(obs)
    flat = model._mask_logits(model.policy_logits(dict(obs), **kw), obs["mask"])
    dims = tuple(int(v) for v in model.action_dims)
    heads = torch.split(flat, list(dims), dim=-1)
    n_agents = int(model.n_agents)
    if len(dims) % n_agents:
        raise ValueError(f"{len(dims)} heads do not divide across {n_agents} agents")
    hpa = len(dims) // n_agents
    return torch.stack([heads[i * hpa] for i in range(n_agents)], dim=1)


def getflag_preserve_loss(
    student: Any,
    anchor: Any,
    obs: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Gated −log π_θ(GET_FLAG) on non-carrying states the anchor commits to GET_FLAG."""
    vec = obs["vec"]
    if vec.dim() != 3 or int(vec.shape[-1]) <= VEC_OWN_CARRYING_IDX:
        raise ValueError(
            f"vec must have shape (B, N, V) with V>{VEC_OWN_CARRYING_IDX}, got {tuple(vec.shape)}"
        )
    carrying = vec[..., VEC_OWN_CARRYING_IDX] > 0.5
    n_agents = int(student.n_agents)
    dims = tuple(int(v) for v in student.action_dims)
    decision = decision_eligible_agents(
        obs["mask"],
        action_dims=dims,
        n_agents=n_agents,
        agent_mask=obs.get("agent_mask"),
    )
    alive = torch.ones_like(carrying, dtype=torch.bool)
    if obs.get("agent_mask") is not None:
        am = obs["agent_mask"]
        if am.dim() == 1:
            am = am.unsqueeze(0)
        alive = am > 0.5

    with torch.no_grad():
        a_logits = masked_macro_logits(anchor, obs)
        anchor_getflag = a_logits.argmax(dim=-1) == GET_FLAG

    s_logits = masked_macro_logits(student, obs)
    logp_gf = s_logits.log_softmax(dim=-1)[..., GET_FLAG]
    gate = (~carrying) & anchor_getflag & decision & alive
    n_gate = int(gate.sum().item())
    n_total = int(gate.numel())
    telemetry = {
        "n_gated": float(n_gate),
        "frac_gated": float(n_gate) / max(1, n_total),
        "frac_carrying": float(carrying.float().mean()),
        "frac_anchor_getflag": float(anchor_getflag.float().mean()),
        "frac_decision": float(decision.float().mean()),
        "student_p_getflag_on_gate": float("nan"),
        "student_p_goto_on_gate": float("nan"),
        "anchor_p_getflag_on_gate": float("nan"),
    }
    if n_gate == 0:
        return s_logits.new_zeros(()), telemetry

    p_s = s_logits.softmax(dim=-1)
    p_a = a_logits.softmax(dim=-1)
    telemetry["student_p_getflag_on_gate"] = float(p_s[..., GET_FLAG][gate].detach().mean())
    telemetry["student_p_goto_on_gate"] = float(p_s[..., GO_TO][gate].detach().mean())
    telemetry["anchor_p_getflag_on_gate"] = float(p_a[..., GET_FLAG][gate].detach().mean())
    loss = -(logp_gf[gate]).mean()
    telemetry["loss"] = float(loss.detach())
    return loss, telemetry


class GetflagPreserveRunner:
    """Interleaved stop-grad GET_FLAG preservation — separate optimizer step."""

    def __init__(
        self,
        student: Any,
        optimizer: Any,
        anchor: Any,
        *,
        lambda_preserve: float,
        cadence: int = 4,
        max_grad_norm: float | None = None,
    ):
        if float(lambda_preserve) <= 0.0:
            raise ValueError(
                "GetflagPreserveRunner must not be constructed with lambda_preserve <= 0. "
                "Disabled means NOT constructing the runner."
            )
        if int(cadence) < 1:
            raise ValueError("cadence must be >= 1")
        self.student = student
        self.optimizer = optimizer
        self.anchor = anchor
        self.lambda_preserve = float(lambda_preserve)
        self.cadence = int(cadence)
        self.max_grad_norm = max_grad_norm
        self.n_ppo_actor_minibatches = 0
        self.n_updates = 0
        self.n_skipped_empty_gate = 0
        self.last_loss = float("nan")
        self.last_telemetry: dict[str, float] = {}
        self.anchor.eval()
        for p in self.anchor.parameters():
            p.requires_grad_(False)

    def note_ppo_minibatch(self, batch: Mapping[str, Any] | None = None) -> bool:
        self.n_ppo_actor_minibatches += 1
        if self.n_ppo_actor_minibatches % self.cadence != 0:
            return False
        if batch is None:
            raise RuntimeError("GetflagPreserveRunner requires the PPO minibatch")
        return self._step(batch)

    def _obs_from_batch(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if "obs_grid" not in batch:
            raise KeyError("PPO minibatch missing obs_grid (GET_FLAG preservation)")
        obs = {
            "grid": batch["obs_grid"],
            "vec": batch["obs_vec"],
            "agent_mask": batch["obs_agent_mask"],
            "mask": batch["obs_mask"],
        }
        for k in _ENTITY_KEYS:
            bk = f"obs_{k}"
            if bk in batch:
                obs[k] = batch[bk]
        return obs

    def _step(self, batch: Mapping[str, Any]) -> bool:
        obs = self._obs_from_batch(batch)
        raw, tel = getflag_preserve_loss(self.student, self.anchor, obs)
        self.last_telemetry = tel
        if int(tel.get("n_gated", 0)) == 0:
            self.n_skipped_empty_gate += 1
            self.last_loss = 0.0
            return False
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.lambda_preserve * raw
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
            "getflag_preserve_lambda": float(self.lambda_preserve),
            "getflag_preserve_cadence": float(self.cadence),
            "getflag_preserve_n_ppo_actor_updates": float(self.n_ppo_actor_minibatches),
            "getflag_preserve_n_updates": float(self.n_updates),
            "getflag_preserve_n_skipped_empty_gate": float(self.n_skipped_empty_gate),
            "getflag_preserve_to_ppo_ratio": float(
                self.n_updates / max(1, self.n_ppo_actor_minibatches)
            ),
            "getflag_preserve_loss": float(self.last_loss),
        }
        for k, v in self.last_telemetry.items():
            if isinstance(v, (int, float)):
                out[f"getflag_preserve_{k}"] = float(v)
        return out
