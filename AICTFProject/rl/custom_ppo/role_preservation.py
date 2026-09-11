"""Role-preservation auxiliary for 4v4 B3 specialists.

Frozen protocol: artifacts/strategic_demand/sppo/4V4_B3_ROLE_PRESERVATION_SPEC.json
(+ TARGET_PROJECTION_AMENDMENT + LOSS_AMENDMENT)

    L_i = L_PPO,i + lambda_role * L_role,i
    L_role = CE(p*_style, q_soft)   # MSE dropped: f* does not separate styles

q_soft uses soft MacroAction attack/defend counts plus PPO-batch global_state
carrier features (existing GLOBAL_STATE_FIELD_NAMES) so escort_pair /
intercept_pair are expressible.

SAPPO-style separate zero_grad/backward/step. Disabled = structurally absent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from macro_actions import MacroAction
from rl.global_state import GLOBAL_STATE_FIELD_NAMES

__all__ = [
    "RolePresRunner",
    "load_role_targets",
    "role_preservation_loss",
    "soft_role_features",
]

N_ROLE = 7
ROLE_NAMES = (
    "all_push",
    "three_attack_one_defend",
    "two_attack_two_defend",
    "one_attack_three_defend",
    "escort_pair",
    "intercept_pair",
    "turtle_defense",
)

# Indices into GLOBAL_STATE_FIELD_NAMES (frozen layout).
_GS = {name: i for i, name in enumerate(GLOBAL_STATE_FIELD_NAMES)}
_IDX_BLUE_CARRYING = _GS["red_flag_captured"]       # blue carrying any
_IDX_RED_CARRYING = _GS["blue_flag_captured"]        # red carrying any
_IDX_CARRIER_SUPPORT = _GS["carrier_teammate_support"]


def load_role_targets(path: str | Path, style: str) -> dict[str, Any]:
    style = str(style).upper()
    if style not in ("GUARD", "BREACH"):
        raise ValueError(f"style must be GUARD or BREACH, got {style!r}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if style not in payload.get("targets", {}):
        raise ValueError(f"{path}: missing targets[{style}]")
    t = payload["targets"][style]
    occ = t["role_occupancy_vector"]
    if len(occ) != N_ROLE:
        raise ValueError(f"role_occupancy_vector must have length {N_ROLE}")
    feat = t["features"]
    return {
        "style": style,
        "style_id": t["style_id"],
        "p_star": torch.tensor(occ, dtype=torch.float32),
        "f_star": torch.tensor(
            [
                float(feat["mean_num_attackers"]),
                float(feat["mean_num_defenders"]),
                float(feat["mean_attack_defense_ratio"]),
            ],
            dtype=torch.float32,
        ),
        "dominant_role": t["dominant_role"],
        "source_path": str(path),
        "n_episodes": int(t["n_episodes"]),
    }


def soft_role_features(
    student: Any,
    obs: Mapping[str, torch.Tensor],
    *,
    temperature: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return ``(f_soft [B,3], q_soft [B,7], telemetry)``."""
    from rl.teacher_distillation import head_logits

    logits = head_logits(student, obs)
    n_agents = int(student.n_agents)
    macro_idx = [2 * i for i in range(n_agents)]
    if macro_idx[-1] >= len(logits):
        raise RuntimeError(
            f"macro head layout mismatch: n_agents={n_agents} n_heads={len(logits)}"
        )

    stacked = torch.stack([logits[i] for i in macro_idx], dim=1)
    probs = F.softmax(stacked, dim=-1)
    p_att = probs[..., int(MacroAction.GET_FLAG)]
    p_def = probs[..., int(MacroAction.GO_HOME)]
    p_goto = probs[..., int(MacroAction.GO_TO)]

    alive = None
    if obs.get("agent_mask") is not None:
        alive = obs["agent_mask"].to(dtype=probs.dtype)
        if alive.dim() == 1:
            alive = alive.unsqueeze(0)
        if alive.shape[-1] != n_agents and alive.numel() == probs.shape[0] * n_agents:
            alive = alive.view(probs.shape[0], n_agents)
    if alive is None or alive.shape[-1] != n_agents:
        alive = torch.ones(probs.shape[0], n_agents, device=probs.device, dtype=probs.dtype)

    n_att = (p_att * alive).sum(dim=1)
    n_def = (p_def * alive).sum(dim=1)
    n_goto = (p_goto * alive).sum(dim=1)
    att_w = n_att + 0.5 * n_goto
    def_w = n_def + 0.5 * n_goto
    ad_ratio = att_w / torch.clamp(att_w + def_w, min=1e-3)
    f_soft = torch.stack([n_att, n_def, ad_ratio], dim=-1)

    # Allocation prototypes (existing 4v4 role_bucket geometry).
    prototypes = f_soft.new_tensor(
        [
            [4.0, 0.0],
            [3.0, 1.0],
            [2.0, 2.0],
            [1.0, 3.0],
            [2.0, 1.0],  # escort stand-in; boosted below when carrying
            [2.0, 1.5],  # intercept stand-in; boosted below when enemy carrying
            [0.0, 4.0],
        ]
    )
    nd = torch.stack([n_att, n_def], dim=-1)
    dist = ((nd.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2).sum(dim=-1)

    # Carrier-aware boosts from global_state (LOSS_AMENDMENT).
    blue_carry = dist.new_zeros(dist.shape[0])
    red_carry = dist.new_zeros(dist.shape[0])
    support = dist.new_zeros(dist.shape[0])
    gs = obs.get("global_state")
    if gs is not None:
        if gs.dim() == 1:
            gs = gs.unsqueeze(0)
        if gs.shape[-1] > max(_IDX_BLUE_CARRYING, _IDX_RED_CARRYING, _IDX_CARRIER_SUPPORT):
            blue_carry = gs[:, _IDX_BLUE_CARRYING].to(dtype=dist.dtype).clamp(0.0, 1.0)
            red_carry = gs[:, _IDX_RED_CARRYING].to(dtype=dist.dtype).clamp(0.0, 1.0)
            support = gs[:, _IDX_CARRIER_SUPPORT].to(dtype=dist.dtype).clamp(0.0, 1.0)

    # Lower distance ⇒ higher softmax mass. Boost escort when blue carries;
    # boost intercept when red carries.
    escort_boost = 4.0 * blue_carry * (0.5 + 0.5 * support)
    intercept_boost = 4.0 * red_carry
    dist = dist.clone()
    dist[:, 4] = dist[:, 4] - escort_boost
    dist[:, 5] = dist[:, 5] - intercept_boost

    q_soft = F.softmax(-dist / max(float(temperature), 1e-3), dim=-1)

    telemetry = {
        "soft_n_attackers": float(n_att.detach().mean()),
        "soft_n_defenders": float(n_def.detach().mean()),
        "soft_ad_ratio": float(ad_ratio.detach().mean()),
        "soft_dominant_role_id": float(q_soft.detach().mean(0).argmax().item()),
        "frac_blue_carrying": float(blue_carry.detach().mean()),
        "frac_red_carrying": float(red_carry.detach().mean()),
    }
    return f_soft, q_soft, telemetry


def role_preservation_loss(
    student: Any,
    obs: Mapping[str, torch.Tensor],
    *,
    p_star: torch.Tensor,
    f_star: torch.Tensor | None = None,
    temperature: float = 0.5,
    include_mse: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """CE(p*, q_soft) + optional MSE(f_soft, f*) when style-intent targets separate."""
    f_soft, q_soft, tel = soft_role_features(student, obs, temperature=temperature)
    device = q_soft.device
    p_star = p_star.to(device=device, dtype=q_soft.dtype)
    p_star = p_star / torch.clamp(p_star.sum(), min=1e-8)

    qbar = q_soft.mean(dim=0)
    qbar = qbar / torch.clamp(qbar.sum(), min=1e-8)
    ce = -(p_star * torch.log(qbar + 1e-8)).sum()
    mse = f_soft.new_zeros(())
    if include_mse and f_star is not None:
        f_star_t = f_star.to(device=device, dtype=f_soft.dtype)
        mse = ((f_soft - f_star_t.unsqueeze(0)) ** 2).mean()
    loss = ce + mse
    tel = {
        **tel,
        "role_ce": float(ce.detach()),
        "role_mse": float(mse.detach()),
        "role_loss": float(loss.detach()),
        "qbar_dominant_id": float(qbar.detach().argmax().item()),
        "pstar_dominant_id": float(p_star.detach().argmax().item()),
    }
    return loss, tel


class RolePresRunner:
    """Interleaved role-preservation step on PPO actor minibatches."""

    def __init__(
        self,
        student: Any,
        optimizer: Any,
        *,
        targets: dict[str, Any],
        lambda_role: float,
        cadence: int = 4,
        temperature: float = 0.5,
        max_grad_norm: float | None = None,
    ):
        if float(lambda_role) <= 0.0:
            raise ValueError(
                "RolePresRunner must not be constructed with lambda_role <= 0. "
                "Disabled means NOT constructing the runner."
            )
        if int(cadence) < 1:
            raise ValueError("cadence must be >= 1")
        self.student = student
        self.optimizer = optimizer
        self.targets = targets
        self.lambda_role = float(lambda_role)
        self.cadence = int(cadence)
        self.temperature = float(temperature)
        self.max_grad_norm = max_grad_norm
        self.n_ppo_actor_minibatches = 0
        self.n_role_updates = 0
        self.last_role_loss = float("nan")
        self.last_telemetry: dict[str, float] = {}

    def note_ppo_minibatch(self, batch: Mapping[str, Any] | None = None) -> bool:
        self.n_ppo_actor_minibatches += 1
        if self.n_ppo_actor_minibatches % self.cadence != 0:
            return False
        if batch is None:
            raise RuntimeError("RolePresRunner requires the PPO minibatch observations")
        self._role_step(batch)
        return True

    def _obs_from_batch(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if "obs_grid" not in batch:
            raise KeyError(
                "PPO minibatch missing obs_grid/obs_vec/obs_mask (role preservation)"
            )
        obs = {
            "grid": batch["obs_grid"],
            "vec": batch["obs_vec"],
            "agent_mask": batch["obs_agent_mask"],
            "mask": batch["obs_mask"],
        }
        if "global_state" in batch:
            obs["global_state"] = batch["global_state"]
        return obs

    def _role_step(self, batch: Mapping[str, Any]) -> None:
        obs = self._obs_from_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        raw, tel = role_preservation_loss(
            self.student,
            obs,
            p_star=self.targets["p_star"],
            f_star=self.targets["f_star"],
            temperature=self.temperature,
        )
        loss = self.lambda_role * raw
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]],
                float(self.max_grad_norm),
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.n_role_updates += 1
        self.last_role_loss = float(loss.detach())
        self.last_telemetry = tel

    def telemetry(self) -> dict[str, float]:
        out = {
            "role_pres_lambda": float(self.lambda_role),
            "role_pres_cadence": float(self.cadence),
            "role_pres_n_ppo_actor_updates": float(self.n_ppo_actor_minibatches),
            "role_pres_n_updates": float(self.n_role_updates),
            "role_pres_to_ppo_ratio": float(
                self.n_role_updates / max(1, self.n_ppo_actor_minibatches)
            ),
            "role_pres_loss": float(self.last_role_loss),
        }
        for k, v in self.last_telemetry.items():
            if isinstance(v, (int, float)):
                out[f"role_pres_{k}"] = float(v)
        return out
