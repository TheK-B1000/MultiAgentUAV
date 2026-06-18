"""Counterfactual / forced-z policy separation objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F

from rl.custom_ppo.update.helpers import StrictFaithfulDictWrapper
from rl.custom_ppo.update.loss_result import (
    LossComponent,
    PairwiseSeparationMeasurement,
    measurement_from_pair_tensor,
)
from rl.custom_ppo.v6i1_cf_loss import v6i1_cf_separation_loss
from rl.ppo_core import TensorDictRolloutBuffer


def z_separation_gate_mask(
    *,
    advantages: torch.Tensor,
    action_entropy: torch.Tensor,
    global_state: torch.Tensor,
    max_action_entropy: float,
    min_abs_advantage: float,
    min_decision_frac: float,
    max_entropy_frac: float,
) -> torch.Tensor:
    """Select tactically meaningful rows for forced-z policy separation."""
    adv = advantages.detach().float().reshape(-1)
    entropy = action_entropy.detach().float().reshape(-1)
    if global_state.dim() != 2 or int(global_state.shape[0]) != int(adv.shape[0]):
        raise ValueError(
            "global_state must be (B, D) and align with advantages for z separation"
        )
    if int(entropy.shape[0]) != int(adv.shape[0]):
        raise ValueError("action_entropy must align with advantages for z separation")

    mask = torch.ones_like(adv, dtype=torch.bool)
    min_adv = max(0.0, float(min_abs_advantage))
    if min_adv > 0.0:
        mask &= adv.abs() >= min_adv

    min_progress = max(0.0, min(1.0, float(min_decision_frac)))
    if min_progress > 0.0:
        if int(global_state.shape[1]) <= 17:
            raise ValueError(
                "global_state needs decision_frac at index 17 for z separation gating"
            )
        mask &= global_state[:, 17].detach().float() >= min_progress

    entropy_frac = max(0.0, min(1.0, float(max_entropy_frac)))
    if entropy_frac < 1.0:
        entropy_ceiling = max(0.0, float(max_action_entropy)) * entropy_frac
        mask &= entropy <= entropy_ceiling
    return mask


def policy_z_separation_loss(
    model: Any,
    obs_batch: dict[str, torch.Tensor],
    z_idx: torch.Tensor,
    *,
    latent_k: int,
    margin: float,
    active_mask: torch.Tensor | None = None,
    subsample_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize collapsed z policies via per-pair JSD hinge (hinge before averaging)."""
    if int(latent_k) <= 1 or z_idx.numel() <= 0:
        zero = torch.zeros((), dtype=torch.float32, device=z_idx.device)
        return zero, {"jsd": zero, "active": zero}

    obs_batch = StrictFaithfulDictWrapper(obs_batch)
    batch_size = int(z_idx.reshape(-1).shape[0])
    device = z_idx.device

    active_fraction = torch.ones((), dtype=torch.float32, device=device)
    if active_mask is not None:
        mask = active_mask.to(device=device, dtype=torch.bool).reshape(-1)
        if int(mask.shape[0]) != batch_size:
            raise ValueError(
                f"active_mask length {int(mask.shape[0])} != batch size {batch_size}"
            )
        active_fraction = mask.float().mean()
        active_indices = torch.where(mask)[0]
        if active_indices.numel() <= 0:
            zero = torch.zeros((), dtype=torch.float32, device=device)
            return zero, {"jsd": zero, "active": zero}
        obs_active: dict[str, Any] = {}
        for key, value in obs_batch.items():
            if isinstance(value, torch.Tensor) and int(value.shape[0]) == batch_size:
                obs_active[key] = value.index_select(0, active_indices)
            else:
                obs_active[key] = value
        obs_batch = StrictFaithfulDictWrapper(obs_active)
        batch_size = int(active_indices.numel())

    max_jsd_rows = 512
    if batch_size > max_jsd_rows:
        indices = torch.randperm(
            batch_size, device=device, generator=subsample_generator
        )[:max_jsd_rows]
        obs_sub = {}
        for k, v in obs_batch.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
                obs_sub[k] = v[indices]
            else:
                obs_sub[k] = v
        obs_sub = StrictFaithfulDictWrapper(obs_sub)
        curr_batch_size = max_jsd_rows
    else:
        obs_sub = obs_batch
        curr_batch_size = int(batch_size)

    logits_list = []
    for k in range(latent_k):
        z_k = torch.full((curr_batch_size,), k, dtype=torch.long, device=device)
        logits_k = model._mask_logits(model.policy_logits(obs_sub, z_idx=z_k), obs_sub.get("mask"))
        logits_list.append(logits_k)

    all_pairwise_js: list[torch.Tensor] = []
    offset = 0
    for _agent_idx in range(int(model.n_agents)):
        for dim in model.per_agent_action_dims:
            width = int(dim)
            p_list = []
            for k in range(latent_k):
                a_k = logits_list[k][:, offset : offset + width]
                p_k = torch.softmax(a_k, dim=-1).clamp_min(1e-8)
                p_list.append(p_k)

            p_stacked = torch.stack(p_list, dim=0)
            p_i = p_stacked.unsqueeze(1)
            p_j = p_stacked.unsqueeze(0)
            m = 0.5 * (p_i + p_j)
            kl_i = (p_i * (p_i.log() - m.log())).sum(dim=-1)
            kl_j = (p_j * (p_j.log() - m.log())).sum(dim=-1)
            js_matrix = 0.5 * kl_i + 0.5 * kl_j
            pair_i, pair_j = torch.triu_indices(int(latent_k), int(latent_k), offset=1, device=device)
            pairwise_js = js_matrix[pair_i, pair_j]
            if pairwise_js.numel() > 0:
                all_pairwise_js.append(pairwise_js)
            offset += width

    if not all_pairwise_js:
        zero = torch.zeros((), dtype=torch.float32, device=z_idx.device)
        return zero, {"jsd": zero, "min_jsd": zero, "active": zero}

    stacked = torch.cat(all_pairwise_js, dim=0)
    margin_t = stacked.new_tensor(float(max(0.0, margin)))
    per_pair_penalty = F.relu(margin_t - stacked)
    loss = per_pair_penalty.mean()
    jsd = stacked.mean()
    return loss, {
        "jsd": jsd.detach(),
        "min_jsd": stacked.min().detach(),
        "active": active_fraction.detach(),
    }


def extract_rollout_resample_subset(
    buffer: TensorDictRolloutBuffer,
    *,
    require_selector_hidden: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str | None]:
    """Return rollout resample rows for the marginal-entropy objective."""
    t_full = int(buffer.pos)
    if t_full <= 0:
        return None, None, "empty_rollout"
    if "global_state" not in buffer.fields:
        raise KeyError("rollout marginal entropy requires global_state in buffer.fields")
    if "z_resampled" not in buffer.fields:
        raise KeyError("rollout marginal entropy requires z_resampled in buffer.fields")

    gs_full = buffer.fields["global_state"][:t_full]
    gs_full = gs_full.reshape(t_full * buffer.n_envs, *gs_full.shape[2:])
    rs_full = buffer.fields["z_resampled"][:t_full].reshape(-1).bool()
    if rs_full.numel() != gs_full.shape[0]:
        raise ValueError(
            f"z_resampled length {rs_full.numel()} != global_state rows {gs_full.shape[0]}"
        )
    if not bool(rs_full.any()):
        return None, None, "no_resample_rows"

    states = gs_full[rs_full]
    hidden: torch.Tensor | None = None
    if require_selector_hidden:
        if "selector_hidden" not in buffer.fields:
            raise KeyError(
                "recurrent selector rollout marginal entropy requires "
                "selector_hidden in buffer.fields"
            )
        sh_full = buffer.fields["selector_hidden"][:t_full]
        sh_full = sh_full.reshape(t_full * buffer.n_envs, *sh_full.shape[2:])
        if sh_full.shape[0] != gs_full.shape[0]:
            raise ValueError("selector_hidden rows misaligned with global_state")
        hidden = sh_full[rs_full]
    return states, hidden, None


@dataclass
class SeparationResult:
    loss: LossComponent
    pairwise_measurement: PairwiseSeparationMeasurement
    diagnostics: dict[str, float]
    train_active: float
    raw_stats: dict[str, Any]


class SeparationObjective:
    """Legacy forced-z and V6I1 counterfactual separation."""

    def __init__(
        self,
        *,
        model: Any,
        cfg: Any,
        hparams: Any,
        runtime: Any,
        latent_state: Any,
        subsample_generator: torch.Generator,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.hparams = hparams
        self.runtime = runtime
        self.latent_state = latent_state
        self.subsample_generator = subsample_generator

    def compute(
        self,
        *,
        obs_batch: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        entropy: torch.Tensor,
        z_idx: torch.Tensor | None,
        separation_coef: float,
        counterfactual_active: bool,
        device: torch.device,
        zero_scalar: torch.Tensor,
    ) -> SeparationResult:
        from rl.custom_ppo.v6i1_phase_runtime import is_v6i1_staged_trainer

        inactive = SeparationResult(
            loss=LossComponent(
                name="separation",
                scaled_loss=zero_scalar,
                raw_value=zero_scalar,
                active=False,
                metrics={},
            ),
            pairwise_measurement=measurement_from_pair_tensor(
                None,
                active_fraction=0.0,
                valid_groups=0,
                reason="missing_pair_jsd",
            ),
            diagnostics={},
            train_active=0.0,
            raw_stats={"jsd": zero_scalar, "active": zero_scalar},
        )
        if float(separation_coef) <= 0.0:
            return replace(
                inactive,
                pairwise_measurement=measurement_from_pair_tensor(
                    None,
                    active_fraction=0.0,
                    valid_groups=0,
                    reason="separation_disabled",
                ),
            )
        if not counterfactual_active:
            return replace(
                inactive,
                pairwise_measurement=measurement_from_pair_tensor(
                    None,
                    active_fraction=0.0,
                    valid_groups=0,
                    reason="phase_counterfactual_inactive",
                ),
            )
        if (
            self.hparams.fixed_latent_strategy
            or z_idx is None
        ):
            return inactive

        cf_margin = float(
            getattr(self.cfg, "latent_cf_jsd_margin", 0.01)
            or getattr(self.hparams, "latent_actor_z_separation_margin", 0.02)
            or 0.01
        )
        if is_v6i1_staged_trainer(self.runtime):
            competence, competence_ready = self.latent_state.compute_competence_scores()
            z_sep_loss, z_sep_stats = v6i1_cf_separation_loss(
                self.model,
                obs_batch,
                latent_k=int(self.hparams.latent_k),
                margin=cf_margin,
                competence=competence,
                competence_ready=bool(competence_ready),
                subsample_generator=self.subsample_generator,
            )
        else:
            max_action_entropy = float(self.model.n_agents) * sum(
                math.log(max(1, int(dim))) for dim in self.model.per_agent_action_dims
            )
            separation_gate = z_separation_gate_mask(
                advantages=advantages,
                action_entropy=entropy,
                global_state=batch["global_state"],
                max_action_entropy=max_action_entropy,
                min_abs_advantage=float(
                    getattr(self.hparams, "latent_actor_z_separation_min_abs_advantage", 0.0) or 0.0
                ),
                min_decision_frac=float(
                    getattr(self.hparams, "latent_actor_z_separation_min_decision_frac", 0.0) or 0.0
                ),
                max_entropy_frac=float(
                    getattr(self.hparams, "latent_actor_z_separation_max_entropy_frac", 1.0)
                    if getattr(self.hparams, "latent_actor_z_separation_max_entropy_frac", 1.0)
                    is not None
                    else 1.0
                ),
            )
            z_sep_loss, z_sep_stats = policy_z_separation_loss(
                self.model,
                obs_batch,
                z_idx.long(),
                latent_k=int(self.hparams.latent_k),
                margin=cf_margin,
                active_mask=separation_gate,
                subsample_generator=self.subsample_generator,
            )

        scaled = float(separation_coef) * z_sep_loss
        pair_jsd = z_sep_stats.get("pair_jsd")
        active_frac = float(z_sep_stats["active"].detach().cpu().item())
        valid_groups = z_sep_stats.get("cf_valid_team_groups", zero_scalar)
        valid_groups_int = (
            int(valid_groups.detach().cpu().item())
            if isinstance(valid_groups, torch.Tensor)
            else int(valid_groups or 0)
        )
        missing_reason = None
        if pair_jsd is None:
            missing_reason = "no_active_rows" if active_frac <= 0.0 else "missing_pair_jsd"
        measurement = measurement_from_pair_tensor(
            pair_jsd,
            active_fraction=active_frac,
            valid_groups=valid_groups_int,
            reason=missing_reason,
        )
        return SeparationResult(
            loss=LossComponent(
                name="separation",
                scaled_loss=scaled,
                raw_value=z_sep_loss.detach(),
                active=True,
                metrics={},
            ),
            pairwise_measurement=measurement,
            diagnostics={},
            train_active=1.0,
            raw_stats=z_sep_stats,
        )
