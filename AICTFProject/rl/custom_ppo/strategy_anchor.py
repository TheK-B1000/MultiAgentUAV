"""SAPPO V1 — strategy-anchor loss over the EXISTING training-side distribution.

Frozen protocol: artifacts/strategic_demand/STRATEGY_ANCHORED_PPO_V1_FROZEN.json

    L_actor = L_PPO + lambda_anchor * L_anchor
    L_anchor = -log pi_theta(a_teacher | o)

The loss was frozen before implementation, so the interface adapts to the loss
rather than the loss being reshaped around whatever API was convenient.

Why this file adds no new distribution machinery
------------------------------------------------
``SharedActorCentralizedCritic.get_distribution()`` is already a public contract
method returning the same ``MultiHeadActionDistribution`` the PPO actor update
uses. This module only evaluates that distribution at a known action. No
sampling, no argmax, no inference wrapper, and no new factorization: the head
structure is whatever ``action_space.nvec`` already defines.

Head ordering is agent-major, mirroring ``action_dims = action_space.nvec`` and
``heads_per_agent = len(action_dims) // n_agents``. A demonstration action of
shape ``(batch, n_agents, heads_per_agent)`` therefore flattens to head order
directly -- the same flattening used when the demonstrations were generated, so
labels cannot silently transpose.

Decision-point masking
----------------------
The engine latches a macro only when ``blue_commit_ticks_left <= 0``. Anchor
supervision applies only to agents that actually had a choice on that step;
locked agents contribute no gradient.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

__all__ = ["action_log_prob", "anchor_loss", "teacher_agreement"]


def _masked_heads(model, obs, *, z_idx=None):
    """Action heads with the SAME legality masking PPO's own update applies.

    This is load-bearing. ``evaluate_actions()`` -- the method the PPO actor
    update calls -- does::

        logits = self._mask_logits(self.policy_logits(obs, z_idx), obs["mask"])

    ``get_distribution()`` does NOT mask. Scoring the teacher action against
    unmasked logits would optimise a different distribution from the one PPO's
    surrogate uses, silently breaking the premise that SAPPO rehearses through
    the policy PPO already trains. Caught by the live compatibility check:
    predict() and get_distribution() agreed on only 43% / 66% of argmax actions.

    Falls back to the unmasked distribution only when the model exposes no
    ``_mask_logits`` or the observation carries no mask, so stub models in unit
    tests still work.
    """
    mask_fn = getattr(model, "_mask_logits", None)
    logits_fn = getattr(model, "policy_logits", None)
    obs_mask = obs.get("mask") if isinstance(obs, dict) else None
    if mask_fn is None or logits_fn is None or obs_mask is None:
        return model.get_distribution(obs, z_idx=z_idx).heads
    from rl.custom_ppo.distributions import ActionHead
    flat = mask_fn(logits_fn(obs, z_idx=z_idx), obs_mask)
    return [ActionHead(h) for h in torch.split(flat, list(model.action_dims), dim=-1)]


def action_log_prob(
    model,
    obs: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    *,
    z_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """log pi_theta(a | o), summed over action heads.

    Parameters
    ----------
    model : SharedActorCentralizedCritic
        The TRAINING-side model, not an inference wrapper.
    obs : dict of tensors
        Observation batch as the policy consumes it.
    actions : LongTensor
        ``(batch, n_agents, heads_per_agent)`` or ``(batch, n_heads)``.
    z_idx : optional LongTensor
        Required when the model uses latent strategy; passed straight through
        so this helper never invents a silent default.

    Returns
    -------
    Tensor ``(batch, n_heads)`` of per-head log-probabilities. The caller
    decides how to reduce, because decision-point masking is per agent-head.
    """
    heads = _masked_heads(model, obs, z_idx=z_idx)
    a = actions.reshape(actions.shape[0], -1).long()
    if a.shape[1] != len(heads):
        raise ValueError(
            f"action encoding mismatch: got {a.shape[1]} action columns for "
            f"{len(heads)} policy heads. Demonstration actions must be "
            f"agent-major and match action_space.nvec exactly; no remapping is "
            f"performed here.")
    out = []
    for h, head in enumerate(heads):
        idx = a[:, h]
        if int(idx.max()) >= head.action_dim or int(idx.min()) < 0:
            raise ValueError(
                f"head {h}: action index out of range [0, {head.action_dim}).")
        logp = head.logits.log_softmax(dim=-1)
        out.append(logp.gather(1, idx.unsqueeze(1)).squeeze(1))
    return torch.stack(out, dim=1)


def anchor_loss(
    model,
    obs: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    decision_mask: Optional[torch.Tensor] = None,
    *,
    z_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean negative log-probability of the teacher action.

    ``decision_mask`` is ``(batch, n_agents)`` and True where that agent had a
    macro decision available. It is broadcast across that agent's heads, so a
    locked agent contributes nothing to the loss.
    """
    per_head = action_log_prob(model, obs, actions, z_idx=z_idx)
    if decision_mask is None:
        return -per_head.mean()

    n_heads = per_head.shape[1]
    n_agents = decision_mask.shape[1]
    if n_heads % n_agents:
        raise ValueError(f"{n_heads} heads do not divide across {n_agents} agents")
    per_agent = n_heads // n_agents
    m = decision_mask.to(per_head.dtype).repeat_interleave(per_agent, dim=1)
    denom = m.sum().clamp_min(1.0)
    return -(per_head * m).sum() / denom


@torch.no_grad()
def teacher_agreement(
    model,
    obs: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    decision_mask: Optional[torch.Tensor] = None,
    *,
    z_idx: Optional[torch.Tensor] = None,
) -> float:
    """Fraction of decision-point heads whose argmax equals the teacher action.

    Diagnostic only. Reported for the train split and the held-out 10% split so
    memorisation is visible, but it is NOT a gate.
    """
    heads = _masked_heads(model, obs, z_idx=z_idx)
    a = actions.reshape(actions.shape[0], -1).long()
    hits = torch.stack([(h.argmax_actions == a[:, i]).float()
                        for i, h in enumerate(heads)], dim=1)
    if decision_mask is None:
        return float(hits.mean())
    n_heads, n_agents = hits.shape[1], decision_mask.shape[1]
    m = decision_mask.to(hits.dtype).repeat_interleave(n_heads // n_agents, dim=1)
    return float((hits * m).sum() / m.sum().clamp_min(1.0))


class AnchorRunner:
    """Interleaved teacher rehearsal — SAPPO V1, Reading 2.

    Frozen semantics (SAPPO_V1_LOSS_SEMANTICS_AMENDMENT.json):

        4 PPO actor minibatches execute unchanged
        -> ZERO GRADS
        -> one anchor-only optimizer step, L = lambda * NLL(teacher action)
        -> ZERO GRADS

    The anchor never shares a backward pass with the PPO surrogate, value,
    entropy, latent, separation, communication, or strategy objectives.

    Stale-gradient discipline
    -------------------------
    The PPO stepper zeroes gradients BEFORE its backward and leaves them
    populated after ``step()``. So gradients from PPO minibatch 4 are still
    resident when rehearsal begins. Without an explicit zero here they would
    ride along into the anchor optimizer step: every counter would still read
    0.25 and the "NLL-only rehearsal" claim would be quietly false. Hence
    zero_grad both BEFORE the anchor backward and AFTER the anchor step.

    Disabled means structurally absent: construct no runner at all. This class
    is never instantiated with lambda=0 as a way of "turning it off", because a
    nominal zero-loss step can still mutate optimizer state, advance counters,
    or apply weight decay.
    """

    def __init__(self, model, optimizer, dataset, *, lambda_anchor: float,
                 cadence: int = 4, max_grad_norm: float | None = None,
                 device: str = "cpu"):
        if lambda_anchor <= 0.0:
            raise ValueError(
                "AnchorRunner must not be constructed with lambda_anchor <= 0. "
                "Disabled anchoring means NOT constructing the runner, so that "
                "no optimizer or scheduler state can be mutated.")
        if cadence < 1:
            raise ValueError("cadence must be >= 1")
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.lambda_anchor = float(lambda_anchor)
        self.cadence = int(cadence)
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.n_ppo_actor_minibatches = 0
        self.n_anchor_updates = 0
        self.last_anchor_loss = float("nan")

    def note_ppo_minibatch(self) -> bool:
        """Call once per completed PPO actor minibatch.

        Returns True iff this call completed a full group and an anchor step
        was performed. No trailing anchor update is ever emitted for a partial
        group -- the ratio is never forced.
        """
        self.n_ppo_actor_minibatches += 1
        if self.n_ppo_actor_minibatches % self.cadence != 0:
            return False
        self._anchor_step()
        return True

    def _anchor_step(self) -> None:
        import torch as _t

        obs, actions, mask = self.dataset.sample(device=self.device)
        # Clear gradients left over from PPO minibatch `cadence`.
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.lambda_anchor * anchor_loss(self.model, obs, actions, mask)
        loss.backward()
        if self.max_grad_norm is not None:
            _t.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]],
                float(self.max_grad_norm))
        self.optimizer.step()
        # Leave no anchor gradients behind for the next PPO minibatch.
        self.optimizer.zero_grad(set_to_none=True)
        self.n_anchor_updates += 1
        self.last_anchor_loss = float(loss.detach())

    def telemetry(self) -> dict:
        """Counters for the frozen cadence check. Measured, never assumed."""
        expected = self.n_ppo_actor_minibatches // self.cadence
        return {
            "anchor_lambda": self.lambda_anchor,
            "anchor_cadence": self.cadence,
            "n_ppo_actor_minibatches": self.n_ppo_actor_minibatches,
            "n_anchor_updates": self.n_anchor_updates,
            "expected_complete_groups": expected,
            "anchor_per_ppo_ratio": (self.n_anchor_updates / self.n_ppo_actor_minibatches
                                     if self.n_ppo_actor_minibatches else 0.0),
            "complete_group_ratio_is_one": (self.n_anchor_updates == expected),
            "last_anchor_loss": self.last_anchor_loss,
        }


class AnchorDataset:
    """Minibatch sampler over a frozen teacher-demonstration file.

    Samples ONLY rows that contain at least one decision point, because rows
    with no available macro decision contribute nothing to the loss and would
    otherwise dilute batches.
    """

    def __init__(self, npz_path, *, batch_size: int = 64, seed: int = 7):
        import numpy as _np
        d = _np.load(str(npz_path))
        self.path = str(npz_path)
        self.run_id = str(d["run_id"][0]) if "run_id" in d.files else None
        self._obs = {k[4:]: d[k] for k in d.files if k.startswith("obs_")}
        self._act = d["actions"]
        self._mask = d["decision_mask"]
        keep = self._mask.any(axis=1)
        if not keep.any():
            raise ValueError(f"{npz_path}: no rows contain a decision point")
        self._idx = keep.nonzero()[0]
        self.batch_size = int(batch_size)
        self._rng = _np.random.default_rng(int(seed))
        self.n_rows = int(self._act.shape[0])
        self.n_usable_rows = int(self._idx.size)

    def sample(self, device: str = "cpu"):
        import torch as _t
        pick = self._rng.choice(self._idx, size=min(self.batch_size, self._idx.size),
                                replace=False)
        obs = {k: _t.from_numpy(v[pick]).to(device) for k, v in self._obs.items()}
        act = _t.from_numpy(self._act[pick]).long().to(device)
        msk = _t.from_numpy(self._mask[pick]).bool().to(device)
        return obs, act, msk

    def describe(self) -> dict:
        return {"path": self.path, "run_id": self.run_id,
                "rows": self.n_rows, "usable_rows_with_decision": self.n_usable_rows,
                "batch_size": self.batch_size}
