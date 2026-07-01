"""V6I7 recurrent router update via truncated BPTT over sequence minibatches.

This module runs AFTER the standard actor/critic PPO minibatch loop each epoch.
It re-runs the GRU forward pass over contiguous chunks (burn_in + seq_len steps)
with gradients flowing through all GRU transitions, computes the router PPO loss
and conditional entropy only at actual decision steps (``router_decision_valid``),
and steps the optimizer.

The actor/critic parameters are NOT updated here — only the GRU (selector_gru)
and q_phi encoder (strategy_encoder) receive gradients from this loop.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from rl.custom_ppo.update.sequence_minibatch import iter_router_sequence_minibatches
from rl.ppo_core import TensorDictRolloutBuffer


class RouterSequenceUpdater:
    """Per-epoch BPTT update for the V6I7 recurrent router.

    Constructed once per ``PPOUpdater.update()`` call; ``update_epoch``
    is called once per training epoch, after the actor/critic loop.
    """

    def __init__(
        self,
        *,
        model: Any,
        cfg: Any,
        hparams: Any,
        optimizer: Any,
        device: Any,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.hparams = hparams
        self.optimizer = optimizer
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_active(self, buffer: TensorDictRolloutBuffer) -> bool:
        """Return True iff this buffer contains V6I7 router fields."""
        return (
            bool(getattr(self.hparams, "use_latent_strategy", False))
            and bool(getattr(self.model, "selector_gru", None) is not None)
            and "router_decision_valid" in buffer.fields
            and "selector_hidden" in buffer.fields
        )

    def update_epoch(
        self,
        buffer: TensorDictRolloutBuffer,
        *,
        ent_coef: float,
    ) -> dict[str, float]:
        """Run one BPTT epoch over all sequence chunks in the buffer.

        Returns a flat dict of scalar metrics to be forwarded to the
        ``UpdateStatsAccumulator``.
        """
        assignment_mode = str(
            getattr(self.cfg, "latent_assignment_mode", "router") or "router"
        )
        train_when_forced = bool(getattr(self.cfg, "train_router_when_forced", False))
        if assignment_mode != "router" and not train_when_forced:
            return {"router_skipped_forced_mode": 1.0}

        burn_in = int(getattr(self.cfg, "recurrent_burn_in", 8) or 8)
        seq_len = int(getattr(self.cfg, "recurrent_seq_len", 32) or 32)
        chunks_per_batch = max(1, int(getattr(self.cfg, "router_chunks_per_batch", 4) or 4))
        clip_range = float(getattr(self.hparams, "clip_range", 0.2) or 0.2)
        router_ppo_coef = float(
            getattr(self.hparams, "latent_strategy_ppo_coef", 0.10) or 0.10
        )
        lam_persist = float(getattr(self.hparams, "latent_lam_p", 0.02) or 0.02)
        router_ent_coef = float(getattr(self.cfg, "router_ent_coef", 0.005) or 0.005)

        total_ppo_loss = 0.0
        total_ent_loss = 0.0
        total_persist_loss = 0.0
        total_decision_count = 0
        n_batches = 0

        for chunk in iter_router_sequence_minibatches(
            buffer,
            burn_in=burn_in,
            seq_len=seq_len,
            chunks_per_batch=chunks_per_batch,
            required_fields=None,
            shuffle=True,
        ):
            loss, stats = self._compute_chunk_loss(
                chunk,
                burn_in=burn_in,
                clip_range=clip_range,
                router_ppo_coef=router_ppo_coef,
                lam_persist=lam_persist,
                router_ent_coef=router_ent_coef,
            )
            if loss is None:
                continue

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._router_params(), max_norm=0.5
            )
            self.optimizer.step()

            total_ppo_loss += stats.get("router_ppo_loss", 0.0)
            total_ent_loss += stats.get("router_ent_loss", 0.0)
            total_persist_loss += stats.get("router_persist_loss", 0.0)
            total_decision_count += stats.get("router_decision_count", 0)
            n_batches += 1

        if n_batches == 0:
            return {}
        return {
            "router_bptt_ppo_loss": total_ppo_loss / n_batches,
            "router_bptt_ent_loss": total_ent_loss / n_batches,
            "router_bptt_persist_loss": total_persist_loss / n_batches,
            "router_bptt_decision_count": float(total_decision_count),
            "router_bptt_batches": float(n_batches),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _router_params(self):
        """Yield parameters for the GRU and q_phi encoder only."""
        gru = getattr(self.model, "selector_gru", None)
        enc = getattr(self.model, "strategy_encoder", None)
        params = []
        if gru is not None:
            params.extend(gru.parameters())
        if enc is not None:
            params.extend(enc.parameters())
        return params

    def _compute_chunk_loss(
        self,
        chunk: dict[str, torch.Tensor],
        *,
        burn_in: int,
        clip_range: float,
        router_ppo_coef: float,
        lam_persist: float,
        router_ent_coef: float,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        """Compute combined router loss for one sequence batch.

        ``chunk`` values have shape ``[chunk_total, B_seq, *]`` with
        ``chunk_total = burn_in + seq_len``.  Only the loss-window
        (``[burn_in:]``) contributes to gradients.
        """
        device = self.device

        gs = chunk["global_state"].to(device)                   # (T, B, state_dim)
        h_start = chunk["selector_hidden_start"].to(device)     # (B, hidden_dim)
        rdv = chunk["router_decision_valid"].to(device).bool()  # (T, B)
        z_stored = chunk["z"].to(device).long()                 # (T, B)
        z_log_prob_old = chunk["z_log_probs"].to(device).float() # (T, B)

        # Use router_advantages (V6I7 opportunity-level GAE), fall back to option_advantages, then advantages.
        if "router_advantages" in chunk:
            adv = chunk["router_advantages"].to(device).float()
        elif "option_advantages" in chunk:
            adv = chunk["option_advantages"].to(device).float()
        else:
            adv = chunk["advantages"].to(device).float()

        terminated = chunk.get("terminated")
        truncated = chunk.get("truncated")
        done_mask = torch.zeros(gs.shape[:2], dtype=torch.float32, device=device)
        if terminated is not None:
            done_mask = done_mask + terminated.to(device).float()
        if truncated is not None:
            done_mask = done_mask + truncated.to(device).float()
        done_mask = done_mask.clamp(0.0, 1.0).bool()

        # BPTT forward — h_start detached so no gradients flow into prior rollout.
        logits_seq, _ = self.model.forward_router_sequence(
            gs, h_start.detach(), done_mask
        )
        # logits_seq: (T, B, K)

        # Only compute loss on loss window (after burn-in).
        logits_loss = logits_seq[burn_in:]      # (seq_len, B, K)
        rdv_loss = rdv[burn_in:]                # (seq_len, B)
        z_loss = z_stored[burn_in:]             # (seq_len, B)
        z_log_prob_old_loss = z_log_prob_old[burn_in:]  # (seq_len, B)
        adv_loss = adv[burn_in:]                # (seq_len, B)

        tau = float(getattr(self.model, "strategy_tau", 1.0) or 1.0)
        dist = Categorical(logits=logits_loss / tau)
        z_log_prob_new = dist.log_prob(z_loss)  # (seq_len, B)

        decision_mask = rdv_loss  # (seq_len, B)
        n_decisions = int(decision_mask.sum().item())
        if n_decisions == 0:
            return None, {}

        # --- Router PPO (clipped) at decision steps only ---
        log_ratio = z_log_prob_new[decision_mask] - z_log_prob_old_loss[decision_mask].detach()
        ratio = log_ratio.exp()
        adv_sel = adv_loss[decision_mask].detach()
        if adv_sel.numel() > 1:
            adv_sel = (adv_sel - adv_sel.mean()) / (adv_sel.std(unbiased=False) + 1e-8)
        surr1 = ratio * adv_sel
        surr2 = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * adv_sel
        ppo_loss = -torch.min(surr1, surr2).mean()

        # --- Conditional entropy at decision steps only ---
        ent_loss = -dist.entropy()[decision_mask].mean()

        # --- Persistence loss at consecutive decision pairs ---
        persist_loss = self._persistence_loss(logits_loss, tau, decision_mask)

        total_loss = (
            router_ppo_coef * ppo_loss
            + router_ent_coef * ent_loss
            + lam_persist * persist_loss
        )
        stats = {
            "router_ppo_loss": float(ppo_loss.detach().cpu().item()),
            "router_ent_loss": float(ent_loss.detach().cpu().item()),
            "router_persist_loss": float(persist_loss.detach().cpu().item()),
            "router_decision_count": float(n_decisions),
        }
        return total_loss, stats

    @staticmethod
    def _persistence_loss(
        logits: torch.Tensor,
        tau: float,
        decision_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean ``1 - p(z_{j+1} = z_j)`` over consecutive decision pairs.

        Uses soft expectation (not sampled z) so no samples needed.
        Only pairs where BOTH steps are decision steps contribute.
        """
        # logits: (seq_len, B, K),  decision_mask: (seq_len, B)
        seq_len, B, K = logits.shape
        device = logits.device

        # Consecutive pair validity: both t and t+1 must be decision steps.
        pair_valid = decision_mask[:-1] & decision_mask[1:]  # (seq_len-1, B)
        if not pair_valid.any():
            return torch.zeros(1, device=device).squeeze()

        probs_prev = F.softmax(logits[:-1] / tau, dim=-1)[pair_valid]   # (N, K)
        probs_next = F.softmax(logits[1:] / tau, dim=-1)[pair_valid]    # (N, K)
        # E[z_{j+1} = z_j] = sum_z p_prev(z) * p_next(z)
        consistency = (probs_prev * probs_next.detach()).sum(dim=-1)     # (N,)
        return (1.0 - consistency).mean()
