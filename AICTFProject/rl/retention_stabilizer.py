"""EMA-teacher retention stabilization for RSCFT.

Implements RSCFT_SPEC.json#EMA_RETENTION:

    theta_bar_0 = theta_0                                  (exact copy of the warm start)
    theta_bar  <- 0.995 * theta_bar + 0.005 * theta         (after each actor update)
    L_ret       = KL( pi_theta_bar || pi_theta )            at decision boundaries only
    lambda_ret  = 0.01

Three things this module is built to guarantee, each of which corresponds to a failure this
program has already paid for once:

1. THE SAME MASKED DISTRIBUTION PPO TRAINS. The KL is computed through
   strategy_anchor._masked_heads, which applies the environment's legality mask exactly as
   evaluate_actions() does. get_distribution() does NOT mask; regularising an unmasked
   distribution would penalise drift in a distribution the actor never optimises.

2. DECISION BOUNDARIES ONLY. A committed agent's logits are not a decision. Retention is
   masked by decision_eligible_agents(), the same predicate EXP2 uses, which reads commitment
   off the legality mask (a locked agent's macro head is one-hot at the latched macro).
   Note this is belt-and-braces: because BOTH teacher and student see the identical one-hot
   mask at a committed agent, that agent's KL is structurally zero anyway -- but relying on
   an emergent property for a load-bearing predicate is exactly the class of mistake
   CCP_FORCED_HEAD_DILUTION.json records, so the mask is applied explicitly and the zero is
   verified empirically in preflight.

3. NO SILENT inf/NaN. The policy is effectively deterministic at logged precision
   (CCP_S2_PRELAUNCH_INTERPRETATION_CAVEATS.json) and the existing telemetry KL returns inf
   under exactly that condition. Here the KL is built from log_softmax and the
   0 * (-inf - -inf) terms that arise at masked actions are zeroed explicitly, then the
   result is checked finite and RAISES rather than propagating a poisoned gradient.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

import torch


class RetentionError(RuntimeError):
    """Retention produced a non-finite value, or was asked for something incoherent."""


class EMATeacher:
    """A frozen exponential-moving-average copy of the actor.

    Receives no optimizer gradients ever: every parameter is requires_grad_(False) and the
    update runs under no_grad. It is a target, not a trained object.
    """

    def __init__(self, model: Any, *, decay: float):
        if not (0.0 < decay < 1.0):
            raise RetentionError(f"decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.n_updates = 0

    @torch.no_grad()
    def update(self, student: Any) -> None:
        """theta_bar <- decay * theta_bar + (1 - decay) * theta.

        Every parameter is averaged, including critic-side ones that the retention loss never
        reads -- averaging them costs a little arithmetic and avoids a fragile "which tensor
        is an actor parameter" heuristic. Buffers are copied outright so the teacher's
        non-learned state cannot drift away from the student's.
        """
        d = self.decay
        for p_bar, p in zip(self.model.parameters(), student.parameters()):
            p_bar.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for b_bar, b in zip(self.model.buffers(), student.buffers()):
            b_bar.copy_(b.detach())
        self.n_updates += 1

    @torch.no_grad()
    def max_abs_param_delta(self, student: Any) -> float:
        """Largest |theta_bar - theta| over all parameters. Zero exactly at initialization."""
        worst = 0.0
        for p_bar, p in zip(self.model.parameters(), student.parameters()):
            worst = max(worst, float((p_bar - p.detach()).abs().max()))
        return worst


def retention_kl(
    student: Any,
    teacher: Any,
    obs: Mapping[str, torch.Tensor],
    *,
    z_idx: torch.Tensor,
    decision_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Mean KL( pi_teacher || pi_student ) over decision-eligible agent heads.

    Parameters
    ----------
    decision_mask : optional (batch, n_agents) bool
        Where a new macro choice is available. When None it is derived from obs["mask"] by
        decision_eligible_agents -- the same predicate EXP2 uses.

    Returns (loss, diagnostics). Raises RetentionError on any non-finite value rather than
    letting it reach .backward().
    """
    from rl.custom_ppo.exp2_teacher_compression import decision_eligible_agents
    from rl.custom_ppo.strategy_anchor import _masked_heads

    heads_s = _masked_heads(student, obs, z_idx=z_idx)
    with torch.no_grad():
        heads_t = _masked_heads(teacher, obs, z_idx=z_idx)
    if len(heads_s) != len(heads_t):
        raise RetentionError(f"head count mismatch: student {len(heads_s)}, teacher {len(heads_t)}")

    n_agents = int(getattr(student, "n_agents"))
    n_heads = len(heads_s)
    if n_heads % n_agents:
        raise RetentionError(f"{n_heads} heads do not divide across {n_agents} agents")
    per_agent = n_heads // n_agents

    if decision_mask is None:
        decision_mask = decision_eligible_agents(
            obs["mask"], action_dims=tuple(int(v) for v in student.action_dims),
            n_agents=n_agents, agent_mask=obs.get("agent_mask"))
    if decision_mask.shape[1] != n_agents:
        raise RetentionError(
            f"decision_mask has {decision_mask.shape[1]} agents, expected {n_agents}")

    # (batch, n_heads): each agent's mask repeated across that agent's heads
    m = decision_mask.to(torch.float32).repeat_interleave(per_agent, dim=1)

    per_head_kl = []
    for h, (hs, ht) in enumerate(zip(heads_s, heads_t)):
        logp_s = hs.logits.log_softmax(dim=-1)
        logp_t = ht.logits.log_softmax(dim=-1)
        p_t = logp_t.exp()
        # At a masked (illegal) action both log-probs are -inf, so p_t * (logp_t - logp_s)
        # evaluates to 0 * nan. Those terms are exactly zero in the KL and are zeroed
        # explicitly rather than being allowed to poison the sum. This is also what makes a
        # committed agent -- one legal action, identical one-hot in both -- contribute
        # exactly 0.0 instead of nan.
        terms = p_t * (logp_t - logp_s)
        terms = torch.where(p_t > 0, terms, torch.zeros_like(terms))
        per_head_kl.append(terms.sum(dim=-1))
    kl = torch.stack(per_head_kl, dim=1)                      # (batch, n_heads)

    if not bool(torch.isfinite(kl).all()):
        raise RetentionError(
            "retention KL produced a non-finite value. This is fail-closed by design: the "
            "policy is effectively deterministic, and a silent inf/NaN here would either "
            "kill the run's gradients or, worse, train on a poisoned objective.")

    denom = m.sum()
    if float(denom) <= 0.0:
        # A legitimate batch in which no agent is at a decision boundary. No pressure and no
        # division by zero; returned as a real zero tensor so .backward() stays safe.
        loss = (kl * 0.0).sum()
        return loss, {"kl_mean": 0.0, "eligible_heads": 0, "rows": int(kl.shape[0]),
                      "empty_batch": True}

    loss = (kl * m).sum() / denom
    if not bool(torch.isfinite(loss)):
        raise RetentionError("retention loss is non-finite after masking")
    return loss, {"kl_mean": float(loss.detach()), "eligible_heads": int(denom),
                  "rows": int(kl.shape[0]), "empty_batch": False}


class RetentionRunner:
    """Applies lambda_ret * L_ret on the on-policy minibatch, then advances the EMA teacher.

    Mirrors the separate-operation shape every auxiliary objective in this program uses
    (CausalSequenceRunner, the SPPPO ranking runner): its own zero_grad/backward/step on the
    SAME optimizer as task PPO, never sharing a backward pass with the PPO surrogate. That is
    an approximation of the literal sum L_PPO + 0.05 L_causal + 0.01 L_ret, and it is the
    same approximation this program already validated for L_causal -- kept identical so the
    only thing differing between the RSCFT arms is retention itself.

    The EMA update runs after the retention step, i.e. after an actor optimizer update, as
    RSCFT_SPEC.json#EMA_RETENTION specifies.
    """

    def __init__(self, trainer, *, lam: float, decay: float, cadence: int = 1):
        if lam <= 0.0:
            raise RetentionError("lambda_ret must be > 0; absent retention means not attaching")
        if cadence < 1:
            raise RetentionError("cadence must be >= 1")
        self.trainer = trainer
        self.lam = float(lam)
        self.cadence = int(cadence)
        self.teacher = EMATeacher(trainer.model, decay=decay)
        self.n_ppo_actor_minibatches = 0
        self.n_retention_updates = 0
        self.n_empty_batches = 0
        self.last_loss = float("nan")
        self.last_diag: dict = {}

    def note_ppo_minibatch(self, batch: Mapping[str, torch.Tensor]) -> bool:
        from rl.custom_ppo.exp2_teacher_compression import _obs_from_batch

        self.n_ppo_actor_minibatches += 1
        if self.n_ppo_actor_minibatches % self.cadence:
            return False

        obs = _obs_from_batch(batch)
        z_idx = batch["z"].long()
        loss, diag = retention_kl(self.trainer.model, self.teacher.model, obs, z_idx=z_idx)
        scaled = self.lam * loss

        opt = self.trainer.optimizer
        opt.zero_grad(set_to_none=True)
        scaled.backward()
        opt.step()

        # after the actor optimizer update, per the frozen rule
        self.teacher.update(self.trainer.model)

        self.n_retention_updates += 1
        self.n_empty_batches += int(bool(diag.get("empty_batch")))
        self.last_loss = float(scaled.detach())
        self.last_diag = diag
        return True

    def telemetry(self) -> dict:
        return {
            "retention_updates": self.n_retention_updates,
            "n_ppo_actor_minibatches": self.n_ppo_actor_minibatches,
            "ema_updates": self.teacher.n_updates,
            "last_loss": self.last_loss,
            "last_kl_mean": self.last_diag.get("kl_mean", float("nan")),
            "last_eligible_heads": self.last_diag.get("eligible_heads", 0),
            "empty_batches": self.n_empty_batches,
            "lambda_ret": self.lam,
            "ema_decay": self.teacher.decay,
        }
