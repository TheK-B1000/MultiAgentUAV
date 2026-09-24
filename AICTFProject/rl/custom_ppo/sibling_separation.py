"""Disagreement-masked sibling separation for 4v4 specialization-preserving PPO.

Frozen protocol:
  artifacts/strategic_demand/sppo/4V4_B3_SPECIALIZATION_PRESERVING_SPEC.json
  (+ 4V4_B3_SPECIALIZATION_PRESERVING_DIVERGENCE_AMENDMENT.json)

    L_i = L_PPO,i + lambda * L_sep,i
    L_sep = -JSD( pi_i(.|o) , stopgrad[ pi_sib(.|o) ] )

The divergence is the in-repo ``rl.teacher_distillation.jsd_per_head`` already
used by the distillation fidelity path: symmetric, bounded by ln 2, and computed
on the SAME legality-masked heads PPO's own update evaluates
(``strategy_anchor._masked_heads``). Bounded divergence is why no ad-hoc clip is
needed -- a hand-rolled unbounded KL would have required one.

Applied only on a frozen disagreement-support dataset (states where the scripted
GUARD and BREACH references disagree) and only on decision-eligible agent-heads.
Separate zero_grad / backward / step on a PPO cadence, mirroring SAPPO -- never
shares a backward pass with the PPO surrogate.

Disabled means structurally absent: construct no runner when lambda<=0 or paths
are empty.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

__all__ = [
    "DisagreementDataset",
    "SiblingSepRunner",
    "sibling_separation_loss",
]

#: Bound of the Jensen-Shannon divergence in nats; JSD in [0, ln 2].
JSD_MAX_NATS = float(np.log(2.0))


def sibling_separation_loss(
    student: Any,
    sibling: Any,
    obs: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return ``(-mean JSD, telemetry)`` over decision-eligible agent-heads.

    Gradient reaches the student only: the sibling's heads are evaluated under
    ``no_grad`` so the reference is a stationary stop-grad target.
    """
    from rl.custom_ppo.exp2_teacher_compression import decision_eligible_agents
    from rl.teacher_distillation import head_logits, jsd_per_head, masked_mean

    student_logits = head_logits(student, obs)
    with torch.no_grad():
        sibling_logits = head_logits(sibling, obs)

    per_head = jsd_per_head(student_logits, sibling_logits)
    decision = decision_eligible_agents(
        obs["mask"],
        action_dims=tuple(int(v) for v in student.action_dims),
        n_agents=int(student.n_agents),
        agent_mask=obs.get("agent_mask"),
    )
    jsd, n_heads = masked_mean(per_head, decision)
    telemetry = {
        "jsd": float(jsd.detach()),
        "decision_heads": float(n_heads),
        "jsd_frac_of_max": float(jsd.detach()) / JSD_MAX_NATS,
    }
    return -jsd, telemetry


class DisagreementDataset:
    """Sampler over frozen GUARD≠BREACH disagreement observation rows."""

    def __init__(self, npz_path: str, *, batch_size: int = 64, seed: int = 7):
        d = np.load(str(npz_path), allow_pickle=False)
        self.path = str(npz_path)
        self.run_id = str(d["run_id"][0]) if "run_id" in d.files else None
        self._obs = {k[4:]: d[k] for k in d.files if k.startswith("obs_")}
        if not self._obs:
            raise ValueError(f"{npz_path}: no obs_* arrays")
        n = next(iter(self._obs.values())).shape[0]
        if n < 1:
            raise ValueError(f"{npz_path}: empty disagreement dataset")
        self.batch_size = int(batch_size)
        self._rng = np.random.default_rng(int(seed))
        self.n_rows = int(n)

    def sample(self, device: str = "cpu") -> dict[str, torch.Tensor]:
        n = min(self.batch_size, self.n_rows)
        pick = self._rng.choice(self.n_rows, size=n, replace=False)
        return {k: torch.from_numpy(v[pick]).to(device) for k, v in self._obs.items()}

    def describe(self) -> dict:
        return {
            "path": self.path,
            "run_id": self.run_id,
            "rows": self.n_rows,
            "batch_size": self.batch_size,
        }


class SiblingSepRunner:
    """Interleaved stop-grad sibling separation — separate optimizer step."""

    def __init__(
        self,
        student: Any,
        optimizer: Any,
        sibling: Any,
        dataset: DisagreementDataset,
        *,
        lambda_sep: float,
        cadence: int = 4,
        max_grad_norm: float | None = None,
        device: str = "cpu",
    ):
        if float(lambda_sep) <= 0.0:
            raise ValueError(
                "SiblingSepRunner must not be constructed with lambda_sep <= 0. "
                "Disabled means NOT constructing the runner."
            )
        if int(cadence) < 1:
            raise ValueError("cadence must be >= 1")
        self.student = student
        self.optimizer = optimizer
        self.sibling = sibling
        self.dataset = dataset
        self.lambda_sep = float(lambda_sep)
        self.cadence = int(cadence)
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.n_ppo_actor_minibatches = 0
        self.n_sep_updates = 0
        self.last_sep_loss = float("nan")
        self.last_jsd = float("nan")
        self.sibling.eval()
        for p in self.sibling.parameters():
            p.requires_grad_(False)

    def note_ppo_minibatch(self) -> bool:
        self.n_ppo_actor_minibatches += 1
        if self.n_ppo_actor_minibatches % self.cadence != 0:
            return False
        self._sep_step()
        return True

    def _sep_step(self) -> None:
        obs = self.dataset.sample(device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        raw_loss, telemetry = sibling_separation_loss(self.student, self.sibling, obs)
        self.last_jsd = telemetry["jsd"]
        loss = self.lambda_sep * raw_loss
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]],
                float(self.max_grad_norm),
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.n_sep_updates += 1
        self.last_sep_loss = float(loss.detach())

    def telemetry(self) -> dict[str, float]:
        return {
            "sibling_sep_lambda": float(self.lambda_sep),
            "sibling_sep_cadence": float(self.cadence),
            "sibling_sep_n_ppo_actor_updates": float(self.n_ppo_actor_minibatches),
            "sibling_sep_n_updates": float(self.n_sep_updates),
            "sibling_sep_to_ppo_ratio": float(
                self.n_sep_updates / max(1, self.n_ppo_actor_minibatches)
            ),
            "sibling_sep_loss": float(self.last_sep_loss),
            "sibling_sep_jsd": float(self.last_jsd),
        }
