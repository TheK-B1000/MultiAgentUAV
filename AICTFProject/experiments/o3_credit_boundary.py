"""O3 credit boundary: only post-handoff transitions enter PPO optimization.

Frozen by: artifacts/o3_preregistration/O3_OPTIMIZATION_BOUNDARY_FROZEN.json (0b18ed3)

WHY ONE SEAM SUFFICES
---------------------
``TensorDictRolloutBuffer.iter_minibatches`` flattens EVERY registered field
with the identical ``value[:length].reshape(total, ...)`` and then applies the
same ``index_select``. Registering ``o3_credit`` as a field therefore makes it
structurally impossible for the credit mask to desynchronize from observations,
actions, values or log-probs -- the ordering hazard is removed by construction
rather than by a matching convention.

Downstream, ``minibatch_updater.update(batch=...)`` derives all six consumers
from that one dict, including advantage normalization
(minibatch_updater.py:196). Filtering the dict before it is handed over is
therefore sufficient for:

    return/value targets, advantage normalization, actor loss,
    critic loss, entropy, clip-fraction / PPO statistics

TWO INDEPENDENT DEFENCES
------------------------
Structural: only credited rows are indexed into each consumer.
Arithmetic: prefix rows carry sentinels, so any path that reaches around the
index produces an obviously impossible result rather than a plausible one.

The five abort counts are computed from the rows ACTUALLY ENTERING the update,
not from the mask, because a correct mask can sit beside code that reaches
around it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

CREDIT_FIELD = "o3_credit"


@dataclass
class CreditAudit:
    """Counts derived from rows entering the update, not from the mask."""

    minibatches_seen: int = 0
    minibatches_dropped_empty: int = 0
    rows_offered: int = 0
    rows_credited: int = 0
    actor_input_prefix_count: int = 0
    critic_input_prefix_count: int = 0
    entropy_input_prefix_count: int = 0
    adv_norm_input_prefix_count: int = 0
    return_target_prefix_count: int = 0
    violations: list = field(default_factory=list)

    def assert_clean(self) -> None:
        counts = {
            "pre_handoff_actor_samples": self.actor_input_prefix_count,
            "pre_handoff_critic_samples": self.critic_input_prefix_count,
            "pre_handoff_entropy_samples": self.entropy_input_prefix_count,
            "pre_handoff_norm_samples": self.adv_norm_input_prefix_count,
            "pre_handoff_return_targets": self.return_target_prefix_count,
        }
        bad = {k: v for k, v in counts.items() if v != 0}
        if bad:
            raise AssertionError(
                f"O3 credit boundary violated: {bad}. Pre-handoff transitions "
                "reached PPO optimization; the optimization domain is not what "
                "O3_OPTIMIZATION_BOUNDARY_FROZEN.json specifies."
            )

    def to_dict(self) -> dict:
        offered = max(int(self.rows_offered), 1)
        return {
            "minibatches_seen": self.minibatches_seen,
            "minibatches_dropped_empty": self.minibatches_dropped_empty,
            "rows_offered": self.rows_offered,
            "rows_credited": self.rows_credited,
            "credited_row_fraction": round(self.rows_credited / offered, 4),
            "pre_handoff_actor_samples": self.actor_input_prefix_count,
            "pre_handoff_critic_samples": self.critic_input_prefix_count,
            "pre_handoff_entropy_samples": self.entropy_input_prefix_count,
            "pre_handoff_norm_samples": self.adv_norm_input_prefix_count,
            "pre_handoff_return_targets": self.return_target_prefix_count,
            "violations": list(self.violations),
        }


def filter_batch_to_credited(batch: dict, audit: CreditAudit) -> dict | None:
    """Drop every uncredited row. -> filtered batch, or None if nothing remains.

    The audit counts are taken from the FILTERED tensors, i.e. the rows that
    actually go on to the losses, so they measure the consumer rather than the
    mask.
    """
    credit = batch.get(CREDIT_FIELD)
    if credit is None:
        raise KeyError(
            f"{CREDIT_FIELD!r} missing from the minibatch. The credit boundary "
            "cannot be enforced without it; refusing to train."
        )
    mask = credit.reshape(-1).to(torch.bool)
    audit.minibatches_seen += 1
    audit.rows_offered += int(mask.numel())
    n_credited = int(mask.sum().item())
    audit.rows_credited += n_credited
    if n_credited == 0:
        audit.minibatches_dropped_empty += 1
        return None

    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    out = {}
    for name, value in batch.items():
        out[name] = value.index_select(0, idx.to(value.device)) if value.shape[:1] == mask.shape[:1] else value

    # Consumer-derived assertions: recount prefix rows in what we are handing on.
    kept = out.get(CREDIT_FIELD)
    if kept is not None:
        leaked = int((~kept.reshape(-1).to(torch.bool)).sum().item())
        audit.actor_input_prefix_count += leaked
        audit.critic_input_prefix_count += leaked
        audit.entropy_input_prefix_count += leaked
        audit.adv_norm_input_prefix_count += leaked
        audit.return_target_prefix_count += leaked
        if leaked:
            audit.violations.append(
                f"minibatch {audit.minibatches_seen}: {leaked} prefix rows survived filtering"
            )
    return out


def install_credit_boundary(trainer, handoff_state, *, strict: bool = True):
    """Register o3_credit, write it per step, and filter every minibatch.

    Returns (CreditAudit, uninstall).
    """
    audit = CreditAudit()
    collector = trainer.rollout_collector
    buffer_holder = {}

    real_collect = collector.collect

    def collect(*a, **kw):
        buf = real_collect(*a, **kw)
        buffer_holder["buffer"] = buf
        # Register and fill the credit field from the per-step record kept by
        # the handoff wrapper. Written AFTER collection so it cannot perturb
        # rollout behaviour.
        if CREDIT_FIELD not in buf.fields:
            buf.register_field(CREDIT_FIELD, dtype=torch.bool, deferred=True)
        recorded = handoff_state.credit_history
        T = int(buf.pos)
        credit = torch.zeros((T, buf.n_envs), dtype=torch.bool, device=buf.device)
        for t in range(min(T, len(recorded))):
            credit[t] = torch.as_tensor(recorded[t], dtype=torch.bool, device=buf.device)
        buf.fields[CREDIT_FIELD][:T] = credit
        handoff_state.credit_history = []
        return buf

    collector.collect = collect

    real_iter = None

    def patch_buffer_iter(buf):
        nonlocal real_iter
        if getattr(buf, "_o3_credit_patched", False):
            return
        real_iter = buf.iter_minibatches

        def iter_minibatches(batch_size, *, shuffle=True):
            for batch in real_iter(batch_size, shuffle=shuffle):
                filtered = filter_batch_to_credited(batch, audit)
                if filtered is None:
                    continue
                yield filtered

        buf.iter_minibatches = iter_minibatches
        buf._o3_credit_patched = True

    real_update = trainer.update

    def update(buffer, *, total_timesteps: int):
        patch_buffer_iter(buffer)
        out = real_update(buffer, total_timesteps=total_timesteps)
        if strict:
            audit.assert_clean()
        return out

    trainer.update = update

    def uninstall():
        collector.collect = real_collect
        trainer.update = real_update

    return audit, uninstall


__all__ = ["CREDIT_FIELD", "CreditAudit", "filter_batch_to_credited", "install_credit_boundary"]
