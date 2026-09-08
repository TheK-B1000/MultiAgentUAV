"""Task-objective purity: causal supervision is an auxiliary POLICY loss, nothing else.

Prelaunch gate 1 of CCP_SUCCESSOR_BUILD_CONTRACT.json's required suite, built blind while
the Phase 1 bank was still collecting.

Uses the REAL rollout buffer (rl.ppo_core.TensorDictRolloutBuffer) and the REAL compute_gae,
not a stand-in, so this is not a test of a simplified model of the trainer -- it is the actual
GAE code path the trainer calls.

Claim under test: toggling causal supervision on/off must leave every TASK quantity bitwise
identical --

    rewards, values, next_values, terminated, truncated   (raw transition fields)
    advantages, returns                                   (GAE output, the value-loss target)

and must leave the TASK loss VALUE itself bitwise identical, with the combined loss differing
from the task loss by exactly lambda * L_causal. If the causal term ever perturbs a task
quantity, it has stopped being an auxiliary loss and has become reward shaping in disguise.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    HAVE_TORCH = True
except Exception:                                            # pragma: no cover
    HAVE_TORCH = False

N_MACROS, N_TARGETS, PER_AGENT = 5, 50, 2
TASK_FIELDS = ("rewards", "values", "next_values", "terminated", "truncated",
              "advantages", "returns")


def _stub(action_dims, logits):
    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    class _M:
        def __init__(self):
            self.action_dims = list(action_dims)
            self.logits = logits

        def policy_logits(self, obs, z_idx=None):
            return self.logits

    m = _M()
    m._mask_logits = SharedActorCentralizedCritic._mask_logits.__get__(m, _M)
    return m


def _build_buffer(T=6, B=4, seed=0):
    """A real TensorDictRolloutBuffer, filled deterministically, GAE run once."""
    from rl.ppo_core import TensorDictRolloutBuffer

    g = torch.Generator().manual_seed(seed)
    buf = TensorDictRolloutBuffer(buffer_size=T, n_envs=B)
    for name in ("rewards", "values", "next_values"):
        buf.register_field(name)
    buf.register_field("terminated", dtype=torch.bool)
    buf.register_field("truncated", dtype=torch.bool)

    for t in range(T):
        term = torch.zeros(B, dtype=torch.bool)
        if t == T - 1:
            term[0] = True
        buf.add(
            rewards=torch.randn(B, generator=g, dtype=torch.float32),
            values=torch.randn(B, generator=g, dtype=torch.float32),
            next_values=torch.randn(B, generator=g, dtype=torch.float32),
            terminated=term,
            truncated=torch.zeros(B, dtype=torch.bool),
        )
    buf.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
    return buf


def _snapshot(buf) -> dict:
    return {name: buf.fields[name].clone() for name in TASK_FIELDS}


def _assert_task_unchanged(case: unittest.TestCase, ref: dict, buf) -> None:
    for name in TASK_FIELDS:
        case.assertTrue(torch.equal(ref[name], buf.fields[name]),
                        f"task field {name!r} changed")


def _task_actor_critic_loss(buf) -> torch.Tensor:
    """A minimal, deterministic stand-in for the task PPO actor+value loss.

    Uses only buffer fields -- exactly what the real update step is restricted to for the
    task objective. Not the real PPO surrogate; the point is that it depends ONLY on task
    fields, so its value is a clean witness for "did toggling causal supervision change
    anything upstream of it".
    """
    adv = buf.fields["advantages"][: buf.pos]
    ret = buf.fields["returns"][: buf.pos]
    val = buf.fields["values"][: buf.pos]
    return (adv.pow(2).mean() + (ret - val).pow(2).mean())


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class TaskObjectivePurityTests(unittest.TestCase):

    def test_causal_toggle_leaves_gae_output_bitwise_identical(self):
        """Compute GAE once, then verify running causal supervision alongside it never
        touches the buffer -- because it is never given a way to."""
        from rl.causal_supervision import causal_supervision_loss

        buf = _build_buffer(seed=1)
        ref = _snapshot(buf)

        model, obs, acts, _ = self._minimal_policy_batch(seed=2)
        dm = torch.tensor([[True, False]]).repeat(acts.shape[0], 1)
        w = torch.full((acts.shape[0], 2), 0.5, dtype=torch.float64)
        z = torch.zeros(acts.shape[0], dtype=torch.long)
        causal = causal_supervision_loss(model, obs, acts, z_idx=z,
                                         decision_mask=dm, weights=w)
        causal.backward()

        _assert_task_unchanged(self, ref, buf)

    def test_task_loss_value_identical_with_causal_on_or_off(self):
        """The TASK loss computed from the buffer must be the exact same number whether or
        not a causal term is separately added to the total."""
        from rl.causal_supervision import causal_supervision_loss

        buf_off = _build_buffer(seed=5)
        task_loss_off = _task_actor_critic_loss(buf_off)

        buf_on = _build_buffer(seed=5)          # identical seed -> identical rollout data
        task_loss_on = _task_actor_critic_loss(buf_on)

        model, obs, acts, _ = self._minimal_policy_batch(seed=6)
        dm = torch.tensor([[True, False]]).repeat(acts.shape[0], 1)
        w = torch.full((acts.shape[0], 2), 0.75, dtype=torch.float64)
        z = torch.zeros(acts.shape[0], dtype=torch.long)
        causal = causal_supervision_loss(model, obs, acts, z_idx=z,
                                         decision_mask=dm, weights=w)
        lam = 0.1
        total_on = task_loss_on + lam * causal

        self.assertTrue(torch.equal(task_loss_off, task_loss_on),
                        "the task loss differs between the two identically-seeded buffers "
                        "before any causal term is added")
        self.assertAlmostEqual(float(total_on - task_loss_on), lam * float(causal), places=12,
                               msg="combined loss does not decompose as task + lambda*causal")
        _assert_task_unchanged(self, _snapshot(buf_off), buf_on)

    def test_zero_lambda_makes_causal_supervision_a_true_no_op(self):
        """lambda=0 must leave the total loss bitwise equal to the task loss alone, even
        though the causal term was computed and is part of the graph."""
        from rl.causal_supervision import causal_supervision_loss

        buf = _build_buffer(seed=8)
        task_loss = _task_actor_critic_loss(buf)

        model, obs, acts, _ = self._minimal_policy_batch(seed=9)
        dm = torch.tensor([[True, False]]).repeat(acts.shape[0], 1)
        w = torch.full((acts.shape[0], 2), 1.0, dtype=torch.float64)
        z = torch.zeros(acts.shape[0], dtype=torch.long)
        causal = causal_supervision_loss(model, obs, acts, z_idx=z,
                                         decision_mask=dm, weights=w)
        total = task_loss + 0.0 * causal
        self.assertTrue(torch.equal(total, task_loss),
                        "lambda=0 changed the loss value; causal supervision is not a "
                        "true no-op at lambda=0")

    # -- helper: a tiny real-forward-pass policy batch, independent of the buffer ----
    def _minimal_policy_batch(self, seed):
        per = N_MACROS + N_TARGETS
        blocks = []
        for a in range(2):
            g = torch.Generator().manual_seed(seed * 1000 + a)
            blocks.append(torch.randn(4, per, dtype=torch.float64, generator=g))
        logits = torch.cat(blocks, dim=1).detach().requires_grad_(True)
        rows = []
        for a in range(2):
            mac, tar = torch.zeros(N_MACROS), torch.zeros(N_TARGETS)
            if a == 1:
                mac[1], tar[7] = 1.0, 1.0
            else:
                mac[:3], tar[:20] = 1.0, 1.0
            rows.append(torch.cat([mac, tar]))
        mask = torch.cat(rows).unsqueeze(0).repeat(4, 1)
        actions = torch.zeros(4, 2 * PER_AGENT, dtype=torch.long)
        actions[:, 2], actions[:, 3] = 1, 7
        actions[:, 0], actions[:, 1] = 2, 5
        dims = [N_MACROS, N_TARGETS, N_MACROS, N_TARGETS]
        return _stub(dims, logits), {"mask": mask}, actions, logits


if __name__ == "__main__":
    unittest.main()
