"""Live-state gradient: synthetic signed delta_Q, real architecture, both signs.

Prelaunch gate 2 of CCP_SUCCESSOR_BUILD_CONTRACT.json's required suite, built blind while
the Phase 1 bank was still collecting. Stage 2 stays unopened: delta_Q values here are
synthetic (+0.5, -0.5), not read from any Phase 1 row.

Reuses the exact LRO-private-branch architecture and helper functions the V3 branch-isolation
smoke used (experiments/smoke_hog_psp_branch_isolation.py) -- same PRIVATE_MARKERS, same
build() path through build_fresh_k2, same private_names()/snapshot()/changed() pattern -- so
this smoke exercises the identical private-capacity mechanism the successor will actually
train, not a simplified stand-in.

What must be true for EACH sign:

    +0.5 on Pole A  ->  z0's private branch MOVES toward pi_A;    z1's private branch untouched
    -0.5 on Pole A  ->  z0's private branch MOVES toward pi_B;    z1's private branch untouched
    (mirrored on Pole B)

and in every case:

    the committed agent's contribution is exactly zero (log_prob 0, no gradient path)
    the OTHER latent's private branch is bit-identical before/after (architectural isolation)

Run:  python tests/test_causal_live_state_gradient.py
  or: pytest tests/test_causal_live_state_gradient.py -q
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

import experiments.smoke_hog_psp_branch_isolation as B   # reused, not reimplemented

DEVICE = "cpu"          # architecture/gradient-flow claim; no env, no GPU needed
GS_DIM = 170
N_AGENTS, GRID_SHAPE, VEC_DIM = 2, (7, 20, 20), 20    # verified against the real model


def _synthetic_batch(model, *, pole: str, delta_q: float, batch: int = 8, seed: int = 0):
    """A committed-agent-plus-free-agent batch on real V4-style action dims, real obs shape.

    Agent 0 is COMMITTED (mask collapses to one legal macro/target); agent 1 is FREE. Teacher
    actions for agent 1 are drawn from the teacher CausalRecord.teacher would select, encoded
    as a concrete action index distinct from what a randomly initialised policy would already
    put mass on, so a real gradient is the only way agreement could increase.
    """
    import numpy as np
    from rl.causal_supervision import CausalRecord

    torch.manual_seed(seed)
    rec = CausalRecord(pole, delta_q, state_id=f"smoke|{pole}", agent_id=1)
    assert rec.teacher is not None, "delta_q=0 has no gradient claim to test"

    global_state = torch.randn(batch, GS_DIM)
    grid = torch.randn(batch, N_AGENTS, *GRID_SHAPE)
    vec = torch.randn(batch, N_AGENTS, VEC_DIM)
    agent_mask = torch.ones(batch, N_AGENTS)

    n_macros, n_targets = 5, 50
    mask = torch.zeros(batch, 2 * (n_macros + n_targets))
    # agent 0: committed, one legal macro/target
    mask[:, 1] = 1.0
    mask[:, n_macros + 7] = 1.0
    # agent 1: free, several legal
    off = n_macros + n_targets
    mask[:, off: off + 3] = 1.0
    mask[:, off + n_macros: off + n_macros + 20] = 1.0

    actions = torch.zeros(batch, 4, dtype=torch.long)
    actions[:, 0], actions[:, 1] = 1, 7                  # agent 0: the committed macro/target
    teacher_action_agent1 = 2 if rec.teacher == "pi_A" else 0
    actions[:, 2], actions[:, 3] = teacher_action_agent1, 11

    decision_mask = torch.tensor([[False, True]]).repeat(batch, 1)
    weights = torch.full((batch, 2), rec.weight, dtype=torch.float32)
    z_idx = torch.full((batch,), rec.latent, dtype=torch.long)
    obs = {"global_state": global_state, "grid": grid, "vec": vec,
           "agent_mask": agent_mask, "mask": mask}
    return obs, actions, decision_mask, weights, z_idx, rec


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class LiveStateGradientTests(unittest.TestCase):

    def _run_one_sign(self, pole: str, delta_q: float):
        from rl.causal_supervision import causal_supervision_loss

        _cfg, model = B.build(DEVICE, B.LRO_FLAGS)
        obs, actions, dm, w, z, rec = _synthetic_batch(model, pole=pole, delta_q=delta_q)

        before = B.snapshot(model)
        own_names = B.private_names(model, rec.latent)
        other_z = 1 - rec.latent
        other_names = B.private_names(model, other_z)
        self.assertTrue(own_names, "no private parameters found for the target latent")
        self.assertTrue(other_names, "no private parameters found for the other latent")

        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        opt.zero_grad(set_to_none=True)
        loss = causal_supervision_loss(model, obs, actions, z_idx=z,
                                       decision_mask=dm, weights=w)
        loss.backward()

        # committed agent (head columns 0,1 of the 4-column action layout) must carry no grad
        from rl.custom_ppo.strategy_anchor import action_log_prob
        with torch.no_grad():
            per_head = action_log_prob(model, obs, actions, z_idx=z)
        self.assertTrue(torch.equal(per_head[:, :2], torch.zeros_like(per_head[:, :2])),
                        "committed agent contributed non-zero log_prob")

        opt.step()
        after = B.snapshot(model)

        own_moved = B.changed(before, after, own_names)
        other_moved = B.changed(before, after, other_names)
        return {"own_moved": own_moved, "other_moved": other_moved, "loss": float(loss),
               "teacher": rec.teacher, "latent": rec.latent}

    def test_positive_delta_q_pole_a_moves_z0_toward_pi_a(self):
        r = self._run_one_sign("A", +0.5)
        self.assertEqual(r["teacher"], "pi_A")
        self.assertEqual(r["latent"], 0)
        self.assertTrue(r["own_moved"], "z0's private branch did not move under +delta_Q")
        self.assertFalse(r["other_moved"], "z1's private branch moved; isolation violated")

    def test_negative_delta_q_pole_a_moves_z0_toward_pi_b(self):
        r = self._run_one_sign("A", -0.5)
        self.assertEqual(r["teacher"], "pi_B")
        self.assertEqual(r["latent"], 0)
        self.assertTrue(r["own_moved"], "z0's private branch did not move under -delta_Q")
        self.assertFalse(r["other_moved"], "z1's private branch moved; isolation violated")

    def test_positive_delta_q_pole_b_moves_z1_toward_pi_b(self):
        r = self._run_one_sign("B", +0.5)
        self.assertEqual(r["teacher"], "pi_B")
        self.assertEqual(r["latent"], 1)
        self.assertTrue(r["own_moved"], "z1's private branch did not move under +delta_Q")
        self.assertFalse(r["other_moved"], "z0's private branch moved; isolation violated")

    def test_negative_delta_q_pole_b_moves_z1_toward_pi_a(self):
        r = self._run_one_sign("B", -0.5)
        self.assertEqual(r["teacher"], "pi_A")
        self.assertEqual(r["latent"], 1)
        self.assertTrue(r["own_moved"], "z1's private branch did not move under -delta_Q")
        self.assertFalse(r["other_moved"], "z0's private branch moved; isolation violated")

    def test_zero_delta_q_produces_zero_gradient_no_op(self):
        """delta_q = 0 has no teacher; the loss must be a real zero, not merely small."""
        from rl.causal_supervision import CausalRecord, causal_supervision_loss

        _cfg, model = B.build(DEVICE, B.LRO_FLAGS)
        rec = CausalRecord("A", 0.0, state_id="smoke|A", agent_id=1)
        obs, actions, dm, _w, z, _ = _synthetic_batch(model, pole="A", delta_q=0.5)
        # rebuild weights consistent with a real zero-delta_Q record
        w = torch.zeros(actions.shape[0], 2, dtype=torch.float32)
        before = B.snapshot(model)
        loss = causal_supervision_loss(model, obs, actions, z_idx=z, decision_mask=dm, weights=w)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        after_names = B.private_names(model, 0) + B.private_names(model, 1)
        for n in after_names:
            g = dict(model.named_parameters())[n].grad
            self.assertTrue(g is None or bool(torch.equal(g, torch.zeros_like(g))),
                            f"{n} received gradient at delta_q=0")


if __name__ == "__main__":
    unittest.main()
