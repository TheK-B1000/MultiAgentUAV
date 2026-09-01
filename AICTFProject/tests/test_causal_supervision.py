"""Prelaunch smokes for the causal-advantage objective, built blind to Phase 1 results.

Covers the first three items of CCP_SUCCESSOR_BUILD_CONTRACT.json's required suite:
decision mask, |delta_Q| weight scaling, and latent routing -- each with a negative control,
because a guard that cannot fail proves nothing.

Reuses the float64 discipline from tests/test_anchor_loss_decision_mask.py: the equivalence
claims are about semantics, but the batches being compared have different shapes and their
.sum() reductions take different accumulation paths. Measuring in float64 keeps the
assertions tight instead of inviting a tolerance fitted to an observed gap.
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


def _build(n_agents, committed, batch=4, seed=0, dtype=torch.float64):
    per = N_MACROS + N_TARGETS
    blocks = []
    for a in range(n_agents):
        g = torch.Generator().manual_seed(seed * 1000 + a)
        blocks.append(torch.randn(batch, per, dtype=dtype, generator=g))
    logits = torch.cat(blocks, dim=1).detach().requires_grad_(True)

    rows = []
    for a in range(n_agents):
        mac, tar = torch.zeros(N_MACROS), torch.zeros(N_TARGETS)
        if a in committed:
            mac[1], tar[7] = 1.0, 1.0
        else:
            mac[:3], tar[:20] = 1.0, 1.0
        rows.append(torch.cat([mac, tar]))
    mask = torch.cat(rows).unsqueeze(0).repeat(batch, 1)

    actions = torch.zeros(batch, n_agents * PER_AGENT, dtype=torch.long)
    for a in range(n_agents):
        actions[:, a * 2] = 1 if a in committed else 2
        actions[:, a * 2 + 1] = 7 if a in committed else 5
    dims = []
    for _ in range(n_agents):
        dims += [N_MACROS, N_TARGETS]
    return _stub(dims, logits), {"mask": mask}, actions, logits


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class CausalSupervisionTests(unittest.TestCase):

    # ---- 1. decision mask ------------------------------------------------
    def test_committed_agent_changes_neither_loss_nor_gradient(self):
        from rl.causal_supervision import causal_supervision_loss
        losses, grads = [], []
        for n_extra in (0, 1, 3):
            n_agents = 1 + n_extra
            model, obs, acts, logits = _build(n_agents, set(range(1, n_agents)), seed=7)
            b = acts.shape[0]
            dm = torch.tensor([[True] + [False] * n_extra]).repeat(b, 1)
            w = torch.ones(b, n_agents, dtype=torch.float64)
            z = torch.zeros(b, dtype=torch.long)
            loss = causal_supervision_loss(model, obs, acts, z_idx=z,
                                           decision_mask=dm, weights=w)
            loss.backward()
            losses.append(float(loss))
            grads.append(logits.grad[:, :N_MACROS + N_TARGETS].clone())
        for i in (1, 2):
            self.assertAlmostEqual(losses[0], losses[i], places=12,
                                   msg=f"committed agents moved the loss: {losses}")
            self.assertTrue(torch.allclose(grads[0], grads[i], atol=1e-12, rtol=0),
                            "committed agents moved the free agent's gradient")

    # ---- 2. |delta_Q| weighting -----------------------------------------
    def test_weight_scales_the_gradient_exactly(self):
        """The loss is weight-normalised, so a UNIFORM rescale must leave it invariant."""
        from rl.causal_supervision import causal_supervision_loss
        ref_loss, ref_grad = None, None
        for scale in (0.25, 0.5, 1.0, 4.0):
            model, obs, acts, logits = _build(2, {1}, seed=9)
            b = acts.shape[0]
            dm = torch.tensor([[True, False]]).repeat(b, 1)
            w = torch.full((b, 2), scale, dtype=torch.float64)
            z = torch.zeros(b, dtype=torch.long)
            loss = causal_supervision_loss(model, obs, acts, z_idx=z, decision_mask=dm, weights=w)
            loss.backward()
            g = logits.grad[:, :N_MACROS + N_TARGETS].clone()
            if ref_loss is None:
                ref_loss, ref_grad = float(loss), g
            else:
                self.assertAlmostEqual(float(loss), ref_loss, places=12,
                                       msg="uniform reweighting changed a normalised loss")
                self.assertTrue(torch.allclose(g, ref_grad, atol=1e-12, rtol=0))

    def test_relative_weights_shift_the_gradient_toward_the_heavier_sample(self):
        """Non-uniform weights must actually re-balance which rows dominate."""
        from rl.causal_supervision import causal_supervision_loss
        outs = []
        for w0 in (0.0, 1.0):
            model, obs, acts, logits = _build(1, set(), batch=2, seed=13)
            dm = torch.tensor([[True], [True]])
            w = torch.tensor([[w0], [1.0]], dtype=torch.float64)
            z = torch.zeros(2, dtype=torch.long)
            loss = causal_supervision_loss(model, obs, acts, z_idx=z, decision_mask=dm, weights=w)
            loss.backward()
            outs.append((float(loss), logits.grad[0].clone(), logits.grad[1].clone()))
        # with w0 = 0 the first row must contribute NO gradient at all
        self.assertTrue(torch.equal(outs[0][1], torch.zeros_like(outs[0][1])),
                        "a zero-weight sample produced gradient")
        self.assertFalse(torch.equal(outs[1][1], torch.zeros_like(outs[1][1])),
                         "a unit-weight sample produced no gradient")
        self.assertNotAlmostEqual(outs[0][0], outs[1][0], places=6,
                                  msg="dropping a sample's weight did not change the loss")

    def test_all_zero_weights_give_zero_loss_and_no_nan(self):
        from rl.causal_supervision import causal_supervision_loss
        model, obs, acts, logits = _build(2, {1}, seed=21)
        b = acts.shape[0]
        dm = torch.tensor([[True, False]]).repeat(b, 1)
        w = torch.zeros(b, 2, dtype=torch.float64)
        z = torch.zeros(b, dtype=torch.long)
        loss = causal_supervision_loss(model, obs, acts, z_idx=z, decision_mask=dm, weights=w)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)),
                        "all-zero weights produced gradient")

    def test_negative_weights_are_refused(self):
        from rl.causal_supervision import causal_supervision_loss, CausalRoutingError
        model, obs, acts, _ = _build(2, {1}, seed=3)
        b = acts.shape[0]
        with self.assertRaises(CausalRoutingError):
            causal_supervision_loss(model, obs, acts, z_idx=torch.zeros(b, dtype=torch.long),
                                    decision_mask=torch.tensor([[True, False]]).repeat(b, 1),
                                    weights=torch.full((b, 2), -0.5, dtype=torch.float64))

    # ---- 3. winner-directed routing -------------------------------------
    def test_positive_delta_q_routes_to_the_pole_matched_specialist(self):
        from rl.causal_supervision import CausalRecord
        a = CausalRecord("s|A|40", "A", 0, +0.5, "single_macro")
        b = CausalRecord("s|B|40", "B", 1, +0.5, "single_macro")
        self.assertEqual((a.latent, a.teacher, a.weight), (0, "pi_A", 0.5))
        self.assertEqual((b.latent, b.teacher, b.weight), (1, "pi_B", 0.5))
        a.assert_routing(); b.assert_routing()

    def test_NEGATIVE_delta_q_routes_to_the_OTHER_specialist(self):
        """The defect this replaced: |delta_Q| keeps magnitude but destroys direction.

        A Pole-A boundary measuring -0.50 means pi_B was causally better THERE. Fixed
        pole-matched routing would carry weight 0.50 and still train z0 toward pi_A --
        training hardest on a decision the measurement showed was worse.
        """
        from rl.causal_supervision import CausalRecord
        a = CausalRecord("s|A|40", "A", 0, -0.5, "single_macro")
        b = CausalRecord("s|B|40", "B", 1, -0.5, "single_macro")
        self.assertEqual(a.teacher, "pi_B", "negative delta_q on Pole A must supervise pi_B")
        self.assertEqual(b.teacher, "pi_A", "negative delta_q on Pole B must supervise pi_A")
        self.assertEqual((a.weight, b.weight), (0.5, 0.5), "magnitude is still |delta_Q|")
        self.assertEqual((a.latent, b.latent), (0, 1), "the LATENT never flips, only the target")
        a.assert_routing(); b.assert_routing()

    def test_zero_delta_q_has_no_teacher_and_no_weight(self):
        from rl.causal_supervision import CausalRecord
        r = CausalRecord("s|A|40", "A", 0, 0.0, "single_macro")
        self.assertIsNone(r.teacher)
        self.assertEqual(r.weight, 0.0)
        r.assert_routing()

    def test_declaring_the_loser_as_teacher_MUST_fail(self):
        """Negative control: a declared teacher inconsistent with the sign is rejected."""
        from rl.causal_supervision import CausalRecord, CausalRoutingError
        for pole, dq, loser in (("A", +0.5, "pi_B"), ("A", -0.5, "pi_A"),
                                ("B", +0.5, "pi_A"), ("B", -0.5, "pi_B")):
            rec = CausalRecord(f"s|{pole}|40", pole, 0, dq, "single_macro")
            with self.assertRaises(CausalRoutingError):
                rec.assert_routing(declared_teacher=loser)

    def test_missing_decision_predicate_is_fatal(self):
        """Absence is an error state: no commit_ticks_left means refuse, never default True."""
        from rl.causal_supervision import decision_mask_from_core, CausalRoutingError

        class _Bare:
            pass

        with self.assertRaises(CausalRoutingError):
            decision_mask_from_core(_Bare(), 2)


if __name__ == "__main__":
    unittest.main()
