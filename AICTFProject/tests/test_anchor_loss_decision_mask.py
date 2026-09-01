"""Regression guard: committed heads must not inflate the anchor-loss denominator.

The defect this locks out (found 2026-09-01, recorded in CCP_FORCED_HEAD_DILUTION.json):
``anchor_loss`` has correct semantics and a docstring promising that "a locked agent
contributes nothing to the loss", but ``paired_rehearsal.py`` and ``oracle_rehearsal.py``
passed ``obs["agent_mask"]`` as the decision mask -- and in the teacher bank that array is
ALL ONES. So committed agents were counted in ``denom = m.sum()`` while contributing
identically zero, scaling the effective lambda to ~0.662 of nominal through OG-PSP, V3, V4.

A committed agent's observation mask is one-hot, and ``_mask_logits`` applies
``masked_fill(mask <= 0, -1e8)``, so its head is degenerate: log_prob = 0 exactly and the
gradient vanishes. That is why this was a normaliser bug and not target contamination --
and it is also why no ordinary test caught it. The loss VALUE stayed well-formed.

These tests exercise the REAL ``_mask_logits`` and the REAL ``anchor_loss``. They assert
behaviour, not variable names, so renaming the mask cannot make them pass spuriously.
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

N_MACROS, N_TARGETS = 5, 50
PER_AGENT = 2                                                # macro head + target head


def _stub(action_dims, logits):
    """Minimal model carrying the REAL _mask_logits, so the masking under test is production code."""
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


def _mask_row(n_agents, committed):
    """Full legality mask. A committed agent gets one-hot macro AND one-hot target."""
    row = []
    for a in range(n_agents):
        mac = torch.zeros(N_MACROS)
        tar = torch.zeros(N_TARGETS)
        if a in committed:
            mac[1] = 1.0                                     # exactly one legal macro
            tar[7] = 1.0                                     # exactly one legal target
        else:
            mac[:3] = 1.0                                    # genuinely free: several legal
            tar[:20] = 1.0
        row.append(torch.cat([mac, tar]))
    return torch.cat(row)


def _build(n_agents, committed, batch=4, seed=0, dtype=torch.float64):
    """Default float64.

    The equivalence claims below are about SEMANTICS -- whether a committed agent changes
    the loss -- but the two batches being compared have different tensor shapes, so their
    ``.sum()`` reductions take different accumulation paths. In float32 that shows up as a
    one-ULP difference with the per-head values bitwise identical, which would tempt a
    loosened tolerance chosen to fit the observed gap. Measuring in float64 instead keeps
    the assertion tight; a separate test bounds the float32 case a priori.
    """
    # Per-agent blocks are drawn from their own seeded generator, so agent 0's logits are
    # IDENTICAL no matter how many agents the batch has. Filling one (batch, n_agents*55)
    # tensor row-major would silently change agent 0's slice as n_agents grows, and the
    # "adding committed agents changes nothing" test would then be comparing two different
    # free agents.
    per = N_MACROS + N_TARGETS
    blocks = []
    for a in range(n_agents):
        g = torch.Generator().manual_seed(seed * 1000 + a)
        blocks.append(torch.randn(batch, per, dtype=dtype, generator=g))
    logits = torch.cat(blocks, dim=1).detach().requires_grad_(True)
    mask = _mask_row(n_agents, committed).unsqueeze(0).repeat(batch, 1)
    actions = torch.zeros(batch, n_agents * PER_AGENT, dtype=torch.long)
    for a in range(n_agents):
        actions[:, a * 2] = 1 if a in committed else 2       # legal under the mask above
        actions[:, a * 2 + 1] = 7 if a in committed else 5
    dims = []
    for _ in range(n_agents):
        dims += [N_MACROS, N_TARGETS]
    return _stub(dims, logits), {"mask": mask}, actions, logits


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class AnchorLossDecisionMaskTests(unittest.TestCase):

    def test_committed_head_log_prob_is_exactly_zero(self):
        """The load-bearing fact: a one-hot mask degenerates the head, so it cannot contribute."""
        from rl.custom_ppo.strategy_anchor import action_log_prob
        model, obs, actions, _ = _build(n_agents=2, committed={1})
        per_head = action_log_prob(model, obs, actions)
        committed_heads = per_head[:, 2:4]
        self.assertTrue(torch.equal(committed_heads, torch.zeros_like(committed_heads)),
                        f"committed heads must have log_prob exactly 0, got {committed_heads}")
        self.assertTrue((per_head[:, 0:2] < 0).all(), "free heads should carry real log-probs")

    def test_mixed_batch_equals_free_agent_only(self):
        """L(one free + one committed) == L(that free agent alone), exactly."""
        from rl.custom_ppo.strategy_anchor import anchor_loss
        mixed, obs_m, act_m, _ = _build(n_agents=2, committed={1}, seed=3)
        solo, obs_s, act_s, _ = _build(n_agents=1, committed=set(), seed=3)
        # the free agent's logits and mask are the leading slice in both builds
        solo.logits = mixed.logits[:, :N_MACROS + N_TARGETS]
        obs_s["mask"] = obs_m["mask"][:, :N_MACROS + N_TARGETS]
        act_s[:] = act_m[:, :PER_AGENT]

        l_mixed = anchor_loss(mixed, obs_m, act_m,
                              torch.tensor([[True, False]]).repeat(act_m.shape[0], 1))
        l_solo = anchor_loss(solo, obs_s, act_s,
                             torch.tensor([[True]]).repeat(act_s.shape[0], 1))
        self.assertLess(abs(float(l_mixed) - float(l_solo)), 1e-12,
                        f"committed agent changed the loss: {float(l_mixed)} vs {float(l_solo)}")

    def test_float32_equivalence_within_an_a_priori_bound(self):
        """Same claim in float32, bounded by eps*sqrt(n)*|L| -- derived, not fitted."""
        from rl.custom_ppo.strategy_anchor import anchor_loss
        mixed, obs_m, act_m, _ = _build(2, {1}, seed=3, dtype=torch.float32)
        solo, obs_s, act_s, _ = _build(1, set(), seed=3, dtype=torch.float32)
        solo.logits = mixed.logits[:, :N_MACROS + N_TARGETS]
        obs_s["mask"] = obs_m["mask"][:, :N_MACROS + N_TARGETS]
        act_s[:] = act_m[:, :PER_AGENT]

        l_mixed = anchor_loss(mixed, obs_m, act_m,
                              torch.tensor([[True, False]]).repeat(act_m.shape[0], 1))
        l_solo = anchor_loss(solo, obs_s, act_s,
                             torch.tensor([[True]]).repeat(act_s.shape[0], 1))
        n_terms = act_m.shape[0] * act_m.shape[1]
        eps32 = float(torch.finfo(torch.float32).eps)
        bound = eps32 * (n_terms ** 0.5) * abs(float(l_solo))
        self.assertLess(abs(float(l_mixed) - float(l_solo)), bound,
                        "float32 gap exceeds the a-priori reduction-order bound, which would "
                        "mean a semantic difference rather than rounding")

    def test_extra_committed_agents_change_neither_loss_nor_gradient(self):
        """Adding committed agents must leave L and dL/d(free logits) untouched."""
        from rl.custom_ppo.strategy_anchor import anchor_loss
        losses, grads = [], []
        for n_extra in (0, 1, 3):
            n_agents = 1 + n_extra
            committed = set(range(1, n_agents))
            model, obs, actions, logits = _build(n_agents, committed, seed=11)
            dm = torch.tensor([[True] + [False] * n_extra]).repeat(actions.shape[0], 1)
            loss = anchor_loss(model, obs, actions, dm)
            loss.backward()
            losses.append(float(loss))
            grads.append(logits.grad[:, :N_MACROS + N_TARGETS].clone())
        for i in (1, 2):
            self.assertAlmostEqual(losses[0], losses[i], places=12,
                                   msg=f"loss moved when committed agents were added: {losses}")
            self.assertTrue(torch.allclose(grads[0], grads[i], atol=1e-12, rtol=0),
                            "gradient on the free agent moved when committed agents were added")

    def test_all_ones_mask_reproduces_the_documented_dilution(self):
        """The actual defect: an all-ones mask scales the loss by n_free / n_total."""
        from rl.custom_ppo.strategy_anchor import anchor_loss
        model, obs, actions, _ = _build(n_agents=2, committed={1}, seed=5)
        b = actions.shape[0]
        correct = anchor_loss(model, obs, actions,
                              torch.tensor([[True, False]]).repeat(b, 1))
        all_ones = anchor_loss(model, obs, actions,
                               torch.tensor([[True, True]]).repeat(b, 1))
        # 2 informative heads out of 4 -> exactly half
        self.assertAlmostEqual(float(all_ones), float(correct) * 0.5, places=12,
                               msg="an all-ones decision mask must dilute by n_free/n_total; "
                                   "if this fails the reduction changed")
        self.assertLess(abs(float(all_ones)), abs(float(correct)),
                        "the diluted loss must be strictly smaller in magnitude")


if __name__ == "__main__":
    unittest.main()
