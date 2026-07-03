"""Pinning tests for the V6I12 paired-advantage router extension.

V6I12 keeps the V6I11 data-collection contract (episode-persistent arcs,
frozen actor, 50 % uniform exploration) and swaps the EXTERNAL regressor
from a raw-return Q-router to a double-centering pair:

    V(context)          — context baseline: E[normalized_return | context]
    A(context, z)       — residual latent advantage, target
                          norm_ret - stopgrad(V(context))

These tests pin:

1. Preset alias resolution + the minimal-diff invariant vs v6i11 (only
   ``experiment_id`` and ``run_tag`` may differ — the trainer-side data
   collection MUST be identical).
2. Model output shapes and opponent one-hot context construction.
3. Double-centering variance reduction: when the return is dominated by a
   context (opponent) component, the advantage target std drops below the
   normalized-return std (~1.0), i.e. V(context) actually absorbs the noise
   that swamped V6I11.
4. stopgrad on the baseline: A's loss does not backprop into V.
5. advantage_gap_ci: a real per-z advantage yields a CI excluding zero;
   flat (z-independent) returns do not.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from rl.global_state import GLOBAL_STATE_DIM
from rl.router.q_value_router import QRouterReplayBuffer
from rl.router.advantage_router import (
    AdvantageRouter,
    ContextualVBaseline,
    advantage_gap_ci,
    train_advantage_router,
)


_OPP_ID_TO_IDX = {7: 0, 8: 1, 9: 2}
_K = 4
_N_OPP = 3
_CTX_DIM = GLOBAL_STATE_DIM + _N_OPP
_OPP_RAW = [7, 8, 9]  # canonical OP8/OP9/OP10


def _arc_record(uid: int, *, z: int, ret: float, opp: int, env: int = 0,
                length: int = 130, reason: str = "episode_end") -> dict:
    # Zero geometry: the opponent one-hot carries all the context signal so
    # V(context) has something learnable and the test is deterministic.
    return {
        "global_state_0": torch.zeros(GLOBAL_STATE_DIM),
        "z": int(z),
        "arc_return": float(ret),
        "opponent_id": int(opp),
        "arc_length": int(length),
        "reason": reason,
        "env_index": int(env),
        "arc_uid": int(uid),
    }


def _build_replay(records: list[dict]) -> QRouterReplayBuffer:
    a_router = AdvantageRouter(
        n_opponents=_N_OPP, opponent_id_to_idx=_OPP_ID_TO_IDX, latent_k=_K
    )
    replay = QRouterReplayBuffer(capacity=100_000, context_dim=_CTX_DIM, latent_k=_K)
    replay.push_many(
        records,
        rollout_index=1,
        opponent_id_to_idx=_OPP_ID_TO_IDX,
        build_context=a_router.build_context_from_record,
    )
    return replay


class V6i12PresetContractTests(unittest.TestCase):
    """The v6i12 preset must be a thin alias of v6i11 (external model only)."""

    _ALIASES = [
        "v6i12",
        "v6i12_advantage_router",
        "v6i12_advantage_router_hardpool",
        "plan_faithful_latent_v6i12_advantage_router_hardpool",
    ]

    def test_all_aliases_resolve_equal(self) -> None:
        import dataclasses

        base = dataclasses.asdict(apply_preset(PPOConfig(), self._ALIASES[0]))
        for alias in self._ALIASES[1:]:
            other = dataclasses.asdict(apply_preset(PPOConfig(), alias))
            self.assertEqual(base, other, f"alias {alias} diverged from {self._ALIASES[0]}")

    def test_minimal_diff_vs_v6i11(self) -> None:
        import dataclasses

        v11 = dataclasses.asdict(
            apply_preset(PPOConfig(), "v6i11_q_router_hardpool")
        )
        v12 = dataclasses.asdict(apply_preset(PPOConfig(), "v6i12"))
        changed = {k for k in v11 if v11[k] != v12[k]}
        self.assertEqual(
            changed,
            {"experiment_id", "run_tag"},
            f"v6i12 must differ from v6i11 only in experiment_id/run_tag; got {changed}",
        )

    def test_arc_collection_inherited(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i12")
        self.assertTrue(bool(getattr(cfg, "latent_arc_credit_enabled", False)))
        self.assertEqual(float(getattr(cfg, "router_uniform_exploration_prob")), 0.5)
        # Internal PPO router channels must remain silent (external model only).
        self.assertEqual(float(getattr(cfg, "latent_arc_credit_coef", -1.0)), 0.0)
        self.assertEqual(float(getattr(cfg, "router_ent_coef", -1.0)), 0.0)
        self.assertEqual(float(getattr(cfg, "latent_lam_h", -1.0)), 0.0)


class V6i12ModelShapeTests(unittest.TestCase):

    def test_v_baseline_forward_shape(self) -> None:
        v = ContextualVBaseline(n_opponents=_N_OPP)
        ctx = torch.randn(5, _CTX_DIM)
        out = v(ctx)
        self.assertEqual(tuple(out.shape), (5,))

    def test_a_router_forward_shape(self) -> None:
        a = AdvantageRouter(n_opponents=_N_OPP, latent_k=_K)
        ctx = torch.randn(5, _CTX_DIM)
        out = a(ctx)
        self.assertEqual(tuple(out.shape), (5, _K))

    def test_opponent_one_hot(self) -> None:
        a = AdvantageRouter(n_opponents=_N_OPP, opponent_id_to_idx=_OPP_ID_TO_IDX)
        gs = torch.zeros(3, GLOBAL_STATE_DIM)
        ctx = a.build_context(gs, [7, 8, 9])
        onehot = ctx[:, GLOBAL_STATE_DIM:]
        self.assertTrue(torch.equal(onehot, torch.eye(_N_OPP)))

    def test_unknown_opponent_zero_onehot(self) -> None:
        a = AdvantageRouter(n_opponents=_N_OPP, opponent_id_to_idx=_OPP_ID_TO_IDX)
        gs = torch.zeros(1, GLOBAL_STATE_DIM)
        ctx = a.build_context(gs, [-1])
        self.assertEqual(float(ctx[:, GLOBAL_STATE_DIM:].sum().item()), 0.0)


class V6i12DoubleCenteringTests(unittest.TestCase):
    """The core V6I12 claim: V(context) absorbs the context return variance."""

    def _context_dominated_records(self) -> list[dict]:
        # Return = large opponent-specific base + small z effect + tiny noise.
        rng = np.random.default_rng(0)
        opp_base = {7: -3.0, 8: 0.0, 9: 3.0}       # dominates raw variance
        z_effect = {0: -0.2, 1: 0.0, 2: 0.2, 3: 0.4}  # small latent signal
        recs: list[dict] = []
        uid = 0
        for opp in _OPP_RAW:
            for z in range(_K):
                for env in range(60):
                    ret = opp_base[opp] + z_effect[z] + float(rng.normal(0, 0.1))
                    recs.append(_arc_record(uid, z=z, ret=ret, opp=opp, env=env))
                    uid += 1
        return recs

    def test_advantage_target_std_below_normalized_return_std(self) -> None:
        replay = _build_replay(self._context_dominated_records())
        v = ContextualVBaseline(n_opponents=_N_OPP)
        a = AdvantageRouter(n_opponents=_N_OPP, opponent_id_to_idx=_OPP_ID_TO_IDX, latent_k=_K)
        v_opt = torch.optim.Adam(v.parameters(), lr=3e-3)
        a_opt = torch.optim.Adam(a.parameters(), lr=3e-3)
        tel = train_advantage_router(
            v, a, replay, v_opt, a_opt, batch_size=256, n_steps=300
        )
        # norm_ret has unit std by construction; if V explains the opponent
        # component, the advantage target std must fall clearly below 1.0.
        self.assertLess(
            tel["advantage_target_std_mean"], 0.9,
            f"V(context) failed to reduce target variance: {tel}",
        )
        self.assertGreater(tel["baseline_r2_mean"], 0.5, tel)
        self.assertTrue(np.isfinite(tel["v_loss_mean"]))
        self.assertTrue(np.isfinite(tel["a_loss_mean"]))
        self.assertGreater(tel["v_grad_norm"], 0.0)
        self.assertGreater(tel["a_grad_norm"], 0.0)

    def test_advantage_gap_ci_detects_real_separation(self) -> None:
        replay = _build_replay(self._context_dominated_records())
        v = ContextualVBaseline(n_opponents=_N_OPP)
        a = AdvantageRouter(n_opponents=_N_OPP, opponent_id_to_idx=_OPP_ID_TO_IDX, latent_k=_K)
        v_opt = torch.optim.Adam(v.parameters(), lr=3e-3)
        a_opt = torch.optim.Adam(a.parameters(), lr=3e-3)
        train_advantage_router(v, a, replay, v_opt, a_opt, batch_size=256, n_steps=300)
        gap = advantage_gap_ci(
            replay, v, a,
            n_opponents=_N_OPP, latent_k=_K,
            opponent_id_to_idx=_OPP_ID_TO_IDX, n_boot=500, seed=0,
        )
        # z3 is best in every opponent cell; the gap CI should exclude zero.
        passing = sum(1 for c in gap.values() if c.get("ci_excludes_zero"))
        self.assertGreaterEqual(passing, 2, gap)

    def test_advantage_gap_ci_flat_when_z_independent(self) -> None:
        rng = np.random.default_rng(1)
        recs: list[dict] = []
        uid = 0
        for opp in _OPP_RAW:
            for z in range(_K):
                for env in range(60):
                    # Return depends only on opponent + noise; z is irrelevant.
                    base = {7: -3.0, 8: 0.0, 9: 3.0}[opp]
                    ret = base + float(rng.normal(0, 1.0))
                    recs.append(_arc_record(uid, z=z, ret=ret, opp=opp, env=env))
                    uid += 1
        replay = _build_replay(recs)
        v = ContextualVBaseline(n_opponents=_N_OPP)
        a = AdvantageRouter(n_opponents=_N_OPP, opponent_id_to_idx=_OPP_ID_TO_IDX, latent_k=_K)
        v_opt = torch.optim.Adam(v.parameters(), lr=3e-3)
        a_opt = torch.optim.Adam(a.parameters(), lr=3e-3)
        train_advantage_router(v, a, replay, v_opt, a_opt, batch_size=256, n_steps=200)
        gap = advantage_gap_ci(
            replay, v, a,
            n_opponents=_N_OPP, latent_k=_K,
            opponent_id_to_idx=_OPP_ID_TO_IDX, n_boot=500, seed=0,
        )
        passing = sum(1 for c in gap.values() if c.get("ci_excludes_zero"))
        self.assertLessEqual(passing, 1, gap)


class V6i12StopGradTests(unittest.TestCase):
    """A's loss must not backprop into the V-baseline (stopgrad target)."""

    def test_a_loss_does_not_touch_v_gradients(self) -> None:
        recs = [
            _arc_record(i, z=i % _K, ret=float(i % 5), opp=_OPP_RAW[i % _N_OPP], env=i)
            for i in range(200)
        ]
        replay = _build_replay(recs)
        v = ContextualVBaseline(n_opponents=_N_OPP)
        a = AdvantageRouter(n_opponents=_N_OPP, opponent_id_to_idx=_OPP_ID_TO_IDX, latent_k=_K)

        ctx, z, ret, _ = replay.sample(128)
        norm_ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        with torch.no_grad():
            v_pred = v(ctx)
        a_target = norm_ret.detach() - v_pred.detach()
        a_pred = a(ctx).gather(1, z.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.huber_loss(a_pred, a_target)
        loss.backward()
        for p in v.parameters():
            self.assertIsNone(p.grad, "V-baseline received gradient from A loss")
        self.assertTrue(any(p.grad is not None for p in a.parameters()))


if __name__ == "__main__":
    unittest.main()
