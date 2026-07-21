"""Pinning tests for the V6I11 contextual Q-value router extension.

Covers the 2026-07-03 hardening pass:

1. Preset EPISODE-PERSISTENT horizon contract (arc == episode) + "internal
   router receives no gradient" invariant.
2. Extraction-before-drain: copied arc records survive a source-buffer reset.
3. Zero-arc / no-insert hard failure aborts (never emits FLAT).
4. Duplicate insertion is rejected by stable record_id.
5. Episode-horizon signal: terminal-finalized fraction + mean arc length.
6. Coverage verdict: a missing z row yields INSUFFICIENT_DATA.
7. Reliable vs noisy separation: clear z differences -> SEPARATING; a large
   apparent mean spread swamped by variance does NOT pass.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from rl.global_state import GLOBAL_STATE_DIM
from rl.router.q_value_router import (
    ArcIntegrityError,
    ContextualQRouter,
    QRouterReplayBuffer,
    check_arc_guards,
    copy_arc_record,
    decide_verdict,
)


# Canonical scheme: OP8->7, OP9->8, OP10->9 (see csv_writers._OPPONENT_TAG_TO_ID).
_OPP_ID_TO_IDX = {7: 0, 8: 1, 9: 2}
_OPP_NAMES = {0: "OP8", 1: "OP9", 2: "OP10"}
_K = 4
_N_OPP = 3
_CTX_DIM = GLOBAL_STATE_DIM + _N_OPP


def _make_router() -> ContextualQRouter:
    return ContextualQRouter(n_opponents=_N_OPP, opponent_id_to_idx=_OPP_ID_TO_IDX, latent_k=_K)


def _arc_record(uid: int, *, z: int, ret: float, opp: int, env: int = 0,
                length: int = 130, reason: str = "episode_end") -> dict:
    return {
        "global_state_0": torch.randn(GLOBAL_STATE_DIM),
        "z": int(z),
        "arc_return": float(ret),
        "opponent_id": int(opp),
        "arc_length": int(length),
        "reason": reason,
        "env_index": int(env),
        "arc_uid": int(uid),
    }


def _spread(mean_mat: np.ndarray) -> dict:
    out = {}
    for oi in range(_N_OPP):
        row = mean_mat[oi]
        out[f"spread_{_OPP_NAMES[oi]}"] = (
            float(np.nanmax(row) - np.nanmin(row)) if not np.all(np.isnan(row)) else float("nan")
        )
    return out


def _buffer_from(records: list[dict]) -> tuple[QRouterReplayBuffer, dict]:
    q = _make_router()
    rb = QRouterReplayBuffer(capacity=20000, context_dim=_CTX_DIM, latent_k=_K)
    stats = rb.push_many(
        records, rollout_index=1,
        opponent_id_to_idx=_OPP_ID_TO_IDX,
        build_context=q.build_context_from_record,
    )
    return rb, stats


class V6i11PresetHorizonContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = apply_preset(PPOConfig(), "v6i11_q_router_hardpool")

    def test_episode_persistent_contract(self) -> None:
        self.assertEqual(self.cfg.strategy_interval, 0)
        self.assertEqual(self.cfg.latent_resample_every_n, 0)
        self.assertEqual(self.cfg.latent_arc_credit_min_len, 1)
        self.assertEqual(self.cfg.recurrent_selector_hidden_dim, 0)

    def test_internal_router_receives_no_gradient(self) -> None:
        self.assertEqual(self.cfg.latent_arc_credit_coef, 0.0)
        self.assertEqual(self.cfg.latent_strategy_ppo_coef, 0.0)
        self.assertEqual(self.cfg.router_ent_coef, 0.0)
        self.assertEqual(self.cfg.latent_lam_h, 0.0)
        self.assertEqual(self.cfg.latent_lam_h_end, 0.0)
        self.assertEqual(self.cfg.latent_lam_p, 0.0)

    def test_data_collection_and_exploration(self) -> None:
        self.assertTrue(self.cfg.latent_arc_credit_enabled)
        self.assertTrue(self.cfg.router_freeze_actor)
        self.assertAlmostEqual(self.cfg.router_uniform_exploration_prob, 0.5)


class V6i11ExtractionBeforeDrainTests(unittest.TestCase):
    def test_copied_records_survive_source_reset(self) -> None:
        # Simulate rollout_strategy_arc_records; copy; then "drain" the source.
        source = [_arc_record(0, z=1, ret=2.0, opp=7)]
        copies = [copy_arc_record(r) for r in source]
        original_tensor = source[0]["global_state_0"]
        # The trainer's reset rebinds the attribute AND we mutate the source in place.
        source[0]["global_state_0"].zero_()
        source.clear()
        # Copy is independent: tensor content preserved, dict intact.
        self.assertFalse(torch.equal(copies[0]["global_state_0"], torch.zeros_like(original_tensor)))
        self.assertEqual(copies[0]["z"], 1)
        self.assertEqual(copies[0]["arc_uid"], 0)


class V6i11HardGuardTests(unittest.TestCase):
    def test_zero_arcs_raises(self) -> None:
        with self.assertRaises(ArcIntegrityError):
            check_arc_guards(records_before_update=0, inserted=0, size_before=0, size_after=0)

    def test_no_insert_raises(self) -> None:
        with self.assertRaises(ArcIntegrityError):
            check_arc_guards(records_before_update=5, inserted=0, size_before=10, size_after=10)

    def test_healthy_passes(self) -> None:
        check_arc_guards(records_before_update=5, inserted=5, size_before=10, size_after=15)


class V6i11DuplicateRejectionTests(unittest.TestCase):
    def test_same_record_id_rejected(self) -> None:
        rb, _ = _buffer_from([_arc_record(7, z=1, ret=2.0, opp=7)])
        size_before = len(rb)
        # Re-push the SAME arc_uid (identity) -> rejected, size unchanged.
        again = rb.push_many(
            [_arc_record(7, z=1, ret=99.0, opp=7)], rollout_index=1,
            opponent_id_to_idx=_OPP_ID_TO_IDX,
            build_context=_make_router().build_context_from_record,
        )
        self.assertEqual(again["inserted"], 0)
        self.assertEqual(again["duplicates_rejected"], 1)
        self.assertEqual(len(rb), size_before)


class V6i11EpisodeHorizonTests(unittest.TestCase):
    def test_terminal_fraction_and_arc_length(self) -> None:
        recs = [_arc_record(i, z=i % _K, ret=float(i % 3), opp=7 + (i % 3),
                            length=130, reason="episode_end") for i in range(60)]
        # Contaminate with a couple of mid-episode (z_change) arcs.
        recs += [_arc_record(1000 + i, z=0, ret=0.0, opp=7, length=32, reason="z_change")
                 for i in range(2)]
        rb, _ = _buffer_from(recs)
        v = rb.validity_report(n_opponents=_N_OPP, latent_k=_K, opponent_id_to_idx=_OPP_ID_TO_IDX)
        self.assertGreater(v["terminal_finalized_fraction"], 0.9)
        self.assertGreater(v["mean_arc_length"], 100.0)
        self.assertTrue(v["no_duplicate_arcs"])


class V6i11VerdictTests(unittest.TestCase):
    def _verdict_for(self, records: list[dict], *, threshold: float = 0.10) -> str:
        rb, _ = _buffer_from(records)
        mean_mat, count_mat = rb.mean_return_matrix(
            n_opponents=_N_OPP, latent_k=_K, opponent_id_to_idx=_OPP_ID_TO_IDX)
        min_cell = float(np.nanmin(count_mat[count_mat > 0]) if np.any(count_mat > 0) else 0)
        v = decide_verdict(
            validity=rb.validity_report(
                n_opponents=_N_OPP, latent_k=_K, opponent_id_to_idx=_OPP_ID_TO_IDX),
            gap_ci=rb.best_second_gap_ci(
                n_opponents=_N_OPP, latent_k=_K, opponent_id_to_idx=_OPP_ID_TO_IDX,
                n_boot=400, seed=0),
            spread=_spread(mean_mat),
            spread_threshold=threshold, min_cell_arcs=min_cell,
            n_opponents=_N_OPP, opp_names=_OPP_NAMES,
        )
        return v[0]

    def test_missing_z_is_insufficient_data(self) -> None:
        # Only z in {0,1,2} ever appears; z3 never -> coverage fails.
        rng = np.random.default_rng(0)
        recs = [_arc_record(i, z=int(rng.integers(0, 3)), ret=float(rng.normal()),
                            opp=7 + (i % 3)) for i in range(300)]
        self.assertEqual(self._verdict_for(recs), "INSUFFICIENT_DATA")

    def test_clear_separation_is_separating(self) -> None:
        rng = np.random.default_rng(1)
        recs = []
        uid = 0
        for _ in range(400):
            for oi, opp in enumerate((7, 8, 9)):
                z = int(rng.integers(0, _K))
                # OP8 and OP9 have z1 clearly best; OP10 flat.
                if oi in (0, 1) and z == 1:
                    ret = 2.0 + rng.normal(0, 0.3)
                else:
                    ret = rng.normal(0, 0.3)
                recs.append(_arc_record(uid, z=z, ret=ret, opp=opp)); uid += 1
        self.assertEqual(self._verdict_for(recs), "SEPARATING")

    def test_noisy_overlap_does_not_separate(self) -> None:
        # Apparent mean spread but HUGE variance -> CI includes zero -> not SEPARATING.
        rng = np.random.default_rng(2)
        recs = []
        uid = 0
        for _ in range(400):
            for oi, opp in enumerate((7, 8, 9)):
                z = int(rng.integers(0, _K))
                mu = 0.15 if z == 1 else 0.0  # small mean gap
                ret = mu + rng.normal(0, 5.0)  # swamping variance
                recs.append(_arc_record(uid, z=z, ret=ret, opp=opp)); uid += 1
        self.assertIn(self._verdict_for(recs), ("FLAT", "WEAK_SEPARATION"))


class V6i11OpponentContextWiringTests(unittest.TestCase):
    """Pin the opponent one-hot: the canonical id scheme must produce a nonzero
    one-hot in the correct row, and an unmapped/-1 id must be all-zero (so the
    replay-validity guards catch missing opponent attribution instead of the
    Q-router silently training on a geometry-only, opponent-blind context)."""

    def test_canonical_ids_set_correct_onehot_row(self) -> None:
        q = _make_router()
        gs = torch.zeros(1, GLOBAL_STATE_DIM)
        # OP8->7 (row 0), OP9->8 (row 1), OP10->9 (row 2).
        for raw, idx in _OPP_ID_TO_IDX.items():
            ctx = q.build_context(gs, [raw])
            onehot = ctx[0, GLOBAL_STATE_DIM:]
            self.assertEqual(int(onehot.argmax()), idx)
            self.assertAlmostEqual(float(onehot.sum()), 1.0)

    def test_unmapped_opponent_is_all_zero_onehot(self) -> None:
        q = _make_router()
        gs = torch.zeros(1, GLOBAL_STATE_DIM)
        # -1 is what _opponent_id_int_from_info returns for unknown/non-scripted,
        # and 10 (would be OP11) is outside the OP8/9/10 grid.
        for bad in (-1, 10):
            ctx = q.build_context(gs, [bad])
            onehot = ctx[0, GLOBAL_STATE_DIM:]
            self.assertAlmostEqual(float(onehot.sum()), 0.0)

    def test_default_map_matches_canonical_scheme(self) -> None:
        from rl.router.q_value_router import _DEFAULT_OPPONENT_ID_TO_IDX
        self.assertEqual(_DEFAULT_OPPONENT_ID_TO_IDX, {7: 0, 8: 1, 9: 2})


if __name__ == "__main__":
    unittest.main()
