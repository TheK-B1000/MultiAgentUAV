"""Router credit-assignment audit and histogram-preserving shuffle tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.custom_ppo.diagnostics.router_credit_audit import (
    audit_feedforward_router_credit_wiring,
    run_synthetic_router_sign_test,
)
from rl.evaluation.router_ablation import (
    build_shuffled_mapping_from_learned_traces,
    learned_z_histogram_from_traces,
    validate_shuffled_mapping_histogram,
)
from rl.latent_marl import StrategyEncoder


def _sample_traces() -> list[dict]:
    traces = []
    for opp in ("OP8", "OP9"):
        for seed in (1000, 1001):
            seq = [0, 3, 3, 2, 0, 1, 3, 2]
            for opp_idx, z in enumerate(seq):
                traces.append(
                    {
                        "opponent": opp,
                        "seed": seed,
                        "episode_index": 0,
                        "opportunity_index": opp_idx,
                        "selected_z": z,
                        "logits": [0.1, 0.2, 0.3, 0.4],
                        "probabilities": [0.1, 0.2, 0.3, 0.4],
                    }
                )
    return traces


class TestHistogramPreservingShuffle(unittest.TestCase):
    def test_global_histogram_matches_learned(self) -> None:
        traces = _sample_traces()
        mapping, meta = build_shuffled_mapping_from_learned_traces(
            traces,
            latent_k=4,
            switch_cadence=32,
        )
        learned_hist = learned_z_histogram_from_traces(traces)
        self.assertEqual(meta["learned_z_histogram"], dict(learned_hist))
        self.assertTrue(meta["histogram_preserved"])
        validation = validate_shuffled_mapping_histogram(traces, mapping)
        self.assertTrue(validation["histogram_preserved"])
        self.assertGreater(validation["reassigned_episode_count"], 0)

    def test_per_episode_multiset_preserved(self) -> None:
        traces = _sample_traces()
        mapping, _ = build_shuffled_mapping_from_learned_traces(
            traces,
            latent_k=4,
            switch_cadence=32,
        )
        for opp in ("OP8", "OP9"):
            for seed in (1000, 1001):
                key = (opp, seed, 0)
                learned_seq = [
                    int(t["selected_z"])
                    for t in traces
                    if t["opponent"] == opp and t["seed"] == seed
                ]
                shuffled_seq = [int(d["selected_z"]) for d in mapping[key][: len(learned_seq)]]
                self.assertEqual(sorted(learned_seq), sorted(shuffled_seq))
                if len(set(learned_seq)) > 1:
                    self.assertNotEqual(learned_seq, shuffled_seq)


class TestSyntheticRouterSignTest(unittest.TestCase):
    def test_update_direction_follows_advantage_sign(self) -> None:
        encoder = StrategyEncoder(state_dim=32, latent_k=4, hidden=16)
        result = run_synthetic_router_sign_test(
            encoder,
            context_dim=32,
            latent_k=4,
            lr=0.1,
        )
        self.assertTrue(result.passed, msg=f"P(z|ctx) did not move with advantage sign: {result}")
        self.assertTrue(result.reversed_passed, msg="Reversed-advantage update did not flip direction")


class TestFeedforwardCreditWiringAudit(unittest.TestCase):
    def test_flags_missing_router_advantages(self) -> None:
        class _Cfg:
            recurrent_selector_hidden_dim = 0
            router_reward_enabled = True

        batch: dict[str, torch.Tensor] = {"router_reward": torch.zeros(4)}
        out = audit_feedforward_router_credit_wiring(_Cfg(), batch)
        self.assertIsNotNone(out["credit_wiring_issue"])

    def test_passes_when_router_advantages_present(self) -> None:
        class _Cfg:
            recurrent_selector_hidden_dim = 0
            router_reward_enabled = True

        batch = {
            "router_advantages": torch.zeros(4),
            "router_reward": torch.zeros(4),
            "advantages": torch.zeros(4),
        }
        out = audit_feedforward_router_credit_wiring(_Cfg(), batch)
        self.assertIsNone(out["credit_wiring_issue"])
        self.assertEqual(out["strategy_ppo_advantage_source"], "router")


if __name__ == "__main__":
    unittest.main()
