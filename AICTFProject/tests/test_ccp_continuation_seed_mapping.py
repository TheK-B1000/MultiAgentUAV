"""The continuation seed mapping must be a pure, frozen function of (state_id, j).

CCP_PHASE1_CAUSAL_BRANCHING_SPEC.json#AMENDMENT_1 requires r_j = H('CCP_PHASE1', state_id, j)
with four properties: both branches receive the same r_j; r_j does not depend on which policy
ran first; r_j does not depend on collection order, wall-clock or process state; and the
mapping is reproducible from the artifact alone.

The continuation-entropy smoke validates that RESEEDING works, but it runs a single policy,
so it cannot establish PAIRING. Pairing has two halves:

  this file          the seed mapping is pure, order-independent and frozen
  the collector      a fail-closed runtime assertion that both branches got the same r_j

Expected values are hardcoded. That is the point: it locks the mapping against silent change,
so a bank collected today stays reproducible against a bank collected later.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SMOKE = ROOT / "experiments" / "ccp_phase1_continuation_entropy.py"


def _seed_fn():
    spec = importlib.util.spec_from_file_location("ccp_entropy_seedmap", SMOKE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.continuation_seed


SEED = _seed_fn()          # module level: storing it on the class would bind it as a method


class ContinuationSeedMappingTests(unittest.TestCase):

    def test_frozen_values(self):
        """Hardcoded so the mapping cannot drift and silently orphan an earlier bank."""
        expected = {
            ("11500021|A|40", 0): 3584840458435560518,
            ("11500021|A|40", 1): 3470152098775640227,
            ("11500021|A|80", 0): 4682613420261417924,
            ("11500021|A|120", 31): 7814427732118546107,
        }
        for (state_id, j), want in expected.items():
            self.assertEqual(SEED(state_id, j), want,
                             f"seed mapping changed for ({state_id}, {j}); an existing "
                             f"continuation bank would no longer be reproducible")

    def test_pure_and_order_independent(self):
        """Same inputs, same output, regardless of the order they are requested in."""
        forward = [SEED("s|A|10", j) for j in range(16)]
        backward = [SEED("s|A|10", j) for j in reversed(range(16))][::-1]
        interleaved = []
        for j in range(16):
            SEED("decoy|B|99", j)                 # unrelated calls in between
            interleaved.append(SEED("s|A|10", j))
        self.assertEqual(forward, backward)
        self.assertEqual(forward, interleaved)

    def test_both_branches_of_a_state_get_the_same_seed(self):
        """The pairing property: r_j depends on (state_id, j) only -- never on the policy."""
        for j in range(8):
            branch_A = SEED("11500021|A|40", j)
            branch_B = SEED("11500021|A|40", j)
            self.assertEqual(branch_A, branch_B)

    def test_distinct_inputs_give_distinct_seeds(self):
        seeds_j = {SEED("11500021|A|40", j) for j in range(64)}
        self.assertEqual(len(seeds_j), 64, "seed collisions across j")
        seeds_s = {SEED(f"11500021|A|{b}", 0) for b in range(64)}
        self.assertEqual(len(seeds_s), 64, "seed collisions across states")

    def test_in_positive_int64_range(self):
        for j in (0, 1, 31, 1000):
            r = SEED("11500021|A|40", j)
            self.assertGreaterEqual(r, 0)
            self.assertLess(r, 2 ** 63 - 1, "must fit torch.Generator.manual_seed")

    # ---- the pairing CONTRACT, not just the mapping -------------------------

    def test_pairing_ledger_both_sides_receive_the_same_seed(self):
        """Record (state_id, j, r_j, seed_A, seed_B) and require seed_A == seed_B == r_j.

        Dispensed in BOTH branch orders. If seeds were drawn from a stream rather than
        computed from (state_id, j), reversing which side asks first would hand the sides
        different values and D_j would stop being a paired difference.
        """
        ledger = []
        for state_id in ("11500021|A|40", "11500021|A|80", "11500033|B|17"):
            for j in range(6):
                r_j = SEED(state_id, j)
                seed_A = SEED(state_id, j)                       # A asks first
                seed_B = SEED(state_id, j)
                ledger.append((state_id, j, r_j, seed_A, seed_B))
                # and again with the order reversed
                seed_B2 = SEED(state_id, j)                      # B asks first
                seed_A2 = SEED(state_id, j)
                ledger.append((state_id, j, r_j, seed_A2, seed_B2))
        self.assertEqual(len(ledger), 36)
        for state_id, j, r_j, seed_A, seed_B in ledger:
            self.assertEqual(seed_A, r_j, f"side A diverged at ({state_id}, {j})")
            self.assertEqual(seed_B, r_j, f"side B diverged at ({state_id}, {j})")

    def test_a_stream_based_seed_source_would_FAIL_this_check(self):
        """Negative control. A guard that cannot fail proves nothing.

        Substitutes an order-dependent seed source -- the shape a naive implementation would
        take -- and requires the pairing assertion to catch it.
        """
        counter = iter(range(10_000))
        stream_seed = lambda state_id, j: next(counter)          # noqa: E731

        caught = False
        for state_id in ("s|A|1",):
            for j in range(3):
                r_j = stream_seed(state_id, j)
                seed_A = stream_seed(state_id, j)
                seed_B = stream_seed(state_id, j)
                if not (seed_A == seed_B == r_j):
                    caught = True
        self.assertTrue(caught, "the pairing check failed to detect an order-dependent seed "
                                "source, so it proves nothing about the real mapping")

    def test_reproduces_on_a_fresh_module_load(self):
        """Artifact-deterministic: a fresh run must reproduce the same tuples."""
        fresh = _seed_fn()
        self.assertIsNot(fresh, SEED)
        for state_id in ("11500021|A|40", "11500021|A|120"):
            for j in (0, 5, 31):
                self.assertEqual(fresh(state_id, j), SEED(state_id, j),
                                 "seed mapping is not reproducible across a fresh load")


if __name__ == "__main__":
    unittest.main()
