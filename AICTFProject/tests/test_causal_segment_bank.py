"""Verify the SEQUENCE-mode segment bank against the REAL frozen Phase 1 result.

Not built blind -- Phase 1 is done and frozen (CCP_PHASE1_CAUSAL_BRANCHING.json,
DECISION_LEVEL_LEVERAGE). This tests the loader's amendments 2-3 rules against the actual
bank, including the one real joint-vs-individual conflict the data contains.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT = ROOT / "artifacts" / "strategic_demand" / "sppo" / "CCP_PHASE1_CAUSAL_BRANCHING.json"


@unittest.skipUnless(RESULT.is_file(), "Phase 1 result not present")
class CausalSegmentBankTests(unittest.TestCase):

    def test_exactly_one_segment_per_pilot_state(self):
        """14 one-free states + 6 both-free states (joint precedence) = 20, always."""
        from rl.causal_segment_bank import build_segment_bank
        bank = build_segment_bank(RESULT)
        self.assertEqual(len(bank), 20)
        start_states = [s.start_state_id for s in bank]
        self.assertEqual(len(set(start_states)), 20, "a state produced more than one segment")

    def test_known_conflict_state_resolves_to_joint_only(self):
        """11500105|B|133: agent0-alone implies pi_A, joint implies pi_B for both agents.
        Joint precedence must win, and the individual segment must not exist in the bank."""
        from rl.causal_segment_bank import build_segment_bank
        bank = build_segment_bank(RESULT)
        at_state = [s for s in bank if s.start_state_id == "11500105|B|133"]
        self.assertEqual(len(at_state), 1, "joint precedence did not collapse to one segment")
        seg = at_state[0]
        self.assertEqual(seg.controlled_agents, (0, 1))
        self.assertEqual(seg.delta_q, 1.0)
        self.assertEqual(seg.teacher, "pi_B")
        self.assertNotIn("agent0", seg.segment_id)
        self.assertNotIn("agent1", seg.segment_id)

    def test_no_individual_segment_exists_wherever_a_joint_segment_does(self):
        from rl.causal_segment_bank import build_segment_bank
        bank = build_segment_bank(RESULT)
        joint_states = {s.start_state_id for s in bank if s.controlled_agents == (0, 1)}
        individual_states = {s.start_state_id for s in bank if len(s.controlled_agents) == 1}
        self.assertEqual(joint_states & individual_states, set(),
                         "a state has both a joint and an individual segment")

    def test_every_segment_passes_its_own_routing_assertion(self):
        from rl.causal_segment_bank import build_segment_bank
        bank = build_segment_bank(RESULT)
        for seg in bank:
            seg.assert_routing()          # must not raise

    def test_latent_never_flips_and_teacher_follows_sign(self):
        from rl.causal_segment_bank import build_segment_bank
        bank = build_segment_bank(RESULT)
        for seg in bank:
            expect_latent = 0 if seg.pole == "A" else 1
            self.assertEqual(seg.latent, expect_latent)
            if seg.delta_q > 0:
                self.assertEqual(seg.teacher, "pi_A" if seg.pole == "A" else "pi_B")
            elif seg.delta_q < 0:
                self.assertEqual(seg.teacher, "pi_B" if seg.pole == "A" else "pi_A")
            else:
                self.assertIsNone(seg.teacher)

    def test_deterministic_and_reproducible(self):
        from rl.causal_segment_bank import build_segment_bank
        a = build_segment_bank(RESULT)
        b = build_segment_bank(RESULT)
        self.assertEqual([s.segment_id for s in a], [s.segment_id for s in b])
        self.assertEqual([s.delta_q for s in a], [s.delta_q for s in b])

    def test_active_until_is_episode_termination_not_invented(self):
        """active_until=None is the frozen encoding for 'through episode end', per
        amendment 2 -- not a placeholder awaiting a value."""
        from rl.causal_segment_bank import build_segment_bank
        bank = build_segment_bank(RESULT)
        for seg in bank:
            self.assertIsNone(seg.active_until)

    def test_two_significant_local_contrasts_are_not_in_the_bank(self):
        """SEQUENCE mode trains on full_takeover only; the significant single_macro
        contrasts (11500105|B|133 agent1, joint) must not appear as their own segments --
        that state IS in the bank, but as the joint full_takeover segment, not a
        single_macro one."""
        from rl.causal_segment_bank import build_segment_bank
        bank = build_segment_bank(RESULT)
        for seg in bank:
            self.assertIn("full_takeover", seg.segment_id)
            self.assertNotIn("single_macro", seg.segment_id)


if __name__ == "__main__":
    unittest.main()
