"""Stage-4 contracts: dual-branch teachers, z+r vs role-only, no Generalist."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class Stage4TeacherDistillTests(unittest.TestCase):
    def test_splice_logits_by_role(self):
        from rl.teacher_distillation import splice_logits_by_role

        # 2 agents, 1 head each, action dim 3
        b, n = 4, 2
        ld = [torch.zeros(b, 3), torch.ones(b, 3)]
        la = [torch.full((b, 3), 2.0), torch.full((b, 3), 3.0)]
        roles = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        out = splice_logits_by_role(ld, la, roles, n)
        self.assertEqual(len(out), 2)
        # batch0 roles=[DEFEND, ATTACK]
        self.assertTrue(torch.equal(out[0][0], torch.zeros(3)))
        self.assertTrue(torch.equal(out[1][0], torch.full((3,), 3.0)))
        # batch1 roles=[ATTACK, DEFEND]
        self.assertTrue(torch.equal(out[0][1], torch.full((3,), 2.0)))
        self.assertTrue(torch.equal(out[1][1], torch.ones(3)))

    def test_role_only_kwargs_drop_z(self):
        from rl.suite_fully_shared_distill import role_only_model_kwargs, fully_shared_model_kwargs

        base = {
            "n_agents": 6, "entity_repair_enabled": True, "entity_hidden_dim": 32,
            "latent_k": 0, "role_conditioning_enabled": False,
        }
        zr = fully_shared_model_kwargs(base, role_conditioning=True)
        ro = role_only_model_kwargs(base)
        self.assertEqual(zr["latent_k"], 2)
        self.assertTrue(zr["role_conditioning_enabled"])
        self.assertEqual(ro["latent_k"], 0)
        self.assertTrue(ro["role_conditioning_enabled"])


class Stage4OrchestratorContractTests(unittest.TestCase):
    def test_auth_record_exists(self):
        p = ROOT / "artifacts/strategic_demand/sppo/STAGE4_6V6_SCHOOL_SUITE_SPEC.json"
        self.assertTrue(p.is_file())
        doc = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(doc["status"], "AUTHORIZED_FOR_SCHOOL_PC")
        self.assertIn("Ours-Shared", " ".join(doc["LADDER_locked"]))
        self.assertIn("Fully Shared+z+r", " ".join(doc["LADDER_locked"]))
        self.assertTrue(any("Role-only" in x for x in doc["LADDER_locked"]))
        self.assertTrue(any("Generalist" in x for x in doc["NOT_AUTHORIZED"]))
        framing = ROOT / "artifacts/strategic_demand/sppo/EXPERIMENTAL_FRAMING_OURS_TEACHERS_SHARED_V1.json"
        self.assertTrue(framing.is_file())
        fr = json.loads(framing.read_text(encoding="utf-8"))
        self.assertIn("Ours-Shared", fr["OURS_FORMS_locked"])
        self.assertEqual(
            fr["SECTION_NAMING_locked"]["new_title"],
            "Strategic Representation Under Parameter Sharing",
        )

    def test_seed_blocks_do_not_collide_with_dual_branch_train(self):
        from experiments import prepare_stage4_baselines as P4
        bl = P4.blocks(6)
        # Dual-branch train uses 26900005/6; Stage4 collection starts at 27002001 lane.
        for eid, lo, hi in bl.values():
            self.assertFalse(26900001 <= lo <= 26900006)
            self.assertFalse(26900001 <= hi <= 26900006)
            self.assertGreaterEqual(lo, 27_000_000)

    def test_orchestrator_chains_stage4_by_default(self):
        text = (ROOT / "6v6" / "run_dual_branch_6v6.py").read_text(encoding="utf-8")
        self.assertIn("technical_seal", text)
        self.assertIn("stage4_dataset", text)
        self.assertIn("Ours-Shared", text)
        self.assertIn("Fully Shared+z+r", text)
        self.assertIn("role_only", text)
        self.assertIn("ugly Delta does NOT stop", text)
        self.assertIn("--skip-stage4", text)
        self.assertNotIn("Sharing ladder is a later stage", text)

    def test_2v2_suite_matches_scale_and_waits(self):
        from experiments import prepare_stage4_baselines as P4
        self.assertEqual(P4.k_sym(2), 1)
        self.assertIn("phase6_package.json", (ROOT / "4v4" / "run_dual_branch_4v4.py").read_text(encoding="utf-8"))
        self.assertEqual(P4.k_sym(4), 2)
        self.assertEqual(P4.k_sym(6), 2)
        text = (ROOT / "2v2" / "run_dual_branch_2v2.py").read_text(encoding="utf-8")
        self.assertIn("SYM_DUAL_BRANCH_2V2_TRAIN_DONE.txt", text)
        self.assertIn("WAITING", text)
        self.assertIn("ugly Delta does NOT stop", text)
        self.assertIn("role_only", text)
        self.assertNotIn('"arm", "generalist"', text)
        reg, primary = P4.SPENT_EVAL[2]
        self.assertEqual(reg, "STANDARDIZED_2V2_SPLIT_K1_CONFIRMATORY_SPECIALIST_CROSSOVER")
        self.assertIn("CONFIRMATORY", primary)
        for _eid, lo, _hi in P4.blocks(2).values():
            self.assertGreaterEqual(lo, 27_000_000)
            self.assertFalse(26900001 <= lo <= 26900002)


if __name__ == "__main__":
    unittest.main()
