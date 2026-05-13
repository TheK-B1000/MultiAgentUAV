"""Guards for OP35 latent preflight / training command matrix."""

from __future__ import annotations

import unittest

from experiments import op35_latent_matrix as m


class Op35LatentMatrixTests(unittest.TestCase):
    def test_default_flat_checkpoint_is_two_blade_hypothesis_run(self) -> None:
        self.assertTrue(
            m.DEFAULT_FLAT_CHECKPOINT_TWO_BLADE.endswith(
                "final_research_hypothesis_flat_opprand_seed42_2v2.zip"
            )
        )
        self.assertIn("hypothesis_runs", m.DEFAULT_FLAT_CHECKPOINT_TWO_BLADE.replace("\\", "/"))
        self.assertIn("20260509_103737", m.DEFAULT_FLAT_CHECKPOINT_TWO_BLADE)

    def test_optional_op35_aligned_flat_path(self) -> None:
        self.assertTrue(
            m.OPTIONAL_FLAT_CHECKPOINT_OP35_ALIGNED.endswith("final_paper_flat_op35_seed42_2v2.zip")
        )
        self.assertIn("paper_runs", m.OPTIONAL_FLAT_CHECKPOINT_OP35_ALIGNED.replace("\\", "/"))

    def test_preflight_with_e3_csv_emits_analyze_not_collect(self) -> None:
        rows = m._preflight_rows(
            python_exe="python",
            flat_ckpt="flat.zip",
            latent_ckpt="latent.zip",
            latent_e3_csv="telemetry_e3_steps.csv",
            agents=2,
            eval_episodes=50,
            device="cpu",
            eval_seed=1,
            collect_e3_steps=4096,
            collect_run_tag="collect_tag",
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1].step, "analyze_e3_mi")
        self.assertIn("analyze_e3_latent_mi.py", rows[1].command)
        self.assertIn("telemetry_e3_steps.csv", rows[1].command)
        self.assertNotIn("train_ppo.py", rows[1].command)

    def test_preflight_without_e3_csv_emits_short_collect(self) -> None:
        rows = m._preflight_rows(
            python_exe="python",
            flat_ckpt="flat.zip",
            latent_ckpt="latent.zip",
            latent_e3_csv=None,
            agents=2,
            eval_episodes=50,
            device="cpu",
            eval_seed=1,
            collect_e3_steps=4096,
            collect_run_tag="collect_tag",
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1].step, "collect_e3_then_mi")
        self.assertIn("rl/train_ppo.py", rows[1].command)
        self.assertIn("--e3-step-telemetry", rows[1].command)
        self.assertIn("latent_a1_plan_faithful", rows[1].command)

    def test_preflight_eval_uses_op3_and_op5(self) -> None:
        rows = m._preflight_rows(
            python_exe="python",
            flat_ckpt="flat.zip",
            latent_ckpt="latent.zip",
            latent_e3_csv="e3.csv",
            agents=2,
            eval_episodes=10,
            device="cpu",
            eval_seed=0,
            collect_e3_steps=100,
            collect_run_tag="x",
        )
        self.assertIn("plot/eval_checkpoint.py", rows[0].command)
        self.assertIn("flat.zip", rows[0].command)
        self.assertIn("OP3", rows[0].command)
        self.assertIn("OP5_RUSHER", rows[0].command)

    def test_train_rows_use_op35_latent_preset_only(self) -> None:
        rows = m._train_rows(
            python_exe="python",
            seeds=[42, 43],
            agents=2,
            device="cpu",
            train_tag_prefix="pfx",
            e3_telemetry=False,
        )
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertIn("hypothesis_latent_opprand_optionb_lamp_coef05_op35", r.command)
            self.assertNotIn("latent_a1_plan_faithful", r.command)
            self.assertIn("--run-tag", r.command)

    def test_train_rows_optional_e3_flag(self) -> None:
        rows = m._train_rows(
            python_exe="python",
            seeds=[7],
            agents=2,
            device="cpu",
            train_tag_prefix="t",
            e3_telemetry=True,
        )
        self.assertIn("--e3-step-telemetry", rows[0].command)

    def test_rows_for_execute_skip_train(self) -> None:
        rows = [
            m.MatrixRow("1_preflight", "a", "d", ("echo", "a")),
            m.MatrixRow("3_train", "b", "d", ("echo", "b")),
        ]
        self.assertEqual(len(m._rows_for_execute(rows, skip_train_on_execute=True)), 1)
        self.assertEqual(m._rows_for_execute(rows, skip_train_on_execute=False), rows)

    def test_resolved_flat_checkpoint(self) -> None:
        self.assertEqual(m._resolved_flat_checkpoint(""), m.DEFAULT_FLAT_CHECKPOINT_TWO_BLADE)
        self.assertEqual(m._resolved_flat_checkpoint("  x.zip  "), "x.zip")

    def test_split_cmd_roundtrip(self) -> None:
        parts = ["python", "plot/x.py", "--checkpoint", "plain.zip"]
        line = m._quote(parts)
        back = m._split_cmd(line)
        self.assertEqual(back, parts)


if __name__ == "__main__":
    unittest.main()
