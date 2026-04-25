import argparse
import unittest

from experiments.phase6_experiment_matrix import _command_rows
from plot.eval_checkpoint import _union_fieldnames


class Phase6HelperTests(unittest.TestCase):
    def test_experiment_matrix_includes_core_ablation_commands(self) -> None:
        args = argparse.Namespace(
            agents=[2],
            seeds=[42],
            steps=100,
            eval_episodes=3,
            eval_opponents=["OP3", "OP4"],
            device="cpu",
            checkpoint_root="checkpoints/phase6",
            python="python",
        )

        rows = _command_rows(args)
        variants = {row["variant"] for row in rows}

        self.assertIn("latent_default", variants)
        self.assertIn("vanilla", variants)
        self.assertIn("no_persistence", variants)
        vanilla = next(row for row in rows if row["variant"] == "vanilla")
        self.assertIn("--no-latent-strategy", vanilla["train_command"])
        default = next(row for row in rows if row["variant"] == "latent_default")
        self.assertIn("plot/eval_checkpoint.py", default["eval_command"])

    def test_eval_checkpoint_field_union_preserves_strategy_columns(self) -> None:
        fields = _union_fieldnames(
            [{"success": 1, "strategy_occupancy_0": 0.25}, {"success": 0, "strategy_occupancy_1": 0.75}],
            ["success"],
        )

        self.assertEqual(fields[0], "success")
        self.assertIn("strategy_occupancy_0", fields)
        self.assertIn("strategy_occupancy_1", fields)


if __name__ == "__main__":
    unittest.main()
