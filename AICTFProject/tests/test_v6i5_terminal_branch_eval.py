import math
import unittest

import torch

from tools.v6i5_state_conditioned_branch_eval import (
    TARGET_BUCKETS,
    classify_recovery_family,
    compute_terminal_pair_rows,
    compute_terminal_summary_rows,
    paired_bootstrap_ci,
    _model_parameter_hash,
    _target_bucket_pairs,
)


def _branch(snapshot_id: str, forced_z: int, *, terminated: bool, final_diff: int, ret: float) -> dict:
    branch_delta = final_diff
    return {
        "checkpoint": "ckpt.zip",
        "opponent": snapshot_id.split("|")[0],
        "state_bucket": snapshot_id.split("|")[1],
        "snapshot_id": snapshot_id,
        "snapshot_source_seed": 7000,
        "snapshot_episode_id": 0,
        "snapshot_step": 64,
        "paired_branch_seed": 910000000,
        "forced_z": forced_z,
        "initial_team_score": 0,
        "initial_enemy_score": 0,
        "final_team_score": max(final_diff, 0),
        "final_enemy_score": max(-final_diff, 0),
        "final_score_differential": final_diff,
        "score_change_from_snapshot": branch_delta,
        "branch_end_score_delta": branch_delta,
        "terminal_score_delta": branch_delta if terminated else "",
        "terminal_reward_or_return": ret,
        "win": int(terminated and final_diff > 0),
        "loss": int(terminated and final_diff < 0),
        "draw": int(terminated and final_diff == 0),
        "terminated_naturally": int(terminated),
        "truncated_by_safety_cap": int(not terminated),
        "branch_steps": 300,
        "time_to_natural_termination": 300 if terminated else "",
        "team_flag_initially_carried": 0,
        "enemy_flag_initially_carried": 1,
        "team_flag_recovered": 1,
        "enemy_flag_capture_completed": 0,
        "own_flag_returned": 1,
        "team_capture_count_after_branch": max(final_diff, 0),
        "enemy_capture_count_after_branch": max(-final_diff, 0),
    }


class V6I5TerminalBranchEvalTests(unittest.TestCase):
    def test_target_bucket_validation_rejects_unsupported_bucket(self) -> None:
        self.assertEqual(_target_bucket_pairs(None), TARGET_BUCKETS)
        with self.assertRaises(ValueError):
            _target_bucket_pairs(["OP5", "leading_late"])

    def test_pair_rows_join_exact_z0_and_z3_snapshot(self) -> None:
        rows = [
            _branch("OP5|enemy_carrying_team_flag|a", 0, terminated=True, final_diff=1, ret=3.0),
            _branch("OP5|enemy_carrying_team_flag|a", 3, terminated=True, final_diff=-1, ret=1.0),
            _branch("OP5|enemy_carrying_team_flag|b", 0, terminated=True, final_diff=4, ret=2.0),
        ]
        pairs = compute_terminal_pair_rows(rows)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["snapshot_id"], "OP5|enemy_carrying_team_flag|a")
        self.assertEqual(pairs[0]["paired_terminal_return_advantage_z0_minus_z3"], 2.0)
        self.assertEqual(pairs[0]["z0_wins_pair"], 1)

    def test_terminal_score_delta_is_unavailable_for_truncated_branch(self) -> None:
        rows = [
            _branch("OP6|team_carrying_enemy_flag|a", 0, terminated=False, final_diff=1, ret=1.0),
            _branch("OP6|team_carrying_enemy_flag|a", 3, terminated=True, final_diff=0, ret=0.5),
        ]
        self.assertEqual(rows[0]["terminal_score_delta"], "")
        pairs = compute_terminal_pair_rows(rows)
        self.assertEqual(pairs[0]["pair_valid_for_terminal_analysis"], 0)
        self.assertEqual(pairs[0]["paired_terminal_return_advantage_z0_minus_z3"], "")

    def test_bootstrap_is_deterministic_under_fixed_seed(self) -> None:
        vals = [1.0, 2.0, -1.0, 0.5]
        self.assertEqual(
            paired_bootstrap_ci(vals, seed=123, resamples=200),
            paired_bootstrap_ci(vals, seed=123, resamples=200),
        )

    def test_summary_and_gate_classification(self) -> None:
        rows = []
        for opp in ("OP5", "OP6", "OP7"):
            sid = f"{opp}|enemy_carrying_team_flag|0"
            rows.append(_branch(sid, 0, terminated=True, final_diff=2, ret=2.0))
            rows.append(_branch(sid, 3, terminated=True, final_diff=0, ret=0.0))
        pairs = compute_terminal_pair_rows(rows)
        summary = compute_terminal_summary_rows(pairs, rows, bootstrap_seed=7, bootstrap_resamples=200)
        self.assertEqual(classify_recovery_family(summary), "VALIDATED")

    def test_model_parameter_hash_changes_only_when_parameters_change(self) -> None:
        model = torch.nn.Linear(2, 1)
        h1 = _model_parameter_hash(model)
        h2 = _model_parameter_hash(model)
        self.assertEqual(h1, h2)
        with torch.no_grad():
            model.weight.add_(1.0)
        self.assertNotEqual(h1, _model_parameter_hash(model))


if __name__ == "__main__":
    unittest.main()
