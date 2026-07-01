"""GPU equivalence test: fresh env per z vs reused env (optional)."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = (
    PROJECT_ROOT
    / "checkpoints"
    / "2v2"
    / "final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip"
)


@unittest.skipUnless(
    os.environ.get("RUN_FORCED_Z_EQUIVALENCE") == "1" and DEFAULT_CKPT.is_file(),
    "Set RUN_FORCED_Z_EQUIVALENCE=1 and provide final repertoire checkpoint",
)
class ForcedZEnvReuseEquivalenceTests(unittest.TestCase):
    def test_fresh_vs_reused_env_block(self) -> None:
        from experiments.forced_z_eval.equivalence import annotate_expected_seeds, compare_forced_z_cells
        from experiments.forced_z_eval.protocol import DEFAULT_LATENTS, ForcedZProtocol
        from experiments.forced_z_eval.runner import load_shared_policy, run_forced_z_episodes

        protocol = ForcedZProtocol(
            checkpoint=str(DEFAULT_CKPT),
            opponents=("OP8",),
            maps=("map_b",),
            latents=DEFAULT_LATENTS,
            episodes_per_cell=5,
            base_seed=42,
            device="cuda",
            collect_behavior_mean=True,
            progress_every=0,
        )
        model = load_shared_policy(protocol, map_name="map_b", cell_seed=protocol.cell_seed(0, 0))
        fresh = run_forced_z_episodes(protocol, env_mode="fresh_per_z", shared_model=model, quiet=True)
        reused = run_forced_z_episodes(protocol, env_mode="reuse_block", shared_model=model, quiet=True)
        annotate_expected_seeds(fresh, protocol)
        annotate_expected_seeds(reused, protocol)
        report = compare_forced_z_cells(
            fresh,
            reused,
            opponents=["OP8"],
            maps=["map_b"],
            latents=DEFAULT_LATENTS,
        )
        self.assertTrue(report.passed, report.summary())


if __name__ == "__main__":
    unittest.main()
