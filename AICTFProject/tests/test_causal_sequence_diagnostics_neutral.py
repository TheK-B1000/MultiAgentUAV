"""Prove the diagnostics wrap is behavior-neutral: it must not exist as an assertion alone.

Two identically-seeded CausalSequenceRunner instances -- one with install_diagnostics_reporter
applied, one without -- run the same number of update steps against the same synthetic bank
and the same starting model weights, sampling the SAME minibatches each step (RNG pinned
externally for this comparison, not a change to the runner's own default behaviour). Their
resulting model parameters must be bitwise identical.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import torch
    HAVE_TORCH = True
except Exception:                                            # pragma: no cover
    HAVE_TORCH = False

import experiments.smoke_hog_psp_branch_isolation as B
from tests.test_causal_sequence_runner import _make_bank, N_AGENTS, GRID_SHAPE, VEC_DIM, GS_DIM


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class DiagnosticsNeutralTests(unittest.TestCase):

    def test_identical_parameters_with_and_without_diagnostics(self):
        from rl.causal_sequence_runner import CausalSequenceRunner
        from rl.causal_sequence_diagnostics import install_diagnostics_reporter, restore

        class _Trainer:
            pass

        import tempfile
        with tempfile.TemporaryDirectory() as d:
            npz, meta = _make_bank(Path(d), bank_hash="MATCH", n_segments=2, rows_per_seg=10)

            def build_runner(model_seed: int, sampling_seed: int):
                torch.manual_seed(model_seed)
                _cfg, model = B.build("cpu", B.LRO_FLAGS)
                trainer = _Trainer()
                trainer.model = model
                trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
                trainer.device = "cpu"
                runner = CausalSequenceRunner(
                    trainer, npz, meta, lam=0.05, cadence=1, batch_rows=6,
                    expected_bank_hash="MATCH", device="cpu")
                runner._rng = torch.Generator(device="cpu").manual_seed(sampling_seed)
                return model, runner

            model_a, runner_a = build_runner(model_seed=42, sampling_seed=777)
            model_b, runner_b = build_runner(model_seed=42, sampling_seed=777)

            original = install_diagnostics_reporter(runner_a, every=1)
            self.assertIsNotNone(original, "install must return the original bound method")

            N_STEPS = 5
            for _ in range(N_STEPS):
                fired_a = runner_a.note_ppo_minibatch()
                fired_b = runner_b.note_ppo_minibatch()
                self.assertEqual(fired_a, fired_b, "wrapped and unwrapped runners diverged "
                                 "on whether an update fired")

            restore(runner_a, original)

            self.assertEqual(runner_a.n_updates, runner_b.n_updates)
            self.assertEqual(runner_a.z0_exposures, runner_b.z0_exposures)
            self.assertEqual(runner_a.z1_exposures, runner_b.z1_exposures)
            self.assertEqual(runner_a.positive_routes, runner_b.positive_routes)
            self.assertEqual(runner_a.negative_routes, runner_b.negative_routes)
            self.assertEqual(runner_a.last_loss, runner_b.last_loss,
                             "loss value differs; the wrap perturbed computation")

            params_a = {n: p.detach().cpu().numpy().copy() for n, p in model_a.named_parameters()}
            params_b = {n: p.detach().cpu().numpy().copy() for n, p in model_b.named_parameters()}
            self.assertEqual(set(params_a), set(params_b))
            for name in params_a:
                self.assertTrue(np.array_equal(params_a[name], params_b[name]),
                                f"parameter {name} diverged between wrapped and unwrapped runs")

    def test_restore_returns_the_exact_original_method(self):
        from rl.causal_sequence_runner import CausalSequenceRunner
        from rl.causal_sequence_diagnostics import install_diagnostics_reporter, restore

        class _Trainer:
            pass

        import tempfile
        with tempfile.TemporaryDirectory() as d:
            npz, meta = _make_bank(Path(d), bank_hash="MATCH")
            _cfg, model = B.build("cpu", B.LRO_FLAGS)
            trainer = _Trainer()
            trainer.model = model
            trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            trainer.device = "cpu"
            runner = CausalSequenceRunner(trainer, npz, meta, lam=0.05, cadence=1,
                                          batch_rows=4, expected_bank_hash="MATCH", device="cpu")
            # Bound methods are not cached by Python -- obj.method is obj.method is False in
            # general, wrapping or not, because __get__ creates a fresh bound-method object
            # on every access. The real invariant is narrower: restore() must reinstate
            # EXACTLY the object install() returned, not merely something behaviourally
            # similar, so both accesses below are pinned to the SAME captured reference.
            before_wrap = runner.note_ppo_minibatch
            original = install_diagnostics_reporter(runner)
            self.assertIsNot(runner.note_ppo_minibatch, before_wrap,
                             "install did not actually wrap the method")
            restore(runner, original)
            self.assertIs(runner.note_ppo_minibatch, original,
                          "restore did not reinstate exactly the object install() returned")
            self.assertEqual(runner.note_ppo_minibatch.__func__,
                             before_wrap.__func__,
                             "the restored method is not the same underlying function")


if __name__ == "__main__":
    unittest.main()
