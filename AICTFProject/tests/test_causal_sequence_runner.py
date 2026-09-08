"""CausalSequenceRunner against a tiny synthetic bank -- fast, no environment, no GPU needed.

Exercises the real class, the real hash-verification refusal, and a real gradient step on
the real LRO private-branch architecture, before the expensive full production wiring
verification is attempted. Not a substitute for that verification; a cheap first filter.
"""
from __future__ import annotations

import json
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

N_AGENTS, GRID_SHAPE, VEC_DIM, GS_DIM = 2, (7, 20, 20), 20, 170
N_MACROS, N_TARGETS = 5, 50


def _make_bank(tmp_dir: Path, *, bank_hash: str, n_segments=2, rows_per_seg=6):
    rows = {"global_state": [], "grid": [], "vec": [], "agent_mask": [], "mask": [],
           "actions": [], "z_idx": [], "decision_mask": [], "weight": [], "segment_idx": []}
    seg_meta = []
    torch.manual_seed(0)
    for s in range(n_segments):
        pole = "A" if s == 0 else "B"
        latent = 0 if pole == "A" else 1
        teacher = "pi_A" if pole == "A" else "pi_B"
        weight = 1.0
        for _ in range(rows_per_seg):
            rows["global_state"].append(torch.randn(GS_DIM).numpy())
            rows["grid"].append(torch.randn(N_AGENTS, *GRID_SHAPE).numpy())
            rows["vec"].append(torch.randn(N_AGENTS, VEC_DIM).numpy())
            rows["agent_mask"].append(np.ones(N_AGENTS, dtype=np.float32))
            mask = np.zeros(N_AGENTS * (N_MACROS + N_TARGETS), dtype=np.float32)
            off = N_MACROS + N_TARGETS
            mask[:3] = 1.0; mask[N_MACROS:N_MACROS + 20] = 1.0
            mask[off + 1] = 1.0; mask[off + N_MACROS + 7] = 1.0
            rows["mask"].append(mask)
            act = np.array([2, 5, 1, 7], dtype=np.int64)
            rows["actions"].append(act)
            rows["z_idx"].append(latent)
            dm = np.array([1.0, 0.0], dtype=np.float32)
            rows["decision_mask"].append(dm)
            rows["weight"].append(np.array([weight, 0.0], dtype=np.float32))
            rows["segment_idx"].append(s)
        seg_meta.append({"segment_id": f"synthetic|{pole}|{s}", "pole": pole, "latent": latent,
                         "teacher": teacher, "weight": weight, "controlled_agents": [0],
                         "start_state_id": f"synthetic|{s}", "episode_rows": rows_per_seg,
                         "live_decision_rows": rows_per_seg})

    for k in rows:
        rows[k] = np.stack(rows[k]) if k not in ("z_idx", "segment_idx") else np.asarray(rows[k])
    rows["decision_mask"] = rows["decision_mask"].astype(bool)

    npz_path = tmp_dir / "bank.npz"
    np.savez_compressed(npz_path, **rows)
    import hashlib
    npz_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    meta_path = tmp_dir / "meta.json"
    meta_path.write_text(json.dumps({
        "status": "FROZEN_ARTIFACT", "segment_bank_hash": bank_hash,
        "npz_sha256": npz_sha, "total_segments_in_causal_bank": 20,
        "nonzero_segments_rolled_out": n_segments, "segments": seg_meta,
    }), encoding="utf-8")
    return npz_path, meta_path


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class CausalSequenceRunnerTests(unittest.TestCase):

    def test_hash_mismatch_is_refused(self):
        from rl.causal_sequence_runner import CausalSequenceRunner
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            npz, meta = _make_bank(Path(d), bank_hash="AAAA")
            with self.assertRaises(RuntimeError):
                CausalSequenceRunner(trainer=None, npz_path=npz, meta_path=meta,
                                     lam=0.05, cadence=1, batch_rows=4,
                                     expected_bank_hash="BBBB")

    def test_hash_match_loads_and_steps(self):
        from rl.causal_sequence_runner import CausalSequenceRunner

        class _Trainer:
            pass

        _cfg, model = B.build("cpu", B.LRO_FLAGS)
        trainer = _Trainer()
        trainer.model = model
        trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        trainer.device = "cpu"

        import tempfile
        with tempfile.TemporaryDirectory() as d:
            npz, meta = _make_bank(Path(d), bank_hash="MATCH")
            runner = CausalSequenceRunner(
                trainer, npz, meta, lam=0.05, cadence=1, batch_rows=8,
                expected_bank_hash="MATCH", device="cpu")

            before = B.snapshot(model)
            z0_before = runner.telemetry()["z0_exposures"]

            fired = runner.note_ppo_minibatch()
            self.assertTrue(fired, "cadence=1 must fire on the first minibatch")
            self.assertEqual(runner.n_updates, 1)

            after = B.snapshot(model)
            z0_names = B.private_names(model, 0)
            z1_names = B.private_names(model, 1)
            self.assertTrue(B.changed(before, after, z0_names) or
                            B.changed(before, after, z1_names),
                            "no private branch moved after a real step")
            tel = runner.telemetry()
            self.assertGreater(tel["z0_exposures"] + tel["z1_exposures"], 0)
            self.assertGreater(tel["positive_routes"] + tel["negative_routes"], 0)

    def test_cadence_gates_updates(self):
        from rl.causal_sequence_runner import CausalSequenceRunner

        class _Trainer:
            pass

        _cfg, model = B.build("cpu", B.LRO_FLAGS)
        trainer = _Trainer()
        trainer.model = model
        trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        trainer.device = "cpu"

        import tempfile
        with tempfile.TemporaryDirectory() as d:
            npz, meta = _make_bank(Path(d), bank_hash="MATCH")
            runner = CausalSequenceRunner(
                trainer, npz, meta, lam=0.05, cadence=3, batch_rows=4,
                expected_bank_hash="MATCH", device="cpu")
            fired = [runner.note_ppo_minibatch() for _ in range(6)]
            self.assertEqual(fired, [False, False, True, False, False, True])
            self.assertEqual(runner.n_updates, 2)


if __name__ == "__main__":
    unittest.main()
