"""CausalSequenceRunner: SEQUENCE-mode causal supervision, interleaved with task PPO.

Mirrors experiments/run_og_psp_production.py's PairedRehearsalRunner exactly -- same hook
name (trainer.oracle_rehearsal_runner, the ONLY generic auxiliary-rehearsal seam the updater
calls: rl/custom_ppo/update/updater.py's _oracle_rehearsal_runner()), same
note_ppo_minibatch()/step() shape, same SEPARATE zero_grad/backward/step on a fixed cadence
using the SAME optimizer as task PPO. This is deliberate: it is the exact machinery V1
through V4 already validated for this style of auxiliary loss, so nothing new is being
designed here, only re-pointed at causal_supervision_loss and the offline sequence bank.

Loads the STATIC array experiments/ccp_build_sequence_bank.py produced -- no live environment
during training, matching paired_rehearsal's pattern. The array's segment_bank_hash is
verified against a FRESH rebuild of the segment bank from the frozen Phase 1 result at
construction time, so a stale or hand-edited artifact cannot be loaded silently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class _NoTiedConcept:
    """Compatibility shim, not a design choice.

    rl/custom_ppo/update/updater.py's shared oracle_rehearsal_runner hook hard-codes
    ``oracle_runner.bank.tied_exposures`` in its per-minibatch telemetry (the same seam
    PairedRehearsalRunner uses, reused here deliberately rather than touching the shared
    updater). 'Tied' means a state where the FIT/CALIB teachers agreed and received zero
    paired-rehearsal pressure by design -- a concept that does not exist for SEQUENCE-mode
    segments. It reports 0 truthfully rather than approximately: zero-weight segments are
    never rolled out into the offline bank in the first place (ccp_build_sequence_bank.py),
    so this runner's data literally contains zero rows of that kind, not zero by convention.
    """
    tied_exposures = 0


class CausalSequenceRunner:
    """Interleaved SEQUENCE-mode causal supervision, one step every ``cadence`` minibatches."""

    def __init__(self, trainer, npz_path: str | Path, meta_path: str | Path, *,
                 lam: float, cadence: int, batch_rows: int, expected_bank_hash: str,
                 device: str = "cpu"):
        import json
        import torch

        if lam <= 0.0:
            raise ValueError("lambda must be > 0; disabled causal supervision means not attaching")
        if cadence < 1:
            raise ValueError("cadence must be >= 1")

        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        if meta["status"] != "FROZEN_ARTIFACT":
            raise RuntimeError(f"sequence bank is not frozen: {meta['status']!r}")
        if meta["segment_bank_hash"] != expected_bank_hash:
            raise RuntimeError(
                f"sequence bank hash {meta['segment_bank_hash']} does not match the "
                f"freshly rebuilt segment bank hash {expected_bank_hash}; the offline "
                f"artifact is stale relative to the frozen Phase 1 result")

        data = np.load(npz_path)
        n = len(data["z_idx"])
        self._t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
        self.obs_all = {
            "global_state": self._t(data["global_state"], torch.float32),
            "grid": self._t(data["grid"], torch.float32),
            "vec": self._t(data["vec"], torch.float32),
            "agent_mask": self._t(data["agent_mask"], torch.float32),
            "mask": self._t(data["mask"], torch.float32),
        }
        self.actions_all = self._t(data["actions"], torch.long)
        self.z_all = self._t(data["z_idx"], torch.long)
        self.decision_mask_all = self._t(data["decision_mask"], torch.bool)
        self.weight_all = self._t(data["weight"], torch.float32)
        self.segment_idx_all = data["segment_idx"]
        self.n_rows = int(n)
        self.meta = meta
        self.bank_hash = expected_bank_hash

        self.trainer = trainer
        self.lam = float(lam)
        self.cadence = int(cadence)
        self.batch_rows = int(batch_rows)
        self.device = device
        self.bank = _NoTiedConcept()          # see class docstring: a hook-compatibility shim

        self.n_ppo_minibatches = 0
        self.n_updates = 0
        self.n_rows_seen = 0
        self.last_loss = float("nan")
        self.z0_exposures = 0
        self.z1_exposures = 0
        self.positive_routes = 0
        self.negative_routes = 0
        self._segment_signs = {int(m_i): (1 if m["weight"] > 0 and self._sign_positive(m)
                                          else -1)
                               for m_i, m in enumerate(meta["segments"])}

    @staticmethod
    def _sign_positive(seg_meta: dict) -> bool:
        pole, teacher = seg_meta["pole"], seg_meta["teacher"]
        matched = "pi_A" if pole == "A" else "pi_B"
        return teacher == matched

    def note_ppo_minibatch(self) -> bool:
        self.n_ppo_minibatches += 1
        if self.n_ppo_minibatches % self.cadence:
            return False
        self.step()
        return True

    def step(self) -> None:
        from rl.causal_supervision import causal_supervision_loss

        import torch
        rng = getattr(self, "_rng", None)
        if rng is None:
            rng = torch.Generator(device="cpu")
            self._rng = rng
        # rng stays CPU (torch.randint with a CUDA generator has separate seeding semantics
        # this runner does not need); the resulting index tensor is moved to self.device
        # explicitly before indexing, since obs_all/actions_all/etc. live there and PyTorch
        # requires matching devices for advanced indexing.
        idx_cpu = torch.randint(0, self.n_rows, (min(self.batch_rows, self.n_rows),),
                                generator=rng)
        idx = idx_cpu.to(self.device)

        obs = {k: v[idx] for k, v in self.obs_all.items()}
        actions = self.actions_all[idx]
        z_idx = self.z_all[idx]
        decision_mask = self.decision_mask_all[idx]
        weights = self.weight_all[idx]
        seg_ids = self.segment_idx_all[idx_cpu.numpy()]

        loss = self.lam * causal_supervision_loss(
            self.trainer.model, obs, actions, z_idx=z_idx,
            decision_mask=decision_mask, weights=weights)

        opt = self.trainer.optimizer
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        self.n_updates += 1
        self.n_rows_seen += int(idx.numel())
        self.last_loss = float(loss.detach())
        self.z0_exposures += int((z_idx == 0).sum())
        self.z1_exposures += int((z_idx == 1).sum())
        for s in seg_ids:
            sign = self._segment_signs.get(int(s), 0)
            if sign > 0:
                self.positive_routes += 1
            elif sign < 0:
                self.negative_routes += 1

    def telemetry(self) -> dict:
        return {
            "updates": self.n_updates, "n_ppo_minibatches": self.n_ppo_minibatches,
            "rows_seen": self.n_rows_seen, "last_loss": self.last_loss,
            "z0_exposures": self.z0_exposures, "z1_exposures": self.z1_exposures,
            "positive_routes": self.positive_routes, "negative_routes": self.negative_routes,
            "segment_bank_hash": self.bank_hash,
            "nonzero_segments": self.meta["nonzero_segments_rolled_out"],
            "total_segments": self.meta["total_segments_in_causal_bank"],
        }
