"""Pin [PPO|diag] latent-field gating for non-latent specialist runs."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from rl.custom_ppo.training_telemetry import TrainingTelemetry


def _core_row() -> dict:
    return {
        "explained_variance": 0.913,
        "value_loss": 0.168,
        "reward_shaping_mean": 0.007,
        "reward_outcome_mean": 0.006,
        "latent_lam_h": 0.005,
        "strategy_entropy": 0.0,
        "strategy_entropy_frac": 0.0,
        "strategy_wr_spread": 0.0,
        "strategy_aux_return_loss": 0.0,
        "strategy_grad_norm": 0.0,
        "strategy_policy_loss": 0.0,
        "strategy_ratio_std": 0.0,
    }


class DiagLatentGateTests(unittest.TestCase):
    def _telemetry(self, *, use_latent: bool, latent_k: int = 4) -> TrainingTelemetry:
        return TrainingTelemetry(
            cfg=SimpleNamespace(
                latent_episode_strategy_ppo=False,
                latent_arc_credit_enabled=False,
                verbose_training=False,
            ),
            hparams=SimpleNamespace(
                use_latent_strategy=use_latent,
                latent_k=latent_k,
                normalize_returns=False,
                latent_sparse_tactical_refresh_enabled=False,
            ),
            curriculum=None,
            reward_shaping_coef=lambda: 1.0,
            runtime=SimpleNamespace(global_step=743424),
        )

    def test_non_latent_diag_omits_structural_z_zeros(self) -> None:
        tel = self._telemetry(use_latent=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            tel.print_update_diagnostics(_core_row(), {})
        line = buf.getvalue().strip()
        self.assertIn("[PPO|diag] steps=743424", line)
        self.assertIn("ev=0.913", line)
        self.assertIn("v_loss=0.168", line)
        self.assertIn("shape/out=0.007/0.006", line)
        for forbidden in (
            "qphi_grad",
            "lamH=",
            "zH=",
            "z_wr_spread",
            "z_aux_ret",
            "z_pi=",
            "z_ratio=",
            "z_occ=",
            "z_wr=",
        ):
            self.assertNotIn(forbidden, line, msg=f"unexpected {forbidden!r} in: {line}")

    def test_latent_diag_still_prints_z_fields(self) -> None:
        tel = self._telemetry(use_latent=True, latent_k=2)
        row = _core_row()
        row["strategy_occupancy_0"] = 0.6
        row["strategy_occupancy_1"] = 0.4
        row["episode_z_0_win_rate"] = 0.5
        row["episode_z_1_win_rate"] = 0.4
        row["strategy_grad_norm"] = 1.25e-4
        buf = io.StringIO()
        with redirect_stdout(buf):
            tel.print_update_diagnostics(row, {})
        text = buf.getvalue()
        self.assertIn("qphi_grad=", text)
        self.assertIn("lamH=", text)
        self.assertIn("zH=", text)
        self.assertIn("z_occ=[0.600,0.400]", text)
        self.assertIn("z_wr=[0.500,0.400]", text)


if __name__ == "__main__":
    unittest.main()
