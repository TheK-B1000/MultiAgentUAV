"""Orchestrator-side construction and attachment of the SPPPO ranking runner.

Frozen protocol: artifacts/strategic_demand/sppo/SPPPO_V1_PROTOCOL.json

The lambda_R = 0 development control is STRUCTURAL ABSENCE. This module returns
None for it and attaches nothing, so no runner object exists, no Q_psi is
loaded, no optimizer state is touched and no RNG is consumed. A runner
constructed with lambda_rank=0 would be a different experiment wearing the
control's name -- ``RankingRunner`` refuses that construction outright.

Attachment happens on ``runtime.sppo_ranking_runner`` AFTER the trainer is
built, which is why ``PPOUpdater`` reads the attribute at use time rather than
caching it. That ordering is the exact seam that silently disabled SAPPO
rehearsal for a full 2x500k campaign.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rl.scorer.ranking import OPPONENT_ID_TO_POLE, POLE_A, POLE_B, RankingRunner

__all__ = ["attach_ranking_runner", "SPPPO_QPSI_SHA256", "SPPPO_MARGIN"]

# Frozen in PHASE0_SCORER_FROZEN.json and SPPPO_V1_PROTOCOL.json respectively.
SPPPO_QPSI_SHA256 = "930051a725e55e4f14e05dfe178e5f1dc7bd8f3d7e3adeba01187958bb7417bf"
SPPPO_MARGIN = 0.04
SPPPO_QPSI_PATH = "artifacts/strategic_demand/phase0_scorer_data/qpsi_frozen.pt"
SPPPO_Z_TO_POLE = {0: POLE_A, 1: POLE_B}          # frozen 16 x z0|A, 16 x z1|B


def attach_ranking_runner(
    runtime,
    model,
    optimizer,
    *,
    lambda_rank: float,
    margin: float = SPPPO_MARGIN,
    cadence: int = 1,
    qpsi_path: str | Path = SPPPO_QPSI_PATH,
    expected_sha256: str = SPPPO_QPSI_SHA256,
    required_n_regimes: int = 1,
    max_grad_norm: float | None = None,
    device: str = "cpu",
) -> Optional[RankingRunner]:
    """Construct and attach the ranking runner, or attach nothing at all.

    Returns the runner, or None for the lambda_R = 0 control. The None case is
    load-bearing: it must leave the runtime with no ``sppo_ranking_runner``
    attribute so the updater's branch is never entered.
    """
    if lambda_rank == 0.0:
        # Structural absence. Do not load Q_psi, do not build a runner, do not
        # set the attribute -- getattr(..., None) in the updater must miss.
        if hasattr(runtime, "sppo_ranking_runner"):
            raise RuntimeError(
                "lambda_rank = 0 is the control, but runtime already carries an "
                "sppo_ranking_runner. The control must be structurally absent.")
        return None
    if lambda_rank < 0.0:
        raise ValueError(f"lambda_rank must be > 0 or exactly 0 (control), got {lambda_rank}")

    # Deferred so the control path never imports torch-heavy scorer machinery.
    from rl.scorer.ranking import load_frozen_qpsi

    if not expected_sha256:
        raise RuntimeError("ranking Q_psi requires a frozen non-empty SHA256")
    qpsi = load_frozen_qpsi(
        qpsi_path,
        expected_sha256=expected_sha256,
        device=device,
        required_n_regimes=int(required_n_regimes),
    )
    runner = RankingRunner(
        model, qpsi, optimizer,
        lambda_rank=float(lambda_rank),
        margin=float(margin),
        cadence=int(cadence),
        z_to_pole=SPPPO_Z_TO_POLE,
        opponent_to_pole=dict(OPPONENT_ID_TO_POLE),
        max_grad_norm=max_grad_norm,
        device=device,
    )
    runtime.sppo_ranking_runner = runner
    return runner
