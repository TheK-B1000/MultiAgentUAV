"""Gate evidence updates from valid CF-batch measurements only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rl.custom_ppo.gate_protocol import is_v6i2_dual_evidence_protocol
from rl.custom_ppo.update.loss_result import PairwiseSeparationMeasurement


@dataclass(frozen=True)
class GateEvidenceUpdate:
    gate_updated: bool
    measurement_valid: bool
    reason: str | None


class ActorInterventionEvidenceUpdater:
    """Apply v6i2 actor-intervention EMA only from valid CF measurements."""

    def update(
        self,
        latent_state: Any,
        measurement: PairwiseSeparationMeasurement,
        *,
        cfg: Any,
        global_step: int,
    ) -> GateEvidenceUpdate:
        if not is_v6i2_dual_evidence_protocol(cfg):
            return GateEvidenceUpdate(
                gate_updated=False,
                measurement_valid=False,
                reason="not_v6i2_protocol",
            )
        if not measurement.valid:
            return GateEvidenceUpdate(
                gate_updated=False,
                measurement_valid=False,
                reason=measurement.reason or "invalid_measurement",
            )
        pair_vals = measurement.as_list()
        if pair_vals is None:
            return GateEvidenceUpdate(
                gate_updated=False,
                measurement_valid=False,
                reason="missing_pair_values",
            )
        updated = bool(
            latent_state.update_cf_pair_jsd_ema(pair_vals, int(global_step))
        )
        return GateEvidenceUpdate(
            gate_updated=updated,
            measurement_valid=True,
            reason=None,
        )
