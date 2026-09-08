"""Build the frozen SEQUENCE-mode segment bank from the Phase 1 causal branching result.

Implements CCP_SUCCESSOR_BUILD_CONTRACT.json amendments 2-3 against
CCP_PHASE1_CAUSAL_BRANCHING.json. Pure transformation: reads the frozen, already-committed
Phase 1 result and produces CausalSegment records. No rollout, no new measurement, no choice
left open -- every rule it applies was frozen before this file existed.

    source           full_takeover contrasts only (amendment 2: SUCCESSOR_MODE = SEQUENCE)
    weighting        continuous, ALL 32 contrasts, w = |delta_Q_hat| -- never filtered to
                     the 10 Holm-significant ones (Phase 1 spec amendment 2, applied as
                     originally frozen)
    routing          winner-directed via CausalSegment's shared _WinnerDirected derivation;
                     the latent never flips
    duration         active_until = episode termination (amendment 2): exactly what
                     full_takeover measured, nothing shorter
    joint precedence (amendment 3): wherever a joint segment exists for a state, it alone
                     supervises both agents there. The individual agent0/agent1 segments for
                     that SAME state are excluded from the returned bank -- structurally, not
                     by a downstream filter someone could forget to apply.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rl.causal_supervision import CausalSegment


def build_segment_bank(phase1_result_path: str | Path) -> list[CausalSegment]:
    """Return the frozen SEQUENCE-mode segment bank.

    Deterministic: the same Phase 1 result file always produces the same bank, in the same
    order (sorted by state_id), so this function has no hidden state and its output is
    reproducible from the artifact alone.
    """
    d = json.loads(Path(phase1_result_path).read_text(encoding="utf-8"))
    if d.get("status") != "FROZEN_RESULT":
        raise RuntimeError(f"Phase 1 result is not frozen: {d.get('status')!r}")

    contrasts = d["contrasts"]
    takeover = {k: v for k, v in contrasts.items() if v["mode"] == "full_takeover"}

    by_state: dict[str, dict[str, dict]] = {}
    for v in takeover.values():
        by_state.setdefault(v["state_id"], {})[v["estimand"]] = v

    bank: list[CausalSegment] = []
    for state_id in sorted(by_state):
        estimands = by_state[state_id]
        joint = estimands.get("joint")
        # JOINT PRECEDENCE (amendment 3): if a joint record exists, it is the ONLY signal
        # this state contributes. Individual agent0/agent1 records are not even inspected
        # for weight -- they cannot leak into the bank through any path here.
        if joint is not None:
            bank.append(CausalSegment(
                pole=joint["pole"], delta_q=joint["delta_Q_hat"],
                segment_id=f"{state_id}|joint|full_takeover",
                start_state_id=state_id, controlled_agents=(0, 1), active_until=None))
            continue
        for estimand, agent in (("agent0", 0), ("agent1", 1)):
            row = estimands.get(estimand)
            if row is None:
                continue
            bank.append(CausalSegment(
                pole=row["pole"], delta_q=row["delta_Q_hat"],
                segment_id=f"{state_id}|{estimand}|full_takeover",
                start_state_id=state_id, controlled_agents=(agent,), active_until=None))

    for seg in bank:
        seg.assert_routing()

    return bank


def segment_bank_hash(bank: list[CausalSegment]) -> str:
    """Deterministic fingerprint over the bank's decision content.

    Used to pin an offline sequence-bank artifact to the exact frozen Phase 1 result that
    produced it, so a stale or mismatched artifact cannot be loaded silently at train time.
    """
    h = hashlib.sha256()
    for seg in bank:
        h.update(f"{seg.segment_id}|{seg.pole}|{seg.delta_q!r}|{seg.controlled_agents}"
                 .encode("utf-8"))
    return h.hexdigest()
