"""Executable contracts for the surgical commitment-mechanism probe.

The probe has run and its result is sealed, so this standing test exercises the POST_RUN
state: G8 pre-run separation reports NOT APPLICABLE and G9 verifies the sealed result
against the frozen spec instead. Before the run it exercises PRE_RUN, where G8 gates.

Two deliberate choices:

* The attestation is written to ``*_CONTRACT_ATTESTATION.json``, NOT to
  ``*_CONTRACT_RESULT.json``. The latter is the pre-run authorization record -- evidence
  that contracts passed before any seed was spent -- and a test run overwriting it is how
  that evidence was previously destroyed (the committed record flipped PASS -> FAIL
  between a856469d and af8f76c2).
* The launcher's own refusal is untouched and is still what prevents a second run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts/strategic_demand/sppo"
LABEL = "ACTION_INTERFACE_COMMITMENT_MECHANISM"
ATTESTATION = SD / f"{LABEL}_CONTRACT_ATTESTATION.json"


@pytest.mark.timeout(120)
def test_action_interface_commitment_mechanism_contracts_pass():
    from experiments.eval_action_interface_commitment_mechanism import run_contracts

    result = run_contracts()
    ATTESTATION.parent.mkdir(parents=True, exist_ok=True)
    ATTESTATION.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert result["overall_pass"], json.dumps(result["gates"], indent=2)


@pytest.mark.timeout(120)
def test_g8_is_mode_aware_and_never_silently_skipped():
    """G8 must be explicit about its mode, not quietly absent or quietly passing."""
    from experiments.eval_action_interface_commitment_mechanism import OUT, run_contracts

    result = run_contracts()
    g8 = result["gates"]["G8_PRIOR_RUN_SEPARATION"]
    mode = result["mode"]
    assert mode in ("PRE_RUN", "IN_FLIGHT", "POST_RUN"), mode
    assert mode == ("POST_RUN" if OUT["result"].exists() else mode), mode

    if mode == "POST_RUN":
        # Not applicable, and the sealed-state gate must have taken over.
        assert g8["applicable"] is False
        assert "G9_SEALED_RESULT_PROVENANCE" in result["gates"], (
            "post-run G8 was retired without G9 replacing it, which would drop the "
            "frozen prior-run-separation intent entirely"
        )
        g9 = result["gates"]["G9_SEALED_RESULT_PROVENANCE"]
        assert g9["applicable"] is True
        assert g9["pass"], json.dumps(g9["checks"], indent=2)
    elif mode == "PRE_RUN":
        assert g8["applicable"] is True and g8["pass"]
        assert "G9_SEALED_RESULT_PROVENANCE" not in result["gates"]
    else:
        # A lock with no result must still be a hard failure.
        assert g8["applicable"] is True and not g8["pass"]
