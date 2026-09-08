"""SRCTF attempt-state machinery and fresh-block enforcement.

Frozen by SRCTF_V1_ERRATUM_02.json. Data spending and evidential validity are
INDEPENDENT: consuming a block is irreversible, but treating its output as
evidence is conditional on the execution being sound.

    INVALID_ATTEMPT     verified failure proven BEFORE consumption
                        -> block unspent, design repairable, one shot intact

    INVALID_EXECUTION   verified failure proven AFTER consumption
                        -> block stays spent, NO rerun, design closes
                           INCONCLUSIVE. May report neither a PASS nor a
                           scientific NO_REVERSAL: the first rewards a bug, the
                           second lets a defect masquerade as a property of the
                           environment.

    VALID               result stands as evidence, PASS or FAIL, and closes
                        the design either way

Invalidating a consumed block requires a DETERMINISTIC, reproducible
demonstration that the defect made the measured quantity not the intended
quantity. A disappointing result, a surprising result, a suspicion, or a defect
that does not change what was measured are all insufficient.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

VALID = "VALID"
INVALID_ATTEMPT = "INVALID_ATTEMPT"
INVALID_EXECUTION = "INVALID_EXECUTION"

# What each state is permitted to report. INVALID_EXECUTION reports nothing.
REPORTABLE = {
    VALID: ("PASS", "FAIL", "NO_REVERSAL"),
    INVALID_ATTEMPT: (),
    INVALID_EXECUTION: (),
}


class AttemptError(RuntimeError):
    """Raised on any attempt-protocol violation. Fails closed."""


@dataclasses.dataclass
class BlockLedger:
    """Records which seed blocks have been consumed. Spending is irreversible."""

    path: Path

    def _load(self) -> dict:
        if not self.path.exists():
            return {"spent": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")

    def is_spent(self, block: int) -> bool:
        return str(int(block)) in self._load()["spent"]

    def require_unspent(self, block: int) -> None:
        if self.is_spent(block):
            rec = self._load()["spent"][str(int(block))]
            raise AttemptError(
                f"block {block} was already consumed by {rec.get('design')!r} on "
                f"{rec.get('utc')}. Blocks are single-use; reusing one would recycle "
                f"data the design has already seen.")

    def spend(self, block: int, *, design: str, utc: str) -> None:
        """Irreversible. Called at the moment the scanner touches the block."""
        d = self._load()
        self.require_unspent(block)
        d["spent"][str(int(block))] = {"design": design, "utc": utc, "state": "CONSUMED"}
        self._save(d)

    def annotate(self, block: int, **fields) -> None:
        """Record an outcome against a spent block. Never un-spends it."""
        d = self._load()
        key = str(int(block))
        if key not in d["spent"]:
            raise AttemptError(f"block {block} is not spent; nothing to annotate")
        d["spent"][key].update(fields)
        self._save(d)


@dataclasses.dataclass(frozen=True)
class AttemptOutcome:
    design: str
    state: str
    block: int
    block_spent: bool
    rerun_allowed: bool
    reportable: tuple
    finding: str | None
    evidence: str | None

    def to_dict(self) -> dict:
        return {**dataclasses.asdict(self), "reportable": list(self.reportable)}


def invalid_attempt(design: str, block: int, *, evidence: str) -> AttemptOutcome:
    """Failure proven BEFORE consumption. Costs nothing."""
    return AttemptOutcome(design, INVALID_ATTEMPT, block, block_spent=False,
                          rerun_allowed=True, reportable=REPORTABLE[INVALID_ATTEMPT],
                          finding=None, evidence=evidence)


def invalid_execution(design: str, block: int, *, deterministic_test: str,
                      shows_measured_quantity_wrong: bool) -> AttemptOutcome:
    """Failure proven AFTER consumption. Block stays spent; design closes INCONCLUSIVE."""
    if not deterministic_test:
        raise AttemptError(
            "INVALID_EXECUTION requires a deterministic, reproducible demonstration. "
            "A suspicion or a disappointing result is never sufficient.")
    if not shows_measured_quantity_wrong:
        raise AttemptError(
            "INVALID_EXECUTION requires the defect to have made the MEASURED quantity not "
            "the intended quantity. A defect that does not change what was measured leaves "
            "the result VALID.")
    return AttemptOutcome(design, INVALID_EXECUTION, block, block_spent=True,
                          rerun_allowed=False, reportable=REPORTABLE[INVALID_EXECUTION],
                          finding="INCONCLUSIVE", evidence=deterministic_test)


def valid(design: str, block: int, *, finding: str) -> AttemptOutcome:
    """A sound execution. Its result is evidence and closes the design."""
    if finding not in REPORTABLE[VALID]:
        raise AttemptError(f"{finding!r} is not a reportable finding; expected one of "
                           f"{REPORTABLE[VALID]}")
    return AttemptOutcome(design, VALID, block, block_spent=True, rerun_allowed=False,
                          reportable=REPORTABLE[VALID], finding=finding, evidence=None)


def assert_reportable(outcome: AttemptOutcome, claim: str) -> None:
    """Guard every scientific claim through the attempt state. Fails closed."""
    if claim not in outcome.reportable:
        raise AttemptError(
            f"{outcome.state} may not report {claim!r}. "
            f"Permitted: {outcome.reportable or '(nothing -- closes INCONCLUSIVE)'}. "
            f"Reporting a PASS would reward a bug; reporting a NO_REVERSAL would let a "
            f"defect masquerade as a property of the environment.")
