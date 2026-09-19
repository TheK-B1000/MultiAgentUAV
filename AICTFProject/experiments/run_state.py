r"""Rule 7 -- sealing is a STATE TRANSITION, not a string the author types.

    RUNNING --> COMPLETE --> AUDITED --> SEALED
                         \-> AUDIT_FAILED   (terminal)

Before this module, every evaluator wrote ``"status": "FROZEN_RESULT"`` as a
literal in its own output dict, and the integrity audit was a *separate script*
someone had to remember to run afterwards. Nothing in the machinery connected
the two. A result could be read, cited, and put in a paper without its audit
ever having executed -- and on a bad day, after its audit had failed.

Here the audit is the gate. ``seal()`` writes the result with ``status``
set by the state machine, and the only path to ``SEALED`` runs through an audit
that passed. A failing audit writes ``AUDIT_FAILED`` and, by default, raises.

WHAT THE AUDIT IS
    A re-derivation of every sealed number *from the persisted raw rows*, plus
    structural integrity checks on those rows. It proves the sealed statistics
    follow from the data on disk, that the data on disk has the shape the frozen
    spec demanded, and that the checkpoints were not substituted.

WHAT THE AUDIT IS NOT  (read this before trusting a SEALED record)
    It is NOT an independent methodological review. The bootstrap re-derivation
    below is a second *implementation*, written to the same contract, so it
    catches stale results, mis-paired arrays, wrong slices, silent truncation
    and rounding drift -- but a methodological choice that is wrong *in the same
    way* in both places would reproduce perfectly and pass. SEALED means
    "internally consistent and structurally sound", never "correct science".
    Rules 3 (frozen spec) and 10 (decision tree) are what carry correctness;
    this rule only guarantees they cannot be skipped silently.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- states ----
RUNNING = "RUNNING"
COMPLETE = "COMPLETE"
AUDITED = "AUDITED"
SEALED = "SEALED"
AUDIT_FAILED = "AUDIT_FAILED"

#: Legal transitions. Absent from every value set: any edge INTO ``SEALED``
#: that does not come from ``AUDITED``. That single omission is the whole rule.
LEGAL_TRANSITIONS: dict[str | None, set[str]] = {
    None: {RUNNING},
    RUNNING: {RUNNING, COMPLETE},          # RUNNING->RUNNING = heartbeat
    COMPLETE: {AUDITED, AUDIT_FAILED},
    AUDITED: {SEALED},
    SEALED: set(),                          # terminal
    AUDIT_FAILED: set(),                    # terminal -- freeze it, do not retry in place
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class IllegalTransition(RuntimeError):
    """Raised when code tries to move a run into a state it cannot reach."""


# ------------------------------------------------------------ state file ----
class RunState:
    """The observable lifecycle of one run, persisted to ``<LABEL>_RUN_STATE.json``.

    The file is the record of record: it holds the current state, the full
    transition history with timestamps, and enough process metadata to tell a
    crashed run from a finished one without reading a log.
    """

    def __init__(self, dirpath: str | Path, label: str) -> None:
        self.path = Path(dirpath) / f"{label}_RUN_STATE.json"
        self.label = label
        self._doc: dict[str, Any] = (
            json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file()
            else {"label": label, "state": None, "history": []}
        )

    @property
    def state(self) -> str | None:
        return self._doc.get("state")

    def _transition(self, new: str, **extra: Any) -> None:
        cur = self.state
        allowed = LEGAL_TRANSITIONS.get(cur, set())
        if new not in allowed:
            raise IllegalTransition(
                f"{self.label}: {cur} -> {new} is not a legal transition "
                f"(legal from {cur}: {sorted(allowed) or 'nothing -- terminal state'}). "
                f"A terminal state is terminal on purpose: freeze it and start a new "
                f"label rather than overwriting a recorded outcome.")
        self._doc["state"] = new
        self._doc["updated_utc"] = _now()
        self._doc["history"].append({"state": new, "utc": _now(), **extra})
        self._doc.update(extra)
        self.path.write_text(json.dumps(self._doc, indent=2), encoding="utf-8")

    def begin(self, **meta: Any) -> "RunState":
        if self.state in (SEALED, AUDIT_FAILED):
            raise IllegalTransition(
                f"{self.label}: already {self.state}; refusing to restart a finished run. "
                f"Use a new OUTPUT_LABEL.")
        if self.state is None:
            self._transition(RUNNING, pid=os.getpid(), host=platform.node(),
                             python=sys.version.split()[0], git_sha=_git_sha(), **meta)
        else:                       # resuming an interrupted RUNNING run
            self._transition(RUNNING, pid=os.getpid(), resumed=True, **meta)
        return self

    def complete(self, **meta: Any) -> "RunState":
        self._transition(COMPLETE, **meta)
        return self


# ------------------------------------------------------------- audit DSL ----
class Derived(NamedTuple):
    """A field that must equal a function of other fields in the same row."""
    desc: str                                   # human-readable, goes in the record
    fn: Callable[[dict[str, Any]], Any]


@dataclass
class Claim:
    """A sealed statistic the audit must re-derive from the rows CSV.

    ``minuend`` / ``subtrahend`` are row selectors (field -> value). With both
    set, the audited vector is the per-seed paired difference, in seed order --
    which is exactly how every crossover delta in this project is defined.
    """
    name: str
    recorded: dict                              # {"mean","lcb95","ucb95"} or {"mean"}
    minuend: dict
    subtrahend: dict | None = None
    value_field: str = "win"


@dataclass
class AuditPlan:
    """Everything the audit needs, declared up front -- alongside the frozen spec."""
    rows_csv: Path
    expected_rows: int
    expected_seeds: Sequence[int]
    group_by: Sequence[str]                     # fields that define one cell
    seed_field: str = "seed"
    int_fields: Sequence[str] = ()              # parsed as int when reading rows
    binary_fields: Sequence[str] = ()           # must contain only {0, 1}
    derived: dict[str, Derived] = field(default_factory=dict)
    checkpoints: dict[str, tuple[Path, str]] = field(default_factory=dict)  # name -> (path, sha)
    spec_path: Path | None = None
    claims: Sequence[Claim] = ()
    n_boot: int = 20000
    alpha: float = 0.05
    rng_seed: int = 7
    tolerance: float = 1e-6                     # sealed values are rounded to 6 dp
    seed_class: str | None = None               # "sealed_confirmatory" | "exploratory" | "smoke"
    experiment_id: str | None = None            # registry owner of the seed block, if any


class CheckResult(NamedTuple):
    name: str
    gating: bool
    passed: bool
    detail: str
    data: dict


# ----------------------------------------------------------- audit engine ----
def _bootstrap(v: np.ndarray, n_boot: int, alpha: float, rng_seed: int) -> dict:
    """Re-derivation of the project's paired percentile bootstrap.

    Mirrors ``experiments/eval_hog_psp_v3.py::_mean_ci`` deliberately, including
    the generator FAMILY: ``default_rng`` is PCG64, and a re-derivation using
    the legacy ``RandomState`` (MT19937) produces close-but-not-identical tails.
    That is ordinary Monte-Carlo variation between two different generators, not
    a defect -- but it means the generator is part of the contract being audited,
    so it is pinned here rather than left to the caller's numpy default.
    """
    v = np.asarray(v, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(v.mean()), "lcb95": float(lo), "ucb95": float(hi)}


def _read_rows(plan: AuditPlan) -> list[dict]:
    ints = set(plan.int_fields) | {plan.seed_field} | set(plan.binary_fields)
    out = []
    with Path(plan.rows_csv).open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append({k: (int(v) if k in ints and v != "" else v) for k, v in r.items()})
    return out


def _select(rows: list[dict], sel: dict) -> list[dict]:
    return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in sel.items())]


def _vector(rows: list[dict], sel: dict, plan: AuditPlan, value_field: str,
            seeds: Sequence[int] | None = None) -> np.ndarray:
    """Per-seed values for one cell, in frozen seed order.

    Seed order is not cosmetic: paired differences are only meaningful if both
    sides are indexed the same way, so this always materialises against
    ``plan.expected_seeds`` rather than against CSV row order.
    """
    seeds = plan.expected_seeds if seeds is None else seeds
    by_seed = {int(r[plan.seed_field]): r for r in _select(rows, sel)}
    missing = [s for s in seeds if s not in by_seed]
    if missing:
        raise KeyError(f"selector {sel} missing seeds {missing[:5]}"
                       f"{'...' if len(missing) > 5 else ''}")
    return np.array([float(by_seed[s][value_field]) for s in seeds], dtype=np.float64)


def run_audit(plan: AuditPlan) -> dict:
    """Execute every check. Returns the audit record; never raises on a failed
    CHECK (that is the caller's decision) -- only on a malformed plan."""
    checks: list[CheckResult] = []

    def add(name, gating, passed, detail, **data):
        checks.append(CheckResult(name, gating, bool(passed), detail, data))

    # 1 -- the spec must have been frozen before any seed was spent (Rule 3)
    if plan.spec_path is not None:
        sp = Path(plan.spec_path)
        if not sp.is_file():
            add("spec_frozen", True, False, f"spec not found: {sp}")
        else:
            st = str(json.loads(sp.read_text(encoding="utf-8")).get("status", ""))
            add("spec_frozen", True, st.startswith("FROZEN"),
                f"spec status {st!r}", spec=sp.name, sha256=sha256_file(sp), status=st)

    # 2 -- rows exist and are exactly the promised size
    rp = Path(plan.rows_csv)
    if not rp.is_file():
        add("rows_present", True, False, f"rows CSV missing: {rp}")
        return _finalise(plan, checks, rows=[])
    rows = _read_rows(plan)
    add("rows_present", True, True, f"{rp.name} ({rp.stat().st_size} bytes)",
        sha256=sha256_file(rp))
    add("row_count", True, len(rows) == plan.expected_rows,
        f"{len(rows)} rows, expected {plan.expected_rows}",
        found=len(rows), expected=plan.expected_rows)

    # 3 -- every cell holds exactly the frozen seed block, once each
    want = set(int(s) for s in plan.expected_seeds)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(str(r.get(k)) for k in plan.group_by), []).append(r)
    bad_cells, dup_cells = [], []
    for key, grp in groups.items():
        seen = [int(r[plan.seed_field]) for r in grp]
        if len(seen) != len(set(seen)):
            dup_cells.append("/".join(key))
        if set(seen) != want:
            bad_cells.append("/".join(key))
    add("cell_seed_block_exact", True, not bad_cells,
        f"{len(groups)} cells; {len(bad_cells)} with a seed set != frozen block "
        f"{min(want)}..{max(want)} (n={len(want)})",
        n_cells=len(groups), offending=bad_cells[:10])
    add("no_duplicate_seeds", True, not dup_cells,
        f"{len(dup_cells)} cell(s) contain a repeated seed", offending=dup_cells[:10])

    # 4 -- pairing: comparing cells is only valid if they share a seed set
    add("pairing_valid", True, not bad_cells and not dup_cells,
        "every cell carries the identical seed set, so paired differences are "
        "seed-matched" if not (bad_cells or dup_cells) else
        "cells do NOT share a seed set -- paired statistics are invalid")

    # 5 -- declared binary fields really are binary
    for f in plan.binary_fields:
        vals = sorted({r[f] for r in rows})
        add(f"binary::{f}", True, set(vals) <= {0, 1},
            f"{f} takes values {vals}", field=f, values=vals)

    # 6 -- derived fields agree with their definitions, row by row
    for f, rule in plan.derived.items():
        bad = [r for r in rows if r.get(f) != rule.fn(r)]
        add(f"derived::{f}", True, not bad,
            f"{f} == {rule.desc}: {len(bad)} disagreement(s) in {len(rows)} rows",
            field=f, definition=rule.desc, n_disagreements=len(bad))

    # 7 -- the checkpoints on disk are still the ones the spec froze
    for name, (p, want_sha) in plan.checkpoints.items():
        p = Path(p)
        if not p.is_file():
            add(f"checkpoint::{name}", True, False, f"missing: {p}")
            continue
        got = sha256_file(p)
        add(f"checkpoint::{name}", True, got == want_sha,
            f"{name} sha {got[:12]}... vs frozen {want_sha[:12]}...",
            path=str(p), sha256=got, frozen_sha256=want_sha)

    # 8 -- every sealed statistic re-derives from the rows on disk
    for c in plan.claims:
        try:
            v = _vector(rows, c.minuend, plan, c.value_field)
            if c.subtrahend is not None:
                v = v - _vector(rows, c.subtrahend, plan, c.value_field)
            got = _bootstrap(v, plan.n_boot, plan.alpha, plan.rng_seed)
        except Exception as exc:                       # noqa: BLE001 -- report, don't crash
            add(f"claim::{c.name}", True, False, f"could not re-derive: {exc}")
            continue
        diffs = {k: abs(got[k] - float(c.recorded[k]))
                 for k in ("mean", "lcb95", "ucb95") if k in c.recorded}
        worst = max(diffs.values()) if diffs else 0.0
        add(f"claim::{c.name}", True, worst <= plan.tolerance,
            f"max |re-derived - recorded| = {worst:.2e} (tol {plan.tolerance:.0e})",
            recorded=c.recorded, rederived={k: round(x, 9) for k, x in got.items()},
            abs_diff={k: round(x, 12) for k, x in diffs.items()})

    # 9 -- REPORT ONLY: split-half sign consistency. A genuinely noisy result may
    #      legitimately flip sign in one half, so this must never gate a seal --
    #      but an unflagged flip is exactly what makes a small delta look solid.
    ordered = sorted(want)
    half = len(ordered) // 2
    halves = (ordered[:half], ordered[half:])
    for c in plan.claims:
        def halfmean(sub: list[int]) -> float:
            v = _vector(rows, c.minuend, plan, c.value_field, sub)
            if c.subtrahend is not None:
                v = v - _vector(rows, c.subtrahend, plan, c.value_field, sub)
            return float(v.mean())
        try:
            a, b = halfmean(halves[0]), halfmean(halves[1])
        except Exception as exc:               # noqa: BLE001
            # Report it. A check that silently disappears is worse than one that
            # fails: the audit record would look clean while carrying one fewer
            # check than the reader believes. (This self-test caught exactly that.)
            add(f"split_half::{c.name}", False, False, f"not computable: {exc}")
            continue
        add(f"split_half::{c.name}", False, True,
            f"first half {a:+.4f} | second half {b:+.4f}"
            f"{'  <-- SIGN FLIP between halves' if a * b < 0 else ''}",
            first_half=round(a, 6), second_half=round(b, 6), sign_flip=bool(a * b < 0))

    # 10 -- seed class (Rule 9 bridge; silent no-op until the registry exists)
    if plan.seed_class is not None:
        try:
            from experiments.seed_registry import check_block          # noqa: PLC0415
            # experiment_id is load-bearing, not decorative. Rule 9 says allocate
            # the block BEFORE spending seeds, so by audit time a compliant run
            # has its own reservation on record. Without an owner, check_block
            # sees that reservation as a foreign overlap and every Rule-9-abiding
            # run fails its own seal. check_block auto-allows ONLY an exact
            # same-experiment, same-range match, so passing the owner cannot wave
            # through a shifted or widened block.
            ok, msg = check_block(min(want), max(want), plan.seed_class,
                                  experiment_id=plan.experiment_id)
            add("seed_class", True, ok, msg, seed_class=plan.seed_class,
                experiment_id=plan.experiment_id)
        except ImportError:
            add("seed_class", False, True,
                f"declared {plan.seed_class!r}; registry not available to verify",
                seed_class=plan.seed_class)

    return _finalise(plan, checks, rows)


def _finalise(plan: AuditPlan, checks: list[CheckResult], rows: list[dict]) -> dict:
    gating = [c for c in checks if c.gating]
    failed = [c for c in gating if not c.passed]
    return {
        "audit_utc": _now(),
        "git_sha": _git_sha(),
        "passed": not failed,
        "n_checks": len(checks),
        "n_gating": len(gating),
        "n_failed": len(failed),
        "failed_checks": [c.name for c in failed],
        "WHAT_A_PASS_MEANS": "the sealed statistics re-derive from the rows on disk and "
                             "those rows have the structure the frozen spec demanded. "
                             "It does NOT certify that the experimental design or the "
                             "statistical method is correct -- a mistake made identically "
                             "in the evaluator and in this audit would reproduce and pass.",
        "bootstrap": {"n_boot": plan.n_boot, "alpha": plan.alpha, "rng_seed": plan.rng_seed,
                      "generator": "numpy.random.default_rng (PCG64)",
                      "tolerance": plan.tolerance},
        "total_rows_seen": len(rows),
        "checks": [{"name": c.name, "gating": c.gating,
                    "result": "PASS" if c.passed else "FAIL",
                    "detail": c.detail, **c.data} for c in checks],
    }


# ------------------------------------------------------------------ seal ----
def seal(*, out_path: str | Path, payload: dict, plan: AuditPlan,
         state: RunState | None = None, strict: bool = True,
         audit_path: str | Path | None = None) -> dict:
    """The ONLY supported way to produce a sealed result.

    Runs the audit, then moves the run COMPLETE -> AUDITED -> SEALED, or
    COMPLETE -> AUDIT_FAILED. ``payload`` must NOT contain ``status``: the state
    machine owns that field, which is the point of the rule.

    With ``strict`` (the default) a failed audit raises ``SystemExit`` *after*
    both artifacts are written, so the failure is recorded and unignorable.
    """
    out_path = Path(out_path)
    if "status" in payload:
        raise ValueError(
            "payload must not set 'status' -- sealing is a state transition, and "
            "hand-written status strings are exactly what Rule 7 exists to prevent.")

    audit = run_audit(plan)
    ap = Path(audit_path) if audit_path else out_path.with_name(
        out_path.stem.replace("_RESULT", "") + "_AUDIT.json")
    ap.write_text(json.dumps({"record": f"audit of {out_path.name}", **audit}, indent=2),
                  encoding="utf-8")

    if state is not None and state.state == RUNNING:
        state.complete(reason="all episodes written")

    if audit["passed"]:
        if state is not None:
            state._transition(AUDITED, audit=ap.name, n_checks=audit["n_checks"])
            state._transition(SEALED, result=out_path.name)
        status = SEALED
    else:
        if state is not None:
            state._transition(AUDIT_FAILED, audit=ap.name,
                              failed_checks=audit["failed_checks"])
        status = AUDIT_FAILED

    out_path.write_text(json.dumps({
        **payload,
        "status": status,
        "sealed_utc": _now(),
        "git_sha": audit["git_sha"],
        "AUDIT": {"path": ap.name, "passed": audit["passed"],
                  "n_checks": audit["n_checks"], "n_gating": audit["n_gating"],
                  "failed_checks": audit["failed_checks"],
                  "rows_sha256": next((c.get("sha256") for c in audit["checks"]
                                       if c["name"] == "rows_present"), None)},
    }, indent=2), encoding="utf-8")

    bar = "=" * 66
    print(f"\n{bar}\n  RUN STATE: {status}   ({audit['n_failed']}/{audit['n_gating']} "
          f"gating checks failed)\n{bar}")
    for c in audit["checks"]:
        if c["result"] == "FAIL" or not c["gating"]:
            mark = "FAIL" if c["result"] == "FAIL" else "note"
            print(f"  [{mark}] {c['name']}: {c['detail']}")
    print(f"  -> {out_path}\n  -> {ap}")

    if not audit["passed"] and strict:
        raise SystemExit(
            f"AUDIT_FAILED: {out_path.name} is NOT sealed. Failed gating checks: "
            f"{', '.join(audit['failed_checks'])}. The result and its audit are on "
            f"disk and frozen in that state -- diagnose the cause; do not re-run "
            f"into the same label.")
    return audit
