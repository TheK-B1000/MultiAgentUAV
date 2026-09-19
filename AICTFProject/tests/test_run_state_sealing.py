"""Rule 7 self-test: prove the seal GATE actually blocks.

A gate that is never shown to reject is not a gate. Each test below corrupts one
thing that has historically gone wrong in this project and asserts the run lands
in AUDIT_FAILED rather than SEALED -- plus the structural test that matters most:
that no code path can reach SEALED without passing through AUDITED.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from experiments.run_state import (AUDIT_FAILED, AUDITED, COMPLETE, RUNNING, SEALED,
                                   AuditPlan, Claim, Derived, IllegalTransition,
                                   RunState, run_audit, seal, sha256_file)

SEEDS = list(range(900001, 900017))          # 16 debug-class seeds
CELLS = [("pi_A", "A"), ("pi_A", "B"), ("pi_B", "A"), ("pi_B", "B")]
FIELDS = ["policy", "pole", "seed", "blue", "red", "win", "margin"]


def _rows():
    """Deterministic synthetic rows with a real, non-zero delta on both axes."""
    rng = np.random.default_rng(11)
    out = []
    for pol, pole in CELLS:
        p = 0.75 if pol[-1] == pole else 0.55       # matched specialist does better
        for s in SEEDS:
            blue = int(rng.random() < p)
            red = 1 - blue
            out.append({"policy": pol, "pole": pole, "seed": s, "blue": blue,
                        "red": red, "win": int(blue > red), "margin": blue - red})
    return out


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plan(tmp_path, rows_csv, claims=(), spec=None, ckpts=None):
    return AuditPlan(
        rows_csv=rows_csv,
        expected_rows=len(CELLS) * len(SEEDS),
        expected_seeds=SEEDS,
        group_by=["policy", "pole"],
        int_fields=["blue", "red", "margin"],
        binary_fields=["win"],
        derived={"win": Derived("int(blue > red)", lambda r: int(r["blue"] > r["red"])),
                 "margin": Derived("blue - red", lambda r: r["blue"] - r["red"])},
        claims=claims,
        spec_path=spec,
        checkpoints=ckpts or {},
        n_boot=2000,                                 # small: this is a unit test
    )


def _delta_claim(rows, name="delta_A", minus=("pi_A", "A"), sub=("pi_B", "A")):
    """Compute the true value the evaluator would seal, so the audit should agree."""
    def vec(pol, pole):
        m = {r["seed"]: r["win"] for r in rows if r["policy"] == pol and r["pole"] == pole}
        return np.array([m[s] for s in SEEDS], dtype=np.float64)
    v = vec(*minus) - vec(*sub)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, v.size, size=(2000, v.size))
    boot = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rec = {"mean": round(float(v.mean()), 6), "lcb95": round(float(lo), 6),
           "ucb95": round(float(hi), 6)}
    return Claim(name=name, recorded=rec,
                 minuend={"policy": minus[0], "pole": minus[1]},
                 subtrahend={"policy": sub[0], "pole": sub[1]}, value_field="win")


@pytest.fixture
def bed(tmp_path):
    rows = _rows()
    csv_path = tmp_path / "rows.csv"
    _write(csv_path, rows)
    spec = tmp_path / "SPEC.json"
    spec.write_text(json.dumps({"status": "FROZEN_BEFORE_ANY_EPISODE"}), encoding="utf-8")
    return tmp_path, rows, csv_path, spec


# --------------------------------------------------------------- happy path --
def test_clean_run_reaches_sealed(bed):
    tmp, rows, csv_path, spec = bed
    st = RunState(tmp, "T").begin()
    assert st.state == RUNNING
    plan = _plan(tmp, csv_path, claims=[_delta_claim(rows)], spec=spec)
    audit = seal(out_path=tmp / "T_RESULT.json", payload={"record": "t"},
                 plan=plan, state=st)
    assert audit["passed"], audit["failed_checks"]
    assert st.state == SEALED
    doc = json.loads((tmp / "T_RESULT.json").read_text(encoding="utf-8"))
    assert doc["status"] == SEALED
    assert doc["AUDIT"]["passed"] is True
    assert doc["AUDIT"]["rows_sha256"] == sha256_file(csv_path)
    # the lifecycle is on disk in order, with no state skipped
    hist = [h["state"] for h in json.loads((tmp / "T_RUN_STATE.json")
                                           .read_text(encoding="utf-8"))["history"]]
    assert hist == [RUNNING, COMPLETE, AUDITED, SEALED]


def test_payload_may_not_set_status(bed):
    tmp, rows, csv_path, spec = bed
    with pytest.raises(ValueError, match="must not set 'status'"):
        seal(out_path=tmp / "X_RESULT.json", payload={"status": "FROZEN_RESULT"},
             plan=_plan(tmp, csv_path, spec=spec))


# ------------------------------------------------------- the gate must bite --
def _expect_failure(tmp, plan, check_prefix):
    st = RunState(tmp, "F").begin()
    audit = seal(out_path=tmp / "F_RESULT.json", payload={"record": "f"},
                 plan=plan, state=st, strict=False)
    assert not audit["passed"]
    assert any(n.startswith(check_prefix) for n in audit["failed_checks"]), \
        f"expected a {check_prefix!r} failure, got {audit['failed_checks']}"
    assert st.state == AUDIT_FAILED
    assert json.loads((tmp / "F_RESULT.json").read_text(encoding="utf-8"))["status"] \
        == AUDIT_FAILED
    return audit


def test_tampered_derived_field_blocks_seal(bed):
    """A 'win' that does not follow from blue/red -- the classic silent corruption."""
    tmp, rows, csv_path, spec = bed
    rows[5]["win"] = 1 - rows[5]["win"]
    _write(csv_path, rows)
    _expect_failure(tmp, _plan(tmp, csv_path, spec=spec), "derived::win")


def test_truncated_rows_block_seal(bed):
    """A crashed run that wrote 63 of 64 rows must never seal as if complete."""
    tmp, rows, csv_path, spec = bed
    _write(csv_path, rows[:-1])
    audit = _expect_failure(tmp, _plan(tmp, csv_path, spec=spec), "row_count")
    assert "cell_seed_block_exact" in audit["failed_checks"]
    assert "pairing_valid" in audit["failed_checks"]


def test_duplicate_seed_blocks_seal(bed):
    """Resume logic double-writing a seed would inflate n and break pairing."""
    tmp, rows, csv_path, spec = bed
    dup = dict(rows[0])
    _write(csv_path, rows + [dup])
    audit = _expect_failure(tmp, _plan(tmp, csv_path, spec=spec), "no_duplicate_seeds")
    assert "pairing_valid" in audit["failed_checks"]


def test_stale_recorded_statistic_blocks_seal(bed):
    """The sealed number no longer follows from the rows -- e.g. a resumed run
    that recomputed from a partially reloaded array."""
    tmp, rows, csv_path, spec = bed
    claim = _delta_claim(rows)
    claim.recorded["mean"] += 0.05
    _expect_failure(tmp, _plan(tmp, csv_path, claims=[claim], spec=spec), "claim::delta_A")


def test_substituted_checkpoint_blocks_seal(bed):
    tmp, rows, csv_path, spec = bed
    ck = tmp / "policy.pt"
    ck.write_bytes(b"weights-v2")
    plan = _plan(tmp, csv_path, spec=spec,
                 ckpts={"pi_A": (ck, sha256_file(ck))})
    st = RunState(tmp, "OK").begin()
    assert seal(out_path=tmp / "OK_RESULT.json", payload={"r": 1}, plan=plan,
                state=st, strict=False)["passed"]
    ck.write_bytes(b"weights-v3")                    # swapped after the spec froze
    _expect_failure(tmp, plan, "checkpoint::pi_A")


def test_unfrozen_spec_blocks_seal(bed):
    tmp, rows, csv_path, spec = bed
    spec.write_text(json.dumps({"status": "DRAFT"}), encoding="utf-8")
    _expect_failure(tmp, _plan(tmp, csv_path, spec=spec), "spec_frozen")


def test_non_binary_outcome_blocks_seal(bed):
    tmp, rows, csv_path, spec = bed
    rows[3]["win"] = 2
    rows[3]["blue"], rows[3]["red"] = 2, 0           # keep 'derived' self-consistent
    rows[3]["margin"] = 2
    _write(csv_path, rows)
    _expect_failure(tmp, _plan(tmp, csv_path, spec=spec), "binary::win")


def test_strict_mode_raises_so_failure_cannot_be_ignored(bed):
    tmp, rows, csv_path, spec = bed
    _write(csv_path, rows[:-1])
    with pytest.raises(SystemExit, match="AUDIT_FAILED"):
        seal(out_path=tmp / "S_RESULT.json", payload={"r": 1},
             plan=_plan(tmp, csv_path, spec=spec), state=RunState(tmp, "S").begin())
    # artifacts are still written: the failure is recorded, not merely raised
    assert json.loads((tmp / "S_RESULT.json").read_text(encoding="utf-8"))["status"] \
        == AUDIT_FAILED
    assert (tmp / "S_AUDIT.json").is_file()


# ------------------------------------------------- structural guarantees -----
def test_sealed_is_unreachable_without_audited(bed):
    """The point of the whole module: there is no edge COMPLETE -> SEALED."""
    tmp, *_ = bed
    st = RunState(tmp, "G").begin()
    st.complete()
    assert st.state == COMPLETE
    with pytest.raises(IllegalTransition, match="COMPLETE -> SEALED"):
        st._transition(SEALED)
    assert st.state == COMPLETE


def test_terminal_states_are_terminal(bed):
    tmp, rows, csv_path, spec = bed
    st = RunState(tmp, "T2").begin()
    seal(out_path=tmp / "T2_RESULT.json", payload={"r": 1},
         plan=_plan(tmp, csv_path, spec=spec), state=st)
    assert st.state == SEALED
    for target in (RUNNING, COMPLETE, AUDITED, AUDIT_FAILED):
        with pytest.raises(IllegalTransition):
            st._transition(target)
    with pytest.raises(IllegalTransition, match="refusing to restart"):
        RunState(tmp, "T2").begin()


def test_audit_failed_cannot_be_overwritten_in_place(bed):
    """A failed audit stays failed. Re-running into the same label is refused,
    so a FAIL can never be quietly replaced by a later PASS."""
    tmp, rows, csv_path, spec = bed
    _write(csv_path, rows[:-1])
    st = RunState(tmp, "AF").begin()
    seal(out_path=tmp / "AF_RESULT.json", payload={"r": 1},
         plan=_plan(tmp, csv_path, spec=spec), state=st, strict=False)
    assert st.state == AUDIT_FAILED
    _write(csv_path, rows)                           # "fix" the data and try again
    with pytest.raises(IllegalTransition, match="refusing to restart"):
        RunState(tmp, "AF").begin()


# ------------------------------------------------------- non-gating checks ---
def test_split_half_is_reported_but_never_gates(bed):
    """A sign flip between halves is information, not grounds to reject: a
    genuinely small, noisy delta can flip legitimately. It must be visible."""
    tmp, rows, csv_path, spec = bed
    # force an extreme split: first half all wins for pi_A, second half all losses
    for r in rows:
        if r["policy"] == "pi_A" and r["pole"] == "A":
            r["blue"] = 1 if r["seed"] < 900009 else 0
        if r["policy"] == "pi_B" and r["pole"] == "A":
            r["blue"] = 0 if r["seed"] < 900009 else 1
        r["red"] = 1 - r["blue"]
        r["win"] = int(r["blue"] > r["red"])
        r["margin"] = r["blue"] - r["red"]
    _write(csv_path, rows)
    audit = run_audit(_plan(tmp, csv_path, claims=[_delta_claim(rows)], spec=spec))
    sh = next(c for c in audit["checks"] if c["name"].startswith("split_half::"))
    assert sh["gating"] is False and sh["sign_flip"] is True
    assert audit["passed"], "a split-half sign flip must not block sealing"
    assert "SIGN FLIP" in sh["detail"]


# ---------------------------------------------------------------------------
# Rule 9 / Rule 7 interaction: the audit must know who OWNS the seed block.
#
# Regression for a real defect found by the first run ever to call seal():
# run_audit called check_block() without an experiment_id, so a run that had
# followed Rule 9 (allocate the block BEFORE spending seeds) saw its OWN
# reservation reported as a foreign overlap and failed its own seal. Every
# Rule-9-compliant run would have hit this.
# ---------------------------------------------------------------------------

def _seed_class_check(plan):
    return next(c for c in run_audit(plan)["checks"] if c["name"] == "seed_class")


def _registry_with(monkeypatch, blocks):
    """Point seed_registry.load at a synthetic registry, so these tests never
    touch artifacts/SEED_REGISTRY.json."""
    from experiments import seed_registry as sr
    monkeypatch.setattr(sr, "load", lambda: {"blocks": blocks})


@pytest.fixture()
def owned_bed(tmp_path, monkeypatch):
    """A block registered in the EXPLORATORY range, owned by MINE."""
    rows = _rows()
    csv_path = tmp_path / "rows.csv"
    _write(csv_path, rows)
    lo, hi = min(SEEDS), max(SEEDS)
    _registry_with(monkeypatch, [{"experiment_id": "MINE", "lo": lo, "hi": hi,
                                  "seed_class": "exploratory", "status": "RESERVED",
                                  "subdivides": None}])
    return tmp_path, csv_path, lo, hi


def test_audit_accepts_a_block_the_run_itself_registered(owned_bed):
    """The defect: this used to FAIL, because the run's own Rule-9 reservation
    looked like someone else's block."""
    tmp_path, csv_path, _lo, _hi = owned_bed
    plan = _plan(tmp_path, csv_path)
    plan = AuditPlan(**{**plan.__dict__, "seed_class": "exploratory",
                        "experiment_id": "MINE"})
    check = _seed_class_check(plan)
    assert check["result"] == "PASS", check["detail"]
    assert "already registered to MINE" in check["detail"]


def test_audit_still_rejects_a_block_owned_by_another_experiment(owned_bed):
    """The guard must keep biting: owner-awareness is not a bypass."""
    tmp_path, csv_path, _lo, _hi = owned_bed
    plan = _plan(tmp_path, csv_path)
    plan = AuditPlan(**{**plan.__dict__, "seed_class": "exploratory",
                        "experiment_id": "SOMEONE_ELSE"})
    check = _seed_class_check(plan)
    assert check["result"] == "FAIL"
    assert "OVERLAPS" in check["detail"]


def test_audit_rejects_an_unowned_block_that_overlaps(owned_bed):
    """Omitting experiment_id keeps the old, strict behaviour -- so existing
    callers that never allocated a block are unaffected by this change."""
    tmp_path, csv_path, _lo, _hi = owned_bed
    plan = _plan(tmp_path, csv_path)
    plan = AuditPlan(**{**plan.__dict__, "seed_class": "exploratory"})
    assert _seed_class_check(plan)["result"] == "FAIL"


def test_owner_cannot_silently_widen_its_own_allocation(tmp_path, monkeypatch):
    """check_block auto-allows ONLY an exact same-range match. A run that spent
    MORE seeds than it reserved must still fail, even under its own name."""
    rows = _rows()
    csv_path = tmp_path / "rows.csv"
    _write(csv_path, rows)
    # registered block is narrower than the seeds actually present in the rows
    _registry_with(monkeypatch, [{"experiment_id": "MINE", "lo": min(SEEDS),
                                  "hi": max(SEEDS) - 4, "seed_class": "exploratory",
                                  "status": "RESERVED", "subdivides": None}])
    plan = _plan(tmp_path, csv_path)
    plan = AuditPlan(**{**plan.__dict__, "seed_class": "exploratory",
                        "experiment_id": "MINE"})
    check = _seed_class_check(plan)
    assert check["result"] == "FAIL", "a widened self-allocation must not be waved through"
    assert "OVERLAPS" in check["detail"]


def test_seed_class_check_is_gating_and_records_the_owner(owned_bed):
    tmp_path, csv_path, _lo, _hi = owned_bed
    plan = _plan(tmp_path, csv_path)
    plan = AuditPlan(**{**plan.__dict__, "seed_class": "exploratory",
                        "experiment_id": "MINE"})
    check = _seed_class_check(plan)
    assert check["gating"] is True
    assert check["experiment_id"] == "MINE"
