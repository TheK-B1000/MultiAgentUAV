"""The pole certification is a SEALED downstream handoff: prove it, and prove it can fail.

Drives the real certification writer (experiments/certify_strategic_demand_scaled.py::main)
at 2v2 with its output directory AND the governing-record lookup redirected to tmp_path,
so no real certification record is ever created or consumed.

The scripted episode runner is replaced by a deterministic one (GUARD wins on Pole A,
BREACH wins on Pole B, or no demand at all). The statistics are then GENUINELY what the
rows imply -- nothing is faked downstream of the evidence -- so the seal's own
re-derivation, and the verdict and handoff invariants, are exercised for real. Episode
physics are not under test here; the seal and the handoff are.

Tampering is injected at the one point that matters: after the evidence is written and
before it is sealed.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REAL_2V2 = (Path(__file__).resolve().parents[1] / "artifacts/strategic_demand/sppo/"
            "STRATEGIC_DEMAND_2v2_CERTIFICATION.json")
N_SEEDS = 8
SMOKE_BASE = 99950301


@pytest.fixture
def cert_env(tmp_path, monkeypatch):
    """Redirect every path, freeze a spec, and install a deterministic episode runner."""
    import experiments.certify_strategic_demand_scaled as C
    import experiments.strategic_demand_searcher as S
    import experiments.train_specialist_scale as T

    monkeypatch.setattr(C, "OUT_DIR", tmp_path)
    monkeypatch.setattr(T, "SD", tmp_path)
    spec = tmp_path / "SMOKE_2V2_CERT_SPEC.json"
    spec.write_text(json.dumps({"status": "FROZEN_BEFORE_ANY_SEED"}), encoding="utf-8")
    calls = {"n": 0}

    def install(demand: bool):
        def run_episode(*, style, genome, seed, device):
            calls["n"] += 1
            pole_a = genome.base_opponent == "OP6"
            if not demand:
                return {"win": 1}                  # every style wins everywhere: no demand
            win = (pole_a and style is S.GUARD) or ((not pole_a) and style is S.BREACH)
            return {"win": int(win)}
        monkeypatch.setattr(S, "run_episode", run_episode)

    def run(extra=()):
        monkeypatch.setattr(sys, "argv", [
            "x", "--team-size", "2", "--n-seeds", str(N_SEEDS), "--seed-base", str(SMOKE_BASE),
            "--device", "cpu", "--spec", str(spec), "--experiment-id", "SMOKE_2V2_CERT",
            "--seed-class", "smoke", *extra])
        return C.main()

    real_existed = REAL_2V2.exists()
    yield {"C": C, "install": install, "run": run, "tmp": tmp_path, "spec": spec, "calls": calls,
           "monkeypatch": monkeypatch}
    assert REAL_2V2.exists() == real_existed, "a smoke must never create a real 2v2 record"


def _record(tmp):
    return json.loads((tmp / "STRATEGIC_DEMAND_2v2_CERTIFICATION.json").read_text(encoding="utf-8"))


def _tamper_before_seal(monkeypatch, fn):
    """Wrap run_state.seal so `fn(plan, payload)` runs after the evidence is written and
    before the audit reads it -- the window a tampered record would have to use."""
    import experiments.run_state as rs
    real = rs.seal

    def wrapped(*, out_path, payload, plan, **kw):
        fn(plan, payload)
        return real(out_path=out_path, payload=payload, plan=plan, **kw)
    monkeypatch.setattr(rs, "seal", wrapped)


def _rewrite_rows(path: Path, mutate):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows = mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------ the positive path -----------------
def test_genuine_certification_is_sealed_and_hands_off_its_poles(cert_env):
    from experiments.pole_attestation import pole_identity, resolve_pole_genome
    cert_env["install"](demand=True)
    assert cert_env["run"]() == 0
    rec = _record(cert_env["tmp"])
    assert rec["status"] == "SEALED" and rec["AUDIT"]["passed"] is True
    assert rec["VERDICT"] == "CERTIFIED"
    assert rec["team_size"] == 2 and rec["seeds"]["n"] == N_SEEDS
    audit = json.loads((cert_env["tmp"] / "STRATEGIC_DEMAND_2v2_CERTIFICATION_AUDIT.json")
                       .read_text(encoding="utf-8"))
    names = {c["name"]: c["result"] for c in audit["checks"]}
    for must in ("spec_frozen", "row_count", "cell_seed_block_exact", "claim::delta_A",
                 "claim::delta_B", "derived::pole_config_hash", "derived::genome_id",
                 "invariant::verdict_matches_evidence",
                 "invariant::handoff_matches_consumer_rebuild", "seed_class"):
        assert names.get(must) == "PASS", (must, names.get(must))
    for p in ("A", "B"):
        consumed = pole_identity(p, 2, resolve_pole_genome(p, 2))
        assert consumed["pole_config_hash"] == rec["POLE_HANDOFF"]["pole_config_hash"][p]
    state = json.loads((cert_env["tmp"] / "STRATEGIC_DEMAND_2v2_CERTIFICATION_RUN_STATE.json")
                       .read_text(encoding="utf-8"))
    assert [h["state"] for h in state["history"]] == ["RUNNING", "COMPLETE", "AUDITED", "SEALED"]


def test_genuine_no_demand_is_sealed_but_not_consumable(cert_env):
    """SEALED and NOT_CERTIFIED: an honest negative. The record is trustworthy, and it
    still must not define poles for any downstream stage."""
    from experiments.pole_attestation import PoleAttestationError, resolve_pole_genome
    cert_env["install"](demand=False)
    assert cert_env["run"]() == 1
    rec = _record(cert_env["tmp"])
    assert rec["status"] == "SEALED" and rec["VERDICT"] == "NOT_CERTIFIED"
    with pytest.raises(PoleAttestationError, match="not CERTIFIED"):
        resolve_pole_genome("B", 2)


# ------------------------------------------------ negative controls -----------------
def test_faked_statistics_cannot_be_sealed(cert_env):
    """Declaring CERTIFIED statistics the rows do not support must land in AUDIT_FAILED."""
    from experiments.pole_attestation import PoleAttestationError, resolve_pole_genome
    cert_env["install"](demand=False)
    cert_env["monkeypatch"].setattr(cert_env["C"], "_mean_ci", lambda vals, **k: {
        "mean": 0.5, "lcb95": 0.1, "ucb95": 0.9, "n": len(list(vals))})
    assert cert_env["run"]() == 1
    rec = _record(cert_env["tmp"])
    assert rec["status"] == "AUDIT_FAILED" and rec["VERDICT"] == "CERTIFIED"
    failed = set(rec["AUDIT"]["failed_checks"])
    assert {"claim::delta_A", "invariant::verdict_matches_evidence"} <= failed
    with pytest.raises(PoleAttestationError):
        resolve_pole_genome("B", 2)      # CERTIFIED on paper, but not trusted


def test_tampered_row_pole_hash_fails_the_seal(cert_env):
    def tamper(plan, payload):
        def m(rows):
            rows[3]["pole_config_hash"] = "0" * 64
            return rows
        _rewrite_rows(plan.rows_csv, m)
    cert_env["install"](demand=True)
    _tamper_before_seal(cert_env["monkeypatch"], tamper)
    assert cert_env["run"]() == 1
    rec = _record(cert_env["tmp"])
    assert rec["status"] == "AUDIT_FAILED"
    assert "derived::pole_config_hash" in rec["AUDIT"]["failed_checks"]


def test_missing_seed_fails_the_seal(cert_env):
    def tamper(plan, payload):
        _rewrite_rows(plan.rows_csv, lambda rows: [r for r in rows if int(r["seed"]) != SMOKE_BASE + 2])
    cert_env["install"](demand=True)
    _tamper_before_seal(cert_env["monkeypatch"], tamper)
    assert cert_env["run"]() == 1
    rec = _record(cert_env["tmp"])
    assert rec["status"] == "AUDIT_FAILED"
    assert {"row_count", "cell_seed_block_exact"} <= set(rec["AUDIT"]["failed_checks"])


def test_tampered_handoff_hash_fails_the_seal(cert_env):
    def tamper(plan, payload):
        payload["POLE_HANDOFF"]["pole_config_hash"]["B"] = "f" * 64
    cert_env["install"](demand=True)
    _tamper_before_seal(cert_env["monkeypatch"], tamper)
    assert cert_env["run"]() == 1
    rec = _record(cert_env["tmp"])
    assert rec["status"] == "AUDIT_FAILED"
    assert "invariant::handoff_matches_consumer_rebuild" in rec["AUDIT"]["failed_checks"]


def test_tampered_pole_overlay_fails_the_seal(cert_env):
    """A pole block edited to a different overlay no longer rebuilds to the pinned pole."""
    def tamper(plan, payload):
        payload["poles"]["B"]["overlay"] = {"lock_defender": 10, "enable_2v1": True}
    cert_env["install"](demand=True)
    _tamper_before_seal(cert_env["monkeypatch"], tamper)
    assert cert_env["run"]() == 1
    assert "invariant::handoff_matches_consumer_rebuild" in _record(cert_env["tmp"])["AUDIT"]["failed_checks"]


def test_flipped_verdict_fails_the_seal(cert_env):
    """The record says CERTIFIED while its own rows imply NOT_CERTIFIED."""
    def tamper(plan, payload):
        payload["VERDICT"] = "CERTIFIED"
    cert_env["install"](demand=False)
    _tamper_before_seal(cert_env["monkeypatch"], tamper)
    assert cert_env["run"]() == 1
    rec = _record(cert_env["tmp"])
    assert rec["status"] == "AUDIT_FAILED"
    assert "invariant::verdict_matches_evidence" in rec["AUDIT"]["failed_checks"]


# ------------------------------------------------ pre-flight: refuse before spending --
@pytest.mark.parametrize("case", ["no_spec", "unfrozen_spec", "no_experiment_id",
                                  "unregistered_block", "one_shot"])
def test_preflight_refuses_before_any_episode(cert_env, case):
    cert_env["install"](demand=True)
    extra = []
    if case == "unfrozen_spec":
        cert_env["spec"].write_text(json.dumps({"status": "DRAFT"}), encoding="utf-8")
    if case == "one_shot":
        (cert_env["tmp"] / "STRATEGIC_DEMAND_2v2_CERTIFICATION.json").write_text("{}", encoding="utf-8")
    argv_fix = {"no_spec": ("--spec", ""), "no_experiment_id": ("--experiment-id", "")}
    if case in argv_fix:
        flag, val = argv_fix[case]
        extra = [flag, val]
    if case == "unregistered_block":
        extra = ["--seed-class", "sealed_confirmatory"]      # a real block that is not reserved
    with pytest.raises(SystemExit):
        cert_env["run"](extra)
    assert cert_env["calls"]["n"] == 0, f"{case}: an episode ran before the pre-flight refused"
