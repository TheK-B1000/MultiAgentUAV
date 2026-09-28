"""Layer 3 (resolved experimental-object identity) must actually catch a wrong pole.

The defect this guards (2026-09-26, SUITE_4V4_POLE_B_IDENTITY_AUDIT.json): the 4v4 suite
collector and sharing evaluator installed opponent_spec.pole_B_genome(4) -- plain OP7, the
pole that FAILED 4v4 certification -- while their frozen specs, the KL teachers and the
Separated arm all used certified B3-3. Layer 2 ("one module per stage") stayed green the
whole time, because both paths went through the same module.

Every case here uses temp files or pure functions; no real artifact is written.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.attest_cross_scale_identity as A
from experiments.pole_attestation import (
    PoleAttestationError, certified_pole_genome, pole_identity, resolve_pole_genome,
)
from experiments.opponent_spec import pole_B_genome

FORBIDDEN = ("pole_A_genome", "pole_B_genome")


# ------------------------------------------------ the regression itself ------------
def test_default_resolution_at_4v4_is_certified_B3_3_not_plain_op7():
    """The silent fallback: with no --pole-b-genome-json, 4v4 Pole B used to resolve to
    plain OP7. It must now be the certified B3-3."""
    g = resolve_pole_genome("B", 4)
    assert g.genome_id == "SDS2_B3_LOCKDEF10_2V1"
    assert dict(g.overlay) == {"lock_defender": 10, "enable_2v1": True, "min_alive_for_defender": 4}
    assert dict(g.overlay) != dict(pole_B_genome(4).overlay), \
        "resolver fell back to plain OP7 -- the exact defect"


def test_6v6_resolution_is_unchanged_by_the_repair():
    """6v6's certified pole equals the canonical one, so the repair must not move it."""
    for p in ("A", "B"):
        from experiments.opponent_spec import pole_A_genome
        canon = pole_A_genome(6) if p == "A" else pole_B_genome(6)
        assert dict(resolve_pole_genome(p, 6).overlay) == dict(canon.overlay)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_uncertified_scale_fails_closed(n, monkeypatch, tmp_path):
    """No governing certification: resolution must refuse, not fall back to a default pole.

    Was pinned to the real 2v2 state ("no certification yet") and went stale when 2v2 was
    certified (5f7c6892). The governing record is forced MISSING instead, at every built
    scale, so the refusal is tested independently of which scales happen to be certified.
    """
    import experiments.pole_attestation as PA
    monkeypatch.setattr(PA, "governing_certification",
                        lambda _n: ("MISSING", tmp_path / f"STRATEGIC_DEMAND_{_n}v{_n}_CERTIFICATION.json"))
    for policy in ("A", "B"):
        with pytest.raises(PoleAttestationError, match="MISSING, not CERTIFIED"):
            resolve_pole_genome(policy, n)


# ------------------------------------------------ static: real call nodes only ------
def test_stage_calling_pole_B_genome_directly_fails(tmp_path: Path):
    m = tmp_path / "stage.py"
    m.write_text("from experiments.opponent_spec import pole_B_genome\n"
                 "genomes = {'OP7': pole_B_genome(4)}\n", encoding="utf-8")
    r = A.static_pole_resolution(m, FORBIDDEN)
    assert r["pass"] is False
    assert r["direct_pole_calls"] == ["line 2: pole_B_genome()"]


def test_stage_on_certified_resolver_passes_even_if_comments_name_the_old_call(tmp_path: Path):
    m = tmp_path / "stage.py"
    m.write_text('"""This stage used to call pole_B_genome(4)."""\n'
                 "# never call pole_B_genome() here\n"
                 "from experiments.pole_attestation import resolve_pole_genome\n"
                 "g = resolve_pole_genome('B', 4)\n", encoding="utf-8")
    r = A.static_pole_resolution(m, FORBIDDEN)
    assert r["pass"] is True and r["direct_pole_calls"] == []


def test_stage_with_no_pole_resolution_at_all_fails(tmp_path: Path):
    """Neither the forbidden call nor the resolver: the stage cannot state its pole."""
    m = tmp_path / "stage.py"
    m.write_text("x = 1\n", encoding="utf-8")
    assert A.static_pole_resolution(m, FORBIDDEN)["pass"] is False


# ------------------------------------------------ comparator ------------------------
@pytest.fixture(scope="module")
def certified_4v4_B():
    return pole_identity("B", 4, resolve_pole_genome("B", 4))


def test_plain_op7_recorded_is_mismatch(certified_4v4_B):
    plain = pole_identity("B", 4, pole_B_genome(4))
    r = A.compare_pole_identity(plain, certified_4v4_B)
    assert r["status"] == "MISMATCH"
    assert r.get("hash_agrees") is False


def test_nothing_recorded_is_unknown_not_match(certified_4v4_B):
    assert A.compare_pole_identity(None, certified_4v4_B)["status"] == "UNKNOWN"


def test_certified_recorded_is_match(certified_4v4_B):
    r = A.compare_pole_identity(dict(certified_4v4_B), certified_4v4_B)
    assert r["status"] == "MATCH" and r["hash_agrees"] is True


def test_no_certification_is_pending(certified_4v4_B):
    assert A.compare_pole_identity(dict(certified_4v4_B), None)["status"] == "PENDING"


# ------------------------------------------------ end to end: feed plain OP7 --------
def _manifest(tmp_path: Path, pole_B_identity: dict) -> str:
    a = pole_identity("A", 4, resolve_pole_genome("A", 4))
    p = tmp_path / "FAKE_4V4_DATASET.json"
    p.write_text(json.dumps({"poles": {"A": a, "B": pole_B_identity}}), encoding="utf-8")
    return p


def _run_layer3_with_dataset(monkeypatch, manifest: Path) -> dict:
    scales = {k: dict(v) for k, v in A.SCALES.items()}
    scales["4v4"]["dataset"] = manifest.name
    monkeypatch.setattr(A, "SCALES", scales)
    monkeypatch.setattr(A, "SD", manifest.parent)
    recipe = json.loads(A.RECIPE.read_text(encoding="utf-8"))
    return A.attest_resolved_objects(recipe["RESOLVED_OBJECT_IDENTITY_required"])


def test_attestation_goes_red_when_a_stage_records_plain_op7(tmp_path, monkeypatch):
    """PI's negative control: deliberately feed plain OP7 into one stage."""
    m = _manifest(tmp_path, pole_identity("B", 4, pole_B_genome(4)))
    out = _run_layer3_with_dataset(monkeypatch, m)
    ds_b = [a for a in out["scales"]["4v4"]["artifacts"]
            if a["artifact"] == "distillation dataset" and a["pole"] == "B"][0]
    assert ds_b["status"] == "MISMATCH"
    assert out["pass"] is False
    assert any(f["status"] == "MISMATCH" for f in out["artifact_failures"])


def test_same_stage_recording_certified_B3_3_matches(tmp_path, monkeypatch):
    m = _manifest(tmp_path, pole_identity("B", 4, resolve_pole_genome("B", 4)))
    out = _run_layer3_with_dataset(monkeypatch, m)
    ds = [a for a in out["scales"]["4v4"]["artifacts"] if a["artifact"] == "distillation dataset"]
    assert {a["status"] for a in ds} == {"MATCH"}


# ------------------------------------------------ resolver fails closed -------------
def test_certified_resolver_refuses_when_candidate_file_disagrees_with_cert(tmp_path, monkeypatch):
    """If the certified candidate genome file no longer produces the certified overlay,
    the resolver must refuse rather than hand a stage an uncertified pole."""
    import experiments.pole_attestation as P
    real = json.loads((Path(P.__file__).resolve().parents[1] / "artifacts/strategic_demand/sppo/"
                       "STRATEGIC_DEMAND_4v4_POLE_B3_3_N192_CERTIFICATION.json").read_text(encoding="utf-8"))
    bad_genome = tmp_path / "tampered_b3.json"
    bad_genome.write_text(json.dumps({"genome_id": "SDS2_B3_LOCKDEF10_2V1", "derived_from": "OP7",
                                      "base_opponent": "OP7",
                                      "overlay": {"lock_defender": 3, "enable_2v1": True},
                                      "opening_hold_steps": 0}), encoding="utf-8")
    real["poles"]["B"]["candidate_source"] = str(bad_genome)
    # Drive the shared rebuild directly: since the 2026-09-26 trust gate, a tampered record
    # under the legacy 4v4 filename is refused EARLIER (content != pin), which would hide
    # this overlay check. The downstream consumer and the certification writer both rebuild
    # through this function.
    with pytest.raises(PoleAttestationError, match="certified"):
        P.rebuild_certified_genome("B", 4, real["poles"]["B"], source_name="tampered cert")


# ------------------------------------------------ trust gate: SEALED or pinned legacy
def _cert_like(tmp_path, name, **fields):
    import experiments.pole_attestation as P
    base = json.loads((Path(P.__file__).resolve().parents[1] / "artifacts/strategic_demand/sppo/"
                       "STRATEGIC_DEMAND_4v4_POLE_B3_3_N192_CERTIFICATION.json").read_text(encoding="utf-8"))
    base.update(fields)
    p = tmp_path / name
    p.write_text(json.dumps(base), encoding="utf-8")
    return p


def test_hand_written_frozen_result_is_not_trusted(tmp_path):
    """A new record that merely SAYS CERTIFIED, without a passing seal, must be refused."""
    from experiments.pole_attestation import assert_certification_trustworthy
    p = _cert_like(tmp_path, "STRATEGIC_DEMAND_2v2_CERTIFICATION.json", status="FROZEN_RESULT")
    with pytest.raises(PoleAttestationError, match="run_state.seal"):
        assert_certification_trustworthy(p)


def test_audit_failed_record_is_not_trusted_even_if_certified(tmp_path):
    from experiments.pole_attestation import assert_certification_trustworthy
    p = _cert_like(tmp_path, "STRATEGIC_DEMAND_2v2_CERTIFICATION.json", VERDICT="CERTIFIED",
                   status="AUDIT_FAILED", AUDIT={"passed": False, "failed_checks": ["claim::delta_B"]})
    with pytest.raises(PoleAttestationError):
        assert_certification_trustworthy(p)


def test_sealed_with_passing_audit_is_trusted(tmp_path):
    from experiments.pole_attestation import assert_certification_trustworthy
    p = _cert_like(tmp_path, "STRATEGIC_DEMAND_2v2_CERTIFICATION.json",
                   status="SEALED", AUDIT={"passed": True})
    assert assert_certification_trustworthy(p) == "SEALED"


def test_sealed_status_with_failed_audit_block_is_not_trusted(tmp_path):
    """status alone is not enough: the audit block must also say passed."""
    from experiments.pole_attestation import assert_certification_trustworthy
    p = _cert_like(tmp_path, "STRATEGIC_DEMAND_2v2_CERTIFICATION.json",
                   status="SEALED", AUDIT={"passed": False})
    with pytest.raises(PoleAttestationError):
        assert_certification_trustworthy(p)


def test_legacy_record_trusted_only_byte_for_byte(tmp_path):
    """The two pre-seal governing records are trusted only while unchanged."""
    from experiments.pole_attestation import assert_certification_trustworthy
    real = Path(__file__).resolve().parents[1] / ("artifacts/strategic_demand/sppo/"
                                                  "STRATEGIC_DEMAND_4v4_POLE_B3_3_N192_CERTIFICATION.json")
    assert assert_certification_trustworthy(real) == "LEGACY_PINNED"
    edited = _cert_like(tmp_path, real.name, VERDICT="CERTIFIED")   # same name, different bytes
    edited.write_text(edited.read_text(encoding="utf-8").replace('"lock_defender": 10', '"lock_defender": 9'),
                      encoding="utf-8")
    with pytest.raises(PoleAttestationError, match="has changed"):
        assert_certification_trustworthy(edited)


# ------------------------------------------------ runtime live check ----------------
def _live_core(genome):
    """Build a 4v4 env with `genome` installed and live as Pole B; caller closes."""
    from experiments.opponent_spec import install_keyed_opponent_overlays
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    R2.AGENTS = 4
    env = R2.build_env("cpu", 99990777)
    c = env.core
    c._bt_profile_override = None
    c._sds_opening_hold_steps = 0
    install_keyed_opponent_overlays(c, {"OP7": genome})
    env.env_method("set_phase", phase_from_tag("OP7"))
    env.env_method("set_next_opponent", "SCRIPTED", "OP7")
    env.reset()
    return env


def test_live_check_refuses_plain_op7_even_though_min_alive_matches(certified_4v4_B):
    """The blind spot: plain OP7 and B3-3 both have min_alive_for_defender=4 at 4v4, so a
    min_alive-only self-check passed on the wrong pole. The full live check must refuse."""
    from experiments.pole_attestation import assert_live_matches_identity
    env = _live_core(pole_B_genome(4))
    try:
        t = env.core._bt_resolved_profile_tensors()
        assert int(t["min_alive_for_defender"].flatten()[0].item()) == 4   # the blind spot
        with pytest.raises(PoleAttestationError, match="lock_defender|enable_2v1"):
            assert_live_matches_identity(env.core, certified_4v4_B, context="negative control")
    finally:
        env.close()


def test_live_check_accepts_certified_B3_3(certified_4v4_B):
    from experiments.pole_attestation import assert_live_matches_identity
    env = _live_core(resolve_pole_genome("B", 4))
    try:
        chk = assert_live_matches_identity(env.core, certified_4v4_B)
        assert chk["live"]["lock_defender"] == 10.0 and chk["live"]["enable_2v1"] is True
    finally:
        env.close()


# ------------------------------------------------ only CERTIFIED defines a pole -----
@pytest.mark.parametrize("verdict", ["NOT_CERTIFIED", "MISSING", "UNREADABLE", "UNKNOWN"])
def test_non_certified_governing_record_cannot_define_a_pole(tmp_path, monkeypatch, verdict):
    """A failed certification must not hand its poles downstream. Without this, the
    safety property 'no valid certification => no canonical experiment' is false: the
    collector and evaluators would consume the poles of a NOT_CERTIFIED record."""
    import experiments.pole_attestation as P
    real = json.loads((Path(P.__file__).resolve().parents[1] / "artifacts/strategic_demand/sppo/"
                       "STRATEGIC_DEMAND_4v4_POLE_B3_3_N192_CERTIFICATION.json").read_text(encoding="utf-8"))
    real["VERDICT"] = verdict
    cert = tmp_path / "STRATEGIC_DEMAND_4v4_CERTIFICATION.json"
    cert.write_text(json.dumps(real), encoding="utf-8")
    monkeypatch.setattr(P, "governing_certification", lambda n: (verdict, cert))
    with pytest.raises(PoleAttestationError, match="not CERTIFIED"):
        certified_pole_genome("B", 4)
    with pytest.raises(PoleAttestationError, match="not CERTIFIED"):
        resolve_pole_genome("A", 4)
