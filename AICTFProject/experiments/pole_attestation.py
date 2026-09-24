r"""Certification is the source of truth for WHICH opponent a run trains against.

THE RULE THIS MODULE EXISTS TO ENFORCE
    No experiment may start unless the live resolved environment is independently
    proven identical to the governing certified configuration. Launch arguments
    alone are never evidence of correctness.

MOTIVATING FAILURE (PI_B3_TRAIN_EVAL_POLE_MISMATCH_INVALIDATION.json)
    pi_B3 was trained against canonical Pole B (SDS_PARENT_OP7) but evaluated
    against the certified B3-3 candidate (SDS2_B3_LOCKDEF10_2V1, which adds
    lock_defender=10 and enable_2v1=True). ~17.5 GPU-hours and five seed blocks
    were spent answering the wrong experiment.

    The old guard demanded --pole-b-genome-json only when the governing
    certification's FILENAME contained the substring "CONFIRMATORY_REDESIGN".
    The B3-3 certification certifies a candidate genome exactly the same way but
    its filename does not contain that substring, so the guard stayed silent.
    A filename cannot carry a scientific guarantee.

    The pre-existing LIVE POLE CHECK could not catch it either: it verified that
    the live overlay matched what the LAUNCHER EXPECTED. The launcher had
    resolved the canonical genome, so expectation and reality agreed. That check
    validates internal consistency -- never agreement with the certification.

THE CHAIN THIS MODULE IMPLEMENTS
    certification -> resolved config -> hash/field equality -> live pole
    attestation -> (caller's known-answer contracts) -> GPU training

    Only the last step costs real time. Everything here runs before it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"


class PoleAttestationError(SystemExit):
    """Fail-closed: raised before any environment construction or GPU work."""


# ----------------------------------------------------------------- normalize --
def _scalar(v: Any) -> Any:
    """Collapse a live BT tensor / numpy scalar / python value to a comparable scalar."""
    if v is None:
        return None
    if hasattr(v, "flatten"):
        try:
            v = v.flatten()[0].item()
        except Exception:  # noqa: BLE001
            return None
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        # 1.0 and 1 and True must compare equal; bools are handled above.
        return float(v)
    return v


def _values_equal(a: Any, b: Any) -> bool:
    a, b = _scalar(a), _scalar(b)
    if a is None or b is None:
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-9
    return a == b


def _canonical_overlay(overlay: dict | None) -> dict:
    """Sorted, JSON-safe overlay for hashing and diffing."""
    out = {}
    for k in sorted((overlay or {}).keys()):
        v = _scalar((overlay or {})[k])
        if isinstance(v, float) and float(v).is_integer():
            v = int(v)
        out[str(k)] = v
    return out


def pole_config_hash(policy: str, n: int, genome_id: str, overlay: dict | None) -> str:
    """Deterministic identity of a pole configuration. Train and eval can both
    print this, so agreement is auditable after the fact rather than assumed."""
    payload = json.dumps(
        {"policy": str(policy), "team_size": int(n), "genome_id": str(genome_id),
         "overlay": _canonical_overlay(overlay)},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------- certification --
def certified_pole(cert_path: Path, policy: str, n: int) -> dict:
    """The pole definition the GOVERNING certification actually certified.

    Fails closed on anything it cannot verify -- a record that cannot express
    the pole it certified is not a basis for spending GPU hours (absence is an
    error state, never a default).
    """
    cert_path = Path(cert_path)
    if not cert_path.is_file():
        raise PoleAttestationError(
            f"FAIL-CLOSED: governing certification record not found: {cert_path}")
    try:
        rec = json.loads(cert_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PoleAttestationError(
            f"FAIL-CLOSED: governing certification {cert_path.name} is unreadable: {exc}")

    poles = rec.get("poles")
    if not isinstance(poles, dict) or not poles:
        raise PoleAttestationError(
            f"FAIL-CLOSED: {cert_path.name} has no 'poles' block, so it cannot state which "
            f"opponent it certified. A run cannot be verified against it. Add the poles block "
            f"to the record (every current certification has one) or use a record that does.")
    if str(policy) not in poles:
        raise PoleAttestationError(
            f"FAIL-CLOSED: {cert_path.name} does not describe pole {policy!r}; "
            f"it describes {sorted(poles)}.")
    cert_n = rec.get("team_size")
    if cert_n is not None and int(cert_n) != int(n):
        raise PoleAttestationError(
            f"FAIL-CLOSED: {cert_path.name} certifies team_size={cert_n}, but this run is "
            f"{n}v{n}. A certification does not transfer across team sizes.")

    spec = dict(poles[str(policy)])
    return {
        "certification_record": cert_path.name,
        "certification_verdict": str(rec.get("VERDICT", "UNKNOWN")),
        "policy": str(policy),
        "team_size": int(n),
        "base": spec.get("base"),
        "overlay": _canonical_overlay(spec.get("overlay")),
        "candidate_genome_id": spec.get("candidate_genome_id"),
        "candidate_source": spec.get("candidate_source"),
    }


def governing_certification(n: int) -> tuple[str, Path]:
    """The certification record that GOVERNS team size ``n``, by the project's
    single precedence order.

    Imported from the launcher rather than re-listed here: two copies of a
    precedence order is one copy too many, and an evaluator that resolved a
    DIFFERENT governing record from the trainer could attest agreement with a
    record the training run was never checked against.
    """
    from experiments.train_specialist_scale import _certification_verdict
    return _certification_verdict(int(n))


# ------------------------------------------------------------------- resolve --
def resolve_pole_genome(policy: str, n: int, pole_b_genome_json: str | None = None):
    """Build the genome this run WILL actually instantiate.

    Factored out of the launcher so preflight and tests resolve through exactly
    the same code path the run uses -- a check that resolves differently from
    the run proves nothing.
    """
    from experiments.opponent_spec import (
        _with_full_team_defender_gate, pole_A_genome, pole_B_genome,
    )
    from experiments.sds_genome import SDSGenome

    if str(policy) == "A":
        if pole_b_genome_json:
            raise PoleAttestationError(
                "FAIL-CLOSED: --pole-b-genome-json was supplied for --policy A. "
                "Pole A has no candidate-genome mechanism; this argument would be silently "
                "ignored and the run would not be what the operator believes it is.")
        return pole_A_genome(int(n))

    if pole_b_genome_json:
        p = Path(pole_b_genome_json)
        if not p.is_file():
            raise PoleAttestationError(f"FAIL-CLOSED: --pole-b-genome-json not found: {p}")
        return _with_full_team_defender_gate(
            SDSGenome.from_dict(json.loads(p.read_text(encoding="utf-8"))), int(n))
    return pole_B_genome(int(n))


# ----------------------------------------------------- pre-GPU field equality --
def assert_resolved_matches_certification(policy: str, n: int, cert_path: Path,
                                          genome, *, is_smoke: bool = False) -> dict:
    """FAIL CLOSED unless the pole this run will instantiate IS the certified pole.

    Runs BEFORE environment construction and before any GPU work. Replaces the
    old filename-substring heuristic entirely: the decision is made by comparing
    the resolved genome id and overlay against the certification record.

    Catches BOTH directions of the failure:
      * a certified candidate genome omitted at launch (the B3-3 incident), and
      * an override supplied when the certification certified the canonical pole.
    """
    cert = certified_pole(cert_path, policy, n)
    live_id = str(getattr(genome, "genome_id", "<unknown>"))
    live_overlay = _canonical_overlay(getattr(genome, "overlay", None))

    expected_id = cert["candidate_genome_id"]
    problems: list[str] = []

    if expected_id is not None and live_id != str(expected_id):
        problems.append(
            f"genome id mismatch: certification {cert['certification_record']} certified "
            f"{expected_id!r} for pole {policy}, but this run resolved {live_id!r}. "
            f"(If the certified genome is a candidate file, pass --pole-b-genome-json "
            f"{cert.get('candidate_source') or '<candidate json>'}.)")
    if expected_id is None and cert["overlay"] != live_overlay:
        problems.append(
            f"the certification certified the CANONICAL pole {policy} "
            f"(overlay {cert['overlay']}), but this run resolved {live_id!r} with overlay "
            f"{live_overlay}. An override was supplied that the certification does not cover.")

    missing = {k: v for k, v in cert["overlay"].items()
               if k not in live_overlay or not _values_equal(live_overlay[k], v)}
    if missing:
        detail = ", ".join(
            f"{k}: certified={v!r} resolved={live_overlay.get(k, '<ABSENT>')!r}"
            for k, v in sorted(missing.items()))
        problems.append(f"overlay field mismatch ({detail})")

    certified_hash = pole_config_hash(policy, n, expected_id or live_id, cert["overlay"])
    live_hash = pole_config_hash(policy, n, live_id, live_overlay)

    attestation = {
        "certification_record": cert["certification_record"],
        "certification_verdict": cert["certification_verdict"],
        "policy": str(policy), "team_size": int(n),
        "certified_genome_id": expected_id or "<canonical>",
        "live_genome_id": live_id,
        "certified_overlay": cert["overlay"],
        "live_overlay": live_overlay,
        "certified_config_hash": certified_hash,
        "live_config_hash": live_hash,
        "hashes_match": bool(certified_hash == live_hash),
        "stage": "pre_gpu_field_equality",
    }

    if problems:
        msg = "\n  - ".join(problems)
        banner = (
            f"\nFAIL-CLOSED: resolved pole does NOT match the governing certification.\n"
            f"  certification : {cert['certification_record']} (VERDICT {cert['certification_verdict']})\n"
            f"  policy        : {policy}   team size: {n}v{n}\n"
            f"  CERTIFIED     : {expected_id or '<canonical>'}  overlay={cert['overlay']}\n"
            f"  LIVE          : {live_id}  overlay={live_overlay}\n"
            f"  certified hash: {certified_hash}\n"
            f"  live hash     : {live_hash}\n"
            f"  - {msg}\n"
            f"No environment is built and no GPU work is done. Launch arguments are never "
            f"evidence of correctness; the certification record is.\n")
        if is_smoke:
            print(banner + "  [SMOKE] non-scientific run -- continuing despite mismatch.\n", flush=True)
            attestation["smoke_override_used"] = True
            return attestation
        raise PoleAttestationError(banner)

    if not attestation["hashes_match"]:
        raise PoleAttestationError(
            f"FAIL-CLOSED: every field compared equal but the config hashes differ "
            f"({certified_hash} vs {live_hash}). This means the hash covers something the "
            f"field comparison does not; refusing to proceed on an unexplained difference.")
    return attestation


# ------------------------------------------------- live, zero-step attestation --
def attest_live_pole(core, policy: str, n: int, attestation: dict, *,
                     context: str = "", is_smoke: bool = False) -> dict:
    """Read the LIVE resolved behaviour tree and prove it carries the certified
    overlay. Runs after the env exists but BEFORE any training step.

    This is the layer that would have caught the B3-3 incident even if the
    resolution logic itself were wrong: it interrogates the instantiated
    opponent, not the launcher's intent.
    """
    resolved = core._bt_resolved_profile_tensors()
    certified = attestation["certified_overlay"]

    unresolvable, mismatched, verified = [], [], {}
    for key, want in certified.items():
        if key not in resolved:
            unresolvable.append(key)
            continue
        got = _scalar(resolved.get(key))
        if not _values_equal(got, want):
            mismatched.append(f"{key}: certified={want!r} live={got!r}")
        else:
            verified[key] = got

    problems = []
    if unresolvable:
        problems.append(
            f"certified overlay key(s) {unresolvable} are NOT readable from the live "
            f"behaviour tree, so they cannot be attested. Absence is an error state: a key "
            f"that cannot be verified must not be assumed correct.")
    if mismatched:
        problems.append("live overlay disagrees with the certification: " + "; ".join(mismatched))

    out = {**attestation, "stage": "live_pole_attestation",
           "live_verified_overlay": verified,
           "unresolvable_keys": unresolvable,
           "context": context}

    if problems:
        banner = (f"\nFAIL-CLOSED: LIVE pole attestation failed{(' -- ' + context) if context else ''}.\n"
                  f"  - " + "\n  - ".join(problems) + "\n")
        if is_smoke:
            print(banner + "  [SMOKE] continuing.\n", flush=True)
            out["smoke_override_used"] = True
            return out
        raise PoleAttestationError(banner)

    out["live_attestation_passed"] = True
    return out


def format_attestation_banner(att: dict) -> str:
    """The prominent startup block. Certified and live are always shown together
    so an operator cannot read one without the other."""
    match = "PASS" if att.get("hashes_match") else "FAIL"
    if att.get("live_attestation_passed"):
        match += " (live attested)"
    return (
        f"  CERTIFIED_OPPONENT: {att['certified_genome_id']}  overlay={att['certified_overlay']}\n"
        f"  LIVE_OPPONENT:      {att['live_genome_id']}  overlay={att['live_overlay']}\n"
        f"  CERTIFIED_HASH:     {att['certified_config_hash'][:16]}...\n"
        f"  LIVE_HASH:          {att['live_config_hash'][:16]}...\n"
        f"  MATCH:              {match}   (record: {att['certification_record']})"
    )


# ------------------------------------------------------ cross-policy parity ----
def cross_policy_parity(n: int, cert_path: Path) -> dict:
    """Diff what pole A and pole B require, so a PAIRED study cannot silently
    assume the two launches are symmetric.

    Had this been printed before the B3-3 runs, the asymmetry -- A needs no
    override, B requires SDS2_B3_LOCKDEF10_2V1 -- would have been visible
    instead of assumed away.
    """
    a, b = certified_pole(cert_path, "A", n), certified_pole(cert_path, "B", n)
    lines = []
    for pol, spec in (("A", a), ("B", b)):
        req = spec["candidate_genome_id"]
        lines.append(
            f"    {pol}: {'requires --pole-b-genome-json ' + str(req) if req else 'no genome override (canonical)'}"
            f"  overlay={spec['overlay']}")
    only_a = {k: v for k, v in a["overlay"].items() if b["overlay"].get(k) != v}
    only_b = {k: v for k, v in b["overlay"].items() if a["overlay"].get(k) != v}
    return {"lines": lines, "overlay_only_in_A": only_a, "overlay_only_in_B": only_b,
            "A_requires_override": a["candidate_genome_id"],
            "B_requires_override": b["candidate_genome_id"]}
