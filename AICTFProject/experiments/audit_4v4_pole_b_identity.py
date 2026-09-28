"""Which Pole B did each 4v4 run ACTUALLY instantiate? (read-only)

The frozen 4v4 records name Pole B = B3-3 (SDS2_B3_LOCKDEF10_2V1:
lock_defender=10, enable_2v1=True, min_alive_for_defender=4). Two resolution paths exist:
``pole_attestation.resolve_pole_genome`` gives B3-3 only when ``--pole-b-genome-json`` is
passed, and ``opponent_spec.pole_B_genome(4)`` always gives plain OP7 with
``min_alive_for_defender=4`` -- the pole that FAILED 4v4 certification.

Every run is classified from its OWN evidence, strongest class first:

  LIVE_ATTESTATION        the run recorded the live resolved profile at run time and a
                          config-hash match against the certification
  RECORDED_IN_RESULT      the run's result record carries a poles.B block
  CODE_PATH_NO_OVERRIDE   no run-time record exists; the producing code (at the commit that
                          ran) resolves Pole B through a fixed function and accepted no
                          Pole-B override argument, and that function is unchanged since
                          2026-09-05. Weakest class -- labelled so.

Hashes use ``pole_attestation.pole_config_hash``, the same function the live attestation
uses, so the plain-OP7 and B3-3 identities are compared on one scale.

Writes SUITE_4V4_POLE_B_IDENTITY_AUDIT.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.opponent_spec import pole_B_genome  # noqa: E402
from experiments.pole_attestation import pole_config_hash  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC4 = ROOT / "artifacts" / "scale_4v4_specialists"
OUT = SD / "SUITE_4V4_POLE_B_IDENTITY_AUDIT.json"
CERT = SD / "STRATEGIC_DEMAND_4v4_POLE_B3_3_N192_CERTIFICATION.json"
N = 4


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_show(commit: str, path: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "show", f"{commit}:{path}"],
                                   text=True, encoding="utf-8")


def _norm(o: dict | None) -> dict:
    return {k: (int(v) if isinstance(v, float) and float(v).is_integer() else v)
            for k, v in sorted((o or {}).items())}



def _is_pole_b_override(arg_line: str) -> bool:
    """True only for an argument that could substitute the Pole-B genome. Deliberately
    narrow: '--n-per-pole' (an episode count) contains 'pole' but overrides nothing, and
    matching it would fail the audit closed on a false positive."""
    s = arg_line.lower()
    return "genome" in s or "pole-b" in s or "pole_b" in s or "opponent" in s


def main() -> int:
    cert = json.loads(CERT.read_text(encoding="utf-8"))
    certB = cert["poles"]["B"]
    b33_id = certB.get("candidate_genome_id") or "SDS2_B3_LOCKDEF10_2V1"
    b33_overlay = _norm(certB["overlay"])
    b33_hash = pole_config_hash("B", N, b33_id, b33_overlay)

    plain = pole_B_genome(N)
    plain_overlay = _norm(dict(plain.overlay or {}))
    plain_hash = pole_config_hash("B", N, plain.genome_id, plain_overlay)
    if plain_hash == b33_hash:
        raise SystemExit("FAIL-CLOSED: plain OP7 and B3-3 hash identically; audit is meaningless")

    def classify(overlay: dict) -> str:
        o = _norm(overlay)
        if o == b33_overlay:
            return "B3_3_CERTIFIED"
        if o == plain_overlay:
            return "PLAIN_OP7_UNCERTIFIED"
        return f"OTHER{o}"

    runs = []

    # 1. corrected pi_B3 TRAINING -- live attestation in its run manifest
    m = json.loads((SPEC4 / "pi_B_specialist_4v4_b3_entity_repair_corrected" /
                    "run_manifest.json").read_text(encoding="utf-8"))
    att = next((v for v in (m, m.get("pole_attestation") or {}) if "live_overlay" in v), None)
    if att is None:  # search one level down
        att = next(v for v in m.values() if isinstance(v, dict) and "live_overlay" in v)
    runs.append({"run": "pi_B3 corrected specialist TRAINING (KL teacher z1)",
                 "role": "teacher", "evidence_class": "LIVE_ATTESTATION",
                 "evidence": "scale_4v4_specialists/pi_B_specialist_4v4_b3_entity_repair_corrected/run_manifest.json",
                 "live_opponent": att.get("LIVE_OPPONENT"), "overlay": _norm(att["live_overlay"]),
                 "live_config_hash": att.get("live_config_hash") or att.get("certified_config_hash"),
                 "hash_match_recorded": att.get("pole_config_hashes_match"),
                 "pole_B": classify(att["live_overlay"])})

    # 2. teacher-pair crossover -- live attestation banner in its launch log
    log = (SD / "corrected_crossover.log").read_bytes().decode("utf-8", errors="replace")
    blk = log.split("POLE B ATTESTATION", 1)[1].split("SPECIALIST CROSSOVER EVAL", 1)[0]
    live_line = next(l for l in blk.splitlines() if "LIVE_OPPONENT" in l)
    live_hash = next(l for l in blk.splitlines() if "LIVE_HASH" in l).split()[-1].rstrip(".")
    match = "MATCH:              PASS" in blk or ("MATCH:" in blk and "PASS" in blk)
    ov = json.loads(live_line.split("overlay=", 1)[1].replace("'", '"').replace("True", "true"))
    runs.append({"run": "teacher-pair CONFIRMATORY crossover (Delta_B -0.156, seeds 18300001..128)",
                 "role": "teacher_reference", "evidence_class": "LIVE_ATTESTATION",
                 "evidence": "corrected_crossover.log (POLE B ATTESTATION banner, written at launch)",
                 "live_opponent": live_line.split("LIVE_OPPONENT:", 1)[1].split("overlay")[0].strip(),
                 "overlay": _norm(ov), "live_config_hash_prefix": live_hash,
                 "hash_match_recorded": bool(match),
                 "live_hash_equals_b33": b33_hash.startswith(live_hash),
                 "pole_B": classify(ov)})

    # 3/4. Separated crossovers -- poles block recorded in the result
    for name, lab in (("SEPARATED exploratory crossover", "DEFEND_ATTACK_SPLIT_POLICY_A_V1"),
                      ("SEPARATED confirmatory crossover (the 4v4 reference PASS)",
                       "DEFEND_ATTACK_SPLIT_POLICY_A_V1_CONFIRMATORY_V1")):
        r = json.loads((SD / f"{lab}_SPECIALIST_CROSSOVER_EVAL_RESULT.json").read_text(encoding="utf-8"))
        b = r["poles"]["B"]
        runs.append({"run": name, "role": "separated_arm", "evidence_class": "RECORDED_IN_RESULT",
                     "evidence": f"{lab}_SPECIALIST_CROSSOVER_EVAL_RESULT.json#poles.B",
                     "candidate_genome_id": b.get("candidate_genome_id"),
                     "overlay": _norm(b["overlay"]), "pole_B": classify(b["overlay"])})

    # 5. suite distillation DATASET collection -- code path, no override
    col = _git_show("ae89f636", "AICTFProject/experiments/collect_suite_distillation_states_4v4.py")
    col_args = [l.strip() for l in col.splitlines() if 'add_argument("--' in l]
    col_override = any(_is_pole_b_override(a) for a in col_args)
    col_uses_plain = '{"OP7": pole_B_genome(N_AGENTS)}' in col
    man = json.loads((SD / "SUITE_DISTILLATION_4V4_DATASET.json").read_text(encoding="utf-8"))
    runs.append({"run": "SUITE_DISTILLATION_4V4 dataset collection (Pole-B states the students trained on)",
                 "role": "suite_student_training_data", "evidence_class": "CODE_PATH_NO_OVERRIDE",
                 "evidence": "collect_suite_distillation_states_4v4.py @ ae89f636 (first commit; the file "
                             "was untracked when it ran on 2026-09-24, so the run-time bytes are not in git)",
                 "manifest_records_pole": "poles" in man,
                 "cli_args": col_args, "pole_B_override_argument_existed": col_override,
                 "installs": "pole_B_genome(N_AGENTS)" if col_uses_plain else "UNKNOWN",
                 "overlay": plain_overlay if col_uses_plain else None,
                 "pole_B": classify(plain_overlay) if (col_uses_plain and not col_override) else "UNKNOWN"})

    # 6. the four student crossovers -- code path, no override
    ev = _git_show("ae89f636", "AICTFProject/experiments/eval_suite_sharing_crossover_4v4.py")
    ev_args = [l.strip() for l in ev.splitlines() if 'add_argument(' in l]
    ev_override = any(_is_pole_b_override(a) for a in ev_args)
    ev_uses_plain = '"B": {"OP7": pole_B_genome(N_AGENTS)}' in ev
    for lab in ("SUITE_FULLY_SHARED_Z_4V4_EXPLORATORY", "SUITE_SHARE_ENCODER_4V4_EXPLORATORY",
                "SUITE_SHARE_BACKBONE_4V4_EXPLORATORY", "SUITE_SHARE_MACRO_4V4_EXPLORATORY"):
        sealed = json.loads((SD / f"{lab}_CROSSOVER_EVAL_RESULT.json").read_text(encoding="utf-8"))
        runs.append({"run": f"{lab} student crossover", "role": "suite_student_eval",
                     "evidence_class": "CODE_PATH_NO_OVERRIDE",
                     "evidence": "eval_suite_sharing_crossover_4v4.py @ ae89f636 (the runner that produced the rows)",
                     "result_records_pole": "poles" in sealed,
                     "cli_args": ev_args, "pole_B_override_argument_existed": ev_override,
                     "installs": "pole_B_genome(N_AGENTS)" if ev_uses_plain else "UNKNOWN",
                     "overlay": plain_overlay if ev_uses_plain else None,
                     "pole_B": classify(plain_overlay) if (ev_uses_plain and not ev_override) else "UNKNOWN",
                     "sealed_status": sealed.get("status"), "sealed_verdict": sealed.get("scientific_verdict")})

    fn_last = subprocess.check_output(
        ["git", "-C", str(REPO), "log", "-1", "--format=%h %ad", "--date=short", "--",
         "AICTFProject/experiments/opponent_spec.py"], text=True).strip()

    on_b33 = [r["run"] for r in runs if r["pole_B"] == "B3_3_CERTIFIED"]
    on_plain = [r["run"] for r in runs if r["pole_B"] == "PLAIN_OP7_UNCERTIFIED"]
    unknown = [r["run"] for r in runs if r["pole_B"] not in ("B3_3_CERTIFIED", "PLAIN_OP7_UNCERTIFIED")]

    print(f"4v4 POLE B IDENTITY AUDIT  {_now()}")
    print(f"  B3-3 (certified)   {b33_id:<24} {b33_overlay}  hash {b33_hash[:16]}")
    print(f"  plain OP7          {plain.genome_id:<24} {plain_overlay}  hash {plain_hash[:16]}")
    print(f"  opponent_spec.py last changed: {fn_last}\n")
    for r in runs:
        tag = {"B3_3_CERTIFIED": "B3-3 ok ", "PLAIN_OP7_UNCERTIFIED": "PLAIN OP7"}.get(r["pole_B"], "UNKNOWN")
        print(f"  [{tag}] {r['evidence_class']:<22} {r['run']}")
    print(f"\n  on certified B3-3: {len(on_b33)}   on plain OP7: {len(on_plain)}   unknown: {len(unknown)}")

    OUT.write_text(json.dumps({
        "record": "4v4 Pole B identity audit", "utc": _now(), "scope": "READ-ONLY",
        "question": "Which Pole B did each 4v4 run actually instantiate?",
        "certified_pole_B": {"genome_id": b33_id, "overlay": b33_overlay,
                             "pole_config_hash": b33_hash, "certification": CERT.name},
        "plain_op7_pole_B": {"genome_id": plain.genome_id, "overlay": plain_overlay,
                             "pole_config_hash": plain_hash,
                             "resolved_by": "opponent_spec.pole_B_genome(4)",
                             "note": "the pole that failed 4v4 certification; B3-3 was designed to replace it"},
        "opponent_spec_last_changed": fn_last,
        "evidence_classes": {
            "LIVE_ATTESTATION": "live resolved profile + config-hash match recorded at run time",
            "RECORDED_IN_RESULT": "poles.B block in the run's own result record",
            "CODE_PATH_NO_OVERRIDE": "no run-time record; producing code resolves Pole B through a fixed "
                                     "function with no override argument; weakest class"},
        "runs": runs,
        "SUMMARY": {"on_certified_B3_3": on_b33, "on_plain_OP7": on_plain, "unknown": unknown},
        "DAMAGE_SCOPE": (
            "Bounded to the two SUITE stages: the 4v4 distillation dataset's Pole-B states and the four "
            "student crossovers. Every teacher and Separated run is on certified B3-3 (two by live "
            "attestation, two by recorded result). Because the dataset itself was collected on plain OP7, "
            "the students were TRAINED on the wrong Pole-B state distribution, so re-evaluating them on "
            "B3-3 would not repair the suite result: the dataset must be recollected and the students "
            "redistilled."),
    }, indent=2), encoding="utf-8")
    print(f"  -> {OUT}")
    return 0 if not unknown else 1


if __name__ == "__main__":
    raise SystemExit(main())
