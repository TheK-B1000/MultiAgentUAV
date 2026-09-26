"""G0 must actually VERIFY frozen hashes, not merely record that files exist.

Before 2026-09-26, G0 of ACTION_INTERFACE_COMMITMENT_MECHANISM passed on
"all protected files present and spec frozen". It hashed the protected artifacts but
never compared the hashes to anything, so when a standing test re-stamped one of them
(the scale-diagnostic contract record, pinned 04ec5c95..., drifted to 6a1e3d9a...) G0
stayed green. These tests pin the repaired behaviour and prove every failure mode
turns G0 red.

Every case drives ``check_prior_artifact_protection`` with files in ``tmp_path``. No real
artifact is read for its content or written.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.eval_action_interface_commitment_mechanism import (
    check_prior_artifact_protection,
)

FROZEN_SPEC = {"status": "FROZEN_BEFORE_EXECUTION"}
NAMES = ("A_SPEC.json", "B_RESULT.json", "C_READING.json")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def world(tmp_path: Path):
    """Three protected files + a pre-run authorization record pinning their raw hashes."""
    files = []
    for i, n in enumerate(NAMES):
        p = tmp_path / n
        p.write_bytes(f'{{\r\n  "id": {i},\r\n  "v": "x"\r\n}}\r\n'.encode())  # CRLF, as on disk
        files.append(p)
    pins = {p.name: _sha(p.read_bytes()) for p in files}
    auth = tmp_path / "AUTH_CONTRACT_RESULT.json"

    def write_auth(table):
        auth.write_text(json.dumps({"gates": {"G0_PRIOR_ARTIFACT_PROTECTION":
                                              {"protected_sha256": table}}}), encoding="utf-8")
    write_auth(pins)
    return {"files": files, "pins": pins, "auth": auth, "write_auth": write_auth}


def _g0(world, mode="POST_RUN"):
    return check_prior_artifact_protection(world["files"], mode, FROZEN_SPEC, world["auth"])


# ---------------------------------------------------------------- positive control ----
def test_untouched_artifacts_pass(world):
    g = _g0(world)
    assert g["pass"] is True
    assert g["action"] == "VERIFY_PINS"
    assert all(f["status"] == "MATCH" for f in g["files"].values())


# ---------------------------------------------------------------- the four PI controls --
def test_modified_file_fails(world):
    """A content change -- even one character -- must turn G0 red."""
    p = world["files"][1]
    p.write_bytes(p.read_bytes().replace(b'"x"', b'"y"'))
    g = _g0(world)
    assert g["pass"] is False
    assert g["files"][p.name]["status"] == "FAIL_CONTENT_CHANGED"
    assert g["failed"] == [p.name]


def test_timestamp_only_change_fails(world):
    """The exact real-world defect: a test re-stamping a utc field. Content changed, so
    G0 must fail -- line-ending tolerance must not swallow a real edit."""
    p = world["files"][0]
    p.write_bytes(p.read_bytes().replace(b'"id": 0', b'"id": 0, "utc": "2026-09-22"'))
    g = _g0(world)
    assert g["pass"] is False
    assert g["files"][p.name]["status"] == "FAIL_CONTENT_CHANGED"


def test_missing_file_fails(world):
    p = world["files"][2]
    p.unlink()
    g = _g0(world)
    assert g["pass"] is False
    assert g["files"][p.name]["status"] == "FAIL_MISSING"


def test_absent_pin_fails(world):
    """A protected file with no frozen pin must fail, not be skipped."""
    table = dict(world["pins"])
    del table[NAMES[0]]
    world["write_auth"](table)
    g = _g0(world)
    assert g["pass"] is False
    assert g["files"][NAMES[0]]["status"] == "FAIL_PIN_ABSENT"


@pytest.mark.parametrize("bad", [
    "04ec5c95",                          # truncated
    "Z" * 64,                            # non-hex
    "04EC5C95E1B9DF88" * 4,              # uppercase hex -- pins are lowercase
    12345,                               # not a string
    "",                                  # empty
])
def test_malformed_pin_fails(world, bad):
    table = dict(world["pins"])
    table[NAMES[1]] = bad
    world["write_auth"](table)
    g = _g0(world)
    assert g["pass"] is False
    assert g["files"][NAMES[1]]["status"] == "FAIL_PIN_MALFORMED"


# ---------------------------------------------------------------- pin-source failures --
def test_missing_pin_source_fails_every_file(world):
    world["auth"].unlink()
    g = _g0(world)
    assert g["pass"] is False
    assert {f["status"] for f in g["files"].values()} == {"FAIL_PIN_SOURCE"}


def test_unreadable_pin_source_fails(world):
    world["auth"].write_text("{not json", encoding="utf-8")
    g = _g0(world)
    assert g["pass"] is False
    assert {f["status"] for f in g["files"].values()} == {"FAIL_PIN_SOURCE"}


def test_pin_table_not_a_mapping_fails(world):
    world["write_auth"](["not", "a", "mapping"])
    g = _g0(world)
    assert g["pass"] is False


def test_unfrozen_spec_fails_even_when_hashes_match(world):
    g = check_prior_artifact_protection(world["files"], "POST_RUN",
                                        {"status": "DRAFT"}, world["auth"])
    assert g["pass"] is False


# ---------------------------------------------------------------- tolerance is narrow --
def test_line_endings_alone_do_not_fail(world):
    """An LF checkout of byte-identical content is the SAME artifact. Only CR/LF may
    differ; the modified/timestamp tests above prove any other change still fails."""
    for p in world["files"]:
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
    g = _g0(world)
    assert g["pass"] is True
    assert {f["matched_form"] for f in g["files"].values()} == {"crlf"}


# ---------------------------------------------------------------- lifecycle --------------
def test_pre_run_records_pins_without_a_pin_source(world):
    """PRE_RUN has no authorization record yet: G0 records the hashes that become pins."""
    world["auth"].unlink()
    g = _g0(world, mode="PRE_RUN")
    assert g["pass"] is True
    assert g["action"] == "RECORD_PINS"
    assert g["protected_sha256"] == world["pins"]


def test_in_flight_verifies_like_post_run(world):
    p = world["files"][0]
    p.write_bytes(p.read_bytes() + b" ")
    assert _g0(world, mode="IN_FLIGHT")["pass"] is False
