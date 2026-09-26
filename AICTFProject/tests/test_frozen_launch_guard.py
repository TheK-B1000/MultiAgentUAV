"""The frozen-spec launch guard must REFUSE every state in which a launch is not authorized.

Motivating incident (2026-09-26): a detached launcher checked `git diff --quiet HEAD -- <spec>`,
which is silent for an UNTRACKED file, so it launched seed 23200001 from a spec that had never
been committed. Every case below runs in a temporary git repository with a temporary seed
registry, so no case can reach a real launch. The committed-clean case is the positive control.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

import pytest

from experiments import frozen_launch_guard as G
from experiments import seed_registry as SR

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

SPEC = "specs/REPAIR_SPEC.json"
CKPT = "ckpts/final_pi_A.zip"


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _spec(ckpt_sha, **over):
    d = {
        "status": "FROZEN_BEFORE_ANY_TRAINING_STEP",
        "SEEDS": {"registry_experiment_id": "T_REPAIR", "A": 20500001, "B": 20500002},
        "SOURCE_CHECKPOINTS": {"A": {"path": CKPT, "sha256": ckpt_sha},
                               "B": {"path": CKPT, "sha256": ckpt_sha}},
        "LAUNCH": {
            "A": f"python t.py --policy A --seed 20500001 --experiment-id T_REPAIR --load-path {CKPT}",
            "B": f"python t.py --policy B --seed 20500002 --experiment-id T_REPAIR --load-path {CKPT}",
        },
    }
    d.update(over)
    return d


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A git repo holding a warm-start checkpoint, plus a registry with a RESERVED pair."""
    monkeypatch.setattr(SR, "REGISTRY", tmp_path / "SEED_REGISTRY.json")
    monkeypatch.setattr(SR, "ROOT", tmp_path)               # manifest scan: tmp artifacts only
    SR.allocate("T_REPAIR", 20500001, 20500002, "exploratory", "repair pair")
    SR.allocate("T_SPENT", 20600001, 20600002, "exploratory", "finished pair")
    SR.set_status("T_SPENT", "SPENT")
    SR.allocate("T_RETIRED", 20700001, 20700002, "exploratory", "retired pair")
    SR.set_status("T_RETIRED", "RETIRED")

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "ckpts").mkdir()
    (repo / CKPT).write_bytes(b"weights")
    (repo / "specs").mkdir()
    sha = hashlib.sha256(b"weights").hexdigest()
    return repo, sha


def _write(repo, d):
    (repo / SPEC).write_text(json.dumps(d, indent=2), encoding="utf-8")


def _commit(repo, msg="spec"):
    _git(repo, "add", SPEC)
    _git(repo, "commit", "-q", "-m", msg)


def _refusals(repo, policy="A"):
    return [name for name, ok, _ in G.check_launch(repo / SPEC, policy, repo) if not ok]


# ------------------------------------------------------------------ positive control
def test_committed_clean_frozen_spec_passes(world):
    repo, sha = world
    _write(repo, _spec(sha))
    _commit(repo)
    results = G.check_launch(repo / SPEC, "A", repo)
    assert results and all(ok for _, ok, _ in results), results
    assert {n for n, _, _ in results} >= {
        "spec_tracked", "spec_no_unstaged", "spec_no_staged", "spec_frozen", "launch_seed",
        "launch_experiment_id", "registry_seed", "seed_never_trained", "warm_start_path",
        "warm_start_sha256"}
    assert _refusals(repo, "B") == []


# ------------------------------------------------------------------ git state
def test_untracked_spec_is_refused(world):
    """The 23200001 incident: `git diff HEAD` would have called this clean."""
    repo, sha = world
    (repo / "README").write_text("x", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "init")
    _write(repo, _spec(sha))
    assert subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", SPEC]).returncode == 0
    assert _refusals(repo) == ["spec_tracked"]


def test_modified_tracked_spec_is_refused(world):
    repo, sha = world
    _write(repo, _spec(sha))
    _commit(repo)
    _write(repo, _spec(sha, note="edited after commit"))
    assert _refusals(repo) == ["spec_no_unstaged"]


def test_staged_but_uncommitted_spec_is_refused(world):
    repo, sha = world
    _write(repo, _spec(sha))
    _commit(repo)
    _write(repo, _spec(sha, note="staged, not committed"))
    _git(repo, "add", SPEC)
    assert _refusals(repo) == ["spec_no_staged"]


def test_unfrozen_spec_is_refused(world):
    repo, sha = world
    _write(repo, _spec(sha, status="DRAFT_NOT_FROZEN"))
    _commit(repo)
    assert _refusals(repo) == ["spec_frozen"]


# ------------------------------------------------------------------ warm start
def test_warm_start_sha_mismatch_is_refused(world):
    repo, sha = world
    _write(repo, _spec(sha))
    _commit(repo)
    (repo / CKPT).write_bytes(b"different weights")
    assert _refusals(repo) == ["warm_start_sha256"]


def test_missing_warm_start_is_refused(world):
    repo, sha = world
    _write(repo, _spec(sha))
    _commit(repo)
    (repo / CKPT).unlink()
    assert _refusals(repo) == ["warm_start_exists"]


def test_launch_load_path_disagreeing_with_pin_is_refused(world):
    repo, sha = world
    d = _spec(sha)
    d["LAUNCH"]["A"] = d["LAUNCH"]["A"].replace(CKPT, "ckpts/other.zip")
    _write(repo, d)
    _commit(repo)
    assert _refusals(repo) == ["warm_start_path"]


# ------------------------------------------------------------------ seeds
@pytest.mark.parametrize("exp_id,seed,expected", [
    ("NOT_REGISTERED", 20500001, ["registry_seed"]),
    ("T_SPENT", 20600001, ["registry_seed"]),
    ("T_RETIRED", 20700001, ["registry_seed"]),
    ("T_REPAIR", 20500009, ["registry_seed"]),              # outside the block
])
def test_unregistered_spent_retired_or_outside_seed_is_refused(world, exp_id, seed, expected):
    repo, sha = world
    d = _spec(sha)
    d["SEEDS"] = {"registry_experiment_id": exp_id, "A": seed, "B": 20500002}
    d["LAUNCH"]["A"] = f"python t.py --seed {seed} --experiment-id {exp_id} --load-path {CKPT}"
    _write(repo, d)
    _commit(repo)
    assert _refusals(repo) == expected


def test_launch_command_seed_disagreeing_with_SEEDS_is_refused(world):
    repo, sha = world
    d = _spec(sha)
    d["LAUNCH"]["A"] = d["LAUNCH"]["A"].replace("--seed 20500001", "--seed 20500002")
    _write(repo, d)
    _commit(repo)
    assert _refusals(repo) == ["launch_seed"]


def test_launch_command_experiment_id_disagreeing_is_refused(world):
    repo, sha = world
    d = _spec(sha)
    d["LAUNCH"]["A"] = d["LAUNCH"]["A"].replace("T_REPAIR", "T_OTHER")
    _write(repo, d)
    _commit(repo)
    assert _refusals(repo) == ["launch_experiment_id"]


def test_seed_already_trained_is_refused(world, tmp_path):
    repo, sha = world
    aborted = tmp_path / "artifacts" / "x_ABORTED_PREMATURE_LAUNCH"
    aborted.mkdir(parents=True)
    (aborted / "run_manifest.json").write_text(
        json.dumps({"record": G.MANIFEST_RECORD, "seed": 20500001}), encoding="utf-8")
    _write(repo, _spec(sha))
    _commit(repo)
    assert _refusals(repo) == ["seed_never_trained"]
    assert _refusals(repo, "B") == []


def test_missing_spec_keys_are_refused_not_defaulted(world):
    repo, sha = world
    d = _spec(sha)
    del d["SOURCE_CHECKPOINTS"]["A"]
    _write(repo, d)
    _commit(repo)
    assert "spec_keys_present" in _refusals(repo)
