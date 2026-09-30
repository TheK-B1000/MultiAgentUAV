"""Executed-code identity (6v6 audit, PI 2026-09-29): a later collection is equivalent to the reference
iff every file the collector executes is identical, or its exact blob change was reviewed and attested."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from experiments import code_identity as CI

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
CLOSURE = ["experiments/collect_suite_distillation_states.py", "rl/policy.py", "gpu_env/core.py"]


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                          capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo, files, msg):
    for rel, text in files.items():
        p = repo / "proj" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    base = _commit(tmp_path, {CLOSURE[0]: "X = 1\n", CLOSURE[1]: "P = 1\n", CLOSURE[2]: "E = 1\n",
                              "experiments/eval_other.py": "V = 1\n"}, "base")
    return tmp_path, base


def eq(r, a, b, attested, files=CLOSURE):
    return CI.executed_code_equivalence(a, b, attested, r / "proj", files=list(files))


def test_identical_closure_is_equivalent_even_if_other_code_changed(repo):
    r, base = repo
    head = _commit(r, {"experiments/eval_other.py": "V = 2\n"}, "evaluator only")   # not executed by collection
    out = eq(r, base, head, [])
    assert out["equivalent"] and out["differing"] == []


def test_unreviewed_change_in_closure_breaks_equivalence(repo):
    r, base = repo
    head = _commit(r, {CLOSURE[1]: "P = 2\n"}, "policy edit")
    out = eq(r, base, head, [])
    assert not out["equivalent"] and [d["file"] for d in out["unreviewed"]] == [CLOSURE[1]]


def test_attested_change_is_accepted_only_for_its_exact_blobs(repo):
    r, base = repo
    mid = _commit(r, {CLOSURE[0]: "X = 1  # provenance\n"}, "reviewed")
    d = CI.executed_code_diff(base, mid, CLOSURE, r / "proj")
    assert len(d) == 1
    assert eq(r, base, mid, d)["equivalent"]
    later = _commit(r, {CLOSURE[0]: "X = 2  # provenance\n"}, "edited again after review")
    out = eq(r, base, later, d)
    assert not out["equivalent"] and out["unreviewed"][0]["file"] == CLOSURE[0]


def test_new_file_in_closure_needs_attesting(repo):
    r, base = repo
    head = _commit(r, {"experiments/code_identity.py": "I = 1\n"}, "new helper the collector imports")
    files = CLOSURE + ["experiments/code_identity.py"]
    out = eq(r, base, head, [], files)
    assert not out["equivalent"] and out["unreviewed"][0]["blob_a"] is None
    assert eq(r, base, head, out["differing"], files)["equivalent"]
