"""Scientific-code identity: an artifact-only commit is equivalent, any executed-code change is not.

Every case runs in a throwaway git repo (never this one), with the project nested one level down like
AICTFProject/ so the prefix handling is exercised.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from experiments import code_identity as CI

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                          capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo, files: dict, msg: str) -> str:
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
    base = _commit(tmp_path, {
        CI.COLLECTOR_REL: "X = 1\n", "experiments/helper.py": "H = 1\n", "rl/policy.py": "P = 1\n",
        "gpu_env/core.py": "E = 1\n", "configs/c.json": "{}\n", "game_manager.py": "G = 1\n",
        "artifacts/rows.csv": "a\n", "docs/notes.md": "n\n",
    }, "base")
    return tmp_path, base


def test_artifact_and_docs_only_commit_is_equivalent(repo):
    r, base = repo
    head = _commit(r, {"artifacts/rows2.csv": "b\n", "artifacts/rows.csv": "a\nb\n", "docs/notes.md": "m\n"}, "art")
    eq = CI.code_equivalence(base, head, r / "proj")
    assert eq["equivalent"] and eq["collector_identical"] and eq["scientific_changed_files"] == []
    assert eq["sha_a"] != eq["sha_b"]                                   # real SHAs kept, not unified
    assert eq["changed_top_level"] == ["artifacts", "docs"]


@pytest.mark.parametrize("rel", [
    CI.COLLECTOR_REL,            # the collector itself
    "experiments/helper.py",     # an experiments/ import
    "rl/policy.py",              # policy code
    "gpu_env/core.py",           # environment code
    "configs/c.json",            # config
    "game_manager.py",           # project-root module
])
def test_any_executed_code_change_breaks_equivalence(repo, rel):
    r, base = repo
    head = _commit(r, {rel: "CHANGED = 2\n", "artifacts/rows2.csv": "b\n"}, f"change {rel}")
    eq = CI.code_equivalence(base, head, r / "proj")
    assert not eq["equivalent"]
    assert eq["scientific_changed_files"] == [rel]
    assert eq["collector_identical"] is (rel != CI.COLLECTOR_REL)
    assert eq["scientific_tree_sha256_a"] != eq["scientific_tree_sha256_b"]


def test_new_scientific_file_breaks_equivalence(repo):
    r, base = repo
    head = _commit(r, {"rl/new_module.py": "N = 1\n"}, "add")
    assert not CI.code_equivalence(base, head, r / "proj")["equivalent"]


def test_unknown_sha_raises_not_equivalent(repo):
    r, base = repo
    with pytest.raises(RuntimeError):
        CI.code_equivalence(base, "0" * 40, r / "proj")


def test_dirty_covers_every_scientific_path_not_only_experiments(repo):
    r, _ = repo
    assert not CI.scientific_dirty(r / "proj")
    (r / "proj" / "artifacts" / "rows.csv").write_text("dirty\n", encoding="utf-8")
    assert not CI.scientific_dirty(r / "proj")                          # artifacts never block
    (r / "proj" / "rl" / "policy.py").write_text("P = 2\n", encoding="utf-8")
    assert CI.scientific_dirty(r / "proj")


def test_attestation_refused_when_not_equivalent(repo, tmp_path_factory, monkeypatch):
    r, base = repo
    head = _commit(r, {"rl/policy.py": "P = 2\n"}, "code")
    monkeypatch.setattr(CI, "ROOT", r / "proj")
    out = tmp_path_factory.mktemp("att") / "ATT.json"
    monkeypatch.setattr("sys.argv", ["code_identity", base, head, "--write", str(out)])
    with pytest.raises(SystemExit, match="REFUSING"):
        CI.main()
    assert not out.exists()


def test_attestation_written_once_and_frozen(repo, tmp_path_factory, monkeypatch):
    r, base = repo
    head = _commit(r, {"artifacts/rows2.csv": "b\n"}, "art")
    monkeypatch.setattr(CI, "ROOT", r / "proj")
    out = tmp_path_factory.mktemp("att") / "ATT.json"
    monkeypatch.setattr("sys.argv", ["code_identity", base, head, "--write", str(out)])
    assert CI.main() == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["status"] == "FROZEN" and rec["equivalence"]["equivalent"]
    with pytest.raises(SystemExit, match="never overwritten"):
        CI.main()
