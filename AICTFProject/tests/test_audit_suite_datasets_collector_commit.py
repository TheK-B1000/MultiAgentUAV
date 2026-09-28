"""Cross-scale suite audit: every shard must carry the manifest's collector commit."""
from __future__ import annotations

import json

import numpy as np

from experiments import audit_suite_datasets_cross_scale as A


def _load(tmp_path, name, **arrays):
    p = tmp_path / name
    np.savez_compressed(p, step=np.arange(3), **arrays)
    return np.load(p, allow_pickle=False)


def test_shard_sha_read_from_fingerprint(tmp_path):
    fp = json.dumps({"team_size": 2, "collector_git_sha": "ff7ca32f"}, sort_keys=True)
    assert A.shard_collector_sha(_load(tmp_path, "a.npz", fingerprint=np.array(fp))) == "ff7ca32f"


def test_shard_without_sha_is_absent_not_matching(tmp_path):
    """A pre-fix shard (no sha in its fingerprint) and a pre-fingerprint shard both read as None."""
    old = json.dumps({"team_size": 2}, sort_keys=True)
    assert A.shard_collector_sha(_load(tmp_path, "b.npz", fingerprint=np.array(old))) is None
    assert A.shard_collector_sha(_load(tmp_path, "c.npz")) is None
