"""Suite dataset collector: Rule 9 on both collection blocks, and fingerprint-exact shard resume."""
from __future__ import annotations

import numpy as np
import pytest

from experiments import collect_suite_distillation_states as C
from experiments import seed_registry as SR


def _spec(a="20500001..20500096", b="20600001..20600096", ids=None):
    return {"SEEDS": {"collection_A": a, "collection_B": b,
                      "registry_experiment_ids": ids if ids is not None else {"collection_A": "T_A", "collection_B": "T_B"}}}


@pytest.fixture
def reg(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "REGISTRY", tmp_path / "SEED_REGISTRY.json")
    SR.allocate("T_A", 20500001, 20500096, "exploratory", "pole A")
    SR.allocate("T_B", 20600001, 20600096, "exploratory", "pole B")
    return tmp_path


def test_registered_reserved_blocks_pass(reg):
    assert C.check_collection_seeds(_spec()) == {"collection_A": (20500001, 20500096, "T_A"),
                                                 "collection_B": (20600001, 20600096, "T_B")}


@pytest.mark.parametrize("spec,needle", [
    (_spec(ids={"collection_A": "T_A"}), "registry_experiment_ids.collection_B is missing"),
    (_spec(ids={"collection_A": "T_A", "collection_B": "NOPE"}), "not registered"),
    (_spec(b="20600001..20600095"), "spec says"),
])
def test_bad_seed_declarations_refuse(reg, spec, needle):
    with pytest.raises(SystemExit, match=needle):
        C.check_collection_seeds(spec)


def test_spent_block_refuses(reg):
    SR.set_status("T_B", "SPENT")
    with pytest.raises(SystemExit, match="SPENT"):
        C.check_collection_seeds(_spec())


def _shard(p, fp, rows=3):
    np.savez_compressed(p, fingerprint=np.array(fp), summary_steps=np.array(40), summary_blue=np.array(1),
                        summary_red=np.array(0), step=np.arange(rows, dtype=np.int32))


def test_shard_reused_only_on_exact_fingerprint(tmp_path):
    p = tmp_path / "A_1.npz"
    assert C.shard_is_resumable(p, "fp") is None                      # absent
    _shard(p, "fp")
    assert C.shard_is_resumable(p, "fp") == {"steps": 40, "blue": 1, "red": 0, "decision_rows": 3}
    assert C.shard_is_resumable(p, "other-run") is None               # different configuration
    q = tmp_path / "A_2.npz"
    np.savez_compressed(q, step=np.arange(3))                          # pre-fingerprint shard
    assert C.shard_is_resumable(q, "fp") is None
    r = tmp_path / "A_3.npz"
    r.write_bytes(b"torn")                                             # corrupt write
    assert C.shard_is_resumable(r, "fp") is None
