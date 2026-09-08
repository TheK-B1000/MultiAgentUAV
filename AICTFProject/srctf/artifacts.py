"""Authoritative classification of scan artifacts. One truth source.

Written after filename-pattern bookkeeping produced a wrong status report:
`^states_.*\\.json$` also matches `states_X.json.manifest.json`. Every consumer
-- resume logic, monitoring summaries, merger readiness, final reporting --
asks this module instead of running its own glob, grep or file count.

Classification is by SCHEMA, never by name. A shard is a shard because it maps a
policy seed to records carrying episode_key and utilities.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

SHARD = "shard"
SHARD_EMPTY = "shard_empty"
MANIFEST = "manifest"
RESULT = "result"
OTHER = "other"
UNREADABLE = "unreadable"

# Cell states, in the vocabulary the C7 Stage 2 crash made necessary.
COMPLETE = "complete"            # shard + manifest
SHARD_ONLY = "shard_only"        # measurement exists, provenance missing
NOT_PRESENT = "not_present"      # never completed


def classify(path: Path) -> str:
    """Classify one JSON artifact by its schema."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return UNREADABLE
    if not isinstance(d, dict):
        return OTHER
    if "states_file_sha256" in d and "policies" in d:
        return MANIFEST
    if d and all(isinstance(v, list) for v in d.values()):
        rows = next(iter(d.values()), [])
        if rows and isinstance(rows[0], dict) and "episode_key" in rows[0] \
                and "utilities" in rows[0]:
            return SHARD
        if not rows:
            return SHARD_EMPTY
    if "verdict" in d:
        return RESULT
    return OTHER


def shard_paths(shard_dir: Path, pseed, opp) -> tuple[Path, Path]:
    sf = Path(shard_dir) / f"states_{pseed}_{opp}.json"
    return sf, Path(str(sf) + ".manifest.json")


def cell_state(shard_dir: Path, pseed, opp) -> str:
    """COMPLETE / SHARD_ONLY / NOT_PRESENT for one (policy, opponent) cell."""
    sf, mf = shard_paths(shard_dir, pseed, opp)
    has_shard = sf.exists() and classify(sf) in (SHARD, SHARD_EMPTY)
    if not has_shard:
        return NOT_PRESENT
    return COMPLETE if (mf.exists() and classify(mf) == MANIFEST) else SHARD_ONLY


@dataclasses.dataclass(frozen=True)
class CellVerdict:
    cell: str
    ok: bool
    checks: dict

    def failures(self) -> list[str]:
        return [k for k, v in self.checks.items() if v is False]


def verify_cell(shard_dir: Path, pseed, opp, *, expected_states: int,
                expected_seed_lo: int, expected_seed_hi: int,
                expected_checkpoint_sha: str | None = None,
                expected_lc_fields: int | None = None) -> CellVerdict:
    """Full end-to-end verification of one completed cell.

    This is the check that distinguishes "another JSON file appeared" from "a
    valid cell landed". Used to confirm the arm-aware manifest fix on the first
    NEW cell rather than trusting that a tested code path also works live.
    """
    sf, mf = shard_paths(shard_dir, pseed, opp)
    checks: dict = {}
    checks["shard_present"] = sf.exists()
    checks["manifest_present"] = mf.exists()
    if not checks["shard_present"]:
        return CellVerdict(f"{pseed}/{opp}", False, checks)

    checks["shard_schema"] = classify(sf) == SHARD
    shard = json.loads(sf.read_text(encoding="utf-8"))
    checks["single_policy_key"] = list(shard) == [str(pseed)]
    rows = shard.get(str(pseed), [])
    checks["state_count"] = len(rows) == expected_states
    opps = {r["episode_key"].split(":")[0] for r in rows}
    checks["opponent_pure"] = opps == {opp}
    seeds = {int(r["episode_key"].split(":")[1]) for r in rows}
    checks["seeds_in_block"] = bool(seeds) and min(seeds) >= expected_seed_lo \
        and max(seeds) <= expected_seed_hi
    if expected_lc_fields is not None:
        checks["legal_context_fields"] = all(
            len(r.get("legal_context", {})) == expected_lc_fields for r in rows)

    if checks["manifest_present"]:
        checks["manifest_schema"] = classify(mf) == MANIFEST
        man = json.loads(mf.read_text(encoding="utf-8"))
        raw = sf.read_text(encoding="utf-8")
        checks["manifest_hash_matches_shard"] = (
            hashlib.sha256(raw.encode("utf-8")).hexdigest() == man.get("states_file_sha256"))
        checks["manifest_names_this_cell"] = (
            man.get("opponents") == [opp] and man.get("policies") == [str(pseed)])
        if expected_checkpoint_sha is not None:
            checks["manifest_checkpoint_sha"] = (
                man.get("policy_checkpoint_sha256", {}).get(str(pseed)) == expected_checkpoint_sha)

    return CellVerdict(f"{pseed}/{opp}", all(v is not False for v in checks.values()), checks)
