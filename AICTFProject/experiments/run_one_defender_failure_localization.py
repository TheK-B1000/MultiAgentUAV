"""EXPLORATORY_ONE_DEFENDER_FAILURE_LOCALIZATION_V1_SPEC.json.

DIAGNOSTIC. EXPLORATORY. NON-GATING. Replays the exact 768 (policy, pole, arm, seed)
episodes already sealed in DEFENDER_INJECTION_CAUSAL_BRIDGE_RESULT.json -- same seeds,
same checkpoints, same code paths, deterministic -- to capture the full per-tick trace
that run was throwing away, purely to look for WHERE the sealed -0.177 (pi_A@B) and
-0.094 (pi_B@B) harms arise. Reports paired (+1D - native) descriptive differences per
mechanism family. Gates nothing, spends no seed, computes no verdict.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.probe_learned_composition as P  # noqa: E402
from experiments.run_routed_composition_outcome import _bootstrap  # noqa: E402
from experiments.tqdm_loop import tqdm_iter  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "EXPLORATORY_ONE_DEFENDER_FAILURE_LOCALIZATION_V1_SPEC.json"
CAUSAL_ROWS = SD / "DEFENDER_INJECTION_CAUSAL_BRIDGE_ROWS.csv"
OUT_ROWS = SD / "EXPLORATORY_ONE_DEFENDER_FAILURE_LOCALIZATION_ROWS.csv"
OUT_RESULT = SD / "EXPLORATORY_ONE_DEFENDER_FAILURE_LOCALIZATION_RESULT.json"

DEVICE = "cuda"
METRIC_FIELDS = (
    "time_to_first_carry", "time_to_first_carry_excl_defender",
    "n_carry_starts", "n_carry_starts_excl_defender",
    "total_carry_ticks", "total_carry_ticks_excl_defender",
    "n_carry_clean_release", "n_carry_tag_caught",
    "red_carrying_ticks", "n_red_carry_starts",
    "tagged_ticks_total", "tagged_fraction", "first_tag_tick", "mean_active_count",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_sealed_rows() -> list[dict]:
    with CAUSAL_ROWS.open(encoding="utf-8") as fh:
        return [{**r, "seed": int(r["seed"]), "defender_id": int(r["defender_id"]), "steps": int(r["steps"]),
                "blue": int(r["blue"]), "red": int(r["red"]), "win": int(r["win"]), "margin": int(r["margin"])}
               for r in csv.DictReader(fh)]


def _n_starts(grid: np.ndarray) -> int:
    """grid: (T, k) bool. Count False->True transitions, summed over the k columns."""
    if grid.shape[0] < 2:
        return 0
    return int((grid[1:] & ~grid[:-1]).sum())


def _first_tick(grid: np.ndarray) -> float:
    """First tick any column is True; censored at T (episode length) if never."""
    any_t = np.asarray(grid).any(axis=1) if grid.ndim > 1 else np.asarray(grid)
    idx = np.flatnonzero(any_t)
    return float(idx[0]) if idx.size else float(grid.shape[0])


def _carry_runs(carrying: np.ndarray, tagged: np.ndarray, alive: np.ndarray, agent_ids: list[int]) -> tuple[int, int]:
    """Simple per-agent scan (deliberately not vectorised -- an independent, easy-to-audit
    implementation, matching the plain-loop 'reference' style used elsewhere in this line).
    Returns (n_clean_release, n_tag_caught)."""
    clean = caught = 0
    T = carrying.shape[0]
    for a in agent_ids:
        was = False
        for t in range(T):
            now = bool(carrying[t, a])
            if was and not now:
                if bool(tagged[t, a]) or not bool(alive[t, a]):
                    caught += 1
                else:
                    clean += 1
            was = now
    return clean, caught


def localization_metrics(tr: dict[str, Any], defender_id: int) -> dict[str, float]:
    alive, tagged, carrying = tr["alive"], tr["tagged"], tr["carrying"]
    red_carrying = tr["red_carrying"]
    T = alive.shape[0]
    active = alive & ~tagged
    all_ids = list(range(P.N_AGENTS))
    excl = [a for a in all_ids if a != defender_id]

    clean_all, caught_all = _carry_runs(carrying, tagged, alive, all_ids)
    return {
        "time_to_first_carry": _first_tick(carrying),
        "time_to_first_carry_excl_defender": _first_tick(carrying[:, excl]),
        "n_carry_starts": _n_starts(carrying),
        "n_carry_starts_excl_defender": _n_starts(carrying[:, excl]),
        "total_carry_ticks": int(carrying.sum()),
        "total_carry_ticks_excl_defender": int(carrying[:, excl].sum()),
        "n_carry_clean_release": clean_all,
        "n_carry_tag_caught": caught_all,
        "red_carrying_ticks": int(red_carrying.sum()),
        "n_red_carry_starts": _n_starts(red_carrying),
        "tagged_ticks_total": int(tagged.sum()),
        "tagged_fraction": float(tagged.sum()) / (P.N_AGENTS * T),
        "first_tag_tick": _first_tick(tagged),
        "mean_active_count": float(active.sum(axis=1).mean()),
    }


SHARD_GLOB = "EXPLORATORY_ONE_DEFENDER_FAILURE_LOCALIZATION_SHARD*_PARTIAL.jsonl"


def _shard_path(i: int, k: int) -> Path:
    return SD / f"EXPLORATORY_ONE_DEFENDER_FAILURE_LOCALIZATION_SHARD{i}OF{k}_PARTIAL.jsonl"


def _load_partial(path: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                out[tuple(o["key"])] = o["row"]
    return out


def replay_shard(shard: int, n_shards: int) -> int:
    """Replay this shard's slice of the 768 sealed cells (cell index % n_shards == shard), appending
    each finished episode to an append-only partial file so a crash or reboot loses at most the
    in-flight episode. Each episode is parity-checked against the sealed record as it is written and
    hard-aborts on any mismatch. Episodes are independent and deterministic, so sharding cannot change
    any result."""
    sealed = _load_sealed_rows()
    mine = [r for i, r in enumerate(sealed) if i % n_shards == shard]
    path = _shard_path(shard, n_shards)
    done = _load_partial(path)
    pending = [r for r in mine if (r["policy"], r["pole"], r["arm"], r["seed"]) not in done]
    print(f"  shard {shard}/{n_shards}: {len(mine)} cells, {len(done)} already recorded, {len(pending)} to run", flush=True)
    if not pending:
        return 0
    g = P.pole_genomes()
    pols = P.load_policies(DEVICE)
    import torch
    with path.open("a", encoding="utf-8") as fh, torch.no_grad():
        for rec in tqdm_iter(pending, desc=f"localization replay shard {shard}/{n_shards} (cuda)",
                             total=len(pending), unit="ep"):
            policy, pole, arm, seed, defender_id = rec["policy"], rec["pole"], rec["arm"], rec["seed"], rec["defender_id"]
            if arm == "native":
                tr = P.run_learned_episode(pols[policy], pole, seed, g, DEVICE)
            else:
                tr = P.run_learned_episode_with_forced_defender(pols[policy], pole, seed, g, DEVICE, defender_id)
            got = (int(tr["blue"]), int(tr["red"]), int(tr["win"]), int(tr["margin"]), int(tr["steps"]))
            want = (rec["blue"], rec["red"], rec["win"], rec["margin"], rec["steps"])
            if got != want:
                raise SystemExit(f"ABORT: {policy}/{pole}/{arm}/{seed}: replayed {got} != sealed {want}")
            row = {"policy": policy, "pole": pole, "arm": arm, "seed": seed, "defender_id": defender_id,
                   "blue": got[0], "red": got[1], "win": got[2], "margin": got[3], "steps": got[4],
                   "sealed_blue": rec["blue"], "sealed_red": rec["red"], **localization_metrics(tr, defender_id)}
            fh.write(json.dumps({"key": [policy, pole, arm, seed], "row": row}) + "\n"); fh.flush()
    print(f"  shard {shard}/{n_shards} complete", flush=True)
    return 0


def analyze() -> int:
    sealed = _load_sealed_rows()
    sealed_by_key = {(r["policy"], r["pole"], r["arm"], r["seed"]): r for r in sealed}

    # Merge every shard's append-only partial file. Absence is an error state: every one of the
    # sealed cells must be present, exactly once, or nothing is reported.
    merged: dict[tuple, dict] = {}
    for path in sorted(SD.glob(SHARD_GLOB)):
        for key, row in _load_partial(path).items():
            if key in merged and merged[key] != row:
                raise SystemExit(f"ABORT: conflicting replays of {key} across shards")
            merged[key] = row
    missing = sorted(set(sealed_by_key) - set(merged))
    extra = sorted(set(merged) - set(sealed_by_key))
    if missing or extra:
        raise SystemExit(f"ABORT: replay incomplete or foreign -- {len(missing)} sealed cell(s) missing "
                         f"(e.g. {missing[:2]}), {len(extra)} unexpected (e.g. {extra[:2]}). Run/resume the shards first.")

    # Parity, re-verified at merge time (each row was already checked when it was written).
    mismatches = []
    for key, rec in sealed_by_key.items():
        row = merged[key]
        got = tuple(int(row[f]) for f in ("blue", "red", "win", "margin", "steps"))
        want = tuple(int(rec[f]) for f in ("blue", "red", "win", "margin", "steps"))
        if got != want:
            mismatches.append(f"{key}: replayed {got} != sealed {want}")
    if mismatches:
        raise SystemExit(f"ABORT: {len(mismatches)} replayed episode(s) disagree with the sealed causal record, "
                         f"e.g. {mismatches[0]}")
    rows = [merged[(r["policy"], r["pole"], r["arm"], r["seed"])] for r in sealed]
    print(f"  parity PASS: all {len(rows)}/{len(sealed)} replayed episodes match the sealed causal record exactly",
         flush=True)

    fields = list(rows[0])
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

    def paired(policy: str, pole: str, metric: str) -> np.ndarray:
        by_seed_native = {r["seed"]: r[metric] for r in rows if r["policy"] == policy and r["pole"] == pole and r["arm"] == "native"}
        by_seed_plus = {r["seed"]: r[metric] for r in rows if r["policy"] == policy and r["pole"] == pole and r["arm"] == "plus_one_defender"}
        seeds = sorted(set(by_seed_native) & set(by_seed_plus))
        return np.asarray([by_seed_plus[s] - by_seed_native[s] for s in seeds], dtype=np.float64)

    comparison: dict[str, dict[str, dict[str, dict]]] = {}
    for pole in P.POLES:
        comparison[pole] = {}
        for policy in ("pi_A", "pi_B"):
            comparison[pole][policy] = {m: _bootstrap(paired(policy, pole, m)) for m in METRIC_FIELDS}

    payload = {
        "record_id": "EXPLORATORY_ONE_DEFENDER_FAILURE_LOCALIZATION_RESULT", "implements": SPEC_PATH.name,
        "utc": _now(), "classification": "DIAGNOSTIC. EXPLORATORY. NON-GATING. Not sealed via the formal audit "
                                         "pipeline used for confirmatory records in this line; the integrity "
                                         "guarantee here is the parity check below, not an AuditPlan.",
        "n_episodes_replayed": len(rows), "n_mismatches": 0,
        "primary_target": "Pole B", "non_gating_contrast": "Pole A",
        "COMPARISON_PLUS1D_MINUS_NATIVE_PAIRED_BY_SEED": comparison,
        "known_gaps": json.loads(SPEC_PATH.read_text(encoding="utf-8"))["KNOWN_GAPS_DISCLOSED_UP_FRONT"],
        "claim_boundary": "Descriptive only. No verdict, no significance-based selection, no automated mechanism "
                          "label. Interpretation is left to the reader applying the PI's own modest mapping "
                          "outside this record. Does not reopen or re-score DEFENDER_INJECTION_CAUSAL_BRIDGE_RESULT.json.",
    }
    OUT_RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  -> {OUT_RESULT}")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("replay", "analyze"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1, dest="n_shards")
    a = ap.parse_args()
    if a.stage == "replay":
        if not 0 <= a.shard < a.n_shards:
            raise SystemExit(f"--shard must be in [0, {a.n_shards})")
        return replay_shard(a.shard, a.n_shards)
    return analyze()


if __name__ == "__main__":
    raise SystemExit(main())
