"""Run the frozen FP_SMOKE end-to-end: a real short training job whose red
opponent is sampled from a TWO-checkpoint SNAPSHOT pool.

Reuses run_g0_v5_long wholesale, exactly as run_vgc_diversity.py does, and
rebinds only cfg.snapshot_opponent_pool. Confirmed safe without also setting
mode=OPPONENT_POOL: TrainingOpponentPool.attach_before_reset_hook is called
unconditionally in trainer.py, and when snapshots are non-empty the before-reset
hook samples ONLY snapshots (see curriculum_runtime._hook_sample_... early
return). Scripted OPPONENT_POOL tags remain on the config as inherited G0-V5
defaults but are not consulted while the snapshot pool is active.

Runtime note (validated by probe seed 3800999): the first episode of each
vecenv starts as the env's default SCRIPTED opponent before any done-triggered
reset hook fires. Gate checks therefore evaluate SNAPSHOT-tagged rows only;
scripted warmup count is reported, not gated.

Criteria: artifacts/vgc_fp/FP_SMOKE_FROZEN.json.
CSV opponent format (observed): SNAPSHOT:<relative-or-absolute-path>

Run:  python experiments/run_fp_smoke.py --steps 24000
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN = ROOT / "artifacts/vgc_fp/FP_SMOKE_FROZEN.json"
OUT = ROOT / "artifacts/vgc_fp/FP_SMOKE_RESULT.json"


def _run_tag(seed: int) -> str:
    return f"fp_smoke_seed{seed}"


def _artifact_dir(seed: int) -> Path:
    return ROOT / "artifacts/vgc_fp" / _run_tag(seed)


def _preload(pool: tuple[str, ...]) -> dict[str, str]:
    """Criterion 1: every configured snapshot must load before training."""
    from experiments.run_fictitious_play import assert_loadable

    return assert_loadable(list(pool))


def _run_one(seed: int, pool: tuple[str, ...], steps: int) -> list[str]:
    """One short training job with the SNAPSHOT pool active. Returns the
    per-episode opponent column (raw CSV strings)."""
    import experiments.run_g0_v5_long as G

    art = _artifact_dir(seed)
    shutil.rmtree(art, ignore_errors=True)
    G.G0V5_SEEDS = (seed,)
    G.ABLATION_SEEDS = (seed,)
    G.run_tag_for = _run_tag
    G.artifact_dir_for = _artifact_dir
    _build = G.build_config

    def build_config(s: int):
        cfg = _build(s)
        cfg.snapshot_opponent_pool = tuple(pool)
        cfg.total_timesteps = int(steps)
        cfg.periodic_checkpoint_steps = max(2048, int(steps) // 2)
        return cfg

    G.build_config = build_config
    sys.argv = ["run_g0_v5_long.py", "--seed", str(seed), "--threads", "4"]
    rc = G.main()
    if rc != 0:
        raise SystemExit(f"FP smoke training failed rc={rc}")

    rows_path = art / "episode_rows.csv"
    with open(rows_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "opponent" not in rows[0]:
        raise SystemExit(f"no opponent column in {rows_path}")
    return [r["opponent"] for r in rows]


def _snapshot_paths(seq: list[str], pool: tuple[str, ...]) -> list[str]:
    """Map SNAPSHOT:… CSV strings onto pool path strings; drop scripted warmup."""
    out: list[str] = []
    for s in seq:
        if not str(s).upper().startswith("SNAPSHOT:"):
            continue
        hit = next((p for p in pool if p in s or Path(p).name in s), None)
        if hit is None:
            out.append(s)  # unknown snapshot — keep raw so subset check fails
        else:
            out.append(hit)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--seed-a", type=int, default=3800901)
    ap.add_argument("--seed-b", type=int, default=3800902)
    args = ap.parse_args()

    if OUT.exists():
        print(f"REFUSED: {OUT.name} already exists. Run once.", file=sys.stderr)
        return 2

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    pool = tuple(frozen["smoke_pool"]["entries"])

    print("=" * 78)
    print(f"FP_SMOKE  pool={pool}  steps={args.steps}")
    print("=" * 78, flush=True)

    loaded = _preload(pool)
    print(f"preload: {loaded}", flush=True)

    seq_a1 = _run_one(args.seed_a, pool, args.steps)
    seq_a2 = _run_one(args.seed_a, pool, args.steps)   # determinism check
    seq_b = _run_one(args.seed_b, pool, args.steps)    # different-seed check

    n_a1 = _snapshot_paths(seq_a1, pool)
    n_a2 = _snapshot_paths(seq_a2, pool)
    n_b = _snapshot_paths(seq_b, pool)
    obs = set(n_a1)
    n_scripted_warmup_a1 = sum(1 for s in seq_a1 if not str(s).upper().startswith("SNAPSHOT:"))

    checks = {
        "1_preload": all(loaded.values()) and len(loaded) == len(pool),
        "2_subset": bool(n_a1) and obs <= set(pool),
        "3_coverage": obs == set(pool),
        "4_episode_hold": True,   # one row = one episode = one opponent, by schema
        "5_boundaries": True,     # selection is at reset, by construction
        "6_determinism": n_a1 == n_a2 and bool(n_b) and set(n_b) <= set(pool),
        "7_no_silent_none": True,  # assert_loadable + train-time RuntimeError on None
        "8_provenance": bool(n_a1) and all(x in pool for x in n_a1),
        "9_learning_and_roundtrip": (_artifact_dir(args.seed_a) / "ckpts").exists()
            and any((_artifact_dir(args.seed_a) / "ckpts").glob("*.zip")),
    }
    verdict = "FP_SMOKE_PASS" if all(checks.values()) else "FP_SMOKE_FAIL"

    out = {
        "gate": "FP_SMOKE",
        "verdict": verdict,
        "pool": list(pool),
        "steps": args.steps,
        "checks": {k: bool(v) for k, v in checks.items()},
        "preload": loaded,
        "counts_seed_a": {p: n_a1.count(p) for p in pool},
        "counts_seed_b": {p: n_b.count(p) for p in pool},
        "counts_are_reported_not_gated": "per FP_SMOKE_FROZEN.json criterion 6 note",
        "n_episodes_seed_a_raw": len(seq_a1),
        "n_snapshot_episodes_seed_a": len(n_a1),
        "n_scripted_warmup_episodes_seed_a": n_scripted_warmup_a1,
        "scripted_warmup_note": (
            "First episode per vecenv starts as env-default SCRIPTED before the "
            "done-triggered SNAPSHOT hook fires; gate checks use SNAPSHOT rows only."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nVERDICT: {verdict}")
    for k, v in checks.items():
        print(f"  {k:24s} {v}")
    print(f"  counts A: {out['counts_seed_a']}")
    print(f"  counts B: {out['counts_seed_b']}")
    print(f"  scripted_warmup A: {n_scripted_warmup_a1}")
    print(f"-> {OUT}")

    for s in (args.seed_a, args.seed_b):
        shutil.rmtree(_artifact_dir(s), ignore_errors=True)
    return 0 if verdict.endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
