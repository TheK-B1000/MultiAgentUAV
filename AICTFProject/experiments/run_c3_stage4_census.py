"""C3 Stage-4 leg 1: fresh natural census on seed block 9810000+.

Frozen cells: artifacts/c3_discovery/C3_STAGE4_CONFIRMATION_FROZEN.json (bebb626)

Collects carrier-pressure anchors on a FRESH seed block using the same frozen
collection function as discovery (``collect_pressure_anchors``), so the natural
leg measures the same phenomenon under the same predicate. It writes to its own
directory: the discovery census under artifacts/c3_discovery/ is the sampling
frame the Stage-3 result rests on and must not be overwritten.

WHY A SEPARATE SCRIPT
---------------------
The discovery runner hardcodes OUT_DIR and DISCOVERY_SEED_BASE as module
constants, and its Stage-1 path calls write_stage1_artifacts(OUT_DIR, ...).
Running it against a fresh block would clobber the discovery census. Reusing the
collection function from a thin wrapper keeps the measurement identical while
making that class of accident impossible.

FAIL-CLOSED
-----------
Refuses a seed base that is not the frozen 9810000, refuses the spent 9800001
block explicitly, and refuses to overwrite an existing fresh census.

Run:  python experiments/run_c3_stage4_census.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.run_c3_decision_proximal_discovery import (  # noqa: E402
    G0_SEEDS,
    _load_runtime_contract,
    collect_pressure_anchors,
)
from experiments.run_g0_v2_seed import OPPONENTS  # noqa: E402
from rl.analysis.c3_discovery_artifacts import write_stage1_artifacts  # noqa: E402

DISCOVERY_DIR = PROJECT_ROOT / "artifacts" / "c3_discovery"
STAGE4_FROZEN = DISCOVERY_DIR / "C3_STAGE4_CONFIRMATION_FROZEN.json"
OUT_DIR = PROJECT_ROOT / "artifacts" / "c3_stage4"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if not STAGE4_FROZEN.exists():
        raise SystemExit(f"REFUSED: Stage-4 freeze missing at {STAGE4_FROZEN}")
    frozen = json.loads(STAGE4_FROZEN.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise SystemExit("REFUSED: Stage-4 cells are not FROZEN")

    seeds_cell = frozen["seeds"]
    frozen_base = int(seeds_cell["base"])
    forbidden = {int(b) for b in seeds_cell["forbidden_spent_blocks"]}
    episodes = int(seeds_cell["episodes_per_cell"])

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-seed-base", type=int, default=frozen_base)
    ap.add_argument("--episodes", type=int, default=episodes)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(frozen["policies"]))
    args = ap.parse_args()

    base = int(args.eval_seed_base)
    if base in forbidden:
        raise SystemExit(
            f"REFUSED: seed block {base} is recorded as SPENT in the frozen cells "
            f"(forbidden: {sorted(forbidden)})"
        )
    if base != frozen_base:
        raise SystemExit(
            f"REFUSED: seed base {base} is not the frozen Stage-4 base {frozen_base}"
        )
    if int(args.episodes) != episodes:
        raise SystemExit(
            f"REFUSED: episodes_per_cell {args.episodes} != frozen {episodes}"
        )

    anchors_path = OUT_DIR / "C3_STAGE1_ANCHORS.jsonl"
    if anchors_path.exists() and anchors_path.stat().st_size > 0:
        raise SystemExit(
            f"REFUSED: a fresh census already exists at {anchors_path}. Refusing "
            "to overwrite Stage-4 leg-1 data."
        )

    contract = _load_runtime_contract()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from experiments.long_session_progress import LongSessionProgress, configure_stdio
    from experiments.run_g0_v2_evaluation import resolve_cnn_channels
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy

    configure_stdio()
    progress = LongSessionProgress(OUT_DIR, name="C3_STAGE4_CENSUS")
    started = time.time()
    print("=" * 78)
    print("C3 STAGE-4 LEG 1 — fresh natural census")
    print(f"frozen cells : {STAGE4_FROZEN.relative_to(PROJECT_ROOT)}")
    print(f"seed block   : {base}..{base + args.episodes - 1}  (fresh)")
    print(f"policies     : {args.seeds}   opponents: {OPPONENTS}")
    print(f"floor        : LCB95(anchors/episode) > {frozen['leg_1_fresh_natural']['floor']}")
    print("writes to    : artifacts/c3_stage4/ (discovery census untouched)")
    print("=" * 78)
    sys.stdout.flush()

    all_anchors: list[dict] = []
    per_policy: dict[int, dict] = {}
    checkpoints: dict[str, dict] = {}

    for seed in args.seeds:
        tag = f"g0_v5_long_seed{seed}"
        ckpt = PROJECT_ROOT / "artifacts" / "g0_v5_long" / tag / "ckpts" / f"final_{tag}.zip"
        payload = read_checkpoint_payload(str(ckpt), map_location="cpu")
        if int(payload.get("global_step", 0)) < 1_000_000:
            raise SystemExit(f"{ckpt}: not the preregistered 1M checkpoint")
        checkpoints[str(seed)] = {
            "checkpoint_path": str(ckpt), "checkpoint_sha256": _sha256(ckpt),
            "global_step": int(payload.get("global_step", 0)),
        }
        policy = load_policy(
            str(ckpt), device=args.device,
            num_cnn_channels=resolve_cnn_channels(payload, context=str(ckpt)),
        )
        progress.set_phase("STAGE4_CENSUS", f"policy_{seed}")
        counts: dict[str, int] = {}
        for opp in OPPONENTS:
            n_cell = 0
            for i in range(args.episodes):
                ev = base + i
                anchors, _feats = collect_pressure_anchors(
                    policy, opponent=opp, seed=ev, device=args.device,
                    response_horizon=contract.h_response,
                )
                for a in anchors:
                    a["train_seed"] = int(seed)
                    a["opponent"] = opp
                    a["eval_seed"] = int(ev)
                    all_anchors.append(a)
                n_cell += len(anchors)
            counts[opp] = n_cell
            progress.log(f"policy={seed} vs {opp}: anchors={n_cell}")
        total = sum(counts.values())
        n_eps = len(OPPONENTS) * args.episodes
        per_policy[str(seed)] = {
            "anchors": total, "episodes": n_eps,
            "anchors_per_episode": round(total / n_eps, 4),
            "by_opponent": counts,
        }
        print(f"  policy {seed}: {total} anchors over {n_eps} episodes "
              f"= {total / n_eps:.3f}/episode", flush=True)

    manifest = {
        "status": "STAGE1_FROZEN",
        "leg": "STAGE4_LEG1_FRESH_NATURAL",
        "science_scope": "CONTROLLABILITY_SCREEN_ONLY",
        "c3_contract_hash": _sha256(DISCOVERY_DIR / "C3_DISCOVERY_PREREG_FROZEN.json"),
        "stage4_frozen_sha256": _sha256(STAGE4_FROZEN),
        "map": frozen["map"], "ruleset": frozen["ruleset"],
        "runtime_contract": {
            "t_trace": contract.t_trace, "h_response": contract.h_response,
            "delta": contract.delta, "utility_name": contract.utility_name,
            "doomed_utility_threshold": contract.doomed_utility_threshold,
            "minimum_fork_rate": contract.minimum_fork_rate,
        },
        "seeds": list(args.seeds), "opponents": list(OPPONENTS),
        "episodes_per_cell": args.episodes,
        "discovery_seed_base": base,
        "n_anchors": len(all_anchors),
        "anchors_by_seed": {k: v["anchors"] for k, v in per_policy.items()},
        "per_policy": per_policy,
        "checkpoints": checkpoints,
        "wall_seconds": round(time.time() - started, 1),
    }
    write_stage1_artifacts(OUT_DIR, anchors=all_anchors, manifest=manifest)

    print("\n" + "=" * 78)
    print(f"fresh census: {len(all_anchors)} anchors")
    for k, v in per_policy.items():
        print(f"  policy {k}: {v['anchors_per_episode']}/episode")
    print(f"\nwrote {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print("Leg-1 verdict requires the frozen analyzer, not these raw rates.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
