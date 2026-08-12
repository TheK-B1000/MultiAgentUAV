"""Unified cross-play evaluator: any policy vs the full OP6-OP12 board.

Extends the existing evaluation stack rather than replacing it. Per-episode
rollout and outcome derivation are imported from run_g0_v2_evaluation
(`run_eval_episode`, `_wr`), so this evaluator cannot disagree with the
historical one about what a win is.

What it adds is the record schema the diversity experiment needs -- policy_id,
method, diversity_condition, seen vs held-out -- and the ability to point at an
arbitrary checkpoint instead of only a G0-V5 training seed.

Handles both team sizes: run_g0_v2_evaluation.AGENTS is rebound per policy,
because patching only one side once produced a 4-agent policy evaluated in a
2v2 env (C7 Stage 0).

Run:
  python experiments/run_crossplay_eval.py --registry artifacts/vgc_diversity/policies.json
  python experiments/run_crossplay_eval.py --auto-d7        # the 6 existing baselines
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN = ROOT / "artifacts/vgc_diversity/VGC_DIVERSITY_SETS_FROZEN.json"
OUT_DIR = ROOT / "artifacts/vgc_diversity/crossplay"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=30).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def auto_d7_registry() -> list[dict]:
    """The six already-trained Mixed-PPO D7 baselines found in the Phase 1 audit."""
    out = []
    for seed in (3200001, 3200002, 3200003):
        tag = f"g0_v5_long_seed{seed}"
        out.append({"policy_id": tag, "checkpoint": f"artifacts/g0_v5_long/{tag}/ckpts/final_{tag}.zip",
                    "method": "Mixed-PPO", "diversity_condition": "D7", "team_size": 2, "seed": seed})
    for seed in (3300001, 3300002, 3300003):
        tag = f"c7_4v4_seed{seed}"
        out.append({"policy_id": tag, "checkpoint": f"artifacts/c7_stage0/{tag}/ckpts/final_{tag}.zip",
                    "method": "Mixed-PPO", "diversity_condition": "D7", "team_size": 4, "seed": seed})
    return out


def evaluate_policy(entry: dict, episodes: int, device: str) -> list[dict]:
    """Evaluate one policy against the full board. Returns per-episode rows."""
    import experiments.run_g0_v2_evaluation as E
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy

    team = int(entry.get("team_size", 2))
    E.AGENTS = team          # BOTH sides of the agent-count binding; see module docstring

    ck = ROOT / entry["checkpoint"]
    if not ck.is_file():
        raise FileNotFoundError(f"{entry['policy_id']}: missing checkpoint {ck}")
    payload = read_checkpoint_payload(str(ck), map_location="cpu")
    policy = load_policy(str(ck), device=device,
                         num_cnn_channels=E.resolve_cnn_channels(payload, context=str(ck)))

    fz = json.loads(FROZEN.read_text(encoding="utf-8"))
    cond = entry.get("diversity_condition")
    trained_on = set(fz["THE_SETS"].get(cond, [])) if cond in fz["THE_SETS"] else set()

    rows = []
    for opp in E.OPPONENTS:
        for i in range(episodes):
            r = E.run_eval_episode(policy, opponent=opp,
                                   seed=E.EVAL_SEED_BASE + i, device=device)
            r.update({"policy_id": entry["policy_id"], "method": entry["method"],
                      "diversity_condition": cond, "team_size": team,
                      "train_seed": entry.get("seed"),
                      "seen_in_training": opp in trained_on})
            rows.append(r)
        wr = E._wr([r for r in rows if r["opponent"] == opp])
        print(f"  {entry['policy_id']:26s} vs {opp:5s} win_rate={wr:.3f} "
              f"({'seen' if opp in trained_on else 'held-out'})", flush=True)
    return rows


def summarize(rows: list[dict], entry: dict) -> dict:
    import experiments.run_g0_v2_evaluation as E
    per_opp = {}
    for opp in sorted({r["opponent"] for r in rows}):
        sub = [r for r in rows if r["opponent"] == opp]
        per_opp[opp] = {"win_rate": round(E._wr(sub), 4), "n": len(sub),
                        "seen": bool(sub[0]["seen_in_training"])}
    seen = [v["win_rate"] for v in per_opp.values() if v["seen"]]
    held = [v["win_rate"] for v in per_opp.values() if not v["seen"]]
    allw = [v["win_rate"] for v in per_opp.values()]
    mean = lambda x: round(sum(x) / len(x), 4) if x else None
    var = (round(sum((w - mean(allw)) ** 2 for w in allw) / len(allw), 5) if allw else None)
    return {
        "policy_id": entry["policy_id"], "method": entry["method"],
        "diversity_condition": entry.get("diversity_condition"),
        "team_size": int(entry.get("team_size", 2)), "train_seed": entry.get("seed"),
        "checkpoint": entry["checkpoint"],
        "checkpoint_sha256": hashlib.sha256((ROOT / entry["checkpoint"]).read_bytes()).hexdigest(),
        "per_opponent": per_opp,
        "seen_avg": mean(seen), "held_out_avg": mean(held), "overall_avg": mean(allw),
        # undefined for D7 by construction -- reported as null, never as zero
        "generalization_gap": (round(mean(seen) - mean(held), 4)
                               if seen and held else None),
        "worst_opponent": min(per_opp, key=lambda k: per_opp[k]["win_rate"]) if per_opp else None,
        "worst_opponent_wr": min(allw) if allw else None,
        "best_opponent_wr": max(allw) if allw else None,
        "variance_across_opponents": var,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="")
    ap.add_argument("--auto-d7", action="store_true")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="crossplay")
    args = ap.parse_args()

    if args.auto_d7:
        registry = auto_d7_registry()
    elif args.registry:
        registry = json.loads((ROOT / args.registry).read_text(encoding="utf-8"))
    else:
        raise SystemExit("give --registry or --auto-d7")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("=" * 78)
    print(f"CROSS-PLAY EVALUATION  policies={len(registry)}  episodes/cell={args.episodes}")
    print("=" * 78, flush=True)

    all_rows, summaries = [], []
    for entry in registry:
        rows = evaluate_policy(entry, args.episodes, args.device)
        all_rows.extend(rows)
        summaries.append(summarize(rows, entry))

    out = {
        "record": "unified cross-play evaluation vs the full OP6-OP12 board",
        "git_commit": _git_commit(),
        "episodes_per_cell": args.episodes,
        "wall_seconds": round(time.time() - started, 1),
        "summaries": summaries,
    }
    (OUT_DIR / f"{args.tag}_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # human-readable matrix
    import experiments.run_g0_v2_evaluation as E
    lines = ["| policy | method | D | seen avg | held-out avg | overall | worst | var |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        f = lambda v: "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {s['policy_id']} | {s['method']} | {s['diversity_condition']} | "
                     f"{f(s['seen_avg'])} | {f(s['held_out_avg'])} | {f(s['overall_avg'])} | "
                     f"{s['worst_opponent']} {f(s['worst_opponent_wr'])} | "
                     f"{f(s['variance_across_opponents'])} |")
    lines += ["", "| policy | " + " | ".join(E.OPPONENTS) + " |",
              "|---" * (len(E.OPPONENTS) + 1) + "|"]
    for s in summaries:
        lines.append(f"| {s['policy_id']} | " + " | ".join(
            f"{s['per_opponent'][o]['win_rate']:.3f}" for o in E.OPPONENTS) + " |")
    (OUT_DIR / f"{args.tag}_matrix.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:2 + len(summaries)]))
    print(f"\n-> {OUT_DIR / (args.tag + '_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
