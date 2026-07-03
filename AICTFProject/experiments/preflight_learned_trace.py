"""Cheap learned-only preflight for the cross-episode shuffle gate.

Runs ONLY the learned_qphi_switching condition over the opponent x map grid
with a few seeds per cell, then asks the single question that decides whether
the full behavioral exam is worth running:

    Does the deterministic learned router produce enough DISTINCT per-episode
    z-signatures for the cross-episode histogram-preserving shuffle to alter
    anything (can_reassign=True), or has it collapsed to one z everywhere
    (cross_episode_gate_untestable=True)?

If untestable, the fixed_z2 / uniform / shuffled conditions add nothing: they
cannot prove contextual routing when the learned policy has no contextual
variation to shuffle. Only resume the full exam when can_reassign=True with at
least two distinct episode signatures.

Usage::

    uv run python experiments/preflight_learned_trace.py \\
      --checkpoint artifacts/ab_router_specialize/treatment/final_treatment.zip \\
      --opponents OP8 OP9 OP10 --maps map_b map_b_split_lane_v2 \\
      --episodes 8 --base-seed 18000 --device cuda \\
      --out artifacts/ab_router_specialize/treatment/preflight_s18000.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.eval_v6i9_router_diagnostic_ablation import (  # noqa: E402
    DiagnosticProtocol,
    _run_condition,
)
from rl.evaluation.router_ablation import (  # noqa: E402
    build_cross_episode_shuffled_mapping_from_learned_traces,
    default_conditions,
)


def _episode_key(t: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(t["opponent"]).upper(),
        int(t["seed"]),
        int(t["episode_index"]),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--anchor-checkpoint",
        default="checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip",
    )
    p.add_argument("--opponents", nargs="+", default=["OP8", "OP9", "OP10"])
    p.add_argument("--maps", nargs="+", default=["map_b", "map_b_split_lane_v2"])
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=18000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from rl.custom_ppo import load_custom_ppo_policy, read_custom_ppo_metadata

    meta = read_custom_ppo_metadata(args.checkpoint)
    cfg_meta = meta.get("cfg") if isinstance(meta.get("cfg"), dict) else {}
    latent_k = int(meta.get("latent_k", 4))
    allowed_latents = cfg_meta.get("router_allowed_latents") if isinstance(cfg_meta, dict) else None
    if not allowed_latents:
        allowed_latents = list(range(latent_k))

    protocol = DiagnosticProtocol(
        checkpoint=str(args.checkpoint),
        anchor_checkpoint=str(args.anchor_checkpoint),
        opponents=tuple(str(o).upper() for o in args.opponents),
        maps=tuple(args.maps),
        episodes_per_cell=int(args.episodes),
        base_seed=int(args.base_seed),
        device=str(args.device),
    )

    # Resolve switch cadence from the checkpoint (same logic as the evaluator).
    probe_env_map = protocol.maps[0]
    from experiments.eval_v6i9_router_diagnostic_ablation import _make_env

    probe_env = _make_env(protocol, probe_env_map, protocol.base_seed)
    model_probe = load_custom_ppo_policy(
        protocol.checkpoint, probe_env.observation_space, probe_env.action_space, device=protocol.device
    )
    switch_cadence = int(getattr(model_probe, "strategy_interval", 0) or 0) or int(
        cfg_meta.get("strategy_interval", 32) or 32
    )
    probe_env.close()

    conditions = {c.name: c for c in default_conditions(latent_k, allowed_latents, switch_cadence)}
    learned = conditions["learned_qphi_switching"]

    print("=" * 72)
    print("[preflight] learned-only trace preflight")
    print(f"[preflight] checkpoint = {protocol.checkpoint}")
    print(f"[preflight] grid       = {len(protocol.opponents)} opp x {len(protocol.maps)} maps")
    print(f"[preflight] episodes   = {protocol.episodes_per_cell}/cell")
    print(f"[preflight] cadence    = {switch_cadence}")
    print("=" * 72)

    _rows, traces, _integrity = _run_condition(
        protocol, learned, switch_cadence=switch_cadence
    )

    # --- Per-(opponent, map) unique z-signatures ---
    seqs: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    ep_map: dict[tuple[str, int, int], str] = {}
    for t in traces:
        k = _episode_key(t)
        seqs.setdefault(k, []).append((int(t["opportunity_index"]), int(t["selected_z"])))
        ep_map.setdefault(k, str(t.get("map", "") or ""))
    episode_sig: dict[tuple[str, int, int], tuple[int, ...]] = {
        k: tuple(z for _i, z in sorted(v)) for k, v in seqs.items()
    }

    cells: dict[tuple[str, str], list[tuple[int, ...]]] = {}
    for k, sig in episode_sig.items():
        cells.setdefault((k[0], ep_map.get(k, "")), []).append(sig)

    per_cell = []
    for (opp, mp), sigs in sorted(cells.items()):
        distinct = {s for s in sigs}
        per_cell.append(
            {
                "opponent": opp,
                "map": mp,
                "episodes": len(sigs),
                "distinct_signatures": len(distinct),
                "example_signatures": [list(s) for s in list(distinct)[:3]],
            }
        )

    argmax_hist = Counter(int(t["selected_z"]) for t in traces)

    # --- Cross-episode testability (the corrected gate) ---
    try:
        _mapping, cross_meta = build_cross_episode_shuffled_mapping_from_learned_traces(
            traces,
            latent_k=latent_k,
            allowed_latents=list(allowed_latents),
            switch_cadence=switch_cadence,
            require_min_contexts=False,
        )
        can_reassign = bool(cross_meta.get("can_reassign", False))
        cross_summary = {
            "can_reassign": can_reassign,
            "cell_count": cross_meta.get("cell_count"),
            "reassignable_episode_count": cross_meta.get("reassignable_episode_count"),
            "reassigned_episode_count": cross_meta.get("reassigned_episode_count"),
            "non_constant_episode_count": cross_meta.get("non_constant_episode_count"),
            "episode_histogram_preserved": cross_meta.get("episode_histogram_preserved"),
        }
    except Exception as exc:  # noqa: BLE001
        can_reassign = False
        cross_summary = {"error": str(exc)}

    total_distinct = len({s for s in episode_sig.values()})
    gate_untestable = (not can_reassign) or total_distinct < 2

    report = {
        "checkpoint": protocol.checkpoint,
        "base_seed": protocol.base_seed,
        "episodes_per_cell": protocol.episodes_per_cell,
        "n_episodes": len(episode_sig),
        "n_decisions": len(traces),
        "argmax_z_histogram": {int(k): int(v) for k, v in sorted(argmax_hist.items())},
        "global_distinct_episode_signatures": total_distinct,
        "per_cell": per_cell,
        "cross_episode": cross_summary,
        "cross_episode_gate_untestable": bool(gate_untestable),
        "recommendation": (
            "STOP: cross-episode shuffle is untestable (learned router has no "
            "contextual variation to shuffle). The full behavioral exam cannot "
            "prove contextual routing."
            if gate_untestable
            else "PROCEED: shuffle is testable (>=2 distinct episode signatures and "
            "can_reassign=True). Running the full behavioral exam is justified."
        ),
    }

    print("\n" + "=" * 72)
    print("[preflight] RESULT")
    print(f"  episodes                 : {report['n_episodes']}")
    print(f"  decisions                : {report['n_decisions']}")
    print(f"  argmax z histogram       : {report['argmax_z_histogram']}")
    print(f"  distinct episode sigs    : {report['global_distinct_episode_signatures']}")
    print(f"  can_reassign             : {cross_summary.get('can_reassign')}")
    print(f"  reassignable_episodes    : {cross_summary.get('reassignable_episode_count')}")
    print(f"  cross_episode_untestable : {report['cross_episode_gate_untestable']}")
    for c in per_cell:
        print(
            f"    {c['opponent']:>5} {c['map']:<20} "
            f"episodes={c['episodes']} distinct_sigs={c['distinct_signatures']}"
        )
    print(f"\n  >>> {report['recommendation']}")
    print("=" * 72)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[preflight] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
