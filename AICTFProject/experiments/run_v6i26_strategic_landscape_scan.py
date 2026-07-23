#!/usr/bin/env python3
"""V6I26 Stage-0 strategic landscape scan (no new long training).

Evaluates archived policies across OP×map cells and reports whether the
environment already supports niches (G_available point estimate, preference
reversals). This is the mandatory cheap gate before LRO birth rounds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v6i26_lro_core import (  # noqa: E402
    default_landscape_policies,
    lro_manifest,
    payoff_tensor_summary,
    select_response_target,
    write_json,
)
from experiments.v6i24_population_config import DEFAULT_MAPS  # noqa: E402
from gpu_env._core._bt_profiles import LRO_AUDITED_OPPONENT_POOL  # noqa: E402

DEFAULT_OPPONENTS = tuple(LRO_AUDITED_OPPONENT_POOL)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I26 strategic landscape scan")
    p.add_argument("--output-dir", default="artifacts/v6i26_landscape_scan_seed1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--episodes-per-cell", type=int, default=4)
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    p.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    p.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Optional extra policy zip (repeatable). Format: id=label=path",
    )
    p.add_argument("--max-decision-steps", type=int, default=240)
    return p.parse_args()


def _load_extra(specs: list[str]):
    from experiments.v6i26_lro_core import LandscapePolicySpec

    out = []
    for raw in specs:
        parts = str(raw).split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"--checkpoint expects id=label=path, got {raw!r}")
        pid, label, path = parts
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        out.append(LandscapePolicySpec(pid, label, path, "checkpoint"))
    return out


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = default_landscape_policies(PROJECT_ROOT)
    policies.extend(_load_extra(list(args.checkpoint or [])))
    if len(policies) < 2:
        print("ERROR: need >=2 existing policy checkpoints for a landscape scan.")
        print("Expected V6I24 probe_05u members and/or V6I23 donor under artifacts/.")
        return 2

    from experiments.run_v6i24_population_eval_gates import (
        _load_policies,
        _make_env,
        collect_payoff_and_features,
        evaluate_cross_fitted_teacher_oracle,
    )

    members = [
        (i, spec.label, Path(spec.path)) for i, spec in enumerate(policies)
    ]
    print("=" * 72)
    print("V6I26 Stage-0: strategic landscape scan")
    print("=" * 72)
    for mid, label, path in members:
        print(f"  [{mid}] {label}: {path}")
    print(f"Cells: {args.opponents} × {args.maps} @ {args.episodes_per_cell} eps")
    print()

    env0 = _make_env(members[0][2], args.maps[0], int(args.seed), args.device, int(args.max_decision_steps))
    try:
        loaded = _load_policies(members, env0.observation_space, env0.action_space, args.device)
    finally:
        env0.close()

    collected = collect_payoff_and_features(
        loaded,
        opponents=list(args.opponents),
        maps=list(args.maps),
        episodes_per_cell=int(args.episodes_per_cell),
        base_seed=int(args.seed),
        device=str(args.device),
        max_decision_steps=int(args.max_decision_steps),
    )
    payoff = np.asarray(collected["payoff_matrix"], dtype=np.float64)
    summary = payoff_tensor_summary(
        payoff,
        policy_labels=list(collected["member_labels"]),
        contexts=list(collected["contexts"]),
    )
    target = select_response_target(
        payoff,
        contexts=list(collected["contexts"]),
        policy_labels=list(collected["member_labels"]),
        episodes_per_cell=int(args.episodes_per_cell),
        prior_strength=float(args.episodes_per_cell),
        max_mixture_weight=0.35,
        aggregate_by_opponent=True,
    )
    try:
        strategic = evaluate_cross_fitted_teacher_oracle(
            collected["returns_kce"],
            context_labels=list(collected["contexts"]),
            member_labels=list(collected["member_labels"]),
            seed=int(args.seed),
        )
    except Exception as exc:
        strategic = {"error": str(exc)}

    # Stage-0 answers only: do archives already complement?
    # G_available > 0  → compress existing repertoire into LRO slots
    # G_available = 0  → manufacture via Stage-1 (do NOT keep fishing archives)
    # parallel rows    → task distribution may lack niches; still allow Stage-1
    #                   once against regret mixture, but flag FAIL_TASK_DISTRIBUTION
    cross_delta = None
    if isinstance(strategic, dict) and "delta" in strategic:
        try:
            cross_delta = float(strategic["delta"])
        except (TypeError, ValueError):
            cross_delta = None
    g_point = float(summary["G_available_point"])
    g_effective = (
        float(cross_delta) if cross_delta is not None else g_point
    )

    if summary["niche_signal"] and g_effective > 0.0:
        decision = "PROMOTE_LRO_BIRTH"
        note = (
            "Archived policies already complement (G_available > 0). "
            "Stage-1 compresses/refines into latent branches."
        )
    elif summary["parallel_rows"] and g_effective <= 0.0:
        decision = "MANUFACTURE_VIA_LRO_STAGE1"
        note = (
            "Payoff rows nearly parallel and G_available = 0 — archives have no "
            "repertoire. Stage-1 must manufacture responses; if G still flat after "
            "a BR round, escalate to geometry/task-distribution search."
        )
    elif g_effective <= 0.0:
        decision = "MANUFACTURE_VIA_LRO_STAGE1"
        note = (
            "Archives contain no harvestable repertoire (G_available = 0). "
            "This does not kill Summer — V6I26 Stage-1 manufactures specialists "
            "via response-oracle training. First success = G_after > G_before."
        )
    else:
        decision = "PROMOTE_LRO_BIRTH"
        note = "Positive G_available without full niche_signal; proceed to Stage-1."

    payload = {
        "experiment": "v6i26_strategic_landscape_scan",
        "lro": lro_manifest(),
        "policies": [p.__dict__ for p in policies],
        "opponents": list(args.opponents),
        "maps": list(args.maps),
        "episodes_per_cell": int(args.episodes_per_cell),
        "seed": int(args.seed),
        "payoff_matrix": payoff.tolist(),
        "winrate_matrix": np.asarray(collected["winrate_matrix"]).tolist(),
        "contexts": list(collected["contexts"]),
        "summary": summary,
        "next_response_target": target,
        "cross_fitted_oracle": strategic,
        "G_available_effective": g_effective,
        "decision": decision,
        "note": note,
        "first_true_v6i26_success": "G_available_after > G_available_before",
    }
    write_json(out_dir / "landscape_scan.json", payload)
    write_json(out_dir / "payoff_summary.json", summary)

    print()
    print("--- Landscape verdict ---")
    print(f"unique_best={summary['unique_best_count']} labels={summary['unique_best_labels']}")
    print(f"G_available_point={summary['G_available_point']:.4f}")
    print(f"G_available_effective={g_effective:.4f}")
    print(f"max_row_distance={summary['max_pairwise_row_distance']:.4f}")
    print(f"decision={decision}")
    print(f"note={note}")
    print(f"Wrote {out_dir / 'landscape_scan.json'}")
    # Always exit 0 unless the scan itself failed — Stage-1 is the next spend.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
