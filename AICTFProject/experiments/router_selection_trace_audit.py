#!/usr/bin/env python3
"""Tiny latent-selection trace audit for router diagnostic conditions.

Runs one opponent × one map × N held-out episode seeds × four selectors.
Records per-episode z trajectories and asserts that dynamic conditions are not aliases.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.latent_selectors import (  # noqa: E402
    FixedLatentSelector,
    LatentSelector,
    LearnedRouterSelector,
    ShuffledAssignmentSelector,
    UniformSelector,
)
from experiments.forced_z_eval.protocol import DEFAULT_MAX_DECISION_STEPS  # noqa: E402
from rl.evaluation.router_ablation import (  # noqa: E402
    build_shuffled_mapping_from_learned_traces,
    learned_z_histogram_from_traces,
    shuffled_mapping_z_histogram,
    validate_shuffled_mapping_histogram,
)

DEFAULT_CHECKPOINT = (
    "checkpoints/2v2/final_v6i9-mapaware-router-feedforward-hardpool-refactor-r1-seed1-mechanism_2v2.zip"
)
DEFAULT_OUT = "artifacts/router_selection_trace_audit"


@dataclass(frozen=True)
class AuditProtocol:
    checkpoint: str
    opponent: str
    map_name: str
    episode_seeds: tuple[int, ...]
    device: str
    base_seed: int = 9000
    max_decision_steps: int = DEFAULT_MAX_DECISION_STEPS


def _make_env(protocol: AuditProtocol, seed: int) -> Any:
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo.inference import read_custom_ppo_metadata

    meta = read_custom_ppo_metadata(protocol.checkpoint)
    agents = int(meta.get("n_blue", 2))
    return GPUCTFVecEnv(
        GPUFieldConfig(
            n_envs=1,
            max_blue_agents=agents,
            max_red_agents=agents,
            map_layout=protocol.map_name,
            max_decision_steps=int(protocol.max_decision_steps),
            aquaticus_profile=True,
            rules_profile="OURS",
            device=protocol.device,
            seed=int(seed),
        )
    )


def _entropy_from_probs(probs: list[float]) -> float:
    arr = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-(arr * np.log(arr)).sum())


def _run_episode(
    protocol: AuditProtocol,
    selector: LatentSelector,
    *,
    model: Any,
    episode_seed: int,
    shuffled_mapping: dict[Any, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from plot.eval_rollout import run_eval_episodes

    env = _make_env(protocol, episode_seed)
    try:
        try:
            env.env_method("set_phase", protocol.opponent)
            env.env_method("set_next_opponent", "SCRIPTED", protocol.opponent)
        except Exception:
            pass
        selector.apply(model, shuffled_mapping=shuffled_mapping)
        if hasattr(model, "opportunity_trace_log"):
            model.opportunity_trace_log = []

        episodes = run_eval_episodes(
            protocol.checkpoint,
            env,
            1,
            protocol.device,
            protocol.opponent,
            deterministic=True,
            fixed_latent_id=getattr(getattr(selector, "condition", None), "fixed_latent_id", None),
            latent_eval_seed=int(episode_seed),
            preloaded_model=model,
            expected_strategy_interval=int(getattr(selector.condition, "strategy_interval", 0)),
            expected_allow_switching=bool(getattr(selector.condition, "allow_switching", False)),
            condition_name=selector.condition.name,
            checkpoint_name=Path(protocol.checkpoint).stem,
            selection_rule=selector.expected_rule(),
        )
    finally:
        env.close()

    ep = episodes[0] if episodes else {}
    traces = [dict(t) for t in getattr(model, "opportunity_trace_log", [])]
    z_seq = [int(t["selected_z"]) for t in traces]
    entropies = [_entropy_from_probs(list(t.get("probabilities") or [])) for t in traces if t.get("probabilities")]

    record = {
        "condition": selector.condition.name,
        "condition_display": selector.name,
        "selection_mode": selector.selection_mode,
        "latent_eval_mode": getattr(model, "latent_eval_mode", ""),
        "selection_rule": selector.expected_rule(),
        "opponent": protocol.opponent.upper(),
        "map": protocol.map_name,
        "episode_seed": int(episode_seed),
        "episode_index": 0,
        "initial_z": z_seq[0] if z_seq else None,
        "selected_z_by_decision": z_seq,
        "z_switch_steps": [i for i, (a, b) in enumerate(zip(z_seq, z_seq[1:])) if a != b],
        "z_occupancy": dict(Counter(z_seq)),
        "router_entropy_mean": float(np.mean(entropies)) if entropies else None,
        "router_entropy_by_opportunity": entropies,
        "router_logits_by_opportunity": [list(t.get("logits") or []) for t in traces],
        "router_probabilities_by_opportunity": [list(t.get("probabilities") or []) for t in traces],
        "selection_mode_executed": getattr(model, "latent_eval_mode", ""),
        "return": float(ep.get("return", ep.get("ep_return", 0.0))),
        "success": int(ep.get("success", 0)),
        "steps": int(ep.get("steps", 0)),
        "strategy_switches": int(ep.get("strategy_switches", 0)),
        "strategy_unique_count": int(ep.get("strategy_unique_count", 0)),
        "strategy_dominant": int(ep.get("strategy_dominant", -1)),
    }
    return record, traces


def _z_histogram(rows: list[dict[str, Any]]) -> Counter[int]:
    hist: Counter[int] = Counter()
    for row in rows:
        for z, count in (row.get("z_occupancy") or {}).items():
            hist[int(z)] += int(count)
    return hist


def _build_assertions(
  rows: list[dict[str, Any]],
  *,
  latent_k: int,
) -> dict[str, Any]:
    by_name = {str(r["condition_display"]): r for r in rows}
    learned_rows = [r for r in rows if r["condition"] == "learned_qphi_switching"]
    uniform_rows = [r for r in rows if r["condition"] == "uniform_random_at_router_opportunities"]
    shuffled_rows = [r for r in rows if r["condition"] == "shuffled_qphi_outputs"]
    fixed_rows = [r for r in rows if str(r["condition"]).startswith("fixed_z")]

    fixed_z = int(fixed_rows[0]["condition"].replace("fixed_z", "")) if fixed_rows else 2
    fixed_ok = bool(fixed_rows) and all(
        int(row.get("strategy_dominant", -1)) == fixed_z
        and int(row.get("strategy_switches", 0)) == 0
        and (
            not row.get("selected_z_by_decision")
            or all(int(z) == fixed_z for z in row.get("selected_z_by_decision") or [])
        )
        for row in fixed_rows
    )

    uniform_z_values = set()
    for row in uniform_rows:
        uniform_z_values.update(int(z) for z in (row.get("selected_z_by_decision") or []))

    learned_vs_uniform_diff = 0
    learned_vs_shuffled_diff = 0
    for lrow, urow, srow in zip(learned_rows, uniform_rows, shuffled_rows):
        if lrow.get("selected_z_by_decision") != urow.get("selected_z_by_decision"):
            learned_vs_uniform_diff += 1
        if lrow.get("selected_z_by_decision") != srow.get("selected_z_by_decision"):
            learned_vs_shuffled_diff += 1

    learned_hist = _z_histogram(learned_rows)
    shuffled_hist = _z_histogram(shuffled_rows)

    return {
        "fixed_z2_all_constant": fixed_ok,
        "fixed_z2_switch_count_zero": all(not row.get("z_switch_steps") for row in fixed_rows),
        "uniform_reaches_multiple_z": len(uniform_z_values) >= 2,
        "uniform_z_values_seen": sorted(uniform_z_values),
        "learned_differs_from_uniform_episodes": learned_vs_uniform_diff,
        "learned_differs_from_uniform": learned_vs_uniform_diff > 0,
        "shuffled_differs_from_learned_episodes": learned_vs_shuffled_diff,
        "shuffled_differs_from_learned": learned_vs_shuffled_diff > 0,
        "shuffled_preserves_marginal_vs_learned": learned_hist == shuffled_hist,
        "learned_z_histogram": dict(learned_hist),
        "shuffled_z_histogram": dict(shuffled_hist),
        "learned_uniform_trace_identical_all": learned_vs_uniform_diff == 0 and len(learned_rows) > 0,
        "learned_shuffled_trace_identical_all": learned_vs_shuffled_diff == 0 and len(learned_rows) > 0,
        "diagnostic_wiring_trusted": (
            learned_vs_uniform_diff > 0 and learned_vs_shuffled_diff > 0 and len(uniform_z_values) >= 2
        ),
        "latent_k": latent_k,
    }


def _trace_equivalence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["condition"]), int(row["episode_seed"]))
        keyed[key] = row

    conditions = sorted({str(r["condition"]) for r in rows})
    out: dict[str, Any] = {}
    for i, left in enumerate(conditions):
        for right in conditions[i + 1 :]:
            seeds = sorted(
                {
                    int(seed)
                    for (cond, seed) in keyed
                    if cond in (left, right)
                }
            )
            matched = [seed for seed in seeds if (left, seed) in keyed and (right, seed) in keyed]
            same_seq = 0
            for seed in matched:
                if keyed[(left, seed)].get("selected_z_by_decision") == keyed[(right, seed)].get("selected_z_by_decision"):
                    same_seq += 1
            out[f"{left}__vs__{right}"] = {
                "n_matched_seeds": len(matched),
                "same_z_sequence_count": same_seq,
                "same_z_sequence_fraction": float(same_seq) / len(matched) if matched else 0.0,
            }
    return out


def run_audit(protocol: AuditProtocol, output_dir: Path) -> dict[str, Any]:
    from rl.custom_ppo import load_custom_ppo_policy, read_custom_ppo_metadata

    output_dir.mkdir(parents=True, exist_ok=True)
    meta = read_custom_ppo_metadata(protocol.checkpoint)
    cfg_meta = meta.get("cfg") if isinstance(meta.get("cfg"), dict) else {}
    latent_k = int(meta.get("latent_k", 4))
    switch_cadence = int(
        cfg_meta.get("latent_resample_every_n")
        or cfg_meta.get("latent_resample_every")
        or cfg_meta.get("strategy_interval")
        or 32
    )

    probe_env = _make_env(protocol, protocol.episode_seeds[0])
    model = load_custom_ppo_policy(
        protocol.checkpoint,
        probe_env.observation_space,
        probe_env.action_space,
        device=protocol.device,
    )
    probe_env.close()

    selectors: list[LatentSelector] = [
        LearnedRouterSelector(strategy_interval=switch_cadence),
        FixedLatentSelector(latent_id=2),
        UniformSelector(strategy_interval=switch_cadence),
        ShuffledAssignmentSelector(strategy_interval=switch_cadence),
    ]

    episode_records: list[dict[str, Any]] = []
    raw_traces: list[dict[str, Any]] = []
    learned_traces_flat: list[dict[str, Any]] = []
    shuffled_mapping: dict[Any, Any] | None = None
    shuffled_meta: dict[str, Any] = {}

    print("Router selection trace audit")
    print(f"  checkpoint : {protocol.checkpoint}")
    print(f"  opponent   : {protocol.opponent}")
    print(f"  map        : {protocol.map_name}")
    print(f"  seeds      : {list(protocol.episode_seeds)}")
    print(f"  cadence    : {switch_cadence}")
    print()

    for selector in selectors:
        if isinstance(selector, ShuffledAssignmentSelector) and shuffled_mapping is None:
            raise RuntimeError("Shuffled selector requires learned traces first")
        print(f"Running {selector.name}...")
        for episode_seed in protocol.episode_seeds:
            record, traces = _run_episode(
                protocol,
                selector,
                model=model,
                episode_seed=episode_seed,
                shuffled_mapping=shuffled_mapping if isinstance(selector, ShuffledAssignmentSelector) else None,
            )
            episode_records.append(record)
            for trace in traces:
                item = dict(trace)
                item["condition"] = selector.condition.name
                item["episode_seed"] = episode_seed
                raw_traces.append(item)
            if selector.condition.name == "learned_qphi_switching":
                learned_traces_flat.extend(traces)
            print(
                f"  seed={episode_seed}: z={record.get('selected_z_by_decision')} "
                f"mode={record.get('selection_mode_executed')} ret={record.get('return'):.3f}"
            )
        if selector.condition.name == "learned_qphi_switching":
            shuffled_mapping, shuffled_meta = build_shuffled_mapping_from_learned_traces(
                learned_traces_flat,
                latent_k=latent_k,
                allowed_latents=list(range(latent_k)),
                switch_cadence=switch_cadence,
                max_decision_steps=protocol.max_decision_steps,
                require_min_contexts=len(protocol.episode_seeds) >= 2,
            )

    assertions = _build_assertions(episode_records, latent_k=latent_k)
    equivalence = _trace_equivalence(episode_records)

    condition_summary: dict[str, Any] = {}
    for selector in selectors:
        rows = [r for r in episode_records if r["condition"] == selector.condition.name]
        condition_summary[selector.name] = {
            "condition": selector.condition.name,
            "selection_mode": selector.selection_mode,
            "n_episodes": len(rows),
            "mean_return": float(np.mean([float(r["return"]) for r in rows])) if rows else None,
            "z_histogram": dict(_z_histogram(rows)),
            "unique_z_sequences": len({tuple(r.get("selected_z_by_decision") or []) for r in rows}),
        }

    report = {
        "protocol": "router_selection_trace_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": protocol.checkpoint,
        "opponent": protocol.opponent,
        "map": protocol.map_name,
        "episode_seeds": list(protocol.episode_seeds),
        "switch_cadence": switch_cadence,
        "shuffled_mapping_meta": shuffled_meta,
        "condition_summary": condition_summary,
        "assertions": assertions,
        "trace_equivalence": equivalence,
        "verdict": {
            "diagnostic_wiring_trusted": assertions["diagnostic_wiring_trusted"],
            "proceed_to_250k": False,
            "note": (
                "Trace audit only; 250k remains blocked until full ablation passes with trusted wiring."
            ),
        },
    }

    jsonl_path = output_dir / "episode_selection_traces.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in episode_records:
            fh.write(json.dumps(row) + "\n")

    with (output_dir / "condition_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(condition_summary, fh, indent=2)
    with (output_dir / "trace_equivalence_report.json").open("w", encoding="utf-8") as fh:
        json.dump({"assertions": assertions, "equivalence": equivalence, "verdict": report["verdict"]}, fh, indent=2)
    with (output_dir / "audit_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print()
    print("=== Assertions ===")
    for key, value in assertions.items():
        print(f"  {key}: {value}")
    print()
    print(f"Wrote {output_dir}")
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latent-selection trace audit for router diagnostics")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--opponent", default="OP8")
    p.add_argument("--map", default="map_b")
    p.add_argument("--base-seed", type=int, default=9000)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    seeds = tuple(int(args.base_seed) + i for i in range(int(args.n_seeds)))
    protocol = AuditProtocol(
        checkpoint=str(args.checkpoint),
        opponent=str(args.opponent).upper(),
        map_name=str(args.map),
        episode_seeds=seeds,
        device=str(args.device),
        base_seed=int(args.base_seed),
    )
    run_audit(protocol, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
