#!/usr/bin/env python3
"""Held-out router diagnostic ablation before committing to 250k router training.

Runs four matched-seed conditions on the OP8/OP9/OP10 × map_b grid:
  learned_router, fixed_z2, uniform_z, shuffled_router

Primary metric: mean episode return on held-out seeds (default base_seed=4242).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.forced_z_eval.protocol import (  # noqa: E402
    DEFAULT_MAPS,
    DEFAULT_MAX_DECISION_STEPS,
    DEFAULT_OPPONENTS,
    ForcedZProtocol,
)
from rl.custom_ppo.diagnostics.frozen_repertoire_hash import compare_frozen_repertoire_hashes  # noqa: E402
from rl.evaluation.router_ablation import (  # noqa: E402
    build_cross_episode_shuffled_mapping_from_learned_traces,
    build_shuffled_mapping_from_learned_traces,
    check_telemetry_invariants,
    configure_condition,
    default_conditions,
    file_sha256,
    get_actor_module,
    hash_module,
)
from rl.evaluation.types import EvalCondition  # noqa: E402

HELD_OUT_BASE_SEED = 4242
DEFAULT_CHECKPOINT = (
    "checkpoints/2v2/final_v6i9-mapaware-router-feedforward-hardpool-refactor-r1-seed1-mechanism_2v2.zip"
)
DEFAULT_ANCHOR = "checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip"

CONDITION_ORDER = (
    "learned_qphi_switching",
    "fixed_z2",
    "uniform_random_at_router_opportunities",
    "shuffled_qphi_outputs",
    "shuffled_qphi_cross_episode",
)
TRACE_AUDIT_CONDITION_ORDER = (
    "fixed_z2",
    "uniform_episode_fixed",
    "qphi_initial_only_no_switch",
    "shuffled_qphi_initial_only_no_switch",
    "learned_qphi_switching",
)
DISPLAY_NAMES = {
    "learned_qphi_switching": "learned_router",
    "fixed_z2": "fixed_z2",
    "uniform_episode_fixed": "uniform_z_episode_fixed",
    "uniform_random_at_router_opportunities": "uniform_z",
    "qphi_initial_only_no_switch": "learned_router_start_only",
    "shuffled_qphi_outputs": "shuffled_router",
    "shuffled_qphi_cross_episode": "shuffled_router_cross_episode",
    "shuffled_qphi_initial_only_no_switch": "shuffled_router_start_only",
}


@dataclass(frozen=True)
class DiagnosticProtocol:
    checkpoint: str
    anchor_checkpoint: str
    opponents: tuple[str, ...]
    maps: tuple[str, ...]
    episodes_per_cell: int
    base_seed: int
    device: str
    max_decision_steps: int = DEFAULT_MAX_DECISION_STEPS

    def cell_seed(self, opponent_index: int, map_index: int) -> int:
        return int(self.base_seed) + 1000 * int(opponent_index) + 100 * int(map_index)


def _make_env(protocol: DiagnosticProtocol, map_name: str, seed: int) -> Any:
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo.inference import read_custom_ppo_metadata

    meta = read_custom_ppo_metadata(protocol.checkpoint)
    agents = int(meta.get("n_blue", 2))
    return GPUCTFVecEnv(
        GPUFieldConfig(
            n_envs=1,
            max_blue_agents=agents,
            max_red_agents=agents,
            map_layout=map_name,
            max_decision_steps=int(protocol.max_decision_steps),
            aquaticus_profile=True,
            rules_profile="OURS",
            device=protocol.device,
            seed=int(seed),
        )
    )


def _fixed_latent_for_condition(condition: EvalCondition) -> int | None:
    if condition.name.startswith("fixed_z") and condition.fixed_latent_id is not None:
        return int(condition.fixed_latent_id)
    return None


def _run_condition(
    protocol: DiagnosticProtocol,
    condition: EvalCondition,
    *,
    switch_cadence: int,
    shuffled_mapping: dict[Any, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from plot.eval_rollout import run_eval_episodes
    from rl.custom_ppo import load_custom_ppo_policy

    first_env = _make_env(protocol, protocol.maps[0], protocol.base_seed)
    obs_space = first_env.observation_space
    act_space = first_env.action_space
    first_env.close()

    model = load_custom_ppo_policy(protocol.checkpoint, obs_space, act_space, device=protocol.device)
    if hasattr(model, "clear_eval_suite_state"):
        model.clear_eval_suite_state()
    configure_condition(model, condition)
    if condition.selection_rule == "shuffled_qphi" and shuffled_mapping is not None:
        if hasattr(model, "inject_shuffled_mapping"):
            model.inject_shuffled_mapping(shuffled_mapping)
    actor_hash_before = hash_module(get_actor_module(model))
    if hasattr(model, "opportunity_trace_log"):
        model.opportunity_trace_log = []

    rows: list[dict[str, Any]] = []
    trace_data: list[dict[str, Any]] = []
    z_id = _fixed_latent_for_condition(condition)
    for opp_idx, opponent in enumerate(protocol.opponents):
        for map_idx, map_name in enumerate(protocol.maps):
            cell_seed = protocol.cell_seed(opp_idx, map_idx)
            if hasattr(model, "set_current_map"):
                model.set_current_map(map_name)
            # Snapshot trace-log position before this cell so new items can be
            # tagged with the correct map name. The cross-episode shuffler groups
            # by (opponent, map); without this tag it degenerates into per-episode
            # singletons (can_reassign=False) regardless of z diversity.
            trace_offset = (
                len(model.opportunity_trace_log)
                if hasattr(model, "opportunity_trace_log")
                else 0
            )
            env = _make_env(protocol, map_name, cell_seed)
            try:
                try:
                    env.env_method("set_phase", opponent)
                    env.env_method("set_next_opponent", "SCRIPTED", opponent)
                except Exception:
                    pass
                episodes = run_eval_episodes(
                    protocol.checkpoint,
                    env,
                    int(protocol.episodes_per_cell),
                    protocol.device,
                    opponent,
                    deterministic=True,
                    fixed_latent_id=z_id,
                    latent_eval_seed=int(cell_seed),
                    preloaded_model=model,
                    expected_strategy_interval=int(condition.strategy_interval),
                    expected_allow_switching=bool(condition.allow_switching),
                    condition_name=condition.name,
                    checkpoint_name=Path(protocol.checkpoint).stem,
                    selection_rule=condition.selection_rule,
                    progress_every=max(5, protocol.episodes_per_cell // 5),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR {condition.name} {opponent} {map_name}: {exc}")
                episodes = []
            finally:
                env.close()

            # Collect new trace items for this cell and tag with map_name so the
            # cross-episode shuffler can group by (opponent, map) correctly.
            if hasattr(model, "opportunity_trace_log"):
                for item in model.opportunity_trace_log[trace_offset:]:
                    row = dict(item)
                    row["condition"] = condition.name
                    row["map"] = map_name
                    trace_data.append(row)

            for ep_idx, ep in enumerate(episodes):
                episode_seed = int(ep.get("episode_seed", int(cell_seed) + int(ep_idx)))
                rows.append(
                    {
                        "condition": condition.name,
                        "condition_display": DISPLAY_NAMES.get(condition.name, condition.name),
                        "opponent": str(opponent).upper(),
                        "map": map_name,
                        "cell_seed": int(cell_seed),
                        "episode_index": int(ep_idx),
                        "episode_seed": episode_seed,
                        "return": float(ep.get("return", ep.get("ep_return", 0.0))),
                        "success": int(ep.get("success", 0)),
                        "steps": int(ep.get("steps", 0)),
                        "blue_score": int(ep.get("blue_score", 0)),
                        "red_score": int(ep.get("red_score", 0)),
                        "score_margin": int(ep.get("win_margin", 0)),
                        "strategy_switches": int(ep.get("strategy_switches", 0)),
                        "strategy_resamples": int(ep.get("strategy_resamples", 0)),
                        "strategy_unique_count": int(ep.get("strategy_unique_count", 0)),
                        "strategy_entropy_mean": float(ep.get("strategy_entropy_mean", float("nan"))),
                        "strategy_dominant": int(ep.get("strategy_dominant", -1)),
                        "checkpoint": protocol.checkpoint,
                    }
                )
            wr = (
                sum(int(e.get("success", 0)) for e in episodes) / len(episodes)
                if episodes
                else float("nan")
            )
            mean_ret = (
                float(np.mean([float(e.get("return", e.get("ep_return", 0.0))) for e in episodes]))
                if episodes
                else float("nan")
            )
            print(
                f"  [{DISPLAY_NAMES.get(condition.name, condition.name)}] "
                f"{opponent} {map_name}: ret={mean_ret:.3f} WR={wr:.1%} ({len(episodes)} eps)"
            )

    check_rows = []
    for row in rows:
        check_row = dict(row)
        check_row["seed"] = int(row["cell_seed"])
        check_rows.append(check_row)
    check_telemetry_invariants(condition, trace_data, check_rows)

    actor_hash_after = hash_module(get_actor_module(model))
    integrity = {
        "actor_hash_before": actor_hash_before,
        "actor_hash_after": actor_hash_after,
        "actor_unchanged": actor_hash_before == actor_hash_after,
        "trace_opportunities": len(trace_data),
    }
    return rows, trace_data, integrity


def _jsonish(value: Any) -> str:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _flatten_trace_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in trace_rows:
        flat = {
            "condition": item.get("condition", ""),
            "opponent": item.get("opponent", ""),
            "map": item.get("map", ""),
            "seed": item.get("seed", ""),
            "environment_seed": item.get("environment_seed", ""),
            "episode_index": item.get("episode_index", ""),
            "opportunity_index": item.get("opportunity_index", ""),
            "step": item.get("step", ""),
            "selected_z": item.get("selected_z", ""),
            "prev_z": item.get("prev_z", ""),
            "switch_occurred": item.get("switch_occurred", ""),
        }
        logits = item.get("logits") or []
        probs = item.get("probabilities") or []
        for idx, val in enumerate(logits):
            flat[f"logit_{idx}"] = float(val)
        for idx, val in enumerate(probs):
            flat[f"prob_{idx}"] = float(val)
        out.append(flat)
    return out


def _episode_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row.get("opponent", "")).upper(),
        str(row.get("map", "")),
        int(row.get("cell_seed", 0)),
        int(row.get("episode_index", 0)),
    )


def _trace_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("opponent", "")).upper(),
        int(row.get("seed", 0)),
        int(row.get("episode_index", 0)),
    )


def _summarize_trace_episodes(
    episode_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trace_by_key: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in trace_rows:
        key = (*_trace_key(trace), str(trace.get("condition", "")))
        trace_by_key[key].append(trace)

    out: list[dict[str, Any]] = []
    for row in episode_rows:
        key = (
            str(row.get("opponent", "")).upper(),
            int(row.get("cell_seed", 0)),
            int(row.get("episode_index", 0)),
            str(row.get("condition", "")),
        )
        traces = sorted(trace_by_key.get(key, []), key=lambda x: int(x.get("opportunity_index", 0)))
        z_seq = [int(t["selected_z"]) for t in traces if t.get("selected_z") not in (None, "")]
        if not z_seq and str(row.get("condition", "")).startswith("fixed_z"):
            fixed_suffix = str(row.get("condition", ""))[len("fixed_z") :]
            if fixed_suffix.isdigit():
                z_seq = [int(fixed_suffix)]
        entropies = []
        for trace in traces:
            probs = trace.get("probabilities") or []
            if probs:
                arr = np.asarray(probs, dtype=np.float64)
                arr = np.clip(arr, 1e-12, 1.0)
                entropies.append(float(-(arr * np.log(arr)).sum()))
        out.append(
            {
                "condition": row.get("condition", ""),
                "condition_display": row.get("condition_display", ""),
                "opponent": row.get("opponent", ""),
                "map": row.get("map", ""),
                "episode_seed": int(row.get("episode_seed", 0)),
                "cell_seed": int(row.get("cell_seed", 0)),
                "episode_index": int(row.get("episode_index", 0)),
                "initial_z": z_seq[0] if z_seq else "",
                "z_sequence": " ".join(str(z) for z in z_seq),
                "z_sequence_len": len(z_seq),
                "z_switches_at_opportunities": sum(1 for a, b in zip(z_seq, z_seq[1:]) if a != b),
                "logged_switches": sum(int(t.get("switch_occurred", 0) or 0) for t in traces),
                "router_entropy_mean": float(np.mean(entropies)) if entropies else float("nan"),
                "return": float(row.get("return", 0.0)),
                "score_margin": int(row.get("score_margin", 0)),
                "episode_length": int(row.get("steps", 0)),
                "strategy_switches": int(row.get("strategy_switches", 0)),
                "strategy_resamples": int(row.get("strategy_resamples", 0)),
                "strategy_unique_count": int(row.get("strategy_unique_count", 0)),
            }
        )
    return out


def _compare_trace_summaries(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition_key = {
        (str(row["condition"]), _episode_key(row)): row
        for row in summary_rows
    }
    conditions = sorted({str(row["condition"]) for row in summary_rows})
    out: dict[str, Any] = {}
    for i, left in enumerate(conditions):
        for right in conditions[i + 1 :]:
            left_keys = {key for cond, key in by_condition_key if cond == left}
            right_keys = {key for cond, key in by_condition_key if cond == right}
            keys = sorted(left_keys & right_keys)
            if not keys:
                continue
            same_initial = 0
            same_sequence = 0
            same_return = 0
            return_deltas: list[float] = []
            for key in keys:
                lrow = by_condition_key[(left, key)]
                rrow = by_condition_key[(right, key)]
                if str(lrow.get("initial_z", "")) == str(rrow.get("initial_z", "")):
                    same_initial += 1
                if str(lrow.get("z_sequence", "")) == str(rrow.get("z_sequence", "")):
                    same_sequence += 1
                lret = float(lrow.get("return", 0.0))
                rret = float(rrow.get("return", 0.0))
                if abs(lret - rret) <= 1e-12:
                    same_return += 1
                return_deltas.append(lret - rret)
            label = f"{left}__vs__{right}"
            out[label] = {
                "n_matched_episodes": len(keys),
                "same_initial_z_count": same_initial,
                "same_initial_z_fraction": float(same_initial) / len(keys),
                "same_z_sequence_count": same_sequence,
                "same_z_sequence_fraction": float(same_sequence) / len(keys),
                "same_return_count": same_return,
                "same_return_fraction": float(same_return) / len(keys),
                "mean_return_delta_left_minus_right": float(np.mean(return_deltas)),
                "max_abs_return_delta": float(np.max(np.abs(return_deltas))) if return_deltas else 0.0,
            }
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[float]] = defaultdict(list)
    by_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        cond = str(row["condition"])
        ret = float(row["return"])
        by_condition[cond].append(ret)
        by_cell[(cond, str(row["opponent"]), str(row["map"]))].append(ret)

    summary: dict[str, Any] = {"global": {}, "per_cell": {}}
    for cond, vals in sorted(by_condition.items()):
        summary["global"][cond] = {
            "display": DISPLAY_NAMES.get(cond, cond),
            "n_episodes": len(vals),
            "mean_return": float(np.mean(vals)) if vals else float("nan"),
            "std_return": float(np.std(vals)) if vals else float("nan"),
            "win_rate": float(np.mean([float(r.get("success", 0)) for r in rows if r["condition"] == cond]))
            if vals
            else float("nan"),
        }
    for key, vals in sorted(by_cell.items()):
        cond, opp, map_name = key
        summary["per_cell"][f"{cond}|{opp}|{map_name}"] = {
            "mean_return": float(np.mean(vals)),
            "n_episodes": len(vals),
        }
    return summary


def _z_histogram_from_trace_summaries(rows: list[dict[str, Any]]) -> dict[int, int]:
    hist: dict[int, int] = {}
    for row in rows:
        for token in str(row.get("z_sequence", "")).split():
            if token:
                z = int(token)
                hist[z] = hist.get(z, 0) + 1
    return hist


def _build_v2_trust_checks(
    all_rows: list[dict[str, Any]],
    trace_summary_rows: list[dict[str, Any]],
    trace_comparison: dict[str, Any],
    frozen: dict[str, Any],
    *,
    episodes_per_cell: int,
) -> dict[str, Any]:
    """Pre-trust checks required before interpreting ablation v2 results."""
    cell_episode_seeds: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in all_rows:
        key = (str(row["opponent"]), str(row["map"]), str(row["condition"]))
        cell_episode_seeds[key].add(int(row["episode_seed"]))

    unique_counts = {f"{k[0]}|{k[1]}|{k[2]}": len(v) for k, v in cell_episode_seeds.items()}
    seeds_per_cell_ok = all(n == int(episodes_per_cell) for n in unique_counts.values())

    seed_sets_by_cell: dict[tuple[str, str], list[set[int]]] = defaultdict(list)
    for (opp, map_name, cond), seeds in cell_episode_seeds.items():
        seed_sets_by_cell[(opp, map_name)].append(seeds)
    same_seed_set_across_conditions = all(
        len({frozenset(s) for s in sets}) == 1 for sets in seed_sets_by_cell.values() if sets
    )

    learned_vs_uniform = trace_comparison.get(
        "learned_qphi_switching__vs__uniform_random_at_router_opportunities", {}
    )
    learned_vs_shuffled = trace_comparison.get("learned_qphi_switching__vs__shuffled_qphi_outputs", {})
    learned_vs_cross = trace_comparison.get(
        "learned_qphi_switching__vs__shuffled_qphi_cross_episode", {}
    )

    learned_summaries = [r for r in trace_summary_rows if r["condition"] == "learned_qphi_switching"]
    shuffled_summaries = [r for r in trace_summary_rows if r["condition"] == "shuffled_qphi_outputs"]
    cross_summaries = [r for r in trace_summary_rows if r["condition"] == "shuffled_qphi_cross_episode"]
    fixed_summaries = [r for r in trace_summary_rows if r["condition"] == "fixed_z2"]

    learned_hist = _z_histogram_from_trace_summaries(learned_summaries)
    shuffled_hist = _z_histogram_from_trace_summaries(shuffled_summaries)
    cross_hist = _z_histogram_from_trace_summaries(cross_summaries)

    def _episode_level_hist(summaries: list[dict[str, Any]]) -> dict[int, int]:
        hist: dict[int, int] = {}
        for row in summaries:
            tok = str(row.get("initial_z", ""))
            if tok == "":
                continue
            z = int(tok)
            hist[z] = hist.get(z, 0) + 1
        return hist

    learned_episode_hist = _episode_level_hist(learned_summaries)
    cross_episode_hist = _episode_level_hist(cross_summaries)

    fixed_z2_rows = [r for r in all_rows if r["condition"] == "fixed_z2"]
    fixed_z2_ok = bool(fixed_z2_rows) and all(
        int(r.get("strategy_dominant", -1)) == 2 for r in fixed_z2_rows
    ) and all(
        not str(r.get("z_sequence", "")) or all(tok == "2" for tok in str(r.get("z_sequence", "")).split())
        for r in fixed_summaries
    )

    within_episode_shuffle_differs = (
        float(learned_vs_shuffled.get("same_z_sequence_fraction", 1.0)) < 1.0
    )
    cross_episode_shuffle_differs = (
        float(learned_vs_cross.get("same_z_sequence_fraction", 1.0)) < 1.0
    )

    return {
        "unique_episode_seeds_per_cell": unique_counts,
        "unique_episode_seeds_per_cell_ok": seeds_per_cell_ok,
        "same_seed_set_across_conditions": same_seed_set_across_conditions,
        "learned_trace_differs_from_uniform": float(
            learned_vs_uniform.get("same_z_sequence_fraction", 1.0)
        )
        < 1.0,
        "within_episode_shuffle_differs_from_learned": within_episode_shuffle_differs,
        "cross_episode_shuffle_differs_from_learned": cross_episode_shuffle_differs,
        # Back-compat alias: the primary testable control is now cross-episode.
        "shuffled_mapping_differs_from_learned": cross_episode_shuffle_differs,
        "shuffled_latent_histogram_preserved": learned_hist == shuffled_hist,
        "cross_episode_latent_histogram_preserved": learned_episode_hist == cross_episode_hist,
        "learned_z_histogram": learned_hist,
        "shuffled_z_histogram": shuffled_hist,
        "cross_episode_z_histogram": cross_hist,
        "learned_episode_z_histogram": learned_episode_hist,
        "cross_episode_episode_z_histogram": cross_episode_hist,
        "fixed_z2_always_selects_z2": fixed_z2_ok,
        "frozen_repertoire_hash_match": bool(frozen.get("frozen_tensor_hash_match", False)),
        "v2_trustworthy": (
            seeds_per_cell_ok
            and same_seed_set_across_conditions
            and float(learned_vs_uniform.get("same_z_sequence_fraction", 1.0)) < 1.0
            and cross_episode_shuffle_differs
            and (learned_episode_hist == cross_episode_hist)
            and fixed_z2_ok
            and bool(frozen.get("frozen_tensor_hash_match", False))
        ),
    }


def _build_verdict(
    summary: dict[str, Any],
    integrity_rows: list[dict[str, Any]],
    frozen: dict[str, Any],
    *,
    trust_checks: dict[str, Any] | None = None,
    trace_comparison: dict[str, Any] | None = None,
    cross_episode_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    global_s = summary["global"]
    learned = global_s.get("learned_qphi_switching", {}).get("mean_return", float("nan"))
    fixed_z2 = global_s.get("fixed_z2", {}).get("mean_return", float("nan"))
    uniform = global_s.get("uniform_random_at_router_opportunities", {}).get("mean_return", float("nan"))
    shuffled = global_s.get("shuffled_qphi_outputs", {}).get("mean_return", float("nan"))
    cross_shuffled = global_s.get("shuffled_qphi_cross_episode", {}).get("mean_return", float("nan"))

    integrity_holds = bool(frozen.get("frozen_tensor_hash_match", False)) and all(
        bool(row.get("actor_unchanged", False)) for row in integrity_rows
    )
    beats_within_shuffled = bool(learned > shuffled) if np.isfinite(learned) and np.isfinite(shuffled) else False

    # Cross-episode gate: only meaningful when the shuffler could actually reassign
    # episodes. When all episodes share the same z-signature (can_reassign=False),
    # the shuffled condition is byte-identical to learned and the gate is vacuous.
    cross_can_reassign = bool((cross_episode_meta or {}).get("can_reassign", True))
    cross_gate_untestable = not cross_can_reassign
    beats_cross_shuffled = (
        bool(learned > cross_shuffled)
        if (np.isfinite(learned) and np.isfinite(cross_shuffled) and cross_can_reassign)
        else False
    )
    # The cross-episode control is the meaningful "context routing" test; the
    # within-episode control is degenerate for an episode-constant router.
    beats_shuffled = beats_cross_shuffled
    no_regression = bool(learned >= fixed_z2) if np.isfinite(learned) and np.isfinite(fixed_z2) else False
    near_fixed_z2 = bool(learned >= fixed_z2 - 0.25) if np.isfinite(learned) and np.isfinite(fixed_z2) else False
    beats_fixed = bool(learned > fixed_z2) if np.isfinite(learned) and np.isfinite(fixed_z2) else False
    beats_uniform = bool(learned > uniform) if np.isfinite(learned) and np.isfinite(uniform) else False
    learned_advantage = beats_uniform or beats_shuffled or beats_fixed

    per_cell = summary.get("per_cell", {})
    learned_cells = {
        k.split("|", 1)[1]: v["mean_return"]
        for k, v in per_cell.items()
        if k.startswith("learned_qphi_switching|")
    }
    fixed_cells = {
        k.split("|", 1)[1]: v["mean_return"]
        for k, v in per_cell.items()
        if k.startswith("fixed_z2|")
    }
    cells_learned_beats_fixed = sum(
        1 for key, lret in learned_cells.items() if key in fixed_cells and lret > fixed_cells[key]
    )
    cells_total = len(learned_cells)

    proceed_to_250k = (
        integrity_holds
        and bool(trust_checks.get("v2_trustworthy", True) if trust_checks else True)
        and beats_uniform
        and beats_shuffled
        and near_fixed_z2
    )

    return {
        "integrity_holds": integrity_holds,
        "frozen_repertoire_match": bool(frozen.get("frozen_tensor_hash_match", False)),
        "no_regression_vs_fixed_z2": no_regression,
        "near_fixed_z2": near_fixed_z2,
        "learned_beats_fixed_z2": beats_fixed,
        "learned_beats_uniform_z": beats_uniform,
        "learned_beats_shuffled_router": beats_shuffled,
        "learned_beats_within_episode_shuffled": beats_within_shuffled,
        "learned_beats_cross_episode_shuffled": beats_cross_shuffled,
        "cross_episode_gate_untestable": cross_gate_untestable,
        "learned_advantage": learned_advantage,
        "cells_learned_beats_fixed_z2": cells_learned_beats_fixed,
        "cells_total": cells_total,
        "proceed_to_250k": proceed_to_250k,
        "deltas": {
            "learned_minus_fixed_z2": float(learned - fixed_z2)
            if np.isfinite(learned) and np.isfinite(fixed_z2)
            else None,
            "learned_minus_uniform_z": float(learned - uniform)
            if np.isfinite(learned) and np.isfinite(uniform)
            else None,
            "learned_minus_shuffled": float(learned - shuffled)
            if np.isfinite(learned) and np.isfinite(shuffled)
            else None,
            "learned_minus_within_episode_shuffled": float(learned - shuffled)
            if np.isfinite(learned) and np.isfinite(shuffled)
            else None,
            "learned_minus_cross_episode_shuffled": float(learned - cross_shuffled)
            if np.isfinite(learned) and np.isfinite(cross_shuffled)
            else None,
        },
        "primary_metric": "mean_return_on_held_out_seeds",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _trace_audit_conditions(all_conditions: dict[str, EvalCondition]) -> dict[str, EvalCondition]:
    out = dict(all_conditions)
    out["shuffled_qphi_initial_only_no_switch"] = EvalCondition(
        name="shuffled_qphi_initial_only_no_switch",
        selection_rule="shuffled_qphi",
        strategy_interval=0,
        allow_switching=False,
        description="Shuffled q_phi output selected once at episode start and held fixed.",
    )
    return out


def run_diagnostic(protocol: DiagnosticProtocol, output_dir: Path, *, trace_audit: bool = False) -> dict[str, Any]:
    from rl.custom_ppo import load_custom_ppo_policy, read_custom_ppo_metadata

    output_dir.mkdir(parents=True, exist_ok=True)
    meta = read_custom_ppo_metadata(protocol.checkpoint)
    cfg_meta = meta.get("cfg") if isinstance(meta.get("cfg"), dict) else {}
    latent_k = int(meta.get("latent_k", 4))
    allowed_latents = cfg_meta.get("router_allowed_latents") if isinstance(cfg_meta, dict) else None
    if not allowed_latents:
        allowed_latents = list(range(latent_k))

    probe_env = _make_env(protocol, protocol.maps[0], protocol.base_seed)
    obs_space = probe_env.observation_space
    act_space = probe_env.action_space
    model_probe = load_custom_ppo_policy(
        protocol.checkpoint, obs_space, act_space, device=protocol.device
    )
    switch_cadence = int(getattr(model_probe, "strategy_interval", 0) or 0)
    if switch_cadence <= 0:
        switch_cadence = int(
            cfg_meta.get("latent_resample_every_n")
            or cfg_meta.get("latent_resample_every")
            or cfg_meta.get("strategy_interval")
            or 32
        )
    probe_env.close()

    from rl.custom_ppo.checkpoints.archive import _torch_load_checkpoint

    anchor_sd = _torch_load_checkpoint(protocol.anchor_checkpoint, map_location="cpu").get("model_state_dict", {})
    candidate_sd = _torch_load_checkpoint(protocol.checkpoint, map_location="cpu").get("model_state_dict", {})
    frozen = compare_frozen_repertoire_hashes(anchor_sd, candidate_sd)

    all_conditions = {c.name: c for c in default_conditions(latent_k, allowed_latents, switch_cadence)}
    all_conditions["shuffled_qphi_cross_episode"] = EvalCondition(
        name="shuffled_qphi_cross_episode",
        selection_rule="shuffled_qphi",
        strategy_interval=switch_cadence,
        allow_switching=True,
        description=(
            "Cross-episode histogram-preserving control: permute which episode "
            "receives which learned z-signature within each (opponent, seed) cell. "
            "Breaks context->z alignment even when routing is episode-constant."
        ),
    )
    if trace_audit:
        all_conditions = _trace_audit_conditions(all_conditions)
        condition_order = TRACE_AUDIT_CONDITION_ORDER
    else:
        condition_order = CONDITION_ORDER
    missing = [name for name in condition_order if name not in all_conditions]
    if missing:
        raise KeyError(f"Missing condition definitions: {missing}")

    print("V6I9 router diagnostic ablation")
    print(f"  checkpoint : {protocol.checkpoint}")
    print(f"  anchor     : {protocol.anchor_checkpoint}")
    print(f"  base_seed  : {protocol.base_seed} (held-out)")
    print(f"  grid       : {len(protocol.opponents)} opponents × {len(protocol.maps)} maps")
    print(f"  episodes   : {protocol.episodes_per_cell}/cell ({protocol.episodes_per_cell * len(protocol.opponents) * len(protocol.maps)} per condition)")
    print(f"  cadence    : {switch_cadence}")
    print(f"  trace audit: {trace_audit}")
    print(f"  frozen hash match: {frozen.get('frozen_tensor_hash_match')}")
    print()

    all_rows: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    learned_traces: list[dict[str, Any]] = []
    learned_start_traces: list[dict[str, Any]] = []
    shuffled_meta: dict[str, Any] = {}
    shuffled_start_meta: dict[str, Any] = {}

    dynamic_shuffled_mapping: dict[Any, Any] | None = None
    start_shuffled_mapping: dict[Any, Any] | None = None
    cross_episode_shuffled_mapping: dict[Any, Any] | None = None
    cross_episode_meta: dict[str, Any] = {}

    for cond_name in condition_order:
        condition = all_conditions[cond_name]
        if condition.name == "shuffled_qphi_outputs" and dynamic_shuffled_mapping is None:
            raise RuntimeError("Dynamic shuffled condition requires learned_qphi_switching traces first.")
        if condition.name == "shuffled_qphi_cross_episode" and cross_episode_shuffled_mapping is None:
            raise RuntimeError("Cross-episode shuffled condition requires learned_qphi_switching traces first.")
        if condition.name == "shuffled_qphi_initial_only_no_switch" and start_shuffled_mapping is None:
            raise RuntimeError("Start-only shuffled condition requires qphi_initial_only_no_switch traces first.")
        print(f"Running {DISPLAY_NAMES.get(condition.name, condition.name)}...")
        if condition.name == "shuffled_qphi_outputs":
            mapping = dynamic_shuffled_mapping
        elif condition.name == "shuffled_qphi_cross_episode":
            mapping = cross_episode_shuffled_mapping
        elif condition.name == "shuffled_qphi_initial_only_no_switch":
            mapping = start_shuffled_mapping
        else:
            mapping = None
        rows, _trace, integrity = _run_condition(
            protocol,
            condition,
            switch_cadence=switch_cadence,
            shuffled_mapping=mapping,
        )
        all_rows.extend(rows)
        all_traces.extend(_trace)
        integrity_rows.append({"condition": condition.name, **integrity})
        if condition.name == "learned_qphi_switching":
            learned_traces = _trace
            dynamic_shuffled_mapping, shuffled_meta = build_shuffled_mapping_from_learned_traces(
                learned_traces,
                latent_k=latent_k,
                allowed_latents=allowed_latents,
                switch_cadence=switch_cadence,
                max_decision_steps=protocol.max_decision_steps,
            )
            cross_episode_shuffled_mapping, cross_episode_meta = (
                build_cross_episode_shuffled_mapping_from_learned_traces(
                    learned_traces,
                    latent_k=latent_k,
                    allowed_latents=allowed_latents,
                    switch_cadence=switch_cadence,
                    max_decision_steps=protocol.max_decision_steps,
                )
            )
        if condition.name == "qphi_initial_only_no_switch":
            learned_start_traces = _trace
            start_shuffled_mapping, shuffled_start_meta = build_shuffled_mapping_from_learned_traces(
                learned_start_traces,
                latent_k=latent_k,
                allowed_latents=allowed_latents,
                switch_cadence=0,
                max_decision_steps=protocol.max_decision_steps,
            )

    summary = _summarize(all_rows)
    trace_summary_rows = _summarize_trace_episodes(all_rows, all_traces)
    trace_comparison = _compare_trace_summaries(trace_summary_rows)
    trust_checks = _build_v2_trust_checks(
        all_rows,
        trace_summary_rows,
        trace_comparison,
        frozen,
        episodes_per_cell=protocol.episodes_per_cell,
    )
    verdict = _build_verdict(
        summary,
        integrity_rows,
        frozen,
        trust_checks=trust_checks,
        trace_comparison=trace_comparison,
        cross_episode_meta=cross_episode_meta,
    )

    manifest = {
        "protocol": "v6i9_router_diagnostic_ablation_v1",
        "trace_audit": bool(trace_audit),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": protocol.checkpoint,
        "checkpoint_sha256": file_sha256(protocol.checkpoint),
        "anchor_checkpoint": protocol.anchor_checkpoint,
        "base_seed": protocol.base_seed,
        "episodes_per_cell": protocol.episodes_per_cell,
        "opponents": list(protocol.opponents),
        "maps": list(protocol.maps),
        "switch_cadence": switch_cadence,
        "conditions": [DISPLAY_NAMES.get(c, c) for c in condition_order],
        "frozen_integrity": frozen,
        "shuffled_mapping": shuffled_meta,
        "cross_episode_shuffled_mapping": cross_episode_meta,
        "shuffled_start_mapping": shuffled_start_meta,
        "condition_integrity": integrity_rows,
        "v2_trust_checks": trust_checks,
        "summary": summary,
        "trace_comparison": trace_comparison,
        "verdict": verdict,
    }

    _write_csv(output_dir / "episode_results.csv", all_rows)
    _write_csv(output_dir / "router_opportunity_traces.csv", _flatten_trace_rows(all_traces))
    _write_csv(output_dir / "router_episode_trace_summary.csv", trace_summary_rows)
    with (output_dir / "router_trace_comparison.json").open("w", encoding="utf-8") as fh:
        json.dump(trace_comparison, fh, indent=2)
    with (output_dir / "diagnostic_report.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print()
    print("=== Summary (mean return, held-out) ===")
    for cond in condition_order:
        block = summary["global"].get(cond, {})
        print(f"  {block.get('display', cond):16s}: {block.get('mean_return', float('nan')):+.4f}  (n={block.get('n_episodes', 0)})")
    print()
    print("=== Trace comparison ===")
    for pair, block in sorted(trace_comparison.items()):
        if "learned_qphi_switching" in pair or "qphi_initial_only_no_switch" in pair:
            print(
                f"  {pair}: same_z_seq={block['same_z_sequence_fraction']:.3f}, "
                f"same_return={block['same_return_fraction']:.3f}"
            )
    print()
    print("=== V2 trust checks ===")
    for key in (
        "unique_episode_seeds_per_cell_ok",
        "same_seed_set_across_conditions",
        "learned_trace_differs_from_uniform",
        "within_episode_shuffle_differs_from_learned",
        "cross_episode_shuffle_differs_from_learned",
        "cross_episode_latent_histogram_preserved",
        "shuffled_latent_histogram_preserved",
        "fixed_z2_always_selects_z2",
        "frozen_repertoire_hash_match",
        "v2_trustworthy",
    ):
        print(f"  {key}: {trust_checks.get(key)}")
    print()
    print("=== Verdict ===")
    for key, value in verdict.items():
        if key != "deltas":
            print(f"  {key}: {value}")
    print(f"  deltas: {verdict['deltas']}")
    print()
    print(f"Wrote {output_dir / 'diagnostic_report.json'}")
    return manifest


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I9 held-out router diagnostic ablation")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--anchor-checkpoint", default=DEFAULT_ANCHOR)
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--base-seed", type=int, default=HELD_OUT_BASE_SEED)
    p.add_argument("--device", default="cuda")
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    p.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--trace-audit",
        action="store_true",
        help="Run the router-path equivalence audit matrix with episode-persistent baselines.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"experiments/router_diagnostic_runs/{ts}")
    protocol = DiagnosticProtocol(
        checkpoint=str(args.checkpoint),
        anchor_checkpoint=str(args.anchor_checkpoint),
        opponents=tuple(str(o).upper() for o in args.opponents),
        maps=tuple(args.maps),
        episodes_per_cell=int(args.episodes),
        base_seed=int(args.base_seed),
        device=str(args.device),
    )
    run_diagnostic(protocol, out_dir, trace_audit=bool(args.trace_audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
