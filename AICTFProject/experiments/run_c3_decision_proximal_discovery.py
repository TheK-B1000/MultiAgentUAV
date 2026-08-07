"""C3 commitment-fork discovery runner.

STATUS
------
C3 preregistration is FROZEN. Rollouts still require
artifacts/c3_discovery/C3_EXECUTION_AUTHORIZATION.json with matching
contract / prereg / runner hashes.

SCIENTIFIC SCOPE
----------------
Stage 3 is a CONTROLLABILITY SCREEN ONLY. A qualified commitment fork is not a
strategy, does not establish latent necessity, and does not authorize O3 by
itself. Fresh confirmation and the Environment-Demand Gate remain separate.
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from experiments.run_g0_v2_evaluation import (
    AGENTS,
    CANONICAL_MAP,
    EPISODE_HORIZON,
    V2_RULES,
)
from experiments.run_g0_v2_seed import OPPONENTS
from rl.analysis.counterfactual_actionability import (
    NO_COMMITMENT_FORK,
    QUALIFIED_COMMITMENT_FORK,
    backward_trace_steps,
    compute_actionability,
    find_earliest_commitment_fork,
    resolve_utility,
    run_counterfactual_branches,
    run_determinism_self_test,
    utility_ceiling_for,
)
from rl.analysis.c3_discovery_artifacts import (
    STAGE3_RESULTS_NAME,
    anchor_key_from_row,
    append_jsonl,
    load_completed_stage3_keys,
    load_stage1_bundle,
    write_stage1_artifacts,
)


DISCOVERY_SEED_BASE = 9_400_000
G0_SEEDS = (3_200_001, 3_200_002, 3_200_003)
OUT_DIR = PROJECT_ROOT / "artifacts" / "c3_discovery"
AUTH_PATH = OUT_DIR / "C3_EXECUTION_AUTHORIZATION.json"
CONTRACT_PATH = OUT_DIR / "C3_DISCOVERY_PREREG_FROZEN.json"
PREREG_PATH = PROJECT_ROOT / "docs" / "c3-decision-proximal-preregistration.md"

CONTROLLABILITY_SCOPE = "CONTROLLABILITY_SCREEN_ONLY"


@dataclass(frozen=True)
class RuntimeContract:
    t_trace: int
    h_response: int
    delta: float
    utility_name: str
    doomed_utility_threshold: float
    minimum_fork_rate: float


def _score_stratum(score_diff: float) -> str:
    if score_diff > 0:
        return "leading"
    if score_diff < 0:
        return "trailing"
    return "tied"


def _parse_runtime_contract(contract: dict) -> RuntimeContract:
    """Fail closed until every frozen Stage-3 runtime cell is present."""
    cells = contract.get("runtime_cells")
    if not isinstance(cells, dict):
        raise ValueError("frozen C3 contract missing object runtime_cells")
    utility = cells.get("U")
    if not isinstance(utility, dict):
        raise ValueError("frozen C3 contract runtime_cells.U must be an object")
    required = {
        "T_trace": cells.get("T_trace"),
        "H_response": cells.get("H_response"),
        "delta": cells.get("delta"),
        "minimum_fork_rate": cells.get("minimum_fork_rate"),
        "U.name": utility.get("name"),
        "U.doomed_at_or_below": utility.get("doomed_at_or_below"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"frozen C3 contract missing runtime cells: {missing}")
    parsed = RuntimeContract(
        t_trace=int(required["T_trace"]),
        h_response=int(required["H_response"]),
        delta=float(required["delta"]),
        utility_name=str(required["U.name"]),
        doomed_utility_threshold=float(required["U.doomed_at_or_below"]),
        minimum_fork_rate=float(required["minimum_fork_rate"]),
    )
    if parsed.t_trace <= 0 or parsed.h_response <= 0:
        raise ValueError("T_trace and H_response must be positive")
    if parsed.delta <= 0.0:
        raise ValueError("delta must be positive")
    if not (0.0 < parsed.minimum_fork_rate <= 1.0):
        raise ValueError("minimum_fork_rate must be in (0, 1]")
    resolve_utility(parsed.utility_name)
    return parsed


def _load_runtime_contract() -> RuntimeContract:
    try:
        payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"C3 frozen contract unreadable: {CONTRACT_PATH}: {exc}") from exc
    try:
        return _parse_runtime_contract(payload)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"C3 frozen runtime contract invalid: {exc}") from exc


def collect_pressure_anchors(
    policy,
    *,
    opponent: str,
    seed: int,
    device: str,
    response_horizon: int,
) -> tuple[list[dict], dict[int, dict]]:
    """Collect natural carrier-pressure crossings as backward-trace anchors."""
    from experiments.eval_v6i9_map_awareness import (
        _adapt_obs_for_policy,
        _done,
        _predict,
        _reset_obs,
        _unpack_step,
    )
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from rl.analysis.decision_proximal_features import DecisionProximalExtractor
    from rl.evaluation.opponent_resolution import (
        get_opponent_key,
        set_opponent,
        validate_opponent_name,
    )

    requested = validate_opponent_name(opponent)
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=AGENTS,
        max_red_agents=AGENTS,
        map_set="train",
        map_layout=CANONICAL_MAP,
        max_decision_steps=EPISODE_HORIZON,
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=int(seed),
        obstacle_obs_channel=True,
        tag_telemetry_enabled=True,
        **V2_RULES,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    model = policy.model if hasattr(policy, "model") else policy
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    extractor = DecisionProximalExtractor()
    anchors: list[dict] = []
    tag_steps: list[int] = []
    capture_steps: list[int] = []
    features_by_step: dict[int, dict] = {}
    try:
        set_opponent(env, requested)
        obs = _reset_obs(env.reset())
        if get_opponent_key(env) != requested:
            raise RuntimeError("opponent drift")
        core.drain_tag_events()
        extractor.reset()

        for step_i in range(EPISODE_HORIZON + 8):
            features = extractor.extract(core)
            features_by_step[step_i] = dataclasses.asdict(features)
            for event in core.drain_tag_events():
                if event.get("event_type") == "capture_scored" and event.get("scoring_team") == "blue":
                    capture_steps.append(step_i)
                if event.get("event_type") == "tagged" and event.get("tagged_team") == "blue":
                    tag_steps.append(step_i)
            if features.is_carrier_pressure_onset:
                row = dataclasses.asdict(features)
                row["pressure_step"] = step_i
                row["score_stratum"] = _score_stratum(features.score_diff)
                anchors.append(row)
            action = _predict(policy, _adapt_obs_for_policy(obs, policy))
            obs, _, done, _infos = _unpack_step(env.step(action))
            if _done(done):
                break
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
        env.close()

    for anchor in anchors:
        pressure_step = int(anchor["pressure_step"])
        anchor["tagged_within_H_response"] = int(
            any(pressure_step < step <= pressure_step + response_horizon for step in tag_steps)
        )
        anchor["captured_within_H_response"] = int(
            any(pressure_step < step <= pressure_step + response_horizon for step in capture_steps)
        )
        anchor["event_role"] = "BACKWARD_TRACE_ANCHOR_ONLY"
    return anchors, features_by_step


def _run_stage_3(
    policy,
    device: str,
    anchors: list[dict],
    contract: RuntimeContract,
    *,
    progress,
    train_seed: int,
    stage3_results_path: Path,
    completed_keys: set[str],
    short_circuit: bool = True,
) -> dict:
    """Replay natural episodes and run the commitment-fork controllability screen."""
    from experiments.eval_v6i9_map_awareness import (
        _adapt_obs_for_policy,
        _done,
        _predict,
        _reset_obs,
        _unpack_step,
    )
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from rl.analysis.c3_discovery_artifacts import read_jsonl
    from rl.evaluation.opponent_resolution import (
        get_opponent_key,
        set_opponent,
        validate_opponent_name,
    )
    from tools.q_probe_local_counterfactual import (
        _restore_env,
        _restore_policy,
        _snapshot_env,
        _snapshot_policy,
    )

    by_episode: dict[tuple[str, int], list[dict]] = defaultdict(list)
    ordered_anchors: list[dict] = []
    for anchor in anchors:
        by_episode[(str(anchor["opponent"]), int(anchor["eval_seed"]))].append(anchor)
        ordered_anchors.append(anchor)

    utility_fn = resolve_utility(contract.utility_name)
    utility_ceiling = utility_ceiling_for(contract.utility_name)
    anchor_results: list[dict] = []
    determinism_checks: list[dict] = []
    timing_rows: list[dict] = []
    model = policy.model if hasattr(policy, "model") else policy
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    n_total = len(ordered_anchors)
    n_done = sum(1 for anchor in ordered_anchors if anchor_key_from_row(anchor) in completed_keys)
    n_skipped_resume = n_done
    stage3_started = time.time()

    try:
        for (opponent, eval_seed), episode_anchors in by_episode.items():
            pending = [
                anchor
                for anchor in episode_anchors
                if anchor_key_from_row(anchor) not in completed_keys
            ]
            if not pending:
                continue

            requested = validate_opponent_name(opponent)
            cfg = GPUFieldConfig(
                n_envs=1,
                max_blue_agents=AGENTS,
                max_red_agents=AGENTS,
                map_set="train",
                map_layout=CANONICAL_MAP,
                max_decision_steps=EPISODE_HORIZON,
                aquaticus_profile=True,
                rules_profile="OURS",
                device=device,
                seed=int(eval_seed),
                obstacle_obs_channel=True,
                tag_telemetry_enabled=True,
                **V2_RULES,
            )
            env = GPUCTFVecEnv(cfg)
            try:
                set_opponent(env, requested)
                obs = _reset_obs(env.reset())
                if get_opponent_key(env) != requested:
                    raise RuntimeError("opponent drift")

                pressure_steps = sorted({int(anchor["pressure_step"]) for anchor in pending})
                candidate_steps = sorted(
                    {
                        candidate
                        for pressure_step in pressure_steps
                        for candidate in backward_trace_steps(pressure_step, contract.t_trace)
                    }
                )
                snapshots: dict[int, tuple[dict, dict, object]] = {}
                for step_i in range(EPISODE_HORIZON + 8):
                    if step_i in candidate_steps:
                        snapshots[step_i] = (
                            _snapshot_env(env),
                            _snapshot_policy(policy),
                            copy.deepcopy(obs),
                        )
                    if candidate_steps and step_i >= candidate_steps[-1]:
                        break
                    action = _predict(policy, _adapt_obs_for_policy(obs, policy))
                    obs, _, done, _infos = _unpack_step(env.step(action))
                    if _done(done):
                        break

                if candidate_steps:
                    first_step = candidate_steps[0]
                    env_snap, policy_snap, candidate_obs = snapshots[first_step]
                    _restore_env(env, env_snap)
                    _restore_policy(policy, policy_snap)
                    passed, difference = run_determinism_self_test(
                        env,
                        policy,
                        candidate_obs,
                        horizon=contract.h_response,
                    )
                    determinism_checks.append(
                        {
                            "episode_key": f"{opponent}:{eval_seed}",
                            "candidate_step": first_step,
                            "passed": bool(passed),
                            "return_abs_difference": float(difference),
                        }
                    )
                    if not passed:
                        raise RuntimeError(
                            f"C3 determinism self-test failed for {opponent}:{eval_seed} "
                            f"at step {first_step}: |delta_return|={difference}"
                        )

                for anchor in pending:
                    pressure_step = int(anchor["pressure_step"])
                    key = anchor_key_from_row(anchor)
                    anchor_t0 = time.time()
                    candidates_searched = 0
                    responses_tested_total = 0
                    legal_alts_total = 0
                    short_circuit_hits = 0

                    def evaluate_candidate(candidate_step: int, _ps=pressure_step):
                        nonlocal responses_tested_total, legal_alts_total, short_circuit_hits
                        if candidate_step not in snapshots:
                            raise RuntimeError(
                                f"natural replay did not capture candidate step {candidate_step} "
                                f"for pressure step {_ps}"
                            )
                        env_snap, policy_snap, candidate_obs = snapshots[candidate_step]
                        _restore_env(env, env_snap)
                        _restore_policy(policy, policy_snap)
                        branch_set = run_counterfactual_branches(
                            env,
                            policy,
                            candidate_obs,
                            candidate_step=candidate_step,
                            response_horizon=contract.h_response,
                            utility_fn=utility_fn,
                            delta=contract.delta,
                            doomed_utility_threshold=contract.doomed_utility_threshold,
                            utility_ceiling=utility_ceiling,
                            short_circuit=short_circuit,
                        )
                        responses_tested_total += int(branch_set.responses_tested)
                        legal_alts_total += int(branch_set.n_legal_alternatives)
                        short_circuit_hits += int(bool(branch_set.short_circuited))
                        return compute_actionability(
                            branch_set,
                            delta=contract.delta,
                            doomed_utility_threshold=contract.doomed_utility_threshold,
                        )

                    def on_candidate(index, n_candidates, candidate_step, evaluation):
                        nonlocal candidates_searched
                        candidates_searched = int(index)
                        progress.heartbeat(
                            done=n_done,
                            total=max(n_total, 1),
                            phase="STAGE3",
                            detail=(
                                f"policy={train_seed} anchor={n_done + 1}/{n_total} "
                                f"candidate_state={index}/{n_candidates} "
                                f"legal_responses={evaluation.n_legal_team_responses} "
                                f"responses_tested={evaluation.responses_tested} "
                                f"fork_found={evaluation.is_actionable} "
                                f"short_circuited={evaluation.short_circuited} "
                                f"elapsed={round(time.time() - stage3_started, 1)}s"
                            ),
                            policy=train_seed,
                            anchor_index=n_done + 1,
                            anchors_total=n_total,
                            candidate_index=index,
                            candidates_total=n_candidates,
                            candidate_step=candidate_step,
                            legal_responses=evaluation.n_legal_team_responses,
                            responses_tested=evaluation.responses_tested,
                            fork_found=bool(evaluation.is_actionable),
                            short_circuited=bool(evaluation.short_circuited),
                        )

                    result = find_earliest_commitment_fork(
                        pressure_step=pressure_step,
                        t_trace=contract.t_trace,
                        evaluate_candidate=evaluate_candidate,
                        on_candidate=on_candidate,
                    )
                    row = dataclasses.asdict(result)
                    elapsed_anchor = time.time() - anchor_t0
                    row.update(
                        {
                            "anchor_key": key,
                            "episode_key": f"{opponent}:{eval_seed}",
                            "opponent": opponent,
                            "eval_seed": eval_seed,
                            "train_seed": int(train_seed),
                            "pressure_step": pressure_step,
                            "o3_authorized": False,
                            "latent_necessity_claim": False,
                            "candidates_searched": candidates_searched,
                            "responses_tested_total": responses_tested_total,
                            "legal_alternatives_total": legal_alts_total,
                            "short_circuit_candidate_hits": short_circuit_hits,
                            "elapsed_seconds": round(elapsed_anchor, 3),
                        }
                    )
                    append_jsonl(stage3_results_path, row)
                    completed_keys.add(key)
                    anchor_results.append(row)
                    timing_rows.append(
                        {
                            "anchor_key": key,
                            "elapsed_seconds": elapsed_anchor,
                            "candidates_searched": candidates_searched,
                            "responses_tested_total": responses_tested_total,
                            "legal_alternatives_total": legal_alts_total,
                            "short_circuit_candidate_hits": short_circuit_hits,
                            "episode_status": row["episode_status"],
                        }
                    )
                    n_done += 1
                    progress.log(
                        f"[STAGE3] policy={train_seed} anchor={n_done}/{n_total} "
                        f"status={row['episode_status']} "
                        f"candidates={candidates_searched} "
                        f"responses_tested={responses_tested_total}/"
                        f"{max(legal_alts_total, responses_tested_total)} "
                        f"elapsed={elapsed_anchor:.1f}s"
                    )
                    progress.heartbeat(
                        done=n_done,
                        total=max(n_total, 1),
                        phase="STAGE3",
                        detail=(
                            f"policy={train_seed} anchor={n_done}/{n_total} "
                            f"fork_found={row['episode_status'] == QUALIFIED_COMMITMENT_FORK} "
                            f"elapsed={round(time.time() - stage3_started, 1)}s"
                        ),
                        policy=train_seed,
                        anchor_index=n_done,
                        anchors_total=n_total,
                        fork_found=row["episode_status"] == QUALIFIED_COMMITMENT_FORK,
                    )
            finally:
                env.close()
    finally:
        if hasattr(model, "train"):
            model.train(was_training)

    if stage3_results_path.exists():
        disk_rows = [
            row
            for row in read_jsonl(stage3_results_path)
            if int(row.get("train_seed", -1)) == int(train_seed)
        ]
        seen = {row["anchor_key"] for row in anchor_results}
        for row in disk_rows:
            if row.get("anchor_key") not in seen:
                anchor_results.append(row)

    qualified = [
        result
        for result in anchor_results
        if result["episode_status"] == QUALIFIED_COMMITMENT_FORK
    ]
    n_anchors = len(anchor_results)
    fork_rate = len(qualified) / n_anchors if n_anchors else 0.0
    return {
        "science_scope": CONTROLLABILITY_SCOPE,
        "n_pressure_anchors": n_anchors,
        "n_qualified_commitment_forks": len(qualified),
        "n_no_commitment_fork": sum(
            result["episode_status"] == NO_COMMITMENT_FORK
            for result in anchor_results
        ),
        "n_skipped_resume": n_skipped_resume,
        "fork_rate": fork_rate,
        "minimum_fork_rate": contract.minimum_fork_rate,
        "clears_frozen_minimum_fork_rate": fork_rate >= contract.minimum_fork_rate,
        "o3_authorized": False,
        "latent_necessity_claim": False,
        "short_circuit": bool(short_circuit),
        "anchor_results": anchor_results,
        "determinism_checks": determinism_checks,
        "timing_rows": timing_rows,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        completed = subprocess.run(
            ["git", "log", "-1", "--format=%H"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return (completed.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def _require_c3_execution_authorization() -> dict:
    """Fail closed before any environment, checkpoint, or rollout work."""
    if not AUTH_PATH.exists():
        raise SystemExit(
            "C3 is DRAFT / NOT FROZEN. Execution is prohibited until "
            f"{AUTH_PATH.relative_to(PROJECT_ROOT)} exists with "
            "status=FROZEN_AND_AUTHORIZED and matching "
            "c3_contract_hash / c3_prereg_commit / runner_commit."
        )
    try:
        auth = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"C3 authorization artifact unreadable: {AUTH_PATH}: {exc}") from exc
    if auth.get("status") != "FROZEN_AND_AUTHORIZED":
        raise SystemExit(
            f"C3 execution refused: authorization status={auth.get('status')!r} "
            "(required 'FROZEN_AND_AUTHORIZED')"
        )
    if not CONTRACT_PATH.exists():
        raise SystemExit(
            "C3 execution refused: frozen machine-readable contract missing at "
            f"{CONTRACT_PATH}"
        )

    contract_hash = _sha256_file(CONTRACT_PATH)
    if str(auth.get("c3_contract_hash") or "") != contract_hash:
        raise SystemExit("C3 execution refused: c3_contract_hash mismatch")
    head = _git_head()
    if str(auth.get("c3_prereg_commit") or "") != head:
        raise SystemExit("C3 execution refused: c3_prereg_commit mismatch")
    if str(auth.get("runner_commit") or "") != head:
        raise SystemExit("C3 execution refused: runner_commit mismatch")
    if not PREREG_PATH.exists():
        raise SystemExit(f"C3 execution refused: prereg missing at {PREREG_PATH}")
    expected_prereg_sha = auth.get("c3_prereg_sha256")
    if expected_prereg_sha and str(expected_prereg_sha) != _sha256_file(PREREG_PATH):
        raise SystemExit("C3 execution refused: c3_prereg_sha256 mismatch")
    return auth


def _build_benchmark_report(
    *,
    stage3_by_seed: dict,
    n_episodes: int,
    n_policies: int,
    n_opponents: int,
    full_episodes_per_cell: int = 30,
) -> dict:
    timing = []
    for stage3 in stage3_by_seed.values():
        timing.extend(stage3.get("timing_rows") or [])
    n_anchors = sum(int(stage3.get("n_pressure_anchors") or 0) for stage3 in stage3_by_seed.values())
    n_forks = sum(
        int(stage3.get("n_qualified_commitment_forks") or 0) for stage3 in stage3_by_seed.values()
    )
    if not timing:
        return {
            "pressure_anchors": n_anchors,
            "forks_found": n_forks,
            "note": "no Stage-3 timing rows",
        }
    mean_candidates = sum(float(row["candidates_searched"]) for row in timing) / len(timing)
    mean_responses = sum(float(row["responses_tested_total"]) for row in timing) / len(timing)
    mean_legal = sum(float(row["legal_alternatives_total"]) for row in timing) / len(timing)
    mean_seconds = sum(float(row["elapsed_seconds"]) for row in timing) / len(timing)
    mean_sc_hits = sum(float(row["short_circuit_candidate_hits"]) for row in timing) / len(timing)
    responses_saved = max(0.0, mean_legal - mean_responses)
    savings_frac = (responses_saved / mean_legal) if mean_legal > 0 else 0.0
    anchors_per_episode = n_anchors / max(n_episodes * n_policies * n_opponents, 1)
    full_jobs = n_policies * n_opponents * full_episodes_per_cell
    projected_anchors = anchors_per_episode * full_jobs
    projected_stage3_hours = (projected_anchors * mean_seconds) / 3600.0
    return {
        "pressure_anchors": n_anchors,
        "forks_found": n_forks,
        "mean_candidates_searched_per_anchor": round(mean_candidates, 3),
        "mean_legal_branches_per_candidate_proxy": round(
            mean_legal / max(mean_candidates, 1e-9), 3
        ),
        "mean_responses_tested_per_anchor": round(mean_responses, 3),
        "mean_legal_alternatives_budget_per_anchor": round(mean_legal, 3),
        "mean_short_circuit_candidate_hits": round(mean_sc_hits, 3),
        "short_circuit_response_savings_frac": round(savings_frac, 4),
        "seconds_per_anchor": round(mean_seconds, 3),
        "anchors_per_episode_cell": round(anchors_per_episode, 4),
        "projected_full_scan_anchors": round(projected_anchors, 1),
        "projected_full_stage3_wall_hours": round(projected_stage3_hours, 2),
        "projection_assumes": {
            "policies": n_policies,
            "opponents": n_opponents if n_opponents > 1 else 7,
            "episodes_per_cell": full_episodes_per_cell,
            "note": "If benchmark used 1 opponent, projection scales opponents to 7.",
        },
    }


def main() -> int:
    auth = _require_c3_execution_authorization()
    contract = _load_runtime_contract()

    from experiments.long_session_progress import LongSessionProgress, configure_stdio
    from experiments.run_g0_v2_evaluation import resolve_cnn_channels
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy

    configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(G0_SEEDS))
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="*",
        default=list(OPPONENTS),
        help="Opponent subset (default: full OP6-OP12). Smoke may use one opponent.",
    )
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--from-stage1",
        type=str,
        default="",
        help="Load frozen Stage-1 anchors/manifest from this directory and skip Stage 1.",
    )
    parser.add_argument(
        "--resume-stage3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip anchors already present in C3_STAGE3_ANCHOR_RESULTS.jsonl (default: true).",
    )
    parser.add_argument(
        "--short-circuit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Existential δ / utility-ceiling short-circuit (default: true).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Write C3_BENCHMARK_REPORT.json with Stage-3 cost extrapolation.",
    )
    args = parser.parse_args()
    opponents = tuple(args.opponents)
    if not opponents:
        raise SystemExit("--opponents must be non-empty")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = LongSessionProgress(OUT_DIR, name="C3_DISCOVERY")
    started = time.time()
    contract_hash = str(auth.get("c3_contract_hash") or _sha256_file(CONTRACT_PATH))
    progress.log("=" * 78)
    progress.log("C3 DISCOVERY - AUTHORIZED (contract hashes verified)")
    progress.log(f"science_scope={CONTROLLABILITY_SCOPE}")
    progress.log(f"authorization={AUTH_PATH}")
    progress.log(f"c3_contract_hash={contract_hash}")
    progress.log(f"runtime_contract={dataclasses.asdict(contract)}")
    progress.log(
        f"seeds={args.seeds} opponents={opponents} episodes/cell={args.episodes} "
        f"from_stage1={bool(args.from_stage1)} resume_stage3={args.resume_stage3} "
        f"short_circuit={args.short_circuit}"
    )
    progress.log("NO STRATEGY CLAIM - NO LATENT-NECESSITY CLAIM - O3 NOT AUTHORIZED HERE")
    progress.log("=" * 78)

    policies: dict[int, object] = {}
    checkpoint_meta: dict[int, dict] = {}
    rows_by_seed: dict[int, list[dict]] = {int(seed): [] for seed in args.seeds}
    all_anchors: list[dict] = []

    for seed in args.seeds:
        tag = f"g0_v5_long_seed{seed}"
        checkpoint = PROJECT_ROOT / "artifacts" / "g0_v5_long" / tag / "ckpts" / f"final_{tag}.zip"
        payload = read_checkpoint_payload(str(checkpoint), map_location="cpu")
        policies[int(seed)] = load_policy(
            str(checkpoint),
            device=args.device,
            num_cnn_channels=resolve_cnn_channels(payload, context=str(checkpoint)),
        )
        checkpoint_meta[int(seed)] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "tag": tag,
        }

    if args.from_stage1:
        stage1_dir = Path(args.from_stage1)
        progress.set_phase("STAGE1_LOAD", f"from={stage1_dir}")
        loaded_anchors, manifest = load_stage1_bundle(stage1_dir)
        if str(manifest.get("c3_contract_hash") or "") != contract_hash:
            raise SystemExit(
                "Stage-1 manifest c3_contract_hash does not match authorized contract hash"
            )
        for anchor in loaded_anchors:
            seed = int(anchor["train_seed"])
            if seed not in rows_by_seed:
                continue
            rows_by_seed[seed].append(anchor)
            all_anchors.append(anchor)
        progress.log(f"loaded_stage1_anchors={len(all_anchors)} from {stage1_dir}")
    else:
        jobs = [
            (seed, opponent, DISCOVERY_SEED_BASE + episode_i)
            for seed in args.seeds
            for opponent in opponents
            for episode_i in range(args.episodes)
        ]
        progress.set_phase("STAGE1", f"natural_pressure_anchor_episodes={len(jobs)}")
        for seed, opponent, eval_seed in progress.bar(jobs, desc="stage1_episodes", unit="ep"):
            anchors, _features = collect_pressure_anchors(
                policies[int(seed)],
                opponent=opponent,
                seed=eval_seed,
                device=args.device,
                response_horizon=contract.h_response,
            )
            for anchor in anchors:
                anchor.update(
                    {
                        "episode_key": f"{opponent}:{eval_seed}",
                        "episode_id": f"{int(seed)}:{opponent}:{int(eval_seed)}",
                        "opponent": opponent,
                        "eval_seed": int(eval_seed),
                        "train_seed": int(seed),
                        "policy": int(seed),
                        "pressure_anchor_step": int(anchor["pressure_step"]),
                        "c3_contract_hash": contract_hash,
                        "checkpoint_sha256": checkpoint_meta[int(seed)]["checkpoint_sha256"],
                        "checkpoint_path": checkpoint_meta[int(seed)]["checkpoint_path"],
                        "map": CANONICAL_MAP,
                        "ruleset": "RULESET_V2_AQUATICUS_10S",
                        "snapshot_restore": {
                            "mode": "deterministic_natural_replay",
                            "references": [
                                "train_seed/policy checkpoint",
                                "opponent",
                                "eval_seed",
                                "pressure_step as backward-trace anchor",
                            ],
                            "note": (
                                "Stage 3 reconstructs candidate snapshots by replaying "
                                "unmodified G0 on (opponent, eval_seed)."
                            ),
                        },
                    }
                )
                rows_by_seed[int(seed)].append(anchor)
                all_anchors.append(anchor)

        stage1_manifest = {
            "status": "STAGE1_FROZEN",
            "science_scope": CONTROLLABILITY_SCOPE,
            "c3_contract_hash": contract_hash,
            "runner_commit": _git_head(),
            "map": CANONICAL_MAP,
            "ruleset": "RULESET_V2_AQUATICUS_10S",
            "runtime_contract": dataclasses.asdict(contract),
            "seeds": [int(seed) for seed in args.seeds],
            "opponents": list(opponents),
            "episodes_per_cell": int(args.episodes),
            "discovery_seed_base": DISCOVERY_SEED_BASE,
            "n_anchors": len(all_anchors),
            "anchors_by_seed": {
                str(seed): len(rows_by_seed[int(seed)]) for seed in args.seeds
            },
            "checkpoints": {str(seed): checkpoint_meta[int(seed)] for seed in args.seeds},
            "stage3_reconstruction": {
                "method": "deterministic_natural_replay",
                "required_fields": [
                    "train_seed",
                    "opponent",
                    "eval_seed",
                    "pressure_step",
                    "checkpoint_sha256",
                    "c3_contract_hash",
                ],
            },
        }
        anchors_path, manifest_path = write_stage1_artifacts(
            OUT_DIR,
            anchors=all_anchors,
            manifest=stage1_manifest,
        )
        progress.log(f"persisted_stage1_anchors={anchors_path}")
        progress.log(f"persisted_stage1_manifest={manifest_path}")

    report: dict = {
        "science_scope": CONTROLLABILITY_SCOPE,
        "contract": dataclasses.asdict(contract),
        "minimum_fork_rate": contract.minimum_fork_rate,
        "stage_1": {
            "unit": "natural carrier-pressure anchor",
            "pressure_is_fork": False,
            "anchors_by_seed": {
                str(seed): len(rows_by_seed[int(seed)]) for seed in args.seeds
            },
            "persisted": True,
            "loaded_from": args.from_stage1 or str(OUT_DIR),
        },
        "stage_2": {},
        "stage_3": {},
        "qualified_commitment_forks": [],
        "o3_authorized": False,
        "latent_necessity_claim": False,
    }

    if args.stage >= 2:
        progress.set_phase("STAGE2", "bounded_backward_trace_candidates")
        report["stage_2"] = {
            "candidate_order": "chronological_earliest_first",
            "pressure_onset_promoted_by_default": False,
            "candidate_states": sum(
                len(backward_trace_steps(int(anchor["pressure_step"]), contract.t_trace))
                for anchor in all_anchors
            ),
        }

    stage3_results_path = OUT_DIR / STAGE3_RESULTS_NAME
    if args.stage >= 3:
        progress.set_phase("STAGE3", "controllability_screen_only")
        completed_keys = (
            load_completed_stage3_keys(stage3_results_path) if args.resume_stage3 else set()
        )
        if completed_keys:
            progress.log(f"stage3_resume_keys={len(completed_keys)} from {stage3_results_path}")
        for seed in progress.bar(list(args.seeds), desc="stage3_policies", unit="policy"):
            stage3 = _run_stage_3(
                policies[int(seed)],
                args.device,
                rows_by_seed[int(seed)],
                contract,
                progress=progress,
                train_seed=int(seed),
                stage3_results_path=stage3_results_path,
                completed_keys=completed_keys,
                short_circuit=bool(args.short_circuit),
            )
            report["stage_3"][str(seed)] = {
                key: value
                for key, value in stage3.items()
                if key != "timing_rows"
            }
            report["stage_3"][str(seed)]["timing_summary"] = {
                "n_timing_rows": len(stage3.get("timing_rows") or []),
                "mean_seconds_per_anchor": (
                    round(
                        sum(r["elapsed_seconds"] for r in stage3["timing_rows"])
                        / len(stage3["timing_rows"]),
                        3,
                    )
                    if stage3.get("timing_rows")
                    else None
                ),
            }
            # Keep timing for benchmark aggregation without bloating the final report.
            report["stage_3"][str(seed)]["_timing_rows"] = stage3.get("timing_rows") or []
            report["qualified_commitment_forks"].extend(
                result
                for result in stage3["anchor_results"]
                if result["episode_status"] == QUALIFIED_COMMITMENT_FORK
            )
            progress.log(
                f"[STAGE3] seed={seed} anchors={stage3['n_pressure_anchors']} "
                f"forks={stage3['n_qualified_commitment_forks']} "
                f"fork_rate={stage3['fork_rate']:.4f} "
                f"resumed_skipped={stage3['n_skipped_resume']}"
            )

    if args.benchmark and args.stage >= 3:
        stage3_for_bench = {}
        for seed, payload in report["stage_3"].items():
            stage3_for_bench[seed] = {
                "n_pressure_anchors": payload.get("n_pressure_anchors"),
                "n_qualified_commitment_forks": payload.get("n_qualified_commitment_forks"),
                "timing_rows": payload.pop("_timing_rows", []),
            }
        for payload in report["stage_3"].values():
            payload.pop("_timing_rows", None)
        bench = _build_benchmark_report(
            stage3_by_seed=stage3_for_bench,
            n_episodes=int(args.episodes),
            n_policies=len(args.seeds),
            n_opponents=len(opponents),
            full_episodes_per_cell=30,
        )
        if len(opponents) == 1 or len(args.seeds) == 1:
            opp_scale = 7 / max(len(opponents), 1)
            pol_scale = 3 / max(len(args.seeds), 1)
            scale = opp_scale * pol_scale
            bench["projected_full_scan_anchors"] = round(
                float(bench.get("projected_full_scan_anchors") or 0.0) * scale, 1
            )
            bench["projected_full_stage3_wall_hours"] = round(
                float(bench.get("projected_full_stage3_wall_hours") or 0.0) * scale, 2
            )
            bench["projection_assumes"]["opponents"] = 7
            bench["projection_assumes"]["policies"] = 3
            bench["projection_assumes"]["scale_applied"] = round(scale, 4)
        (OUT_DIR / "C3_BENCHMARK_REPORT.json").write_text(
            json.dumps(bench, indent=2, default=str, allow_nan=False),
            encoding="utf-8",
        )
        progress.log(f"benchmark={OUT_DIR / 'C3_BENCHMARK_REPORT.json'}")
        progress.log(f"benchmark_summary={bench}")
    else:
        for payload in report.get("stage_3", {}).values():
            if isinstance(payload, dict):
                payload.pop("_timing_rows", None)

    (OUT_DIR / "C3_DISCOVERY.json").write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    if all_anchors:
        with (OUT_DIR / "C3_PRESSURE_ANCHORS.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_anchors[0].keys()))
            writer.writeheader()
            writer.writerows(all_anchors)

    if args.stage >= 3 and report["qualified_commitment_forks"]:
        (OUT_DIR / "C3_QUALIFIED_COMMITMENT_FORKS.json").write_text(
            json.dumps(
                {
                    "science_scope": CONTROLLABILITY_SCOPE,
                    "o3_authorized": False,
                    "forks": report["qualified_commitment_forks"],
                },
                indent=2,
                default=str,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    elif args.stage >= 3:
        (OUT_DIR / "C3_NO_QUALIFIED_STRATEGIC_FORK.json").write_text(
            json.dumps(
                {
                    "result": "clean negative",
                    "episode_status": NO_COMMITMENT_FORK,
                    "message": "No pressure anchor had an upstream state satisfying R1-R4.",
                    "o3_authorized": False,
                    "latent_necessity_claim": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    progress.set_phase("COMPLETE", f"forks={len(report['qualified_commitment_forks'])}")
    progress.log(f"report={OUT_DIR / 'C3_DISCOVERY.json'}")
    progress.log(f"wall={round(time.time() - started, 1)}s")
    progress.log("NO C3 CONFIRMATION RUN - NO O3 AUTHORIZATION - NO LATENT CLAIM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
