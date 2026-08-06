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


def _run_stage_3(policy, device: str, anchors: list[dict], contract: RuntimeContract) -> dict:
    """Replay natural episodes and run the commitment-fork controllability screen."""
    from experiments.eval_v6i9_map_awareness import (
        _adapt_obs_for_policy,
        _done,
        _predict,
        _reset_obs,
        _unpack_step,
    )
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
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
    for anchor in anchors:
        by_episode[(str(anchor["opponent"]), int(anchor["eval_seed"]))].append(anchor)

    utility_fn = resolve_utility(contract.utility_name)
    anchor_results: list[dict] = []
    determinism_checks: list[dict] = []
    model = policy.model if hasattr(policy, "model") else policy
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    try:
        for (opponent, eval_seed), episode_anchors in by_episode.items():
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

                pressure_steps = sorted({int(anchor["pressure_step"]) for anchor in episode_anchors})
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

                for pressure_step in pressure_steps:
                    def evaluate_candidate(candidate_step: int):
                        if candidate_step not in snapshots:
                            raise RuntimeError(
                                f"natural replay did not capture candidate step {candidate_step} "
                                f"for pressure step {pressure_step}"
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
                        )
                        return compute_actionability(
                            branch_set,
                            delta=contract.delta,
                            doomed_utility_threshold=contract.doomed_utility_threshold,
                        )

                    result = find_earliest_commitment_fork(
                        pressure_step=pressure_step,
                        t_trace=contract.t_trace,
                        evaluate_candidate=evaluate_candidate,
                    )
                    row = dataclasses.asdict(result)
                    row.update(
                        {
                            "episode_key": f"{opponent}:{eval_seed}",
                            "opponent": opponent,
                            "eval_seed": eval_seed,
                            "o3_authorized": False,
                            "latent_necessity_claim": False,
                        }
                    )
                    anchor_results.append(row)
            finally:
                env.close()
    finally:
        if hasattr(model, "train"):
            model.train(was_training)

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
        "fork_rate": fork_rate,
        "minimum_fork_rate": contract.minimum_fork_rate,
        "clears_frozen_minimum_fork_rate": fork_rate >= contract.minimum_fork_rate,
        "o3_authorized": False,
        "latent_necessity_claim": False,
        "anchor_results": anchor_results,
        "determinism_checks": determinism_checks,
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
    args = parser.parse_args()
    opponents = tuple(args.opponents)
    if not opponents:
        raise SystemExit("--opponents must be non-empty")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = LongSessionProgress(OUT_DIR, name="C3_DISCOVERY")
    started = time.time()
    progress.log("=" * 78)
    progress.log("C3 DISCOVERY - AUTHORIZED (contract hashes verified)")
    progress.log(f"science_scope={CONTROLLABILITY_SCOPE}")
    progress.log(f"authorization={AUTH_PATH}")
    progress.log(f"c3_contract_hash={auth.get('c3_contract_hash')}")
    progress.log(f"runtime_contract={dataclasses.asdict(contract)}")
    progress.log(f"seeds={args.seeds} opponents={opponents} episodes/cell={args.episodes}")
    progress.log("NO STRATEGY CLAIM - NO LATENT-NECESSITY CLAIM - O3 NOT AUTHORIZED HERE")
    progress.log("=" * 78)

    policies: dict[int, object] = {}
    rows_by_seed: dict[int, list[dict]] = {int(seed): [] for seed in args.seeds}
    all_anchors: list[dict] = []
    jobs = [
        (seed, opponent, DISCOVERY_SEED_BASE + episode_i)
        for seed in args.seeds
        for opponent in opponents
        for episode_i in range(args.episodes)
    ]

    progress.set_phase("STAGE1", f"natural_pressure_anchor_episodes={len(jobs)}")
    for seed in args.seeds:
        tag = f"g0_v5_long_seed{seed}"
        checkpoint = PROJECT_ROOT / "artifacts" / "g0_v5_long" / tag / "ckpts" / f"final_{tag}.zip"
        payload = read_checkpoint_payload(str(checkpoint), map_location="cpu")
        policies[int(seed)] = load_policy(
            str(checkpoint),
            device=args.device,
            num_cnn_channels=resolve_cnn_channels(payload, context=str(checkpoint)),
        )

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
                    "opponent": opponent,
                    "eval_seed": eval_seed,
                    "train_seed": int(seed),
                }
            )
            rows_by_seed[int(seed)].append(anchor)
            all_anchors.append(anchor)

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

    if args.stage >= 3:
        progress.set_phase("STAGE3", "controllability_screen_only")
        for seed in progress.bar(list(args.seeds), desc="stage3_policies", unit="policy"):
            stage3 = _run_stage_3(
                policies[int(seed)],
                args.device,
                rows_by_seed[int(seed)],
                contract,
            )
            report["stage_3"][str(seed)] = stage3
            report["qualified_commitment_forks"].extend(
                result
                for result in stage3["anchor_results"]
                if result["episode_status"] == QUALIFIED_COMMITMENT_FORK
            )
            progress.log(
                f"[STAGE3] seed={seed} anchors={stage3['n_pressure_anchors']} "
                f"forks={stage3['n_qualified_commitment_forks']} "
                f"fork_rate={stage3['fork_rate']:.4f}"
            )

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
