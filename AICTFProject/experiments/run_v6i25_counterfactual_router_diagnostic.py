#!/usr/bin/env python3
"""V6I25 counterfactual geometry→z router diagnostic (corrected protocol).

Pipeline
--------
1. Load V6I23 donor; freeze actor / z-paths; reinitialize ``q_phi`` fresh.
2. Collect matched-seed all-z rollouts (OP8–OP12 × both maps); capture real
   episode-start ``global_state`` (no opponent ID in router input).
3. Stage A: cross-fitted geometry oracle vs best-fixed on held-out seeds.
   Stop with ``FAIL_SIGNAL`` if paired CI does not exclude zero.
4. Stage B: train ``q_phi`` with soft Q-targets ``softmax(Q̂/τ)`` from train
   means; evaluate router vs best-fixed / uniform / context-oracle.
5. Fresh online rollouts on unused seeds.

Verdicts: PASS / PARTIAL / FAIL_SIGNAL / FAIL_ROUTER
(see ``rl.router.counterfactual_router.decide_v6i25_verdict``).

Example
-------
::

    uv run python experiments/run_v6i25_counterfactual_router_diagnostic.py \\
        --checkpoint artifacts/v6i23_population_birth_5u_seed1/final_v6i23_population_birth_5u_seed1_2v2.zip \\
        --output-dir artifacts/v6i25_cf_router_smoke_seed1 \\
        --episodes-per-cell 8 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.forced_z_eval.protocol import (  # noqa: E402
    ForcedZProtocol,
    DEFAULT_BASE_SEED,
    DEFAULT_LATENTS,
    DEFAULT_MAX_DECISION_STEPS,
)
from experiments.v6i24_population_config import DEFAULT_MAPS, DEFAULT_OPPONENTS  # noqa: E402
from rl.router.counterfactual_router import (  # noqa: E402
    assert_valid_geometry_context,
    assign_cross_fitted_z,
    build_geometry_q_table,
    decide_v6i25_verdict,
    freeze_non_router_parameters,
    geometry_context_report,
    paired_delta_ci,
    predict_router_z,
    prepare_v6i7_episode_start_context,
    reinitialize_q_phi,
    soft_targets_from_geometry_q,
    stage_a_signal_validation,
    stage_b_router_eval,
    train_counterfactual_router,
    train_test_split_indices,
)

REWARD_KEY_CHOICES = ("success", "return", "win_margin")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I25 counterfactual router diagnostic")
    p.add_argument("--checkpoint", required=True, help="V6I23 donor zip")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    p.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    p.add_argument("--latents", nargs="+", type=int, default=list(DEFAULT_LATENTS))
    p.add_argument("--episodes-per-cell", type=int, default=32)
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-decision-steps", type=int, default=DEFAULT_MAX_DECISION_STEPS)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--spread-floor", type=float, default=0.05)
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--geometry-decimals", type=int, default=4)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--gap-recovery-threshold", type=float, default=0.5)
    p.add_argument(
        "--reward-key",
        choices=REWARD_KEY_CHOICES,
        default="success",
        help="Scalar R for Q̂(c,z); default success matches Stage-C WR oracle.",
    )
    p.add_argument("--online-episodes-per-cell", type=int, default=16)
    p.add_argument(
        "--online-seed-offset",
        type=int,
        default=100_000,
        help="Added to base_seed for fresh online confirmation seeds.",
    )
    p.add_argument("--skip-online", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _reward_from_episode(ep: dict[str, Any], key: str) -> float:
    if key == "success":
        return float(int(ep.get("success", 0)))
    if key == "win_margin":
        return float(ep.get("win_margin", 0))
    return float(ep.get("return", float("nan")))


def _make_env(checkpoint: str, map_name: str, seed: int, device: str, max_steps: int):
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo.inference import read_custom_ppo_metadata

    meta = read_custom_ppo_metadata(checkpoint)
    agents = int(meta.get("n_blue", 2))
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=agents,
        max_red_agents=agents,
        map_layout=map_name,
        max_decision_steps=int(max_steps),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=int(seed),
    )
    return GPUCTFVecEnv(cfg)


def _set_opponent(env: Any, opponent: str) -> None:
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
    except Exception:
        pass


def probe_episode_start_contexts(
    protocol: ForcedZProtocol,
) -> dict[tuple[str, str, int], np.ndarray]:
    """Capture real t=0 geometry for every (opponent, map, episode_index)."""
    out: dict[tuple[str, str, int], np.ndarray] = {}
    for opp_idx, opponent in enumerate(protocol.opponents):
        for map_idx, map_name in enumerate(protocol.maps):
            cell_seed = protocol.cell_seed(opp_idx, map_idx)
            env = _make_env(
                protocol.checkpoint,
                map_name,
                cell_seed,
                protocol.device,
                protocol.max_decision_steps,
            )
            try:
                for ep_idx in range(int(protocol.episodes_per_cell)):
                    ep_seed = protocol.episode_seed(cell_seed, ep_idx)
                    import random

                    random.seed(ep_seed)
                    np.random.seed(ep_seed)
                    torch.manual_seed(ep_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(ep_seed)
                    if hasattr(env, "seed"):
                        env.seed(ep_seed)
                    _set_opponent(env, opponent)
                    env.reset()
                    try:
                        raw = env.state()[0]
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(
                            f"global_state missing at episode start "
                            f"({opponent}, {map_name}, ep={ep_idx}): {exc}"
                        ) from exc
                    if raw is None:
                        raise RuntimeError(
                            f"global_state is None at episode start "
                            f"({opponent}, {map_name}, ep={ep_idx})"
                        )
                    ctx = prepare_v6i7_episode_start_context(raw)
                    out[(str(opponent), str(map_name), int(ep_idx))] = ctx
            finally:
                env.close()
    return out


def build_matched_cf_table(
    cells: dict[tuple[str, int, str], list[dict[str, Any]]],
    contexts: dict[tuple[str, str, int], np.ndarray],
    *,
    opponents: list[str],
    maps: list[str],
    latents: tuple[int, ...],
    reward_key: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Stack matched-seed R(c,z) rows with geometry context (no opponent features)."""
    rows_meta: list[dict[str, Any]] = []
    ctx_list: list[np.ndarray] = []
    ret_list: list[np.ndarray] = []
    k = len(latents)
    for opponent in opponents:
        for map_name in maps:
            ep_lists = [cells.get((opponent, int(z), map_name), []) for z in latents]
            n = min((len(eps) for eps in ep_lists), default=0)
            if n == 0:
                continue
            for ep_idx in range(n):
                key = (opponent, map_name, ep_idx)
                if key not in contexts:
                    raise KeyError(f"Missing episode-start context for {key}")
                ctx = assert_valid_geometry_context(contexts[key], name=f"context{key}")
                r = np.empty(k, dtype=np.float64)
                for j, z in enumerate(latents):
                    r[j] = _reward_from_episode(ep_lists[j][ep_idx], reward_key)
                ctx_list.append(ctx)
                ret_list.append(r)
                rows_meta.append(
                    {
                        "opponent": opponent,
                        "map": map_name,
                        "episode_index": int(ep_idx),
                        "episode_seed": int(ep_lists[0][ep_idx].get("episode_seed", -1)),
                    }
                )
    if not ctx_list:
        raise RuntimeError("No matched CF rows collected")
    contexts_arr = np.stack(ctx_list, axis=0)
    returns_arr = np.stack(ret_list, axis=0)
    report = geometry_context_report(contexts_arr)
    if int(report["n_unique_contexts"]) <= 1:
        raise RuntimeError(
            "number_of_unique_contexts <= 1: geometry has no learnable variation "
            f"(report={report})"
        )
    if float(report["context_abs_sum_mean"]) <= 0.0:
        raise RuntimeError(f"all-zero contexts: {report}")
    return contexts_arr, returns_arr, rows_meta


def _load_policy(checkpoint: str, device: str):
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.custom_ppo.inference import read_custom_ppo_metadata

    meta = read_custom_ppo_metadata(checkpoint)
    agents = int(meta.get("n_blue", 2))
    # Tiny env only to read spaces; map unused for load.
    env = GPUCTFVecEnv(
        GPUFieldConfig(
            n_envs=1,
            max_blue_agents=agents,
            max_red_agents=agents,
            map_layout=DEFAULT_MAPS[0],
            max_decision_steps=8,
            aquaticus_profile=True,
            rules_profile="OURS",
            device=device,
            seed=0,
        )
    )
    try:
        policy = load_custom_ppo_policy(
            checkpoint, env.observation_space, env.action_space, device=device
        )
    finally:
        env.close()
    return policy


def _run_condition_online(
    policy,
    *,
    checkpoint: str,
    opponents: list[str],
    maps: list[str],
    episodes_per_cell: int,
    base_seed: int,
    device: str,
    max_steps: int,
    mode: str,
    fixed_z: int | None = None,
) -> np.ndarray:
    """Fresh-seed online returns (success) under router / fixed / uniform."""
    from plot.eval_rollout import run_eval_episodes

    model = getattr(policy, "model", policy)
    rewards: list[float] = []
    for opp_idx, opponent in enumerate(opponents):
        for map_idx, map_name in enumerate(maps):
            cell_seed = int(base_seed) + 1000 * opp_idx + 100 * map_idx
            env = _make_env(checkpoint, map_name, cell_seed, device, max_steps)
            try:
                if hasattr(policy, "fixed_latent_strategy"):
                    if mode == "fixed":
                        policy.fixed_latent_strategy = True
                        policy.fixed_latent_strategy_id = int(fixed_z or 0)
                    else:
                        policy.fixed_latent_strategy = False
                if mode == "uniform" and hasattr(policy, "set_latent_eval_mode"):
                    policy.set_latent_eval_mode("uniform_random", seed=cell_seed)
                elif mode == "router" and hasattr(policy, "set_latent_eval_mode"):
                    policy.set_latent_eval_mode("normal", seed=cell_seed)
                eps = run_eval_episodes(
                    checkpoint,
                    env,
                    int(episodes_per_cell),
                    device,
                    opponent,
                    fixed_latent_id=int(fixed_z) if mode == "fixed" else None,
                    deterministic=True,
                    latent_eval_seed=cell_seed,
                    preloaded_model=policy,
                    progress_every=0,
                )
                for ep in eps:
                    rewards.append(_reward_from_episode(ep, "success"))
            finally:
                env.close()
    _ = model  # silence unused when policy wrapper only
    return np.asarray(rewards, dtype=np.float64)


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    summary_path = out_dir / "summary.json"
    if summary_path.is_file() and not args.force:
        print(f"[v6i25] {summary_path} exists; pass --force to overwrite")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    opponents = [str(o).upper() for o in args.opponents]
    maps = [str(m) for m in args.maps]
    latents = tuple(int(z) for z in args.latents)
    ckpt = str(Path(args.checkpoint).resolve())
    if not Path(ckpt).is_file():
        raise FileNotFoundError(ckpt)

    protocol = ForcedZProtocol(
        checkpoint=ckpt,
        opponents=tuple(opponents),
        maps=tuple(maps),
        latents=latents,
        episodes_per_cell=int(args.episodes_per_cell),
        base_seed=int(args.base_seed),
        max_decision_steps=int(args.max_decision_steps),
        device=str(args.device),
        collect_behavior_mean=False,
        progress_every=0 if args.quiet else 25,
    )

    if not args.quiet:
        print("[v6i25] Stage 0: probe episode-start global_state")
    contexts = probe_episode_start_contexts(protocol)
    ctx_arr_probe = np.stack(list(contexts.values()), axis=0)
    geom_report = geometry_context_report(ctx_arr_probe)
    if not args.quiet:
        print(f"[v6i25] geometry report: {json.dumps(geom_report)}")

    if not args.quiet:
        print("[v6i25] Stage 0b: matched-seed forced-z collection")
    from experiments.forced_z_eval.runner import run_forced_z_episodes

    cells = run_forced_z_episodes(protocol, quiet=bool(args.quiet))
    contexts_arr, returns_arr, rows_meta = build_matched_cf_table(
        cells,
        contexts,
        opponents=opponents,
        maps=maps,
        latents=latents,
        reward_key=str(args.reward_key),
    )
    geom_report = geometry_context_report(contexts_arr)

    n = contexts_arr.shape[0]
    train_idx, test_idx = train_test_split_indices(
        n, test_frac=float(args.test_frac), seed=int(args.base_seed)
    )
    train_ctx, train_ret = contexts_arr[train_idx], returns_arr[train_idx]
    test_ctx, test_ret = contexts_arr[test_idx], returns_arr[test_idx]

    q_table = build_geometry_q_table(
        train_ctx, train_ret, decimals=int(args.geometry_decimals)
    )
    stage_a = stage_a_signal_validation(
        test_ret,
        test_ctx,
        q_table,
        train_returns_for_best_fixed=train_ret,
        decimals=int(args.geometry_decimals),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.base_seed) + 7,
    )
    if not args.quiet:
        print(
            f"[v6i25] Stage A: ctx_oracle={stage_a.context_oracle_mean:.4f} "
            f"best_fixed(z={stage_a.best_fixed_z})={stage_a.best_fixed_mean:.4f} "
            f"delta={stage_a.delta:.4f} CI=[{stage_a.ci_low:.4f},{stage_a.ci_high:.4f}] "
            f"signal_ok={stage_a.signal_ok}"
        )

    summary: dict[str, Any] = {
        "checkpoint": ckpt,
        "opponents": opponents,
        "maps": maps,
        "latents": list(latents),
        "reward_key": str(args.reward_key),
        "n_rows": n,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "geometry_report": geom_report,
        "geometry_q_n_keys": len(q_table.q_by_key),
        "stage_a": stage_a.asdict(),
        "stage_b": None,
        "online": None,
        "verdict": "FAIL_SIGNAL",
        "train": None,
    }

    if not stage_a.signal_ok:
        summary["verdict"] = "FAIL_SIGNAL"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("[v6i25] FAIL_SIGNAL — stop before router training")
        return 2

    # --- Stage B: soft-Q train ---
    if not args.quiet:
        print("[v6i25] Stage B: load donor, reinit q_phi, soft-Q train")
    policy = _load_policy(ckpt, args.device)
    model = getattr(policy, "model", policy)
    frozen = freeze_non_router_parameters(model)
    reset = reinitialize_q_phi(model)
    if not args.quiet:
        print(f"[v6i25] frozen_non_router={len(frozen)} reinitialized={reset}")

    targets, q_rows = soft_targets_from_geometry_q(
        train_ctx,
        q_table,
        temperature=float(args.temperature),
        decimals=int(args.geometry_decimals),
    )
    train_result = train_counterfactual_router(
        model,
        torch.as_tensor(train_ctx, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
        q_values=torch.as_tensor(q_rows, dtype=torch.float32),
        n_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        spread_floor=float(args.spread_floor),
        device=args.device,
        seed=int(args.base_seed) + 11,
        loss_mode="soft_q",
    )
    summary["train"] = {
        "n_steps": train_result.n_steps,
        "loss_mean": train_result.loss_mean,
        "n_rows": train_result.n_rows,
        "n_rows_used_frac": train_result.n_rows_used,
        "loss_mode": train_result.loss_mode,
        "temperature": float(args.temperature),
        "reinitialized": reset,
        "n_frozen_params": len(frozen),
    }

    with torch.no_grad():
        router_z = (
            predict_router_z(
                model, torch.as_tensor(test_ctx, dtype=torch.float32, device=args.device)
            )
            .detach()
            .cpu()
            .numpy()
        )
    ctx_oracle_z = assign_cross_fitted_z(
        test_ctx, q_table, decimals=int(args.geometry_decimals), fallback_z=stage_a.best_fixed_z
    )
    stage_b = stage_b_router_eval(
        test_ret,
        router_z,
        context_oracle_z=ctx_oracle_z,
        best_fixed_z=stage_a.best_fixed_z,
        gap_recovery_threshold=float(args.gap_recovery_threshold),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.base_seed) + 13,
    )
    summary["stage_b"] = stage_b.asdict()
    verdict = decide_v6i25_verdict(
        stage_a, stage_b, gap_recovery_threshold=float(args.gap_recovery_threshold)
    )
    summary["verdict"] = verdict
    if not args.quiet:
        print(
            f"[v6i25] Stage B: router={stage_b.router_mean:.4f} "
            f"best_fixed={stage_b.best_fixed_mean:.4f} "
            f"ctx_oracle={stage_b.context_oracle_mean:.4f} "
            f"uniform={stage_b.uniform_mean:.4f} "
            f"recovery={stage_b.gap_recovery} "
            f"verdict={verdict}"
        )

    # Geometry-only lookup baseline = context oracle (already in stage_b).
    summary["geometry_lookup_baseline_mean"] = stage_b.context_oracle_mean

    if not args.skip_online:
        if not args.quiet:
            print("[v6i25] Stage C: fresh online confirmation seeds")
        online_base = int(args.base_seed) + int(args.online_seed_offset)
        online_router = _run_condition_online(
            policy,
            checkpoint=ckpt,
            opponents=opponents,
            maps=maps,
            episodes_per_cell=int(args.online_episodes_per_cell),
            base_seed=online_base,
            device=args.device,
            max_steps=int(args.max_decision_steps),
            mode="router",
        )
        online_fixed = _run_condition_online(
            policy,
            checkpoint=ckpt,
            opponents=opponents,
            maps=maps,
            episodes_per_cell=int(args.online_episodes_per_cell),
            base_seed=online_base,
            device=args.device,
            max_steps=int(args.max_decision_steps),
            mode="fixed",
            fixed_z=stage_a.best_fixed_z,
        )
        online_uniform = _run_condition_online(
            policy,
            checkpoint=ckpt,
            opponents=opponents,
            maps=maps,
            episodes_per_cell=int(args.online_episodes_per_cell),
            base_seed=online_base,
            device=args.device,
            max_steps=int(args.max_decision_steps),
            mode="uniform",
        )
        # Truncate to common length if any cell failed partially.
        m = min(online_router.size, online_fixed.size, online_uniform.size)
        paired = paired_delta_ci(
            online_router[:m],
            online_fixed[:m],
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.base_seed) + 17,
        )
        summary["online"] = {
            "n": m,
            "router_mean": float(online_router[:m].mean()) if m else float("nan"),
            "best_fixed_mean": float(online_fixed[:m].mean()) if m else float("nan"),
            "uniform_mean": float(online_uniform[:m].mean()) if m else float("nan"),
            "router_minus_best_fixed": asdict(paired),
            "base_seed": online_base,
        }
        if not args.quiet:
            print(
                f"[v6i25] online: router={summary['online']['router_mean']:.4f} "
                f"fixed={summary['online']['best_fixed_mean']:.4f} "
                f"uniform={summary['online']['uniform_mean']:.4f} "
                f"delta_CI=[{paired.ci_low:.4f},{paired.ci_high:.4f}]"
            )

    # Persist table for audit.
    np.savez_compressed(
        out_dir / "cf_table.npz",
        contexts=contexts_arr,
        returns=returns_arr,
        train_idx=train_idx,
        test_idx=test_idx,
        router_z_test=router_z if stage_a.signal_ok else np.array([]),
        ctx_oracle_z_test=ctx_oracle_z if stage_a.signal_ok else np.array([]),
    )
    (out_dir / "rows_meta.json").write_text(
        json.dumps(rows_meta, indent=2), encoding="utf-8"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[v6i25] verdict={summary['verdict']} wrote {summary_path}")
    return 0 if summary["verdict"] in ("PASS", "PARTIAL") else 3


if __name__ == "__main__":
    raise SystemExit(main())
