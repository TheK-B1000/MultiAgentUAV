#!/usr/bin/env python3
"""V6I26 Stage-1 Latent Response-Oracle birth round.

Screening criterion:
  delta_G_available = G_after - G_before > 0

The default 4 episodes/cell run is screening only. It may emit
PROMISING_DIRECTION, never ACCEPT.

Acceptance (all required):
  G_after > G_before
  AND CI95(delta_G) lower bound > 0
  AND branch adds a nonredundant payoff row
  AND competence stays above its floor
  AND forced-z behavior is nonredundant
  AND >=32 episodes/cell
  AND repetition across >=3 training seeds

Target selection guardrail: the archive landscape defines the evaluation
surface, but the branch and target/anchor mixture are selected from the current
forced-z payoff matrix. Do not chase a single 4-episode accident. Final G uses
cross-fitted matched-seed forced-z evaluation on the LRO model itself.
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
    accept_lro_round,
    behavior_distinctness_summary,
    cell_means_from_episode_df,
    diagnose_lro_reject,
    lro_manifest,
    payoff_tensor_summary,
    select_current_response_target,
    select_response_target,
    summarize_training_learning_signal,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I26 LRO oracle birth round")
    p.add_argument(
        "--landscape-scan",
        default="artifacts/v6i26_landscape_scan_op8_12_seed1/landscape_scan.json",
        help="Stage-0 scan JSON (payoff tensor for smoothed regret mixture).",
    )
    p.add_argument(
        "--checkpoint",
        default="artifacts/v6i23_population_birth_5u_seed1/final_v6i23_population_birth_5u_seed1_2v2.zip",
    )
    p.add_argument("--output-dir", default="artifacts/v6i26_lro_round1_seed1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument(
        "--updates",
        type=int,
        default=25,
        help="PPO updates for this BR round (25u default manufacture).",
    )
    p.add_argument(
        "--branch",
        type=int,
        default=None,
        help="Override branch z to train (default: selected from current forced-z payoff).",
    )
    p.add_argument("--eval-episodes-per-cell", type=int, default=4)
    p.add_argument(
        "--saturation-cutoff",
        type=float,
        default=0.90,
        help="Exclude current forced-z contexts with coverage at or above this payoff.",
    )
    p.add_argument(
        "--target-fraction",
        type=float,
        default=0.75,
        help="Training mixture mass assigned to uncovered target contexts.",
    )
    p.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=5,
        help="Save run-relative checkpoints every N PPO updates (0 disables).",
    )
    p.add_argument(
        "--behavior-distance-threshold",
        type=float,
        default=None,
        help="Forced-z behavior distance required for the trained branch.",
    )
    p.add_argument(
        "--competence-floor",
        type=float,
        default=None,
        help="Min mean payoff for acceptance (default: 0.75 * median init row).",
    )
    p.add_argument("--skip-post-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--reuse-forced-z-before",
        default=None,
        help=(
            "Reuse an existing forced-z out-dir (episode_results.csv) for current "
            "payoff selection / G_before instead of re-running eval."
        ),
    )
    return p.parse_args()


def _mixture_cells(mixture: dict) -> list[tuple[str, str, float]]:
    cells: list[tuple[str, str, float]] = []
    for ctx, w in mixture.items():
        if "|" not in ctx:
            continue
        opp, mp = ctx.split("|", 1)
        cells.append((opp.upper(), mp, float(w)))
    return cells


def _subset_payoff_matrix(payoff: np.ndarray, contexts: list[str], selected: list[str]) -> np.ndarray:
    index = {str(ctx): i for i, ctx in enumerate(contexts)}
    missing = [ctx for ctx in selected if str(ctx) not in index]
    if missing:
        raise ValueError(f"selected contexts missing from payoff matrix: {missing}")
    return np.asarray(payoff, dtype=np.float64)[:, [index[str(ctx)] for ctx in selected]]


def _load_forced_z_payoff_matrix(
    out_dir: Path,
    *,
    opponents: list[str],
    maps: list[str],
    latent_k: int = 4,
) -> dict:
    """Rebuild payoff/winrate matrices from an existing forced-z out-dir."""
    import pandas as pd

    csv_path = out_dir / "episode_results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing episode_results.csv under {out_dir}")
    df = pd.read_csv(csv_path)
    payoff, contexts = cell_means_from_episode_df(
        df, opponents=opponents, maps=maps, latent_k=latent_k, metric="win_margin"
    )
    winrate, _ = cell_means_from_episode_df(
        df, opponents=opponents, maps=maps, latent_k=latent_k, metric="success"
    )
    labels = [f"z{z}" for z in range(int(latent_k))]
    summary = payoff_tensor_summary(payoff, policy_labels=labels, contexts=contexts)
    oracle_path = out_dir / "oracle_report.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8")) if oracle_path.is_file() else {}
    behavior_path = out_dir / "behavior_report.json"
    behavior_report = (
        json.loads(behavior_path.read_text(encoding="utf-8")) if behavior_path.is_file() else None
    )
    return {
        "payoff_matrix": payoff,
        "winrate_matrix": winrate,
        "contexts": contexts,
        "member_labels": labels,
        "summary": summary,
        "oracle_report": oracle,
        "behavior_report": behavior_report,
        "behavior_report_path": str(behavior_path) if behavior_path.is_file() else None,
        "G_available": float(summary["G_available_point"]),
        "forced_z_dir": str(out_dir),
        "reused": True,
    }


def _forced_z_payoff_matrix(
    checkpoint: Path,
    *,
    opponents: list[str],
    maps: list[str],
    episodes_per_cell: int,
    seed: int,
    device: str,
    out_dir: Path,
    latent_k: int = 4,
    max_decision_steps: int = 240,
) -> dict:
    """Matched-seed forced-z payoff via canonical ``run_forced_z_eval``.

    G_available uses context-holdout ``payoff_tensor_summary`` (not hindsight
    per-episode oracle), matching the Stage-0 / Stage-1 ΔG contract.
    """
    import subprocess

    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "python",
        "experiments/run_forced_z_eval.py",
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(out_dir),
        "--inherit-training-config",
        "--episodes",
        str(int(episodes_per_cell)),
        "--device",
        str(device),
        "--base-seed",
        str(int(seed)),
        "--oracle-metric",
        "win_margin",
        "--max-decision-steps",
        str(int(max_decision_steps)),
        "--progress-every",
        "8",
        "--opponents",
        *opponents,
        "--maps",
        *maps,
    ]
    print("  exec:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    if rc != 0:
        raise RuntimeError(f"forced-z eval failed rc={rc}")

    csv_path = out_dir / "episode_results.csv"
    df = pd.read_csv(csv_path)
    latents = list(range(int(latent_k)))
    contexts = [f"{o}|{m}" for o in opponents for m in maps]
    payoff = np.zeros((len(latents), len(contexts)), dtype=np.float64)
    winrate = np.zeros_like(payoff)
    for zi, z in enumerate(latents):
        for ci, ctx in enumerate(contexts):
            opp, mp = ctx.split("|", 1)
            sub = df[(df["latent_z"] == z) & (df["opponent"] == opp) & (df["map"] == mp)]
            if len(sub) == 0:
                payoff[zi, ci] = float("nan")
                winrate[zi, ci] = float("nan")
            else:
                payoff[zi, ci] = float(sub["win_margin"].mean())
                winrate[zi, ci] = float(sub["success"].mean())
    labels = [f"z{z}" for z in latents]
    summary = payoff_tensor_summary(payoff, policy_labels=labels, contexts=contexts)
    oracle_path = out_dir / "oracle_report.json"
    oracle = {}
    if oracle_path.is_file():
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    behavior_path = out_dir / "behavior_report.json"
    behavior_report = None
    if behavior_path.is_file():
        behavior_report = json.loads(behavior_path.read_text(encoding="utf-8"))
    # Locked G: context-holdout point estimate (cross-fitted style).
    g = float(summary["G_available_point"])
    return {
        "payoff_matrix": payoff,
        "winrate_matrix": winrate,
        "contexts": contexts,
        "member_labels": labels,
        "summary": summary,
        "oracle_report": oracle,
        "behavior_report": behavior_report,
        "behavior_report_path": str(behavior_path) if behavior_path.is_file() else None,
        "G_available": g,
        "forced_z_dir": str(out_dir),
    }


def _distribution_logits(dist: object) -> "torch.Tensor | None":
    """Resolve concatenated action logits from a MultiHead / Categorical dist."""
    import torch

    logits = getattr(dist, "logits", None)
    if callable(logits):
        try:
            logits = logits()
        except TypeError:
            return None
    if logits is None:
        inner = getattr(dist, "distribution", None)
        if inner is not None:
            return _distribution_logits(inner)
        heads = getattr(dist, "heads", None)
        if heads is not None:
            try:
                return torch.cat([h.logits for h in heads], dim=-1)
            except Exception:  # noqa: BLE001
                return None
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(logits, (list, tuple)) and logits:
        try:
            return torch.cat(list(logits), dim=-1)
        except Exception:  # noqa: BLE001
            return None
    return None


def _branch_kl_from_init(
    init_ckpt: Path,
    trained_ckpt: Path,
    *,
    branch_z: int,
    opponents: list[str],
    maps: list[str],
    steps_per_cell: int,
    seed: int,
    device: str,
) -> float:
    """Mean action KL(init||trained) on shared states; NaN if APIs unavailable."""
    import torch

    from experiments.run_v6i24_donor_teacher_kl import _kl_cat
    from experiments.run_v6i24_population_eval_gates import (
        _collect_shared_history,
        _load_policies,
        _make_env,
        _obs_batch,
    )

    env0 = _make_env(init_ckpt, maps[0], int(seed), device, 240)
    try:
        init_pol = _load_policies(
            [(0, "init", init_ckpt)], env0.observation_space, env0.action_space, device
        )[0]["policy"]
        tr_pol = _load_policies(
            [(1, "trained", trained_ckpt)],
            env0.observation_space,
            env0.action_space,
            device,
        )[0]["policy"]
    finally:
        env0.close()

    for pol in (init_pol, tr_pol):
        model = getattr(pol, "policy", pol)
        if hasattr(model, "fixed_latent_strategy"):
            model.fixed_latent_strategy = True
            model.fixed_latent_strategy_id = int(branch_z)

    kls: list[float] = []
    for ci, (opp, mp) in enumerate((o, m) for o in opponents for m in maps):
        env = _make_env(init_ckpt, mp, int(seed) + ci, device, 240)
        try:
            snaps = _collect_shared_history(
                init_pol,
                env,
                opponent=opp,
                n_steps=int(steps_per_cell),
                seed=int(seed) + ci,
            )
            if not snaps:
                continue
            obs = _obs_batch(snaps, device)
            with torch.no_grad():
                mi = getattr(init_pol, "policy", init_pol)
                mt = getattr(tr_pol, "policy", tr_pol)
                if not hasattr(mi, "get_distribution") or not hasattr(mt, "get_distribution"):
                    continue
                # Force branch z on the observation path when supported.
                z = torch.full(
                    (int(obs["grid"].shape[0]),),
                    int(branch_z),
                    device=obs["grid"].device,
                    dtype=torch.long,
                )
                try:
                    di = mi.get_distribution(obs, z_idx=z)
                    dt = mt.get_distribution(obs, z_idx=z)
                except TypeError:
                    di = mi.get_distribution(obs)
                    dt = mt.get_distribution(obs)
                logits_i = _distribution_logits(di)
                logits_t = _distribution_logits(dt)
                if logits_i is None or logits_t is None:
                    continue
                kls.append(float(_kl_cat(logits_i, logits_t).mean().item()))
        finally:
            env.close()
    return float(np.mean(kls)) if kls else float("nan")


def main() -> int:
    args = _parse_args()
    scan_path = Path(args.landscape_scan)
    if not scan_path.is_file():
        print(f"ERROR: landscape scan missing: {scan_path}")
        return 2
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    decision = str(scan.get("decision") or "")

    payoff = np.asarray(scan.get("payoff_matrix") or [], dtype=np.float64)
    contexts = list(scan.get("contexts") or [])
    labels = list(scan.get("policy_labels") or scan.get("member_labels") or [])
    if payoff.size == 0 or not contexts:
        # Fall back to precomputed landscape surface (legacy).
        landscape_target = dict(scan.get("next_response_target") or {})
    else:
        if not labels:
            labels = [f"p{i}" for i in range(payoff.shape[0])]
        eps = int(scan.get("episodes_per_cell") or args.eval_episodes_per_cell)
        landscape_target = select_response_target(
            payoff,
            contexts=contexts,
            policy_labels=labels,
            episodes_per_cell=eps,
            prior_strength=float(eps),
            max_mixture_weight=0.35,
            aggregate_by_opponent=True,
        )

    surface_contexts = [str(ctx) for ctx in contexts]
    if not surface_contexts:
        surface_contexts = list((landscape_target.get("mixture_weights") or {}).keys())
    surface_cells = _mixture_cells({ctx: 1.0 for ctx in surface_contexts})
    if not surface_cells:
        print("ERROR: empty landscape evaluation surface")
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(args.updates) * int(args.n_envs) * int(args.n_steps)

    opponents = sorted({o for o, _, _ in surface_cells})
    maps = sorted({m for _, m, _ in surface_cells})

    round_log: dict = {
        "experiment": "v6i26_lro_oracle_round",
        "lro": lro_manifest(),
        "landscape_scan": str(scan_path),
        "landscape_decision": decision,
        "landscape_target": landscape_target,
        "selection_source": "current_forced_z_payoff_after_G_before",
        "breakthrough_criterion": "delta_G_available = G_after - G_before > 0",
        "screening_rule": [
            "G_after > G_before",
            "nonredundant_payoff_row",
            "competence_above_floor",
            "forced_z_behavior_nonredundant",
        ],
        "acceptance_rule": [
            "screening_rule passes",
            "eval_episodes_per_cell >= 32",
            "CI95(delta_G) lower bound > 0",
            "repetition across >= 3 training seeds",
        ],
        "G_before": None,
        "G_before_full_surface": None,
        "targeted_regret_mixture": None,
        "branch_selected": None,
        "branch_KL_from_initialization": None,
        "niche_payoff_improvement": None,
        "general_competence_change": None,
        "G_after": None,
        "delta_G_available": None,
        "accept_reject": None,
        "evaluation_surface_cells": [
            {"opponent": o, "map": m, "weight": w} for o, m, w in surface_cells
        ],
        "mixture_cells": [],
        "saturation_cutoff": float(args.saturation_cutoff),
        "target_fraction": float(args.target_fraction),
        "updates": int(args.updates),
        "eval_episodes_per_cell": int(args.eval_episodes_per_cell),
        "checkpoint_every_updates": int(args.checkpoint_every_updates),
        "checkpoint_diagnostics": [
            "target_cell_payoff",
            "competence",
            "behavior_distance_from_nearest_branch",
            "branch_KL_from_initialization",
            "inactive_branch_drift",
        ],
        "run_tag": None,
    }
    write_json(out_dir / "stage1_round_log.json", round_log)

    print("=" * 72)
    print("V6I26 Stage-1 LRO (manufacture / refine)")
    print("=" * 72)
    print(f"decision={decision}")
    print(f"updates={args.updates}  steps={steps}")
    print("current forced-z payoff will choose branch and target mixture after G_before")
    print("landscape mixture top:", landscape_target.get("mixture_top"))
    print(
        "screening: delta_G>0 AND nonredundant payoff row AND competence "
        "floor AND behavior distance"
    )
    print("acceptance: screening + 32 eps/cell + CI95(delta_G)>0 + >=3 training seeds")

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file() and not args.dry_run:
        print(f"ERROR: checkpoint missing: {ckpt}")
        return 2
    if args.dry_run and not ckpt.is_file() and args.reuse_forced_z_before is None:
        print("ERROR: dry-run without --reuse-forced-z-before still needs a checkpoint")
        return 2

    # --- G_before on the LRO init model (forced-z), not archive teachers ---
    reuse_dir = Path(args.reuse_forced_z_before) if args.reuse_forced_z_before else None
    if reuse_dir is not None:
        print(f"[stage1] reusing forced-z before from {reuse_dir} ...", flush=True)
        before = _load_forced_z_payoff_matrix(
            reuse_dir, opponents=opponents, maps=maps
        )
    else:
        if args.dry_run:
            print("ERROR: dry-run without --reuse-forced-z-before would require a full eval")
            return 2
        print("[stage1] measuring G_before on init checkpoint (forced-z)...", flush=True)
        before = _forced_z_payoff_matrix(
            ckpt,
            opponents=opponents,
            maps=maps,
            episodes_per_cell=int(args.eval_episodes_per_cell),
            seed=int(args.seed),
            device=str(args.device),
            out_dir=out_dir / "forced_z_before",
        )
    competence_floor = args.competence_floor
    if competence_floor is None:
        means = before["payoff_matrix"].mean(axis=1)
        competence_floor = float(0.75 * float(np.median(means)))

    # Saturation / headroom MUST use winrate (≈[0,1]), not win_margin.
    current_target = select_current_response_target(
        before["winrate_matrix"],
        contexts=before["contexts"],
        policy_labels=before["member_labels"],
        saturation_cutoff=float(args.saturation_cutoff),
        target_fraction=float(args.target_fraction),
        competence_floor=None,
    )
    current_target["selection_metric"] = "winrate_success"
    current_target["competence_floor_margin"] = float(competence_floor)
    branch = int(args.branch) if args.branch is not None else int(
        current_target["branch_to_train_index"]
    )
    branch = int(branch) % 4
    if args.branch is not None:
        current_target["branch_override"] = True
        current_target["branch_to_train_index"] = branch
        current_target["branch_to_train_label"] = f"z{branch}"
    cells = _mixture_cells(dict(current_target.get("mixture_weights") or {}))
    if not cells:
        print("ERROR: empty current-payoff response mixture")
        return 2
    train_opponents = sorted({o for o, _, _ in cells})
    train_maps = sorted({m for _, m, _ in cells})
    selected_contexts = [f"{o}|{m}" for o, m, _ in cells]
    pay_b = _subset_payoff_matrix(before["payoff_matrix"], before["contexts"], selected_contexts)
    before_selected_summary = payoff_tensor_summary(
        pay_b,
        policy_labels=before["member_labels"],
        contexts=selected_contexts,
    )
    g_before = float(before_selected_summary["G_available_point"])
    run_tag = f"v6i26_lro_z{branch}_r1_{args.updates}u_seed{args.seed}"

    target_mass = sum(
        float(w)
        for ctx, w in (current_target.get("mixture_weights") or {}).items()
        if ctx in set(current_target.get("target_contexts") or [])
    )
    anchor_mass = sum(
        float(w)
        for ctx, w in (current_target.get("mixture_weights") or {}).items()
        if ctx in set(current_target.get("anchor_contexts") or [])
    )
    target_cov = float(current_target.get("target_coverage") or 0.0)
    target_headroom = float(current_target.get("target_headroom") or 0.0)
    best_on_target = int(current_target.get("current_best_z_on_target") or -1)
    n_excluded = len(current_target.get("excluded_saturated_contexts") or [])
    n_contexts = len(before["contexts"])
    selection_gates = {
        "target_not_saturated": bool(target_cov < float(args.saturation_cutoff)),
        "branch_not_dominant_on_target": bool(branch != best_on_target),
        "branch_has_meaningful_headroom": bool(target_headroom > 0.05),
        "target_mixture_70_80": bool(0.70 <= target_mass <= 0.80),
        "anchor_mixture_20_30": bool(0.20 <= anchor_mass <= 0.30),
        "g_before_uses_selected_cells": True,
    }
    selection_gates["all_pass"] = all(selection_gates.values())

    round_log.update(
        {
            "G_before": g_before,
            "G_before_full_surface": float(before["G_available"]),
            "G_before_detail": {
                "summary": before_selected_summary,
                "full_surface_summary": before["summary"],
                "oracle_report": before.get("oracle_report"),
                "payoff_matrix": pay_b.tolist(),
                "full_surface_payoff_matrix": before["payoff_matrix"].tolist(),
                "winrate_matrix": before["winrate_matrix"].tolist(),
                "contexts": selected_contexts,
                "full_surface_contexts": before["contexts"],
                "forced_z_dir": before.get("forced_z_dir"),
                "reused": bool(before.get("reused")),
            },
            "targeted_regret_mixture": current_target,
            "branch_selected": branch,
            "mixture_cells": [
                {"opponent": o, "map": m, "weight": w} for o, m, w in cells
            ],
            "selection_gates": selection_gates,
            "target_mixture_mass": float(target_mass),
            "anchor_mixture_mass": float(anchor_mass),
            "run_tag": run_tag,
            "competence_floor": float(competence_floor),
            "dry_run": bool(args.dry_run),
        }
    )
    write_json(out_dir / "stage1_round_log.json", round_log)
    write_json(out_dir / "current_response_target.json", current_target)
    write_json(out_dir / "selection_gates.json", selection_gates)

    print(f"  G_before_selected={g_before:.4f}", flush=True)
    print(f"  G_before_full_surface={float(before['G_available']):.4f}", flush=True)
    print(
        f"  selected branch z={branch} target={current_target.get('target_context')}",
        flush=True,
    )
    print(
        f"  target_coverage={target_cov:.4f} headroom={target_headroom:.4f} "
        f"best_z_on_target={best_on_target}",
        flush=True,
    )
    print(
        f"  mixture target_mass={target_mass:.3f} anchor_mass={anchor_mass:.3f} "
        f"excluded_saturated={n_excluded}/{n_contexts}",
        flush=True,
    )
    print(f"  mixture_top={current_target.get('mixture_top')}", flush=True)
    print(f"  selection_gates={selection_gates}", flush=True)

    if args.dry_run:
        print("[dry-run] selection complete; skipping training/eval")
        return 0 if selection_gates["all_pass"] else 1

    if not selection_gates["all_pass"]:
        print("ERROR: selection gates failed — refusing to train on a saturated/ill-posed target")
        round_log["accept_reject"] = "REJECT"
        round_log["error"] = "selection_gates_failed"
        write_json(out_dir / "stage1_round_log.json", round_log)
        return 1

    opponents = train_opponents
    maps = train_maps

    from rl.config.ppo_config import PPOConfig
    from rl.presets import PRESET_REGISTRY
    from rl.train_ppo import train_ppo

    cfg = PRESET_REGISTRY["v6i26"](PPOConfig())
    cfg.seed = int(args.seed) + 17 * branch
    cfg.device = str(args.device)
    cfg.n_envs = int(args.n_envs)
    cfg.n_steps = int(args.n_steps)
    cfg.load_path = str(ckpt)
    cfg.load_weights_only = True
    cfg.additional_timesteps = steps
    cfg.checkpoint_dir = str(out_dir)
    cfg.run_tag = run_tag
    cfg.fresh_metrics_csv = True
    cfg.fixed_latent_strategy = True
    cfg.fixed_latent_strategy_id = int(branch)
    cfg.training_cell_distribution = tuple(cells)
    cfg.opponent_pool = tuple(opponents)
    cfg.freeze_return_norm_after_load = True
    if int(args.checkpoint_every_updates) > 0:
        cfg.periodic_checkpoint_steps = (
            int(args.checkpoint_every_updates) * int(args.n_envs) * int(args.n_steps)
        )
        round_log["checkpoint_step_interval"] = int(cfg.periodic_checkpoint_steps)
        round_log["checkpoint_update_labels"] = list(
            range(
                int(args.checkpoint_every_updates),
                int(args.updates) + 1,
                int(args.checkpoint_every_updates),
            )
        )
        write_json(out_dir / "stage1_round_log.json", round_log)

    print(f"[stage1] training response branch z={branch} ...", flush=True)
    train_ppo(cfg)

    metrics_candidates = sorted(out_dir.glob(f"*{run_tag}*metrics*.csv")) + sorted(
        out_dir.glob("*metrics*.csv")
    )
    if metrics_candidates:
        try:
            learning = summarize_training_learning_signal(metrics_candidates[0])
            round_log["learning_signal"] = learning
            write_json(out_dir / "learning_signal.json", learning)
            print(
                f"  learning_signal={learning.get('status')} "
                f"approx_kl_mean={learning.get('approx_kl', {}).get('mean')} "
                f"clip_mean={learning.get('clip_fraction', {}).get('mean')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            round_log["learning_signal_error"] = str(exc)

    finals = sorted(out_dir.glob(f"final_{run_tag}*.zip"))
    final_ckpt = finals[-1] if finals else None
    round_log["final_checkpoint"] = str(final_ckpt) if final_ckpt else None
    write_json(out_dir / "stage1_round_log.json", round_log)
    if final_ckpt is None:
        print("ERROR: no final checkpoint written")
        round_log["accept_reject"] = "REJECT"
        round_log["error"] = "missing_final_checkpoint"
        write_json(out_dir / "stage1_round_log.json", round_log)
        return 3

    if args.skip_post_eval:
        print("Skipping post-eval (--skip-post-eval). Re-run without it to gate.")
        return 0

    print("[stage1] measuring G_after + KL + screening...", flush=True)
    after = _forced_z_payoff_matrix(
        final_ckpt,
        opponents=opponents,
        maps=maps,
        episodes_per_cell=int(args.eval_episodes_per_cell),
        seed=int(args.seed),  # matched seeds vs before
        device=str(args.device),
        out_dir=out_dir / "forced_z_after",
    )
    g_after = float(after["G_available"])
    pay_b = _subset_payoff_matrix(before["payoff_matrix"], before["contexts"], after["contexts"])
    pay_a = after["payoff_matrix"]
    niche_imp = float(pay_a[branch].max() - pay_b[branch].max())
    competence_change = float(pay_a[branch].mean() - pay_b[branch].mean())

    try:
        kl = _branch_kl_from_init(
            ckpt,
            final_ckpt,
            branch_z=branch,
            opponents=opponents,
            maps=maps,
            steps_per_cell=32,
            seed=int(args.seed) + 7,
            device=str(args.device),
        )
    except Exception as exc:  # noqa: BLE001
        kl = float("nan")
        round_log["branch_KL_error"] = str(exc)

    behavior_distinctness = behavior_distinctness_summary(
        after.get("behavior_report"),
        branch_idx=branch,
        min_branch_distance=args.behavior_distance_threshold,
    )
    decision_gate = accept_lro_round(
        g_before=g_before,
        g_after=g_after,
        payoff_after=pay_a,
        branch_idx=branch,
        competence_floor=float(competence_floor),
        behavior_distinctness=behavior_distinctness,
        require_behavior_distinctness=True,
        episodes_per_cell=int(args.eval_episodes_per_cell),
        ci95_low_delta_g=None,
        training_seed_count=1,
    )
    diagnosis = None
    if decision_gate["verdict"] == "PROMISING_DIRECTION":
        print(
            "  screening passed but this is PROMISING_DIRECTION only; "
            "run Phase-2 confirmation before acceptance."
        )
    elif not decision_gate["accepted"]:
        if not decision_gate.get("behavior_distinctness_pass"):
            diagnosis = {
                "diagnosis_code": "BEHAVIOR_REDUNDANT",
                "meaning": (
                    "The branch did not clear forced-z behavior distance, "
                    "so the payoff change is not accepted as a distinct strategy."
                ),
                "next_action": (
                    "Broaden or retarget the response mixture; do not train the router yet."
                ),
                "signals": {
                    "branch_idx": branch,
                    "behavior_distinctness": behavior_distinctness,
                    "delta_G_available": float(g_after - g_before),
                    "niche_payoff_improvement": niche_imp,
                    "general_competence_change": competence_change,
                },
                "escalate_to_task_niches": False,
            }
        else:
            diagnosis = diagnose_lro_reject(
                branch_kl=float(kl) if kl == kl else float("nan"),
                niche_payoff_improvement=niche_imp,
                general_competence_change=competence_change,
                delta_g=float(g_after - g_before),
            )
        print(f"  diagnosis={diagnosis['diagnosis_code']}: {diagnosis['meaning']}")
        print(f"  next={diagnosis['next_action']}")
        print("  note: one reject does not kill LRO — diagnose then adjust BR.")

    round_log.update(
        {
            "G_after": g_after,
            "delta_G_available": float(g_after - g_before),
            "branch_KL_from_initialization": kl,
            "niche_payoff_improvement": niche_imp,
            "general_competence_change": competence_change,
            "accept_reject": decision_gate["verdict"],
            "behavior_distinctness": behavior_distinctness,
            "acceptance": decision_gate,
            "rejection_diagnosis": diagnosis,
            "G_after_detail": {
                "summary": after["summary"],
                "oracle_report": after.get("oracle_report"),
                "payoff_matrix": pay_a.tolist(),
                "contexts": after["contexts"],
                "forced_z_dir": after.get("forced_z_dir"),
                "behavior_report_path": after.get("behavior_report_path"),
            },
        }
    )
    write_json(out_dir / "stage1_round_log.json", round_log)
    write_json(out_dir / "acceptance.json", decision_gate)
    if diagnosis is not None:
        write_json(out_dir / "rejection_diagnosis.json", diagnosis)

    print("--- Stage-1 result ---")
    print(f"G_before={g_before:.4f}  G_after={g_after:.4f}  ΔG={g_after - g_before:.4f}")
    print(f"branch_KL≈{kl:.4f}  niche_imp={niche_imp:.4f}  competence_Δ={competence_change:.4f}")
    print(
        "behavior_nearest="
        f"{behavior_distinctness.get('branch_nearest_behavior_distance')} "
        f"pass={decision_gate.get('behavior_distinctness_pass')}"
    )
    print(f"verdict={decision_gate['verdict']}")
    print(f"Wrote {out_dir / 'stage1_round_log.json'}")
    # Exit 4 = rejected this round (informative). Screening candidates exit 0
    # because the next action is confirmation, not failure handling.
    return 0 if decision_gate["verdict"] in {"ACCEPT", "PROMISING_DIRECTION"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
