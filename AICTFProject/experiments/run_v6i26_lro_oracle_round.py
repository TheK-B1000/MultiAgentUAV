#!/usr/bin/env python3
"""V6I26 Stage-1 Latent Response-Oracle birth round.

Breakthrough criterion (locked):
  ΔG_available = G_after − G_before > 0

Acceptance (all required):
  G_after > G_before
  AND branch adds a nonredundant payoff row
  AND competence stays above its floor

Mixture guardrail: smoothed / opponent-aggregated regret with capped weights —
do not chase a single 4-episode accident. Final G uses cross-fitted matched-seed
forced-z evaluation on the LRO model itself.
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
    diagnose_lro_reject,
    lro_manifest,
    payoff_tensor_summary,
    select_response_target,
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
        help="Override branch z to train (default: from smoothed target).",
    )
    p.add_argument("--eval-episodes-per-cell", type=int, default=4)
    p.add_argument(
        "--competence-floor",
        type=float,
        default=None,
        help="Min mean payoff for acceptance (default: 0.75 * median init row).",
    )
    p.add_argument("--skip-post-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _mixture_cells(mixture: dict) -> list[tuple[str, str, float]]:
    cells: list[tuple[str, str, float]] = []
    for ctx, w in mixture.items():
        if "|" not in ctx:
            continue
        opp, mp = ctx.split("|", 1)
        cells.append((opp.upper(), mp, float(w)))
    return cells


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
    # Locked G: context-holdout point estimate (cross-fitted style).
    g = float(summary["G_available_point"])
    return {
        "payoff_matrix": payoff,
        "winrate_matrix": winrate,
        "contexts": contexts,
        "member_labels": labels,
        "summary": summary,
        "oracle_report": oracle,
        "G_available": g,
        "forced_z_dir": str(out_dir),
    }


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
                di = mi.get_distribution(obs)
                dt = mt.get_distribution(obs)
                logits_i = getattr(di, "logits", None)
                logits_t = getattr(dt, "logits", None)
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
        # Fall back to precomputed target (legacy).
        target = dict(scan.get("next_response_target") or {})
    else:
        if not labels:
            labels = [f"p{i}" for i in range(payoff.shape[0])]
        eps = int(scan.get("episodes_per_cell") or args.eval_episodes_per_cell)
        target = select_response_target(
            payoff,
            contexts=contexts,
            policy_labels=labels,
            episodes_per_cell=eps,
            prior_strength=float(eps),
            max_mixture_weight=0.35,
            aggregate_by_opponent=True,
        )

    branch = int(args.branch) if args.branch is not None else int(
        target.get("branch_to_train_index", 0)
    )
    # Map archive policy index → latent z slot (same K=4 indexing by default).
    branch = int(branch) % 4
    mixture = dict(target.get("mixture_weights") or {})
    cells = _mixture_cells(mixture)
    if not cells:
        print("ERROR: empty smoothed mixture")
        return 2

    # Keep only cells with material weight to avoid near-zero noise cells in PPO.
    cells_sorted = sorted(cells, key=lambda t: -t[2])
    mass = 0.0
    trimmed: list[tuple[str, str, float]] = []
    for o, m, w in cells_sorted:
        if w < 0.05 and trimmed:
            continue
        trimmed.append((o, m, w))
        mass += w
        if mass >= 0.90 and len(trimmed) >= 2:
            break
    if len(trimmed) >= 2:
        s = sum(w for _, _, w in trimmed)
        cells = [(o, m, w / s) for o, m, w in trimmed]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(args.updates) * int(args.n_envs) * int(args.n_steps)
    run_tag = f"v6i26_lro_z{branch}_r1_{args.updates}u_seed{args.seed}"

    opponents = sorted({o for o, _, _ in cells})
    maps = sorted({m for _, m, _ in cells})

    round_log: dict = {
        "experiment": "v6i26_lro_oracle_round",
        "lro": lro_manifest(),
        "landscape_scan": str(scan_path),
        "landscape_decision": decision,
        "breakthrough_criterion": "delta_G_available = G_after - G_before > 0",
        "acceptance_rule": [
            "G_after > G_before",
            "nonredundant_payoff_row",
            "competence_above_floor",
        ],
        "G_before": None,
        "targeted_regret_mixture": target,
        "branch_selected": branch,
        "branch_KL_from_initialization": None,
        "niche_payoff_improvement": None,
        "general_competence_change": None,
        "G_after": None,
        "delta_G_available": None,
        "accept_reject": None,
        "mixture_cells": [{"opponent": o, "map": m, "weight": w} for o, m, w in cells],
        "updates": int(args.updates),
        "run_tag": run_tag,
    }
    write_json(out_dir / "stage1_round_log.json", round_log)

    print("=" * 72)
    print("V6I26 Stage-1 LRO (manufacture / refine)")
    print("=" * 72)
    print(f"decision={decision}")
    print(f"branch z={branch}  updates={args.updates}  steps={steps}")
    print(f"smoothed target={target.get('target_context')} regret={target.get('target_regret')}")
    print("mixture top:", target.get("mixture_top"))
    print("acceptance: ΔG>0 AND nonredundant AND competence floor")
    if args.dry_run:
        print("[dry-run] wrote stage1_round_log.json only")
        return 0

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"ERROR: checkpoint missing: {ckpt}")
        return 2

    # --- G_before on the LRO init model (forced-z), not archive teachers ---
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
    g_before = float(before["G_available"])
    round_log["G_before"] = g_before
    round_log["G_before_detail"] = {
        "summary": before["summary"],
        "oracle_report": before.get("oracle_report"),
        "payoff_matrix": before["payoff_matrix"].tolist(),
        "contexts": before["contexts"],
        "forced_z_dir": before.get("forced_z_dir"),
    }
    write_json(out_dir / "stage1_round_log.json", round_log)
    print(f"  G_before={g_before:.4f}", flush=True)

    competence_floor = args.competence_floor
    if competence_floor is None:
        means = before["payoff_matrix"].mean(axis=1)
        competence_floor = float(0.75 * float(np.median(means)))
    round_log["competence_floor"] = float(competence_floor)

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

    print(f"[stage1] training response branch z={branch} ...", flush=True)
    train_ppo(cfg)

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

    print("[stage1] measuring G_after + KL + acceptance...", flush=True)
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
    pay_b = before["payoff_matrix"]
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

    decision_gate = accept_lro_round(
        g_before=g_before,
        g_after=g_after,
        payoff_after=pay_a,
        branch_idx=branch,
        competence_floor=float(competence_floor),
    )
    diagnosis = None
    if not decision_gate["accepted"]:
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
            "acceptance": decision_gate,
            "rejection_diagnosis": diagnosis,
            "G_after_detail": {
                "summary": after["summary"],
                "oracle_report": after.get("oracle_report"),
                "payoff_matrix": pay_a.tolist(),
                "contexts": after["contexts"],
                "forced_z_dir": after.get("forced_z_dir"),
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
    print(f"verdict={decision_gate['verdict']}")
    print(f"Wrote {out_dir / 'stage1_round_log.json'}")
    # Exit 4 = rejected this round (informative). Not a kill signal for LRO.
    return 0 if decision_gate["accepted"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
