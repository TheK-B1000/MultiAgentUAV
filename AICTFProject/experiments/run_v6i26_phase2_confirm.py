#!/usr/bin/env python3
"""V6I26 Phase-2 confirm: large forced-z ΔG + bootstrap CI (seed 1).

Re-evaluates init and post-LRO checkpoints at matched seeds with many episodes
per cell, then reports bootstrap CI95(ΔG) plus the forced-z behavior strategy
gate. Numeric promotion remains CI lower bound > 0; strategy promotion requires
both numeric promotion and behavior nonredundancy.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v6i26_lro_core import (  # noqa: E402
    behavior_distinctness_summary,
    payoff_tensor_summary,
    write_json,
)
from gpu_env._core._bt_profiles import LRO_AUDITED_OPPONENT_POOL  # noqa: E402

DEFAULT_MAPS = ("map_b_split_lane", "map_b_split_lane_v2")
DEFAULT_INIT = (
    "artifacts/v6i23_population_birth_5u_seed1/"
    "final_v6i23_population_birth_5u_seed1_2v2.zip"
)
DEFAULT_FINAL = (
    "artifacts/v6i26_lro_niches_round1_seed1/"
    "final_v6i26_lro_z3_r1_25u_seed1.zip"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I26 Phase-2 large-eval ΔG confirm")
    p.add_argument("--init-checkpoint", default=DEFAULT_INIT)
    p.add_argument("--final-checkpoint", default=DEFAULT_FINAL)
    p.add_argument(
        "--output-dir",
        default="artifacts/v6i26_lro_niches_round1_seed1/phase2_confirm",
    )
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--opponents", nargs="+", default=list(LRO_AUDITED_OPPONENT_POOL))
    p.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--skip-eval", action="store_true", help="Only analyze existing CSVs")
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument(
        "--candidate-branch",
        type=int,
        default=None,
        help="Candidate z branch to test for behavior distinctness.",
    )
    p.add_argument(
        "--behavior-distance-threshold",
        type=float,
        default=None,
        help="Forced-z behavior distance required for strategy acceptance.",
    )
    return p.parse_args()


def _run_forced_z(
    checkpoint: Path,
    out_dir: Path,
    *,
    opponents: list[str],
    maps: list[str],
    episodes: int,
    seed: int,
    device: str,
    max_decision_steps: int,
) -> None:
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
        str(int(episodes)),
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
    print("exec:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    if rc != 0:
        raise RuntimeError(f"forced-z eval failed rc={rc} out={out_dir}")


def _cell_means_from_df(
    df: pd.DataFrame,
    *,
    opponents: list[str],
    maps: list[str],
    latent_k: int = 4,
    metric: str = "win_margin",
) -> tuple[np.ndarray, list[str]]:
    contexts = [f"{o}|{m}" for o in opponents for m in maps]
    payoff = np.zeros((latent_k, len(contexts)), dtype=np.float64)
    for zi in range(latent_k):
        for ci, ctx in enumerate(contexts):
            opp, mp = ctx.split("|", 1)
            sub = df[(df["latent_z"] == zi) & (df["opponent"] == opp) & (df["map"] == mp)]
            if len(sub) == 0:
                payoff[zi, ci] = float("nan")
            else:
                payoff[zi, ci] = float(sub[metric].mean())
    return payoff, contexts


def _g_from_payoff(payoff: np.ndarray, contexts: list[str]) -> float:
    labels = [f"z{i}" for i in range(payoff.shape[0])]
    summary = payoff_tensor_summary(payoff, policy_labels=labels, contexts=contexts)
    return float(summary["G_available_point"])


def _infer_branch_idx(path: str | Path) -> int | None:
    match = re.search(r"(?:^|_)z(\d+)(?:_|\.|$)", Path(path).name)
    return int(match.group(1)) if match else None


def _read_json_if_present(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_delta_g(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    opponents: list[str],
    maps: list[str],
    n_boot: int,
    seed: int,
    latent_k: int = 4,
) -> dict:
    """Episode-resample bootstrap of ΔG within each (opp, map, z) cell."""
    rng = np.random.default_rng(int(seed) + 17)
    contexts = [f"{o}|{m}" for o in opponents for m in maps]
    cells: list[tuple[int, str, str]] = []
    before_vals: list[np.ndarray] = []
    after_vals: list[np.ndarray] = []
    for z in range(latent_k):
        for opp in opponents:
            for mp in maps:
                b = df_before[
                    (df_before["latent_z"] == z)
                    & (df_before["opponent"] == opp)
                    & (df_before["map"] == mp)
                ]["win_margin"].to_numpy(dtype=np.float64)
                a = df_after[
                    (df_after["latent_z"] == z)
                    & (df_after["opponent"] == opp)
                    & (df_after["map"] == mp)
                ]["win_margin"].to_numpy(dtype=np.float64)
                if b.size == 0 or a.size == 0:
                    continue
                cells.append((z, opp, mp))
                before_vals.append(b)
                after_vals.append(a)

    def _payoff_from_means(means: dict[tuple[int, str, str], float]) -> np.ndarray:
        p = np.full((latent_k, len(contexts)), np.nan, dtype=np.float64)
        for zi in range(latent_k):
            for ci, ctx in enumerate(contexts):
                opp, mp = ctx.split("|", 1)
                p[zi, ci] = means.get((zi, opp, mp), float("nan"))
        return p

    point_before, _ = _cell_means_from_df(
        df_before, opponents=opponents, maps=maps, latent_k=latent_k
    )
    point_after, _ = _cell_means_from_df(
        df_after, opponents=opponents, maps=maps, latent_k=latent_k
    )
    g_b = _g_from_payoff(point_before, contexts)
    g_a = _g_from_payoff(point_after, contexts)
    delta_point = float(g_a - g_b)

    deltas = np.empty(int(n_boot), dtype=np.float64)
    for bi in range(int(n_boot)):
        means_b: dict[tuple[int, str, str], float] = {}
        means_a: dict[tuple[int, str, str], float] = {}
        for (z, opp, mp), b, a in zip(cells, before_vals, after_vals):
            ib = rng.integers(0, b.size, size=b.size)
            ia = rng.integers(0, a.size, size=a.size)
            means_b[(z, opp, mp)] = float(b[ib].mean())
            means_a[(z, opp, mp)] = float(a[ia].mean())
        pb = _payoff_from_means(means_b)
        pa = _payoff_from_means(means_a)
        # Drop all-NaN columns if any cell missing
        if np.isnan(pb).any() or np.isnan(pa).any():
            # fill NaN with column mean of available z (rare)
            for p in (pb, pa):
                for ci in range(p.shape[1]):
                    col = p[:, ci]
                    if np.isnan(col).any():
                        fill = float(np.nanmean(col)) if np.isfinite(col).any() else 0.0
                        col = np.where(np.isnan(col), fill, col)
                        p[:, ci] = col
        deltas[bi] = _g_from_payoff(pa, contexts) - _g_from_payoff(pb, contexts)

    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "G_before": g_b,
        "G_after": g_a,
        "delta_G": delta_point,
        "bootstrap_n": int(n_boot),
        "bootstrap_mean_delta_G": float(deltas.mean()),
        "CI95_low": float(lo),
        "CI95_high": float(hi),
        "CI95_delta_G_gt_0": bool(float(lo) > 0.0),
        "n_cells_bootstrapped": len(cells),
        "episodes_hint": {
            "before_rows": int(len(df_before)),
            "after_rows": int(len(df_after)),
        },
    }


def main() -> int:
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    before_dir = out / "forced_z_before_32"
    after_dir = out / "forced_z_after_32"
    opponents = [str(o).upper() for o in args.opponents]
    maps = list(args.maps)

    if not args.skip_eval:
        init_ck = Path(args.init_checkpoint)
        final_ck = Path(args.final_checkpoint)
        if not init_ck.is_file():
            print(f"ERROR: missing init checkpoint {init_ck}")
            return 2
        if not final_ck.is_file():
            print(f"ERROR: missing final checkpoint {final_ck}")
            return 2
        print("[phase2] evaluating INIT (G_before) at", args.episodes, "eps/cell", flush=True)
        _run_forced_z(
            init_ck,
            before_dir,
            opponents=opponents,
            maps=maps,
            episodes=int(args.episodes),
            seed=int(args.seed),
            device=str(args.device),
            max_decision_steps=int(args.max_decision_steps),
        )
        print("[phase2] evaluating FINAL (G_after) at", args.episodes, "eps/cell", flush=True)
        _run_forced_z(
            final_ck,
            after_dir,
            opponents=opponents,
            maps=maps,
            episodes=int(args.episodes),
            seed=int(args.seed),
            device=str(args.device),
            max_decision_steps=int(args.max_decision_steps),
        )

    before_csv = before_dir / "episode_results.csv"
    after_csv = after_dir / "episode_results.csv"
    if not before_csv.is_file() or not after_csv.is_file():
        print("ERROR: missing episode_results.csv — run without --skip-eval first")
        return 3

    df_b = pd.read_csv(before_csv)
    df_a = pd.read_csv(after_csv)
    report = _bootstrap_delta_g(
        df_b,
        df_a,
        opponents=opponents,
        maps=maps,
        n_boot=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    report["protocol"] = {
        "episodes_per_cell": int(args.episodes),
        "opponents": opponents,
        "maps": maps,
        "base_seed": int(args.seed),
        "init_checkpoint": str(args.init_checkpoint),
        "final_checkpoint": str(args.final_checkpoint),
        "promotion_rule": "CI95_low(delta_G) > 0",
    }
    candidate_branch = args.candidate_branch
    if candidate_branch is None:
        candidate_branch = _infer_branch_idx(args.final_checkpoint)
    report["candidate_branch"] = candidate_branch
    behavior_report = _read_json_if_present(after_dir / "behavior_report.json")
    if candidate_branch is None:
        strategy_distinctness = {
            "behavior_measurement_valid": False,
            "branch_behavior_nonredundant": False,
            "verdict": "BEHAVIOR_DISTINCT_FAIL",
            "reason": "candidate_branch_unknown",
        }
    else:
        strategy_distinctness = behavior_distinctness_summary(
            behavior_report,
            branch_idx=int(candidate_branch),
            min_branch_distance=args.behavior_distance_threshold,
        )
    report["strategy_distinctness"] = strategy_distinctness
    report["verdict"] = (
        "PHASE2_PASS" if report["CI95_delta_G_gt_0"] else "PHASE2_HOLD_OR_FAIL"
    )
    strategy_pass = bool(
        report["CI95_delta_G_gt_0"]
        and strategy_distinctness.get("branch_behavior_nonredundant")
    )
    report["phase2_strategy_verdict"] = (
        "PHASE2_STRATEGY_PASS" if strategy_pass else "PHASE2_STRATEGY_HOLD_OR_FAIL"
    )
    write_json(out / "phase2_seed1_confirm.json", report)
    print("=" * 72)
    print("Phase-2 seed-1 confirm")
    print(f"  G_before={report['G_before']:.4f}  G_after={report['G_after']:.4f}")
    print(f"  ΔG={report['delta_G']:.4f}")
    print(f"  CI95=[{report['CI95_low']:.4f}, {report['CI95_high']:.4f}]")
    print(f"  CI95>0? {report['CI95_delta_G_gt_0']}  verdict={report['verdict']}")
    print(
        "  behavior_strategy_pass? "
        f"{strategy_distinctness.get('branch_behavior_nonredundant')} "
        f"verdict={report['phase2_strategy_verdict']}"
    )
    print("Wrote", out / "phase2_seed1_confirm.json")
    return 0 if report["CI95_delta_G_gt_0"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
