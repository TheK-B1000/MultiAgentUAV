#!/usr/bin/env python3
"""Build the step-4 proof table for the Summer Plan latent comparison.

For each run_tag, this script pulls together:

    Training-time q_phi metrics  (from *_metrics.csv final row)
        zH (frac), MI(z;phase), MI(z;outcome), MI(z;opponent), switch_fraction
    Training-time WR             (from *_episodes.csv, last K=500 episodes)
        overall WR, WR vs OP3, WR vs OP5, WR vs OP6
    Held-out eval WR             (from eval_op4_zero_shot/op4_zero_shot_comparison.csv)
        OP4 WR

Outputs:
    {out_dir}/proof_table.csv     wide table, one row per run_tag
    stdout                         pretty-printed comparison table

All MI columns are stored in *nats* by the trainer; this script converts to *bits*
(divide by ln 2) for human readability.

Usage
-----

Default (compares 5 hard-pool ablations + 3 phase-aux variants if they exist):

    python experiments/build_proof_table.py

Custom run set:

    python experiments/build_proof_table.py --run-tags \\
        plan_faithful_no_latent_hardpool_1m_2v2 \\
        plan_faithful_latent_no_entropy_hardpool_1m_2v2 \\
        plan_faithful_latent_phaseaux_001_hardpool_1m_2v2
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


DEFAULT_RUN_TAGS: tuple[str, ...] = (
    "plan_faithful_no_latent_hardpool_1m_2v2",
    "plan_faithful_latent_k1_hardpool_1m_2v2",
    "plan_faithful_latent_no_persistence_hardpool_1m_2v2",
    "plan_faithful_latent_persist_entropy_hardpool_1m_2v2",
    "plan_faithful_latent_no_entropy_hardpool_1m_2v2",
    "plan_faithful_latent_phaseaux_001_hardpool_1m_2v2",
    "plan_faithful_latent_phaseaux_005_hardpool_1m_2v2",
    "plan_faithful_latent_phaseaux_010_hardpool_1m_2v2",
)

LN2 = math.log(2.0)


def _normalize_opponent_label(raw: str) -> str:
    s = str(raw).strip()
    if s.upper().startswith("SCRIPTED:"):
        s = s.split(":", 1)[1]
    return s.upper()


def _final_metrics_row(metrics_path: str) -> dict:
    if not os.path.isfile(metrics_path):
        return {}
    df = pd.read_csv(metrics_path)
    if df.empty:
        return {}
    last = df.tail(1).iloc[0]

    def _get(col: str, default: float = float("nan")) -> float:
        if col in df.columns:
            try:
                return float(last[col])
            except (TypeError, ValueError):
                return default
        return default

    return {
        "zH_frac": _get("strategy_entropy_frac"),
        "zH_nats": _get("strategy_entropy"),
        "MI_z_phase_bits": _get("latent_mi_z_phase_nats") / LN2,
        "MI_z_outcome_bits": _get("latent_mi_z_outcome_nats") / LN2,
        "MI_z_opp_bits": _get("latent_mi_z_opponent_nats") / LN2,
        "switch_fraction": _get("strategy_switch_fraction"),
        "qphi_phase_acc": _get("latent_strategy_aux_phase_acc"),
        "latent_lam_h": _get("latent_lam_h"),
        "latent_lam_p": _get("latent_lam_p"),
        "latent_K": _get("latent_strategy_n"),
    }


def _train_wr_by_opponent(episodes_path: str, *, tail: int = 500) -> dict:
    if not os.path.isfile(episodes_path):
        return {}
    df = pd.read_csv(episodes_path)
    if df.empty:
        return {}
    df = df.tail(int(tail))
    if "success" not in df.columns or "opponent" not in df.columns:
        return {}
    out: dict[str, float] = {}
    df = df.copy()
    df["__opp__"] = df["opponent"].map(_normalize_opponent_label)
    df["__win__"] = (df["success"].astype(int) == 1).astype(int)
    out["WR_train_overall"] = float(df["__win__"].mean()) if len(df) else float("nan")
    for opp, sub in df.groupby("__opp__"):
        out[f"WR_train_vs_{opp}"] = float(sub["__win__"].mean()) if len(sub) else float("nan")
    return out


def _op4_wr(comparison_path: str, run_tag: str) -> Optional[float]:
    if not os.path.isfile(comparison_path):
        return None
    df = pd.read_csv(comparison_path)
    sel = df[(df["run_tag"] == run_tag) & (df["opponent"].astype(str).str.upper() == "OP4")]
    if sel.empty:
        return None
    # Prefer the largest-N row (more episodes => tighter estimate)
    sel = sel.sort_values("episodes", ascending=False)
    return float(sel.iloc[0]["win_rate"])


def build_row(*, run_tag: str, checkpoint_dir: str) -> dict:
    metrics_path = os.path.join(checkpoint_dir, f"{run_tag}_metrics.csv")
    episodes_path = os.path.join(checkpoint_dir, f"{run_tag}_episodes.csv")
    op4_path = os.path.join(checkpoint_dir, "eval_op4_zero_shot", "op4_zero_shot_comparison.csv")

    row: dict = {"run_tag": run_tag}
    row.update(_final_metrics_row(metrics_path))
    row.update(_train_wr_by_opponent(episodes_path))
    op4 = _op4_wr(op4_path, run_tag)
    if op4 is not None:
        row["WR_OP4_holdout"] = op4
    return row


def _fmt_pct(v: float) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "   -  "
    return f"{100.0 * float(v):5.1f}%"


def _fmt_bits(v: float) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "  -   "
    return f"{float(v):+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join("checkpoints", "2v2"),
        help="Directory containing *_metrics.csv, *_episodes.csv, and eval_op4_zero_shot/.",
    )
    parser.add_argument(
        "--run-tags",
        nargs="+",
        default=None,
        help=f"Explicit run_tag list (default: {len(DEFAULT_RUN_TAGS)} hard-pool + phase-aux tags).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output dir for proof_table.csv (default: <checkpoint-dir>/analysis/).",
    )
    parser.add_argument(
        "--tail-episodes",
        type=int,
        default=500,
        help="Use the last N episodes for training-time WR (default: 500).",
    )
    args = parser.parse_args()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(checkpoint_dir, "analysis"))
    os.makedirs(out_dir, exist_ok=True)

    run_tags = args.run_tags or list(DEFAULT_RUN_TAGS)

    rows: list[dict] = []
    for tag in run_tags:
        row = build_row(run_tag=tag, checkpoint_dir=checkpoint_dir)
        if len(row) <= 1:
            print(f"[build_proof_table] skip {tag} (no artifacts found)")
            continue
        rows.append(row)

    if not rows:
        print("[build_proof_table] no runs to report.")
        return

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "proof_table.csv")
    df.to_csv(out_path, index=False)
    print(f"[build_proof_table] wrote {out_path}\n")

    # Pretty print: WR table first, then meaning table
    print("=" * 110)
    print("PROOF TABLE 1/2 — Win rates (training tail vs held-out OP4)")
    print("=" * 110)
    header = (
        f"  {'run_tag':<58}  {'WR_all':>7}  {'OP3':>7}  {'OP5':>7}  {'OP6':>7}  {'OP4_ho':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in df.iterrows():
        print(
            f"  {str(r['run_tag']):<58}  "
            f"{_fmt_pct(r.get('WR_train_overall')):>7}  "
            f"{_fmt_pct(r.get('WR_train_vs_OP3')):>7}  "
            f"{_fmt_pct(r.get('WR_train_vs_OP5')):>7}  "
            f"{_fmt_pct(r.get('WR_train_vs_OP6')):>7}  "
            f"{_fmt_pct(r.get('WR_OP4_holdout')):>7}"
        )

    print("\n" + "=" * 110)
    print("PROOF TABLE 2/2 — Latent meaning (final PPO update; MI in bits)")
    print("=" * 110)
    header2 = (
        f"  {'run_tag':<58}  {'zH_frac':>7}  {'MI_phase':>9}  {'MI_outc':>9}  {'MI_opp':>9}  {'switch%':>7}"
    )
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for _, r in df.iterrows():
        zH_frac = r.get("zH_frac")
        sw = r.get("switch_fraction")
        print(
            f"  {str(r['run_tag']):<58}  "
            f"{(f'{float(zH_frac):.3f}' if zH_frac == zH_frac else '  -  '):>7}  "
            f"{_fmt_bits(r.get('MI_z_phase_bits')):>9}  "
            f"{_fmt_bits(r.get('MI_z_outcome_bits')):>9}  "
            f"{_fmt_bits(r.get('MI_z_opp_bits')):>9}  "
            f"{(f'{100*float(sw):.1f}%' if sw == sw else '  -  '):>7}"
        )

    # Highlight the success criteria for phase-aux variants
    phase_variants = df[df["run_tag"].astype(str).str.contains("phaseaux", case=False)]
    if not phase_variants.empty:
        print("\n[build_proof_table] phase-aux success criteria check (target: MI_phase > 0.01 bits, zH_frac < 0.95):")
        for _, r in phase_variants.iterrows():
            mi = r.get("MI_z_phase_bits", float("nan"))
            zh = r.get("zH_frac", float("nan"))
            ok_mi = (isinstance(mi, float) and np.isfinite(mi) and mi > 0.01)
            ok_zh = (isinstance(zh, float) and np.isfinite(zh) and zh < 0.95)
            verdict = "PASS" if (ok_mi and ok_zh) else ("partial" if (ok_mi or ok_zh) else "fail")
            print(
                f"  {r['run_tag']:<58}  MI_phase={mi:.4f} bits  zH_frac={zh:.3f}  -> {verdict}"
            )


if __name__ == "__main__":
    main()
