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
        "z_wr_spread": _get("strategy_wr_spread"),
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


def _eval_wr(comparison_path: str, run_tag: str, opponent: str) -> Optional[float]:
    if not os.path.isfile(comparison_path):
        return None
    try:
        df = pd.read_csv(comparison_path)
        opp_upper = opponent.upper()
        # Handle variations in opponent naming (e.g. OP5 vs OP5_RUSHER)
        def _match_opp(val: str) -> bool:
            v = str(val).strip().upper().replace("_RUSHER", "").replace("_TURTLE", "")
            o = opp_upper.replace("_RUSHER", "").replace("_TURTLE", "")
            return v == o
        
        sel = df[(df["run_tag"] == run_tag) & (df["opponent"].map(_match_opp))]
        if sel.empty:
            return None
        # Prefer the largest-N row (more episodes => tighter estimate)
        sel = sel.sort_values("episodes", ascending=False)
        return float(sel.iloc[0]["win_rate"])
    except Exception:
        return None


def build_row(*, run_tag: str, checkpoint_dir: str) -> dict:
    metrics_path = os.path.join(checkpoint_dir, f"{run_tag}_metrics.csv")
    episodes_path = os.path.join(checkpoint_dir, f"{run_tag}_episodes.csv")
    op4_path = os.path.join(checkpoint_dir, "eval_op4_zero_shot", "op4_zero_shot_comparison.csv")

    row: dict = {"run_tag": run_tag}
    row.update(_final_metrics_row(metrics_path))
    row.update(_train_wr_by_opponent(episodes_path))
    
    # Extract evaluation win rates
    for opp in ["OP3", "OP5", "OP6", "OP4"]:
        val = _eval_wr(op4_path, run_tag, opp)
        if val is not None:
            row[f"WR_eval_{opp}"] = val
        else:
            row[f"WR_eval_{opp}"] = float("nan")
    return row


def _fmt_pct(v: float) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "  -  "
    return f"{100.0 * float(v):4.1f}%"


def _fmt_bits(v: float) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "  -   "
    return f"{float(v):+.4f}"


def compute_verdict(row: dict, baseline_row: Optional[dict]) -> str:
    run_tag = str(row.get("run_tag", ""))
    if "no_latent" in run_tag:
        return "baseline"
    if "latent_k1" in run_tag:
        return "sanity"
    
    mi_phase = row.get("MI_z_phase_bits", float("nan"))
    zh_frac = row.get("zH_frac", float("nan"))
    z_spread = row.get("z_wr_spread", float("nan"))
    
    op4_wr = row.get("WR_eval_OP4", float("nan"))
    op3_wr = row.get("WR_eval_OP3", float("nan"))
    op5_wr = row.get("WR_eval_OP5", float("nan"))
    op6_wr = row.get("WR_eval_OP6", float("nan"))
    
    # 1. Latent meaningfulness checks
    ok_mi = (isinstance(mi_phase, float) and np.isfinite(mi_phase) and mi_phase > 0.01)
    ok_zh = (isinstance(zh_frac, float) and np.isfinite(zh_frac) and 0.40 <= zh_frac <= 0.95)
    ok_spread = (isinstance(z_spread, float) and np.isfinite(z_spread) and z_spread > 0.05)
    
    # 2. Performance checks
    ok_perf = True
    reasons = []
    
    if baseline_row is not None:
        base_op4 = baseline_row.get("WR_eval_OP4", float("nan"))
        base_op3 = baseline_row.get("WR_eval_OP3", float("nan"))
        base_op5 = baseline_row.get("WR_eval_OP5", float("nan"))
        base_op6 = baseline_row.get("WR_eval_OP6", float("nan"))
        
        # OP4 holdout >= baseline OP4 - 2 percentage points
        if isinstance(op4_wr, float) and np.isfinite(op4_wr) and isinstance(base_op4, float) and np.isfinite(base_op4):
            if op4_wr < base_op4 - 0.02:
                ok_perf = False
                reasons.append("OP4 drop")
        # Seen opponents not worse by > 3 percentage points
        for name, wr, base_wr in [("OP3", op3_wr, base_op3), ("OP5", op5_wr, base_op5), ("OP6", op6_wr, base_op6)]:
            if isinstance(wr, float) and np.isfinite(wr) and isinstance(base_wr, float) and np.isfinite(base_wr):
                if wr < base_wr - 0.03:
                    ok_perf = False
                    reasons.append(f"{name} drop")
                    
    if ok_mi and ok_zh and ok_spread and ok_perf:
        return "PASS"
    elif not ok_mi and not ok_zh and not ok_spread:
        return "FAIL"
    else:
        parts = []
        if not ok_mi: parts.append("low MI")
        if not ok_zh: parts.append("collapse")
        if not ok_spread: parts.append("low spread")
        parts.extend(reasons)
        return "PARTIAL (" + ",".join(parts) + ")"


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

    # Find baseline row
    baseline_row = next((r for r in rows if "no_latent" in str(r.get("run_tag", ""))), None)

    # Unified pretty-print table
    print("=" * 145)
    print("UNIFIED LATENT EVALUATION PROOF TABLE")
    print("=" * 145)
    header = (
        f"  {'Run (tag)':<46}  {'WR OP3':>7}  {'WR OP5':>7}  {'WR OP6':>7}  {'WR OP4':>7}  {'zH_frac':>7}  {'MI_phase':>8}  {'MI_outc':>8}  {'MI_opp':>8}  {'z_spread':>8}  {'Verdict':<15}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in rows:
        run_tag = r.get("run_tag", "")
        short_tag = run_tag
        for pfx in ["plan_faithful_", "_hardpool_1m_2v2", "_1m_2v2"]:
            short_tag = short_tag.replace(pfx, "")
            
        verdict = compute_verdict(r, baseline_row)
        
        op3 = _fmt_pct(r.get("WR_eval_OP3"))
        op5 = _fmt_pct(r.get("WR_eval_OP5"))
        op6 = _fmt_pct(r.get("WR_eval_OP6"))
        op4 = _fmt_pct(r.get("WR_eval_OP4"))
        
        zh = r.get("zH_frac")
        zh_str = f"{float(zh):.3f}" if (isinstance(zh, float) and np.isfinite(zh)) else "  -  "
        
        mi_p = _fmt_bits(r.get("MI_z_phase_bits"))
        mi_o = _fmt_bits(r.get("MI_z_outcome_bits"))
        mi_opp = _fmt_bits(r.get("MI_z_opp_bits"))
        
        sp = r.get("z_wr_spread")
        sp_str = f"{float(sp):+.3f}" if (isinstance(sp, float) and np.isfinite(sp)) else "  -  "
        
        print(
            f"  {short_tag:<46}  "
            f"{op3:>7}  "
            f"{op5:>7}  "
            f"{op6:>7}  "
            f"{op4:>7}  "
            f"{zh_str:>7}  "
            f"{mi_p:>8}  "
            f"{mi_o:>8}  "
            f"{mi_opp:>8}  "
            f"{sp_str:>8}  "
            f"{verdict:<15}"
        )
    print("=" * 145)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
