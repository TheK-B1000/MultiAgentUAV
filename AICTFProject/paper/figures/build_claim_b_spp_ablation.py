"""Claim B figure: SP-PPO ablation -- valid non-recovery of specialization.

Separate from Claim A (architecture / sharing). Plots sealed

  PPO  vs  PPO+CSC  vs  PPO+CSC+SPFT

as Delta_A / Delta_B with 95% CIs and PASS/FAIL gate marks.

Mapping to sealed artifacts (methodology CSC/SPFT sections):
  PPO              = CCP-S2 CONTROL          (task PPO only)
  PPO+CSC          = CCP-S2 TREATMENT        (causal correction)
  PPO+CSC+SPFT     = RSCFT RETENTION treatment (causal + EMA retention)

Source numbers from CCP_S2_EVAL_INTEGRITY.json and RSCFT_EVAL_RESULT.json.
All three arms FAIL the specialization crossover gate.

Visual point: Share-Encoder PASS (Claim A) is NOT evidence that SP-PPO beat PPO.

Run:  python paper/figures/build_claim_b_spp_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figures.figure_style import COLORS, LINESTYLES, MARKERS, TWO_COLUMN, apply_style, save_figure

SD = ROOT / "artifacts/strategic_demand/sppo"


def _arm(label: str, da: dict, db: dict) -> dict:
    gate = (
        da["mean"] > 0 and da["lcb95"] > 0 and db["mean"] > 0 and db["lcb95"] > 0
    )
    return {
        "label": label,
        "da": da["mean"] * 100,
        "da_lo": (da["mean"] - da["lcb95"]) * 100,
        "da_hi": (da["ucb95"] - da["mean"]) * 100,
        "db": db["mean"] * 100,
        "db_lo": (db["mean"] - db["lcb95"]) * 100,
        "db_hi": (db["ucb95"] - db["mean"]) * 100,
        "gate": "PASS" if gate else "FAIL",
    }


def _load_arms() -> list[dict]:
    ccp = json.loads((SD / "CCP_S2_EVAL_INTEGRITY.json").read_text(encoding="utf-8"))
    deltas = ccp["checks"]["3_deltas_and_gammas_recomputed"] if "checks" in ccp else None
    if deltas is None:
        # tolerate alternate nesting
        def find(obj, key):
            if isinstance(obj, dict):
                if key in obj:
                    return obj[key]
                for v in obj.values():
                    got = find(v, key)
                    if got is not None:
                        return got
            return None
        block = find(ccp, "3_deltas_and_gammas_recomputed")
        assert block is not None
        deltas = block

    rscft = json.loads((SD / "RSCFT_EVAL_RESULT.json").read_text(encoding="utf-8"))
    t = rscft["PRIMARY_GATE"]["RETENTION_treatment"]

    return [
        _arm("PPO", deltas["delta_A_CONTROL"], deltas["delta_B_CONTROL"]),
        _arm("PPO+CSC", deltas["delta_A_TREATMENT"], deltas["delta_B_TREATMENT"]),
        _arm(
            "PPO+CSC+SPFT",
            t["delta_A"],
            t["delta_B"],
        ),
    ]


def main() -> dict:
    apply_style()
    arms = _load_arms()
    assert all(a["gate"] == "FAIL" for a in arms), arms

    fig, ax = plt.subplots(figsize=(TWO_COLUMN * 0.72, 2.9))
    xs = np.arange(len(arms))
    da = [a["da"] for a in arms]
    da_lo = [a["da_lo"] for a in arms]
    da_hi = [a["da_hi"] for a in arms]
    db = [a["db"] for a in arms]
    db_lo = [a["db_lo"] for a in arms]
    db_hi = [a["db_hi"] for a in arms]

    ax.axhline(0.0, color=COLORS["zero"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.errorbar(
        xs - 0.08, da, yerr=[da_lo, da_hi], color=COLORS["A"], ls=LINESTYLES["A"],
        marker=MARKERS["A"], ms=6, mec="white", mew=0.5, lw=1.4,
        capsize=3, elinewidth=0.8, label=r"$\Delta_A$", zorder=2,
    )
    ax.errorbar(
        xs + 0.08, db, yerr=[db_lo, db_hi], color=COLORS["B"], ls=LINESTYLES["B"],
        marker=MARKERS["B"], ms=6, mec="white", mew=0.5, lw=1.4,
        capsize=3, elinewidth=0.8, label=r"$\Delta_B$", zorder=2,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([a["label"] for a in arms], fontsize=9)
    ax.set_ylim(-40, 25)
    ax.set_ylabel(r"Specialization $\Delta$ (pp)")
    ax.set_title(
        r"Claim B: SP-PPO ablation -- specialization not recovered",
        fontsize=9.5, fontweight="bold",
    )
    ax.text(
        0.5, 0.04,
        "gate " + " · ".join(a["gate"] for a in arms) + "   (none clears LCB > 0 on both poles)",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="#C62828",
        fontweight="bold",
    )
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # light FAIL band below zero to emphasize the point
    ax.axhspan(-40, 0, color="#FFEBEE", zorder=0, alpha=0.55)

    caption = (
        "Claim B only -- do not merge with Claim A. "
        "Matched sealed arms: PPO = CCP-S2 control; PPO+CSC = CCP-S2 treatment; "
        "PPO+CSC+SPFT = RSCFT retention treatment. "
        "All three FAIL the crossover gate. "
        "Share-Encoder PASS (Claim A) is an architectural sharing result, "
        "not evidence that SP-PPO outperformed PPO. "
        "Sources: CCP_S2_EVAL_INTEGRITY.json, RSCFT_EVAL_RESULT.json."
    )
    fig.text(0.5, -0.08, caption, ha="center", va="top", fontsize=7.2, style="italic")
    fig.subplots_adjust(bottom=0.28, top=0.88, left=0.12, right=0.98)

    paths = save_figure(fig, "fig_claim_b_spp_ablation")
    print(paths)
    return paths


if __name__ == "__main__":
    main()
