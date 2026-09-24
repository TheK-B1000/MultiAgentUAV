"""Progressive sharing ladder -- sealed Delta_A / Delta_B with 95% CIs.

Flagship quantitative Claim-A figure. Reads the sealed rung records and plots
exact specialization means with bootstrap CIs across

  Share-0 -> Share-Encoder -> Share-Backbone -> Share-Macro

so the preserved -> degraded -> failed transition is visible without relying
only on PASS/FAIL labels. Paired within-seed D vs Share-0 stays as the second
panel (detectable loss iff UCB95(D)<0).

Sources (authoritative):
  RUNG0_LADDER_REFERENCE.json          POOLED_N128
  RUNG1_LADDER_EVAL_RESULT.json        OWN_GATE_N128 / PRIMARY_WITHIN_SEED
  RUNG2_LADDER_EVAL_RESULT.json        OWN_GATE_N128 / PRIMARY_WITHIN_SEED
  RUNG3_LADDER_EVAL_RESULT.json        OWN_GATE_N128 / PRIMARY_WITHIN_SEED

Also writes paper/data/sharing_ladder_sealed_deltas.json for table/caption use.

Run:  python paper/figures/build_sharing_ladder_2v2.py
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
DATA = ROOT / "paper/data"

LADDER = [
    {
        "label": "Share-0",
        "sub": "Independent experts",
        "delta_src": ("RUNG0_LADDER_REFERENCE.json", "POOLED_N128"),
        "d_src": None,
    },
    {
        "label": "Share-Encoder",
        "sub": "Shared CNN",
        "delta_src": ("RUNG1_LADDER_EVAL_RESULT.json", "OWN_GATE_N128"),
        "d_src": ("RUNG1_LADDER_EVAL_RESULT.json", "PRIMARY_WITHIN_SEED"),
    },
    {
        "label": "Share-Backbone",
        "sub": "+ MLP backbone",
        "delta_src": ("RUNG2_LADDER_EVAL_RESULT.json", "OWN_GATE_N128"),
        "d_src": ("RUNG2_LADDER_EVAL_RESULT.json", "PRIMARY_WITHIN_SEED"),
    },
    {
        "label": "Share-Macro",
        "sub": "+ macro outputs",
        "delta_src": ("RUNG3_LADDER_EVAL_RESULT.json", "OWN_GATE_N128"),
        "d_src": ("RUNG3_LADDER_EVAL_RESULT.json", "PRIMARY_WITHIN_SEED"),
    },
]


def _load(name: str) -> dict:
    return json.loads((SD / name).read_text(encoding="utf-8"))


def _gate_from_block(g: dict) -> str:
    """Joint gate: both poles mean>0 and LCB95>0 (sealed `passes` when present)."""
    if "passes" in g and isinstance(g["passes"], bool):
        return "PASS" if g["passes"] else "FAIL"
    da, db = g["delta_A"], g["delta_B"]
    if "passes" in da and "passes" in db:
        return "PASS" if (da["passes"] and db["passes"]) else "FAIL"
    ok = (
        da["mean"] > 0 and da["lcb95"] > 0
        and db["mean"] > 0 and db["lcb95"] > 0
    )
    return "PASS" if ok else "FAIL"


def _pp(x: float) -> float:
    return float(x) * 100.0


def _delta_record(blob: dict, key: str) -> dict:
    g = blob[key]
    da, db = g["delta_A"], g["delta_B"]
    return {
        "delta_A": {
            "mean_pp": _pp(da["mean"]),
            "lcb95_pp": _pp(da["lcb95"]),
            "ucb95_pp": _pp(da["ucb95"]),
            "passes": bool(da.get("passes", da["mean"] > 0 and da["lcb95"] > 0)),
        },
        "delta_B": {
            "mean_pp": _pp(db["mean"]),
            "lcb95_pp": _pp(db["lcb95"]),
            "ucb95_pp": _pp(db["ucb95"]),
            "passes": bool(db.get("passes", db["mean"] > 0 and db["lcb95"] > 0)),
        },
        "gate": _gate_from_block(g),
    }


def _d_record(blob: dict, key: str) -> dict:
    g = blob[key]
    da, db = g["D_A"], g["D_B"]
    return {
        "D_A": {
            "mean_pp": _pp(da["mean"]),
            "lcb95_pp": _pp(da["lcb95"]),
            "ucb95_pp": _pp(da["ucb95"]),
            "detectable_loss": bool(da["ucb95"] < 0),
        },
        "D_B": {
            "mean_pp": _pp(db["mean"]),
            "lcb95_pp": _pp(db["lcb95"]),
            "ucb95_pp": _pp(db["ucb95"]),
            "detectable_loss": bool(db["ucb95"] < 0),
        },
    }


def _fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.1f} [{lo:.1f}, {hi:.1f}]"


def _collect() -> list[dict]:
    rows = []
    for rung in LADDER:
        fname, key = rung["delta_src"]
        rec = _delta_record(_load(fname), key)
        row = {
            "label": rung["label"],
            "sub": rung["sub"],
            "source": {"file": fname, "key": key},
            **rec,
            "paired_D": None,
        }
        if rung["d_src"] is not None:
            f2, k2 = rung["d_src"]
            row["paired_D"] = {
                "source": {"file": f2, "key": k2},
                **_d_record(_load(f2), k2),
            }
        else:
            row["paired_D"] = {
                "source": None,
                "D_A": {"mean_pp": 0.0, "lcb95_pp": 0.0, "ucb95_pp": 0.0, "detectable_loss": False},
                "D_B": {"mean_pp": 0.0, "lcb95_pp": 0.0, "ucb95_pp": 0.0, "detectable_loss": False},
                "note": "reference (D ≡ 0 by definition)",
            }
        rows.append(row)
    return rows


def _write_sidecar(rows: list[dict]) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "sharing_ladder_sealed_deltas.json"
    payload = {
        "record": "Sealed progressive-sharing specialization deltas (2v2)",
        "units": "percentage points (pp); means and percentile-bootstrap 95% CIs",
        "n_seeds": 128,
        "bootstrap": {"samples": 20000, "alpha": 0.05, "rng_seed": 7},
        "gate_criterion": "PASS iff mean(Delta)>0 AND LCB95(Delta)>0 on BOTH poles",
        "reading": (
            "Share-0 and Share-Encoder preserve the joint gate; "
            "Share-Backbone still PASSes but with lower Delta_A; "
            "Share-Macro FAILs because LCB95(Delta_A)=0."
        ),
        "rungs": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> dict:
    apply_style()
    rows = _collect()
    sidecar = _write_sidecar(rows)

    xs = np.arange(len(rows))
    da = [r["delta_A"]["mean_pp"] for r in rows]
    da_lo = [r["delta_A"]["mean_pp"] - r["delta_A"]["lcb95_pp"] for r in rows]
    da_hi = [r["delta_A"]["ucb95_pp"] - r["delta_A"]["mean_pp"] for r in rows]
    db = [r["delta_B"]["mean_pp"] for r in rows]
    db_lo = [r["delta_B"]["mean_pp"] - r["delta_B"]["lcb95_pp"] for r in rows]
    db_hi = [r["delta_B"]["ucb95_pp"] - r["delta_B"]["mean_pp"] for r in rows]
    gates = [r["gate"] for r in rows]

    dA = [r["paired_D"]["D_A"]["mean_pp"] for r in rows]
    dA_lo = [r["paired_D"]["D_A"]["mean_pp"] - r["paired_D"]["D_A"]["lcb95_pp"] for r in rows]
    dA_hi = [r["paired_D"]["D_A"]["ucb95_pp"] - r["paired_D"]["D_A"]["mean_pp"] for r in rows]
    dB = [r["paired_D"]["D_B"]["mean_pp"] for r in rows]
    dB_lo = [r["paired_D"]["D_B"]["mean_pp"] - r["paired_D"]["D_B"]["lcb95_pp"] for r in rows]
    dB_hi = [r["paired_D"]["D_B"]["ucb95_pp"] - r["paired_D"]["D_B"]["mean_pp"] for r in rows]

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(TWO_COLUMN, 3.35), gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    # --- left: sealed absolute specialization (flagship panel) ---
    ax0.axhline(0.0, color=COLORS["zero"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    # soft FAIL band: LCB must clear above 0; shade below 0 only
    ax0.axhspan(-5, 0, color="#FFEBEE", alpha=0.55, zorder=0)
    ax0.errorbar(
        xs - 0.07, da, yerr=[da_lo, da_hi], color=COLORS["A"], ls=LINESTYLES["A"],
        marker=MARKERS["A"], ms=6, mec="white", mew=0.5, lw=1.4,
        capsize=3.5, elinewidth=0.9, label=r"$\Delta_A$", zorder=3,
    )
    ax0.errorbar(
        xs + 0.07, db, yerr=[db_lo, db_hi], color=COLORS["B"], ls=LINESTYLES["B"],
        marker=MARKERS["B"], ms=6, mec="white", mew=0.5, lw=1.4,
        capsize=3.5, elinewidth=0.9, label=r"$\Delta_B$", zorder=3,
    )

    # Annotate exact sealed mean [LCB, UCB] above each point
    for i, r in enumerate(rows):
        a, b = r["delta_A"], r["delta_B"]
        y_top = max(a["ucb95_pp"], b["ucb95_pp"]) + 2.2
        ax0.text(
            i, y_top,
            f"A {_fmt_ci(a['mean_pp'], a['lcb95_pp'], a['ucb95_pp'])}\n"
            f"B {_fmt_ci(b['mean_pp'], b['lcb95_pp'], b['ucb95_pp'])}",
            ha="center", va="bottom", fontsize=5.8, color="#222222",
            linespacing=1.15,
        )
        gate_color = "#2E7D32" if r["gate"] == "PASS" else "#C62828"
        ax0.text(
            i, -3.2, r["gate"], ha="center", va="top",
            fontsize=7.5, fontweight="bold", color=gate_color,
        )

    ax0.set_xticks(xs)
    ax0.set_xticklabels([r["label"] for r in rows], fontsize=8)
    for i, r in enumerate(rows):
        ax0.text(i, -7.5, r["sub"], ha="center", va="top", fontsize=6.2, color="#555555")
    ax0.set_ylim(-12, 55)
    ax0.set_ylabel(r"Specialization $\Delta$ (pp)")
    ax0.set_title(
        r"Sealed $\Delta_A,\Delta_B$ with 95% CIs (n=128)",
        fontsize=9.5, fontweight="bold",
    )
    ax0.legend(loc="upper right", frameon=False, fontsize=8)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # Reading strip: preserved -> degraded -> failed
    reading = "preserved  →  preserved  →  degraded (still PASS)  →  failed"
    ax0.text(
        0.5, 1.02, reading, transform=ax0.transAxes,
        ha="center", va="bottom", fontsize=7.2, color="#333333", style="italic",
    )

    # --- right: paired loss vs Share-0 ---
    ax1.axhline(0.0, color=COLORS["zero"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax1.errorbar(
        xs - 0.07, dA, yerr=[dA_lo, dA_hi], color=COLORS["A"], ls=LINESTYLES["A"],
        marker=MARKERS["A"], ms=5.5, mec="white", mew=0.5, lw=1.3,
        capsize=3, elinewidth=0.8, label=r"$D_A$ vs Share-0", zorder=2,
    )
    ax1.errorbar(
        xs + 0.07, dB, yerr=[dB_lo, dB_hi], color=COLORS["B"], ls=LINESTYLES["B"],
        marker=MARKERS["B"], ms=5.5, mec="white", mew=0.5, lw=1.3,
        capsize=3, elinewidth=0.8, label=r"$D_B$ vs Share-0", zorder=2,
    )
    # Mark detectable Pole-A losses
    for i, r in enumerate(rows):
        if r["paired_D"]["D_A"]["detectable_loss"]:
            ax1.plot(i - 0.07, dA[i], marker="x", color=COLORS["A"], ms=8, zorder=4)
        if r["paired_D"]["D_B"]["detectable_loss"]:
            ax1.plot(i + 0.07, dB[i], marker="x", color=COLORS["B"], ms=8, zorder=4)

    ax1.set_xticks(xs)
    ax1.set_xticklabels([r["label"] for r in rows], fontsize=8)
    ax1.set_ylim(-40, 20)
    ax1.set_ylabel(r"Paired change $D$ (pp)")
    ax1.set_title(r"Detectable loss iff UCB$_{95}(D)<0$", fontsize=9.5, fontweight="bold")
    ax1.legend(loc="lower left", frameon=False, fontsize=7.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    assert gates == ["PASS", "PASS", "PASS", "FAIL"], gates

    caption = (
        "Claim A quantitative ladder. Exact sealed means and percentile-bootstrap 95% CIs "
        "(n=128 matched seeds). Share-0 is the bit-exact expert-dispatch control; "
        "Encoder/Backbone/Macro share one distillation recipe (sharing axis only among those three). "
        "Share-0/Encoder preserve the joint crossover gate; "
        "Share-Backbone still PASSes with lower Delta_A; Share-Macro FAILs because "
        "LCB95(Delta_A)=0. Right: paired D vs Share-0 (x = detectable loss). "
        "Not an SP-PPO result."
    )
    fig.text(0.5, -0.05, caption, ha="center", va="top", fontsize=7.2, style="italic")
    fig.subplots_adjust(wspace=0.28, bottom=0.24, top=0.86, left=0.07, right=0.99)

    paths = save_figure(fig, "fig_sharing_ladder_2v2")
    paths["sidecar"] = str(sidecar)
    print(paths)
    return paths


if __name__ == "__main__":
    main()
