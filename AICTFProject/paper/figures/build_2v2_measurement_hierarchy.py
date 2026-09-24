"""Imitation ≠ strategy separation ≠ closed-loop specialization (2v2 sharing ladder).

Compact table figure from sealed RUNG*_STUDENT_FROZEN + RUNG*_LADDER_EVAL /
RUNG0_LADDER_REFERENCE. Makes the methodology warning visual:

  high expert agreement / preserved JS  =/=>  crossover gate PASS

Share-0 has no distilled-student KL/agreement (bit-exact expert dispatch);
those cells are marked N/A.

Run:  python paper/figures/build_2v2_measurement_hierarchy.py
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

from paper.figures.figure_style import COLORS, TWO_COLUMN, apply_style, save_figure

SD = ROOT / "artifacts/strategic_demand/sppo"

ROWS = [
    {
        "label": "Share-0",
        "agree": "N/A (exact)",
        "kl": "N/A",
        "js": "N/A*",
        "eval": "RUNG0_LADDER_REFERENCE.json",
        "gate_key": "POOLED_N128",
        "frozen": None,
    },
    {
        "label": "Share-Encoder",
        "eval": "RUNG1_LADDER_EVAL_RESULT.json",
        "gate_key": "OWN_GATE_N128",
        "frozen": "RUNG1_STUDENT_FROZEN.json",
    },
    {
        "label": "Share-Backbone",
        "eval": "RUNG2_LADDER_EVAL_RESULT.json",
        "gate_key": "OWN_GATE_N128",
        "frozen": "RUNG2_STUDENT_FROZEN.json",
    },
    {
        "label": "Share-Macro",
        "eval": "RUNG3_LADDER_EVAL_RESULT.json",
        "gate_key": "OWN_GATE_N128",
        "frozen": "RUNG3_STUDENT_FROZEN.json",
    },
]


def _load(name: str) -> dict:
    return json.loads((SD / name).read_text(encoding="utf-8"))


def _fmt_delta(d: dict) -> str:
    return f"{d['mean']*100:+.1f} [{d['lcb95']*100:+.1f}, {d['ucb95']*100:+.1f}]"


def _row_cells(spec: dict) -> list[str]:
    gate_blob = _load(spec["eval"])[spec["gate_key"]]
    da, db = gate_blob["delta_A"], gate_blob["delta_B"]
    if "passes" in gate_blob:
        gate = "PASS" if gate_blob["passes"] else "FAIL"
    else:
        # RUNG0_LADDER_REFERENCE#POOLED_N128 stores passes only on each delta.
        gate = "PASS" if (da.get("passes") and db.get("passes")) else "FAIL"

    if spec["frozen"] is None:
        agree = spec["agree"]
        kl = spec["kl"]
        js = spec["js"]
    else:
        hold = _load(spec["frozen"])["final_holdout"]
        agree = (
            f"{hold['holdout_agree_z0_vs_piA']:.3f} / "
            f"{hold['holdout_agree_z1_vs_piB']:.3f}"
        )
        kl = f"{hold['holdout_kl_A']:.3f} / {hold['holdout_kl_B']:.3f}"
        js = f"{hold['holdout_student_z0_z1_jsd']:.3f}"

    return [
        spec["label"],
        agree,
        kl,
        js,
        _fmt_delta(da),
        _fmt_delta(db),
        gate,
    ]


def main() -> dict:
    apply_style()
    headers = [
        "Model",
        r"Agree $z_0\!\sim\!\pi_A$ / $z_1\!\sim\!\pi_B$",
        r"Holdout KL $A$ / $B$",
        r"JS $z_0,z_1$",
        r"$\Delta_A$ (pp)",
        r"$\Delta_B$ (pp)",
        "Gate",
    ]
    cells = [_row_cells(r) for r in ROWS]

    fig, ax = plt.subplots(figsize=(TWO_COLUMN, 2.55))
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.55)

    # style header + gate column
    n_rows = len(cells) + 1
    n_cols = len(headers)
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#F0F0F0")
        cell.set_text_props(weight="bold")
    for i, row in enumerate(cells, start=1):
        gate_cell = table[i, n_cols - 1]
        if row[-1] == "PASS":
            gate_cell.set_facecolor("#E8F5E9")
            gate_cell.set_text_props(color="#2E7D32", weight="bold")
        else:
            gate_cell.set_facecolor("#FFEBEE")
            gate_cell.set_text_props(color="#C62828", weight="bold")
        # highlight Share-Macro agreement still high-ish but FAIL
        if row[0] == "Share-Macro":
            for j in (1, 2, 3):
                table[i, j].set_facecolor("#FFF8E1")

    caption = (
        r"Measurement hierarchy on the sealed 2v2 sharing ladder. "
        r"Imitation (agreement/KL) and strategy separation (JS) can remain strong while "
        r"closed-loop specialization ($\Delta$, gate) fails -- Share-Macro still imitates "
        r"but OWN_GATE FAIL via $\Delta_A$ LCB$=0$. "
        r"*Share-0 is expert dispatch (no distilled student); teacher A--B JS $=0.179$. "
        r"Sources: RUNG{{1,2,3}}_STUDENT_FROZEN + ladder EVAL / RUNG0_LADDER_REFERENCE."
    )
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=7.0, style="italic")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.28)

    paths = save_figure(fig, "fig_2v2_measurement_hierarchy")
    print(paths)
    # also dump machine-readable companion
    out = {
        "headers": ["Model", "agree", "holdout_KL", "JS_z0_z1", "delta_A", "delta_B", "gate"],
        "rows": cells,
    }
    (ROOT / "paper/data/2v2_measurement_hierarchy.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    return paths


if __name__ == "__main__":
    main()
