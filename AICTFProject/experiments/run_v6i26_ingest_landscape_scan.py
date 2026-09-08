#!/usr/bin/env python3
"""Ingest an existing V6I24/V6I26 payoff eval JSON into LRO Stage-0 summary.

Avoids re-rolling episodes when a slim eval matrix already exists.
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
    lro_manifest,
    payoff_tensor_summary,
    select_response_target,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest payoff matrix → LRO Stage-0")
    p.add_argument(
        "--eval-json",
        default="artifacts/v6i24_population_seed1/probe_05u/eval_gates_slim/v6i24_eval_gates.json",
    )
    p.add_argument("--output-dir", default="artifacts/v6i26_landscape_scan_seed1")
    p.add_argument("--margin", type=float, default=0.10)
    return p.parse_args()


def _extract_payoff(payload: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """Best-effort extract P[policy, context] from eval_gates JSON shapes."""
    # Shape A0: V6I24 eval_gates strategic block
    strategic = payload.get("strategic") or {}
    if "payoff_matrix" in strategic:
        labels = list(
            strategic.get("member_labels")
            or strategic.get("policy_labels")
            or []
        )
        contexts = list(strategic.get("context_labels") or [])
        return (
            np.asarray(strategic["payoff_matrix"], dtype=np.float64),
            labels,
            contexts,
        )

    # Shape A: explicit matrix
    if "payoff_matrix" in payload and "policy_labels" in payload:
        labels = list(payload["policy_labels"])
        contexts = list(payload.get("contexts") or payload.get("context_labels") or [])
        return np.asarray(payload["payoff_matrix"], dtype=np.float64), labels, contexts

    # Shape B: nested cells keyed by policy / context
    cells = payload.get("cells") or payload.get("payoff_cells") or {}
    if cells:
        # expect {policy: {context: {payoff|wr: ...}}}
        labels = sorted(cells.keys())
        contexts = sorted({c for pol in cells.values() for c in pol.keys()})
        mat = np.zeros((len(labels), len(contexts)), dtype=np.float64)
        for i, lab in enumerate(labels):
            for j, ctx in enumerate(contexts):
                entry = cells[lab].get(ctx, {})
                if isinstance(entry, (int, float)):
                    mat[i, j] = float(entry)
                else:
                    mat[i, j] = float(entry.get("payoff", entry.get("wr", 0.0)))
        return mat, labels, contexts

    # Shape C: flat list of rows
    rows = payload.get("payoff_rows") or payload.get("results") or []
    if rows:
        labels = []
        contexts = []
        seen_l, seen_c = {}, {}
        for r in rows:
            lab = str(r.get("policy") or r.get("label") or r.get("member"))
            ctx = str(r.get("context") or f"{r.get('opponent')}|{r.get('map')}")
            if lab not in seen_l:
                seen_l[lab] = len(labels)
                labels.append(lab)
            if ctx not in seen_c:
                seen_c[ctx] = len(contexts)
                contexts.append(ctx)
        mat = np.full((len(labels), len(contexts)), np.nan, dtype=np.float64)
        for r in rows:
            lab = str(r.get("policy") or r.get("label") or r.get("member"))
            ctx = str(r.get("context") or f"{r.get('opponent')}|{r.get('map')}")
            mat[seen_l[lab], seen_c[ctx]] = float(
                r.get("payoff", r.get("wr", r.get("success", 0.0)))
            )
        if np.isnan(mat).any():
            raise ValueError("incomplete payoff_rows matrix")
        return mat, labels, contexts

    raise KeyError(
        "Could not find payoff_matrix / cells / payoff_rows in eval JSON. "
        "Keys: " + ", ".join(sorted(payload.keys())[:40])
    )


def main() -> int:
    args = _parse_args()
    path = Path(args.eval_json)
    if not path.is_file():
        print(f"ERROR: missing {path}")
        return 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        payoff, labels, contexts = _extract_payoff(payload)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR extracting payoff: {exc}")
        # Fallback: parse from known slim-eval nested structure used by runner.
        summary = payload.get("summary") or {}
        matrix = payload.get("payoff") or payload.get("mean_payoff")
        if matrix is None:
            return 3
        labels = list(payload.get("member_labels") or payload.get("policies") or [])
        contexts = list(payload.get("context_keys") or [])
        payoff = np.asarray(matrix, dtype=np.float64)

    summary = payoff_tensor_summary(
        payoff, policy_labels=labels, contexts=contexts, margin=float(args.margin)
    )
    target = select_response_target(
        payoff, contexts=contexts, policy_labels=labels
    )
    if summary["niche_signal"] and float(summary["G_available_point"]) > 0.0:
        decision = "PROMOTE_LRO_BIRTH"
        note = "Archived policies already complement (G_available > 0)."
    elif float(summary["G_available_point"]) <= 0.0:
        decision = "MANUFACTURE_VIA_LRO_STAGE1"
        note = (
            "G_available = 0 — archives have no repertoire. "
            "Stage-1 manufactures; first success is G_after > G_before."
        )
    else:
        decision = "PROMOTE_LRO_BIRTH"
        note = "Positive G_available; proceed to Stage-1."

    out = {
        "experiment": "v6i26_strategic_landscape_scan",
        "source": "ingest_existing_eval",
        "source_path": str(path),
        "lro": lro_manifest(),
        "policy_labels": labels,
        "contexts": contexts,
        "payoff_matrix": payoff.tolist(),
        "summary": summary,
        "next_response_target": target,
        "G_available_effective": float(summary["G_available_point"]),
        "decision": decision,
        "note": note,
        "first_true_v6i26_success": "G_available_after > G_available_before",
    }
    out_dir = Path(args.output_dir)
    write_json(out_dir / "landscape_scan.json", out)
    print(json.dumps({"decision": decision, **{k: summary[k] for k in (
        "unique_best_count",
        "cells_with_margin_ge",
        "G_available_point",
        "max_pairwise_row_distance",
        "niche_signal",
        "parallel_rows",
    )}}, indent=2))
    print(f"Wrote {out_dir / 'landscape_scan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
