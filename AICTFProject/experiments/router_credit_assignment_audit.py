#!/usr/bin/env python3
"""Router credit-assignment audit for v6i9 feedforward router stage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.custom_ppo.diagnostics.router_credit_audit import (  # noqa: E402
    audit_feedforward_router_credit_wiring,
    audit_router_rollout_dump,
    offline_feedforward_predictability,
    run_synthetic_router_sign_test,
    summarize_router_advantages,
)
from rl.evaluation.router_ablation import (  # noqa: E402
    build_shuffled_mapping_from_learned_traces,
    learned_z_histogram_from_traces,
    validate_shuffled_mapping_histogram,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Router credit-assignment audit")
    p.add_argument(
        "--mode",
        choices=("all", "shuffle", "sign-test", "buffer-audit", "offline-predict"),
        default="all",
    )
    p.add_argument("--trace-json", default=None, help="Learned-router opportunity traces JSON list")
    p.add_argument(
        "--buffer",
        "--buffer-npz",
        dest="buffer",
        default=None,
        help="Rollout dump .pt from dump_router_rollout_audit.py",
    )
    p.add_argument("--forced-z-csv", default=None, help="Forced-z episode_results.csv for offline predictability")
    p.add_argument("--out-dir", default="artifacts/router_credit_audit")
    p.add_argument("--latent-k", type=int, default=4)
    p.add_argument("--switch-cadence", type=int, default=32)
    return p.parse_args()


def _load_traces(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "traces" in payload:
        return list(payload["traces"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported trace JSON format in {path}")


def _load_buffer(path: str) -> dict[str, Any]:
    return dict(torch.load(path, map_location="cpu", weights_only=False))


def _run_shuffle_audit(traces: list[dict[str, Any]], *, latent_k: int, switch_cadence: int) -> dict[str, Any]:
    mapping, meta = build_shuffled_mapping_from_learned_traces(
        traces,
        latent_k=latent_k,
        switch_cadence=switch_cadence,
        require_min_contexts=len({(t["opponent"], t["seed"], t["episode_index"]) for t in traces}) >= 2,
    )
    hist_check = validate_shuffled_mapping_histogram(traces, mapping)
    return {
        "learned_histogram": dict(learned_z_histogram_from_traces(traces)),
        "mapping_meta": meta,
        "histogram_validation": hist_check,
    }


def _run_sign_test(*, latent_k: int) -> dict[str, Any]:
    from rl.latent_marl import StrategyEncoder

    context_dim = 35
    encoder = StrategyEncoder(state_dim=context_dim, latent_k=latent_k, hidden=64)
    result = run_synthetic_router_sign_test(encoder, context_dim=context_dim, latent_k=latent_k)
    return {
        "passed": result.passed,
        "reversed_passed": result.reversed_passed,
        "p_z0_context_a_before": result.p_z0_context_a_before,
        "p_z0_context_a_after": result.p_z0_context_a_after,
        "p_z1_context_b_before": result.p_z1_context_b_before,
        "p_z1_context_b_after": result.p_z1_context_b_after,
        "details": result.details,
    }


def _run_buffer_audit(path: str, out_dir: Path) -> dict[str, Any]:
    payload = _load_buffer(path)
    meta = payload.get("metadata") or {}

    batch = {
        "router_advantages": payload["raw_router_advantages"],
        "advantages": payload["actor_advantages"],
        "router_reward": payload.get("router_reward"),
        "router_decision_valid": payload["router_decision_mask"],
    }
    cfg_dict = payload.get("cfg") or {}
    cfg = SimpleNamespace(**cfg_dict)
    wiring = audit_feedforward_router_credit_wiring(cfg, batch)

    mask = payload["router_decision_mask"].bool()
    adv_summary = summarize_router_advantages(
        router_advantages=payload["raw_router_advantages"],
        actor_advantages=payload["actor_advantages"],
        returns=payload.get("router_returns", payload["returns"]),
        selected_z=payload["selected_z"],
        resample_mask=mask,
    )

    artifact_dir = out_dir / Path(path).stem
    artifact_report = audit_router_rollout_dump(payload, out_dir=artifact_dir)

    return {
        "wiring": wiring,
        "advantage_summary": adv_summary,
        "artifact_dir": str(artifact_dir),
        "artifacts": artifact_report,
        "metadata": meta,
    }


def _run_offline_predict(csv_path: str) -> dict[str, Any]:
    import csv

    rows: list[dict[str, Any]] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    def _seed_of(row: dict[str, Any]) -> int:
        for col in ("seed", "episode_seed", "cell_seed"):
            if col in row and str(row[col]).strip() != "":
                return int(float(row[col]))
        raise KeyError("no seed-like column (seed/episode_seed/cell_seed) in forced-z CSV")

    def _forced_z_of(row: dict[str, Any]) -> int:
        for col in ("forced_z", "z", "latent_z", "fixed_latent_id"):
            if col in row and str(row[col]).strip() != "":
                return int(float(row[col]))
        return -1

    by_context: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["opponent"]), str(row.get("map", row.get("map_name", ""))), _seed_of(row))
        by_context.setdefault(key, []).append(row)

    samples: list[dict[str, Any]] = []
    fixed_z2_returns: list[float] = []
    for (_opp, _map, _seed), cells in by_context.items():
        best = max(cells, key=lambda r: float(r.get("return", r.get("ep_return", 0.0))))
        z2 = [r for r in cells if _forced_z_of(r) == 2]
        if z2:
            fixed_z2_returns.append(float(z2[0].get("return", z2[0].get("ep_return", 0.0))))
        samples.append(
            {
                "best_z": _forced_z_of(best),
                "return": float(best.get("return", best.get("ep_return", 0.0))),
                "best_return": float(best.get("return", best.get("ep_return", 0.0))),
                "feature_seed": int(_seed),
                "opponent_code": hash(_opp) % 97,
                "map_code": hash(_map) % 13,
            }
        )

    def feature_fn(sample: dict[str, Any]) -> torch.Tensor:
        return torch.tensor(
            [
                float(sample["feature_seed"]) / 10000.0,
                float(sample["opponent_code"]) / 100.0,
                float(sample["map_code"]) / 20.0,
                1.0,
            ],
            dtype=torch.float32,
        )

    fixed_z2_mean = float(sum(fixed_z2_returns) / len(fixed_z2_returns)) if fixed_z2_returns else 0.0
    report = offline_feedforward_predictability(
        samples,
        feature_fn=feature_fn,
        fixed_z2_return=fixed_z2_mean,
    )
    return {
        "n_contexts": len(samples),
        "fixed_z2_mean_return": fixed_z2_mean,
        "accuracy": report.accuracy,
        "top2_accuracy": report.top2_accuracy,
        "beats_chance": report.beats_chance,
        "beats_fixed_z2": report.beats_fixed_z2,
        "mean_regret_vs_best_z": report.mean_regret_vs_best_z,
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"mode": args.mode}

    if args.mode in ("all", "shuffle"):
        if not args.trace_json:
            print("shuffle audit skipped: provide --trace-json with learned opportunity traces")
        else:
            traces = _load_traces(args.trace_json)
            report["shuffle_audit"] = _run_shuffle_audit(
                traces, latent_k=args.latent_k, switch_cadence=args.switch_cadence
            )
            print("shuffle histogram preserved:", report["shuffle_audit"]["histogram_validation"]["histogram_preserved"])

    if args.mode in ("all", "sign-test"):
        report["sign_test"] = _run_sign_test(latent_k=args.latent_k)
        print(
            "synthetic sign test:",
            "PASS" if report["sign_test"]["passed"] and report["sign_test"]["reversed_passed"] else "FAIL",
        )

    if args.mode in ("all", "buffer-audit"):
        if not args.buffer:
            print("buffer audit skipped: provide --buffer path/to/update_0001.pt")
        else:
            report["buffer_audit"] = _run_buffer_audit(args.buffer, out_dir)
            issue = report["buffer_audit"].get("wiring", {}).get("credit_wiring_issue")
            if issue:
                print("CREDIT WIRING ISSUE:", issue)
            verdict = report["buffer_audit"]["artifacts"].get("verdict", {})
            print("audit verdict:", json.dumps(verdict, indent=2))

    if args.mode in ("all", "offline-predict"):
        if not args.forced_z_csv:
            print("offline predictability skipped: provide --forced-z-csv")
        else:
            report["offline_predictability"] = _run_offline_predict(args.forced_z_csv)
            print(
                "offline predictor beats fixed z2:",
                report["offline_predictability"]["beats_fixed_z2"],
            )

    out_path = out_dir / "router_credit_audit_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
