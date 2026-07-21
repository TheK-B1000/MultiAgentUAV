"""Verify feedforward router smoke gates against repertoire anchor."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_ANCHOR = Path(r"K:\AICTF_Checkpoint_Archive\v6i9-repertoire-refactor-r1\frozen_repertoire_anchor.json")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_metrics_csv(run_tag: str) -> Path | None:
    candidates = sorted(
        (_REPO / "checkpoints/2v2").glob(f"*{run_tag}*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _max_numeric(rows: list[dict], key: str) -> float:
    vals = []
    for row in rows:
        if key not in row or row[key] in ("", None):
            continue
        try:
            vals.append(float(row[key]))
        except ValueError:
            continue
    return max(vals) if vals else 0.0


def _min_numeric(rows: list[dict], key: str) -> float:
    vals = []
    for row in rows:
        if key not in row or row[key] in ("", None):
            continue
        try:
            vals.append(float(row[key]))
        except ValueError:
            continue
    return min(vals) if vals else float("inf")


def _occupancy_ok(rows: list[dict], latent_k: int = 4) -> tuple[bool, str]:
    keys = [f"latent_z_occupancy_z{z}" for z in range(latent_k)]
    if not any(key in rows[-1] for key in keys):
        keys = [f"router_z_occupancy_z{z}" for z in range(latent_k)]
    last = rows[-1]
    occ = []
    for key in keys:
        if key in last and last[key] not in ("", None):
            occ.append(float(last[key]))
    if not occ:
        return True, "occupancy columns unavailable (non-fatal at smoke)"
    total = sum(occ)
    if total <= 0:
        return False, "all occupancy zero"
    shares = [v / total for v in occ]
    if max(shares) >= 0.999:
        return False, "100% single-z collapse"
    if sum(1 for s in shares if s > 0.01) < latent_k:
        return False, f"only {sum(1 for s in shares if s > 0.01)} latents reachable"
    return True, "ok"


def _credit_telemetry_from_checkpoint(smoke_checkpoint: Path) -> dict[str, float]:
    from rl.custom_ppo.checkpoints.archive import _torch_load_checkpoint

    payload = _torch_load_checkpoint(str(smoke_checkpoint), map_location="cpu")
    last_stats = dict(payload.get("last_stats", {}) or {})
    keys = [
        "strategy_advantage_source",
        "router_decision_count",
        "router_advantage_std",
        "router_advantage_mean",
        "router_advantage_positive_fraction",
        "strategy_policy_grad_norm",
        "router_entropy_grad_norm",
        "strategy_policy_to_router_entropy_grad_ratio",
        "feedforward_router_entropy_loss",
        "strategy_policy_loss",
        "strategy_encoder_grad_norm",
    ]
    return {k: float(last_stats[k]) for k in keys if k in last_stats}


def verify(
    *,
    metrics_csv: Path,
    anchor_path: Path,
    smoke_checkpoint: Path | None,
    recurrent_hidden_dim: int,
) -> dict:
    anchor = _read_json(anchor_path)
    with metrics_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in metrics csv: {metrics_csv}")

    gates: dict[str, dict] = {}
    gates["frozen_repertoire"] = {
        "shared_actor_grad_norm_max": _max_numeric(rows, "shared_actor_grad_norm"),
        "shared_actor_max_abs_delta_max": _max_numeric(rows, "shared_actor_max_abs_delta"),
        "z_specific_grad_norm_max": _max_numeric(rows, "z_specific_grad_norm"),
        "z_specific_max_abs_delta_max": _max_numeric(rows, "z_specific_max_abs_delta"),
        "pass": (
            _max_numeric(rows, "shared_actor_grad_norm") == 0.0
            and _max_numeric(rows, "shared_actor_max_abs_delta") == 0.0
            and _max_numeric(rows, "z_specific_grad_norm") == 0.0
            and _max_numeric(rows, "z_specific_max_abs_delta") == 0.0
        ),
    }

    gates["router_mechanism"] = {
        "strategy_encoder_grad_norm_max": _max_numeric(rows, "strategy_encoder_grad_norm"),
        "router_grad_norm_max": _max_numeric(rows, "router_grad_norm"),
        "pass": (
            _max_numeric(rows, "strategy_encoder_grad_norm") > 0.0
            and _max_numeric(rows, "router_grad_norm") > 0.0
        ),
    }

    occ_ok, occ_msg = _occupancy_ok(rows)
    gates["occupancy"] = {"pass": occ_ok, "detail": occ_msg}

    gates["recurrent_inactive"] = {
        "recurrent_selector_hidden_dim": recurrent_hidden_dim,
        "pass": int(recurrent_hidden_dim) == 0,
    }

    gates["router_credit"] = {
        "pass": False,
        "detail": "smoke checkpoint not provided",
    }
    credit: dict[str, float] = {}
    if smoke_checkpoint is not None and smoke_checkpoint.is_file():
        credit = _credit_telemetry_from_checkpoint(smoke_checkpoint)
        ratio = credit.get("strategy_policy_to_router_entropy_grad_ratio")
        gates["router_credit"] = {
            "strategy_advantage_source": credit.get("strategy_advantage_source"),
            "router_decision_count": credit.get("router_decision_count"),
            "router_advantage_std": credit.get("router_advantage_std"),
            "strategy_policy_grad_norm": credit.get("strategy_policy_grad_norm"),
            "router_entropy_grad_norm": credit.get("router_entropy_grad_norm"),
            "strategy_policy_to_router_entropy_grad_ratio": ratio,
            "feedforward_router_entropy_loss": credit.get("feedforward_router_entropy_loss"),
            "pass": (
                credit.get("strategy_advantage_source") == 2.0
                and (credit.get("router_decision_count") or 0) > 0
                and (credit.get("router_advantage_std") or 0) > 0
                and (credit.get("strategy_policy_grad_norm") or 0) > 0
                and (credit.get("router_entropy_grad_norm") or 0) > 0
                and ratio is not None
                and math.isfinite(float(ratio))
                and (credit.get("feedforward_router_entropy_loss") or 0) != 0
            ),
        }

    hash_report = {
        "frozen_tensor_hash_match": None,
        "shared_actor_max_abs_delta": None,
        "z_specific_max_abs_delta": None,
    }
    if smoke_checkpoint is not None and smoke_checkpoint.is_file():
        from rl.custom_ppo.checkpoints.archive import _torch_load_checkpoint
        from rl.custom_ppo.diagnostics.frozen_repertoire_hash import compare_frozen_repertoire_hashes

        anchor_ckpt = Path(anchor.get("checkpoint_source", ""))
        if not anchor_ckpt.is_file():
            anchor_ckpt = Path(r"K:\AICTF_Checkpoint_Archive\v6i9-repertoire-refactor-r1") / (
                "final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip"
            )
        anchor_sd = _torch_load_checkpoint(str(anchor_ckpt), map_location="cpu").get("model_state_dict", {})
        smoke_sd = _torch_load_checkpoint(str(smoke_checkpoint), map_location="cpu").get("model_state_dict", {})
        hash_report = compare_frozen_repertoire_hashes(anchor_sd, smoke_sd)
        gates["frozen_tensor_hash"] = {
            "pass": bool(hash_report.get("frozen_tensor_hash_match")),
            **hash_report,
        }
    else:
        gates["frozen_tensor_hash"] = {"pass": None, "detail": "smoke checkpoint not provided"}

    all_required = [
        gates["frozen_repertoire"]["pass"],
        gates["router_mechanism"]["pass"],
        gates["occupancy"]["pass"],
        gates["recurrent_inactive"]["pass"],
        gates["router_credit"]["pass"],
    ]
    if gates["frozen_tensor_hash"]["pass"] is not None:
        all_required.append(bool(gates["frozen_tensor_hash"]["pass"]))

    report = {
        "metrics_csv": str(metrics_csv),
        "anchor_path": str(anchor_path),
        "rows": len(rows),
        "global_step_last": rows[-1].get("global_step"),
        "router_credit_telemetry": credit,
        "gates": gates,
        "smoke_pass": all(all_required),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="v6i9-mapaware-router-feedforward-hardpool-refactor-r1-seed1-smoke")
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--smoke-checkpoint", type=Path, default=None)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    metrics_csv = args.metrics_csv or _latest_metrics_csv(args.run_tag)
    if metrics_csv is None or not metrics_csv.is_file():
        raise SystemExit(f"Could not find metrics csv for run tag {args.run_tag!r}")
    if not args.anchor.is_file():
        raise SystemExit(f"Missing anchor sidecar: {args.anchor}")

    report = verify(
        metrics_csv=metrics_csv,
        anchor_path=args.anchor,
        smoke_checkpoint=args.smoke_checkpoint,
        recurrent_hidden_dim=int(args.recurrent_hidden_dim),
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"[verify] wrote {args.out}")
    else:
        print(text)
    print(f"[verify] smoke_pass={report['smoke_pass']}")
    return 0 if report["smoke_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
