#!/usr/bin/env python3
"""V6I23 pre-registered population-birth diagnostic (5u → 25u total).

Scientific question
-------------------
Do independent per-z action heads under a frozen shared trunk need more
optimization time to separate π(a|s,z), or does the frozen representation
constrain every specialist to the same behavioral basin?

This is a closed diagnostic — not an open continuation chain. Router stays
locked. No new diversity losses or architecture changes.

Protocol (locked)
-----------------
Start: artifacts/.../v6i23_population_birth_5u_seed1/final_*.zip
Milestones (birth PPO updates total): 10u, 15u, 25u
Sampling: forced balanced_episode z
Trainable: active adapter + active per-z head only (Stage-2 freeze)
Shared trunk / shared action_head: frozen
Matched-seed probes at every milestone (same base_seed / cells)

Tracked
-------
pairwise head L2, adapter contribution magnitude,
CF action-JSD mean/max, entropy-aware (non-tie) argmax disagreement,
forced-z payoff rows / Stage-C / oracle margin / behavior pair distance.

Decision rule
-------------
PROMOTE if:
  CF JSD mean > 0.05 on ≥2 cells with pairs>0.05, OR
  non-tie argmax disagree > 0.20,
  AND forced-z payoff rows begin separating.

KEEP_INVESTIGATING if:
  JSD clearly rising but below threshold, OR
  trajectory/payoff separation precedes action-JSD.

STOP_EARLY_ESCALATE if (checked at 10u and 15u):
  JSD stuck ~1e-4..1e-3 AND head L2 rises while actions stay flat.
  Then escalate to four full independent policies → distill.
  Do NOT add soft regularizers.

Usage
-----
# Full orchestrated run (trains each segment then probes):
uv run python experiments/run_v6i23_population_birth_prereg_diagnostic.py --run

# Probe-only at an existing milestone checkpoint:
uv run python experiments/run_v6i23_population_birth_prereg_diagnostic.py \\
    --probe-only --milestone 10 \\
    --checkpoint artifacts/v6i23_popbirth_prereg/ckpt_10u/final_*.zip
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_PRESET = "v6i23"
_ROOT = PROJECT_ROOT / "artifacts" / "v6i23_popbirth_prereg"
_START_CKPT = (
    PROJECT_ROOT
    / "artifacts"
    / "v6i23_population_birth_5u_seed1"
    / "final_v6i23_population_birth_5u_seed1_2v2.zip"
)
_START_U = 5
_MILESTONES = (10, 15, 25)
_STEPS_PER_UPDATE = 4 * 256  # n_envs * n_steps
_PROBE_BASE_SEED = 42
_OPPONENTS = ("OP8", "OP9", "OP10", "OP11", "OP12")
_MAPS = ("map_b_split_lane", "map_b_split_lane_v2")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ckpt_dir(milestone: int) -> Path:
    return _ROOT / f"ckpt_{milestone:02d}u"


def _probe_dir(milestone: int) -> Path:
    return _ROOT / "probes" / f"{milestone:02d}u"


def _decision_path() -> Path:
    return _ROOT / "decision_log.json"


def _load_decision_log() -> dict[str, Any]:
    path = _decision_path()
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "protocol": "v6i23_population_birth_prereg_to_25u",
        "classification": "SUMMER-COMPATIBLE EXTENSION (diagnostic)",
        "router": "disabled",
        "created_at_utc": _utc(),
        "milestones": {},
        "verdict": "RUNNING",
    }


def _save_decision_log(log: dict[str, Any]) -> None:
    _ROOT.mkdir(parents=True, exist_ok=True)
    path = _decision_path()
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[prereg] wrote {path}")


def _run(cmd: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    print("[prereg] exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _train_segment(*, load: Path, milestone: int, additional_updates: int) -> Path:
    out = _ckpt_dir(milestone)
    out.mkdir(parents=True, exist_ok=True)
    steps = int(additional_updates) * _STEPS_PER_UPDATE
    tag = f"v6i23_popbirth_prereg_{milestone:02d}u_seed1"
    _run(
        [
            "uv",
            "run",
            "python",
            "rl/train_ppo.py",
            "--preset",
            _PRESET,
            "--load",
            str(load),
            "--load-weights-only",
            "--additional-steps",
            str(steps),
            "--n-envs",
            "4",
            "--n-steps",
            "256",
            "--n-epochs",
            "1",
            "--device",
            "cuda",
            "--run-tag",
            tag,
            "--checkpoint-dir",
            str(out),
            "--fresh-metrics-csv",
            "--episode-log-every",
            "0",
            "--periodic-checkpoint-steps",
            "0",
            "--no-progress-bar",
        ]
    )
    final = out / f"final_{tag}_2v2.zip"
    if not final.is_file():
        # Fallback: any final_*.zip in out
        finals = sorted(out.glob("final_*.zip"))
        if not finals:
            raise FileNotFoundError(f"No final checkpoint in {out}")
        final = finals[-1]
    return final


def _run_geometry_diag(checkpoint: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Reuse the existing diag script; capture JSON sidecar we write here.
    from experiments.run_v6i23_population_birth import _find_actor  # type: ignore
    from experiments.dump_router_rollout_audit import _build_audit_trainer

    _, _, _, trainer = _build_audit_trainer(
        preset=_PRESET,
        checkpoint=str(checkpoint),
        device="cuda",
        seed=1,
    )
    actor = _find_actor(trainer.model)
    if actor is None:
        raise RuntimeError("latent actor not found")

    heads = getattr(actor, "latent_action_heads", None)
    adapters = getattr(actor, "latent_adapters", None)
    head_l2: list[float] = []
    head_pair: list[dict[str, float]] = []
    if heads is not None:
        w0 = heads[0].weight.detach().float()
        for k, head in enumerate(heads):
            w = head.weight.detach().float()
            head_l2.append(float(w.norm().item()))
            if k > 0:
                head_pair.append(
                    {
                        "pair": f"0-{k}",
                        "weight_l2": float((w - w0).norm().item()),
                    }
                )
    adapter_l2 = []
    if adapters is not None:
        for k, ad in enumerate(adapters):
            adapter_l2.append(float(ad.weight.detach().float().norm().item()))

    # Offline α‖A(h)‖/‖h‖ on random local features
    alpha = float(getattr(actor, "_latent_z_alpha", 0.1) or 0.1)
    ratio = float("nan")
    if adapters is not None:
        import torch

        local = torch.randn(64, actor.local_feature_dim, device=next(actor.parameters()).device)
        z = torch.zeros(64, dtype=torch.long, device=local.device)
        pieces = [local.float()]
        if actor.strategy_embedding is not None:
            pieces.append(actor.strategy_embedding(z) * actor.z_embed_scale)
        h = actor.body(torch.cat(pieces, dim=-1))
        a_out = adapters[0](h)
        ratio = float((alpha * a_out.norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-8)).mean().item())

    payload = {
        "checkpoint": str(checkpoint),
        "head_weight_l2": head_l2,
        "head_pair_l2_vs_z0": head_pair,
        "head_pair_l2_mean": float(
            sum(p["weight_l2"] for p in head_pair) / max(1, len(head_pair))
        ),
        "adapter_weight_l2": adapter_l2,
        "adapter_weight_l2_mean": float(sum(adapter_l2) / max(1, len(adapter_l2))),
        "adapter_alpha_norm_ratio_mean": ratio,
        "active_z_only": bool(getattr(actor, "_population_birth_active_z_only", False)),
        "per_z_heads": heads is not None,
    }
    path = out_dir / "geometry_diag.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[prereg] geometry: head_pair_L2_mean={payload['head_pair_l2_mean']:.4f} "
          f"adapter_L2_mean={payload['adapter_weight_l2_mean']:.4f} "
          f"alpha_ratio={ratio:.4f}")
    return payload


def _run_jsd_probe(checkpoint: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "uv",
            "run",
            "python",
            "experiments/run_per_cell_action_jsd_probe.py",
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(out_dir),
            "--device",
            "cuda",
            "--base-seed",
            str(_PROBE_BASE_SEED),
            "--opponents",
            *_OPPONENTS,
            "--maps",
            *_MAPS,
        ]
    )
    report = json.loads((out_dir / "action_jsd_probe_report.json").read_text(encoding="utf-8"))
    return report["summary"]


def _run_forced_z(checkpoint: Path, out_dir: Path, *, episodes: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "uv",
            "run",
            "python",
            "experiments/run_forced_z_eval.py",
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(out_dir),
            "--inherit-training-config",
            "--opponents",
            *_OPPONENTS,
            "--maps",
            *_MAPS,
            "--episodes",
            str(episodes),
            "--oracle-metric",
            "win_margin",
            "--device",
            "cuda",
            "--progress-every",
            "8",
            "--base-seed",
            str(_PROBE_BASE_SEED),
        ]
    )
    stage_c = json.loads((out_dir / "stage_c_report.json").read_text(encoding="utf-8"))
    oracle = json.loads((out_dir / "oracle_report.json").read_text(encoding="utf-8"))
    behavior = json.loads((out_dir / "behavior_report.json").read_text(encoding="utf-8"))
    pairwise = behavior.get("pairwise_summary", behavior)
    return {
        "stage_c_pass": bool(stage_c.get("passed", False)),
        "oracle_gap": oracle.get("oracle_gap", stage_c.get("margin_advantage")),
        "margin_advantage": stage_c.get("margin_advantage"),
        "wr_advantage": stage_c.get("wr_advantage"),
        "unique_best_z": stage_c.get("unique_best_z_count", oracle.get("unique_best_z_count")),
        "behavior_pair_distance_mean": pairwise.get(
            "forced_z_behavior_pair_distance_mean",
            pairwise.get("behavior_pair_distance_mean"),
        ),
        "pairs_above_threshold": pairwise.get(
            "forced_z_behavior_pairs_above_threshold",
            pairwise.get("pairs_above_threshold"),
        ),
    }


def _decide(milestone: int, geom: dict, jsd: dict, fz: dict, prev_jsd_mean: float | None) -> str:
    jsd_mean = float(jsd.get("mean_of_cell_jsd_means") or 0.0)
    cells_jsd = int(jsd.get("cells_with_any_pair_above_0_05") or 0)
    nontie_max = float(jsd.get("max_of_cell_non_tie_argmax_disagree") or 0.0)
    nontie_cells = int(jsd.get("cells_with_non_tie_disagree_gt_0_20") or 0)
    head_l2 = float(geom.get("head_pair_l2_mean") or 0.0)
    rising = (
        prev_jsd_mean is not None
        and jsd_mean > max(prev_jsd_mean * 1.5, prev_jsd_mean + 5e-4)
        and jsd_mean > 5e-4
    )
    payoff_sep = bool(fz.get("stage_c_pass")) and (
        (fz.get("unique_best_z") or 0) >= 2
        or (fz.get("oracle_gap") is not None and float(fz["oracle_gap"]) > 0.25)
    )
    # Functional action separation required. Non-tie disagree alone is insufficient
    # when CF JSD stays near-zero (near-boundary argmax flips under peaked policies).
    action_sep = cells_jsd >= 2 or (nontie_max > 0.20 and jsd_mean > 0.01 and nontie_cells >= 2)

    if action_sep and payoff_sep:
        return "PROMOTE"

    flat_jsd = jsd_mean < 1e-3
    param_only = head_l2 > 0.02 and flat_jsd
    if milestone in (10, 15, 25) and flat_jsd and param_only and not rising:
        return "STOP_EARLY_ESCALATE"

    if rising or (payoff_sep and jsd_mean >= 1e-3) or (milestone < 25 and rising):
        return "KEEP_INVESTIGATING"

    if milestone >= 25 and flat_jsd:
        return "STOP_EARLY_ESCALATE"

    return "KEEP_INVESTIGATING"


def probe_milestone(checkpoint: Path, milestone: int, *, episodes: int = 2) -> dict[str, Any]:
    probe = _probe_dir(milestone)
    geom = _run_geometry_diag(checkpoint, probe / "geometry")
    jsd = _run_jsd_probe(checkpoint, probe / "action_jsd")
    fz = _run_forced_z(checkpoint, probe / "forced_z_eps2", episodes=episodes)
    return {"geometry": geom, "jsd_summary": jsd, "forced_z": fz}


def main() -> int:
    p = argparse.ArgumentParser(description="V6I23 pre-registered 5→25u diagnostic")
    p.add_argument("--run", action="store_true", help="Train+probe all remaining milestones")
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--milestone", type=int, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--start-checkpoint", type=str, default=str(_START_CKPT))
    p.add_argument("--forced-z-episodes", type=int, default=2)
    p.add_argument(
        "--stop-after-milestone",
        type=int,
        default=None,
        help="Optional early stop after completing this milestone (e.g. 10 for staged runs)",
    )
    args = p.parse_args()

    _ROOT.mkdir(parents=True, exist_ok=True)
    protocol_note = _ROOT / "PROTOCOL.txt"
    if not protocol_note.is_file():
        protocol_note.write_text(__doc__ or "", encoding="utf-8")

    if args.probe_only:
        if args.milestone is None or not args.checkpoint:
            print("ERROR: --probe-only requires --milestone and --checkpoint")
            return 2
        result = probe_milestone(
            Path(args.checkpoint), int(args.milestone), episodes=int(args.forced_z_episodes)
        )
        log = _load_decision_log()
        log["milestones"][str(args.milestone)] = {
            "at_utc": _utc(),
            "checkpoint": args.checkpoint,
            **result,
            "decision": _decide(
                int(args.milestone),
                result["geometry"],
                result["jsd_summary"],
                result["forced_z"],
                None,
            ),
        }
        _save_decision_log(log)
        print(json.dumps(log["milestones"][str(args.milestone)]["decision"], indent=2))
        return 0

    if not args.run:
        p.print_help()
        return 2

    start = Path(args.start_checkpoint)
    if not start.is_file():
        print(f"ERROR: start checkpoint missing: {start}")
        return 2

    log = _load_decision_log()
    log["start_checkpoint"] = str(start)
    log["updated_at_utc"] = _utc()
    current_ckpt = start
    current_u = _START_U
    prev_jsd: float | None = None
    # Seed prev_jsd from existing 5u probe if present
    five_jsd = (
        PROJECT_ROOT
        / "artifacts"
        / "v6i23_population_birth_5u_seed1"
        / "action_jsd_probe"
        / "action_jsd_probe_report.json"
    )
    if five_jsd.is_file():
        prev_jsd = float(
            json.loads(five_jsd.read_text(encoding="utf-8"))["summary"]["mean_of_cell_jsd_means"]
        )
        log["milestones"]["5"] = {
            "at_utc": _utc(),
            "checkpoint": str(start),
            "jsd_summary": json.loads(five_jsd.read_text(encoding="utf-8"))["summary"],
            "decision": "BASELINE_5U_FAIL_GATE",
            "note": "pre-registered baseline; heads moved, CF JSD flat",
        }

    for milestone in _MILESTONES:
        if str(milestone) in log["milestones"] and log["milestones"][str(milestone)].get(
            "checkpoint"
        ):
            # Resume support: skip completed milestones
            ck = Path(log["milestones"][str(milestone)]["checkpoint"])
            if ck.is_file():
                print(f"[prereg] skip completed milestone {milestone}u ({ck})")
                current_ckpt = ck
                current_u = milestone
                prev_jsd = float(
                    log["milestones"][str(milestone)]
                    .get("jsd_summary", {})
                    .get("mean_of_cell_jsd_means", prev_jsd or 0.0)
                )
                continue

        add_u = milestone - current_u
        if add_u <= 0:
            raise RuntimeError(f"bad schedule: current={current_u} milestone={milestone}")
        print(f"\n======== TRAIN {current_u}u -> {milestone}u (+{add_u} updates) ========\n")
        ckpt = _train_segment(load=current_ckpt, milestone=milestone, additional_updates=add_u)
        print(f"\n======== PROBE @ {milestone}u ========\n")
        result = probe_milestone(ckpt, milestone, episodes=int(args.forced_z_episodes))
        decision = _decide(
            milestone,
            result["geometry"],
            result["jsd_summary"],
            result["forced_z"],
            prev_jsd,
        )
        entry = {
            "at_utc": _utc(),
            "checkpoint": str(ckpt),
            "trained_from": str(current_ckpt),
            "additional_updates": add_u,
            **result,
            "decision": decision,
            "prev_jsd_mean": prev_jsd,
        }
        log["milestones"][str(milestone)] = entry
        log["verdict"] = decision
        log["updated_at_utc"] = _utc()
        _save_decision_log(log)
        print(f"[prereg] milestone {milestone}u decision: {decision}")

        current_ckpt = ckpt
        current_u = milestone
        prev_jsd = float(result["jsd_summary"].get("mean_of_cell_jsd_means") or 0.0)

        if decision == "STOP_EARLY_ESCALATE":
            print(
                "[prereg] STOP_EARLY_ESCALATE - do not add soft regularizers; "
                "escalate to four independent policies -> distill."
            )
            break
        if decision == "PROMOTE":
            print("[prereg] PROMOTE - freeze specialists; router training unblocked by birth gate.")
            break
        if args.stop_after_milestone is not None and milestone >= int(args.stop_after_milestone):
            print(f"[prereg] stop-after-milestone={args.stop_after_milestone} reached")
            break

    _save_decision_log(log)
    print(f"[prereg] final verdict: {log.get('verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
