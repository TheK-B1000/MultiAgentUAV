#!/usr/bin/env python3
"""V6I26 Stage-2/3 distill + sparse-router scaffold (gated on niche PASS).

Refuses to run unless a landscape / LRO gate JSON reports niche_signal or
primary_pass. Distillation maps accepted teacher/branch checkpoints into
π(a|s,z) slots; router training is a separate follow-up command.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v6i26_lro_core import POD_TO_Z, lro_manifest, write_json  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I26 distill/route scaffold (gated)")
    p.add_argument(
        "--gate-json",
        required=True,
        help="landscape_scan.json or eval gates JSON with niche_signal/primary_pass",
    )
    p.add_argument(
        "--branches",
        nargs="+",
        default=[],
        help="Accepted branch zips as z=path (e.g. 0=.../z0.zip 1=.../z1.zip)",
    )
    p.add_argument(
        "--base-checkpoint",
        default="artifacts/v6i23_population_birth_5u_seed1/final_v6i23_population_birth_5u_seed1_2v2.zip",
    )
    p.add_argument("--output-dir", default="artifacts/v6i26_distill_seed1")
    p.add_argument(
        "--train-router",
        action="store_true",
        help="After distill, launch sparse-router training (Stage 2).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _gate_allows(payload: dict) -> tuple[bool, str]:
    acceptance = payload.get("acceptance")
    if isinstance(acceptance, dict) and "behavior_distinctness_required" in acceptance:
        return _gate_allows(acceptance)
    strategy_verdict = payload.get("phase2_strategy_verdict")
    if strategy_verdict is not None:
        if strategy_verdict == "PHASE2_STRATEGY_PASS":
            return True, "phase2_strategy_verdict=PHASE2_STRATEGY_PASS"
        return False, f"phase2_strategy_verdict={strategy_verdict}"
    if "behavior_distinctness_required" in payload:
        if bool(payload.get("accepted")) and bool(payload.get("branch_behavior_nonredundant")):
            return True, "accepted LRO branch with behavior distinctness"
        return False, "accepted branch lacks forced-z behavior distinctness"
    summary = payload.get("summary") or {}
    if bool(summary.get("niche_signal")):
        return True, "landscape niche_signal"
    if bool(payload.get("primary_pass")) or bool(payload.get("overall_pass")):
        return True, "eval primary_pass"
    if str(payload.get("decision") or "").startswith("PROMOTE"):
        return True, f"decision={payload.get('decision')}"
    return False, "no niche / promote signal in gate JSON"


def main() -> int:
    args = _parse_args()
    gate_path = Path(args.gate_json)
    if not gate_path.is_file():
        print(f"ERROR: gate JSON missing: {gate_path}")
        return 2
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    ok, reason = _gate_allows(gate)
    if not ok:
        print("REFUSED: distill/route blocked until G_available / niche PASS.")
        print(f"  gate={gate_path} reason={reason}")
        print("  Run Stage-0 landscape scan and Stage-1 LRO rounds first.")
        return 3

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    branch_map: dict[int, str] = {}
    for item in args.branches:
        if "=" not in item:
            print(f"ERROR: --branches entries must be z=path, got {item!r}")
            return 2
        z_s, path = item.split("=", 1)
        z = int(z_s)
        if not Path(path).is_file():
            print(f"ERROR: branch zip missing: {path}")
            return 2
        branch_map[z] = path

    contract = {
        "experiment": "v6i26_distill_and_route",
        "lro": lro_manifest(),
        "gate": str(gate_path),
        "gate_reason": reason,
        "base_checkpoint": args.base_checkpoint,
        "branch_map": {str(k): v for k, v in sorted(branch_map.items())},
        "pod_to_z": dict(POD_TO_Z),
        "status": "CONTRACT_ONLY" if args.dry_run or not branch_map else "READY",
        "next": [
            "Merge accepted branch weights into latent_branch_trunks/action_heads for each z",
            "Re-evaluate distilled context/phase oracle > best fixed z",
            "If PASS: train sparse q_phi with switch cost (flag/score/interval events)",
            "Headline: LRO-Summer vs K=1 / matched non-latent / end-to-end Summer",
        ],
    }
    write_json(out_dir / "distill_contract.json", contract)

    if not branch_map:
        print("Gate PASS, but no --branches provided.")
        print(f"Wrote contract: {out_dir / 'distill_contract.json'}")
        print("Re-run with --branches 0=path 1=path 2=path 3=path to materialize distill.")
        return 0

    if args.dry_run:
        print("[dry-run] distill contract only")
        return 0

    # Materialize: copy base checkpoint as distill scaffold and record branch sources.
    # Full weight-surgery (copy per-z trunks from independent BR zips) is deferred to
    # a dedicated merge helper once LRO rounds produce accepted branches.
    base = Path(args.base_checkpoint)
    if not base.is_file():
        print(f"ERROR: base checkpoint missing: {base}")
        return 2
    dest = out_dir / "distill_scaffold_base.zip"
    shutil.copy2(base, dest)
    contract["distill_scaffold"] = str(dest)
    contract["status"] = "SCAFFOLD_READY_AWAITING_WEIGHT_MERGE"
    write_json(out_dir / "distill_contract.json", contract)
    print(f"Scaffold ready at {dest}")
    print("TODO weight-merge: map each accepted BR zip's active z modules into scaffold slots.")

    if args.train_router:
        print("Router training deferred until weight-merge completes and distilled oracle PASS.")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
