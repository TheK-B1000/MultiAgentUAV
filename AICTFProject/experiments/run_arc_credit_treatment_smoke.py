#!/usr/bin/env python3
"""One-update smoke for the recurrent running-mean arc-credit treatment.

Runs exactly ONE PPO update from the protected repertoire anchor and checks
the treatment gates: arc credit enabled, running-mean baseline, finite
advantages, frozen actor unchanged, router parameters moved.

Example::

    python experiments/run_arc_credit_treatment_smoke.py \\
      --preset v6i9_arc_credit_running_mean_hardpool \\
      --checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip \\
      --device cuda --seed 1 \\
      --output artifacts/router_credit_audit/arc_treatment_smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.dump_router_rollout_audit import _build_audit_trainer  # noqa: E402
from rl.custom_ppo.diagnostics.arc_credit_smoke import (  # noqa: E402
    evaluate_arc_credit_treatment_gates,
    frozen_actor_z_fingerprint,
    router_fingerprint,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arc-credit treatment one-update smoke")
    p.add_argument("--preset", default="v6i9_arc_credit_running_mean_hardpool")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--output", default="artifacts/router_credit_audit/arc_treatment_smoke.json"
    )
    return p.parse_args()


def _router_decision_count(buffer) -> int:
    field = buffer.fields.get("router_decision_valid")
    if field is None:
        return 0
    return int(field[: buffer.pos].bool().sum().item())


def main() -> None:
    args = _parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg, resolved, env, trainer = _build_audit_trainer(
        preset=args.preset,
        checkpoint=str(checkpoint),
        device=args.device,
        seed=args.seed,
    )

    # Preset-level hard gates (fail before spending a rollout).
    if not bool(getattr(cfg, "latent_arc_credit_enabled", False)):
        raise RuntimeError("Treatment preset must set latent_arc_credit_enabled=True")
    if float(getattr(cfg, "latent_strategy_ppo_coef", 1.0) or 0.0) != 0.0:
        raise RuntimeError(
            "Treatment must zero latent_strategy_ppo_coef (remove biased critic credit)"
        )
    if str(getattr(cfg, "latent_arc_credit_baseline", "")) != "running_mean":
        raise RuntimeError("Treatment baseline must be running_mean")
    # Recurrent (GRU) and feedforward router variants are both valid for this treatment.

    try:
        frozen_before = frozen_actor_z_fingerprint(trainer.model)
        router_before = router_fingerprint(trainer.model)

        buffer = trainer.collect_rollout()
        decision_count = _router_decision_count(buffer)
        stats = trainer.update(
            buffer, total_timesteps=int(getattr(trainer, "global_step", 0) + buffer.pos)
        )

        frozen_after = frozen_actor_z_fingerprint(trainer.model)
        router_after = router_fingerprint(trainer.model)

        report = evaluate_arc_credit_treatment_gates(
            cfg=cfg,
            arc_stats=stats,
            router_decision_count=decision_count,
            frozen_hash_before=frozen_before,
            frozen_hash_after=frozen_after,
            router_hash_before=router_before,
            router_hash_after=router_after,
        )
        report["preset"] = args.preset
        report["checkpoint"] = str(checkpoint)
        report["seed"] = int(args.seed)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("[arc-treatment-smoke] gates:")
        for gate, passed in report["gates"].items():
            print(f"    {'PASS' if passed else 'FAIL'}  {gate}")
        print("[arc-treatment-smoke] telemetry:")
        for key, value in report["telemetry"].items():
            print(f"    {key} = {value}")
        print(f"[arc-treatment-smoke] wrote {out}")
        if not report["gates"]["all_passed"]:
            sys.exit(1)
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
