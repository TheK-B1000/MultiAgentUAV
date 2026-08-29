"""RASR-PPO S0/R1/R2/R3 contract builder and gated launcher.

Contract mode constructs configs only. It does not construct an environment,
open DEV/FINAL blocks, or spend a policy-training step.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RASR_DIR = ROOT / "artifacts" / "strategic_demand" / "rasrppo"
PROTOCOL = RASR_DIR / "RASR_PPO_CAUSAL_LADDER_PROTOCOL.json"
IMPLEMENTATION_GATE = RASR_DIR / "RASR_PPO_IMPLEMENTATION_GATE.json"
SCORER_QUALIFICATION = RASR_DIR / "RASR_SCORER_QUALIFICATION.json"
REGIME_QPSI = RASR_DIR / "qpsi_regime_frozen.pt"
TRAIN_SEED = 10_400_001
TRAIN_RANGE = (10_400_001, 10_400_032)
DEV_RANGE = (10_500_001, 10_500_096)
OLD_QPSI_SHA = "930051a725e55e4f14e05dfe178e5f1dc7bd8f3d7e3adeba01187958bb7417bf"

RUN_TAGS = {
    "S0": "rasrppo_s0_same_block_control_seed10400001_1m",
    "R1": "rasrppo_r1_regime_scorer_seed10400001_1m",
    "R2": "rasrppo_r2_private_critic_seed10400001_1m",
    "R3": "rasrppo_r3_directed_identity_seed10400001_1m",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(arm: str):
    """Build one ladder arm from the frozen SPPPO production treatment."""
    arm = arm.upper()
    if arm not in RUN_TAGS:
        raise ValueError(f"unknown RASR-PPO arm {arm!r}")
    from experiments.run_sppo_production import build_production_config

    cfg, parent_contract = build_production_config()
    cfg.seed = TRAIN_SEED
    cfg.total_timesteps = 1_000_000
    cfg.run_tag = RUN_TAGS[arm]
    out = RASR_DIR / arm
    cfg.checkpoint_dir = str(out / "ckpts")
    cfg.metrics_csv_path = str(out / "metrics.csv")
    cfg.episode_csv_path = str(out / "episode_rows.csv")
    cfg.periodic_checkpoint_steps = 0
    cfg.load_path = None

    cfg.rasr_regime_qpsi = arm in {"R1", "R2", "R3"}
    cfg.rasr_private_critic_heads = arm in {"R2", "R3"}
    cfg.rasr_directed_identity = arm == "R3"
    if cfg.rasr_regime_qpsi:
        cfg.rasr_regime_qpsi_path = str(REGIME_QPSI.relative_to(ROOT))
        if REGIME_QPSI.is_file():
            cfg.rasr_regime_qpsi_sha256 = _sha256(REGIME_QPSI)
    else:
        cfg.sppo_qpsi_sha256 = OLD_QPSI_SHA
    return cfg, parent_contract


def configure_rasr_live_environment(env, cfg, *, contract):
    """Install the inherited 16/0/0/16 treatment on RASR's reserved blocks."""
    from experiments.run_exp2b_specialization_preserving_compression import (
        configure_exp2b_live_environment,
    )

    return configure_exp2b_live_environment(
        env,
        cfg,
        contract=contract,
        allow_development_seed=False,
        training_seed_range=TRAIN_RANGE,
        development_seed_range=DEV_RANGE,
        manifest_key="rasrppo_protocol",
        context_label="RASR-PPO assigned-pole training construction",
    )


def _implementation_gate(*, require_policy_launch: bool) -> dict:
    """Load the gate schema used by DEV and policy launch boundaries.

    DEV collection requires PASS + ``dev_collection_authorized=true``.
    A 1M policy launch additionally requires
    ``policy_launch_authorized=true`` and a separate scorer qualification PASS.
    Contract-only mode calls neither gate.
    """
    if not IMPLEMENTATION_GATE.is_file():
        raise RuntimeError(
            "RASR-PPO GATED ACTION REFUSED: implementation gate is absent: "
            f"{IMPLEMENTATION_GATE}. Contract-only mode remains available."
        )
    gate = json.loads(IMPLEMENTATION_GATE.read_text(encoding="utf-8"))
    if gate.get("verdict") != "PASS" or gate.get("dev_collection_authorized") is not True:
        raise RuntimeError(
            "RASR-PPO GATED ACTION REFUSED: implementation gate must say "
            "verdict=PASS and dev_collection_authorized=true"
        )
    if require_policy_launch and gate.get("policy_launch_authorized") is not True:
        raise RuntimeError(
            "RASR-PPO LAUNCH REFUSED: implementation gate must additionally say "
            "policy_launch_authorized=true"
        )
    if require_policy_launch:
        if not SCORER_QUALIFICATION.is_file():
            raise RuntimeError(
                "RASR-PPO LAUNCH REFUSED: scorer qualification artifact is absent: "
                f"{SCORER_QUALIFICATION}"
            )
        qualification = json.loads(SCORER_QUALIFICATION.read_text(encoding="utf-8"))
        if qualification.get("verdict") != "PASS":
            raise RuntimeError(
                "RASR-PPO LAUNCH REFUSED: scorer qualification verdict must be PASS"
            )
    return gate


def require_dev_collection_gate() -> dict:
    """Public boundary for future DEV collectors."""
    return _implementation_gate(require_policy_launch=False)


def _launch_gate() -> dict:
    return _implementation_gate(require_policy_launch=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=tuple(RUN_TAGS))
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    cfg, parent_contract = build_config(args.arm)
    if cfg.rasr_regime_qpsi and not cfg.rasr_regime_qpsi_sha256 and args.launch:
        raise RuntimeError(
            "RASR-PPO LAUNCH REFUSED: four-regime Q_psi artifact/hash is absent"
        )
    contract = {
        "method": "RASR-PPO",
        "arm": args.arm,
        "classification": "SUMMER-COMPATIBLE EXTENSION",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "train_seed": int(cfg.seed),
        "total_timesteps": int(cfg.total_timesteps),
        "run_tag": cfg.run_tag,
        "output_dir": str((RASR_DIR / args.arm).relative_to(ROOT)),
        "scientific_flags": {
            "rasr_regime_qpsi": cfg.rasr_regime_qpsi,
            "rasr_private_critic_heads": cfg.rasr_private_critic_heads,
            "rasr_directed_identity": cfg.rasr_directed_identity,
        },
        "regime_qpsi_sha256": cfg.rasr_regime_qpsi_sha256,
        "resolved_config": dataclasses.asdict(cfg),
        "implementation_gate_path": str(IMPLEMENTATION_GATE.relative_to(ROOT)),
        "scorer_qualification_path": str(SCORER_QUALIFICATION.relative_to(ROOT)),
    }
    print(json.dumps(contract, indent=2, default=str))
    if not args.launch:
        print("\nCONTRACT ONLY. No environment, DEV collection, or training step spent.")
        return 0

    _launch_gate()
    from rl.training.orchestrator import orchestrate_training_run

    (RASR_DIR / args.arm).mkdir(parents=True, exist_ok=False)
    orchestrate_training_run(
        cfg,
        pre_rollout_env_setup=partial(
            configure_rasr_live_environment, contract=parent_contract
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
