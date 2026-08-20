"""SAPPO V1 — continue pi_A and pi_B from their 1M terminals to 1.5M with rehearsal.

Frozen protocol: artifacts/strategic_demand/STRATEGY_ANCHORED_PPO_V1_FROZEN.json
Semantics:       artifacts/strategic_demand/SAPPO_V1_LOSS_SEMANTICS_AMENDMENT.json

Reuses R1's own config construction rather than rebuilding the run from CLI
flags. A first attempt passed only checkpoint/steps/seed and was rejected by the
ruleset identity guard:

    ruleset_id mismatch: 'RULESET_V3_M1_OWN_FLAG_HOME' != 'RULESET_V2_AQUATICUS_10S'

It had also silently defaulted to opponent_pool ['OP1','OP2','OP3']. So it would
have trained the wrong opponent under the wrong ruleset. Starting from
build_r1_config() means M1, the pole, the pool and the whole G0-V5 preset are
inherited from the run being continued, and only the intended fields move.

The ONLY intended differences from R1 are:

    run_tag / artifact paths   (a continuation writes elsewhere)
    load_path                  (resume the 1M terminal)
    total_timesteps            (checkpoint_step + 500k)
    sappo_anchor_*             (the intervention under test)

Run:  python experiments/run_sappo_continuation.py --policy ALL
      python experiments/run_sappo_continuation.py --policy ALL --dry-run
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_r1_repertoire_training import (  # noqa: E402
    build_r1_config, configure_r1_live_environment,
)
from rl.training.orchestrator import orchestrate_training_run  # noqa: E402

SD = ROOT / "artifacts/strategic_demand"
R1 = SD / "r1_training"
OUT = SD / "sappo_continuation"
DEMO = SD / "sappo_demonstrations"
PROTOCOL = SD / "STRATEGY_ANCHORED_PPO_V1_FROZEN.json"
SEMANTICS = SD / "SAPPO_V1_LOSS_SEMANTICS_AMENDMENT.json"
COMPAT = SD / "sappo_live_compat.json"

ADDITIONAL_STEPS = 500_000
LAMBDA_ANCHOR = 0.10
CADENCE = 4

SPECS = {
    "A": {"resume": R1 / "r1_pi_A_specialist_seed7100001/ckpts/final_r1_pi_A_specialist_seed7100001.zip",
          "anchor": "anchor_A_train.npz", "label": "pi_A_specialist"},
    "B": {"resume": R1 / "r1_pi_B_specialist_seed7200001/ckpts/final_r1_pi_B_specialist_seed7200001.zip",
          "anchor": "anchor_B_train.npz", "label": "pi_B_specialist"},
}

# Fields a CONTINUATION is allowed to move relative to its R1 parent config.
ALLOWED_CONTINUATION_DIFFS = {
    "run_tag", "checkpoint_dir", "metrics_csv_path", "episode_csv_path",
    "load_path", "total_timesteps",
    "sappo_anchor_dataset", "sappo_anchor_lambda", "sappo_anchor_cadence",
    "sappo_anchor_batch_size",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def preflight() -> dict:
    for f, why in ((PROTOCOL, "SAPPO freeze"), (SEMANTICS, "semantics amendment"),
                   (COMPAT, "live compatibility check"),
                   (DEMO / "manifest.json", "demonstration manifest")):
        if not f.is_file():
            raise SystemExit(f"REFUSING: {why} missing: {f}")
    compat = json.loads(COMPAT.read_text(encoding="utf-8"))
    if not compat.get("ALL_PASS"):
        raise SystemExit("REFUSING: live compatibility check did not pass")
    man = json.loads((DEMO / "manifest.json").read_text(encoding="utf-8"))
    if not man.get("all_files_share_run_id"):
        raise SystemExit("REFUSING: demonstration files do not share one run_id")
    return man


def build_continuation_config(policy: str, man: dict):
    """R1 config for this policy, moved ONLY along the continuation axes."""
    cfg, contract = build_r1_config(policy)
    parent = dataclasses.asdict(cfg)

    spec = SPECS[policy]
    ckpt = spec["resume"]
    if not ckpt.is_file():
        raise SystemExit(f"REFUSING: R1 terminal checkpoint missing: {ckpt}")
    if not ckpt.name.startswith("final_"):
        raise SystemExit(f"REFUSING: not a terminal checkpoint: {ckpt.name}")
    anchor = DEMO / spec["anchor"]
    if not anchor.is_file():
        raise SystemExit(f"REFUSING: anchor dataset missing: {anchor}")

    run_tag = f"sappo_{spec['label']}_1p5M_seed{int(cfg.seed)}"
    art = OUT / run_tag
    if (art / "result_summary.json").is_file():
        raise SystemExit(f"REFUSING: {art/'result_summary.json'} exists; "
                         "continuation already run, no re-run and no checkpoint shopping")

    cfg.run_tag = run_tag
    cfg.checkpoint_dir = str(art / "ckpts")
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")
    cfg.load_path = str(ckpt)
    # Resolved after load: total_timesteps = checkpoint_step + ADDITIONAL_STEPS.
    cfg.additional_steps = ADDITIONAL_STEPS
    cfg.total_timesteps = int(cfg.total_timesteps) + ADDITIONAL_STEPS
    cfg.sappo_anchor_dataset = str(anchor)
    cfg.sappo_anchor_lambda = LAMBDA_ANCHOR
    cfg.sappo_anchor_cadence = CADENCE

    child = dataclasses.asdict(cfg)
    diff = {k for k in child if child[k] != parent.get(k)}
    unexpected = sorted(diff - ALLOWED_CONTINUATION_DIFFS)
    if unexpected:
        raise SystemExit(
            f"REFUSING: continuation changed unexpected config fields {unexpected}. "
            "Only run identity, resume, budget and anchor fields may move.")

    manifest = {
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "semantics": str(SEMANTICS.relative_to(ROOT)),
        "policy": policy, "run_tag": run_tag,
        "resume_from": str(ckpt.relative_to(ROOT)),
        "resume_checkpoint_sha256_16": _sha(ckpt),
        "additional_steps": ADDITIONAL_STEPS,
        "cumulative_target": cfg.total_timesteps,
        "seed": int(cfg.seed),
        "own_flag_home_required_to_score": bool(cfg.own_flag_home_required_to_score),
        "opponent_pool": list(cfg.opponent_pool),
        "anchor_dataset": str(anchor.relative_to(ROOT)),
        "anchor_dataset_sha256_16": _sha(anchor),
        "demo_run_id": man.get("run_id"),
        "lambda_anchor": LAMBDA_ANCHOR, "cadence": CADENCE,
        "changed_vs_R1_parent": sorted(diff),
        "scored_checkpoint": "1.5M terminal only; no checkpoint shopping",
        "utc": _now(),
    }
    return cfg, contract, art, manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("A", "B", "ALL"), required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    man = preflight()
    todo = ["A", "B"] if a.policy == "ALL" else [a.policy]
    print(f"SAPPO V1 CONTINUATION  {_now()}")
    print(f"  additional steps {ADDITIONAL_STEPS:,}  (1.0M -> ~1.5M cumulative)")
    print(f"  anchor           lambda={LAMBDA_ANCHOR}, cadence 1:{CADENCE}, no decay")
    print(f"  demo run_id      {man.get('run_id')}")
    print(f"  generalist       NOT continued")

    for pol in todo:
        cfg, contract, art, manifest = build_continuation_config(pol, man)
        print(f"\n  [{pol}] {cfg.run_tag}")
        print(f"      resume  {Path(cfg.load_path).name}  sha={manifest['resume_checkpoint_sha256_16']}")
        print(f"      pole    pool={manifest['opponent_pool']}  M1={manifest['own_flag_home_required_to_score']}")
        print(f"      anchor  {Path(cfg.sappo_anchor_dataset).name}  sha={manifest['anchor_dataset_sha256_16']}")
        print(f"      budget  -> {cfg.total_timesteps:,}")
        print(f"      changed vs R1 parent: {manifest['changed_vs_R1_parent']}")
        if a.dry_run:
            continue
        art.mkdir(parents=True, exist_ok=True)
        (art / "sappo_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                 encoding="utf-8")
        orchestrate_training_run(
            cfg,
            pre_rollout_env_setup=partial(configure_r1_live_environment,
                                          policy=pol, config_contract=contract,
                                          # continuation declares its own budget;
                                          # the drift guard stays active
                                          expected_steps=int(cfg.total_timesteps)),
        )
        print(f"      done -> {art}")

    if a.dry_run:
        print("\nDRY RUN -- nothing launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
