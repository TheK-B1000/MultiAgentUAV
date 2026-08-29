"""Fit and freeze the RASR-PPO four-regime Q_psi on Phase-0 TRAIN only.

The target, 128/32 inner split, row weighting, optimizer, early stopping,
and RNG seed are inherited unchanged from ``phase0_fit_qpsi.py``. This script
must not be used for DEV qualification.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase0_fit_qpsi import (  # noqa: E402
    TEACHERS,
    TRAIN_CFG,
    _eval_loss,
    _ordering_diagnostic,
    _tensors,
)
from experiments.phase0_scorer_common import (  # noqa: E402
    HELDOUT_SEEDS,
    INNER_FIT_SEEDS,
    INNER_VAL_SEEDS,
    TRAIN_SEEDS,
    load_split,
    sha256_file,
)
from rl.scorer.qpsi import QPsi, QPsiConfig  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "strategic_demand" / "rasrppo"
WEIGHTS = OUT_DIR / "qpsi_regime_frozen.pt"
RECORD = OUT_DIR / "RASR_REGIME_QPSI_FROZEN.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if WEIGHTS.is_file() or RECORD.is_file():
        raise SystemExit("REFUSING: RASR regime scorer output already exists")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    torch.manual_seed(TRAIN_CFG["torch_seed"])
    np.random.seed(TRAIN_CFG["torch_seed"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"RASR R1 -- FIT FOUR-REGIME Q_psi  {_now()}")
    print(f"  fit       {INNER_FIT_SEEDS[0]}..{INNER_FIT_SEEDS[-1]}")
    print(f"  inner-val {INNER_VAL_SEEDS[0]}..{INNER_VAL_SEEDS[-1]}")
    print(f"  held-out  {HELDOUT_SEEDS[0]}..{HELDOUT_SEEDS[-1]} NOT OPENED")

    started = time.perf_counter()
    fit_split = load_split(INNER_FIT_SEEDS)
    val_split = load_split(INNER_VAL_SEEDS)
    if fit_split.heldout_opened or val_split.heldout_opened:
        raise SystemExit("REFUSING: held-out data opened during RASR fitting")
    fit_data = _tensors(fit_split, device)
    val_data = _tensors(val_split, device)
    grid, vec, agent_mask, pole, a1, a2, target, weight = fit_data

    model = QPsi(QPsiConfig(n_regimes=4)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CFG["lr"],
        weight_decay=TRAIN_CFG["weight_decay"],
    )
    best = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(TRAIN_CFG["max_epochs"]):
        model.train()
        permutation = torch.randperm(len(target))
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        for start in range(0, len(target), TRAIN_CFG["batch_size"]):
            rows = permutation[start:start + TRAIN_CFG["batch_size"]]
            prediction = model(
                grid[rows].to(device),
                vec[rows].to(device),
                agent_mask[rows].to(device),
                pole[rows].to(device),
                a1[rows].to(device),
                a2[rows].to(device),
            )
            batch_weight = weight[rows].to(device)
            loss = (
                batch_weight * (prediction - target[rows].to(device)).square()
            ).sum() / batch_weight.sum().clamp_min(1e-9)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_weight = float(batch_weight.sum())
            train_loss_sum += float(loss.detach()) * total_weight
            train_weight_sum += total_weight

        val_loss = _eval_loss(model, val_data, device)
        history.append({
            "epoch": epoch,
            "train_wmse": round(train_loss_sum / max(train_weight_sum, 1e-9), 6),
            "val_wmse": round(val_loss, 6),
        })
        if val_loss < best - 1e-6:
            best = val_loss
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= TRAIN_CFG["early_stop_patience"]:
                break

    if best_state is None:
        raise RuntimeError("RASR regime scorer fit produced no checkpoint")
    model.load_state_dict(best_state)
    torch.save({"config": model.cfg.to_dict(), "state_dict": best_state}, WEIGHTS)
    weights_sha = sha256_file(WEIGHTS)

    # Train-side diagnostic only; this does not open or qualify on RASR DEV.
    from rl.custom_ppo import load_custom_ppo_policy
    import experiments.r2_learned_crossover as R2

    probe = R2.build_env(device, TRAIN_SEEDS[0])
    obs_space, action_space = probe.observation_space, probe.action_space
    probe.close()
    teachers = {
        name: load_custom_ppo_policy(
            str(path), obs_space, action_space, device=device
        )
        for name, path in TEACHERS.items()
    }
    ordering = _ordering_diagnostic(model, val_split, teachers, device)

    record = {
        "record": "RASR-PPO frozen four-regime action-conditioned scorer Q_psi",
        "status": "FROZEN_BEFORE_ANY_RASR_DEV_ACCESS",
        "utc": _now(),
        "protocol": "artifacts/strategic_demand/rasrppo/RASR_PPO_CAUSAL_LADDER_PROTOCOL.json",
        "target": "terminal blue_score - red_score win margin (unchanged)",
        "architecture": {
            "shared": ["conv", "pole_emb", "trunk", "action_emb"],
            "regime_specific": ["b", "u", "v", "P", "Q"],
            "n_regimes": 4,
            "config": model.cfg.to_dict(),
        },
        "training_config": TRAIN_CFG,
        "seed_sets": {
            "inner_fit": [INNER_FIT_SEEDS[0], INNER_FIT_SEEDS[-1], len(INNER_FIT_SEEDS)],
            "inner_val": [INNER_VAL_SEEDS[0], INNER_VAL_SEEDS[-1], len(INNER_VAL_SEEDS)],
            "held_out_not_opened": [
                HELDOUT_SEEDS[0], HELDOUT_SEEDS[-1], len(HELDOUT_SEEDS)
            ],
        },
        "held_out_seeds_opened": 0,
        "best_inner_val_weighted_mse": best,
        "train_side_ordering_diagnostic_not_dev_gate": ordering,
        "weights": {
            "path": str(WEIGHTS.relative_to(ROOT)),
            "sha256": weights_sha,
        },
        "epoch_history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"frozen -> {WEIGHTS} sha256={weights_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
