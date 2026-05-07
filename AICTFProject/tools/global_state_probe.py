"""Freeze-and-probe: how well does the CTDE global_state predict strategy-relevant scalars?

Uses a frozen checkpoint and on-policy rollouts (same as ``critic_ceiling``). For each
decision, logs ``global_state`` (pre-action) with targets derived from post-step ``info``:

- ``score_diff``: (blue_score - red_score) / score_limit
- ``time_frac``: decision_steps / max_decision_steps

High held-out R² means the summary exposes score pressure and clock well enough for
linear/nonlinear probes; low R² still points to an information bottleneck.

Temporal alignment: each row pairs **pre-action** ``global_state`` (same as the rollout buffer)
with **post-step** ``info`` targets. That is usually fine for slowly changing targets like
``score_diff`` and ``time_frac``. For **event-sharp** labels (e.g. carrier bits that flip in
one step), use same-step state–info pairing or shift indices — do not reuse this target
definition blindly.

Example:
    python tools/global_state_probe.py checkpoints/2v2/final_ppo_latent_fixed_op3_2v2.zip --rollouts 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ceiling_path = ROOT / "tools" / "critic_ceiling.py"
_spec = importlib.util.spec_from_file_location("critic_ceiling", _ceiling_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load {_ceiling_path}")
_ceiling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ceiling)
_cfg_from_checkpoint = _ceiling._cfg_from_checkpoint
_make_trainer = _ceiling._make_trainer
_fit_models = _ceiling._fit_models


def _collect_probe_rows(trainer: object, rollouts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict] = []
    setattr(trainer, "_global_state_probe_rows", rows)
    try:
        for _ in range(max(1, int(rollouts))):
            trainer.collect_rollout()
    finally:
        delattr(trainer, "_global_state_probe_rows")
    if not rows:
        raise RuntimeError("no probe rows collected")
    x = np.stack([r["global_state"] for r in rows], axis=0).astype(np.float32, copy=False)
    y_s = np.asarray([r["score_diff"] for r in rows], dtype=np.float32)
    y_t = np.asarray([r["time_frac"] for r in rows], dtype=np.float32)
    return x, y_s, y_t


def _print_target(name: str, y: np.ndarray) -> None:
    print(f"[probe] {name}: mean={float(np.mean(y)):.4f} std={float(np.std(y)):.4f} min={float(np.min(y)):.4f} max={float(np.max(y)):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe global_state -> score_diff / time_frac R².")
    parser.add_argument("model", type=Path, help="Custom PPO checkpoint path.")
    parser.add_argument("--rollouts", type=int, default=2, help="Number of consecutive collect_rollout passes.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    model_path = args.model
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    cfg = _cfg_from_checkpoint(model_path, device=args.device)
    trainer, env = _make_trainer(cfg, model_path)
    try:
        x, y_score, y_time = _collect_probe_rows(trainer, int(args.rollouts))
    finally:
        env.close()

    print(f"samples={x.shape[0]} global_state_dim={x.shape[1]} rollouts={int(args.rollouts)}")
    _print_target("score_diff_norm", y_score)
    _print_target("time_frac", y_time)

    for target_name, y in ("score_diff_norm", y_score), ("time_frac", y_time):
        print(f"\n=== target={target_name} ===")
        print("model,train_r2,test_r2")
        best = ("", float("-inf"))
        for name, train_r2, test_r2 in _fit_models(x, y, seed=int(args.seed), test_size=float(args.test_size)):
            print(f"{name},{train_r2:.4f},{test_r2:.4f}")
            if test_r2 > best[1]:
                best = (name, test_r2)
        print(f"best_test_r2: {best[1]:.4f} ({best[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
