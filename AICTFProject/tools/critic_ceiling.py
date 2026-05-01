"""Estimate the supervised predictability ceiling for the centralized critic.

This collects one or more frozen PPO rollouts, then fits fresh supervised
regressors from ``global_state -> return`` with a held-out split. The held-out
R^2 is an empirical ceiling for any critic that only sees the 14-d global state.

Example:
    python tools/critic_ceiling.py checkpoints/2v2/final_latent_fix_v4_retnorm_vf256_2v2.zip
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from rl.custom_ppo import CustomPPOTrainer, _torch_load_checkpoint
from rl.curriculum import phase_from_tag
from rl.stress_schedule import STRESS_BY_PHASE
from rl.train_ppo import PPOConfig, _apply_initial_opponent_params


def _cfg_from_checkpoint(path: Path, *, device: str | None) -> PPOConfig:
    payload = _torch_load_checkpoint(str(path), map_location="cpu")
    raw_cfg = payload.get("cfg") if isinstance(payload, dict) else {}
    valid = {f.name for f in fields(PPOConfig)}
    cfg_kwargs = {k: v for k, v in dict(raw_cfg or {}).items() if k in valid}
    cfg = PPOConfig(**cfg_kwargs)
    if device is not None:
        cfg.device = device
    cfg.enable_progress_bar = False
    cfg.enable_metrics_csv = False
    cfg.metrics_csv_path = None
    cfg.episode_csv_path = None
    return cfg


def _make_trainer(cfg: PPOConfig, model_path: Path) -> tuple[CustomPPOTrainer, GPUCTFVecEnv]:
    max_agents = max(1, int(getattr(cfg, "max_blue_agents", 2)))
    gpu_cfg = GPUFieldConfig(
        n_envs=max(1, int(cfg.n_envs)),
        n_agents_per_team=max_agents,
        map_set=str(getattr(cfg, "map_set", "train")).lower(),
        max_decision_steps=max(1, int(cfg.max_decision_steps)),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=str(cfg.device),
        seed=int(cfg.seed),
    )
    env = GPUCTFVecEnv(gpu_cfg)
    env.env_method("set_stress_schedule", STRESS_BY_PHASE)
    env.env_method("set_dynamics_config", {"rules_profile": "OURS"})
    opponent_tag = str(getattr(cfg, "fixed_opponent_tag", "OP3")).upper()
    phase = phase_from_tag(opponent_tag)
    env.env_method("set_phase", phase)
    env.env_method("set_next_opponent", "SCRIPTED", opponent_tag)
    _apply_initial_opponent_params(env, cfg, gpu_cfg, opponent_tag=opponent_tag, phase=phase)

    trainer = CustomPPOTrainer(
        env,
        cfg,
        learning_rate=float(cfg.learning_rate),
        clip_range=float(cfg.clip_range),
        ent_coef=float(cfg.ent_coef),
        n_epochs=int(cfg.n_epochs),
        batch_size=int(cfg.batch_size),
        value_clip_range=getattr(cfg, "clip_range_vf", cfg.clip_range),
    )
    trainer.load(str(model_path))
    return trainer, env


def _collect_dataset(trainer: CustomPPOTrainer, rollouts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    for idx in range(max(1, int(rollouts))):
        buffer = trainer.collect_rollout()
        length = int(buffer.pos)
        x = buffer.fields["global_state"][:length].detach().cpu().numpy().reshape(-1, buffer.fields["global_state"].shape[-1])
        y = buffer.fields["returns"][:length].detach().cpu().numpy().reshape(-1)
        v = buffer.fields["values"][:length].detach().cpu().numpy().reshape(-1)
        xs.append(x.astype(np.float32, copy=False))
        ys.append(y.astype(np.float32, copy=False))
        vs.append(v.astype(np.float32, copy=False))
        print(f"[critic_ceiling] collected rollout {idx + 1}/{rollouts}: samples={x.shape[0]}")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(vs, axis=0)


def _fit_models(x: np.ndarray, y: np.ndarray, *, seed: int, test_size: float) -> list[tuple[str, float, float]]:
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(test_size),
        random_state=int(seed),
        shuffle=True,
    )
    models: list[tuple[str, Any]] = [
        ("mean_baseline", DummyRegressor(strategy="mean")),
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        (
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=31, learning_rate=0.05, random_state=int(seed)),
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=int(seed),
            ),
        ),
        (
            "mlp_128x128",
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(128, 128),
                    activation="relu",
                    early_stopping=True,
                    max_iter=500,
                    random_state=int(seed),
                ),
            ),
        ),
    ]

    results: list[tuple[str, float, float]] = []
    for name, model in models:
        model.fit(x_train, y_train)
        train_r2 = float(r2_score(y_train, model.predict(x_train)))
        test_r2 = float(r2_score(y_test, model.predict(x_test)))
        results.append((name, train_r2, test_r2))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate global_state -> return supervised R^2 ceiling.")
    parser.add_argument("model", type=Path, help="Custom PPO checkpoint path.")
    parser.add_argument("--rollouts", type=int, default=1, help="Number of PPO rollouts to collect.")
    parser.add_argument("--device", type=str, default=None, help="Override checkpoint device.")
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
        x, y, v = _collect_dataset(trainer, args.rollouts)
    finally:
        env.close()

    print(f"samples: {x.shape[0]} | global_state_dim: {x.shape[1]}")
    print(f"returns: mean={float(np.mean(y)):.4f} std={float(np.std(y)):.4f} min={float(np.min(y)):.4f} max={float(np.max(y)):.4f}")
    critic_ev = trainer._explained_variance(torch.as_tensor(v), torch.as_tensor(y))
    print(f"checkpoint_critic_ev_on_collected_rollout: {critic_ev:.4f}")
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
