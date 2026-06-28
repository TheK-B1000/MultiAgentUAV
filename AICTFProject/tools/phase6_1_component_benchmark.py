#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo import CustomPPOTrainer
from rl.train_ppo import _clamp_runtime_config_for_team_size, _resolve_initial_opponent_and_phase
from rl.training.env_factory import build_training_env


def _parse_values(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(item.strip() for item in str(value).split(',') if item.strip())
    return out


def _sync(device: str) -> None:
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def _time(device: str, fn):
    _sync(device)
    start = time.perf_counter()
    value = fn()
    _sync(device)
    return time.perf_counter() - start, value


def _apply_checkpoint_config(cfg: PPOConfig, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    try:
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as exc:
        print(f'[component] checkpoint config hydration skipped: {exc}')
        return
    raw_cfg = payload.get('cfg') if isinstance(payload, dict) else None
    if not isinstance(raw_cfg, dict):
        return
    for key, value in raw_cfg.items():
        if hasattr(cfg, key):
            try:
                setattr(cfg, key, value)
            except Exception:
                pass


def _ensure_device(cfg: PPOConfig) -> None:
    if not str(cfg.device).startswith('cuda'):
        return
    torch.zeros(1, device=cfg.device)


def _cfg(args: argparse.Namespace, env_count: int) -> PPOConfig:
    cfg = PPOConfig()
    _apply_checkpoint_config(cfg, args.checkpoint)
    cfg.seed = 42
    cfg.device = args.device
    cfg.n_envs = env_count
    cfg.n_steps = args.n_steps
    cfg.batch_size = max(1024, env_count * args.n_steps)
    cfg.n_epochs = 1
    cfg.training_telemetry_mode = 'off'
    cfg.training_events_jsonl_path = ''
    cfg.telemetry_events_jsonl_path = ''
    cfg.load_path = args.checkpoint
    cfg.checkpoint_dir = str(Path(args.output_dir) / 'tmp_checkpoints')
    cfg.mode = 'FIXED_OPPONENT'
    cfg.fixed_opponent_tag = str(args.opponent).upper()
    cfg.opponent_randomize = False
    cfg.opponent_pool = (str(args.opponent).upper(),)
    cfg.opponent_pool_weights = ()
    cfg.map_layout = str(args.map_layout).lower()
    cfg.use_latent_strategy = True
    cfg.enable_metrics_csv = False
    cfg.gpu_native_env = True
    return cfg


def _run_sample(args: argparse.Namespace, env_count: int, sample_index: int) -> list[dict[str, Any]]:
    cfg = _cfg(args, env_count)
    _ensure_device(cfg)
    max_agents = max(1, int(getattr(cfg, 'max_blue_agents', 2)))
    rows: list[dict[str, Any]] = []

    def record(component: str, duration: float, extra: dict[str, Any] | None = None) -> None:
        row = {
            'label': args.label,
            'env_count': env_count,
            'sample_index': sample_index,
            'component': component,
            'duration_seconds': duration,
            'device': args.device,
            'n_steps': args.n_steps,
            'batch_size': cfg.batch_size,
        }
        if extra:
            row.update(extra)
        rows.append(row)

    curriculum, initial_phase, initial_opponent_tag = _resolve_initial_opponent_and_phase(cfg, max_agents)
    _clamp_runtime_config_for_team_size(cfg, max_agents)
    d, env = _time(args.device, lambda: build_training_env(cfg, initial_phase=initial_phase, initial_opponent_tag=initial_opponent_tag))
    record('environment_build', d)
    d, trainer = _time(args.device, lambda: CustomPPOTrainer(
        env=env,
        cfg=cfg,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        n_epochs=1,
        batch_size=cfg.batch_size,
        value_clip_range=0.2,
        curriculum=curriculum,
    ))
    record('trainer_build', d)
    if args.checkpoint:
        d, _ = _time(args.device, lambda: trainer.load(args.checkpoint))
        record('checkpoint_load', d)
    d, rollout = _time(args.device, trainer.collect_rollout)
    record('complete_rollout', d, {'transitions': env_count * args.n_steps})
    d, _ = _time(args.device, lambda: trainer.update(rollout, total_timesteps=1000000))
    record('complete_optimization_phase', d, {'samples': env_count * args.n_steps})
    try:
        env.close()
    except Exception:
        pass
    trainer.telemetry.close_e3_step_telemetry()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--map', '--map-layout', dest='map_layout', default='map_b_split_lane')
    parser.add_argument('--opponent', default='OP9')
    parser.add_argument('--env-counts', nargs='+', default=['16', '64'])
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n-steps', type=int, default=64)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for env_count in [int(x) for x in _parse_values(args.env_counts)]:
        for sample_index in range(args.samples):
            all_rows.extend(_run_sample(args, env_count, sample_index))
    sample_path = out / f'component_benchmark_samples_{args.label}.csv'
    with sample_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in all_rows for k in row}))
        writer.writeheader()
        writer.writerows(all_rows)
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in all_rows:
        grouped.setdefault((int(row['env_count']), str(row['component'])), []).append(float(row['duration_seconds']))
    summary = {
        'label': args.label,
        'device': args.device,
        'checkpoint': args.checkpoint,
        'map_layout': args.map_layout,
        'opponent': args.opponent,
        'samples': args.samples,
        'results': [
            {
                'env_count': env_count,
                'component': component,
                'median_duration_seconds': statistics.median(values),
                'mean_duration_seconds': statistics.mean(values),
                'samples': len(values),
            }
            for (env_count, component), values in sorted(grouped.items())
        ],
    }
    summary_path = out / f'component_benchmark_summary_{args.label}.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'samples': str(sample_path), 'summary': str(summary_path)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
