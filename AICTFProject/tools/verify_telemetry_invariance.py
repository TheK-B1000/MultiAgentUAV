#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo import CustomPPOTrainer
from rl.train_ppo import _clamp_runtime_config_for_team_size, _resolve_initial_opponent_and_phase, set_global_seed
from rl.training.env_factory import build_training_env


def _tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype.is_floating_point or b.dtype.is_floating_point:
        return torch.allclose(a, b, atol=0.0, rtol=0.0)
    return torch.equal(a, b)


def _clone_state_dict(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _clone_state_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clone_state_dict(v) for v in obj]
    return obj


def _state_equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return _tensor_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_state_equal(x, y) for x, y in zip(a, b))
    return a == b


def _rng_snapshot() -> dict[str, Any]:
    py_state = random.getstate()
    np_state = np.random.get_state()
    out = {
        "python_random_state": repr(py_state),
        "numpy_random_state": {
            "kind": np_state[0],
            "state": np_state[1].tolist(),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch_cpu_rng": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        out["torch_cuda_rng_all"] = [state.cpu().tolist() for state in torch.cuda.get_rng_state_all()]
    return out


def _run_mode(mode: str, output_dir: Path) -> dict[str, Any]:
    set_global_seed(42)
    random.seed(42)
    np.random.seed(42)
    cfg = PPOConfig()
    cfg.seed = 42
    cfg.device = "cpu"
    cfg.n_envs = 2
    cfg.n_steps = 4
    cfg.batch_size = 8
    cfg.n_epochs = 1
    cfg.training_telemetry_mode = mode
    cfg.checkpoint_dir = str(output_dir)
    cfg.mode = "FIXED_OPPONENT"
    cfg.fixed_opponent_tag = "OP3"
    cfg.use_latent_strategy = True
    cfg.enable_metrics_csv = False
    cfg.gpu_native_env = True
    cfg.run_tag = f"telemetry_invariance_{mode}"
    cfg.training_events_jsonl_path = str(output_dir / f"events_{mode}.jsonl")
    cfg.performance_samples_path = str(output_dir / f"perf_{mode}.csv")
    cfg.performance_summary_path = str(output_dir / f"summary_{mode}.json")

    max_agents = max(1, int(getattr(cfg, "max_blue_agents", 2)))
    curriculum, initial_phase, initial_opponent_tag = _resolve_initial_opponent_and_phase(cfg, max_agents)
    _clamp_runtime_config_for_team_size(cfg, max_agents)
    env = build_training_env(cfg, initial_phase=initial_phase, initial_opponent_tag=initial_opponent_tag)
    trainer = CustomPPOTrainer(
        env=env,
        cfg=cfg,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        n_epochs=1,
        batch_size=cfg.batch_size,
        value_clip_range=0.2,
        curriculum=curriculum,
    )
    rollout = trainer.collect_rollout()
    buffer = {key: tensor.detach().cpu().clone() for key, tensor in rollout.fields.items()}
    trainer.update(rollout, total_timesteps=1000)
    params = {name: p.detach().cpu().clone() for name, p in trainer.model.named_parameters()}
    optimizer = _clone_state_dict(trainer.optimizers.primary.state_dict())
    rng = _rng_snapshot()
    env.close()
    trainer.telemetry.close_e3_step_telemetry()
    return {"buffer": buffer, "params": params, "optimizer": optimizer, "rng": rng}


def _compare(reference: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    buffer_results = {key: _tensor_equal(reference["buffer"][key], other["buffer"][key]) for key in reference["buffer"]}
    param_results = {key: _tensor_equal(reference["params"][key], other["params"][key]) for key in reference["params"]}
    optimizer_equal = _state_equal(reference["optimizer"], other["optimizer"])
    rng_equal = _state_equal(reference["rng"], other["rng"])
    return {
        "buffer_all_equal": all(buffer_results.values()),
        "buffer_field_results": buffer_results,
        "model_parameters_equal": all(param_results.values()),
        "optimizer_state_equal": optimizer_equal,
        "final_rng_state_equal": rng_equal,
        "pass": all(buffer_results.values()) and all(param_results.values()) and optimizer_equal and rng_equal,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output.parent)) as td:
        root = Path(td)
        runs = {mode: _run_mode(mode, root) for mode in ("off", "basic", "full")}
    comparisons = {
        "basic_vs_off": _compare(runs["off"], runs["basic"]),
        "full_vs_off": _compare(runs["off"], runs["full"]),
    }
    buffer_keys = sorted(runs["off"]["buffer"].keys())
    category_map = {
        "action_sequence": [key for key in buffer_keys if "action" in key],
        "latent_sequence": [key for key in buffer_keys if key in {"z_idx", "z_indices", "latent_z"} or "latent" in key or key.startswith("z_")],
        "rewards": [key for key in buffer_keys if "reward" in key],
        "done_and_truncation_sequence": [key for key in buffer_keys if key in {"dones", "terminated", "truncated", "episode_starts"} or "done" in key or "trunc" in key],
        "advantages": [key for key in buffer_keys if "advantage" in key],
        "returns": [key for key in buffer_keys if "return" in key],
        "buffer_contents": buffer_keys,
        "model_parameters": sorted(runs["off"]["params"].keys()),
        "optimizer_state": ["primary"],
        "final_rng_states": sorted(runs["off"]["rng"].keys()),
    }
    result = {
        "status": "PASS" if all(c["pass"] for c in comparisons.values()) else "FAIL",
        "device": "cpu",
        "modes": ["off", "basic", "full"],
        "category_map": category_map,
        "comparisons": comparisons,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
