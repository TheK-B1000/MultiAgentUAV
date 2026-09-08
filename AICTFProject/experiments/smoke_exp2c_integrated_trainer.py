"""One-rollout EXP2C lifecycle smoke on frozen development seed 8800001."""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_exp2c_mode_specific_actor_compression import (
    build_exp2c_config,
    configure_exp2c_live_environment,
)
from rl.training.orchestrator import orchestrate_training_run


def main() -> int:
    cfg, contract = build_exp2c_config()
    smoke_dir = ROOT / ".test_runs" / "exp2c_integrated_trainer"
    if smoke_dir.exists():
        raise RuntimeError(f"smoke directory must not already exist: {smoke_dir}")
    cfg.seed = 8_800_001
    cfg.total_timesteps = int(cfg.n_steps) * int(cfg.n_envs)
    cfg.run_tag = "exp2c_integrated_trainer_smoke_seed8800001"
    cfg.checkpoint_dir = str(smoke_dir / "ckpts")
    cfg.metrics_csv_path = str(smoke_dir / "metrics.csv")
    cfg.episode_csv_path = str(smoke_dir / "episode_rows.csv")
    orchestrate_training_run(
        cfg,
        pre_rollout_env_setup=partial(
            configure_exp2c_live_environment,
            contract=contract,
            allow_development_seed=True,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
