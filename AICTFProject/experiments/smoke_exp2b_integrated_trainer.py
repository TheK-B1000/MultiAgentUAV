"""One-rollout EXP2B lifecycle smoke on the frozen development block."""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_exp2b_specialization_preserving_compression import (
    build_exp2b_config,
    configure_exp2b_live_environment,
)
from rl.training.orchestrator import orchestrate_training_run


def main() -> int:
    cfg, contract = build_exp2b_config()
    smoke_dir = ROOT / ".test_runs" / "exp2b_integrated_trainer"
    if smoke_dir.exists():
        raise RuntimeError(f"smoke directory must not already exist: {smoke_dir}")
    cfg.seed = 8_500_001
    cfg.total_timesteps = int(cfg.n_steps) * int(cfg.n_envs)
    cfg.run_tag = "exp2b_integrated_trainer_smoke_seed8500001"
    cfg.checkpoint_dir = str(smoke_dir / "ckpts")
    cfg.metrics_csv_path = str(smoke_dir / "metrics.csv")
    cfg.episode_csv_path = str(smoke_dir / "episode_rows.csv")
    orchestrate_training_run(
        cfg,
        pre_rollout_env_setup=partial(
            configure_exp2b_live_environment,
            contract=contract,
            allow_development_seed=True,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
