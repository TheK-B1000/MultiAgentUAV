from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from ray.rllib.algorithms.algorithm import Algorithm

from env_factory import make_parallel_env, build_multiagent_specs
from metrics import episode_record, aggregate_records, write_episode_csv, write_summary


def run_eval(config: Dict[str, Any], checkpoint_path: str, out_dir: Path) -> Dict[str, Any]:
    eval_cfg = config.get("evaluation", {})
    episodes_per_seed = int(eval_cfg.get("episodes_per_seed", 50))
    seeds = list(eval_cfg.get("seeds", [1, 2, 3, 4, 5]))

    _, policy_mapping_fn, _ = build_multiagent_specs(config)
    algo = Algorithm.from_checkpoint(checkpoint_path)

    records: List[Dict[str, Any]] = []
    for seed in seeds:
        env = make_parallel_env(config, seed=seed)
        for ep in range(episodes_per_seed):
            obs, infos = env.reset(seed=int(seed) * 100000 + ep)
            done = False
            step_count = 0
            last_info = {}
            while not done:
                act_dict = {}
                for agent_id, agent_obs in obs.items():
                    policy_id = policy_mapping_fn(agent_id)
                    action = algo.compute_single_action(agent_obs, policy_id=policy_id, explore=False)
                    act_dict[agent_id] = action
                obs, rewards, terminated, truncated, infos = env.step(act_dict)
                step_count += 1
                done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
                if infos:
                    # any agent info has the same global_state
                    last_info = next(iter(infos.values()))
            records.append(episode_record(seed=seed, episode_idx=ep, steps=step_count, info=last_info))
        env.close()

    summary = aggregate_records(records)
    write_episode_csv(out_dir / "eval_episodes.csv", records)
    write_summary(out_dir / "summary.json", out_dir / "summary.csv", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Pyquaticus RLlib checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to RLlib checkpoint dir/file.")
    parser.add_argument("--out-dir", required=True, help="Output directory for evaluation results.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    summary = run_eval(cfg, args.checkpoint, Path(args.out_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
