"""
Phase 0: Reality-Controlled Evaluation Truth Harness.

Single entry point: run_fixed_eval_suite(checkpoint, opponents, seeds, ...).
- Frozen world: deterministic, no training wrappers, no opponent matchmaking.
- Logs per opponent: win/loss/draw, score diff, time-to-first-score, collisions (enter + per-tick).
- Writes manifest.json for paper-grade reproducibility (extend with checkpoint hash, env config hash).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rl.episode_result import parse_episode_result
from rl.determinism import SeedAuthority


def _get_git_hash() -> Optional[str]:
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return (out or "").strip() or None
    except Exception:
        return None


def _checkpoint_hash(path: str) -> Optional[str]:
    """SHA256 of checkpoint file (or first 10MB for large files)."""
    if not path or not os.path.isfile(path):
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                if f.tell() >= 10 * (1 << 20):
                    break
        return h.hexdigest()
    except Exception:
        return None


@dataclass
class OpponentEvalResult:
    """Per-opponent aggregate from fixed eval."""
    opponent_kind: str
    opponent_key: str
    wins: int
    losses: int
    draws: int
    win_rate: float
    score_diff_mean: float  # blue - red
    time_to_first_score_mean: Optional[float]
    time_to_game_over_mean: Optional[float]
    collision_events_per_episode_mean: float
    collisions_per_episode_mean: float
    collision_free_rate: float
    episodes_total: int
    seeds: List[int]


@dataclass
class EvalSuiteManifest:
    """Paper-grade eval manifest (Phase 0.2)."""
    code_version_git_hash: Optional[str] = None
    checkpoint_path: str = ""
    checkpoint_sha256: Optional[str] = None
    env_schema_version: int = 1
    vec_schema_version: int = 1
    opponents: List[Dict[str, str]] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    episodes_per_opponent_per_seed: int = 10
    realism_tier: Optional[str] = None
    deterministic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code_version_git_hash": self.code_version_git_hash,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "env_schema_version": self.env_schema_version,
            "vec_schema_version": self.vec_schema_version,
            "opponents": list(self.opponents),
            "seeds": list(self.seeds),
            "episodes_per_opponent_per_seed": self.episodes_per_opponent_per_seed,
            "realism_tier": self.realism_tier,
            "deterministic": self.deterministic,
        }


def run_fixed_eval_suite(
    checkpoint_path: str,
    opponents: Optional[List[Tuple[str, str]]] = None,
    seeds: Optional[List[int]] = None,
    episodes_per_opponent_per_seed: int = 10,
    realism_tier: Optional[str] = None,
    output_dir: Optional[str] = None,
    *,
    deterministic: bool = True,
    max_decision_steps: int = 900,
) -> Tuple[Dict[str, OpponentEvalResult], EvalSuiteManifest]:
    """
    Run fixed eval suite: frozen world, no training wrappers.

    Args:
        checkpoint_path: Path to PPO .zip.
        opponents: List of (kind, key) e.g. [("SCRIPTED","OP1"), ("SCRIPTED","OP2"), ...].
            Default: OP1, OP2, OP3, OP3_HARD, SPECIES:BALANCED, RUSHER, CAMPER.
        seeds: List of seeds; each seed runs episodes_per_opponent_per_seed per opponent.
            Default: [42].
        episodes_per_opponent_per_seed: Episodes per (opponent, seed).
        realism_tier: Optional label (e.g. "baseline", "stress_OP3").
        output_dir: Where to write results.json and manifest.json.
        deterministic: Use deterministic policy and seeding.
        max_decision_steps: Env max_decision_steps.

    Returns:
        (results_by_opponent_key, manifest).
    """
    from rl.train_ppo import PPOConfig, _make_env_fn

    if opponents is None:
        opponents = [
            ("SCRIPTED", "OP1"),
            ("SCRIPTED", "OP2"),
            ("SCRIPTED", "OP3"),
            ("SCRIPTED", "OP3_HARD"),
            ("SPECIES", "BALANCED"),
            ("SPECIES", "RUSHER"),
            ("SPECIES", "CAMPER"),
        ]
    if seeds is None:
        seeds = [42]

    cfg = PPOConfig(
        seed=seeds[0],
        max_decision_steps=max_decision_steps,
        mode="FIXED_OPPONENT",
        fixed_opponent_tag="OP3",
    )

    seed_authority = SeedAuthority(base_seed=seeds[0], deterministic=deterministic)
    seed_authority.set_all_seeds()

    model = PPO.load(checkpoint_path, device="cpu")

    results_by_key: Dict[str, List[Dict[str, Any]]] = {}

    for (opp_kind, opp_key) in opponents:
        opp_full = f"{opp_kind}:{opp_key}"
        results_by_key[opp_full] = []

        for seed in seeds:
            cfg.seed = int(seed)
            seed_authority.base_seed = int(seed)
            seed_authority.set_all_seeds()

            env_fn = _make_env_fn(cfg, default_opponent=(opp_kind, opp_key), rank=seed)
            env = DummyVecEnv([env_fn])
            try:
                env.env_method("set_next_opponent", opp_kind, opp_key)
            except Exception:
                pass

            for _ in range(episodes_per_opponent_per_seed):
                obs, _ = env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, _, dones, infos = env.step(action)
                    done = dones[0]
                    if done:
                        info = infos[0] if isinstance(infos, list) else infos
                        summary = parse_episode_result(info)
                        if summary:
                            results_by_key[opp_full].append({
                                "win": summary.blue_score > summary.red_score,
                                "loss": summary.blue_score < summary.red_score,
                                "draw": summary.blue_score == summary.red_score,
                                "blue_score": summary.blue_score,
                                "red_score": summary.red_score,
                                "score_diff": summary.blue_score - summary.red_score,
                                "time_to_first_score": summary.time_to_first_score,
                                "time_to_game_over": summary.time_to_game_over,
                                "collision_events_per_episode": getattr(summary, "collision_events_per_episode", summary.collisions_per_episode),
                                "collisions_per_episode": summary.collisions_per_episode,
                                "collision_free": summary.collision_free_episode == 1,
                            })
                        break
            env.close()

    # Aggregate per opponent
    out_results: Dict[str, OpponentEvalResult] = {}
    for opp_full, episodes in results_by_key.items():
        if not episodes:
            continue
        kind, key = opp_full.split(":", 1) if ":" in opp_full else ("SCRIPTED", opp_full)
        wins = sum(e["win"] for e in episodes)
        losses = sum(e["loss"] for e in episodes)
        draws = sum(e["draw"] for e in episodes)
        n = len(episodes)
        tfs = [e["time_to_first_score"] for e in episodes if e.get("time_to_first_score") is not None]
        tgo = [e["time_to_game_over"] for e in episodes if e.get("time_to_game_over") is not None]
        out_results[opp_full] = OpponentEvalResult(
            opponent_kind=kind,
            opponent_key=key,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=wins / max(1, n),
            score_diff_mean=float(np.mean([e["score_diff"] for e in episodes])),
            time_to_first_score_mean=float(np.mean(tfs)) if tfs else None,
            time_to_game_over_mean=float(np.mean(tgo)) if tgo else None,
            collision_events_per_episode_mean=float(np.mean([e["collision_events_per_episode"] for e in episodes])),
            collisions_per_episode_mean=float(np.mean([e["collisions_per_episode"] for e in episodes])),
            collision_free_rate=sum(1 for e in episodes if e["collision_free"]) / max(1, n),
            episodes_total=n,
            seeds=list(seeds),
        )

    manifest = EvalSuiteManifest(
        code_version_git_hash=_get_git_hash(),
        checkpoint_path=os.path.abspath(checkpoint_path),
        checkpoint_sha256=_checkpoint_hash(checkpoint_path),
        opponents=[{"kind": k, "key": v} for k, v in opponents],
        seeds=list(seeds),
        episodes_per_opponent_per_seed=episodes_per_opponent_per_seed,
        realism_tier=realism_tier,
        deterministic=deterministic,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_dict = {
            k: asdict(v) for k, v in out_results.items()
        }
        with open(os.path.join(output_dir, "fixed_eval_results.json"), "w") as f:
            json.dump(results_dict, f, indent=2)
        with open(os.path.join(output_dir, "manifest.json"), "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        print(f"[run_fixed_eval_suite] Wrote {output_dir}/fixed_eval_results.json and manifest.json")

    return out_results, manifest


def main():
    import argparse
    p = argparse.ArgumentParser(description="Phase 0: Fixed eval suite + manifest")
    p.add_argument("checkpoint", help="Path to PPO .zip")
    p.add_argument("--opponents", default="OP1,OP2,OP3,OP3_HARD,SPECIES:BALANCED,SPECIES:RUSHER,SPECIES:CAMPER",
                   help="Comma-separated: OP1,OP2,OP3, or SCRIPTED:OP1,SPECIES:RUSHER")
    p.add_argument("--seeds", default="42,123,456", help="Comma-separated seeds")
    p.add_argument("--episodes", type=int, default=10, help="Episodes per opponent per seed")
    p.add_argument("--output-dir", default="eval_suite_out", help="Output directory")
    p.add_argument("--realism-tier", default=None, help="Realism tier label")
    args = p.parse_args()

    opponents = []
    for part in args.opponents.split(","):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            opponents.append((k.strip(), v.strip()))
        else:
            opponents.append(("SCRIPTED", part))
    seeds = [int(s) for s in args.seeds.split(",")]

    results, manifest = run_fixed_eval_suite(
        args.checkpoint,
        opponents=opponents,
        seeds=seeds,
        episodes_per_opponent_per_seed=args.episodes,
        realism_tier=args.realism_tier,
        output_dir=args.output_dir,
    )
    print("\n[run_fixed_eval_suite] Per-opponent summary:")
    for k, r in results.items():
        print(f"  {k}: WR={r.win_rate:.2%} (W/L/D={r.wins}/{r.losses}/{r.draws}) "
              f"score_diff_mean={r.score_diff_mean:.2f} collision_free={r.collision_free_rate:.2%}")


if __name__ == "__main__":
    main()
