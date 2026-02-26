from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Pyquaticus repo layout: MultiAgentUAV/pyquaticus/pyquaticus/ (package with config, envs, ...).
# Add the pyquaticus repo root so "import pyquaticus" finds the inner package.
_run_dir = Path(__file__).resolve().parent
_project_root = _run_dir.parent
_pyquaticus_root = _project_root / "pyquaticus"
if _pyquaticus_root.is_dir() and str(_pyquaticus_root) not in sys.path:
    sys.path.insert(0, str(_pyquaticus_root))
elif str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import ray
import yaml
from ray.rllib.algorithms.ppo import PPOConfig

from env_factory import register_pyquaticus_env, build_multiagent_specs
from eval import run_eval
from league_controller import LeagueController


def _safe_cmd_output(cmd: str, cwd: str | None = None) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, cwd=cwd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _write_repro_bundle(run_dir: Path, cfg: Dict[str, Any]) -> None:
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit_main_repo": _safe_cmd_output("git rev-parse HEAD"),
        "git_commit_pyquaticus": _safe_cmd_output("git rev-parse HEAD", cwd=str(Path.cwd() / "pyquaticus")),
        "python_version": _safe_cmd_output("python --version"),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "requirements_freeze.txt").write_text(_safe_cmd_output("pip freeze"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RLlib PPO on Pyquaticus with reproducibility artifacts.")
    parser.add_argument(
        "--config",
        default="research_pyquaticus/configs/aquaticus_2v2.yaml",
        help="YAML config path.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = str(cfg.get("exp_name", "pyquaticus_exp"))
    seed = int(cfg.get("seed", 42))
    mode = str(cfg.get("mode", "CURRICULUM_LEAGUE")).upper()
    root = Path(cfg.get("results_root", "research_pyquaticus/results"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{exp_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_repro_bundle(run_dir, cfg)

    # So Ray workers can import pyquaticus (they start with a fresh interpreter).
    _pythonpath = os.pathsep.join([str(_pyquaticus_root), str(_project_root)])
    if os.environ.get("PYTHONPATH"):
        _pythonpath = _pythonpath + os.pathsep + os.environ["PYTHONPATH"]
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": _pythonpath}},
    )
    env_name = register_pyquaticus_env(cfg)
    league = LeagueController(cfg)
    policies, policy_mapping_fn, policies_to_train, callbacks_cls = build_multiagent_specs(cfg, league_controller=league)

    rllib_cfg = cfg.get("rllib", {})
    ppo = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env=env_name)
        .framework(str(rllib_cfg.get("framework", "torch")))
        .env_runners(
            num_env_runners=int(rllib_cfg.get("num_env_runners", 8)),
            num_cpus_per_env_runner=float(rllib_cfg.get("num_cpus_per_env_runner", 0.25)),
        )
        .training(
            train_batch_size=int(rllib_cfg.get("train_batch_size", 4096)),
            minibatch_size=int(rllib_cfg.get("minibatch_size", rllib_cfg.get("sgd_minibatch_size", 256))),
            num_sgd_iter=int(rllib_cfg.get("num_sgd_iter", 8)),
            gamma=float(rllib_cfg.get("gamma", 0.99)),
            lr=float(rllib_cfg.get("lr", 3e-4)),
            clip_param=float(rllib_cfg.get("clip_param", 0.2)),
            entropy_coeff=float(rllib_cfg.get("entropy_coeff", 0.01)),
            vf_loss_coeff=float(rllib_cfg.get("vf_loss_coeff", 1.0)),
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
        .callbacks(callbacks_cls)
        .debugging(seed=seed)
    )

    algo = ppo.build_algo()

    checkpoint_every = int(rllib_cfg.get("checkpoint_every_iters", 10))
    flat_iters = int(rllib_cfg.get("training_iterations", 50))
    train_log_path = run_dir / "train_log.jsonl"
    final_checkpoint = None

    with train_log_path.open("w", encoding="utf-8") as logf:
        global_iter = 0
        phases = list(cfg.get("curriculum", {}).get("phases", []))
        if not phases:
            phases = [{"name": "OP3", "train_iterations": flat_iters}]

        for phase in phases:
            phase_name = str(phase.get("name", "OP3")).upper()
            phase_iters = int(phase.get("train_iterations", flat_iters))
            eval_every = int(phase.get("eval_every_iters", 0))
            eval_episodes = int(phase.get("eval_episodes", 10))
            min_wr = float(phase.get("min_winrate", 0.0))
            min_eps = int(phase.get("min_episodes", 0))
            league.phase_idx = next((i for i, p in enumerate(league.phases) if p["name"] == phase_name), league.phase_idx)
            league.phase_episode_count = 0
            # Tag: [PPO|LEAGUE] | [PPO|CURRICULUM] | [PPO|SELF_PLAY] — same format for all modes
            if mode == "SELF_PLAY":
                ppo_tag = "[PPO|SELF_PLAY]"
            elif league.league_mode:
                ppo_tag = "[PPO|LEAGUE]"
            else:
                ppo_tag = "[PPO|CURRICULUM]"
            # Phase start: same field order as per-iter line (phase, train_iters, opp, W, L, D, elo)
            default_opp = f"SCRIPTED:{phase_name}" if not league.league_mode else ("SCRIPTED:OP3" if not league.snapshots else league.snapshots[-1])
            if mode == "SELF_PLAY":
                default_opp = league.snapshots[-1] if league.snapshots else "SCRIPTED:OP3"
            print(
                f"{ppo_tag} phase={phase_name} train_iters={phase_iters} opp={default_opp} "
                f"W={league.total_wins} | L={league.total_losses} | D={league.total_draws} elo={league.learner_rating:.1f}"
            )

            for _ in range(phase_iters):
                global_iter += 1
                result = algo.train()
                cm = result.get("custom_metrics") or {}
                # Score (blue:red) and result (WIN/LOSS/TIE) from batch means
                b, r = cm.get("blue_score"), cm.get("red_score")
                if isinstance(b, (list, tuple)) and b:
                    b = sum(b) / len(b)
                if isinstance(r, (list, tuple)) and r:
                    r = sum(r) / len(r)
                try:
                    sb = int(round(b)) if b is not None else 0
                    sr = int(round(r)) if r is not None else 0
                except (TypeError, ValueError):
                    sb, sr = 0, 0
                score_str = f"{sb}:{sr}"
                if b is not None and r is not None:
                    if b > r:
                        result_str = "WIN"
                    elif b < r:
                        result_str = "LOSS"
                    else:
                        result_str = "TIE"
                else:
                    result_str = "?"
                # Opponent: last from batch (custom_metrics may be list)
                opp_raw = cm.get("opponent_key", "?")
                opp_str = opp_raw[-1] if isinstance(opp_raw, (list, tuple)) and opp_raw else (opp_raw if isinstance(opp_raw, str) else "?")
                ep_total = result.get("episodes_total") or result.get("episodes_this_iter") or global_iter
                print(
                    f"{ppo_tag} ep={ep_total} result={result_str} score={score_str} phase={phase_name} "
                    f"opp={opp_str} W={league.total_wins} | L={league.total_losses} | D={league.total_draws} elo={league.learner_rating:.1f}"
                )
                # Log only JSON-serializable metrics (result can contain classes, ABCMeta, etc.)
                def _safe(v):
                    if v is None or isinstance(v, (bool, int, str, float)):
                        return v
                    if isinstance(v, (list, tuple)):
                        return [_safe(x) for x in v]
                    if isinstance(v, dict):
                        return {str(k): _safe(x) for k, x in v.items() if isinstance(k, (str, int))}
                    if hasattr(v, "item"):
                        return float(v.item())  # numpy scalar
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return str(v)
                result_safe = {k: _safe(v) for k, v in result.items() if isinstance(k, str)}
                log_row = {
                    "iter": global_iter,
                    "phase": phase_name,
                    "mode": mode,
                    "league_mode": bool(league.league_mode),
                    "learner_elo": league.learner_rating,
                    "result": result_safe,
                }
                logf.write(json.dumps(log_row) + "\n")

                # Optional small eval during phase for early-advance gate.
                if eval_every > 0 and (global_iter % eval_every == 0):
                    small_cfg = dict(cfg)
                    small_cfg["evaluation"] = {"episodes_per_seed": eval_episodes, "seeds": [seed]}
                    ck = algo.save(checkpoint_dir=str(run_dir / "checkpoints"))
                    temp_ckpt = ck.checkpoint.path
                    small_summary = run_eval(small_cfg, temp_ckpt, run_dir / "eval_during_train" / f"iter_{global_iter:06d}")
                    wr = float(small_summary.get("win_rate", 0.0))
                    print(f"{ppo_tag} phase={phase_name} interim_win_rate={wr:.3f} target={min_wr:.3f}")
                    if league.phase_episode_count >= min_eps and wr >= min_wr:
                        print(f"{ppo_tag} early-advance phase={phase_name} at iter={global_iter}")
                        break

                if global_iter % checkpoint_every == 0:
                    ckpt = algo.save(checkpoint_dir=str(run_dir / "checkpoints"))
                    final_checkpoint = ckpt.checkpoint.path
                    print(f"[train] checkpoint: {final_checkpoint}")
                    # Snapshot registration and self-play baseline: red uses frozen copy of learner.
                    if mode in ("CURRICULUM_LEAGUE", "SELF_PLAY"):
                        snap_key = league.add_snapshot(final_checkpoint)
                        print(f"[league] registered snapshot: {snap_key}")
                        try:
                            learner = algo.get_policy("learner_policy")
                            snapshot = algo.get_policy("snapshot_policy")
                            if learner is not None and snapshot is not None:
                                snapshot.set_weights(learner.get_weights())
                                print("[self-play] copied learner_policy -> snapshot_policy")
                        except Exception as e:
                            print(f"[self-play] weight copy skipped: {e}")

            league.maybe_advance_phase()
            league.maybe_enable_league_mode()
            if mode == "CURRICULUM_NO_LEAGUE":
                league.league_mode = False

    if not final_checkpoint:
        ckpt = algo.save(checkpoint_dir=str(run_dir / "checkpoints"))
        final_checkpoint = ckpt.checkpoint.path

    summary = run_eval(cfg, final_checkpoint, run_dir / "eval")
    (run_dir / "final_checkpoint.txt").write_text(final_checkpoint + os.linesep, encoding="utf-8")
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] run_dir={run_dir}")
    print(f"[done] final_checkpoint={final_checkpoint}")
    print(f"[done] eval_summary={json.dumps(summary)}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
