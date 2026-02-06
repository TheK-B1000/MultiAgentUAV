"""
Fixed evaluation suite: deterministic episodes on fixed opponents (no learning).

Step 2: Add a fixed eval suite to track performance on known opponents
independent of training dynamics.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.episode_result import parse_episode_result


class FixedEvalCallback(BaseCallback):
    """
    Run deterministic evaluation on fixed opponents periodically.
    Tracks performance independent of training matchmaking.
    When curriculum is provided, updates curriculum._fixed_eval_wr for fixed-eval gating.
    """
    
    def __init__(
        self,
        *,
        cfg: Any,
        eval_every_episodes: int = 500,
        episodes_per_opponent: int = 10,
        curriculum: Optional[Any] = None,
    ):
        super().__init__(verbose=1)
        self.cfg = cfg
        self.eval_every_episodes = eval_every_episodes
        self.episodes_per_opponent = episodes_per_opponent
        self.curriculum = curriculum
        
        # Fixed opponents for eval
        self.fixed_opponents: List[Tuple[str, str]] = [
            ("SCRIPTED", "OP1"),
            ("SCRIPTED", "OP2"),
            ("SCRIPTED", "OP3"),
            ("SCRIPTED", "OP3_HARD"),
            ("SPECIES", "BALANCED"),
        ]
        
        # Results storage: opponent_key -> [win_rate, episode_count]
        self.eval_results: Dict[str, List[float]] = {}
        self.eval_episode_counts: Dict[str, int] = {}
        
        # Golden snapshots (picked once, never replaced)
        self.golden_snapshots: List[str] = []
        
        # Episode tracking
        self.episode_idx = 0
        
    def _run_fixed_eval(self, model: Any) -> None:
        """Run deterministic evaluation on all fixed opponents."""
        if model is None:
            return
        
        print(f"[FixedEval] Running evaluation suite (ep={self.episode_idx})...")
        
        from rl.train_ppo import _make_env_fn
        
        for opp_kind, opp_key in self.fixed_opponents:
            opp_full_key = f"{opp_kind}:{opp_key}"
            
            # Create eval environment
            eval_env_fn = _make_env_fn(
                self.cfg,
                default_opponent=(opp_kind, opp_key),
                rank=9999,  # Different seed
            )
            eval_env = DummyVecEnv([eval_env_fn])
            
            # Set opponent
            try:
                eval_env.env_method("set_next_opponent", opp_kind, opp_key)
            except Exception:
                pass
            
            wins = 0
            losses = 0
            draws = 0
            
            # Run deterministic episodes
            for ep in range(self.episodes_per_opponent):
                obs = eval_env.reset()  # VecEnv.reset() returns obs only, not (obs, info)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, infos = eval_env.step(action)
                    if done[0]:
                        info = infos[0] if isinstance(infos, list) else infos
                        summary = parse_episode_result(info)
                        if summary:
                            if summary.blue_score > summary.red_score:
                                wins += 1
                            elif summary.blue_score < summary.red_score:
                                losses += 1
                            else:
                                draws += 1
                        break
            
            eval_env.close()
            
            total = wins + losses + draws
            win_rate = (wins / max(1, total)) * 100.0
            
            # Store results
            if opp_full_key not in self.eval_results:
                self.eval_results[opp_full_key] = []
                self.eval_episode_counts[opp_full_key] = 0
            
            self.eval_results[opp_full_key].append(win_rate)
            self.eval_episode_counts[opp_full_key] += total
            
            # Keep last 10 eval runs
            if len(self.eval_results[opp_full_key]) > 10:
                self.eval_results[opp_full_key].pop(0)
            
            print(
                f"[FixedEval] {opp_full_key}: "
                f"{wins}W/{losses}L/{draws}D "
                f"(WR={win_rate:.1f}%, avg={np.mean(self.eval_results[opp_full_key]):.1f}%)"
            )
            
            # Log to tensorboard
            self.logger.record(f"fixed_eval/{opp_full_key}_win_rate", win_rate)
            self.logger.record(f"fixed_eval/{opp_full_key}_avg_win_rate", np.mean(self.eval_results[opp_full_key]))
            
            # Update curriculum fixed-eval WR for fixed-eval gating (Phase 2 Deliverable G)
            if self.curriculum is not None and hasattr(self.curriculum, "set_fixed_eval_wr"):
                self.curriculum.set_fixed_eval_wr(opp_full_key, win_rate / 100.0)
        
        print("[FixedEval] Evaluation complete")
    
    def _on_step(self) -> bool:
        """Run fixed eval periodically."""
        # Track episode completions
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        for i, done in enumerate(dones):
            if done:
                self.episode_idx += 1
        
        # Run fixed eval periodically
        if (self.episode_idx > 0 and 
            self.episode_idx % self.eval_every_episodes == 0):
            self._run_fixed_eval(self.model)
        return True
