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
from rl.curriculum import STRESS_BY_PHASE


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
        self._last_eval_episode = -1  # Track last episode we ran eval for (prevent duplicate runs)
        
    def _get_opponents_for_phase(self, phase: str) -> List[Tuple[str, str]]:
        """Get opponents to test based on current curriculum phase."""
        phase = str(phase).upper().strip()
        
        # Check if we're in CURRICULUM_NO_LEAGUE mode (no league-only opponents)
        is_no_league = False
        if self.cfg is not None:
            mode = str(getattr(self.cfg, "mode", "")).upper().strip()
            is_no_league = (mode == "CURRICULUM_NO_LEAGUE")
        
        if phase == "OP1":
            # OP1 phase: only test OP1
            return [("SCRIPTED", "OP1")]
        elif phase == "OP2":
            # OP2 phase: test OP1 and OP2
            return [("SCRIPTED", "OP1"), ("SCRIPTED", "OP2")]
        elif phase == "OP3":
            # OP3 phase: test opponents based on mode
            if is_no_league:
                # CURRICULUM_NO_LEAGUE: only test OP1, OP2, OP3 (no league-only opponents)
                return [("SCRIPTED", "OP1"), ("SCRIPTED", "OP2"), ("SCRIPTED", "OP3")]
            else:
                # CURRICULUM_LEAGUE or other modes: test all opponents including league-only
                return self.fixed_opponents
        else:
            # Unknown phase: default to OP1 only
            return [("SCRIPTED", "OP1")]
    
    def _run_fixed_eval(self, model: Any) -> None:
        """Run deterministic evaluation on fixed opponents matching current curriculum phase."""
        if model is None:
            return
        
        # Determine which opponents to test based on curriculum phase
        current_phase = "OP1"  # Default
        if self.curriculum is not None and hasattr(self.curriculum, "phase"):
            current_phase = str(self.curriculum.phase).upper().strip()
        
        opponents_to_test = self._get_opponents_for_phase(current_phase)
        
        print(f"[FixedEval] Running evaluation suite (ep={self.episode_idx}, phase={current_phase})...")
        print(f"[FixedEval] Testing opponents: {[f'{k}:{v}' for k, v in opponents_to_test]}")
        
        # CRITICAL: Set model to eval mode (disables dropout, batch norm updates, etc.)
        if hasattr(model, "policy") and hasattr(model.policy, "set_training_mode"):
            model.policy.set_training_mode(False)
        
        from rl.train_ppo import _make_env_fn
        
        for opp_kind, opp_key in opponents_to_test:
            opp_full_key = f"{opp_kind}:{opp_key}"
            
            # Create eval environment
            eval_env_fn = _make_env_fn(
                self.cfg,
                default_opponent=(opp_kind, opp_key),
                rank=9999,  # Different seed
            )
            eval_env = DummyVecEnv([eval_env_fn])
            
            # CRITICAL: Match training environment setup (stress schedule + phase)
            # Training sets these, so FixedEval must too for fair comparison
            try:
                eval_env.env_method("set_stress_schedule", STRESS_BY_PHASE)
            except Exception:
                pass
            try:
                eval_env.env_method("set_phase", current_phase)
            except Exception:
                pass
            
            wins = 0
            losses = 0
            draws = 0
            
            # Run deterministic episodes
            for ep in range(self.episodes_per_opponent):
                # Set opponent BEFORE reset (so it's applied during reset)
                try:
                    eval_env.env_method("set_next_opponent", opp_kind, opp_key)
                except Exception:
                    pass
                
                obs = eval_env.reset()  # VecEnv.reset() returns obs only, not (obs, info)
                
                # Verify opponent is set after reset and log for debugging
                try:
                    # Double-check opponent is set (reset should have applied it, but ensure it's correct)
                    eval_env.env_method("set_next_opponent", opp_kind, opp_key)
                    # Verify opponent was actually applied by checking game_field
                    if ep == 0 and self.verbose:  # Only print for first episode
                        try:
                            # Try to get opponent info from environment
                            opponent_info = eval_env.env_method("get_opponent_info", indices=[0])
                            if opponent_info:
                                print(f"[FixedEval] Episode 1 opponent verification: {opponent_info[0] if isinstance(opponent_info, list) else opponent_info}")
                        except Exception:
                            pass
                except Exception:
                    pass
                
                done = False
                step_count = 0
                # Use same max_decision_steps as training
                max_decision_steps = int(getattr(self.cfg, "max_decision_steps", 400))
                # Safety timeout: allow some extra steps for episodes to complete naturally
                # Training uses 400 decision steps, so allow up to 400*2 = 800 steps
                max_steps = max_decision_steps * 2
                
                while not done and step_count < max_steps:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = eval_env.step(action)
                    # DummyVecEnv returns dones as array [bool] for single env
                    if isinstance(dones, (list, np.ndarray)):
                        done = bool(dones[0])
                    else:
                        done = bool(dones)
                    step_count += 1
                    
                    if done:
                        info = infos[0] if isinstance(infos, list) else infos
                        
                        # Debug: Check what's in info
                        if self.verbose and ep == 0:  # Only print for first episode to avoid spam
                            print(f"[FixedEval] Episode {ep+1} ended at step {step_count}. "
                                  f"Info keys: {list(info.keys()) if isinstance(info, dict) else 'not dict'}")
                            if isinstance(info, dict) and "episode_result" in info:
                                ep_result = info["episode_result"]
                                print(f"[FixedEval] episode_result keys: {list(ep_result.keys()) if isinstance(ep_result, dict) else 'not dict'}")
                        
                        summary = parse_episode_result(info)
                        if summary:
                            # Verify opponent matches what we requested
                            actual_opponent = summary.opponent_key() if hasattr(summary, 'opponent_key') else None
                            expected_opponent = f"{opp_kind}:{opp_key}"
                            if actual_opponent and actual_opponent.upper() != expected_opponent.upper():
                                print(f"[FixedEval] WARNING: Opponent mismatch! Expected {expected_opponent}, got {actual_opponent}")
                            elif not actual_opponent:
                                # Fallback: try to get from summary fields
                                if hasattr(summary, 'scripted_tag') and summary.scripted_tag:
                                    actual_opponent = f"SCRIPTED:{summary.scripted_tag}"
                                elif hasattr(summary, 'species_tag') and summary.species_tag:
                                    actual_opponent = f"SPECIES:{summary.species_tag}"
                                elif hasattr(summary, 'opponent_snapshot') and summary.opponent_snapshot:
                                    actual_opponent = f"SNAPSHOT:{summary.opponent_snapshot}"
                            
                            if summary.blue_score > summary.red_score:
                                wins += 1
                            elif summary.blue_score < summary.red_score:
                                losses += 1
                            else:
                                draws += 1
                            if self.verbose and ep == 0:
                                decision_steps = summary.decision_steps if hasattr(summary, 'decision_steps') else 'N/A'
                                opponent_str = actual_opponent or expected_opponent
                                print(f"[FixedEval] Episode {ep+1} result: blue={summary.blue_score}, red={summary.red_score}, "
                                      f"opponent={opponent_str}, step_count={step_count}, decision_steps={decision_steps}, "
                                      f"terminated={info.get('terminated', 'N/A')}, truncated={info.get('truncated', 'N/A')}")
                        else:
                            # No summary - episode ended but no result parsed
                            # Try to extract scores directly from info as fallback
                            blue_score = int(info.get("blue_score", info.get("episode_result", {}).get("blue_score", 0) if isinstance(info.get("episode_result"), dict) else 0)))
                            red_score = int(info.get("red_score", info.get("episode_result", {}).get("red_score", 0) if isinstance(info.get("episode_result"), dict) else 0)))
                            if blue_score > red_score:
                                wins += 1
                            elif red_score > blue_score:
                                losses += 1
                            else:
                                draws += 1
                            if self.verbose:
                                print(f"[FixedEval] Episode {ep+1} ended but parse_episode_result returned None. "
                                      f"Using fallback: blue={blue_score}, red={red_score}, steps={step_count}")
                                if isinstance(info, dict):
                                    print(f"[FixedEval] Info dict: {list(info.keys())}")
                                    if "episode_result" in info:
                                        print(f"[FixedEval] episode_result type: {type(info['episode_result'])}, value: {info['episode_result']}")
                        break
                
                if step_count >= max_steps and not done:
                    # Episode timed out - count as draw
                    draws += 1
                    if self.verbose:
                        print(f"[FixedEval] Episode {ep+1} timed out after {max_steps} steps (max_decision_steps={max_decision_steps})")
            
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
        
        # Run fixed eval periodically (only once per episode milestone)
        # CRITICAL: Check if we've already run eval for this episode to prevent duplicate runs
        # (can happen with parallel environments where multiple episodes complete in same step)
        if (self.episode_idx > 0 and 
            self.episode_idx % self.eval_every_episodes == 0 and
            self.episode_idx != self._last_eval_episode):
            self._last_eval_episode = self.episode_idx
            self._run_fixed_eval(self.model)
        return True
