"""
Evaluation mode separation for research safety.

Ensures evaluation runs are deterministic by disabling stochastic features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalModeConfig:
    """
    Configuration for evaluation mode.
    
    When enabled, stochastic features are disabled unless explicitly allowed.
    """
    enabled: bool = False
    allow_action_noise: bool = False  # Action execution noise
    allow_opponent_params_random: bool = False  # Opponent param sampling randomness
    allow_curriculum_random: bool = False  # Curriculum randomness
    lock_opponent_identity: bool = True  # Lock opponent identity list
    lock_phase_schedule: bool = True  # Lock phase schedule
    lock_n_agents: bool = True  # Lock number of agents
    lock_physics_stress: bool = True  # Lock physics/stress schedule flags
    
    def get_disabled_features(self) -> List[str]:
        """Get list of disabled stochastic features."""
        disabled = []
        if not self.allow_action_noise:
            disabled.append("action_noise")
        if not self.allow_opponent_params_random:
            disabled.append("opponent_params_random")
        if not self.allow_curriculum_random:
            disabled.append("curriculum_random")
        return disabled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "enabled": self.enabled,
            "allow_action_noise": self.allow_action_noise,
            "allow_opponent_params_random": self.allow_opponent_params_random,
            "allow_curriculum_random": self.allow_curriculum_random,
            "lock_opponent_identity": self.lock_opponent_identity,
            "lock_phase_schedule": self.lock_phase_schedule,
            "lock_n_agents": self.lock_n_agents,
            "lock_physics_stress": self.lock_physics_stress,
            "disabled_features": self.get_disabled_features(),
        }


class EvalModeManager:
    """Manages evaluation mode state and enforces deterministic evaluation."""
    
    def __init__(self, config: Optional[EvalModeConfig] = None) -> None:
        """
        Initialize evaluation mode manager.
        
        Args:
            config: Evaluation mode configuration; if None, creates default (disabled)
        """
        self.config = config if config is not None else EvalModeConfig()
    
    def is_enabled(self) -> bool:
        """Check if evaluation mode is enabled."""
        return self.config.enabled
    
    def should_disable_action_noise(self) -> bool:
        """Check if action noise should be disabled."""
        return self.is_enabled() and not self.config.allow_action_noise
    
    def should_disable_opponent_params_random(self) -> bool:
        """Check if opponent params randomness should be disabled."""
        return self.is_enabled() and not self.config.allow_opponent_params_random
    
    def should_disable_curriculum_random(self) -> bool:
        """Check if curriculum randomness should be disabled."""
        return self.is_enabled() and not self.config.allow_curriculum_random
    
    def should_lock_opponent_identity(self) -> bool:
        """Check if opponent identity should be locked."""
        return self.is_enabled() and self.config.lock_opponent_identity
    
    def should_lock_phase_schedule(self) -> bool:
        """Check if phase schedule should be locked."""
        return self.is_enabled() and self.config.lock_phase_schedule
    
    def should_lock_n_agents(self) -> bool:
        """Check if number of agents should be locked."""
        return self.is_enabled() and self.config.lock_n_agents
    
    def should_lock_physics_stress(self) -> bool:
        """Check if physics/stress schedule should be locked."""
        return self.is_enabled() and self.config.lock_physics_stress
    
    def get_disabled_features(self) -> List[str]:
        """Get list of disabled stochastic features."""
        return self.config.get_disabled_features()


__all__ = ["EvalModeConfig", "EvalModeManager"]
