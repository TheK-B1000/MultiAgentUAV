"""
Seed authority and determinism logging for research safety.

Centralizes all RNG seeding and logs determinism configuration for reproducibility.
"""
from __future__ import annotations

import os
import random
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class RNGConfig:
    """RNG configuration for a single run."""
    python_seed: int
    numpy_seed: int
    torch_seed: Optional[int] = None
    torch_deterministic: bool = False
    action_noise_seed_offset: int = 1000  # Offset for action noise RNG
    opponent_params_seed_offset: int = 1  # Offset for opponent params RNG
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        d = {
            "python_seed": self.python_seed,
            "numpy_seed": self.numpy_seed,
            "action_noise_seed_offset": self.action_noise_seed_offset,
            "opponent_params_seed_offset": self.opponent_params_seed_offset,
        }
        if self.torch_seed is not None:
            d["torch_seed"] = self.torch_seed
            d["torch_deterministic"] = self.torch_deterministic
        return d


@dataclass
class DeterminismManifest:
    """Complete determinism manifest for an experiment run."""
    rng_config: RNGConfig
    git_commit_hash: Optional[str] = None
    git_branch: Optional[str] = None
    env_schema_version: int = 1
    vec_schema_version: int = 1
    macro_order_hash: Optional[str] = None
    opponent_identifiers: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_mode: bool = False
    stochastic_features_disabled: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "rng_config": self.rng_config.to_dict(),
            "git_commit_hash": self.git_commit_hash,
            "git_branch": self.git_branch,
            "env_schema_version": self.env_schema_version,
            "vec_schema_version": self.vec_schema_version,
            "macro_order_hash": self.macro_order_hash,
            "opponent_identifiers": list(self.opponent_identifiers),
            "evaluation_mode": bool(self.evaluation_mode),
            "stochastic_features_disabled": list(self.stochastic_features_disabled),
        }


class SeedAuthority:
    """
    Central seed authority for all RNG seeding.
    
    Single source of truth for setting seeds across:
    - Python random
    - NumPy
    - PyTorch (if available)
    - Action noise RNGs
    - Opponent params RNGs
    """
    
    def __init__(self, base_seed: int, deterministic: bool = False) -> None:
        """
        Initialize seed authority.
        
        Args:
            base_seed: Base seed for all RNGs
            deterministic: If True, enable PyTorch deterministic mode
        """
        self.base_seed = int(base_seed)
        self.deterministic = bool(deterministic)
        self._rng_config: Optional[RNGConfig] = None
        self._manifest: Optional[DeterminismManifest] = None
    
    def set_all_seeds(self) -> RNGConfig:
        """
        Set all RNG seeds using base_seed.
        
        Returns:
            RNGConfig with all seed values
        """
        # Python random
        random.seed(self.base_seed)
        
        # NumPy
        np.random.seed(self.base_seed)
        
        # PyTorch (if available)
        torch_seed = None
        torch_deterministic = False
        if TORCH_AVAILABLE:
            torch_seed = self.base_seed
            torch.manual_seed(torch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(torch_seed)
            if self.deterministic:
                torch_deterministic = True
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except AttributeError:
                    pass
        
        self._rng_config = RNGConfig(
            python_seed=self.base_seed,
            numpy_seed=self.base_seed,
            torch_seed=torch_seed,
            torch_deterministic=torch_deterministic,
        )
        
        return self._rng_config
    
    def get_action_noise_seed(self, agent_idx: int = 0) -> int:
        """Get seed for action noise RNG (per agent)."""
        return self.base_seed + RNGConfig.action_noise_seed_offset + int(agent_idx)
    
    def get_opponent_params_seed(self, episode_seed: Optional[int] = None) -> int:
        """Get seed for opponent params RNG."""
        if episode_seed is not None:
            return int(episode_seed) + RNGConfig.opponent_params_seed_offset
        return self.base_seed + RNGConfig.opponent_params_seed_offset
    
    def get_env_seed(self, rank: int = 0) -> int:
        """Get seed for environment (for vectorized envs)."""
        return self.base_seed + int(rank)
    
    def get_git_info(self) -> Dict[str, Optional[str]]:
        """Get git commit hash and branch name."""
        commit_hash = None
        branch = None
        
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                commit_hash = result.stdout.strip()
        except Exception:
            pass
        
        try:
            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                branch = result.stdout.strip()
        except Exception:
            pass
        
        return {"git_commit_hash": commit_hash, "git_branch": branch}
    
    def create_manifest(
        self,
        env_schema_version: int = 1,
        vec_schema_version: int = 1,
        macro_order_hash: Optional[str] = None,
        opponent_identifiers: Optional[List[Dict[str, Any]]] = None,
        evaluation_mode: bool = False,
        stochastic_features_disabled: Optional[List[str]] = None,
    ) -> DeterminismManifest:
        """
        Create determinism manifest for experiment logging.
        
        Args:
            env_schema_version: Environment schema version
            vec_schema_version: Vector observation schema version
            macro_order_hash: Hash of macro order (for validation)
            opponent_identifiers: List of opponent identifiers used
            evaluation_mode: Whether in evaluation mode
            stochastic_features_disabled: List of disabled stochastic features
        
        Returns:
            DeterminismManifest
        """
        if self._rng_config is None:
            self.set_all_seeds()
        
        git_info = self.get_git_info()
        
        self._manifest = DeterminismManifest(
            rng_config=self._rng_config,
            git_commit_hash=git_info["git_commit_hash"],
            git_branch=git_info["git_branch"],
            env_schema_version=int(env_schema_version),
            vec_schema_version=int(vec_schema_version),
            macro_order_hash=macro_order_hash,
            opponent_identifiers=list(opponent_identifiers) if opponent_identifiers else [],
            evaluation_mode=bool(evaluation_mode),
            stochastic_features_disabled=list(stochastic_features_disabled) if stochastic_features_disabled else [],
        )
        
        return self._manifest
    
    def get_manifest(self) -> Optional[DeterminismManifest]:
        """Get current manifest (if created)."""
        return self._manifest


__all__ = ["SeedAuthority", "RNGConfig", "DeterminismManifest"]
