"""Population member: one independently trained policy in the V6I24 population."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class PopulationMemberConfig:
    """Configuration for one population member.
    
    Each member gets a distinct opponent pressure to encourage behavioral
    diversity across the population.
    """
    member_id: int
    label: str  # e.g. "balanced", "historical", "exploiter", "coverage"
    opponent_tags: tuple[str, ...] = ("OP8", "OP9", "OP10", "OP11", "OP12")
    opponent_weights: Optional[tuple[float, ...]] = None
    map_pool: tuple[str, ...] = ("map_b_split_lane", "map_b_split_lane_v2")
    seed_offset: int = 0  # Added to base seed for per-member RNG


class PopulationMember:
    """One independently trained policy in the population.
    
    Wraps a complete ``SharedActorCentralizedCritic`` model, its own AdamW
    optimizer, return normalizer, and rollout buffer.  Each member trains
    against its own opponent distribution specified by ``PopulationMemberConfig``.
    
    The model is initialized from a competent checkpoint (weights-only load)
    but gets a fresh optimizer and normalization state.
    """
    
    def __init__(
        self,
        config: PopulationMemberConfig,
        source_checkpoint: Path,
        base_cfg: Any,  # PPOConfig
        device: str = "cuda",
        base_seed: int = 1,
    ) -> None:
        from rl.config.ppo_config import PPOConfig
        from rl.custom_ppo.policy import SharedActorCentralizedCritic
        from rl.custom_ppo.inference import (
            _torch_load_checkpoint,
            _load_model_state_dict_compat,
        )
        from rl.custom_ppo.trainer_config import build_model_kwargs
        from rl.custom_ppo.return_normalization import ReturnNormalizer
        
        self.config = config
        self.member_id = config.member_id
        self.label = config.label
        self.device = torch.device(device)
        self.seed = base_seed + config.seed_offset
        self._updates_completed = 0
        self._steps_collected = 0
        
        # Build a fresh model with NO latent strategy conditioning
        member_cfg = copy.deepcopy(base_cfg)
        member_cfg.use_latent_strategy = False
        member_cfg.seed = self.seed
        member_cfg.opponent_pool = config.opponent_tags
        if config.opponent_weights is not None:
            member_cfg.opponent_pool_weights = config.opponent_weights
        
        # Build model kwargs from config
        model_kwargs = build_model_kwargs(
            member_cfg,
            observation_space=None,  # Will be set from checkpoint
            action_space=None,
        )
        
        # Load checkpoint to get observation/action space info
        ckpt = _torch_load_checkpoint(str(source_checkpoint), map_location=device)
        ckpt_cfg = ckpt.get("cfg", {})
        
        # Build model with checkpoint's architecture but no latent strategy
        self.model = SharedActorCentralizedCritic(
            observation_space=ckpt.get("observation_space"),
            action_space=ckpt.get("action_space"),
            latent_k=0,  # No latent conditioning
            **{k: v for k, v in model_kwargs.items() 
               if k not in ("observation_space", "action_space", "latent_k")},
        )
        
        # Load weights from checkpoint (compatible load, ignoring latent-specific params)
        _load_model_state_dict_compat(
            self.model,
            ckpt["model_state_dict"],
            strict=False,  # Allow missing latent keys
        )
        self.model = self.model.to(self.device)
        
        # Fresh optimizer — independent optimization state
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(member_cfg.learning_rate),
            eps=1e-5,
        )
        
        # Fresh return normalizer
        self.return_normalizer = ReturnNormalizer(
            gamma=float(member_cfg.gamma),
            enabled=bool(member_cfg.normalize_returns),
        )
        
        # Opponent RNG — seeded per-member for reproducibility
        self.opponent_rng = np.random.default_rng(self.seed + 9000 + self.member_id)
        
        self._member_cfg = member_cfg
    
    @property
    def updates_completed(self) -> int:
        return self._updates_completed
    
    @property
    def steps_collected(self) -> int:
        return self._steps_collected
    
    def sample_opponent(self) -> str:
        """Sample an opponent tag from this member's pressure distribution."""
        tags = list(self.config.opponent_tags)
        if self.config.opponent_weights is not None:
            weights = np.array(self.config.opponent_weights, dtype=np.float64)
            weights = weights / weights.sum()  # normalize
            return str(self.opponent_rng.choice(tags, p=weights))
        return str(self.opponent_rng.choice(tags))
    
    def save_checkpoint(self, path: Path) -> None:
        """Save this member's model and optimizer state."""
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "member_id": self.member_id,
            "label": self.label,
            "updates_completed": self._updates_completed,
            "steps_collected": self._steps_collected,
            "return_norm_mean": float(self.return_normalizer.mean),
            "return_norm_var": float(self.return_normalizer.var),
            "return_norm_count": int(self.return_normalizer.count),
            "config": {
                "member_id": self.member_id,
                "label": self.label,
                "opponent_tags": list(self.config.opponent_tags),
                "opponent_weights": (
                    list(self.config.opponent_weights)
                    if self.config.opponent_weights is not None
                    else None
                ),
                "map_pool": list(self.config.map_pool),
                "seed_offset": self.config.seed_offset,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))
    
    def load_checkpoint(self, path: Path) -> None:
        """Load this member's model and optimizer state."""
        payload = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self._updates_completed = int(payload.get("updates_completed", 0))
        self._steps_collected = int(payload.get("steps_collected", 0))
    
    def __repr__(self) -> str:
        return (
            f"PopulationMember(id={self.member_id}, label={self.label!r}, "
            f"updates={self._updates_completed}, steps={self._steps_collected})"
        )
