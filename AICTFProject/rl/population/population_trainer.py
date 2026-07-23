"""PopulationTrainer: orchestrates K independently trained policies.

Round-robin training on a single GPU: each member does one rollout+update
cycle, then the next member runs.  This is equivalent to parallel training
with synchronized checkpointing but uses only one GPU's worth of memory.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from rl.population.population_member import PopulationMember, PopulationMemberConfig
from rl.population.pressure_rotation import rotate_pressures


# Default pressure archetypes using existing scripted opponents.
# π0: balanced calibrated hardpool (uniform OP8-OP12)
# π1: historical/easier skill level (OP3-OP7)
# π2: exploiter-heavy (weighted toward hardest OP11-OP12)
# π3: coverage-focused (initially uniform, rotated by payoff matrix)
DEFAULT_PRESSURE_CONFIGS: list[PopulationMemberConfig] = [
    PopulationMemberConfig(
        member_id=0,
        label="balanced",
        opponent_tags=("OP8", "OP9", "OP10", "OP11", "OP12"),
        opponent_weights=None,  # uniform
        seed_offset=0,
    ),
    PopulationMemberConfig(
        member_id=1,
        label="historical",
        opponent_tags=("OP3", "OP5", "OP6", "OP7"),
        opponent_weights=None,  # uniform
        seed_offset=100,
    ),
    PopulationMemberConfig(
        member_id=2,
        label="exploiter",
        opponent_tags=("OP8", "OP9", "OP10", "OP11", "OP12"),
        opponent_weights=(0.05, 0.05, 0.10, 0.40, 0.40),  # heavy on OP11/OP12
        seed_offset=200,
    ),
    PopulationMemberConfig(
        member_id=3,
        label="coverage",
        opponent_tags=("OP8", "OP9", "OP10", "OP11", "OP12"),
        opponent_weights=None,  # initially uniform, rotated by payoff matrix
        seed_offset=300,
    ),
]


@dataclass
class PopulationTrainingState:
    """Tracks global population training progress."""
    global_update: int = 0
    member_updates: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    member_steps: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    probe_results: dict[int, dict[str, Any]] = field(default_factory=dict)


class PopulationTrainer:
    """Trains K independent policies via round-robin on a single GPU.
    
    Each training cycle:
    1. For each member k in 0..K-1:
       a. Load member k's model onto GPU
       b. Collect one rollout from member k's environment
       c. Run one PPO update on member k's data
       d. Save member k's model off GPU (if memory constrained)
    2. Increment global update counter
    3. If at a probe boundary, run evaluation gates
    4. If at a rotation boundary, rotate coverage-member pressures
    """
    
    def __init__(
        self,
        source_checkpoint: Path,
        base_cfg: Any,  # PPOConfig
        member_configs: Optional[list[PopulationMemberConfig]] = None,
        device: str = "cuda",
        seed: int = 1,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.source_checkpoint = Path(source_checkpoint)
        self.base_cfg = base_cfg
        self.device = device
        self.seed = seed
        self.output_dir = Path(output_dir) if output_dir else Path("artifacts/v6i24")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        configs = member_configs or DEFAULT_PRESSURE_CONFIGS
        self.k = len(configs)
        
        print(f"[PopulationTrainer] Initializing {self.k} members from {self.source_checkpoint}")
        self.members: list[PopulationMember] = []
        for cfg in configs:
            print(f"  member[{cfg.member_id}] label={cfg.label!r} "
                  f"opponents={cfg.opponent_tags} weights={cfg.opponent_weights}")
            member = PopulationMember(
                config=cfg,
                source_checkpoint=self.source_checkpoint,
                base_cfg=self.base_cfg,
                device=device,
                base_seed=seed,
            )
            self.members.append(member)
        
        self.state = PopulationTrainingState(
            member_updates=[0] * self.k,
            member_steps=[0] * self.k,
        )
    
    def save_all_checkpoints(self, tag: str = "") -> list[Path]:
        """Save checkpoints for all members."""
        paths = []
        for member in self.members:
            suffix = f"_{tag}" if tag else ""
            path = self.output_dir / f"member_{member.member_id}_{member.label}{suffix}.pt"
            member.save_checkpoint(path)
            paths.append(path)
            print(f"  saved: {path}")
        return paths
    
    def print_population_summary(self) -> None:
        """Print parameter divergence summary across the population."""
        print("\n" + "=" * 72)
        print("Population parameter divergence summary")
        print("=" * 72)
        
        # Collect flat parameter vectors for each member
        param_vecs = []
        for member in self.members:
            params = []
            for p in member.model.parameters():
                params.append(p.detach().float().cpu().reshape(-1))
            param_vecs.append(torch.cat(params))
        
        # Pairwise L2 distances
        print("Pairwise parameter L2 distance:")
        for i in range(self.k):
            for j in range(i + 1, self.k):
                d = float((param_vecs[i] - param_vecs[j]).norm())
                print(f"  ({i}:{self.members[i].label}, {j}:{self.members[j].label}) L2={d:.4f}")
        
        # Per-member parameter norms
        print("Per-member total parameter L2 norm:")
        for i, vec in enumerate(param_vecs):
            print(f"  member[{i}] ({self.members[i].label}): {float(vec.norm()):.4f}")
        print()
    
    def __repr__(self) -> str:
        return (
            f"PopulationTrainer(k={self.k}, "
            f"source={self.source_checkpoint.name}, "
            f"global_update={self.state.global_update})"
        )

    def learn(
        self,
        total_updates_per_member: int,
        probe_at_updates: Sequence[int] = (5, 10, 25),
        rotation_interval: int = 10,
    ) -> PopulationTrainingState:
        """Train all K members via round-robin.
        
        TODO: This requires tight integration with GPUCTFVecEnv and the
        rollout collection loop. The current implementation is a scaffold
        that will be filled in when the environment interface is ready.
        
        The round-robin loop structure:
        for update in range(total_updates_per_member):
            for member in self.members:
                # 1. Set member's opponent pressure on the environment
                # 2. Collect rollout with member's model
                # 3. Compute GAE with member's critic
                # 4. Run PPO update with member's optimizer
                # 5. Update member's return normalizer
            
            # Probe gates at specified boundaries
            if update + 1 in probe_at_updates:
                self.save_all_checkpoints(tag=f"probe_{update+1}u")
                self.print_population_summary()
            
            # Rotate coverage pressures
            if (update + 1) % rotation_interval == 0:
                # TODO: Run forced-policy eval to get payoff matrix
                # TODO: Call rotate_pressures() on coverage member
                pass
            
            self.state.global_update = update + 1
        
        self.save_all_checkpoints(tag="final")
        self.print_population_summary()
        return self.state
