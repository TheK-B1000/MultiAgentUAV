"""
Evaluation harness for deterministic evaluation runs.

Loads eval.yaml config and runs deterministic evaluation suite.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from rl.determinism import DeterminismManifest, SeedAuthority
from rl.eval_mode import EvalModeConfig, EvalModeManager


class EvalHarness:
    """
    Evaluation harness for running deterministic evaluation suites.
    
    Loads configuration from configs/eval.yaml and ensures deterministic execution.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize evaluation harness.
        
        Args:
            config_path: Path to eval.yaml config; if None, uses default
        """
        if config_path is None:
            config_path = os.path.join("configs", "eval.yaml")
        self.config_path = config_path
        self.config: Optional[Dict[str, Any]] = None
        self.seed_authority: Optional[SeedAuthority] = None
        self.eval_mode_manager: Optional[EvalModeManager] = None
        self.manifest: Optional[DeterminismManifest] = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load evaluation configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise RuntimeError("PyYAML not available. Install with: pip install pyyaml")
        
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Evaluation config not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        return self.config
    
    def initialize_seed_authority(self) -> SeedAuthority:
        """Initialize seed authority from config."""
        if self.config is None:
            self.load_config()
        
        seed_cfg = self.config.get("seeds", {})
        base_seed = int(seed_cfg.get("base_seed", 42))
        deterministic = bool(seed_cfg.get("deterministic", True))
        
        self.seed_authority = SeedAuthority(base_seed=base_seed, deterministic=deterministic)
        self.seed_authority.set_all_seeds()
        
        return self.seed_authority
    
    def initialize_eval_mode(self) -> EvalModeManager:
        """Initialize evaluation mode manager from config."""
        if self.config is None:
            self.load_config()
        
        eval_cfg = self.config.get("eval_mode", {})
        eval_config = EvalModeConfig(
            enabled=bool(eval_cfg.get("enabled", True)),
            allow_action_noise=bool(eval_cfg.get("allow_action_noise", False)),
            allow_opponent_params_random=bool(eval_cfg.get("allow_opponent_params_random", False)),
            allow_curriculum_random=bool(eval_cfg.get("allow_curriculum_random", False)),
            lock_opponent_identity=bool(eval_cfg.get("lock_opponent_identity", True)),
            lock_phase_schedule=bool(eval_cfg.get("lock_phase_schedule", True)),
            lock_n_agents=bool(eval_cfg.get("lock_n_agents", True)),
            lock_physics_stress=bool(eval_cfg.get("lock_physics_stress", True)),
        )
        
        self.eval_mode_manager = EvalModeManager(config=eval_config)
        
        return self.eval_mode_manager
    
    def create_manifest(
        self,
        env_schema_version: int = 1,
        vec_schema_version: int = 1,
        macro_order_hash: Optional[str] = None,
    ) -> DeterminismManifest:
        """
        Create determinism manifest for this evaluation run.
        
        Args:
            env_schema_version: Environment schema version
            vec_schema_version: Vector observation schema version
            macro_order_hash: Hash of macro order
        
        Returns:
            DeterminismManifest
        """
        if self.seed_authority is None:
            self.initialize_seed_authority()
        
        if self.config is None:
            self.load_config()
        
        # Collect opponent identifiers from config
        opponents = self.config.get("opponents", [])
        opponent_identifiers = []
        for opp in opponents:
            opponent_identifiers.append({
                "kind": str(opp.get("kind", "SCRIPTED")),
                "key": str(opp.get("key", "OP1")),
                "episodes_per_seed": int(opp.get("episodes_per_seed", 10)),
            })
        
        # Get disabled features from eval mode
        if self.eval_mode_manager is None:
            self.initialize_eval_mode()
        disabled_features = self.eval_mode_manager.get_disabled_features()
        
        self.manifest = self.seed_authority.create_manifest(
            env_schema_version=env_schema_version,
            vec_schema_version=vec_schema_version,
            macro_order_hash=macro_order_hash,
            opponent_identifiers=opponent_identifiers,
            evaluation_mode=True,
            stochastic_features_disabled=disabled_features,
        )
        
        return self.manifest
    
    def save_manifest(self, output_path: Optional[str] = None) -> str:
        """
        Save determinism manifest to JSON file.
        
        Args:
            output_path: Output path; if None, uses config output.manifest_file
        
        Returns:
            Path to saved manifest file
        """
        if self.manifest is None:
            self.create_manifest()
        
        if output_path is None:
            if self.config is None:
                self.load_config()
            output_cfg = self.config.get("output", {})
            output_path = output_cfg.get("manifest_file", "eval_manifest.json")
        
        manifest_dict = self.manifest.to_dict()
        
        with open(output_path, "w") as f:
            json.dump(manifest_dict, f, indent=2)
        
        return output_path
    
    def get_seed_list(self) -> List[int]:
        """Get list of seeds for evaluation runs."""
        if self.config is None:
            self.load_config()
        
        seed_cfg = self.config.get("seeds", {})
        seed_list = seed_cfg.get("seed_list", [42])
        
        return [int(s) for s in seed_list]
    
    def get_opponents(self) -> List[Dict[str, Any]]:
        """Get list of opponents from config."""
        if self.config is None:
            self.load_config()
        
        return list(self.config.get("opponents", []))
    
    def get_env_config(self) -> Dict[str, Any]:
        """Get environment configuration from config."""
        if self.config is None:
            self.load_config()
        
        return dict(self.config.get("environment", {}))


__all__ = ["EvalHarness"]
