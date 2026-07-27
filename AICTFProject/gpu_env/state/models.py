"""Core state mixin: constructor, RNG seed management, and random helpers.

``_CoreStateMixin.__init__`` establishes all shared scalar attributes (B, Nb, Nr,
rows, cols, device, _rng, etc.) then calls ``_build_macro_targets()``,
``_alloc_state()``, and ``reset_all()`` — each of which is defined in one of the
other state sub-mixins.  Because all sub-mixins are composed into ``_StateMixin``
via inheritance, Python's MRO resolves those method calls correctly at runtime.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .._config import GPUFieldConfig
from .._constants import MAP_SET_SEED_OFFSETS


class _CoreStateMixin:
    """Establishes core scalar state and RNG; owns seed management and random helpers."""

    def __init__(self, cfg: GPUFieldConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.B = int(cfg.n_envs)
        self.Nb = int(cfg.max_blue_agents)
        self.Nr = int(cfg.max_red_agents)
        self.rows = int(cfg.map_rows)
        self.cols = int(cfg.map_cols)
        self.max_steps = int(cfg.max_decision_steps)
        self.max_sim_steps = int(cfg.max_decision_steps) * max(
            1,
            int(
                max(
                    cfg.macro_commit_go_to_ticks,
                    cfg.macro_commit_grab_ticks,
                    cfg.macro_commit_get_flag_ticks,
                    cfg.macro_commit_place_ticks,
                    cfg.macro_commit_go_home_ticks,
                )
            ),
        )
        self.score_limit = int(cfg.score_limit)
        self.dt = float(cfg.decision_interval_seconds) * 0.99
        self.max_dist = math.sqrt(float(self.cols * self.cols + self.rows * self.rows))
        self.map_set = str(cfg.map_set).lower()
        self.map_layout = str(cfg.map_layout).lower()
        self._map_seed_offset = int(MAP_SET_SEED_OFFSETS[self.map_set])

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(int(cfg.seed) + self._map_seed_offset)

        self._phase: List[str] = ["OP3"] * self.B
        self._league_mode = torch.zeros((self.B,), dtype=torch.bool, device=self.device)
        self._stress_schedule: Optional[dict] = None
        self._opponent_kind: List[str] = ["SCRIPTED"] * self.B
        self._opponent_key: List[str] = ["OP3"] * self.B
        self._phase_tensor_cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        self._red_control_mask: Optional[torch.Tensor] = None
        self._red_control_mask_dirty = True
        self._snapshot_policy_cache: Dict[str, Tuple[float, Optional[object]]] = {}
        self.rules_profile = str(cfg.rules_profile).upper()

        self.blue_scripted = False
        self._blue_style_id = 0  # 0 = no style (legacy generic blue brain); see _scripted_blue_styles.py

        self._build_macro_targets()
        self._alloc_state()
        self._init_map_pool_state()
        self.reset_all()

    def reseed(self, seed: int) -> None:
        self.cfg.seed = int(seed)
        self._rng.manual_seed(int(seed) + self._map_seed_offset)

    # ------------------------------------------------------------------
    # Random helpers — used by sub-mixins that need per-episode sampling
    # ------------------------------------------------------------------

    def _rand_uniform(self, shape: Sequence[int], lo: float, hi: float) -> torch.Tensor:
        t = torch.rand(tuple(shape), generator=self._rng, device=self.device)
        return lo + (hi - lo) * t

    def _randn(self, shape: Sequence[int]) -> torch.Tensor:
        return torch.randn(tuple(shape), generator=self._rng, device=self.device)
