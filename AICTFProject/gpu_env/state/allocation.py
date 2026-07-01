"""State allocation dispatcher and macro-target builder.

``_AllocationMixin._alloc_state`` is the single entry point called from
``_CoreStateMixin.__init__``.  It delegates to the individual ``_alloc_*``
methods defined in the other state sub-mixins.  ``_build_macro_targets`` builds
the fixed GoTo / PlaceMine target grid used throughout the environment.
"""
from __future__ import annotations

import torch


class _AllocationMixin:
    """Owns the top-level allocation dispatcher and macro-target construction."""

    def _alloc_state(self) -> None:
        B, Nb, Nr = self.B, self.Nb, self.Nr
        dev = self.device
        f32 = torch.float32

        self._alloc_episode_state(B, Nb, Nr, dev)
        self._alloc_map_state(B, dev, f32)
        self._alloc_agent_state(B, Nb, Nr, dev, f32)
        self._alloc_flags_and_scores(B, dev, f32)
        self._alloc_runtime_buffers(B, Nb, Nr, dev, f32)
        self._alloc_mine_state(B, Nb, Nr, dev, f32)
        self._alloc_metric_buffers(B, dev, f32)
        self._alloc_navigation_telemetry_buffers(B, Nb, Nr, dev)
        self._alloc_bt_state(B, Nr, dev, f32)

    def _build_macro_targets(self) -> None:
        """Build a fixed set of 2D macro targets for GoTo/PlaceMine.

        Follows the paper's categorical-distribution variant: ~50 predetermined
        locations on a coarse 5×10 grid covering the full field.  The first
        ``n_mine_pickups`` slots are overwritten with mine-pickup coordinates so
        the policy can intentionally route to those action-relevant points.
        """
        num_x = 5
        num_y = 10
        max_x = float(max(0, self.cols - 1))
        max_y = float(max(0, self.rows - 1))

        xs = []
        ys = []
        for ix in range(num_x):
            x = max_x * (ix / float(max(1, num_x - 1)))
            for iy in range(num_y):
                y = max_y * (iy / float(max(1, num_y - 1)))
                xs.append(x)
                ys.append(y)

        targets = torch.stack(
            [
                torch.tensor(xs, dtype=torch.float32, device=self.device),
                torch.tensor(ys, dtype=torch.float32, device=self.device),
            ],
            dim=1,
        )
        pickup_positions = [
            (min(3.0, max_x), min(5.0, max_y)),
            (min(3.0, max_x), min(14.0, max_y)),
            (max(0.0, max_x - 3.0), min(5.0, max_y)),
            (max(0.0, max_x - 3.0), min(14.0, max_y)),
        ]
        n_pickup_targets = min(
            int(getattr(self.cfg, "n_mine_pickups", 4)),
            len(pickup_positions),
            int(targets.shape[0]),
        )
        for k in range(n_pickup_targets):
            targets[k, 0] = pickup_positions[k][0]
            targets[k, 1] = pickup_positions[k][1]
        self._macro_targets = targets
