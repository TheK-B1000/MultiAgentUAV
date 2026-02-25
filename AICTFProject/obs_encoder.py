from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, TYPE_CHECKING

import gymnasium
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class ObsEncoderSpec:
    in_channels: int = 7
    height: int = 20
    width: int = 20
    latent_dim: int = 128
    pool_hw: Tuple[int, int] = (2, 2)


class ObsEncoder(nn.Module):
    """
    CNN encoder for spatial observations (default: 7-channel CTF map).

    Accepted input shapes:
      - [C, H, W]              (unbatched, channels-first)
      - [B, C, H, W]           (batched, channels-first)
      - [H, W, C]              (unbatched, channels-last)
      - [B, H, W, C]           (batched, channels-last)

    Output:
      - latent: [B, latent_dim]

    Design goals (research-grade RL):
      - No BatchNorm (unstable with non-iid on-policy batches)
      - Orthogonal init for Conv/Linear (PPO/MAPPO-friendly)
      - Fixed latent size via AdaptiveAvgPool2d (robust to runtime H/W drift)
      - Strict shape validation (fail fast, no silent permute mistakes)
    """

    def __init__(
        self,
        in_channels: int = 7,
        height: int = 20,
        width: int = 20,
        latent_dim: int = 128,
        pool_hw: Tuple[int, int] = (2, 2),   # ✅ default per your request
    ) -> None:
        super().__init__()

        ph, pw = int(pool_hw[0]), int(pool_hw[1])
        if ph <= 0 or pw <= 0:
            raise ValueError(f"pool_hw must be positive, got {pool_hw}")

        self.spec = ObsEncoderSpec(
            in_channels=int(in_channels),
            height=int(height),
            width=int(width),
            latent_dim=int(latent_dim),
            pool_hw=(ph, pw),
        )

        C = self.spec.in_channels

        # Feature extractor: simple + stable
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # ✅ Force a stable pooled spatial size
        self.pool = nn.AdaptiveAvgPool2d((ph, pw))
        self._conv_out_hw: Tuple[int, int] = (ph, pw)

        # Compute flat dim robustly
        with torch.no_grad():
            dummy = torch.zeros(1, C, self.spec.height, self.spec.width)
            y = self.pool(self.conv(dummy))
            flat_dim = int(y.shape[1] * y.shape[2] * y.shape[3])
        self._flat_dim: int = flat_dim

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_dim, self.spec.latent_dim),
            nn.ReLU(inplace=True),
        )

        self.apply(self._init_weights)

    # -----------------------------
    # Public metadata
    # -----------------------------
    @property
    def flat_dim(self) -> int:
        return int(self._flat_dim)

    @property
    def conv_out_hw(self) -> Tuple[int, int]:
        return self._conv_out_hw

    # -----------------------------
    # Init
    # -----------------------------
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # -----------------------------
    # Shape handling
    # -----------------------------
    def _ensure_bchw(self, obs: Union[torch.Tensor, "np.ndarray"]) -> torch.Tensor:
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs)

        C = self.spec.in_channels

        if obs.dim() == 3:
            # [C,H,W]
            if obs.shape[0] == C:
                obs = obs.unsqueeze(0)
            # [H,W,C]
            elif obs.shape[-1] == C:
                obs = obs.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(
                    f"Unbatched obs must be [C,H,W] or [H,W,C] with C={C}, got {tuple(obs.shape)}"
                )

        elif obs.dim() == 4:
            # [B,C,H,W]
            if obs.shape[1] == C:
                pass
            # [B,H,W,C]
            elif obs.shape[-1] == C:
                obs = obs.permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    f"Batched obs must be [B,C,H,W] or [B,H,W,C] with C={C}, got {tuple(obs.shape)}"
                )
        else:
            raise ValueError(f"ObsEncoder expects 3D or 4D tensor, got dim={obs.dim()} shape={tuple(obs.shape)}")

        if obs.dim() != 4 or obs.shape[1] != C:
            raise ValueError(f"Internal error: expected [B,{C},H,W], got {tuple(obs.shape)}")

        return obs

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, obs: Union[torch.Tensor, "np.ndarray"]) -> torch.Tensor:
        x = self._ensure_bchw(obs).float().contiguous()
        x = self.conv(x)
        x = self.pool(x)
        z = self.fc(x)
        return z


class CTFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible features extractor for the Dict observation space
    produced by BatchedCTFCore / GPUCTFVecEnv.

    Handles the 5D grid tensor [B, N_agents, C, H, W] by folding agents into
    the batch dimension for the shared CNN, then concatenating per-agent CNN
    latents with the vector obs, masking dead agents, and projecting through
    a final linear head.

    Expected observation keys (set by GPUCTFVecEnv):
        grid        [B, N, C, H, W]   spatial per-agent obs
        vec         [B, N, vec_dim]    normalised scalars per agent
        agent_mask  [B, N]             1.0 alive / 0.0 dead
        mask        [B, flat]          action-validity mask
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_latent: int = 128,
        features_dim: int = 256,
        pool_hw: Tuple[int, int] = (2, 2),
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)

        grid_shape = observation_space["grid"].shape    # (N, C, H, W)
        vec_shape = observation_space["vec"].shape       # (N, vec_dim)
        mask_shape = observation_space["mask"].shape      # (flat_mask,)

        self.n_agents = int(grid_shape[0])
        in_channels = int(grid_shape[1])
        h, w = int(grid_shape[2]), int(grid_shape[3])
        vec_dim = int(vec_shape[1])
        flat_mask = int(mask_shape[0])

        self.cnn = ObsEncoder(
            in_channels=in_channels,
            height=h,
            width=w,
            latent_dim=cnn_latent,
            pool_hw=pool_hw,
        )

        per_agent = cnn_latent + vec_dim
        combined = self.n_agents * per_agent + flat_mask

        self.head = nn.Sequential(
            nn.Linear(combined, features_dim),
            nn.ReLU(inplace=True),
        )
        nn.init.orthogonal_(self.head[0].weight, gain=nn.init.calculate_gain("relu"))
        nn.init.constant_(self.head[0].bias, 0.0)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        grid = obs["grid"]
        vec = obs["vec"]
        agent_mask = obs["agent_mask"]
        mask = obs["mask"]

        B, N, C, H, W = grid.shape

        cnn_out = self.cnn(grid.reshape(B * N, C, H, W))   # [B*N, cnn_latent]
        cnn_out = cnn_out.reshape(B, N, -1)                 # [B, N, cnn_latent]

        per_agent = torch.cat([cnn_out, vec], dim=-1)       # [B, N, cnn_latent+vec_dim]
        per_agent = per_agent * agent_mask.unsqueeze(-1)     # zero dead agents

        flat = per_agent.reshape(B, -1)                      # [B, N*(cnn_latent+vec_dim)]
        combined = torch.cat([flat, mask], dim=-1)           # [B, total]
        return self.head(combined)


__all__ = ["ObsEncoder", "ObsEncoderSpec", "CTFeaturesExtractor"]
