# obs_encoder.py
import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """
    CNN encoder for the 7-channel spatial observation map.

    Supports:
      - Unbatched: [C,H,W]
      - Batched:   [B,C,H,W]
      - Channels-last: [B,H,W,C] or [H,W,C] (auto-permute if C == in_channels)

    Bulletproofing:
      - flat_dim computed via dummy forward (no pooling math assumptions)
      - AdaptiveAvgPool2d locks conv output to a fixed spatial size
      - handles non-contiguous tensors safely
    """

    def __init__(
        self,
        in_channels: int = 7,
        height: int = 40,
        width: int = 30,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.height = int(height)
        self.width = int(width)
        self.latent_dim = int(latent_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Determine conv output shape (robust)
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.height, self.width)
            y = self.conv(dummy)
            out_h, out_w = int(y.shape[-2]), int(y.shape[-1])

        # Lock output size (even if runtime H/W changes)
        self.pool = nn.AdaptiveAvgPool2d((out_h, out_w))
        self._conv_out_hw = (out_h, out_w)

        # Compute flat_dim AFTER pooling (airtight)
        with torch.no_grad():
            y2 = self.pool(self.conv(dummy))
            flat_dim = int(y2.shape[1] * y2.shape[2] * y2.shape[3])
        self._flat_dim = flat_dim

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_dim, self.latent_dim),
            nn.ReLU(inplace=True),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _ensure_bchw(self, obs: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs)

        # [C,H,W] -> [1,C,H,W]
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        # [B,H,W,C] -> [B,C,H,W] if channels-last
        if obs.dim() == 4 and obs.shape[1] != self.in_channels and obs.shape[-1] == self.in_channels:
            obs = obs.permute(0, 3, 1, 2)

        if obs.dim() != 4:
            raise ValueError(f"ObsEncoder expects [B,C,H,W] or [C,H,W], got {tuple(obs.shape)}")

        if obs.shape[1] != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got {obs.shape[1]} with shape {tuple(obs.shape)}")

        return obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self._ensure_bchw(obs).float()
        # Optional: uncomment for deterministic memory layout/perf
        # obs = obs.contiguous()

        x = self.conv(obs)
        x = self.pool(x)  # always enforce fixed spatial size
        latent = self.fc(x)
        return latent
