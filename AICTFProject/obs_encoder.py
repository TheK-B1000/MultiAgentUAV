# obs_encoder.py
import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """
    CNN encoder for the 7-channel spatial observation map.

    Intended input from GameField.build_observation:
        - Unbatched: [C,H,W]  = [7,40,30]
        - Batched:   [B,C,H,W]

    Also supports accidental channels-last batches:
        - [B,H,W,C]  -> will auto-permute to [B,C,H,W] if C matches in_channels.

    Bulletproofing:
      - flat_dim computed via dummy forward (no pooling math assumptions)
      - AdaptiveAvgPool2d locks conv output to a fixed spatial size, so FC always matches
      - handles non-contiguous tensors safely
    """

    def __init__(
        self,
        in_channels: int = 7,
        height: int = 40,    # CNN_ROWS
        width: int = 30,     # CNN_COLS
        latent_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.height = int(height)
        self.width = int(width)
        self.latent_dim = int(latent_dim)

        # Feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H,W -> floor(H/2),floor(W/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> floor(H/4),floor(W/4)
        )

        # Determine conv output shape using a dummy forward pass (robust)
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.height, self.width)
            y = self.conv(dummy)
            out_h, out_w = int(y.shape[-2]), int(y.shape[-1])
            flat_dim = int(y.numel())  # since batch=1

        # Lock conv output size so FC stays valid even if H/W changes at runtime
        self.pool = nn.AdaptiveAvgPool2d((out_h, out_w))
        self._conv_out_hw = (out_h, out_w)
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
        """
        Convert obs to [B,C,H,W] if possible.
        """
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs)

        # [C,H,W] -> [1,C,H,W]
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        # If channels-last: [B,H,W,C] and C matches in_channels -> permute
        if obs.dim() == 4 and obs.shape[1] != self.in_channels and obs.shape[-1] == self.in_channels:
            obs = obs.permute(0, 3, 1, 2).contiguous()

        if obs.dim() != 4:
            raise ValueError(f"ObsEncoder expects [B,C,H,W] or [C,H,W], got {tuple(obs.shape)}")

        if obs.shape[1] != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got {obs.shape[1]} with shape {tuple(obs.shape)}")

        return obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            latent: [B, latent_dim]
        """
        obs = self._ensure_bchw(obs).float()

        x = self.conv(obs)  # [B,64,h,w] (depends on input)
        # Force stable spatial size for FC
        if (int(x.shape[-2]), int(x.shape[-1])) != self._conv_out_hw:
            x = self.pool(x)

        latent = self.fc(x)  # [B,latent_dim]
        return latent
