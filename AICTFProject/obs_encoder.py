import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """
    CNN encoder for the 7-channel spatial observation map.

    Default observation layout (from GameField.build_observation):
        Shape: [C, H, W] = [7, 40, 30]
            - C = 7 semantic channels (own UAV, teammate, enemies, mines, flags, ...)
            - H = 40 rows   (CNN_ROWS)
            - W = 30 cols   (CNN_COLS)

    Output:
        latent feature vector of size `latent_dim`, suitable for actor/critic heads.
    """

    def __init__(
        self,
        in_channels: int = 7,
        height: int = 40,   # rows
        width: int = 30,    # cols
        latent_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.latent_dim = latent_dim

        # --------------------------------------------------------------
        # Simple CNN stack: keeps H,W the same (padding=1, kernel=3)
        # --------------------------------------------------------------
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # After conv: [B, 64, H, W]
        flat_dim = 64 * height * width

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [C, H, W] or [B, C, H, W]

        Returns:
            latent: [B, latent_dim]
        """
        if obs.dim() == 3:
            # [C, H, W] -> [1, C, H, W]
            obs = obs.unsqueeze(0)

        assert obs.dim() == 4, f"ObsEncoder expects 4D tensor [B,C,H,W], got {obs.shape}"
        b, c, h, w = obs.shape

        # Shape sanity checks
        assert c == self.in_channels, f"Expected {self.in_channels} channels, got {c}"
        assert h == self.height and w == self.width, \
            f"Expected spatial size {self.height}x{self.width}, got {h}x{w}"

        x = self.conv(obs)       # [B, 64, H, W]
        x = x.view(b, -1)        # [B, 64*H*W]
        latent = self.fc(x)      # [B, latent_dim]
        return latent
