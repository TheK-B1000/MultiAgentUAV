import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """
    CNN over a 7-channel 20x20 spatial observation.

    Observation layout (from GameField.build_observation):

        Shape: [C, H, W] = [7, 20, 20]

        Channels:
          0: Own UAV position
          1: Teammate UAVs (same side, excluding self)
          2: Enemy UAVs
          3: Friendly mines
          4: Enemy mines
          5: Own flag
          6: Enemy flag
    """

    def __init__(
        self,
        in_channels: int = 7,
        height: int = 20,
        width: int = 20,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.latent_dim = latent_dim

        # Simple CNN stack: keep H,W = 20x20 via padding
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # After conv: channels = 64, H=W=20 → flat_dim = 64 * 20 * 20
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
        obs: [B, C, H, W] or [C, H, W]
        returns:
            latent: [B, latent_dim]
        """
        if obs.dim() == 3:
            # [C, H, W] -> [1, C, H, W]
            obs = obs.unsqueeze(0)

        assert obs.dim() == 4, f"ObsEncoder expects 4D tensor [B,C,H,W], got {obs.shape}"
        b, c, h, w = obs.shape
        assert c == self.in_channels, f"Expected {self.in_channels} channels, got {c}"
        assert h == self.height and w == self.width, \
            f"Expected spatial size {self.height}x{self.width}, got {h}x{w}"

        x = self.conv(obs)                 # [B, 64, 20, 20]
        x = x.view(b, -1)                  # [B, 64*20*20]
        latent = self.fc(x)                # [B, latent_dim]
        return latent
