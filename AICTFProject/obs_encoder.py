import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """
    CNN over a 7-channel HxW spatial observation + small extra feature vector.

    Spatial observation layout (from GameField.build_observation):

        Shape: [C, H, W] = [7, H, W]

        Channels:
          0: Own UAV position
          1: Teammate UAVs (same side, excluding self)
          2: Enemy UAVs
          3: Friendly mines
          4: Enemy mines
          5: Own flag
          6: Enemy flag

    The encoder:
      - runs a small CNN over the 7xHxW map
      - flattens it
      - concatenates a small extra vector (payload/time/decision/MAVId)
      - projects to latent_dim.
    """

    def __init__(
        self,
        in_channels: int = 7,
        height: int = 30,
        width: int = 40,
        latent_dim: int = 128,
        extra_dim: int = 7,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        self.extra_dim = extra_dim

        # Simple CNN stack: keep H,W via padding
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # After conv: channels = 64, H,W unchanged → flat_dim = 64 * H * W
        flat_dim = 64 * height * width

        # FC takes [conv_flat || extra_vec] → latent_dim
        self.fc = nn.Sequential(
            nn.Linear(flat_dim + extra_dim, latent_dim),
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

    def forward(
        self,
        obs: torch.Tensor,
        extra: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        obs  : [B, C, H, W] or [C, H, W]
        extra: [B, extra_dim] or [extra_dim] or None

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

        x = self.conv(obs)             # [B, 64, H, W]
        x = x.view(b, -1)              # [B, 64*H*W]

        # Handle extra vector
        if extra is None:
            # Use zeros if no extra features are passed
            extra_t = torch.zeros(
                b,
                self.extra_dim,
                device=obs.device,
                dtype=obs.dtype,
            )
        else:
            if extra.dim() == 1:
                extra = extra.unsqueeze(0)  # [1, extra_dim]
            assert extra.shape[0] == b, \
                f"Extra vector batch mismatch: got {extra.shape[0]}, expected {b}"
            assert extra.shape[1] == self.extra_dim, \
                f"Expected extra_dim={self.extra_dim}, got {extra.shape[1]}"
            extra_t = extra

        x_cat = torch.cat([x, extra_t], dim=-1)  # [B, flat_dim + extra_dim]
        latent = self.fc(x_cat)                  # [B, latent_dim]
        return latent
