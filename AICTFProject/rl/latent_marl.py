"""
Latent team strategy (CTDE): q_φ(z|s_g), π(a|o,z), centralized V(s_g,z).

See MARL plan: auxiliary loss −λ_H H(q) on rollout global states; env samples z via the same φ (no grad).
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo import PPO
from torch import distributions as thd

from rl.ctf_cnn_extractor import TokenizedCombinedExtractor
from rl.global_state import GLOBAL_STATE_DIM
from rl.train_ppo import MaskedMultiInputPolicy


class StrategyEncoder(nn.Module):
    """MLP: global state → logits over K strategies."""

    def __init__(self, state_dim: int, latent_k: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_k),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class PiWithZFeatureExtractor(BaseFeaturesExtractor):
    """Tokenized CNN+vec features concatenated with a linear projection of z_onehot."""

    def __init__(
        self,
        observation_space: gym.Space,
        cnn_output_dim: int = 256,
        normalized_image: bool = True,
        z_embed_dim: int = 32,
    ):
        assert isinstance(observation_space, spaces.Dict)
        latent_k = int(observation_space.spaces["z_onehot"].shape[0])
        keys = ["grid", "vec", "agent_mask"]
        if "context" in observation_space.spaces:
            keys.append("context")
        sub = spaces.Dict({k: observation_space.spaces[k] for k in keys})
        base = TokenizedCombinedExtractor(sub, cnn_output_dim, normalized_image)
        zd = int(z_embed_dim)
        super().__init__(observation_space, base.features_dim + zd)
        self.base = base
        self._pi_obs_keys: tuple[str, ...] = tuple(keys)
        self._latent_k = latent_k
        self._z_embed_dim = zd
        self.z_proj = nn.Linear(latent_k, zd)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        sub = {k: obs[k] for k in self._pi_obs_keys}
        x = self.base(sub)
        z = obs["z_onehot"].float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return torch.cat([x, self.z_proj(z)], dim=-1)


class LatentMaskedMultiInputPolicy(MaskedMultiInputPolicy):
    """
    Actor: CNN(o) || embed(z). Critic: MLP(global_state || z). Strategy: q_φ(global_state).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule,
        *,
        z_embed_dim: int = 32,
        vf_hidden: int = 128,
        strategy_hidden: int = 128,
        ortho_init: bool = True,
        **kwargs,
    ):
        self._latent_k = int(observation_space.spaces["z_onehot"].shape[0])
        self._global_state_dim = int(GLOBAL_STATE_DIM)
        self._z_embed_dim = int(z_embed_dim)
        self._vf_hidden = int(vf_hidden)

        fe_kw = dict(kwargs.pop("features_extractor_kwargs", {}) or {})
        fe_kw.setdefault("cnn_output_dim", 256)
        fe_kw.setdefault("normalized_image", True)
        fe_kw["z_embed_dim"] = int(z_embed_dim)
        kwargs["features_extractor_class"] = PiWithZFeatureExtractor
        kwargs["features_extractor_kwargs"] = fe_kw

        super().__init__(observation_space, action_space, lr_schedule, ortho_init=ortho_init, **kwargs)

        self.strategy_encoder = StrategyEncoder(self._global_state_dim, self._latent_k, hidden=int(strategy_hidden))
        self.vf_mlp = nn.Sequential(
            nn.Linear(self._global_state_dim + self._latent_k, self._vf_hidden),
            nn.Tanh(),
            nn.Linear(self._vf_hidden, self._vf_hidden),
            nn.Tanh(),
        )
        self.value_net = nn.Linear(self._vf_hidden, 1)

        if ortho_init:
            for mod, gain in (
                (self.strategy_encoder, np.sqrt(2)),
                (self.vf_mlp, np.sqrt(2)),
                (self.value_net, 1.0),
            ):
                mod.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _values_from_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        gs = obs["global_state"].float()
        z = obs["z_onehot"].float()
        if gs.dim() == 1:
            gs = gs.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        vf_in = torch.cat([gs, z], dim=-1)
        return self.value_net(self.vf_mlp(vf_in))

    def forward(self, obs: dict[str, torch.Tensor], deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        if isinstance(obs, dict) and "mask" in obs:
            logits = self._apply_action_mask(logits, obs["mask"])
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self._values_from_obs(obs)
        return actions, values, log_prob

    def evaluate_actions(self, obs: dict[str, torch.Tensor], actions: torch.Tensor):
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        if isinstance(obs, dict) and "mask" in obs:
            logits = self._apply_action_mask(logits, obs["mask"])
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        log_prob = distribution.log_prob(actions)
        values = self._values_from_obs(obs)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._values_from_obs(obs)


class LatentStrategyPPO(PPO):
    """PPO with extra term −λ_H H(q_φ(z|s_g)) (encourages higher strategy entropy when λ_H > 0)."""

    def __init__(self, *args, latent_lam_h: float = 0.01, latent_lam_p: float = 0.0, **kwargs):
        self.latent_lam_h = float(latent_lam_h)
        self.latent_lam_p = float(latent_lam_p)
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses, strat_entropy_vals = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                strat_ent_mean: Optional[torch.Tensor] = None
                if self.latent_lam_h != 0.0 and isinstance(rollout_data.observations, dict):
                    gs = rollout_data.observations.get("global_state")
                    enc = getattr(self.policy, "strategy_encoder", None)
                    if gs is not None and enc is not None:
                        logits = enc(gs)
                        strat_ent_mean = thd.Categorical(logits=logits).entropy().mean()
                        loss = loss - self.latent_lam_h * strat_ent_mean
                        strat_entropy_vals.append(float(strat_ent_mean.detach().item()))

                if self.latent_lam_p != 0.0:
                    pass  # Reserved for temporal persistence (Option B); not used by default.

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if strat_entropy_vals:
            self.logger.record("train/strategy_entropy", np.mean(strat_entropy_vals))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
