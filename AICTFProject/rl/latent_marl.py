"""
Latent team strategy MARL path for the paper-aligned CTDE setup.

This module implements:
  - q_phi(z | s_g): discrete strategy encoder from global state
  - pi_i(a_i | o_i, z): shared per-agent policy conditioned on local obs + shared z
  - Q(s_g, a_joint, z): centralized action-conditioned critic

The actor keeps decentralized execution faithful by computing each agent's logits
from only that agent's local observation and the shared latent strategy embedding.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo import PPO
from torch import distributions as thd

from rl.ctf_cnn_extractor import TokenizedCombinedExtractor
from rl.global_state import GLOBAL_STATE_DIM
from rl.networks import CNNEncoder
from rl.train_ppo import MaskedMultiInputPolicy


def expected_strategy_switch_penalty(logits: torch.Tensor, prev_z_idx: torch.Tensor) -> torch.Tensor:
    """
    Differentiable persistence proxy: E[1(z_t != z_{t-1})] = 1 - p(z_{t-1} | s_t).

    This matches the paper's switch-indicator loss in expectation while still
    allowing gradients to flow into q_phi.
    """

    probs = torch.softmax(logits, dim=-1)
    prev = prev_z_idx.long().clamp(min=0, max=probs.shape[-1] - 1).unsqueeze(-1)
    stay_prob = probs.gather(-1, prev).squeeze(-1)
    return 1.0 - stay_prob


class StrategyEncoder(nn.Module):
    """Global-state encoder q_phi(z | s_g) with the paper's 128-128-ReLU MLP."""

    def __init__(self, state_dim: int, latent_k: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_k),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class LatentMaskedMultiInputPolicy(MaskedMultiInputPolicy):
    """
    Shared per-agent actor with a centralized action-conditioned critic.

    Actor:
      pi_i(a_i | o_i, z) with shared parameters across all blue agents

    Critic:
      Q(s_g, a_joint, z) where the input is [global_state, joint_action_onehot, z_onehot]
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule,
        *,
        z_embed_dim: int = 16,
        vf_hidden: int = 128,
        strategy_hidden: int = 128,
        ortho_init: bool = True,
        **kwargs,
    ):
        self._latent_k = int(observation_space.spaces["z_onehot"].shape[0])
        self._global_state_dim = int(GLOBAL_STATE_DIM)
        self._z_embed_dim = int(z_embed_dim)
        self._vf_hidden = int(vf_hidden)

        grid_shape = observation_space.spaces["grid"].shape
        vec_shape = observation_space.spaces["vec"].shape
        self._n_agents = int(grid_shape[0])
        self._grid_channels = int(grid_shape[1])
        self._grid_rows = int(grid_shape[2])
        self._grid_cols = int(grid_shape[3])
        self._vec_dim = int(vec_shape[1])

        action_dims = [int(v) for v in getattr(action_space, "nvec", [])]
        if not action_dims:
            raise ValueError("LatentMaskedMultiInputPolicy requires a MultiDiscrete action space.")
        if len(action_dims) % self._n_agents != 0:
            raise ValueError(
                f"Action dims {action_dims} do not divide evenly across {self._n_agents} agents."
            )
        self._action_dims = tuple(action_dims)
        self._n_action_heads_per_agent = len(action_dims) // self._n_agents
        self._per_agent_action_dims = tuple(action_dims[: self._n_action_heads_per_agent])
        for agent_idx in range(self._n_agents):
            start = agent_idx * self._n_action_heads_per_agent
            end = start + self._n_action_heads_per_agent
            if tuple(action_dims[start:end]) != self._per_agent_action_dims:
                raise ValueError("Paper-aligned latent policy expects identical action heads per agent.")
        self._per_agent_action_dim = int(sum(self._per_agent_action_dims))
        self._joint_action_onehot_dim = int(sum(self._action_dims))

        fe_kw = dict(kwargs.pop("features_extractor_kwargs", {}) or {})
        fe_kw.setdefault("cnn_output_dim", 256)
        fe_kw.setdefault("normalized_image", True)
        kwargs["features_extractor_class"] = TokenizedCombinedExtractor
        kwargs["features_extractor_kwargs"] = fe_kw
        kwargs.setdefault("net_arch", dict(pi=[], vf=[]))

        super().__init__(observation_space, action_space, lr_schedule, ortho_init=ortho_init, **kwargs)

        self.strategy_encoder = StrategyEncoder(self._global_state_dim, self._latent_k, hidden=int(strategy_hidden))
        self.strategy_embedding = nn.Embedding(self._latent_k, self._z_embed_dim)
        self.local_grid_encoder = CNNEncoder(
            (self._grid_channels, self._grid_rows, self._grid_cols),
            feature_dim=256,
        )

        actor_input_dim = 256 + self._vec_dim + self._z_embed_dim
        self.agent_policy_net = nn.Sequential(
            nn.Linear(actor_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.agent_action_head = nn.Linear(256, self._per_agent_action_dim)

        self.q_mlp = nn.Sequential(
            nn.Linear(self._global_state_dim + self._joint_action_onehot_dim + self._latent_k, self._vf_hidden),
            nn.ReLU(),
            nn.Linear(self._vf_hidden, self._vf_hidden),
            nn.ReLU(),
            nn.Linear(self._vf_hidden, 1),
        )

        if ortho_init:
            for mod, gain in (
                (self.strategy_encoder, np.sqrt(2)),
                (self.local_grid_encoder, np.sqrt(2)),
                (self.agent_policy_net, np.sqrt(2)),
                (self.agent_action_head, 0.01),
                (self.q_mlp, 1.0),
            ):
                mod.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _z_indices(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        z_idx = obs.get("z_idx")
        if z_idx is None:
            z = obs["z_onehot"].float()
            if z.dim() == 1:
                z = z.unsqueeze(0)
            return z.argmax(dim=-1).long()
        if z_idx.dim() == 1:
            z_idx = z_idx.unsqueeze(0)
        return z_idx.squeeze(-1).long().clamp(min=0, max=self._latent_k - 1)

    def _actor_logits_from_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        grid = obs["grid"].float()
        vec = obs["vec"].float()
        if grid.dim() == 4:
            grid = grid.unsqueeze(0)
        if vec.dim() == 2:
            vec = vec.unsqueeze(0)

        batch_size = grid.shape[0]
        z_idx = self._z_indices(obs)
        z_emb = self.strategy_embedding(z_idx).unsqueeze(1).expand(batch_size, self._n_agents, self._z_embed_dim)

        grid_flat = grid.reshape(batch_size * self._n_agents, *grid.shape[2:])
        grid_feat = self.local_grid_encoder(grid_flat).reshape(batch_size, self._n_agents, -1)

        agent_mask = obs.get("agent_mask")
        if agent_mask is not None:
            if agent_mask.dim() == 1:
                agent_mask = agent_mask.unsqueeze(0)
            mask = agent_mask.float().unsqueeze(-1)
            grid_feat = grid_feat * mask
            vec = vec * mask

        actor_in = torch.cat([grid_feat, vec, z_emb], dim=-1)
        hidden = self.agent_policy_net(actor_in.reshape(batch_size * self._n_agents, -1))
        logits = self.agent_action_head(hidden).reshape(batch_size, self._n_agents, self._per_agent_action_dim)
        return logits.reshape(batch_size, self._n_agents * self._per_agent_action_dim)

    def _masked_distribution(self, obs: dict[str, torch.Tensor]):
        logits = self._actor_logits_from_obs(obs)
        if isinstance(obs, dict) and "mask" in obs:
            logits = self._apply_action_mask(logits, obs["mask"])
        return self.action_dist.proba_distribution(action_logits=logits)

    def _joint_action_onehot(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        actions = actions.long()
        chunks = []
        for col, dim in enumerate(self._action_dims):
            a = actions[:, col].clamp(min=0, max=dim - 1)
            chunks.append(F.one_hot(a, num_classes=dim).float())
        return torch.cat(chunks, dim=-1)

    def _q_values_from_obs_actions(self, obs: dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        gs = obs["global_state"].float()
        z = obs["z_onehot"].float()
        if gs.dim() == 1:
            gs = gs.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        joint_action = self._joint_action_onehot(actions)
        critic_in = torch.cat([gs, joint_action, z], dim=-1)
        return self.q_mlp(critic_in)

    def get_distribution(self, obs: dict[str, torch.Tensor]):
        return self._masked_distribution(obs)

    def _predict(self, observation: dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def forward(self, obs: dict[str, torch.Tensor], deterministic: bool = False):
        distribution = self._masked_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self._q_values_from_obs_actions(obs, actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: dict[str, torch.Tensor], actions: torch.Tensor):
        distribution = self._masked_distribution(obs)
        log_prob = distribution.log_prob(actions)
        values = self._q_values_from_obs_actions(obs, actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        distribution = self._masked_distribution(obs)
        greedy_actions = distribution.get_actions(deterministic=True)
        return self._q_values_from_obs_actions(obs, greedy_actions)


class LatentStrategyPPO(PPO):
    """
    PPO with the paper-aligned latent regularizers:
      L = L_marl + lambda_p * L_persist - lambda_H * H(q_phi(z | s_g))
    """

    def __init__(self, *args, latent_lam_h: float = 0.01, latent_lam_p: float = 0.02, **kwargs):
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
        persist_losses, switch_rates, resample_rates = [], [], []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        approx_kl_divs = []
        last_loss: Optional[torch.Tensor] = None
        for epoch in range(self.n_epochs):
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

                if isinstance(rollout_data.observations, dict):
                    gs = rollout_data.observations.get("global_state")
                    enc = getattr(self.policy, "strategy_encoder", None)
                    if gs is not None and enc is not None:
                        logits = enc(gs)
                        strat_dist = thd.Categorical(logits=logits)
                        strat_ent_mean = strat_dist.entropy().mean()
                        if self.latent_lam_h != 0.0:
                            loss = loss - self.latent_lam_h * strat_ent_mean
                        strat_entropy_vals.append(float(strat_ent_mean.detach().item()))

                        z_resampled = rollout_data.observations.get("z_resampled")
                        z_prev_idx = rollout_data.observations.get("z_prev_idx")
                        if self.latent_lam_p != 0.0 and z_resampled is not None and z_prev_idx is not None:
                            if z_resampled.dim() > 1:
                                z_resampled = z_resampled.squeeze(-1)
                            if z_prev_idx.dim() > 1:
                                z_prev_idx = z_prev_idx.squeeze(-1)
                            valid = z_resampled.float() > 0.5
                            resample_rates.append(float(valid.float().mean().detach().item()))
                            if torch.any(valid):
                                per_sample_penalty = expected_strategy_switch_penalty(logits, z_prev_idx)
                                persist_loss = per_sample_penalty[valid].mean()
                                loss = loss + self.latent_lam_p * persist_loss
                                persist_losses.append(float(persist_loss.detach().item()))

                        z_switch = rollout_data.observations.get("z_switch")
                        if z_switch is not None:
                            if z_switch.dim() > 1:
                                z_switch = z_switch.squeeze(-1)
                            switch_rates.append(float(z_switch.float().mean().detach().item()))

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
                last_loss = loss.detach()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        if last_loss is not None:
            self.logger.record("train/loss", float(last_loss.item()))
        self.logger.record("train/explained_variance", explained_var)
        if strat_entropy_vals:
            self.logger.record("train/strategy_entropy", np.mean(strat_entropy_vals))
        if persist_losses:
            self.logger.record("train/strategy_persistence_loss", np.mean(persist_losses))
        if switch_rates:
            self.logger.record("train/strategy_switch_rate", np.mean(switch_rates))
        if resample_rates:
            self.logger.record("train/strategy_resample_rate", np.mean(resample_rates))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
