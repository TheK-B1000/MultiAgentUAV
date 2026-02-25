from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Core GPU components
from game_field_gpu import GPUCTFVecEnv, CNN_ROWS, CNN_COLS, NUM_CNN_CHANNELS
from obs_encoder import ObsEncoder

@dataclass
class MAPPOConfig:
    seed: int = 42
    n_envs: int = 64          # Run 64 games in parallel on GPU
    update_every: int = 2048   # Steps per update (Total samples = n_envs * update_every)
    ppo_epochs: int = 10
    batch_size: int = 512      # Minibatch size for SGD
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class CentralizedActorCritic(nn.Module):
    """MAPPO Network: Shared Actor and a Centralized Critic."""
    def __init__(self, n_actions: int, n_agents: int, latent_dim: int = 128):
        super().__init__()
        self.n_agents = n_agents
        
        # 1. Feature Extractor (Shared by Actor and Critic)
        self.encoder = ObsEncoder(NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS, latent_dim)
        
        # 2. Actor (Policy Head) - Shared across all agents
        self.actor = nn.Sequential(
            nn.Linear(latent_dim + 12, 128), # latent + vec features
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        # 3. Centralized Critic - Sees all agents' data to estimate team value
        # Input size: (latent_dim + 12) * n_agents
        self.critic = nn.Sequential(
            nn.Linear((latent_dim + 12) * n_agents, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def get_value(self, grid: torch.Tensor, vec: torch.Tensor):
        # grid: [B, N, C, H, W], vec: [B, N, V]
        B, N = grid.shape[0], grid.shape[1]
        features = self.encoder(grid.view(-1, *grid.shape[2:]))
        features = torch.cat([features, vec.view(-1, 12)], dim=-1)
        
        # Concatenate all agents' features for the central critic
        central_state = features.view(B, -1) 
        return self.critic(central_state) # [B, 1]

    def get_action_and_value(self, grid: torch.Tensor, vec: torch.Tensor, mask: torch.Tensor, action=None):
        B, N = grid.shape[0], grid.shape[1]
        
        # Extract features
        features = self.encoder(grid.view(-1, *grid.shape[2:]))
        features = torch.cat([features, vec.view(-1, 12)], dim=-1)
        
        # Actor processing
        logits = self.actor(features).view(B, N, -1)
        logits[mask == 0] = -1e9 # Action Masking
        
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        # Central Critic
        central_state = features.view(B, -1)
        value = self.critic(central_state)
        
        return action, probs.log_prob(action), probs.entropy(), value

def train_mappo_gpu():
    cfg = MAPPOConfig()
    device = torch.device(cfg.device)
    
    # 1. Setup GPU Env
    env = GPUCTFVecEnv(num_envs=cfg.n_envs, device=cfg.device)
    n_actions = 5 * 8 # 40 actions
    
    policy = CentralizedActorCritic(n_actions, env.Nb).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
    
    # 2. Storage for Rollouts (Pre-allocated on GPU)
    obs_grid = torch.zeros((cfg.update_every, cfg.n_envs, env.Nb, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS)).to(device)
    obs_vec = torch.zeros((cfg.update_every, cfg.n_envs, env.Nb, 12)).to(device)
    actions = torch.zeros((cfg.update_every, cfg.n_envs, env.Nb)).to(device)
    masks = torch.zeros((cfg.update_every, cfg.n_envs, env.Nb, n_actions)).to(device)
    logprobs = torch.zeros((cfg.update_every, cfg.n_envs, env.Nb)).to(device)
    rewards = torch.zeros((cfg.update_every, cfg.n_envs)).to(device)
    values = torch.zeros((cfg.update_every, cfg.n_envs)).to(device)
    dones = torch.zeros((cfg.update_every, cfg.n_envs)).to(device)

    next_obs, _ = env.reset()
    global_step = 0

    

    # 3. Training Loop
    while global_step < cfg.total_steps:
        # A. Collect Rollouts
        for step in range(cfg.update_every):
            global_step += cfg.n_envs
            
            # Save current state
            obs_grid[step] = next_obs['grid']
            obs_vec[step] = next_obs['vec']
            masks[step] = next_obs['mask']
            
            with torch.no_grad():
                action, logp, ent, val = policy.get_action_and_value(
                    next_obs['grid'], next_obs['vec'], next_obs['mask']
                )
                values[step] = val.squeeze()
                actions[step] = action
                logprobs[step] = logp

            # Step Environment
            # Split actions [B, N] into [B, N, 2] for (macro, target)
            env_a = torch.stack([action // 8, action % 8], dim=-1)
            next_obs, reward, terminated, truncated, _ = env.step(env_a)
            
            rewards[step] = reward.sum(dim=1) # Team-based reward
            dones[step] = terminated | truncated

        # B. Compute Returns and Advantages (GAE)
        with torch.no_grad():
            next_value = policy.get_value(next_obs['grid'], next_obs['vec']).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.update_every)):
                if t == cfg.update_every - 1:
                    nextnonterminal = 1.0 - dones[t].float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t].float()
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # C. Optimize Policy (PPO Update)
        # Flatten batch dimensions for the update
        b_grid = obs_grid.reshape(-1, env.Nb, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS)
        b_vec = obs_vec.reshape(-1, env.Nb, 12)
        b_masks = masks.reshape(-1, env.Nb, n_actions)
        b_actions = actions.reshape(-1, env.Nb)
        b_logprobs = logprobs.reshape(-1, env.Nb)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        for epoch in range(cfg.ppo_epochs):
            inds = np.arange(b_grid.shape[0])
            np.random.shuffle(inds)
            for start in range(0, len(inds), cfg.batch_size):
                end = start + cfg.batch_size
                mb_idx = inds[start:end]

                _, newlogp, entropy, newvalue = policy.get_action_and_value(
                    b_grid[mb_idx], b_vec[mb_idx], b_masks[mb_idx], b_actions[mb_idx]
                )
                
                # Policy Loss
                ratio = (newlogp - b_logprobs[mb_idx]).exp()
                # Average over agents for multi-agent policy loss
                mb_adv = b_advantages[mb_idx].unsqueeze(1).expand_as(ratio)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((newvalue.squeeze() - b_returns[mb_idx]) ** 2).mean()

                loss = pg_loss + cfg.entropy_coef * -entropy.mean() + v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        print(f"Global Step: {global_step} | Team Reward: {rewards.mean():.4f}")

if __name__ == "__main__":
    train_mappo_gpu() 