from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List

# Core GPU components
from game_field_gpu import GPUCTFVecEnv, CNN_ROWS, CNN_COLS, NUM_CNN_CHANNELS
from obs_encoder import ObsEncoder

@dataclass
class QMIXConfig:
    seed: int = 42
    n_envs: int = 64          # Run 64 games in parallel
    total_steps: int = 2_000_000
    replay_capacity: int = 100_000
    batch_size: int = 32      # Batch of trajectories
    lr: float = 5e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 100_000
    target_update_every: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class AgentQNet(nn.Module):
    def __init__(self, n_actions: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = ObsEncoder(NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS, latent_dim)
        # We add +12 to the latent dim to account for the 'vec' features (agent ID, flags, etc.)
        self.head = nn.Sequential(
            nn.Linear(latent_dim + 12, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        # grid: [Batch*Nb, C, H, W], vec: [Batch*Nb, V]
        z = self.encoder(grid)
        combined = torch.cat([z, vec], dim=-1)
        return self.head(combined)

class QMixer(nn.Module):
    """Mixes individual Q-values into a total team Q-value using a monotonic hypernetwork."""
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetwork for weight 1
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        # Hypernetwork for weight 2
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        # agent_qs: [B, Nb], states: [B, StateDim]
        bs = agent_qs.size(0)
        
        w1 = torch.abs(self.hyper_w1(states)).view(bs, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(bs, 1, self.embed_dim)
        
        # First layer
        hidden = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        
        w2 = torch.abs(self.hyper_w2(states)).view(bs, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(bs, 1, 1)
        
        # Output layer
        return torch.bmm(hidden, w2) + b2

def train_qmix_gpu():
    cfg = QMIXConfig()
    device = torch.device(cfg.device)
    
    # 1. Setup GPU Env
    env = GPUCTFVecEnv(num_envs=cfg.n_envs, device=cfg.device)
    n_agents = env.Nb
    n_actions = 5 * 8 # 5 macros * 8 targets
    
    # 2. Networks
    # Note: State dimension for QMIX is usually just the flattened concatenation of all obs
    # For simplicity here, we use a global pooling of the grid as the "state"
    state_dim = 256 
    
    q_net = AgentQNet(n_actions).to(device)
    target_q_net = AgentQNet(n_actions).to(device)
    mixer = QMixer(n_agents, state_dim).to(device)
    target_mixer = QMixer(n_agents, state_dim).to(device)
    
    optimizer = optim.Adam(list(q_net.parameters()) + list(mixer.parameters()), lr=cfg.lr)
    
    # 3. Training Loop
    obs, _ = env.reset()
    global_step = 0
    
    print(f"Starting GPU QMIX with {cfg.n_envs} parallel environments...")
    
    

    while global_step < cfg.total_steps:
        # Epsilon-greedy exploration
        epsilon = max(cfg.epsilon_end, cfg.epsilon_start - global_step / cfg.epsilon_decay)
        
        with torch.no_grad():
            # Flatten batch and agents for the CNN
            b, nb, c, h, w = obs['grid'].shape
            q_values = q_net(obs['grid'].view(-1, c, h, w), obs['vec'].view(-1, 12))
            q_values = q_values.view(b, nb, n_actions)
            
            # Masking invalid actions
            mask = obs['mask'] # [B, Nb, 40]
            q_values[mask == 0] = -1e9
            
            # Select actions
            if np.random.rand() < epsilon:
                # Random choice from valid actions
                actions = torch.stack([torch.multinomial(m.float(), 1).squeeze() for m in mask.view(-1, n_actions)])
                actions = actions.view(b, nb)
            else:
                actions = q_values.argmax(dim=-1)
        
        # Step environment
        # Convert flat actions back to (macro, target) for the GPU env
        macro_actions = actions // 8
        target_actions = actions % 8
        env_actions = torch.stack([macro_actions, target_actions], dim=-1)
        
        next_obs, rewards, terminals, truncateds, infos = env.step(env_actions)
        
        # In a real QMIX implementation, you would store these in a Replay Buffer
        # For brevity, we assume a standard replay buffer .add() here.
        
        obs = next_obs
        global_step += cfg.n_envs
        
        if global_step % 1000 == 0:
            print(f"Step: {global_step} | Epsilon: {epsilon:.2f} | Avg Reward: {rewards.mean():.4f}")

if __name__ == "__main__":
    train_qmix_gpu()