import argparse
import os
import sys
import numpy as np
import torch
from rl.config.ppo_config import PPOConfig
from rl.training.env_factory import build_training_env
from rl.custom_ppo.inference import load_custom_ppo_policy
from rl.behavior_telemetry import compute_behavior_telemetry_batch, BEHAVIOR_TELEMETRY_NAMES

def main():
    parser = argparse.ArgumentParser(description="Evaluate counterfactual z-swaps on a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint .pth")
    parser.add_argument("--steps", type=int, default=100, help="Total evaluation steps")
    parser.add_argument("--split-step", type=int, default=20, help="Step at which to force z swap")
    parser.add_argument("--seed", type=int, default=42, help="Seed for environment and generation")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # 1. Create a dummy config to initialize the env
    cfg = PPOConfig()
    cfg.preset = "plan_faithful_latent_strategic"
    cfg.use_latent_strategy = True
    cfg.n_envs = 1  # evaluate 1 env at a time
    cfg.seed = args.seed
    cfg.device = "cpu"

    # Build the environment (using standard PHASE1 and SCRIPTED OP3 opponent)
    print("Building environment...")
    env = build_training_env(cfg, initial_phase="PHASE1", initial_opponent_tag="OP3")

    try:
        # 2. Load the policy
        print(f"Loading custom PPO policy from {args.checkpoint}...")
        policy = load_custom_ppo_policy(args.checkpoint, env.observation_space, env.action_space, device="cpu")
        
        # We will run 4 runs, one for each z in [0, 1, 2, 3]
        trajectories = {}
        for z_forced in range(4):
            print(f"Running trajectory with forced z = {z_forced}...")
            # Set deterministic seeds to align initial phase and steps exactly
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            
            obs = env.reset()
            policy.reset_strategy()
            
            history = []
            
            # Step loop
            for step in range(args.steps):
                # Apply forced z starting at split_step
                if step >= args.split_step:
                    policy.fixed_latent_strategy = True
                    policy.fixed_latent_strategy_id = z_forced
                else:
                    policy.fixed_latent_strategy = False
                
                # Predict action
                action, _ = policy.predict(obs, deterministic=True)
                
                # Step env (actions shape is (1, action_dim))
                obs, rewards, dones, infos = env.step(action[None, :])
                
                # Extract behavior telemetry
                actions_t = torch.as_tensor(action[None, :], dtype=torch.long)
                with torch.no_grad():
                    beh_t = compute_behavior_telemetry_batch(env.core, actions_t)
                beh_val = beh_t[0].cpu().numpy()
                
                # Record metrics
                step_data = {
                    "step": step,
                    "z": int(policy.strategy_info().get("strategy", -1)),
                    "z_prob_0": float(policy.strategy_info().get("strategy_prob_0", 0.0)),
                    "z_prob_1": float(policy.strategy_info().get("strategy_prob_1", 0.0)),
                    "z_prob_2": float(policy.strategy_info().get("strategy_prob_2", 0.0)),
                    "z_prob_3": float(policy.strategy_info().get("strategy_prob_3", 0.0)),
                }
                for idx, name in enumerate(BEHAVIOR_TELEMETRY_NAMES):
                    step_data[name] = float(beh_val[idx])
                history.append(step_data)
                
                if dones[0]:
                    break
            trajectories[z_forced] = history
            
        # 3. Compare trajectories and print results
        print("\n=== Counterfactual z-Swap Results ===")
        print(f"Average behavior metrics starting at step {args.split_step} (after z-swap):")
        
        # Calculate average behavior metrics after split_step for each forced z
        z_metrics = {}
        for z_forced, history in trajectories.items():
            after_split = history[args.split_step:]
            if not after_split:
                print(f"Forced z={z_forced}: trajectory ended before split step.")
                continue
            
            z_metrics[z_forced] = {
                "spread": np.mean([d["team_spread"] for d in after_split]),
                "attackers": np.mean([d["num_attackers"] for d in after_split]),
                "defenders": np.mean([d["num_defenders"] for d in after_split]),
                "ratio": np.mean([d["attack_defense_ratio"] for d in after_split]),
            }
            m = z_metrics[z_forced]
            print(f"Forced z={z_forced}: avg_spread={m['spread']:.4f}, avg_attackers={m['attackers']:.2f}, avg_defenders={m['defenders']:.2f}, avg_ratio={m['ratio']:.4f}")
        
        # Evaluate divergence
        if len(z_metrics) >= 2:
            spreads = [m["spread"] for m in z_metrics.values()]
            attackers = [m["attackers"] for m in z_metrics.values()]
            ratios = [m["ratio"] for m in z_metrics.values()]
            
            spread_range = max(spreads) - min(spreads)
            attacker_range = max(attackers) - min(attackers)
            ratio_range = max(ratios) - min(ratios)
            
            print("\nDivergence Range Across Forced Strategies:")
            print(f"  Team Spread Range          : {spread_range:.4f}")
            print(f"  Num Attackers Range        : {attacker_range:.2f}")
            print(f"  Attack/Defense Ratio Range : {ratio_range:.4f}")
            
            if ratio_range > 0.05 or attacker_range > 0.2:
                print("\nRESULT: PASS")
                print("The actor shows significant sensitivity to z! Changing z successfully alters blue's tactical policy.")
            else:
                print("\nRESULT: FAIL")
                print("The actor is mostly insensitive to z (behavior metrics did not separate meaningfully).")
                print("Latent is decorative confetti. Check actor training pull (latent_strategy_ppo_coef).")
        else:
            print("\nError: Too few valid trajectories to evaluate divergence.")
            
    finally:
        env.close()

if __name__ == "__main__":
    main()
