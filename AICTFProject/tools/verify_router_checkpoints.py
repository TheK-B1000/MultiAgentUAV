import os
import hashlib
import torch
import glob

def calculate_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_hash_of_actor_weights(state_dict):
    # Get all keys that belong to the actor (starts with 'actor' or 'latent_actor' or contains 'actor')
    actor_keys = sorted([k for k in state_dict.keys() if 'actor' in k])
    h = hashlib.sha256()
    for k in actor_keys:
        tensor = state_dict[k]
        # Clone to cpu, convert to numpy float32 for deterministic hashing
        arr = tensor.cpu().numpy().tobytes()
        h.update(k.encode('utf-8'))
        h.update(arr)
    return h.hexdigest()

def verify():
    source_path = "checkpoints/4v4_phasea_window/ckpt_v6i5_ppo_then_cf_8x_phasea_window_400k_500k_4v4_516096.zip"
    print(f"Loading source actor checkpoint: {source_path}")
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    source_step = source_payload.get("global_step")
    source_updates = source_payload.get("updates_completed")
    source_actor_hash = get_hash_of_actor_weights(source_payload["model_state_dict"])
    source_sha256 = calculate_sha256(source_path)
    print(f"  SHA256: {source_sha256}")
    print(f"  global_step: {source_step}")
    print(f"  updates_completed: {source_updates}")
    print(f"  Actor Weights Hash: {source_actor_hash}")
    
    # Now find all checkpoints in the router stage
    router_checkpoints = sorted(glob.glob("checkpoints/4v4_phasea_window_router_z0_z3/*.zip"))
    
    print("\nVerifying Router Checkpoints:")
    for path in router_checkpoints:
        filename = os.path.basename(path)
        sha256 = calculate_sha256(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        
        step = payload.get("global_step")
        updates = payload.get("updates_completed")
        actor_hash = get_hash_of_actor_weights(payload["model_state_dict"])
        
        # Check router optimizer step
        router_opt = payload.get("router_optimizer_state_dict")
        router_opt_steps = []
        if router_opt and "state" in router_opt:
            for p_state in router_opt["state"].values():
                if "step" in p_state:
                    step_val = p_state["step"]
                    if isinstance(step_val, torch.Tensor):
                        router_opt_steps.append(int(step_val.item()))
                    else:
                        router_opt_steps.append(int(step_val))
        
        max_router_opt_step = max(router_opt_steps) if router_opt_steps else None
        
        # Compare actor hash
        actor_frozen = (actor_hash == source_actor_hash)
        
        print(f"\nCheckpoint: {filename}")
        print(f"  File SHA256: {sha256}")
        print(f"  global_step: {step}")
        print(f"  updates_completed: {updates}")
        print(f"  Max Router Opt Step: {max_router_opt_step}")
        print(f"  Actor Weights Hash: {actor_hash}")
        print(f"  Actor Frozen? {actor_frozen}")
        
        # Also print configuration details if present
        cfg = payload.get("cfg", {})
        if cfg:
            print(f"  router_allowed_latents: {cfg.get('router_allowed_latents')}")
            print(f"  use_latent_strategy: {cfg.get('use_latent_strategy')}")
            print(f"  fixed_latent_strategy: {cfg.get('fixed_latent_strategy')}")

if __name__ == "__main__":
    verify()
