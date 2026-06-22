"""Offline diagnostics tool to audit v6i5 CF loss geometry, PPO conflicts, and cancellation."""

from __future__ import annotations

import os
import sys
import json
import csv
import copy
import hashlib
import dataclasses
import numpy as np
import torch
import torch.nn.functional as F

# Insert project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from rl.training.env_factory import build_training_env
from rl.custom_ppo.trainer import CustomPPOTrainer
from rl.custom_ppo.v6i1_cf_loss import v6i1_cf_separation_loss, _FaithfulObsGuard
from rl.custom_ppo.curriculum_gates import PAIR_ORDER
from rl.custom_ppo.trainer_optimizers import collect_actor_parameters
from rl.custom_ppo.update.param_registry import classify_parameter_name
from rl.ppo_core import ppo_policy_loss

# Set devices and seeds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIAG_SEED = 42

def compute_fingerprint(model: torch.nn.Module) -> str:
    """Compute sha256 parameter fingerprint of the model."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def compute_module_hash(module: torch.nn.Module | None) -> str:
    """Compute sha256 parameter and buffer fingerprint of a sub-module."""
    if module is None:
        return "None"
    h = hashlib.sha256()
    for name, param in sorted(module.named_parameters()):
        h.update(name.encode())
        h.update(param.detach().cpu().numpy().tobytes())
    for name, buf in sorted(module.named_buffers()):
        h.update(name.encode())
        h.update(buf.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def get_norm_state_dict(trainer) -> dict:
    return {
        "return_norm": trainer.return_norm.state_dict(),
        "strategy_return_norm": trainer.strategy_return_norm.state_dict()
    }

def compute_norm_hash(trainer) -> str:
    """Compute hash of the return normalizers state dict."""
    norm_dict = get_norm_state_dict(trainer)
    norm_str = json.dumps(norm_dict, sort_keys=True)
    return hashlib.sha256(norm_str.encode()).hexdigest()

def compute_tensor_hash(t: torch.Tensor | None) -> str:
    if t is None:
        return "None"
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()

def compute_batch_hash(batch: dict) -> str:
    h = hashlib.sha256()
    for k, v in sorted(batch.items()):
        if isinstance(v, torch.Tensor):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def classify_actor_pathway(name: str) -> str:
    """Classify actor parameter names into disjoint groups."""
    if "strategy_embedding" in name:
        return "z_embedding"
    if "actor_cnn" in name:
        return "local_encoder"
    if "latent_actor.body" in name:
        return "shared_trunk"
    if "latent_actor.action_head" in name:
        return "policy_head"
    return "other"

def get_actor_param_groups(model: torch.nn.Module) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    """Get disjoint actor parameter groups."""
    groups = {
        "z_embedding": [],
        "local_encoder": [],
        "shared_trunk": [],
        "policy_head": [],
        "other": [],
    }
    actor_params = collect_actor_parameters(model)
    actor_param_set = set(actor_params)
    for name, param in model.named_parameters():
        if param not in actor_param_set:
            continue
        pathway = classify_actor_pathway(name)
        groups[pathway].append((name, param))
    
    # Assert disjointness
    all_names = []
    for gname, plist in groups.items():
        for name, _ in plist:
            all_names.append(name)
    assert len(all_names) == len(set(all_names)), f"Parameter groups are not disjoint: {all_names}"
    return groups

def extract_obs_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract standard observations from buffer batch dict."""
    return {
        "grid": batch["obs_grid"],
        "vec": batch["obs_vec"],
        "mask": batch.get("obs_mask"),
        "agent_mask": batch.get("obs_agent_mask"),
    }

def compute_pair_losses(model, obs_batch, margin, latent_k, device) -> list[torch.Tensor]:
    """Compute Relu(margin - pair_jsd) for each of the 6 pairs without detaching to preserve gradients."""
    obs_batch = _FaithfulObsGuard(obs_batch)
    batch_size = 0
    for value in obs_batch.values():
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            batch_size = int(value.shape[0])
            break
    if batch_size <= 0:
        return [torch.zeros((), device=device) for _ in range(6)]

    max_rows = 512
    if batch_size > max_rows:
        obs_sub = {}
        for key, value in obs_batch.items():
            if isinstance(value, torch.Tensor) and int(value.shape[0]) == batch_size:
                obs_sub[key] = value[:max_rows]
            else:
                obs_sub[key] = value
        obs_batch = _FaithfulObsGuard(obs_sub)
        curr_batch_size = max_rows
    else:
        curr_batch_size = batch_size

    logits_list = []
    for k in range(int(latent_k)):
        z_k = torch.full((curr_batch_size,), k, dtype=torch.long, device=device)
        logits_k = model._mask_logits(
            model.policy_logits(obs_batch, z_idx=z_k, detach_local_features=True),
            obs_batch.get("mask"),
        )
        logits_list.append(logits_k)

    pair_count = len(PAIR_ORDER)
    pair_js_sum = [logits_list[0].new_zeros((curr_batch_size,)) for _ in range(pair_count)]
    pair_js_count = [0 for _ in range(pair_count)]
    n_heads = len(model.per_agent_action_dims)

    offset = 0
    for _agent_idx in range(int(model.n_agents)):
        for head_idx, dim in enumerate(model.per_agent_action_dims):
            width = int(dim)
            p_stacked = []
            for k in range(int(latent_k)):
                a_k = logits_list[k][:, offset : offset + width]
                p_stacked.append(torch.softmax(a_k, dim=-1).clamp_min(1e-8))
            p_stacked_t = torch.stack(p_stacked, dim=0)
            p_i = p_stacked_t.unsqueeze(1)
            p_j = p_stacked_t.unsqueeze(0)
            m = 0.5 * (p_i + p_j)
            kl_i = (p_i * (p_i.log() - m.log())).sum(dim=-1)
            kl_j = (p_j * (p_j.log() - m.log())).sum(dim=-1)
            js_matrix = 0.5 * kl_i + 0.5 * kl_j
            for pair_idx, (zi, zj) in enumerate(PAIR_ORDER):
                if zi >= int(latent_k) or zj >= int(latent_k):
                    continue
                pair_js_sum[pair_idx] = pair_js_sum[pair_idx] + js_matrix[zi, zj]
                pair_js_count[pair_idx] += 1
            offset += width

    pair_losses = []
    margin_t = logits_list[0].new_tensor(float(max(0.0, margin)))
    for pair_idx in range(6):
        denom = max(1, pair_js_count[pair_idx])
        pair_d_p = pair_js_sum[pair_idx] / float(denom)
        pair_losses.append(F.relu(margin_t - pair_d_p).mean())
    return pair_losses

def get_gradients(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    """Compute gradients of loss w.r.t parameters, handling None and cloning."""
    if not loss.requires_grad:
        return [torch.zeros_like(p) for p in parameters]
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    clean = []
    for p, g in zip(parameters, grads):
        if g is None:
            clean.append(torch.zeros_like(p))
        else:
            clean.append(g.detach().clone())
    return clean

def file_hash(path: str) -> str:
    """Compute sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Start of v6i5 CF Diagnostics ===")
    
    # 1. Verify Checkpoint and Resolved Configuration
    checkpoint_path = "checkpoints/4v4_diag/ckpt_v6i5_phase_a_diag_150k_audit_cf_r2_4v4_150000.zip"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Selected Phase A checkpoint not found: {checkpoint_path}")
    
    ckpt_hash = file_hash(checkpoint_path)
    print(f"Loaded Checkpoint: {checkpoint_path}")
    print(f"Checkpoint SHA256: {ckpt_hash}")
    
    # Read prior run config to log baseline metrics
    prior_config_path = "checkpoints/4v4_diag/v6i5_cf_sweep_8x_150k_4v4_run_config.json"
    with open(prior_config_path, "r", encoding="utf-8") as f:
        prior_data = json.load(f)
    prior_resolved = prior_data.get("resolved_ppo_config", {})
    
    print("\n--- Calibration Constants ---")
    print(f"Baseline Canonical CF Coefficient: 1.0")
    print(f"Sweep Multiplier: 8.0")
    print(f"Resolved Effective CF Coefficient: {prior_resolved.get('latent_cf_coef_max', 8.0)}")
    
    # Create config matching checkpoint parameters
    cfg = PPOConfig()
    apply_preset(cfg, "v6i5")
    cfg.device = str(DEVICE)
    cfg.n_envs = 32
    cfg.n_steps = 256
    cfg.batch_size = 512
    cfg.max_blue_agents = 4
    cfg.map_layout = "map_b_split_lane_v2"
    
    # 2. Build Disjoint Training and Held-Out Banks
    print("\n--- Constructing Disjoint Environments ---")
    cfg.seed = 42
    env_train = build_training_env(cfg, initial_phase="A", initial_opponent_tag="OP5")
    
    cfg.seed = 9999
    env_heldout = build_training_env(cfg, initial_phase="A", initial_opponent_tag="OP5")
    
    # Derive keyword args for trainer
    learning_rate = float(cfg.learning_rate)
    ent_coef = float(cfg.ent_coef)
    clip_range = float(cfg.clip_range)
    n_epochs = int(cfg.n_epochs)
    batch_size = int(cfg.batch_size)
    if cfg.max_blue_agents > 2:
        learning_rate *= 0.75
    rollout_size = max(1, int(cfg.n_steps) * max(1, int(cfg.n_envs)))
    if batch_size > rollout_size:
        batch_size = rollout_size

    # Load trainer for train and heldout data collection
    trainer_train = CustomPPOTrainer(
        env_train,
        cfg,
        learning_rate=learning_rate,
        clip_range=clip_range,
        ent_coef=ent_coef,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )
    trainer_train.load(checkpoint_path)
    
    trainer_heldout = CustomPPOTrainer(
        env_heldout,
        cfg,
        learning_rate=learning_rate,
        clip_range=clip_range,
        ent_coef=ent_coef,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )
    trainer_heldout.load(checkpoint_path)
    
    print("Collecting disjoint rollout buffers...")
    # Seed generators to control stochasticity
    torch.manual_seed(DIAG_SEED)
    np.random.seed(DIAG_SEED)
    
    rollout_train = trainer_train.collect_rollout()
    rollout_heldout = trainer_heldout.collect_rollout()
    
    # Extract fixed minibatches
    train_iter = rollout_train.iter_minibatches(batch_size=512, shuffle=True)
    heldout_iter = rollout_heldout.iter_minibatches(batch_size=512, shuffle=True)
    B_train = next(train_iter)
    B_heldout = next(heldout_iter)
    
    # Assert disjointness of train and heldout banks
    # Verify environment seeds and data references are unique
    assert env_train.cfg.seed != env_heldout.cfg.seed, "Environment seeds are not disjoint!"
    
    # Check that there is no overlap in global states (source IDs/trajectories)
    train_gs = B_train["global_state"]
    heldout_gs = B_heldout["global_state"]
    overlap_count = 0
    for row in train_gs:
        match = (heldout_gs == row).all(dim=-1).any().item()
        if match:
            overlap_count += 1
    assert overlap_count == 0, f"Overlap detected between train and held-out global states! Overlap count: {overlap_count}"
    print("Training and heldout banks successfully constructed and verified disjoint.")
    
    # Clean up collect environments
    env_train.close()
    env_heldout.close()
    
    # 3. Freeze starting state and verify pristine copies
    print("\n--- Freezing Actor and Model Copying ---")
    reference_model = copy.deepcopy(trainer_train.model)
    cf_optimized_model = copy.deepcopy(trainer_train.model)
    
    # Ensure requires_grad is frozen on reference and only actor is trainable on opt
    for name, param in reference_model.named_parameters():
        param.requires_grad = False
    
    for name, param in cf_optimized_model.named_parameters():
        group = classify_parameter_name(name)
        param.requires_grad = (group == "actor")
        
    cf_optimized_model.eval()
    reference_model.eval()
    
    # Record starting state hashes for integrity check
    torch_rng_state_start = torch.get_rng_state()
    numpy_rng_state_start = np.random.get_state()
    
    actor_hash_start = compute_module_hash(trainer_train.model.latent_actor)
    critic_hash_start = compute_module_hash(trainer_train.model.critic)
    router_hash_start = compute_module_hash(trainer_train.model.strategy_encoder)
    
    norm_hash_train_start = compute_norm_hash(trainer_train)
    norm_hash_heldout_start = compute_norm_hash(trainer_heldout)
    
    batch_train_hash_start = compute_batch_hash(B_train)
    batch_heldout_hash_start = compute_batch_hash(B_heldout)
    
    ref_fingerprint_start = compute_fingerprint(reference_model)
    opt_fingerprint_start = compute_fingerprint(cf_optimized_model)
    
    # 4. Authoritative Loss checks
    print("\n--- Authoritative Loss Check ---")
    obs_batch_train = extract_obs_batch(B_train)
    obs_batch_heldout = extract_obs_batch(B_heldout)
    cf_margin = float(getattr(cfg, "latent_cf_jsd_margin", 0.01))
    competence = np.ones(cfg.latent_k, dtype=np.float32)
    competence_ready = True
    
    loss_direct, stats_direct = v6i1_cf_separation_loss(
        model=cf_optimized_model,
        obs_batch=obs_batch_train,
        latent_k=cfg.latent_k,
        margin=cf_margin,
        competence=competence,
        competence_ready=competence_ready,
        weak_pair_ema=None,
        weak_pair_boost=0.0,
        worst_pair_coef=0.0,
        require_competence=False,
    )
    
    # Call loss via trainer updater separation objective
    from rl.custom_ppo.update.separation_objectives import SeparationObjective
    separation_objective = SeparationObjective(
        model=trainer_train.model,
        cfg=cfg,
        hparams=trainer_train.hparams,
        runtime=trainer_train,
        latent_state=trainer_train.latent_state,
        subsample_generator=trainer_train.updater._z_separation_generator,
    )
    sep_res = separation_objective.compute(
        obs_batch=obs_batch_train,
        batch=B_train,
        advantages=B_train["advantages"],
        entropy=B_train["advantages"],
        z_idx=B_train["z"],
        separation_coef=1.0,
        counterfactual_active=True,
        device=DEVICE,
        zero_scalar=torch.zeros((), device=DEVICE),
    )
    
    # For verification, we compute the loss directly using the exact same arguments as the updater:
    comp_scores, comp_ready = trainer_train.latent_state.compute_competence_scores()
    weak_pair_ema = getattr(trainer_train.latent_state, "cf_pair_jsd_ema", None)
    weak_pair_boost = float(getattr(cfg, "latent_cf_weak_pair_boost", 0.0) or 0.0)
    worst_pair_coef = float(getattr(cfg, "latent_cf_worst_pair_coef", 0.0) or 0.0)
    require_competence = bool(getattr(cfg, "latent_cf_require_competence", False))
    
    loss_direct_verification, stats_direct_verification = v6i1_cf_separation_loss(
        model=cf_optimized_model,
        obs_batch=obs_batch_train,
        latent_k=cfg.latent_k,
        margin=cf_margin,
        competence=comp_scores,
        competence_ready=comp_ready,
        weak_pair_ema=weak_pair_ema,
        weak_pair_boost=weak_pair_boost,
        worst_pair_coef=worst_pair_coef,
        require_competence=require_competence,
    )
    
    abs_diff = torch.abs(loss_direct_verification - sep_res.loss.raw_value).item()
    print(f"Direct Verification Loss: {loss_direct_verification.item():.6f}, Updater Loss: {sep_res.loss.raw_value.item():.6f}, Diff: {abs_diff:.6e}")
    assert abs_diff < 1e-4, f"Loss verification failed! Direct verification loss ({loss_direct_verification.item():.6f}) differs from training loss ({sep_res.loss.raw_value.item():.6f})"
    print("Authoritative loss alignment verified.")
    
    # 5. One-Step SGD Sign Test (Refinement 1)
    print("\n--- One-Step SGD Sign Test ---")
    sgd_test_model = copy.deepcopy(cf_optimized_model)
    sgd_actor_params = [p for p in sgd_test_model.parameters() if p.requires_grad]
    
    loss_direct_before, stats_direct_before = v6i1_cf_separation_loss(
        model=sgd_test_model,
        obs_batch=obs_batch_train,
        latent_k=cfg.latent_k,
        margin=cf_margin,
        competence=competence,
        competence_ready=competence_ready,
        weak_pair_ema=None,
        weak_pair_boost=0.0,
        worst_pair_coef=0.0,
        require_competence=False,
    )
    jsd_before = stats_direct_before["jsd"].item()
    pair_jsd_before = stats_direct_before["pair_jsd"].tolist()
    
    g_CF_sgd = get_gradients(loss_direct_before, sgd_actor_params)
    
    # One tiny SGD step: θ' = θ - η * g
    eta = 1e-4
    with torch.no_grad():
        for p, g in zip(sgd_actor_params, g_CF_sgd):
            p.add_(g, alpha=-eta)
            
    loss_direct_after, stats_direct_after = v6i1_cf_separation_loss(
        model=sgd_test_model,
        obs_batch=obs_batch_train,
        latent_k=cfg.latent_k,
        margin=cf_margin,
        competence=competence,
        competence_ready=competence_ready,
        weak_pair_ema=None,
        weak_pair_boost=0.0,
        worst_pair_coef=0.0,
        require_competence=False,
    )
    jsd_after = stats_direct_after["jsd"].item()
    pair_jsd_after = stats_direct_after["pair_jsd"].tolist()
    
    print(f"CF Loss Before: {loss_direct_before.item():.6f} | After: {loss_direct_after.item():.6f}")
    print(f"Aggregate JSD Before: {jsd_before:.6f} | After: {jsd_after:.6f}")
    print(f"JSD Change: {jsd_after - jsd_before:+.6f}")
    
    # Check if update lowers aggregate JSD unexpectedly
    if jsd_after < jsd_before - 1e-7:
        raise AssertionError(f"One-step CF update lowered aggregate JSD unexpectedly: {jsd_before:.6f} -> {jsd_after:.6f}")
    print("SGD Sign Test passed.")
    
    # 6. PPO vs CF gradient alignment (Refinement 3)
    print("\n--- PPO vs. CF Gradient Alignment ---")
    actor_params = [p for p in cf_optimized_model.parameters() if p.requires_grad]
    
    # PPO Policy Loss
    values_norm, action_log_prob, entropy, aux = cf_optimized_model.evaluate_actions(
        obs_batch_train,
        B_train["global_state"],
        B_train["actions"],
        z_idx=B_train["z"],
        router_context=B_train.get("router_context"),
    )
    advantages = B_train["advantages"]
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        
    policy_loss, ppo_stats = ppo_policy_loss(
        action_log_prob,
        B_train["log_probs"],
        advantages,
        cfg.clip_range,
    )
    
    g_PPO = get_gradients(policy_loss, actor_params)
    
    # Load actual competence scores from trainer_train latent_state for effective scaled loss
    comp_scores, comp_ready = trainer_train.latent_state.compute_competence_scores()
    
    # Gated loss (effective)
    loss_gated, stats_gated = v6i1_cf_separation_loss(
        model=cf_optimized_model,
        obs_batch=obs_batch_train,
        latent_k=cfg.latent_k,
        margin=cf_margin,
        competence=comp_scores,
        competence_ready=comp_ready,
        weak_pair_ema=getattr(trainer_train.latent_state, "cf_pair_jsd_ema", None),
        weak_pair_boost=cfg.latent_cf_weak_pair_boost,
        worst_pair_coef=cfg.latent_cf_worst_pair_coef,
        require_competence=cfg.latent_cf_require_competence,
    )
    
    g_CF_raw = get_gradients(loss_direct, actor_params)
    
    # Effective CF gradient
    effective_coef = 4.0
    scaled_cf_loss = effective_coef * loss_gated
    g_CF_eff = get_gradients(scaled_cf_loss, actor_params)
    
    # Disjoint parameter grouping setup
    param_groups = get_actor_param_groups(cf_optimized_model)
    
    alignment_report = {}
    
    # Helper to compute metrics for a parameter group
    def compute_group_alignment_metrics(g_ppo, g_cf_raw, g_cf_eff, parameters, filter_fn):
        ppo_flat = torch.cat([g.flatten() for p, g in zip(parameters, g_ppo) if filter_fn(p)])
        raw_flat = torch.cat([g.flatten() for p, g in zip(parameters, g_cf_raw) if filter_fn(p)])
        eff_flat = torch.cat([g.flatten() for p, g in zip(parameters, g_cf_eff) if filter_fn(p)])
        
        norm_ppo = torch.norm(ppo_flat).item()
        norm_raw = torch.norm(raw_flat).item()
        norm_eff = torch.norm(eff_flat).item()
        
        cos_raw = torch.dot(ppo_flat, raw_flat).item() / (norm_ppo * norm_raw + 1e-12) if norm_ppo > 1e-12 and norm_raw > 1e-12 else float("nan")
        cos_eff = torch.dot(ppo_flat, eff_flat).item() / (norm_ppo * norm_eff + 1e-12) if norm_ppo > 1e-12 and norm_eff > 1e-12 else float("nan")
        
        return cos_raw, cos_eff, norm_ppo, norm_raw, norm_eff
        
    for pathway, plist in param_groups.items():
        if pathway == "other" and not plist:
            continue
        filter_fn = lambda p: any(p is param for name, param in plist)
        cos_raw, cos_eff, norm_ppo, norm_raw, norm_eff = compute_group_alignment_metrics(
            g_PPO, g_CF_raw, g_CF_eff, actor_params, filter_fn
        )
        
        pathway_status = "cooperative" if cos_raw > 0.05 else ("conflict" if cos_raw < -0.05 else "orthogonal")
        if np.isnan(cos_raw):
            pathway_status = "undefined"
            
        print(f"Pathway: {pathway:15} | Cosine Raw: {cos_raw: .4f} | Cosine Eff: {cos_eff: .4f} | PPO norm: {norm_ppo:.4e} | CF Raw norm: {norm_raw:.4e} | CF Eff norm: {norm_eff:.4e}")
        alignment_report[pathway] = {
            "ppo_raw_cf_cosine": cos_raw,
            "ppo_effective_cf_cosine": cos_eff,
            "raw_cf_grad_norm": norm_raw,
            "effective_cf_grad_norm": norm_eff,
            "ppo_grad_norm": norm_ppo,
            "status_raw": pathway_status,
        }
        
    # Global Alignment
    cos_raw, cos_eff, norm_ppo, norm_raw, norm_eff = compute_group_alignment_metrics(
        g_PPO, g_CF_raw, g_CF_eff, actor_params, lambda p: True
    )
    alignment_report["global"] = {
        "ppo_raw_cf_cosine": cos_raw,
        "ppo_effective_cf_cosine": cos_eff,
        "raw_cf_grad_norm": norm_raw,
        "effective_cf_grad_norm": norm_eff,
        "ppo_grad_norm": norm_ppo,
        "status_raw": "cooperative" if cos_raw > 0.05 else ("conflict" if cos_raw < -0.05 else "orthogonal"),
    }
    
    # 7. Competence Gate Algebra Audit
    print("\n--- Competence Gate Algebra Audit ---")
    print(f"Competence Ready: {comp_ready}")
    print(f"Raw Competence Scores: {comp_scores}")
    
    g_raw = get_gradients(loss_direct, actor_params)
    g_gated = get_gradients(loss_gated, actor_params)
    
    m = float(loss_gated.item() / (loss_direct.item() + 1e-8))
    
    g_raw_flat = torch.cat([g.flatten() for g in g_raw])
    g_gated_flat = torch.cat([g.flatten() for g in g_gated])
    
    g_raw_norm = torch.norm(g_raw_flat).item()
    g_gated_norm = torch.norm(g_gated_flat).item()
    aggregate_gradient_norm = g_raw_norm
    
    predicted_norm = m * g_raw_norm
    norm_diff = abs(g_gated_norm - predicted_norm)
    print(f"Raw Loss: {loss_direct.item():.6f} | Gated Loss: {loss_gated.item():.6f} | Ratio (m): {m:.4f}")
    print(f"Raw Grad Norm: {g_raw_norm:.6e} | Gated Grad Norm: {g_gated_norm:.6e} | Predicted gated norm: {predicted_norm:.6e} | Diff: {norm_diff:.6e}")
    
    gate_audit = {
        "configured_cf_coefficient": 8.0,
        "coefficient_schedule_value": 0.5,
        "competence_requirement_enabled": cfg.latent_cf_require_competence,
        "raw_competence_scores": comp_scores.tolist(),
        "competence_gate_ready": comp_ready,
        "raw_loss": loss_direct.item(),
        "gated_loss": loss_gated.item(),
        "raw_gradient_norm": g_raw_norm,
        "gated_gradient_norm": g_gated_norm,
        "algebraic_ratio": m,
        "algebraic_norm_diff": norm_diff,
    }
    
    # 8. Pairwise Cancellation Test (Refinement 2)
    print("\n--- Pairwise Cancellation Test ---")
    pair_losses_grad = compute_pair_losses(cf_optimized_model, obs_batch_train, cf_margin, cfg.latent_k, DEVICE)
    
    pair_grads = []
    pair_losses = []
    pair_jsds = []
    
    for pair_idx, (zi, zj) in enumerate(PAIR_ORDER):
        pair_jsd_val = stats_direct["pair_jsd"][pair_idx].item()
        pair_loss_t = pair_losses_grad[pair_idx]
        
        pair_jsds.append(pair_jsd_val)
        pair_losses.append(pair_loss_t.item())
        
        g_p = get_gradients(pair_loss_t, actor_params)
        g_p_flat = torch.cat([g.flatten() for g in g_p])
        pair_grads.append(g_p_flat)
        
    # Build 6x6 pairwise cosine matrix
    pairwise_cosines = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            norm_i = torch.norm(pair_grads[i]).item()
            norm_j = torch.norm(pair_grads[j]).item()
            if norm_i < 1e-12 or norm_j < 1e-12:
                pairwise_cosines[i, j] = float("nan")
            else:
                dot = torch.dot(pair_grads[i], pair_grads[j]).item()
                pairwise_cosines[i, j] = dot / (norm_i * norm_j)
                
    # Calculate cooperation/cancellation ratio with vector norms:
    # C = ||sum_p g_p||_2 / (sum_p ||g_p||_2 + eps)
    sum_grads = torch.stack(pair_grads).sum(dim=0)
    norm_sum = torch.norm(sum_grads).item()
    sum_norms = sum(torch.norm(g).item() for g in pair_grads)
    cooperation_ratio = norm_sum / (sum_norms + 1e-12)
    
    print(f"Sum of Gradients Norm: {norm_sum:.6e}")
    print(f"Sum of Individual Norms: {sum_norms:.6e}")
    print(f"Cooperation Ratio C: {cooperation_ratio:.4f} (near 0 = cancellation, near 1 = cooperation)")
    
    # Verify mathematical identity: L_raw = sum(L_p) / 6.0 -> g_raw_reconstructed = sum(g_p) / 6.0
    g_raw_reconstructed = sum_grads / 6.0
    weighted_pair_gradient_sum_error_raw = torch.norm(g_raw_flat - g_raw_reconstructed).item()
    print(f"Identity check (Raw): norm(g_raw_flat - g_raw_reconstructed) = {weighted_pair_gradient_sum_error_raw:.6e}")
    assert weighted_pair_gradient_sum_error_raw < 1e-4, f"Pair-gradient aggregation cannot reproduce the authoritative loss gradient! Error: {weighted_pair_gradient_sum_error_raw}"
    
    # Reconstructed gated/effective gradient
    gated_weights = stats_gated["cf_pair_weight"]
    gated_weight_sum = gated_weights.sum().item()
    worst_pair_idx = int(stats_gated["cf_worst_pair_index"].item())
    worst_coef = float(stats_gated["cf_worst_pair_coef"].item())
    
    if gated_weight_sum > 1e-8:
        weighted_sum_g_p = torch.zeros_like(sum_grads)
        for idx in range(6):
            weighted_sum_g_p = weighted_sum_g_p + gated_weights[idx].item() * pair_grads[idx]
        g_gated_reconstructed = weighted_sum_g_p / gated_weight_sum
        g_gated_reconstructed = g_gated_reconstructed + worst_coef * pair_grads[worst_pair_idx]
    else:
        g_gated_reconstructed = torch.zeros_like(sum_grads)
        
    weighted_pair_gradient_sum_error_gated = torch.norm(g_gated_flat - g_gated_reconstructed).item()
    print(f"Identity check (Gated): norm(g_gated_flat - g_gated_reconstructed) = {weighted_pair_gradient_sum_error_gated:.6e}")
    
    # 9. Frozen-batch CF-only Optimization Loop
    print("\n--- Frozen-batch CF-only Optimization Loop ---")
    opt_lr = 1e-4
    optimizer = torch.optim.Adam(actor_params, lr=opt_lr)
    
    # Save step 0 parameter values to compute deltas
    step_0_params = {name: p.clone().detach() for name, p in cf_optimized_model.named_parameters() if p.requires_grad}
    
    train_curve = []
    heldout_curve = []
    recorded_steps = [0, 1, 2, 5, 10, 25, 50, 100, 200]
    
    for step in range(201):
        # Forward pass on Train
        loss_t, stats_t = v6i1_cf_separation_loss(
            model=cf_optimized_model,
            obs_batch=obs_batch_train,
            latent_k=cfg.latent_k,
            margin=cf_margin,
            competence=competence,
            competence_ready=competence_ready,
            weak_pair_ema=None,
            weak_pair_boost=0.0,
            worst_pair_coef=0.0,
            require_competence=False,
        )
        
        # Forward pass on Heldout (no grad)
        with torch.no_grad():
            loss_h, stats_h = v6i1_cf_separation_loss(
                model=cf_optimized_model,
                obs_batch=obs_batch_heldout,
                latent_k=cfg.latent_k,
                margin=cf_margin,
                competence=competence,
                competence_ready=competence_ready,
                weak_pair_ema=None,
                weak_pair_boost=0.0,
                worst_pair_coef=0.0,
                require_competence=False,
            )
            
        # Backward pass to populate grads
        optimizer.zero_grad()
        loss_t.backward(retain_graph=True)
        
        # Compute gradient norms
        act_grad_norm_sq = 0.0
        z_grad_norm = 0.0
        for name, param in cf_optimized_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                g_sq = (param.grad.detach() ** 2).sum().item()
                if classify_actor_pathway(name) == "z_embedding":
                    z_grad_norm = g_sq ** 0.5
                else:
                    act_grad_norm_sq += g_sq
        act_grad_norm = act_grad_norm_sq ** 0.5
        
        # Compute parameter deltas
        act_delta_sq = 0.0
        z_delta = 0.0
        for name, param in cf_optimized_model.named_parameters():
            if param.requires_grad:
                p_0 = step_0_params[name]
                d_sq = ((param.detach() - p_0) ** 2).sum().item()
                if classify_actor_pathway(name) == "z_embedding":
                    z_delta = d_sq ** 0.5
                else:
                    act_delta_sq += d_sq
        act_delta = act_delta_sq ** 0.5
        
        # Check safety limits and NaNs
        if torch.isnan(loss_t) or torch.isinf(loss_t):
            print(f"Safety Stop: NaN/Inf Loss at step {step}!")
            break
        if np.isnan(act_grad_norm) or np.isnan(z_grad_norm):
            print(f"Safety Stop: NaN Gradient Norm at step {step}!")
            break
            
        # Record curves
        train_jsd = stats_t["jsd"].item()
        heldout_jsd = stats_h["jsd"].item()
        
        if step in recorded_steps:
            print(f"Step {step:3d} | Train Loss: {loss_t.item():.6f} | Train JSD: {train_jsd:.6f} | Heldout JSD: {heldout_jsd:.6f} | Actor Delta: {act_delta:.4e} | z-embed Delta: {z_delta:.4e}")
            train_curve.append({
                "step": step,
                "loss": loss_t.item(),
                "jsd": train_jsd,
                "pair_jsds": stats_t["pair_jsd"].tolist(),
                "actor_delta": act_delta,
                "z_embed_delta": z_delta,
                "actor_grad_norm": act_grad_norm,
                "z_embed_grad_norm": z_grad_norm,
            })
            heldout_curve.append({
                "step": step,
                "loss": loss_h.item(),
                "jsd": heldout_jsd,
                "pair_jsds": stats_h["pair_jsd"].tolist(),
            })
            
        # Step optimizer
        if step < 200:
            optimizer.step()
            
    # Verify reference and non-actor components are unchanged (Refinement 4)
    ref_fingerprint_end = compute_fingerprint(reference_model)
    opt_fingerprint_end = compute_fingerprint(cf_optimized_model)
    
    # Assert reference fingerprint remains unchanged
    assert ref_fingerprint_start == ref_fingerprint_end, "Pristine reference model copy was mutated!"
    
    # Calculate hashes after diagnostics
    actor_hash_end = compute_module_hash(cf_optimized_model.latent_actor)
    ref_actor_hash_end = compute_module_hash(reference_model.latent_actor)
    
    critic_hash_train_end = compute_module_hash(trainer_train.model.critic)
    critic_hash_opt_end = compute_module_hash(cf_optimized_model.critic)
    critic_hash_ref_end = compute_module_hash(reference_model.critic)
    
    router_hash_train_end = compute_module_hash(trainer_train.model.strategy_encoder)
    router_hash_opt_end = compute_module_hash(cf_optimized_model.strategy_encoder)
    router_hash_ref_end = compute_module_hash(reference_model.strategy_encoder)
    
    norm_hash_train_end = compute_norm_hash(trainer_train)
    norm_hash_heldout_end = compute_norm_hash(trainer_heldout)
    
    batch_train_hash_end = compute_batch_hash(B_train)
    batch_heldout_hash_end = compute_batch_hash(B_heldout)
    
    # Assertions for reference model and non-actor modules
    assert actor_hash_start == ref_actor_hash_end, "Reference actor was mutated during diagnostics!"
    assert critic_hash_start == critic_hash_opt_end, "Critic in optimized model was mutated!"
    assert critic_hash_start == critic_hash_ref_end, "Critic in reference model was mutated!"
    assert router_hash_start == router_hash_opt_end, "Router in optimized model was mutated!"
    assert router_hash_start == router_hash_ref_end, "Router in reference model was mutated!"
    
    # Assert normalizers are unchanged
    assert norm_hash_train_start == norm_hash_train_end, "Train normalizer was mutated!"
    assert norm_hash_heldout_start == norm_hash_heldout_end, "Heldout normalizer was mutated!"
    
    # Assert batches are unchanged
    assert batch_train_hash_start == batch_train_hash_end, "Train batch was mutated!"
    assert batch_heldout_hash_start == batch_heldout_hash_end, "Heldout batch was mutated!"
    
    torch_rng_state_end = torch.get_rng_state()
    numpy_rng_state_end = np.random.get_state()
    rng_torch_equal = torch.equal(torch_rng_state_start, torch_rng_state_end)
    rng_numpy_equal = all(np.array_equal(a, b) for a, b in zip(numpy_rng_state_start, numpy_rng_state_end))
    
    print("\n--- State Integrity Checks ---")
    print(f"Actor start hash:         {actor_hash_start[:16]}")
    print(f"Actor end hash (opt):     {actor_hash_end[:16]}")
    print(f"Actor end hash (ref):     {ref_actor_hash_end[:16]}")
    print(f"Critic start hash:        {critic_hash_start[:16]}")
    print(f"Critic end hash (opt):    {critic_hash_opt_end[:16]}")
    print(f"Critic end hash (ref):    {critic_hash_ref_end[:16]}")
    print(f"Router start hash:        {router_hash_start[:16]}")
    print(f"Router end hash (opt):    {router_hash_opt_end[:16]}")
    print(f"Router end hash (ref):    {router_hash_ref_end[:16]}")
    print(f"Normalizer train start/end match: {norm_hash_train_start == norm_hash_train_end}")
    print(f"Normalizer heldout start/end match: {norm_hash_heldout_start == norm_hash_heldout_end}")
    print(f"Batch train start/end match: {batch_train_hash_start == batch_train_hash_end}")
    print(f"Batch heldout start/end match: {batch_heldout_hash_start == batch_heldout_hash_end}")
    print(f"Torch RNG state matches:   {rng_torch_equal}")
    print(f"Numpy RNG state matches:   {rng_numpy_equal}")
    print("Fingerprint check: pristine copies and non-actor components match start and end exactly.")
    
    # 10. Output machine-readable artifacts
    artifact_dir = "C:/Users/K-B/.gemini/antigravity/brain/f82565a1-de91-45a6-a045-88d907692492"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Write train curve CSV
    train_curve_path = os.path.join(artifact_dir, "v6i5_cf_only_train_curve.csv")
    with open(train_curve_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "jsd", "actor_delta", "z_embed_delta", "actor_grad_norm", "z_embed_grad_norm"])
        for row in train_curve:
            writer.writerow([row["step"], row["loss"], row["jsd"], row["actor_delta"], row["z_embed_delta"], row["actor_grad_norm"], row["z_embed_grad_norm"]])
            
    # Write heldout curve CSV
    heldout_curve_path = os.path.join(artifact_dir, "v6i5_cf_only_heldout_curve.csv")
    with open(heldout_curve_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "jsd"])
        for row in heldout_curve:
            writer.writerow([row["step"], row["loss"], row["jsd"]])
            
    # Write alignment JSON
    alignment_path = os.path.join(artifact_dir, "v6i5_cf_ppo_alignment.json")
    with open(alignment_path, "w", encoding="utf-8") as f:
        json.dump(alignment_report, f, indent=2)
        
    # Write gate audit JSON
    gate_audit_path = os.path.join(artifact_dir, "v6i5_cf_gate_audit.json")
    with open(gate_audit_path, "w", encoding="utf-8") as f:
        json.dump(gate_audit, f, indent=2)
        
    # Write cosines CSV
    cosines_path = os.path.join(artifact_dir, "v6i5_cf_pairwise_cosines.csv")
    with open(cosines_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair"] + [f"pair_{idx}" for idx in range(6)])
        for idx in range(6):
            label = f"{PAIR_ORDER[idx][0]}{PAIR_ORDER[idx][1]}"
            writer.writerow([label] + pairwise_cosines[idx].tolist())
            
    # Write manifest JSON
    manifest = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": ckpt_hash,
        "git_commit": prior_data.get("git_sha"),
        "preset": "v6i5",
        "resolved_configuration": prior_resolved,
        "train_seed": 42,
        "heldout_seed": 9999,
        "diagnostic_seed": DIAG_SEED,
        "device": str(DEVICE),
        "offline_optimizer_settings": {"type": "Adam", "lr": opt_lr},
    }
    manifest_path = os.path.join(artifact_dir, "v6i5_cf_diagnostics_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    # Write report JSON
    report = {
        "one_step_sgd_test": {
            "loss_before": loss_direct_before.item(),
            "loss_after": loss_direct_after.item(),
            "jsd_before": jsd_before,
            "jsd_after": jsd_after,
            "pair_jsd_before": pair_jsd_before,
            "pair_jsd_after": pair_jsd_after,
        },
        "cancellation_ratio_metrics": {
            "cooperation_ratio": cooperation_ratio,
            "aggregate_gradient_norm": aggregate_gradient_norm,
            "sum_pair_gradient_norms": sum_norms,
            "weighted_pair_gradient_sum_error_raw": weighted_pair_gradient_sum_error_raw,
            "weighted_pair_gradient_sum_error_gated": weighted_pair_gradient_sum_error_gated,
        },
        "integrity_hashes": {
            "actor_hash_start": actor_hash_start,
            "actor_hash_end": actor_hash_end,
            "ref_actor_hash_end": ref_actor_hash_end,
            "critic_hash_start": critic_hash_start,
            "critic_hash_opt_end": critic_hash_opt_end,
            "critic_hash_ref_end": critic_hash_ref_end,
            "router_hash_start": router_hash_start,
            "router_hash_opt_end": router_hash_opt_end,
            "router_hash_ref_end": router_hash_ref_end,
            "norm_hash_train_start": norm_hash_train_start,
            "norm_hash_train_end": norm_hash_train_end,
            "norm_hash_heldout_start": norm_hash_heldout_start,
            "norm_hash_heldout_end": norm_hash_heldout_end,
            "batch_train_hash_start": batch_train_hash_start,
            "batch_train_hash_end": batch_train_hash_end,
            "batch_heldout_hash_start": batch_heldout_hash_start,
            "batch_heldout_hash_end": batch_heldout_hash_end,
            "rng_torch_equal": rng_torch_equal,
            "rng_numpy_equal": rng_numpy_equal,
        },
        "train_curve": train_curve,
        "heldout_curve": heldout_curve,
        "alignment": alignment_report,
        "gate_audit": gate_audit,
        "pairwise_cosines": pairwise_cosines.tolist(),
        "cooperation_ratio_status": "cancellation" if cooperation_ratio < 0.25 else "cooperative",
    }
    report_path = os.path.join(artifact_dir, "v6i5_cf_diagnostics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nDiagnostics completed. Outputs written to: {artifact_dir}")
    print("=== End of v6i5 CF Diagnostics ===")

if __name__ == "__main__":
    main()
