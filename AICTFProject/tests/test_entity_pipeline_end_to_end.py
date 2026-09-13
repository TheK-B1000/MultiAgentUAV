"""End-to-end anchors: real env extraction -> rollout-shaped batch -> real
policy, including the REAL sealed B3-3 4v4 checkpoint.

These close the loop the policy-level tests (tests/test_entity_residual.py)
could not: that the identity guarantees survive contact with production
extraction code, not only synthetic tensors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gpu_env._core._entity_obs import build_entity_tensors, flatten_for_policy
from rl.custom_ppo.entity_residual import EntityResidualEncoder, augmented_local_in

ROOT = Path(__file__).resolve().parents[1]
CKPT_A = ROOT / "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3/ckpts/final_pi_A_specialist_4v4_b3.zip"


def _live_4v4_env(seed=17600001):
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome
    from experiments.strategic_demand_searcher import apply_genome_to_core
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    S.AGENTS = 4
    g = pole_A_genome(4)
    cfg = GPUFieldConfig(n_envs=1, max_blue_agents=4, max_red_agents=4, map_set="train",
                         map_layout=S.MAP, max_decision_steps=S.MAX_STEPS, aquaticus_profile=True,
                         rules_profile="OURS", device="cpu", seed=seed, obstacle_obs_channel=True,
                         tag_telemetry_enabled=True, own_flag_home_required_to_score=True, **S.RULESET)
    env = GPUCTFVecEnv(cfg); core = env.core
    opp = g.base_opponent
    env.env_method("set_phase", opp)
    env.env_method("set_next_opponent", "SCRIPTED", opp)
    apply_genome_to_core(core, g)
    core.blue_scripted = True
    core.set_blue_style(S.GUARD)
    env.reset(); apply_genome_to_core(core, g); core.drain_tag_events()
    return env, core


def _advance_ticks(env, core, n=20):
    """Let the scripted world evolve a bit so agents are not all at spawn --
    a real, non-degenerate rollout state."""
    import numpy as np
    for _ in range(n):
        env.step_async(env.action_space.sample() * 0)
        _o, _r, d, _i = env.step_wait()
        if bool(np.asarray(d).any()):
            break


# --------------------------------------------- 4: permutation through the real path
def test_permutation_safety_through_real_extraction_and_policy():
    """Take REAL env-derived entity tensors (not synthetic), permute the
    teammate/enemy SLOT order together with their validity masks, and confirm
    the full extraction -> flatten -> policy pipeline gives identical logits."""
    from gymnasium import spaces

    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    env, core = _live_4v4_env()
    try:
        _advance_ticks(env, core, 15)
        d = build_entity_tensors(core, "blue")
        tm, tm_v, en, en_v = flatten_for_policy(d)     # (B*N, K, F) real geometry

        obs_space, act_space = env.observation_space, env.action_space
        torch.manual_seed(0)
        policy = SharedActorCentralizedCritic(obs_space, act_space,
                                              strategy_encoder_enabled=False, latent_k=0)
        mod = EntityResidualEncoder(out_dim=policy._local_actor_in_dim)
        with torch.no_grad():                          # move off zero-init so g is nontrivial
            for p in mod.parameters():
                p.add_(torch.randn_like(p) * 0.3)

        obs = core.get_obs_tensors("blue")
        aug1, _, _, _ = augmented_local_in(policy, obs, tm, tm_v, en, en_v, mod)

        pt = torch.randperm(tm.shape[1]); pe = torch.randperm(en.shape[1])
        aug2, _, _, _ = augmented_local_in(policy, obs, tm[:, pt], tm_v[:, pt],
                                           en[:, pe], en_v[:, pe], mod)
        assert torch.allclose(aug1, aug2, atol=1e-5), \
            "real-geometry entity permutation changed the fused local_in"
    finally:
        env.close()


# ------------------------------------- 5: warm-start equivalence, real checkpoint
@pytest.mark.skipif(not CKPT_A.is_file(), reason="sealed B3-3 checkpoint not present")
def test_warm_start_equivalence_with_real_sealed_checkpoint_and_real_entities():
    """Load the REAL sealed pi_A_specialist_4v4_b3 checkpoint. Run a REAL
    observation through it two ways: (a) the checkpoint's own forward pass,
    (b) the entity-augmented path with REAL, non-empty entity tensors and a
    freshly-initialised EntityResidualEncoder. Logits must match to the
    tolerance already frozen in the policy-level contract."""
    from rl.custom_ppo import load_custom_ppo_policy

    env, core = _live_4v4_env()
    try:
        obs_space, act_space = env.observation_space, env.action_space
        inference = load_custom_ppo_policy(str(CKPT_A), obs_space, act_space, device="cpu")
        policy = inference.model
        policy.eval()
        _advance_ticks(env, core, 25)
        obs = core.get_obs_tensors("blue")
        d = build_entity_tensors(core, "blue")
        tm, tm_v, en, en_v = flatten_for_policy(d)

        with torch.no_grad():
            base_logits = policy.policy_logits(obs)

            mod = EntityResidualEncoder(out_dim=policy._local_actor_in_dim)  # fresh, zero-init
            aug_local_in, local_base, _, _ = augmented_local_in(
                policy, obs, tm, tm_v, en, en_v, mod)
            assert torch.equal(aug_local_in, local_base), \
                "fresh module must contribute exactly 0 even with real checkpoint + real entities"
            batch = int(obs["grid"].shape[0])
            aug_flat = policy.latent_actor(aug_local_in)
            aug_logits = aug_flat.reshape(batch, policy.n_agents * policy.per_agent_logits)

        assert torch.allclose(aug_logits, base_logits, atol=1e-5), \
            "REAL sealed checkpoint: augmented-path logits must equal base logits at warm start"
    finally:
        env.close()


# ------------------------------------------------- 6: the pathway can wake up
def test_entity_pathway_receives_gradient_and_updates():
    """Zero-init is intentional at t=0, but the pathway must not be a
    permanent trapdoor. One optimizer step must move the projection weight,
    produce no NaNs, and make the augmented logits diverge from the (fixed)
    base logits once real entity content has a nonzero-weighted path."""
    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    env, core = _live_4v4_env()
    try:
        obs_space, act_space = env.observation_space, env.action_space
        torch.manual_seed(0)
        policy = SharedActorCentralizedCritic(obs_space, act_space,
                                              strategy_encoder_enabled=False, latent_k=0)
        for p in policy.parameters():                    # freeze the base network:
            p.requires_grad_(False)                       # isolate the entity path's own gradient
        mod = EntityResidualEncoder(out_dim=policy._local_actor_in_dim)
        opt = torch.optim.SGD(mod.parameters(), lr=1.0)  # large lr: a step must be visible
        _advance_ticks(env, core, 10)
        obs = core.get_obs_tensors("blue")
        d = build_entity_tensors(core, "blue")
        tm, tm_v, en, en_v = flatten_for_policy(d)

        with torch.no_grad():
            base_logits = policy.policy_logits(obs)

        proj_w_before = mod.proj.weight.detach().clone()
        assert torch.equal(proj_w_before, torch.zeros_like(proj_w_before))

        aug_local_in, local_base, _, _ = augmented_local_in(policy, obs, tm, tm_v, en, en_v, mod)
        batch = int(obs["grid"].shape[0])
        aug_flat = policy.latent_actor(aug_local_in)
        aug_logits = aug_flat.reshape(batch, policy.n_agents * policy.per_agent_logits)

        loss = aug_logits.pow(2).sum()          # arbitrary differentiable scalar
        opt.zero_grad(); loss.backward()

        assert mod.proj.weight.grad is not None, "entity projection received NO gradient"
        assert not torch.isnan(mod.proj.weight.grad).any(), "NaN gradient"
        assert float(mod.proj.weight.grad.abs().sum()) > 0.0, "gradient is exactly zero"

        opt.step()
        assert not torch.equal(mod.proj.weight, proj_w_before), \
            "optimizer step did not change the entity projection weight"
        assert not torch.isnan(mod.proj.weight).any()

        with torch.no_grad():
            aug_local_in2, _, _, _ = augmented_local_in(policy, obs, tm, tm_v, en, en_v, mod)
            aug_flat2 = policy.latent_actor(aug_local_in2)
            aug_logits2 = aug_flat2.reshape(batch, policy.n_agents * policy.per_agent_logits)
        assert not torch.allclose(aug_logits2, base_logits, atol=1e-5), \
            "after one optimizer step, real entity content must be able to move logits"
    finally:
        env.close()


# ------------------ 7: warm start (--load-path) is initialization, not resume
@pytest.mark.skipif(not CKPT_A.is_file(), reason="sealed B3-3 checkpoint not present")
def test_warm_start_reset_progress_starts_fresh_global_step():
    """4V4_ENTITY_REPAIR_SPEC.json commits to "same training budget (steps) as
    the sealed B3-3 baseline" for the repaired specialists. B3-3's own
    global_step is 1,001,472 -- if warm-starting carried that over, a
    "1,000,000-step budget" would silently mean total_timesteps=2,001,472 (an
    unintended DOUBLE budget), or worse, immediately satisfy `global_step <
    total_timesteps` and train for zero steps (the actual failure this
    surfaced as during launcher testing). warm_start_reset_progress=True must
    make trainer.load(...) treat the checkpoint as pure weight initialization:
    global_step/updates_completed reset to 0, while model weights still load
    and remain behaviourally identical to the checkpoint at that instant."""
    from rl.config.ppo_config import PPOConfig
    from rl.custom_ppo.entity_residual import EntityResidualEncoder
    from rl.custom_ppo.trainer import CustomPPOTrainer

    env, core = _live_4v4_env()
    try:
        checkpoint_global_step = int(torch.load(str(CKPT_A), map_location="cpu",
                                                weights_only=False)["global_step"])
        assert checkpoint_global_step > 0, "fixture assumption: sealed checkpoint has trained steps"

        cfg = PPOConfig()
        cfg.device = "cpu"
        cfg.n_steps = 8
        cfg.use_latent_strategy = False
        cfg.normalize_returns = False
        cfg.entity_repair_enabled = True
        cfg.allow_active_actor_module_migration = True  # entity_encoder is new vs. old optimizer
        n_envs = int(env.num_envs)
        trainer = CustomPPOTrainer(env, cfg, learning_rate=1e-4, clip_range=0.2,
                                  ent_coef=0.01, n_epochs=1, batch_size=cfg.n_steps * n_envs)
        assert trainer.model.entity_encoder is not None

        _advance_ticks(env, core, 10)
        obs = core.get_obs_tensors("blue")
        d = build_entity_tensors(core, "blue")
        tm, tm_v, en, en_v = d["teammates"], d["teammates_valid"], d["enemies"], d["enemies_valid"]
        with torch.no_grad():
            pre_load_logits = trainer.model.policy_logits(
                obs, teammates=tm, teammates_valid=tm_v, enemies=en, enemies_valid=en_v)

        trainer.load(str(CKPT_A), reset_progress=True)

        assert trainer.global_step == 0, \
            f"reset_progress=True must zero global_step, got {trainer.global_step}"
        assert trainer._updates_completed == 0

        with torch.no_grad():
            post_load_logits = trainer.model.policy_logits(
                obs, teammates=tm, teammates_valid=tm_v, enemies=en, enemies_valid=en_v)
        assert not torch.allclose(pre_load_logits, post_load_logits, atol=1e-5), \
            "warm start must actually change the weights (loaded from a trained checkpoint, " \
            "not left at random init) -- equal logits would mean the load silently no-op'd"
    finally:
        env.close()
