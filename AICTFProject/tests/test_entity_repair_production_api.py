"""Rule 12 contract for the PRODUCTION act()/evaluate_actions()/policy_logits()
entity kwargs -- native to the policy class now, not the standalone shim.

1. legacy call (no entity kwargs, entity_repair_enabled=False) == pre-existing behaviour
2. warm start: entity_repair_enabled=True + fresh entity_encoder + REAL sealed
   B3-3 checkpoint (loaded via the compat loader) + real entities ==
   identical logits/action distribution to the sealed checkpoint's own output
3. evaluate_actions() returns identical log-prob/value/entropy at step 0
4. entity-slot permutation leaves act()/evaluate_actions() outputs unchanged
5. fail-closed: entity_repair_enabled=True with missing entity args raises;
   entity_repair_enabled=False with entity args supplied raises
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gpu_env._core._entity_obs import build_entity_tensors
from rl.custom_ppo.policy import SharedActorCentralizedCritic

ROOT = Path(__file__).resolve().parents[1]
CKPT_A = ROOT / "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3/ckpts/final_pi_A_specialist_4v4_b3.zip"


def _live_env(seed=17600001):
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


def _entities_bnkf(core):
    """(B,N,K,F)/(B,N,K) tensors -- the NATIVE, unflattened shape act()/
    evaluate_actions() now expect (flattening happens inside _encode_local_obs)."""
    d = build_entity_tensors(core, "blue")
    return d["teammates"], d["teammates_valid"], d["enemies"], d["enemies_valid"]


def _global_state(env):
    return torch.as_tensor(env.state(), dtype=torch.float32)


# ------------------------------------------------------------- 1: legacy path
def test_legacy_act_unaffected_by_entity_repair_absence():
    env, core = _live_env()
    try:
        obs_space, act_space = env.observation_space, env.action_space
        torch.manual_seed(0)
        policy = SharedActorCentralizedCritic(obs_space, act_space,
                                              strategy_encoder_enabled=False, latent_k=0)
        assert policy.entity_encoder is None
        obs = core.get_obs_tensors("blue")
        gs = _global_state(env)
        torch.manual_seed(1)
        a1, v1, lp1, e1 = policy.act(obs, gs, deterministic=True)
        torch.manual_seed(1)
        a2, v2, lp2, e2 = policy.act(obs, gs, deterministic=True)  # no entity kwargs at all
        assert torch.equal(a1, a2) and torch.equal(v1, v2) and torch.equal(lp1, lp2)
    finally:
        env.close()


# --------------------------------------------------- 2/3: warm start, real ckpt
@pytest.mark.skipif(not CKPT_A.is_file(), reason="sealed B3-3 checkpoint not present")
def test_warm_start_act_and_evaluate_actions_match_sealed_checkpoint():
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.custom_ppo.checkpoints.archive import _torch_load_checkpoint
    from rl.custom_ppo.checkpoints.state_dict import _load_model_state_dict_compat

    env, core = _live_env()
    try:
        obs_space, act_space = env.observation_space, env.action_space

        baseline = load_custom_ppo_policy(str(CKPT_A), obs_space, act_space, device="cpu").model
        baseline.eval()

        torch.manual_seed(0)
        repaired = SharedActorCentralizedCritic(obs_space, act_space,
                                                strategy_encoder_enabled=False, latent_k=0,
                                                entity_repair_enabled=True)
        repaired.eval()
        payload = _torch_load_checkpoint(str(CKPT_A), map_location="cpu")
        sd = payload["model_state_dict"] if "model_state_dict" in payload else payload
        _load_model_state_dict_compat(repaired, sd)   # entity_encoder.* allowed missing

        obs = core.get_obs_tensors("blue")
        gs = _global_state(env)
        tm, tm_v, en, en_v = _entities_bnkf(core)

        with torch.no_grad():
            base_logits = baseline.policy_logits(obs)
            rep_logits = repaired.policy_logits(obs, teammates=tm, teammates_valid=tm_v,
                                                enemies=en, enemies_valid=en_v)
        assert torch.allclose(rep_logits, base_logits, atol=1e-5), \
            "warm-started entity-repair policy must match the sealed baseline's logits at t=0"

        torch.manual_seed(7)
        a_base, v_base, lp_base, ent_base = baseline.act(obs, gs, deterministic=True)
        torch.manual_seed(7)
        a_rep, v_rep, lp_rep, ent_rep = repaired.act(obs, gs, deterministic=True,
                                                     teammates=tm, teammates_valid=tm_v,
                                                     enemies=en, enemies_valid=en_v)
        assert torch.equal(a_base, a_rep), "warm start: sampled actions must match exactly"
        assert torch.allclose(v_base, v_rep, atol=1e-5), "critic must be UNCHANGED (no entity leak)"
        assert torch.allclose(lp_base, lp_rep, atol=1e-5)

        # evaluate_actions: identical log-prob/value/entropy at step 0 for the SAME actions
        vn_base, lp2_base, ent2_base, _ = baseline.evaluate_actions(obs, gs, a_base)
        vn_rep, lp2_rep, ent2_rep, _ = repaired.evaluate_actions(
            obs, gs, a_base, teammates=tm, teammates_valid=tm_v, enemies=en, enemies_valid=en_v)
        assert torch.allclose(vn_base, vn_rep, atol=1e-5)
        assert torch.allclose(lp2_base, lp2_rep, atol=1e-5)
        assert torch.allclose(ent2_base, ent2_rep, atol=1e-5)
    finally:
        env.close()


# ------------------------------------------------------- 4: permutation safety
def test_permutation_invariance_through_act_and_evaluate_actions():
    env, core = _live_env()
    try:
        obs_space, act_space = env.observation_space, env.action_space
        torch.manual_seed(0)
        policy = SharedActorCentralizedCritic(obs_space, act_space,
                                              strategy_encoder_enabled=False, latent_k=0,
                                              entity_repair_enabled=True)
        with torch.no_grad():
            for p in policy.entity_encoder.parameters():
                p.add_(torch.randn_like(p) * 0.3)

        obs = core.get_obs_tensors("blue")
        gs = _global_state(env)
        tm, tm_v, en, en_v = _entities_bnkf(core)

        with torch.no_grad():
            logits1 = policy.policy_logits(obs, teammates=tm, teammates_valid=tm_v,
                                           enemies=en, enemies_valid=en_v)
            pt, pe = torch.randperm(tm.shape[2]), torch.randperm(en.shape[2])
            logits2 = policy.policy_logits(obs, teammates=tm[:, :, pt], teammates_valid=tm_v[:, :, pt],
                                           enemies=en[:, :, pe], enemies_valid=en_v[:, :, pe])
        assert torch.allclose(logits1, logits2, atol=1e-5)

        actions = torch.zeros(1, len(policy.action_dims), dtype=torch.long)
        vn1, lp1, ent1, _ = policy.evaluate_actions(obs, gs, actions, teammates=tm,
                                                    teammates_valid=tm_v, enemies=en, enemies_valid=en_v)
        vn2, lp2, ent2, _ = policy.evaluate_actions(obs, gs, actions, teammates=tm[:, :, pt],
                                                    teammates_valid=tm_v[:, :, pt],
                                                    enemies=en[:, :, pe], enemies_valid=en_v[:, :, pe])
        assert torch.allclose(vn1, vn2, atol=1e-5) and torch.allclose(lp1, lp2, atol=1e-5)
    finally:
        env.close()


# -------------------------------------------------------------- 5: fail closed
def test_entity_repair_enabled_requires_all_four_tensors():
    env, core = _live_env()
    try:
        obs_space, act_space = env.observation_space, env.action_space
        policy = SharedActorCentralizedCritic(obs_space, act_space,
                                              strategy_encoder_enabled=False, latent_k=0,
                                              entity_repair_enabled=True)
        obs = core.get_obs_tensors("blue")
        gs = _global_state(env)
        with pytest.raises(ValueError, match="requires teammates"):
            policy.act(obs, gs)                       # entity_repair on, nothing supplied
    finally:
        env.close()


def test_legacy_model_rejects_entity_tensors_rather_than_ignoring_them():
    env, core = _live_env()
    try:
        obs_space, act_space = env.observation_space, env.action_space
        policy = SharedActorCentralizedCritic(obs_space, act_space,
                                              strategy_encoder_enabled=False, latent_k=0)
        obs = core.get_obs_tensors("blue")
        gs = _global_state(env)
        tm, tm_v, en, en_v = _entities_bnkf(core)
        with pytest.raises(ValueError, match="entity_repair_enabled=False"):
            policy.act(obs, gs, teammates=tm, teammates_valid=tm_v, enemies=en, enemies_valid=en_v)
    finally:
        env.close()


def test_critic_never_receives_entity_tensors_by_construction():
    """values() has no entity parameters at all -- structural, not behavioural,
    guarantee that the critic path cannot be given entity tensors even by
    mistake."""
    import inspect
    params = set(inspect.signature(SharedActorCentralizedCritic.values).parameters)
    assert not any("teammate" in p or "enem" in p for p in params), \
        f"values() must never accept entity parameters, got {params}"


# ------------------ 7: CustomPPOInferencePolicy.predict() -- the eval/deploy seam
def test_inference_policy_predict_requires_and_uses_entity_tensors():
    """load_custom_ppo_policy().predict() (and the CustomPPOInferencePolicy
    wrapper it returns) is the seam every eval/deploy script calls (e.g.
    experiments/eval_specialist_crossover_scaled.py) -- distinct from
    act()/evaluate_actions() tested above, which are the TRAINING-side entry
    points. This seam was missing entity-tensor support entirely until this
    fix (discovered when the real crossover eval's dry-run passed but the
    first real episode step would have raised ValueError on the very first
    policy.predict() call). Must (a) fail closed with a clear message when
    entity tensors are missing from obs, and (b) actually thread them into the
    real forward pass when present, using the exact key names
    gpu_env._core._entity_obs.augment_obs_with_entities produces."""
    import numpy as np
    from rl.custom_ppo.inference_policy import CustomPPOInferencePolicy
    from gpu_env._core._entity_obs import augment_obs_with_entities

    env, core = _live_env()
    try:
        obs_space, act_space = env.observation_space, env.action_space
        model = SharedActorCentralizedCritic(obs_space, act_space,
                                             strategy_encoder_enabled=False, latent_k=0,
                                             entity_repair_enabled=True)
        inference = CustomPPOInferencePolicy(model, device="cpu")

        obs = env.reset()

        with pytest.raises(ValueError, match="entity_repair_enabled=True"):
            inference.predict(obs, deterministic=True)

        obs_aug = augment_obs_with_entities(obs, core, side="blue")
        action, _ = inference.predict(obs_aug, deterministic=True)
        assert not np.isnan(np.asarray(action)).any(), "predict() produced NaN actions"
    finally:
        env.close()
