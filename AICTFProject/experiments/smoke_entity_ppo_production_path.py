r"""Step 5 (final gate): one real PPO smoke through the ACTUAL production path.

    rollout (CustomPPOTrainer.collect_rollout, real GPU env, real GAE)
      -> real TensorDictRolloutBuffer minibatches
      -> real clipped PPO loss (CustomPPOTrainer.update)
      -> optimizer step
      -> checkpoint save / reload

NOT a hand-rolled REINFORCE loop (that was the earlier mechanical smoke). This
exercises rl/custom_ppo/trainer.py, rollout/collector.py, rollout/buffer_writer.py,
update/minibatch_updater.py and rl/ppo_core.py's GAE/minibatch code completely
unmodified in their CONTROL FLOW -- only the entity kwargs I added are new.

SCOPE NOTE: entity_encoder is attached to trainer.model POST-CONSTRUCTION and
its parameters are added to the existing shared optimizer's param group,
rather than wiring entity_repair_enabled through PPOConfig/build_model_kwargs.
This avoids touching PPOConfig (541-entry preset snapshot regeneration risk,
and the file already carries unrelated local modifications per git status) for
what is a plumbing smoke, not the final training launch config. Proper
PPOConfig + CLI wiring for the actual frozen launch is the next, separate,
low-risk mechanical step -- named explicitly at the end of this script's output.

    python -m experiments.smoke_entity_ppo_production_path
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LABEL = "ENTITY_PPO_PRODUCTION_SMOKE"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_env(n_envs=4, n_agents=4, seed=17600001, device="cpu"):
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome
    from experiments.strategic_demand_searcher import apply_genome_to_core
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    S.AGENTS = n_agents
    g = pole_A_genome(n_agents)
    cfg = GPUFieldConfig(n_envs=n_envs, max_blue_agents=n_agents, max_red_agents=n_agents,
                         map_set="train", map_layout=S.MAP, max_decision_steps=S.MAX_STEPS,
                         aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
                         obstacle_obs_channel=True, tag_telemetry_enabled=True,
                         own_flag_home_required_to_score=True, **S.RULESET)
    env = GPUCTFVecEnv(cfg); core = env.core
    opp = g.base_opponent
    env.env_method("set_phase", opp)
    env.env_method("set_next_opponent", "SCRIPTED", opp)
    apply_genome_to_core(core, g)
    core.blue_scripted = False       # blue is POLICY-controlled, matching real training
    return env


def main() -> int:
    from rl.config.ppo_config import PPOConfig
    from rl.custom_ppo.entity_residual import EntityResidualEncoder
    from rl.custom_ppo.trainer import CustomPPOTrainer

    print(f"{LABEL}  {_now()}")

    env = _make_env(n_envs=4, n_agents=4)
    cfg = PPOConfig()
    cfg.device = "cpu"
    cfg.n_steps = 8
    cfg.use_latent_strategy = False
    cfg.normalize_returns = False
    n_envs = int(env.num_envs)
    total_samples = cfg.n_steps * n_envs

    trainer = CustomPPOTrainer(env, cfg, learning_rate=1e-4, clip_range=0.2,
                              ent_coef=0.01, n_epochs=1, batch_size=total_samples)

    # ---- attach the entity pathway post-construction (see module docstring) --
    model = trainer.model
    model.entity_repair_enabled = True
    model.entity_encoder = EntityResidualEncoder(out_dim=model._local_actor_in_dim)
    model.entity_encoder.to(trainer.device)
    trainer.optimizers.primary.add_param_group(
        {"params": list(model.entity_encoder.parameters())})
    print(f"  attached entity_encoder ({sum(p.numel() for p in model.entity_encoder.parameters())} "
         f"params) to trainer.model and trainer.optimizers.primary")

    proj_w_before = model.entity_encoder.proj.weight.detach().clone()
    assert torch.equal(proj_w_before, torch.zeros_like(proj_w_before))

    t0 = time.time()
    print(f"  collect_rollout() ... (n_envs={n_envs}, n_steps={cfg.n_steps})", flush=True)
    rollout = trainer.collect_rollout()
    print(f"    done in {time.time() - t0:.1f}s")

    assert "obs_teammates" in rollout.fields, \
        "entity fields were not registered in the rollout buffer -- wiring did not activate"
    for name in ("obs_teammates", "obs_teammates_valid", "obs_enemies", "obs_enemies_valid"):
        v = rollout.fields[name][: rollout.pos]
        assert not torch.isnan(v.float()).any(), f"NaN in {name}"
    print(f"  rollout buffer entity fields present and NaN-free: "
         f"obs_teammates{tuple(rollout.fields['obs_teammates'].shape)}, "
         f"obs_enemies{tuple(rollout.fields['obs_enemies'].shape)}")

    t0 = time.time()
    print(f"  update() -- real clipped PPO loss over real GAE advantages ...", flush=True)
    stats = trainer.update(rollout, total_timesteps=cfg.total_timesteps)
    print(f"    done in {time.time() - t0:.1f}s")
    # Some diagnostic stats (e.g. actor_jsd_update_start) use NaN as a legitimate
    # "not applicable when this feature is inactive" sentinel -- checking every
    # key blanket would be MY bug, not a real one. Check only the stats that
    # bear directly on whether this update step was healthy.
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
        v = stats.get(k)
        if isinstance(v, (int, float)) and (v != v):
            raise AssertionError(f"NaN in core training stat {k!r}")
    print(f"  update stats (subset): "
         f"policy_loss={stats.get('policy_loss', 'n/a')} "
         f"value_loss={stats.get('value_loss', 'n/a')} "
         f"entropy={stats.get('entropy', 'n/a')}")

    grad = model.entity_encoder.proj.weight.grad
    assert grad is not None, "entity_encoder.proj received NO gradient from the real PPO update"
    assert not torch.isnan(grad).any(), "NaN gradient in entity_encoder.proj"
    grad_norm = float(grad.norm())
    assert grad_norm > 0.0, "entity_encoder.proj gradient is exactly zero"
    print(f"  entity_encoder.proj gradient norm (from REAL PPO loss): {grad_norm:.6f}")

    proj_w_after = model.entity_encoder.proj.weight.detach().clone()
    assert not torch.equal(proj_w_after, proj_w_before), \
        "optimizer.step() did not change the entity projection weight"
    assert not torch.isnan(proj_w_after).any()
    print(f"  entity_encoder.proj weight CHANGED after one real PPO update (wake-up confirmed)")

    # ---- checkpoint save / reload -------------------------------------------
    ckpt_path = SD / f"{LABEL}_checkpoint.zip"
    trainer.save(str(ckpt_path))
    assert ckpt_path.is_file(), "trainer.save() did not produce a file"
    print(f"  checkpoint saved: {ckpt_path.name} ({ckpt_path.stat().st_size} bytes)")

    obs = env.core.get_obs_tensors("blue")
    from gpu_env._core._entity_obs import build_entity_tensors
    d = build_entity_tensors(env.core, "blue")
    with torch.no_grad():
        pre_reload_logits = model.policy_logits(
            obs, teammates=d["teammates"], teammates_valid=d["teammates_valid"],
            enemies=d["enemies"], enemies_valid=d["enemies_valid"])

    # NOTE: load_custom_ppo_policy (the CLI-level convenience wrapper) cannot be
    # used here yet -- it reconstructs the model shape from the checkpoint's
    # SAVED CONFIG via build_model_kwargs, which does not know entity_repair
    # (not yet a PPOConfig field; see module docstring/REMAINING_BEFORE_LAUNCH).
    # It correctly REJECTED that mismatch with "unexpected: entity_encoder.*"
    # when first tried here -- that is the loader's fail-closed check doing its
    # job, not a bug. The lower-level compat loader, given a model ALREADY
    # constructed with entity_repair_enabled=True (exactly as
    # test_entity_repair_production_api.py's warm-start test already proved
    # works against the real sealed checkpoint), is the correct tool.
    from rl.custom_ppo.checkpoints.archive import _torch_load_checkpoint
    from rl.custom_ppo.checkpoints.state_dict import _load_model_state_dict_compat
    from rl.custom_ppo.entity_residual import EntityResidualEncoder as _E
    from rl.custom_ppo.policy import SharedActorCentralizedCritic as _M

    reloaded = _M(env.observation_space, env.action_space, entity_repair_enabled=True)
    payload = _torch_load_checkpoint(str(ckpt_path), map_location="cpu")
    sd = payload["model_state_dict"] if "model_state_dict" in payload else payload
    _load_model_state_dict_compat(reloaded, sd)
    reloaded.eval()
    with torch.no_grad():
        post_reload_logits = reloaded.policy_logits(
            obs, teammates=d["teammates"], teammates_valid=d["teammates_valid"],
            enemies=d["enemies"], enemies_valid=d["enemies_valid"])
    reload_ok = torch.allclose(pre_reload_logits, post_reload_logits, atol=1e-5)
    print(f"  checkpoint reload equivalence (fresh entity_repair_enabled=True model, "
         f"loaded via the real compat loader): {'MATCH' if reload_ok else 'MISMATCH'}")
    assert reload_ok, "reloaded model (including its TRAINED entity_encoder weights) " \
                      "must reproduce pre-save logits exactly"

    env.close()

    rec = {"record": f"{LABEL} real production PPO path smoke", "status": "SMOKE_PASSED",
          "utc": _now(), "n_envs": n_envs, "n_steps": cfg.n_steps,
          "entity_proj_grad_norm": grad_norm, "checkpoint_reload_match": reload_ok,
          "update_stats_keys": sorted(stats.keys()),
          "NOT_A_CLAIM": ["closed-loop performance", "that PPOConfig/CLI wiring is complete "
                          "(entity_encoder was attached post-construction here, not via a "
                          "frozen config field)"],
          "REMAINING_BEFORE_LAUNCH": "add entity_repair_enabled/entity_hidden_dim to PPOConfig "
                                     "+ build_model_kwargs + train_specialist_scale.py CLI flag, "
                                     "then regenerate the preset snapshot"}
    out = SD / f"{LABEL}_RESULT.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  ALL CHECKS PASSED\n  -> {out}\n  -> {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
