r"""Step 7: one real vectorized rollout smoke of the entity pipeline.

NOT training. Proves the full mechanical chain works at real batch shapes,
with the entity-augmented actor genuinely driving blue's actions (not scripted
zero-actions):

    env state (n_envs>1, vectorized)
      -> entity extraction (gpu_env._core._entity_obs)
      -> flatten to policy batch shape
      -> augmented_local_in -> policy.latent_actor -> logits
      -> real mask + real Categorical sampling (policy._mask_logits / _categoricals)
      -> env.step with the SAMPLED actions
      -> a tiny REINFORCE-style loss (genuinely differentiable, genuinely
         dependent on real env data -- not real PPO/GAE, which is out of
         scope for a mechanical smoke)
      -> optimizer.step()
      -> checkpoint save
      -> checkpoint reload
      -> reload equivalence check (known-answer contract on the round trip
         itself: reloaded logits on a fixed held-out observation must be
         BIT-IDENTICAL to pre-save logits)

tqdm + incremental metrics.csv, matching Rule 1/2. Artifacts are audited at
the end of this script, not left for a human to remember to check.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LABEL = "ENTITY_PIPELINE_ROLLOUT_SMOKE"
N_ENVS = 4
N_AGENTS = 4
N_STEPS = 40


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_batched_env(n_envs, n_agents, device="cpu", seed=17600001):
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
    env.env_method("set_next_opponent", "SCRIPTED", opp)   # red stays scripted
    apply_genome_to_core(core, g)
    core.blue_scripted = False                             # blue is POLICY-controlled
    env.reset(); apply_genome_to_core(core, g); core.drain_tag_events()
    return env, core


def _augmented_logits(policy, entity_module, obs, tm, tm_v, en, en_v):
    from rl.custom_ppo.entity_residual import augmented_local_in
    aug_local_in, _, _, _ = augmented_local_in(policy, obs, tm, tm_v, en, en_v, entity_module)
    batch = int(obs["grid"].shape[0])
    flat = policy.latent_actor(aug_local_in)
    return flat.reshape(batch, policy.n_agents * policy.per_agent_logits)


def main() -> int:
    from gpu_env._core._entity_obs import build_entity_tensors, flatten_for_policy
    from rl.custom_ppo.entity_residual import EntityResidualEncoder
    from rl.custom_ppo.policy import SharedActorCentralizedCritic
    from experiments.tqdm_loop import set_postfix, tqdm_iter

    print(f"{LABEL}  {_now()}")
    print(f"  n_envs={N_ENVS} n_agents={N_AGENTS} n_steps={N_STEPS}  device=cpu\n", flush=True)

    env, core = _make_batched_env(N_ENVS, N_AGENTS)
    obs_space, act_space = env.observation_space, env.action_space

    torch.manual_seed(0)
    policy = SharedActorCentralizedCritic(obs_space, act_space,
                                          strategy_encoder_enabled=False, latent_k=0)
    entity_module = EntityResidualEncoder(out_dim=policy._local_actor_in_dim)
    with torch.no_grad():                     # move off pure zero-init so the smoke
        for p in entity_module.parameters():  # exercises a genuinely nonzero pathway
            p.add_(torch.randn_like(p) * 0.05)
    opt = torch.optim.Adam(list(policy.parameters()) + list(entity_module.parameters()), lr=1e-4)

    ROWS = SD / f"{LABEL.lower()}_rows.csv"
    with ROWS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "mean_reward", "loss", "grad_norm_entity"])
        w.writeheader()

    log_probs_buf, rewards_buf = [], []
    t0 = time.time()
    bar = tqdm_iter(range(N_STEPS), desc=LABEL, unit="step")
    for step in bar:
        obs = core.get_obs_tensors("blue")
        obs["global_state"] = env.state() if hasattr(env, "state") else None
        d = build_entity_tensors(core, "blue")
        tm, tm_v, en, en_v = flatten_for_policy(d)

        logits = _augmented_logits(policy, entity_module, obs, tm, tm_v, en, en_v)
        masked = policy._mask_logits(logits, obs.get("mask"))
        dists = list(policy._categoricals(masked))
        actions = torch.stack([dist.sample() for dist in dists], dim=1)
        logp = sum(dist.log_prob(a) for dist, a in zip(dists, actions.unbind(dim=1)))

        env.step_async(actions.reshape(-1).numpy())
        _o, r, done, _info = env.step_wait()
        reward = torch.as_tensor(np.asarray(r), dtype=torch.float32)
        log_probs_buf.append(logp); rewards_buf.append(reward)

        set_postfix(bar, f"mean_r={float(reward.mean()):.3f}")
        with ROWS.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=["step", "mean_reward", "loss",
                                          "grad_norm_entity"]).writerow(
                {"step": step, "mean_reward": round(float(reward.mean()), 4),
                 "loss": "", "grad_norm_entity": ""})
            fh.flush(); os.fsync(fh.fileno())

        if bool(np.asarray(done).any()):
            env.reset()

    # ---- one REINFORCE-style optimizer step over the accumulated buffer ----
    all_logp = torch.cat(log_probs_buf)
    all_r = torch.cat(rewards_buf)
    loss = -(all_logp * (all_r - all_r.mean())).mean()
    opt.zero_grad(); loss.backward()
    grad_norm = float(entity_module.proj.weight.grad.norm()) if entity_module.proj.weight.grad is not None else 0.0
    assert not any(torch.isnan(p.grad).any() for p in entity_module.parameters() if p.grad is not None), \
        "NaN gradient in entity module"
    opt.step()
    print(f"\n  optimizer step: loss={float(loss):.4f}  entity_proj_grad_norm={grad_norm:.6f}", flush=True)
    with ROWS.open("a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=["step", "mean_reward", "loss",
                                      "grad_norm_entity"]).writerow(
            {"step": "OPT_STEP", "mean_reward": "", "loss": round(float(loss), 6),
             "grad_norm_entity": round(grad_norm, 6)})

    # ---- checkpoint save / reload equivalence -------------------------------
    CKPT = SD / f"{LABEL}_checkpoint.pt"
    torch.save({"policy": policy.state_dict(), "entity_module": entity_module.state_dict()},
              CKPT)

    held_out = core.get_obs_tensors("blue")
    held_out["global_state"] = env.state() if hasattr(env, "state") else None
    d2 = build_entity_tensors(core, "blue")
    tm2, tm2_v, en2, en2_v = flatten_for_policy(d2)
    with torch.no_grad():
        pre_reload_logits = _augmented_logits(policy, entity_module, held_out, tm2, tm2_v, en2, en2_v)

    policy2 = SharedActorCentralizedCritic(obs_space, act_space,
                                           strategy_encoder_enabled=False, latent_k=0)
    entity_module2 = EntityResidualEncoder(out_dim=policy2._local_actor_in_dim)
    payload = torch.load(CKPT, map_location="cpu")
    policy2.load_state_dict(payload["policy"])
    entity_module2.load_state_dict(payload["entity_module"])
    with torch.no_grad():
        post_reload_logits = _augmented_logits(policy2, entity_module2, held_out, tm2, tm2_v, en2, en2_v)

    reload_ok = torch.equal(pre_reload_logits, post_reload_logits)
    print(f"  checkpoint reload equivalence: {'EXACT MATCH' if reload_ok else 'MISMATCH'}")
    env.close()

    # ---- audit the artifacts, not just trust they were written -------------
    print("\n  AUDIT:")
    n_rows = sum(1 for _ in ROWS.open(encoding="utf-8")) - 1
    print(f"    rows file: {ROWS.name}  ({n_rows} data rows, expected {N_STEPS + 1})")
    assert n_rows == N_STEPS + 1, f"expected {N_STEPS + 1} rows, found {n_rows}"
    assert CKPT.is_file(), "checkpoint file missing"
    print(f"    checkpoint: {CKPT.name}  ({CKPT.stat().st_size} bytes)")
    assert reload_ok, "REGRESSION: checkpoint reload did not reproduce logits exactly"
    print(f"    reload equivalence: PASS")

    rec = {"record": f"{LABEL} mechanical pipeline smoke", "status": "SMOKE_PASSED",
          "utc": _now(), "n_envs": N_ENVS, "n_agents": N_AGENTS, "n_steps": N_STEPS,
          "final_loss": round(float(loss), 6), "entity_proj_grad_norm": round(grad_norm, 6),
          "checkpoint_reload_exact_match": reload_ok, "rows_file": ROWS.name,
          "n_rows_audited": n_rows, "elapsed_s": round(time.time() - t0, 1),
          "NOT_A_CLAIM": ["real PPO correctness", "GAE/advantage estimation",
                          "trained policy quality", "closed-loop performance"]}
    out = SD / f"{LABEL}_RESULT.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  -> {out}\n  -> {ROWS}\n  -> {CKPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
