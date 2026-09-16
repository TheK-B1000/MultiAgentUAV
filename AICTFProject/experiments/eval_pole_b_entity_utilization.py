r"""Pole-B diagnosis, step 1: does the trained policy actually USE the entity stream?

Cheapest possible cut at the first fork in
4V4_ENTITY_REPAIR_ASYMMETRIC_RECOVERY_READING.json#DIAGNOSTIC_FORK, measured at
the POLICY level rather than by spending hundreds of episodes:

    g            = entity_encoder(teammates, enemies)        the geometry residual
    local_in_aug = _encode_local_obs(obs, entities)          what the actor actually sees
    local_in_base= local_in_aug - g                          what it would have seen pre-repair

    residual_ratio = ||g|| / ||local_in_base||               how load-bearing is geometry
    + logits under FULL / TEAMMATES_ZEROED / ENEMIES_ZEROED / BOTH_ZEROED
    + per-arm action agreement and KL vs FULL

WHY THIS IS VALID AS AN ABLATION, NOT AN APPROXIMATION
    EntityResidualEncoder is bias-free throughout, so zeroing an entity feature
    block sends that block's encoder output to EXACTLY zero (0 in -> 0 out at
    every linear/ReLU), and the masked mean pool of all-zero encodings is
    exactly zero. So:
        BOTH_ZEROED      -> g == 0 exactly  -> the literal pre-repair pathway
        TEAMMATES_ZEROED -> g == proj(concat(0, e_pooled))
        ENEMIES_ZEROED   -> g == proj(concat(t_pooled, 0))
    These are exact interventions on the residual, not noisy perturbations.
    Contract 1 below asserts the exactness live rather than trusting this note.

WHAT THIS CANNOT ANSWER
    Whether the usage is PERFORMANCE-relevant. A policy can lean hard on a
    feature and still lose. That needs the episode-level win-rate ablation and
    is deliberately a separate, more expensive run.

    python -m experiments.eval_pole_b_entity_utilization
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LABEL = "POLE_B_ENTITY_UTILIZATION"
SEED_LO, SEED_HI = 17_900_001, 17_900_128          # POLE_B_ENTITY_DIAGNOSIS, exploratory
EPISODES_PER_CELL = 4
SAMPLE_EVERY = 8

CKPT_A3 = ROOT / "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3_entity_repair/ckpts/final_pi_A_specialist_4v4_b3_entity_repair.zip"
CKPT_B3 = ROOT / "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_b3_entity_repair/ckpts/final_pi_B_specialist_4v4_b3_entity_repair.zip"
POLE_B_GENOME = SD / "pole_b2_candidates" / "B3-3_lockdef10_2v1.json"

ARMS = ("FULL", "TEAMMATES_ZEROED", "ENEMIES_ZEROED", "BOTH_ZEROED")
BASE_KEY = {"A": "OP6", "B": "OP7"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _entity_arm(d: dict, arm: str) -> dict:
    """Exact intervention on the entity tensors (see module docstring)."""
    out = dict(d)
    if arm in ("TEAMMATES_ZEROED", "BOTH_ZEROED"):
        out["teammates"] = torch.zeros_like(out["teammates"])
    if arm in ("ENEMIES_ZEROED", "BOTH_ZEROED"):
        out["enemies"] = torch.zeros_like(out["enemies"])
    return out


def _flat(d: dict):
    from gpu_env._core._entity_obs import flatten_for_policy
    return flatten_for_policy(d)


def _logits(model, obs, d):
    return model.policy_logits(obs, teammates=d["teammates"], teammates_valid=d["teammates_valid"],
                               enemies=d["enemies"], enemies_valid=d["enemies_valid"])


def _g(model, d):
    tm, tmv, en, env = _flat(d)
    return model.entity_encoder(tm, tmv, en, env)


def _per_head_kl(model, logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """Mean KL(p||q) summed over the per-agent categorical heads."""
    total = None
    for dp, dq in zip(model._categoricals(logits_p), model._categoricals(logits_q)):
        kl = torch.distributions.kl_divergence(dp, dq)
        total = kl if total is None else total + kl
    return float(total.mean().item())


def _contracts(model, obs, d) -> dict:
    """Rule 12: known-answer contracts BEFORE the main loop."""
    res = {}
    with torch.no_grad():
        # 1. BOTH_ZEROED must drive the residual to EXACTLY zero, for these
        #    trained (not initial) weights -- the bias-free guarantee.
        g0 = _g(model, _entity_arm(d, "BOTH_ZEROED"))
        res["both_zeroed_residual_is_exactly_zero"] = bool(torch.equal(g0, torch.zeros_like(g0)))

        # 2. The trained residual must NOT be identically zero on real entities,
        #    otherwise every arm is trivially identical and the probe is vacuous.
        g_full = _g(model, d)
        res["full_residual_is_nonzero"] = bool(g_full.abs().sum().item() > 0.0)

        # 3. Partial arms must equal a directly-constructed half-residual.
        tm, tmv, en, env = _flat(d)
        t_pooled = model.entity_encoder.teammate_enc(tm)
        e_pooled = model.entity_encoder.enemy_enc(en)
        from rl.custom_ppo.entity_residual import _masked_mean_pool
        t_p = _masked_mean_pool(t_pooled, tmv)
        e_p = _masked_mean_pool(e_pooled, env)
        direct_tm_zero = model.entity_encoder.proj(
            torch.cat([torch.zeros_like(t_p), e_p], dim=-1))
        arm_tm_zero = _g(model, _entity_arm(d, "TEAMMATES_ZEROED"))
        res["teammates_zeroed_matches_direct_construction"] = bool(
            torch.allclose(direct_tm_zero, arm_tm_zero, atol=1e-6))

        # 4. local_in_base recovered by subtraction must be entity-INVARIANT:
        #    the CNN/vec pathway cannot depend on entity content (NO_MUTATION).
        aug_full, _, _ = model._encode_local_obs(
            obs, teammates=d["teammates"], teammates_valid=d["teammates_valid"],
            enemies=d["enemies"], enemies_valid=d["enemies_valid"])
        base_from_full = aug_full - g_full
        dz = _entity_arm(d, "BOTH_ZEROED")
        aug_zero, _, _ = model._encode_local_obs(
            obs, teammates=dz["teammates"], teammates_valid=dz["teammates_valid"],
            enemies=dz["enemies"], enemies_valid=dz["enemies_valid"])
        res["base_pathway_is_entity_invariant"] = bool(
            torch.allclose(base_from_full, aug_zero, atol=1e-6))

        # 5. Determinism: same state twice -> identical logits.
        res["logits_deterministic"] = bool(torch.equal(_logits(model, obs, d),
                                                       _logits(model, obs, d)))
    return res


def _build_env(device, seed, pole, pole_b_genome):
    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    from rl.curriculum import phase_from_tag
    R2.AGENTS = 4
    genomes = {"OP6": pole_A_genome(4)} if pole == "A" else {"OP7": pole_b_genome}
    env = R2.build_env(device, seed)
    core = env.core
    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0
    install_keyed_opponent_overlays(core, genomes)
    key = BASE_KEY[pole]
    env.env_method("set_phase", phase_from_tag(key))
    env.env_method("set_next_opponent", "SCRIPTED", key)
    obs = env.reset()
    obs["global_state"] = env.state()
    assert_live_opponent_batch(core, genomes, allowed_keys=(key,), context=f"{LABEL} {pole} {seed}")
    resolved = core._bt_resolved_profile_tensors()
    got = resolved.get("min_alive_for_defender")
    got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
    if got_val != 4:
        raise SystemExit(f"FAIL-CLOSED: live pole {pole} min_alive_for_defender={got_val}, expected 4")
    return env, core, obs


def run_cell(policy_name: str, ckpt: Path, pole: str, pole_b_genome, device: str) -> dict:
    """Roll the policy out on `pole`, sampling states, and measure entity usage."""
    from gpu_env._core._entity_obs import augment_obs_with_entities, build_entity_tensors
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    from rl.custom_ppo import load_custom_ppo_policy
    import experiments.r2_learned_crossover as R2

    env, core, _ = _build_env(device, SEED_LO, pole, pole_b_genome)
    obs_space, act_space = env.observation_space, env.action_space
    env.close()

    inference = load_custom_ppo_policy(str(ckpt), obs_space, act_space, device=device)
    model = inference.model
    model.eval()
    if model.entity_encoder is None:
        raise SystemExit(f"FAIL-CLOSED: {policy_name} has no entity_encoder; wrong checkpoint?")

    ratios, kls, agree = [], {a: [] for a in ARMS if a != "FULL"}, {a: [] for a in ARMS if a != "FULL"}
    g_norms, base_norms = [], []
    contracts_checked = None
    n_states = 0

    seeds = list(range(SEED_LO, SEED_LO + EPISODES_PER_CELL))
    bar = tqdm_iter(seeds, desc=f"{LABEL} {policy_name}@Pole{pole}", unit="ep")
    for seed in bar:
        set_postfix(bar, f"seed={seed} states={n_states}")
        env, core, obs = _build_env(device, seed, pole, pole_b_genome)
        try:
            inference.reset_strategy()
            obs = augment_obs_with_entities(obs, core, side="blue")
            for step in range(R2.MAX_STEPS):
                if step % SAMPLE_EVERY == 0:
                    d = build_entity_tensors(core, "blue")
                    obs_t = inference._tensor_obs(inference._batched_obs(obs))
                    if contracts_checked is None:
                        contracts_checked = _contracts(model, obs_t, d)
                        failed = [k for k, v in contracts_checked.items() if not v]
                        if failed:
                            raise SystemExit(
                                f"FAIL-CLOSED: known-answer contracts failed for "
                                f"{policy_name}@Pole{pole}: {failed}. Refusing to collect "
                                f"a single measurement against an unverified probe.")
                    with torch.no_grad():
                        g_full = _g(model, d)
                        aug, _, _ = model._encode_local_obs(
                            obs_t, teammates=d["teammates"], teammates_valid=d["teammates_valid"],
                            enemies=d["enemies"], enemies_valid=d["enemies_valid"])
                        base = aug - g_full
                        gn = g_full.norm(dim=-1)
                        bn = base.norm(dim=-1).clamp(min=1e-8)
                        ratios.extend((gn / bn).tolist())
                        g_norms.extend(gn.tolist())
                        base_norms.extend(base.norm(dim=-1).tolist())

                        lg = {a: _logits(model, obs_t, _entity_arm(d, a)) for a in ARMS}
                        for a in ARMS:
                            if a == "FULL":
                                continue
                            kls[a].append(_per_head_kl(model, lg["FULL"], lg[a]))
                            am_f = torch.stack([c.probs.argmax(-1) for c in model._categoricals(lg["FULL"])])
                            am_a = torch.stack([c.probs.argmax(-1) for c in model._categoricals(lg[a])])
                            agree[a].append(float((am_f == am_a).float().mean().item()))
                    n_states += 1
                action, _ = inference.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _r, done, _info = env.step_wait()
                obs["global_state"] = env.state()
                obs = augment_obs_with_entities(obs, core, side="blue")
                if bool(np.asarray(done).any()):
                    break
        finally:
            env.close()

    return {
        "policy": policy_name, "pole": pole, "n_states": n_states,
        "n_agent_rows": len(ratios),
        "contracts": contracts_checked,
        "residual_ratio_g_over_base": {
            "mean": float(np.mean(ratios)), "median": float(np.median(ratios)),
            "p90": float(np.percentile(ratios, 90)), "max": float(np.max(ratios)),
        },
        "g_norm": {"mean": float(np.mean(g_norms)), "median": float(np.median(g_norms))},
        "base_norm": {"mean": float(np.mean(base_norms))},
        "ablation_vs_FULL": {
            a: {"mean_kl": float(np.mean(kls[a])),
                "action_agreement": float(np.mean(agree[a]))}
            for a in ARMS if a != "FULL"
        },
    }


def main() -> int:
    from experiments.opponent_spec import _with_full_team_defender_gate
    from experiments.run_lock import RunLock
    from experiments.sds_genome import SDSGenome

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pole_b_genome = _with_full_team_defender_gate(
        SDSGenome.from_dict(json.loads(POLE_B_GENOME.read_text(encoding="utf-8"))), 4)

    out_path = SD / f"{LABEL}_RESULT.json"
    if out_path.is_file():
        raise SystemExit(f"REFUSING: {out_path.name} exists; one-shot per label")

    print(f"{LABEL}  {_now()}  device={device}")
    print(f"  seeds        {SEED_LO}..{SEED_LO + EPISODES_PER_CELL - 1} "
          f"(exploratory block POLE_B_ENTITY_DIAGNOSIS {SEED_LO}-{SEED_HI})")
    print(f"  pole B       candidate genome {pole_b_genome.genome_id!r} overlay={dict(pole_b_genome.overlay or {})}")
    print(f"  arms         {ARMS}")
    print(f"  NOT a claim  performance relevance -- policy-level usage only\n", flush=True)

    cells = [("pi_B3", CKPT_B3, "B"),      # the subject: the regime that FAILED
             ("pi_A3", CKPT_A3, "A"),      # positive reference: the regime that PASSED
             ("pi_A3", CKPT_A3, "B")]      # A in B's environment: did A's usage persist?

    results = []
    lock_path = SD / f"{LABEL}.run.lock"
    with RunLock(lock_path, run_id=LABEL):
        for name, ckpt, pole in cells:
            if not ckpt.is_file():
                raise SystemExit(f"REFUSING: checkpoint missing: {ckpt}")
            r = run_cell(name, ckpt, pole, pole_b_genome, device)
            results.append(r)
            rr = r["residual_ratio_g_over_base"]
            ab = r["ablation_vs_FULL"]["BOTH_ZEROED"]
            print(f"\n  {name}@Pole{pole}: residual ||g||/||base|| mean={rr['mean']:.4f} "
                  f"median={rr['median']:.4f} p90={rr['p90']:.4f}")
            print(f"    BOTH_ZEROED vs FULL: mean_kl={ab['mean_kl']:.6f} "
                  f"action_agreement={ab['action_agreement']:.4f}")
            for a in ("TEAMMATES_ZEROED", "ENEMIES_ZEROED"):
                x = r["ablation_vs_FULL"][a]
                print(f"    {a:17s} vs FULL: mean_kl={x['mean_kl']:.6f} "
                      f"action_agreement={x['action_agreement']:.4f}")
            # incremental write (Rule 2): never hold results only in RAM
            out_path.write_text(json.dumps(
                {"record": f"{LABEL} (partial)", "status": "RUNNING", "utc": _now(),
                 "device": device, "cells": results}, indent=2), encoding="utf-8")

    out_path.write_text(json.dumps({
        "record": f"{LABEL} policy-level entity-utilization probe",
        "status": "COMPLETE_DIAGNOSTIC", "utc": _now(), "device": device,
        "arm": "DIAGNOSTIC", "confirmatory": False,
        "implements": "4V4_ENTITY_REPAIR_ASYMMETRIC_RECOVERY_READING.json#WHAT_HAPPENS_NEXT.2_diagnose_B_only",
        "seed_block": [SEED_LO, SEED_HI], "seeds_used": [SEED_LO, SEED_LO + EPISODES_PER_CELL - 1],
        "episodes_per_cell": EPISODES_PER_CELL, "sample_every_steps": SAMPLE_EVERY,
        "checkpoints": {"pi_A3": str(CKPT_A3.name), "pi_B3": str(CKPT_B3.name)},
        "cells": results,
        "NOT_A_CLAIM": [
            "that entity usage is or is not PERFORMANCE-relevant -- this measures the policy's "
            "dependence on the entity stream, not win rate",
            "any confirmatory status -- exploratory seeds, diagnostic arm",
        ],
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
