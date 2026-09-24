"""Forced-z crossover eval for Fully Shared Strategy-Conditioned checkpoints.

Asks the same question as eval_rung1_crossover_scaled / specialist crossover,
but for ONE shared ``pi_phi(a|o,z)`` under CLOSEST_DEFENDS role allocation:

    delta_A = V(z_A, A) - V(z_B, A)
    delta_B = V(z_B, B) - V(z_A, B)

PASS iff both means > 0 AND both LCB95 > 0. Bootstrap identical
(n_boot=20000, alpha=0.05, rng_seed=7). No router — z is forced.

Sealing goes through ``experiments.run_state.seal``.

Run:
  python experiments/eval_fully_shared_z_crossover_scaled.py --team-size 4 \\
      --checkpoint <ckpt.zip> \\
      --spec artifacts/strategic_demand/sppo/FULLY_SHARED_STRATEGY_CONDITIONED_4V4_V1_SPEC.json \\
      --seed-base 22510001 --n-seeds 64 \\
      --label FULLY_SHARED_Z_4V4_V1 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval_hog_psp_v3 import _mean_ci  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7
BASE_KEY = {"A": "OP6", "B": "OP7"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-size", type=int, required=True, choices=(4, 6))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--n-seeds", type=int, default=64)
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--role-k-defend",
        type=int,
        default=0,
        help="0 => CLOSEST_DEFENDS k=N/2 (4v4 default). Positive overrides k.",
    )
    args = ap.parse_args()

    N = int(args.team_size)
    label = str(args.label)
    seeds = list(range(int(args.seed_base), int(args.seed_base) + int(args.n_seeds)))

    OUT = SD / f"{label}_CROSSOVER_EVAL_RESULT.json"
    ROWS_CSV = SD / f"{label.lower()}_crossover_eval_rows.csv"
    PREAUDIT_FLAG = SD / f"{label}_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json"

    spec_path = Path(args.spec)
    if not spec_path.is_file():
        raise SystemExit(f"REFUSING: spec not found: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")

    ck = Path(args.checkpoint)
    if not ck.is_file():
        raise SystemExit(f"REFUSING: checkpoint missing: {ck}")
    if OUT.is_file() or ROWS_CSV.is_file() or PREAUDIT_FLAG.is_file():
        raise SystemExit(f"REFUSING: an output for label {label!r} already exists; one-shot")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch,
        install_keyed_opponent_overlays,
        pole_A_genome,
        pole_B_genome,
    )
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo.checkpoints.loader import load_custom_ppo_policy
    from rl.custom_ppo.rule_role_assignment import RoleHoldState, roles_from_core
    from gpu_env._core._entity_obs import augment_obs_with_entities

    R2.AGENTS = N
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    genomes_by_pole = {
        "A": {"OP6": pole_A_genome(N)},
        "B": {"OP7": pole_B_genome(N)},
    }

    print(f"FULLY-SHARED z CROSSOVER EVAL  {label}  {N}v{N}  {_now()}")
    print(f"  spec       {spec_path.name}  [{spec.get('status')}]  arm={spec.get('arm', 'n/a')}")
    print(f"  checkpoint {ck}  sha {_sha(ck)[:12]}...")
    print(f"  seeds      {seeds[0]}..{seeds[-1]} (n={len(seeds)}), SHARED across z and poles")
    print(f"  allocator  CLOSEST_DEFENDS role_fixed_for_episode "
          f"k={'N/2' if int(args.role_k_defend) == 0 else int(args.role_k_defend)}")
    print(f"  gate       delta_A > 0 & LCB95 > 0; delta_B symmetric")
    print(f"  bootstrap  n={N_BOOT}, alpha={ALPHA}, rng_seed={BOOTSTRAP_SEED}\n", flush=True)

    probe = R2.build_env(device, seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    grid_agents = int(obs_space.spaces["grid"].shape[0])
    probe.close()
    if grid_agents != N:
        raise SystemExit(f"FAIL-CLOSED: env grid agent dim {grid_agents} != team size {N}")

    policy = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
    model = policy.model

    # C1 / C2 / C5 contracts at eval load time.
    if not bool(getattr(model, "uses_latent_strategy", False)):
        raise SystemExit("REFUSING: checkpoint must be latent-conditioned (uses_latent_strategy)")
    if int(getattr(model, "latent_k", 0) or 0) != 2:
        raise SystemExit(f"REFUSING: latent_k must be 2; got {getattr(model, 'latent_k', None)}")
    if not bool(getattr(model, "role_conditioning_enabled", False)):
        raise SystemExit("REFUSING: checkpoint must have role_conditioning_enabled (CLOSEST_DEFENDS)")
    if bool(getattr(model, "assignment_conditioning_enabled", False)):
        raise SystemExit("REFUSING: assignment conditioning must be off")
    # C2: no second actor / no split frozen model attached on the inference policy.
    if getattr(policy, "split_attack_defend_frozen_model", None) is not None:
        raise SystemExit("REFUSING: C2 — split_attack_defend frozen model must not be present")

    hold_kwargs: dict = {"fixed_for_episode": True}
    if int(args.role_k_defend) > 0:
        hold_kwargs["k_defend"] = int(args.role_k_defend)

    def _attach_roles(obs, core, hold: RoleHoldState, *, force: bool):
        roles = roles_from_core(core, hold, force=force, advance_age=True)
        out = dict(obs)
        out["roles"] = roles.detach().cpu().numpy().astype(np.float32)
        return out

    def run_cell(z: int, pole: str, seed: int) -> dict:
        env = R2.build_env(device, seed)
        core = env.core
        hold = RoleHoldState(n_agents=N, **hold_kwargs)
        try:
            # C5: force z; never sample q_phi.
            policy.fixed_latent_strategy = True
            policy.fixed_latent_strategy_id = int(z)
            if hasattr(policy, "reset_strategy"):
                policy.reset_strategy()
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            genomes = genomes_by_pole[pole]
            install_keyed_opponent_overlays(core, genomes)
            key = BASE_KEY[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            if bool(getattr(model, "entity_encoder", None) is not None):
                obs = augment_obs_with_entities(obs, core)
            obs = _attach_roles(obs, core, hold, force=True)
            assert_live_opponent_batch(
                core, genomes, allowed_keys=(key,),
                context=f"{label} z{z}@Pole{pole} seed {seed}",
            )
            resolved = core._bt_resolved_profile_tensors()
            got = resolved.get("min_alive_for_defender")
            got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
            if got_val != N:
                raise SystemExit(
                    f"FAIL-CLOSED: live pole {pole} resolves min_alive_for_defender={got_val}, "
                    f"expected {N}"
                )
            terminal = None
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                if bool(getattr(model, "entity_encoder", None) is not None):
                    obs = augment_obs_with_entities(obs, core)
                obs = _attach_roles(obs, core, hold, force=False)
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    res = (i0 or {}).get("episode_result") or {}
                    terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                    break
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            blue, red = terminal
            return {"blue": blue, "red": red, "win": int(blue > red), "margin": blue - red}
        finally:
            env.close()

    if args.dry_run:
        for pole in ("A", "B"):
            env = R2.build_env(device, 99_990_000 + N)
            try:
                core = env.core
                core._bt_profile_override = None
                install_keyed_opponent_overlays(core, genomes_by_pole[pole])
                key = BASE_KEY[pole]
                env.env_method("set_phase", phase_from_tag(key))
                env.env_method("set_next_opponent", "SCRIPTED", key)
                env.reset()
                resolved = core._bt_resolved_profile_tensors()
                got = resolved.get("min_alive_for_defender")
                got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
                print(
                    f"  dry-run pole {pole}: live min_alive_for_defender={got_val} "
                    f"(expect {N}) {'OK' if got_val == N else 'MISMATCH'}"
                )
                if got_val != N:
                    raise SystemExit("FAIL-CLOSED: dry-run live pole mismatch")
            finally:
                env.close()
        print(
            "\n  --dry-run: spec frozen, checkpoint latent+role OK, poles resolve at N. "
            "NO episodes run, NOTHING written."
        )
        return 0

    from experiments.tqdm_loop import set_postfix, tqdm_iter

    cells = [(z, pole, seed) for z in (0, 1) for pole in ("A", "B") for seed in seeds]
    rows = []
    bar = tqdm_iter(cells, desc=f"{label} crossover", unit="ep")
    for z, pole, seed in bar:
        set_postfix(bar, f"z{z}@Pole{pole} seed={seed}")
        rows.append({"z": z, "pole": pole, "seed": seed, **run_cell(z, pole, seed)})
        if seed == seeds[-1]:
            wr = np.mean([r["win"] for r in rows if r["z"] == z and r["pole"] == pole])
            print(f"  z{z} on Pole {pole}: win rate {wr:.4f}", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def wins(z, pole):
        by = {r["seed"]: r["win"] for r in rows if r["z"] == z and r["pole"] == pole}
        return np.array([by[s] for s in seeds], dtype=np.float64)

    delta_a = _mean_ci(wins(0, "A") - wins(1, "A"))
    delta_b = _mean_ci(wins(1, "B") - wins(0, "B"))
    delta_a["passes"] = bool(delta_a["mean"] > 0 and delta_a["lcb95"] > 0)
    delta_b["passes"] = bool(delta_b["mean"] > 0 and delta_b["lcb95"] > 0)

    tie_or_reversal = [
        k for k, d in (("delta_A", delta_a), ("delta_B", delta_b)) if d["mean"] <= 0.0
    ]
    if tie_or_reversal:
        PREAUDIT_FLAG.write_text(
            json.dumps(
                {
                    "record": f"{label} crossover EVAL integrity audit REQUIRED",
                    "status": "FLAGGED",
                    "utc": _now(),
                    "implements": f"{spec_path.name}#EVALUATION.tie_or_reversal",
                    "triggered_by": tie_or_reversal,
                    "point_estimates": {
                        "delta_A": delta_a["mean"],
                        "delta_B": delta_b["mean"],
                    },
                    "rule": "requires a row-level integrity audit before any verdict-bearing "
                    f"result. Raw rows: {ROWS_CSV.name}.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  TIE/REVERSAL on {tie_or_reversal} -- integrity audit REQUIRED.")
        print(f"  -> {PREAUDIT_FLAG}")
        return 0

    gate_passes = bool(delta_a["passes"] and delta_b["passes"])
    print("\n  PRIMARY GATE")
    print(
        f"    delta_A {delta_a['mean']:+.4f} [{delta_a['lcb95']:+.4f}, {delta_a['ucb95']:+.4f}]"
        f" {'PASS' if delta_a['passes'] else 'FAIL'}"
    )
    print(
        f"    delta_B {delta_b['mean']:+.4f} [{delta_b['lcb95']:+.4f}, {delta_b['ucb95']:+.4f}]"
        f" {'PASS' if delta_b['passes'] else 'FAIL'}"
    )
    print(f"\n  GATE: {'PASS' if gate_passes else 'FAIL'}")

    import experiments.run_state as rs

    state = rs.RunState(SD, label)
    if state.state is None:
        state.begin(
            checkpoint=str(ck),
            seed_base=int(args.seed_base),
            n_seeds=len(seeds),
            team_size=N,
        )

    claims = [
        rs.Claim(
            name="delta_A",
            recorded={
                "mean": delta_a["mean"],
                "lcb95": delta_a["lcb95"],
                "ucb95": delta_a["ucb95"],
            },
            minuend={"z": 0, "pole": "A"},
            subtrahend={"z": 1, "pole": "A"},
            value_field="win",
        ),
        rs.Claim(
            name="delta_B",
            recorded={
                "mean": delta_b["mean"],
                "lcb95": delta_b["lcb95"],
                "ucb95": delta_b["ucb95"],
            },
            minuend={"z": 1, "pole": "B"},
            subtrahend={"z": 0, "pole": "B"},
            value_field="win",
        ),
    ]
    plan = rs.AuditPlan(
        rows_csv=ROWS_CSV,
        expected_rows=len(rows),
        expected_seeds=seeds,
        group_by=("z", "pole"),
        seed_field="seed",
        int_fields=("z", "seed", "blue", "red", "margin"),
        binary_fields=("win",),
        derived={},
        checkpoints={"pi_phi": (ck, _sha(ck))},
        spec_path=spec_path,
        claims=claims,
        n_boot=N_BOOT,
        alpha=ALPHA,
        rng_seed=BOOTSTRAP_SEED,
        seed_class="exploratory" if int(args.n_seeds) < 128 else "sealed_confirmatory",
        experiment_id=label,
    )

    # status is owned by seal(); do not set it here.
    payload = {
        "record": f"{label} crossover EVAL",
        "one_shot": True,
        "utc": _now(),
        "arm": spec.get("arm", "EXPLORATORY"),
        "confirmatory": bool(spec.get("confirmatory", False)),
        "implements": f"{spec_path.name}#EVALUATION_after_training_not_during",
        "team_size": N,
        "device": device,
        "architecture": "fully_shared_pi_phi_a_given_o_z_under_CLOSEST_DEFENDS",
        "role_fixed_for_episode": True,
        "forced_z": True,
        "router_absent": True,
        "split_attack_defend_absent": True,
        "checkpoint_sha256": _sha(ck),
        "seeds": {
            "block": [seeds[0], seeds[-1]],
            "n": len(seeds),
            "shared_across_z_and_poles": True,
        },
        "poles": {
            p: {
                "base": BASE_KEY[p],
                "overlay": dict(
                    (pole_A_genome(N) if p == "A" else pole_B_genome(N)).overlay or {}
                ),
            }
            for p in ("A", "B")
        },
        "PRIMARY_GATE": {
            "delta_A": delta_a,
            "delta_B": delta_b,
            "passes": gate_passes,
        },
        "bootstrap": {
            "procedure": "paired percentile bootstrap over evaluation seeds",
            "samples": N_BOOT,
            "alpha": ALPHA,
            "rng_seed": BOOTSTRAP_SEED,
        },
        "no_model_selection_occurred": True,
        "total_episodes": len(rows),
    }

    rs.seal(out_path=OUT, payload=payload, plan=plan, state=state, strict=False)
    print(f"\n  -> {OUT} (sealed)")
    return 0 if gate_passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
