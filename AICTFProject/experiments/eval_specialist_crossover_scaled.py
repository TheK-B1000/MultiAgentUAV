"""Team-size-parameterized specialist crossover evaluation.

Generalizes eval_specialist_baseline.py to any team size. Asks whether two LEARNED
specialists cross over against the two poles:

    delta_A_spec = V(pi_A, A) - V(pi_B, A)
    delta_B_spec = V(pi_B, B) - V(pi_A, B)

PASS iff both means > 0 AND both LCB95 > 0 -- the identical criterion the 2v2 specialist
baseline used. Same paired percentile bootstrap (n_boot=20000, alpha=0.05, rng_seed=7).

BOTH scaling gaps in the 2v2 evaluator are fixed here, not inherited:
  * it called pole_A_genome() with no argument (defaults to the 2v2 defender gate)
  * it installed NO overlay for Pole B (correct only at 2v2, where canonical OP7 is natively
    min_alive=2)
This evaluator resolves both poles at the live team size and asserts the live opponent.

Tie/reversal convention matches eval_specialist_baseline.py exactly: any delta with mean <= 0
writes a PREAUDIT flag and STOPS before any verdict-bearing record is written.

Run:  python experiments/eval_specialist_crossover_scaled.py --team-size 4 \
          --spec artifacts/strategic_demand/sppo/EXPLORATORY_4V4_LEARNED_SPECIALIST_SPEC.json \
          --pi-a-path <ckpt> --pi-b-path <ckpt> --seed-base 13200001 --n-seeds 64 \
          --label EXPLORATORY_4V4 --device cuda
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
POLICIES = ("pi_A", "pi_B")
BASE_KEY = {"A": "OP6", "B": "OP7"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-size", type=int, required=True, choices=(2, 4, 6))
    ap.add_argument("--spec", required=True, help="frozen spec governing this evaluation")
    ap.add_argument("--pi-a-path", required=True)
    ap.add_argument("--pi-b-path", required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--n-seeds", type=int, default=64)
    ap.add_argument("--label", required=True,
                    help="output prefix, e.g. EXPLORATORY_4V4; never omitted so an exploratory "
                         "result cannot be written under a confirmatory name")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify spec, checkpoints, seeds and the live pole resolution; run "
                         "no episodes and write nothing")
    ap.add_argument("--pole-b-genome-json", default=None,
                    help="path to a JSON SDSGenome candidate for Pole B (e.g. a certified "
                         "B2 from a confirmatory-redesign track), used instead of the "
                         "canonical pole_B_genome(N). Without this, an evaluation of "
                         "specialists trained against a redesigned Pole B would silently "
                         "score them against the OLD, different Pole B.")
    ap.add_argument("--role-fixed-for-episode", action="store_true",
                    help="assign roles exactly once per episode (RoleHoldState."
                         "fixed_for_episode) instead of the periodic H_r=8 hold, for a "
                         "role-conditioned policy under evaluation. Default off preserves "
                         "existing RULE_BASED_ROLE_CONDITIONING evaluation behavior "
                         "unchanged. See DEFEND_TEACHER_ROLE_CONDITIONING_A_V1_SPEC.json "
                         "EXECUTION_BOUNDARY_locked.")
    ap.add_argument("--frozen-attack-path", default="",
                    help="path to a frozen (non-role-conditioned) checkpoint used for "
                         "ATTACK-role agent slots when evaluating a split-policy pi_A "
                         "(DEFEND_ATTACK_SPLIT_POLICY_A_V1_SPEC.json). --pi-a-path is then "
                         "the DEFEND-trained pi_D; the two models are spliced by role every "
                         "tick, exactly as at training time (rl.custom_ppo.split_attack_defend"
                         ".splice_actions, deterministic=True, no teacher, no N'). Requires "
                         "--role-fixed-for-episode and a role-conditioned --pi-a-path. "
                         "Default empty = ordinary single-policy evaluation, unchanged.")
    ap.add_argument("--frozen-attack-path-sha256", default="",
                    help="expected sha256 of --frozen-attack-path (fail-closed on mismatch)")
    ap.add_argument("--role-k-defend", type=int, default=0,
                    help="explicit CLOSEST_DEFENDS defender count. 0 = default role_k(N)=N/2. "
                         "6v6 locked closure uses 1 (5A/1D). Passed to RoleHoldState.k_defend.")
    args = ap.parse_args()

    N = int(args.team_size)
    label = str(args.label)
    seeds = list(range(int(args.seed_base), int(args.seed_base) + int(args.n_seeds)))

    OUT = SD / f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
    ROWS_CSV = SD / f"{label.lower()}_specialist_crossover_eval_rows.csv"
    PREAUDIT_FLAG = SD / f"{label}_SPECIALIST_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json"

    # ---- preflight: fail closed on everything that must already be true --------------
    spec_path = Path(args.spec)
    if not spec_path.is_file():
        raise SystemExit(f"REFUSING: spec not found: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")

    paths = {}
    for name, p in (("pi_A", args.pi_a_path), ("pi_B", args.pi_b_path)):
        ck = Path(p)
        if not ck.is_file():
            raise SystemExit(f"REFUSING: {name} checkpoint missing: {ck}")
        paths[name] = ck
    if OUT.is_file() or ROWS_CSV.is_file() or PREAUDIT_FLAG.is_file():
        raise SystemExit(f"REFUSING: an output for label {label!r} already exists; one-shot")

    frozen_attack_path_str = str(args.frozen_attack_path or "")
    frozen_attack_ckpt_path: Path | None = None
    if frozen_attack_path_str:
        if not args.role_fixed_for_episode:
            raise SystemExit("REFUSING: --frozen-attack-path requires --role-fixed-for-episode")
        frozen_attack_ckpt_path = Path(frozen_attack_path_str)
        if not frozen_attack_ckpt_path.is_file():
            raise SystemExit(f"REFUSING: frozen attack checkpoint missing: {frozen_attack_ckpt_path}")
        expected_fap = str(args.frozen_attack_path_sha256 or "").lower()
        actual_fap = _sha(frozen_attack_ckpt_path)
        if expected_fap and actual_fap != expected_fap:
            raise SystemExit(
                f"REFUSING: frozen attack checkpoint hash mismatch for {frozen_attack_ckpt_path}: "
                f"{actual_fap} != {expected_fap}"
            )

    import torch
    from experiments.opponent_spec import (
        _with_full_team_defender_gate, assert_live_opponent_batch,
        install_keyed_opponent_overlays, pole_A_genome, pole_B_genome,
    )
    from experiments.sds_genome import SDSGenome
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy
    from gpu_env._core._entity_obs import augment_obs_with_entities

    # Team size reaches the env builder by NAME, not by sweep (this module's global is what
    # R2.build_env feeds into GPUFieldConfig).
    R2.AGENTS = N

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    # Rule 14: the EVALUATOR is an experiment too. It must prove its poles are the
    # certified poles rather than trusting the operator to remember a flag -- the
    # same hole that let pi_B3 TRAIN against canonical OP7
    # (PI_B3_TRAIN_EVAL_POLE_MISMATCH_INVALIDATION.json). Both poles are attested,
    # before any episode.
    from experiments.pole_attestation import (
        assert_resolved_matches_certification, format_attestation_banner,
        governing_certification, resolve_pole_genome,
    )
    _cert_verdict, _cert_path = governing_certification(N)
    pole_attestations = {}
    for _pol in ("A", "B"):
        _g = resolve_pole_genome(_pol, N, args.pole_b_genome_json if _pol == "B" else None)
        pole_attestations[_pol] = assert_resolved_matches_certification(
            _pol, N, _cert_path, _g, is_smoke=False)
        print(f"  POLE {_pol} ATTESTATION vs {_cert_path.name}:")
        print(format_attestation_banner(pole_attestations[_pol]))
    pole_b_resolved = resolve_pole_genome("B", N, args.pole_b_genome_json)

    # Both poles resolved at the LIVE team size. At N=2 pole_B_genome(2) carries no overlay,
    # reproducing the 2v2 evaluator exactly; at N>2 the Pole-B overlay is required.
    genomes_by_pole = {
        "A": {"OP6": pole_A_genome(N)},
        "B": {"OP7": pole_b_resolved} if N != 2 else {},
    }

    print(f"SPECIALIST CROSSOVER EVAL  {label}  {N}v{N}  {_now()}")
    print(f"  spec       {spec_path.name}  [{spec.get('status')}]  arm={spec.get('arm', 'n/a')}")
    print(f"  pi_A       {paths['pi_A']}  sha {_sha(paths['pi_A'])[:12]}...")
    print(f"  pi_B       {paths['pi_B']}  sha {_sha(paths['pi_B'])[:12]}...")
    if frozen_attack_ckpt_path is not None:
        print(f"  pi_A is a SPLIT POLICY: DEFEND slots -> pi_A path above (pi_D); "
              f"ATTACK slots -> frozen_attack_path={frozen_attack_ckpt_path}  "
              f"sha {_sha(frozen_attack_ckpt_path)[:12]}...")
    print(f"  seeds      {seeds[0]}..{seeds[-1]} (n={len(seeds)}), SHARED across policies and poles")
    print(f"  poles      A: OP6+{dict(pole_A_genome(N).overlay or {})}   "
          f"B: OP7+{dict(pole_b_resolved.overlay or {})}")
    print(f"  gate       delta_A_spec > 0 & LCB95 > 0; delta_B_spec symmetric")
    print(f"  bootstrap  n={N_BOOT}, alpha={ALPHA}, rng_seed={BOOTSTRAP_SEED}\n", flush=True)

    probe = R2.build_env(device, seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    grid_agents = int(obs_space.spaces["grid"].shape[0])
    probe.close()
    if grid_agents != N:
        raise SystemExit(f"FAIL-CLOSED: env grid agent dim {grid_agents} != team size {N}")

    policies = {n: load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
                for n, p in paths.items()}
    for n, pol in policies.items():
        if getattr(pol.model, "uses_latent_strategy", False):
            raise SystemExit(f"REFUSING: {n} is latent-conditioned; specialists must be single-strategy")

    frozen_attack_policy = None
    if frozen_attack_ckpt_path is not None:
        if not bool(getattr(policies["pi_A"].model, "role_conditioning_enabled", False)):
            raise SystemExit(
                "REFUSING: --frozen-attack-path requires --pi-a-path to be role-conditioned "
                "(it is pi_D, the DEFEND-trained half of the split policy)"
            )
        frozen_attack_policy = load_custom_ppo_policy(
            str(frozen_attack_ckpt_path), obs_space, act_space, device=device
        )
        if bool(getattr(frozen_attack_policy.model, "uses_latent_strategy", False)):
            raise SystemExit("REFUSING: --frozen-attack-path model must be a non-latent specialist")
        if bool(getattr(frozen_attack_policy.model, "role_conditioning_enabled", False)):
            raise SystemExit(
                "REFUSING: --frozen-attack-path model must NOT be role-conditioned -- it plays "
                "its own native behavior for whichever slots CLOSEST_DEFENDS assigns to ATTACK"
            )
        if tuple(frozen_attack_policy.model.action_dims) != tuple(policies["pi_A"].model.action_dims):
            raise SystemExit("REFUSING: --frozen-attack-path action space differs from pi_A")

    # Rule-role policies require obs['roles'] at predict time (fail-closed in
    # CustomPPOInferencePolicy). Inject the same geometric RoleHoldState path
    # used in training (RULE_BASED_ROLE_CONDITIONING_SPEC: H_r=8). Non-role
    # policies are untouched. Assignment-v1 policies similarly need
    # obs['assignment'] via AssignmentHoldState (H_a=8).
    from rl.custom_ppo.rule_role_assignment import RoleHoldState, roles_from_core
    from rl.custom_ppo.guard_assignment import AssignmentHoldState, assignment_from_core

    def _maybe_attach_roles(obs, core, hold: RoleHoldState | None, *, force: bool):
        if hold is None:
            return obs
        roles = roles_from_core(core, hold, force=force, advance_age=True)
        out = dict(obs)
        out["roles"] = roles.detach().cpu().numpy().astype(np.float32)
        return out

    def _maybe_attach_assignment(obs, core, hold: AssignmentHoldState | None, *, force: bool):
        if hold is None:
            return obs
        feat = assignment_from_core(core, hold, force=force, advance_age=True)
        out = dict(obs)
        out["assignment"] = feat.detach().cpu().numpy().astype(np.float32)
        return out

    def _composite_predict(trained_policy, attack_policy, obs) -> np.ndarray:
        """DEFEND_ATTACK_SPLIT_POLICY_A_V1_SPEC: splice trained pi_D's own
        DEFEND-slot actions with the frozen pi_A's ATTACK-slot actions -- the
        SAME pure splice function used at training time
        (rl.custom_ppo.split_attack_defend.splice_actions), applied here with
        deterministic=True and no teacher/N' of any kind."""
        from rl.custom_ppo.split_attack_defend import splice_actions

        trained_action, _ = trained_policy.predict(obs, deterministic=True)
        attack_action, _ = attack_policy.predict(obs, deterministic=True)
        n_agents = int(trained_policy.model.n_agents)
        heads_per_agent = int(trained_policy.model.heads_per_agent)
        trained_t = torch.as_tensor(np.asarray(trained_action), dtype=torch.long).reshape(1, -1)
        attack_t = torch.as_tensor(np.asarray(attack_action), dtype=torch.long).reshape(1, -1)
        roles_t = torch.as_tensor(np.asarray(obs["roles"]), dtype=torch.float32).reshape(1, n_agents)
        is_defend = roles_t < 0.5
        exec_t = splice_actions(trained_t, attack_t, is_defend, heads_per_agent)
        return exec_t.reshape(-1).numpy().astype(np.int64)

    def run_cell(policy, pole: str, seed: int, *, attack_policy=None) -> dict:
        env = R2.build_env(device, seed)
        core = env.core
        role_hold = None
        assignment_hold = None
        if bool(getattr(policy.model, "role_conditioning_enabled", False)):
            hold_ticks = int(getattr(policy.model, "role_hold_ticks", 0) or 0)
            if hold_ticks < 1:
                # Checkpoints may only store the enable bit; SPEC locks H_r=8.
                hold_ticks = 8
            k_defend = int(getattr(args, "role_k_defend", 0) or 0)
            role_hold = RoleHoldState(
                int(env.num_envs), int(policy.model.n_agents),
                hold_ticks=hold_ticks, device=device,
                fixed_for_episode=bool(args.role_fixed_for_episode),
                k_defend=(k_defend if k_defend > 0 else None),
            )
        if bool(getattr(policy.model, "assignment_conditioning_enabled", False)):
            hold_ticks = int(getattr(policy.model, "assignment_hold_ticks", 0) or 0)
            if hold_ticks < 1:
                hold_ticks = 8
            assignment_hold = AssignmentHoldState(
                int(env.num_envs), int(policy.model.n_agents),
                hold_ticks=hold_ticks, device=device,
            )
        try:
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
            # Additive, unconditional: harmless no-op for a non-entity-repair
            # policy (predict() only reads these keys when the loaded model's
            # entity_encoder is not None); required for pi_A3/pi_B3.
            obs = augment_obs_with_entities(obs, core, side="blue")
            obs = _maybe_attach_roles(obs, core, role_hold, force=True)
            obs = _maybe_attach_assignment(obs, core, assignment_hold, force=True)
            assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                                       context=f"{label} {pole} seed {seed}")
            # Fail closed on the size-normalized gate, read from the live BT tensors.
            resolved = core._bt_resolved_profile_tensors()
            got = resolved.get("min_alive_for_defender")
            got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
            if got_val != N:
                raise SystemExit(f"FAIL-CLOSED: live pole {pole} resolves "
                                 f"min_alive_for_defender={got_val}, expected {N}")
            terminal = None
            for _ in range(R2.MAX_STEPS):
                if attack_policy is not None:
                    action = _composite_predict(policy, attack_policy, obs)
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                obs = augment_obs_with_entities(obs, core, side="blue")
                obs = _maybe_attach_roles(obs, core, role_hold, force=False)
                obs = _maybe_attach_assignment(obs, core, assignment_hold, force=False)
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
        # Prove the live pole resolves at N for both poles without spending a seed.
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
                print(f"  dry-run pole {pole}: live min_alive_for_defender={got_val} "
                      f"(expect {N}) {'OK' if got_val == N else 'MISMATCH'}")
                if got_val != N:
                    raise SystemExit("FAIL-CLOSED: dry-run live pole mismatch")
            finally:
                env.close()
        print("\n  --dry-run: spec frozen, checkpoints present, env at N, both poles resolve "
              "at N. NO episodes run, NOTHING written.")
        return 0

    from experiments.tqdm_loop import set_postfix, tqdm_iter

    cells = [(name, pole, seed) for name in POLICIES for pole in ("A", "B") for seed in seeds]
    rows = []
    bar = tqdm_iter(cells, desc=f"{label} crossover", unit="ep")
    for name, pole, seed in bar:
        set_postfix(bar, f"{name}@Pole{pole} seed={seed}")
        cell_attack_policy = frozen_attack_policy if name == "pi_A" else None
        rows.append({"policy": name, "pole": pole, "seed": seed,
                     **run_cell(policies[name], pole, seed, attack_policy=cell_attack_policy)})
        if seed == seeds[-1]:
            wr = np.mean([r["win"] for r in rows if r["policy"] == name and r["pole"] == pole])
            print(f"  {name:5s} on Pole {pole}: win rate {wr:.4f}", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def wins(name, pole):
        by = {r["seed"]: r["win"] for r in rows if r["policy"] == name and r["pole"] == pole}
        return np.array([by[s] for s in seeds], dtype=np.float64)

    delta_a = _mean_ci(wins("pi_A", "A") - wins("pi_B", "A"))
    delta_b = _mean_ci(wins("pi_B", "B") - wins("pi_A", "B"))
    delta_a["passes"] = bool(delta_a["mean"] > 0 and delta_a["lcb95"] > 0)
    delta_b["passes"] = bool(delta_b["mean"] > 0 and delta_b["lcb95"] > 0)

    tie_or_reversal = [k for k, d in (("delta_A", delta_a), ("delta_B", delta_b))
                       if d["mean"] <= 0.0]
    if tie_or_reversal:
        PREAUDIT_FLAG.write_text(json.dumps({
            "record": f"{label} specialist crossover EVAL integrity audit REQUIRED",
            "status": "FLAGGED", "utc": _now(),
            "implements": f"{spec_path.name}#EVALUATION.tie_or_reversal",
            "triggered_by": tie_or_reversal,
            "point_estimates": {"delta_A": delta_a["mean"], "delta_B": delta_b["mean"]},
            "rule": "requires a row-level integrity audit before any verdict-bearing result. "
                    f"Raw rows: {ROWS_CSV.name}.",
        }, indent=2), encoding="utf-8")
        print(f"\n  TIE/REVERSAL on {tie_or_reversal} -- integrity audit REQUIRED.")
        print(f"  -> {PREAUDIT_FLAG}")
        return 0

    gate_passes = bool(delta_a["passes"] and delta_b["passes"])
    print("\n  PRIMARY GATE")
    print(f"    delta_A_spec {delta_a['mean']:+.4f} [{delta_a['lcb95']:+.4f}, {delta_a['ucb95']:+.4f}]"
          f" {'PASS' if delta_a['passes'] else 'FAIL'}")
    print(f"    delta_B_spec {delta_b['mean']:+.4f} [{delta_b['lcb95']:+.4f}, {delta_b['ucb95']:+.4f}]"
          f" {'PASS' if delta_b['passes'] else 'FAIL'}")
    print(f"\n  GATE: {'PASS' if gate_passes else 'FAIL'}")

    OUT.write_text(json.dumps({
        "record": f"{label} specialist crossover EVAL", "status": "FROZEN_RESULT",
        "one_shot": True, "utc": _now(),
        "arm": spec.get("arm", "n/a"), "confirmatory": bool(spec.get("confirmatory", False)),
        "implements": f"{spec_path.name}#EVALUATION",
        "team_size": N, "device": device,
        "role_fixed_for_episode": bool(args.role_fixed_for_episode),
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds), "shared_across_policies": True},
        "poles": {p: {"base": BASE_KEY[p],
                      "overlay": dict((pole_A_genome(N) if p == "A" else pole_b_resolved).overlay or {}),
                      "candidate_genome_id": (pole_b_resolved.genome_id
                                              if p == "B" and args.pole_b_genome_json else None)}
                  for p in ("A", "B")},
        "pole_attestations": {p: {k: pole_attestations[p][k] for k in (
            "certification_record", "certified_genome_id", "live_genome_id",
            "certified_overlay", "live_overlay", "certified_config_hash",
            "live_config_hash", "hashes_match")} for p in ("A", "B")},
        "PRIMARY_GATE": {"delta_A": delta_a, "delta_B": delta_b, "passes": gate_passes},
        "checkpoints": {n: _sha(paths[n]) for n in POLICIES},
        "split_policy_pi_A": (
            {"pi_D_path": str(paths["pi_A"]), "frozen_attack_path": str(frozen_attack_ckpt_path),
             "frozen_attack_sha256": _sha(frozen_attack_ckpt_path)}
            if frozen_attack_ckpt_path is not None else None
        ),
        "bootstrap": {"procedure": "paired percentile bootstrap over evaluation seeds",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
        "no_model_selection_occurred": True, "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0 if gate_passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
