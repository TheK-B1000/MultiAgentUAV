"""Forced-z crossover eval for the suite's distilled sharing arms (and the Generalist), any team size.

Loads suite ``.pt`` students (not PPO ``.zip``), forces z, augments entity
tensors, and records per-seed wins. Sharing arms:

    delta_A = V(z0, A) - V(z1, A)
    delta_B = V(z1, B) - V(z0, B)

(gate carried as provenance: both means > 0 AND both LCB95 > 0; n_boot=20000, alpha=0.05, rng=7).
The Generalist has no z (GENERALIST_DEFINITION_V1): it plays each pole once per seed and its
sealed claims are V(pi_G, A) and V(pi_G, B); Delta_G against the Separated reference is formed
later from the two sealed row sets on the same seeds.

Implements the scale's STANDARDIZED_<N>V<N>_SHARING_EVAL_SPEC.json. Team size is an argument;
there is no module-level team size (CROSS_SCALE_CANONICAL_RECIPE_V1.json#STAGE_IMPLEMENTATIONS_required).
Seeds come from the spec's shared registry block (every arm of the scale on the same seeds); the
block is SPENT only once every label declared on it has sealed. Every finished episode is appended
to a fingerprinted PARTIAL file, so an interrupted run continues with --resume.

Run:
  python experiments/eval_suite_sharing_crossover.py --team-size 2 --arm share_encoder --dry-run
  python experiments/eval_suite_sharing_crossover.py --team-size 2 --arm share_encoder --device cuda --resume
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_state as rs  # noqa: E402
from experiments import seed_registry as sr  # noqa: E402
from experiments.eval_hog_psp_v3 import _mean_ci  # noqa: E402
from experiments.eval_specialist_crossover_scaled import (  # noqa: E402
    post_hoc_block, shared_block_all_sealed, shared_block_owner,
)

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7
BASE_KEY = {"A": "OP6", "B": "OP7"}
SUPPORTED_TEAM_SIZES = (2, 4, 6)
ARM_KEY = {
    "fully_shared": "fully_shared_z",
    "generalist": "generalist",
    "role_only": "role_only",
    "share_encoder": "share_encoder",
    "share_backbone": "share_backbone",
    "share_macro": "share_macro",
}


def _spec_path(n: int, tag: str = "") -> Path:
    return SD / f"STANDARDIZED_{n}V{n}{'_' + tag if tag else ''}_SHARING_EVAL_SPEC.json"


def _family(n: int, tag: str = "") -> str:
    if tag == "STAGE4":
        return f"{n}v{n}_stage4"
    if tag == "STAGE4_OWN50":
        return f"{n}v{n}_stage4_own50"
    return f"{n}v{n}" + (f"_{tag.lower()}" if tag else "")


def _is_stage4(tag: str) -> bool:
    """STAGE4 = historical-seed diagnostic; STAGE4_OWN50 = clean own-top50 re-score."""
    return tag in ("STAGE4", "STAGE4_OWN50")


def resolve_seeds(spec: dict, label: str) -> dict:
    """The evaluation seeds. Default: a RESERVED shared block (lo..hi). When SEEDS names a frozen
    seed_ids_file, a POST-HOC evaluation on exactly those seeds of an already-SPENT block, authorized
    by the spec's POST_HOC_MATCHED_ROLE_ABLATIONS entry for this label (the same rule as the
    specialist evaluator); no seed is spent and the block status never changes."""
    S = spec["SEEDS"]
    reg_id = str(S["registry_experiment_id"])
    lo, hi = (int(x) for x in str(S["block"]).split(".."))
    if not S.get("seed_ids_file"):
        seeds = list(range(lo, hi + 1))
        if len(seeds) != int(S["n"]):
            raise SystemExit(f"FAIL-CLOSED: spec block {lo}..{hi} is not n={S['n']}")
        return {"post_hoc": None, "seeds": seeds, "lo": lo, "hi": hi, "reg_id": reg_id,
                "seed_class": str(S["seed_class"])}
    f = ROOT / S["seed_ids_file"]
    if not f.is_file() or _sha(f) != S.get("seed_ids_sha256"):
        raise SystemExit(f"FAIL-CLOSED: {S['seed_ids_file']} missing or sha != SEEDS.seed_ids_sha256")
    seeds = sorted({int(x) for x in json.loads(f.read_text(encoding="utf-8"))})
    if len(seeds) != int(S["n"]):
        raise SystemExit(f"FAIL-CLOSED: {f.name} holds {len(seeds)} seeds, spec n={S['n']}")
    ph = post_hoc_block(spec, label, reg_id, lo, hi, SD, seeds=seeds)
    return {"post_hoc": ph, "seeds": seeds, "lo": lo, "hi": hi, "reg_id": reg_id,
            "seed_class": ph["seed_class"]}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_partial(partial: Path, fingerprint: dict) -> dict:
    """Finished cells of an interrupted run, keyed (z, pole, seed). Same rules as the Separated
    evaluator: the fingerprint must match exactly, a torn last line is dropped, any other
    malformed line refuses."""
    lines = partial.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}
    try:
        head = json.loads(lines[0])
    except ValueError:
        raise SystemExit(f"REFUSING: {partial.name} has an unreadable fingerprint line")
    if head.get("fingerprint") != fingerprint:
        raise SystemExit(f"REFUSING: {partial.name} was written by a different run configuration; "
                         f"recorded {head.get('fingerprint')} vs now {fingerprint}")
    done: dict = {}
    for i, ln in enumerate(lines[1:], start=2):
        try:
            row = json.loads(ln)
        except ValueError:
            if i == len(lines):
                break
            raise SystemExit(f"REFUSING: {partial.name} line {i} is malformed")
        done[(int(row["z"]), row["pole"], int(row["seed"]))] = row
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=tuple(ARM_KEY))
    ap.add_argument(
        "--team-size", type=int, required=True, choices=SUPPORTED_TEAM_SIZES,
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run of this arm from its PARTIAL rows file "
                         "(fingerprint must match); with no PARTIAL file it starts fresh")
    ap.add_argument(
        "--spec-tag",
        default="",
        choices=("", "SYM", "STAGE4", "STAGE4_OWN50"),
        help="SYM = symmetric-role family; STAGE4 = dual-branch z+r / role-only "
             "(historical-seed diagnostic); STAGE4_OWN50 = same students re-scored on "
             "dual-branch own top-50 (clean Stage-4 comparison; new OWN50_* labels)",
    )
    args = ap.parse_args()

    N_AGENTS = int(args.team_size)
    STAG = str(args.spec_tag)
    if _is_stage4(STAG) and args.arm == "generalist":
        raise SystemExit("REFUSING: Stage 4 forbids Generalist; use --arm role_only")
    SPEC_PATH = _spec_path(N_AGENTS, STAG)
    if not SPEC_PATH.is_file():
        raise SystemExit(
            f"FAIL-CLOSED: {SPEC_PATH.name} not found. Each scale needs its own frozen "
            f"eval spec pinning that scale's arms, seeds and poles."
        )
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: {SPEC_PATH.name} not frozen: {spec.get('status')!r}")

    arm_key = ARM_KEY[args.arm]
    if _is_stage4(STAG) and args.arm == "fully_shared":
        arm_key = "fully_shared_z_r"
    if arm_key not in spec["ARMS"]:
        raise SystemExit(f"REFUSING: SPEC missing ARMS[{arm_key!r}] — pin after distill freeze")
    arm = spec["ARMS"][arm_key]
    is_generalist = args.arm == "generalist"
    is_role_only = args.arm == "role_only"
    no_z = is_generalist or is_role_only
    label = str(arm["label"])
    rsd = resolve_seeds(spec, label)
    REG_ID, seed_class, lo, hi, seeds = rsd["reg_id"], rsd["seed_class"], rsd["lo"], rsd["hi"], rsd["seeds"]
    posthoc = rsd["post_hoc"]

    OUT = SD / f"{label}_CROSSOVER_EVAL_RESULT.json"
    ROWS_CSV = SD / f"{label.lower()}_crossover_eval_rows.csv"
    PARTIAL = SD / f"{label.lower()}_crossover_eval_rows.PARTIAL.jsonl"
    PREAUDIT_FLAG = SD / f"{label}_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json"
    log_arm = arm_key
    LOG = SD / "suite_sharing_std" / _family(N_AGENTS, STAG) / log_arm / "crossover_eval.log"
    EXP_ID = label

    ck = ROOT / arm["checkpoint"]
    if not ck.is_file():
        raise SystemExit(f"REFUSING: checkpoint missing: {ck}")
    ck_sha = _sha(ck)
    if ck_sha != arm["sha256"]:
        raise SystemExit(f"REFUSING: checkpoint sha mismatch vs SPEC pin")

    if posthoc is not None:
        shared_block = {}
        print(f"  Rule 9: POST-HOC on SPENT block {REG_ID} {lo}..{hi} ({len(seeds)} frozen seeds; "
              f"matched to {posthoc['primary_record']}); no seeds spent, block status unchanged")
    else:
        shared_block = shared_block_owner(REG_ID, label, lo, hi, seed_class)
        print(f"  Rule 9: shared block {REG_ID} {lo}..{hi} [{seed_class}/RESERVED], "
              f"label {label} declared among {shared_block.get('shared_by_labels')}")

    if OUT.is_file() or ROWS_CSV.is_file() or PREAUDIT_FLAG.is_file():
        raise SystemExit(f"REFUSING: an output for label {label!r} already exists; one-shot")
    if PARTIAL.is_file() and not args.resume and not args.dry_run:
        raise SystemExit(f"REFUSING: {PARTIAL.name} exists (an interrupted run of this label); "
                         f"pass --resume to continue it")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch,
        install_keyed_opponent_overlays,
    )
    from experiments.pole_attestation import (
        assert_live_matches_identity, pole_identity, resolve_pole_genome,
    )
    import experiments.r2_learned_crossover as R2
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    from gpu_env._core._entity_obs import augment_obs_with_entities
    from rl.curriculum import phase_from_tag

    R2.AGENTS = N_AGENTS
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    # Poles come from the governing certification, never from pole_*_genome() directly.
    # This evaluator's 4v4 predecessor installed pole_B_genome(4) -- plain OP7 -- while its
    # frozen spec named B3-3 (SUITE_4V4_POLE_B_IDENTITY_AUDIT.json). The spec's POLES
    # block must also agree with the certified pole, or the evaluation refuses to start.
    POLE_GENOMES = {p: resolve_pole_genome(p, N_AGENTS) for p in ("A", "B")}
    POLE_IDENTITY = {p: pole_identity(p, N_AGENTS, g) for p, g in POLE_GENOMES.items()}
    for p, spec_pole in (spec.get("POLES") or {}).items():
        if not isinstance(spec_pole, dict):
            continue                     # notes such as "source" sit beside the pole entries
        if p in POLE_IDENTITY and spec_pole.get("pole_config_hash") not in (None, POLE_IDENTITY[p]["pole_config_hash"]):
            raise SystemExit(
                f"FAIL-CLOSED: {SPEC_PATH.name} pins pole {p} hash {spec_pole['pole_config_hash'][:12]}, "
                f"but the certified pole resolves to {POLE_IDENTITY[p]['pole_config_hash'][:12]}")
        want = {k: (int(v) if isinstance(v, float) and float(v).is_integer() else v)
                for k, v in sorted((spec_pole.get("overlay") or {}).items())}
        if p in POLE_IDENTITY and "overlay" in spec_pole and POLE_IDENTITY[p]["overlay"] != want:
            raise SystemExit(
                f"FAIL-CLOSED: {SPEC_PATH.name} names pole {p} overlay {want}, but the "
                f"certified pole resolves to {POLE_IDENTITY[p]['overlay']}. Spec and "
                f"certification must agree before any seed is spent.")
    genomes_by_pole = {
        "A": {"OP6": POLE_GENOMES["A"]},
        "B": {"OP7": POLE_GENOMES["B"]},
    }

    print(f"SUITE {N_AGENTS}V{N_AGENTS} CROSSOVER EVAL  {label}  {_now()}  device={device}")
    print(f"  arm        {args.arm}")
    print(f"  checkpoint {ck.relative_to(ROOT)}  sha {ck_sha[:12]}...")
    print(f"  seeds      {seeds[0]}..{seeds[-1]} (n={len(seeds)}, {seed_class})")
    print(f"  bootstrap  n={N_BOOT}, alpha={ALPHA}, rng_seed={BOOTSTRAP_SEED}\n", flush=True)

    probe = R2.build_env(device, seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    if int(obs_space.spaces["grid"].shape[0]) != N_AGENTS:
        raise SystemExit(f"FAIL-CLOSED: env agent dim != {N_AGENTS}")
    probe.close()

    if args.arm in ("fully_shared", "generalist", "role_only"):
        from rl.custom_ppo.inference_policy import CustomPPOInferencePolicy
        from rl import suite_fully_shared_distill as FS

        if is_role_only:
            model, cfg, payload = FS.load_role_only(str(ck), obs_space, act_space, device=device)
        else:
            loader = FS.load_generalist if is_generalist else FS.load_fully_shared
            model, payload = loader(str(ck), obs_space, act_space, device=device)
            cfg = dict(payload.get("cfg") or {})
        if not no_z:
            cfg["fixed_latent_strategy"] = True
        policy = CustomPPOInferencePolicy(model, device=device, cfg=cfg)
        needs_entity = getattr(model, "entity_encoder", None) is not None
    else:
        from rl import ladder_rung1 as L1

        rung = {"share_encoder": 1, "share_backbone": 2, "share_macro": 3}[args.arm]
        if rung == 1:
            model, branch_cfg, _ = L1.load_rung1(str(ck), obs_space, act_space, device=device)
        else:
            model, branch_cfg, _ = L1.load_rung(
                rung, str(ck), obs_space, act_space, device=device,
            )
        policy = L1.make_dispatch_policy(model, branch_cfg, device=device)
        needs_entity = bool(getattr(model, "entity_repair_enabled", False))

    want_k = 0 if no_z else 2
    if int(getattr(model, "latent_k", 0) or 0) != want_k:
        raise SystemExit(f"REFUSING: latent_k must be {want_k}; got {getattr(model, 'latent_k', None)}")
    if not needs_entity:
        raise SystemExit(f"REFUSING: suite {N_AGENTS}v{N_AGENTS} students are entity-repair; refusing non-entity eval")

    needs_roles = bool(getattr(model, "role_conditioning_enabled", False)) or (
        hasattr(model, "branch") and any(
            bool(getattr(model.branch[z], "role_conditioning_enabled", False)) for z in ("z0", "z1")
        )
    )
    if _is_stage4(STAG) and not needs_roles:
        raise SystemExit("REFUSING: Stage-4 students require role_conditioning_enabled")
    k_defend = -(-N_AGENTS // 3) if STAG in ("SYM", "STAGE4", "STAGE4_OWN50") else {2: 1, 4: 2, 6: 1}[N_AGENTS]

    def _attach_roles(obs, core, hold, *, force: bool):
        from rl.custom_ppo.rule_role_assignment import roles_from_core
        roles = roles_from_core(core, hold, force=force, advance_age=True)
        out = dict(obs)
        out["roles"] = roles.detach().cpu().numpy().astype(np.float32)
        return out

    def force_z(z: int) -> None:
        if no_z:
            return                   # pi(a|o) or pi(a|o,r): there is no z to force
        policy.fixed_latent_strategy = True
        policy.fixed_latent_strategy_id = int(z)
        if hasattr(policy, "reset_strategy"):
            policy.reset_strategy()

    def run_cell(z: int, pole: str, seed: int) -> dict:
        from rl.custom_ppo.rule_role_assignment import RoleHoldState
        env = R2.build_env(device, seed)
        core = env.core
        role_hold = None
        if needs_roles:
            role_hold = RoleHoldState(
                int(env.num_envs), N_AGENTS, hold_ticks=8, device=device,
                fixed_for_episode=True, k_defend=k_defend,
            )
        try:
            force_z(z)
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            genomes = genomes_by_pole[pole]
            install_keyed_opponent_overlays(core, genomes)
            key = BASE_KEY[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            obs = augment_obs_with_entities(obs, core, side="blue")
            if needs_roles:
                obs = _attach_roles(obs, core, role_hold, force=True)
            assert_live_opponent_batch(
                core, genomes, allowed_keys=(key,),
                context=f"{label} z{z}@Pole{pole} seed {seed}",
            )
            resolved = core._bt_resolved_profile_tensors()
            got = resolved.get("min_alive_for_defender")
            got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
            if got_val != N_AGENTS:
                raise SystemExit(
                    f"FAIL-CLOSED: pole {pole} min_alive_for_defender={got_val}, expected {N_AGENTS}"
                )
            assert_live_matches_identity(core, POLE_IDENTITY[pole],
                                         context=f"{label} z{z}@Pole{pole} seed {seed}")
            terminal = None
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                obs = augment_obs_with_entities(obs, core, side="blue")
                if needs_roles:
                    obs = _attach_roles(obs, core, role_hold, force=False)
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    res = (i0 or {}).get("episode_result") or {}
                    terminal = (
                        int(res.get("blue_score", 0)),
                        int(res.get("red_score", 0)),
                    )
                    break
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            blue, red = terminal
            return {"blue": blue, "red": red, "win": int(blue > red), "margin": blue - red}
        finally:
            env.close()

    if args.dry_run:
        # One smoke step to prove entity+predict path (z forced for the sharing arms).
        from rl.custom_ppo.rule_role_assignment import RoleHoldState
        env = R2.build_env(device, 99_991_004)
        try:
            core = env.core
            install_keyed_opponent_overlays(core, genomes_by_pole["A"])
            env.env_method("set_phase", phase_from_tag("OP6"))
            env.env_method("set_next_opponent", "SCRIPTED", "OP6")
            obs = env.reset()
            obs["global_state"] = env.state()
            obs = augment_obs_with_entities(obs, core, side="blue")
            if needs_roles:
                hold = RoleHoldState(int(env.num_envs), N_AGENTS, hold_ticks=8, device=device,
                                     fixed_for_episode=True, k_defend=k_defend)
                obs = _attach_roles(obs, core, hold, force=True)
            force_z(0)
            action, _ = policy.predict(obs, deterministic=True)
            print(f"  dry-run predict OK  action_shape={np.asarray(action).shape}")
        finally:
            env.close()
        for pole in ("A", "B"):
            env = R2.build_env(device, 99_990_000 + N_AGENTS)
            try:
                core = env.core
                install_keyed_opponent_overlays(core, genomes_by_pole[pole])
                key = BASE_KEY[pole]
                env.env_method("set_phase", phase_from_tag(key))
                env.env_method("set_next_opponent", "SCRIPTED", key)
                env.reset()
                resolved = core._bt_resolved_profile_tensors()
                got = resolved.get("min_alive_for_defender")
                got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
                print(f"  dry-run pole {pole}: min_alive={got_val} "
                      f"{'OK' if got_val == N_AGENTS else 'MISMATCH'}")
                if got_val != N_AGENTS:
                    raise SystemExit("FAIL-CLOSED: dry-run pole mismatch")
                chk = assert_live_matches_identity(core, POLE_IDENTITY[pole],
                                                   context=f"dry-run pole {pole}")
                print(f"  dry-run pole {pole}: LIVE == certified {POLE_IDENTITY[pole]['genome_id']} "
                      f"on {chk['live']}")
            finally:
                env.close()
        print("\n  --dry-run PASS: nothing written.")
        return 0

    state = rs.RunState(SD, label)
    state.begin(checkpoint=str(ck), seed_base=seeds[0], n_seeds=len(seeds),
                team_size=N_AGENTS, arm=args.arm)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    zs = (0,) if no_z else (0, 1)
    cells = [(z, pole, seed) for z in zs for pole in ("A", "B") for seed in seeds]
    fingerprint = {"label": label, "arm": args.arm, "team_size": N_AGENTS, "checkpoint_sha256": ck_sha,
                   "seeds": [lo, hi], "spec_sha256": _sha(SPEC_PATH),
                   "poles": {p: POLE_IDENTITY[p]["pole_config_hash"] for p in ("A", "B")},
                   "stage4": _is_stage4(STAG), "k_defend": k_defend if needs_roles else None}
    if posthoc is not None:          # only present when used, so earlier PARTIAL files still resume
        fingerprint["seed_ids"] = seeds
    done = load_partial(PARTIAL, fingerprint) if PARTIAL.is_file() else {}
    if done:
        print(f"  RESUME: {len(done)} finished episode(s) read from {PARTIAL.name}", flush=True)
    else:
        PARTIAL.write_text(json.dumps({"fingerprint": fingerprint}) + "\n", encoding="utf-8")
    rows = []
    bar = tqdm_iter(cells, desc=f"{label}", unit="ep")
    for z, pole, seed in bar:
        set_postfix(bar, f"z{z}@Pole{pole} seed={seed}")
        row = done.get((z, pole, seed))
        if row is None:
            row = {"z": z, "pole": pole, "seed": seed, **run_cell(z, pole, seed)}
            with PARTIAL.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        rows.append({k: row[k] for k in ("z", "pole", "seed", "blue", "red", "win", "margin")})
        if seed == seeds[-1]:
            wr = float(np.mean([r["win"] for r in rows if r["z"] == z and r["pole"] == pole]))
            tag = "role_only" if is_role_only else ("pi_G" if is_generalist else f"z{z}")
            print(f"  {tag} on Pole {pole}: win rate {wr:.4f}", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def wins(z, pole):
        by = {r["seed"]: r["win"] for r in rows if r["z"] == z and r["pole"] == pole}
        return np.array([by[s] for s in seeds], dtype=np.float64)

    tie_or_reversal: list = []
    if no_z:
        v_a, v_b = _mean_ci(wins(0, "A")), _mean_ci(wins(0, "B"))
        name = "role_only" if is_role_only else "pi_G"
        print(f"\n  V({name}, A) {v_a['mean']:.4f} [{v_a['lcb95']:.4f}, {v_a['ucb95']:.4f}]")
        print(f"  V({name}, B) {v_b['mean']:.4f} [{v_b['lcb95']:.4f}, {v_b['ucb95']:.4f}]")
        claims = [rs.Claim(name=f"V_pole_{p}", recorded={k: v[k] for k in ("mean", "lcb95", "ucb95")},
                           minuend={"z": 0, "pole": p}, value_field="win")
                  for p, v in (("A", v_a), ("B", v_b))]
        primary = {"V_pole_A": v_a, "V_pole_B": v_b,
                   "note": ("Role-only has no z; report per-pole value. Strategy preservation is "
                            "read from behavioral signatures vs Fully Shared+z+r, not from Delta.")
                   if is_role_only else
                   ("no crossover delta for a single policy; Delta_G is formed against the "
                    "Separated reference sealed on the same seeds (GENERALIST_DEFINITION_V1)")}
        gate_passes = None
    else:
        delta_a = _mean_ci(wins(0, "A") - wins(1, "A"))
        delta_b = _mean_ci(wins(1, "B") - wins(0, "B"))
        delta_a["passes"] = bool(delta_a["mean"] > 0 and delta_a["lcb95"] > 0)
        delta_b["passes"] = bool(delta_b["mean"] > 0 and delta_b["lcb95"] > 0)
        gate_passes = bool(delta_a["passes"] and delta_b["passes"])
        print(f"\n  delta_A {delta_a['mean']:+.4f} [{delta_a['lcb95']:+.4f}, {delta_a['ucb95']:+.4f}]")
        print(f"  delta_B {delta_b['mean']:+.4f} [{delta_b['lcb95']:+.4f}, {delta_b['ucb95']:+.4f}]")
        print(f"  (frozen gate, provenance only: {'PASS' if gate_passes else 'FAIL'})")
        tie_or_reversal = [
            k for k, d in (("delta_A", delta_a), ("delta_B", delta_b)) if d["mean"] <= 0.0
        ]
        if tie_or_reversal:
            PREAUDIT_FLAG.write_text(json.dumps({
                "record": f"{label} crossover EVAL integrity audit REQUIRED",
                "status": "FLAGGED", "utc": _now(),
                "triggered_by": tie_or_reversal,
                "point_estimates": {"delta_A": delta_a["mean"], "delta_B": delta_b["mean"]},
                "raw_rows": str(ROWS_CSV.relative_to(ROOT)),
            }, indent=2), encoding="utf-8")
            print(f"  TIE/REVERSAL on {tie_or_reversal} -- integrity FLAG written; still sealing rows.")
            print(f"  -> {PREAUDIT_FLAG}")
        claims = [
            rs.Claim(name="delta_A", recorded={k: delta_a[k] for k in ("mean", "lcb95", "ucb95")},
                     minuend={"z": 0, "pole": "A"}, subtrahend={"z": 1, "pole": "A"}, value_field="win"),
            rs.Claim(name="delta_B", recorded={k: delta_b[k] for k in ("mean", "lcb95", "ucb95")},
                     minuend={"z": 1, "pole": "B"}, subtrahend={"z": 0, "pole": "B"}, value_field="win"),
        ]
        primary = {"delta_A": delta_a, "delta_B": delta_b, "passes": gate_passes}

    plan = rs.AuditPlan(
        rows_csv=ROWS_CSV, expected_rows=len(rows), expected_seeds=seeds,
        group_by=("z", "pole"), seed_field="seed",
        int_fields=("z", "seed", "blue", "red", "margin"), binary_fields=("win",), derived={},
        checkpoints={"student": (ck, ck_sha)}, spec_path=SPEC_PATH, claims=claims,
        n_boot=N_BOOT, alpha=ALPHA, rng_seed=BOOTSTRAP_SEED,
        seed_class=seed_class, experiment_id=REG_ID,
        registry_block=((lo, hi) if posthoc is not None else None),
    )
    # status is owned by seal(); do not set it here. Sealed != gate PASS.
    payload = {
        "record": f"{label} crossover EVAL",
        "one_shot": True,
        "utc": _now(),
        "arm": seed_class.upper(),
        "confirmatory": seed_class == "sealed_confirmatory",
        "implements": f"{SPEC_PATH.name}#EVALUATION",
        "suite_arm": args.arm,
        "team_size": N_AGENTS,
        "device": device,
        "checkpoint": str(ck.relative_to(ROOT)),
        "checkpoint_sha256": ck_sha,
        "seeds": {
            "block": [lo, hi], "n": len(seeds),
            "shared_across_z_and_poles": True,
            "seed_class": seed_class, "registry_experiment_id": REG_ID,
            "shared_by_labels": shared_block.get("shared_by_labels"),
            "seed_ids": (seeds if posthoc is not None else None),
            "post_hoc": (None if posthoc is None else
                         {"primary_record": posthoc["primary_record"], "block_status_unchanged": True}),
        },
        # The resolved experimental object each pole was evaluated on (Layer 3 reads this).
        "poles": POLE_IDENTITY,
        "PRIMARY_GATE": primary,
        "bootstrap": {
            "procedure": "paired percentile bootstrap over evaluation seeds",
            "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED,
        },
        "no_model_selection_occurred": True,
        "total_episodes": len(rows),
        "claim_boundary": str(spec.get("claim_boundary", "")),
        "integrity_flag": (str(PREAUDIT_FLAG.relative_to(ROOT)) if tie_or_reversal else None),
    }
    rs.seal(out_path=OUT, payload=payload, plan=plan, state=state, strict=False)
    sealed = json.loads(OUT.read_text(encoding="utf-8"))
    PARTIAL.unlink(missing_ok=True)       # the sealed rows CSV is now the record
    if posthoc is None and shared_block_all_sealed(shared_block, SD):
        sr.set_status(REG_ID, "SPENT", note=f"all shared labels sealed (last: {label})")
    print(f"\n  -> {OUT} ({sealed.get('status')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
