"""Collect SUITE_DISTILLATION_<N>V<N> under CLOSEST_DEFENDS(k), for any suite team size.

Pole A acts with the sealed DEFEND_ATTACK_SPLIT composite (pi_D + frozen attack)
so the CLOSEST_DEFENDS allocator is live in the state distribution. Pole B acts
with entity-repair pi_B -- or, when the spec declares
ACTING_DEPLOYMENT_locked.construction == "symmetric", with the same split composite
built from pi_B (pi_DB + frozen pi_B), and k_defend is the spec's own
(SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json). Entity tensors and roles are stored so
entity-repair KL teachers can be queried at distillation time.

Implements SUITE_DISTILLATION_<N>V<N>_SPEC.json. One-shot: refuses if the dataset
manifest already exists.

Run:
  python experiments/collect_suite_distillation_states.py --team-size 4 --device cuda
  python experiments/collect_suite_distillation_states.py --team-size 2 --device cpu --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
#: One collector for every suite scale: team size is an argument, and every path and knob
#: below is derived from it. There is no module-level team size
#: (CROSS_SCALE_CANONICAL_RECIPE_V1.json#STAGE_IMPLEMENTATIONS_required).
#: CLOSEST_DEFENDS defender count per scale -- the one scale knob besides N.
K_DEFEND_BY_SCALE = {2: 1, 4: 2, 6: 1}
SUPPORTED_TEAM_SIZES = (2, 4, 6)


#: --dataset-tag names a NEW collection at a scale whose default paths are taken by an older,
#: frozen one (e.g. the corrected 4v4 recollection, tag "V2", beside the invalidated plain-OP7
#: SUITE_DISTILLATION_4V4 set). Naming only: it changes no knob of the collection itself.
def _tag(tag: str) -> str:
    return f"_{tag}" if tag else ""


def _spec_path(n: int, tag: str = "") -> Path:
    return SD / f"SUITE_DISTILLATION_{n}V{n}{_tag(tag)}_SPEC.json"


#: The symmetric-role family's tag. Its shards live in their own tree, never beside an asymmetric set.
SYMMETRIC_TAG = "SYM"
SYMMETRIC_ROOT = SD / "suite_distillation_symmetric"


def _out_dir(n: int, smoke: bool, tag: str = "") -> Path:
    if tag == SYMMETRIC_TAG:
        return SYMMETRIC_ROOT / (f"{n}v{n}" + ("_SMOKE" if smoke else "")) / "states"
    stem = f"suite_distillation_{n}v{n}{_tag(tag).lower()}" + ("_SMOKE" if smoke else "")
    return SD / stem / "states"


def _manifest(n: int, smoke: bool, tag: str = "") -> Path:
    return SD / f"SUITE_DISTILLATION_{n}V{n}{_tag(tag)}_DATASET{'_SMOKE' if smoke else ''}.json"


ENTITY_KEYS = ("teammates", "teammates_valid", "enemies", "enemies_valid")
STORE_KEYS = (
    "grid", "vec", "agent_mask", "mask", "global_state",
    *ENTITY_KEYS, "roles", "decision_mask", "step",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_identity() -> dict:
    # The git sha is provenance; the scientific-tree hash is the experimental identity (a commit
    # that only adds artifacts moves HEAD but not the tree). Dirty = any scientific path, not only
    # experiments/ (see experiments/code_identity.py).
    from experiments import code_identity as CI
    return {"git_sha": CI._git(ROOT, "rev-parse", "HEAD").strip(), "git_dirty": CI.scientific_dirty(ROOT),
            "scientific_tree_sha256": CI.scientific_tree_sha256("HEAD", ROOT)}


def check_collection_seeds(spec: dict) -> dict:
    """Rule 9 for dataset collection: every collection block must be registered and RESERVED.

    The spec names each block ("lo..hi") and the registry id that owns it
    (SEEDS.registry_experiment_ids.<block>). Returns {block: (lo, hi, id)}. Refuses on a
    missing id, an unregistered id, a range that differs from the registered one, or a block
    that is not RESERVED -- a dataset is collected once, from seeds nobody else spent.
    """
    from experiments import seed_registry as SR
    seeds = spec["SEEDS"]
    ids = seeds.get("registry_experiment_ids") or {}
    out = {}
    for blk in ("collection_A", "collection_B"):
        lo, hi = (int(x) for x in str(seeds[blk]).split(".."))
        rid = ids.get(blk)
        if not rid:
            raise SystemExit(f"REFUSING (Rule 9): spec SEEDS.registry_experiment_ids.{blk} is missing")
        b = next((x for x in SR.load()["blocks"] if x["experiment_id"] == rid), None)
        if b is None:
            raise SystemExit(f"REFUSING (Rule 9): {rid} is not registered")
        if (b["lo"], b["hi"]) != (lo, hi):
            raise SystemExit(f"REFUSING (Rule 9): {rid} is {b['lo']}..{b['hi']}, spec says {lo}..{hi}")
        if b["status"] != "RESERVED":
            raise SystemExit(f"REFUSING (Rule 9): {rid} is {b['status']}")
        out[blk] = (lo, hi, rid)
    return out


def shard_is_resumable(shard: Path, fingerprint: str) -> dict | None:
    """Summary of an already-written shard from an interrupted run, or None to (re)collect it.

    A shard is reused only if its embedded fingerprint equals this run's exactly (pins, poles,
    allocator, seeds, spec); a shard without one, or from another configuration, is never
    reused. Each episode is a fresh env from its own seed with deterministic actions.
    """
    if not shard.is_file():
        return None
    try:
        z = np.load(shard, allow_pickle=False)
        if "fingerprint" not in z.files or str(z["fingerprint"]) != fingerprint:
            return None
        return {"steps": int(z["summary_steps"]), "blue": int(z["summary_blue"]),
                "red": int(z["summary_red"]), "decision_rows": int(z["step"].shape[0])}
    except Exception:  # noqa: BLE001 -- a torn/corrupt shard is simply re-collected
        return None


def spec_is_symmetric(spec: dict) -> bool:
    """True when the spec builds Pole B the same way as Pole A (pi_DB + frozen pi_B)."""
    c = (spec.get("ACTING_DEPLOYMENT_locked") or {}).get("construction", "asymmetric")
    if c not in ("asymmetric", "symmetric"):
        raise SystemExit(f"REFUSING: unknown ACTING_DEPLOYMENT_locked.construction {c!r}")
    return c == "symmetric"


def check_symmetric_acting(act: dict, kl: dict) -> None:
    """Fail closed unless a symmetric spec names both composites and keeps the defenders out of the
    teacher set: Pole_A = {pi_D, frozen_attack_pi_A}, Pole_B = {pi_D, frozen_attack_pi_B}, each pin a
    {path, sha256}; the attackers are the KL teachers; a defender is never a teacher."""
    need = {"Pole_A": ("pi_D", "frozen_attack_pi_A"), "Pole_B": ("pi_D", "frozen_attack_pi_B")}
    for pole, keys in need.items():
        blk = act.get(pole)
        for k in keys:
            pin = blk.get(k) if isinstance(blk, dict) else None
            if not (isinstance(pin, dict) and pin.get("path") and len(str(pin.get("sha256", ""))) == 64):
                raise SystemExit(f"REFUSING: symmetric spec needs ACTING_DEPLOYMENT_locked.{pole}.{k} "
                                 f"as {{path, sha256}}")
    if act["Pole_A"]["frozen_attack_pi_A"]["sha256"] != kl["pi_A"]["sha256"]:
        raise SystemExit("REFUSING: symmetric Pole A attacker != KL teacher pi_A")
    if act["Pole_B"]["frozen_attack_pi_B"]["sha256"] != kl["pi_B"]["sha256"]:
        raise SystemExit("REFUSING: symmetric Pole B attacker != KL teacher pi_B")
    teachers = {kl["pi_A"]["sha256"], kl["pi_B"]["sha256"]}
    da, db = act["Pole_A"]["pi_D"]["sha256"], act["Pole_B"]["pi_D"]["sha256"]
    if da in teachers or db in teachers or da == db:
        raise SystemExit("REFUSING: a defender checkpoint equals a teacher or the other defender")


def check_tag_matches_construction(spec: dict, tag: str) -> bool:
    """The symmetric construction runs only under --dataset-tag SYM, and that tag only with it, so a
    symmetric set can never land on an asymmetric path (or the reverse). Returns SYMMETRIC."""
    sym = spec_is_symmetric(spec)
    if sym != (tag == SYMMETRIC_TAG):
        raise SystemExit(f"REFUSING: --dataset-tag {tag or '(none)'} with a "
                         f"{'symmetric' if sym else 'asymmetric'} spec; the symmetric construction "
                         f"runs only under --dataset-tag {SYMMETRIC_TAG}, and that tag only with it")
    return sym


def acting_action(pole: str, obs, pi_D, frozen_attack, pi_B, pi_DB):
    """Pole A: split composite (pi_D on DEFEND, frozen pi_A on ATTACK). Pole B: plain pi_B, or -- in
    the symmetric construction (pi_DB given) -- the same split composite built from pi_B."""
    if pole == "A":
        return _composite_predict(pi_D, frozen_attack, obs)
    if pi_DB is not None:
        return _composite_predict(pi_DB, pi_B, obs)
    action, _ = pi_B.predict(obs, deterministic=True)
    return action


def symmetric_k_defend(spec: dict, n_agents: int) -> int:
    """k for a symmetric spec: the spec's ALLOCATOR_locked.k_defend, which must be ceil(N/3)."""
    k = int(spec["ALLOCATOR_locked"]["k_defend"])
    want = -(-n_agents // 3)
    if k != want:
        raise SystemExit(f"REFUSING: symmetric k_defend={k} != ceil({n_agents}/3)={want}")
    return k


def _composite_predict(trained_policy, attack_policy, obs) -> np.ndarray:
    from rl.custom_ppo.split_attack_defend import splice_actions
    import torch

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="NON-SCIENTIFIC plumbing smoke: disposable 999xxxxx seeds, "
             "_SMOKE paths, never touches the real manifest.",
    )
    ap.add_argument("--n-per-pole", type=int, default=None)
    ap.add_argument("--team-size", type=int, required=True, choices=SUPPORTED_TEAM_SIZES)
    ap.add_argument("--dataset-tag", default="",
                    help="name a new collection beside an older frozen one at the same scale "
                         "(e.g. V2 -> SUITE_DISTILLATION_4V4_V2_SPEC/_DATASET); naming only")
    ap.add_argument("--resume", action="store_true",
                    help="reuse shards from an interrupted run of THIS collection whose embedded "
                         "fingerprint matches exactly; without it, existing shards refuse")
    args = ap.parse_args()
    device = args.device
    smoke = bool(args.smoke)
    N_AGENTS = int(args.team_size)
    TAG = str(args.dataset_tag or "").strip().upper()
    if TAG and not TAG.isalnum():
        raise SystemExit(f"REFUSING: --dataset-tag must be alphanumeric, got {TAG!r}")
    SPEC_PATH = _spec_path(N_AGENTS, TAG)

    if not SPEC_PATH.is_file():
        raise SystemExit(
            f"FAIL-CLOSED: {SPEC_PATH.name} not found. Each scale needs its own frozen "
            f"collection spec naming the teachers, acting policies, seeds and poles."
        )
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: {SPEC_PATH.name} not frozen: {spec.get('status')!r}")

    SYMMETRIC = check_tag_matches_construction(spec, TAG)
    K_DEFEND = (symmetric_k_defend(spec, N_AGENTS) if SYMMETRIC else K_DEFEND_BY_SCALE[N_AGENTS])
    n_per_pole = int(args.n_per_pole) if args.n_per_pole else int(spec["DATASET"]["n_per_pole"])
    if smoke:
        n_per_pole = max(min(n_per_pole, 12), 10)
        seed_a_base, seed_b_base = 99_920_001, 99_920_101
    else:
        seeds_spec = spec["SEEDS"]
        seed_blocks = check_collection_seeds(spec)
        seed_a_base = seed_blocks["collection_A"][0]
        seed_b_base = seed_blocks["collection_B"][0]
        for blk, pole in (("collection_A", "A"), ("collection_B", "B")):
            lo, hi, _rid = seed_blocks[blk]
            if hi - lo + 1 != n_per_pole:
                raise SystemExit(f"REFUSING: {blk} {lo}..{hi} holds {hi - lo + 1} seeds, "
                                 f"n_per_pole is {n_per_pole}")
    out_dir = _out_dir(N_AGENTS, smoke, TAG)
    manifest = _manifest(N_AGENTS, smoke, TAG)

    seeds = {
        "A": list(range(seed_a_base, seed_a_base + n_per_pole)),
        "B": list(range(seed_b_base, seed_b_base + n_per_pole)),
    }

    if manifest.is_file():
        raise SystemExit(
            f"REFUSING: {manifest.name} exists; "
            f"{'smoke rerun' if smoke else 'the suite dataset is collected once'}"
        )
    if smoke and out_dir.is_dir():
        import shutil
        shutil.rmtree(out_dir)
    if not smoke and out_dir.is_dir() and any(out_dir.glob("*.npz")) and not args.resume:
        raise SystemExit(f"REFUSING: {out_dir} already holds shards from an interrupted "
                         f"collection; pass --resume to continue it")

    # Propagate team size by name (same trap as collect_distillation_states_scale).
    import experiments.collect_distillation_states as C
    import experiments.r2_learned_crossover as R2
    C.N_AGENTS = N_AGENTS
    R2.AGENTS = N_AGENTS

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch,
        install_keyed_opponent_overlays,
    )
    from experiments.pole_attestation import (
        assert_live_matches_identity, pole_identity, resolve_pole_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    from experiments.tqdm_loop import tqdm_iter
    from gpu_env._core._entity_obs import augment_obs_with_entities
    from rl.causal_supervision import decision_mask_from_core
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.custom_ppo.rule_role_assignment import RoleHoldState, roles_from_core

    # Poles come from the governing certification, never from pole_*_genome() directly:
    # the 4v4 dataset was once collected on plain OP7 while this spec named B3-3
    # (SUITE_4V4_POLE_B_IDENTITY_AUDIT.json). The frozen spec's own POLES block must also
    # agree with the certified pole, or the collection refuses to start.
    POLE_GENOMES = {p: resolve_pole_genome(p, N_AGENTS) for p in ("A", "B")}
    POLE_IDENTITY = {p: pole_identity(p, N_AGENTS, g) for p, g in POLE_GENOMES.items()}
    spec_poles = spec.get("POLES") or {}
    for p in ("A", "B"):
        spec_pole = spec_poles.get(p)
        if not isinstance(spec_pole, dict) or not isinstance(spec_pole.get("overlay"), dict):
            raise SystemExit(f"FAIL-CLOSED: {SPEC_PATH.name} POLES.{p}.overlay is missing; the spec "
                             f"must name each pole so it can be checked against the certification")
        want = {k: (int(v) if isinstance(v, float) and float(v).is_integer() else v)
                for k, v in sorted((spec_pole.get("overlay") or {}).items())}
        if p in POLE_IDENTITY and POLE_IDENTITY[p]["overlay"] != want:
            raise SystemExit(
                f"FAIL-CLOSED: {SPEC_PATH.name} names pole {p} overlay {want}, but the "
                f"certified pole resolves to {POLE_IDENTITY[p]['overlay']}. Spec and "
                f"certification must agree before any state is collected.")
    for p, ident in POLE_IDENTITY.items():
        print(f"  pole {p}  {ident['genome_id']:<24} {ident['overlay']}  "
              f"hash {ident['pole_config_hash'][:12]}  ({ident['certification_record']})")

    act = spec["ACTING_DEPLOYMENT_locked"]
    kl = spec["KL_TEACHERS_locked"]
    if SYMMETRIC:
        check_symmetric_acting(act, kl)
    pins = {
        "pi_D": act["Pole_A"]["pi_D"],
        "frozen_attack": act["Pole_A"]["frozen_attack_pi_A"],
        "pi_B_act": act["Pole_B"]["frozen_attack_pi_B"] if SYMMETRIC else act["Pole_B"],
        "pi_A_kl": kl["pi_A"],
        "pi_B_kl": kl["pi_B"],
    }
    if SYMMETRIC:
        pins["pi_DB"] = act["Pole_B"]["pi_D"]
    paths = {}
    for name, pin in pins.items():
        p = ROOT / pin["path"]
        if not p.is_file() or _sha(p) != pin["sha256"]:
            raise SystemExit(f"REFUSING: {name} missing or sha mismatch: {p}")
        paths[name] = p
    # KL and acting attack/B share pins; refuse drift.
    if pins["frozen_attack"]["sha256"] != pins["pi_A_kl"]["sha256"]:
        raise SystemExit("REFUSING: frozen_attack sha != KL pi_A sha")
    if pins["pi_B_act"]["sha256"] != pins["pi_B_kl"]["sha256"]:
        raise SystemExit("REFUSING: acting pi_B sha != KL pi_B sha")

    probe = R2.build_env(device, seeds["A"][0])
    obs_space, act_space = probe.observation_space, probe.action_space
    grid_dim = int(obs_space.spaces["grid"].shape[0])
    probe.close()
    if grid_dim != N_AGENTS:
        raise SystemExit(f"FAIL-CLOSED: probe grid dim {grid_dim} != {N_AGENTS}")

    pi_D = load_custom_ppo_policy(str(paths["pi_D"]), obs_space, act_space, device=device)
    frozen_attack = load_custom_ppo_policy(
        str(paths["frozen_attack"]), obs_space, act_space, device=device
    )
    pi_B = load_custom_ppo_policy(str(paths["pi_B_act"]), obs_space, act_space, device=device)
    pi_DB = (load_custom_ppo_policy(str(paths["pi_DB"]), obs_space, act_space, device=device)
             if SYMMETRIC else None)
    if pi_DB is not None:
        if not bool(getattr(pi_DB.model, "role_conditioning_enabled", False)):
            raise SystemExit("REFUSING: pi_DB must be role-conditioned")
        if getattr(pi_DB.model, "entity_encoder", None) is None:
            raise SystemExit("REFUSING: pi_DB expects entity repair")
        if bool(getattr(pi_B.model, "role_conditioning_enabled", False)):
            raise SystemExit("REFUSING: frozen pi_B (B-side attack) must NOT be role-conditioned")
        if bool(getattr(pi_DB.model, "uses_latent_strategy", False)):
            raise SystemExit("REFUSING: pi_DB is latent-conditioned")

    if not bool(getattr(pi_D.model, "role_conditioning_enabled", False)):
        raise SystemExit("REFUSING: pi_D must be role-conditioned")
    if bool(getattr(frozen_attack.model, "role_conditioning_enabled", False)):
        raise SystemExit("REFUSING: frozen attack must NOT be role-conditioned")
    if getattr(pi_D.model, "entity_encoder", None) is None:
        raise SystemExit("REFUSING: pi_D expects entity repair")
    if getattr(frozen_attack.model, "entity_encoder", None) is None:
        raise SystemExit("REFUSING: frozen attack expects entity repair")
    if getattr(pi_B.model, "entity_encoder", None) is None:
        raise SystemExit("REFUSING: pi_B expects entity repair")
    for name, pol in (("pi_D", pi_D), ("frozen_attack", frozen_attack), ("pi_B", pi_B)):
        if bool(getattr(pol.model, "uses_latent_strategy", False)):
            raise SystemExit(f"REFUSING: {name} is latent-conditioned")

    alloc = spec["ALLOCATOR_locked"]
    if int(alloc["k_defend"]) != K_DEFEND:
        raise SystemExit(f"REFUSING: SPEC k_defend={alloc['k_defend']} != {K_DEFEND}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"SUITE DISTILLATION {N_AGENTS}V{N_AGENTS} COLLECT  {_now()}  device={device}"
          f"{'  [SMOKE]' if smoke else ''}")
    print(f"  CLOSEST_DEFENDS k={K_DEFEND} fixed_for_episode={alloc['fixed_for_episode']}")
    print(f"  n_per_pole={n_per_pole}  A={seeds['A'][0]}..{seeds['A'][-1]}  "
          f"B={seeds['B'][0]}..{seeds['B'][-1]}\n", flush=True)

    def _attach_roles(obs, core, hold: RoleHoldState, *, force: bool):
        roles = roles_from_core(core, hold, force=force, advance_age=True)
        out = dict(obs)
        out["roles"] = roles.detach().cpu().numpy().astype(np.float32)
        return out

    ident = _git_identity()
    # The collector's own code is part of a shard's identity: a real collection refuses to run
    # from uncommitted collector code, and the commit is inside every shard's fingerprint, so
    # --resume can never splice shards from two collector versions into one dataset.
    if not smoke and ident["git_dirty"]:
        raise SystemExit("REFUSING: scientific code (experiments/ rl/ gpu_env/ configs/ root *.py) has "
                         "uncommitted changes; a dataset must be "
                         "collected by committed collector code (its git sha is recorded per shard)")
    fingerprint = json.dumps({
        "team_size": N_AGENTS, "k_defend": K_DEFEND, "allocator": alloc,
        "pins": {k: v["sha256"] for k, v in pins.items()},
        "poles": {p: POLE_IDENTITY[p]["pole_config_hash"] for p in ("A", "B")},
        "seeds": {k: [v[0], v[-1]] for k, v in seeds.items()},
        "spec_sha256": _sha(SPEC_PATH), "device": str(device),
        "collector_git_sha": ident["git_sha"],
    }, sort_keys=True)

    shards, totals = [], {
        "A": {"episodes": 0, "steps": 0, "decision_rows": 0, "wins": 0},
        "B": {"episodes": 0, "steps": 0, "decision_rows": 0, "wins": 0},
    }

    for pole in ("A", "B"):
        ep_iter = tqdm_iter(
            list(enumerate(seeds[pole])),
            desc=f"suite{N_AGENTS}v{N_AGENTS} pole {pole}",
            total=n_per_pole,
            unit="ep",
        )
        for ep_i, seed in ep_iter:
            shard = out_dir / f"{pole}_{seed}.npz"
            prior = shard_is_resumable(shard, fingerprint) if args.resume else None
            if prior is not None:
                shards.append({"pole": pole, "episode": ep_i, "seed": seed, "steps": prior["steps"],
                               "decision_rows": prior["decision_rows"], "blue": prior["blue"],
                               "red": prior["red"], "file": str(shard.relative_to(ROOT)),
                               "resumed": True})
                tt = totals[pole]
                tt["episodes"] += 1
                tt["steps"] += prior["steps"]
                tt["decision_rows"] += prior["decision_rows"]
                tt["wins"] += int(prior["blue"] > prior["red"])
                continue
            env = R2.build_env(device, seed)
            core = env.core
            role_hold = RoleHoldState(
                int(env.num_envs),
                N_AGENTS,
                hold_ticks=int(alloc["hold_ticks_H_r"]),
                device=device,
                fixed_for_episode=bool(alloc["fixed_for_episode"]),
                k_defend=K_DEFEND,
            )
            try:
                pi_D.reset_strategy()
                frozen_attack.reset_strategy()
                pi_B.reset_strategy()
                if pi_DB is not None:
                    pi_DB.reset_strategy()
                core._bt_profile_override = None
                core._sds_opening_hold_steps = 0
                genomes = (
                    {"OP6": POLE_GENOMES["A"]} if pole == "A"
                    else {"OP7": POLE_GENOMES["B"]}
                )
                install_keyed_opponent_overlays(core, genomes)
                key = P0.POLES[pole]
                env.env_method("set_phase", phase_from_tag(key))
                env.env_method("set_next_opponent", "SCRIPTED", key)
                obs = env.reset()
                obs["global_state"] = env.state()
                obs = augment_obs_with_entities(obs, core, side="blue")
                # Roles always attached under CD so Pole A composite + stored
                # allocator audit see the same geometric assignment.
                obs = _attach_roles(obs, core, role_hold, force=True)
                assert_live_opponent_batch(
                    core, genomes, allowed_keys=(key,),
                    context=f"suite distill {N_AGENTS}v{N_AGENTS} {pole} seed {seed}",
                )
                resolved = core._bt_resolved_profile_tensors()
                got = resolved.get("min_alive_for_defender")
                got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
                if got_val != N_AGENTS:
                    raise SystemExit(
                        f"FAIL-CLOSED: pole {pole} min_alive_for_defender="
                        f"{got_val}, expected {N_AGENTS}"
                    )
                # min_alive is 4 on BOTH plain OP7 and certified B3-3, so it is not a pole check.
                assert_live_matches_identity(
                    core, POLE_IDENTITY[pole],
                    context=f"suite distill {N_AGENTS}v{N_AGENTS} {pole} seed {seed}")

                rows = {k: [] for k in STORE_KEYS}
                steps, terminal = 0, None
                for t in range(R2.MAX_STEPS):
                    d = decision_mask_from_core(core, N_AGENTS, side="blue")
                    d_np = np.asarray(d.detach().cpu())[0].copy()
                    if d_np.any():
                        for k in ("grid", "vec", "agent_mask", "mask", "global_state",
                                  *ENTITY_KEYS, "roles"):
                            rows[k].append(np.asarray(obs[k])[0].copy())
                        rows["decision_mask"].append(d_np)
                        rows["step"].append(t)

                    action = acting_action(pole, obs, pi_D, frozen_attack, pi_B, pi_DB)
                    env.step_async(action)
                    obs, _r, done, info = env.step_wait()
                    obs["global_state"] = env.state()
                    obs = augment_obs_with_entities(obs, core, side="blue")
                    obs = _attach_roles(obs, core, role_hold, force=False)
                    steps += 1
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
            finally:
                env.close()

            n_rows = len(rows["step"])
            save = {
                "fingerprint": np.array(fingerprint),
                "summary_steps": np.array(steps, dtype=np.int64),
                "summary_blue": np.array(terminal[0], dtype=np.int64),
                "summary_red": np.array(terminal[1], dtype=np.int64),
                "grid": np.asarray(rows["grid"], dtype=np.float32),
                "vec": np.asarray(rows["vec"], dtype=np.float32),
                "agent_mask": np.asarray(rows["agent_mask"], dtype=np.float32),
                "mask": np.asarray(rows["mask"], dtype=np.float32),
                "global_state": np.asarray(rows["global_state"], dtype=np.float32),
                "teammates": np.asarray(rows["teammates"], dtype=np.float32),
                "teammates_valid": np.asarray(rows["teammates_valid"], dtype=bool),
                "enemies": np.asarray(rows["enemies"], dtype=np.float32),
                "enemies_valid": np.asarray(rows["enemies_valid"], dtype=bool),
                "roles": np.asarray(rows["roles"], dtype=np.float32),
                "decision_mask": np.asarray(rows["decision_mask"], dtype=bool),
                "step": np.asarray(rows["step"], dtype=np.int32),
                "pole": np.full((n_rows,), 0 if pole == "A" else 1, dtype=np.int8),
                "episode": np.full((n_rows,), ep_i, dtype=np.int32),
                "seed": np.full((n_rows,), seed, dtype=np.int64),
            }
            np.savez_compressed(shard, **save)
            shards.append({
                "pole": pole, "episode": ep_i, "seed": seed, "steps": steps,
                "decision_rows": n_rows, "blue": terminal[0], "red": terminal[1],
                "file": str(shard.relative_to(ROOT)),
            })
            tt = totals[pole]
            tt["episodes"] += 1
            tt["steps"] += steps
            tt["decision_rows"] += n_rows
            tt["wins"] += int(terminal[0] > terminal[1])

    for pole in ("A", "B"):
        if totals[pole]["decision_rows"] <= 0:
            raise SystemExit(f"REFUSING: pole {pole} produced zero decision-bearing rows")

    manifest.write_text(json.dumps({
        "record": f"Suite distillation state set ({N_AGENTS}v{N_AGENTS})"
                  + (" -- NON-SCIENTIFIC SMOKE" if smoke else ""),
        "status": "SMOKE_NOT_SCIENTIFIC" if smoke else "FROZEN_DATASET",
        "utc": _now(),
        "implements": f"{SPEC_PATH.name}#DATASET",
        "team_size": N_AGENTS,
        "dataset_tag": TAG or None,
        # The resolved experimental object each pole's states were collected against --
        # Layer 3 of the cross-scale identity attestation reads this.
        "poles": POLE_IDENTITY,
        "allocator": {
            "rule": "CLOSEST_DEFENDS",
            "k_defend": K_DEFEND,
            "fixed_for_episode": bool(alloc["fixed_for_episode"]),
            "hold_ticks_H_r": int(alloc["hold_ticks_H_r"]),
        },
        "acting": {
            "Pole_A": "DEFEND_ATTACK_SPLIT(pi_D + frozen_attack)",
            "Pole_B": ("DEFEND_ATTACK_SPLIT(pi_DB + frozen pi_B)" if SYMMETRIC
                       else "pi_B entity-repair"),
            "construction": "symmetric" if SYMMETRIC else "asymmetric",
            **({"pi_DB": {"path": str(pins["pi_DB"]["path"]), "sha256": pins["pi_DB"]["sha256"]}}
               if SYMMETRIC else {}),
            "pi_D": {"path": str(pins["pi_D"]["path"]), "sha256": pins["pi_D"]["sha256"]},
            "frozen_attack": {
                "path": str(pins["frozen_attack"]["path"]),
                "sha256": pins["frozen_attack"]["sha256"],
            },
            "pi_B": {"path": str(pins["pi_B_act"]["path"]), "sha256": pins["pi_B_act"]["sha256"]},
        },
        "teachers": {
            "pi_A": {"path": str(pins["pi_A_kl"]["path"]), "sha256": pins["pi_A_kl"]["sha256"]},
            "pi_B": {"path": str(pins["pi_B_kl"]["path"]), "sha256": pins["pi_B_kl"]["sha256"]},
        },
        "seeds": {k: [v[0], v[-1]] for k, v in seeds.items()},
        "seed_registry": (None if smoke else {blk: {"block": [lo, hi], "experiment_id": rid}
                                               for blk, (lo, hi, rid) in seed_blocks.items()}),
        "collector": {"module": "experiments/collect_suite_distillation_states.py",
                      "spec": SPEC_PATH.name, "spec_sha256": _sha(SPEC_PATH), **ident},
        "fingerprint": json.loads(fingerprint),
        # Self-description of the symmetric family: the defenders choose which states are visited;
        # the KL teachers stay the repaired pi_A / pi_B.
        **({"dataset_mode": "symmetric_roles", "symmetric_roles": {
            "team_size": N_AGENTS, "k_defend": K_DEFEND,
            "episodes_per_regime": n_per_pole,
            "pole_A": {"attacker": pins["frozen_attack"], "defender": pins["pi_D"]},
            "pole_B": {"attacker": pins["pi_B_act"], "defender": pins["pi_DB"]},
            "teacher_A": pins["pi_A_kl"], "teacher_B": pins["pi_B_kl"],
            "defenders_are_not_teachers": True,
            "spec": SPEC_PATH.name, "spec_sha256": _sha(SPEC_PATH), "git_sha": ident["git_sha"],
        }} if SYMMETRIC else {}),
        "resumed_shards": sum(1 for s_ in shards if s_.get("resumed")),
        "device": device,
        "decision_rows_only": True,
        "stored_entity_tensors": True,
        "stored_roles": True,
        "totals": {
            p: {**t, "teacher_deployment_win_rate": t["wins"] / max(1, t["episodes"])}
            for p, t in totals.items()
        },
        "shards": shards,
    }, indent=2), encoding="utf-8")
    if not smoke:
        from experiments import seed_registry as SR
        for blk, (_lo, _hi, rid) in seed_blocks.items():
            SR.set_status(rid, "SPENT", note=f"{manifest.name} written ({blk})")
    print(f"\n  A: {totals['A']}\n  B: {totals['B']}\n  -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
