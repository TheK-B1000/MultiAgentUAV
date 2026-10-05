"""Shared pieces for the three frozen 2v2 strengthening studies.

    BEHAVIOR_SIGNATURES_2V2_V1, REPLICATION_2V2_DUAL_BRANCH_V1, Z_INTERVENTION_2V2_V1
    (artifacts/strategic_demand/sppo/2v2_strengthening/)

The episode loop mirrors eval_specialist_crossover_scaled.run_cell (Ours, dual-branch
composite) and eval_suite_sharing_crossover.run_cell (Fully Shared+z+r) step for step:
same env builder, opponent installation, pole attestation, role allocator (CLOSEST_DEFENDS,
H_r = 8, fixed for the episode, k = 1 at 2v2) and deterministic actions. The only addition is
an optional per-tick callback that READS state; it never alters an action.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N = 2
K_DEFEND = 1
BASE_KEY = {"A": "OP6", "B": "OP7"}
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
STRENGTH_DIR = SD / "2v2_strengthening"
DEPLOY_MANIFEST = ROOT / "2v2" / "dual_branch_deploy_manifest.json"
N_BOOT, ALPHA, BOOT_SEED = 20000, 0.05, 7
R_HOME = 4.5


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_frozen_spec(name: str) -> dict:
    p = STRENGTH_DIR / f"{name}_SPEC.json"
    spec = json.loads(p.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN":
        raise SystemExit(f"REFUSING: {p.name} status {spec.get('status')!r} != FROZEN")
    return spec


def check_block(experiment_id: str, lo: int, hi: int, *, smoke: bool) -> None:
    """Smoke runs must stay in 999xxxxx; real runs must match a RESERVED registry block exactly."""
    from experiments.seed_registry import SMOKE_LO, SMOKE_HI, load
    if smoke:
        if not (SMOKE_LO <= lo <= hi <= SMOKE_HI):
            raise SystemExit(f"REFUSING: smoke seeds {lo}..{hi} outside {SMOKE_LO}..{SMOKE_HI}")
        return
    blocks = [b for b in load()["blocks"] if b["experiment_id"] == experiment_id]
    if len(blocks) != 1 or (blocks[0]["lo"], blocks[0]["hi"]) != (lo, hi):
        raise SystemExit(f"REFUSING: {experiment_id} {lo}..{hi} is not the registered block")
    if blocks[0]["status"] != "RESERVED":
        raise SystemExit(f"REFUSING: {experiment_id} block status {blocks[0]['status']} (already spent?)")


# ------------------------------------------------------------------------- statistics
def mean_ci(values) -> dict:
    """Paired percentile bootstrap with the seed as the resampling unit (eval_hog_psp_v3._mean_ci)."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(v.mean()), "lcb95": float(lo), "ucb95": float(hi), "n": int(len(v))}


def sign_flip_p(d, n_perm: int = 20000, rng_seed: int = BOOT_SEED) -> float:
    """Two-sided paired sign-flip permutation p-value for mean(d) = 0."""
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        return float("nan")
    obs = abs(d.mean())
    rng = np.random.default_rng(rng_seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1))


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down over exactly the given family. Returns adjusted p and reject flag."""
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    out, running, stop = {}, 0.0, False
    for i, k in enumerate(keys):
        adj = min(1.0, (m - i) * pvals[k])
        running = max(running, adj)
        reject = (not stop) and pvals[k] <= alpha / (m - i)
        if not reject:
            stop = True
        out[k] = {"p": pvals[k], "p_holm": running, "reject": bool(reject)}
    return out


# ------------------------------------------------------------------------- poles
def resolve_poles():
    from experiments.pole_attestation import (
        assert_resolved_matches_certification, governing_certification,
        pole_identity, resolve_pole_genome,
    )
    _v, cert = governing_certification(N)
    genomes, identity = {}, {}
    for pole in ("A", "B"):
        g = resolve_pole_genome(pole, N, None)
        assert_resolved_matches_certification(pole, N, cert, g, is_smoke=False)
        genomes[pole] = {BASE_KEY[pole]: g}
        identity[pole] = pole_identity(pole, N, g)
    return genomes, identity, cert.name


def spaces(device: str):
    import experiments.r2_learned_crossover as R2
    R2.AGENTS = N
    probe = R2.build_env(device, 99_900_640)
    try:
        return probe.observation_space, probe.action_space
    finally:
        probe.close()


# ------------------------------------------------------------------------- actors
class OursActor:
    """Dual-branch composite: DEFEND slots -> pi_X final, ATTACK slots -> exported attack branch."""
    needs_roles = True

    def __init__(self, defend_policy, attack_policy):
        self.defend, self.attack = defend_policy, attack_policy

    def reset(self) -> None:
        for p in (self.defend, self.attack):
            if hasattr(p, "reset_strategy"):
                p.reset_strategy()

    def act(self, obs) -> np.ndarray:
        import torch
        from rl.custom_ppo.split_attack_defend import splice_actions
        trained, _ = self.defend.predict(obs, deterministic=True)
        attack, _ = self.attack.predict(obs, deterministic=True)
        n_agents = int(self.defend.model.n_agents)
        hpa = int(self.defend.model.heads_per_agent)
        t = torch.as_tensor(np.asarray(trained), dtype=torch.long).reshape(1, -1)
        a = torch.as_tensor(np.asarray(attack), dtype=torch.long).reshape(1, -1)
        roles = torch.as_tensor(np.asarray(obs["roles"]), dtype=torch.float32).reshape(1, n_agents)
        return splice_actions(t, a, roles < 0.5, hpa).reshape(-1).numpy().astype(np.int64)


class StudentActor:
    """Fully Shared+z+r with z forced for the episode (eval_suite_sharing_crossover.force_z)."""
    needs_roles = True

    def __init__(self, policy, z: int):
        self.policy, self.z = policy, int(z)

    def reset(self) -> None:
        self.policy.fixed_latent_strategy = True
        self.policy.fixed_latent_strategy_id = self.z
        if hasattr(self.policy, "reset_strategy"):
            self.policy.reset_strategy()

    def act(self, obs) -> np.ndarray:
        action, _ = self.policy.predict(obs, deterministic=True)
        return np.asarray(action)


def load_ours(device: str, manifest_path: Path = DEPLOY_MANIFEST, pins: dict | None = None) -> dict:
    """{'A': OursActor, 'B': OursActor} from a dual-branch deploy manifest, sha-checked."""
    from rl.custom_ppo import load_custom_ppo_policy
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    osp, asp = spaces(device)
    out = {}
    for s in ("A", "B"):
        pols = {}
        for half in ("defend", "attack"):
            ent = man[f"pi_{s}_{half}"]
            p = ROOT / ent["path"]
            got = sha256(p)
            want = (pins or {}).get(f"pi_{s}_{half}", ent["sha256"])
            if got != ent["sha256"] or got != want:
                raise SystemExit(f"FAIL-CLOSED: pi_{s}_{half} sha {got[:12]} != pinned {want[:12]}")
            pols[half] = load_custom_ppo_policy(str(p), osp, asp, device=device)
        if not bool(getattr(pols["defend"].model, "role_conditioning_enabled", False)):
            raise SystemExit(f"REFUSING: pi_{s} defend half must be role-conditioned")
        if bool(getattr(pols["attack"].model, "role_conditioning_enabled", False)):
            raise SystemExit(f"REFUSING: pi_{s} attack half must NOT be role-conditioned")
        out[s] = OursActor(pols["defend"], pols["attack"])
    return out


def load_student(device: str, ckpt: str, sha: str):
    from rl import suite_fully_shared_distill as FS
    from rl.custom_ppo.inference_policy import CustomPPOInferencePolicy
    p = ROOT / ckpt
    got = sha256(p)
    if got != sha:
        raise SystemExit(f"FAIL-CLOSED: student sha {got[:12]} != pinned {sha[:12]}")
    osp, asp = spaces(device)
    model, payload = FS.load_fully_shared(str(p), osp, asp, device=device)
    cfg = dict(payload.get("cfg") or {})
    cfg["fixed_latent_strategy"] = True
    if int(getattr(model, "latent_k", 0) or 0) != 2:
        raise SystemExit("REFUSING: Fully Shared+z+r must have latent_k = 2")
    if not bool(getattr(model, "role_conditioning_enabled", False)):
        raise SystemExit("REFUSING: Fully Shared+z+r must be role-conditioned")
    return CustomPPOInferencePolicy(model, device=device, cfg=cfg)


# ------------------------------------------------------------------------- episode
def snapshot(core) -> dict[str, np.ndarray]:
    def a(x, dt=None):
        v = x.detach().cpu().numpy()[0]
        return v.astype(dt) if dt is not None else v
    return {
        "pos": a(core.blue_pos, np.float32),
        "alive": a(core.blue_alive, bool),
        "tagged": a(core.blue_tagged, bool),
        "carrying": a(core.blue_carrying, bool),
        "own_home": a(core.blue_flag_home, np.float32).reshape(-1)[:2],
        "enemy_flag": a(core.red_flag_pos, np.float32).reshape(-1)[:2],
        "enemy_home": a(core.red_flag_home, np.float32).reshape(-1)[:2],
        "cols": np.float32(core.cols),
    }


def run_episode(actor, pole: str, seed: int, device: str, genomes: dict, identity: dict,
                on_tick: Callable[[Any, dict, np.ndarray, int], None] | None = None,
                context: str = "") -> dict:
    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import assert_live_opponent_batch, install_keyed_opponent_overlays
    from experiments.pole_attestation import assert_live_matches_identity
    from gpu_env._core._entity_obs import augment_obs_with_entities
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo.rule_role_assignment import RoleHoldState, roles_from_core

    R2.AGENTS = N
    env = R2.build_env(device, seed)
    core = env.core
    hold = RoleHoldState(int(env.num_envs), N, hold_ticks=8, device=device,
                         fixed_for_episode=True, k_defend=K_DEFEND)

    def attach(obs, force):
        roles = roles_from_core(core, hold, force=force, advance_age=True)
        out = dict(obs)
        out["roles"] = roles.detach().cpu().numpy().astype(np.float32)
        return out

    ctx = f"{context} {pole} seed {seed}"
    try:
        actor.reset()
        core._bt_profile_override = None
        core._sds_opening_hold_steps = 0
        g = genomes[pole]
        install_keyed_opponent_overlays(core, g)
        key = BASE_KEY[pole]
        env.env_method("set_phase", phase_from_tag(key))
        env.env_method("set_next_opponent", "SCRIPTED", key)
        obs = env.reset()
        obs["global_state"] = env.state()
        obs = augment_obs_with_entities(obs, core, side="blue")
        obs = attach(obs, True)
        assert_live_opponent_batch(core, g, allowed_keys=(key,), context=ctx)
        got = core._bt_resolved_profile_tensors().get("min_alive_for_defender")
        got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
        if got_val != N:
            raise SystemExit(f"FAIL-CLOSED: pole {pole} min_alive_for_defender={got_val}, expected {N}")
        assert_live_matches_identity(core, identity[pole], context=ctx)
        terminal, steps = None, 0
        for t in range(R2.MAX_STEPS):
            action = actor.act(obs)
            if on_tick is not None:
                on_tick(core, obs, action, t)
            env.step_async(action)
            obs, _r, done, info = env.step_wait()
            obs["global_state"] = env.state()
            obs = augment_obs_with_entities(obs, core, side="blue")
            obs = attach(obs, False)
            steps += 1
            if bool(np.asarray(done).any()):
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                res = (i0 or {}).get("episode_result") or {}
                terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                break
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        return {"blue": terminal[0], "red": terminal[1], "win": int(terminal[0] > terminal[1]),
                "margin": terminal[0] - terminal[1], "steps": steps}
    finally:
        env.close()


# ------------------------------------------------------------------------- resume
def read_partial(path: Path) -> dict:
    """Rows from ``path`` and every shard file beside it (``<stem>.shard*.jsonl``), keyed by row key."""
    done = {}
    files = [path] + sorted(path.parent.glob(f"{path.stem}.shard*.jsonl"))
    for f in files:
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done[r["key"]] = r
    return done


def shard_path(path: Path, shard: str | None) -> Path:
    """Where this process appends: the shared partial, or its own shard file for --shard i/n."""
    if not shard:
        return path
    i, n = (int(x) for x in shard.split("/"))
    return path.parent / f"{path.stem}.shard{i}of{n}.jsonl"


def in_shard(index: int, shard: str | None) -> bool:
    if not shard:
        return True
    i, n = (int(x) for x in shard.split("/"))
    if not (0 <= i < n):
        raise SystemExit(f"REFUSING: bad --shard {shard}")
    return index % n == i


def append_partial(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
