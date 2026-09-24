r"""Pole-B reward / advantage / credit diagnostic (post-GETFLAG failure).

Governed by artifacts/strategic_demand/sppo/POLE_B_CREDIT_ADVANTAGE_DIAGNOSTIC_SPEC.json.
DIAGNOSTIC / EXPLORATORY — not specialization recovery, not a training repair launcher.

    python -m experiments.eval_pole_b_credit_advantage_diagnostic [--dry-run]
    python -m experiments.eval_pole_b_credit_advantage_diagnostic --device cuda
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.eval_pole_b_behavioral_diagnosis as D  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LABEL = "POLE_B_CREDIT_ADVANTAGE_DIAGNOSTIC"
SPEC = SD / f"{LABEL}_SPEC.json"
EXPERIMENT_ID = LABEL

SEED_LO, N_SEEDS = 19_400_001, 24
SCALE = ROOT / "artifacts" / "scale_4v4_specialists"
EXPL = ROOT / "artifacts" / "exploratory_scale_4v4_specialists"
CORR = SCALE / "pi_B_specialist_4v4_b3_entity_repair_corrected" / "ckpts"

MACRO_NAMES = D.MACRO_NAMES
DEFEND_RADIUS = 4.0
PRESSURE_RADIUS = 3.0
N_BOOT, ALPHA, BOOTSTRAP_SEED = 5_000, 0.05, 17

POLICIES: list[tuple[str, Path]] = [
    ("B_t500k", CORR / "ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_500000.zip"),
    ("B_final", CORR / "final_pi_B_specialist_4v4_b3_entity_repair_corrected.zip"),
    (
        "GETFLAG",
        EXPL
        / "exploratory_pi_B_specialist_4v4_b3_entity_repair_getflag_preserve"
        / "ckpts"
        / "final_exploratory_pi_B_specialist_4v4_b3_entity_repair_getflag_preserve.zip",
    ),
    (
        "Assignment_v1",
        EXPL
        / "exploratory_pi_B_specialist_4v4_b3_entity_repair_assignment_v1"
        / "ckpts"
        / "final_exploratory_pi_B_specialist_4v4_b3_entity_repair_assignment_v1.zip",
    ),
    (
        "A3",
        SCALE / "pi_A_specialist_4v4_b3_entity_repair" / "ckpts" / "final_pi_A_specialist_4v4_b3_entity_repair.zip",
    ),
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _info0(info) -> dict:
    i0 = info[0] if isinstance(info, (list, tuple)) else info
    return dict(i0 or {})


def _macro_bucket(macro: int, near_own: bool) -> str:
    if macro == 2:
        return "GET_FLAG"
    if macro == 4:
        return "GO_HOME"
    if macro == 0:
        return "GUARD" if near_own else "GO_TO"
    if macro in (1, 3):
        return "GUARD" if near_own else "OTHER"
    return "OTHER"


@dataclass
class PolicyRuntime:
    label: str
    path: Path
    inference: object
    gamma: float
    gae_lambda: float
    return_norm: object
    role_hold: object | None
    assignment_hold: object | None


def _load_return_norm(payload: dict):
    from rl.custom_ppo.return_normalization import ReturnNormalizer

    cfg = payload.get("cfg") or {}
    enabled = bool(cfg.get("normalize_returns", True))
    rn = ReturnNormalizer(enabled=enabled)
    rn.load_state_dict(
        {
            "mean": float(payload.get("return_norm_mean", 0.0) or 0.0),
            "var": float(payload.get("return_norm_var", 1.0) or 1.0),
            "count": float(payload.get("return_norm_count", 1e-4) or 1e-4),
        }
    )
    return rn


def _privileged_zi_keys(core) -> list[int]:
    from gpu_env._core._scripted_blue_styles import gate2b_defender_hold_radius
    from rl.custom_ppo.guard_assignment import assign_guard_v2_responsibilities, zi_discrete_key

    home = core.blue_flag_home
    if home.dim() == 3:
        home = home[:, 0, :]
    eflag = core.red_flag_pos
    if eflag.dim() == 3:
        eflag = eflag[:, 0, :]
    on_our = core._is_on_home_side("blue", core.red_x)
    zi = assign_guard_v2_responsibilities(
        own_x=core.blue_x,
        own_y=core.blue_y,
        own_alive=core.blue_alive.bool(),
        home_xy=home,
        enemy_x=core.red_x,
        enemy_y=core.red_y,
        enemy_alive=core.red_alive.bool(),
        enemy_tagged=core.red_tagged.bool(),
        enemy_flag_xy=eflag,
        on_our_side=on_our,
        defense_radius=float(gate2b_defender_hold_radius(core.cfg)),
        midline_fn=lambda tx: core._is_on_home_side("blue", tx),
    )
    key = zi_discrete_key(zi["responsibility"], zi["assigned_entity"])
    return [int(key[0, i].item()) for i in range(4)]


def _per_agent_geom(core) -> list[dict]:
    bx = D._np(core.blue_x).reshape(-1)
    by = D._np(core.blue_y).reshape(-1)
    rx = D._np(core.red_x).reshape(-1)
    ry = D._np(core.red_y).reshape(-1)
    bf = D._np(core.blue_flag_pos).reshape(-1)
    rf = D._np(core.red_flag_pos).reshape(-1)
    carrying = D._np(core.blue_carrying).reshape(-1).astype(bool)
    out = []
    for i in range(4):
        d_enemy = float(np.hypot(bx[i] - rf[0], by[i] - rf[1]))
        d_own = float(np.hypot(bx[i] - bf[0], by[i] - bf[1]))
        n_near = int(np.sum(np.hypot(rx - bx[i], ry - by[i]) <= PRESSURE_RADIUS))
        out.append(
            {
                "carrying": bool(carrying[i]),
                "near_enemy_flag": d_enemy < DEFEND_RADIUS,
                "near_own_flag": d_own < DEFEND_RADIUS,
                "pressured": n_near >= 2,
                "d_enemy": d_enemy,
            }
        )
    return out


def _attach_conditioning(obs, core, rt: PolicyRuntime, *, force: bool):
    from rl.custom_ppo.rule_role_assignment import RoleHoldState, roles_from_core
    from rl.custom_ppo.guard_assignment import assignment_from_core

    out = dict(obs)
    if rt.role_hold is not None:
        roles = roles_from_core(core, rt.role_hold, force=force, advance_age=True)
        out["roles"] = roles.detach().cpu().numpy().astype(np.float32)
    if rt.assignment_hold is not None:
        feat = assignment_from_core(core, rt.assignment_hold, force=force, advance_age=True)
        out["assignment"] = feat.detach().cpu().numpy().astype(np.float32)
    return out


def _critic_value(rt: PolicyRuntime, obs: dict, device: str) -> float:
    gs = np.asarray(obs["global_state"], dtype=np.float32)
    if gs.ndim == 1:
        gs = gs[None, :]
    gs_t = torch.as_tensor(gs, dtype=torch.float32, device=device)
    roles_t = None
    assign_t = None
    if "roles" in obs:
        r = np.asarray(obs["roles"], dtype=np.float32)
        if r.ndim == 1:
            r = r[None, :]
        roles_t = torch.as_tensor(r, dtype=torch.float32, device=device)
    if "assignment" in obs:
        a = np.asarray(obs["assignment"], dtype=np.float32)
        if a.ndim == 2:
            a = a[None, ...]
        assign_t = torch.as_tensor(a, dtype=torch.float32, device=device)
    with torch.no_grad():
        v_norm = rt.inference.model.values(gs_t, team_roles=roles_t, team_assignment=assign_t)
        v = rt.return_norm.denormalize(v_norm)
    return float(v.reshape(-1)[0].cpu().item())


def _mean_ci(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "lcb95": float("nan"), "ucb95": float("nan")}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(values.mean()), "lcb95": float(lo), "ucb95": float(hi)}


def run_episode(rt: PolicyRuntime, seed: int, genome, device: str) -> tuple[dict, list[dict]]:
    from gpu_env._core._entity_obs import augment_obs_with_entities
    import experiments.r2_learned_crossover as R2

    env, core, obs = D._build_env(device, seed, "B", genome)
    try:
        rt.inference.reset_strategy()
        if rt.role_hold is not None:
            rt.role_hold.reset_envs(torch.ones(1, dtype=torch.bool, device=device))
        if rt.assignment_hold is not None:
            rt.assignment_hold.reset_envs(torch.ones(1, dtype=torch.bool, device=device))
        obs = augment_obs_with_entities(obs, core, side="blue")
        obs = _attach_conditioning(obs, core, rt, force=True)

        team_rewards: list[float] = []
        values_pre: list[float] = []
        terminated_flags: list[bool] = []
        truncated_flags: list[bool] = []
        step_traces: list[dict] = []
        terminal_scores = None
        post_done_value = None

        for t in range(R2.MAX_STEPS):
            st = D._snapshot(core)
            geom_pre = _per_agent_geom(core)
            zi_keys = _privileged_zi_keys(core)
            v_pre = _critic_value(rt, obs, device)
            action, _ = rt.inference.predict(obs, deterministic=True)
            flat = np.asarray(action).reshape(-1)
            macros = [int(flat[2 * i]) for i in range(4)]
            targets = [int(flat[2 * i + 1]) for i in range(4)]

            env.step_async(action)
            obs_next, r, done, info = env.step_wait()
            obs_next["global_state"] = env.state()
            obs_next = augment_obs_with_entities(obs_next, core, side="blue")
            obs_next = _attach_conditioning(obs_next, core, rt, force=False)

            inf = _info0(info)
            r_team = float(np.asarray(r).reshape(-1)[0])
            term = bool(inf.get("terminated", False))
            trunc = bool(inf.get("truncated", False))
            team_rewards.append(r_team)
            values_pre.append(v_pre)
            terminated_flags.append(term)
            truncated_flags.append(trunc)

            geom_post = _per_agent_geom(core)
            for i in range(4):
                fp_delta = geom_pre[i]["d_enemy"] - geom_post[i]["d_enemy"]
                m = macros[i]
                bucket = _macro_bucket(m, geom_pre[i]["near_own_flag"])
                useful = (not geom_pre[i]["carrying"]) and (
                    m == 2 or (m == 0 and fp_delta > 0)
                )
                idle = (not geom_pre[i]["carrying"]) and (
                    m == 4 or (m == 0 and fp_delta <= 0)
                )
                step_traces.append(
                    {
                        "policy": rt.label,
                        "seed": seed,
                        "t": t,
                        "agent_i": i,
                        "macro": MACRO_NAMES.get(m, str(m)),
                        "target": targets[i],
                        "macro_bucket": bucket,
                        "team_reward": r_team,
                        "reward_sparse": float(inf.get("reward_sparse", 0.0) or 0.0),
                        "reward_offense": float(inf.get("reward_offense", 0.0) or 0.0),
                        "reward_pbrs": float(inf.get("reward_pbrs", 0.0) or 0.0),
                        "reward_team": float(inf.get("reward_team", 0.0) or 0.0),
                        "reward_failure": float(inf.get("reward_failure", 0.0) or 0.0),
                        "reward_terminal": float(inf.get("reward_terminal", 0.0) or 0.0),
                        "value_pre_step": v_pre,
                        "carrying": int(geom_pre[i]["carrying"]),
                        "near_enemy_flag": int(geom_pre[i]["near_enemy_flag"]),
                        "near_own_flag": int(geom_pre[i]["near_own_flag"]),
                        "pressured": int(geom_pre[i]["pressured"]),
                        "mean_d_enemy_flag": geom_pre[i]["d_enemy"],
                        "flag_progress_delta": fp_delta,
                        "privileged_zi_key": zi_keys[i],
                        "useful_B": int(useful),
                        "idle_retreat_B": int(idle),
                        "blue_score": st["blue_score"],
                        "red_score": st["red_score"],
                    }
                )

            obs = obs_next
            if bool(np.asarray(done).any()):
                res = inf.get("episode_result") or {}
                terminal_scores = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                post_done_value = _critic_value(rt, obs, device)
                break
        else:
            st_end = D._snapshot(core)
            terminal_scores = (st_end["blue_score"], st_end["red_score"])
            post_done_value = _critic_value(rt, obs, device)
            truncated_flags[-1] = True

        T = len(team_rewards)
        if T == 0:
            raise RuntimeError(f"empty episode policy={rt.label} seed={seed}")

        next_values = np.zeros(T, dtype=np.float64)
        for ti in range(T - 1):
            next_values[ti] = values_pre[ti + 1]
        if terminated_flags[-1]:
            next_values[-1] = 0.0
        else:
            next_values[-1] = float(post_done_value or 0.0)

        from rl.ppo_core import compute_gae

        rew_t = torch.tensor(team_rewards, dtype=torch.float32).unsqueeze(1)
        val_t = torch.tensor(values_pre, dtype=torch.float32).unsqueeze(1)
        nxt_t = torch.tensor(next_values, dtype=torch.float32).unsqueeze(1)
        term_t = torch.tensor(terminated_flags, dtype=torch.bool).unsqueeze(1)
        trunc_t = torch.tensor(truncated_flags, dtype=torch.bool).unsqueeze(1)
        adv_t, ret_t = compute_gae(
            rew_t,
            val_t,
            nxt_t,
            term_t,
            trunc_t,
            gamma=float(rt.gamma),
            gae_lambda=float(rt.gae_lambda),
        )
        adv = adv_t.reshape(-1).cpu().numpy()
        ret = ret_t.reshape(-1).cpu().numpy()
        for ti in range(T):
            for row in step_traces:
                if row["t"] == ti:
                    row["advantage_team"] = float(adv[ti])
                    row["return_team"] = float(ret[ti])

        blue_s, red_s = terminal_scores or (0, 0)
        ep = {
            "policy": rt.label,
            "seed": seed,
            "steps": T,
            "episode_return": float(sum(team_rewards)),
            "won": int(blue_s > red_s),
            "blue_score": blue_s,
            "red_score": red_s,
        }
        return ep, step_traces
    finally:
        env.close()


def _summarize_policy(ep_rows: list[dict], step_rows: list[dict]) -> dict:
    out: dict = {
        "mean_episode_return": float(np.mean([e["episode_return"] for e in ep_rows])),
        "win_rate_pole_B": float(np.mean([e["won"] for e in ep_rows])),
        "n_episodes": len(ep_rows),
        "n_agent_steps": len(step_rows),
    }
    macro_counts = Counter(r["macro_bucket"] for r in step_rows)
    total = max(1, sum(macro_counts.values()))
    out["macro_fraction"] = {k: macro_counts[k] / total for k in sorted(macro_counts)}
    bucket_adv: dict[str, list[float]] = defaultdict(list)
    for r in step_rows:
        bucket_adv[r["macro_bucket"]].append(float(r["advantage_team"]))
    out["E_advantage_given_bucket"] = {
        k: float(np.mean(v)) for k, v in sorted(bucket_adv.items())
    }
    fp = np.array([r["flag_progress_delta"] for r in step_rows], dtype=np.float64)
    av = np.array([r["advantage_team"] for r in step_rows], dtype=np.float64)
    if fp.std() > 1e-9 and av.std() > 1e-9:
        out["corr_advantage_flag_progress_delta"] = float(np.corrcoef(fp, av)[0, 1])
    else:
        out["corr_advantage_flag_progress_delta"] = float("nan")

    useful_adv = [r["advantage_team"] for r in step_rows if r["useful_B"]]
    idle_adv = [r["advantage_team"] for r in step_rows if r["idle_retreat_B"]]
    mu_u = float(np.mean(useful_adv)) if useful_adv else float("nan")
    mu_i = float(np.mean(idle_adv)) if idle_adv else float("nan")
    ratio = mu_u / mu_i if idle_adv and abs(mu_i) > 1e-9 else float("nan")
    out["G_signal"] = {
        "mean_adv_useful_B": mu_u,
        "mean_adv_idle_retreat_B": mu_i,
        "difference": mu_u - mu_i if useful_adv and idle_adv else float("nan"),
        "ratio_useful_over_idle": ratio,
        "n_useful_steps": len(useful_adv),
        "n_idle_steps": len(idle_adv),
    }
    # Episode-level bootstrap on per-episode mean advantages (clustered by seed).
    by_seed_u: dict[int, list[float]] = defaultdict(list)
    by_seed_i: dict[int, list[float]] = defaultdict(list)
    for r in step_rows:
        if r["useful_B"]:
            by_seed_u[r["seed"]].append(float(r["advantage_team"]))
        if r["idle_retreat_B"]:
            by_seed_i[r["seed"]].append(float(r["advantage_team"]))
    ep_u = np.array([np.mean(v) for v in by_seed_u.values()], dtype=np.float64)
    ep_i = np.array([np.mean(v) for v in by_seed_i.values()], dtype=np.float64)
    if ep_u.size and ep_i.size:
        rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
        n = min(len(ep_u), len(ep_i))
        idx = rng.integers(0, n, size=(N_BOOT, n))
        boot_ratio = ep_u[idx].mean(axis=1) / np.maximum(ep_i[idx].mean(axis=1), 1e-9)
        lo, hi = np.percentile(boot_ratio, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        out["G_signal"]["ratio_ci95"] = {
            "mean": float(np.mean(boot_ratio)),
            "lcb95": float(lo),
            "ucb95": float(hi),
        }
    return out


def _build_policy_runtimes(obs_space, act_space, device: str) -> dict[str, PolicyRuntime]:
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.custom_ppo.rule_role_assignment import RoleHoldState
    from rl.custom_ppo.guard_assignment import AssignmentHoldState

    runtimes: dict[str, PolicyRuntime] = {}
    for label, path in POLICIES:
        payload = read_checkpoint_payload(str(path), map_location="cpu")
        cfg = payload.get("cfg") or {}
        inf = load_custom_ppo_policy(str(path), obs_space, act_space, device=device)
        inf.model.eval()
        role_hold = None
        assignment_hold = None
        if bool(getattr(inf.model, "role_conditioning_enabled", False)):
            ht = int(getattr(inf.model, "role_hold_ticks", 0) or 0) or 8
            role_hold = RoleHoldState(1, int(inf.model.n_agents), hold_ticks=ht, device=device)
        if bool(getattr(inf.model, "assignment_conditioning_enabled", False)):
            ht = int(getattr(inf.model, "assignment_hold_ticks", 0) or 0) or 8
            assignment_hold = AssignmentHoldState(
                1, int(inf.model.n_agents), hold_ticks=ht, device=device
            )
        runtimes[label] = PolicyRuntime(
            label=label,
            path=path,
            inference=inf,
            gamma=float(cfg.get("gamma", 0.99) or 0.99),
            gae_lambda=float(cfg.get("gae_lambda", 0.95) or 0.95),
            return_norm=_load_return_norm(payload),
            role_hold=role_hold,
            assignment_hold=assignment_hold,
        )
    return runtimes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from experiments.pole_attestation import assert_resolved_matches_certification, governing_certification, resolve_pole_genome
    from experiments.run_lock import RunLock
    from experiments.tqdm_loop import set_postfix, tqdm_iter

    if not SPEC.is_file():
        raise SystemExit(f"REFUSING: frozen spec missing: {SPEC}")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")

    out_path = SD / f"{LABEL}_RESULT.json"
    step_csv = SD / "pole_b_credit_advantage_diagnostic_step_rows.csv"
    ep_csv = SD / "pole_b_credit_advantage_diagnostic_episode_rows.csv"
    if not args.dry_run:
        import experiments.seed_registry as R

        doc = R.load()
        b = next((x for x in doc["blocks"] if x["experiment_id"] == EXPERIMENT_ID), None)
        if b is None or b["lo"] != SEED_LO or b["hi"] != SEED_LO + N_SEEDS - 1:
            raise SystemExit(
                f"FAIL-CLOSED (Rule 9): seed block for {EXPERIMENT_ID!r} not "
                f"registered as {SEED_LO}..{SEED_LO + N_SEEDS - 1}"
            )
        for p in (out_path, step_csv, ep_csv):
            if p.is_file():
                raise SystemExit(f"REFUSING: output already exists (one-shot): {p}")

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    seeds = list(range(SEED_LO, SEED_LO + N_SEEDS))
    _v, cert_path = governing_certification(4)

    print(f"{LABEL}  {_now()}  device={device}")
    print(f"  spec     {SPEC.name} [{spec.get('status')}]")
    print(f"  seeds    {seeds[0]}..{seeds[-1]} (n={len(seeds)}), SHARED across all policies")

    genome = resolve_pole_genome("B", 4, str(D.POLE_B_GENOME))
    attestation = assert_resolved_matches_certification("B", 4, cert_path, genome, is_smoke=False)
    print(
        f"  pole B   {attestation['live_genome_id']} "
        f"MATCH={'PASS' if attestation['hashes_match'] else 'FAIL'}"
    )

    for lbl, p in POLICIES:
        if not p.is_file():
            raise SystemExit(f"REFUSING: missing checkpoint {lbl}: {p}")
        print(f"  {lbl:14s} sha {_sha(p)[:12]}...")

    env, _c, _o = D._build_env(device, seeds[0], "B", genome)
    obs_space, act_space = env.observation_space, env.action_space
    env.close()

    runtimes = _build_policy_runtimes(obs_space, act_space, device)
    shas = {lbl: _sha(p) for lbl, p in POLICIES}
    contracts = {
        "K1_five_distinct_checkpoints": len(set(shas.values())) == 5,
        "K2_all_entity_encoder": all(
            rt.inference.model.entity_encoder is not None for rt in runtimes.values()
        ),
        "K3_pole_attested": attestation["hashes_match"],
        "K4_assignment_v1_has_assignment_hold": runtimes["Assignment_v1"].assignment_hold is not None,
    }
    print("\n  known-answer contracts ...")
    for k, v in contracts.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    if not all(contracts.values()):
        raise SystemExit(f"FAIL-CLOSED: contracts failed {[k for k, v in contracts.items() if not v]}")

    if args.dry_run:
        print("\n  --dry-run: spec frozen, checkpoints present, pole attested, contracts OK.")
        print("  NO episodes run, NOTHING written.")
        return 0

    step_fields = [
        "policy", "seed", "t", "agent_i", "macro", "target", "macro_bucket",
        "team_reward", "reward_sparse", "reward_offense", "reward_pbrs", "reward_team",
        "reward_failure", "reward_terminal", "value_pre_step", "advantage_team", "return_team",
        "carrying", "near_enemy_flag", "near_own_flag", "pressured", "mean_d_enemy_flag",
        "flag_progress_delta", "privileged_zi_key", "useful_B", "idle_retreat_B",
        "blue_score", "red_score",
    ]
    ep_fields = ["policy", "seed", "steps", "episode_return", "won", "blue_score", "red_score"]

    summaries: dict[str, dict] = {}
    all_ep: list[dict] = []

    with RunLock(SD / f"{LABEL}.run.lock", run_id=LABEL):
        with step_csv.open("w", newline="", encoding="utf-8") as sf, ep_csv.open(
            "w", newline="", encoding="utf-8"
        ) as ef:
            sw = csv.DictWriter(sf, fieldnames=step_fields, extrasaction="ignore")
            ew = csv.DictWriter(ef, fieldnames=ep_fields, extrasaction="ignore")
            sw.writeheader()
            ew.writeheader()
            for label, _path in POLICIES:
                rt = runtimes[label]
                ep_rows: list[dict] = []
                step_rows: list[dict] = []
                bar = tqdm_iter(seeds, desc=f"{LABEL} {label}", unit="ep")
                for seed in bar:
                    set_postfix(bar, f"seed={seed}")
                    ep, steps = run_episode(rt, seed, genome, device)
                    ep_rows.append(ep)
                    all_ep.append(ep)
                    ew.writerow(ep)
                    for row in steps:
                        sw.writerow(row)
                        step_rows.append(row)
                    ef.flush()
                    sf.flush()
                    summaries[label] = _summarize_policy(ep_rows, step_rows)
                    out_path.write_text(
                        json.dumps(
                            {
                                "record": LABEL,
                                "status": "RUNNING",
                                "utc": _now(),
                                "summaries_partial": summaries,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                s = summaries[label]
                print(
                    f"\n  {label}: win={s['win_rate_pole_B']:.3f} "
                    f"ret={s['mean_episode_return']:.3f} "
                    f"G_signal_ratio={s['G_signal'].get('ratio_useful_over_idle', float('nan')):.3f}",
                    flush=True,
                )

    gf = summaries.get("GETFLAG", {}).get("macro_fraction", {})
    b5 = summaries.get("B_t500k", {}).get("macro_fraction", {})
    bf = summaries.get("B_final", {}).get("macro_fraction", {})
    g_retention = {
        "GET_FLAG_mass": {
            "B_t500k": b5.get("GET_FLAG", float("nan")),
            "B_final": bf.get("GET_FLAG", float("nan")),
            "GETFLAG": gf.get("GET_FLAG", float("nan")),
            "Assignment_v1": summaries.get("Assignment_v1", {}).get("macro_fraction", {}).get(
                "GET_FLAG", float("nan")
            ),
        }
    }

    out_path.write_text(
        json.dumps(
            {
                "record": f"{LABEL} Pole-B credit/advantage diagnostic",
                "status": "COMPLETE_DIAGNOSTIC",
                "utc": _now(),
                "device": device,
                "arm": "DIAGNOSTIC",
                "confirmatory": False,
                "implements": SPEC.name,
                "decision_tree": "4V4_POST_GETFLAG_DECISION_TREE.json steps 2-3",
                "question": spec.get("THE_QUESTION"),
                "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds), "shared_across_policies": True},
                "checkpoints_sha256_prefix": {k: v[:12] for k, v in shas.items()},
                "contracts": contracts,
                "pole_attestation": {
                    k: attestation[k]
                    for k in (
                        "certified_genome_id",
                        "live_genome_id",
                        "certified_config_hash",
                        "live_config_hash",
                        "hashes_match",
                    )
                },
                "summaries": summaries,
                "G_retention": g_retention,
                "advantage_caveats": spec.get("ADVANTAGE_CONTRACT"),
                "NOT_A_CLAIM": spec.get("FORBIDDEN"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  -> {out_path}\n  -> {step_csv}\n  -> {ep_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
