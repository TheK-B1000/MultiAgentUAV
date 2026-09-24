"""Read-only probe: what composition do the existing learned 6v6 specialists behave like?

DIAGNOSTIC. No training, no parameter change, no outcome-based selection. It loads
the two existing specialist checkpoints, runs them on the certified 6v6 poles, and
records per-tick state so that an instrument can infer an EFFECTIVE composition.

Why an instrument is needed. The scripted composition sweeps assigned roles by
construction and their defenders used macro 7 (DEFEND), which is evaluation-only.
The learned policies' production action space has 5 macros (GO_TO, GRAB_MINE,
GET_FLAG, PLACE_MINE, GO_HOME) and CANNOT select DEFEND. A learned agent can only
defend by standing near home. So its role has to be inferred from behaviour.

The instrument is calibrated on SCRIPTED episodes, where the true composition is
known, using the same environment and the same code path. Learned readings are
interpretable only to the extent the instrument recovers the scripted truth.

Two env configs are used, and they are the SAME GAME: n_macros only sizes the
observation mask and the action decode. Scripted defenders need n_macros=8 (macro 7
in a 5-macro env silently aliases to GET_FLAG via 7 mod 5 = 2); the learned policies
were trained at n_macros=5. Scripted 6A/0D uses only macros valid in both and its
positions were verified bit-identical across the two.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_actions import MacroAction  # noqa: E402

N_AGENTS = 6
BASE_KEY = {"A": "OP6", "B": "OP7"}
POLES = ("A", "B")
LEARNED = ("pi_A", "pi_B")
CHECKPOINTS = {
    "pi_A": ROOT / "artifacts" / "6v6_results" / "specialists" / "final_pi_A_specialist_6v6.zip",
    "pi_B": ROOT / "artifacts" / "6v6_results" / "specialists" / "final_pi_B_specialist_6v6.zip",
}
SHARE0_RESULT = ROOT / "artifacts" / "strategic_demand" / "sppo" / "SHARE0_6V6_TEACHER_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
SHARE0_ROWS = ROOT / "artifacts" / "strategic_demand" / "sppo" / "share0_6v6_teacher_specialist_crossover_eval_rows.csv"


# ------------------------------------------------------------------ env / setup

def build_probe_env(device: str, seed: int, n_macros: int):
    """R2.build_env with the ONE change the scripted calibration needs (n_macros).
    Every other field, including the deliberately unset obstacle_obs_channel, matches
    the learned evaluation exactly."""
    import experiments.r2_learned_crossover as R2
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    R2.AGENTS = N_AGENTS
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=N_AGENTS, max_red_agents=N_AGENTS, map_set="train",
        map_layout=R2.MAP, max_decision_steps=R2.MAX_STEPS, aquaticus_profile=True,
        rules_profile="OURS", device=device, seed=int(seed), tag_telemetry_enabled=True,
        own_flag_home_required_to_score=True, n_macros=int(n_macros), **R2.RULESET)
    return GPUCTFVecEnv(cfg)


def pole_genomes() -> dict[str, dict]:
    from experiments.opponent_spec import pole_A_genome
    from experiments.pole_attestation import resolve_pole_genome
    return {"A": {"OP6": pole_A_genome(N_AGENTS)}, "B": {"OP7": resolve_pole_genome("B", N_AGENTS, None)}}


def setup_episode(env, pole: str, genomes: dict[str, dict]):
    """Identical opponent installation to eval_specialist_crossover_scaled.run_cell."""
    from experiments.opponent_spec import assert_live_opponent_batch, install_keyed_opponent_overlays
    from gpu_env._core._entity_obs import augment_obs_with_entities
    from rl.curriculum import phase_from_tag
    core = env.core
    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0
    install_keyed_opponent_overlays(core, genomes[pole])
    key = BASE_KEY[pole]
    env.env_method("set_phase", phase_from_tag(key))
    env.env_method("set_next_opponent", "SCRIPTED", key)
    obs = env.reset()
    obs["global_state"] = env.state()
    obs = augment_obs_with_entities(obs, core, side="blue")
    assert_live_opponent_batch(core, genomes[pole], allowed_keys=(key,), context="composition probe")
    got = core._bt_resolved_profile_tensors().get("min_alive_for_defender")
    got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
    if got_val != N_AGENTS:
        raise SystemExit(f"FAIL-CLOSED: live pole {pole} resolves min_alive_for_defender={got_val}, expected {N_AGENTS}")
    return obs


# --------------------------------------------------------------------- recording

def resolved_intent(core, action) -> tuple[np.ndarray, np.ndarray]:
    """Where the env is actually sending each blue agent this tick, READ-ONLY.

    Reproduces the pipeline in gpu_env/_core/_step.py without calling it: a macro is
    COMMITTED for a few ticks, so the effective macro/target is the committed one while
    commit_ticks_left > 0 and the requested one otherwise; the env then resolves it to a
    target position with _build_targets_from_action (GET_FLAG -> enemy flag, GO_HOME ->
    own home, DEFEND -> a state-derived point near own flag, GO_TO -> the decoded target,
    and any carrier -> home). Nothing here mutates the core.

    This is the common currency between a scripted DEFEND and a learned GO_TO: both end
    up as a position the agent is being sent to.
    """
    import torch
    a = torch.as_tensor(np.asarray(action).reshape(1, N_AGENTS, 2), dtype=torch.long, device=core.device)
    req_m = torch.remainder(a[..., 0], int(core.cfg.n_macros))
    req_t = torch.remainder(a[..., 1], int(core.cfg.n_targets))
    fresh = core.blue_commit_ticks_left <= 0
    m = torch.where(fresh, req_m, core.blue_commit_macro)
    tg = torch.where(fresh, req_t, core.blue_commit_target)
    tx, ty = core._build_targets_from_action(m, tg, side="blue")
    xy = torch.stack([tx, ty], dim=-1)[0].detach().cpu().numpy().astype(np.float32)
    return xy, m[0].detach().cpu().numpy().astype(np.int16)


def _snap(core) -> dict[str, np.ndarray]:
    def a(x, dtype=None):
        arr = x.detach().cpu().numpy()[0]
        return arr.astype(dtype) if dtype is not None else arr
    return {
        "pos": a(core.blue_pos, np.float32),
        "alive": a(core.blue_alive, bool),
        "tagged": a(core.blue_tagged, bool),
        "carrying": a(core.blue_carrying, bool),
        "flag_pos": a(core.blue_flag_pos, np.float32),
        "flag_home": a(core.blue_flag_home, np.float32),
        "red_carrying": a(core.red_carrying, bool) if hasattr(core, "red_carrying") else np.zeros(N_AGENTS, bool),
    }


def _finish(ticks: list[dict], actions: list[np.ndarray], true_defend: np.ndarray | None,
            blue: int, red: int) -> dict[str, Any]:
    T = len(ticks)
    out = {k: np.stack([t[k] for t in ticks]) for k in ticks[0]}
    out["action"] = np.stack(actions).astype(np.int16)             # (T, N, 2) macro, target
    out["true_defend"] = (np.zeros(N_AGENTS, bool) if true_defend is None else true_defend.astype(bool))
    out.update(steps=T, blue=int(blue), red=int(red), win=int(blue > red), margin=int(blue - red))
    return out


def run_learned_episode(policy, pole: str, seed: int, genomes: dict[str, dict], device: str,
                        max_ticks: int | None = None) -> dict[str, Any]:
    """Mirrors eval_specialist_crossover_scaled.run_cell action-for-action, plus a
    pre-step state snapshot. The learned env is the production n_macros=5 config."""
    import experiments.r2_learned_crossover as R2
    from gpu_env._core._entity_obs import augment_obs_with_entities
    env = build_probe_env(device, seed, n_macros=5)
    core = env.core
    try:
        policy.reset_strategy()
        obs = setup_episode(env, pole, genomes)
        ticks, actions = [], []
        terminal = None
        horizon = R2.MAX_STEPS if max_ticks is None else min(int(max_ticks), R2.MAX_STEPS)
        for _ in range(horizon):
            action, _ = policy.predict(obs, deterministic=True)
            snap = _snap(core)
            snap["intent"], snap["eff_macro"] = resolved_intent(core, action)
            ticks.append(snap)
            actions.append(np.asarray(action).reshape(N_AGENTS, 2).copy())
            env.step_async(action)
            obs, _r, done, info = env.step_wait()
            obs["global_state"] = env.state()
            obs = augment_obs_with_entities(obs, core, side="blue")
            if bool(np.asarray(done).any()):
                res = (info[0] if isinstance(info, (list, tuple)) else info)
                res = (res or {}).get("episode_result") or {}
                terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                break
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        return _finish(ticks, actions, None, *terminal)
    finally:
        env.close()


def run_scripted_episode(composition: str, pole: str, seed: int, genomes: dict[str, dict], device: str,
                         max_ticks: int | None = None, roles: tuple[int, ...] | None = None) -> dict[str, Any]:
    """A fixed scripted composition in the SAME env family (n_macros=8, required for the
    DEFEND macro). true_defend records the ground-truth role of each agent."""
    import experiments.r2_learned_crossover as R2
    from experiments.run_pyquaticus_6v6_role_composition_sweep import action_for_roles_n, composition_roles_n
    env = build_probe_env(device, seed, n_macros=8)
    core = env.core
    try:
        setup_episode(env, pole, genomes)
        if roles is None:
            roles = composition_roles_n(composition, N_AGENTS)
        ticks, actions = [], []
        terminal = None
        horizon = R2.MAX_STEPS if max_ticks is None else min(int(max_ticks), R2.MAX_STEPS)
        for _ in range(horizon):
            action = action_for_roles_n(core, roles)
            snap = _snap(core)
            snap["intent"], snap["eff_macro"] = resolved_intent(core, action)
            ticks.append(snap)
            actions.append(action.reshape(N_AGENTS, 2).copy())
            env.step_async(action)
            _o, _r, done, info = env.step_wait()
            if bool(np.asarray(done).any()):
                res = (info[0] if isinstance(info, (list, tuple)) else info)
                res = (res or {}).get("episode_result") or {}
                terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                break
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        return _finish(ticks, actions, np.asarray(roles) == 1, *terminal)
    finally:
        env.close()


# ------------------------------------------------------- forced DEFEND injection
# CAUSAL_BRIDGE_DEFENDER_INJECTION_V1: give a learned rollout exactly one real
# defender without touching the policy's action space at all.
#
# A learned policy's action head has n_macros=5 (GO_TO, GRAB_MINE, GET_FLAG,
# PLACE_MINE, GO_HOME) and can never emit MacroAction.DEFEND=7: the env's own
# decode is `torch.remainder(raw_macro, cfg.n_macros)`, applied unconditionally
# to whatever integer reaches it, so 7 always aliases to 2 (GET_FLAG) at
# n_macros=5. Widening the env to n_macros=8 is not an option either: that
# changes the observation's macro-mask size, which the checkpoint was trained
# against and would silently corrupt every agent's inference.
#
# Writing macro id 7 into blue_commit_macro directly (bypassing the decode, as
# MacroAction's own docstring describes: "reachable only by a caller that
# builds a macro-id tensor directly") was tried first and is WRONG at
# n_macros=5: gpu_env/_core/_observations.py::_build_action_mask builds each
# agent's action mask with `macro_mask.scatter_(2, commit_macro.unsqueeze(-1),
# 1.0)` into a dimension sized cfg.n_macros -- a one-hot over 5 slots has no
# representation for index 7, and CUDA hard-crashes (device-side assert,
# index out of bounds) the instant that scatter runs, every tick, because a
# permanently-committed agent (commit_ticks_left > 0) hits this branch
# unconditionally. Confirmed live: identical scripted episode via
# run_scripted_episode_with_forced_override reproduces the target/position
# trace exactly at n_macros=8 and crashes at n_macros=5 with exactly this
# stack.
#
# The correct injection point -- and what "the resolved-target/control layer"
# in the frozen spec means -- is one layer downstream of the macro id, at
# gpu_env/_core/_rules.py::_build_targets_from_action's OUTPUT (tx, ty), which
# is a continuous float pair with no n_macros-sized representation at all.
# install_forced_defend_target() monkey-patches this one instance's bound
# method: every call still runs the real, untouched implementation first
# (whatever the agent's own, safely-representable committed macro is), then
# calls that SAME real implementation a second time with a synthetic
# all-DEFEND macro tensor -- exactly the read-only, no-core-mutation pattern
# resolved_intent() already uses above -- purely to read what DEFEND's
# live-state target would be for the overridden agent(s), and splices only
# that slice into the real (tx, ty). DEFEND's target math is never
# reimplemented; commit_macro itself is never touched, so the action-mask
# scatter, and the unrelated action_success/action_failed_punishment reward
# bookkeeping (which does not enumerate DEFEND at all -- see _step.py), see
# only the agent's real, small, always-valid committed macro. Verified against
# the game's OWN state machine (gpu_env/_core/_step.py::_advance_blue_macros):
# blue_commit_macro is read in exactly three places (the two above, plus
# movement resolution itself, which is equality-based on the macro value and
# has no indexing constraint), so leaving it untouched has no other effect.

def install_forced_defend_target(core, agent_id: int, *, side: str = "blue") -> None:
    """Idempotent per-core-instance patch (safe to call once per overridden
    agent per episode; does not affect any other env/core object)."""
    if getattr(core, "_forced_defend_targets", None) is None:
        core._forced_defend_targets: dict[str, set[int]] = {}
        real = core._build_targets_from_action

        def patched(macro, targ, side="blue"):
            tx, ty = real(macro, targ, side=side)
            ids = core._forced_defend_targets.get(side)
            if ids:
                defend_macro = macro.clone()
                for i in ids:
                    defend_macro[0, i] = int(MacroAction.DEFEND)
                dtx, dty = real(defend_macro, targ, side=side)
                tx, ty = tx.clone(), ty.clone()
                for i in ids:
                    tx[0, i] = dtx[0, i]
                    ty[0, i] = dty[0, i]
            return tx, ty

        core._build_targets_from_action = patched
    core._forced_defend_targets.setdefault(side, set()).add(int(agent_id))


def defender_id_for_seed(seed: int, n_agents: int = N_AGENTS) -> int:
    """Deterministic seed -> agent-id rotation, so the overridden slot is not a
    fixed spawn position. `seed % n_agents` cycles ids 0..n_agents-1 exactly
    uniformly over any seed block whose length is a multiple of n_agents."""
    return int(seed) % int(n_agents)


def run_learned_episode_with_forced_defender(policy, pole: str, seed: int, genomes: dict[str, dict],
                                             device: str, defender_id: int,
                                             max_ticks: int | None = None) -> dict[str, Any]:
    """Identical to run_learned_episode, except agent `defender_id`'s resolved
    target is DEFEND's for the whole episode (install_forced_defend_target).
    The other five agents' actions come from the policy exactly as in
    run_learned_episode and are never touched; the overridden agent's own
    policy-predicted action is still computed (so the network's OTHER five
    outputs are unaffected) but its own macro/target choice is moot -- its
    target is unconditionally replaced before movement."""
    import experiments.r2_learned_crossover as R2
    from gpu_env._core._entity_obs import augment_obs_with_entities
    env = build_probe_env(device, seed, n_macros=5)
    core = env.core
    try:
        policy.reset_strategy()
        obs = setup_episode(env, pole, genomes)
        install_forced_defend_target(core, defender_id)
        ticks, actions = [], []
        terminal = None
        horizon = R2.MAX_STEPS if max_ticks is None else min(int(max_ticks), R2.MAX_STEPS)
        for _ in range(horizon):
            action, _ = policy.predict(obs, deterministic=True)
            snap = _snap(core)
            snap["intent"], snap["eff_macro"] = resolved_intent(core, action)
            snap["eff_macro"][defender_id] = int(MacroAction.DEFEND)  # telemetry only; see install_forced_defend_target
            ticks.append(snap)
            actions.append(np.asarray(action).reshape(N_AGENTS, 2).copy())
            env.step_async(action)
            obs, _r, done, info = env.step_wait()
            obs["global_state"] = env.state()
            obs = augment_obs_with_entities(obs, core, side="blue")
            if bool(np.asarray(done).any()):
                res = (info[0] if isinstance(info, (list, tuple)) else info)
                res = (res or {}).get("episode_result") or {}
                terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                break
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        return _finish(ticks, actions, None, *terminal)
    finally:
        env.close()


def run_scripted_episode_with_forced_override(composition: str, pole: str, seed: int, genomes: dict[str, dict],
                                              device: str, override_id: int, *, n_macros: int,
                                              override_raw_macro: int = int(MacroAction.GO_TO),
                                              max_ticks: int | None = None,
                                              roles: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Contract-test fixture ONLY: a scripted composition where agent `override_id`
    is fed `override_raw_macro` in the raw action array (deliberately NOT DEFEND)
    but has its resolved target forced to DEFEND every tick regardless
    (install_forced_defend_target). Used to prove the injection reproduces a
    natural scripted DEFEND bit-for-bit, at any n_macros including 5 (the
    production learned-env value), never to generate reported evidence."""
    import experiments.r2_learned_crossover as R2
    from experiments.run_pyquaticus_6v6_role_composition_sweep import action_for_roles_n, composition_roles_n
    env = build_probe_env(device, seed, n_macros=n_macros)
    core = env.core
    try:
        setup_episode(env, pole, genomes)
        if roles is None:
            roles = composition_roles_n(composition, N_AGENTS)
        install_forced_defend_target(core, override_id)
        ticks, actions = [], []
        terminal = None
        horizon = R2.MAX_STEPS if max_ticks is None else min(int(max_ticks), R2.MAX_STEPS)
        for _ in range(horizon):
            action = action_for_roles_n(core, roles)
            action[0, override_id, 0] = int(override_raw_macro)
            action[0, override_id, 1] = 0
            snap = _snap(core)
            snap["intent"], snap["eff_macro"] = resolved_intent(core, action)
            ticks.append(snap)
            actions.append(action.reshape(N_AGENTS, 2).copy())
            env.step_async(action)
            _o, _r, done, info = env.step_wait()
            if bool(np.asarray(done).any()):
                res = (info[0] if isinstance(info, (list, tuple)) else info)
                res = (res or {}).get("episode_result") or {}
                terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                break
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        true_defend = np.asarray(roles) == 1
        true_defend[override_id] = True
        return _finish(ticks, actions, true_defend, *terminal)
    finally:
        env.close()


def load_policies(device: str) -> dict[str, Any]:
    from rl.custom_ppo import load_custom_ppo_policy
    probe = build_probe_env(device, 99_900_640, n_macros=5)
    osp, asp = probe.observation_space, probe.action_space
    probe.close()
    pols = {n: load_custom_ppo_policy(str(p), osp, asp, device=device) for n, p in CHECKPOINTS.items()}
    for n, pol in pols.items():
        if getattr(pol.model, "uses_latent_strategy", False):
            raise SystemExit(f"REFUSING: {n} is latent-conditioned; specialists must be single-strategy")
    return pols


# ------------------------------------------------------------- role assignments

def roles_for(composition: str, assignment: str) -> tuple[int, ...]:
    """Ground-truth defender placement. PREFIX puts the D defenders on agent ids 0..D-1,
    the assignment every scripted sweep used. SUFFIX puts them on the LAST D ids. Agents
    2 and 3 spawn inside the instrument radius, so the two assignments differ in spawn
    geometry while asking for the same composition. Both give the same team of D
    defenders and N-D attackers."""
    d = int(composition.split("_")[1][:-1])
    if assignment == "PREFIX":
        return tuple(1 if i < d else 0 for i in range(N_AGENTS))
    if assignment == "SUFFIX":
        return tuple(1 if i >= N_AGENTS - d else 0 for i in range(N_AGENTS))
    raise ValueError(f"unknown assignment {assignment!r}")


# ---------------------------------------------------------------------- instrument

def home_distance(trace: dict[str, Any]) -> np.ndarray:
    """(T, N) distance of every agent from ITS OWN flag's home position."""
    return np.linalg.norm(trace["pos"] - trace["flag_home"][:, None, :], axis=-1)


def instrument(trace: dict[str, Any], r_home: float) -> dict[str, np.ndarray]:
    """Per-tick defend-like reading.

    An agent is ACTIVE when alive and not tagged (a tagged agent is heading home to
    reset and is neither attacking nor defending). An active agent is DEFEND-LIKE when it
    is not carrying and lies within r_home of its own flag's home; otherwise FORWARD-LIKE.
    Reads only positions and status flags. It never sees an action, a role or a policy.
    """
    d = home_distance(trace)
    active = trace["alive"] & ~trace["tagged"]
    defend = active & ~trace["carrying"] & (d <= r_home)
    n_active = active.sum(axis=1)
    k = defend.sum(axis=1)
    share = np.where(n_active > 0, k / np.maximum(n_active, 1), np.nan)
    return {"defend_like": defend, "active": active, "k": k, "n_active": n_active, "share": share, "d": d}


def instrument_reference(trace: dict[str, Any], r_home: float) -> dict[str, np.ndarray]:
    """A second, deliberately naive implementation of instrument(): plain Python loops
    over ticks and agents, sharing no vectorised code with it. The run re-derives its
    per-episode numbers with THIS one from the saved traces and aborts on disagreement."""
    T = len(trace["pos"])
    k = np.zeros(T, dtype=np.int64)
    n_active = np.zeros(T, dtype=np.int64)
    share = np.full(T, np.nan)
    for t in range(T):
        hx, hy = float(trace["flag_home"][t][0]), float(trace["flag_home"][t][1])
        for i in range(N_AGENTS):
            if not (bool(trace["alive"][t][i]) and not bool(trace["tagged"][t][i])):
                continue
            n_active[t] += 1
            dist = ((float(trace["pos"][t][i][0]) - hx) ** 2 + (float(trace["pos"][t][i][1]) - hy) ** 2) ** 0.5
            if (not bool(trace["carrying"][t][i])) and dist <= r_home:
                k[t] += 1
        if n_active[t] > 0:
            share[t] = k[t] / n_active[t]
    return {"k": k, "n_active": n_active, "share": share}


# ------------------------------------------------- intent instrument (pre-declared)

R_INTENT = 4.5      # declared once, before the intent data were read; not scanned


def intent_instrument(trace: dict[str, Any], r_t: float = R_INTENT) -> dict[str, np.ndarray]:
    """Per-tick HOME-DIRECTED intent reading.

    An agent is HOME-DIRECTED when it is alive, untagged, not carrying, and the env's
    resolved target for it lies within r_t of its own flag's home. Reads the resolved
    target, never the raw macro id and never the position. The per-tick masks let the
    caller slice by game state: a friendly carrier present (the adapter then sends every
    attacker home), a friendly agent tagged, or the phase of the episode.
    """
    d_t = np.linalg.norm(trace["intent"] - trace["flag_home"][:, None, :], axis=-1)
    active = trace["alive"] & ~trace["tagged"]
    home_dir = active & ~trace["carrying"] & (d_t <= r_t)
    n_active = active.sum(axis=1)
    k = home_dir.sum(axis=1)
    share = np.where(n_active > 0, k / np.maximum(n_active, 1), np.nan)
    return {"home_directed": home_dir, "active": active, "k": k, "n_active": n_active, "share": share,
            "any_carry": trace["carrying"].any(axis=1), "any_tagged": trace["tagged"].any(axis=1),
            "d_target": d_t}


def intent_instrument_reference(trace: dict[str, Any], r_t: float = R_INTENT) -> dict[str, np.ndarray]:
    """Naive loop implementation sharing no vectorised code with intent_instrument."""
    T = len(trace["intent"])
    k = np.zeros(T, dtype=np.int64)
    n_active = np.zeros(T, dtype=np.int64)
    share = np.full(T, np.nan)
    for t in range(T):
        hx, hy = float(trace["flag_home"][t][0]), float(trace["flag_home"][t][1])
        for i in range(N_AGENTS):
            if not (bool(trace["alive"][t][i]) and not bool(trace["tagged"][t][i])):
                continue
            n_active[t] += 1
            dist = ((float(trace["intent"][t][i][0]) - hx) ** 2 + (float(trace["intent"][t][i][1]) - hy) ** 2) ** 0.5
            if (not bool(trace["carrying"][t][i])) and dist <= r_t:
                k[t] += 1
        if n_active[t] > 0:
            share[t] = k[t] / n_active[t]
    return {"k": k, "n_active": n_active, "share": share}
