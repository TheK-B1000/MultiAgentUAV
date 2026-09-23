"""Viability ladder for CLOSEST_SPLIT_LEARNED_DEFEND_6V6 on this PC.

25k checkpoint, then a non-claim n=16 screen (k=1 vs k=3, both poles).
Promising -> do not keep training here; the full 200k belongs on the other PC.
Unclear -> continue locally to 50k and screen again.
Obvious collapse -> stop.

Not a confirmatory result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_routed_composition_outcome import _bootstrap  # noqa: E402
from experiments.tqdm_loop import tqdm_iter  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
CKPT_DIR = (
    ROOT / "artifacts" / "exploratory_scale_6v6_specialists"
    / "exploratory_pi_A_specialist_6v6_closest_split_defend_6v6_v1" / "ckpts"
)
RUN_TAG = "exploratory_pi_A_specialist_6v6_closest_split_defend_6v6_v1"
FROZEN_ATTACK = (
    ROOT / "artifacts" / "scale_6v6_specialists" / "pi_A_specialist_6v6"
    / "ckpts" / "final_pi_A_specialist_6v6.zip"
)
FROZEN_SHA = "8ac6a41c66ac2d83cf906f9a83a80aa60f4bff11d1a1a97c5a11fa551a5a5d82"
N_AGENTS = 6
N_SEEDS = 16
WATCH_STEPS = (25_000, 50_000)
STEP_SEED_BASE = {25_000: 22_300_001, 50_000: 22_400_001}
COLLAPSE_WR = 0.25
POLE_A_CATASTROPHE = -0.25
ARMS = (("k1", 1), ("k3", 3))
POLES = ("A", "B")
BASE_KEY = {"A": "OP6", "B": "OP7"}


def _out(step: int) -> Path:
    return SD / f"CLOSEST_SPLIT_VIABILITY_6V6_{step}.json"


def _ckpt(step: int) -> Path:
    periodic = CKPT_DIR / f"ckpt_{RUN_TAG}_{step}.zip"
    if periodic.is_file():
        return periodic
    final = CKPT_DIR / f"final_{RUN_TAG}.zip"
    if step == 25_000 and final.is_file():
        return final
    return periodic


def decide(cell_wr: dict, contrasts: dict) -> str:
    """Point estimates only. n=16 is a viability check, not a gate."""
    d_b = float(contrasts["Delta_B"]["mean"])
    d_a = float(contrasts["Delta_A"]["mean"])
    k1_dead = cell_wr["k1_poleA"] < COLLAPSE_WR and cell_wr["k1_poleB"] < COLLAPSE_WR
    k3_dead = cell_wr["k3_poleA"] < COLLAPSE_WR and cell_wr["k3_poleB"] < COLLAPSE_WR
    if k1_dead:
        return "obvious_collapse"
    if d_b > 0 and d_a > POLE_A_CATASTROPHE and not k1_dead and not k3_dead:
        return "promising"
    return "unclear"


def run_screen(step: int, device: str = "cuda") -> dict:
    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays,
    )
    from experiments.pole_attestation import resolve_pole_genome
    from gpu_env._core._entity_obs import augment_obs_with_entities
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.custom_ppo.rule_role_assignment import RoleHoldState, roles_from_core
    from rl.custom_ppo.split_attack_defend import splice_actions
    import experiments.r2_learned_crossover as R2

    ckpt = _ckpt(step)
    if not ckpt.is_file():
        raise SystemExit(f"REFUSING: checkpoint missing: {ckpt}")
    if _out(step).is_file():
        raise SystemExit(f"REFUSING: watch record already exists: {_out(step)}")
    seeds = list(range(STEP_SEED_BASE[step], STEP_SEED_BASE[step] + N_SEEDS))
    R2.AGENTS = N_AGENTS
    genomes = {"A": {"OP6": resolve_pole_genome("A", N_AGENTS)},
               "B": {"OP7": resolve_pole_genome("B", N_AGENTS)}}

    probe = R2.build_env(device, seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    pi_d = load_custom_ppo_policy(str(ckpt), obs_space, act_space, device=device)
    pi_a = load_custom_ppo_policy(str(FROZEN_ATTACK), obs_space, act_space, device=device)
    if not bool(getattr(pi_d.model, "role_conditioning_enabled", False)):
        raise SystemExit("REFUSING: checkpoint is not role-conditioned pi_D")
    if bool(getattr(pi_a.model, "role_conditioning_enabled", False)):
        raise SystemExit("REFUSING: frozen attacker must be the native pi_A")

    def composite(obs, hold_roles) -> np.ndarray:
        d_act, _ = pi_d.predict(obs, deterministic=True)
        a_act, _ = pi_a.predict(obs, deterministic=True)
        n_agents = int(pi_d.model.n_agents)
        heads = int(pi_d.model.heads_per_agent)
        d_t = torch.as_tensor(np.asarray(d_act), dtype=torch.long).reshape(1, -1)
        a_t = torch.as_tensor(np.asarray(a_act), dtype=torch.long).reshape(1, -1)
        roles_t = torch.as_tensor(np.asarray(hold_roles), dtype=torch.float32).reshape(1, n_agents)
        return splice_actions(d_t, a_t, roles_t < 0.5, heads).reshape(-1).numpy().astype(np.int64)

    def episode(pole: str, seed: int, k: int) -> dict:
        env = R2.build_env(device, seed)
        core = env.core
        hold = RoleHoldState(
            int(env.num_envs), N_AGENTS, hold_ticks=8, fixed_for_episode=True,
            device=device, k_defend=k,
        )
        try:
            pi_d.reset_strategy()
            pi_a.reset_strategy()
            install_keyed_opponent_overlays(core, genomes[pole])
            key = BASE_KEY[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            obs = augment_obs_with_entities(obs, core, side="blue")
            roles = roles_from_core(core, hold, force=True)
            obs["roles"] = roles.detach().cpu().numpy().astype(np.float32)
            assert_live_opponent_batch(core, genomes[pole], allowed_keys=(key,),
                                       context=f"collapse-watch {pole} seed {seed} k={k}")
            terminal = None
            for _ in range(R2.MAX_STEPS):
                action = composite(obs, obs["roles"])
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                obs = augment_obs_with_entities(obs, core, side="blue")
                roles = roles_from_core(core, hold, force=False)
                obs["roles"] = roles.detach().cpu().numpy().astype(np.float32)
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    res = (i0 or {}).get("episode_result") or {}
                    terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                    break
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            blue, red = terminal
            return {"arm": f"k{k}", "k": k, "pole": pole, "seed": seed,
                    "blue": blue, "red": red, "win": int(blue > red), "margin": blue - red}
        finally:
            env.close()

    cells = [(arm, k, pole, s) for s in seeds for pole in POLES for arm, k in ARMS]
    rows = []
    with torch.no_grad():
        for arm, k, pole, seed in tqdm_iter(cells, desc=f"collapse-watch {step}", unit="ep"):
            rows.append(episode(pole, seed, k))

    def wr(k: int, pole: str) -> float:
        return float(np.mean([r["win"] for r in rows if r["k"] == k and r["pole"] == pole]))

    def paired(pole: str) -> np.ndarray:
        by = {(r["k"], r["seed"]): r["win"] for r in rows if r["pole"] == pole}
        return np.asarray([by[(1, s)] - by[(3, s)] for s in seeds], dtype=np.float64)

    cell_wr = {f"k{k}_pole{pole}": wr(k, pole) for k in (1, 3) for pole in POLES}
    contrasts = {"Delta_A": _bootstrap(paired("A")), "Delta_B": _bootstrap(paired("B"))}
    decision = decide(cell_wr, contrasts)
    k1_dead = cell_wr["k1_poleA"] < COLLAPSE_WR and cell_wr["k1_poleB"] < COLLAPSE_WR
    k3_dead = cell_wr["k3_poleA"] < COLLAPSE_WR and cell_wr["k3_poleB"] < COLLAPSE_WR
    payload = {
        "record_id": f"CLOSEST_SPLIT_VIABILITY_6V6_{step}",
        "classification": "NON-CLAIM viability screen. n=16. Not a crossover result.",
        "step": step,
        "checkpoint": str(ckpt),
        "frozen_attack_sha256": FROZEN_SHA,
        "seeds": seeds,
        "CELL_WIN_RATES": cell_wr,
        "WIN_RATE_CONTRASTS_nonclaim": contrasts,
        "decision": decision,
        "both_compositions_destroyed": bool(k1_dead and k3_dead),
        "reading_rule": {
            "promising": "Delta_B>0, Delta_A>-0.25, neither composition dead on both poles",
            "obvious_collapse": "k=1 win rate < 0.25 on both poles",
            "unclear": "otherwise; continue to 50k once, then stop on this PC",
        },
        "rows": rows,
    }
    _out(step).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "step": step,
        "decision": decision,
        "cell_win_rates": cell_wr,
        "Delta_A": contrasts["Delta_A"]["mean"],
        "Delta_B": contrasts["Delta_B"]["mean"],
    }, indent=2), flush=True)
    return payload


def _alive(pid: int) -> bool:
    import subprocess
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
    return str(pid) in (r.stdout or "")


def _wait_for_finished_ckpt(step: int, pid: int) -> Path:
    print(f"waiting for {step} checkpoint and train pid {pid} to exit", flush=True)
    while True:
        path = _ckpt(step)
        if path.is_file() and not _alive(pid):
            return path
        if not _alive(pid) and not path.is_file():
            raise SystemExit(f"train pid {pid} exited before a {step} checkpoint existed")
        time.sleep(60)


def _train_cmd(total: int, resume: str | None) -> list[str]:
    cmd = [
        "experiments/train_specialist_scale.py",
        "--team-size", "6", "--policy", "A", "--seed", "22200001", "--device", "cuda",
        "--total-timesteps", str(total),
        "--run-label-suffix", "_closest_split_defend_6v6_v1",
        "--role-conditioning-enabled", "--role-fixed-for-episode",
        "--role-conditioning-allow-pre-entity-base",
        "--role-k-defend-choices", "1,3",
        "--split-attack-defend-enabled",
        "--split-attack-defend-frozen-ckpt",
        "artifacts/scale_6v6_specialists/pi_A_specialist_6v6/ckpts/final_pi_A_specialist_6v6.zip",
        "--split-attack-defend-frozen-ckpt-sha256", FROZEN_SHA,
        "--exploratory-spec",
        "artifacts/strategic_demand/sppo/CLOSEST_SPLIT_LEARNED_DEFEND_6V6_V1_SPEC.json",
    ]
    if resume:
        cmd.extend(["--resume", resume])
    else:
        cmd.extend([
            "--load-path",
            "artifacts/scale_6v6_specialists/pi_A_specialist_6v6/ckpts/final_pi_A_specialist_6v6.zip",
        ])
    return cmd


def _write_other_pc(step: int, ckpt: Path) -> None:
    cmd = _train_cmd(200_000, str(ckpt))
    text = (
        "Promising viability screen. Do not continue the long run on this PC.\n"
        "On the other PC, from AICTFProject, resume this checkpoint through 200k:\n\n"
        + " ".join(cmd) + "\n"
    )
    path = SD / "CLOSEST_SPLIT_LEARNED_DEFEND_6V6_OTHER_PC.txt"
    path.write_text(text, encoding="utf-8")
    print(text, flush=True)


def _launch_resume(total: int, ckpt: Path) -> int:
    import subprocess
    log = SD / "closest_split_learned_defend_6v6_v1.log"
    err = SD / "closest_split_learned_defend_6v6_v1.err"
    proc = subprocess.Popen(
        [sys.executable, *_train_cmd(total, str(ckpt))],
        cwd=str(ROOT),
        stdout=open(log, "a", encoding="utf-8"),
        stderr=open(err, "a", encoding="utf-8"),
    )
    print(f"resumed training to {total} pid={proc.pid}", flush=True)
    return int(proc.pid)


def watch(train_pid: int, device: str) -> int:
    print(f"viability ladder: 25k screen, then 50k only if unclear. train pid={train_pid}", flush=True)
    ckpt25 = _wait_for_finished_ckpt(25_000, train_pid)
    payload = run_screen(25_000, device=device)
    decision = payload["decision"]
    if decision == "promising":
        _write_other_pc(25_000, ckpt25)
        return 0
    if decision == "obvious_collapse":
        print("obvious collapse at 25k; stopping", flush=True)
        return 3
    print("25k unclear; continuing locally to 50k", flush=True)
    pid50 = _launch_resume(50_000, ckpt25)
    ckpt50 = _wait_for_finished_ckpt(50_000, pid50)
    payload50 = run_screen(50_000, device=device)
    if payload50["decision"] == "promising":
        _write_other_pc(50_000, ckpt50)
        return 0
    if payload50["decision"] == "obvious_collapse":
        print("obvious collapse at 50k; stopping", flush=True)
        return 3
    print("still unclear at 50k; not starting 200k on this PC", flush=True)
    (SD / "CLOSEST_SPLIT_VIABILITY_6V6_UNCLEAR.json").write_text(
        json.dumps({"decision": "unclear_at_50k", "checkpoint": str(ckpt50)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("screen", "watch"), required=True)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--train-pid", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.stage == "screen":
        if a.step not in STEP_SEED_BASE:
            raise SystemExit(f"--step must be one of {sorted(STEP_SEED_BASE)}")
        run_screen(a.step, device=a.device)
        return 0
    if a.train_pid <= 0:
        raise SystemExit("--train-pid is required for --stage watch")
    return watch(a.train_pid, a.device)


if __name__ == "__main__":
    raise SystemExit(main())
