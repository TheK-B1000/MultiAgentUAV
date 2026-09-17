"""Capture per-tick blue role proxies + red BT roles for heartbeat plots.

DIAGNOSTIC. Replays one sealed confirmatory seed for (pi_A, pi_B) x (Pole A, B)
and writes agent-step rows rich enough for swimlane + red overlay figures.

Fidelity: terminal (blue, red) must match the sealed crossover CSV row.

    python -m experiments.capture_role_heartbeat_timeline --seed 16700001
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_actions import MacroAction

N_AGENTS = 4
BASE_KEY = {"A": "OP6", "B": "OP7"}
MACRO_NAMES = {int(m): m.name for m in MacroAction}
RED_ROLE_NAMES = {
    0: "ATTACKER",
    1: "DEFENDER",
    2: "ESCORT",
    3: "INTERCEPTOR",
    4: "FLAG_RETR",
    5: "COUNTER",
    6: "2V1_WING",
}

DEFAULT_PI_A = (
    "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3/"
    "ckpts/final_pi_A_specialist_4v4_b3.zip"
)
DEFAULT_PI_B = (
    "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_b3/"
    "ckpts/final_pi_B_specialist_4v4_b3.zip"
)
DEFAULT_POLE_B = (
    "artifacts/strategic_demand/sppo/pole_b2_candidates/B3-3_lockdef10_2v1.json"
)
DEFAULT_SEALED = (
    "artifacts/strategic_demand/sppo/"
    "confirmatory_b3_3_4v4_specialist_crossover_eval_rows.csv"
)
DEFAULT_OUT = (
    "artifacts/strategic_demand/sppo/role_heartbeat_timeline_agent_step_rows.csv"
)

FIELDNAMES = [
    "policy",
    "pole",
    "seed",
    "t",
    "agent",
    "team",
    "carrying",
    "own_half",
    "enemy_half",
    "near_enemy_flag",
    "near_own_flag",
    "macro",
    "blue_role",
    "red_role",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sealed_terminal(csv_path: Path, seed: int, policy: str, pole: str) -> tuple[int, int]:
    with csv_path.open(encoding="utf-8") as fh:
        rows = [
            r
            for r in csv.DictReader(fh)
            if int(r["seed"]) == seed and r.get("pole") == pole and r.get("policy") == policy
        ]
    if len(rows) != 1:
        raise SystemExit(
            f"FIDELITY: expected 1 sealed row for policy={policy} pole={pole} "
            f"seed={seed}, got {len(rows)}"
        )
    return int(rows[0]["blue"]), int(rows[0]["red"])


def _pole_genomes(n: int, pole_b_json: Path):
    from experiments.opponent_spec import (
        _with_full_team_defender_gate,
        pole_A_genome,
    )
    from experiments.sds_genome import SDSGenome

    pole_b = _with_full_team_defender_gate(
        SDSGenome.from_dict(json.loads(pole_b_json.read_text(encoding="utf-8"))),
        n,
    )
    return {
        "A": {"OP6": pole_A_genome(n)},
        "B": {"OP7": pole_b},
        "_pole_b_source": f"CANDIDATE_OVERRIDE:{pole_b.genome_id} from {pole_b_json}",
    }


def blue_role_proxy(
    *,
    carrying: bool,
    macro: str,
    own_half: bool,
    near_own_flag: bool,
    near_enemy_flag: bool,
) -> str:
    """Macro + geometry role proxy (same contract as the figure builder)."""
    if carrying:
        return "CARRIER"
    if macro == "GET_FLAG":
        return "OFFENSE"
    if macro == "GO_HOME":
        return "DEFENSE"
    if own_half and (near_own_flag or macro in ("GO_TO", "PLACE_MINE", "GRAB_MINE")):
        # Home-side linger without an offensive macro counts as defense posture.
        if not near_enemy_flag:
            return "DEFENSE"
    if near_enemy_flag and not own_half:
        return "OFFENSE"
    return "TRANSIT"


def capture_episode(env, core, policy, max_steps: int, obs) -> tuple[list[dict], tuple[int, int]]:
    rows: list[dict] = []
    terminal = None
    mid = float(core.cfg.map_cols) * 0.5
    flag_near = 6.0

    for t in range(max_steps):
        with torch.no_grad():
            action, _ = policy.predict(obs, deterministic=True)
        act = np.asarray(action).reshape(-1)
        macros = act[0::2].astype(int)

        bx = core.blue_x[0].detach().cpu().numpy()
        by = core.blue_y[0].detach().cpu().numpy()
        ba = core.blue_alive[0].detach().cpu().numpy().astype(bool)
        bc = core.blue_carrying[0].detach().cpu().numpy().astype(bool)
        own_half = core._is_on_home_side("blue", core.blue_x)[0].detach().cpu().numpy().astype(bool)

        rfx = float(core.red_flag_pos[0, 0].item())
        rfy = float(core.red_flag_pos[0, 1].item())
        bfx = float(core.blue_flag_pos[0, 0].item())
        bfy = float(core.blue_flag_pos[0, 1].item())
        d_enemy = np.sqrt((bx - rfx) ** 2 + (by - rfy) ** 2)
        d_own = np.sqrt((bx - bfx) ** 2 + (by - bfy) ** 2)

        red_roles = []
        if hasattr(core, "bt_red_role"):
            red_roles = [int(x) for x in core.bt_red_role[0].detach().cpu().tolist()]

        for i in range(N_AGENTS):
            if not ba[i]:
                continue
            macro = MACRO_NAMES.get(int(macros[i]), str(int(macros[i])))
            near_e = bool(d_enemy[i] < flag_near)
            near_o = bool(d_own[i] < flag_near)
            role = blue_role_proxy(
                carrying=bool(bc[i]),
                macro=macro,
                own_half=bool(own_half[i]),
                near_own_flag=near_o,
                near_enemy_flag=near_e,
            )
            rows.append(
                {
                    "t": t,
                    "agent": i,
                    "team": "blue",
                    "carrying": int(bc[i]),
                    "own_half": int(own_half[i]),
                    "enemy_half": int(not own_half[i]),
                    "near_enemy_flag": int(near_e),
                    "near_own_flag": int(near_o),
                    "macro": macro,
                    "blue_role": role,
                    "red_role": "",
                }
            )

        for j, rid in enumerate(red_roles):
            rows.append(
                {
                    "t": t,
                    "agent": j,
                    "team": "red",
                    "carrying": int(bool(core.red_carrying[0, j].item()))
                    if hasattr(core, "red_carrying")
                    else 0,
                    "own_half": "",
                    "enemy_half": "",
                    "near_enemy_flag": "",
                    "near_own_flag": "",
                    "macro": "",
                    "blue_role": "",
                    "red_role": RED_ROLE_NAMES.get(rid, str(rid)),
                }
            )

        env.step_async(action)
        obs, _r, done, info = env.step_wait()
        obs["global_state"] = env.state()
        if bool(np.asarray(done).any()):
            i0 = info[0] if isinstance(info, (list, tuple)) else info
            res = (i0 or {}).get("episode_result") or {}
            terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
            break

    if terminal is None:
        terminal = (int(core.blue_score[0].item()), int(core.red_score[0].item()))
    return rows, terminal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=16_700_001)
    ap.add_argument("--pi-a-path", default=DEFAULT_PI_A)
    ap.add_argument("--pi-b-path", default=DEFAULT_PI_B)
    ap.add_argument("--pole-b-genome-json", default=DEFAULT_POLE_B)
    ap.add_argument("--sealed-csv", default=DEFAULT_SEALED)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--poles", default="A,B", help="Comma poles to capture")
    args = ap.parse_args()

    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import (
        assert_live_opponent_batch,
        install_keyed_opponent_overlays,
    )
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    R2.AGENTS = N_AGENTS
    sealed = Path(args.sealed_csv)
    out_path = Path(args.out)
    genomes = _pole_genomes(N_AGENTS, Path(args.pole_b_genome_json))
    pole_b_source = genomes.pop("_pole_b_source")
    poles = [p.strip() for p in str(args.poles).split(",") if p.strip()]

    probe = R2.build_env(args.device, int(args.seed))
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policies = {
        "pi_A": load_custom_ppo_policy(str(args.pi_a_path), obs_space, act_space, device=args.device),
        "pi_B": load_custom_ppo_policy(str(args.pi_b_path), obs_space, act_space, device=args.device),
    }

    print(f"ROLE HEARTBEAT CAPTURE  {_now()}")
    print("  DIAGNOSTIC_NOT_A_GATE")
    print(f"  seed={args.seed} poles={poles}")
    print(f"  pole_B {pole_b_source}")
    print(f"  out {out_path}\n", flush=True)

    all_rows: list[dict] = []
    fidelity: list[dict] = []
    work = [(pole, name) for pole in poles for name in ("pi_A", "pi_B")]
    bar = tqdm_iter(work, desc="role-heartbeat capture", unit="ep")
    for pole, name in bar:
        set_postfix(bar, f"{name}@Pole{pole}")
        env = R2.build_env(args.device, int(args.seed))
        core = env.core
        try:
            policies[name].reset_strategy()
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            gmap = genomes[pole]
            install_keyed_opponent_overlays(core, gmap)
            key = BASE_KEY[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            assert_live_opponent_batch(
                core, gmap, allowed_keys=(key,),
                context=f"role heartbeat {name}@{pole} seed {args.seed}",
            )
            rows, terminal = capture_episode(env, core, policies[name], args.max_steps, obs)
            expected = _sealed_terminal(sealed, int(args.seed), name, pole)
            ok = terminal == expected
            fidelity.append(
                {
                    "policy": name,
                    "pole": pole,
                    "seed": int(args.seed),
                    "expected": list(expected),
                    "got": list(terminal),
                    "match": ok,
                }
            )
            if not ok:
                raise SystemExit(
                    f"FIDELITY MISMATCH {name}@Pole{pole} seed={args.seed}: "
                    f"expected={expected} got={terminal}"
                )
            for r in rows:
                r["policy"] = name
                r["pole"] = pole
                r["seed"] = int(args.seed)
                all_rows.append(r)
        finally:
            env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})

    meta = {
        "utc": _now(),
        "seed": int(args.seed),
        "poles": poles,
        "n_rows": len(all_rows),
        "fidelity": fidelity,
        "pole_b_source": pole_b_source,
        "out": str(out_path),
        "role_proxy": (
            "CARRIER if carrying; GET_FLAG→OFFENSE; GO_HOME→DEFENSE; "
            "own_half+(near_own_flag|linger)→DEFENSE; near_enemy_flag+enemy_half→OFFENSE; else TRANSIT"
        ),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
