"""CCP-S2 bank construction, stage B: the canonical r_bank(x) trajectory rollout.

Implements CCP_S2_SPEC.json#BANK_CONSTRUCTION.ANTI_TRAJECTORY_SHOPPING_RULE. For every unit
that Stage A routed to a teacher (t* is not None, w > 0), roll that already-selected teacher
EXACTLY ONCE from the boundary state, under a canonical seed derived INDEPENDENTLY of the M
measurement seeds:

    r_bank(x) = sha256("S2_BANK|<state_id>|<estimand>")  -- mirrors continuation_seed()'s
    construction byte for byte, different domain string, no j.

The loophole this closes: teacher selection and weight come EXCLUSIVELY from the frozen Stage-A
measurement (CCP_S2_CAUSAL_BANK_ROUTING.json). This rollout's own terminal outcome is recorded
for provenance but MUST NOT alter t* or w -- it only supplies (obs, action) trajectory data.
Nothing here re-opens routing, and no unit is rolled more than once.

Intervention semantics are identical to the measurement collector's: replay the recorded prefix
under the incumbent, assert the manifest's free set holds at s_t, seed ONLY the env RNG, then
run the full remaining episode with the teacher controlling intervened_agents(estimand) and the
incumbent controlling the rest (SEQUENCE takeover, full-episode).

Supervision targeting: w_ti = w for agents the estimand actually intervenes on, 0 for the
others. A unit measured by intervening on agent0 makes no causal claim about agent1, so agent1
carries no weight and contributes nothing to numerator or denominator.

Deliberately NOT used: rl/causal_supervision.py's CausalRecord/CausalSegment. Those derive the
teacher from a single signed teacher-vs-teacher delta_q (the predecessor's estimand). CCP-S2's
estimand is two independent incumbent-relative advantages with its own frozen routing rule, so
wrapping a CCP-S2 unit in CausalRecord could silently substitute a different teacher. Per
CCP_S2_SPEC.json#CAUSAL_OBJECTIVE only causal_supervision_loss and the decision-mask machinery
are reused unchanged; the weight source and routing derivation are CCP-S2's own.

Run:  python experiments/ccp_s2_bank_rollout.py --device cuda
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

import experiments.ccp_s2_collect as C

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
ROUTING = SD / "CCP_S2_CAUSAL_BANK_ROUTING.json"
MANIFEST = SD / "CCP_S2_STATE_MANIFEST_AMENDMENT.json"
TRAJ_DIR = SD / "ccp_s2_bank_traj"
OUT = SD / "CCP_S2_CAUSAL_BANK_STAGE_B.json"

N_AGENTS = 2


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bank_seed(state_id: str, estimand: str) -> int:
    """The canonical S2_BANK mapping -- independent of every CCP_S2_MEASURE seed."""
    h = hashlib.sha256(f"S2_BANK|{state_id}|{estimand}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 63 - 1)


def roll_unit(unit: dict, st: dict, incumbent, teachers: dict, R2, device: str,
              env_ctx: dict) -> dict:
    """Roll t*(x) once under r_bank(x), recording every step's supervision material."""
    from rl.causal_supervision import decision_mask_from_core

    pole = st["pole"]
    z = C.POLE_LATENT[pole]
    estimand = unit["estimand"]
    ident = f"{unit['state_id']}|{estimand}"
    r_bank = bank_seed(unit["state_id"], estimand)

    env, obs, core = C.setup_env(R2, env_ctx["P0"], env_ctx["phase_from_tag"],
                                 env_ctx["install_keyed_opponent_overlays"],
                                 env_ctx["pole_A_genome"],
                                 env_ctx["assert_live_opponent_batch"],
                                 device, st["seed"], pole)
    try:
        teacher = teachers[unit["t_star"]]
        if getattr(teacher, "fixed_latent_strategy", None) is not None:
            teacher.fixed_latent_strategy = False
        teacher.reset_strategy()

        obs = C.replay_prefix(R2, incumbent, env, obs, st["actions"], z, ident)

        f0 = bool((core.blue_commit_ticks_left[0, 0] <= 0).item())
        f1 = bool((core.blue_commit_ticks_left[0, 1] <= 0).item())
        expect = {"agent0_only": (True, False), "agent1_only": (False, True),
                  "both_free": (True, True)}[st["free_set"]]
        if (f0, f1) != expect:
            raise SystemExit(f"REFUSING: free set at s_t is {(f0, f1)}, manifest says "
                             f"{st['free_set']} for {ident}")

        core._rng.manual_seed(int(r_bank))                    # ONLY the env RNG
        targets = C.intervened_agents(estimand)

        grids, vecs, amasks, masks, gstates = [], [], [], [], []
        teach_acts, inc_acts, dmasks = [], [], []
        info, steps = None, 0
        for _ in range(R2.MAX_STEPS):
            d = decision_mask_from_core(core, N_AGENTS, side="blue")   # (1, N_AGENTS) bool
            a_inc, _ = incumbent.predict(obs, deterministic=True)
            a_teach, _ = teacher.predict(obs, deterministic=True)
            inc = np.asarray(a_inc).ravel().copy()
            tch = np.asarray(a_teach).ravel().copy()

            grids.append(np.asarray(obs["grid"])[0].copy())
            vecs.append(np.asarray(obs["vec"])[0].copy())
            amasks.append(np.asarray(obs["agent_mask"])[0].copy())
            masks.append(np.asarray(obs["mask"])[0].copy())
            gstates.append(np.asarray(obs["global_state"])[0].copy())
            teach_acts.append(tch)
            inc_acts.append(inc)
            dmasks.append(np.asarray(d.detach().cpu())[0].copy())

            act = inc.copy()
            for i in targets:
                act[i * 2] = tch[i * 2]
                act[i * 2 + 1] = tch[i * 2 + 1]
            env.step_async(act)
            obs, _r, done, info = env.step_wait()
            obs["global_state"] = env.state()
            steps += 1
            if bool(np.asarray(done).any()):
                break

        # w_ti: the unit's weight on intervened agents only; zero elsewhere. A unit that
        # intervened on agent0 makes no causal claim about agent1.
        weights = np.zeros((steps, N_AGENTS), dtype=np.float32)
        for i in targets:
            weights[:, i] = float(unit["w"])

        dmask = np.asarray(dmasks, dtype=bool)                       # (steps, N_AGENTS)
        teach = np.asarray(teach_acts)                               # (steps, 2*N_AGENTS)
        incum = np.asarray(inc_acts)
        # a supervised decision = agent free to commit AND carrying weight
        supervised = dmask & (weights > 0)
        # teacher/incumbent disagreement at those supervised decisions
        disagree = np.zeros_like(supervised)
        for i in range(N_AGENTS):
            differs = (teach[:, i * 2] != incum[:, i * 2]) | (teach[:, i * 2 + 1] != incum[:, i * 2 + 1])
            disagree[:, i] = supervised[:, i] & differs

        TRAJ_DIR.mkdir(parents=True, exist_ok=True)
        npz = TRAJ_DIR / f"{ident.replace('|', '__')}.npz"
        np.savez_compressed(
            npz, grid=np.asarray(grids, dtype=np.float32), vec=np.asarray(vecs, dtype=np.float32),
            agent_mask=np.asarray(amasks, dtype=np.float32), mask=np.asarray(masks, dtype=np.float32),
            global_state=np.asarray(gstates, dtype=np.float32),
            teacher_actions=teach, incumbent_actions=incum,
            decision_mask=dmask, weights=weights, z_idx=np.full((steps,), z, dtype=np.int64))

        res = C.outcome(core, info)          # provenance ONLY -- must not alter t* or w
        return {
            "unit": ident, "state_id": unit["state_id"], "estimand": estimand,
            "pole": pole, "free_set": st["free_set"], "latent_supervised": z,
            "t_star": unit["t_star"], "w": unit["w"], "intervened_agents": list(targets),
            "r_bank": r_bank, "steps": steps, "trajectory_file": str(npz.relative_to(ROOT)),
            "n_supervised_decisions": int(supervised.sum()),
            "n_supervised_decisions_with_disagreement": int(disagree.sum()),
            "n_free_ticks_any_agent": int(dmask.any(axis=1).sum()),
            "canonical_rollout_outcome_provenance_only": res,
        }
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot -- each unit is rolled exactly once")
    routing = json.loads(ROUTING.read_text(encoding="utf-8"))
    if routing["status"] != "FROZEN_RESULT":
        raise SystemExit(f"REFUSING: Stage A routing not frozen: {routing['status']!r}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    states_by_id = {s["state_id"]: s for s in manifest["states"]}

    units = [u for u in routing["units"] if u["t_star"] is not None]
    if not units:
        raise SystemExit("REFUSING: no routed units; Stage B has nothing to roll")
    print(f"CCP-S2 BANK STAGE B  {_now()}")
    print(f"  {len(units)} routed units to roll (of {len(routing['units'])} active)\n", flush=True)

    device, incumbent, teachers, R2, env_ctx = C.load_runtime(args.device)

    records = []
    for n, u in enumerate(units, 1):
        rec = roll_unit(u, states_by_id[u["state_id"]], incumbent, teachers, R2, device, env_ctx)
        records.append(rec)
        print(f"  [{n}/{len(units)}] {rec['unit']:28s} t*={rec['t_star']:5s} "
              f"steps={rec['steps']:3d} supervised={rec['n_supervised_decisions']:3d} "
              f"disagree={rec['n_supervised_decisions_with_disagreement']:3d}", flush=True)

    total_sup = sum(r["n_supervised_decisions"] for r in records)
    total_dis = sum(r["n_supervised_decisions_with_disagreement"] for r in records)
    zero_sup = [r["unit"] for r in records if r["n_supervised_decisions"] == 0]

    OUT.write_text(json.dumps({
        "record": "CCP-S2 causal bank, stage B canonical rollouts",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "implements": "CCP_S2_SPEC.json#BANK_CONSTRUCTION.ANTI_TRAJECTORY_SHOPPING_RULE",
        "seed_mapping": "r_bank(x) = sha256('S2_BANK|<state_id>|<estimand>')[:8], independent "
                        "of every CCP_S2_MEASURE seed",
        "rolled_once_each": True,
        "outcome_did_not_alter_routing": "t* and w are copied verbatim from "
                                         "CCP_S2_CAUSAL_BANK_ROUTING.json; the canonical "
                                         "rollout's terminal outcome is recorded for provenance "
                                         "only and was never consulted",
        "N_causal_anchors": len(records),
        "N_usable_commitment_level_supervision_targets": total_sup,
        "N_supervision_targets_with_teacher_disagreement": total_dis,
        "zero_supervision_units": zero_sup,
        "supervision_note": "supervision targets are highly correlated descendants of "
                            f"{len(records)} causal anchors -- they are NOT independent causal "
                            "observations and must not be reported as such",
        "units": records,
    }, indent=2), encoding="utf-8")

    print(f"\n  N_causal_anchors                       {len(records)}")
    print(f"  N_usable_commitment_supervision_targets {total_sup}")
    print(f"  ... of which teacher != incumbent       {total_dis}")
    print(f"  zero-supervision units                  {len(zero_sup)} {zero_sup}")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
