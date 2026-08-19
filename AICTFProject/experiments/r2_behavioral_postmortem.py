"""Behavioral postmortem on the R2 specialists — DESCRIPTIVE, NOT A GATE.

R2 is closed: R2_LEARNED_CROSSOVER_NOT_CONFIRMED (670bc828). Nothing here
reopens it, re-scores it, or feeds R3. No win-rate contrast is computed.

The question this answers is the one the payoff numbers cannot:

    Do pi_A and pi_B actually BEHAVE differently, or did they produce noisy
    payoff differences while converging to effectively the same policy?

That distinction decides the branch after the 1M->2M continuation experiment:

    behaviourally distinct  -> training robustness / seed variance problem
    behaviourally identical -> strategy-discovery collapse, needs anchoring

Method: matched-state counterfactual. A single rollout is driven by one policy
while BOTH policies are queried on every observation. Action disagreement is
therefore measured on identical states, with no confound from the two policies
visiting different parts of the state space. Each pole is driven twice, once by
each policy, so disagreement is measured on both policies' own state
distributions.

Strategic telemetry recorded alongside, using the same predicates as the frozen
V3 definitions:

    commitment   >= 2 live untagged BLUE on RED's half (the R4/observability
                 commitment event)
    home defense >= 1 live untagged BLUE on BLUE's half
    occupancy    mean fraction of live untagged BLUE on RED's half

Seeds reuse the R2 block deliberately: matched states require matched seeds, and
since no contrast is computed the spent block cannot be re-spent by this script.

Run:  python experiments/r2_behavioral_postmortem.py --device cuda --n 48
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

import experiments.r2_learned_crossover as R2                      # noqa: E402
from experiments.opponent_spec import (                            # noqa: E402
    assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
)
from rl.curriculum import phase_from_tag                           # noqa: E402

OUT_DIR = ROOT / "artifacts/strategic_demand/r2_postmortem"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def episode(models: dict, driver: str, pole: str, seed: int, device: str) -> dict:
    """One rollout driven by `driver`, with every policy queried each step."""
    base_key = R2.POLES[pole]
    genomes = {"OP6": pole_A_genome()} if pole == "A" else {}
    env = R2.build_env(device, seed)
    core = env.core
    try:
        core._bt_profile_override = None
        core._sds_opening_hold_steps = 0
        install_keyed_opponent_overlays(core, genomes)
        env.env_method("set_phase", phase_from_tag(base_key))
        env.env_method("set_next_opponent", "SCRIPTED", base_key)
        obs = env.reset()
        assert_live_opponent_batch(core, genomes, allowed_keys=(base_key,),
                                   context=f"postmortem {pole} seed {seed}")

        disagree = 0
        steps = 0
        step_stats: list = []
        term = None

        for _ in range(R2.MAX_STEPS):
            acts = {name: m.predict(obs, deterministic=True)[0]
                    for name, m in models.items()}
            a_drv = acts[driver]
            other = "pi_B" if driver == "pi_A" else "pi_A"
            if not np.array_equal(np.asarray(acts[driver]).ravel(),
                                  np.asarray(acts[other]).ravel()):
                disagree += 1

            env.step_async(a_drv)
            obs, _r, done, info = env.step_wait()
            steps += 1

            # Accumulate telemetry ON DEVICE. Calling .item() per step forces a
            # CUDA sync every step and made this ~40s/episode versus ~3s for
            # the R2 scorer; the sync happens once per episode instead.
            live = core.blue_alive[0] & (~core.blue_tagged[0])
            on_red = core._is_on_home_side("red", core.blue_x)[0]
            on_own = core._is_on_home_side("blue", core.blue_x)[0]
            step_stats.append(torch.stack([
                (live & on_red).sum(),
                (live & on_own).sum(),
                live.sum(),
            ]))

            if bool(np.asarray(done).any()):
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                er = (i0 or {}).get("episode_result") or {}
                term = (int(er.get("blue_score", 0)), int(er.get("red_score", 0)))
                break
        if term is None:
            term = (int(core.blue_score[0]), int(core.red_score[0]))

        # single sync for the whole episode
        stats = torch.stack(step_stats).cpu().numpy() if step_stats else np.zeros((1, 3))
        n_on_red, n_on_own, n_live = stats[:, 0], stats[:, 1], np.maximum(stats[:, 2], 1)
        commit_mask = n_on_red >= 2
        commit_steps = int(commit_mask.sum())
        defend_steps = int((n_on_own >= 1).sum())
        occupancy = (n_on_red / n_live).tolist()
        t_commit = int(np.argmax(commit_mask)) + 1 if commit_mask.any() else None

        return {
            "pole": pole, "driver": driver, "episode_seed": seed,
            "steps": steps,
            "action_disagreement_rate": disagree / max(1, steps),
            "commit_frac": commit_steps / max(1, steps),
            "defend_frac": defend_steps / max(1, steps),
            "mean_enemy_half_occupancy": float(np.mean(occupancy)) if occupancy else 0.0,
            "t_commit": t_commit if t_commit is not None else -1,
            "blue_score": term[0], "red_score": term[1],
        }
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=48, help="seeds per (pole, driver) cell")
    a = ap.parse_args()

    from rl.custom_ppo import load_custom_ppo_policy
    probe = R2.build_env(a.device, R2.SEED_BASE)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    models = {n: load_custom_ppo_policy(str(R2.POLICIES[n]), obs_space, act_space,
                                        device=a.device)
              for n in ("pi_A", "pi_B")}

    print(f"R2 BEHAVIORAL POSTMORTEM (descriptive, not a gate)  {_now()}")
    print(f"  {a.n} seeds x 2 poles x 2 drivers = {a.n * 4} episodes")

    rows = []
    for pole in ("A", "B"):
        for driver in ("pi_A", "pi_B"):
            print(f"  driving {driver} on pole {pole} ...", flush=True)
            for i in range(a.n):
                rows.append(episode(models, driver, pole, R2.SEED_BASE + i, a.device))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "episode_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(pole, driver, key):
        v = [r[key] for r in rows if r["pole"] == pole and r["driver"] == driver]
        return float(np.mean(v)) if v else float("nan")

    summary = {
        "record": "R2 behavioral postmortem", "utc": _now(),
        "status": "DESCRIPTIVE ONLY. R2 remains NOT CONFIRMED (670bc828). "
                  "No win-rate contrast computed; nothing here feeds R3.",
        "n_per_cell": a.n,
        "cells": {},
    }
    print("\n" + "=" * 74)
    print(f"{'pole/driver':<16}{'disagree':>10}{'commit_f':>10}{'defend_f':>10}"
          f"{'occupancy':>11}{'t_commit':>10}")
    for pole in ("A", "B"):
        for driver in ("pi_A", "pi_B"):
            cell = {k: agg(pole, driver, k) for k in
                    ("action_disagreement_rate", "commit_frac", "defend_frac",
                     "mean_enemy_half_occupancy", "t_commit")}
            summary["cells"][f"{pole}|{driver}"] = cell
            print(f"{pole + ' / ' + driver:<16}"
                  f"{cell['action_disagreement_rate']:>10.3f}"
                  f"{cell['commit_frac']:>10.3f}{cell['defend_frac']:>10.3f}"
                  f"{cell['mean_enemy_half_occupancy']:>11.3f}"
                  f"{cell['t_commit']:>10.1f}")
    print("=" * 74)

    dis = float(np.mean([r["action_disagreement_rate"] for r in rows]))
    summary["overall_action_disagreement_rate"] = dis
    summary["reading"] = (
        "near 0 => the specialists converged to effectively the same policy "
        "(strategy-discovery collapse). substantially above 0 => they are "
        "behaviourally distinct and the R2 failure is about stability/variance, "
        "not about absence of learned difference.")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"overall matched-state action disagreement: {dis:.4f}")
    print(f"written: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
