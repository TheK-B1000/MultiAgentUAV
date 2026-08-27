"""D2 -- collect additional stolen-flag branch pairs. Implements D2_SPEC_FROZEN.json.

Reuses Phase 0's exact machinery unchanged: P0.source_policy_for(), P0.rollout()
(to regenerate the full per-step prefix -- verified deterministic; same seed +
same policy reproduces the exact trajectory already on record), and
P0.branch_at() (replay-to-state + branch-both-teachers, teacher-consistent
continuation). Only the SET OF DECISION POINTS branched is new.

Zero new seeds: every point comes from the 128 already-audited INNER_FIT seeds.

Selection is frozen and deterministic: shuffle the 71 eligible seeds with
rng_seed=11, then within each seed shuffle its own stolen-flag decision steps
with the same RNG stream, take up to cap_per_seed=8 per seed in that order,
until 150 total points are selected. The first 50 selected (in draw order) are
the rung_2 delta; the next 100 are the additional rung_3 delta -- this makes
rung_2 subset rung_3 by construction.

Run:  python experiments/d2_collect_stolen_density.py --device cuda
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts/strategic_demand"
SPPO = SD / "sppo"
COLL = SD / "phase0_scorer_data/full_collection_rebuild_per_branch"
OUT_DIR = SPPO / "d2_density" / "supplement_shards"
SELECTION_OUT = SPPO / "D2_POINT_SELECTION.json"
SPEC = SPPO / "D2_SPEC_FROZEN.json"

INNER_FIT = list(range(6_500_001, 6_500_129))
RNG_SEED, CAP_PER_SEED, TOTAL_NEEDED = 11, 8, 150


def select_points():
    """Deterministic TRUE round-robin selection. Returns [(seed, step), ...] in draw order.

    Per D2_SPEC_AMENDMENT_1: breadth before depth. Every eligible seed contributes
    one point per pass before any seed contributes a second, so the density
    increase reflects added STATE DIVERSITY rather than a few prolific seeds.
    """
    from rl.scorer.qpsi import QPsi, QPsiConfig
    m = QPsi(QPsiConfig())
    import experiments.phase0_collect_scorer_data as P0

    per_seed_steps: dict[int, list[int]] = {}
    for seed in INNER_FIT:
        z = np.load(COLL / "seed_shards" / f"seed_{seed}.npz", allow_pickle=True)
        src = P0.source_policy_for(seed, "B")
        pol_idx = 0 if src == "pi_A" else 1
        mask = (z["plain_policy"] == pol_idx) & (z["plain_pole"] == 1)
        if not mask.any():
            continue
        vec = torch.as_tensor(z["plain_obs_vec"][mask][:, 0], dtype=torch.float32)
        regime = m.regime_from_vec(vec).numpy()
        stolen = regime >= 2
        steps = z["plain_step"][mask][stolen].tolist()
        if steps:
            per_seed_steps[seed] = steps

    rng = np.random.default_rng(RNG_SEED)
    eligible = list(per_seed_steps)
    rng.shuffle(eligible)                              # fixed seed order, once
    for seed in eligible:
        rng.shuffle(per_seed_steps[seed])               # fixed per-seed draw order, once
    cursor = {seed: 0 for seed in eligible}

    selected: list[tuple[int, int]] = []
    while len(selected) < TOTAL_NEEDED:
        progressed = False
        for seed in eligible:                            # one pass = one point per seed
            if len(selected) >= TOTAL_NEEDED:
                break
            i = cursor[seed]
            if i >= min(CAP_PER_SEED, len(per_seed_steps[seed])):
                continue
            selected.append((seed, int(per_seed_steps[seed][i])))
            cursor[seed] += 1
            progressed = True
        if not progressed:
            raise SystemExit(
                f"REFUSING: exhausted all eligible seeds at cap {CAP_PER_SEED}/seed "
                f"with only {len(selected)}/{TOTAL_NEEDED} points; cap or pool too small")
    return selected


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_ANY_NEW_BRANCH_IS_COLLECTED":
        raise SystemExit("REFUSING: D2 spec is not in the expected pre-collection state")
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise SystemExit(f"REFUSING: {OUT_DIR} is not empty; collection is one-shot")

    selected = select_points()
    if len(selected) != TOTAL_NEEDED:
        raise SystemExit(f"REFUSING: selected {len(selected)} points, needed exactly {TOTAL_NEEDED}")

    sel_record = {
        "record": "D2 point selection", "utc": datetime.now(timezone.utc).isoformat(),
        "rng_seed": RNG_SEED, "cap_per_seed": CAP_PER_SEED, "total": TOTAL_NEEDED,
        "n_seeds_touched": len(set(s for s, _ in selected)),
        "rung_2_delta": selected[:50], "rung_3_additional_delta": selected[50:150],
        "all_selected_in_draw_order": selected,
    }
    SELECTION_OUT.write_text(json.dumps(sel_record, indent=2), encoding="utf-8")
    print(f"selected {len(selected)} points across {sel_record['n_seeds_touched']} seeds")
    print(f"  -> {SELECTION_OUT}\n")

    from rl.custom_ppo import load_custom_ppo_policy
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2

    probe = R2.build_env(device, INNER_FIT[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    models = {k: load_custom_ppo_policy(str(v), obs_space, act_space, device=device)
             for k, v in P0.TEACHERS.items()}

    by_seed: dict[int, list[int]] = {}
    for seed, step in selected:
        by_seed.setdefault(seed, []).append(step)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for si, (seed, steps) in enumerate(sorted(by_seed.items())):
        src = P0.source_policy_for(seed, "B")
        r = P0.rollout(models[src], "B", seed, device, record_prefix=True)   # regenerate prefix
        have = set(r["decision_steps"])
        missing = [s for s in steps if s not in have]
        if missing:
            raise SystemExit(f"REFUSING: seed {seed} steps {missing} not decision points on replay")

        rows = {k: [] for k in ("branch_obs_grid", "branch_obs_vec", "branch_obs_agent_mask",
                                "branch_obs_mask", "branch_pole",
                                "branch_pi_A_action", "branch_pi_A_return", "branch_pi_A_blue", "branch_pi_A_red",
                                "branch_pi_B_action", "branch_pi_B_return", "branch_pi_B_blue", "branch_pi_B_red")}
        for t in steps:
            b = P0.branch_at(models, "B", seed, r["prefix"], t, device)
            if b is None:
                raise SystemExit(f"REFUSING: seed {seed} step {t} branch returned None (episode ended early)")
            rows["branch_obs_grid"].append(b["obs"]["grid"])
            rows["branch_obs_vec"].append(b["obs"]["vec"])
            rows["branch_obs_agent_mask"].append(b["obs"]["agent_mask"])
            rows["branch_obs_mask"].append(b["obs"]["mask"])
            rows["branch_pole"].append(1)
            for tag in ("pi_A", "pi_B"):
                br = b["branches"][tag]
                rows[f"branch_{tag}_action"].append(br["action"])
                rows[f"branch_{tag}_return"].append([float("nan")])   # not fit on; margin comes from blue/red
                rows[f"branch_{tag}_blue"].append(br["blue"])
                rows[f"branch_{tag}_red"].append(br["red"])
            n_done += 1

        np.savez(OUT_DIR / f"seed_{seed}.npz",
                **{k: np.asarray(v) for k, v in rows.items()})
        if (si + 1) % 5 == 0 or si == len(by_seed) - 1:
            print(f"  {si+1}/{len(by_seed)} seeds, {n_done}/{TOTAL_NEEDED} branches collected", flush=True)

    print(f"\nD2 collection complete: {n_done} new branch pairs across {len(by_seed)} seeds")
    print(f"  -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
