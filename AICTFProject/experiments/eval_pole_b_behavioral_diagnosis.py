r"""Behavioural diagnosis: why does A3 handle Pole B better than B3, which trained on it?

Governed by artifacts/strategic_demand/sppo/POLE_B_BEHAVIORAL_DIAGNOSIS_SPEC.json
(frozen before any episode, with every measured quantity verified obtainable first).

Three policies form a time machine:

    A3      -- trained for Pole A, yet handles Pole B well (~0.70-0.74)
    B100k   -- early B: competent but undifferentiated
    Bfinal  -- fully trained B, worst on its own pole (~0.59)

A behaviour present in A3, partial in B100k, absent in Bfinal is a behaviour
TRAINING DESTROYED -- a far more specific repair target than a static difference.

This measures WHAT the policies do, not whether they win. No delta is computed
and no gate claim is possible from this script.

    python -m experiments.eval_pole_b_behavioral_diagnosis [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LABEL = "POLE_B_BEHAVIORAL_DIAGNOSIS"
SPEC = SD / f"{LABEL}_SPEC.json"
EXPERIMENT_ID = LABEL

SEED_LO, N_SEEDS = 18_600_001, 32
SCALE = ROOT / "artifacts" / "scale_4v4_specialists"
POLE_B_GENOME = SD / "pole_b2_candidates" / "B3-3_lockdef10_2v1.json"

MACRO_NAMES = {0: "GO_TO", 1: "GRAB_MINE", 2: "GET_FLAG", 3: "PLACE_MINE", 4: "GO_HOME"}
DEFEND_RADIUS = 4.0
PRESSURE_RADIUS = 3.0

POLICIES = [
    ("A3", SCALE / "pi_A_specialist_4v4_b3_entity_repair/ckpts/final_pi_A_specialist_4v4_b3_entity_repair.zip"),
    ("B100k", SCALE / "pi_B_specialist_4v4_b3_entity_repair_corrected/ckpts/ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_100000.zip"),
    ("Bfinal", SCALE / "pi_B_specialist_4v4_b3_entity_repair_corrected/ckpts/final_pi_B_specialist_4v4_b3_entity_repair_corrected.zip"),
]
BASE_KEY = {"A": "OP6", "B": "OP7"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def _pole_genome(pole: str):
    from experiments.opponent_spec import _with_full_team_defender_gate, pole_A_genome
    from experiments.sds_genome import SDSGenome
    if pole == "A":
        return pole_A_genome(4)
    return _with_full_team_defender_gate(
        SDSGenome.from_dict(json.loads(POLE_B_GENOME.read_text(encoding="utf-8"))), 4)


def _build_env(device, seed, pole, genome):
    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import assert_live_opponent_batch, install_keyed_opponent_overlays
    from rl.curriculum import phase_from_tag
    R2.AGENTS = 4
    key = BASE_KEY[pole]
    genomes = {key: genome}
    env = R2.build_env(device, seed)
    core = env.core
    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0
    install_keyed_opponent_overlays(core, genomes)
    env.env_method("set_phase", phase_from_tag(key))
    env.env_method("set_next_opponent", "SCRIPTED", key)
    obs = env.reset()
    obs["global_state"] = env.state()
    assert_live_opponent_batch(core, genomes, allowed_keys=(key,), context=f"{LABEL} {pole} {seed}")
    return env, core, obs


def _snapshot(core) -> dict:
    """Per-step world state. Only fields verified present before the spec froze."""
    bx, by = _np(core.blue_x).reshape(-1), _np(core.blue_y).reshape(-1)
    rx, ry = _np(core.red_x).reshape(-1), _np(core.red_y).reshape(-1)
    bf = _np(core.blue_flag_pos).reshape(-1)
    rf = _np(core.red_flag_pos).reshape(-1)
    return {
        "bx": bx, "by": by, "rx": rx, "ry": ry,
        "own_flag": (float(bf[0]), float(bf[1])),
        "enemy_flag": (float(rf[0]), float(rf[1])),
        "b_tagged": _np(core.blue_tagged).reshape(-1).astype(bool),
        "b_carrying": _np(core.blue_carrying).reshape(-1).astype(bool),
        "b_alive": _np(core.blue_alive).reshape(-1).astype(bool),
        "r_tagged": _np(core.red_tagged).reshape(-1).astype(bool),
        "blue_score": int(_np(core.blue_score).reshape(-1)[0]),
        "red_score": int(_np(core.red_score).reshape(-1)[0]),
    }


def _step_features(action: np.ndarray, st: dict) -> dict:
    """Behavioural features for ONE decision step, per the frozen spec."""
    a = np.asarray(action).reshape(-1)
    macros = [int(a[2 * i]) for i in range(4)]
    targets = [int(a[2 * i + 1]) for i in range(4)]
    cmds = list(zip(macros, targets))

    bx, by, rx, ry = st["bx"], st["by"], st["rx"], st["ry"]
    ofx, ofy = st["own_flag"]
    efx, efy = st["enemy_flag"]

    d_own = np.hypot(bx - ofx, by - ofy)
    d_enemy = np.hypot(bx - efx, by - efy)

    pair = [float(np.hypot(bx[i] - bx[j], by[i] - by[j]))
            for i in range(4) for j in range(i + 1, 4)]

    # 2v1: how many reds are within PRESSURE_RADIUS of each blue
    reds_near = []
    for i in range(4):
        reds_near.append(int(np.sum(np.hypot(rx - bx[i], ry - by[i]) <= PRESSURE_RADIUS)))
    pressured = [i for i in range(4) if reds_near[i] >= 2]
    support_dist = []
    for i in pressured:
        others = [j for j in range(4) if j != i]
        support_dist.append(float(min(np.hypot(bx[i] - bx[j], by[i] - by[j]) for j in others)))

    return {
        "macros": macros, "targets": targets,
        "all_identical_cmd": int(len(set(cmds)) == 1),
        "distinct_cmds": len(set(cmds)),
        "distinct_macros": len(set(macros)),
        "n_get_flag": sum(1 for m in macros if m == 2),
        "n_go_home": sum(1 for m in macros if m == 4),
        "n_go_to": sum(1 for m in macros if m == 0),
        "n_mine": sum(1 for m in macros if m in (1, 3)),
        "mean_pair_dist": float(np.mean(pair)),
        "n_defenders": int(np.sum(d_own <= DEFEND_RADIUS)),
        "n_attackers": int(np.sum(d_enemy <= DEFEND_RADIUS)),
        "mean_d_own_flag": float(np.mean(d_own)),
        "mean_d_enemy_flag": float(np.mean(d_enemy)),
        "n_tagged": int(np.sum(st["b_tagged"])),
        "any_carrying": int(np.any(st["b_carrying"])),
        "n_pressured": len(pressured),
        "mean_support_dist": (float(np.mean(support_dist)) if support_dist else float("nan")),
    }


def run_cell(inference, label, pole, seeds, device, genome, w, fh, collect_states=False):
    from gpu_env._core._entity_obs import augment_obs_with_entities
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    import experiments.r2_learned_crossover as R2

    per_step, per_ep, saved_states = [], [], []
    bar = tqdm_iter(seeds, desc=f"{LABEL} {label}@Pole{pole}", unit="ep")
    for seed in bar:
        set_postfix(bar, f"seed={seed}")
        env, core, obs = _build_env(device, seed, pole, genome)
        try:
            inference.reset_strategy()
            obs = augment_obs_with_entities(obs, core, side="blue")
            prev_macros, switches, steps = None, 0, 0
            ep_rows = []
            for t in range(R2.MAX_STEPS):
                st = _snapshot(core)
                action, _ = inference.predict(obs, deterministic=True)
                f = _step_features(action, st)
                if prev_macros is not None:
                    switches += sum(1 for i in range(4) if f["macros"][i] != prev_macros[i])
                prev_macros = f["macros"]
                steps += 1
                ep_rows.append(f)
                if collect_states and (t % 8 == 0):
                    saved_states.append({"seed": seed, "t": t,
                                         "obs": {k: np.copy(np.asarray(v)) for k, v in obs.items()
                                                 if k != "global_state"},
                                         "global_state": np.copy(np.asarray(obs["global_state"]))})
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                obs = augment_obs_with_entities(obs, core, side="blue")
                if bool(np.asarray(done).any()):
                    # The vec env auto-resets on episode completion INSIDE
                    # step_wait(), so core.blue_score/red_score are already
                    # zeroed for the NEXT episode by the time we could read them
                    # here. The terminal score lives only in `info`, captured at
                    # this exact instant -- matches the already-validated
                    # pattern in eval_specialist_crossover_scaled.py::run_cell.
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    res = (i0 or {}).get("episode_result") or {}
                    terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                    break
            else:
                terminal = None
            if terminal is None:
                # Timeout: no done ever fired, so core's live score is the
                # real terminal state, not a post-reset artifact.
                st_end = _snapshot(core)
                terminal = (st_end["blue_score"], st_end["red_score"])
            blue_score, red_score = terminal
            won = int(blue_score > red_score)
            agg = {k: float(np.nanmean([r[k] for r in ep_rows])) for k in ep_rows[0]
                   if k not in ("macros", "targets")}
            agg.update({"policy": label, "pole": pole, "seed": seed, "steps": steps,
                        "macro_switch_rate": switches / max(1, (steps - 1) * 4),
                        "won": won, "blue_score": blue_score, "red_score": red_score})
            per_ep.append(agg)
            w.writerow(agg)
            fh.flush()
            per_step.extend(ep_rows)
        finally:
            env.close()
    return per_ep, per_step, saved_states


def summarize(per_ep, per_step) -> dict:
    keys = [k for k in per_ep[0] if k not in ("policy", "pole", "seed")]
    out = {k: float(np.nanmean([e[k] for e in per_ep])) for k in keys}
    macro_counts = Counter()
    for r in per_step:
        for m in r["macros"]:
            macro_counts[MACRO_NAMES.get(m, str(m))] += 1
    total = max(1, sum(macro_counts.values()))
    out["macro_fraction"] = {k: macro_counts[k] / total for k in sorted(macro_counts)}
    out["n_agent_steps"] = total
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from experiments.pole_attestation import (
        assert_resolved_matches_certification, governing_certification, resolve_pole_genome,
    )
    from experiments.run_lock import RunLock
    from rl.custom_ppo import load_custom_ppo_policy

    if not SPEC.is_file():
        raise SystemExit(f"REFUSING: frozen spec missing: {SPEC}")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")

    out_path = SD / f"{LABEL}_RESULT.json"
    rows_csv = SD / f"{LABEL.lower()}_episode_rows.csv"
    if not args.dry_run:
        import experiments.seed_registry as R
        doc = R.load()
        b = next((x for x in doc["blocks"] if x["experiment_id"] == EXPERIMENT_ID), None)
        if b is None or b["lo"] != SEED_LO or b["hi"] != SEED_LO + N_SEEDS - 1:
            raise SystemExit(f"FAIL-CLOSED (Rule 9): seed block for {EXPERIMENT_ID!r} not "
                             f"registered as {SEED_LO}..{SEED_LO + N_SEEDS - 1}")
        if out_path.is_file() or rows_csv.is_file():
            raise SystemExit(f"REFUSING: output for {LABEL} already exists; one-shot")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    seeds = list(range(SEED_LO, SEED_LO + N_SEEDS))
    _v, cert_path = governing_certification(4)

    print(f"{LABEL}  {_now()}  device={device}")
    print(f"  spec      {SPEC.name} [{spec.get('status')}]")
    print(f"  question  {spec.get('THE_QUESTION')}")
    print(f"  seeds     {seeds[0]}..{seeds[-1]} (n={len(seeds)}), SHARED across all 6 cells")

    # Rule 14: attest BOTH poles before any episode.
    genomes, attestations = {}, {}
    for pole in ("A", "B"):
        g = resolve_pole_genome(pole, 4, str(POLE_B_GENOME) if pole == "B" else None)
        attestations[pole] = assert_resolved_matches_certification(pole, 4, cert_path, g, is_smoke=False)
        genomes[pole] = g
        print(f"  pole {pole}    {attestations[pole]['live_genome_id']} "
              f"MATCH={'PASS' if attestations[pole]['hashes_match'] else 'FAIL'}")

    for lbl, p in POLICIES:
        if not p.is_file():
            raise SystemExit(f"REFUSING: checkpoint missing for {lbl}: {p}")
        print(f"  {lbl:7s}   sha {_sha(p)[:12]}...")

    env, _c, _o = _build_env(device, seeds[0], "B", genomes["B"])
    obs_space, act_space = env.observation_space, env.action_space
    env.close()
    policies = {lbl: load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
                for lbl, p in POLICIES}
    for pol in policies.values():
        pol.model.eval()

    # ---- known-answer contracts -------------------------------------------------
    shas = {lbl: _sha(p) for lbl, p in POLICIES}
    contracts = {
        "K1_three_distinct_checkpoints": len(set(shas.values())) == 3,
        "K2_all_have_entity_encoder": all(policies[l].model.entity_encoder is not None
                                          for l, _ in POLICIES),
        "K3_action_decode_shape": tuple(policies["A3"].model.per_agent_action_dims) == (5, 50),
        "K4_pole_attestations_passed": all(a["hashes_match"] for a in attestations.values()),
    }
    print("\n  known-answer contracts ...")
    for k, v in contracts.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    if not all(contracts.values()):
        raise SystemExit(f"FAIL-CLOSED: contracts failed "
                         f"{[k for k, v in contracts.items() if not v]}")

    if args.dry_run:
        print("\n  --dry-run: spec frozen, 3 distinct checkpoints, both poles attested, action "
              "decode verified. NO episodes run, NOTHING written.")
        return 0

    fields = ["policy", "pole", "seed", "steps", "won", "blue_score", "red_score",
              "macro_switch_rate", "all_identical_cmd", "distinct_cmds", "distinct_macros",
              "n_get_flag", "n_go_home", "n_go_to", "n_mine", "mean_pair_dist",
              "n_defenders", "n_attackers", "mean_d_own_flag", "mean_d_enemy_flag",
              "n_tagged", "any_carrying", "n_pressured", "mean_support_dist"]

    summaries, bfinal_states = {}, []
    with RunLock(SD / f"{LABEL}.run.lock", run_id=LABEL):
        with rows_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for pole in ("B", "A"):
                for lbl, _p in POLICIES:
                    collect = (pole == "B" and lbl == "Bfinal")
                    per_ep, per_step, states = run_cell(
                        policies[lbl], lbl, pole, seeds, device, genomes[pole], w, fh,
                        collect_states=collect)
                    if collect:
                        bfinal_states = states
                    s = summarize(per_ep, per_step)
                    summaries[f"{lbl}@Pole{pole}"] = s
                    print(f"\n  {lbl}@Pole{pole}: win={s['won']:.3f} "
                          f"identical_cmd={s['all_identical_cmd']:.3f} "
                          f"distinct_cmds={s['distinct_cmds']:.2f} "
                          f"switch={s['macro_switch_rate']:.3f} "
                          f"spread={s['mean_pair_dist']:.2f} "
                          f"def={s['n_defenders']:.2f} att={s['n_attackers']:.2f}", flush=True)
                    print(f"      macros: { {k: round(v,3) for k,v in s['macro_fraction'].items()} }")
                    out_path.write_text(json.dumps(
                        {"record": f"{LABEL} (partial)", "status": "RUNNING", "utc": _now(),
                         "summaries": summaries}, indent=2), encoding="utf-8")

    # ---- counterfactual: replay Bfinal's Pole-B states through the other policies --
    counterfactual = {}
    if bfinal_states:
        print(f"\n  counterfactual replay over {len(bfinal_states)} Bfinal Pole-B states ...",
              flush=True)
        base_macros = {}
        for other in ("Bfinal", "A3", "B100k"):
            agree, subs, total = 0, Counter(), 0
            for i, s in enumerate(bfinal_states):
                o = dict(s["obs"]); o["global_state"] = s["global_state"]
                a, _ = policies[other].predict(o, deterministic=True)
                m = [int(np.asarray(a).reshape(-1)[2 * k]) for k in range(4)]
                if other == "Bfinal":
                    base_macros[i] = m
                    continue
                for k in range(4):
                    total += 1
                    if m[k] == base_macros[i][k]:
                        agree += 1
                    else:
                        subs[f"{MACRO_NAMES.get(base_macros[i][k])}->{MACRO_NAMES.get(m[k])}"] += 1
            if other != "Bfinal":
                counterfactual[other] = {
                    "macro_agreement_with_Bfinal": agree / max(1, total),
                    "n_agent_states": total,
                    "top_substitutions": dict(subs.most_common(8)),
                }
                print(f"    {other}: macro agreement with Bfinal = "
                      f"{counterfactual[other]['macro_agreement_with_Bfinal']:.3f}")
                print(f"      top substitutions: {counterfactual[other]['top_substitutions']}")

    out_path.write_text(json.dumps({
        "record": f"{LABEL} behavioural diagnosis on Pole B (with Pole-A reference)",
        "status": "COMPLETE_DIAGNOSTIC", "utc": _now(), "device": device,
        "arm": "DIAGNOSTIC", "confirmatory": False,
        "implements": f"{SPEC.name}",
        "question": spec.get("THE_QUESTION"),
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds), "shared_across_cells": True},
        "checkpoints": shas, "contracts": contracts,
        "pole_attestations": {p: {k: a[k] for k in ("certified_genome_id", "live_genome_id",
                                                    "certified_config_hash", "live_config_hash",
                                                    "hashes_match")}
                              for p, a in attestations.items()},
        "summaries": summaries,
        "counterfactual_replay_on_Bfinal_PoleB_states": counterfactual,
        "NOT_A_CLAIM": spec.get("WHAT_THIS_EXPERIMENT_MAY_NOT_CLAIM"),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {out_path}\n  -> {rows_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
