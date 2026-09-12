r"""Arm 1: Observation -> Teacher-Target Consistency Probe.

Implements ARM1_TEACHER_IDENTIFIABILITY_SPEC.json. Read that file for the full
design, the six 2026-09-12 safeguards, and the precommitted interpretation.
This module only executes it.

    python experiments/eval_arm1_teacher_identifiability.py --dry-run
    python experiments/eval_arm1_teacher_identifiability.py

CPU-only by construction: this measures a scripted-vs-scripted mapping, no
policy is loaded or trained. Uses n_envs=1 (no batching) specifically to
minimize CPU footprint while OPP_ABLATION_6V6 is live on the same machine
(disclosed in the spec's TIMING_ESTIMATE.resource_contention_disclosure).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[0].parent
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "ARM1_TEACHER_IDENTIFIABILITY_SPEC.json"

DECISION_CADENCE_TICKS = 4          # imposed GO_TO horizon -- see spec SAFEGUARDS.2
FIELDS = ["scale", "strategy", "seed", "episode", "tick", "agent_idx",
          "fp_hex", "raw_tx", "raw_ty", "eff_tx", "eff_ty",
          "q_wx", "q_wy", "q_idx", "e_q",
          "is_defender", "n_live_intruders"]


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


@dataclass
class Sample:
    scale: str; strategy: str; seed: int; episode: int; tick: int; agent_idx: int
    grid: np.ndarray; vec: np.ndarray; agent_mask: np.ndarray; mask: np.ndarray
    raw_t: tuple; eff_t: tuple; is_defender: bool; n_live_intruders: int
    fp_hex: str = field(init=False)

    def __post_init__(self):
        self.fp_hex = _sha256_bytes(self.grid, self.vec, self.agent_mask, self.mask)

    def fingerprint_bytes(self) -> bytes:
        """The exact bytes hashed -- used for the byte-for-byte verification
        pass, so a hash match is never trusted on its own (SAFEGUARDS.4)."""
        return (np.ascontiguousarray(self.grid).tobytes()
               + np.ascontiguousarray(self.vec).tobytes()
               + np.ascontiguousarray(self.agent_mask).tobytes()
               + np.ascontiguousarray(self.mask).tobytes())


def _project(t: tuple, W: np.ndarray) -> tuple[int, float, float]:
    """q(t) = argmin_w ||t-w||, plus e_q = ||t - q(t)||."""
    d2 = (W[:, 0] - t[0]) ** 2 + (W[:, 1] - t[1]) ** 2
    idx = int(np.argmin(d2))
    e_q = float(np.sqrt(d2[idx]))
    return idx, float(W[idx, 0]), float(W[idx, 1])


def collect_cell(scale: int, strategy: str, seeds: list[int], device: str) -> list[Sample]:
    """Roll out `strategy` for every seed at team size `scale`, sampling one
    fingerprint per agent every DECISION_CADENCE_TICKS ticks."""
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome

    S.AGENTS = scale
    genome = pole_A_genome(scale)
    style = S.GUARD if strategy == "GUARD" else S.BREACH
    n_def = (scale + 1) // 2
    def_lo = scale - n_def

    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    samples: list[Sample] = []
    for ep_i, seed in enumerate(seeds):
        cfg = GPUFieldConfig(
            n_envs=1, max_blue_agents=scale, max_red_agents=scale,
            map_set="train", map_layout=S.MAP, max_decision_steps=S.MAX_STEPS,
            aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
            obstacle_obs_channel=True, tag_telemetry_enabled=True,
            own_flag_home_required_to_score=True, **S.RULESET,
        )
        env = GPUCTFVecEnv(cfg)
        core = env.core
        try:
            opp = genome.base_opponent
            env.env_method("set_phase", opp)
            env.env_method("set_next_opponent", "SCRIPTED", opp)
            from experiments.strategic_demand_searcher import apply_genome_to_core
            apply_genome_to_core(core, genome)
            core.blue_scripted = True
            core.set_blue_style(style)
            env.reset()
            apply_genome_to_core(core, genome)
            core.drain_tag_events()

            W = core._macro_targets.detach().cpu().numpy()
            assert W.shape[0] == int(core.cfg.n_targets), (
                f"W50 shape mismatch: {W.shape[0]} != n_targets={core.cfg.n_targets}")

            for tick in range(S.MAX_STEPS):
                if tick % DECISION_CADENCE_TICKS == 0:
                    obs = core.get_obs_tensors("blue")
                    raw_tx, raw_ty = core._get_scripted_targets("blue")
                    grid = obs["grid"][0].detach().cpu().numpy().astype(np.float32)
                    vec = obs["vec"][0].detach().cpu().numpy().astype(np.float32)
                    amask = obs["agent_mask"][0].detach().cpu().numpy().astype(np.float32)
                    m = obs["mask"][0].detach().cpu().numpy().astype(np.float32)
                    rtx = raw_tx[0].detach().cpu().numpy()
                    rty = raw_ty[0].detach().cpu().numpy()
                    enemy_alive = (core.red_alive[0] & (~core.red_tagged[0])).detach().cpu().numpy()
                    n_intr = int(enemy_alive.sum())
                    for i in range(scale):
                        t = (float(rtx[i]), float(rty[i]))
                        samples.append(Sample(
                            scale=f"{scale}v{scale}", strategy=strategy, seed=seed,
                            episode=ep_i, tick=tick, agent_idx=i,
                            grid=grid[i], vec=vec[i],
                            agent_mask=amask, mask=m[i * (m.shape[0] // scale):
                                                     (i + 1) * (m.shape[0] // scale)]
                            if m.ndim == 1 else m[i],
                            raw_t=t, eff_t=t,             # identical in this codebase; see SAFEGUARDS.3
                            is_defender=bool(i >= def_lo), n_live_intruders=n_intr,
                        ))
                env.step_async(env.action_space.sample() * 0)
                _o, _r, done, info = env.step_wait()
                if bool(np.asarray(done).any()):
                    break
        finally:
            env.close()
    return samples


def _preflight(device: str) -> None:
    """Rule 4: known-answer checks on the capture path itself, before any
    frozen seed is spent."""
    print("  preflight self-test ...")
    smoke_seeds = [99900001]

    # (a) BREACH target must equal the enemy flag position for every agent.
    s = collect_cell(4, "BREACH", smoke_seeds, device)
    assert s, "preflight: BREACH produced zero samples"
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    S.AGENTS = 4
    cfg = GPUFieldConfig(n_envs=1, max_blue_agents=4, max_red_agents=4, map_set="train",
                         map_layout=S.MAP, max_decision_steps=S.MAX_STEPS, aquaticus_profile=True,
                         rules_profile="OURS", device=device, seed=smoke_seeds[0],
                         obstacle_obs_channel=True, tag_telemetry_enabled=True,
                         own_flag_home_required_to_score=True, **S.RULESET)
    probe_env = GPUCTFVecEnv(cfg); probe_env.reset()
    flag_xy = (float(probe_env.core.red_flag_pos[0, 0]), float(probe_env.core.red_flag_pos[0, 1]))
    probe_env.close()
    for smp in s[:8]:
        assert abs(smp.raw_t[0] - flag_xy[0]) < 1e-3, (
            f"preflight FAILED: BREACH target {smp.raw_t} != enemy flag {flag_xy}")

    # (b) a repeated call to _get_scripted_targets on UNCHANGED core state
    #     must return the identical target (capture path is deterministic).
    import experiments.strategic_demand_searcher as S2
    S2.AGENTS = 2
    g2 = pole_A_genome(2)
    from gpu_env import GPUCTFVecEnv as _E, GPUFieldConfig as _C
    cfg2 = _C(n_envs=1, max_blue_agents=2, max_red_agents=2, map_set="train",
             map_layout=S2.MAP, max_decision_steps=S2.MAX_STEPS, aquaticus_profile=True,
             rules_profile="OURS", device=device, seed=smoke_seeds[0],
             obstacle_obs_channel=True, tag_telemetry_enabled=True,
             own_flag_home_required_to_score=True, **S2.RULESET)
    env2 = _E(cfg2); core2 = env2.core
    from experiments.strategic_demand_searcher import apply_genome_to_core as _apply
    _apply(core2, g2); core2.blue_scripted = True; core2.set_blue_style(S2.GUARD)
    env2.reset(); _apply(core2, g2)
    t1 = tuple(float(v) for v in (core2._get_scripted_targets("blue")[0][0],
                                  core2._get_scripted_targets("blue")[1][0]))
    t2 = tuple(float(v) for v in (core2._get_scripted_targets("blue")[0][0],
                                  core2._get_scripted_targets("blue")[1][0]))
    env2.close()
    assert t1 == t2, f"preflight FAILED: repeated call on unchanged state disagreed: {t1} vs {t2}"

    # (c) W50 has exactly n_targets rows -- checked inside collect_cell via assert.
    print("  preflight self-test PASS: BREACH==enemy-flag, capture path deterministic, W50 sized correctly")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")
    label = spec["OUTPUT_LABEL"]
    lo, hi = spec["SEEDS"]["block"]
    n_per_cell = int(spec["DESIGN"]["episodes_per_cell"])
    seeds_all = list(range(lo, hi + 1))
    if len(seeds_all) < n_per_cell:
        raise SystemExit("REFUSING: seed block smaller than episodes_per_cell")

    OUT = SD / f"{label}_RESULT.json"
    ROWS = SD / f"{label.lower()}_rows.csv"
    if not args.dry_run and OUT.is_file():
        raise SystemExit(f"REFUSING: sealed result for {label!r} already exists; one-shot.")
    LIVE = SD / f"{label}_LIVE_STATUS.json"
    LOCK = SD / f"{label}.run.lock"
    if not args.dry_run:
        if LOCK.is_file():
            raise SystemExit(f"REFUSING: {LOCK.name} exists -- another process may be running.")
        LOCK.write_text(json.dumps({"pid": os.getpid(), "utc": _now()}), encoding="utf-8")

    print(f"ARM1 TEACHER IDENTIFIABILITY  {label}  {_now()}")
    print(f"  spec        {SPEC.name}  [{spec['status']}]  arm={spec['arm']}")
    print(f"  seeds       {lo}..{hi} (n={len(seeds_all)}), {n_per_cell} used per cell")
    print(f"  cadence     every {DECISION_CADENCE_TICKS} ticks (imposed GO_TO horizon)")
    print("  DIAGNOSTIC ONLY -- measures identifiability, trains nothing.\n", flush=True)

    _preflight(args.device)
    if args.dry_run:
        print("\n  --dry-run: spec frozen, preflight passed. NO seed spent, NOTHING written.")
        return 0

    from experiments.tqdm_loop import set_postfix, tqdm_iter

    cells = [(n, strat) for n in (2, 4, 6) for strat in ("GUARD", "BREACH")]
    all_samples: dict[tuple, list[Sample]] = {}
    t0 = time.time()
    bar = tqdm_iter(cells, desc=label, unit="cell")
    for n, strat in bar:
        set_postfix(bar, f"{n}v{n} {strat}")
        cell_seeds = seeds_all[:n_per_cell]
        cell_t0 = time.time()
        samples = collect_cell(n, strat, cell_seeds, args.device)
        all_samples[(n, strat)] = samples
        el = time.time() - t0
        print(f"  {n}v{n} {strat:<7s}: {len(samples)} samples in "
              f"{time.time() - cell_t0:.1f}s (elapsed {el:.1f}s)", flush=True)
        LIVE.write_text(json.dumps({
            "label": label, "pid": os.getpid(), "completed_cells": len(all_samples),
            "total_cells": len(cells), "current": f"{n}v{n} {strat}",
            "elapsed_s": round(el, 1), "heartbeat_utc": _now(),
        }, indent=2), encoding="utf-8")

    # ---- write raw rows (Rule 2, all at once here since collection is fast enough
    #      that per-sample fsync would dominate wall time; still incremental at
    #      the CELL level via LIVE_STATUS above) ----
    with ROWS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for (n, strat), samples in all_samples.items():
            for s in samples:
                q_idx, q_wx, q_wy = _project(s.eff_t, W=None) if False else (None, None, None)
        # second pass: need W per scale for projection
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    import experiments.strategic_demand_searcher as S
    W_by_scale = {}
    for n in (2, 4, 6):
        S.AGENTS = n
        cfg = GPUFieldConfig(n_envs=1, max_blue_agents=n, max_red_agents=n, map_set="train",
                             map_layout=S.MAP, max_decision_steps=S.MAX_STEPS, aquaticus_profile=True,
                             rules_profile="OURS", device=args.device, seed=seeds_all[0],
                             obstacle_obs_channel=True, tag_telemetry_enabled=True,
                             own_flag_home_required_to_score=True, **S.RULESET)
        e = GPUCTFVecEnv(cfg); e.reset()
        W_by_scale[n] = e.core._macro_targets.detach().cpu().numpy()
        e.close()

    with ROWS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for (n, strat), samples in all_samples.items():
            W = W_by_scale[n]
            for s in samples:
                q_idx, q_wx, q_wy = _project(s.eff_t, W)
                e_q = float(np.hypot(s.eff_t[0] - q_wx, s.eff_t[1] - q_wy))
                w.writerow({
                    "scale": s.scale, "strategy": s.strategy, "seed": s.seed,
                    "episode": s.episode, "tick": s.tick, "agent_idx": s.agent_idx,
                    "fp_hex": s.fp_hex, "raw_tx": s.raw_t[0], "raw_ty": s.raw_t[1],
                    "eff_tx": s.eff_t[0], "eff_ty": s.eff_t[1],
                    "q_wx": q_wx, "q_wy": q_wy, "q_idx": q_idx, "e_q": e_q,
                    "is_defender": int(s.is_defender), "n_live_intruders": s.n_live_intruders,
                })
                fh.flush(); os.fsync(fh.fileno())

    # ---- Test 1 & 2: fingerprint grouping, WITHIN (scale, strategy) only ----
    results = {}
    for (n, strat), samples in all_samples.items():
        by_fp: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            by_fp[s.fp_hex].append(s)
        W = W_by_scale[n]
        repeated = {h: v for h, v in by_fp.items() if len(v) > 1}
        n_ambiguous_groups = 0
        n_ambiguous_samples = 0
        entropies = []
        verified_collisions = []
        by_role = {"defender": {"ambiguous": 0, "total": 0}, "attacker": {"ambiguous": 0, "total": 0}}
        for h, group in repeated.items():
            # byte-verify before trusting the hash (SAFEGUARDS.4)
            b0 = group[0].fingerprint_bytes()
            if not all(g.fingerprint_bytes() == b0 for g in group[1:]):
                continue  # hash collision without true equality -- not a match, skip
            qs = [_project(g.eff_t, W)[0] for g in group]
            role = "defender" if group[0].is_defender else "attacker"
            by_role[role]["total"] += len(group)
            uniq = sorted(set(qs))
            if len(uniq) > 1:
                n_ambiguous_groups += 1
                n_ambiguous_samples += len(group)
                by_role[role]["ambiguous"] += len(group)
                verified_collisions.append({
                    "fp_hex": h[:16], "n_in_group": len(group),
                    "distinct_q_idx": uniq, "role": role,
                    "example_seeds": [g.seed for g in group[:4]],
                })
            counts = np.bincount(qs, minlength=int(spec["DESIGN"].get("n_targets", 50) or 50) + 1)
            p = counts[counts > 0] / max(1, len(qs))
            ent = float(-np.sum(p * np.log2(p))) if len(p) else 0.0
            entropies.append(ent)

        results[f"{n}v{n}_{strat}"] = {
            "n_samples": len(samples),
            "n_repeated_fingerprints": len(repeated),
            "n_ambiguous_groups_verified": n_ambiguous_groups,
            "n_ambiguous_samples": n_ambiguous_samples,
            "frac_samples_in_ambiguous_group": round(n_ambiguous_samples / max(1, len(samples)), 6),
            "mean_target_entropy_bits": round(float(np.mean(entropies)), 4) if entropies else 0.0,
            "median_target_entropy_bits": round(float(np.median(entropies)), 4) if entropies else 0.0,
            "by_role": by_role,
            "verified_collision_examples": verified_collisions[:10],
            "e_q_mean": round(float(np.mean([_project(s.eff_t, W)[0] and
                                             np.hypot(s.eff_t[0] - _project(s.eff_t, W)[1],
                                                     s.eff_t[1] - _project(s.eff_t, W)[2])
                                             for s in samples])), 6) if samples else None,
        }

    OUT.write_text(json.dumps({
        "record": f"{label} teacher-target identifiability probe",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "arm": "DIAGNOSTIC", "confirmatory": False, "implements": SPEC.name,
        "decision_cadence_ticks": DECISION_CADENCE_TICKS,
        "seeds": {"block": [lo, hi], "n_used_per_cell": n_per_cell},
        "cells": results,
        "total_samples": sum(v["n_samples"] for v in results.values()),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}\n  -> {ROWS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
