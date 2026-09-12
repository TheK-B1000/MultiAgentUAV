"""6v6 opponent-channel ablation -- exploratory mechanism diagnostic.

Implements 6V6_OPPONENT_CHANNEL_ABLATION_SPEC.json.

QUESTION
--------
Do the frozen 6v6 specialists materially depend on opponent-position information
at inference time? The track-hold experiments showed no detectable loss when
opponent tracks were merely AGED (frozen at t=0). This destroys the information
instead of aging it.

WHERE OPPONENT POSITION REACHES THE ACTION (audited, not assumed)
-----------------------------------------------------------------
Exactly two paths:
  * grid channel 2   -- _build_grid_obs, _scatter_points(grid[:,i], 2, ex, ey, elive)
  * vec feature 11   -- _build_vec_obs, "nearest enemy distance" normalised to [0,1]

Everything else in the vec schema is own-team / flag / mine / pickup / time /
agent-id / score. _build_action_mask reads no enemy field, so legality is
unaffected.

The GLOBAL STATE does carry opponent information (rl/global_state.py reads
core.red_x/red_y), but policy.act() uses it only for self.values() -- the critic.
Action selection comes from policy_logits(obs) masked by obs["mask"]. So at
inference the actionable opponent surface is exactly the two paths above, and
global_state is deliberately left untouched.

WHY "ZERO" IS NOT ZERO
----------------------
Setting vec feature 11 to 0.0 would mean "an enemy is at zero distance" --
maximally alarming, the OPPOSITE of absent. The schema's own convention for an
absent object is visible in features 16/17 (pickups, friendly mines): when the
mask is inactive the distance is set to max_dist, normalising to 1.0. ZERO_OPP
therefore uses 1.0 there, and an empty channel 2 (which is exactly what the
env's native sensor dropout already produces).

RNG DISCIPLINE
--------------
RANDOM_OPP draws from a SEPARATE numpy Generator seeded per (seed, condition).
The env's native sensor model would have drawn from the env RNG, desynchronising
ablated episodes from FULL and destroying the paired design.

Run:
  ./.venv/Scripts/python.exe experiments/eval_opponent_ablation_6v6.py --dry-run
  ./.venv/Scripts/python.exe experiments/eval_opponent_ablation_6v6.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "6V6_OPPONENT_CHANNEL_ABLATION_SPEC.json"

N = 6
BASE_KEY = {"A": "OP6", "B": "OP7"}
ENEMY_GRID_CHANNEL = 2
NEAREST_ENEMY_VEC_IDX = 11
CONDITIONS = ("FULL", "ZERO_OPP", "RANDOM_OPP")
N_BOOT, ALPHA, BOOT_SEED = 20_000, 0.05, 7


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mean_ci(v):
    v = np.asarray(v, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    b = v[idx].mean(axis=1)
    lo, hi = np.percentile(b, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": round(float(v.mean()), 6),
            "lcb95": round(float(lo), 6), "ucb95": round(float(hi), 6)}


class OppAblation:
    """Remove or corrupt opponent-position information in the OBSERVATION only.

    ZERO_OPP post-processes the two builder outputs (surgical and exactly the
    schema's absent-object encoding). RANDOM_OPP substitutes enemy_x/enemy_y at
    the _side_tensors seam so the random positions flow through the unmodified
    pipeline and yield a plausible channel-2 scatter and a correctly derived
    feature 11.

    The _side_tensors substitution is gated to fire ONLY inside the two
    observation builders: _mines.py and _rules.py also call _side_tensors, and a
    blanket patch would corrupt mine mechanics and win determination.
    """

    def __init__(self, core, condition: str, seed: int, cols: float, rows: float):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.core, self.condition, self.seed = core, condition, seed
        self.cols, self.rows = float(cols), float(rows)
        self._installed = False
        self._active = False
        self._rng = np.random.default_rng(seed + 550_001)  # NOT the env RNG
        self._cycle_xy = None   # synthetic positions shared within one observation cycle
        self._orig_side = core._side_tensors
        self._orig_grid = core._build_grid_obs
        self._orig_vec = core._build_vec_obs
        self._orig_get = core.get_obs_tensors

    # ---- RANDOM_OPP: substitute positions upstream of every derived feature ----
    def _side_tensors(self, side: str):
        d = self._orig_side(side)
        if side == "blue" and self._active and self.condition == "RANDOM_OPP":
            d = dict(d)
            if self._cycle_xy is None:
                # ONE synthetic opponent set per observation cycle. Both the grid
                # scatter and the nearest-enemy distance must derive from the SAME
                # set, or the two channels would describe different worlds.
                shape = tuple(d["enemy_x"].shape)
                self._cycle_xy = (
                    d["enemy_x"].new_tensor(self._rng.uniform(0.0, self.cols, size=shape)),
                    d["enemy_y"].new_tensor(self._rng.uniform(0.0, self.rows, size=shape)),
                )
            d["enemy_x"], d["enemy_y"] = self._cycle_xy
        return d

    def _get_obs_tensors(self, *a, **kw):
        """Start a new observation cycle: redraw the synthetic opponent set once."""
        self._cycle_xy = None
        return self._orig_get(*a, **kw)

    # ---- ZERO_OPP: post-process exactly the two carriers ----
    def _grid(self, *a, **kw):
        self._active = True
        try:
            g = self._orig_grid(*a, **kw)
        finally:
            self._active = False
        side = kw.get("side", a[0] if a else "blue")
        if side == "blue" and self.condition == "ZERO_OPP":
            g = g.clone()
            g[:, :, ENEMY_GRID_CHANNEL, :, :] = 0.0
        return g

    def _vec(self, *a, **kw):
        self._active = True
        try:
            v = self._orig_vec(*a, **kw)
        finally:
            self._active = False
        side = kw.get("side", a[0] if a else "blue")
        if side == "blue" and self.condition == "ZERO_OPP":
            v = v.clone()
            v[..., NEAREST_ENEMY_VEC_IDX] = 1.0   # schema's "no such object" value
        return v

    def install(self):
        if self.condition == "FULL":
            return self
        self.core._side_tensors = self._side_tensors
        self.core._build_grid_obs = self._grid
        self.core._build_vec_obs = self._vec
        self.core.get_obs_tensors = self._get_obs_tensors
        self._installed = True
        return self

    def uninstall(self):
        if self._installed:
            self.core._side_tensors = self._orig_side
            self.core._build_grid_obs = self._orig_grid
            self.core._build_vec_obs = self._orig_vec
            self.core.get_obs_tensors = self._orig_get
            self._installed = False


def differential_selftest(build_env, install_pole, device, cols, rows) -> None:
    """Prove ONLY grid channel 2 and vec feature 11 change, and truth is intact."""
    print("\n  DIFFERENTIAL SELF-TEST (no pilot seed spent)")
    seed = 99_990_066
    env = build_env(device, seed)
    core = env.core
    try:
        install_pole(core, "A")
        env.reset()
        for _ in range(8):
            env.step_async(np.zeros((1, 2 * N), dtype=np.int64)); env.step_wait()

        base = core.get_obs_tensors("blue")
        g0, v0 = base["grid"].clone(), base["vec"].clone()
        m0 = base["mask"].clone()
        truth = (core.red_x.clone(), core.red_y.clone())

        for cond in ("ZERO_OPP", "RANDOM_OPP"):
            ab = OppAblation(core, cond, seed, cols, rows).install()
            o = core.get_obs_tensors("blue")
            g1, v1, m1 = o["grid"], o["vec"], o["mask"]

            other_g = [c for c in range(g0.shape[2]) if c != ENEMY_GRID_CHANNEL]
            g_other_same = torch.equal(g0[:, :, other_g], g1[:, :, other_g])
            g_ch2_changed = not torch.equal(g0[:, :, ENEMY_GRID_CHANNEL],
                                            g1[:, :, ENEMY_GRID_CHANNEL])
            other_v = [i for i in range(v0.shape[-1]) if i != NEAREST_ENEMY_VEC_IDX]
            v_other_same = torch.equal(v0[..., other_v], v1[..., other_v])
            v11_changed = not torch.equal(v0[..., NEAREST_ENEMY_VEC_IDX],
                                          v1[..., NEAREST_ENEMY_VEC_IDX])
            mask_same = torch.equal(m0, m1)
            truth_same = (torch.equal(truth[0], core.red_x)
                          and torch.equal(truth[1], core.red_y))

            print(f"    [{cond}]")
            print(f"      grid ch2 changed          : {g_ch2_changed}   (must be True)")
            print(f"      all OTHER grid channels   : {'identical' if g_other_same else 'CHANGED'}")
            print(f"      vec[11] changed           : {v11_changed}   (must be True)")
            print(f"      all OTHER vec features    : {'identical' if v_other_same else 'CHANGED'}")
            print(f"      action mask               : {'identical' if mask_same else 'CHANGED'}")
            print(f"      world truth red_x/red_y   : {'intact' if truth_same else 'MUTATED'}")
            if cond == "ZERO_OPP":
                z_ok = bool(torch.all(g1[:, :, ENEMY_GRID_CHANNEL] == 0))
                o_ok = bool(torch.all(v1[..., NEAREST_ENEMY_VEC_IDX] == 1.0))
                print(f"      ch2 all zero              : {z_ok}   (must be True)")
                print(f"      vec[11] == 1.0 everywhere : {o_ok}   (must be True)")
                if not (z_ok and o_ok):
                    raise SystemExit("FAIL-CLOSED: ZERO_OPP did not produce the neutral encoding.")
            if cond == "RANDOM_OPP":
                # grid and vec must describe the SAME synthetic opponent set.
                # Recompute nearest-enemy distance from the cached synthetic
                # positions and check it reproduces vec[11] exactly.
                ex, ey = ab._cycle_xy
                ox, oy = core.blue_x, core.blue_y
                dx = ex[:, None, :] - ox[:, :, None]
                dy = ey[:, None, :] - oy[:, :, None]
                dmin = torch.sqrt(dx * dx + dy * dy + 1e-8).min(dim=2).values
                expect = torch.clamp(dmin / max(1e-6, core.max_dist), 0.0, 1.0)
                consistent = torch.allclose(expect, v1[..., NEAREST_ENEMY_VEC_IDX], atol=1e-5)
                print(f"      grid/vec same synthetic set: {consistent}   (must be True)")
                if not consistent:
                    raise SystemExit(
                        "FAIL-CLOSED: RANDOM_OPP grid and vec derived from DIFFERENT "
                        "synthetic opponent sets -- the two channels would describe "
                        "different worlds.")
            if not g_ch2_changed or not v11_changed:
                raise SystemExit(f"FAIL-CLOSED: {cond} did not alter the intended channels.")
            if not (g_other_same and v_other_same and mask_same):
                raise SystemExit(f"FAIL-CLOSED: {cond} altered fields outside the opponent channels.")
            if not truth_same:
                raise SystemExit(f"FAIL-CLOSED: {cond} mutated world truth.")
            ab.uninstall()

        o = core.get_obs_tensors("blue")
        if not (torch.equal(o["grid"], g0) and torch.equal(o["vec"], v0)):
            raise SystemExit("FAIL-CLOSED: uninstall did not restore truthful observation.")
        print("    uninstall restores truth  : OK")
        print("    SELF-TEST PASS -- only the two opponent carriers change; truth intact.")
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")
    lo, hi = spec["SEEDS"]["block"]
    seeds = list(range(int(lo), int(hi) + 1))
    label = spec["OUTPUT_LABEL"]
    OUT = SD / f"{label}_RESULT.json"
    ROWS = SD / f"{label.lower()}_rows.csv"
    # One-shot applies to the SEALED RESULT. A bare rows file means an interrupted
    # run, which we resume rather than discard.
    if not args.dry_run and OUT.is_file():
        raise SystemExit(f"REFUSING: sealed result for {label!r} already exists; one-shot.")
    LIVE = SD / f"{label}_LIVE_STATUS.json"
    LOCK = SD / f"{label}.run.lock"
    if not args.dry_run:
        if LOCK.is_file():
            raise SystemExit(
                f"REFUSING: {LOCK.name} exists -- another process may be writing these "
                f"artifacts. Remove it only if you are certain no run is live.")
        LOCK.write_text(json.dumps({"pid": os.getpid(), "utc": _now()}), encoding="utf-8")

    ck = {}
    for who in ("pi_A", "pi_B"):
        p = ROOT / spec["POLICIES"][who]["path"]
        if not p.is_file():
            raise SystemExit(f"REFUSING: checkpoint missing: {p}")
        got = _sha(p)
        if got != spec["POLICIES"][who]["sha256"]:
            raise SystemExit(f"REFUSING: {who} sha mismatch\n  disk  {got}\n  frozen {spec['POLICIES'][who]['sha256']}")
        ck[who] = p

    global torch
    import torch
    from experiments.opponent_spec import (assert_live_opponent_batch,
                                           install_keyed_opponent_overlays,
                                           pole_A_genome, pole_B_genome)
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    R2.AGENTS = N
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    genomes_by_pole = {"A": {"OP6": pole_A_genome(N)}, "B": {"OP7": pole_B_genome(N)}}

    def install_pole(core, pole):
        core._bt_profile_override = None
        core._sds_opening_hold_steps = 0
        install_keyed_opponent_overlays(core, genomes_by_pole[pole])
        return genomes_by_pole[pole]

    print(f"6v6 OPPONENT-CHANNEL ABLATION  {label}  {_now()}")
    print(f"  spec        {SPEC.name}  [{spec['status']}]  arm={spec['arm']}")
    print(f"  policies    pi_A sha {_sha(ck['pi_A'])[:12]}...  pi_B sha {_sha(ck['pi_B'])[:12]}...")
    print(f"  seeds       {seeds[0]}..{seeds[-1]} (n={len(seeds)}), shared across all conditions")
    print(f"  conditions  {list(CONDITIONS)}  (PERMUTE dropped: provably vacuous)")
    print(f"  episodes    2 policies x 2 poles x 3 conditions x {len(seeds)} = {2*2*3*len(seeds)}")
    print("  MECHANISM DIAGNOSTIC ONLY -- not a specialization claim.\n", flush=True)

    probe = R2.build_env(device, seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    cols, rows_ = float(probe.core.cols - 1), float(probe.core.rows - 1)
    if int(obs_space.spaces["grid"].shape[0]) != N:
        probe.close(); raise SystemExit("FAIL-CLOSED: env grid agent dim != 6")
    probe.close()

    differential_selftest(R2.build_env, install_pole, device, cols, rows_)

    if args.dry_run:
        print("\n  --dry-run: spec frozen, checkpoints sha-verified, self-test passed. "
              "NO seed spent, NOTHING written.")
        return 0

    pol = {w: load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
           for w, p in ck.items()}

    def run(who, pole, cond, seed):
        env = R2.build_env(device, seed)
        core = env.core
        ab = None
        try:
            p = pol[who]
            p.reset_strategy()
            genomes = install_pole(core, pole)
            key = BASE_KEY[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                                       context=f"{label} {who}@{pole} {cond} {seed}")
            ab = OppAblation(core, cond, seed, cols, rows_).install()
            terminal = None
            for _ in range(R2.MAX_STEPS):
                action, _ = p.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    res = (i0 or {}).get("episode_result") or {}
                    terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                    break
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            b, r = terminal
            return {"policy": who, "pole": pole, "condition": cond, "seed": seed,
                    "blue": b, "red": r, "win": int(b > r), "margin": b - r}
        finally:
            if ab is not None:
                ab.uninstall()
            env.close()

    from experiments.tqdm_loop import set_postfix, tqdm_iter
    cells = [(w, p, c, s) for w in ("pi_A", "pi_B") for p in ("A", "B")
             for c in CONDITIONS for s in seeds]

    # ---- RESUME: reload any rows a previous interrupted run already produced ----
    FIELDS = ["policy", "pole", "condition", "seed", "blue", "red", "win", "margin"]
    rows, done = [], set()
    if ROWS.is_file():
        with ROWS.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rec = {k: (int(r[k]) if k not in ("policy", "pole", "condition") else r[k])
                       for k in FIELDS}
                rows.append(rec)
                done.add((rec["policy"], rec["pole"], rec["condition"], rec["seed"]))
        print(f"  RESUME: {len(done)} episode(s) already on disk; skipping those.", flush=True)
    else:
        with ROWS.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    todo = [c for c in cells if (c[0], c[1], c[2], c[3]) not in done]
    t0 = time.time()
    bar = tqdm_iter(todo, desc=label, unit="ep")
    for w, p, c, s in bar:
        set_postfix(bar, f"{w}@{p} {c} seed={s}")
        rec = run(w, p, c, s)
        rows.append(rec)
        # ---- INCREMENTAL: append + flush after EVERY episode ----
        with ROWS.open("a", newline="", encoding="utf-8") as fh:
            wcsv = csv.DictWriter(fh, fieldnames=FIELDS)
            wcsv.writerow({k: rec[k] for k in FIELDS})
            fh.flush(); os.fsync(fh.fileno())

        n_done = len(rows)
        el = time.time() - t0
        rate = el / max(1, n_done - len(done))
        latest = None
        if s == seeds[-1]:
            wr = float(np.mean([r["win"] for r in rows
                                if r["policy"] == w and r["pole"] == p and r["condition"] == c]))
            latest = {"cell": f"{w}@Pole{p}/{c}", "win_rate": round(wr, 4)}
            print(f"  {w}@Pole{p} {c:<11s}: win rate {wr:.4f}", flush=True)
        # ---- LIVE_STATUS heartbeat ----
        LIVE.write_text(json.dumps({
            "label": label, "pid": os.getpid(), "device": device,
            "completed": n_done, "total": len(cells),
            "frac": round(n_done / len(cells), 4),
            "current": {"policy": w, "pole": p, "condition": c, "seed": s},
            "latest_cell_result": latest,
            "started_utc": _now() if n_done == len(done) + 1 else None,
            "elapsed_s": round(el, 1),
            "eta_s": round(rate * (len(cells) - n_done), 1),
            "sec_per_ep": round(rate, 2),
            "checkpoint_sha256": {k: _sha(v) for k, v in ck.items()},
            "heartbeat_utc": _now(),
        }, indent=2), encoding="utf-8")

    def arr(w, p, c):
        d = {r["seed"]: r["win"] for r in rows
             if r["policy"] == w and r["pole"] == p and r["condition"] == c}
        return np.array([d[s] for s in seeds], dtype=np.float64)

    out = {}
    for w in ("pi_A", "pi_B"):
        for p in ("A", "B"):
            f = arr(w, p, "FULL")
            cell = {"V_full": round(float(f.mean()), 6)}
            for c in ("ZERO_OPP", "RANDOM_OPP"):
                v = arr(w, p, c)
                cell[f"V_{c}"] = round(float(v.mean()), 6)
                cell[f"L_{c}"] = mean_ci(f - v)
                cell[f"flips_{c}"] = int((f != v).sum())
            out[f"{w}@Pole{p}"] = cell

    OUT.write_text(json.dumps({
        "record": f"{label} opponent-channel ablation",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "arm": "DIAGNOSTIC", "confirmatory": False, "implements": SPEC.name,
        "WHAT_THIS_IS": "exploratory mechanism diagnostic of inference-time dependence on "
                        "opponent-position observation channels",
        "WHAT_THIS_IS_NOT": "not a specialization certification, not architecture or "
                            "checkpoint selection, and not evidence of global "
                            "opponent-agnosticism outside the tested channels",
        "ablated_channels": {"grid_channel": ENEMY_GRID_CHANNEL,
                             "vec_feature": NEAREST_ENEMY_VEC_IDX,
                             "zero_encoding": "grid ch2 = 0; vec[11] = 1.0 (schema's absent-object value)"},
        "global_state_left_untouched": "carries opponent info but feeds only the critic; "
                                       "policy.act() selects actions from policy_logits(obs)",
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds), "shared": True},
        "bootstrap": {"n_boot": N_BOOT, "alpha": ALPHA, "rng_seed": BOOT_SEED},
        "cells": out, "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}\n  -> {ROWS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
