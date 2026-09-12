"""6v6 track-hold (stale opponent localization) dose-response pilot.

Implements 6V6_TRACK_HOLD_DOSE_PILOT_V1.json.

WHAT THIS MEASURES
------------------
ABSOLUTE task performance of the frozen 6v6 Share-Encoder (Rung-1) controller
when some Red tracks go stale. It is NOT a specialization-robustness test: the
6v6 crossover did not establish joint specialization, so no Delta-based claim
is made or implied here. Outcomes are binary win rate and

    L_K(z) = V_full(z) - V_K(z)

reported SEPARATELY for each fixed strategy code z, never collapsed.

THE INTERVENTION
----------------
Observation-side only. World truth is never modified: the simulator's Red poses
continue normally, and Red's own behaviour tree acts on truth throughout.

Blue's view of Red originates at exactly one place -- ``_side_tensors("blue")``,
whose ``enemy_x``/``enemy_y`` entries are ``red_x``/``red_y``. Every derived
opponent geometry in the observation (relative dx/dy, distances, grid enemy
channel) is computed downstream of that dict, so substituting there satisfies
the frozen leakage rule by construction: the filter acts BEFORE any derived
feature is built.

``_side_tensors`` is also called by ``_mines.py`` and ``_rules.py`` (game
mechanics). A blanket patch would corrupt mine logic and win determination and
violate ``world_truth_unchanged``. The substitution is therefore gated on a flag
that is raised ONLY inside the two observation builders that consume enemy
position -- ``_build_grid_obs`` and ``_build_vec_obs``. ``_build_action_mask``
was inspected and reads only own-team fields (own_alive / own_carrying /
own_mine_charges / own_x / own_y), so action legality is provably unaffected.

NESTED DOSES
------------
For episode seed s: rng = default_rng(s + 900001); perm = permutation(N_red);
K1 = perm[:1] subset K2 = perm[:2] subset K3 = perm[:3]. The same permutation is
used for both z codes, so the dose-response is paired by seed AND by which Red
agents are stale.

Run:
  ./.venv/Scripts/python.exe experiments/eval_track_hold_6v6.py --dry-run
  ./.venv/Scripts/python.exe experiments/eval_track_hold_6v6.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "6V6_TRACK_HOLD_DOSE_PILOT_V1.json"
AMENDMENT = SD / "6V6_TRACK_HOLD_CHECKPOINT_AMENDMENT.json"

N_TEAM = 6
BASE_KEY = {"A": "OP6", "B": "OP7"}
# Each code is evaluated on its INTENDED regime -- this is own-regime absolute
# performance, giving one V(z) per code (2 codes x 4 doses x 20 seeds = 160 ep).
POLE_FOR_Z = {0: "A", 1: "B"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def held_indices(seed: int, k: int, n_red: int, offset: int) -> np.ndarray:
    """Nested prefixes of one deterministic permutation per seed."""
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(seed + offset)
    return rng.permutation(n_red)[:k].astype(np.int64)


class TrackHold:
    """Freeze Blue's perceived position of selected Red agents at their t0 pose.

    Installed AFTER env.reset(), so the captured pose is the true initial one
    (the frozen t0 rule: Blue receives correct initial Red positions, then the
    selected tracks are held there). Because the held values ARE the t0 values,
    the first observation is identical with or without the hook.
    """

    def __init__(self, core, idx: np.ndarray):
        self.core = core
        self.idx = idx
        self._installed = False
        self._active = False
        # snapshot of TRUE red pose at t0 -- the only thing Blue is allowed to keep
        self.held_x = core.red_x.clone()
        self.held_y = core.red_y.clone()
        self._orig_side = core._side_tensors
        self._orig_grid = core._build_grid_obs
        self._orig_vec = core._build_vec_obs

    def _side_tensors(self, side: str):
        d = self._orig_side(side)
        if side == "blue" and self._active and self.idx.size:
            d = dict(d)
            ex, ey = d["enemy_x"].clone(), d["enemy_y"].clone()
            ex[:, self.idx] = self.held_x[:, self.idx]
            ey[:, self.idx] = self.held_y[:, self.idx]
            d["enemy_x"], d["enemy_y"] = ex, ey
        return d

    def _wrap(self, fn):
        def inner(*a, **kw):
            self._active = True
            try:
                return fn(*a, **kw)
            finally:
                self._active = False
        return inner

    def install(self):
        if self.idx.size == 0:
            return self  # K=0 full-information baseline: genuinely no hook
        self.core._side_tensors = self._side_tensors
        self.core._build_grid_obs = self._wrap(self._orig_grid)
        self.core._build_vec_obs = self._wrap(self._orig_vec)
        self._installed = True
        return self

    def uninstall(self):
        if self._installed:
            self.core._side_tensors = self._orig_side
            self.core._build_grid_obs = self._orig_grid
            self.core._build_vec_obs = self._orig_vec
            self._installed = False

    def perceived_enemy(self):
        """What Blue currently sees for Red (for the integrity self-test).

        Reads through ``core._side_tensors`` -- the LIVE hook -- not through this
        object's own method, so that after ``uninstall()`` this reports truthful
        observation exactly as the env would. Reading ``self._side_tensors``
        here would substitute unconditionally and silently pass a broken
        uninstall.
        """
        self._active = True
        try:
            d = self.core._side_tensors("blue")
            return d["enemy_x"].clone(), d["enemy_y"].clone()
        finally:
            self._active = False


def integrity_selftest(build_env, install_pole, device) -> None:
    """Prove the hook does what it claims before spending a single pilot seed."""
    print("\n  INTEGRITY SELF-TEST (no pilot seed spent)")
    seed = 99_990_006
    env = build_env(device, seed)
    core = env.core
    try:
        install_pole(core, "A")
        env.reset()
        idx = held_indices(seed, 3, N_TEAM, 900001)
        hold = TrackHold(core, idx).install()
        t0_x = core.red_x.clone()

        for _ in range(12):
            act = np.zeros((1, 2 * N_TEAM), dtype=np.int64)
            env.step_async(act)
            env.step_wait()

        truth_x = core.red_x
        seen_x, _ = hold.perceived_enemy()
        moved = (truth_x - t0_x).abs().max().item()
        held_err = (seen_x[:, idx] - t0_x[:, idx]).abs().max().item()
        free = np.setdiff1d(np.arange(N_TEAM), idx)
        free_err = (seen_x[:, free] - truth_x[:, free]).abs().max().item()

        print(f"    held indices                 : {idx.tolist()}")
        print(f"    world truth moved since t0   : {moved:.4f}  (must be > 0)")
        print(f"    held tracks vs t0 pose       : {held_err:.6f}  (must be 0)")
        print(f"    free tracks vs live truth    : {free_err:.6f}  (must be 0)")

        if moved <= 0:
            raise SystemExit("FAIL-CLOSED: world truth did not advance; test is vacuous.")
        if held_err > 1e-6:
            raise SystemExit("FAIL-CLOSED: held tracks are not frozen at their t0 pose.")
        if free_err > 1e-6:
            raise SystemExit("FAIL-CLOSED: unheld tracks do not follow live truth.")

        hold.uninstall()
        after_x, _ = hold.perceived_enemy()
        if (after_x - truth_x).abs().max().item() > 1e-6:
            raise SystemExit("FAIL-CLOSED: uninstall did not restore truthful observation.")
        print("    uninstall restores truth     : OK")
        print("    SELF-TEST PASS -- observation-side only, world truth intact.")
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
    if not AMENDMENT.is_file():
        raise SystemExit(
            "REFUSING: the spec forbids running before the checkpoint path+sha256 are "
            f"filled by amendment. Missing: {AMENDMENT.name}")
    amd = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if not str(amd.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: amendment not frozen: {amd.get('status')!r}")

    ck = ROOT / amd["CHECKPOINT"]["path"]
    if not ck.is_file():
        raise SystemExit(f"REFUSING: checkpoint missing: {ck}")
    got_sha = _sha(ck)
    if got_sha != amd["CHECKPOINT"]["sha256"]:
        raise SystemExit(f"REFUSING: checkpoint sha mismatch.\n  on disk: {got_sha}\n"
                         f"  frozen : {amd['CHECKPOINT']['sha256']}")

    seeds = list(spec["SEED_BLOCK"]["eval_seeds"])
    offset = int(spec["OCCLUSION_SELECTION"]["FIXED_OCCLUSION_OFFSET"])
    label = amd.get("OUTPUT_LABEL", "TRACK_HOLD_6V6")
    OUT = SD / f"{label}_RESULT.json"
    ROWS = SD / f"{label.lower()}_rows.csv"
    if not args.dry_run and (OUT.is_file() or ROWS.is_file()):
        raise SystemExit(f"REFUSING: output for {label!r} already exists; one-shot.")

    import torch
    from experiments.opponent_spec import (assert_live_opponent_batch,
                                           install_keyed_opponent_overlays,
                                           pole_A_genome, pole_B_genome)
    import experiments.r2_learned_crossover as R2
    import rl.ladder_rung1 as L1
    from rl.curriculum import phase_from_tag

    R2.AGENTS = N_TEAM
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    genomes_by_pole = {"A": {"OP6": pole_A_genome(N_TEAM)},
                       "B": {"OP7": pole_B_genome(N_TEAM)}}

    def install_pole(core, pole):
        core._bt_profile_override = None
        core._sds_opening_hold_steps = 0
        install_keyed_opponent_overlays(core, genomes_by_pole[pole])
        return genomes_by_pole[pole]

    print(f"TRACK-HOLD 6v6 PILOT  {label}  {_now()}")
    print(f"  spec        {SPEC.name}  [{spec['status']}]  arm={spec.get('claim_scope')}")
    print(f"  amendment   {AMENDMENT.name}")
    print(f"  checkpoint  {ck.name}  sha {got_sha[:12]}...")
    print(f"  seeds       {seeds[0]}..{seeds[-1]} (n={len(seeds)})")
    print(f"  doses       K=0(full),1,2,3 nested; offset={offset}")
    print(f"  codes       z0 on Pole {POLE_FOR_Z[0]}, z1 on Pole {POLE_FOR_Z[1]} (own regime)")
    print(f"  episodes    2 x 4 x {len(seeds)} = {2 * 4 * len(seeds)}")
    print("  MEASURES    absolute win rate + L_K = V_full - V_K. NOT specialization.\n",
          flush=True)

    probe = R2.build_env(device, seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    if int(obs_space.spaces["grid"].shape[0]) != N_TEAM:
        probe.close()
        raise SystemExit("FAIL-CLOSED: env grid agent dim != 6")
    probe.close()

    model, branch_cfg, _ = L1.load_rung(1, str(ck), obs_space, act_space, device=device)
    policy = L1.make_dispatch_policy(model, branch_cfg, device=device)

    integrity_selftest(R2.build_env, install_pole, device)

    if args.dry_run:
        print("\n  --dry-run: spec+amendment frozen, checkpoint sha verified, env at N=6, "
              "hook self-test passed. NO pilot seed spent, NOTHING written.")
        return 0

    def run_episode(z: int, k: int, seed: int) -> dict:
        pole = POLE_FOR_Z[z]
        env = R2.build_env(device, seed)
        core = env.core
        hold = None
        try:
            policy.fixed_latent_strategy = True
            policy.fixed_latent_strategy_id = int(z)
            policy.reset_strategy()
            genomes = install_pole(core, pole)
            key = BASE_KEY[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                                       context=f"{label} z{z} K{k} seed {seed}")
            idx = held_indices(seed, k, N_TEAM, offset)
            hold = TrackHold(core, idx).install()

            terminal, steps = None, 0
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                steps += 1
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    res = (i0 or {}).get("episode_result") or {}
                    terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                    break
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            blue, red = terminal
            return {"z": z, "K": k, "seed": seed, "held": ",".join(map(str, idx.tolist())),
                    "blue": blue, "red": red, "win": int(blue > red),
                    "margin": blue - red, "steps": steps}
        finally:
            if hold is not None:
                hold.uninstall()
            env.close()

    from experiments.tqdm_loop import set_postfix, tqdm_iter

    cells = [(z, k, s) for z in (0, 1) for k in (0, 1, 2, 3) for s in seeds]
    rows = []
    bar = tqdm_iter(cells, desc=f"{label}", unit="ep")
    for z, k, s in bar:
        set_postfix(bar, f"z{z} K{k} seed={s}")
        rows.append(run_episode(z, k, s))
        if s == seeds[-1]:
            wr = np.mean([r["win"] for r in rows if r["z"] == z and r["K"] == k])
            print(f"  z{z}  K={k}: win rate {wr:.4f}", flush=True)

    with ROWS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def wr(z, k):
        v = [r["win"] for r in rows if r["z"] == z and r["K"] == k]
        return float(np.mean(v))

    per_code = {}
    for z in (0, 1):
        v_full = wr(z, 0)
        per_code[f"z{z}"] = {
            "pole": POLE_FOR_Z[z],
            "V_full": round(v_full, 6),
            "V_K": {f"K{k}": round(wr(z, k), 6) for k in (1, 2, 3)},
            "L_K": {f"K{k}": round(v_full - wr(z, k), 6) for k in (1, 2, 3)},
            "mean_margin": {f"K{k}": round(float(np.mean(
                [r["margin"] for r in rows if r["z"] == z and r["K"] == k])), 4)
                for k in (0, 1, 2, 3)},
        }

    OUT.write_text(json.dumps({
        "record": f"{label} absolute track-hold dose-response",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "arm": "EXPLORATORY_INFERENCE_ONLY", "confirmatory": False,
        "implements": f"{SPEC.name} + {AMENDMENT.name}",
        "WHAT_THIS_IS": "absolute task-performance robustness of the frozen 6v6 "
                        "Share-Encoder controller under stale opponent tracks",
        "WHAT_THIS_IS_NOT": "NOT robustness of strategic specialization. The 6v6 "
                            "crossover did not establish joint specialization, so no "
                            "Delta-based or latent-strategy robustness claim follows "
                            "from these numbers.",
        "checkpoint_sha256": got_sha,
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds)},
        "doses": "K=0 (full information), 1, 2, 3 stale Red tracks; nested prefixes",
        "occlusion_offset": offset,
        "per_code": per_code,
        "codes_not_collapsed": True,
        "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}\n  -> {ROWS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
