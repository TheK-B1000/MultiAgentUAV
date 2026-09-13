r"""Projected-teacher oracle: can the student's ACTION interface execute the strategy?

Implements PROJECTED_TEACHER_ORACLE_SPEC.json.

ORIGINAL  : blue_scripted=True -- the teacher drives continuous targets directly.
PROJECTED : blue_scripted=False -- the engine consumes real student actions. At
            each ACTUAL decision boundary (blue_commit_ticks_left <= 0) the
            teacher is queried with full privileged state, its target is passed
            through the validated adapter, and the resulting (macro, target_idx)
            is submitted. Between boundaries the previous action is held and the
            engine's own commitment logic governs.

The commitment rule is the critical fidelity detail: an every-tick teacher
override would give the oracle reactivity the learned student can never have.

    python -m experiments.eval_projected_teacher_oracle --scales 2 4 --promote
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "PROJECTED_TEACHER_ORACLE_SPEC.json"
LABEL = "PROJECTED_TEACHER_ORACLE"
FIELDS = ["scale", "strategy", "pole", "arm", "seed", "blue", "red", "win",
          "n_decisions", "n_waypoint_actions"]


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _boot(d: np.ndarray, n_boot=20000, alpha=0.05, seed=7) -> dict:
    if d.size == 0:
        return {"mean": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    b = d[idx].mean(axis=1)
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": round(float(d.mean()), 6), "lcb95": round(float(lo), 6),
            "ucb95": round(float(hi), 6), "n": int(d.size)}


def _make_env(scale, pole, style, seed, device, go_to_horizon: int | None = None):
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome, pole_B_genome
    from experiments.strategic_demand_searcher import apply_genome_to_core
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    S.AGENTS = scale
    g = pole_A_genome(scale) if pole == "A" else pole_B_genome(scale)
    cfg = GPUFieldConfig(n_envs=1, max_blue_agents=scale, max_red_agents=scale,
        map_set="train", map_layout=S.MAP, max_decision_steps=S.MAX_STEPS,
        aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
        obstacle_obs_channel=True, tag_telemetry_enabled=True,
        own_flag_home_required_to_score=True, **S.RULESET)
    if go_to_horizon is not None:
        cfg.macro_commit_go_to_ticks = int(go_to_horizon)
    env = GPUCTFVecEnv(cfg); core = env.core
    opp = g.base_opponent
    env.env_method("set_phase", opp)
    env.env_method("set_next_opponent", "SCRIPTED", opp)
    apply_genome_to_core(core, g)
    # set_blue_style does NOT enable scripted blue -- its own docstring says so
    # explicitly ("callers must also set self.blue_scripted = True"). An earlier
    # version of this file asserted the opposite in a comment, which left blue
    # action-controlled with all-zero actions: blue drove to waypoint 0 and
    # scored 0 in every episode. Order matches experiments/strategic_demand_searcher.py.
    core.blue_scripted = True
    core.set_blue_style(style)
    env.reset()
    apply_genome_to_core(core, g)
    core.drain_tag_events()
    return env, core, S


def harness_anchor(scale: int, device: str, seeds: tuple[int, ...],
                   go_to_horizon: int | None = None) -> list[str]:
    """KNOWN-ANSWER ANCHOR the oracle must pass before spending episodes:
    the ORIGINAL arm must reproduce experiments/strategic_demand_searcher.run_episode
    EXACTLY on the same seed. Rule 11 was applied to the adapter but not to the
    harness, and that gap let a fully broken ORIGINAL arm run for 40 episodes."""
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome
    S.AGENTS = scale
    g = pole_A_genome(scale)
    fails = []
    for seed in seeds:
        ref = S.run_episode(style=S.GUARD, genome=g, seed=seed, device=device)
        mine = run_episode(scale, "GUARD", "A", "ORIGINAL", seed, device,
                           go_to_horizon=go_to_horizon)
        if not (ref["win"] == mine["win"] and ref["blue_score"] == mine["blue"]
                and ref["red_score"] == mine["red"]):
            fails.append(f"{scale}v{scale} seed={seed}: reference "
                         f"b={ref['blue_score']} r={ref['red_score']} w={ref['win']} "
                         f"!= oracle-ORIGINAL b={mine['blue']} r={mine['red']} w={mine['win']}")
    return fails


def run_episode(scale, strategy, pole, arm, seed, device,
                go_to_horizon: int | None = None) -> dict:
    import experiments.strategic_demand_searcher as S0
    from experiments.teacher_action_adapter import adapt
    style = S0.GUARD if strategy == "GUARD" else S0.BREACH
    env, core, S = _make_env(scale, pole, style, seed, device,
                             go_to_horizon=go_to_horizon)
    n_dec = n_wp = 0
    try:
        if arm == "PROJECTED":
            # engine consumes real student actions; _blue_style_active() depends
            # only on _blue_style_id, so the teacher stays queryable.
            core.blue_scripted = False
        W = core._macro_targets.detach().cpu().numpy()
        action = np.zeros((scale, 2), dtype=np.int64)
        have = False
        term = None
        for _t in range(S.MAX_STEPS):
            if arm == "PROJECTED":
                # decision boundary per agent, read BEFORE the step consumes it
                nc = (core.blue_commit_ticks_left[0] <= 0).detach().cpu().numpy()
                btx, bty = core._get_scripted_targets("blue")
                for i in range(scale):
                    if have and not bool(nc[i]):
                        continue                    # still committed; hold action
                    a = adapt(core, float(btx[0, i]), float(bty[0, i]), i, W)
                    n_dec += 1
                    if a.uses_waypoint:
                        n_wp += 1
                        action[i] = (a.macro, a.target_idx)
                    elif a.macro is not None:
                        action[i] = (a.macro, 0)
                    else:
                        action[i] = (0, 0)          # forced-home: action irrelevant
                have = True
                env.step_async(action.reshape(-1))
            else:
                env.step_async(env.action_space.sample() * 0)
            _o, _r, done, info = env.step_wait()
            if bool(np.asarray(done).any()):
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                er = (i0 or {}).get("episode_result") or {}
                term = (int(er.get("blue_score", 0)), int(er.get("red_score", 0)))
                break
        if term is None:
            term = (int(core.blue_score[0]), int(core.red_score[0]))
        b, r = term
        return {"scale": f"{scale}v{scale}", "strategy": strategy, "pole": pole,
                "arm": arm, "seed": seed, "blue": b, "red": r, "win": int(b > r),
                "n_decisions": n_dec, "n_waypoint_actions": n_wp}
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--n-seeds", type=int, default=24)
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--go-to-horizon", type=int, default=None,
                    help="Override GPUFieldConfig.macro_commit_go_to_ticks "
                         "(default: cfg=4). Use 1 for the repaired projected validation.")
    ap.add_argument("--spec", type=str, default=None,
                    help="Alternate frozen SPEC path (required when --go-to-horizon != 4).")
    ap.add_argument("--label", type=str, default=None,
                    help="Output label / artifact prefix (defaults from SPEC OUTPUT_LABEL).")
    a = ap.parse_args()

    go_to_h = a.go_to_horizon
    if go_to_h is not None and go_to_h < 1:
        raise SystemExit("REFUSING: --go-to-horizon must be >= 1")

    spec_path = Path(a.spec) if a.spec else SPEC
    if not spec_path.is_absolute():
        # allow bare filename under SD, or relative to CWD / ROOT
        cand = [spec_path, SD / spec_path.name, ROOT / spec_path]
        spec_path = next((p for p in cand if p.is_file()), cand[0])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")

    label = a.label or str(spec.get("OUTPUT_LABEL") or LABEL)
    # Protect the sealed h=4 oracle artifacts from accidental overwrite.
    if go_to_h is not None and int(go_to_h) != 4 and label == LABEL:
        raise SystemExit(
            "REFUSING: --go-to-horizon != 4 would overwrite sealed PROJECTED_TEACHER_ORACLE "
            "artifacts. Pass --spec REPAIRED_GO_TO_H1_PROJECTED_SPEC.json "
            "(or another non-oracle --label)."
        )
    if go_to_h is None and label != LABEL and a.spec is None:
        raise SystemExit("REFUSING: custom --label without --spec / --go-to-horizon")

    if "SEEDS" in spec and "block" in spec["SEEDS"]:
        lo, _ = spec["SEEDS"]["block"]
    else:
        # Repair SPEC reuses the sealed oracle seed block by explicit design.
        lo = 17600001
    seeds = list(range(int(lo), int(lo) + a.n_seeds))
    ROWS = SD / f"{label.lower()}_rows.csv"
    LIVE = SD / f"{label}_LIVE_STATUS.json"
    LOCK = SD / f"{label}.run.lock"
    if LOCK.is_file():
        raise SystemExit(f"REFUSING: {LOCK.name} exists.")
    LOCK.write_text(json.dumps({"pid": os.getpid(), "utc": _now(),
                                "go_to_horizon": go_to_h}), encoding="utf-8")

    # RULE 11 gate: adapter anchors must pass before any episode.
    from experiments.teacher_action_adapter import golden_anchors
    import experiments.strategic_demand_searcher as S0
    env, core, _ = _make_env(a.scales[0], "A", S0.GUARD, seeds[0], a.device,
                             go_to_horizon=go_to_h)
    try:
        fails = golden_anchors(core, core._macro_targets.detach().cpu().numpy(), 0)
        effective_h = int(core.cfg.macro_commit_go_to_ticks)
    finally:
        env.close()
    if fails:
        LOCK.unlink(missing_ok=True)
        raise SystemExit("RULE 11 ADAPTER ANCHORS FAILED -- oracle refuses to run:\n  " + "\n  ".join(fails))
    # HARNESS anchor: the ORIGINAL arm must reproduce the reference implementation.
    hfails = []
    for sc in a.scales:
        hfails += harness_anchor(sc, a.device, tuple(seeds[:3]), go_to_horizon=go_to_h)
    if hfails:
        LOCK.unlink(missing_ok=True)
        raise SystemExit("HARNESS ANCHOR FAILED -- the ORIGINAL arm does not reproduce "
                         "strategic_demand_searcher.run_episode. Oracle refuses to run:\n  "
                         + "\n  ".join(hfails))
    print(f"{label}  {_now()}\n  rule-11 adapter anchors: PASS\n  harness anchor "
          f"(ORIGINAL == reference run_episode): PASS")
    print(f"  matrix {{GUARD,BREACH}} x {{A,B}} x scales {a.scales}, n={len(seeds)} paired seeds")
    print(f"  macro_commit_go_to_ticks={effective_h}"
          f"{' (CLI override)' if go_to_h is not None else ' (cfg default)'}")
    print(f"  PROJECTED re-queries the teacher ONLY at real commitment boundaries\n", flush=True)

    done_keys = set()
    rows = []
    if ROWS.is_file():
        with ROWS.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
                done_keys.add((r["scale"], r["strategy"], r["pole"], r["arm"], int(r["seed"])))
        print(f"  RESUME: {len(done_keys)} episodes already on disk", flush=True)
    else:
        with ROWS.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    from experiments.tqdm_loop import set_postfix, tqdm_iter
    cells = [(n, st, p, arm) for n in a.scales for st in ("GUARD", "BREACH")
             for p in ("A", "B") for arm in ("ORIGINAL", "PROJECTED")]
    t0 = time.time()
    bar = tqdm_iter(cells, desc=label, unit="cell")
    for n, st, p, arm in bar:
        set_postfix(bar, f"{n}v{n} {st} pole{p} {arm}")
        wins = []
        for seed in seeds:
            key = (f"{n}v{n}", st, p, arm, seed)
            if key in done_keys:
                wins.append(int(next(r["win"] for r in rows if (r["scale"], r["strategy"],
                            r["pole"], r["arm"], int(r["seed"])) == key)))
                continue
            rec = run_episode(n, st, p, arm, seed, a.device, go_to_horizon=go_to_h)
            wins.append(rec["win"])
            with ROWS.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=FIELDS).writerow(rec)
                fh.flush(); os.fsync(fh.fileno())
            rows.append({k: str(v) for k, v in rec.items()})
        print(f"  {n}v{n} {st:<7s} pole{p} {arm:<9s} V={np.mean(wins):.4f}", flush=True)
        LIVE.write_text(json.dumps({"label": label, "pid": os.getpid(),
            "go_to_horizon": effective_h,
            "current": f"{n}v{n} {st} {p} {arm}", "elapsed_s": round(time.time() - t0, 1),
            "heartbeat_utc": _now()}, indent=2), encoding="utf-8")

    def V(n, st, p, arm):
        d = {int(r["seed"]): int(r["win"]) for r in rows
             if r["scale"] == f"{n}v{n}" and r["strategy"] == st
             and r["pole"] == p and r["arm"] == arm}
        return np.array([d[s] for s in seeds if s in d], dtype=float)

    out = {}
    for n in a.scales:
        cell = {}
        for st in ("GUARD", "BREACH"):
            for p in ("A", "B"):
                o, pr = V(n, st, p, "ORIGINAL"), V(n, st, p, "PROJECTED")
                m = min(o.size, pr.size)
                cell[f"{st}@{p}"] = {
                    "V_original": round(float(o.mean()), 4) if o.size else None,
                    "V_projected": round(float(pr.mean()), 4) if pr.size else None,
                    "L_paired_original_minus_projected": _boot(o[:m] - pr[:m]) if m else None,
                }
        for arm in ("ORIGINAL", "PROJECTED"):
            gA, bA = V(n, "GUARD", "A", arm), V(n, "BREACH", "A", arm)
            bB, gB = V(n, "BREACH", "B", arm), V(n, "GUARD", "B", arm)
            mA, mB = min(gA.size, bA.size), min(bB.size, gB.size)
            cell[f"delta_{arm}"] = {
                "delta_A": _boot(gA[:mA] - bA[:mA]) if mA else None,
                "delta_B": _boot(bB[:mB] - gB[:mB]) if mB else None,
            }
            da, db = cell[f"delta_{arm}"]["delta_A"], cell[f"delta_{arm}"]["delta_B"]
            cell[f"gate_{arm}"] = (
                da is not None and db is not None
                and da["lcb95"] is not None and db["lcb95"] is not None
                and float(da["lcb95"]) > 0.0 and float(db["lcb95"]) > 0.0
            )
        out[f"{n}v{n}"] = cell
        d_o, d_p = cell["delta_ORIGINAL"], cell["delta_PROJECTED"]
        print(f"\n  {n}v{n} SPECIALIZATION CONTRAST  "
              f"(PRIMARY GATE = LCB95(delta_A)>0 AND LCB95(delta_B)>0)")
        for nm, dd, gkey in (("ORIGINAL ", d_o, "gate_ORIGINAL"),
                             ("PROJECTED", d_p, "gate_PROJECTED")):
            da, db = dd["delta_A"], dd["delta_B"]
            gate = "PASS" if cell[gkey] else "FAIL"
            print(f"    {nm}  delta_A={da['mean']:+.4f} [{da['lcb95']:+.4f},{da['ucb95']:+.4f}]"
                  f"   delta_B={db['mean']:+.4f} [{db['lcb95']:+.4f},{db['ucb95']:+.4f}]"
                  f"   gate={gate}")

    cfg_sig = json.dumps({"scales": a.scales, "n_seeds": a.n_seeds,
                          "device": a.device, "seed_lo": seeds[0],
                          "go_to_horizon": effective_h}, sort_keys=True)
    run_id = f"{_now().replace(':', '').replace('-', '')}_{hashlib.sha256(cfg_sig.encode()).hexdigest()[:8]}"
    rec = {"record": f"{label} closed-loop action-interface test", "status": "FROZEN_RESULT",
           "utc": _now(), "run_id": run_id, "arm": "DIAGNOSTIC", "confirmatory": False,
           "study_class": spec.get("study_class", "DIAGNOSTIC"),
           "implements": [spec_path.name],
           "config_signature": json.loads(cfg_sig),
           "elapsed_s": round(time.time() - t0, 1), "results": out}
    p = SD / f"{label}_{run_id}_RESULT.json"
    if p.exists():
        raise SystemExit(f"REFUSING: {p.name} exists.")
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    if a.promote:
        (SD / f"{label}_CANONICAL.json").write_text(json.dumps({
            "record": f"CANONICAL {label}", "points_to": p.name, "run_id": run_id,
            "promoted_utc": _now(), "config_signature": json.loads(cfg_sig)}, indent=2),
            encoding="utf-8")
    LOCK.unlink(missing_ok=True)
    print(f"\n  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
