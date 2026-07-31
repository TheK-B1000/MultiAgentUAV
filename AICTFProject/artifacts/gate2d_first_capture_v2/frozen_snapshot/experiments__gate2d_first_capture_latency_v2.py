#!/usr/bin/env python3
"""GATE 2D -- does a defender DELAY the enemy while costing us tempo?

Identical to Gate 2C except for ONE change: the formal manipulation criterion.

Gate 2C returned INVALID_TREATMENT_SEPARATION at +0.2257 against a 0.25 floor.
That was the RULER, not the treatment. ``blue_home_defense_fraction`` is a
binary per-step "is anyone near home" checkbox: ONE_DEFENDER already sits near
its ceiling, while BOTH_ATTACK still registers home presence during flag
returns, tags and resets. The contrast compresses, and no defender hold radius
could reliably open a 0.25 gap on a nearly-full checkbox.

Gate 2D counts VEHICLES instead, symmetric with the attacker metric that
separated cleanly (0.820 vs 0.472 agents):

    mean_num_home_defenders = mean over steps of the number of BLUE agents that
        are alive, NOT carrying the enemy flag, in authoritative BLUE territory,
        and within the declared home-defense zone.

Controllers, opponent, map, ruleset, horizon, capture-ledger latency outcomes,
and the home-defense radius are all UNCHANGED and were not tuned from the Gate
2C results. ``blue_home_defense_fraction`` is retained as description only.

--- inherited Gate 2C contract ---

Gate 2B was correct in construction but hit a genuine ceiling: on map_a vs OP6
both teams reliably capture in both treatments (first capture around step 31 of
240), so max-flag-progress pinned at 2.0 and both contrasts were exactly zero.

Gate 2C keeps the game identical -- same map, ruleset, opponent, horizon,
paired seeds, and the same BOTH_ATTACK / ONE_DEFENDER controllers -- and
changes only the outcome measure, from WHETHER a team scores to HOW FAST:

    T_capture = decision step of the first authoritative capture
                241 if no capture occurred within the 240-step horizon

    defense_benefit = RED  T_capture under ONE_DEFENDER
                    - RED  T_capture under BOTH_ATTACK      (positive = delayed)

    offense_cost    = BLUE T_capture under ONE_DEFENDER
                    - BLUE T_capture under BOTH_ATTACK      (positive = slower)

PASS requires the manipulation check AND both paired LCB95 above zero.

CAPTURE TIMING IS TAKEN FROM AN AUTHORITATIVE EVENT LEDGER, never from
post-step score state. Reading ``core.blue_score`` after ``step_wait()`` on a
terminal step returns the POST-RESET value: Gate 2B's rows reported 0-0 for
episodes that actually contained four captures. The ledger records the score at
the instant it is awarded, so a reset cannot wash it off the scoreboard.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BOTH_ATTACK = "BLUE_BOTH_ATTACK_V2"
ONE_DEFENDER = "BLUE_ONE_DEFENDER_V2"
STYLES = (BOTH_ATTACK, ONE_DEFENDER)

MAP = "map_a"
RESOLVED_MAP = "map_a_open"
OPPONENT = "OP6"
MAX_DECISION_STEPS = 240
NO_CAPTURE = MAX_DECISION_STEPS + 1     # censoring value for "never scored"
EARLY_WINDOW = 60                       # first observed capture was ~step 31
AGENTS = 2
SEED_BASE = 2_200_001                   # fresh CONFIRMATION block.
# Retired: 2000001..2000032 (Gate 2C), 2100001..2100004 (Gate 2D dev smoke).

RULESET = dict(taggers_required=1, tag_min_interval_seconds=10.0,
               tag_nearest_only=True, tag_channel_seconds=0.0,
               suppression_attackers_required=2)
RULESET_ID = "RULESET_V2_AQUATICUS_10S"

HOME_DEFENSE_RADIUS = 8.0
MIN_MANIPULATION_SEPARATION = 0.25
ROW_KEY = ("episode_seed", "blue_style", "map", "opponent", "ruleset_id")


def run_treatment(style: str, seed: int, device: str, opponent: str) -> dict:
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from rl.ruleset_identity import fingerprint

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
        map_set="train", map_layout=MAP, max_decision_steps=MAX_DECISION_STEPS,
        aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
        obstacle_obs_channel=True, tag_telemetry_enabled=True, **RULESET,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
        actual = (env.env_method("get_opponent_key")[0] or "").strip().upper()
        if actual != opponent:
            raise RuntimeError(f"opponent mismatch: {actual!r} != {opponent!r}")
        core.blue_scripted = True
        core.set_blue_style(style)
        env.reset()
        core.drain_tag_events()

        ledger: list[dict] = []            # authoritative capture record
        n = 0
        home_def_steps = 0
        atk_agent_steps = def_agent_steps = 0
        home_defender_agent_steps = 0
        b_tags = r_tags = denials = 0
        first_contact = first_pickup = first_drop = None
        carry_steps = 0
        prev_b = core.blue_carrying[0].detach().cpu().numpy().copy()

        for _ in range(MAX_DECISION_STEPS):
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, _i = env.step_wait()
            n += 1
            terminal = bool(np.asarray(done).any())

            for e in core.drain_tag_events():
                et = e.get("event_type")
                if et == "capture_scored":
                    ledger.append({"step": n, "team": e["scoring_team"],
                                   "score_after": e["score_after"],
                                   "seq": e["event_sequence"]})
                elif et == "tag_success":
                    if e.get("tagger_team") == "blue":
                        b_tags += 1
                    else:
                        r_tags += 1
                elif et == "tag_denied":
                    denials += 1

            if terminal:
                break

            bx = core.blue_x
            on_enemy = core._is_on_home_side("red", bx)[0].detach().cpu().numpy()
            atk_agent_steps += int(on_enemy.sum())
            def_agent_steps += int((~on_enemy.astype(bool)).sum())

            hf = core.blue_flag_home[0]
            d_home = np.hypot((bx[0] - hf[0]).detach().cpu().numpy(),
                              (core.blue_y[0] - hf[1]).detach().cpu().numpy())
            if bool((d_home <= HOME_DEFENSE_RADIUS).any()):
                home_def_steps += 1     # descriptive checkbox only

            # Formal manipulation metric, in AGENT UNITS: alive, not carrying
            # the enemy flag, in authoritative BLUE territory, inside the zone.
            in_zone = d_home <= HOME_DEFENSE_RADIUS
            in_own_territory = core._is_on_home_side("blue", bx)[0].detach().cpu().numpy()
            alive_b = core.blue_alive[0].detach().cpu().numpy().astype(bool)
            carrying_b = core.blue_carrying[0].detach().cpu().numpy().astype(bool)
            home_defender_agent_steps += int(
                (in_zone & in_own_territory.astype(bool) & alive_b & (~carrying_b)).sum())

            rf = core.red_flag_pos[0]
            d_flag = np.hypot((bx[0] - rf[0]).detach().cpu().numpy(),
                              (core.blue_y[0] - rf[1]).detach().cpu().numpy())
            if first_contact is None and bool((d_flag <= 1.5).any()):
                first_contact = n

            cb = core.blue_carrying[0].detach().cpu().numpy()
            if cb.any():
                carry_steps += 1
            if first_pickup is None and bool(((~prev_b) & cb).any()):
                first_pickup = n
            if first_drop is None and bool((prev_b & (~cb)).any()):
                first_drop = n
            prev_b = cb.copy()

        # --- everything score-related comes from the ledger --------------
        b_caps = [c for c in ledger if c["team"] == "blue"]
        r_caps = [c for c in ledger if c["team"] == "red"]
        t_blue = b_caps[0]["step"] if b_caps else NO_CAPTURE
        t_red = r_caps[0]["step"] if r_caps else NO_CAPTURE

        row = {
            "episode_seed": seed, "blue_style": style,
            "map": MAP, "resolved_map": RESOLVED_MAP, "opponent": opponent,
            "blue_first_capture_step": t_blue,
            "red_first_capture_step": t_red,
            "blue_captured": int(bool(b_caps)), "red_captured": int(bool(r_caps)),
            "blue_score": len(b_caps), "red_score": len(r_caps),
            "capture_count": len(ledger),
            "win_margin": len(b_caps) - len(r_caps),
            "blue_captures_by_60": sum(1 for c in b_caps if c["step"] <= EARLY_WINDOW),
            "red_captures_by_60": sum(1 for c in r_caps if c["step"] <= EARLY_WINDOW),
            "blue_home_defense_fraction": home_def_steps / max(1, n),
            "blue_forward_commitment_fraction": atk_agent_steps / max(1, n * AGENTS),
            "mean_num_attackers": atk_agent_steps / max(1, n),
            "mean_num_defenders": def_agent_steps / max(1, n),
            "mean_num_home_defenders": home_defender_agent_steps / max(1, n),
            "blue_tag_successes": b_tags, "red_tag_successes": r_tags,
            "cooldown_denials": denials,
            "first_flag_contact_step": first_contact if first_contact else NO_CAPTURE,
            "first_pickup_step": first_pickup if first_pickup else NO_CAPTURE,
            "first_flag_drop_step": first_drop if first_drop else NO_CAPTURE,
            "carrier_return_steps": carry_steps,
            "episode_steps": n,
        }
        row.update(fingerprint(cfg))
        return row
    finally:
        env.close()


def paired_ci(d: np.ndarray, rng, n_boot: int, alpha: float):
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    b = d[idx].mean(axis=1)
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def verify_rows(rows: list, seeds: int) -> tuple[bool, list]:
    problems = []
    expected = seeds * len(STYLES)
    if len(rows) != expected:
        problems.append(f"row count {len(rows)} != {expected}")
    keys = Counter(tuple(str(r[k]) for k in ROW_KEY) for r in rows)
    if any(v > 1 for v in keys.values()):
        problems.append("duplicate row keys")
    per_seed = Counter(r["episode_seed"] for r in rows)
    if any(c != len(STYLES) for c in per_seed.values()):
        problems.append(f"seeds without exactly {len(STYLES)} rows")
    for field, want in (("map", {MAP}), ("ruleset_id", {RULESET_ID})):
        vals = {r[field] for r in rows}
        if vals != want:
            problems.append(f"mixed {field}: {sorted(vals)}")
    if len({r["opponent"] for r in rows}) != 1:
        problems.append("mixed opponent")
    return (not problems), problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--opponent", default=OPPONENT)
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out-dir", default="artifacts/gate2d_first_capture_v2")
    args = p.parse_args()

    out = PROJECT_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def emit(s: str = ""):
        print(s, flush=True)
        lines.append(s)

    emit("=" * 84)
    emit(f"GATE 2D -- first-capture latency under {RULESET_ID}, {MAP}, vs {args.opponent}")
    emit(f"T_capture = first authoritative capture step; {NO_CAPTURE} = none within horizon")
    emit("Capture timing from the event ledger, never post-step score state.")
    emit("=" * 84)

    rows = []
    for i in range(args.seeds):
        seed = SEED_BASE + i
        for style in STYLES:
            rows.append(run_treatment(style, seed, args.device, args.opponent))
        if (i + 1) % 8 == 0 or i == 0:
            emit(f"  paired seed {i + 1}/{args.seeds}")

    ok, problems = verify_rows(rows, args.seeds)
    emit(f"\nINTEGRITY  rows={len(rows)} expected={args.seeds * len(STYLES)}  "
         f"{'OK' if ok else 'FAIL'}")
    for pr in problems:
        emit(f"    {pr}")
    if not ok:
        emit("\nGATE 2D: ABORT -- integrity failure; refusing to analyze")
        (out / "gate2d_report.txt").write_text("\n".join(lines))
        return 1

    by = {s: sorted([r for r in rows if r["blue_style"] == s],
                    key=lambda r: r["episode_seed"]) for s in STYLES}

    def col(style, k):
        return np.array([r[k] for r in by[style]], dtype=float)

    rng = np.random.default_rng(0)

    # ---- manipulation --------------------------------------------------
    d_home = (col(ONE_DEFENDER, "mean_num_home_defenders")
              - col(BOTH_ATTACK, "mean_num_home_defenders"))
    d_fwd = (col(BOTH_ATTACK, "mean_num_attackers")
             - col(ONE_DEFENDER, "mean_num_attackers"))
    m_home, lo_home, hi_home = paired_ci(d_home, rng, args.n_boot, args.alpha)
    m_fwd, lo_fwd, hi_fwd = paired_ci(d_fwd, rng, args.n_boot, args.alpha)
    home_ok = lo_home > 0 and m_home >= MIN_MANIPULATION_SEPARATION
    fwd_ok = lo_fwd > 0 and m_fwd >= MIN_MANIPULATION_SEPARATION
    manip_ok = home_ok and fwd_ok

    emit("\nMANIPULATION CHECK")
    emit(f"  ONE_DEFENDER extra home defenders (agents): {m_home:+.4f}  "
         f"CI95=[{lo_home:+.4f}, {hi_home:+.4f}]  [{'PASS' if home_ok else 'FAIL'}]")
    emit(f"  BOTH_ATTACK extra attackers (agents): {m_fwd:+.4f}  "
         f"CI95=[{lo_fwd:+.4f}, {hi_fwd:+.4f}]  [{'PASS' if fwd_ok else 'FAIL'}]")
    emit(f"  manipulation: {'PASS' if manip_ok else 'FAIL'}")

    # ---- primary timing contrasts --------------------------------------
    d_def = (col(ONE_DEFENDER, "red_first_capture_step")
             - col(BOTH_ATTACK, "red_first_capture_step"))
    d_off = (col(ONE_DEFENDER, "blue_first_capture_step")
             - col(BOTH_ATTACK, "blue_first_capture_step"))
    m_def, lo_def, hi_def = paired_ci(d_def, rng, args.n_boot, args.alpha)
    m_off, lo_off, hi_off = paired_ci(d_off, rng, args.n_boot, args.alpha)
    def_ok, off_ok = lo_def > 0, lo_off > 0

    emit(f"\nPRIMARY CONTRASTS (paired, n={args.seeds}, steps)")
    emit(f"  RED  T_capture  BOTH={col(BOTH_ATTACK,'red_first_capture_step').mean():7.2f}  "
         f"ONE={col(ONE_DEFENDER,'red_first_capture_step').mean():7.2f}")
    emit(f"  BLUE T_capture  BOTH={col(BOTH_ATTACK,'blue_first_capture_step').mean():7.2f}  "
         f"ONE={col(ONE_DEFENDER,'blue_first_capture_step').mean():7.2f}")
    emit(f"  defense_benefit (RED delayed)  = {m_def:+.3f}  "
         f"CI95=[{lo_def:+.3f}, {hi_def:+.3f}]  [{'PASS' if def_ok else 'FAIL'}]")
    emit(f"  offense_cost    (BLUE slowed)  = {m_off:+.3f}  "
         f"CI95=[{lo_off:+.3f}, {hi_off:+.3f}]  [{'PASS' if off_ok else 'FAIL'}]")

    emit("\nSUPPORTING (descriptive)")
    sup = ["blue_captured", "red_captured", "blue_score", "red_score",
           "blue_captures_by_60", "red_captures_by_60", "first_flag_contact_step",
           "first_pickup_step", "first_flag_drop_step", "carrier_return_steps",
           "blue_tag_successes", "red_tag_successes", "cooldown_denials",
           "mean_num_attackers", "mean_num_home_defenders",
           "blue_home_defense_fraction", "episode_steps"]
    emit(f"  {'metric':<28s}{'BOTH_ATTACK':>14s}{'ONE_DEFENDER':>14s}")
    for k in sup:
        emit(f"  {k:<28s}{col(BOTH_ATTACK, k).mean():>14.3f}"
             f"{col(ONE_DEFENDER, k).mean():>14.3f}")

    all_t = np.concatenate([col(s, t) for s in STYLES
                            for t in ("red_first_capture_step", "blue_first_capture_step")])
    non_discriminating = bool(np.std(all_t) < 1e-9)

    if not manip_ok:
        verdict, detail = "INVALID_TREATMENT_SEPARATION", "controller problem, not environment failure"
    elif non_discriminating:
        verdict, detail = ("NON_DISCRIMINATING_SCENARIO",
                           "capture times identical across treatments; screen OP6-OP12 "
                           "for a harder admitted opponent -- do NOT change rules")
    elif def_ok and off_ok:
        verdict, detail = "PASS", "V2 demonstrates a real same-map offense/defense trade-off"
    elif def_ok:
        verdict, detail = ("PARTIAL_DEFENSE_ONLY",
                           "defender helps, offensive sacrifice not established")
    elif off_ok:
        verdict, detail = ("PARTIAL_COST_ONLY",
                           "defender costs offense but does not protect home effectively")
    else:
        verdict, detail = "FAIL", "neither timing contrast clears zero"

    emit("\n" + "=" * 84)
    emit(f"GATE 2D: {verdict}")
    emit(f"  {detail}")
    emit("=" * 84)

    with open(out / "episode_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "gate": "gate2d_first_capture_latency",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ruleset_id": RULESET_ID, "ruleset": RULESET,
        "map": MAP, "resolved_map": RESOLVED_MAP, "opponent": args.opponent,
        "treatments": {"both_attack": BOTH_ATTACK, "one_defender": ONE_DEFENDER},
        "paired_seeds": args.seeds,
        "seed_block": [SEED_BASE, SEED_BASE + args.seeds - 1],
        "no_capture_value": NO_CAPTURE,
        "capture_source": "authoritative capture_scored event ledger",
        "integrity": {"ok": ok, "problems": problems},
        "manipulation": {
            "min_separation": MIN_MANIPULATION_SEPARATION,
            "home_defenders_agents": {"mean": m_home, "ci95": [lo_home, hi_home],
                                     "passed": home_ok, "units": "agents"},
            "forward_commit_agents": {"mean": m_fwd, "ci95": [lo_fwd, hi_fwd],
                                      "passed": fwd_ok, "units": "agents"},
            "passed": manip_ok},
        "defense_benefit": {"mean": m_def, "ci95": [lo_def, hi_def], "passed": def_ok,
                            "units": "decision steps"},
        "offense_cost": {"mean": m_off, "ci95": [lo_off, hi_off], "passed": off_ok,
                         "units": "decision steps"},
        "supporting": {k: {"both_attack": float(col(BOTH_ATTACK, k).mean()),
                           "one_defender": float(col(ONE_DEFENDER, k).mean())} for k in sup},
        "verdict": verdict, "detail": detail, "passed": verdict == "PASS",
    }
    (out / "paired_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "manifest.json").write_text(json.dumps({
        "experiment": "gate2d_first_capture_latency", "ruleset_id": RULESET_ID, **RULESET,
        "map": MAP, "resolved_map": RESOLVED_MAP, "opponent": args.opponent,
        "horizon": MAX_DECISION_STEPS, "no_capture_value": NO_CAPTURE,
        "paired_seeds": args.seeds, "seed_block": [SEED_BASE, SEED_BASE + args.seeds - 1],
        "styles": list(STYLES), "row_key": list(ROW_KEY),
        "created_utc": summary["created_utc"],
    }, indent=2))

    def sha(pth: Path) -> str:
        return hashlib.sha256(pth.read_bytes()).hexdigest() if pth.exists() else "MISSING"

    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=str(PROJECT_ROOT),
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unavailable"

    crit = ["experiments/gate2d_first_capture_latency_v2.py",
            "gpu_env/_core/_scripted_blue_styles.py", "gpu_env/_core/_rules.py",
            "gpu_env/_config.py"]
    (out / "run_provenance.json").write_text(json.dumps({
        "recorded_utc": summary["created_utc"],
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
        "critical_file_sha256": {c: sha(PROJECT_ROOT / c) for c in crit},
        "artifact_sha256": {n: sha(out / n) for n in
                            ("episode_rows.csv", "paired_summary.json", "manifest.json")},
        "ruleset_fingerprint": {"ruleset_id": RULESET_ID, **RULESET},
    }, indent=2))
    (out / "gate2d_report.txt").write_text("\n".join(lines))
    emit(f"\n[done] {out}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
