#!/usr/bin/env python3
"""GATE 2B -- does committing one vehicle to defense pay for itself on map_a?

The original Gate 2 was VOID: every episode ended 0-0, so score-difference
metrics had zero variance and could not measure anything. Adding episodes to a
structurally zero metric only produces more zeros, and lengthening the horizon
would change the primary task rather than repair the measurement.

Gate 2B replaces score with CONTINUOUS FLAG PROGRESS, which stays informative
when nobody completes a capture:

    0.0 -> 1.0   travel from the reset position toward the enemy flag
    1.0          enemy flag reached / acquired
    1.0 -> 2.0   carrier progress from the enemy flag back toward home
    2.0          capture

Each episode retains the MAXIMUM progress reached, never the final value: a
carrier tagged at 1.8 demonstrated 1.8 of progress, and final-value scoring
would erase exactly the signal this gate exists to detect.

Paired contrasts, 32 seed-identical pairs, only the blue controller differs:

    defense_benefit = RED max progress under BOTH_ATTACK
                    - RED max progress under ONE_DEFENDER
    offense_cost    = BLUE max progress under BOTH_ATTACK
                    - BLUE max progress under ONE_DEFENDER

PASS requires the manipulation check AND both paired LCB95 above zero. Without
the offensive cost there is no opportunity cost, so allocation is a free choice
and a policy pool has no reason to contain different allocations -- the exact
condition that collapsed the strategy space under RULESET_V1.

MEASUREMENT RULE: authoritative environment state only. Positions, flag
locations, possession, and side predicates all come from the engine. Nothing
here recomputes geometry -- the engine's midline is (cols - 1) * 0.5 with
inclusive bounds, and reimplementing it as cols * 0.5 is a half-cell error that
already caused a full round of false rule violations.

A pass does NOT show that latent strategies exist. It shows they could
rationally exist.
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
AGENTS = 2
SEED_BASE = 1_900_001          # fresh block, unused by any prior experiment

RULESET = dict(taggers_required=1, tag_min_interval_seconds=10.0,
               tag_nearest_only=True, tag_channel_seconds=0.0,
               suppression_attackers_required=2)
RULESET_ID = "RULESET_V2_AQUATICUS_10S"

HOME_DEFENSE_RADIUS = 8.0
# Predeclared practical-separation floor. Statistical significance alone is not
# enough: the treatments must differ by a usable margin, not by dust.
MIN_MANIPULATION_SEPARATION = 0.25

ROW_KEY = ("episode_seed", "blue_style", "map", "opponent", "ruleset_id")


# ----------------------------------------------------------------------
# Phase 4: authoritative flag progress
# ----------------------------------------------------------------------

CAPTURE_PROGRESS = 2.0
# Non-capture progress is capped strictly below 2.0 so floating-point rounding
# can never promote "carrier standing on home" into "captured".
NON_CAPTURE_CAP = 2.0 - 1e-6
FLAG_HOME_EPS = 1e-3


class TeamProgress:
    """Progress toward a LEGAL CAPTURE for one team, from authoritative state.

    2.0 is reserved for an actual capture, detected from the authoritative score
    delta. A carrier that physically reaches its own home while the team's own
    flag is missing cannot score, so it must not be credited as if it had: the
    return term is gated on the team's own flag being home.

        0.0 - 1.0   approaching / reaching the enemy flag
        1.0         enemy flag possessed, legal scoring possibly blocked
        1.0 - 2.0   returning while our own flag is safely home
        2.0         actual capture only

    ``reset_state`` is captured immediately after reset and supplies the true
    starting positions -- never a hardcoded spawn.
    """

    def __init__(self, core, team: str):
        self.core = core
        self.team = team
        self.reset_state = None
        self.max_progress = 0.0
        self.max_raw_return_fraction = 0.0   # descriptive, ungated

    def _fields(self):
        c = self.core
        if self.team == "blue":
            return (c.blue_x[0], c.blue_y[0], c.blue_alive[0], c.blue_carrying[0],
                    c.red_flag_pos[0], c.blue_flag_home[0], int(c.blue_score[0]),
                    c.blue_flag_pos[0])
        return (c.red_x[0], c.red_y[0], c.red_alive[0], c.red_carrying[0],
                c.blue_flag_pos[0], c.red_flag_home[0], int(c.red_score[0]),
                c.red_flag_pos[0])

    def anchor(self):
        x, y, _al, _cr, tflag, hflag, score, _own = self._fields()
        xs = x.detach().cpu().numpy()
        ys = y.detach().cpu().numpy()
        tf = tflag.detach().cpu().numpy()
        hf = hflag.detach().cpu().numpy()
        self.reset_state = {
            "d_start_to_enemy_flag": np.maximum(
                np.hypot(xs - tf[0], ys - tf[1]), 1e-6),
            "d_enemy_flag_to_home": max(
                float(np.hypot(tf[0] - hf[0], tf[1] - hf[1])), 1e-6),
            "start_score": int(score),
        }
        self.max_progress = 0.0
        self.max_raw_return_fraction = 0.0

    def sample(self) -> float:
        x, y, alive, carry, tflag, hflag, score, own_flag = self._fields()
        rs = self.reset_state
        xs = x.detach().cpu().numpy()
        ys = y.detach().cpu().numpy()
        tf = tflag.detach().cpu().numpy()
        hf = hflag.detach().cpu().numpy()
        of = own_flag.detach().cpu().numpy()
        al = alive.detach().cpu().numpy().astype(bool)
        cr = carry.detach().cpu().numpy().astype(bool)

        # Ungated physical return fraction -- descriptive evidence that a carrier
        # got home even when scoring was illegal.
        raw_return = np.clip(
            1.0 - np.hypot(xs - hf[0], ys - hf[1]) / rs["d_enemy_flag_to_home"],
            0.0, 1.0)
        if cr.any():
            self.max_raw_return_fraction = max(
                self.max_raw_return_fraction, float(raw_return[cr & al].max())
                if (cr & al).any() else 0.0)

        if int(score) > rs["start_score"]:
            p = CAPTURE_PROGRESS
        else:
            # A capture is only legally available while our own flag is home.
            own_flag_home = bool(
                np.hypot(of[0] - hf[0], of[1] - hf[1]) <= FLAG_HOME_EPS)
            approach = np.clip(
                1.0 - np.hypot(xs - tf[0], ys - tf[1]) / rs["d_start_to_enemy_flag"],
                0.0, 1.0)
            returning = 1.0 + raw_return * float(own_flag_home)
            per_agent = np.where(cr, returning, approach)
            per_agent = np.where(al, per_agent, 0.0)
            p = float(per_agent.max()) if per_agent.size else 0.0
            p = min(p, NON_CAPTURE_CAP)

        self.max_progress = max(self.max_progress, p)
        return p


def compute_team_flag_progress(core, team: str, reset_state: dict) -> float:
    """Stateless single-sample form of the metric (shared definition)."""
    tp = TeamProgress(core, team)
    tp.reset_state = reset_state
    return tp.sample()


# ----------------------------------------------------------------------
# Phase 5: paired harness
# ----------------------------------------------------------------------

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

        blue, red = TeamProgress(core, "blue"), TeamProgress(core, "red")
        blue.anchor()
        red.anchor()

        n = 0
        home_def_steps = 0
        atk_agent_steps = def_agent_steps = 0
        b_tags = r_tags = denials = 0
        b_pick = r_pick = b_drop = r_drop = 0
        first_tag = None
        prev_b = core.blue_carrying[0].detach().cpu().numpy().copy()
        prev_r = core.red_carrying[0].detach().cpu().numpy().copy()

        for _ in range(MAX_DECISION_STEPS):
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, _i = env.step_wait()
            n += 1
            terminal = bool(np.asarray(done).any())

            for e in core.drain_tag_events():
                et = e.get("event_type")
                if et == "tag_success":
                    if first_tag is None:
                        first_tag = n
                    if e.get("tagger_team") == "blue":
                        b_tags += 1
                    else:
                        r_tags += 1
                elif et == "tag_denied":
                    denials += 1

            if terminal:
                # Post-step state belongs to episode N+1; stop sampling here.
                break

            blue.sample()
            red.sample()

            bx = core.blue_x
            on_enemy = core._is_on_home_side("red", bx)[0].detach().cpu().numpy()
            atk_agent_steps += int(on_enemy.sum())
            def_agent_steps += int((~on_enemy.astype(bool)).sum())

            hf = core.blue_flag_home[0]
            d_home = np.hypot(
                (bx[0] - hf[0]).detach().cpu().numpy(),
                (core.blue_y[0] - hf[1]).detach().cpu().numpy())
            if bool((d_home <= HOME_DEFENSE_RADIUS).any()):
                home_def_steps += 1

            cb = core.blue_carrying[0].detach().cpu().numpy()
            cr = core.red_carrying[0].detach().cpu().numpy()
            b_pick += int(((~prev_b) & cb).sum())
            b_drop += int((prev_b & (~cb)).sum())
            r_pick += int(((~prev_r) & cr).sum())
            r_drop += int((prev_r & (~cr)).sum())
            prev_b, prev_r = cb.copy(), cr.copy()

        bs, rs_ = int(core.blue_score[0]), int(core.red_score[0])
        row = {
            "episode_seed": seed, "blue_style": style,
            "map": MAP, "resolved_map": RESOLVED_MAP, "opponent": opponent,
            "blue_max_flag_progress": blue.max_progress,
            "blue_raw_carrier_return_fraction": blue.max_raw_return_fraction,
            "red_raw_carrier_return_fraction": red.max_raw_return_fraction,
            "red_max_flag_progress": red.max_progress,
            "blue_score": bs, "red_score": rs_, "win_margin": bs - rs_,
            "blue_home_defense_fraction": home_def_steps / max(1, n),
            "blue_forward_commitment_fraction": atk_agent_steps / max(1, n * AGENTS),
            "mean_num_attackers": atk_agent_steps / max(1, n),
            "mean_num_defenders": def_agent_steps / max(1, n),
            "blue_tag_successes": b_tags, "red_tag_successes": r_tags,
            "cooldown_denials": denials,
            "blue_flag_pickups": b_pick, "red_flag_pickups": r_pick,
            "blue_flag_drops": b_drop, "red_flag_drops": r_drop,
            "time_to_first_tag": first_tag if first_tag else n,
            "episode_steps": n,
        }
        row.update(fingerprint(cfg))
        return row
    finally:
        env.close()


# ----------------------------------------------------------------------
# Phases 6-8: manipulation, contrasts, integrity, artifacts
# ----------------------------------------------------------------------

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
    dups = {k: v for k, v in keys.items() if v > 1}
    if dups:
        problems.append(f"{len(dups)} duplicate row keys")
    per_seed = Counter(r["episode_seed"] for r in rows)
    bad = {s: c for s, c in per_seed.items() if c != len(STYLES)}
    if bad:
        problems.append(f"{len(bad)} seeds without exactly {len(STYLES)} rows")
    for field, want in (("map", {MAP}), ("opponent", None),
                        ("ruleset_id", {RULESET_ID})):
        vals = {r[field] for r in rows}
        if len(vals) != 1 or (want and vals != want):
            problems.append(f"mixed {field}: {sorted(vals)}")
    return (not problems), problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--opponent", default=OPPONENT)
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out-dir", default="artifacts/gate2b_affordance_v2b")
    args = p.parse_args()

    out = PROJECT_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def emit(s: str = ""):
        print(s, flush=True)
        lines.append(s)

    emit("=" * 84)
    emit(f"GATE 2B -- strategic affordance under {RULESET_ID}, {MAP}, vs {args.opponent}")
    emit("Progress: 0->1 approach, 1 flag, 1->2 return, 2 capture (episode MAXIMUM)")
    emit("=" * 84)

    rows = []
    for i in range(args.seeds):
        seed = SEED_BASE + i
        for style in STYLES:
            rows.append(run_treatment(style, seed, args.device, args.opponent))
        if (i + 1) % 8 == 0 or i == 0:
            emit(f"  paired seed {i + 1}/{args.seeds}")

    ok, problems = verify_rows(rows, args.seeds)
    emit("\nINTEGRITY")
    emit(f"  rows={len(rows)} expected={args.seeds * len(STYLES)}  "
         f"{'OK' if ok else 'FAIL'}")
    for pr in problems:
        emit(f"    {pr}")
    if not ok:
        emit("\nGATE 2B: ABORT -- integrity failure; refusing to analyze")
        (out / "gate2b_report.txt").write_text("\n".join(lines))
        return 1

    by = {s: [r for r in rows if r["blue_style"] == s] for s in STYLES}
    for s in STYLES:
        by[s].sort(key=lambda r: r["episode_seed"])

    def col(style, k):
        return np.array([r[k] for r in by[style]], dtype=float)

    rng = np.random.default_rng(0)

    # ---- Phase 6: manipulation check -----------------------------------
    d_home = (col(ONE_DEFENDER, "blue_home_defense_fraction")
              - col(BOTH_ATTACK, "blue_home_defense_fraction"))
    # Decision 2 (pre-32-seed amendment): the manipulation is how many VEHICLES
    # are committed forward, so the 0.25 floor applies to mean attacker COUNT,
    # not to the team-normalized fraction (which halves it by construction at
    # N=2). The fraction is retained in the artifact as description only.
    d_fwd = (col(BOTH_ATTACK, "mean_num_attackers")
             - col(ONE_DEFENDER, "mean_num_attackers"))
    m_home, lo_home, hi_home = paired_ci(d_home, rng, args.n_boot, args.alpha)
    m_fwd, lo_fwd, hi_fwd = paired_ci(d_fwd, rng, args.n_boot, args.alpha)
    home_ok = lo_home > 0 and m_home >= MIN_MANIPULATION_SEPARATION
    fwd_ok = lo_fwd > 0 and m_fwd >= MIN_MANIPULATION_SEPARATION
    manip_ok = home_ok and fwd_ok

    emit("\nMANIPULATION CHECK (must pass before outcomes are interpreted)")
    emit(f"  practical-separation floor: {MIN_MANIPULATION_SEPARATION}")
    emit(f"  ONE_DEFENDER extra home defense : {m_home:+.4f}  "
         f"CI95=[{lo_home:+.4f}, {hi_home:+.4f}]  [{'PASS' if home_ok else 'FAIL'}]")
    emit(f"  BOTH_ATTACK extra attackers (agents): {m_fwd:+.4f}  "
         f"CI95=[{lo_fwd:+.4f}, {hi_fwd:+.4f}]  [{'PASS' if fwd_ok else 'FAIL'}]")
    emit(f"  manipulation: {'PASS' if manip_ok else 'FAIL'}")

    # ---- Phase 7: primary contrasts ------------------------------------
    d_def = (col(BOTH_ATTACK, "red_max_flag_progress")
             - col(ONE_DEFENDER, "red_max_flag_progress"))
    d_off = (col(BOTH_ATTACK, "blue_max_flag_progress")
             - col(ONE_DEFENDER, "blue_max_flag_progress"))
    m_def, lo_def, hi_def = paired_ci(d_def, rng, args.n_boot, args.alpha)
    m_off, lo_off, hi_off = paired_ci(d_off, rng, args.n_boot, args.alpha)
    def_ok, off_ok = lo_def > 0, lo_off > 0

    emit(f"\nPRIMARY CONTRASTS (paired, n={args.seeds})")
    emit(f"  RED progress   BOTH={col(BOTH_ATTACK,'red_max_flag_progress').mean():.4f}  "
         f"ONE={col(ONE_DEFENDER,'red_max_flag_progress').mean():.4f}")
    emit(f"  BLUE progress  BOTH={col(BOTH_ATTACK,'blue_max_flag_progress').mean():.4f}  "
         f"ONE={col(ONE_DEFENDER,'blue_max_flag_progress').mean():.4f}")
    emit(f"  defense_benefit = {m_def:+.4f}  CI95=[{lo_def:+.4f}, {hi_def:+.4f}]  "
         f"[{'PASS' if def_ok else 'FAIL'}]")
    emit(f"  offense_cost    = {m_off:+.4f}  CI95=[{lo_off:+.4f}, {hi_off:+.4f}]  "
         f"[{'PASS' if off_ok else 'FAIL'}]")

    emit("\nSUPPORTING TELEMETRY (mechanism evidence, not gates)")
    sup = ["blue_tag_successes", "red_tag_successes", "cooldown_denials",
           "blue_flag_pickups", "red_flag_pickups", "blue_flag_drops",
           "red_flag_drops", "time_to_first_tag", "mean_num_attackers",
           "blue_raw_carrier_return_fraction", "red_raw_carrier_return_fraction",
           "mean_num_defenders", "episode_steps"]
    emit(f"  {'metric':<28s}{'BOTH_ATTACK':>14s}{'ONE_DEFENDER':>14s}")
    for k in sup:
        emit(f"  {k:<28s}{col(BOTH_ATTACK, k).mean():>14.4f}"
             f"{col(ONE_DEFENDER, k).mean():>14.4f}")

    no_variance = (float(np.std(np.concatenate([
        col(BOTH_ATTACK, "blue_max_flag_progress"),
        col(ONE_DEFENDER, "blue_max_flag_progress"),
        col(BOTH_ATTACK, "red_max_flag_progress"),
        col(ONE_DEFENDER, "red_max_flag_progress")]))) < 1e-9)

    if not manip_ok:
        verdict = "INVALID_TREATMENT_SEPARATION"
        detail = "controllers did not create distinct allocations; fix controllers, not rules"
    elif no_variance:
        verdict = "INCONCLUSIVE_NO_INTERACTION"
        detail = "progress has no variance; scenario never reaches meaningful interaction"
    elif def_ok and off_ok:
        verdict = "PASS"
        detail = "RULESET_V2 creates a real same-map offense/defense trade-off"
    elif def_ok:
        verdict = "PARTIAL_DEFENSE_ONLY"
        detail = "defense useful but apparently free; not enough strategic tension yet"
    elif off_ok:
        verdict = "PARTIAL_COST_ONLY"
        detail = "defender sacrifices offense without protecting home"
    else:
        verdict = "FAIL"
        detail = "neither contrast clears zero"

    emit("\n" + "=" * 84)
    emit(f"GATE 2B: {verdict}")
    emit(f"  {detail}")
    emit("=" * 84)
    emit("A pass does NOT show latent strategies exist -- only that they could")
    emit("rationally exist. Confirm with a larger paired block before spending")
    emit("the full G0-v2 budget.")

    # ---- Phase 8: artifacts --------------------------------------------
    fields = list(rows[0].keys())
    with open(out / "episode_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "gate": "gate2b_affordance", "created_utc": datetime.now(timezone.utc).isoformat(),
        "ruleset_id": RULESET_ID, "ruleset": RULESET,
        "map": MAP, "resolved_map": RESOLVED_MAP, "opponent": args.opponent,
        "treatments": {"both_attack": BOTH_ATTACK, "one_defender": ONE_DEFENDER},
        "paired_seeds": args.seeds,
        "seed_block": [SEED_BASE, SEED_BASE + args.seeds - 1],
        "integrity": {"ok": ok, "problems": problems, "rows": len(rows)},
        "manipulation": {
            "min_separation": MIN_MANIPULATION_SEPARATION,
            "home_defense": {"mean": m_home, "ci95": [lo_home, hi_home], "passed": home_ok},
            "forward_commit_agents": {"mean": m_fwd, "ci95": [lo_fwd, hi_fwd],
                                      "passed": fwd_ok, "units": "agents"},
            "passed": manip_ok,
        },
        "defense_benefit": {"mean": m_def, "ci95": [lo_def, hi_def], "passed": def_ok},
        "offense_cost": {"mean": m_off, "ci95": [lo_off, hi_off], "passed": off_ok},
        "supporting": {k: {"both_attack": float(col(BOTH_ATTACK, k).mean()),
                           "one_defender": float(col(ONE_DEFENDER, k).mean())}
                       for k in sup},
        "verdict": verdict, "detail": detail,
        "passed": verdict == "PASS",
    }
    (out / "paired_summary.json").write_text(json.dumps(summary, indent=2))

    manifest = {
        "experiment": "gate2b_affordance", "created_utc": summary["created_utc"],
        "ruleset_id": RULESET_ID, **RULESET,
        "map": MAP, "resolved_map": RESOLVED_MAP, "opponent": args.opponent,
        "horizon": MAX_DECISION_STEPS, "agents": AGENTS,
        "paired_seeds": args.seeds, "seed_block": [SEED_BASE, SEED_BASE + args.seeds - 1],
        "styles": list(STYLES), "row_key": list(ROW_KEY),
        "domain_randomization": False, "tag_telemetry": True,
        "expected_rows": args.seeds * len(STYLES),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def sha(pth: Path) -> str:
        return hashlib.sha256(pth.read_bytes()).hexdigest() if pth.exists() else "MISSING"

    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=str(PROJECT_ROOT),
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unavailable"

    crit = ["experiments/gate2b_affordance_scenarios_v2.py",
            "gpu_env/_core/_scripted_blue_styles.py", "gpu_env/_core/_rules.py",
            "gpu_env/_config.py", "plot/eval_rollout.py"]
    (out / "run_provenance.json").write_text(json.dumps({
        "recorded_utc": summary["created_utc"],
        "command": "experiments/gate2b_affordance_scenarios_v2.py",
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
        "critical_file_sha256": {c: sha(PROJECT_ROOT / c) for c in crit},
        "artifact_sha256": {n: sha(out / n) for n in
                            ("episode_rows.csv", "paired_summary.json", "manifest.json")},
        "ruleset_fingerprint": {"ruleset_id": RULESET_ID, **RULESET},
    }, indent=2))

    (out / "gate2b_report.txt").write_text("\n".join(lines))
    emit(f"\n[done] {out}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
