"""Phase 7B/7C -- offense-vs-defense interaction assay under stock RULESET_V2.

Pre-authorized mechanism diagnostics. These do NOT gate anything and cannot
upgrade a failed or invalid Gate A; they explain it.

7B  1 attacker : 1 prepared defender     can one attacker penetrate one defender?
7C  2 attackers : 1 prepared defender     does numerical advantage activate, and
                                          which event fires first?

Team sizes use the EXISTING config knobs (max_blue_agents / max_red_agents)
rather than an invented neutralisation mechanism.

FIRST-EVENT ORDERING. Suppression emits no event -- `_kill_agents` has exactly
one caller, inside `_apply_suppression` (gpu_env/_core/_rules.py:467), so an
alive True->False transition is unambiguously a suppression. Tags come from the
`tag_success` telemetry event. Per encounter we classify whichever happens
first:

    TAG_FIRST | SUPPRESSION_FIRST | FLAG_CONTACT_FIRST | DISENGAGEMENT | NO_EVENT

The stock-V2 primary diagnostic runs at suppression_range_cells = 2.0. The
frozen ladder values 2.50 / 2.75 / 3.00 are run as MECHANISM COUNTERFACTUALS
ONLY -- they adopt no ruleset, authorize no training, and no intermediate values
are searched.

Frozen prediction retained: raising suppression range may improve coordinated
breach against FORTRESS while simultaneously letting a rushing pair suppress a
lone home defender more easily, so the joint effect may be non-monotonic and
larger is not automatically better.

Run:  python experiments/phase7_interaction_assay.py --seeds 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAP = "map_a"
MAX_STEPS = 240
RULESET = dict(taggers_required=1, tag_min_interval_seconds=10.0,
               tag_nearest_only=True, tag_channel_seconds=0.0,
               suppression_attackers_required=2)
SEED_BASE = 1_990_001          # fresh block for the Option-B primary assay
SUPPRESSION_RUNGS = (2.0, 2.50, 2.75, 3.00)   # 2.0 = stock V2 primary


def run_episode(*, opponent: str, seed: int, suppression_range: float,
                blue_style: str = "BLUE_BOTH_ATTACK_V2",
                n_blue: int = 2, n_red: int = 2, device: str = "cuda") -> dict:
    # HARD SCOPE GUARD. The project studies multi-agent maritime CTF at 2v2 or
    # larger. Deleting an agent changes the strategic object rather than
    # weakening it -- OP7_DEEP_FORTRESS is a two-role system, so Nr=1 is not a
    # weaker fortress, it is a different opponent. Mechanism must be isolated
    # via telemetry inside the full game.
    if n_blue < 2 or n_red < 2:
        raise ValueError(
            f"NO EXPERIMENT BELOW 2v2 (got n_blue={n_blue}, n_red={n_red}). "
            "Isolate mechanism through in-game telemetry, not agent removal.")

    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=n_blue, max_red_agents=n_red,
        map_set="train", map_layout=MAP, max_decision_steps=MAX_STEPS,
        aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
        obstacle_obs_channel=True, tag_telemetry_enabled=True,
        suppression_range_cells=float(suppression_range), **RULESET,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
        core.blue_scripted = True
        core.set_blue_style(blue_style)
        env.reset()
        core.drain_tag_events()

        prev_alive_r = core.red_alive[0].detach().cpu().numpy().astype(bool).copy()
        prev_alive_b = core.blue_alive[0].detach().cpu().numpy().astype(bool).copy()
        prev_carry_b = core.blue_carrying[0].detach().cpu().numpy().astype(bool).copy()

        first_event = None
        first_event_step = None
        t_tag = t_supp = t_contact = None
        tags_on_blue = tags_by_blue = 0
        supp_red = supp_blue = 0
        pickups = drops = 0
        term_scores = None
        n = 0

        for step_i in range(MAX_STEPS):
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, info = env.step_wait()
            n += 1
            terminal = bool(np.asarray(done).any())
            if terminal:
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                er = i0.get("episode_result") or {}
                term_scores = (int(er.get("blue_score", i0.get("blue_score", 0))),
                               int(er.get("red_score", i0.get("red_score", 0))))

            for e in core.drain_tag_events():
                if e.get("event_type") == "tag_success":
                    if e.get("tagger_team") == "blue":
                        tags_by_blue += 1
                    else:
                        tags_on_blue += 1
                        if t_tag is None:
                            t_tag = step_i
                            if first_event is None:
                                first_event, first_event_step = "TAG_FIRST", step_i

            alive_r = core.red_alive[0].detach().cpu().numpy().astype(bool)
            alive_b = core.blue_alive[0].detach().cpu().numpy().astype(bool)
            # alive True->False is a suppression: _kill_agents has one caller.
            killed_r = int((prev_alive_r & (~alive_r)).sum())
            killed_b = int((prev_alive_b & (~alive_b)).sum())
            supp_red += killed_r
            supp_blue += killed_b
            if killed_r and t_supp is None:
                t_supp = step_i
                if first_event is None:
                    first_event, first_event_step = "SUPPRESSION_FIRST", step_i

            carry_b = core.blue_carrying[0].detach().cpu().numpy().astype(bool)
            newly = int(((~prev_carry_b) & carry_b).sum())
            dropped = int((prev_carry_b & (~carry_b)).sum())
            pickups += newly
            drops += dropped
            if newly and t_contact is None:
                t_contact = step_i
                if first_event is None:
                    first_event, first_event_step = "FLAG_CONTACT_FIRST", step_i

            prev_alive_r, prev_alive_b, prev_carry_b = alive_r.copy(), alive_b.copy(), carry_b.copy()
            if terminal:
                break

        if first_event is None:
            first_event = "NO_EVENT" if pickups == 0 and tags_on_blue == 0 else "DISENGAGEMENT"

        bs, rs_ = term_scores if term_scores else (
            int(core.blue_score[0]), int(core.red_score[0]))
        return {
            "seed": seed, "n_blue": n_blue, "n_red": n_red,
            "blue_style": blue_style,
            "opponent": opponent, "suppression_range": suppression_range,
            "first_event": first_event, "first_event_step": first_event_step,
            "t_first_tag_on_blue": t_tag, "t_first_suppression_of_red": t_supp,
            "t_first_flag_contact": t_contact,
            "tags_on_blue": tags_on_blue, "tags_by_blue": tags_by_blue,
            "suppressions_of_red": supp_red, "suppressions_of_blue": supp_blue,
            "blue_pickups": pickups, "blue_drops": drops,
            "blue_score": bs, "red_score": rs_,
            "breach": int(pickups > 0), "capture": int(bs > 0),
            "episode_steps": n,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    def frac(k):
        return float(np.mean([r[k] for r in rows]))
    ev = [r["first_event"] for r in rows]
    n = len(rows)
    return {
        "n": n,
        "breach_rate": frac("breach"),
        "capture_rate": frac("capture"),
        "mean_blue_score": frac("blue_score"),
        "mean_red_score": frac("red_score"),
        "mean_pickups": frac("blue_pickups"),
        "mean_drops": frac("blue_drops"),
        "mean_tags_on_blue": frac("tags_on_blue"),
        "mean_suppressions_of_red": frac("suppressions_of_red"),
        "mean_episode_steps": frac("episode_steps"),
        "P_tag_first": ev.count("TAG_FIRST") / n,
        "P_suppression_first": ev.count("SUPPRESSION_FIRST") / n,
        "P_flag_contact_first": ev.count("FLAG_CONTACT_FIRST") / n,
        "P_disengagement": ev.count("DISENGAGEMENT") / n,
        "P_no_event": ev.count("NO_EVENT") / n,
        "first_event_counts": {k: ev.count(k) for k in set(ev)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts/phase7/interaction_assay.json")
    a = ap.parse_args()

    out_p = PROJECT_ROOT / a.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {"record": "Phase 7B/7C interaction assay, stock RULESET_V2",
                     "map": MAP, "ruleset": RULESET, "seed_base": SEED_BASE,
                     "seeds": a.seeds, "arms": {}}
    all_rows: list[dict] = []

    # PI SCOPE DECISION: NO EXPERIMENT BELOW 2v2.
    #
    # Both the original asymmetric design (n_red=1) and the Option-B revision
    # (n_blue=1) are discarded. Deleting an agent changes the strategic object:
    # OP7_DEEP_FORTRESS is intrinsically a two-role system (deep flag defender +
    # return-cut interceptor), and the project's scope is multi-agent maritime
    # CTF at 2v2 or larger. Mechanism is therefore isolated through TELEMETRY
    # INSIDE the full 2v2 game, never by removing agents.
    #
    # Every arm below is 2 BLUE vs 2 RED on map_a under stock RULESET_V2. The
    # only manipulated variables are BLUE's allocation (ONE_DEFENDER vs
    # BOTH_ATTACK) and, as diagnostic counterfactuals only, suppression range.
    #
    # OP6/FAST_RAID arms exist to test the frozen symmetric-danger prediction:
    # raising suppression may help BOTH_ATTACK breach FORTRESS while also
    # helping a rushing pair suppress our lone home defender.
    arms = [
        ("7B_ONE_DEFENDER_vs_OP7", "BLUE_ONE_DEFENDER_V2", "OP7", SUPPRESSION_RUNGS),
        ("7B_BOTH_ATTACK_vs_OP7",  "BLUE_BOTH_ATTACK_V2",  "OP7", SUPPRESSION_RUNGS),
        ("7C_ONE_DEFENDER_vs_OP6", "BLUE_ONE_DEFENDER_V2", "OP6", SUPPRESSION_RUNGS),
        ("7C_BOTH_ATTACK_vs_OP6",  "BLUE_BOTH_ATTACK_V2",  "OP6", SUPPRESSION_RUNGS),
    ]

    for name, style, opp, rungs in arms:
        for rung in rungs:
            key = f"{name}_supp{rung:g}"
            rows = []
            for i in range(a.seeds):
                try:
                    rows.append(run_episode(opponent=opp, seed=SEED_BASE + i,
                                            suppression_range=rung,
                                            blue_style=style,
                                            device=a.device))
                except Exception as exc:
                    print(f"  {key} seed {SEED_BASE+i}: ERROR {type(exc).__name__}: {exc}",
                          flush=True)
                    break
            if not rows:
                continue
            s = summarize(rows)
            s["is_stock_v2_primary"] = (rung == 2.0)
            s["diagnostic_only"] = (rung != 2.0)
            results["arms"][key] = s
            all_rows.extend(rows)
            print(f"{key:24s} breach={s['breach_rate']:.3f} cap={s['capture_rate']:.3f} "
                  f"P(tag1st)={s['P_tag_first']:.3f} P(supp1st)={s['P_suppression_first']:.3f} "
                  f"steps={s['mean_episode_steps']:.0f}", flush=True)

    results["note"] = ("Rungs above 2.0 are mechanism counterfactuals only. They "
                       "adopt no ruleset, authorize no PPO training, and no "
                       "intermediate values were searched.")
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_p.parent / "interaction_assay_rows.json").write_text(
        json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\n-> {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
