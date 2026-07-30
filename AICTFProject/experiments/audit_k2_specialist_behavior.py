#!/usr/bin/env python3
"""K=2 LRO behavior audit: are piR and piS behaviorally distinct, or is piR
simply executing the same strategy better?

The 1M payoff gate already FAILED (piR dominates both frozen contexts). Payoff
alone cannot distinguish two explanations:

  (a) piR learned a genuine hybrid RUSH/SPLIT policy   -> behaviors differ
  (b) both families converged on the same strategy and
      piR merely executes it better                    -> behaviors match

This script separates them. It is a DIAGNOSTIC -- it defines no gate and
cannot change the 1M verdict.

Two measurement families
------------------------
1. Matched-observation policy divergence. Every reference rollout is driven by
   one policy; at each decision step ALL six policies are queried on the
   *identical* observation, so JSD / KL / argmax-disagreement compare decisions,
   not trajectories.

   The load-bearing control is WITHIN-family divergence. A raw between-family
   JSD is uninterpretable on its own: two seeds of the *same* family already
   differ. The audit therefore reports

       separation_ratio = between_family / mean(within_piR, within_piS)

   ratio ~ 1.0  -> families are not distinguishable; seed-to-seed spread inside
                   a family is as large as the gap between families
                   => explanation (b)
   ratio >> 1.0 -> families occupy genuinely different policy regions
                   => explanation (a)

   Divergence is symmetrized across the two on-policy state distributions:
   JSD(a,b) = 0.5 * [ mean JSD on a's states + mean JSD on b's states ].

2. Tactical trajectory statistics, measured for each policy on its OWN
   rollouts: lane occupancy, agent separation, carrier-return paths,
   home-defense time, screening/interposition, capture timing.

Evaluation invariants match run_k2_specialist_cross_eval.py exactly
(map_b_split_lane, 2v2, 240 decision steps, deterministic, no domain
randomization, n_envs=1) so the audit describes the same policies the payoff
gate scored.

Audit seed blocks (1_030_001 / 1_040_001) are deliberately disjoint from the
payoff evaluation blocks (1_010_001 / 1_020_001), from every training seed
(901xxx / 902xxx), and from all context-confirmation blocks.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "plot") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "plot"))

CONTEXTS = {
    "C_RUSH": {"opponent": "OP11_ADAPTIVE_EXPLOITER", "seed_base": 1_030_001},
    "C_SPLIT": {"opponent": "OP9_SPLIT_LANE_FEINT", "seed_base": 1_040_001},
}
MAP = "map_b_split_lane"
MAX_DECISION_STEPS = 240
AGENTS = 2

FAMILIES = {
    "piR": {"dir": "checkpoints/k2v2_piR", "stem": "k2v2_piR_op11_mapb_s{seed}_2v2",
            "seeds": [901001, 901002, 901003]},
    "piS": {"dir": "checkpoints/k2v2_piS", "stem": "k2v2_piS_op9_mapb_s{seed}_2v2",
            "seeds": [902001, 902002, 902003]},
}

# Interposition corridor half-width, in grid cells, around the
# nearest-red -> blue-carrier segment.
SCREEN_CORRIDOR = 3.0
# Radius around the blue flag home counted as "defending home".
HOME_DEFENSE_RADIUS = 8.0
# separation_ratio at or above this reads as "families are distinguishable".
SEPARATION_RATIO_THRESHOLD = 1.5


def ckpt_path(family: str, seed: int, step: int) -> Path:
    spec = FAMILIES[family]
    return PROJECT_ROOT / spec["dir"] / f"ckpt_{spec['stem'].format(seed=seed)}_{step}.zip"


def policy_key(family: str, seed: int) -> str:
    return f"{family}/s{seed}"


ALL_KEYS = [policy_key(f, s) for f, spec in FAMILIES.items() for s in spec["seeds"]]
KEY_FAMILY = {policy_key(f, s): f for f, spec in FAMILIES.items() for s in spec["seeds"]}


# ----------------------------------------------------------------------
# policy distribution access
# ----------------------------------------------------------------------

def head_probs(policy, obs_t: dict) -> list[torch.Tensor]:
    """Per-head action probabilities at one observation.

    Returns one ``(1, dim)`` tensor per MultiDiscrete head. For 2v2 the heads
    are [a0_macro, a0_target, a1_macro, a1_target].

    Deliberately NOT wrapped in try/except: if the policy internals move, this
    must fail loudly rather than silently degrade every divergence number to
    NaN.
    """
    model = policy.model
    with torch.no_grad():
        logits = model._mask_logits(model.policy_logits(obs_t, z_idx=None), obs_t.get("mask"))
        return [d.probs for d in model._categoricals(logits)]


def jsd_bits(pa: torch.Tensor, pb: torch.Tensor) -> float:
    """Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint support)."""
    eps = 1e-12
    a = pa.clamp_min(eps)
    b = pb.clamp_min(eps)
    m = 0.5 * (a + b)
    kl_am = (a * (a.log() - m.log())).sum(dim=-1)
    kl_bm = (b * (b.log() - m.log())).sum(dim=-1)
    return float((0.5 * (kl_am + kl_bm)).mean().item() / np.log(2.0))


def kl_bits(pa: torch.Tensor, pb: torch.Tensor) -> float:
    """KL(pa || pb) in bits."""
    eps = 1e-12
    a = pa.clamp_min(eps)
    b = pb.clamp_min(eps)
    return float((a * (a.log() - b.log())).sum(dim=-1).mean().item() / np.log(2.0))


# ----------------------------------------------------------------------
# per-step tactical state
# ----------------------------------------------------------------------

def read_core_state(core) -> dict:
    """Snapshot the env-0 tactical state as plain numpy / floats."""
    def g(t):
        return t[0].detach().cpu().numpy()

    alive = g(core.blue_alive).astype(bool)
    tagged = (g(core.blue_tagged).astype(bool)
              if hasattr(core, "blue_tagged") else np.zeros_like(alive))
    return {
        "bx": g(core.blue_x).astype(float),
        "by": g(core.blue_y).astype(float),
        "carrying": g(core.blue_carrying).astype(bool),
        "alive": alive,
        "tagged": tagged,
        "rx": g(core.red_x).astype(float),
        "ry": g(core.red_y).astype(float),
        "red_alive": g(core.red_alive).astype(bool),
        "blue_flag": g(core.blue_flag_pos).astype(float),
        "red_flag": g(core.red_flag_pos).astype(float),
        "blue_score": int(core.blue_score[0].item()),
    }


def _point_segment_frac_and_dist(px, py, ax, ay, bx, by):
    """Projection fraction along AB, and perpendicular distance, of point P."""
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom < 1e-9:
        return 0.0, float(np.hypot(px - ax, py - ay))
    t = ((px - ax) * vx + (py - ay) * vy) / denom
    cx, cy = ax + t * vx, ay + t * vy
    return float(t), float(np.hypot(px - cx, py - cy))


class TacticalAccumulator:
    """Aggregates tactical statistics over one policy's own rollouts."""

    def __init__(self, rows: int, cols: int):
        self.rows = float(rows)
        self.cols = float(cols)
        self.mid_x = float(cols) * 0.5
        self.lane_edges = (float(rows) / 3.0, 2.0 * float(rows) / 3.0)
        self.episodes: list[dict] = []

    def new_episode(self):
        self._n = 0
        self._lane = np.zeros(3, dtype=float)   # agent-steps in bottom / mid / top
        self._opposed_lane_steps = 0
        self._sep_sum = 0.0
        self._ysep_sum = 0.0
        self._pair_live_steps = 0
        self._home_half_agent_steps = 0.0
        self._all_home_steps = 0
        self._home_defense_steps = 0
        self._carry_steps = 0
        self._carry_return_gain = 0.0
        self._carry_path_len = 0.0
        self._carry_net_home = 0.0
        self._screen_steps = 0
        self._screen_opportunities = 0
        self._first_pickup = None
        self._first_capture = None
        self._captures = 0
        self._flag_contact = None
        self._prev_carrier_xy = None
        self._prev_score = 0

    def _lane_of(self, y: float) -> int:
        lo, hi = self.lane_edges
        return 0 if y < lo else (1 if y < hi else 2)

    def step(self, t: int, st: dict):
        self._n += 1
        bx, by = st["bx"], st["by"]
        live = st["alive"] & (~st["tagged"])

        # --- lane occupancy (agent-steps, live agents only) ---
        for i in range(len(bx)):
            if live[i]:
                self._lane[self._lane_of(by[i])] += 1.0
        # both agents live, in the two OUTER lanes, on opposite sides
        if len(bx) == 2 and live.all():
            if {self._lane_of(by[0]), self._lane_of(by[1])} == {0, 2}:
                self._opposed_lane_steps += 1

        # --- separation (only meaningful with both agents live) ---
        if len(bx) == 2 and live.all():
            self._sep_sum += float(np.hypot(bx[0] - bx[1], by[0] - by[1]))
            self._ysep_sum += abs(float(by[0] - by[1]))
            self._pair_live_steps += 1

        # --- home-half occupancy / home defense ---
        n_live = int(live.sum())
        if n_live > 0:
            self._home_half_agent_steps += float(np.sum(bx[live] < self.mid_x)) / n_live
            if np.all(bx[live] < self.mid_x):
                self._all_home_steps += 1
            fx, fy = st["blue_flag"][0], st["blue_flag"][1]
            near_home = np.hypot(bx - fx, by - fy) <= HOME_DEFENSE_RADIUS
            if np.any(near_home & live & (~st["carrying"])):
                self._home_defense_steps += 1

        # --- first contact with the enemy flag ---
        if self._flag_contact is None:
            rf = st["red_flag"]
            if np.any((np.hypot(bx - rf[0], by - rf[1]) <= 1.5) & live):
                self._flag_contact = t

        # --- carrying: return path + screening ---
        carrying = st["carrying"] & live
        if carrying.any():
            ci = int(np.argmax(carrying))
            if self._first_pickup is None:
                self._first_pickup = t
            self._carry_steps += 1
            # progress measured from the red flag home toward blue home (low x)
            self._carry_return_gain += float(st["red_flag"][0] - bx[ci])
            cxy = (bx[ci], by[ci])
            if self._prev_carrier_xy is not None:
                self._carry_path_len += float(np.hypot(cxy[0] - self._prev_carrier_xy[0],
                                                       cxy[1] - self._prev_carrier_xy[1]))
                self._carry_net_home += float(self._prev_carrier_xy[0] - cxy[0])
            self._prev_carrier_xy = cxy

            # screening: teammate interposed between nearest live red and carrier
            red_live = st["red_alive"].astype(bool)
            if red_live.any() and len(bx) == 2:
                oi = 1 - ci
                if live[oi]:
                    d = np.hypot(st["rx"] - cxy[0], st["ry"] - cxy[1])
                    d = np.where(red_live, d, np.inf)
                    ri = int(np.argmin(d))
                    self._screen_opportunities += 1
                    frac, perp = _point_segment_frac_and_dist(
                        bx[oi], by[oi], st["rx"][ri], st["ry"][ri], cxy[0], cxy[1]
                    )
                    if 0.0 < frac < 1.0 and perp <= SCREEN_CORRIDOR:
                        self._screen_steps += 1
        else:
            self._prev_carrier_xy = None

        # --- capture timing ---
        if st["blue_score"] > self._prev_score:
            self._captures += st["blue_score"] - self._prev_score
            if self._first_capture is None:
                self._first_capture = t
            self._prev_score = st["blue_score"]

    def end_episode(self):
        n = max(1, self._n)
        lane_tot = max(1.0, self._lane.sum())
        carry = max(1, self._carry_steps)
        pair = max(1, self._pair_live_steps)
        # Timing fields censor at episode length when the event never happened;
        # *_occurred lets the reader separate "fast" from "never".
        self.episodes.append({
            "steps": self._n,
            "lane_bottom_frac": self._lane[0] / lane_tot,
            "lane_mid_frac": self._lane[1] / lane_tot,
            "lane_top_frac": self._lane[2] / lane_tot,
            "opposed_lane_frac": self._opposed_lane_steps / n,
            "mean_agent_sep": self._sep_sum / pair,
            "mean_y_sep": self._ysep_sum / pair,
            "home_half_occupancy": self._home_half_agent_steps / n,
            "all_home_frac": self._all_home_steps / n,
            "home_defense_frac": self._home_defense_steps / n,
            "carry_steps": self._carry_steps,
            "carry_frac": self._carry_steps / n,
            "mean_return_progress": self._carry_return_gain / carry,
            "carry_path_efficiency": (
                self._carry_net_home / self._carry_path_len
                if self._carry_path_len > 1e-6 else float("nan")
            ),
            "screen_frac": (
                self._screen_steps / self._screen_opportunities
                if self._screen_opportunities > 0 else float("nan")
            ),
            "first_pickup_step": self._first_pickup if self._first_pickup is not None else self._n,
            "pickup_occurred": float(self._first_pickup is not None),
            "first_capture_step": self._first_capture if self._first_capture is not None else self._n,
            "capture_occurred": float(self._first_capture is not None),
            "flag_contact_step": self._flag_contact if self._flag_contact is not None else self._n,
            "flag_contact_occurred": float(self._flag_contact is not None),
            "pickup_to_capture": (
                self._first_capture - self._first_pickup
                if (self._first_capture is not None and self._first_pickup is not None)
                else float("nan")
            ),
            "captures": self._captures,
        })

    def summary(self) -> dict:
        if not self.episodes:
            return {}
        out = {"n_episodes": len(self.episodes),
               "mean_steps": float(np.mean([e["steps"] for e in self.episodes]))}
        for k in (k for k in self.episodes[0] if k != "steps"):
            vals = np.array([e[k] for e in self.episodes], dtype=float)
            out[k] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
        return out


# ----------------------------------------------------------------------
# rollout
# ----------------------------------------------------------------------

def run_reference_rollouts(ref_key, policies, env, ctx, n_episodes, macro_head_idx):
    """Drive the env with ``ref_key``; query every policy at each matched obs.

    Returns ``(per_episode_divergence_rows, tactical_accumulator)``.

    Divergence is accumulated PER EPISODE rather than pooled, so the analyzer
    can bootstrap confidence intervals over episodes and can slice by
    observation source (which policy generated the states). Pooling here would
    throw that away irrecoverably and cost another full GPU run to recover.
    """
    core = env.core
    tac = TacticalAccumulator(core.rows, core.cols)
    ep_rows: list[dict] = []

    ref = policies[ref_key]
    seed_base = int(ctx["seed_base"])

    for ep_idx in range(n_episodes):
        div = defaultdict(lambda: {"jsd_all": 0.0, "jsd_macro": 0.0, "kl_ab": 0.0,
                                   "kl_ba": 0.0, "argmax_dis": 0.0, "macro_dis": 0.0, "n": 0})
        # Same seeding convention as run_eval_episodes' legacy path, so episode
        # k presents the same environment to every reference policy.
        ep_seed = seed_base + ep_idx
        random.seed(ep_seed)
        np.random.seed(ep_seed)
        torch.manual_seed(ep_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ep_seed)
        if hasattr(env, "seed"):
            env.seed(ep_seed)
        for p in policies.values():
            if hasattr(p, "reset_strategy"):
                p.reset_strategy()

        obs = env.reset()
        tac.new_episode()
        t = 0
        while True:
            single = {
                k: v[0] if hasattr(v, "shape") and len(v.shape) > 1 and v.shape[0] == 1 else v
                for k, v in obs.items()
            }
            try:
                single["global_state"] = env.state()[0]
            except Exception:
                pass

            # One shared tensor obs -> every policy sees byte-identical input.
            obs_t = ref._tensor_obs(ref._batched_obs(single))
            probs = {k: head_probs(p, obs_t) for k, p in policies.items()}
            argmax = {k: [int(ph.argmax(dim=-1).item()) for ph in v] for k, v in probs.items()}

            n_heads = len(probs[ref_key])
            for a, b in combinations(ALL_KEYS, 2):
                pa, pb = probs[a], probs[b]
                d = div[(a, b)]
                d["jsd_all"] += float(np.mean([jsd_bits(pa[h], pb[h]) for h in range(n_heads)]))
                d["jsd_macro"] += float(np.mean([jsd_bits(pa[h], pb[h]) for h in macro_head_idx]))
                d["kl_ab"] += float(np.mean([kl_bits(pa[h], pb[h]) for h in range(n_heads)]))
                d["kl_ba"] += float(np.mean([kl_bits(pb[h], pa[h]) for h in range(n_heads)]))
                d["argmax_dis"] += float(np.mean([argmax[a][h] != argmax[b][h] for h in range(n_heads)]))
                d["macro_dis"] += float(np.mean([argmax[a][h] != argmax[b][h] for h in macro_head_idx]))
                d["n"] += 1

            tac.step(t, read_core_state(core))

            act = np.asarray([argmax[ref_key][h] for h in range(n_heads)], dtype=np.int64)
            env.step_async(act)
            obs, _rew, done, _infos = env.step_wait()
            t += 1
            if bool(np.asarray(done).any()) or t >= MAX_DECISION_STEPS:
                break
        tac.end_episode()

        for (a, b), d in div.items():
            n = max(1.0, d["n"])
            ep_rows.append({
                "obs_source": ref_key,
                "obs_source_family": KEY_FAMILY[ref_key],
                "episode_index": ep_idx,
                "episode_seed": seed_base + ep_idx,
                "policy_a": a, "policy_b": b,
                "pair_type": ("between" if KEY_FAMILY[a] != KEY_FAMILY[b]
                              else f"within_{KEY_FAMILY[a]}"),
                "jsd_all_bits": d["jsd_all"] / n,
                "jsd_macro_bits": d["jsd_macro"] / n,
                "kl_ab_bits": d["kl_ab"] / n,
                "kl_ba_bits": d["kl_ba"] / n,
                "argmax_disagreement": d["argmax_dis"] / n,
                "macro_disagreement": d["macro_dis"] / n,
                "n_matched_steps": int(n),
            })

    return ep_rows, tac


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=int, nargs="+",
                   default=[200_000, 300_000, 500_000, 1_000_000])
    p.add_argument("--contexts", nargs="+", default=list(CONTEXTS))
    p.add_argument("--episodes", type=int, default=12,
                   help="Episodes per reference policy per cell (6 references per cell).")
    p.add_argument("--out-dir", default="artifacts/k2v2_specialist_behavior_audit")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = sorted({str(ckpt_path(f, s, st))
                      for st in args.checkpoints
                      for f, spec in FAMILIES.items() for s in spec["seeds"]
                      if not ckpt_path(f, s, st).exists()})
    if missing:
        print("[abort] missing checkpoints:", file=sys.stderr)
        for m in missing:
            print("   ", m, file=sys.stderr)
        return 1

    manifest = {
        "experiment": "k2v2_specialist_behavior_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC -- defines no gate; does not modify the 1M payoff verdict",
        "contexts": {k: f"{CONTEXTS[k]['opponent']}|{MAP}" for k in args.contexts},
        "audit_seed_blocks": {
            k: [CONTEXTS[k]["seed_base"], CONTEXTS[k]["seed_base"] + args.episodes - 1]
            for k in args.contexts
        },
        "seed_disjointness": (
            "Audit blocks 1_030_001 / 1_040_001 are disjoint from the payoff eval blocks "
            "1_010_001 / 1_020_001, from all training seeds (901xxx / 902xxx), and from "
            "every context-confirmation block."
        ),
        "episodes_per_reference": args.episodes,
        "checkpoints": list(args.checkpoints),
        "invariants": {"map": MAP, "agents": AGENTS, "max_decision_steps": MAX_DECISION_STEPS,
                       "deterministic": True, "domain_randomization": False, "n_envs": 1},
        "families": {k: v["seeds"] for k, v in FAMILIES.items()},
        "separation_ratio_definition": (
            "between_family_divergence / mean(within_piR, within_piS); ~1.0 means the "
            "families are indistinguishable given seed-to-seed spread, which supports "
            "'same strategy, better execution' over 'piR learned a hybrid'."
        ),
        "separation_ratio_threshold": SEPARATION_RATIO_THRESHOLD,
        "observation_bank": (
            "Balanced by construction: every one of the 6 policies (3 piR + 3 piS) serves "
            "as observation source for an equal number of episodes, in BOTH contexts. A "
            "bank generated only by piR could miss states where piS behaves distinctly, so "
            "obs_source is recorded per row and the analyzer slices by it."
        ),
        "two_separations_measured": {
            "counterfactual": (
                "divergence_episodes.csv -- masked-logit JSD/KL and argmax disagreement at "
                "byte-identical observations: do the networks DECIDE differently given the "
                "same information and the same legal actions?"
            ),
            "on_policy": (
                "tactical_episodes.csv -- lane occupancy, agent separation, carrier-return "
                "route, home-defense time, screening/interposition, capture timing: do the "
                "full TRAJECTORIES differ when each policy drives the environment?"
            ),
        },
        "s902002_handling": (
            "The collapsed piS seed s902002 is RETAINED in the full-family result. It may "
            "inflate within_piS and so deflate the separation ratio; the analyzer therefore "
            "also reports a clearly labeled sensitivity slice over s902001/s902003, which "
            "is diagnostic only and never a gate."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] {out_dir / 'manifest.json'}")

    if args.dry_run:
        for st in args.checkpoints:
            for c in args.contexts:
                for k in ALL_KEYS:
                    print(f"  [dry] step={st:>8} {c} ref={k}")
        return 0

    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo import load_custom_ppo_policy

    div_path = out_dir / "divergence_episodes.csv"
    tac_path = out_dir / "tactical_episodes.csv"
    div_fields = ["checkpoint_step", "context", "obs_source", "obs_source_family",
                  "episode_index", "episode_seed", "policy_a", "policy_b", "pair_type",
                  "jsd_all_bits", "jsd_macro_bits", "kl_ab_bits", "kl_ba_bits",
                  "argmax_disagreement", "macro_disagreement", "n_matched_steps"]
    div_fh = open(div_path, "w", newline="")
    div_w = csv.DictWriter(div_fh, fieldnames=div_fields, extrasaction="ignore")
    div_w.writeheader()

    tac_rows: list[dict] = []
    ratios: list[tuple] = []

    try:
        for step in args.checkpoints:
            for ctx_name in args.contexts:
                ctx = CONTEXTS[ctx_name]
                cfg = GPUFieldConfig(
                    n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
                    map_set="train", map_layout=MAP,
                    max_decision_steps=MAX_DECISION_STEPS,
                    aquaticus_profile=True, rules_profile="OURS",
                    device=args.device, seed=int(ctx["seed_base"]),
                )
                env = GPUCTFVecEnv(cfg)
                try:
                    env.env_method("set_phase", ctx["opponent"])
                    env.env_method("set_next_opponent", "SCRIPTED", ctx["opponent"])
                    try:
                        from rl.stress_schedule import STRESS_BY_PHASE
                        env.env_method("set_stress_schedule", STRESS_BY_PHASE)
                    except Exception:
                        pass
                    actual = (env.env_method("get_opponent_key")[0] or "").strip().upper()
                    if actual != ctx["opponent"].strip().upper():
                        print(f"[abort] opponent mismatch: core has {actual!r}, "
                              f"requested {ctx['opponent']!r}", file=sys.stderr)
                        return 1

                    policies = {}
                    for fam, spec in FAMILIES.items():
                        for sd in spec["seeds"]:
                            pol = load_custom_ppo_policy(
                                str(ckpt_path(fam, sd, step)),
                                env.observation_space, env.action_space, device=args.device,
                            )
                            if hasattr(pol, "fixed_latent_strategy"):
                                pol.fixed_latent_strategy = False
                            policies[policy_key(fam, sd)] = pol

                    heads = len(env.action_space.nvec)
                    macro_head_idx = [h for h in range(heads) if h % 2 == 0]
                    print(f"\n=== step={step:,} {ctx_name} ({ctx['opponent']}) | "
                          f"{heads} heads, macro heads {macro_head_idx} ===", flush=True)

                    cell_div: list[dict] = []
                    for ref_key in ALL_KEYS:
                        print(f"  [ref] {ref_key} x {args.episodes} eps", flush=True)
                        ep_rows, tac = run_reference_rollouts(
                            ref_key, policies, env, ctx, args.episodes, macro_head_idx
                        )
                        for r in ep_rows:
                            r["checkpoint_step"] = step
                            r["context"] = ctx_name
                            div_w.writerow(r)
                        cell_div.extend(ep_rows)
                        # Per-episode tactical rows (not just the mean) so the
                        # analyzer can bootstrap these too.
                        for ep_i, ep in enumerate(tac.episodes):
                            tac_rows.append({
                                "checkpoint_step": step, "context": ctx_name,
                                "policy": ref_key, "family": KEY_FAMILY[ref_key],
                                "episode_index": ep_i,
                                "episode_seed": int(ctx["seed_base"]) + ep_i,
                                **ep,
                            })
                    div_fh.flush()

                    # Live per-cell readout. The authoritative statistics
                    # (matrices, CIs, sensitivity) come from the analyzer.
                    by_type = defaultdict(list)
                    for r in cell_div:
                        by_type[r["pair_type"]].append((r["jsd_all_bits"],
                                                        r["macro_disagreement"]))
                    print(f"  {'pair type':14s} {'mean JSD (bits)':>17s} {'macro argmax dis':>18s}")
                    for t in ("within_piR", "within_piS", "between"):
                        if by_type[t]:
                            j = float(np.mean([x[0] for x in by_type[t]]))
                            m = float(np.mean([x[1] for x in by_type[t]]))
                            print(f"  {t:14s} {j:>17.5f} {m:>18.4f}")
                    wr = [x[0] for x in by_type["within_piR"]]
                    ws = [x[0] for x in by_type["within_piS"]]
                    bt = [x[0] for x in by_type["between"]]
                    if wr and ws and bt:
                        within = float(np.mean([np.mean(wr), np.mean(ws)]))
                        ratio = float(np.mean(bt)) / within if within > 1e-12 else float("inf")
                        ratios.append((step, ctx_name, ratio))
                        verdict = ("families distinguishable"
                                   if ratio >= SEPARATION_RATIO_THRESHOLD
                                   else "NOT distinguishable -- same behavior region")
                        print(f"  separation_ratio = {ratio:.3f}  ({verdict})  "
                              f"[point estimate; see analyzer for CI]")
                finally:
                    env.close()
    finally:
        div_fh.close()

    if tac_rows:
        keys = list(dict.fromkeys(k for r in tac_rows for k in r))
        with open(tac_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(tac_rows)

    if ratios:
        print("\n=== separation_ratio trajectory (point estimates, no CI) ===")
        print(f"{'step':>10s} {'context':>10s} {'ratio':>8s}")
        for step, ctx_name, ratio in ratios:
            print(f"{step:>10,} {ctx_name:>10s} {ratio:>8.3f}")

    print(f"\n[done] counterfactual divergence -> {div_path}")
    print(f"[done] on-policy tactical        -> {tac_path}")
    print("\nRun experiments/analyze_k2_specialist_behavior.py for the pairwise matrices, "
          "bootstrap CIs, and the s902002 sensitivity slice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
