"""The four O1 retention gates, scored on natural episodes.

Protocol: docs/o1-response-oracle-preregistration.md
Constants: artifacts/o1_preregistration/O1_PREREGISTRATION.json (frozen 2026-08-04)

Every threshold, seed block and population definition is read out of the frozen
JSON. This script decides retention; nothing else does. The training panels in
run_o1_response_oracle.py are diagnostics and do not feed in here.

NO INJECTION ANYWHERE IN THIS FILE
----------------------------------
Injected C1 starts are a training device with two declared artifacts (a full
horizon that a natural mid-episode C1 does not have, and no real prefix). If O1
were scored on them it could win by exploiting the injector. Every rollout below
uses ordinary resets, and ``assert not row["injected"]`` holds throughout.

THE PAIRED-PREFIX CONSTRUCTION
------------------------------
For each (opponent, evaluation seed) the same seed is played twice:

    arm A   G0 for the whole episode
    arm B   G0 until C1_active first fires at t*, then O1 to the horizon

Identical seeds give identical trajectories up to t*, so the pair differs only
in what happened after the handoff. Comparisons are paired on the episode, which
is also the bootstrap cluster unit.

Arm B *is* the selector: gate 3's "selector" and gate 1's treatment arm are the
same rollout, scored on different populations.

Run:  python experiments/run_o1_gates.py                    # all three O1 seeds
      python experiments/run_o1_gates.py --episodes 5       # smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.o1_rollout import (  # noqa: E402
    LOST_AFTER_LEADING_DEFINITION,
    paired_bootstrap_delta,
    run_c1_episode,
)

PREREG_PATH = PROJECT_ROOT / "artifacts" / "o1_preregistration" / "O1_PREREGISTRATION.json"
OUT_DIR = PROJECT_ROOT / "artifacts" / "o1_gates"

G0_SEEDS = (3_200_001, 3_200_002, 3_200_003)


def load_prereg() -> dict:
    if not PREREG_PATH.is_file():
        raise FileNotFoundError(f"frozen preregistration missing: {PREREG_PATH}")
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


# --- checkpoint loading -----------------------------------------------------


def g0_checkpoint(seed: int) -> Path:
    tag = f"g0_v5_long_seed{seed}"
    return PROJECT_ROOT / "artifacts" / "g0_v5_long" / tag / "ckpts" / f"final_{tag}.zip"


def o1_checkpoint(seed: int) -> Path:
    tag = f"o1_response_oracle_seed{seed}"
    return PROJECT_ROOT / "artifacts" / "o1_response_oracle" / tag / "ckpts" / f"final_{tag}.zip"


def load_policy_at(ckpt: Path, *, device: str, min_step: int):
    """Load a checkpoint, refusing anything below the preregistered step."""
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy
    from experiments.run_g0_v2_evaluation import resolve_cnn_channels

    if not ckpt.is_file():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    payload = read_checkpoint_payload(str(ckpt), map_location="cpu")
    step = int(payload.get("global_step", 0))
    if step < min_step:
        raise ValueError(
            f"{ckpt}: global_step={step:,} is below the preregistered primary "
            f"checkpoint {min_step:,}. Intermediate checkpoints are not evidence."
        )
    channels = resolve_cnn_channels(payload, context=str(ckpt))
    return load_policy(str(ckpt), device=device, num_cnn_channels=channels)


# --- rollout collection -----------------------------------------------------


def collect_rows(g0, o1, *, opponents, episodes: int, seed_base: int, device: str) -> dict:
    """Three arms over one shared seed grid.

    A         G0 throughout
    B         G0 until first C1, then O1   (the selector)
    O1_only   O1 throughout                (gate 2's and gate 3's fixed O1)
    """
    arms: dict[str, list[dict]] = {"A": [], "B": [], "O1_only": []}
    for opp in opponents:
        for i in range(episodes):
            s = seed_base + i
            a = run_c1_episode(g0, opponent=opp, seed=s, device=device)
            b = run_c1_episode(g0, policy_b=o1, opponent=opp, seed=s, device=device)
            o = run_c1_episode(o1, opponent=opp, seed=s, device=device)
            for key, row in (("A", a), ("B", b), ("O1_only", o)):
                assert not row["injected"], "gates must never score injected episodes"
                arms[key].append(row)
        wr = statistics.fmean([r["win"] for r in arms["A"] if r["opponent"] == opp])
        print(f"    vs {opp}: G0 win_rate={wr:.3f} (n={episodes})")
    return arms


def _index(rows: list[dict]) -> dict:
    return {r["episode_key"]: r for r in rows}


def _wr(rows) -> float:
    rows = list(rows)
    return statistics.fmean([r["win"] for r in rows]) if rows else float("nan")


# --- the gates --------------------------------------------------------------


def gate_1_o1_owns_c1(arms: dict, thresholds: dict, stats_cfg: dict) -> dict:
    """Among natural C1 episodes, does the handoff preserve more leads?

    Population membership is decided on arm A, where G0 played the whole
    episode: whether C1 occurs is a property of the situation G0 walks into, and
    must not be redefined by the treatment.
    """
    a_idx, b_idx = _index(arms["A"]), _index(arms["B"])
    keys = [k for k, r in a_idx.items() if r["c1_fired"] and k in b_idx]

    pairs = [(float(a_idx[k]["lead_preserved"]), float(b_idx[k]["lead_preserved"]))
             for k in keys]
    ci = paired_bootstrap_delta(
        pairs, resamples=stats_cfg["resamples"], seed=stats_cfg["seed"]
    )
    thr = float(thresholds["threshold"])
    enough = len(keys) >= int(thresholds["min_c1_episodes"])
    delta = ci["delta"]
    passed = bool(
        enough and delta is not None and delta >= thr and ci["excludes_zero"]
    )
    return {
        "gate": "1_o1_owns_c1",
        "population": "natural episodes in which C1_active fires (decided on arm A)",
        "metric": "lead_preserved = NOT lost_after_leading",
        "metric_definition": LOST_AFTER_LEADING_DEFINITION,
        "n_c1_episodes": len(keys),
        "min_required": int(thresholds["min_c1_episodes"]),
        "sufficient_support": enough,
        "lead_preserved_A": round(statistics.fmean([p[0] for p in pairs]), 4) if pairs else None,
        "lead_preserved_B": round(statistics.fmean([p[1] for p in pairs]), 4) if pairs else None,
        "threshold": thr,
        **{k: ci[k] for k in ("delta", "ci_low", "ci_high", "excludes_zero", "n_pairs")},
        "PASS": passed,
    }


def gate_2_g0_retains_anchor(arms: dict, thresholds: dict) -> dict:
    """On episodes where C1 never fires, O1 must not be BETTER than G0.

    An O1 that wins the anchor too is a better generalist, not a complementary
    specialist, and the right response would be to replace G0 rather than birth
    z1. O1 being worse here is expected and permitted.
    """
    a_idx, o_idx = _index(arms["A"]), _index(arms["O1_only"])
    keys = [k for k, r in a_idx.items() if not r["c1_fired"] and k in o_idx]

    wr_g0 = _wr(a_idx[k] for k in keys)
    wr_o1 = _wr(o_idx[k] for k in keys)
    tol = float(thresholds["threshold"])
    enough = len(keys) >= int(thresholds["min_anchor_episodes"])
    passed = bool(enough and not np.isnan(wr_o1) and wr_o1 <= wr_g0 + tol)
    return {
        "gate": "2_g0_retains_an_anchor",
        "population": "natural episodes in which C1_active never fires",
        "n_anchor_episodes": len(keys),
        "min_required": int(thresholds["min_anchor_episodes"]),
        "sufficient_support": enough,
        "win_rate_G0": None if np.isnan(wr_g0) else round(wr_g0, 4),
        "win_rate_O1": None if np.isnan(wr_o1) else round(wr_o1, 4),
        "tolerance": tol,
        "requirement": "WR(O1) <= WR(G0) + tolerance",
        "margin_over_G0": None if np.isnan(wr_o1) else round(wr_o1 - wr_g0, 4),
        "PASS": passed,
        "note": (
            "Failing because O1 is BETTER on the anchor is not a near miss. It "
            "means O1 dominates and G0 should be replaced, not that z1 should "
            "be born."
        ),
    }


def gate_3_selector_beats_best_fixed(arms: dict, thresholds: dict, stats_cfg: dict) -> dict:
    """Over the whole pool, the selector must beat whichever fixed policy is best."""
    a_idx, b_idx, o_idx = _index(arms["A"]), _index(arms["B"]), _index(arms["O1_only"])
    keys = [k for k in a_idx if k in b_idx and k in o_idx]

    wr_g0 = _wr(a_idx[k] for k in keys)
    wr_o1 = _wr(o_idx[k] for k in keys)
    wr_sel = _wr(b_idx[k] for k in keys)

    best_name, best_idx = ("G0", a_idx) if wr_g0 >= wr_o1 else ("O1", o_idx)
    pairs = [(float(best_idx[k]["win"]), float(b_idx[k]["win"])) for k in keys]
    ci = paired_bootstrap_delta(
        pairs, resamples=stats_cfg["resamples"], seed=stats_cfg["seed"]
    )
    thr = float(thresholds["threshold"])
    delta = ci["delta"]
    passed = bool(delta is not None and delta >= thr and ci["excludes_zero"])
    return {
        "gate": "3_selector_beats_best_fixed",
        "population": "the full evaluation pool",
        "n_episodes": len(keys),
        "win_rate_G0_always": None if np.isnan(wr_g0) else round(wr_g0, 4),
        "win_rate_O1_always": None if np.isnan(wr_o1) else round(wr_o1, 4),
        "win_rate_selector": None if np.isnan(wr_sel) else round(wr_sel, 4),
        "best_fixed": best_name,
        "threshold": thr,
        **{k: ci[k] for k in ("delta", "ci_low", "ci_high", "excludes_zero", "n_pairs")},
        "PASS": passed,
        "selector": "within-episode, one-way, switches to O1 at first C1 onset",
    }


# --- gate 4: behaviour ------------------------------------------------------


def build_observation_bank(policies: dict, *, opponents, episodes: int,
                           seed_base: int, device: str, out_csv: Path) -> int:
    """Pairwise divergence at natural C1 onset states, in the analyzer's format.

    For every reference policy, episodes are rolled out under that policy and
    the observations at C1-active decisions are recorded. Every policy is then
    scored on those byte-identical observations with identical action masking,
    so the comparison is of behaviour at the same state rather than of the
    states each policy happens to reach.

    ``obs_source`` records which policy generated the states, and the analyzer
    symmetrises over each pair's own two endpoints -- that is what stops a pair
    being flattered by one side's home turf.
    """
    from experiments.audit_k2_specialist_behavior import head_probs, jsd_bits

    keys = sorted(policies)
    fields = ["checkpoint_step", "context", "obs_source", "episode_index",
              "episode_seed", "policy_a", "policy_b", "pair_type",
              "jsd_all_bits", "jsd_macro_bits", "argmax_disagreement",
              "macro_disagreement"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()

        for ref_key in keys:
            ref_policy = policies[ref_key]
            ep_index = 0
            for opp in opponents:
                for i in range(episodes):
                    bank: list[dict] = []

                    def _capture(step_i, ctx, obs, is_b, _bank=bank):
                        from experiments.c1_context import c1_active_from_context

                        if not c1_active_from_context(ctx):
                            return
                        # Copy: the env may hand back views onto buffers it
                        # overwrites in place, which would silently collapse the
                        # bank onto the last observation of the episode.
                        _bank.append({k: np.array(v, copy=True)
                                      for k, v in obs.items()})

                    run_c1_episode(
                        ref_policy, opponent=opp, seed=seed_base + i,
                        device=device, on_step=_capture,
                    )
                    if not bank:
                        ep_index += 1
                        continue

                    # Per-policy head distributions on the SAME observations.
                    probs: dict[str, list] = {}
                    for k in keys:
                        p = policies[k]
                        from experiments.eval_v6i9_map_awareness import _adapt_obs_for_policy

                        per_obs = []
                        for obs in bank:
                            obs_t = _as_tensor_obs(_adapt_obs_for_policy(obs, p), device)
                            per_obs.append(head_probs(p, obs_t))
                        probs[k] = per_obs

                    n_obs = len(bank)
                    for ai in range(len(keys)):
                        for bi in range(ai + 1, len(keys)):
                            ka, kb = keys[ai], keys[bi]
                            jsd_all = jsd_macro = arg_dis = mac_dis = 0.0
                            for t in range(n_obs):
                                ha, hb = probs[ka][t], probs[kb][t]
                                per_head = [jsd_bits(x, y) for x, y in zip(ha, hb)]
                                jsd_all += float(np.mean(per_head))
                                # Heads alternate [macro, target] per agent.
                                macro_heads = per_head[0::2]
                                jsd_macro += float(np.mean(macro_heads))
                                am = [int(x.argmax(dim=-1).item()) for x in ha]
                                bm = [int(y.argmax(dim=-1).item()) for y in hb]
                                arg_dis += float(np.mean([x != y for x, y in zip(am, bm)]))
                                mac_dis += float(np.mean(
                                    [x != y for x, y in zip(am[0::2], bm[0::2])]
                                ))
                            w.writerow({
                                "checkpoint_step": 1_000_000,
                                "context": "C1",
                                "obs_source": ref_key,
                                "episode_index": ep_index,
                                "episode_seed": seed_base + i,
                                "policy_a": ka,
                                "policy_b": kb,
                                "pair_type": (
                                    "within" if ka.split("/")[0] == kb.split("/")[0]
                                    else "between"
                                ),
                                "jsd_all_bits": jsd_all / n_obs,
                                "jsd_macro_bits": jsd_macro / n_obs,
                                "argmax_disagreement": arg_dis / n_obs,
                                "macro_disagreement": mac_dis / n_obs,
                            })
                            written += 1
                    ep_index += 1
    return written


def _as_tensor_obs(obs, device: str) -> dict:
    out = {}
    for k, v in obs.items():
        t = torch.as_tensor(np.asarray(v), device=device)
        if t.is_floating_point():
            t = t.float()
        out[k] = t
    return out


def gate_4_behaviour_distinct(bank_csv: Path, *, families: dict, stats_cfg: dict,
                              n_boot: int = 4000) -> dict:
    """LCB95(B_distinct) > 0, using the analyzer's statistic unchanged."""
    from collections import defaultdict

    from experiments.analyze_k2_behavior_gate import b_distinct, load, pair_values

    rows = load(bank_csv, 1_000_000, "jsd_all_bits")
    if not rows:
        return {"gate": "4_behaviour_is_distinct", "PASS": False,
                "error": "observation bank is empty", "insufficient_support": True}

    pairs = pair_values(rows, balanced=True)
    keys = sorted({k for pr in pairs for k in pr})
    fams = {f: [k for k in keys if k.split("/")[0] == f] for f in families}
    if not all(len(v) >= 2 for v in fams.values()):
        return {"gate": "4_behaviour_is_distinct", "PASS": False,
                "error": f"each family needs >= 2 seeds; got "
                         f"{ {f: len(v) for f, v in fams.items()} }",
                "insufficient_support": True}

    point, med_b, q_w = b_distinct(pairs)

    rng = np.random.default_rng(stats_cfg["seed"])
    by_src_ep = defaultdict(list)
    for r in rows:
        by_src_ep[(r["obs_source"], r["episode_index"])].append(r)
    srcs = sorted({s for (s, _e) in by_src_ep})
    eps_by_src = {s: sorted({e for (s2, e) in by_src_ep if s2 == s}) for s in srcs}

    boots = np.empty(n_boot)
    for i in range(n_boot):
        resampled = []
        for s in srcs:
            eps = eps_by_src[s]
            for j in rng.integers(0, len(eps), len(eps)):
                resampled.extend(by_src_ep[(s, eps[j])])
        pm = pair_values(resampled, balanced=True)
        drawn = {f: [v[k] for k in rng.integers(0, len(v), len(v))]
                 for f, v in fams.items()}
        boots[i] = b_distinct(pm, drawn)[0]

    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "gate": "4_behaviour_is_distinct",
        "statistic": "B_distinct = median(JSD_between) - Q_0.95(JSD_within)",
        "metric": "jsd_all_bits",
        "families": {f: len(v) for f, v in fams.items()},
        "median_between": round(float(med_b), 6),
        "q95_within": round(float(q_w), 6),
        "B_distinct": round(float(point), 6),
        "ci_low": round(float(lo), 6),
        "ci_high": round(float(hi), 6),
        "LCB95": round(float(lo), 6),
        "PASS": bool(lo > 0),
        "level": "family, not per-seed",
        "d_policy_not_used": (
            "D_policy passed on a checkpoint that had collapsed into a single "
            "dominant generalist; it tests distinguishability, not distinction."
        ),
    }


# --- main -------------------------------------------------------------------


def main() -> int:
    prereg = load_prereg()
    o1_seeds = [int(s) for s in prereg["training"]["seeds"]]
    ev = prereg["evaluation"]
    gates_cfg = prereg["gates"]
    stats_cfg = {
        "resamples": int(prereg["statistics"]["resamples"]),
        "seed": int(prereg["statistics"]["seed"]),
    }
    min_step = int(prereg["training"]["primary_checkpoint"])

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30,
                    help="episodes per opponent per O1 seed (protocol default 30)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, nargs="*", default=o1_seeds)
    ap.add_argument("--behaviour-episodes", type=int, default=6,
                    help="episodes per opponent per reference policy for gate 4")
    ap.add_argument("--skip-gate4", action="store_true",
                    help="run gates 1-3 only (gate 4 needs all seeds of both families)")
    args = ap.parse_args()

    from experiments.run_g0_v2_seed import OPPONENTS

    seed_base = int(ev["eval_seed_base"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("=" * 78)
    print("O1 RETENTION GATES — natural episodes only, no injection")
    print(f"protocol: {PREREG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"eval seeds {seed_base}..{seed_base + args.episodes - 1}  "
          f"opponents={OPPONENTS}")
    print(f"O1 seeds: {args.seeds}   G0 seeds: {list(G0_SEEDS)}")
    print("=" * 78)

    # Thresholds come from the frozen JSON; support minima are protocol-level.
    min_c1 = int(ev["minimum_support"]["natural_c1_episodes_per_o1_seed"])
    t1 = {"threshold": gates_cfg["gate_1_o1_owns_c1"]["threshold"],
          "min_c1_episodes": min_c1}
    t2 = {"threshold": gates_cfg["gate_2_g0_retains_an_anchor"]["threshold"],
          "min_anchor_episodes": min_c1}
    t3 = {"threshold": gates_cfg["gate_3_selector_beats_best_fixed"]["threshold"]}

    per_seed: list[dict] = []
    all_rows: list[dict] = []
    for i, o1_seed in enumerate(args.seeds):
        # Each O1 seed is paired against the G0 seed of the same index, so no
        # O1 seed gets to pick a favourable G0 opponent.
        g0_seed = G0_SEEDS[i % len(G0_SEEDS)]
        print(f"\n--- O1 seed {o1_seed} vs G0 seed {g0_seed} ---")
        g0 = load_policy_at(g0_checkpoint(g0_seed), device=args.device, min_step=min_step)
        o1 = load_policy_at(o1_checkpoint(o1_seed), device=args.device, min_step=min_step)

        arms = collect_rows(g0, o1, opponents=OPPONENTS, episodes=args.episodes,
                            seed_base=seed_base, device=args.device)
        for arm, rows in arms.items():
            for r in rows:
                all_rows.append({"o1_seed": o1_seed, "g0_seed": g0_seed, "arm": arm, **r})

        r1 = gate_1_o1_owns_c1(arms, t1, stats_cfg)
        r2 = gate_2_g0_retains_anchor(arms, t2)
        r3 = gate_3_selector_beats_best_fixed(arms, t3, stats_cfg)
        per_seed.append({"o1_seed": o1_seed, "g0_seed": g0_seed,
                         "gate_1": r1, "gate_2": r2, "gate_3": r3})

        print(f"  gate 1  C1 eps={r1['n_c1_episodes']} "
              f"lead_kept A={r1['lead_preserved_A']} B={r1['lead_preserved_B']} "
              f"delta={r1['delta']} CI=[{r1['ci_low']},{r1['ci_high']}] -> {r1['PASS']}")
        print(f"  gate 2  anchor eps={r2['n_anchor_episodes']} "
              f"WR G0={r2['win_rate_G0']} O1={r2['win_rate_O1']} -> {r2['PASS']}")
        print(f"  gate 3  WR G0={r3['win_rate_G0_always']} O1={r3['win_rate_O1_always']} "
              f"sel={r3['win_rate_selector']} delta={r3['delta']} -> {r3['PASS']}")

    if all_rows:
        with open(OUT_DIR / "gate_episodes.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    # --- gate 4: one family-level verdict, not per-seed ---------------------
    gate4: dict
    if args.skip_gate4:
        gate4 = {"gate": "4_behaviour_is_distinct", "PASS": None,
                 "skipped": "--skip-gate4"}
    else:
        print("\n--- gate 4: behaviour bank at natural C1 onsets ---")
        policies = {}
        for s in G0_SEEDS:
            policies[f"G0/s{s}"] = load_policy_at(
                g0_checkpoint(s), device=args.device, min_step=min_step)
        for s in args.seeds:
            policies[f"O1/s{s}"] = load_policy_at(
                o1_checkpoint(s), device=args.device, min_step=min_step)
        bank = OUT_DIR / "behaviour_bank.csv"
        n = build_observation_bank(
            policies, opponents=OPPONENTS, episodes=args.behaviour_episodes,
            seed_base=seed_base, device=args.device, out_csv=bank)
        print(f"  bank rows: {n}")
        gate4 = gate_4_behaviour_distinct(
            bank, families={"G0": None, "O1": None}, stats_cfg=stats_cfg)
        print(f"  B_distinct={gate4.get('B_distinct')} "
              f"LCB95={gate4.get('LCB95')} -> {gate4.get('PASS')}")

    def _n_pass(key: str) -> int:
        return sum(1 for e in per_seed if e[key]["PASS"])

    min_seeds = 2
    g1_ok = _n_pass("gate_1") >= min_seeds
    g2_ok = _n_pass("gate_2") >= min_seeds
    g3_ok = _n_pass("gate_3") >= min_seeds
    g4_ok = bool(gate4.get("PASS"))
    retained = bool(g1_ok and g2_ok and g3_ok and g4_ok)

    report = {
        "evaluation": "O1 retention gates",
        "protocol": str(PREREG_PATH.relative_to(PROJECT_ROOT)),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": "RETAINED" if retained else "NOT_RETAINED",
        "scored_on_injected_episodes": False,
        "lost_after_leading_definition": LOST_AFTER_LEADING_DEFINITION,
        "eval_seed_base": seed_base,
        "episodes_per_opponent": args.episodes,
        "min_seeds_passing": min_seeds,
        "gate_summary": {
            "gate_1_o1_owns_c1": f"{_n_pass('gate_1')}/{len(per_seed)} -> {g1_ok}",
            "gate_2_g0_retains_an_anchor": f"{_n_pass('gate_2')}/{len(per_seed)} -> {g2_ok}",
            "gate_3_selector_beats_best_fixed": f"{_n_pass('gate_3')}/{len(per_seed)} -> {g3_ok}",
            "gate_4_behaviour_is_distinct": f"family-level -> {g4_ok}",
        },
        "per_seed": per_seed,
        "gate_4": gate4,
        "retention_rule": prereg["retention_rule"],
        "on_failure": prereg["on_failure"],
        "wall_seconds": round(time.time() - started, 2),
    }
    (OUT_DIR / "O1_GATES.json").write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False), encoding="utf-8")

    print("\n" + "=" * 78)
    for k, v in report["gate_summary"].items():
        print(f"  {k}: {v}")
    print(f"\nO1 VERDICT: {report['verdict']}")
    if not retained:
        print("z1 is NOT born. Do not retune thresholds, promote an intermediate")
        print("checkpoint, or select the best O1 seed — see the protocol, section 8.")
    print(f"report: {OUT_DIR / 'O1_GATES.json'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
