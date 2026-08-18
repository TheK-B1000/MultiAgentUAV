"""Opponent-only Strategic Demand Searcher under frozen RULESET_V3_M1.

Exploratory. Search numbers are NOT Gate B evidence. Do not train PPO.

    python experiments/strategic_demand_searcher.py --generations 6 --pop 8
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.sds_genome import (  # noqa: E402
    ANCHOR_B,
    LEGAL_A_BASES,
    SDSGenome,
    apply_genome_to_core,
    canonical_parent,
    degeneracy_penalty,
    development_eligible,
    mutate,
    recombine,
)

GUARD = "BLUE_ONE_DEFENDER_V2"
BREACH = "BLUE_BOTH_ATTACK_V2"
MAP = "map_a"
MAX_STEPS = 240
AGENTS = 2
SEARCH_SEED_BASE = 2_400_001
MUTATE_SEED_BASE = 2_410_001
RULESET = dict(
    taggers_required=1,
    tag_min_interval_seconds=10.0,
    tag_nearest_only=True,
    tag_channel_seconds=0.0,
    suppression_attackers_required=2,
)
INTENT_PERSIST = 4
PROMOTE_DELTA_G = 0.05


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_episode(*, style: str, genome: SDSGenome, seed: int, device: str) -> dict:
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
        map_set="train", map_layout=MAP, max_decision_steps=MAX_STEPS,
        aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
        obstacle_obs_channel=True, tag_telemetry_enabled=True,
        own_flag_home_required_to_score=True, **RULESET,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        opp = genome.base_opponent
        env.env_method("set_phase", opp)
        env.env_method("set_next_opponent", "SCRIPTED", opp)
        apply_genome_to_core(core, genome)
        core.blue_scripted = True
        core.set_blue_style(style)
        env.reset()
        apply_genome_to_core(core, genome)
        core.drain_tag_events()

        term_scores = None
        intent_run = 0
        t_intent = None
        t_commit = None
        n = 0
        for _ in range(MAX_STEPS):
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, info = env.step_wait()
            n += 1
            terminal = bool(np.asarray(done).any())
            if terminal:
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                er = (i0 or {}).get("episode_result") or {}
                term_scores = (
                    int(er.get("blue_score", 0)),
                    int(er.get("red_score", 0)),
                )
                break
            red_alive = core.red_alive[0] & (~core.red_tagged[0])
            blue_alive = core.blue_alive[0] & (~core.blue_tagged[0])
            red_on_blue = core._is_on_home_side("blue", core.red_x)[0]
            blue_on_red = core._is_on_home_side("red", core.blue_x)[0]
            live_red_on_blue = (~red_alive) | red_on_blue
            live_blue_on_red = (~blue_alive) | blue_on_red
            if int(red_alive.sum().item()) >= 2 and bool(live_red_on_blue.all().item()):
                intent_run += 1
                if t_intent is None and intent_run >= INTENT_PERSIST:
                    t_intent = n
            else:
                intent_run = 0
            if (t_commit is None
                    and int(blue_alive.sum().item()) >= 2
                    and bool(live_blue_on_red.all().item())):
                t_commit = n
        if term_scores is None:
            term_scores = (int(core.blue_score[0]), int(core.red_score[0]))
        b, r = term_scores
        return {
            "blue_score": b,
            "red_score": r,
            "win": int(b > r),
            "draw": int(b == r),
            "steps": n,
            "t_intent": t_intent,
            "t_commit": t_commit,
            "zero_zero": int(b == 0 and r == 0),
            "total_score": b + r,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def paired_eval(genome: SDSGenome, *, n: int, seed_base: int, device: str) -> dict:
    guard_w = []
    breach_w = []
    totals = []
    zz = []
    gaps = []
    for i in range(n):
        seed = seed_base + i
        g = run_episode(style=GUARD, genome=genome, seed=seed, device=device)
        b = run_episode(style=BREACH, genome=genome, seed=seed, device=device)
        guard_w.append(g["win"])
        breach_w.append(b["win"])
        totals.extend([g["total_score"], b["total_score"]])
        zz.extend([g["zero_zero"], b["zero_zero"]])
        ti = b["t_intent"] if b["t_intent"] is not None else (MAX_STEPS + 1)
        tc = b["t_commit"] if b["t_commit"] is not None else (MAX_STEPS + 1)
        gaps.append(ti - tc)
    gw = float(np.mean(guard_w))
    bw = float(np.mean(breach_w))
    delta = gw - bw
    frac00 = float(np.mean(zz))
    mean_tot = float(np.mean(totals))
    pen = degeneracy_penalty(frac00, mean_tot)
    mean_gap = float(np.mean(gaps))
    return {
        "genome": genome.to_dict(),
        "n": n,
        "seed_base": seed_base,
        "guard_wr": gw,
        "breach_wr": bw,
        "delta_G": delta,
        "frac_0_0": frac00,
        "mean_total_score": mean_tot,
        "degeneracy_penalty": pen,
        "mean_intent_minus_commit": mean_gap,
        "precommitment_uncertain": bool(mean_gap > 0.0),
        "J_local": float(delta - pen),
        "search_not_gate_B": True,
    }


def evaluate_B(n: int, seed_base: int, device: str) -> dict:
    g = canonical_parent(ANCHOR_B)
    # delta_B is BREACH - GUARD, i.e. -delta_G on the fortress parent.
    out = paired_eval(g, n=n, seed_base=seed_base, device=device)
    out["delta_B"] = float(out["breach_wr"] - out["guard_wr"])
    out["J_local"] = float(out["delta_B"] - out["degeneracy_penalty"])
    return out


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--pop", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="artifacts/strategic_demand/searcher")
    p.add_argument("--no-screen", action="store_true")
    p.add_argument("--screen-only", action="store_true",
                   help="Evaluate legal A bases + OP7 anchor; do not mutate.")
    p.add_argument("--mutate-from-screen", action="store_true",
                   help="Resume screen archive, skip re-screen, mutate. Never touches 2500001.")
    p.add_argument("--resume-archive", default="",
                   help="Path to a prior archive.json (default: screen archive).")
    args = p.parse_args()
    out_dir = PROJECT_ROOT / args.out
    if args.mutate_from_screen:
        args.no_screen = True
        if args.out == "artifacts/strategic_demand/searcher":
            out_dir = PROJECT_ROOT / "artifacts/strategic_demand/searcher_mutate"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2410001 if args.mutate_from_screen else 2400001)

    print("=" * 74)
    print("STRATEGIC DEMAND SEARCHER  RULESET_V3_M1 frozen")
    print("Search results are NOT Gate B. No PPO. Block 2500001 is not used.")
    print("=" * 74)

    screen_dir = PROJECT_ROOT / "artifacts/strategic_demand/searcher"
    b_path = screen_dir / "anchor_B_cheap.json"
    if args.mutate_from_screen and b_path.is_file():
        b_est = json.loads(b_path.read_text(encoding="utf-8"))
        delta_B = float(b_est["delta_B"])
        print(f"[{_now()}] reuse screen cheap B=OP7 delta_B={delta_B:+.3f} "
              "(not Gate B; not 2500001)")
    else:
        print(f"[{_now()}] cheap B=OP7 estimate ({8} paired)")
        b_est = evaluate_B(8, SEARCH_SEED_BASE + 50_000, args.device)
        print(f"    delta_B={b_est['delta_B']:+.3f}  (search estimate, not Gate B)")
        _write(out_dir / "anchor_B_cheap.json", b_est)
        delta_B = float(b_est["delta_B"])

    archive = []
    seen_ids = set()
    seed_cursor = MUTATE_SEED_BASE if args.mutate_from_screen else SEARCH_SEED_BASE
    resume_path = Path(args.resume_archive) if args.resume_archive else (
        screen_dir / "archive.json")
    if args.mutate_from_screen and resume_path.is_file():
        prior = json.loads(resume_path.read_text(encoding="utf-8"))
        for row in prior.get("rows") or []:
            archive.append(row)
            seen_ids.add(row["genome"]["genome_id"])
        print(f"[{_now()}] resumed {len(archive)} screen rows; "
              f"mutation seeds from {seed_cursor}")
        _write(out_dir / "parents_from_screen.json", prior)

    def consider(genome: SDSGenome, tag: str) -> dict:
        nonlocal seed_cursor
        if genome.genome_id in seen_ids:
            for row in archive:
                if row["genome"]["genome_id"] == genome.genome_id:
                    return row
        seen_ids.add(genome.genome_id)
        print(f"  stage1 {tag} {genome.genome_id} base={genome.base_opponent} "
              f"hold={genome.opening_hold_steps}")
        s1 = paired_eval(genome, n=8, seed_base=seed_cursor, device=args.device)
        seed_cursor += 8
        print(f"    dG={s1['delta_G']:+.3f}  J={s1['J_local']:+.3f}  "
              f"gap={s1['mean_intent_minus_commit']:+.1f}")
        rec = dict(s1)
        rec["stage"] = 1
        rec["tag"] = tag
        rec["J"] = float(min(s1["delta_G"], delta_B) - s1["degeneracy_penalty"])
        if s1["delta_G"] > PROMOTE_DELTA_G:
            print("    PROMOTING to 16 paired")
            s2 = paired_eval(
                genome, n=16, seed_base=seed_cursor, device=args.device)
            seed_cursor += 16
            rec = dict(s2)
            rec["stage"] = 2
            rec["tag"] = tag
            rec["stage1_delta_G"] = s1["delta_G"]
            rec["J"] = float(min(s2["delta_G"], delta_B) - s2["degeneracy_penalty"])
            print(f"    stage2 dG={s2['delta_G']:+.3f}  J={rec['J']:+.3f}  "
                  f"gap={s2['mean_intent_minus_commit']:+.1f}")
        rec["delta_B_anchor"] = delta_B
        rec["not_gate_B"] = True
        rec["development_eligible"] = development_eligible(rec, promote=PROMOTE_DELTA_G)
        rec["confirmation_2500001"] = False
        if rec["development_eligible"]:
            print("    DEVELOPMENT_ELIGIBLE under frozen J/gap/promote "
                  "(does NOT spend 2500001)")
        archive.append(rec)
        _write(out_dir / "archive.json", {"updated": _now(), "rows": archive})
        return rec

    if not args.no_screen:
        print(f"[{_now()}] screen legal A bases")
        for base in LEGAL_A_BASES:
            consider(canonical_parent(base), "screen")

    if args.screen_only:
        archive.sort(key=lambda r: r.get("J", -9), reverse=True)
        summary = {
            "protocol_id": "STRATEGIC_DEMAND_SEARCHER_V1",
            "finished_utc": _now(),
            "mode": "screen_only",
            "search_results_are_gate_B": False,
            "ruleset": "RULESET_V3_M1",
            "anchor_B_cheap_delta_B": delta_B,
            "best": archive[0] if archive else None,
            "n_archive": len(archive),
            "confirmation": "NOT RUN. Freeze genomes, then fresh block 2500001.",
            "ppo": "NOT STARTED",
        }
        _write(out_dir / "summary.json", summary)
        print("=" * 74)
        print("SCREEN FINISHED — not Gate B, not V3_STRATEGIC_DEMAND = VALIDATED")
        if archive:
            best = archive[0]
            print(f"best J={best.get('J'):+.3f}  dG={best.get('delta_G'):+.3f}  "
                  f"id={best['genome']['genome_id']}")
        print("=" * 74)
        return 0

    library = [canonical_parent(b) for b in LEGAL_A_BASES]
    pop = list(library)
    k_init = 0
    while len(pop) < args.pop:
        src = library[k_init % len(library)]
        pop.append(mutate(src, rng, new_id=f"SDS_INIT_{k_init}"))
        k_init += 1

    for gen in range(args.generations):
        print(f"[{_now()}] generation {gen + 1}/{args.generations}")
        scored = []
        for g in pop:
            rec = consider(g, f"gen{gen}")
            scored.append((float(rec.get("J", -9.0)), g))
        scored.sort(key=lambda t: t[0], reverse=True)
        elites = [g for _, g in scored[: max(2, args.pop // 3)]]
        breed = elites + library
        nxt = list(elites)
        k = 0
        while len(nxt) < args.pop:
            if rng.random() < 0.5:
                parent = breed[int(rng.integers(0, len(breed)))]
                child = mutate(parent, rng, new_id=f"SDS_G{gen}_{k}")
            else:
                a = breed[int(rng.integers(0, len(breed)))]
                b = breed[int(rng.integers(0, len(breed)))]
                child = recombine(a, b, rng, new_id=f"SDS_X{gen}_{k}")
            nxt.append(child)
            k += 1
        pop = nxt

    archive.sort(key=lambda r: r.get("J", -9), reverse=True)
    summary = {
        "protocol_id": "STRATEGIC_DEMAND_SEARCHER_V1",
        "finished_utc": _now(),
        "search_results_are_gate_B": False,
        "ruleset": "RULESET_V3_M1",
        "anchor_B_cheap_delta_B": delta_B,
        "best": archive[0] if archive else None,
        "n_archive": len(archive),
        "confirmation": "NOT RUN. 2500001 remains pristine. Development-eligible only is not confirmation.",
        "ppo": "NOT STARTED",
        "mode": "mutate_from_screen" if args.mutate_from_screen else "search",
        "n_development_eligible": sum(
            1 for r in archive if r.get("development_eligible")),
    }
    _write(out_dir / "summary.json", summary)
    print("=" * 74)
    print("SEARCH FINISHED — not Gate B, not V3_STRATEGIC_DEMAND = VALIDATED")
    if archive:
        best = archive[0]
        print(f"best J={best.get('J'):+.3f}  dG={best.get('delta_G'):+.3f}  "
              f"id={best['genome']['genome_id']}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
