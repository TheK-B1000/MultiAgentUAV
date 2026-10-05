"""BEHAVIOR_SIGNATURES_2V2_V1: are the 2v2 A and B strategies behaviourally different?

Frozen spec: artifacts/strategic_demand/sppo/2v2_strengthening/BEHAVIOR_SIGNATURES_2V2_V1_SPEC.json

Read-only rollouts of Ours (dual-branch teachers) and Fully Shared+z+r; every strategy plays
every pole on the same seeds. Each episode is reduced to ONE value per metric (the episode is
the statistical unit), strategies are paired by seed, and the 10 pole-specific tests per system
are Holm-corrected as one family.

    # smoke (999xxxxx only, output under 2v2_strengthening/smoke/)
    .venv/Scripts/python.exe experiments/run_behavior_signatures_2v2.py --smoke
    # confirmatory (after the 4v4 chain finishes)
    .venv/Scripts/python.exe experiments/run_behavior_signatures_2v2.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.strengthening_2v2_common as C  # noqa: E402

SPEC_ID = "BEHAVIOR_SIGNATURES_2V2_V1"
SYSTEMS = ("Ours", "Fully_Shared_z_r")
STRATS = ("A", "B")
POLES = ("A", "B")
TESTED = ("M1_home_share", "M2_forward_share", "M3_enemy_flag_distance", "M4_spacing", "M5_first_grab_tick")
N_MACROS = 5
HORIZON = 240
SMOKE_SEEDS = (99971001, 99971002)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ per-episode metrics
class EpisodeRecorder:
    """Accumulates per-tick readings and reduces them to one value per metric."""

    def __init__(self):
        self.home, self.fwd, self.dist, self.spacing = [], [], [], []
        self.first_grab = None
        self.macro_counts = np.zeros(N_MACROS, dtype=np.int64)

    def tick(self, s: dict, macros: np.ndarray, decided: np.ndarray, t: int) -> None:
        pos = s["pos"]
        active = s["alive"] & ~s["tagged"]
        if self.first_grab is None and bool((s["alive"] & s["carrying"]).any()):
            self.first_grab = int(t)
        if active.any():
            d_home = np.linalg.norm(pos - s["own_home"][None, :], axis=-1)
            self.home.append(float((active & ~s["carrying"] & (d_home <= C.R_HOME)).sum() / active.sum()))
            mid = float(s["cols"]) * 0.5
            enemy_right = float(s["enemy_home"][0]) > mid
            on_enemy = (pos[:, 0] > mid) if enemy_right else (pos[:, 0] < mid)
            self.fwd.append(float((active & on_enemy).sum() / active.sum()))
            d_flag = np.linalg.norm(pos[active] - s["enemy_flag"][None, :], axis=-1)
            self.dist.append(float(d_flag.mean()))
        if active.size == 2 and bool(active.all()):
            self.spacing.append(float(np.linalg.norm(pos[0] - pos[1])))
        for m, d in zip(macros, decided):
            if d:
                self.macro_counts[int(m) % N_MACROS] += 1

    def values(self) -> dict:
        tot = int(self.macro_counts.sum())
        out = {
            "M1_home_share": float(np.mean(self.home)) if self.home else None,
            "M2_forward_share": float(np.mean(self.fwd)) if self.fwd else None,
            "M3_enemy_flag_distance": float(np.mean(self.dist)) if self.dist else None,
            "M4_spacing": float(np.mean(self.spacing)) if self.spacing else None,
            "M5_first_grab_tick": float(self.first_grab) if self.first_grab is not None else float(HORIZON),
            "grab_occurred": int(self.first_grab is not None),
            "n_decisions": tot,
        }
        for i in range(N_MACROS):
            out[f"M6_macro{i}_share"] = float(self.macro_counts[i] / tot) if tot else None
        return out


# ------------------------------------------------------------------ analysis (pure)
def paired_diffs(rows: list[dict], system: str, pole: str, metric: str) -> tuple[np.ndarray, int]:
    by = {(r["strategy"], int(r["seed"])): r[metric] for r in rows
          if r["system"] == system and r["pole"] == pole}
    seeds = sorted({s for (_st, s) in by})
    d, dropped = [], 0
    for s in seeds:
        a, b = by.get(("A", s)), by.get(("B", s))
        if a is None or b is None:
            dropped += 1
            continue
        d.append(float(a) - float(b))
    return np.asarray(d), dropped


def analyse(rows: list[dict]) -> dict:
    out = {}
    for system in SYSTEMS:
        tests, effects = {}, {}
        for m in TESTED:
            for pole in POLES:
                d, dropped = paired_diffs(rows, system, pole, m)
                key = f"{m}@pole{pole}"
                tests[key] = C.sign_flip_p(d)
                effects[key] = {**C.mean_ci(d), "dropped_pairs": dropped} if d.size else {"n": 0, "dropped_pairs": dropped}
        h = C.holm(tests)
        sigs = {}
        for m in TESTED:
            ea, eb = effects[f"{m}@poleA"], effects[f"{m}@poleB"]
            same = ("mean" in ea and "mean" in eb and np.sign(ea["mean"]) == np.sign(eb["mean"])
                    and ea["mean"] != 0)
            both = h[f"{m}@poleA"]["reject"] and h[f"{m}@poleB"]["reject"]
            sigs[m] = {"signature": bool(same and both), "same_sign": bool(same), "both_rejected_holm": bool(both),
                       "direction": ("A>B" if ea.get("mean", 0) > 0 else "A<B") if same else None}
        n_sig = sum(v["signature"] for v in sigs.values())
        desc = {}
        for st in STRATS:
            for pole in POLES:
                sub = [r for r in rows if r["system"] == system and r["strategy"] == st and r["pole"] == pole]
                cell = {}
                for k in [f"M6_macro{i}_share" for i in range(N_MACROS)] + ["grab_occurred"] + list(TESTED):
                    v = [r[k] for r in sub if r[k] is not None]
                    cell[k] = C.mean_ci(v) if v else None
                desc[f"{st}@pole{pole}"] = cell
        opp = {}
        for st in STRATS:
            for m in TESTED:
                byp = {(r["pole"], int(r["seed"])): r[m] for r in rows
                       if r["system"] == system and r["strategy"] == st}
                d = [byp[("A", s)] - byp[("B", s)] for (p, s) in byp
                     if p == "A" and byp.get(("B", s)) is not None and byp[("A", s)] is not None]
                opp[f"{st}:{m}"] = C.mean_ci(d) if d else None
        out[system] = {"tests": h, "effects": effects, "signatures": sigs, "n_signatures": n_sig,
                       "distinct_supported": bool(n_sig >= 2), "descriptive_cells": desc,
                       "opponent_effect_descriptive": opp}
    ours, fs = out["Ours"]["signatures"], out["Fully_Shared_z_r"]["signatures"]
    out["preserved_by_Fully_Shared_z_r"] = {
        m: bool(ours[m]["signature"] and fs[m]["signature"] and ours[m]["direction"] == fs[m]["direction"])
        for m in TESTED}
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default=None, help="i/n: run only every n-th episode starting at i; no seal")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import torch
    from rl.causal_supervision import decision_mask_from_core

    spec = C.load_frozen_spec(SPEC_ID)
    lo, hi = spec["design"]["seeds"]["block"]
    seeds = list(SMOKE_SEEDS) if args.smoke else list(range(lo, hi + 1))
    C.check_block(SPEC_ID, seeds[0], seeds[-1], smoke=args.smoke)
    out_dir = C.STRENGTH_DIR / ("smoke" if args.smoke else "results") / SPEC_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{SPEC_ID}_RESULT.json"
    if result_path.exists() and not args.smoke:
        raise SystemExit(f"REFUSING: {result_path.name} already sealed (write-once)")
    partial = out_dir / "rows.PARTIAL.jsonl"
    if args.smoke and not args.shard:
        for f in [partial, *partial.parent.glob(f"{partial.stem}.shard*.jsonl")]:
            if f.exists():
                f.unlink()
    write_to = C.shard_path(partial, args.shard)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    genomes, identity, cert = C.resolve_poles()
    ours = C.load_ours(device, pins=spec["systems"]["Ours"]["pins"])
    st = spec["systems"]["Fully_Shared_z_r"]
    student = C.load_student(device, st["checkpoint"], st["sha256"])
    actors = {("Ours", s): ours[s] for s in STRATS}
    actors.update({("Fully_Shared_z_r", s): C.StudentActor(student, 0 if s == "A" else 1) for s in STRATS})

    done = C.read_partial(partial)
    total = len(actors) * len(POLES) * len(seeds)
    print(f"{SPEC_ID} {'SMOKE' if args.smoke else 'CONFIRMATORY'}  seeds {seeds[0]}..{seeds[-1]}  "
          f"episodes {total}  resumed {len(done)}  poles attested vs {cert}", flush=True)
    idx = -1
    for (system, strat), actor in actors.items():
        for pole in POLES:
            for seed in seeds:
                idx += 1
                key = f"{system}|{strat}|{pole}|{seed}"
                if key in done or not C.in_shard(idx, args.shard):
                    continue
                rec = EpisodeRecorder()

                def on_tick(core, obs, action, t, rec=rec):
                    s = C.snapshot(core)
                    macros = np.asarray(action).reshape(C.N, -1)[:, 0]
                    decided = np.asarray(decision_mask_from_core(core, C.N, side="blue").detach().cpu())[0]
                    rec.tick(s, macros, decided, t)

                res = C.run_episode(actor, pole, seed, device, genomes, identity, on_tick=on_tick,
                                    context=f"{SPEC_ID} {system} z{strat}")
                row = {"key": key, "system": system, "strategy": strat, "pole": pole, "seed": seed,
                       **rec.values(), **res}
                C.append_partial(write_to, row)
                done[key] = row
                print(f"  [{len(done)}/{total}] {key}  steps {res['steps']}  "
                      f"home {row['M1_home_share']:.3f} fwd {row['M2_forward_share']:.3f}", flush=True)

    if args.shard:
        print(f"shard {args.shard} finished; run without --shard to merge and seal", flush=True)
        return 0
    done = C.read_partial(partial)
    if len(done) != total:
        raise SystemExit(f"REFUSING to seal: {len(done)}/{total} episodes present")
    rows = list(done.values())
    fields = list(rows[0].keys())
    with (out_dir / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    analysis = analyse(rows)
    result = {"id": SPEC_ID, "status": "SMOKE" if args.smoke else "SEALED", "utc": _now(),
              "spec_sha256": C.sha256(C.STRENGTH_DIR / f"{SPEC_ID}_SPEC.json"),
              "seeds": [seeds[0], seeds[-1]], "n_seeds": len(seeds), "device": device,
              "pole_certification": cert, "analysis": analysis,
              "note": "wins/scores in rows.csv are traceability only (spec: not_evidence)"}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.smoke:
        from experiments.seed_registry import set_status
        set_status(SPEC_ID, "SPENT", note=f"sealed {result_path.name}")
    for system in SYSTEMS:
        a = analysis[system]
        print(f"\n{system}: signatures {a['n_signatures']}/5  distinct_supported={a['distinct_supported']}")
        for m, v in a["signatures"].items():
            ea, eb = a["effects"][f"{m}@poleA"], a["effects"][f"{m}@poleB"]
            print(f"  {m:24s} A-B poleA {ea.get('mean', float('nan')):+.3f}  poleB {eb.get('mean', float('nan')):+.3f}"
                  f"  holm {a['tests'][m + '@poleA']['p_holm']:.4f}/{a['tests'][m + '@poleB']['p_holm']:.4f}"
                  f"  {'SIGNATURE' if v['signature'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
