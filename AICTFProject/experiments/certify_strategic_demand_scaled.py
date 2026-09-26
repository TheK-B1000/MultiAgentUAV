"""Strategic-demand certification at a given team size.

Implements SIZE_NORMALIZED_POLE_SEMANTICS_SPEC.json#CERTIFICATION_PROTOCOL.

Asks the same question the 2v2 certification asked, with the size-normalized probe and
pole gate: do the frozen poles demand OPPOSITE strategic responses?

    delta_A = WR(GUARD, pole A) - WR(BREACH, pole A)      GUARD should win on the wide zone
    delta_B = WR(BREACH, pole B) - WR(GUARD, pole B)      BREACH should win on the tight one

CERTIFIED iff LCB95(delta_A) > 0 AND LCB95(delta_B) > 0 -- the same rule and the same
bootstrap (n_boot=20000, alpha=0.05, rng_seed=7, seed as the resampling unit) used
everywhere else in this project. No threshold is adjusted for team size.

GUARD and BREACH are run on the SAME seed, so the comparison is paired per seed.

No policy is trained, loaded or updated: both blue styles are scripted.

Run:  python experiments/certify_strategic_demand_scaled.py --team-size 4 --n-seeds 64 --device cpu
      python experiments/certify_strategic_demand_scaled.py --team-size 3 --variant guard_distributed_v2_n192 --seed-base 12321001 --n-seeds 192 --device cpu
      python experiments/certify_strategic_demand_scaled.py --team-size 4 --n-seeds 2 --probe
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "artifacts" / "strategic_demand" / "sppo"

# Certification seed blocks, disjoint per team size and from every prior block.
# N=3 v1 default is reserved but unused: GUARD_DISTRIBUTED_V2 is the first 3v3
# certification (see 3V3_STRATEGIC_DEMAND_N192_AMENDMENT.json).
SEED_BASE = {3: 12_300_001, 4: 12_400_001, 6: 12_600_001}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mean_ci(vals, n_boot=20000, alpha=0.05, rng_seed=7):
    a = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = a[idx].mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"mean": float(a.mean()), "lcb95": lo, "ucb95": hi, "n": int(a.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-size", type=int, required=True, choices=(2, 3, 4, 6))
    ap.add_argument("--n-seeds", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--probe", action="store_true",
                    help="timing probe only; writes no certification record")
    ap.add_argument("--variant", default="v1",
                    help="tag distinguishing this GUARD implementation in the output "
                        "filename and record, e.g. 'v1' or 'guard_distributed_v2'. "
                        "Prevents a v2 run from colliding with (or overwriting) a v1 record.")
    ap.add_argument("--seed-base", type=int, default=None,
                    help="override SEED_BASE[team_size]; REQUIRED to differ from the v1 "
                        "block whenever --variant is not 'v1', so a fresh block is spent "
                        "explicitly rather than by an unstated default")
    ap.add_argument("--pole-b-genome-json", default=None,
                    help="path to a JSON SDSGenome (genome_id/derived_from/base_opponent/"
                        "overlay/opening_hold_steps) to substitute for Pole B, e.g. a "
                        "candidate from a confirmatory-redesign ladder. Defaults to None, "
                        "which reproduces the exact pre-existing behaviour "
                        "(pole_B_genome(N), canonical OP7 + the size-normalized defender "
                        "gate) byte-for-byte -- every existing 2v2/4v4/6v6 certification "
                        "call is unaffected by this flag's mere existence.")
    ap.add_argument("--spec", default=None,
                    help="frozen certification spec governing this run (required unless --probe)")
    ap.add_argument("--experiment-id", default=None,
                    help="seed-registry owner of the block (required unless --probe)")
    ap.add_argument("--seed-class", default="sealed_confirmatory",
                    choices=("sealed_confirmatory", "exploratory", "smoke"))
    args = ap.parse_args()

    N = int(args.team_size)
    n_seeds = int(args.n_seeds)
    variant = str(args.variant)

    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import (
        _with_full_team_defender_gate, expected_profile, pole_A_genome, pole_B_genome,
    )
    from experiments.sds_genome import SDSGenome

    # Team size must reach the scripted-episode env: it reads this module global.
    S.AGENTS = N
    if int(S.AGENTS) != N:
        raise SystemExit("FAIL-CLOSED: could not set team size on the episode runner")

    gA = pole_A_genome(N)
    if args.pole_b_genome_json:
        p = Path(args.pole_b_genome_json)
        if not p.is_file():
            raise SystemExit(f"FAIL-CLOSED: --pole-b-genome-json not found: {p}")
        raw = json.loads(p.read_text(encoding="utf-8"))
        candidate = SDSGenome.from_dict(raw)
        if candidate.base_opponent != "OP7":
            raise SystemExit(f"FAIL-CLOSED: --pole-b-genome-json base_opponent="
                             f"{candidate.base_opponent!r}, expected 'OP7' -- Pole B "
                             f"candidates are OP7 overlays, not a different dispatch key")
        gB = _with_full_team_defender_gate(candidate, N)
        print(f"  Pole B SOURCE  candidate genome {candidate.genome_id!r} from "
              f"{p.name} (NOT canonical pole_B_genome({N}))")
    else:
        gB = pole_B_genome(N)
    pA, pB = expected_profile("OP6", gA), expected_profile("OP7", gB)

    # Fail closed if the size normalization did not reach the resolved profiles.
    for tag, prof in (("A", pA), ("B", pB)):
        got = int(getattr(prof, "min_alive_for_defender", -1))
        if got != N:
            raise SystemExit(f"FAIL-CLOSED: pole {tag} resolved min_alive_for_defender={got}, "
                             f"expected {N} under the size-normalized semantics")

    v1_base = SEED_BASE.get(N)
    base = int(args.seed_base) if args.seed_base is not None else v1_base
    if base is None:
        raise SystemExit(f"FAIL-CLOSED: no default seed base for team size {N}; "
                         f"pass --seed-base explicitly")
    if variant != "v1" and base == v1_base and not args.probe:
        raise SystemExit(
            f"FAIL-CLOSED: --variant={variant!r} but --seed-base resolved to the v1 block "
            f"({v1_base}). A non-v1 variant must spend a FRESH block, passed explicitly via "
            f"--seed-base, not the v1 default.")
    # Timing probes must never spend the certification block. Offset far above any
    # preregistered strategic-demand seed range used in this program.
    cert_base = base
    if args.probe:
        base = cert_base + 50_000_000

    print("=" * 78)
    print(f"STRATEGIC DEMAND CERTIFICATION  {N}v{N}  variant={variant}  device={args.device}")
    print("=" * 78)
    print(f"  utc            {_now()}")
    print(f"  seeds          {base if not args.probe else 'PROBE'}"
          f"..+{n_seeds}   (paired: GUARD and BREACH share each seed)")
    print(f"  GUARD          {(N + 1) // 2} of {N} defend  (indices "
          f"{list(range(N - (N + 1) // 2, N))})")
    print(f"  pole A         OP6+overlay  min_alive={getattr(pA, 'min_alive_for_defender', None)}"
          f"  zone_frac={getattr(pA, 'defender_zone_frac', None)}"
          f"  threat_radius={getattr(pA, 'threat_radius', None)}")
    print(f"  pole B         OP7+overlay  min_alive={getattr(pB, 'min_alive_for_defender', None)}"
          f"  zone_frac={getattr(pB, 'defender_zone_frac', None)}"
          f"  threat_radius={getattr(pB, 'threat_radius', None)}"
          f"  lock_defender={getattr(pB, 'lock_defender', None)}"
          f"  lock_attacker={getattr(pB, 'lock_attacker', None)}"
          f"{'  [CANDIDATE: ' + gB.genome_id + ']' if args.pole_b_genome_json else ''}")
    print(f"  gate           LCB95(delta_A) > 0 AND LCB95(delta_B) > 0")
    print("=" * 78, flush=True)

    # ---- identity + pre-flight, BEFORE any seed is spent ------------------------------
    # The certification record is the HANDOFF ARTIFACT every downstream stage consumes: it
    # pins each pole's identity with the same hash pole_attestation.pole_identity computes.
    from experiments.pole_attestation import _canonical_overlay, pole_config_hash

    def _ident(policy: str, g) -> dict:
        ov = _canonical_overlay(g.overlay)
        return {"genome_id": str(g.genome_id), "derived_from": str(g.derived_from),
                "base_opponent": str(g.base_opponent),
                "opening_hold_steps": int(getattr(g, "opening_hold_steps", 0) or 0),
                "pole_config_hash": pole_config_hash(policy, N, g.genome_id, ov)}
    identA, identB = _ident("A", gA), _ident("B", gB)

    suffix = "" if variant == "v1" else f"_{variant.upper()}"
    label = f"STRATEGIC_DEMAND_{N}v{N}{suffix}_CERTIFICATION"
    out = OUT_DIR / f"{label}.json"
    rows_csv = OUT_DIR / f"{label.lower()}_rows.csv"
    seeds_list = list(range(base, base + n_seeds))
    state = spec_path = None
    seed_class, experiment_id = str(args.seed_class), args.experiment_id
    if not args.probe:
        import experiments.run_state as rs
        from experiments import seed_registry as sr
        # One-shot, checked before the first episode: a collision found AFTER 768 episodes
        # would waste the whole seed block.
        existing = [p.name for p in (out, rows_csv) if p.exists()]
        if existing:
            raise SystemExit(f"REFUSING: {existing} already exist; certification is one-shot")
        if not args.spec:
            raise SystemExit("REFUSING: --spec <frozen certification spec> is required; a "
                             "certification that spends seeds must be governed by a frozen spec")
        spec_path = Path(args.spec)
        if not spec_path.is_file() and (OUT_DIR / args.spec).is_file():
            spec_path = OUT_DIR / args.spec
        if not spec_path.is_file():
            raise SystemExit(f"FAIL-CLOSED: spec not found: {args.spec}")
        spec_status = str(json.loads(spec_path.read_text(encoding="utf-8")).get("status", ""))
        if not spec_status.startswith("FROZEN"):
            raise SystemExit(f"REFUSING: {spec_path.name} is {spec_status!r}, not FROZEN")
        if not experiment_id:
            raise SystemExit("REFUSING: --experiment-id is required (the registry owner of "
                             "the seed block)")
        ok, msg = sr.check_block(seeds_list[0], seeds_list[-1], seed_class,
                                 experiment_id=experiment_id)
        if not ok:
            raise SystemExit(f"REFUSING (Rule 9): {msg}")
        state = rs.RunState(OUT_DIR, label)
        state.begin(seed_base=base, n_seeds=n_seeds, team_size=N, variant=variant,
                    seed_class=seed_class, spec=spec_path.name,
                    pole_config_hash={"A": identA["pole_config_hash"],
                                      "B": identB["pole_config_hash"]})
        print(f"  pre-flight     spec {spec_path.name} [{spec_status}]; registry: {msg}")
        print(f"  pole identity  A {identA['genome_id']} {identA['pole_config_hash'][:12]}   "
              f"B {identB['genome_id']} {identB['pole_config_hash'][:12]}", flush=True)

    from experiments.tqdm_loop import set_postfix, tqdm_iter

    rows = []
    t0 = time.time()
    bar = tqdm_iter(range(n_seeds), desc=f"certify demand {N}v{N}", unit="seed")
    for i in bar:
        seed = base + i
        set_postfix(bar, f"seed={seed}")
        r = {"seed": seed}
        for pole, genome in (("A", gA), ("B", gB)):
            for style_tag, style in (("guard", S.GUARD), ("breach", S.BREACH)):
                ep = S.run_episode(style=style, genome=genome, seed=seed, device=args.device)
                r[f"{pole}_{style_tag}"] = int(ep["win"])
        rows.append(r)
        if (i + 1) % 8 == 0 or i == 0:
            el = time.time() - t0
            print(f"  seed {i + 1}/{n_seeds}  elapsed {el:6.1f}s  "
                  f"({el / (i + 1):.2f}s/seed, 4 episodes each)", flush=True)

    dA = np.array([r["A_guard"] - r["A_breach"] for r in rows], dtype=float)
    dB = np.array([r["B_breach"] - r["B_guard"] for r in rows], dtype=float)
    ciA, ciB = _mean_ci(dA), _mean_ci(dB)

    # Standing rule: any tie or reversal on a gated quantity triggers a mandatory row-level
    # integrity audit, written BEFORE its rows are interpreted. Checked before `certified` is
    # even computed, matching the ladder eval scripts' convention -- this script previously
    # lacked this gate (found retroactively: 4v4 GUARD_DISTRIBUTED_V2 tied at delta_A=0.0000
    # exactly and was manually audited after the fact rather than caught here).
    tie_or_reversal = None
    if ciA["mean"] == 0.0:
        tie_or_reversal = "delta_A exactly zero (tie)"
    elif ciA["mean"] < 0:
        tie_or_reversal = f"delta_A reversed (mean={ciA['mean']:+.4f})"
    elif ciB["mean"] == 0.0:
        tie_or_reversal = "delta_B exactly zero (tie)"
    elif ciB["mean"] < 0:
        tie_or_reversal = f"delta_B reversed (mean={ciB['mean']:+.4f})"

    if tie_or_reversal is not None:
        n_tied = int(np.sum(np.array([r["A_guard"] for r in rows]) ==
                            np.array([r["A_breach"] for r in rows])))
        n_guard_only = int(np.sum((np.array([r["A_guard"] for r in rows]) == 1) &
                                  (np.array([r["A_breach"] for r in rows]) == 0)))
        n_breach_only = int(np.sum((np.array([r["A_guard"] for r in rows]) == 0) &
                                   (np.array([r["A_breach"] for r in rows]) == 1)))
        seeds_seen = [r["seed"] for r in rows]
        audit = {
            "record": f"Row-level integrity audit ({N}v{N}, variant={variant})",
            "status": "FLAGGED", "utc": _now(),
            "triggered_by": tie_or_reversal,
            "n_rows": len(rows),
            "n_unique_seeds": len(set(seeds_seen)),
            "seed_range": [min(seeds_seen), max(seeds_seen)] if seeds_seen else None,
            "poleA_split": {"tied": n_tied, "guard_only": n_guard_only,
                            "breach_only": n_breach_only},
            "point_estimates": {"delta_A": ciA["mean"], "delta_B": ciB["mean"]},
            "note": ("a tie or reversal is a legitimate possible outcome, not necessarily an "
                     "evaluator defect. This audit checks for duplicate/missing seeds and a "
                     "symmetric win/loss split before any interpretation is offered."),
            "classification": ("GENUINE -- seeds unique, range as expected, split symmetric"
                               if len(set(seeds_seen)) == len(seeds_seen) else
                               "SUSPECT -- duplicate or missing seeds, investigate before reading"),
        }
        audit_out = OUT_DIR / f"TIE_REVERSAL_AUDIT_{N}v{N}_{variant.upper()}.json"
        print(f"\n  TIE/REVERSAL on {tie_or_reversal}")
        print(f"  classification: {audit['classification']}")
        if args.probe:
            print("  --probe: tie/reversal audit NOT written.")
        else:
            audit_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
            print(f"  integrity audit written: {audit_out}")
            if audit["classification"] != "GENUINE -- seeds unique, range as expected, split symmetric":
                raise SystemExit(
                    f"REFUSING to write the frozen certification result: the tie/reversal audit "
                    f"classified this run as SUSPECT. Investigate {audit_out} before proceeding; "
                    f"a tie or reversal is a legitimate outcome only once the rows themselves are "
                    f"verified clean.")

    certified = bool(ciA["lcb95"] > 0 and ciB["lcb95"] > 0)

    cells = {
        "poleA_guard_wr": float(np.mean([r["A_guard"] for r in rows])),
        "poleA_breach_wr": float(np.mean([r["A_breach"] for r in rows])),
        "poleB_guard_wr": float(np.mean([r["B_guard"] for r in rows])),
        "poleB_breach_wr": float(np.mean([r["B_breach"] for r in rows])),
    }

    print("\n  cell win rates")
    for k, v in cells.items():
        print(f"    {k:20s} {v:.4f}")
    print("\n  PRIMARY")
    print(f"    delta_A (GUARD-BREACH on A) {ciA['mean']:+.4f} "
          f"[{ciA['lcb95']:+.4f}, {ciA['ucb95']:+.4f}]")
    print(f"    delta_B (BREACH-GUARD on B) {ciB['mean']:+.4f} "
          f"[{ciB['lcb95']:+.4f}, {ciB['ucb95']:+.4f}]")
    print(f"\n  {N}v{N} STRATEGIC DEMAND: {'CERTIFIED' if certified else 'NOT CERTIFIED'}")

    if args.probe:
        print("\n  --probe: no record written.")
        return 0

    # ---- seal --------------------------------------------------------------------
    # Evidence in the sealer's contract: one row per (seed, pole, style) episode, each
    # carrying the identity of the pole it was actually run against. The record's status
    # is owned by run_state.seal (SEALED / AUDIT_FAILED); VERDICT (CERTIFIED /
    # NOT_CERTIFIED) is the separate scientific result. Downstream consumption requires
    # BOTH: pole_attestation.certified_pole_genome refuses anything but VERDICT=CERTIFIED
    # in a record that is SEALED with a passing audit.
    long_rows = []
    for r in rows:
        for pole, ident in (("A", identA), ("B", identB)):
            for style in ("guard", "breach"):
                long_rows.append({"seed": int(r["seed"]), "pole": pole, "style": style,
                                  "win": int(r[f"{pole}_{style}"]),
                                  "genome_id": ident["genome_id"],
                                  "pole_config_hash": ident["pole_config_hash"]})
    with rows_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(long_rows[0]))
        w.writeheader()
        w.writerows(long_rows)

    PIN = {"A": identA["pole_config_hash"], "B": identB["pole_config_hash"]}
    GID = {"A": identA["genome_id"], "B": identB["genome_id"]}
    verdict = "CERTIFIED" if certified else "NOT_CERTIFIED"
    poles_block = {
        "A": {"base": "OP6", "overlay": dict(gA.overlay or {}), **identA,
              "min_alive_for_defender": getattr(pA, "min_alive_for_defender", None),
              "defender_zone_frac": getattr(pA, "defender_zone_frac", None),
              "threat_radius": getattr(pA, "threat_radius", None)},
        "B": {"base": "OP7", "overlay": dict(gB.overlay or {}), **identB,
              "min_alive_for_defender": getattr(pB, "min_alive_for_defender", None),
              "defender_zone_frac": getattr(pB, "defender_zone_frac", None),
              "threat_radius": getattr(pB, "threat_radius", None),
              "lock_defender": getattr(pB, "lock_defender", None),
              "lock_attacker": getattr(pB, "lock_attacker", None),
              "candidate_genome_id": gB.genome_id if args.pole_b_genome_json else None,
              "candidate_source": (str(args.pole_b_genome_json)
                                   if args.pole_b_genome_json else None)},
    }
    handoff = {
        "role": "the authoritative pole definition for every downstream stage at this team "
                "size; consumed through pole_attestation.resolve_pole_genome, which refuses "
                "unless VERDICT is CERTIFIED and the record is SEALED with a passing audit",
        "team_size": N,
        "pole_config_hash": dict(PIN),
        "hash_function": "experiments/pole_attestation.py::pole_config_hash",
        "downstream_must": "record pole_identity(...) per pole, whose pole_config_hash must "
                           "equal the value above and whose certification_sha256 is this "
                           "file's sha256; Layer 3 of attest_cross_scale_identity checks it",
    }

    def _vec(rs_rows, pole, style):
        m = {int(x["seed"]): int(x["win"]) for x in rs_rows
             if x["pole"] == pole and x["style"] == style}
        return np.array([m[s] for s in seeds_list], dtype=np.float64)

    def _verdict_matches_evidence(rs_rows):
        """Recompute the verdict from the rows on disk with the AUDIT's bootstrap."""
        a = rs._bootstrap(_vec(rs_rows, "A", "guard") - _vec(rs_rows, "A", "breach"),
                          20000, 0.05, 7)
        b = rs._bootstrap(_vec(rs_rows, "B", "breach") - _vec(rs_rows, "B", "guard"),
                          20000, 0.05, 7)
        want = "CERTIFIED" if (a["lcb95"] > 0 and b["lcb95"] > 0) else "NOT_CERTIFIED"
        # Read the verdict from the PAYLOAD being sealed (late-bound), not a local: a check
        # against a local would pass even if the written record said something else.
        stated = payload["VERDICT"]
        return (want == stated,
                f"rows imply {want} (LCB95 delta_A {a['lcb95']:+.4f}, delta_B "
                f"{b['lcb95']:+.4f}); record states {stated}")

    def _handoff_matches_consumer_rebuild(_rs_rows):
        """Rebuild each pole with the SAME code downstream consumption uses, and require
        its hash to equal the pinned handoff, the pole block and the per-row identity."""
        from experiments.pole_attestation import rebuild_certified_genome
        ok, msgs = True, []
        # Every value read from the PAYLOAD being sealed (late-bound), so a record altered
        # after these dicts were built is still caught. Chain: rows <-> PIN (derived check)
        # <-> the record's pole blocks and POLE_HANDOFF <-> the consumer's rebuild.
        for p in ("A", "B"):
            block = payload["poles"][p]
            g = rebuild_certified_genome(p, N, block, source_name="pre-seal handoff")
            h = pole_config_hash(p, N, g.genome_id, _canonical_overlay(g.overlay))
            good = (h == PIN[p] == payload["POLE_HANDOFF"]["pole_config_hash"][p]
                    == block["pole_config_hash"])
            ok = ok and good
            msgs.append(f"{p}: consumer rebuild {h[:12]} {'==' if good else '!='} pinned {PIN[p][:12]}")
        return (ok, "; ".join(msgs))

    plan = rs.AuditPlan(
        rows_csv=rows_csv, expected_rows=4 * n_seeds, expected_seeds=seeds_list,
        group_by=("pole", "style"), seed_field="seed", int_fields=("seed",),
        binary_fields=("win",),
        derived={
            "pole_config_hash": rs.Derived("pinned certified hash of the row's pole",
                                           lambda r: PIN[r["pole"]]),
            "genome_id": rs.Derived("certified genome id of the row's pole",
                                    lambda r: GID[r["pole"]]),
        },
        spec_path=spec_path,
        claims=[
            rs.Claim("delta_A", {k: ciA[k] for k in ("mean", "lcb95", "ucb95")},
                     minuend={"pole": "A", "style": "guard"},
                     subtrahend={"pole": "A", "style": "breach"}),
            rs.Claim("delta_B", {k: ciB[k] for k in ("mean", "lcb95", "ucb95")},
                     minuend={"pole": "B", "style": "breach"},
                     subtrahend={"pole": "B", "style": "guard"}),
        ],
        n_boot=20000, alpha=0.05, rng_seed=7,
        seed_class=seed_class, experiment_id=experiment_id,
        invariants={"verdict_matches_evidence": _verdict_matches_evidence,
                    "handoff_matches_consumer_rebuild": _handoff_matches_consumer_rebuild},
    )
    # status is owned by seal(); it must not appear in the payload.
    payload = {
        "record": f"Strategic demand certification {N}v{N} ({variant})",
        "one_shot": True, "utc": _now(),
        "implements": "SIZE_NORMALIZED_POLE_SEMANTICS_SPEC.json#CERTIFICATION_PROTOCOL",
        "governing_spec": spec_path.name,
        "variant": variant, "supersedes": None if variant == "v1" else f"v1 ({v1_base})",
        "team_size": N, "device": args.device,
        "guard_defenders": (N + 1) // 2,
        "guard_defender_indices": list(range(N - (N + 1) // 2, N)),
        "seeds": {"base": base, "n": n_seeds, "paired": True, "seed_class": seed_class,
                  "registry_experiment_id": experiment_id},
        "poles": poles_block,
        "cell_win_rates": cells,
        "PRIMARY": {"delta_A": ciA, "delta_B": ciB},
        "bootstrap": {"procedure": "paired percentile bootstrap over seeds",
                      "samples": 20000, "alpha": 0.05, "rng_seed": 7},
        "gate": "LCB95(delta_A) > 0 AND LCB95(delta_B) > 0",
        "VERDICT": verdict,
        "VERDICT_vs_status": "VERDICT is the scientific result; 'status' is set by "
                             "run_state.seal and says whether that result re-derives from the "
                             "evidence on disk. Downstream stages require CERTIFIED and SEALED.",
        "POLE_HANDOFF": handoff,
        "rows_csv": rows_csv.name,
        "tie_reversal_audit": (f"TIE_REVERSAL_AUDIT_{N}v{N}_{variant.upper()}.json"
                               if tie_or_reversal is not None else None),
        "total_episodes": 4 * n_seeds,
        "note_if_not_certified": (
            "SIZE_NORMALIZED_POLE_SEMANTICS_SPEC.json#NORMALIZATION_1_GUARD_PROBE records that "
            "multiple defenders CONVERGE on a single threat rather than covering distinct "
            "intruders. That stacking is a candidate explanation for a failure at N>2 and must "
            "be acknowledged before concluding the poles do not demand different behaviour."),
    }
    audit = rs.seal(out_path=out, payload=payload, plan=plan, state=state, strict=False)
    if seed_class != "smoke":
        sr.set_status(experiment_id, "SPENT",
                      note=f"{N}v{N} certification sealed "
                           f"{'SEALED' if audit['passed'] else 'AUDIT_FAILED'}; VERDICT={verdict}")
    if not audit["passed"]:
        print(f"  !! certification AUDIT_FAILED ({audit['failed_checks']}): the record is not "
              f"trusted, and no downstream stage will consume it.")
        return 1

    # Round-trip: consume the sealed record exactly as every downstream stage will.
    if certified:
        from experiments.pole_attestation import (
            governing_certification, pole_identity, resolve_pole_genome,
        )
        _v, gov = governing_certification(N)
        if gov.resolve() != out.resolve():
            print(f"  !! handoff NOT governing: {N}v{N} resolves to {gov.name}, not {out.name}. "
                  f"Downstream stages will not consume this record.")
            return 1
        for pol in ("A", "B"):
            got = pole_identity(pol, N, resolve_pole_genome(pol, N))
            ok = got["pole_config_hash"] == PIN[pol]
            print(f"  handoff round-trip pole {pol}: consumer hash "
                  f"{got['pole_config_hash'][:12]} {'==' if ok else '!='} recorded "
                  f"{PIN[pol][:12]}  ({'OK' if ok else 'MISMATCH'})")
            if not ok:
                return 1
    return 0 if certified else 1


if __name__ == "__main__":
    raise SystemExit(main())
