"""Mechanical preflight on the assigned S2 training seed, before either arm launches.

Same discipline CCP_S2_SEED_ASSIGNMENT.json required of the collection block: the seed is not
trusted because it looks disjoint, it is checked. Every check is mechanical against the frozen
records and the real config objects the runner will actually build -- not against a
description of them.

The offset check exists because the assigned seed is NOT the only namespace the run consumes:
run_ccp_s2_production.py derives an environment range of (seed, seed+320). A seed that is
itself disjoint can still produce a run that overlaps a reserved block.

Run:  python experiments/ccp_s2_training_seed_preflight.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SEED_RECORD = SD / "CCP_S2_TRAINING_SEED_ASSIGNMENT.json"
SEED_BLOCKS = SD / "CCP_S2_SEED_ASSIGNMENT.json"
OUT = SD / "CCP_S2_TRAINING_SEED_PREFLIGHT.json"

ENV_RANGE_SPAN = 320          # run_ccp_s2_production.py: (seed, seed + 320)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_block(s: str) -> range:
    lo, hi = s.split("..")
    return range(int(lo), int(hi) + 1)


def main() -> int:
    import experiments.run_ccp_s2_production as P

    rec = json.loads(SEED_RECORD.read_text(encoding="utf-8"))
    if rec["status"] != "FROZEN_APPEND_ONLY_AMENDMENT":
        raise SystemExit(f"REFUSING: seed record not frozen: {rec['status']!r}")
    blocks = json.loads(SEED_BLOCKS.read_text(encoding="utf-8"))["SEED_BLOCKS"]

    seed = int(rec["training_seed"])
    env_range = range(seed, seed + ENV_RANGE_SPAN + 1)
    collection = _parse_block(blocks["collection_state_source"]["block"])
    gap = _parse_block(blocks["intentional_gap"]["block"])
    eval_blk = _parse_block(blocks["eval_block"]["block"])

    checks = []

    def check(name: str, passed: bool, detail: str):
        checks.append({"check": name, "PASS": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    print(f"CCP-S2 TRAINING SEED PREFLIGHT  {_now()}")
    print(f"  assigned seed {seed}, derived env range {env_range.start}..{env_range.stop - 1}\n")

    # 1-3 disjointness of the seed itself
    check("seed_disjoint_from_collection", seed not in collection,
          f"{seed} vs {collection.start}..{collection.stop - 1}")
    check("seed_disjoint_from_gap", seed not in gap,
          f"{seed} vs {gap.start}..{gap.stop - 1}")
    check("seed_disjoint_from_sealed_eval", seed not in eval_blk,
          f"{seed} vs {eval_blk.start}..{eval_blk.stop - 1}")

    # 4 the derived range -- the accidental-offset check
    overlaps = {name: sorted(set(env_range) & set(blk))
                for name, blk in (("collection", collection), ("gap", gap), ("eval", eval_blk))}
    bad = {k: v for k, v in overlaps.items() if v}
    check("derived_env_range_disjoint_from_all_reserved_blocks", not bad,
          f"{env_range.start}..{env_range.stop - 1} overlaps nothing" if not bad
          else f"OVERLAP {bad}")

    # 5 namespace consistency
    same_ns = all(str(x).startswith("117") for x in (seed, env_range.stop - 1,
                                                     collection.start, eval_blk.start))
    check("namespace_consistency_117xxxxx", same_ns,
          f"seed {seed}, range end {env_range.stop - 1} share the 117xxxxx program namespace")

    # 6 the record's own claimed range matches what the runner will actually derive
    claimed = _parse_block(rec["DERIVED_ENVIRONMENT_RANGE"]["range"])
    check("record_matches_runner_derived_range",
          (claimed.start, claimed.stop) == (env_range.start, env_range.stop),
          f"record says {claimed.start}..{claimed.stop - 1}, runner derives "
          f"{env_range.start}..{env_range.stop - 1}")

    # 7 the runner reads exactly this seed, through its own code path
    got = P.training_seed()
    check("runner_reads_assigned_seed", got == seed, f"training_seed() returned {got}")

    # 8-9 matched-arm usage + deterministic init, checked on the REAL config objects
    ckpt = P.incumbent_checkpoint()
    c_cfg = P.build_config("control", seed, ckpt)
    t_cfg = P.build_config("treatment", seed, ckpt)
    allowed_to_differ = {"run_tag", "checkpoint_dir", "metrics_csv_path", "episode_csv_path"}
    diffs = {}
    for k in sorted(set(vars(c_cfg)) | set(vars(t_cfg))):
        cv, tv = getattr(c_cfg, k, "<missing>"), getattr(t_cfg, k, "<missing>")
        if cv != tv:
            diffs[k] = (str(cv)[:60], str(tv)[:60])
    unexpected = {k: v for k, v in diffs.items() if k not in allowed_to_differ}
    check("arms_differ_only_in_run_tag_and_output_paths", not unexpected,
          f"differing fields: {sorted(diffs)}" if not unexpected
          else f"UNEXPECTED DIFFERENCES: {unexpected}")
    check("arms_share_training_seed", int(c_cfg.seed) == int(t_cfg.seed) == seed,
          f"control {c_cfg.seed}, treatment {t_cfg.seed}")
    check("arms_share_horizon",
          int(c_cfg.total_timesteps) == int(t_cfg.total_timesteps) == P.TOTAL_STEPS,
          f"{c_cfg.total_timesteps} == {t_cfg.total_timesteps} == {P.TOTAL_STEPS}")
    # the field that actually governs a WARM-START run: learn() stops when global_step
    # REACHES total_timesteps, and the incumbent restores global_step ~1.0M, so an absolute
    # 500k budget trains for zero steps and still exits COMPLETE
    check("warm_start_horizon_is_additional_not_absolute",
          int(getattr(c_cfg, "additional_timesteps", 0)) == P.TOTAL_STEPS
          and int(getattr(t_cfg, "additional_timesteps", 0)) == P.TOTAL_STEPS,
          f"additional_timesteps control={getattr(c_cfg, 'additional_timesteps', 0)}, "
          f"treatment={getattr(t_cfg, 'additional_timesteps', 0)} (both must be {P.TOTAL_STEPS})")
    check("deterministic_init_same_incumbent_weights",
          c_cfg.load_path == t_cfg.load_path == str(ckpt)
          and bool(c_cfg.load_weights_only) and bool(t_cfg.load_weights_only),
          f"both load {Path(str(ckpt)).name}, load_weights_only=True (fresh optimizer)")

    # 10 no arm writes into the other's outputs
    check("arm_outputs_disjoint",
          c_cfg.checkpoint_dir != t_cfg.checkpoint_dir
          and c_cfg.metrics_csv_path != t_cfg.metrics_csv_path,
          "control and treatment write to separate directories")

    all_pass = all(c["PASS"] for c in checks)
    verdict = "PASS" if all_pass else "FAIL"
    OUT.write_text(json.dumps({
        "record": "CCP-S2 training seed mechanical preflight", "status": "FROZEN_RESULT",
        "utc": _now(), "VERDICT": verdict,
        "training_seed": seed,
        "derived_env_range": f"{env_range.start}..{env_range.stop - 1}",
        "checks": checks,
        "n_checks": len(checks), "n_passed": sum(c["PASS"] for c in checks),
        "authorizes_if_pass": "experiments/run_ccp_s2_production.py --arm control "
                              "--verify-steps (wiring smoke), then the full 500k run",
    }, indent=2), encoding="utf-8")
    print(f"\n  {sum(c['PASS'] for c in checks)}/{len(checks)} checks passed")
    print(f"  VERDICT: {verdict}")
    print(f"  -> {OUT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
