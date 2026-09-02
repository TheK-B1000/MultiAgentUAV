"""CCP-S2 compute-budget amendment, stage 3: reconcile already-collected rows.

Implements CCP_S2_COMPUTE_BUDGET_AMENDMENT.json#ALREADY_COMPLETED_ROWS_RULE. Classifies every
row already written under the original 10176-job plan (ccp_s2_rows/worker_*.jsonl) as RETAINED
(its job_id is a member of the amended 2496-job set) or ARCHIVED_UNUSED (it is not). This
classification uses job_id membership ONLY -- state_id/estimand/arm/j identifiers already fixed
at collection time -- and never reads a row's outcome (blue/red/win/margin/continuation_steps)
to decide its status. Nothing is deleted or moved; this is an accounting record, frozen once,
required before the amended collector's worker mode will run.

Run:  python experiments/ccp_s2_amend_reconcile.py
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
AMENDMENT = SD / "CCP_S2_COMPUTE_BUDGET_AMENDMENT.json"
MANIFEST = SD / "CCP_S2_STATE_MANIFEST_AMENDMENT.json"
ROWS_DIR = SD / "ccp_s2_rows"
OUT = SD / "CCP_S2_AMENDMENT_RECONCILIATION.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if amendment["status"] != "FROZEN_APPEND_ONLY_AMENDMENT":
        raise SystemExit(f"REFUSING: compute-budget amendment not frozen: {amendment['status']!r}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_SELECTION":
        raise SystemExit(f"REFUSING: amended manifest not frozen: {manifest['status']!r}")

    import experiments.ccp_s2_collect_amended as A
    jobs, counts = A.build_jobs(manifest)
    A.validate(jobs, counts)
    amended_ids = {j["job_id"] for j in jobs}
    print(f"CCP-S2 AMENDMENT RECONCILIATION  {_now()}")
    print(f"  amended job set: {len(amended_ids)} ids\n", flush=True)

    retained_by_worker, archived_by_worker = {}, {}
    retained_ids, archived_ids = [], []
    for p in sorted(ROWS_DIR.glob("worker_*.jsonl")):
        n_ret = n_arc = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)                    # parsed only for job_id, never for outcome
            jid = row["job_id"]
            if jid in amended_ids:
                retained_ids.append(jid); n_ret += 1
            else:
                archived_ids.append(jid); n_arc += 1
        retained_by_worker[p.name] = n_ret
        archived_by_worker[p.name] = n_arc
        print(f"  {p.name}: {n_ret} retained, {n_arc} archived_unused", flush=True)

    if len(set(retained_ids)) != len(retained_ids):
        raise SystemExit("REFUSING: a retained job_id appears more than once across worker files "
                         "-- two workers wrote the same job, which should be structurally "
                         "impossible under the fixed i%workers sharding")

    n_retained, n_archived = len(retained_ids), len(archived_ids)
    n_amended_remaining = len(amended_ids) - n_retained
    print(f"\n  retained {n_retained} / {len(amended_ids)} amended jobs already satisfied")
    print(f"  archived_unused {n_archived} rows (valid original-plan measurements, out of "
          f"amended scope, preserved on disk, never used)")
    print(f"  remaining to collect under the amendment: {n_amended_remaining}")

    OUT.write_text(json.dumps({
        "record": "CCP-S2 amendment reconciliation", "status": "FROZEN_RESULT", "one_shot": True,
        "utc": _now(),
        "implements": "CCP_S2_COMPUTE_BUDGET_AMENDMENT.json#ALREADY_COMPLETED_ROWS_RULE",
        "classified_by": "job_id membership in the amended 2496-job set only -- no row's "
                         "outcome field (blue/red/win/margin/continuation_steps) was read to "
                         "decide RETAINED vs ARCHIVED_UNUSED",
        "amended_job_set_size": len(amended_ids),
        "n_retained": n_retained, "n_archived_unused": n_archived,
        "n_amended_remaining_to_collect": n_amended_remaining,
        "retained_by_worker_file": retained_by_worker,
        "archived_by_worker_file": archived_by_worker,
        "retained_job_ids": sorted(retained_ids),
        "archived_unused_job_ids": sorted(archived_ids),
        "disposition": "no file moved, edited, or deleted. worker_*.jsonl retain every row as "
                       "written; the amended collector's own resume logic (done_ids) already "
                       "skips retained job_ids automatically. archived_unused rows are inert -- "
                       "present, valid original-plan measurements, permanently excluded from "
                       "bank construction / the advantage estimator / routing / any statistic "
                       "of this program.",
        "authorizes_if_frozen": "experiments/ccp_s2_collect_amended.py worker mode",
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
