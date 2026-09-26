r"""Rule 9 -- seed classes, made machine-checkable.

Three classes, never overlapping:

    smoke                debug/plumbing. Reserved family 999xxxxx. Cheap, disposable.
    exploratory          diagnostics, pilots, power estimates, anything you may look
                         at and then change your mind about.
    sealed_confirmatory  spent once, against a frozen spec, to answer a
                         pre-registered question. Never reused. Never spent on
                         debugging. If the code changes materially afterwards, the
                         result stays frozen rather than being quietly replaced.

The rule was already practised; what was missing was anything that could *refuse*.
A launcher could silently re-spend a confirmatory block and nothing would notice
until someone compared two specs by eye. This module is the thing that notices.

    python experiments/seed_registry.py list
    python experiments/seed_registry.py check 17500001 17500032 --class exploratory
    python experiments/seed_registry.py allocate MY_EXP 32 --class exploratory \
        --purpose "..." --spec MY_SPEC.json
    python experiments/seed_registry.py audit

NESTING IS REAL AND LEGITIMATE. A 320-seed collection bank subdivided into four
shards is not an overlap violation, so an entry may declare ``subdivides`` naming
its parent. Anything else that overlaps is an error.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "artifacts" / "SEED_REGISTRY.json"

CLASSES = ("smoke", "exploratory", "sealed_confirmatory")
STATUSES = ("RESERVED", "SPENT", "RETIRED")

#: The smoke family is reserved by number so a debug run can never *accidentally*
#: land on real seeds -- the failure that motivated the rule.
SMOKE_LO, SMOKE_HI = 99_900_000, 99_999_999


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load() -> dict:
    if not REGISTRY.is_file():
        return {"record": "seed allocation registry (Rule 9)", "blocks": []}
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def save(doc: dict) -> None:
    doc["updated_utc"] = _now()
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _overlap(a: dict, b: dict) -> bool:
    return a["lo"] <= b["hi"] and b["lo"] <= a["hi"]


def _nested(child: dict, parent: dict) -> bool:
    return parent["lo"] <= child["lo"] and child["hi"] <= parent["hi"]


def check_block(lo: int, hi: int, seed_class: str,
                experiment_id: str | None = None,
                subdivides: str | None = None) -> tuple[bool, str]:
    """Is this block safe to spend? Returns ``(ok, human-readable reason)``.

    Called by ``run_state.run_audit`` as a gating check, and by launchers before
    a single seed is spent.
    """
    if seed_class not in CLASSES:
        return False, f"unknown seed class {seed_class!r}; expected one of {CLASSES}"
    if lo > hi:
        return False, f"malformed block {lo}..{hi}"

    in_smoke = SMOKE_LO <= lo and hi <= SMOKE_HI
    if seed_class == "smoke" and not in_smoke:
        return False, (f"smoke block {lo}..{hi} is outside the reserved smoke family "
                       f"{SMOKE_LO}..{SMOKE_HI}")
    if seed_class != "smoke" and in_smoke:
        return False, (f"{seed_class} block {lo}..{hi} lies inside the reserved smoke "
                       f"family -- those seeds are disposable and may already be dirty")

    doc = load()
    me = {"lo": lo, "hi": hi}
    parent = None
    if subdivides:
        parent = next((b for b in doc["blocks"] if b["experiment_id"] == subdivides), None)
        if parent is None:
            return False, f"declared subdivides={subdivides!r} but no such registered block"
        if not _nested(me, parent):
            return False, (f"{lo}..{hi} is not contained in parent {subdivides} "
                           f"({parent['lo']}..{parent['hi']})")

    clashes = []
    for b in doc["blocks"]:
        # Exact match to your OWN prior reservation is the only auto-allow. A
        # request under the same experiment_id but a DIFFERENT range (bigger,
        # shifted, partially overlapping) is NOT waved through -- it falls into
        # the normal overlap check below, same as anyone else's block, so a
        # launcher cannot silently expand its own allocation by re-checking with
        # a wider range instead of going through allocate(). This was a real bug:
        # the previous version skipped this entry entirely on a name match,
        # which meant a mismatched self-request with no OTHER overlapping block
        # would be reported "free" even though it clashed with its own record.
        if experiment_id and b["experiment_id"] == experiment_id \
                and b["lo"] == lo and b["hi"] == hi:
            return True, (f"already registered to {experiment_id} "
                          f"(status {b['status']}) -- same block, no conflict")
        if not _overlap(b, me):
            continue
        # RETIRED is still a clash: retiring a block records that it was spent
        # and superseded, not that its seeds became free again. Seeds are never
        # reused, retired or not.
        if parent is not None and b["experiment_id"] == subdivides:
            continue                                    # declared nesting
        if b.get("subdivides") and parent is None and _nested(b, me):
            continue                                    # we are the parent
        clashes.append(b)

    if clashes:
        lines = "; ".join(f"{c['experiment_id']} {c['lo']}..{c['hi']} "
                          f"[{c['seed_class']}/{c['status']}]" for c in clashes[:4])
        return False, (f"block {lo}..{hi} OVERLAPS {len(clashes)} registered block(s): "
                       f"{lines}. Seeds are not reusable across experiments; pick a "
                       f"fresh block (see `seed_registry.py next {hi - lo + 1}`).")
    return True, f"block {lo}..{hi} ({seed_class}, n={hi - lo + 1}) is free"


def next_free(n: int, seed_class: str = "exploratory", stride: int = 100_000) -> int:
    """Lowest unused block start on the project's 100k-stride convention."""
    doc = load()
    used = [b for b in doc["blocks"] if b["status"] != "RETIRED"]
    top = max((b["hi"] for b in used if b["hi"] < SMOKE_LO), default=10_000_000)
    start = ((top // stride) + 1) * stride + 1
    while not check_block(start, start + n - 1, seed_class)[0]:
        start += stride
    return start


def allocate(experiment_id: str, lo: int, hi: int, seed_class: str, purpose: str,
             spec: str | None = None, subdivides: str | None = None,
             status: str = "RESERVED") -> dict:
    ok, msg = check_block(lo, hi, seed_class, experiment_id, subdivides)
    if not ok:
        raise SystemExit(f"REFUSING to allocate: {msg}")
    doc = load()
    if any(b["experiment_id"] == experiment_id for b in doc["blocks"]):
        raise SystemExit(f"REFUSING: experiment_id {experiment_id!r} already registered. "
                         f"Seed blocks are append-only; use a distinct id.")
    entry = {"experiment_id": experiment_id, "purpose": purpose,
             "seed_class": seed_class, "lo": lo, "hi": hi, "n": hi - lo + 1,
             "status": status, "spec": spec, "subdivides": subdivides,
             "allocated_utc": _now()}
    doc["blocks"].append(entry)
    doc["blocks"].sort(key=lambda b: (b["lo"], b["hi"]))
    save(doc)
    return entry


def reconcile_spent(
    experiment_id: str,
    lo: int,
    hi: int,
    seed_class: str,
    purpose: str,
    *,
    spec: str | None = None,
    cited_by: list[str] | None = None,
    note: str | None = None,
    arm: str | None = None,
    registration_origin: str = "RETROACTIVE_RECONCILIATION",
    historically_pre_registered: bool = False,
) -> dict:
    """Record a block already spent without prior reservation.

    Collision protection going forward only. Does **not** pretend Rule 9 was
    followed originally (``historically_pre_registered=false`` by default).
    """
    if seed_class not in CLASSES:
        raise SystemExit(f"unknown seed class {seed_class!r}; expected one of {CLASSES}")
    if lo > hi:
        raise SystemExit(f"malformed block {lo}..{hi}")
    doc = load()
    if any(b["experiment_id"] == experiment_id for b in doc["blocks"]):
        raise SystemExit(f"REFUSING: experiment_id {experiment_id!r} already registered")
    ok, msg = check_block(lo, hi, seed_class, experiment_id=experiment_id)
    if not ok:
        raise SystemExit(f"REFUSING to reconcile: {msg}")
    entry: dict[str, Any] = {
        "experiment_id": experiment_id,
        "purpose": purpose,
        "seed_class": seed_class,
        "lo": lo,
        "hi": hi,
        "n": hi - lo + 1,
        "status": "SPENT",
        "spec": spec,
        "subdivides": None,
        "allocated_utc": _now(),
        "spent_utc": _now(),
        "registration_origin": registration_origin,
        "historically_pre_registered": historically_pre_registered,
        "cited_by": list(cited_by or []),
    }
    if arm is not None:
        entry["arm"] = arm
    if note:
        entry["note"] = note
    doc["blocks"].append(entry)
    doc["blocks"].sort(key=lambda b: (b["lo"], b["hi"]))
    save(doc)
    return entry


def set_status(experiment_id: str, status: str, note: str | None = None) -> dict:
    if status not in STATUSES:
        raise SystemExit(f"unknown status {status!r}; expected {STATUSES}")
    doc = load()
    b = next((x for x in doc["blocks"] if x["experiment_id"] == experiment_id), None)
    if b is None:
        raise SystemExit(f"no such experiment_id: {experiment_id}")
    if b["seed_class"] == "sealed_confirmatory" and b["status"] == "SPENT" \
            and status != "RETIRED":
        raise SystemExit(
            f"REFUSING: {experiment_id} is a SPENT confirmatory block. Confirmatory "
            f"seeds are spent exactly once. RETIRED is the only legal move from here.")
    b["status"] = status
    b[f"{status.lower()}_utc"] = _now()
    if note:
        b.setdefault("notes", []).append({"utc": _now(), "note": note})
    save(doc)
    return b


def audit_registry() -> tuple[bool, list[str]]:
    """Self-consistency of the registry itself: undeclared overlaps, bad classes,
    unclassified entries."""
    doc = load()
    problems = []
    blocks = doc["blocks"]
    for i, a in enumerate(blocks):
        if a["seed_class"] not in CLASSES:
            problems.append(f"{a['experiment_id']}: unknown class {a['seed_class']!r} "
                            f"-- classify it from the spec, do not guess")
        if a["status"] not in STATUSES:
            problems.append(f"{a['experiment_id']}: unknown status {a['status']!r}")
        for b in blocks[i + 1:]:
            if not _overlap(a, b):
                continue
            if b.get("subdivides") == a["experiment_id"] and _nested(b, a):
                continue
            if a.get("subdivides") == b["experiment_id"] and _nested(a, b):
                continue
            if a.get("subdivides") and a["subdivides"] == b.get("subdivides"):
                if not _overlap(a, b):
                    continue
            problems.append(
                f"UNDECLARED OVERLAP: {a['experiment_id']} {a['lo']}..{a['hi']} "
                f"vs {b['experiment_id']} {b['lo']}..{b['hi']}")
    return not problems, problems


# --------------------------------------------------------------------- CLI ---
def main() -> int:
    ap = argparse.ArgumentParser(description="Rule 9 seed registry")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="show every registered block")
    p.add_argument("--class", dest="cls", choices=CLASSES)

    p = sub.add_parser("check", help="is a block free to spend?")
    p.add_argument("lo", type=int); p.add_argument("hi", type=int)
    p.add_argument("--class", dest="cls", required=True, choices=CLASSES)
    p.add_argument("--subdivides")
    # Without this, a launcher verifying the block it already reserved is told it
    # collides with itself -- which would train people to ignore the refusal.
    p.add_argument("--experiment-id", dest="experiment_id",
                   help="the block's own experiment id, so it does not self-collide")

    p = sub.add_parser("next", help="suggest the next free block start")
    p.add_argument("n", type=int)
    p.add_argument("--class", dest="cls", default="exploratory", choices=CLASSES)

    p = sub.add_parser("allocate", help="reserve a block (refuses on overlap)")
    p.add_argument("experiment_id"); p.add_argument("n", type=int)
    p.add_argument("--class", dest="cls", required=True, choices=CLASSES)
    p.add_argument("--purpose", required=True)
    p.add_argument("--lo", type=int, help="explicit start; default = next free")
    p.add_argument("--spec"); p.add_argument("--subdivides")

    p = sub.add_parser("status", help="mark a block SPENT or RETIRED")
    p.add_argument("experiment_id"); p.add_argument("status", choices=STATUSES)
    p.add_argument("--note")

    sub.add_parser("audit", help="check the registry against itself")
    a = ap.parse_args()

    if a.cmd == "list":
        doc = load()
        bl = [b for b in doc["blocks"] if not a.cls or b["seed_class"] == a.cls]
        print(f"{len(bl)} block(s) in {REGISTRY.relative_to(ROOT)}\n")
        print(f"{'range':<24}{'n':>6}  {'class':<20}{'status':<10}experiment")
        for b in bl:
            print(f"{b['lo']}..{b['hi']:<12}{b['n']:>6}  {b['seed_class']:<20}"
                  f"{b['status']:<10}{b['experiment_id']}"
                  f"{'  (subdivides ' + b['subdivides'] + ')' if b.get('subdivides') else ''}")
        return 0

    if a.cmd == "check":
        ok, msg = check_block(a.lo, a.hi, a.cls, experiment_id=a.experiment_id,
                              subdivides=a.subdivides)
        print(("OK   " if ok else "REFUSE ") + msg)
        return 0 if ok else 1

    if a.cmd == "next":
        print(next_free(a.n, a.cls))
        return 0

    if a.cmd == "allocate":
        lo = a.lo if a.lo is not None else next_free(a.n, a.cls)
        e = allocate(a.experiment_id, lo, lo + a.n - 1, a.cls, a.purpose,
                     spec=a.spec, subdivides=a.subdivides)
        print(f"allocated {e['lo']}..{e['hi']} (n={e['n']}) to {e['experiment_id']} "
              f"[{e['seed_class']}/{e['status']}]")
        return 0

    if a.cmd == "status":
        b = set_status(a.experiment_id, a.status, a.note)
        print(f"{b['experiment_id']} -> {b['status']}")
        return 0

    if a.cmd == "audit":
        ok, problems = audit_registry()
        n = len(load()["blocks"])
        if ok:
            print(f"registry CLEAN: {n} blocks, no undeclared overlaps.")
            return 0
        print(f"registry has {len(problems)} problem(s) across {n} blocks:")
        for pr in problems:
            print(f"  {pr}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
