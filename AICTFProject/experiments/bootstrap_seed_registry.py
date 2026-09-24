r"""One-time backfill of the Rule 9 seed registry from artifacts already on disk.

Rule 9 was practised long before it was codified, so the registry starts with
history rather than empty. This script reads the structured seed declarations out
of the sealed artifacts and records them.

It classifies ONLY from evidence found in the artifact itself (``arm``,
``confirmatory``, ``status``, the smoke family's reserved numbers). Anything it
cannot determine is written as ``UNCLASSIFIED`` and will fail
``seed_registry.py audit`` until a human resolves it. That is deliberate: a
guessed classification in a registry whose entire job is to be trusted is worse
than an admitted gap. (See the standing rule: absence is an error state.)

    python experiments/bootstrap_seed_registry.py --dry-run
    python experiments/bootstrap_seed_registry.py --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.seed_registry import REGISTRY, SMOKE_LO, _nested, _overlap, save

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"

SEED_KEYS = ("block", "seed_block", "range", "seed_range", "seeds")


def _walk(node, path, out, fname):
    if isinstance(node, dict):
        for k in SEED_KEYS:
            v = node.get(k)
            if (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v)
                    and 1000 < v[0] <= v[1] < 10**9 and v[1] - v[0] < 100_000):
                out.append({"lo": v[0], "hi": v[1], "file": fname,
                            "json_path": "/".join(path + [k])})
        for k, v in node.items():
            _walk(v, path + [k], out, fname)
    elif isinstance(node, list):
        for i, v in enumerate(node[:100]):
            _walk(v, path + [str(i)], out, fname)


#: Records that cite a seed block downstream of whoever allocated it.
_DERIVATIVE = ("INTEGRITY", "AUDIT", "TIE_REVERSAL", "RETIREMENT", "READING",
               "POSTMORTEM", "ANALYSIS", "RECONCILIATION")


def _derivative(name: str) -> bool:
    return any(w in name.upper() for w in _DERIVATIVE)


def _classify(doc: dict, lo: int) -> tuple[str, str]:
    """(seed_class, evidence). Never guesses.

    Evidence is ranked by how directly it states the contract. ``one_shot: true``
    and a ``PRIMARY_GATE`` block are this project's literal confirmatory contract
    -- spent once, against a pre-registered gate -- so they outrank record names.
    """
    if lo >= SMOKE_LO:
        return "smoke", "lies in the reserved 999xxxxx smoke family"
    arm = str(doc.get("arm", "")).upper()
    conf = doc.get("confirmatory")
    if conf is True or arm == "CONFIRMATORY":
        return "sealed_confirmatory", f"artifact declares arm={arm!r} confirmatory={conf!r}"
    if conf is False or arm in ("DIAGNOSTIC", "EXPLORATORY", "PILOT"):
        return "exploratory", f"artifact declares arm={arm!r} confirmatory={conf!r}"
    name = (str(doc.get("record_id", "")) + " " + str(doc.get("record", ""))).upper()
    # an EXPLORATORY/DIAGNOSTIC name beats one_shot: a diagnostic can also be
    # one-shot without being a confirmatory seed spend
    if any(w in name for w in ("EXPLORATORY", "DIAGNOSTIC", "SMOKE", "PILOT")):
        return "exploratory", "record name declares EXPLORATORY/DIAGNOSTIC/PILOT"
    if "CONFIRMATORY" in name:
        return "sealed_confirmatory", "record name declares CONFIRMATORY"
    if doc.get("one_shot") is True:
        return "sealed_confirmatory", "artifact declares one_shot=true (spent-once contract)"
    if "PRIMARY_GATE" in doc:
        return "sealed_confirmatory", "artifact carries a pre-registered PRIMARY_GATE"
    return "UNCLASSIFIED", "no arm/confirmatory/one_shot/PRIMARY_GATE field and no decisive name"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.write or a.dry_run):
        raise SystemExit("pass --dry-run or --write")

    found: dict[tuple[int, int], dict] = {}
    for p in sorted(SD.glob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            continue
        if not isinstance(doc, dict):
            continue
        hits: list[dict] = []
        _walk(doc, [], hits, p.name)
        for h in hits:
            key = (h["lo"], h["hi"])
            cls, why = _classify(doc, h["lo"])
            prev = found.get(key)
            retired = "RETIRED" in doc or "RETIREMENT" in p.name.upper()
            # Prefer the record that ALLOCATED the block over one that merely cites
            # it. An integrity audit or tie-reversal record names the same seeds but
            # is downstream of the allocation, so it is the wrong provenance to
            # record and often the wrong thing to classify from.
            better = prev is None \
                or (prev["seed_class"] == "UNCLASSIFIED" and cls != "UNCLASSIFIED") \
                or (_derivative(prev["spec"]) and not _derivative(p.name)
                    and cls != "UNCLASSIFIED")
            if better:
                cited = (prev or {}).get("cited_by", [])
                found[key] = {"lo": h["lo"], "hi": h["hi"], "seed_class": cls,
                              "evidence": why, "spec": p.name,
                              "purpose": str(doc.get("record") or doc.get("record_id")
                                             or p.stem)[:160],
                              "retired": retired or (prev or {}).get("retired", False),
                              "cited_by": sorted(set(cited + [p.name]))}
            else:
                found[key]["cited_by"] = sorted(set(found[key]["cited_by"] + [p.name]))
                found[key]["retired"] = found[key].get("retired", False) or retired

    blocks = sorted(found.values(), key=lambda b: (b["lo"], -(b["hi"] - b["lo"])))

    # detect legitimate nesting (a collection bank subdivided into shards)
    for i, b in enumerate(blocks):
        b["experiment_id"] = f"HIST_{b['lo']}_{b['hi']}"
        b["n"] = b["hi"] - b["lo"] + 1
        b["status"] = "RETIRED" if b.pop("retired", False) else "SPENT"
        b["subdivides"] = None
        b["backfilled"] = True
    for b in blocks:
        for parent in blocks:
            if parent is b or not _nested(b, parent):
                continue
            if parent["n"] > b["n"]:
                b["subdivides"] = parent["experiment_id"]
                break

    undeclared = []
    for i, x in enumerate(blocks):
        for y in blocks[i + 1:]:
            if not _overlap(x, y):
                continue
            if y.get("subdivides") == x["experiment_id"] or \
               x.get("subdivides") == y["experiment_id"]:
                continue
            if x.get("subdivides") and x["subdivides"] == y.get("subdivides"):
                continue
            undeclared.append((x, y))

    n_unc = sum(1 for b in blocks if b["seed_class"] == "UNCLASSIFIED")
    by_cls: dict[str, int] = {}
    for b in blocks:
        by_cls[b["seed_class"]] = by_cls.get(b["seed_class"], 0) + 1

    print(f"harvested {len(blocks)} distinct seed blocks from {SD.relative_to(ROOT)}")
    for c, n in sorted(by_cls.items()):
        print(f"  {c:<22}{n}")
    print(f"  nested (declared)     {sum(1 for b in blocks if b['subdivides'])}")
    print(f"  UNDECLARED OVERLAPS   {len(undeclared)}")
    for x, y in undeclared[:10]:
        print(f"    {x['lo']}..{x['hi']} ({x['spec']})  vs  {y['lo']}..{y['hi']} ({y['spec']})")
    if n_unc:
        print(f"\n{n_unc} block(s) could not be classified from artifact evidence and are "
              f"recorded as UNCLASSIFIED.\n`seed_registry.py audit` will fail until each "
              f"is resolved by hand. That is the intended behaviour -- a guessed class "
              f"is worse than an admitted gap.")

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    save({"record": "seed allocation registry (Rule 9)",
          "rule": "three seed classes, never overlapping; confirmatory seeds are spent "
                  "exactly once and never on debugging",
          "backfilled_from": str(SD.relative_to(ROOT)),
          "backfill_note": "HIST_* ids are historical blocks recovered from sealed "
                           "artifacts after Rule 9 was codified. Their classification "
                           "comes from each artifact's own arm/confirmatory fields.",
          "blocks": blocks})
    print(f"\nwrote {REGISTRY.relative_to(ROOT)} ({len(blocks)} blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
