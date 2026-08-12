"""Report scan state from the artifacts themselves, not from filename patterns.

Written because filename-pattern bookkeeping produced a wrong status report:
`^states_.*\\.json$` also matches `states_X.json.manifest.json`, so a shard
directory with 5 shards and 5 manifests was reported as 10 shards.

Every classification here loads the file and inspects its SCHEMA. A shard is a
shard because it maps a policy seed to a list of state records, not because of
what it is called. Cells are enumerated from the frozen cell list, so a cell that
was never started is distinguishable from one that failed.

Run:  python experiments/scan_status.py --shard-dir artifacts/c7_stage2/shards_4v4 --arm 4v4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARM_SEEDS = {"2v2": [3200001, 3200002, 3200003], "4v4": [3300001, 3300002, 3300003]}


from srctf.artifacts import COMPLETE, NOT_PRESENT, SHARD_ONLY, cell_state, classify  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--arm", default="4v4", choices=tuple(ARM_SEEDS))
    ap.add_argument("--opponent-set", default="historical")
    args = ap.parse_args()

    from srctf.opponent_sets import get as opponent_set

    d = Path(args.shard_dir)
    if not d.exists():
        print(f"shard dir absent: {d}")
        return 1

    kinds: dict[str, list[str]] = {}
    for p in sorted(d.glob("*.json")):
        kinds.setdefault(classify(p), []).append(p.name)

    print(f"shard dir: {d}")
    for k in sorted(kinds):
        print(f"  {k:12s} {len(kinds[k])}")

    cells = [(p, o) for p in ARM_SEEDS[args.arm] for o in opponent_set(args.opponent_set)]
    complete, states_only, absent = [], [], []
    for pseed, opp in cells:
        st = cell_state(d, pseed, opp)
        {COMPLETE: complete, SHARD_ONLY: states_only, NOT_PRESENT: absent}[st].append(
            f"{pseed}/{opp}")

    print(f"\ncells ({args.arm}, {len(cells)} total)")
    print(f"  complete (shard+manifest) : {len(complete)}")
    print(f"  shard only, no manifest   : {len(states_only)}  {states_only if states_only else ''}")
    print(f"  not present               : {len(absent)}")

    if complete:
        n = []
        for c in complete:
            pseed, opp = c.split("/")
            shard = json.loads((d / f"states_{pseed}_{opp}.json").read_text(encoding="utf-8"))
            n.append(len(next(iter(shard.values()))))
        print(f"  states per complete cell  : min {min(n)} max {max(n)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
