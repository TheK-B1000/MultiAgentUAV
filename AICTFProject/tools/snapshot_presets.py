"""Regenerate ``tests/preset_snapshots.json`` from the current preset registry.

Use this **intentionally** when you have changed a training preset's resolved
config and want to update the golden file the snapshot test compares against.
Do not run this just to make a failing test pass — that defeats the regression
guarantee. If the test surprises you, audit the diff first.

Usage::

    python tools/snapshot_presets.py            # writes tests/preset_snapshots.json
    python tools/snapshot_presets.py --check    # exits non-zero if regen would change anything
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if regenerating would change the snapshot file (no write).",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Override snapshot path (default: tests/preset_snapshots.json next to the suite).",
    )
    args = parser.parse_args(argv)

    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    from tests.test_preset_resolution import SNAPSHOT_PATH, resolve_all_presets

    out_path = args.path or SNAPSHOT_PATH
    resolved = resolve_all_presets()
    new_text = json.dumps(resolved, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not os.path.isfile(out_path):
            print(f"[snapshot-presets] missing: {out_path}", file=sys.stderr)
            return 2
        with open(out_path, "r", encoding="utf-8") as f:
            old_text = f.read()
        if old_text != new_text:
            print(
                f"[snapshot-presets] {out_path} is OUT OF DATE. "
                "Re-run without --check to update.",
                file=sys.stderr,
            )
            return 1
        print(f"[snapshot-presets] {out_path} is up to date.")
        return 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"[snapshot-presets] wrote {len(resolved)} preset entries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
