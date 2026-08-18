"""Audit a candidate evaluation seed block against every seed the campaign has
ever used, in the working tree AND in git history.

Motivating incident (2026-08-18): the reserved strategic-demand confirmation
block 2500001 was found, before any episode was spent, to collide with the
training run g0_v2_seed2500001. The replacement candidate 2600001 collided with
the g0_v2 penalty-ablation training seeds. Neither collision was visible
without an explicit audit.

The campaign's hygiene rule is: evaluation seeds disjoint from ALL training
seeds. This tool enforces it over a FULL intended range, not just the base.

Four independent checks, because each has a blind spot the others cover:

  1. DECLARED   integers in a genuine seed context (JSON key containing
                "seed", CSV column containing "seed", filename tagged
                seed<N>, python assignment to a NAME containing SEED).
                This is the set the hygiene rule is really about.

  2. BROAD      any 6-12 digit integer in any tracked text file. Deliberately
                over-inclusive (catches global_step, timesteps, CSV payload)
                and therefore conservative: a BROAD hit is a warning, not
                necessarily a real seed.

  3. SPANS      any declared (base, n) pair whose span overlaps the window.
                Catches a block that reaches INTO the range from a base
                sitting outside it.

  4. HISTORY    git log -G over all branches. A working-tree scan cannot see
                seeds used by deleted artifacts -- block 11000000 (VGC-4)
                appears in 5 historical commits and nowhere on disk. This
                check is load-bearing, not ceremony. It runs with a positive
                control so a broken query cannot masquerade as a clean result.

Margin: rl/population/population_trainer.py assigns per-member seed_offset up
to 300, so a declared training seed T can occupy T..T+300. The default margin
is far larger than that.

Usage:
    python scripts/audit_seed_block.py --base 5000001 --n 32
    python scripts/audit_seed_block.py --base 5000001 --n 32 --json out.json
    python scripts/audit_seed_block.py --suggest

Exit code 0 = CLEAN, 1 = COLLISION, 2 = audit could not be trusted.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", ".git", "__pycache__", "node_modules", ".pytest_cache",
             "wandb", ".mypy_cache"}
TEXT_EXT = {".py", ".json", ".md", ".txt", ".csv", ".yaml", ".yml", ".cfg", ".ini"}
INT_RE = re.compile(r"\b(\d{6,12})\b")
PY_SEED_ASSIGN = re.compile(r"\b([A-Za-z_]*SEED[A-Za-z_]*)\s*=\s*(\d[\d_]{5,})", re.I)
MAX_FILE_BYTES = 60_000_000
POPULATION_SEED_OFFSET_MAX = 300     # rl/population/population_trainer.py

BASE_KEYS = {"seed_base", "base", "seed", "eval_seed_base", "seed_block",
             "episode_seed_base", "start_seed", "seed_start"}
N_KEYS = {"n", "n_seeds", "seeds", "n_paired", "n_episodes", "episodes",
          "n_eval_episodes", "count"}

# A block known to be dirty, used to prove the history query actually works.
POSITIVE_CONTROL = (2600001, 32)


# Files that RESERVE a block rather than USE it. Without excluding these, an
# amendment record naming its own block makes that block look permanently
# dirty, and the tool can never certify the very block it was written to
# certify. The distinction (reserved vs used) is semantic and cannot be
# inferred, so it is declared explicitly here and echoed in the audit output
# -- an exclusion that is invisible is an exclusion that hides collisions.
SELF_REFERENCE_DEFAULTS = (
    "scripts/audit_seed_block.py",
    "artifacts/strategic_demand/CONFIRMATION_SEED_BLOCK_AMENDMENT.json",
    "artifacts/strategic_demand/CONFIRMATION_BLOCK_AUDIT_",
)


def _excluded(rel: str, extra: tuple[str, ...]) -> bool:
    norm = rel.replace("\\", "/")
    return any(norm.startswith(pat) or norm == pat
               for pat in SELF_REFERENCE_DEFAULTS + extra)


def _files():
    for p in ROOT.rglob("*"):
        if p.is_file() and not any(x in SKIP_DIRS for x in p.parts):
            yield p


def build_inventories(exclude: tuple[str, ...] = ()) -> tuple[dict, dict, list]:
    declared: dict[int, set[str]] = {}
    broad: dict[int, set[str]] = {}
    blocks: list[tuple[str, int, int, str]] = []

    def add(d, v, src):
        if 1_000_000 <= v <= 99_999_999:
            d.setdefault(v, set()).add(src)

    def walk_json(obj, src, key=""):
        if isinstance(obj, dict):
            base = n = None
            for k, v in obj.items():
                kl = k.lower()
                if kl in BASE_KEYS and isinstance(v, int) and v >= 1_000_000:
                    base = v
                if kl in N_KEYS and isinstance(v, int) and 0 < v <= 100_000:
                    n = v
            if base is not None:
                blocks.append((key or "<root>", base, n or 1, src))
            for k, v in obj.items():
                walk_json(v, src, k)
        elif isinstance(obj, list):
            for v in obj:
                walk_json(v, src, key)
        elif isinstance(obj, int):
            if "seed" in key.lower():
                add(declared, obj, f"{src}::{key}")
        elif isinstance(obj, str) and "seed" in key.lower():
            for m in INT_RE.finditer(obj):
                add(declared, int(m.group(1)), f"{src}::{key}")

    for p in _files():
        ext = p.suffix.lower()
        rel = str(p.relative_to(ROOT))
        if _excluded(rel, exclude):
            continue
        m = re.search(r"seed(\d{6,12})", p.name)
        if m:
            add(declared, int(m.group(1)), rel)
        for part in p.parts:
            m2 = re.search(r"seed(\d{6,12})", part)
            if m2:
                add(declared, int(m2.group(1)), rel)
        if ext not in TEXT_EXT:
            continue
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for mm in INT_RE.finditer(txt):
            add(broad, int(mm.group(1)), rel)
        if ext == ".json":
            try:
                walk_json(json.loads(txt), rel)
            except Exception:
                pass
        elif ext == ".csv":
            try:
                rows = list(csv.DictReader(txt.splitlines()))
                if rows:
                    cols = [c for c in rows[0] if c and "seed" in c.lower()]
                    for r in rows:
                        for c in cols:
                            try:
                                add(declared, int(str(r[c]).strip()), f"{rel}::{c}")
                            except (ValueError, TypeError):
                                pass
            except Exception:
                pass
        elif ext == ".py":
            for mm in PY_SEED_ASSIGN.finditer(txt):
                add(declared, int(mm.group(2).replace("_", "")), f"{rel}::{mm.group(1)}")

    return declared, broad, blocks


def window_regex(lo: int, n: int) -> str:
    """Word-anchored alternation matching exactly lo..lo+n-1."""
    return r"\b(" + "|".join(str(v) for v in range(lo, lo + n)) + r")\b"


def history_hits(lo: int, n: int, timeout: int = 540,
                 exclude: tuple[str, ...] = ()):
    """Return (commits, ok). ok=False means the query could not be trusted.

    Reservation files are excluded via pathspec for the same reason they are
    excluded from the working-tree scan: once this amendment is committed, the
    commit that RESERVES a block would otherwise make that block look used.
    """
    pathspec = ["--", "."]
    for pat in SELF_REFERENCE_DEFAULTS + exclude:
        pathspec.append(f":(exclude,glob){pat}*" if pat.endswith("_")
                        else f":(exclude,glob){pat}")
    try:
        out = subprocess.run(
            ["git", "log", "--all", "--oneline", "--perl-regexp",
             "-G", window_regex(lo, n)] + pathspec,
            cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [], False
    if out.returncode != 0:
        return [], False
    return [ln for ln in out.stdout.splitlines() if ln.strip()], True


def audit(base: int, n: int, margin: int, *, skip_history: bool = False,
          exclude: tuple[str, ...] = ()) -> dict:
    lo, hi = base, base + n - 1
    declared, broad, blocks = build_inventories(exclude)

    d_in = sorted(v for v in declared if lo <= v <= hi)
    b_in = sorted(v for v in broad if lo <= v <= hi)
    spans = [{"label": lbl, "base": bb, "n": nn, "source": s}
             for lbl, bb, nn, s in blocks if not (bb + nn - 1 < lo or bb > hi)]

    below = [v for v in declared if v < lo]
    above = [v for v in declared if v > hi]
    gap_below = lo - max(below) if below else None
    gap_above = min(above) - hi if above else None

    res = {
        "block": {"base": base, "n": n, "range": f"{lo}..{hi}"},
        "margin_required": margin,
        "population_seed_offset_max": POPULATION_SEED_OFFSET_MAX,
        "declared_inventory_size": len(declared),
        "broad_inventory_size": len(broad),
        "excluded_as_reservation_records": list(SELF_REFERENCE_DEFAULTS + exclude),
        "exclusion_caveat": ("these paths RESERVE a block rather than USE it and are "
                            "skipped in every check; any other file naming a seed in "
                            "the window counts as a collision"),
        "check_1_declared_hits": [{"seed": v, "sources": sorted(declared[v])[:4]}
                                  for v in d_in],
        "check_2_broad_hits": [{"int": v, "sources": sorted(broad[v])[:4]}
                               for v in b_in],
        "check_3_overlapping_blocks": spans,
        "gap_below": gap_below,
        "gap_above": gap_above,
    }

    if skip_history:
        res["check_4_history"] = {"skipped": True}
        hist_ok, hist_hits = True, []
    else:
        ctrl, ctrl_ok = history_hits(*POSITIVE_CONTROL, exclude=exclude)
        hist_hits, hist_ok = history_hits(lo, n, exclude=exclude)
        control_valid = ctrl_ok and len(ctrl) > 0
        res["check_4_history"] = {
            "commits": hist_hits,
            "query_ok": hist_ok,
            "positive_control_block":
                f"{POSITIVE_CONTROL[0]}..{POSITIVE_CONTROL[0] + POSITIVE_CONTROL[1] - 1}",
            "positive_control_commits": len(ctrl),
            "positive_control_valid": control_valid,
            "note": ("control returned hits, so an empty result for the candidate "
                     "is a true negative" if control_valid else
                     "CONTROL FAILED -- an empty candidate result cannot be trusted"),
        }
        if not control_valid:
            res["verdict"] = "UNTRUSTED"
            return res

    margin_ok = ((gap_below is None or gap_below > margin)
                 and (gap_above is None or gap_above > margin))
    clean = (not d_in and not spans and not b_in and not hist_hits
             and hist_ok and margin_ok)
    res["margin_ok"] = margin_ok
    res["verdict"] = "CLEAN" if clean else "COLLISION"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, help="first seed of the candidate block")
    ap.add_argument("--n", type=int, default=32, help="block width (default 32)")
    ap.add_argument("--margin", type=int, default=5000,
                    help="required clear margin either side (default 5000)")
    ap.add_argument("--skip-history", action="store_true",
                    help="skip the git-history check (fast, blind to deleted artifacts)")
    ap.add_argument("--json", help="write the full audit record here")
    ap.add_argument("--suggest", action="store_true",
                    help="list round bases that are clean (working tree only)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="extra path prefix to treat as a reservation record, "
                         "not a use. Repeatable. Echoed into the audit record.")
    a = ap.parse_args()
    extra = tuple(a.exclude)

    if a.suggest:
        declared, broad, _ = build_inventories(extra)
        print(f"declared={len(declared)} broad={len(broad)}")
        print(f"{'base':>10}  verdict (working tree only; confirm with --base)")
        for base in range(2_000_001, 12_000_001, 100_000):
            lo, hi = base, base + a.n - 1
            if any(lo <= v <= hi for v in declared) or any(lo <= v <= hi for v in broad):
                continue
            below = [v for v in declared if v < lo]
            above = [v for v in declared if v > hi]
            gb = lo - max(below) if below else 10 ** 9
            ga = min(above) - hi if above else 10 ** 9
            if gb > a.margin and ga > a.margin:
                print(f"{base:>10}  CLEAN  (gaps -{gb} / +{ga})")
        return 0

    if a.base is None:
        ap.error("--base is required unless --suggest is given")

    res = audit(a.base, a.n, a.margin, skip_history=a.skip_history, exclude=extra)
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nVERDICT: {res['verdict']}  for {res['block']['range']}")
    return {"CLEAN": 0, "COLLISION": 1}.get(res["verdict"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
