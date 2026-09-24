r"""Rule 8 -- training runs export EVERYTHING, and the bundle is validated.

Generalises ``export_6v6_results.sh`` to 2v2 / 4v4 / 6v6 (and any future scale),
because the failure it was written for was never specific to 6v6:

    The original 6v6 exporter copied only terminal checkpoints, manifests and
    specs. ``metrics.csv``, ``episode_rows.csv`` and every intermediate
    checkpoint were silently omitted, and became unrecoverable once the source
    machine no longer had them. That permanently removed the ability to diagnose
    6v6 training dynamics -- the exact question we later needed to answer.

Two things prevent a repeat, and the second matters more than the first:

  ``--strict``   refuses to call a bundle complete, naming every missing file.
  ``verify``     re-checks a bundle against its own MANIFEST.json *on the
                 receiving machine*, by sha256. The original loss was discovered
                 long after the source was gone; verification at the destination
                 catches it while recovery is still possible.

    python experiments/export_run_bundle.py 6v6 --strict
    python experiments/export_run_bundle.py 4v4 --suffix _b3
    python experiments/export_run_bundle.py verify artifacts/4v4_results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, cwd=ROOT, capture_output=True, text=True,
                                  timeout=15).stdout.strip()
        except Exception:                                    # noqa: BLE001
            return "unknown"
    return {"sha": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


#: Per-run files that must be exported. These are the training DYNAMICS -- small
#: text files whose absence cost us the 6v6 diagnosis.
PER_RUN_REQUIRED = ("metrics.csv", "episode_rows.csv", "training_manifest.json",
                    "result_summary.json")
PER_RUN_OPTIONAL = ("evaluation_manifest.json", "run_manifest.json")


class Bundle:
    def __init__(self, out: Path, strict: bool, plan_only: bool = False) -> None:
        self.out, self.strict, self.plan_only = out, strict, plan_only
        self.inventory: dict[str, dict] = {}
        self.missing: list[str] = []
        if not plan_only:
            out.mkdir(parents=True, exist_ok=True)

    def take(self, src: Path, rel: str, desc: str, required: bool = True) -> bool:
        if not src.is_file():
            self.inventory[rel] = {"present": False, "desc": desc, "required": required,
                                   "source": str(src)}
            if required:
                self.missing.append(f"{rel}  ({desc})  <- {src}")
            return False
        if self.plan_only:
            # Inventory without copying, so completeness can be checked before
            # committing gigabytes of disk. No sha: hashing means reading anyway.
            self.inventory[rel] = {"present": True, "bytes": src.stat().st_size,
                                   "desc": desc, "required": required}
            return True
        dst = self.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.inventory[rel] = {"present": True, "bytes": dst.stat().st_size,
                               "sha256": _sha(dst), "desc": desc, "required": required}
        return True

    def take_glob(self, srcdir: Path, pattern: str, reldir: str, desc: str,
                  required: bool = True) -> int:
        files = sorted(srcdir.glob(pattern)) if srcdir.is_dir() else []
        for f in files:
            self.take(f, f"{reldir}/{f.name}", desc, required=False)
        self.inventory[reldir + "/"] = {"present": bool(files), "n_files": len(files),
                                        "files": [f.name for f in files], "desc": desc,
                                        "required": required}
        if required and not files:
            self.missing.append(f"{reldir}/  ({desc}) -- NO files matched {pattern}")
        return len(files)

    def finish(self, meta: dict) -> int:
        if self.plan_only:
            total = sum(v.get("bytes", 0) for v in self.inventory.values())
            n = sum(1 for v in self.inventory.values() if v.get("present") and "bytes" in v)
            print(f"\n  PLAN ONLY -- nothing copied.")
            print(f"  would export {n} file(s), {total / 1e6:.1f} MB -> {self.out}")
            if self.missing:
                print(f"  {len(self.missing)} required artifact(s) would be MISSING:")
                for m in self.missing:
                    print(f"     MISSING: {m}")
                return 1 if self.strict else 0
            print("  every required artifact is present at the source.")
            return 0
        doc = {"record": "run export bundle manifest (Rule 8)",
               "rule": "a transfer is NOT complete until every required artifact is "
                       "present. sha256 is recorded per file so the RECEIVING machine "
                       "can verify integrity while recovery is still possible.",
               "created_utc": _now(), "git": _git(),
               "host": platform.node(), "python": sys.version.split()[0],
               **meta,
               "complete": not self.missing, "n_missing": len(self.missing),
               "n_files": sum(1 for v in self.inventory.values() if v.get("present")
                              and "sha256" in v),
               "inventory": self.inventory}
        (self.out / "MANIFEST.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")

        total = sum(v.get("bytes", 0) for v in self.inventory.values())
        print(f"\n  bundle   {self.out}")
        print(f"  files    {doc['n_files']}   ({total / 1e6:.1f} MB)")
        print(f"  manifest {self.out / 'MANIFEST.json'}")
        if self.missing:
            print(f"\n  INCOMPLETE -- {len(self.missing)} required artifact(s) missing:")
            for m in self.missing:
                print(f"     MISSING: {m}")
            if self.strict:
                print("\n  FAILING (--strict). Do NOT treat this run as transferred.\n"
                      "  Recover the missing files from the source machine BEFORE it is "
                      "wiped -- that is exactly the loss this rule exists to prevent.")
                return 1
            print("  (permissive: run with --strict before declaring a transfer complete)")
        else:
            print("\n  COMPLETE -- every required artifact present, hashes recorded.")
        return 0


def export_scale(scale: str, suffix: str, out: Path, strict: bool,
                 extra_specs: list[str], plan_only: bool = False) -> int:
    n = scale[0]
    src_root = ROOT / "artifacts" / f"scale_{scale}_specialists"
    if not src_root.is_dir():
        raise SystemExit(f"no such scale directory: {src_root}")

    b = Bundle(out, strict, plan_only)
    runs = []
    print(f"=== {scale} specialists (suffix {suffix!r}) ===")
    for P in ("A", "B"):
        label = f"pi_{P}_specialist_{scale}{suffix}"
        d = src_root / label
        runs.append(label)
        if not d.is_dir():
            b.missing.append(f"specialists/{label}/ -- run directory not found: {d}")
            b.inventory[f"specialists/{label}/"] = {"present": False, "required": True,
                                                    "desc": "training run directory"}
            continue
        # terminal checkpoint
        b.take(d / "ckpts" / f"final_{label}.zip", f"specialists/final_{label}.zip",
               f"terminal checkpoint pi_{P}")
        # training dynamics -- the files the original exporter dropped
        for f in PER_RUN_REQUIRED:
            b.take(d / f, f"specialists/{label}__{f}", f"{f} pi_{P}")
        for f in PER_RUN_OPTIONAL:
            b.take(d / f, f"specialists/{label}__{f}", f"{f} pi_{P}", required=False)
        for cfg in d.glob("*_run_config.json"):
            b.take(cfg, f"specialists/{label}__{cfg.name}", "run config", required=False)
        # intermediate checkpoints -- required for any checkpoint-over-time trajectory
        k = b.take_glob(d / "ckpts", f"ckpt_{label}_*.zip",
                        f"specialists/{label}__ckpts", f"intermediate checkpoints pi_{P}")
        print(f"  pi_{P}: {k} intermediate checkpoint(s)")

    print(f"=== sealed artifacts matching {scale} ===")
    pats = [f"*{scale.upper()}*", f"*{n}V{n}*", f"*{n}v{n}*"] + list(extra_specs)
    seen: set[str] = set()
    for pat in pats:
        for f in sorted(SD.glob(f"{pat}.json")):
            if f.name in seen:
                continue
            seen.add(f.name)
            b.take(f, f"sealed/{f.name}", "sealed spec / result / audit", required=False)
    for pat in (f"*{n}v{n}*rows.csv", f"*{scale}*rows.csv"):
        for f in sorted(SD.glob(pat)):
            b.take(f, f"sealed/{f.name}", "raw episode rows", required=False)
    print(f"  {len(seen)} sealed JSON artifact(s)")

    return b.finish({"scale": scale, "suffix": suffix, "runs": runs,
                     "source_root": str(src_root.relative_to(ROOT))})


def verify(bundle: Path) -> int:
    """Re-check a bundle against its own manifest. Run this on the machine that
    RECEIVED the transfer, before the source is wiped."""
    mf = bundle / "MANIFEST.json"
    if not mf.is_file():
        raise SystemExit(f"no MANIFEST.json in {bundle} -- this bundle was not produced "
                         f"by export_run_bundle.py and cannot be verified.")
    doc = json.loads(mf.read_text(encoding="utf-8"))
    bad, ok_n = [], 0
    for rel, e in doc["inventory"].items():
        if not e.get("present") or "sha256" not in e:
            continue
        p = bundle / rel
        if not p.is_file():
            bad.append(f"LOST:      {rel}")
        elif _sha(p) != e["sha256"]:
            bad.append(f"CORRUPTED: {rel}")
        else:
            ok_n += 1
    print(f"verifying {bundle}")
    print(f"  manifest created {doc.get('created_utc')} on {doc.get('host')}")
    print(f"  git {doc.get('git', {}).get('sha', '?')[:12]} "
          f"({'dirty' if doc.get('git', {}).get('dirty') else 'clean'})")
    print(f"  {ok_n} file(s) verified by sha256")
    if not doc.get("complete", True):
        print(f"  NOTE: this bundle was recorded INCOMPLETE at creation "
              f"({doc.get('n_missing')} required artifact(s) were already missing).")
    if bad:
        print(f"\n  FAILED -- {len(bad)} problem(s):")
        for x in bad:
            print(f"     {x}")
        return 1
    if not doc.get("complete", True):
        return 1
    print("\n  VERIFIED -- bundle is complete and every file matches its hash.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Rule 8 run-bundle exporter")
    ap.add_argument("target", help="a scale (2v2/4v4/6v6) or the word 'verify'")
    ap.add_argument("bundle", nargs="?", help="bundle directory, for 'verify'")
    ap.add_argument("--suffix", default="", help="run-label suffix, e.g. _b3")
    ap.add_argument("--out", help="output directory (default artifacts/<scale>_results)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any required artifact is missing")
    ap.add_argument("--plan", action="store_true",
                    help="inventory what WOULD be exported; copy nothing")
    ap.add_argument("--spec-glob", action="append", default=[],
                    help="extra sealed-artifact name patterns to include")
    a = ap.parse_args()

    if a.target == "verify":
        if not a.bundle:
            raise SystemExit("usage: export_run_bundle.py verify <bundle-dir>")
        return verify(Path(a.bundle))

    if not a.target.count("v") == 1:
        raise SystemExit(f"unrecognised target {a.target!r}; expected e.g. 4v4 or 'verify'")
    out = Path(a.out) if a.out else ROOT / "artifacts" / f"{a.target}{a.suffix}_results"
    return export_scale(a.target, a.suffix, out, a.strict, a.spec_glob, a.plan)


if __name__ == "__main__":
    raise SystemExit(main())
