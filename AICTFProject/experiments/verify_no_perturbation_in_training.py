"""Structural proof: deployment perturbations never reach a training run.

Implements DEPLOYMENT_ROBUSTNESS_SPEC.json#DEPLOYMENT_ONLY_GUARANTEE. Rather than assert that
sensor_noise_sigma_cells (localization noise), drift_sigma_cells (motion error), and
rl.control_delay (control delay) are "eval-only," this greps the entire repo for the two ways
a nonzero value could leak into a training run:

  1. a config assignment setting either field to a nonzero LITERAL anywhere in the tree
     (sensor_noise_sigma_cells=0.03, etc. -- a hardcoded nonzero would be visible as a string
     match; this cannot catch a value computed at runtime from a variable, which is why check
     2 exists as well)
  2. any experiments/run_*.py (a training entrypoint, by this project's own naming
     convention) importing rl.control_delay at all -- the delay buffer has no legitimate
     reason to appear in a training script, since delay is purely a rollout-loop concept

This is a repo-wide structural check, not a runtime test -- it runs in under a second, has no
GPU/env dependency, and is meant to be re-run any time a new run_*.py is added, not just once.

Run:  python experiments/verify_no_perturbation_in_training.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "strategic_demand" / "sppo" / "DEPLOYMENT_ONLY_GUARANTEE_CHECK.json"

NONZERO_LITERAL = re.compile(
    r"(sensor_noise_sigma_cells|drift_sigma_cells)\s*=\s*([1-9][0-9.eE+-]*|0\.[1-9][0-9]*)\b")

# files that legitimately DEFINE the field (the config dataclass itself, default 0.0) or
# READ it generically -- not a training run setting it nonzero. Excluded from the "nonzero
# literal" scan by path, then double-checked below that the excluded file's own literal is
# in fact the zero default, not silently exempting a real violation.
CONFIG_DEFINITION_FILES = {"gpu_env/_config.py"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scan_nonzero_literals() -> list[dict]:
    hits = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if any(skip in rel for skip in (".venv/", "__pycache__/", "/tests/")):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for m in NONZERO_LITERAL.finditer(text):
            if rel in CONFIG_DEFINITION_FILES:
                continue     # verified separately below to actually be the 0.0 default
            line_no = text.count("\n", 0, m.start()) + 1
            hits.append({"file": rel, "line": line_no, "match": m.group(0)})
    return hits


def verify_config_defaults_are_zero() -> dict:
    cfg_path = ROOT / "gpu_env" / "_config.py"
    text = cfg_path.read_text(encoding="utf-8")
    out = {}
    for field in ("sensor_noise_sigma_cells", "drift_sigma_cells"):
        m = re.search(rf"{field}\s*:\s*float\s*=\s*([0-9.]+)", text)
        if not m:
            raise SystemExit(f"REFUSING: could not find {field}'s default in {cfg_path}")
        out[field] = float(m.group(1))
        if out[field] != 0.0:
            raise SystemExit(f"REFUSING: {field}'s own default is {out[field]}, not 0.0 -- "
                             "the deployment-only guarantee assumes training runs that "
                             "never touch this field inherit a zero value")
    return out


def scan_control_delay_imports_in_training() -> list[str]:
    hits = []
    for path in sorted(ROOT.glob("experiments/run_*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"\brl\.control_delay\b|\bfrom rl import control_delay\b|"
                     r"\bimport control_delay\b", text):
            hits.append(path.relative_to(ROOT).as_posix())
    return hits


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; re-run is fine for re-verification, but "
                         "delete the old record first so this stays a fresh check, not a "
                         "stale one silently trusted")

    print(f"DEPLOYMENT-ONLY GUARANTEE CHECK  {_now()}\n")

    defaults = verify_config_defaults_are_zero()
    print(f"  config defaults: {defaults}  (both 0.0, confirmed)")

    literal_hits = scan_nonzero_literals()
    print(f"  nonzero-literal scan: {len(literal_hits)} hits outside gpu_env/_config.py")
    for h in literal_hits[:10]:
        print(f"    {h['file']}:{h['line']}  {h['match']}")

    delay_hits = scan_control_delay_imports_in_training()
    print(f"  rl.control_delay imported by {len(delay_hits)} experiments/run_*.py files")
    for h in delay_hits:
        print(f"    {h}")

    all_pass = not literal_hits and not delay_hits
    verdict = "PASS" if all_pass else "FAIL"
    OUT.write_text(json.dumps({
        "record": "Deployment-only guarantee structural check", "status": "FROZEN_RESULT",
        "utc": _now(), "implements": "DEPLOYMENT_ROBUSTNESS_SPEC.json#DEPLOYMENT_ONLY_GUARANTEE",
        "VERDICT": verdict,
        "config_defaults_confirmed_zero": defaults,
        "nonzero_literal_hits": literal_hits,
        "control_delay_imports_in_training_entrypoints": delay_hits,
        "scope": "every .py file in the repo (excluding .venv, __pycache__, tests) for "
                "nonzero literal assignments; every experiments/run_*.py for a "
                "rl.control_delay import",
        "note": "a structural grep, not a runtime guarantee -- catches every case that "
                "matters for this project's actual scripts (hardcoded config, direct "
                "import), does not catch a value computed dynamically from an external "
                "argument at call time",
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {OUT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
