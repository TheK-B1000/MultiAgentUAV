"""Freeze SAC-RFT terminal checkpoints after both production arms complete.

Run:  python experiments/sac_rft_freeze_terminal_checkpoints.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
OUT = SD / "SAC_RFT_MODELS_FROZEN.json"
ARMS = ("control", "treatment")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    payload = {
        "record": "SAC-RFT terminal models frozen",
        "status": "FROZEN_MODELS",
        "utc": _now(),
        "implements": "SAC_RFT_SPEC.json",
        "EVAL_STATE_AT_FREEZE": {"touched": False, "block": "11804001..11804064"},
    }
    for arm in ARMS:
        rec_path = SD / f"SAC_RFT_{arm.upper()}_RESULT.json"
        if not rec_path.is_file():
            raise SystemExit(f"REFUSING: missing {rec_path.name}")
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        if rec.get("VERDICT") != "COMPLETE":
            raise SystemExit(f"REFUSING: {arm} VERDICT={rec.get('VERDICT')!r}")
        man = rec["launch_manifest"]
        ck = ROOT / man["outputs"]["checkpoint_dir"] / f"final_sac_rft_{arm}.zip"
        if not ck.is_file():
            # orchestrator may use final_<run_tag>.zip
            alt = list(Path(ROOT / man["outputs"]["checkpoint_dir"]).glob("final_*.zip"))
            if len(alt) != 1:
                raise SystemExit(f"REFUSING: cannot locate unique final checkpoint under "
                                 f"{man['outputs']['checkpoint_dir']}: {[p.name for p in alt]}")
            ck = alt[0]
        rtel = rec.get("retention_telemetry") or {}
        if arm == "control":
            if rtel.get("teacher_kind") != "ema" or int(rtel.get("ema_updates", 0)) <= 0:
                raise SystemExit("REFUSING: CONTROL retention telemetry is not a live EMA path")
        else:
            if rtel.get("teacher_kind") != "frozen_anchor":
                raise SystemExit("REFUSING: TREATMENT retention telemetry is not frozen_anchor")
            if int(rtel.get("ema_updates", 0)) != 0:
                raise SystemExit("REFUSING: TREATMENT reported ema_updates > 0")
            if rec.get("frozen_anchor_moved"):
                raise SystemExit("REFUSING: TREATMENT frozen anchor moved")
        key = arm.upper()
        payload[key] = {
            "TERMINAL_CHECKPOINT": {
                "path": str(ck.relative_to(ROOT)),
                "sha256": _sha(ck),
            },
            "TERMINAL_RECORD_VALIDITY": {
                "verdict": "VALID",
                "result_record": str(rec_path.relative_to(ROOT)),
                "steps_advanced": rec.get("steps_advanced"),
                "retention_path": rtel,
            },
        }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"FROZEN -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
