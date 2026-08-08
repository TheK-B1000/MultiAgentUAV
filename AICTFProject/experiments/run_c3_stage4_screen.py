"""C3 Stage-4 leg 2: fresh counterfactual commitment-fork screen.

Frozen cells: artifacts/c3_discovery/C3_STAGE4_CONFIRMATION_FROZEN.json (bebb626)
Draw:         artifacts/c3_stage4/C3_STAGE4_SAMPLE_MANIFEST.json

Reuses the discovery runner's ``_run_stage_3`` unchanged, so the counterfactual
measurement is identical to the one that produced C3_PASS. Only the anchor set,
the seed block and the output location differ.

Written as a wrapper because the discovery runner writes Stage-3 results to a
module-constant path under artifacts/c3_discovery/. Pointing it at the fresh
block would mix Stage-4 outcomes into the Stage-3 result file that C3_PASS rests
on. ``_run_stage_3`` already takes ``stage3_results_path`` as a parameter, so
the wrapper supplies the Stage-4 location and nothing is reimplemented.

Run:  python experiments/run_c3_stage4_screen.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from experiments.run_c3_decision_proximal_discovery import (  # noqa: E402
    _load_runtime_contract,
    _run_stage_3,
)
from rl.analysis.c3_discovery_artifacts import (  # noqa: E402
    STAGE3_RESULTS_NAME,
    anchor_key_from_row,
    load_completed_stage3_keys,
    load_stage1_bundle,
)

DISCOVERY_DIR = PROJECT_ROOT / "artifacts" / "c3_discovery"
STAGE4_DIR = PROJECT_ROOT / "artifacts" / "c3_stage4"
FROZEN = DISCOVERY_DIR / "C3_STAGE4_CONFIRMATION_FROZEN.json"
SAMPLE = STAGE4_DIR / "C3_STAGE4_SAMPLE_MANIFEST.json"
LOCK = STAGE4_DIR / "STAGE4_SCREEN_RUNNING.lock"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _require_authorization() -> tuple[dict, dict]:
    if not FROZEN.exists():
        raise SystemExit(f"REFUSED: Stage-4 freeze missing at {FROZEN}")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise SystemExit("REFUSED: Stage-4 cells are not FROZEN")
    if not SAMPLE.exists():
        raise SystemExit(
            f"REFUSED: sampled-anchor manifest missing at {SAMPLE}. Build it "
            "with experiments/build_c3_stage4_sample.py first."
        )
    man = json.loads(SAMPLE.read_text(encoding="utf-8"))
    if str(man.get("frozen_cells_sha256") or "") != _sha256(FROZEN):
        raise SystemExit(
            "REFUSED: the Stage-4 freeze changed after the draw was made. The "
            "draw is only valid against the cells it was drawn under."
        )
    if not man.get("drawn_before_any_stage4_outcome"):
        raise SystemExit("REFUSED: the draw does not assert it preceded outcomes")
    if not man.get("selected_anchor_ids"):
        raise SystemExit("REFUSED: the draw selects no anchors")
    return frozen, man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    frozen, man = _require_authorization()
    contract = _load_runtime_contract()

    if LOCK.exists():
        raise SystemExit(
            f"REFUSED: another Stage-4 screen holds {LOCK}. Remove it only if "
            "that process is genuinely dead."
        )

    anchors, manifest = load_stage1_bundle(STAGE4_DIR)
    selected = set(man["selected_anchor_ids"])
    by_policy: dict[int, list[dict]] = defaultdict(list)
    kept = 0
    for a in anchors:
        if anchor_key_from_row(a) in selected:
            by_policy[int(a["train_seed"])].append(a)
            kept += 1
    if kept != len(selected):
        raise SystemExit(
            f"REFUSED: fresh census contains {kept} of {len(selected)} selected "
            "anchors; manifest and census disagree."
        )

    from experiments.long_session_progress import LongSessionProgress, configure_stdio
    from experiments.run_g0_v2_evaluation import resolve_cnn_channels
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy

    configure_stdio()
    STAGE4_DIR.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({"pid": __import__("os").getpid(),
                                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                               indent=2), encoding="utf-8")
    progress = LongSessionProgress(STAGE4_DIR, name="C3_STAGE4_SCREEN")
    results_path = STAGE4_DIR / STAGE3_RESULTS_NAME
    started = time.time()

    print("=" * 78)
    print("C3 STAGE-4 LEG 2 — fresh counterfactual screen")
    print(f"seed block   : {frozen['seeds']['range']}")
    print(f"draw         : {len(selected)} anchors, seed {man['sampling_seed']}")
    print(f"results      : {results_path.relative_to(PROJECT_ROOT)}")
    print("measurement  : _run_stage_3 from the discovery runner, unchanged")
    print("=" * 78)
    sys.stdout.flush()

    try:
        completed = load_completed_stage3_keys(results_path) if args.resume else set()
        if completed:
            progress.log(f"resume: {len(completed)} anchors already done")
        for seed in sorted(by_policy):
            tag = f"g0_v5_long_seed{seed}"
            ckpt = PROJECT_ROOT / "artifacts" / "g0_v5_long" / tag / "ckpts" / f"final_{tag}.zip"
            payload = read_checkpoint_payload(str(ckpt), map_location="cpu")
            policy = load_policy(
                str(ckpt), device=args.device,
                num_cnn_channels=resolve_cnn_channels(payload, context=str(ckpt)),
            )
            progress.set_phase("STAGE4_SCREEN", f"policy_{seed}")
            _run_stage_3(
                policy, args.device, by_policy[seed], contract,
                progress=progress, train_seed=int(seed),
                stage3_results_path=results_path,
                completed_keys=completed, short_circuit=True,
            )
    finally:
        try:
            LOCK.unlink()
        except FileNotFoundError:
            pass

    print("\n" + "=" * 78)
    print(f"leg 2 screen complete in {round(time.time() - started, 1)}s")
    print("Run experiments/analyze_c3_stage4.py for the frozen verdict.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
