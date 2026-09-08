"""Specialist difficulty anchor on the CCP-S2 and RSCFT seed blocks.

WHY THIS EXISTS. Every eval before CCP-S2 scored pi_A on Pole A and pi_B on Pole B alongside
z0/z1, giving a per-block difficulty reference. eval_ccp_s2.py and eval_rscft.py deliberately
omitted those cells -- their frozen PRIMARY_GATE only needed z0/z1, so the specialists were
dropped to keep the eval minimal. That choice left the 11701xxx and 11704xxx blocks with NO
difficulty anchor, which means an absolute statement like "V(z0,A) fell from 0.8438 to 0.5469"
cannot currently be separated from "those blocks were simply harder."

This measures exactly the missing cells: pi_A@PoleA and pi_B@PoleB, on both blocks, matching
the cell convention every pre-CCP-S2 eval used.

WHAT THIS IS NOT. There is no gate here, no pass/fail, no verdict. It is a calibration
measurement of two FROZEN, sha-pinned checkpoints that were never trained on or tuned against
these seeds. It supports an EXPLORATORY, post-hoc analysis
(LATENT_DOMINANCE_DRIFT_ANALYSIS.json) and inherits that status: nothing here may be reported
as a confirmatory finding.

SEED REUSE. These blocks were already opened by CCP-S2's and RSCFT's own one-shot evals.
Scoring a DIFFERENT, previously-unevaluated policy on the same seeds does not reopen or
contaminate those results -- the one-shot rule protects against re-scoring the SAME arms to
fish for a better number, which is not what this does.

Run:  python experiments/eval_specialist_anchor.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval_hog_psp_v3 import _mean_ci

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
PHASE1 = SD / "CCP_PHASE1_PILOT_MANIFEST.json"
OUT = SD / "SPECIALIST_ANCHOR_RESULT.json"
ROWS_CSV = SD / "specialist_anchor_eval_rows.csv"

BLOCKS = {
    "CCP_S2": list(range(11_701_001, 11_701_065)),
    "RSCFT": list(range(11_704_001, 11_704_065)),
}
# match every pre-CCP-S2 eval's convention: each specialist on its OWN pole only
CELLS = (("pi_A", "A"), ("pi_B", "B"))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if OUT.is_file() or ROWS_CSV.is_file():
        raise SystemExit("REFUSING: specialist anchor output already exists; one-shot")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    manifest = json.loads(PHASE1.read_text(encoding="utf-8"))
    teachers_meta = manifest["TEACHER_POLICIES"]
    paths = {}
    for name in ("pi_A", "pi_B"):
        p = ROOT / teachers_meta[name]["path"]
        if not p.is_file():
            raise SystemExit(f"REFUSING: {name} checkpoint missing: {p}")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != teachers_meta[name]["sha256"]:
            raise SystemExit(f"REFUSING: {name} sha mismatch -- frozen manifest disagrees")
        paths[name] = p

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"SPECIALIST DIFFICULTY ANCHOR  {_now()}")
    print("  both specialist checkpoints sha256 VERIFIED against the frozen Phase-1 manifest")
    print(f"  blocks: " + ", ".join(f"{k} ({v[0]}..{v[-1]})" for k, v in BLOCKS.items()))
    print(f"  cells per block: {CELLS}  -> {len(BLOCKS)*len(CELLS)*64} episodes total")
    print("  EXPLORATORY calibration -- no gate, no verdict\n", flush=True)

    probe = R2.build_env(device, BLOCKS["CCP_S2"][0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policies = {n: load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
                for n, p in paths.items()}

    def run_episode(policy, pole: str, seed: int) -> dict:
        env = R2.build_env(device, seed)
        core = env.core
        try:
            policy.reset_strategy()
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            genomes = {"OP6": pole_A_genome()} if pole == "A" else {}
            install_keyed_opponent_overlays(core, genomes)
            key = P0.POLES[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                                       context=f"anchor {pole} seed {seed}")
            terminal = None
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    res = (i0 or {}).get("episode_result") or {}
                    terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                    break
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            blue, red = terminal
            return {"blue": blue, "red": red, "win": int(blue > red), "margin": blue - red}
        finally:
            env.close()

    rows, summary = [], {}
    for block, seeds in BLOCKS.items():
        summary[block] = {}
        for name, pole in CELLS:
            for s in seeds:
                rows.append({"block": block, "policy": name, "pole": pole, "seed": s,
                             **run_episode(policies[name], pole, s)})
            w = np.array([r["win"] for r in rows
                          if r["block"] == block and r["policy"] == name and r["pole"] == pole],
                         dtype=np.float64)
            ci = _mean_ci(w)
            summary[block][f"{name}@{pole}"] = ci
            print(f"  {block:8s} {name}@{pole}: {ci['mean']:.4f} "
                  f"[{ci['lcb95']:.4f}, {ci['ucb95']:.4f}]", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    OUT.write_text(json.dumps({
        "record": "Specialist difficulty anchor on the CCP-S2 and RSCFT seed blocks",
        "status": "EXPLORATORY_CALIBRATION", "one_shot": True, "utc": _now(),
        "why": "eval_ccp_s2.py and eval_rscft.py omitted the pi_A@A / pi_B@B reference cells "
               "that every pre-CCP-S2 eval recorded, leaving those blocks without a difficulty "
               "anchor. This supplies exactly those cells.",
        "not_preregistered": True,
        "no_gate_no_verdict": True,
        "checkpoints": {n: {"path": teachers_meta[n]["path"],
                            "sha256": teachers_meta[n]["sha256"]} for n in ("pi_A", "pi_B")},
        "blocks": {k: [v[0], v[-1]] for k, v in BLOCKS.items()},
        "summary": summary,
        "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
