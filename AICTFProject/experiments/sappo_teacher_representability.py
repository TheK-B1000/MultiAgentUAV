"""SAPPO V1 prerequisite — is the validated teacher representable in PPO's action space?

Strategy-Anchored PPO clones pi_A toward scripted GUARD and pi_B toward scripted
BREACH. That only makes sense if those teachers can be EXPRESSED as PPO actions.
They cannot be cloned directly, because the two controllers do not share an
action space:

    scripted teacher : writes continuous (target_x, target_y) every step,
                       bypassing macros entirely
    PPO              : emits (macro, target_idx) per agent, where target_idx
                       indexes 50 FIXED waypoints, and the choice persists for
                       blue_commit_ticks_left steps before it can change

So a teacher demonstration must be PROJECTED into PPO's action space, and the
projection is lossy on two axes: spatially (continuous -> 50 waypoints) and
temporally (per-step retargeting -> committed macros).

This probe measures whether the projection preserves the thing that matters.
It is NOT the strategic claim -- that is already validated for the true scripted
teachers. It asks the narrower engineering question:

    Does a PPO-action-space replay of the projected teacher still reproduce the
    teacher's strategic advantage on its own pole?

Three arms per pole, on the same paired seeds:

    TRUE      the scripted style itself (the validated teacher)
    PROJECTED the teacher's continuous target projected each step to the nearest
              of the 50 macro waypoints, then issued as a PPO action
    OPPOSITE  the other scripted style, as the reference contrast

If PROJECTED tracks TRUE, the teacher is representable and SAPPO V1 is well
posed. If PROJECTED collapses toward OPPOSITE, then cloning toward the projected
teacher would anchor the student to something that is NOT the validated
strategy, and the anchor dataset must be built differently before any training.

Training-only seeds. No evaluation block is touched.

Run:  python experiments/sappo_teacher_representability.py --device cuda --n 24
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.r2_learned_crossover as R2                      # noqa: E402
from experiments.opponent_spec import (                            # noqa: E402
    assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
)
from rl.curriculum import phase_from_tag                           # noqa: E402

GUARD = "BLUE_ONE_DEFENDER_V2"
BREACH = "BLUE_BOTH_ATTACK_V2"
TEACHER_FOR = {"A": GUARD, "B": BREACH}
OPPOSITE_FOR = {"A": BREACH, "B": GUARD}

# Training-only block, disjoint from every evaluation block.
SEED_BASE = 7_900_001
OUT_DIR = ROOT / "artifacts/strategic_demand/sappo_representability"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prep(env, core, pole: str, seed: int):
    base_key = R2.POLES[pole]
    genomes = {"OP6": pole_A_genome()} if pole == "A" else {}
    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0
    install_keyed_opponent_overlays(core, genomes)
    env.env_method("set_phase", phase_from_tag(base_key))
    env.env_method("set_next_opponent", "SCRIPTED", base_key)
    obs = env.reset()
    assert_live_opponent_batch(core, genomes, allowed_keys=(base_key,),
                               context=f"representability {pole} seed {seed}")
    return obs


def _terminal(core, info):
    i0 = info[0] if isinstance(info, (list, tuple)) else info
    er = (i0 or {}).get("episode_result") or {}
    return int(er.get("blue_score", 0)), int(er.get("red_score", 0))


def run_scripted(pole: str, style: str, seed: int, device: str) -> dict:
    """Arm TRUE / OPPOSITE: the scripted controller drives directly."""
    env = R2.build_env(device, seed)
    core = env.core
    try:
        _prep(env, core, pole, seed)
        core.blue_scripted = True
        core.set_blue_style(style)
        env.reset()
        _prep(env, core, pole, seed)
        core.blue_scripted = True
        core.set_blue_style(style)
        zero = np.zeros(1 * 2 * 2, dtype=np.float32)
        term = None
        for _ in range(R2.MAX_STEPS):
            env.step_async(zero)
            _o, _r, done, info = env.step_wait()
            if bool(np.asarray(done).any()):
                term = _terminal(core, info)
                break
        if term is None:
            term = (int(core.blue_score[0]), int(core.red_score[0]))
        return {"blue": term[0], "red": term[1], "win": int(term[0] > term[1])}
    finally:
        env.close()


def run_projected(pole: str, style: str, seed: int, device: str) -> dict:
    """Arm PROJECTED: read the teacher's continuous target each step, snap it to
    the nearest macro waypoint, and issue that as a PPO action.

    The macro is inferred from the same rule the engine uses for intent:
    a carrier is going home, everyone else is going to a place.
    """
    from macro_actions import MacroAction
    env = R2.build_env(device, seed)
    core = env.core
    try:
        _prep(env, core, pole, seed)
        # Scripted ON only to COMPUTE targets; blue is driven by our actions.
        core.blue_scripted = True
        core.set_blue_style(style)
        wp = core._macro_targets            # [n_targets, 2] waypoint table
        proj_err = []
        term = None
        for _ in range(R2.MAX_STEPS):
            tx, ty = core._assign_blue_style_targets()
            t = torch.stack([tx[0], ty[0]], dim=-1)          # [Nb, 2]
            d = torch.cdist(t.unsqueeze(0).float(), wp.unsqueeze(0).float())[0]
            idx = torch.argmin(d, dim=-1)                    # [Nb]
            proj_err.append(torch.gather(d, 1, idx.unsqueeze(1)).squeeze(1))
            carrying = core.blue_carrying[0]
            macro = torch.where(carrying,
                                torch.full_like(idx, int(MacroAction.GO_HOME)),
                                torch.full_like(idx, int(MacroAction.GO_TO)))
            act = torch.stack([macro, idx], dim=-1).reshape(-1).cpu().numpy()
            # Blue must be action-driven for this arm.
            core.blue_scripted = False
            env.step_async(act.astype(np.int64))
            _o, _r, done, info = env.step_wait()
            core.blue_scripted = True       # restore so targets stay computable
            if bool(np.asarray(done).any()):
                term = _terminal(core, info)
                break
        if term is None:
            term = (int(core.blue_score[0]), int(core.red_score[0]))
        err = torch.cat(proj_err).float().mean().item() if proj_err else float("nan")
        return {"blue": term[0], "red": term[1], "win": int(term[0] > term[1]),
                "mean_projection_error_cells": err}
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()

    print(f"SAPPO TEACHER REPRESENTABILITY PROBE  {_now()}")
    print(f"  training-only seeds {SEED_BASE}..{SEED_BASE + a.n - 1}")
    print(f"  arms: TRUE (scripted) | PROJECTED (snapped to 50 waypoints) | OPPOSITE")

    out = {}
    for pole in ("A", "B"):
        teacher, opposite = TEACHER_FOR[pole], OPPOSITE_FOR[pole]
        arms = {"TRUE": [], "PROJECTED": [], "OPPOSITE": [],
                "PROJECTED_OPPOSITE": []}
        errs = []
        for i in range(a.n):
            seed = SEED_BASE + i
            arms["TRUE"].append(run_scripted(pole, teacher, seed, a.device)["win"])
            pr = run_projected(pole, teacher, seed, a.device)
            arms["PROJECTED"].append(pr["win"])
            errs.append(pr["mean_projection_error_cells"])
            arms["OPPOSITE"].append(run_scripted(pole, opposite, seed, a.device)["win"])
            arms["PROJECTED_OPPOSITE"].append(
                run_projected(pole, opposite, seed, a.device)["win"])
            print(f"  pole {pole} seed {seed}: TRUE={arms['TRUE'][-1]} "
                  f"PROJ={arms['PROJECTED'][-1]} OPP={arms['OPPOSITE'][-1]} "
                  f"PROJ_OPP={arms['PROJECTED_OPPOSITE'][-1]}", flush=True)
        out[pole] = {k: float(np.mean(v)) for k, v in arms.items()}
        out[pole]["mean_projection_error_cells"] = float(np.nanmean(errs))
        out[pole]["n"] = a.n

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import hashlib
    invocation = f"n={a.n};device={a.device};seeds={SEED_BASE};arms=TRUE,PROJECTED,OPPOSITE,PROJECTED_OPPOSITE"
    rec = {
        "record": "SAPPO teacher representability probe", "utc": _now(),
        "run_identity": {
            "invocation": invocation,
            "invocation_sha256": hashlib.sha256(invocation.encode()).hexdigest()[:16],
            "arms": ["TRUE", "PROJECTED", "OPPOSITE", "PROJECTED_OPPOSITE"],
            "why_stamped": ("a stale duplicate probe once shared this output path. A summary "
                            "must be attributable to its exact invocation, not merely exist."),
        },
        "status": "ENGINEERING PREREQUISITE, not a strategic claim",
        "question": "does a PPO-action-space replay of the projected teacher "
                    "reproduce the teacher's advantage on its own pole?",
        "seeds": f"{SEED_BASE}..{SEED_BASE + a.n - 1} (training-only)",
        "results": out,
        "frozen_criterion": {
            "primary": "PROJECTED must preserve the STRATEGIC ORDERING, not reproduce TRUE "
                       "exactly: on pole A, PROJECTED(GUARD) > PROJECTED(BREACH); on pole B, "
                       "PROJECTED(BREACH) > PROJECTED(GUARD).",
            "why_projected_vs_projected": "both sides must be projected, because that is the "
                                          "pair SAPPO would actually clone toward. Comparing a "
                                          "projected teacher against a SCRIPTED opposite would "
                                          "flatter the projection.",
            "question_is_not": "is projected GUARD pixel-perfect GUARD?",
            "question_is": "does the PPO-action-space version still occupy the validated GUARD "
                           "strategic basin?",
        },
        "temporal_projection_verified": (
            "the engine enforces macro commitment itself: blue_commit_ticks_left gates whether "
            "an issued action latches, so issuing a projected target every step is NOT per-step "
            "control. Measured on a live core: latched target changed 2 times over 39 steps "
            "while issued targets changed 4 times, with ticks cycling 3->2->1->0. The probe "
            "therefore does not overestimate representability on the temporal axis."),
        "dataset_implication": (
            "demonstration labels must still be collected ONLY at PPO decision points "
            "(blue_commit_ticks_left <= 0), one label per macro decision rather than one per "
            "physics step, or the dataset is dominated by locked-in steps where no decision is "
            "being made."),
        "reading": ("PROJECTED close to TRUE and clearly above PROJECTED_OPPOSITE => the teacher is "
                    "representable and SAPPO V1 is well posed. PROJECTED collapsing toward "
                    "OPPOSITE => cloning toward the projected teacher would anchor the "
                    "student to something that is NOT the validated strategy, and the anchor "
                    "dataset must be built differently before any training."),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"{'pole':<6}{'TRUE':>9}{'PROJ':>9}{'OPP':>9}{'PROJ_OPP':>10}"
          f"{'PROJ-PROJ_OPP':>15}{'proj_err':>10}")
    for pole in ("A", "B"):
        r = out[pole]
        gap = r["PROJECTED"] - r["PROJECTED_OPPOSITE"]
        print(f"{pole:<6}{r['TRUE']:>9.3f}{r['PROJECTED']:>9.3f}{r['OPPOSITE']:>9.3f}"
              f"{r['PROJECTED_OPPOSITE']:>10.3f}{gap:>+15.3f}"
              f"{r['mean_projection_error_cells']:>10.3f}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
