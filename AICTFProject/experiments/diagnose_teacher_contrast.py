"""Teacher-contrast feasibility: do pi_A and pi_B prescribe different actions?

The final latent attempt hinges on this. The corrected treatment presents BOTH
latents with the SAME resolved state:

    (s, z0) -> pi_A(s)        (s, z1) -> pi_B(s)

A z-independent policy cannot satisfy both targets ONLY where the teachers actually
disagree. If they mostly agree, action-level anchoring cannot encode the strategic
distinction no matter how large the bank, and the latent line should stop.

Measured DECISION-MASKED throughout. An earlier ad-hoc pass reported 33.4% identical
action vectors by averaging over inactive heads, which is the same masking error that
made the FIT/CALIB diagnostic look contradictory. Disagreement on a locked head is
not usable supervision, so only heads the anchor loss actually trains are counted.

Reports argmax disagreement and teacher-teacher JSD, split by preference direction,
pole, regime, and horizon. Existing resolved states only -- no new collection, no
EVAL, no model training.

Run:  python experiments/diagnose_teacher_contrast.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand"
DATA = SD / "stratified_regime_data" / "seed_shards"
OUT = SD / "sppo" / "TEACHER_CONTRAST_DIAGNOSTIC.json"

FIT = (10_700_001, 10_700_096)
CALIB = (10_700_097, 10_700_128)
CELL_RE = re.compile(r"^(?P<pole>[AB])_r(?P<regime>\d)_(?P<hor>not_late|late)$")

# Frozen decision rule, stated before the numbers are read.
SUBSTANTIAL_CONTRAST = 0.20   # >=20% of decision-masked heads disagreeing


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_resolved(lo: int, hi: int) -> dict:
    keys = ("grid", "vec", "amask", "mask", "pi_a", "pi_b", "delta", "cell")
    b: dict[str, list] = {k: [] for k in keys}
    for s in range(lo, hi + 1):
        with np.load(DATA / f"seed_{s}.npz", allow_pickle=False) as z:
            d = ((z["branch_pi_B_blue"].astype(np.int64) - z["branch_pi_B_red"].astype(np.int64))
                 - (z["branch_pi_A_blue"].astype(np.int64) - z["branch_pi_A_red"].astype(np.int64)))
            keep = np.nonzero(d != 0)[0]
            if not len(keep):
                continue
            b["grid"].append(z["branch_obs_grid"][keep][:, 0])
            b["vec"].append(z["branch_obs_vec"][keep][:, 0])
            b["amask"].append(z["branch_obs_agent_mask"][keep][:, 0])
            b["mask"].append(z["branch_obs_mask"][keep][:, 0])
            b["pi_a"].append(z["branch_pi_A_action"][keep])
            b["pi_b"].append(z["branch_pi_B_action"][keep])
            b["delta"].append(d[keep])
            b["cell"].extend(str(c) for c in z["branch_cell"][keep])
    out = {k: (np.concatenate(v) if k != "cell" else np.array(v)) for k, v in b.items()}
    out["a_preferred"] = out["delta"] < 0
    out["b_preferred"] = out["delta"] > 0
    return out


def head_mask(amask: np.ndarray, n_heads: int) -> np.ndarray:
    """(N, n_heads) bool: which heads the anchor loss actually trains."""
    n_agents = amask.shape[1]
    return np.repeat(amask.astype(bool), n_heads // n_agents, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.custom_ppo import load_custom_ppo_policy

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; this diagnostic is one-shot")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    probe = R2.build_env(device, FIT[0])
    osp, asp = probe.observation_space, probe.action_space
    probe.close()
    teachers = {k: load_custom_ppo_policy(str(v), osp, asp, device=device)
                for k, v in P0.TEACHERS.items()}

    def teacher_probs(pol, split, idx):
        sub = {"grid": split["grid"][idx], "vec": split["vec"][idx],
               "agent_mask": split["amask"][idx], "mask": split["mask"][idx]}
        ot = pol._tensor_obs(pol._batched_obs(sub))
        with torch.no_grad():
            return [p.detach() for p in pol.get_distribution(ot, z_idx=None).probabilities()]

    def jsd_per_state(split) -> np.ndarray:
        """Mean JSD (bits) across decision-masked heads, per state."""
        n = len(split["delta"])
        vals = []
        for i in range(0, n, 128):
            idx = np.arange(i, min(i + 128, n))
            pa = teacher_probs(teachers["pi_A"], split, idx)
            pb = teacher_probs(teachers["pi_B"], split, idx)
            per_head = []
            for p, q in zip(pa, pb):
                m = 0.5 * (p + q)
                eps = 1e-8
                kl = lambda x, y: (x * ((x + eps).log2() - (y + eps).log2())).sum(-1)
                per_head.append((0.5 * kl(p, m) + 0.5 * kl(q, m)).cpu().numpy())
            per_head = np.stack(per_head, axis=1)
            hm = head_mask(split["amask"][idx], per_head.shape[1])
            vals.append((per_head * hm).sum(1) / np.maximum(hm.sum(1), 1))
        return np.concatenate(vals)

    report = {}
    for name, (lo, hi) in (("FIT", FIT), ("CALIB", CALIB)):
        sp = load_resolved(lo, hi)
        n_heads = sp["pi_a"].shape[1]
        hm = head_mask(sp["amask"], n_heads)
        disagree = (sp["pi_a"] != sp["pi_b"]) & hm

        per_state_rate = disagree.sum(1) / np.maximum(hm.sum(1), 1)
        any_disagree = disagree.any(1)
        overall = float(disagree.sum() / max(1, hm.sum()))
        jsd = jsd_per_state(sp)

        def block(mask, label):
            if not mask.any():
                return {"n": 0}
            return {"n": int(mask.sum()),
                    "head_disagreement_rate": round(float(per_state_rate[mask].mean()), 4),
                    "states_with_any_disagreement": round(float(any_disagree[mask].mean()), 4),
                    "teacher_jsd_bits_mean": round(float(jsd[mask].mean()), 4)}

        by_cell, by_pole, by_regime, by_hor = {}, defaultdict(list), defaultdict(list), defaultdict(list)
        for i, c in enumerate(sp["cell"]):
            m = CELL_RE.match(c)
            if not m:
                continue
            by_pole[m["pole"]].append(i)
            by_regime[f"r{m['regime']}"].append(i)
            by_hor[m["hor"]].append(i)
        for cell in sorted(set(sp["cell"])):
            sel = sp["cell"] == cell
            by_cell[cell] = block(sel, cell)

        def group(d):
            out = {}
            for k, idxs in sorted(d.items()):
                sel = np.zeros(len(sp["delta"]), dtype=bool)
                sel[np.array(idxs)] = True
                out[k] = block(sel, k)
            return out

        report[name] = {
            "n_resolved_states": int(len(sp["delta"])),
            "n_heads_total": int(hm.sum()),
            "OVERALL": {
                "head_disagreement_rate": round(overall, 4),
                "states_with_any_disagreement": round(float(any_disagree.mean()), 4),
                "teacher_jsd_bits_mean": round(float(jsd.mean()), 4)},
            "by_preference": {"A_preferred": block(sp["a_preferred"], "A"),
                              "B_preferred": block(sp["b_preferred"], "B")},
            "by_pole": group(by_pole),
            "by_regime": group(by_regime),
            "by_horizon": group(by_hor),
            "by_cell": by_cell,
        }
        o = report[name]["OVERALL"]
        print(f"=== {name} ({report[name]['n_resolved_states']} resolved states) ===")
        print(f"  decision-masked head disagreement : {o['head_disagreement_rate']:.4f}")
        print(f"  states with >=1 head disagreeing  : {o['states_with_any_disagreement']:.4f}")
        print(f"  teacher-teacher JSD (bits)        : {o['teacher_jsd_bits_mean']:.4f}")
        for k, v in report[name]["by_preference"].items():
            print(f"    {k:12s} n={v['n']:4d}  disagree {v['head_disagreement_rate']:.4f}  "
                  f"jsd {v['teacher_jsd_bits_mean']:.4f}")
        print()

    fit_rate = report["FIT"]["OVERALL"]["head_disagreement_rate"]
    substantial = fit_rate >= SUBSTANTIAL_CONTRAST
    verdict = ("SUBSTANTIAL_TEACHER_CONTRAST" if substantial
               else "INSUFFICIENT_TEACHER_CONTRAST")
    reading = (
        "The teachers prescribe different actions often enough that paired latent "
        "supervision -- (s,z0)->pi_A and (s,z1)->pi_B on the SAME state -- creates "
        "targets no z-independent policy can satisfy. The corrected final latent "
        "treatment is feasible."
        if substantial else
        "The teachers mostly agree at the action level on resolved states. Paired "
        "latent supervision would supply almost no contradictory targets, so a "
        "z-independent policy could still satisfy both. Action-level anchoring cannot "
        "encode the strategic distinction regardless of bank size, and the latent line "
        "should stop.")

    OUT.write_text(json.dumps({
        "record": "Teacher-contrast feasibility for the corrected final latent treatment",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "question": ("How often do frozen pi_A and pi_B prescribe DIFFERENT actions on "
                     "the same resolved state?"),
        "why_it_decides_the_fork": (
            "The corrected treatment gives both latents the same state with different "
            "teacher targets. That only forces z to matter where the teachers disagree."),
        "metric_note": (
            "DECISION-MASKED throughout. An earlier ad-hoc pass reported 33.4% identical "
            "action vectors by averaging over inactive heads -- the same masking error "
            "that made the FIT/CALIB crossed comparison look contradictory. Only heads "
            "the anchor loss trains are counted."),
        "threshold_frozen_before_reading": {
            "substantial_contrast_at": SUBSTANTIAL_CONTRAST,
            "basis": "fraction of decision-masked heads on which the teachers disagree, FIT"},
        "splits": report,
        "VERDICT": verdict,
        "reading": reading,
        "consequence": {
            "substantial": "freeze paired-latent rehearsal + expanded bank; fresh 1M final attempt",
            "insufficient": "stop latent work; pivot to the SAPPO-centred ICRA paper"},
        "does_not_alter": "the frozen V1 NOT_CONFIRMED result",
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")
    print(f"  VERDICT: {verdict}")
    print(f"  {reading}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
