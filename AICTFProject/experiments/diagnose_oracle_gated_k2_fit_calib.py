"""FIT vs CALIB latent-separation diagnostic for oracle-gated K=2 V1.

Measures whether z0/z1 became different on the rehearsal distribution but failed
to generalize (memorization) or never separated even on FIT (interference).

Uses the frozen V1 terminal checkpoint only. FIT and CALIB branch states from the
stratified collection shards; EVAL is never opened.

Metrics (resolvable states only, Delta != 0):
  - z0 argmax agreement with pi_A on A-preferred states (Delta < 0)
  - z1 argmax agreement with pi_B on B-preferred states (Delta > 0)
  - mean JSD between z0 and z1 action distributions
  - breakdown by pole, regime, horizon (16 cells)

This is a DIAGNOSTIC. It does not alter the frozen V1 verdict and defines no gate.

Run:  python experiments/diagnose_oracle_gated_k2_fit_calib.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
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
DATA = SD / "stratified_regime_data"
FROZEN = SD / "sppo" / "ORACLE_GATED_K2_MODEL_FROZEN.json"
OUT = SD / "sppo" / "ORACLE_GATED_K2_FIT_CALIB_DIAGNOSTIC.json"

FIT_LO, FIT_HI = 10_700_001, 10_700_096
CALIB_LO, CALIB_HI = 10_700_097, 10_700_128
EVAL_LO = 10_700_129
BATCH = 128
CELL_RE = re.compile(r"^([AB])_r(\d)_(not_late|late)$")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(lo: int, hi: int) -> dict:
    """Resolvable branch states in [lo, hi] with teacher actions."""
    keys = ("grid", "vec", "amask", "mask", "pi_a", "pi_b", "pole", "cell", "delta", "seed")
    buckets: dict[str, list] = {k: [] for k in keys}
    for s in range(lo, hi + 1):
        path = DATA / "seed_shards" / f"seed_{s}.npz"
        with np.load(path, allow_pickle=False) as z:
            d = ((z["branch_pi_B_blue"].astype(np.int64) - z["branch_pi_B_red"].astype(np.int64))
                 - (z["branch_pi_A_blue"].astype(np.int64) - z["branch_pi_A_red"].astype(np.int64)))
            keep = np.nonzero(d != 0)[0]
            if not len(keep):
                continue
            buckets["grid"].append(z["branch_obs_grid"][keep][:, 0])
            buckets["vec"].append(z["branch_obs_vec"][keep][:, 0])
            buckets["amask"].append(z["branch_obs_agent_mask"][keep][:, 0])
            buckets["mask"].append(z["branch_obs_mask"][keep][:, 0])
            buckets["pi_a"].append(z["branch_pi_A_action"][keep])
            buckets["pi_b"].append(z["branch_pi_B_action"][keep])
            buckets["pole"].append(z["branch_pole"][keep].astype(np.int64))
            buckets["cell"].extend(str(c) for c in z["branch_cell"][keep])
            buckets["delta"].append(d[keep])
            buckets["seed"].append(np.full(len(keep), s, dtype=np.int64))
    if not buckets["grid"]:
        raise SystemExit(f"REFUSING: no resolvable states in [{lo}, {hi}]")
    out = {k: (np.concatenate(v) if k != "cell" else np.array(v)) for k, v in buckets.items()}
    out["a_preferred"] = out["delta"] < 0
    out["b_preferred"] = out["delta"] > 0
    return out


def _obs_batch(split: dict, idx: np.ndarray, device: str) -> dict:
    import torch
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return {
        "grid": t(split["grid"][idx], torch.float32),
        "vec": t(split["vec"][idx], torch.float32),
        "agent_mask": t(split["amask"][idx], torch.float32),
        "mask": t(split["mask"][idx], torch.float32),
    }


def _agreement(model, obs, actions, z_val: int, device: str) -> np.ndarray:
    """Per-state head agreement fraction (decision-masked)."""
    import torch
    from rl.custom_ppo.strategy_anchor import _masked_heads

    with torch.no_grad():
        z = torch.full((obs["grid"].shape[0],), z_val, dtype=torch.long, device=device)
        heads = _masked_heads(model, obs, z_idx=z)
        acts = torch.as_tensor(actions, dtype=torch.long, device=device).reshape(actions.shape[0], -1)
        hits = []
        for h, head in enumerate(heads):
            hits.append((head.argmax_actions == acts[:, h]).float())
        per_head = torch.stack(hits, dim=1)
        n_agents = obs["agent_mask"].shape[1]
        n_heads = per_head.shape[1]
        m = obs["agent_mask"].bool().repeat_interleave(n_heads // n_agents, dim=1)
        return (per_head * m.float()).sum(dim=1) / m.float().sum(dim=1).clamp_min(1.0)


def _jsd_mean(model, obs, device: str) -> np.ndarray:
    """Per-state mean JSD across heads between z0 and z1."""
    import torch
    from rl.custom_ppo.diagnostics.counterfactual import _jsd_from_logits
    from rl.custom_ppo.strategy_anchor import _masked_heads

    with torch.no_grad():
        n = obs["grid"].shape[0]
        z0 = torch.zeros(n, dtype=torch.long, device=device)
        z1 = torch.ones(n, dtype=torch.long, device=device)
        h0 = _masked_heads(model, obs, z_idx=z0)
        h1 = _masked_heads(model, obs, z_idx=z1)
        jsds = []
        for a, b in zip(h0, h1):
            jsds.append(_jsd_from_logits(a.logits, b.logits))
        return torch.stack(jsds, dim=1).mean(dim=1).cpu().numpy()


def score_split(model, split: dict, device: str) -> dict:
    n = len(split["delta"])
    agree_z0_a = np.zeros(n, dtype=np.float64)
    agree_z1_b = np.zeros(n, dtype=np.float64)
    jsd = np.zeros(n, dtype=np.float64)
    for start in range(0, n, BATCH):
        idx = np.arange(start, min(start + BATCH, n))
        obs = _obs_batch(split, idx, device)
        agree_z0_a[idx] = _agreement(model, obs, split["pi_a"][idx], 0, device).cpu().numpy()
        agree_z1_b[idx] = _agreement(model, obs, split["pi_b"][idx], 1, device).cpu().numpy()
        jsd[idx] = _jsd_mean(model, obs, device)

    def _summ(mask: np.ndarray, values: np.ndarray) -> dict:
        if not mask.any():
            return {"n": 0, "mean": None}
        v = values[mask]
        return {"n": int(mask.sum()), "mean": float(v.mean())}

    by_cell: dict[str, dict] = {}
    for cell in np.unique(split["cell"]):
        m = split["cell"] == cell
        by_cell[str(cell)] = {
            "n_resolvable": int(m.sum()),
            "z0_match_pi_A_on_A_pref": _summ(m & split["a_preferred"], agree_z0_a),
            "z1_match_pi_B_on_B_pref": _summ(m & split["b_preferred"], agree_z1_b),
            "z0_z1_jsd": _summ(m, jsd),
        }

    by_pole: dict[str, dict] = {}
    for pole_id, name in ((0, "A"), (1, "B")):
        m = split["pole"] == pole_id
        by_pole[name] = {
            "z0_match_pi_A_on_A_pref": _summ(m & split["a_preferred"], agree_z0_a),
            "z1_match_pi_B_on_B_pref": _summ(m & split["b_preferred"], agree_z1_b),
            "z0_z1_jsd": _summ(m, jsd),
        }

    by_regime: dict[str, dict] = {}
    by_horizon: dict[str, dict] = {}
    for cell, block in by_cell.items():
        m = CELL_RE.match(cell)
        if not m:
            continue
        regime, horizon = f"r{m.group(2)}", m.group(3)
        for bucket, key in ((by_regime, regime), (by_horizon, horizon)):
            if key not in bucket:
                bucket[key] = defaultdict(lambda: {"n": 0, "sum": 0.0})
            for metric in ("z0_match_pi_A_on_A_pref", "z1_match_pi_B_on_B_pref", "z0_z1_jsd"):
                sub = block[metric]
                if sub["n"]:
                    bucket[key][metric]["n"] += sub["n"]
                    bucket[key][metric]["sum"] += sub["mean"] * sub["n"]

    def _finalize(bucket: dict) -> dict:
        out = {}
        for key, metrics in bucket.items():
            out[key] = {}
            for metric, acc in metrics.items():
                if acc["n"]:
                    out[key][metric] = {"n": acc["n"], "mean": acc["sum"] / acc["n"]}
                else:
                    out[key][metric] = {"n": 0, "mean": None}
        return out

    return {
        "n_resolvable": n,
        "z0_match_pi_A_on_A_pref": _summ(split["a_preferred"], agree_z0_a),
        "z1_match_pi_B_on_B_pref": _summ(split["b_preferred"], agree_z1_b),
        "z0_z1_jsd_mean": {"n": n, "mean": float(jsd.mean())},
        "by_pole": by_pole,
        "by_regime": _finalize(by_regime),
        "by_horizon": _finalize(by_horizon),
        "by_cell": by_cell,
    }


def classify_fork(fit: dict, calib: dict) -> dict:
    """Fork from FIT vs CALIB specialist teacher-match (primary) and JSD (secondary)."""
    def _m(block: dict, key: str) -> float | None:
        return block[key]["mean"]

    z0_fit = _m(fit, "z0_match_pi_A_on_A_pref")
    z1_fit = _m(fit, "z1_match_pi_B_on_B_pref")
    z0_cal = _m(calib, "z0_match_pi_A_on_A_pref")
    z1_cal = _m(calib, "z1_match_pi_B_on_B_pref")
    jsd_fit = _m(fit, "z0_z1_jsd_mean")
    jsd_cal = _m(calib, "z0_z1_jsd_mean")

    fit_floor = min(z0_fit or 0.0, z1_fit or 0.0)
    cal_floor = min(z0_cal or 0.0, z1_cal or 0.0)
    fit_strong = fit_floor >= 0.70
    cal_weaker = cal_floor < fit_floor - 0.10
    fit_weak = fit_floor < 0.50

    if fit_strong and cal_weaker:
        fork = "MEMORIZATION_GENERALIZATION"
        v2 = "larger_diverse_oracle_bank_same_1M"
    elif fit_weak:
        fork = "LATENT_INTERFERENCE_COLLAPSE"
        v2 = "stronger_latent_isolation_same_1M"
    else:
        fork = "MIXED_OR_BORDERLINE"
        v2 = "PI_REVIEW_REQUIRED"

    return {
        "z0_match_pi_A_on_A_pref": {"FIT": z0_fit, "CALIB": z0_cal},
        "z1_match_pi_B_on_B_pref": {"FIT": z1_fit, "CALIB": z1_cal},
        "z0_z1_jsd": {"FIT": jsd_fit, "CALIB": jsd_cal},
        "fit_specialist_floor": fit_floor,
        "calib_specialist_floor": cal_floor,
        "fit_strong_by_teacher_match": fit_strong,
        "calib_weaker_than_fit": cal_weaker,
        "fit_weak_by_teacher_match": fit_weak,
        "recommended_fork": fork,
        "recommended_v2_lever": v2,
        "rule": (
            "strong FIT teacher-match + weaker CALIB teacher-match -> memorization; "
            "weak FIT teacher-match -> interference; JSD reported separately"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; this diagnostic is one-shot")
    if not (DATA / "COLLECTION_COMPLETE.json").is_file():
        raise SystemExit("REFUSING: stratified collection is not complete")

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    ck_path = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    expected = frozen["TERMINAL_CHECKPOINT"]["sha256"]
    actual = _sha256(ck_path)
    if actual != expected:
        raise SystemExit(f"REFUSING: checkpoint sha mismatch\n  expected {expected}\n  actual   {actual}")

    import torch
    import experiments.r2_learned_crossover as R2
    from rl.custom_ppo import load_custom_ppo_policy

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    probe = R2.build_env(device, FIT_LO)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policy = load_custom_ppo_policy(str(ck_path), obs_space, act_space, device=device)
    model = policy.model if hasattr(policy, "model") else policy
    model.eval()

    print(f"ORACLE-GATED K=2 FIT/CALIB DIAGNOSTIC  {_now()}")
    print(f"  checkpoint sha256 verified")
    print(f"  FIT   {FIT_LO}..{FIT_HI}")
    print(f"  CALIB {CALIB_LO}..{CALIB_HI}")
    print(f"  EVAL  {EVAL_LO}+ sealed\n", flush=True)

    fit = load_split(FIT_LO, FIT_HI)
    calib = load_split(CALIB_LO, CALIB_HI)
    fit_scores = score_split(model, fit, device)
    calib_scores = score_split(model, calib, device)
    fork = classify_fork(fit_scores, calib_scores)

    record = {
        "record": "Oracle-gated K=2 V1 FIT vs CALIB latent-separation diagnostic",
        "status": "FROZEN_RESULT",
        "one_shot": True,
        "utc": _now(),
        "checkpoint": {"path": str(ck_path.relative_to(ROOT)), "sha256": actual},
        "v1_verdict_unchanged": "ORACLE_GATED_K2_CROSSOVER_NOT_CONFIRMED",
        "FIT": fit_scores,
        "CALIB": calib_scores,
        "fork": fork,
        "consequence": (
            "Diagnostic only. V2 lever selection follows the recommended_fork; "
            "V2 training is not authorized until a fresh V2 run spec is frozen."
        ),
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"  FIT   z0->pi_A on A-pref  {fit_scores['z0_match_pi_A_on_A_pref']}")
    print(f"  FIT   z1->pi_B on B-pref  {fit_scores['z1_match_pi_B_on_B_pref']}")
    print(f"  FIT   z0/z1 JSD           {fit_scores['z0_z1_jsd_mean']}")
    print(f"  CALIB z0->pi_A on A-pref  {calib_scores['z0_match_pi_A_on_A_pref']}")
    print(f"  CALIB z1->pi_B on B-pref  {calib_scores['z1_match_pi_B_on_B_pref']}")
    print(f"  CALIB z0/z1 JSD           {calib_scores['z0_z1_jsd_mean']}")
    print(f"\n  FORK: {fork['recommended_fork']}")
    print(f"  V2 lever: {fork['recommended_v2_lever']}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
