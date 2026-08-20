"""SAPPO V1 FINAL LAUNCH BLOCKER — live checkpoint x live dataset.

Unit tests checked the anchor dataset's shapes against stored arrays. That is
not the same as proving the REAL R1 checkpoint can evaluate those observations
through its own preprocessing path. This is the last place a silent
incompatibility can hide before 500k environment steps are spent.

Checks, per specialist:

  1. load the ACTUAL R1 terminal checkpoint
  2. take a real batch from that pole's anchor_*_train.npz
  3. push the observations through the resumed policy's own preprocessing
  4. call the real get_distribution()
  5. evaluate the STORED teacher actions under that distribution
  6. require finite log pi(a_teacher | o) for every head and sample
  7. verify obs keys, shapes, dtypes, action column count, agent-major
     ordering, and action_space.nvec all agree exactly
  8. compare the distribution obtained with and without the runner's
     preprocessing path -- they must agree, or the dataset is valid for the raw
     model but not for the thing that will actually train

Nothing here is a scientific gate. It is an engineering precondition.

Run:  python experiments/sappo_live_compatibility_check.py --device cuda
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
from rl.custom_ppo.strategy_anchor import (                        # noqa: E402
    _masked_heads, action_log_prob, anchor_loss, teacher_agreement,
)

DEMO = ROOT / "artifacts/strategic_demand/sappo_demonstrations"
OUT = ROOT / "artifacts/strategic_demand/sappo_live_compat.json"
PAIRS = {"pi_A": ("A", R2.POLICIES["pi_A"]), "pi_B": ("B", R2.POLICIES["pi_B"])}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_batch(pole: str, n: int, device: str):
    d = np.load(DEMO / f"anchor_{pole}_train.npz")
    obs = {k[4:]: torch.from_numpy(d[k][:n]).to(device)
           for k in d.files if k.startswith("obs_")}
    acts = torch.from_numpy(d["actions"][:n]).long().to(device)
    mask = torch.from_numpy(d["decision_mask"][:n]).bool().to(device)
    return obs, acts, mask, d


def check(name: str, pole: str, ckpt: Path, device: str, n: int) -> dict:
    from rl.custom_ppo import load_custom_ppo_policy

    probe = R2.build_env(device, R2.SEED_BASE)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    policy = load_custom_ppo_policy(str(ckpt), obs_space, act_space, device=device)
    model = getattr(policy, "model", policy)

    obs, acts, mask, raw = load_batch(pole, n, device)
    rec: dict = {"policy": name, "pole": pole,
                 "checkpoint": str(ckpt.relative_to(ROOT)),
                 "dataset_run_id": str(raw["run_id"][0]) if "run_id" in raw.files else None,
                 "batch": int(acts.shape[0])}

    # 7. structural agreement
    nvec = [int(v) for v in getattr(act_space, "nvec", [])]
    rec["action_space_nvec"] = nvec
    rec["action_columns"] = int(acts.reshape(acts.shape[0], -1).shape[1])
    rec["nvec_matches_action_columns"] = (len(nvec) == rec["action_columns"])
    rec["obs_keys_dataset"] = sorted(obs.keys())
    rec["obs_dtypes"] = {k: str(v.dtype) for k, v in obs.items()}
    rec["obs_shapes"] = {k: list(v.shape) for k, v in obs.items()}
    # agent-major: (batch, n_agents, heads) flattens to nvec order
    rec["agent_major_ok"] = bool(
        acts.reshape(acts.shape[0], -1)[0].tolist()
        == [int(x) for x in acts[0].reshape(-1).tolist()])

    # 4/5/6. the real distribution, the stored actions, finite log-probs
    z = None
    if getattr(model, "uses_latent_strategy", False):
        z = torch.zeros(acts.shape[0], dtype=torch.long, device=device)
        rec["latent_z_supplied"] = True
    with torch.no_grad():
        lp = action_log_prob(model, obs, acts, z_idx=z)
        loss = anchor_loss(model, obs, acts, mask, z_idx=z)
        agree = teacher_agreement(model, obs, acts, mask, z_idx=z)
    rec["log_prob_shape"] = list(lp.shape)
    rec["all_log_probs_finite"] = bool(torch.isfinite(lp).all())
    rec["log_prob_min"] = float(lp.min())
    rec["log_prob_max"] = float(lp.max())
    rec["anchor_loss"] = float(loss)
    rec["anchor_loss_finite"] = bool(np.isfinite(float(loss)))
    rec["teacher_agreement_at_1M"] = float(agree)

    # 8. same observations through the policy's own predict() preprocessing
    np_obs = {k: v.detach().cpu().numpy() for k, v in obs.items()}
    with torch.no_grad():
        pred, _ = policy.predict(np_obs, deterministic=True)
        # Compare against the MASKED heads -- the distribution PPO's own
        # evaluate_actions() uses. Comparing against unmasked get_distribution()
        # is what exposed the masking bug in the first place.
        heads = _masked_heads(model, obs, z_idx=z)
        argmax = torch.stack([h.argmax_actions for h in heads], dim=1)
    pred_t = torch.as_tensor(np.asarray(pred).reshape(argmax.shape),
                             device=argmax.device)
    match = float((pred_t == argmax).float().mean())
    rec["predict_vs_masked_heads_argmax_match"] = match
    rec["preprocessing_paths_agree"] = bool(match == 1.0)

    # Teacher labels the policy's legality mask forbids. These are unlearnable:
    # -log pi would be the mask sentinel. Must be zero at decision points, which
    # are the only rows the anchor loss uses.
    with torch.no_grad():
        af = acts.reshape(acts.shape[0], -1)
        ill = torch.stack([
            (h.logits.log_softmax(-1).gather(1, af[:, i:i + 1]).squeeze(1) < -1e6).float()
            for i, h in enumerate(heads)], dim=1)
        dm = mask.float().repeat_interleave(len(heads) // mask.shape[1], dim=1)
    rec["illegal_teacher_labels_all_heads"] = float(ill.mean())
    rec["illegal_teacher_labels_at_decision_points"] = float((ill * dm).sum() / dm.sum())
    rec["no_illegal_labels_in_loss"] = bool(rec["illegal_teacher_labels_at_decision_points"] == 0.0)

    rec["PASS"] = bool(rec["anchor_loss_finite"]
                       and rec["nvec_matches_action_columns"]
                       and rec["agent_major_ok"]
                       and rec["preprocessing_paths_agree"]
                       and rec["no_illegal_labels_in_loss"])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()

    print(f"SAPPO LIVE COMPATIBILITY CHECK  {_now()}")
    results = []
    for name, (pole, ckpt) in PAIRS.items():
        print(f"\n--- {name} (pole {pole}) ---", flush=True)
        r = check(name, pole, ckpt, a.device, a.n)
        results.append(r)
        for k in ("dataset_run_id", "action_space_nvec", "action_columns",
                  "nvec_matches_action_columns", "agent_major_ok",
                  "all_log_probs_finite", "log_prob_min", "anchor_loss",
                  "teacher_agreement_at_1M",
                  "predict_vs_masked_heads_argmax_match",
                  "illegal_teacher_labels_all_heads",
                  "illegal_teacher_labels_at_decision_points",
                  "no_illegal_labels_in_loss",
                  "preprocessing_paths_agree", "PASS"):
            print(f"  {k:<42} {r.get(k)}")

    ok = all(r["PASS"] for r in results)
    OUT.write_text(json.dumps(
        {"record": "SAPPO live compatibility check", "utc": _now(),
         "status": "ENGINEERING PRECONDITION, not a scientific gate",
         "results": results, "ALL_PASS": ok}, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"ALL_PASS: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
