"""Rung 0 equivalence preflight -- the seal does not open unless this is bit-exact.

Implements RUNG0_INTERPRETATION_AMENDMENT.json#MANDATORY_EQUIVALENCE_PREFLIGHT_BEFORE_THE_SEAL_OPENS.

Verifies wrapper(z0) == pi_A and wrapper(z1) == pi_B on fixed NON-SEALED states, on CUDA (the
device the sealed ladder runs on -- device-dependence was established in
ONPOLICY_TD_DEVICE_AMENDMENT.json, so a cpu check would not certify the cuda instrument):

    masked logits bit-exact, argmax equal on every head, MACRO heads (0,2) and TARGET heads
    (1,3) checked separately, both specialist checkpoint sha256 verified against the frozen
    SPECIALIST_BASELINE_SPEC.json hashes, and the forced-z path exercised through the SAME
    predict() interface the sealed evaluator uses rather than a bespoke call path.

Run:  python experiments/rung0_equivalence_preflight.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LADDER = SD / "SHARING_LADDER_SPEC.json"
AMENDMENT = SD / "RUNG0_INTERPRETATION_AMENDMENT.json"
SPECIALISTS = SD / "SPECIALIST_BASELINE_SPEC.json"
DATASET = SD / "TEACHER_DISTILLATION_DATASET.json"
OUT = SD / "RUNG0_EQUIVALENCE_PREFLIGHT.json"

N_STATES = 256
OBS_KEYS = ("grid", "vec", "agent_mask", "mask", "global_state")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ladder = json.loads(LADDER.read_text(encoding="utf-8"))
    if ladder["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: ladder spec not frozen: {ladder['status']!r}")
    amend = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if amend["status"] != "FROZEN_APPEND_ONLY_AMENDMENT":
        raise SystemExit("REFUSING: Rung-0 interpretation amendment not frozen")
    if args.device != "cuda":
        raise SystemExit("REFUSING: the amendment requires this preflight on cuda, the device "
                         "the sealed ladder uses")

    import torch
    import experiments.r2_learned_crossover as R2
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.rung0_dispatch import Rung0DispatchPolicy, verify_equivalence

    if not torch.cuda.is_available():
        raise SystemExit("REFUSING: cuda unavailable")
    device = args.device

    tspec = json.loads(SPECIALISTS.read_text(encoding="utf-8"))["MODELS_UNDER_TEST"]
    paths, shas = {}, {}
    for n in ("pi_A", "pi_B"):
        p = ROOT / tspec[n]["path"]
        got = _sha(p)
        if got != tspec[n]["sha256"]:
            raise SystemExit(f"REFUSING: {n} sha mismatch against SPECIALIST_BASELINE_SPEC.json")
        paths[n], shas[n] = p, got

    probe = R2.build_env(device, 11_960_001)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    specialists = {n: load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
                   for n, p in paths.items()}
    wrapper = Rung0DispatchPolicy(specialists["pi_A"], specialists["pi_B"])

    # non-sealed states already on disk; no new rollouts, no sealed seeds
    man = json.loads(DATASET.read_text(encoding="utf-8"))
    rows = {k: [] for k in OBS_KEYS}
    got = 0
    for sh in man["shards"]:
        d = np.load(ROOT / sh["file"])
        if int(d["step"].shape[0]) == 0:
            continue
        take = min(N_STATES - got, int(d["step"].shape[0]))
        for k in OBS_KEYS:
            rows[k].append(d[k][:take])
        got += take
        if got >= N_STATES:
            break
    obs = {k: torch.as_tensor(np.concatenate(v, axis=0), dtype=torch.float32, device=device)
           for k, v in rows.items()}

    print(f"RUNG 0 EQUIVALENCE PREFLIGHT  {_now()}  device={device}")
    print(f"  pi_A sha {shas['pi_A'][:12]}...  pi_B sha {shas['pi_B'][:12]}...  (both VERIFIED)")
    print(f"  {got} non-sealed states from TEACHER_DISTILLATION_DATASET.json\n", flush=True)

    report = verify_equivalence(wrapper, obs, device=device)

    # exercise the SAME interface the sealed evaluator uses, not a bespoke path
    obs_np = {k: v.detach().cpu().numpy() for k, v in obs.items()}
    single = {k: v[:1] for k, v in obs_np.items()}
    iface = {}
    for z, name in ((0, "pi_A"), (1, "pi_B")):
        wrapper.fixed_latent_strategy = True
        wrapper.fixed_latent_strategy_id = z
        wrapper.reset_strategy()
        a_w, _ = wrapper.predict(single, deterministic=True)
        specialists[name].reset_strategy()
        a_s, _ = specialists[name].predict(single, deterministic=True)
        same = [int(x) for x in np.asarray(a_w).ravel()] == [int(x) for x in np.asarray(a_s).ravel()]
        iface[f"z{z}_predict_matches_{name}"] = bool(same)
        iface[f"z{z}_action"] = [int(x) for x in np.asarray(a_w).ravel()]

    for k, v in report.items():
        if k.startswith("z"):
            print(f"  {k}: max|dlogit|={v['max_abs_logit_delta']:.3e}  bit_exact={v['logits_bit_exact']}  "
                  f"macro_heads_equal={v['macro_heads_equal']}  target_heads_equal={v['target_heads_equal']}  "
                  f"per_head_argmax={v['argmax_equal_per_head']}")
    for k, v in iface.items():
        if k.endswith("pi_A") or k.endswith("pi_B"):
            print(f"  {k}: {v}")

    ok = bool(report["ALL_EXACT"]) and all(v for k, v in iface.items() if isinstance(v, bool))
    print(f"\n  VERDICT: {'PASS -- Rung 0 seal may open' if ok else 'FAIL -- seal stays shut'}")

    OUT.write_text(json.dumps({
        "record": "Rung 0 equivalence preflight", "status": "FROZEN_RESULT", "utc": _now(),
        "device": device,
        "implements": "RUNG0_INTERPRETATION_AMENDMENT.json#MANDATORY_EQUIVALENCE_PREFLIGHT_BEFORE_THE_SEAL_OPENS",
        "specialists": {n: {"path": tspec[n]["path"], "sha256": shas[n]} for n in ("pi_A", "pi_B")},
        "n_states_checked": got, "states_source": "TEACHER_DISTILLATION_DATASET.json (non-sealed)",
        "logit_and_head_equivalence": report,
        "predict_interface_parity": iface,
        "VERDICT": "PASS" if ok else "FAIL",
        "rule": "a wrapper that is not bit-exact is not a positive control; on FAIL the Rung 0 seal does not open",
    }, indent=2), encoding="utf-8")
    print(f"  -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
