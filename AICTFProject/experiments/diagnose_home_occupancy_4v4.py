"""Read-only home-occupancy diagnostic for the qualified 4v4 specialists on Pole B.

DIAGNOSTIC under HOME_OCCUPANCY_DIAGNOSTIC_4V4_V1_SPEC.json (frozen before any
episode). It plays A3 and B2 natively (k = 0) on Pole B over the same 16 throwaway
seeds and, at every decision tick, counts how many agents sit in the home region.
No action is changed and nothing is trained. The episode path mirrors
eval_specialist_crossover_scaled.run_cell for a plain policy; the only addition is
a read-only state snapshot before each env step.

The home region is the learned-composition probe's frozen position instrument
(R_POS = 4.5 cells around own flag home), adopted unchanged.

    ./.venv/Scripts/python.exe experiments/diagnose_home_occupancy_4v4.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N = 4
POLE = "B"
KEY = "OP7"
SPEC_PATH = ROOT / "artifacts" / "strategic_demand" / "sppo" / "HOME_OCCUPANCY_DIAGNOSTIC_4V4_V1_SPEC.json"
OUT = ROOT / "artifacts" / "strategic_demand" / "sppo" / "home_occupancy_4v4"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def runs_over(mask: np.ndarray) -> list[int]:
    """Lengths of maximal runs of True in a 1-D boolean array."""
    out, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def summarize(episodes: list[np.ndarray], cap: int, sustain: int) -> dict:
    """Over-cap metrics from per-episode home_count arrays."""
    all_ticks = np.concatenate(episodes)
    periods = [r for ep in episodes for r in runs_over(ep > cap)]
    ever = sum(bool((ep > cap).any()) for ep in episodes)
    sustained = sum(any(r >= sustain for r in runs_over(ep > cap)) for ep in episodes)
    n_ep = len(episodes)
    return {
        "n_episodes": n_ep,
        "n_ticks": int(all_ticks.size),
        "pct_ticks_over_cap": float(100.0 * (all_ticks > cap).mean()),
        "n_episodes_ever_over_cap": int(ever),
        "pct_episodes_ever_over_cap": float(100.0 * ever / n_ep),
        "n_episodes_sustained_over_cap": int(sustained),
        "pct_episodes_sustained_over_cap": float(100.0 * sustained / n_ep),
        "over_cap_periods": {
            "count": len(periods),
            "mean_ticks": float(np.mean(periods)) if periods else 0.0,
            "median_ticks": float(np.median(periods)) if periods else 0.0,
            "max_ticks": int(max(periods)) if periods else 0,
        },
        "home_count_distribution": {
            str(c): float((all_ticks == c).mean()) for c in range(N + 1)
        },
    }


def decide(primary: dict[str, dict], rule: dict) -> str:
    go = any(m["pct_ticks_over_cap"] > rule["pct_ticks"] or
             m["pct_episodes_sustained_over_cap"] >= rule["pct_episodes_sustained"]
             for m in primary.values())
    return "GO" if go else "NO_GO"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN":
        raise SystemExit("REFUSING: spec is not FROZEN")
    result_path = OUT / "HOME_OCCUPANCY_DIAGNOSTIC_4V4_V1_RESULT.json"
    if result_path.exists():
        raise SystemExit(f"REFUSING: {result_path.name} already written (write-once)")
    lo, hi = spec["seeds"]["block"]
    seeds = list(range(lo, hi + 1))
    cap = int(spec["cap"])
    r_home = float(spec["home_region"]["radius_cells"])
    sustain = 8
    rule = {"pct_ticks": 5.0, "pct_episodes_sustained": 25.0}

    import torch
    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import assert_live_opponent_batch, install_keyed_opponent_overlays
    from experiments.pole_attestation import (
        assert_resolved_matches_certification, governing_certification, resolve_pole_genome,
    )
    from experiments.probe_learned_composition import _snap
    from gpu_env._core._entity_obs import augment_obs_with_entities
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    R2.AGENTS = N
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    paths = {}
    for name, ent in spec["policies"].items():
        p = ROOT / ent["path"]
        got = _sha(p)
        if got != ent["sha256"]:
            raise SystemExit(f"FAIL-CLOSED: {name} sha {got[:12]} != frozen {ent['sha256'][:12]}")
        paths[name] = p

    _verdict, cert_path = governing_certification(N)
    genome = resolve_pole_genome(POLE, N, None)
    assert_resolved_matches_certification(POLE, N, cert_path, genome, is_smoke=False)
    genomes = {KEY: genome}
    print(f"HOME OCCUPANCY DIAGNOSTIC 4v4  pole {POLE} ({KEY}, attested vs {cert_path.name})")
    print(f"  seeds {lo}..{hi}  cap {cap}  r_home {r_home}  device {device}", flush=True)

    probe = R2.build_env(device, seeds[0])
    osp, asp = probe.observation_space, probe.action_space
    probe.close()
    policies = {n: load_custom_ppo_policy(str(p), osp, asp, device=device) for n, p in paths.items()}
    for n, pol in policies.items():
        if getattr(pol.model, "uses_latent_strategy", False) or \
                getattr(pol.model, "role_conditioning_enabled", False):
            raise SystemExit(f"REFUSING: {n} must be a plain single-strategy, non-role specialist")

    OUT.mkdir(parents=True, exist_ok=True)
    rows_path = OUT / "home_occupancy_4v4_ticks.csv"
    per_policy: dict[str, dict[str, list]] = {n: {"primary": [], "raw": []} for n in policies}
    terminals: dict[str, list] = {n: [] for n in policies}
    with rows_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "seed", "tick", "home_count", "home_count_raw", "n_active"])
        for name, policy in policies.items():
            for seed in seeds:
                env = R2.build_env(device, seed)
                core = env.core
                prim, raw = [], []
                terminal = None
                try:
                    policy.reset_strategy()
                    core._bt_profile_override = None
                    core._sds_opening_hold_steps = 0
                    install_keyed_opponent_overlays(core, genomes)
                    env.env_method("set_phase", phase_from_tag(KEY))
                    env.env_method("set_next_opponent", "SCRIPTED", KEY)
                    obs = env.reset()
                    obs["global_state"] = env.state()
                    obs = augment_obs_with_entities(obs, core, side="blue")
                    assert_live_opponent_batch(core, genomes, allowed_keys=(KEY,),
                                               context=f"home occupancy {name} seed {seed}")
                    got = core._bt_resolved_profile_tensors().get("min_alive_for_defender")
                    got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
                    if got_val != N:
                        raise SystemExit(f"FAIL-CLOSED: min_alive_for_defender={got_val}, expected {N}")
                    for t in range(R2.MAX_STEPS):
                        action, _ = policy.predict(obs, deterministic=True)
                        s = _snap(core)
                        d = np.linalg.norm(s["pos"] - s["flag_home"][None, :], axis=-1)
                        active = s["alive"] & ~s["tagged"]
                        k = int((active & ~s["carrying"] & (d <= r_home)).sum())
                        k_raw = int((s["alive"] & (d <= r_home)).sum())
                        prim.append(k)
                        raw.append(k_raw)
                        w.writerow([name, seed, t, k, k_raw, int(active.sum())])
                        env.step_async(action)
                        obs, _r, done, info = env.step_wait()
                        obs["global_state"] = env.state()
                        obs = augment_obs_with_entities(obs, core, side="blue")
                        if bool(np.asarray(done).any()):
                            i0 = info[0] if isinstance(info, (list, tuple)) else info
                            res = (i0 or {}).get("episode_result") or {}
                            terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                            break
                    if terminal is None:
                        terminal = (int(core.blue_score[0]), int(core.red_score[0]))
                finally:
                    env.close()
                per_policy[name]["primary"].append(np.asarray(prim))
                per_policy[name]["raw"].append(np.asarray(raw))
                terminals[name].append({"seed": seed, "blue": terminal[0], "red": terminal[1],
                                        "ticks": len(prim)})
                ep = np.asarray(prim)
                print(f"  {name} seed {seed}: ticks {len(prim)}  max home {ep.max()}  "
                      f"over-cap ticks {int((ep > cap).sum())}", flush=True)

    primary = {n: summarize(v["primary"], cap, sustain) for n, v in per_policy.items()}
    raw = {n: summarize(v["raw"], cap, sustain) for n, v in per_policy.items()}
    verdict = decide(primary, rule)
    result = {
        "id": spec["id"],
        "status": "SEALED",
        "spec": SPEC_PATH.name,
        "spec_sha256": _sha(SPEC_PATH),
        "device": device,
        "seeds": [lo, hi],
        "cap": cap,
        "r_home": r_home,
        "sustain_ticks": sustain,
        "decision_rule": rule,
        "decision": verdict,
        "primary": primary,
        "raw_secondary_transparency_only": raw,
        "terminals_traceability_only_not_evidence": terminals,
        "rows_csv": rows_path.name,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nDECISION:", verdict)
    print(json.dumps(primary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
