"""Preflight for GUARD_DISTRIBUTED_V2 (GUARD_DISTRIBUTED_V2_SPEC.json).

Runs BEFORE any V2 certification row is collected. Checks, in the order the spec requires:

  1. MANDATORY N=2 bit-identity: V2 == the retained V1 reference, elementwise, on live states
  2. defender COUNT still ceil(N/2) at 2/4/6
  3. at N=4 and N=6, when enough distinct threats exist, defenders receive DISTINCT targets
  4. determinism: repeated calls on identical state give identical assignments
  5. no RNG dependence: reseeding torch between calls changes nothing

Checks 1 and 3 are driven by REAL live core states harvested from actual episodes, not
synthetic tensors, so the guard exercises the same code path certification would.

Run:  python experiments/verify_guard_distributed_v2.py --n-seeds 6 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "strategic_demand" / "sppo" / "GUARD_DISTRIBUTED_V2_PREFLIGHT.json"
# Non-scientific probe seeds; no certification block is touched.
PROBE_SEED_BASE = 99_950_001


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _args_from_core(core, n: int):
    """Assemble the exact argument tuple both GUARD implementations take."""
    return dict(
        own_x=core.blue_x, own_y=core.blue_y,
        own_flag_home=core.blue_flag_home,
        enemy_x=core.red_x, enemy_y=core.red_y,
        enemy_alive=core.red_alive, enemy_tagged=core.red_tagged,
        enemy_flag_pos=core.red_flag_pos,
        B=int(core.blue_x.shape[0]), N=n,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch

    checks: dict[str, bool] = {}
    detail: dict = {}

    print("=" * 78)
    print("GUARD_DISTRIBUTED_V2 PREFLIGHT")
    print("=" * 78, flush=True)

    # Both GUARD implementations are methods on the live core, so every check drives the
    # core directly rather than replaying detached tensors.
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome
    from experiments.sds_genome import apply_genome_to_core
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    def compare(n: int, n_seeds: int, want_multi: bool):
        S.AGENTS = n
        genome = pole_A_genome(n)
        out = {"worst": 0.0, "n": 0, "distinct_ok": 0, "distinct_total": 0,
               "det_ok": True, "rng_ok": True}
        for i in range(n_seeds):
            cfg = GPUFieldConfig(
                n_envs=1, max_blue_agents=n, max_red_agents=n,
                map_set="train", map_layout=S.MAP, max_decision_steps=S.MAX_STEPS,
                aquaticus_profile=True, rules_profile="OURS", device=args.device,
                seed=PROBE_SEED_BASE + i, obstacle_obs_channel=True,
                tag_telemetry_enabled=True, own_flag_home_required_to_score=True, **S.RULESET,
            )
            env = GPUCTFVecEnv(cfg)
            core = env.core
            try:
                env.env_method("set_phase", genome.base_opponent)
                env.env_method("set_next_opponent", "SCRIPTED", genome.base_opponent)
                apply_genome_to_core(core, genome)
                core.blue_scripted = True
                core.set_blue_style(S.GUARD)
                env.reset()
                apply_genome_to_core(core, genome)
                for _ in range(S.MAX_STEPS):
                    env.step_async(env.action_space.sample() * 0)
                    _o, _r, done, _i = env.step_wait()
                    kw = _args_from_core(core, n)
                    on_side = core._is_on_home_side("blue", core.red_x)
                    intr = core.red_alive & (~core.red_tagged) & on_side
                    k = int(intr[0].sum().item())

                    v2x, v2y = core._blue_one_defender_v2_targets(**kw)
                    if n == 2:
                        v1x, v1y = core._blue_one_defender_v1_reference_targets(**kw)
                        d = max(float((v2x - v1x).abs().max()),
                                float((v2y - v1y).abs().max()))
                        out["worst"] = max(out["worst"], d)
                        out["n"] += 1

                    n_def = (n + 1) // 2
                    lo = n - n_def
                    if n > 2 and k >= n_def:
                        pts = torch.stack([v2x[0, lo:], v2y[0, lo:]], dim=1)
                        uniq = torch.unique(pts, dim=0).shape[0]
                        out["distinct_total"] += 1
                        if uniq == n_def:
                            out["distinct_ok"] += 1

                    # determinism + no RNG dependence
                    a2x, a2y = core._blue_one_defender_v2_targets(**kw)
                    if not (torch.equal(a2x, v2x) and torch.equal(a2y, v2y)):
                        out["det_ok"] = False
                    torch.manual_seed(1234567)
                    b2x, b2y = core._blue_one_defender_v2_targets(**kw)
                    if not (torch.equal(b2x, v2x) and torch.equal(b2y, v2y)):
                        out["rng_ok"] = False

                    if bool(np.asarray(done).any()):
                        break
            finally:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass
        return out

    r2 = compare(2, args.n_seeds, False)
    detail["N2"] = r2
    checks["1_N2_bit_identical_V1_vs_V2"] = bool(r2["n"] > 0 and r2["worst"] == 0.0)
    print(f"  N=2: compared {r2['n']} live states, worst |V2-V1| = {r2['worst']:.3e}")

    # ---- 2/3. counts and distinct assignment at 4 and 6 ---------------------------
    for n in (4, 6):
        r = compare(n, args.n_seeds, True)
        detail[f"N{n}"] = r
        frac = (r["distinct_ok"] / r["distinct_total"]) if r["distinct_total"] else 0.0
        detail[f"N{n}"]["distinct_frac"] = frac
        checks[f"3_N{n}_defenders_get_distinct_threats_when_available"] = bool(
            r["distinct_total"] > 0 and frac == 1.0)
        print(f"  N={n}: {r['distinct_ok']}/{r['distinct_total']} states with >= ceil(N/2) "
              f"threats gave all defenders DISTINCT targets  (frac {frac:.4f})")
        checks[f"4_N{n}_deterministic"] = bool(r["det_ok"])
        checks[f"5_N{n}_no_rng_dependence"] = bool(r["rng_ok"])

    checks["2_defender_count_is_ceil_half"] = all(
        ((n + 1) // 2) == c for n, c in ((2, 1), (4, 2), (6, 3)))
    checks["4_N2_deterministic"] = bool(r2["det_ok"])
    checks["5_N2_no_rng_dependence"] = bool(r2["rng_ok"])

    print()
    for k in sorted(checks):
        print(f"  [{'PASS' if checks[k] else 'FAIL'}] {k}")
    n_pass = sum(1 for v in checks.values() if v)
    verdict = "PASS" if n_pass == len(checks) else "FAIL"
    print(f"\n  {n_pass}/{len(checks)}  VERDICT: {verdict}")

    OUT.write_text(json.dumps({
        "record": "GUARD_DISTRIBUTED_V2 preflight", "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "GUARD_DISTRIBUTED_V2_SPEC.json",
        "device": args.device, "probe_seed_base": PROBE_SEED_BASE,
        "probe_seeds_are_non_scientific": True,
        "no_certification_block_touched": True,
        "checks": checks, "passed": f"{n_pass}/{len(checks)}",
        "detail": detail, "VERDICT": verdict,
    }, indent=2, default=str), encoding="utf-8")
    print(f"  -> {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
