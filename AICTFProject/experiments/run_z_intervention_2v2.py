"""Z_INTERVENTION_2V2_V1: does changing only z steer Fully Shared+z+r toward the matching teacher?

Frozen spec: artifacts/strategic_demand/sppo/2v2_strengthening/Z_INTERVENTION_2V2_V1_SPEC.json

Probe states come from the student's own natural episodes (z in {A,B} x pole in {A,B}, same
seeds). At every decision-boundary agent-state the student's logits are computed twice on the
IDENTICAL observation tensors, once with z = A (z_idx 0) and once with z = B (z_idx 1), and the
role-gated dual-branch teachers are queried on the same tensors -- the exact objects and the
exact distance (forward KL per head over legality-masked logits, rl/teacher_distillation) the
Stage-4 distillation used. Nothing here changes an action.

    C_A = d(T_A, z=B) - d(T_A, z=A)      C_B = d(T_B, z=A) - d(T_B, z=B)

averaged per probe seed (the statistical unit), 95% bootstrap over seeds.

    .venv/Scripts/python.exe experiments/run_z_intervention_2v2.py --smoke
    .venv/Scripts/python.exe experiments/run_z_intervention_2v2.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.strengthening_2v2_common as C  # noqa: E402

SPEC_ID = "Z_INTERVENTION_2V2_V1"
BEHAVIOR_SPEC_ID = "BEHAVIOR_SIGNATURES_2V2_V1"   # teacher pins live there
SMOKE_SEEDS = (99972001, 99972002)
OBS_KEYS = ("grid", "vec", "agent_mask", "mask", "global_state")
OPTIONAL_KEYS = ("teammates", "teammates_valid", "enemies", "enemies_valid", "roles", "assignment")
FIELDS = ("seed", "collect_z", "pole", "tick", "agent", "role", "dA_z0", "dA_z1", "dB_z0", "dB_z1",
          "jsd_z0_z1", "argmax_flip", "kl_TA_TB", "kl_TB_TA")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_torch(obs: dict, device: str) -> dict:
    """run_teacher_distillation.to_torch conventions, applied to one live (batch-1) obs dict."""
    import torch
    out = {k: torch.as_tensor(np.asarray(obs[k]), dtype=torch.float32, device=device) for k in OBS_KEYS}
    for k in OPTIONAL_KEYS:
        if k in obs:
            dt = torch.bool if k.endswith("_valid") else torch.float32
            out[k] = torch.as_tensor(np.asarray(obs[k]), dtype=dt, device=device)
    return out


def per_agent(per_head, n_agents: int) -> np.ndarray:
    """(1, n_heads) -> (n_agents,) summed over each agent's heads."""
    v = per_head.detach().cpu().numpy()[0]
    return v.reshape(n_agents, -1).sum(axis=1)


def probe_state(student_model, teachers: dict, obs_t: dict, decided: np.ndarray) -> list[dict]:
    import torch
    from rl.teacher_distillation import dual_branch_composite_logits, head_logits, jsd_per_head, kl_per_head
    n = int(decided.shape[0])
    z0 = torch.zeros((1,), dtype=torch.long, device=obs_t["grid"].device)
    z1 = torch.ones((1,), dtype=torch.long, device=obs_t["grid"].device)
    with torch.no_grad():
        ls0 = head_logits(student_model, obs_t, z_idx=z0)
        ls1 = head_logits(student_model, obs_t, z_idx=z1)
        ta = dual_branch_composite_logits(teachers["A"]["defend"], teachers["A"]["attack"], obs_t)
        tb = dual_branch_composite_logits(teachers["B"]["defend"], teachers["B"]["attack"], obs_t)
        vals = {
            "dA_z0": per_agent(kl_per_head(ta, ls0), n), "dA_z1": per_agent(kl_per_head(ta, ls1), n),
            "dB_z0": per_agent(kl_per_head(tb, ls0), n), "dB_z1": per_agent(kl_per_head(tb, ls1), n),
            "jsd_z0_z1": per_agent(jsd_per_head(ls0, ls1), n),
            "kl_TA_TB": per_agent(kl_per_head(ta, tb), n), "kl_TB_TA": per_agent(kl_per_head(tb, ta), n),
        }
        hpa = len(ls0) // n
        flip = np.zeros(n, dtype=bool)
        for i, (a, b) in enumerate(zip(ls0, ls1)):
            if bool((a.argmax(-1) != b.argmax(-1)).any()):
                flip[i // hpa] = True
    roles = np.asarray(obs_t["roles"].detach().cpu())[0]
    rows = []
    for i in range(n):
        if decided[i]:
            rows.append({"agent": i, "role": "DEFEND" if roles[i] < 0.5 else "ATTACK",
                         "argmax_flip": int(flip[i]), **{k: float(v[i]) for k, v in vals.items()}})
    return rows


def analyse(rows: list[dict]) -> dict:
    def seed_means(sub, fn):
        by = {}
        for r in sub:
            by.setdefault(int(r["seed"]), []).append(fn(r))
        return np.asarray([np.mean(v) for _s, v in sorted(by.items())])

    ca = lambda r: r["dA_z1"] - r["dA_z0"]   # noqa: E731
    cb = lambda r: r["dB_z0"] - r["dB_z1"]   # noqa: E731
    out = {"primary": {}, "secondary": {}, "breakdown": {}}
    out["primary"]["C_A"] = C.mean_ci(seed_means(rows, ca))
    out["primary"]["C_B"] = C.mean_ci(seed_means(rows, cb))
    out["primary"]["z_steers_toward_matching_teacher"] = bool(
        out["primary"]["C_A"]["lcb95"] > 0 and out["primary"]["C_B"]["lcb95"] > 0)
    for k in ("jsd_z0_z1", "argmax_flip", "kl_TA_TB", "kl_TB_TA"):
        out["secondary"][k] = C.mean_ci(seed_means(rows, lambda r, k=k: r[k]))
    for z in ("A", "B"):
        for pole in ("A", "B"):
            sub = [r for r in rows if r["collect_z"] == z and r["pole"] == pole]
            if sub:
                out["breakdown"][f"collect_z{z}@pole{pole}"] = {
                    "C_A": C.mean_ci(seed_means(sub, ca)), "C_B": C.mean_ci(seed_means(sub, cb)),
                    "jsd_z0_z1": C.mean_ci(seed_means(sub, lambda r: r["jsd_z0_z1"]))}
    for role in ("ATTACK", "DEFEND"):
        sub = [r for r in rows if r["role"] == role]
        if sub:
            out["breakdown"][f"role_{role}"] = {
                "C_A": C.mean_ci(seed_means(sub, ca)), "C_B": C.mean_ci(seed_means(sub, cb)),
                "n_agent_states": len(sub)}
    out["n_agent_states"] = len(rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default=None, help="i/n: run only every n-th episode starting at i; no seal")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import torch
    from rl.causal_supervision import decision_mask_from_core

    spec = C.load_frozen_spec(SPEC_ID)
    pins = C.load_frozen_spec(BEHAVIOR_SPEC_ID)["systems"]["Ours"]["pins"]
    lo, hi = spec["state_collection"]["block"]
    seeds = list(SMOKE_SEEDS) if args.smoke else list(range(lo, hi + 1))
    C.check_block(SPEC_ID, seeds[0], seeds[-1], smoke=args.smoke)
    out_dir = C.STRENGTH_DIR / ("smoke" if args.smoke else "results") / SPEC_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{SPEC_ID}_RESULT.json"
    if result_path.exists() and not args.smoke:
        raise SystemExit(f"REFUSING: {result_path.name} already sealed (write-once)")
    partial = out_dir / "agent_states.PARTIAL.jsonl"
    if args.smoke and not args.shard:
        for f in [partial, *partial.parent.glob(f"{partial.stem}.shard*.jsonl")]:
            if f.exists():
                f.unlink()
    write_to = C.shard_path(partial, args.shard)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    genomes, identity, cert = C.resolve_poles()
    subj = spec["subject"]
    student = C.load_student(device, subj["checkpoint"], subj["sha256"])
    ours = C.load_ours(device, pins=pins)
    teachers = {s: {"defend": ours[s].defend.model.eval(), "attack": ours[s].attack.model.eval()} for s in ("A", "B")}

    done_eps = {}
    for r in C.read_partial(partial).values():
        done_eps[r["key"]] = r
    print(f"{SPEC_ID} {'SMOKE' if args.smoke else 'CONFIRMATORY'}  seeds {seeds[0]}..{seeds[-1]}  "
          f"episodes {4 * len(seeds)}  resumed {len(done_eps)}  poles vs {cert}", flush=True)
    idx = -1
    for z in ("A", "B"):
        actor = C.StudentActor(student, 0 if z == "A" else 1)
        for pole in ("A", "B"):
            for seed in seeds:
                idx += 1
                key = f"{z}|{pole}|{seed}"
                if key in done_eps or not C.in_shard(idx, args.shard):
                    continue
                ep_rows: list[dict] = []

                def on_tick(core, obs, action, t, ep_rows=ep_rows):
                    decided = np.asarray(decision_mask_from_core(core, C.N, side="blue").detach().cpu())[0]
                    if decided.any():
                        for r in probe_state(student.model, teachers, to_torch(obs, device), decided):
                            ep_rows.append({"seed": seed, "collect_z": z, "pole": pole, "tick": t, **r})

                res = C.run_episode(actor, pole, seed, device, genomes, identity, on_tick=on_tick,
                                    context=f"{SPEC_ID} collect z{z}")
                C.append_partial(write_to, {"key": key, "episode": res, "rows": ep_rows})
                done_eps[key] = {"key": key, "episode": res, "rows": ep_rows}
                ca = np.mean([r["dA_z1"] - r["dA_z0"] for r in ep_rows]) if ep_rows else float("nan")
                cb = np.mean([r["dB_z0"] - r["dB_z1"] for r in ep_rows]) if ep_rows else float("nan")
                print(f"  [{len(done_eps)}/{4 * len(seeds)}] {key}  agent-states {len(ep_rows)}  "
                      f"C_A {ca:+.4f}  C_B {cb:+.4f}", flush=True)

    if args.shard:
        print(f"shard {args.shard} finished; run without --shard to merge and seal", flush=True)
        return 0
    done_eps = C.read_partial(partial)
    if len(done_eps) != 4 * len(seeds):
        raise SystemExit(f"REFUSING to seal: {len(done_eps)}/{4 * len(seeds)} episodes present")
    rows = [r for e in done_eps.values() for r in e["rows"]]
    with (out_dir / "agent_states.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r[k] for k in FIELDS} for r in rows)
    analysis = analyse(rows)
    result = {"id": SPEC_ID, "status": "SMOKE" if args.smoke else "SEALED", "utc": _now(),
              "spec_sha256": C.sha256(C.STRENGTH_DIR / f"{SPEC_ID}_SPEC.json"),
              "seeds": [seeds[0], seeds[-1]], "n_seeds": len(seeds), "device": device,
              "pole_certification": cert, "analysis": analysis}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.smoke:
        from experiments.seed_registry import set_status
        set_status(SPEC_ID, "SPENT", note=f"sealed {result_path.name}")
    p = analysis["primary"]
    print(f"\nC_A {p['C_A']['mean']:+.4f} [{p['C_A']['lcb95']:+.4f}, {p['C_A']['ucb95']:+.4f}]  "
          f"C_B {p['C_B']['mean']:+.4f} [{p['C_B']['lcb95']:+.4f}, {p['C_B']['ucb95']:+.4f}]  "
          f"steers={p['z_steers_toward_matching_teacher']}  agent-states {analysis['n_agent_states']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
