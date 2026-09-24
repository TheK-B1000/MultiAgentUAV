"""Collect GUARD≠BREACH disagreement observations for 4v4 sibling separation.

Frozen by: artifacts/strategic_demand/sppo/4V4_B3_SPECIALIZATION_PRESERVING_SPEC.json

Stores ONLY decision-eligible states where projected GUARD and BREACH discrete
actions disagree on at least one agent. The resulting NPZ is the support of m(o)=1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.opponent_spec import (  # noqa: E402
    assert_live_opponent_batch,
    install_keyed_opponent_overlays,
    pole_A_genome,
    _with_full_team_defender_gate,
)
from experiments.sds_genome import SDSGenome  # noqa: E402
from experiments.sappo_teacher_representability import BREACH, GUARD  # noqa: E402
from rl.curriculum import phase_from_tag  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _project_style_action(core, style: str):
    from macro_actions import MacroAction

    was = bool(getattr(core, "blue_scripted", False))
    core.blue_scripted = True
    core.set_blue_style(style)
    wp = core._macro_targets
    tx, ty = core._assign_blue_style_targets()
    t = torch.stack([tx[0], ty[0]], dim=-1)
    d = torch.cdist(t.unsqueeze(0).float(), wp.unsqueeze(0).float())[0]
    idx = torch.argmin(d, dim=-1)
    carrying = core.blue_carrying[0]
    macro = torch.where(
        carrying,
        torch.full_like(idx, int(MacroAction.GO_HOME)),
        torch.full_like(idx, int(MacroAction.GO_TO)),
    )
    act = torch.stack([macro, idx], dim=-1)
    core.blue_scripted = was
    return act


def _obs_to_numpy(obs) -> dict[str, np.ndarray]:
    if isinstance(obs, dict):
        out = {}
        for k, v in obs.items():
            if torch.is_tensor(v):
                out[k] = v.detach().cpu().numpy().copy()
            else:
                out[k] = np.asarray(v).copy()
        return out
    raise TypeError("expected dict observation")


def collect_episode(*, pole: str, seed: int, device: str, team_size: int,
                    pole_b_genome) -> list[dict[str, np.ndarray]]:
    import experiments.r2_learned_crossover as R2

    # Force team size into G0 chain then rebuild env at N.
    from experiments.train_scale import _propagate_team_size

    _propagate_team_size(team_size)
    base_key = "OP6" if pole == "A" else "OP7"
    genomes = {"OP6": pole_A_genome(team_size)}
    if pole == "B":
        genomes["OP7"] = pole_b_genome

    env = R2.build_env(device, seed)
    core = env.core
    rows: list[dict[str, np.ndarray]] = []
    try:
        core._bt_profile_override = None
        core._sds_opening_hold_steps = 0
        install_keyed_opponent_overlays(core, genomes)
        env.env_method("set_phase", phase_from_tag(base_key))
        env.env_method("set_next_opponent", "SCRIPTED", base_key)
        obs = env.reset()
        assert_live_opponent_batch(
            core, genomes, allowed_keys=(base_key,),
            context=f"disagreement {pole} seed {seed}",
        )
        # Drive the episode with BREACH projection (arbitrary driver); we only
        # *label* disagreement between GUARD and BREACH, we do not need the
        # driver to match either specialist.
        for _ in range(int(getattr(R2, "MAX_STEPS", 400))):
            avail = core.blue_commit_ticks_left[0] <= 0
            if bool(avail.any().item()):
                a_g = _project_style_action(core, GUARD)
                a_b = _project_style_action(core, BREACH)
                disagree = (a_g != a_b).any(dim=-1) & avail
                if bool(disagree.any().item()):
                    row = _obs_to_numpy(obs)
                    # Keep only env-0 slice if batched.
                    sliced = {}
                    for k, v in row.items():
                        sliced[k] = v[0] if v.ndim >= 1 and v.shape[0] >= 1 else v
                    rows.append(sliced)
            act = _project_style_action(core, BREACH if pole == "B" else GUARD)
            core.blue_scripted = False
            env.step_async(act.reshape(-1).cpu().numpy().astype(np.int64))
            obs, _r, done, _info = env.step_wait()
            if bool(np.asarray(done).any()):
                break
        return rows
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-size", type=int, required=True, choices=[4])
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--n-episodes", type=int, default=64)
    ap.add_argument("--pole-b-genome-json", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    raw = json.loads(Path(args.pole_b_genome_json).read_text(encoding="utf-8"))
    pole_b = _with_full_team_defender_gate(
        SDSGenome.from_dict(raw), int(args.team_size)
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".collector.lock"
    run_id = hashlib.sha256(
        f"{_now()}|{args.seed_base}|{args.n_episodes}|{args.device}".encode()
    ).hexdigest()[:12]
    try:
        fd = __import__("os").open(str(lock), __import__("os").O_CREAT
                                   | __import__("os").O_EXCL
                                   | __import__("os").O_WRONLY)
        __import__("os").write(fd, run_id.encode())
        __import__("os").close(fd)
    except FileExistsError:
        raise SystemExit(f"REFUSING: {lock} exists — another collector owns this directory.")

    print(f"GUARD/BREACH DISAGREEMENT COLLECTION  {_now()}  run_id={run_id}")
    print(f"  seeds {args.seed_base}..{args.seed_base + args.n_episodes - 1}")
    print(f"  pole_B genome {args.pole_b_genome_json}")

    all_rows: list[dict[str, np.ndarray]] = []
    n_ep = int(args.n_episodes)
    for i in range(n_ep):
        pole = "A" if i < (n_ep // 2) else "B"
        seed = int(args.seed_base) + i
        rows = collect_episode(
            pole=pole, seed=seed, device=args.device,
            team_size=int(args.team_size), pole_b_genome=pole_b,
        )
        all_rows.extend(rows)
        print(f"  ep {i+1}/{n_ep} pole={pole} seed={seed} disagreement_rows={len(rows)}")

    if not all_rows:
        raise SystemExit("FAIL-CLOSED: collected zero disagreement rows")

    keys = sorted(all_rows[0].keys())
    payload = {"run_id": np.asarray([run_id])}
    for k in keys:
        payload[f"obs_{k}"] = np.stack([r[k] for r in all_rows], axis=0)
    npz_path = out / "disagreement_rows.npz"
    np.savez_compressed(npz_path, **payload)
    manifest = {
        "record": "4v4 B3 GUARD/BREACH disagreement dataset",
        "utc": _now(),
        "run_id": run_id,
        "implements": "4V4_B3_SPECIALIZATION_PRESERVING_SPEC.json",
        "seed_base": int(args.seed_base),
        "n_episodes": n_ep,
        "n_disagreement_rows": len(all_rows),
        "npz": str(npz_path.as_posix()),
        "GUARD": GUARD,
        "BREACH": BREACH,
        "pole_b_genome_json": str(Path(args.pole_b_genome_json).as_posix()),
    }
    (out / "collection_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"  wrote {npz_path}  rows={len(all_rows)}")
    lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
