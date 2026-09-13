r"""BC representation-sufficiency smoke: does teacher-action predictability improve
exactly according to the counterexample ladder?

Three input configurations, IDENTICAL network and parameter count. Only the
INFORMATION differs -- the entity streams are zero-masked, so every config has
the same capacity and the same optimisation problem shape:

    BASE        quantized grid + vec only            (current student)
    +TEAMMATES  + exact teammate entity stream
    +BOTH       + exact teammate AND enemy entity stream

PREDICTION, from OBSERVATION_LADDER (4v4):
    defender-perturbation counterexamples  9 -> 0 -> 0   (teammates fix these)
    enemy-perturbation counterexamples    10 -> 10 -> 0  (only enemies fix these)
So +TEAMMATES should improve prediction but LEAVE a residual error attributable
to enemy-geometry ambiguity, and +BOTH should close that residual. If instead
+TEAMMATES already closes everything, the ladder's mechanism is wrong.

Entity features are RELATIVE and permutation-invariantly pooled (shared per-entity
MLP + mean pool), so the same architecture works at 2v2/4v4/6v6 -- not twelve
concatenated absolute coordinates.

The grid is KEPT in every config. This is an augmentation test, not a replacement
test: the CNN keeps handling map/objective geometry.

    python -m experiments.eval_bc_representation_smoke --scale 4 --n-seeds 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LABEL = "BC_REPRESENTATION_SMOKE"
CONFIGS = ("BASE", "TEAMMATES", "BOTH")
ENT_F = 6          # dx, dy, dist, alive, carrying, tagged


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect(scale, strategy, seeds, device="cpu"):
    """Teacher rollouts -> (grid, vec, teammates, enemies, target_xy, macro)."""
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome
    from experiments.strategic_demand_searcher import apply_genome_to_core
    from experiments.teacher_action_adapter import adapt
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    S.AGENTS = scale
    g = pole_A_genome(scale)
    style = S.GUARD if strategy == "GUARD" else S.BREACH
    G, V, T, E, Y, M = [], [], [], [], [], []

    bar = tqdm_iter(seeds, desc=f"collect {scale}v{scale} {strategy}", unit="seed")
    for seed in bar:
        set_postfix(bar, f"n={len(Y)}")
        cfg = GPUFieldConfig(n_envs=1, max_blue_agents=scale, max_red_agents=scale,
            map_set="train", map_layout=S.MAP, max_decision_steps=S.MAX_STEPS,
            aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
            obstacle_obs_channel=True, tag_telemetry_enabled=True,
            own_flag_home_required_to_score=True, **S.RULESET)
        env = GPUCTFVecEnv(cfg); core = env.core
        try:
            opp = g.base_opponent
            env.env_method("set_phase", opp)
            env.env_method("set_next_opponent", "SCRIPTED", opp)
            apply_genome_to_core(core, g)
            core.blue_scripted = True
            core.set_blue_style(style)
            env.reset(); apply_genome_to_core(core, g); core.drain_tag_events()
            W = core._macro_targets.detach().cpu().numpy()

            for _t in range(S.MAX_STEPS):
                obs = core.get_obs_tensors("blue")
                grid = obs["grid"][0].detach().cpu().numpy().astype(np.float32)
                vec = obs["vec"][0].detach().cpu().numpy().astype(np.float32)
                bx = core.blue_x[0].cpu().numpy(); by = core.blue_y[0].cpu().numpy()
                rx = core.red_x[0].cpu().numpy(); ry = core.red_y[0].cpu().numpy()
                ba = core.blue_alive[0].cpu().numpy().astype(np.float32)
                bc = core.blue_carrying[0].cpu().numpy().astype(np.float32)
                bt = core.blue_tagged[0].cpu().numpy().astype(np.float32)
                ra = core.red_alive[0].cpu().numpy().astype(np.float32)
                rc = core.red_carrying[0].cpu().numpy().astype(np.float32)
                rt = core.red_tagged[0].cpu().numpy().astype(np.float32)
                btx, bty = core._get_scripted_targets("blue")

                for i in range(scale):
                    # teammates: every other blue agent, RELATIVE to i
                    tm = []
                    for j in range(scale):
                        if j == i:
                            continue
                        dx, dy = bx[j] - bx[i], by[j] - by[i]
                        tm.append([dx, dy, float(np.hypot(dx, dy)), ba[j], bc[j], bt[j]])
                    en = []
                    for k in range(scale):
                        dx, dy = rx[k] - bx[i], ry[k] - by[i]
                        en.append([dx, dy, float(np.hypot(dx, dy)), ra[k], rc[k], rt[k]])
                    a = adapt(core, float(btx[0, i]), float(bty[0, i]), i, W)
                    G.append(grid[i]); V.append(vec[i])
                    T.append(np.array(tm, dtype=np.float32))
                    E.append(np.array(en, dtype=np.float32))
                    Y.append([float(btx[0, i]), float(bty[0, i])])
                    M.append(int(a.macro) if a.macro is not None else 0)
                env.step_async(env.action_space.sample() * 0)
                _o, _r, d, _i = env.step_wait()
                if bool(np.asarray(d).any()):
                    break
        finally:
            env.close()
    return (np.stack(G), np.stack(V), np.stack(T), np.stack(E),
            np.array(Y, dtype=np.float32), np.array(M, dtype=np.int64))


class Net(nn.Module):
    """Identical in every config. Entity streams are zero-masked to ablate
    INFORMATION without changing parameter count or optimisation shape."""

    def __init__(self, n_ch, n_vec):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.vmlp = nn.Sequential(nn.Linear(n_vec, 32), nn.ReLU())
        self.tenc = nn.Sequential(nn.Linear(ENT_F, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.eenc = nn.Sequential(nn.Linear(ENT_F, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(32 + 32 + 32 + 32, 64), nn.ReLU())
        self.xy = nn.Linear(64, 2)
        self.mac = nn.Linear(64, 5)

    def forward(self, g, v, t, e):
        h = torch.cat([self.cnn(g), self.vmlp(v),
                       self.tenc(t).mean(dim=1), self.eenc(e).mean(dim=1)], dim=1)
        h = self.head(h)
        return self.xy(h), self.mac(h)


def train_eval(data, config, seed=0, epochs=30, device="cpu"):
    G, V, T, E, Y, M = data
    n = len(Y)
    rng = np.random.default_rng(12345)          # SAME split for every config
    perm = rng.permutation(n)
    ntr = int(0.8 * n)
    tr, va = perm[:ntr], perm[ntr:]

    Tm = T.copy(); Em = E.copy()
    if config == "BASE":
        Tm[:] = 0.0; Em[:] = 0.0
    elif config == "TEAMMATES":
        Em[:] = 0.0

    torch.manual_seed(seed)                     # SAME init for every config
    net = Net(G.shape[1], V.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    tg = torch.tensor(G); tv = torch.tensor(V); tt = torch.tensor(Tm)
    te = torch.tensor(Em); ty = torch.tensor(Y); tm = torch.tensor(M)

    bs = 256
    for _ep in range(epochs):
        net.train()
        idx = tr[np.random.default_rng(_ep).permutation(len(tr))]
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            opt.zero_grad()
            pxy, pm = net(tg[b], tv[b], tt[b], te[b])
            loss = nn.functional.mse_loss(pxy, ty[b]) + nn.functional.cross_entropy(pm, tm[b])
            loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        pxy, pm = net(tg[va], tv[va], tt[va], te[va])
        dist = torch.norm(pxy - ty[va], dim=1)
        acc = (pm.argmax(1) == tm[va]).float().mean().item()
    return {"config": config, "n_train": len(tr), "n_val": len(va),
            "val_mean_target_dist_cells": round(float(dist.mean()), 4),
            "val_median_target_dist_cells": round(float(dist.median()), 4),
            "val_macro_accuracy": round(acc, 4),
            "n_params": sum(p.numel() for p in net.parameters())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--strategy", default="GUARD")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()
    seeds = list(range(17600001, 17600001 + a.n_seeds))

    print(f"{LABEL}  {_now()}")
    print(f"  {a.scale}v{a.scale} {a.strategy}@A, seeds {seeds[0]}..{seeds[-1]}")
    print(f"  configs {CONFIGS} -- SAME net, SAME init, SAME split; only info differs")
    print(f"  PREDICTION: TEAMMATES improves but leaves an enemy-geometry residual; "
          f"BOTH closes it\n", flush=True)

    t0 = time.time()
    data = collect(a.scale, a.strategy, seeds)
    print(f"  collected {len(data[4])} samples in {time.time()-t0:.0f}s", flush=True)

    res = {}
    for c in CONFIGS:
        r = train_eval(data, c, epochs=a.epochs)
        res[c] = r
        print(f"  {c:<11s} val target dist: mean={r['val_mean_target_dist_cells']:.4f} "
              f"median={r['val_median_target_dist_cells']:.4f}  "
              f"macro acc={r['val_macro_accuracy']:.4f}  params={r['n_params']}", flush=True)

    base = res["BASE"]["val_mean_target_dist_cells"]
    tm = res["TEAMMATES"]["val_mean_target_dist_cells"]
    bo = res["BOTH"]["val_mean_target_dist_cells"]
    print(f"\n  improvement BASE->TEAMMATES: {base - tm:+.4f} cells")
    print(f"  improvement TEAMMATES->BOTH: {tm - bo:+.4f} cells")

    cfg = json.dumps({"scale": a.scale, "strategy": a.strategy, "n_seeds": a.n_seeds,
                      "epochs": a.epochs}, sort_keys=True)
    rid = f"{_now().replace(':','').replace('-','')}_{hashlib.sha256(cfg.encode()).hexdigest()[:8]}"
    rec = {"record": f"{LABEL} representation-sufficiency smoke",
           "status": "FROZEN_RESULT", "utc": _now(), "run_id": rid,
           "arm": "DIAGNOSTIC", "confirmatory": False,
           "study_class": "MECHANISTIC_FOLLOW_UP",
           "builds_on": "OBSERVATION_LADDER (teammates alone insufficient: enemy ce 10 -> 10)",
           "control": "identical architecture and parameter count in all configs; entity "
                      "streams zero-masked to ablate information only",
           "config_signature": json.loads(cfg), "elapsed_s": round(time.time() - t0, 1),
           "results": res,
           "READING": "If TEAMMATES improves over BASE but BOTH improves further, teacher "
                      "predictability tracks the counterexample ladder and the repair is "
                      "behaving for the reason we think. If TEAMMATES alone closes "
                      "everything, the ladder's mechanism is wrong.",
           "NOT_A_CLAIM": ["that PPO can learn this", "closed-loop performance",
                           "that entity features are sufficient for specialization"]}
    p = SD / f"{LABEL}_{rid}_RESULT.json"
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    if a.promote:
        (SD / f"{LABEL}_CANONICAL.json").write_text(json.dumps(
            {"record": f"CANONICAL {LABEL}", "points_to": p.name, "run_id": rid,
             "promoted_utc": _now()}, indent=2), encoding="utf-8")
    print(f"  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
