"""O3 demand gate — evaluator for the frozen contract at d89e9f8.

Implements O3_DEMAND_GATE_POPULATIONS_FROZEN.json literally. No population,
metric, threshold or label is chosen here; every one is read from the frozen
artifacts and the run fails closed if they cannot be verified.

Three arms per (opponent, eval_seed), all on the same seed:
    A   G0 for the whole episode
    B   G0 until the first C_fork firing, then O3 to terminal   (the selector)
    O   O3 for the whole episode

    C_fork      delta = E[U | B] - E[U | A]           final score differential
    C_anchor    delta = E[U | A@boundary] - E[U | O@boundary]
                scored AT each arm's own first firing, never at episode end
    whole-game  A vs O, competence and global dominance only
    G_available V_selector(B) - V_best_fixed(max of A, O)

Run:  python experiments/run_o3_demand_gate.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

PREREG = PROJECT_ROOT / "artifacts" / "o3_preregistration"
FROZEN = PREREG / "O3_DEMAND_GATE_POPULATIONS_FROZEN.json"
OUT = PROJECT_ROOT / "artifacts" / "o3_demand_gate"
RESULT = OUT / "O3_DEMAND_GATE_RESULT.json"

G0_TAG = "g0_v5_long_seed3200001"
G0_CKPT = PROJECT_ROOT / "artifacts" / "g0_v5_long" / G0_TAG / "ckpts" / f"final_{G0_TAG}.zip"
O3_CKPT = PROJECT_ROOT / "artifacts" / "o3_run_3400001" / "ckpts" / "final_o3_run_3400001.zip"
PROTOCOL_COMMIT = "d89e9f8"
RULESET = "RULESET_V2_AQUATICUS_10S"

RESAMPLES = 2000
BOOT_SEED = 12345
LCB_PCT = 2.5


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _identity(ckpt: Path) -> dict:
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.ruleset_identity import ARTIFACT_IDENTITY_KEY

    payload = read_checkpoint_payload(str(ckpt), map_location="cpu")
    ai = payload.get(ARTIFACT_IDENTITY_KEY, {})
    return {
        "path": str(ckpt.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(ckpt),
        "global_step": int(payload.get("global_step", 0)),
        "ruleset_id": str(ai.get("ruleset_id")),
        "canonical_map": str(ai.get("canonical_map")),
        "formal_result_eligible": bool(ai.get("formal_result_eligible")),
        "identity_override_used": bool(ai.get("identity_override_used")),
        "payload": payload,
    }


def guards(frozen: dict) -> tuple[dict, list[str]]:
    """Fail-closed on only what could invalidate the run."""
    problems: list[str] = []

    commit = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--",
         "AICTFProject/artifacts/o3_preregistration/O3_DEMAND_GATE_POPULATIONS_FROZEN.json"],
        cwd=str(PROJECT_ROOT.parent), capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    if not commit.startswith(PROTOCOL_COMMIT[:7]):
        problems.append(f"protocol commit {commit!r} != {PROTOCOL_COMMIT}")
    if frozen.get("status") != "FROZEN":
        problems.append("demand-gate populations are not FROZEN")

    ids = {}
    for name, ck in (("G0", G0_CKPT), ("O3", O3_CKPT)):
        if not ck.is_file():
            problems.append(f"{name} checkpoint missing: {ck}")
            continue
        i = _identity(ck)
        ids[name] = {k: v for k, v in i.items() if k != "payload"}
        ids[name + "_payload"] = i["payload"]
        if i["ruleset_id"] != RULESET:
            problems.append(f"{name} ruleset {i['ruleset_id']} != {RULESET}")
        if i["canonical_map"] != "map_a":
            problems.append(f"{name} map {i['canonical_map']} != map_a")
        if not i["formal_result_eligible"]:
            problems.append(f"{name} formal_result_eligible is false")
        if i["identity_override_used"]:
            problems.append(f"{name} identity_override_used is true")
        if i["global_step"] < 1_000_000:
            problems.append(f"{name} global_step {i['global_step']} < 1,000,000")

    if RESULT.exists():
        problems.append(f"a demand-gate result already exists at {RESULT.name}")
    if not torch.cuda.is_available():
        problems.append("CUDA required; O3 and G0 artifacts are CUDA-produced")

    return ids, problems


def run_arm(policy_prefix, policy_suffix, *, opponent: str, seed: int, device: str) -> dict:
    """One episode. policy_suffix=None means policy_prefix controls throughout.

    Records the score differential at the FIRST C_fork firing (the anchor
    quantity) and at terminal (the whole-game / C_fork quantity).
    """
    from experiments.eval_v6i9_map_awareness import (
        _adapt_obs_for_policy, _done, _predict, _reset_obs, _unpack_step,
    )
    from experiments.run_g0_v2_evaluation import (
        AGENTS, CANONICAL_MAP, EPISODE_HORIZON, V2_RULES, legal_context,
    )
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from rl.analysis.c_fork_detector import CForkDetector
    from rl.evaluation.opponent_resolution import (
        get_opponent_key, set_opponent, validate_opponent_name,
    )

    requested = validate_opponent_name(opponent)
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
        map_set="train", map_layout=CANONICAL_MAP,
        max_decision_steps=EPISODE_HORIZON, aquaticus_profile=True,
        rules_profile="OURS", device=device, seed=int(seed),
        obstacle_obs_channel=True, tag_telemetry_enabled=True, **V2_RULES,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    models = []
    for p in (policy_prefix, policy_suffix):
        if p is None:
            continue
        m = p.model if hasattr(p, "model") else p
        models.append((m, getattr(m, "training", False)))
        if hasattr(m, "eval"):
            m.eval()

    det = CForkDetector()
    det.reset()
    trigger_step = None
    anchor_score = None
    blue = red = 0.0
    obs_blue_max = obs_red_max = 0.0
    try:
        set_opponent(env, requested)
        obs = _reset_obs(env.reset())
        if get_opponent_key(env) != requested:
            raise RuntimeError("opponent drift")
        core.drain_tag_events()
        active = False
        for step_i in range(EPISODE_HORIZON + 8):
            ctx = legal_context(core)
            obs_blue_max = max(obs_blue_max, ctx["blue_score"])
            obs_red_max = max(obs_red_max, ctx["red_score"])
            fired = det.step(core, step_i)
            if fired and trigger_step is None:
                trigger_step = step_i
                # C_anchor: scored AT the boundary, never at episode end.
                anchor_score = ctx["blue_score"] - ctx["red_score"]
            if fired:
                active = True
            actor = policy_suffix if (active and policy_suffix is not None) else policy_prefix
            action = _predict(actor, _adapt_obs_for_policy(obs, actor))
            obs, _, done, _infos = _unpack_step(env.step(action))
            for e in core.drain_tag_events():
                if e.get("event_type") != "capture_scored":
                    continue
                after = int(e.get("score_after", 0))
                if e.get("scoring_team") == "blue":
                    blue = max(blue, after)
                else:
                    red = max(red, after)
            if _done(done):
                break
        final_blue = float(max(blue, obs_blue_max))
        final_red = float(max(red, obs_red_max))
    finally:
        for m, was in models:
            if hasattr(m, "train"):
                m.train(was)
        env.close()

    if anchor_score is None:          # never fired: anchor read at terminal
        anchor_score = final_blue - final_red
    return {
        "opponent": requested, "eval_seed": int(seed),
        "episode_key": f"{requested}:{seed}",
        "score_diff": final_blue - final_red,
        "win": int(final_blue > final_red),
        "anchor_score": float(anchor_score),
        "trigger_step": trigger_step,
        "fired": int(trigger_step is not None),
    }


def paired_bootstrap(pairs: list[tuple[float, float]], *, rng) -> dict:
    if len(pairs) < 2:
        return {"delta": None, "lcb95": None, "ucb95": None, "n_pairs": len(pairs)}
    d = np.asarray([b - a for a, b in pairs], dtype=float)
    draws = d[rng.integers(0, d.size, (RESAMPLES, d.size))].mean(axis=1)
    return {
        "delta": round(float(d.mean()), 6),
        "lcb95": round(float(np.percentile(draws, LCB_PCT)), 6),
        "ucb95": round(float(np.percentile(draws, 100 - LCB_PCT)), 6),
        "n_pairs": len(pairs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    ids, problems = guards(frozen)
    print("=" * 74)
    print("O3 DEMAND GATE")
    print(f"  protocol: {PROTOCOL_COMMIT}   guards: {'PASS' if not problems else 'FAIL'}")
    for p in problems:
        print(f"    - {p}")
    if problems:
        return 3

    base = int(frozen["inherited_unchanged_from_de6fb58"]["held_out_seed_block"][0])
    from experiments.run_g0_v2_evaluation import resolve_cnn_channels
    from experiments.run_g0_v2_seed import OPPONENTS
    from rl.evaluation.checkpoint import load_policy

    g0 = load_policy(str(G0_CKPT), device=args.device,
                     num_cnn_channels=resolve_cnn_channels(ids["G0_payload"], context=str(G0_CKPT)))
    o3 = load_policy(str(O3_CKPT), device=args.device,
                     num_cnn_channels=resolve_cnn_channels(ids["O3_payload"], context=str(O3_CKPT)))

    print(f"  seeds {base}..{base + args.episodes - 1}  opponents {OPPONENTS}")
    print("=" * 74)
    started = time.time()
    A, B, O = [], [], []
    for opp in OPPONENTS:
        for i in range(args.episodes):
            s = base + i
            A.append(run_arm(g0, None, opponent=opp, seed=s, device=args.device))
            B.append(run_arm(g0, o3, opponent=opp, seed=s, device=args.device))
            O.append(run_arm(o3, None, opponent=opp, seed=s, device=args.device))
        print(f"  {opp}: done", flush=True)

    rng = np.random.default_rng(BOOT_SEED)
    ia = {r["episode_key"]: r for r in A}
    ib = {r["episode_key"]: r for r in B}
    io = {r["episode_key"]: r for r in O}
    keys = [k for k in ia if k in ib and k in io]

    fork = paired_bootstrap([(ia[k]["score_diff"], ib[k]["score_diff"]) for k in keys], rng=rng)
    anchor = paired_bootstrap([(io[k]["anchor_score"], ia[k]["anchor_score"]) for k in keys], rng=rng)

    v_a = statistics.fmean(ia[k]["score_diff"] for k in keys)
    v_o = statistics.fmean(io[k]["score_diff"] for k in keys)
    v_sel = statistics.fmean(ib[k]["score_diff"] for k in keys)
    best_fixed, best_map = ("G0", ia) if v_a >= v_o else ("O3", io)
    gav = paired_bootstrap([(best_map[k]["score_diff"], ib[k]["score_diff"]) for k in keys], rng=rng)

    wr_g0 = statistics.fmean(ia[k]["win"] for k in keys)
    wr_o3 = statistics.fmean(io[k]["win"] for k in keys)
    floor_ok = bool(wr_o3 >= 0.50 * wr_g0)

    fork_pass = bool(fork["delta"] is not None and fork["delta"] > 0 and fork["lcb95"] > 0)
    anchor_pass = bool(anchor["delta"] is not None and anchor["delta"] > 0 and anchor["lcb95"] > 0)
    gav_pass = bool(gav["lcb95"] is not None and gav["lcb95"] > 0)

    if not floor_ok:
        verdict = "COMPETENCE_FLOOR_FAIL"
    elif not fork_pass:
        verdict = "NEGATIVE_DEVELOPMENT"
    elif not anchor_pass:
        verdict = "BETTER_GENERALIST"
    elif not gav_pass:
        verdict = "CROSSOVER_NO_REPERTOIRE_VALUE"
    else:
        verdict = "COMPLEMENTARY_PAIR"

    claims = frozen["outcomes_and_permitted_claims"][verdict]
    doc = {
        "record": "O3 demand gate result",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_commit": PROTOCOL_COMMIT,
        "device": args.device, "torch": torch.__version__, "cuda": torch.version.cuda,
        "identities": {k: v for k, v in ids.items() if not k.endswith("_payload")},
        "seed_block": [base, base + args.episodes - 1],
        "n_episodes_per_arm": len(keys),
        "c_fork": {**fork, "PASS": fork_pass,
                   "mean_G0": round(v_a, 4), "mean_selector_B": round(v_sel, 4)},
        "c_anchor": {**anchor, "PASS": anchor_pass,
                     "mean_G0_at_boundary": round(statistics.fmean(ia[k]["anchor_score"] for k in keys), 4),
                     "mean_O3_at_boundary": round(statistics.fmean(io[k]["anchor_score"] for k in keys), 4)},
        "whole_game": {"G0_score_diff": round(v_a, 4), "O3_score_diff": round(v_o, 4),
                       "G0_win_rate": round(wr_g0, 4), "O3_win_rate": round(wr_o3, 4)},
        "g_available": {**gav, "best_fixed": best_fixed, "PASS": gav_pass,
                        "V_selector": round(v_sel, 4)},
        "competence_floor": {"threshold": "O3 wr >= 0.50 x G0 wr",
                             "G0_wr": round(wr_g0, 4), "O3_wr": round(wr_o3, 4),
                             "PASS": floor_ok},
        "handoff": {"fired_fraction_armA": round(statistics.fmean(ia[k]["fired"] for k in keys), 4),
                    "fired_fraction_armO": round(statistics.fmean(io[k]["fired"] for k in keys), 4)},
        "verdict": verdict,
        "permitted_claim": claims["permitted_claim"],
        "prohibited_claim": claims.get("prohibited_claim"),
        "wall_seconds": round(time.time() - started, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"  C_fork   delta {fork['delta']}  LCB95 {fork['lcb95']}  -> {fork_pass}")
    print(f"  C_anchor delta {anchor['delta']}  LCB95 {anchor['lcb95']}  -> {anchor_pass}")
    print(f"  whole    G0 {v_a:.4f} (wr {wr_g0:.3f})   O3 {v_o:.4f} (wr {wr_o3:.3f})")
    print(f"  G_avail  delta {gav['delta']}  LCB95 {gav['lcb95']}  best_fixed {best_fixed} -> {gav_pass}")
    print(f"  floor    {floor_ok}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  permitted: {claims['permitted_claim']}")
    print(f"  wrote {RESULT.relative_to(PROJECT_ROOT)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
