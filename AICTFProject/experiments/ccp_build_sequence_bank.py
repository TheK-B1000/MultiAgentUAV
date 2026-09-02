"""Build the offline SEQUENCE-mode training bank from the frozen 20-segment causal bank.

A CausalSegment (rl/causal_segment_bank.py) says WHO should follow WHICH teacher, weighted
by how much, starting at a specific reconstructed state. It does not itself carry the
(observation, action) pairs a policy-gradient loss needs. This script produces those pairs,
once, offline -- so the production runner samples from a static array exactly the way
PairedRehearsalRunner already does, rather than driving a live environment mid-training.

For each segment: replay its prefix (the manifest's own stored actions, asserted against
V4's live output -- the same fail-closed check Phase 1's collector used), verify the free set
at s_t still matches what the manifest recorded, then roll out to episode end with the
segment's controlled_agents driven by its winner-directed teacher and every other agent
driven by V4 under the pole-matched latent -- the exact intervention full_takeover measured,
because active_until=episode-termination means training on anything else would train on a
claim Phase 1 never tested.

Only the 7 NON-ZERO-weight segments get rollout data collected. This is a compute
optimisation, not a significance filter: a zero-weight segment contributes exactly zero to
the loss regardless of what data it is paired with (verified in gate 1), so collecting its
rollout would cost real time to produce rows nothing will ever read. The 13 zero-weight
segments remain in the frozen CausalSegment bank and in the continuous-weighting record;
they are simply not worth rolling out.

Every stored row carries an EXPLICIT decision_mask per controlled agent (was this agent
actually free to act here) alongside its weight, so committed-agent exclusion is provable two
structurally independent ways at train time, matching the pattern used throughout this
project's other guards.

One-shot. Run:  python experiments/ccp_build_sequence_bank.py --device cuda
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

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
MANIFEST = SD / "CCP_PHASE1_PILOT_MANIFEST.json"
PHASE1_RESULT = SD / "CCP_PHASE1_CAUSAL_BRANCHING.json"
OUT_NPZ = SD / "ccp_sequence_bank.npz"
OUT_META = SD / "CCP_SEQUENCE_BANK.json"

N_MACROS, N_TARGETS = 5, 50
POLE_LATENT = {"A": 0, "B": 1}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if OUT_NPZ.is_file() or OUT_META.is_file():
        raise SystemExit(f"REFUSING: {OUT_NPZ} or {OUT_META} exists; one-shot")

    from rl.causal_segment_bank import build_segment_bank, segment_bank_hash

    bank = build_segment_bank(PHASE1_RESULT)
    if len(bank) != 20:
        raise SystemExit(f"REFUSING: expected 20 segments, got {len(bank)}")
    bank_hash = segment_bank_hash(bank)
    nonzero = [s for s in bank if s.weight > 0]
    if len(nonzero) != 7:
        raise SystemExit(f"REFUSING: expected 7 non-zero segments, got {len(nonzero)}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    states_by_id = {s["state_id"]: s for s in manifest["states"]}

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome)
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    frozen = json.loads((SD / "HOG_PSP_V4_MODEL_FROZEN.json").read_text(encoding="utf-8"))
    v4_ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]

    teacher_paths = {}
    for name, meta in manifest["TEACHER_POLICIES"].items():
        if name.startswith("pi_"):
            p = ROOT / meta["path"]
            if not p.is_file():
                raise SystemExit(f"REFUSING: {name} checkpoint missing at {p}")
            import hashlib as _h
            got = _h.sha256(p.read_bytes()).hexdigest()
            if got != meta["sha256"]:
                raise SystemExit(f"REFUSING: {name} sha256 mismatch")
            teacher_paths[name] = p

    probe = R2.build_env(device, states_by_id[nonzero[0].start_state_id]["seed"])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    v4 = load_custom_ppo_policy(str(v4_ck), obs_space, act_space, device=device)
    (v4.model if hasattr(v4, "model") else v4).eval()
    teachers = {}
    for name, p in teacher_paths.items():
        pol = load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
        (pol.model if hasattr(pol, "model") else pol).eval()
        teachers[name] = pol

    print(f"CCP SEQUENCE BANK BUILD  {_now()}")
    print(f"  segment bank hash {bank_hash}")
    print(f"  {len(nonzero)} non-zero segments to roll out\n", flush=True)

    def setup(seed: int, pole: str):
        env = R2.build_env(device, seed)
        core = env.core
        core._bt_profile_override = None
        core._sds_opening_hold_steps = 0
        genomes = {"OP6": pole_A_genome()} if pole == "A" else {}
        install_keyed_opponent_overlays(core, genomes)
        key = P0.POLES[pole]
        env.env_method("set_phase", phase_from_tag(key))
        env.env_method("set_next_opponent", "SCRIPTED", key)
        obs = env.reset()
        obs["global_state"] = env.state()
        assert_live_opponent_batch(core, genomes, allowed_keys=(key,), context="seq bank")
        return env, obs, core

    def obs_row(obs):
        return {k: np.asarray(obs[k])[0].copy() for k in ("global_state", "grid", "vec", "mask")}, \
               np.asarray(obs["agent_mask"])[0].copy() if "agent_mask" in obs else np.ones(2, np.float32)

    rows = {k: [] for k in ("global_state", "grid", "vec", "agent_mask", "mask",
                            "actions", "z_idx", "decision_mask", "weight", "segment_idx")}
    seg_meta = []

    for seg_idx, seg in enumerate(nonzero):
        st = states_by_id.get(seg.start_state_id)
        if st is None:
            raise SystemExit(f"REFUSING: {seg.start_state_id} not in the manifest")
        pole, seed, prefix = seg.pole, st["seed"], st["actions"]
        z = POLE_LATENT[pole]
        teacher_pol = teachers[seg.teacher]

        env, obs, core = setup(seed, pole)
        try:
            v4.fixed_latent_strategy = True
            v4.fixed_latent_strategy_id = z
            v4.reset_strategy()
            if getattr(teacher_pol, "fixed_latent_strategy", None) is not None:
                teacher_pol.fixed_latent_strategy = False
            teacher_pol.reset_strategy()

            for i, want in enumerate(prefix):
                a, _ = v4.predict(obs, deterministic=True)
                teacher_pol.predict(obs, deterministic=True)     # keep any context aligned
                got = [int(x) for x in np.asarray(a).ravel()]
                if got != list(want):
                    raise SystemExit(f"REFUSING: prefix divergence at step {i} for "
                                     f"{seg.segment_id}: {got} != {list(want)}")
                env.step_async(a)
                obs, _r, done, _i = env.step_wait()
                obs["global_state"] = env.state()
                if bool(np.asarray(done).any()):
                    raise SystemExit(f"REFUSING: episode ended inside the prefix for "
                                     f"{seg.segment_id}")

            f0 = bool((core.blue_commit_ticks_left[0, 0] <= 0).item())
            f1 = bool((core.blue_commit_ticks_left[0, 1] <= 0).item())
            actual_free = {a for a, free in ((0, f0), (1, f1)) if free}
            if not set(seg.controlled_agents) <= actual_free:
                raise SystemExit(f"REFUSING: {seg.segment_id} controlled_agents "
                                 f"{seg.controlled_agents} not free at s_t (free={actual_free})")

            n_rows = 0
            for _step in range(R2.MAX_STEPS):
                obs_fields, amask = obs_row(obs)
                a_v4, _ = v4.predict(obs, deterministic=True)
                act = np.asarray(a_v4).ravel().copy()
                dmask = np.zeros(2, dtype=np.float32)
                for ag in seg.controlled_agents:
                    free = bool((core.blue_commit_ticks_left[0, ag] <= 0).item())
                    dmask[ag] = 1.0 if free else 0.0
                if any(dmask[ag] > 0 for ag in seg.controlled_agents):
                    a_t, _ = teacher_pol.predict(obs, deterministic=True)
                    tsp = np.asarray(a_t).ravel()
                    for ag in seg.controlled_agents:
                        if dmask[ag] > 0:
                            act[ag * 2] = tsp[ag * 2]
                            act[ag * 2 + 1] = tsp[ag * 2 + 1]
                else:
                    teacher_pol.predict(obs, deterministic=True)   # keep context aligned

                w = np.zeros(2, dtype=np.float32)
                for ag in seg.controlled_agents:
                    w[ag] = seg.weight if dmask[ag] > 0 else 0.0

                rows["global_state"].append(obs_fields["global_state"])
                rows["grid"].append(obs_fields["grid"])
                rows["vec"].append(obs_fields["vec"])
                rows["agent_mask"].append(amask)
                rows["mask"].append(obs_fields["mask"])
                rows["actions"].append(act.astype(np.int64))
                rows["z_idx"].append(z)
                rows["decision_mask"].append(dmask)
                rows["weight"].append(w)
                rows["segment_idx"].append(seg_idx)
                n_rows += 1

                env.step_async(act)
                obs, _r, done, _i = env.step_wait()
                obs["global_state"] = env.state()
                if bool(np.asarray(done).any()):
                    break
        finally:
            env.close()

        live_rows = sum(1 for i in range(len(rows["segment_idx"]) - n_rows, len(rows["segment_idx"]))
                        if rows["decision_mask"][i].sum() > 0)
        seg_meta.append({"segment_id": seg.segment_id, "pole": seg.pole, "latent": seg.latent,
                         "teacher": seg.teacher, "weight": seg.weight,
                         "controlled_agents": list(seg.controlled_agents),
                         "start_state_id": seg.start_state_id,
                         "episode_rows": n_rows, "live_decision_rows": live_rows})
        print(f"  {seg.segment_id:45s} {n_rows:4d} rows ({live_rows} live)  "
              f"teacher={seg.teacher}", flush=True)

    for k in rows:
        rows[k] = np.stack(rows[k]) if k not in ("z_idx", "segment_idx") else np.asarray(rows[k])

    total_rows = len(rows["z_idx"])
    total_live = sum(m["live_decision_rows"] for m in seg_meta)
    if total_live == 0:
        raise SystemExit("REFUSING: zero live-decision rows collected across all segments")

    np.savez_compressed(OUT_NPZ, **rows)
    import hashlib
    npz_sha = hashlib.sha256(OUT_NPZ.read_bytes()).hexdigest()

    OUT_META.write_text(json.dumps({
        "record": "CCP SEQUENCE-mode offline bank", "status": "FROZEN_ARTIFACT", "utc": _now(),
        "segment_bank_hash": bank_hash,
        "npz_path": str(OUT_NPZ.relative_to(ROOT)).replace("\\", "/"),
        "npz_sha256": npz_sha,
        "total_segments_in_causal_bank": 20,
        "nonzero_segments_rolled_out": len(nonzero),
        "zero_weight_segments_skipped": 20 - len(nonzero),
        "why_skipped_is_not_a_filter": ("a zero-weight segment contributes exactly zero to "
                                       "the loss regardless of paired data; skipping its "
                                       "rollout is a compute optimisation over the already-"
                                       "frozen continuous-weighting bank, not a new "
                                       "significance-based inclusion rule"),
        "total_rows": int(total_rows),
        "total_live_decision_rows": int(total_live),
        "segments": seg_meta,
        "fields": {k: list(v.shape) for k, v in rows.items()},
        "teacher_checkpoints": {k: {"path": str(v.relative_to(ROOT)).replace("\\", "/")}
                                for k, v in teacher_paths.items()},
        "v4_checkpoint_sha256": frozen["TERMINAL_CHECKPOINT"]["sha256"],
    }, indent=2), encoding="utf-8")
    print(f"\n  total rows {total_rows}, live-decision rows {total_live}")
    print(f"  -> {OUT_NPZ}")
    print(f"  -> {OUT_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
