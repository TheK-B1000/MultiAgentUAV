"""Team-size-parameterized wrapper around collect_distillation_states.py.

Implements the SCALED_DISTILLATION_AUDIT_RESULT recorded in
SCALING_IS_ADDITIVE_DEADLINE_RULE.json. The distillation CORE
(rl.teacher_distillation, rl.causal_supervision) is already team-size generic -- it derives
n_agents from decision_mask.shape[1] and asserts on mismatch. Exactly two things are NOT
generic and are set explicitly here, by name, rather than through any attribute sweep:

  1. experiments.collect_distillation_states.N_AGENTS
     Named N_AGENTS, not AGENTS -- train_scale.py's _propagate_team_size sweep (which looks
     for an attribute literally named AGENTS) does NOT catch it. Recorded as a naming trap
     in SCALING_IS_ADDITIVE_DEADLINE_RULE.json rather than papered over with a broader sweep;
     the PI's instruction after that finding was explicit handling of both names, not a more
     "clever" generic mechanism.
  2. experiments.r2_learned_crossover.AGENTS
     Consumed by R2.build_env's GPUFieldConfig construction.

Both are propagated by NAME, not swept, deliberately.

This wrapper does not (yet) invoke a full collection run -- that needs real pi_A/pi_B
checkpoints at the target size, which do not exist until specialist smokes/training are
authorized and complete. Its ``--check-only`` mode proves team size reaches the real
construction path (module globals, GPUFieldConfig, decision_mask_from_core's own agent-count
assertion) without requiring a checkpoint at all, which is what "audited" can mean before any
scaled specialist exists.

Run:
  python experiments/collect_distillation_states_scale.py --team-size 4 --check-only --device cpu
  python experiments/collect_distillation_states_scale.py --team-size 4 \
      --pi-a-path <ckpt> --pi-b-path <ckpt> --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_TEAM_SIZES = (4, 6)


def _propagate_by_name(team_size: int) -> dict:
    """Set team size on EXACTLY the two non-generic globals, by their real names."""
    import experiments.collect_distillation_states as C
    import experiments.r2_learned_crossover as R2

    before = {"collect_distillation_states.N_AGENTS": C.N_AGENTS,
              "r2_learned_crossover.AGENTS": R2.AGENTS}
    C.N_AGENTS = int(team_size)
    R2.AGENTS = int(team_size)
    after = {"collect_distillation_states.N_AGENTS": C.N_AGENTS,
             "r2_learned_crossover.AGENTS": R2.AGENTS}
    return {"before": before, "after": after}


def check_only(team_size: int, device: str) -> int:
    """Prove team size reaches the real construction path -- no checkpoint required.

    Builds the actual probe env collect_distillation_states.py builds (R2.build_env),
    inspects its observation space, and separately proves decision_mask_from_core's own
    agent-count assertion fires on a deliberate mismatch -- the safety net that protects this
    path even where no explicit check exists.
    """
    n = int(team_size)
    prop = _propagate_by_name(n)
    print(f"[collect_distillation_states_scale] propagated by name: {prop['after']}")

    import experiments.r2_learned_crossover as R2

    env = R2.build_env(device, 99_900_000 + n)
    try:
        grid_dim = int(env.observation_space.spaces["grid"].shape[0])
        print(f"  probe env grid agent dim: {grid_dim}  (expect {n})")
        if grid_dim != n:
            raise SystemExit(f"FAIL-CLOSED: probe env grid dim {grid_dim} != team size {n}; "
                             f"R2.AGENTS did not reach GPUFieldConfig")

        from rl.causal_supervision import CausalRoutingError, decision_mask_from_core

        core = env.core
        env.reset()
        try:
            decision_mask_from_core(core, n, side="blue")
            print(f"  decision_mask_from_core(core, {n}): OK at the matching size")
        except CausalRoutingError as e:
            raise SystemExit(f"FAIL-CLOSED: decision_mask_from_core rejected the MATCHING "
                             f"team size {n}: {e}")

        # Deliberate mismatch: prove the safety net actually fires, not just that it exists.
        wrong = n + 1
        try:
            decision_mask_from_core(core, wrong, side="blue")
            raise SystemExit(f"FAIL-CLOSED: decision_mask_from_core accepted a WRONG agent "
                             f"count ({wrong} against a live {n}-agent core) instead of "
                             f"raising. The safety net this wrapper relies on did not fire.")
        except CausalRoutingError:
            print(f"  decision_mask_from_core(core, {wrong}) correctly REJECTED the mismatch "
                  f"against a live {n}-agent core")
    finally:
        env.close()

    print(f"\n  CHECK-ONLY: PASS  (team size {n} reaches R2.build_env's GPUFieldConfig, "
          f"and the agent-count safety net fires on a real mismatch)")
    print("  NOTE: this does not exercise a real teacher checkpoint or the full episode "
          "collection loop; it exercises exactly what does not depend on one.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-size", type=int, required=True, choices=SUPPORTED_TEAM_SIZES)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--check-only", action="store_true",
                    help="prove team size reaches env construction; no checkpoint needed")
    ap.add_argument("--pi-a-path", default=None)
    ap.add_argument("--pi-b-path", default=None)
    ap.add_argument("--n-per-pole", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="NON-SCIENTIFIC plumbing smoke: uses disposable 999xxxxx seeds "
                         "(never the frozen collection block) and writes to a _SMOKE-suffixed "
                         "manifest/dir that can never collide with or block the real one. Use "
                         "this to exercise the collection loop against a placeholder "
                         "checkpoint before real teachers exist.")
    args = ap.parse_args()

    if args.check_only:
        return check_only(int(args.team_size), args.device)

    if not (args.pi_a_path and args.pi_b_path):
        raise SystemExit("FAIL-CLOSED: without --check-only, --pi-a-path and --pi-b-path are "
                         "required (there is no default scaled teacher checkpoint).")
    for tag, p in (("pi_A", args.pi_a_path), ("pi_B", args.pi_b_path)):
        if not Path(p).is_file():
            raise SystemExit(f"FAIL-CLOSED: {tag} checkpoint not found: {p}")

    return collect(int(args.team_size), args.device, args.pi_a_path, args.pi_b_path,
                    args.n_per_pole, smoke=args.smoke)


def collect(team_size: int, device: str, pi_a_path: str, pi_b_path: str,
            n_per_pole_override: int | None, smoke: bool = False) -> int:
    """Real scaled collection: pi_A on Pole A, pi_B on Pole B, at the live team size.

    Reads its seed blocks and n_per_pole from the frozen TEACHER_DISTILLATION_{n}V{n}_SPEC.json
    for this team size (preregister-before-collect: this record must already exist and be
    frozen, or this refuses). Fixes two bugs the ORIGINAL 2v2-only collect_distillation_states.py
    would carry unmodified at scale (documented in eval_specialist_crossover_scaled.py's own
    docstring as the same class of bug already found and fixed there):
      1. pole_A_genome() must be called at the LIVE team size, not the 2v2 default.
      2. Pole B needs its size-normalized overlay installed for team_size != 2 -- the 2v2
         script installs {} for pole B, correct ONLY because canonical OP7 is natively
         min_alive=2. At N=6 that omission would silently collect against the WRONG Pole B.
    """
    import hashlib
    import json as _json
    from datetime import datetime, timezone

    import numpy as np
    import torch

    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays,
        pole_A_genome, pole_B_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.causal_supervision import decision_mask_from_core
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    n = int(team_size)
    _propagate_by_name(n)

    SD = PROJECT_ROOT / "artifacts" / "strategic_demand" / "sppo"
    spec_path = SD / f"TEACHER_DISTILLATION_{n}V{n}_SPEC.json"
    if not spec_path.is_file():
        raise SystemExit(f"FAIL-CLOSED: {spec_path.name} not found. Preregister-before-collect: "
                         f"freeze this team size's distillation spec (seed blocks, n_per_pole, "
                         f"output paths) before any row is collected.")
    spec = _json.loads(spec_path.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"FAIL-CLOSED: {spec_path.name} not frozen: {spec.get('status')!r}")

    n_per_pole = int(n_per_pole_override) if n_per_pole_override else int(spec["DATASET"]["n_per_pole"])
    if smoke:
        # Disposable, non-scientific seeds -- NEVER the frozen collection block. A real
        # collection run must never be blockable (or contaminated) by a plumbing smoke.
        seed_a_base, seed_b_base = 99_900_000 + 1000 * n, 99_900_000 + 1000 * n + 500
        # Floor of 10 (not just a cap) -- the holdout rule is (episode % 10 == 9), so ANY
        # n_per_pole below 10 gives an EMPTY holdout by construction, silently breaking every
        # downstream fit-check that reads it (a real seam this smoke mode exists to catch).
        # A caller's smaller --n-per-pole request is honored only above that floor.
        n_per_pole = max(min(n_per_pole, 12), 10)
        OUT_DIR = SD / f"teacher_distillation_{n}v{n}_SMOKE" / "states"
        MANIFEST = SD / f"TEACHER_DISTILLATION_{n}V{n}_DATASET_SMOKE.json"
    else:
        # A retirement amendment (if one exists for this team size) governs seed selection in
        # preference to the base spec -- e.g. TEACHER_DISTILLATION_6V6_SEED_RETIREMENT_AMENDMENT
        # retired 4 seeds a plumbing-smoke mistake spent before --smoke mode existed, and shifted
        # the block to replace them. The original spec is never edited (append-only discipline).
        retirement_path = SD / f"TEACHER_DISTILLATION_{n}V{n}_SEED_RETIREMENT_AMENDMENT.json"
        if retirement_path.is_file():
            retirement = _json.loads(retirement_path.read_text(encoding="utf-8"))
            if not str(retirement.get("status", "")).startswith("FROZEN"):
                raise SystemExit(f"FAIL-CLOSED: {retirement_path.name} not frozen: {retirement.get('status')!r}")
            repl = retirement["REPLACEMENT_BLOCK"]
            seed_a_base = int(str(repl["collection_A"]).split("..")[0])
            seed_b_base = int(str(repl["collection_B"]).split("..")[0])
            print(f"  seed source: {retirement_path.name} (base spec's original block superseded)")
        else:
            seeds_spec = spec["SEEDS"]
            seed_a_base = int(str(seeds_spec["collection_A"]).split("..")[0])
            seed_b_base = int(str(seeds_spec["collection_B"]).split("..")[0])
        OUT_DIR = SD / f"teacher_distillation_{n}v{n}" / "states"
        MANIFEST = SD / f"TEACHER_DISTILLATION_{n}V{n}_DATASET.json"
    SEEDS = {"A": list(range(seed_a_base, seed_a_base + n_per_pole)),
             "B": list(range(seed_b_base, seed_b_base + n_per_pole))}

    if MANIFEST.is_file():
        raise SystemExit(f"REFUSING: {MANIFEST.name} exists; {'smoke rerun' if smoke else 'the dataset is collected once'}")
    if smoke and OUT_DIR.is_dir():
        import shutil
        shutil.rmtree(OUT_DIR)  # smoke dirs are disposable by design; a real dir is never touched

    paths = {"pi_A": Path(pi_a_path), "pi_B": Path(pi_b_path)}
    shas = {name: _sha(p) for name, p in paths.items()}

    probe = R2.build_env(device, SEEDS["A"][0])
    obs_space, act_space = probe.observation_space, probe.action_space
    grid_dim = int(obs_space.spaces["grid"].shape[0])
    probe.close()
    if grid_dim != n:
        raise SystemExit(f"FAIL-CLOSED: probe env grid dim {grid_dim} != team size {n}")

    teachers = {name: load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
                for name, p in paths.items()}
    for name, pol in teachers.items():
        if getattr(pol.model, "uses_latent_strategy", False):
            raise SystemExit(f"REFUSING: {name} is latent-conditioned; teachers must be single-strategy")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"COLLECT DISTILLATION STATES ({n}v{n})  {_now()}  device={device}")
    print(f"  spec {spec_path.name}  n_per_pole={n_per_pole}  seeds A={SEEDS['A'][0]}..{SEEDS['A'][-1]} "
          f"B={SEEDS['B'][0]}..{SEEDS['B'][-1]}\n", flush=True)

    shards, totals = [], {"A": {"episodes": 0, "steps": 0, "decision_rows": 0, "wins": 0},
                          "B": {"episodes": 0, "steps": 0, "decision_rows": 0, "wins": 0}}
    for pole in ("A", "B"):
        teacher = teachers["pi_A" if pole == "A" else "pi_B"]
        for ep_i, seed in enumerate(SEEDS[pole]):
            env = R2.build_env(device, seed)
            core = env.core
            try:
                teacher.reset_strategy()
                core._bt_profile_override = None
                core._sds_opening_hold_steps = 0
                # Both poles resolved at the LIVE team size -- the fix relative to the 2v2-only
                # collector, which only ever needed Pole A's overlay and left Pole B implicit.
                genomes = {"OP6": pole_A_genome(n)} if pole == "A" else (
                    {"OP7": pole_B_genome(n)} if n != 2 else {})
                install_keyed_opponent_overlays(core, genomes)
                key = P0.POLES[pole]
                env.env_method("set_phase", phase_from_tag(key))
                env.env_method("set_next_opponent", "SCRIPTED", key)
                obs = env.reset()
                obs["global_state"] = env.state()
                assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                                           context=f"distill collect {n}v{n} {pole} seed {seed}")
                rows = {k: [] for k in ("grid", "vec", "agent_mask", "mask", "global_state",
                                        "decision_mask", "step")}
                steps, terminal = 0, None
                for t in range(R2.MAX_STEPS):
                    d = decision_mask_from_core(core, n, side="blue")
                    d_np = np.asarray(d.detach().cpu())[0].copy()
                    if d_np.any():
                        rows["grid"].append(np.asarray(obs["grid"])[0].copy())
                        rows["vec"].append(np.asarray(obs["vec"])[0].copy())
                        rows["agent_mask"].append(np.asarray(obs["agent_mask"])[0].copy())
                        rows["mask"].append(np.asarray(obs["mask"])[0].copy())
                        rows["global_state"].append(np.asarray(obs["global_state"])[0].copy())
                        rows["decision_mask"].append(d_np)
                        rows["step"].append(t)
                    action, _ = teacher.predict(obs, deterministic=True)
                    env.step_async(action)
                    obs, _r, done, info = env.step_wait()
                    obs["global_state"] = env.state()
                    steps += 1
                    if bool(np.asarray(done).any()):
                        i0 = info[0] if isinstance(info, (list, tuple)) else info
                        res = (i0 or {}).get("episode_result") or {}
                        terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                        break
                if terminal is None:
                    terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            finally:
                env.close()
            n_rows = len(rows["step"])
            shard = OUT_DIR / f"{pole}_{seed}.npz"
            np.savez_compressed(
                shard,
                grid=np.asarray(rows["grid"], dtype=np.float32),
                vec=np.asarray(rows["vec"], dtype=np.float32),
                agent_mask=np.asarray(rows["agent_mask"], dtype=np.float32),
                mask=np.asarray(rows["mask"], dtype=np.float32),
                global_state=np.asarray(rows["global_state"], dtype=np.float32),
                decision_mask=np.asarray(rows["decision_mask"], dtype=bool),
                step=np.asarray(rows["step"], dtype=np.int32),
                pole=np.full((n_rows,), 0 if pole == "A" else 1, dtype=np.int8),
                episode=np.full((n_rows,), ep_i, dtype=np.int32),
                seed=np.full((n_rows,), seed, dtype=np.int64),
            )
            shards.append({"pole": pole, "episode": ep_i, "seed": seed, "steps": steps,
                           "decision_rows": n_rows, "blue": terminal[0], "red": terminal[1],
                           "file": str(shard.relative_to(PROJECT_ROOT))})
            tt = totals[pole]
            tt["episodes"] += 1; tt["steps"] += steps; tt["decision_rows"] += n_rows
            tt["wins"] += int(terminal[0] > terminal[1])
            if (ep_i + 1) % 16 == 0:
                print(f"  pole {pole}: {ep_i + 1}/{n_per_pole} episodes, "
                      f"{tt['decision_rows']} decision rows so far", flush=True)

    for pole in ("A", "B"):
        tt = totals[pole]
        if tt["decision_rows"] <= 0:
            raise SystemExit(f"REFUSING: pole {pole} produced zero decision-bearing rows")
    MANIFEST.write_text(_json.dumps({
        "record": f"Teacher-distillation state set ({n}v{n})"
                  + (" -- NON-SCIENTIFIC SMOKE" if smoke else ""),
        "status": "SMOKE_NOT_SCIENTIFIC" if smoke else "FROZEN_DATASET", "utc": _now(),
        "implements": f"{spec_path.name}#DATASET", "team_size": n,
        "teachers": {name: {"path": str(p), "sha256": shas[name]} for name, p in paths.items()},
        "seeds": {k: [v[0], v[-1]] for k, v in SEEDS.items()},
        "device": device, "decision_rows_only": True,
        "totals": {p: {**t, "teacher_deployment_win_rate": t["wins"] / max(1, t["episodes"])}
                   for p, t in totals.items()},
        "shards": shards,
    }, indent=2), encoding="utf-8")
    print(f"\n  A: {totals['A']}\n  B: {totals['B']}\n  -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
