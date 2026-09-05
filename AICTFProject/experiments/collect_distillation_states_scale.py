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
    args = ap.parse_args()

    if args.check_only:
        return check_only(int(args.team_size), args.device)

    if not (args.pi_a_path and args.pi_b_path):
        raise SystemExit("FAIL-CLOSED: without --check-only, --pi-a-path and --pi-b-path are "
                         "required (there is no default scaled teacher checkpoint).")
    for tag, p in (("pi_A", args.pi_a_path), ("pi_B", args.pi_b_path)):
        if not Path(p).is_file():
            raise SystemExit(f"FAIL-CLOSED: {tag} checkpoint not found: {p}")

    raise SystemExit(
        "NOT YET WIRED: a full scaled collection run additionally needs its own frozen "
        "TEACHER_DISTILLATION_SPEC-equivalent (dataset size, output paths, seed block "
        "disjoint from every 2v2 block) before it writes rows, per this project's "
        "preregister-before-collect rule. That spec does not exist yet because no scaled "
        "specialist checkpoint exists yet for it to reference. Use --check-only until "
        "specialist training clears its own gate.")


if __name__ == "__main__":
    raise SystemExit(main())
