"""Portable team-size launcher for nominal 4v4 / 6v6 POLE SPECIALIST PPO training.

Produces pi_A (Pole A) and pi_B (Pole B) at a given team size -- the first stage of the
route that succeeded at 2v2:

    strategic demand -> specialists -> direct latent distillation -> crossover eval

WHY THIS WRAPPER EXISTS
-----------------------
``run_r1_repertoire_training.py`` resolves the poles through ``pole_A_genome()`` /
``pole_B_genome()``. Those default to ``n_agents=2`` so that all 57 existing 2v2 callers
stay byte-for-byte unchanged. Without this wrapper a "4v4 specialist" would therefore train
in a 4-agent environment while being given the 2v2 strategic pole: the run would look
perfectly healthy and answer the wrong scientific question. That is the same class of bug as
the C7 dual-AGENTS trap, and it is why every check below FAILS CLOSED on the LIVE resolved
profile rather than on a config field.

Team size reaches the run through exactly two paths, and both are asserted:
  1. ``AGENTS`` on the G0 module chain -> ``build_g0_v5_config`` -> ``cfg.max_blue_agents``
  2. ``cfg.max_blue_agents`` -> the R1 seam -> ``pole_{A,B}_genome(N)`` -> live BT profile

PRODUCTION TRAINING IS GATED ON CERTIFICATION
---------------------------------------------
A size may only be trained for production once
``STRATEGIC_DEMAND_<N>v<N>_CERTIFICATION.json`` exists with VERDICT == CERTIFIED. Without it
this script refuses unless ``--smoke`` (which is non-scientific and writes nothing eligible).

This launcher NEVER starts distillation, evaluation or robustness testing.

Run:
  python experiments/train_specialist_scale.py --team-size 4 --policy A --seed <SEED> --device cuda
  python experiments/train_specialist_scale.py --team-size 4 --policy A --seed 99900044 --smoke --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_scale import (  # noqa: E402
    SMOKE_SEED_MAX,
    SMOKE_SEED_MIN,
    _git_dirty,
    _git_sha,
    _now,
    _propagate_team_size,
)

SUPPORTED_TEAM_SIZES = (4, 6)
SMOKE_TIMESTEPS = 5_000
SD = PROJECT_ROOT / "artifacts" / "strategic_demand" / "sppo"
BASE_KEY = {"A": "OP6", "B": "OP7"}


#: Certification records in PRECEDENCE ORDER, most authoritative first.
#: GUARD_DISTRIBUTED_V2 supersedes v1 because v1's GUARD probe carried a MEASURED defect
#: (DEFENDER_STACKING_DIAGNOSTIC.json: every defender received the identical target, so
#: defensive capacity did not scale with defender count). Both records are retained; only
#: which one GATES production training changes. This is precedence, not deletion -- the v1
#: verdict stays on disk and in the history exactly as recorded.
#: CONFIRMATORY_REDESIGN_N192 supersedes both at 4v4: after GUARD_DISTRIBUTED_V2 also failed
#: (exploratory learned-specialist crossover confirmed a genuine Δ_B reversal, not a probe
#: defect), 4v4 was rebuilt on a new Pole-B2 (4V4_CONFIRMATORY_REDESIGN_C2_SPEC.json) and
#: properly powered (n=192, matching this Pole-A genome's own 2v2 precedent) rather than the
#: n=64 every prior 4v4 attempt used. Both older records are retained unmodified.
#: POLE_B3_3_N192 supersedes CONFIRMATORY_REDESIGN_N192 at 4v4: the C2/B2-1 construction
#: certified cleanly (scripted probe) but its own learned-specialist crossover eval FAILED
#: (CONFIRMATORY_REDESIGN_C2_4V4_SPECIALIST_CROSSOVER_EVAL_RESULT.json). B3-3 is a NEW pole
#: construction (coordinated 2v1 flanking) selected specifically because it also cleared a
#: learned-contestability screen B2-1 lacked (4V4_POLE_B3_DIFFICULTY_SCREEN.json). Both older
#: certification records are retained unmodified; only which one GATES production changes.
_CERT_PRECEDENCE = (
    "STRATEGIC_DEMAND_{n}v{n}_POLE_B3_3_N192_CERTIFICATION.json",
    "STRATEGIC_DEMAND_{n}v{n}_CONFIRMATORY_REDESIGN_N192_CERTIFICATION.json",
    "STRATEGIC_DEMAND_{n}v{n}_GUARD_DISTRIBUTED_V2_CERTIFICATION.json",
    "STRATEGIC_DEMAND_{n}v{n}_CERTIFICATION.json",
)


def _certification_verdict(n: int) -> tuple[str, Path]:
    """Resolve the authoritative certification verdict for team size ``n``.

    Returns the FIRST record found in precedence order, so a corrected-probe certification
    governs over a superseded one. If none exists, reports MISSING against the v1 path.
    """
    for tmpl in _CERT_PRECEDENCE:
        p = SD / tmpl.format(n=n)
        if not p.is_file():
            continue
        try:
            return str(json.loads(p.read_text(encoding="utf-8")).get("VERDICT", "UNKNOWN")), p
        except Exception:  # noqa: BLE001
            return "UNREADABLE", p
    return "MISSING", SD / _CERT_PRECEDENCE[-1].format(n=n)


def assert_live_pole_matches_team_size(env, policy: str, n: int) -> dict:
    """FAIL CLOSED on the LIVE resolved behaviour-tree profile, not on a config field."""
    from experiments.opponent_spec import pole_A_genome, pole_B_genome

    core = env.core
    want = pole_A_genome(n) if policy == "A" else pole_B_genome(n)
    detail = {"policy": policy, "base_key": BASE_KEY[policy],
              "expected_overlay": dict(want.overlay or {})}

    resolved = None
    for attr in ("_bt_resolved_profile_tensors",):
        fn = getattr(core, attr, None)
        if callable(fn):
            try:
                resolved = fn()
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"FAIL-CLOSED: could not read the live resolved profile: {e}")
    if resolved is None:
        raise SystemExit("FAIL-CLOSED: core exposes no _bt_resolved_profile_tensors(); the "
                         "live pole cannot be verified and this run must not proceed.")

    got = resolved.get("min_alive_for_defender")
    try:
        got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"FAIL-CLOSED: unreadable min_alive_for_defender in the live "
                         f"profile ({got!r}): {e}")
    detail["live_min_alive_for_defender"] = got_val
    if got_val != n:
        raise SystemExit(
            f"FAIL-CLOSED: the LIVE pole resolves min_alive_for_defender={got_val} but the "
            f"team size is {n}. This run would train a {n}v{n} specialist against the "
            f"{got_val}-agent pole definition. Refusing.")
    return detail


def _verify_live_pole(cfg, policy: str, n: int) -> dict:
    """Build a throwaway env, install the pole overlay, and assert the LIVE profile."""
    from experiments.opponent_spec import (install_keyed_opponent_overlays, pole_A_genome,
                                           pole_B_genome)
    from rl.training.env_factory import build_training_env

    env = None
    try:
        env = build_training_env(cfg, initial_phase="phase1",
                                 initial_opponent_tag=BASE_KEY[policy])
        core = env.core
        core._bt_profile_override = None
        core._sds_opening_hold_steps = 0
        genomes = {"OP6": pole_A_genome(n)} if policy == "A" else {}
        if n != 2:
            genomes["OP7"] = pole_B_genome(n)
        install_keyed_opponent_overlays(core, genomes)
        detail = assert_live_pole_matches_team_size(env, policy, n)
        detail["installed_overlay_keys"] = sorted(genomes)
        detail["grid_agent_dim"] = int(env.observation_space.spaces["grid"].shape[0])
        if detail["grid_agent_dim"] != n:
            raise SystemExit(f"FAIL-CLOSED: live env grid agent dim "
                             f"{detail['grid_agent_dim']} != team size {n}")
        return detail
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Nominal 4v4/6v6 pole-specialist PPO launcher.")
    ap.add_argument("--team-size", type=int, required=True, choices=SUPPORTED_TEAM_SIZES)
    ap.add_argument("--policy", required=True, choices=("A", "B"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="NON-SCIENTIFIC short run; requires a 999xxxxx seed")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--exploratory-spec", default=None,
                    help="path to a FROZEN spec with arm=EXPLORATORY. The ONLY way to train "
                         "at a size whose strategic demand is NOT_CERTIFIED. Artifacts are "
                         "routed to an exploratory_ prefix and labelled so they can never be "
                         "mistaken for confirmatory records.")
    ap.add_argument("--confirmatory-redesign-spec", default=None,
                    help="path to a FROZEN spec with arm=CONFIRMATORY_REDESIGN and "
                         "confirmatory=true. Lets ONE named policy in a new, not-yet-certified "
                         "pole redesign start training before the OTHER pole's certification "
                         "screen finishes, when the spec explicitly states that policy's own "
                         "pole is unchanged from an already-frozen definition and so does not "
                         "depend on the pending pole. Unlike --exploratory-spec this is NOT an "
                         "exploratory bypass: artifacts land in the normal production directory "
                         "and are labelled arm=CONFIRMATORY_REDESIGN (pending), because if the "
                         "joint certification later passes, this IS the confirmatory checkpoint "
                         "-- not a separate run to be repeated. If the joint certification never "
                         "passes, this checkpoint simply never reaches confirmatory status; nothing "
                         "here fabricates a certification that did not happen.")
    ap.add_argument("--pole-b-genome-json", default=None,
                    help="path to a JSON SDSGenome candidate for Pole B (e.g. a certified "
                         "B2 from a confirmatory-redesign track), used instead of the "
                         "canonical pole_B_genome(N). REQUIRED whenever --policy B and the "
                         "GOVERNING certification record for this team size is itself a "
                         "confirmatory-redesign record (name contains 'CONFIRMATORY_REDESIGN') "
                         "-- otherwise the run would silently train against the OLD, "
                         "uncertified canonical Pole B while the certification banner claims "
                         "CERTIFIED.")
    ap.add_argument("--run-label-suffix", default="",
                    help="appended verbatim to the artifact label/checkpoint directory (e.g. "
                         "'_b3'), so a NEW pole-redesign track never collides with an older "
                         "track's already-sealed checkpoints at the same team size. Discovered "
                         "as a real gap when the B3 track's first dry-run hit the C2 track's "
                         "still-present pi_A_specialist_4v4/pi_B_specialist_4v4 directories "
                         "('already holds N checkpoint(s); refusing to overwrite'). Those "
                         "checkpoints are referenced by frozen records "
                         "(CONFIRMATORY_REDESIGN_C2_PI_{A2,B2}_FROZEN.json) and must never be "
                         "renamed or overwritten -- a NEW track takes a new label instead. "
                         "Empty by default so every existing invocation is unaffected.")
    ap.add_argument("--sibling-sep-lambda", type=float, default=0.0,
                    help="specialization-preserving sibling separation coefficient. 0 = OFF "
                         "(structurally absent). See 4V4_B3_SPECIALIZATION_PRESERVING_SPEC.json.")
    ap.add_argument("--sibling-ckpt", default="",
                    help="path to frozen opposite-pole specialist zip used as stop-grad sibling")
    ap.add_argument("--sibling-ckpt-sha256", default="",
                    help="optional expected sha256 of --sibling-ckpt (fail-closed on mismatch)")
    ap.add_argument("--sibling-sep-dataset", default="",
                    help="path to disagreement NPZ or directory containing disagreement_rows.npz")
    ap.add_argument("--role-pres-lambda", type=float, default=0.0,
                    help="role-preservation coefficient. 0 = OFF (structurally absent). "
                         "See 4V4_B3_ROLE_PRESERVATION_SPEC.json.")
    ap.add_argument("--role-pres-targets", default="",
                    help="path to frozen role_targets.json from collect_role_targets_4v4_b3.py")
    ap.add_argument("--role-pres-style", default="",
                    help="GUARD or BREACH — which frozen target this specialist should match")
    ap.add_argument("--resume", default="",
                    help="resume from an existing checkpoint zip in this run's ckpt dir "
                         "(crash recovery). Sets load_path, appends metrics, and allows "
                         "the pre-existing zip set that a fresh launch would refuse.")
    args = ap.parse_args()

    n, policy, seed = int(args.team_size), args.policy, int(args.seed)
    is_smoke = bool(args.smoke)

    # ---- exploratory arm: separately gated, separately labelled, never a bypass ----------
    exploratory = None
    if args.exploratory_spec:
        sp = Path(args.exploratory_spec)
        if not sp.is_file():
            raise SystemExit(f"FAIL-CLOSED: --exploratory-spec not found: {sp}")
        exploratory = json.loads(sp.read_text(encoding="utf-8"))
        if not str(exploratory.get("status", "")).startswith("FROZEN"):
            raise SystemExit(f"FAIL-CLOSED: exploratory spec is not frozen: "
                             f"{exploratory.get('status')!r}")
        if exploratory.get("arm") != "EXPLORATORY" or exploratory.get("confirmatory", True):
            raise SystemExit("FAIL-CLOSED: --exploratory-spec must carry arm=EXPLORATORY and "
                             "confirmatory=false; a confirmatory spec cannot be used to bypass "
                             "the certification gate")
        declared = (exploratory.get("TRAINING", {}).get("seeds") or {})
        key = f"pi_{policy}_{n}v{n}"
        if key not in declared:
            raise SystemExit(f"FAIL-CLOSED: exploratory spec does not preregister {key}; it "
                             f"declares {sorted(declared)}. This spec does not cover this "
                             f"team size / policy.")
        if not is_smoke and int(declared[key]) != seed:
            raise SystemExit(f"FAIL-CLOSED: exploratory spec preregisters seed "
                             f"{declared[key]} for {key}, but --seed={seed}. A production "
                             f"exploratory run must use the preregistered seed exactly.")

    # ---- confirmatory redesign: one named policy may start before the OTHER pole's own -----
    # certification screen finishes, when its own pole is frozen and unchanged. Not a
    # certification bypass in the exploratory sense -- confirmatory=true, normal artifact path.
    redesign = None
    if args.confirmatory_redesign_spec:
        if args.exploratory_spec:
            raise SystemExit("FAIL-CLOSED: --exploratory-spec and --confirmatory-redesign-spec "
                             "are mutually exclusive arms")
        sp = Path(args.confirmatory_redesign_spec)
        if not sp.is_file():
            raise SystemExit(f"FAIL-CLOSED: --confirmatory-redesign-spec not found: {sp}")
        redesign = json.loads(sp.read_text(encoding="utf-8"))
        if not str(redesign.get("status", "")).startswith("FROZEN"):
            raise SystemExit(f"FAIL-CLOSED: confirmatory-redesign spec is not frozen: "
                             f"{redesign.get('status')!r}")
        if redesign.get("arm") != "CONFIRMATORY_REDESIGN" or not redesign.get("confirmatory", False):
            raise SystemExit("FAIL-CLOSED: --confirmatory-redesign-spec must carry "
                             "arm=CONFIRMATORY_REDESIGN and confirmatory=true")
        declared_r = (redesign.get("TRAINING", {}).get("seeds") or {})
        key = f"pi_{policy}_{n}v{n}"
        if key not in declared_r:
            raise SystemExit(f"FAIL-CLOSED: confirmatory-redesign spec does not preregister "
                             f"{key}; it declares {sorted(declared_r)}.")
        if not is_smoke and int(declared_r[key]) != seed:
            raise SystemExit(f"FAIL-CLOSED: confirmatory-redesign spec preregisters seed "
                             f"{declared_r[key]} for {key}, but --seed={seed}.")
        unchanged = redesign.get("POLES_UNCHANGED_FROM_FROZEN", [])
        if policy not in unchanged:
            raise SystemExit(
                f"FAIL-CLOSED: confirmatory-redesign spec does not list pole {policy} in "
                f"POLES_UNCHANGED_FROM_FROZEN ({unchanged}). This bypass exists ONLY so a "
                f"policy whose OWN pole is already frozen/unchanged can start before the "
                f"other pole's screen finishes -- policy {policy}'s pole is not attested as "
                f"unchanged, so this spec does not authorize starting it early.")

    if is_smoke and not (SMOKE_SEED_MIN <= seed <= SMOKE_SEED_MAX):
        raise SystemExit(f"FAIL-CLOSED: --smoke requires a seed in "
                         f"[{SMOKE_SEED_MIN}, {SMOKE_SEED_MAX}], got {seed}")
    if not is_smoke and SMOKE_SEED_MIN <= seed <= SMOKE_SEED_MAX:
        raise SystemExit(f"FAIL-CLOSED: seed {seed} is reserved for non-scientific smokes")

    verdict, cert_path = _certification_verdict(n)
    if not is_smoke and verdict != "CERTIFIED" and exploratory is None and redesign is None:
        raise SystemExit(
            f"FAIL-CLOSED: {n}v{n} strategic demand is {verdict} ({cert_path.name}). "
            f"Production specialist training is gated on CERTIFIED: a pole pair that does not "
            f"demand different behaviour cannot support complementary specialists, and "
            f"training them would spend GPU hours on an unanswerable question. "
            f"(An EXPLORATORY arm may proceed only via --exploratory-spec <frozen spec>; a "
            f"named policy in a pending redesign may proceed only via "
            f"--confirmatory-redesign-spec <frozen spec>.)")
    if exploratory is not None:
        print("!" * 78)
        print(f"!!  EXPLORATORY ARM  --  {n}v{n} strategic demand is {verdict}; this run is NOT")
        print(f"!!  confirmatory. Governed by {Path(args.exploratory_spec).name}.")
        print(f"!!  Artifacts are prefixed exploratory_ and labelled arm=EXPLORATORY.")
        print("!" * 78, flush=True)
    if redesign is not None:
        print("!" * 78)
        print(f"!!  CONFIRMATORY REDESIGN (PENDING)  --  {n}v{n} joint strategic demand is "
              f"{verdict}; pole {policy} is attested unchanged and starts ahead of the other "
              f"pole's certification screen. Governed by "
              f"{Path(args.confirmatory_redesign_spec).name}.")
        print(f"!!  This checkpoint becomes confirmatory ONLY IF the joint certification "
              f"later passes.")
        print("!" * 78, flush=True)

    # ---- Pole B genome source: fail closed if the GOVERNING certification is itself a -----
    # confirmatory-redesign record. That record certified a SPECIFIC candidate genome, not
    # canonical OP7; training "policy B" via the default pole_B_genome(N) would silently
    # answer a different, uncertified question while the banner above still says CERTIFIED.
    pole_b_override = None
    if policy == "B":
        if args.pole_b_genome_json:
            pbp = Path(args.pole_b_genome_json)
            if not pbp.is_file():
                raise SystemExit(f"FAIL-CLOSED: --pole-b-genome-json not found: {pbp}")
            from experiments.sds_genome import SDSGenome
            from experiments.opponent_spec import _with_full_team_defender_gate
            pole_b_override = _with_full_team_defender_gate(
                SDSGenome.from_dict(json.loads(pbp.read_text(encoding="utf-8"))), n)
            print(f"  Pole B SOURCE  candidate genome {pole_b_override.genome_id!r} from "
                  f"{pbp.name} (NOT canonical pole_B_genome({n}))", flush=True)
        elif not is_smoke and "CONFIRMATORY_REDESIGN" in cert_path.name:
            raise SystemExit(
                f"FAIL-CLOSED: the governing certification for {n}v{n} is a confirmatory-"
                f"redesign record ({cert_path.name}), which certified a SPECIFIC Pole-B "
                f"candidate genome, not canonical OP7. --policy B requires "
                f"--pole-b-genome-json naming that candidate, or this run would silently "
                f"train against the wrong, uncertified Pole B.")

    try:
        import torch
        torch.set_num_threads(max(1, int(args.threads)))
    except Exception:  # noqa: BLE001
        pass

    touched = _propagate_team_size(n)
    print(f"[specialist_scale] team size {n} propagated to {len(touched)} module(s)")

    import experiments.run_r1_repertoire_training as R

    # The R1 seam asserts seed/budget against POLICIES, so the scaled run declares its own
    # entry there rather than bypassing the guard.
    spec = dict(R.POLICIES[policy])
    spec["seed"] = seed
    if is_smoke and args.total_timesteps is None:
        spec["steps"] = SMOKE_TIMESTEPS
    if args.total_timesteps is not None:
        spec["steps"] = int(args.total_timesteps)
    _prefix = "exploratory_" if exploratory is not None else ""
    _suffix = str(args.run_label_suffix or "")
    spec["label"] = (f"{_prefix}smoke_pi_{policy}_specialist_{n}v{n}{_suffix}" if is_smoke
                     else f"{_prefix}pi_{policy}_specialist_{n}v{n}{_suffix}")
    R.POLICIES[policy] = spec

    cfg, contract = R.build_r1_config(policy)
    _root = (f"exploratory_scale_{n}v{n}_specialists" if exploratory is not None
             else f"scale_{n}v{n}_specialists")
    art = PROJECT_ROOT / "artifacts" / _root / spec["label"]
    cfg.run_tag = spec["label"]
    cfg.device = str(args.device)
    cfg.checkpoint_dir = str(art / "ckpts")
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")
    if is_smoke:
        cfg.formal_run = False
        cfg.periodic_checkpoint_steps = max(1, int(cfg.total_timesteps) // 2)

    sep_lam = float(args.sibling_sep_lambda or 0.0)
    if sep_lam > 0.0:
        if not args.sibling_ckpt or not args.sibling_sep_dataset:
            raise SystemExit(
                "FAIL-CLOSED: --sibling-sep-lambda > 0 requires --sibling-ckpt and "
                "--sibling-sep-dataset"
            )
        cfg.sibling_sep_lambda = sep_lam
        cfg.sibling_sep_ckpt = str(args.sibling_ckpt)
        cfg.sibling_sep_ckpt_sha256 = str(args.sibling_ckpt_sha256 or "")
        cfg.sibling_sep_dataset = str(args.sibling_sep_dataset)
        # Frozen SPEC defaults.
        cfg.sibling_sep_cadence = 4
        cfg.sibling_sep_batch_size = 64
        # Pin expected hashes for the vanilla B3 siblings when not overridden.
        if not cfg.sibling_sep_ckpt_sha256:
            _known = {
                "final_pi_A_specialist_4v4_b3.zip":
                    "011fbc0b43460fffda28b1c30ea105586ce8030511c495e5433d53d4a608a1b0",
                "final_pi_B_specialist_4v4_b3.zip":
                    "57d354e10c56aab6f548faa43a2c76360e03c5c8840ef7ff079b4cbe5c809aa8",
            }
            cfg.sibling_sep_ckpt_sha256 = _known.get(Path(cfg.sibling_sep_ckpt).name, "")
    elif args.sibling_ckpt or args.sibling_sep_dataset:
        raise SystemExit(
            "FAIL-CLOSED: sibling ckpt/dataset provided but --sibling-sep-lambda is 0; "
            "refusing a silently disabled intervention"
        )

    role_lam = float(args.role_pres_lambda or 0.0)
    if role_lam > 0.0:
        if sep_lam > 0.0:
            raise SystemExit(
                "FAIL-CLOSED: role preservation and sibling separation are mutually "
                "exclusive (JSD SPEC is RETIRED_UNUSED)"
            )
        if not args.role_pres_targets or not args.role_pres_style:
            raise SystemExit(
                "FAIL-CLOSED: --role-pres-lambda > 0 requires --role-pres-targets and "
                "--role-pres-style"
            )
        style = str(args.role_pres_style).upper()
        if style not in ("GUARD", "BREACH"):
            raise SystemExit(f"FAIL-CLOSED: --role-pres-style must be GUARD or BREACH, got {style!r}")
        expected = "GUARD" if policy == "A" else "BREACH"
        if style != expected:
            raise SystemExit(
                f"FAIL-CLOSED: policy {policy} must use style {expected}, got {style}"
            )
        if not Path(args.role_pres_targets).is_file():
            raise SystemExit(f"FAIL-CLOSED: role targets missing: {args.role_pres_targets}")
        cfg.role_pres_lambda = role_lam
        cfg.role_pres_targets = str(args.role_pres_targets)
        cfg.role_pres_style = style
        cfg.role_pres_cadence = 4
        cfg.role_pres_temperature = 0.5
    elif args.role_pres_targets or args.role_pres_style:
        raise SystemExit(
            "FAIL-CLOSED: role targets/style provided but --role-pres-lambda is 0; "
            "refusing a silently disabled intervention"
        )

    if int(getattr(cfg, "max_blue_agents", -1)) != n:
        raise SystemExit(f"FAIL-CLOSED: cfg.max_blue_agents={getattr(cfg,'max_blue_agents',None)} "
                         f"!= team size {n}; AGENTS propagation did not reach build_r1_config")

    ck = Path(cfg.checkpoint_dir)
    resume_path = str(args.resume or "").strip()
    if resume_path:
        rp = Path(resume_path)
        if not rp.is_file():
            raise SystemExit(f"FAIL-CLOSED: --resume not found: {rp}")
        # Must live under this run's checkpoint dir (no silent cross-run warm-start).
        try:
            rp.resolve().relative_to(ck.resolve())
        except ValueError:
            raise SystemExit(
                f"FAIL-CLOSED: --resume {rp} is not under this run's checkpoint dir {ck}"
            )
        cfg.load_path = str(rp)
        cfg.fresh_metrics_csv = False
        print(f"  RESUME from {rp.name} (metrics will append; total_timesteps stays "
              f"{int(cfg.total_timesteps):,})")
    else:
        existing = sorted(ck.glob("*.zip")) if ck.is_dir() else []
        if existing:
            raise SystemExit(
                f"FAIL-CLOSED: {ck} already holds {len(existing)} checkpoint(s); "
                f"refusing to overwrite. Choose another seed, move them aside, or "
                f"pass --resume <ckpt.zip> for crash recovery."
            )

    sha, dirty = _git_sha(), _git_dirty()
    print("=" * 78)
    print(f"SPECIALIST_SCALE  pi_{policy}  {n}v{n}   "
          f"{'[SMOKE - NON-SCIENTIFIC]' if is_smoke else '[PRODUCTION]'}"
          f"{'  [EXPLORATORY ARM - NOT CONFIRMATORY]' if exploratory is not None else ''}"
          f"{'  [CONFIRMATORY REDESIGN - PENDING JOINT CERT]' if redesign is not None else ''}"
          f"{'  [RESUME]' if resume_path else ''}")
    print("=" * 78)
    print(f"  utc              {_now()}")
    print(f"  git sha          {sha}{'  (DIRTY)' if dirty else ''}")
    print(f"  pole             {policy}  (base {BASE_KEY[policy]})")
    print(f"  certification    {verdict}  ({cert_path.name})")
    print(f"  team size        {n}v{n}")
    print(f"  seed             {seed}")
    print(f"  device           {cfg.device}")
    print(f"  total timesteps  {int(cfg.total_timesteps):,}")
    if resume_path:
        print(f"  resume           {cfg.load_path}")
    print(f"  opponent pool    {getattr(cfg, 'opponent_pool', None)}  "
          f"fixed={getattr(cfg, 'fixed_opponent_tag', None)}")
    print(f"  max_blue_agents  {cfg.max_blue_agents}")
    print(f"  checkpoint dir   {cfg.checkpoint_dir}")
    if float(getattr(cfg, "sibling_sep_lambda", 0.0) or 0.0) > 0.0:
        print(f"  sibling_sep      lambda={cfg.sibling_sep_lambda}  "
              f"ckpt={cfg.sibling_sep_ckpt}  dataset={cfg.sibling_sep_dataset}")
    if float(getattr(cfg, "role_pres_lambda", 0.0) or 0.0) > 0.0:
        print(f"  role_pres        lambda={cfg.role_pres_lambda}  "
              f"style={cfg.role_pres_style}  targets={cfg.role_pres_targets}")
    print("=" * 78, flush=True)

    # FAIL CLOSED on the LIVE resolved pole, before any step. Builds a throwaway env,
    # installs the same overlay the R1 seam installs, and reads the behaviour tree's own
    # resolved tensors -- the only authority that cannot be fooled by an unapplied override.
    live = _verify_live_pole(cfg, policy, n)
    print("  LIVE POLE CHECK: PASS")
    for k, v in live.items():
        print(f"    {k}: {v}")

    if args.dry_run:
        # A dry run starts nothing, so it must leave nothing behind. Writing the manifest
        # here would create a production-named artifact claiming the full step budget with no
        # training behind it -- later indistinguishable from an aborted production run.
        print("  --dry-run: config resolved and checks passed. NOT training, NOT writing "
              "a run manifest.")
        return 0

    art.mkdir(parents=True, exist_ok=True)
    (art / "run_manifest.json").write_text(json.dumps({
        "record": "train_specialist_scale run manifest", "utc": _now(),
        "smoke_non_scientific": is_smoke, "team_size": n, "policy": policy,
        "pole_base": BASE_KEY[policy], "seed": seed, "device": cfg.device,
        "git_sha": sha, "git_dirty": dirty,
        "total_timesteps": int(cfg.total_timesteps),
        "certification_verdict": verdict,
        "checkpoint_dir": cfg.checkpoint_dir,
        "recipe_inherited_from": "experiments/run_r1_repertoire_training.py::build_r1_config",
        "pole_b_source": (f"CANDIDATE_OVERRIDE:{pole_b_override.genome_id}"
                         if pole_b_override is not None else "canonical_pole_B_genome"),
        "arm": ("EXPLORATORY" if exploratory is not None
               else "CONFIRMATORY_REDESIGN_PENDING" if redesign is not None
               else "CONFIRMATORY"),
        "confirmatory": exploratory is None,
        "confirmatory_pending_joint_certification": redesign is not None,
        "exploratory_spec": (Path(args.exploratory_spec).name if exploratory is not None
                             else None),
        "confirmatory_redesign_spec": (Path(args.confirmatory_redesign_spec).name
                                       if redesign is not None else None),
        "distillation_started_by_this_script": False,
        "evaluation_started_by_this_script": False,
        "sibling_sep_lambda": float(getattr(cfg, "sibling_sep_lambda", 0.0) or 0.0),
        "sibling_sep_ckpt": str(getattr(cfg, "sibling_sep_ckpt", "") or ""),
        "sibling_sep_dataset": str(getattr(cfg, "sibling_sep_dataset", "") or ""),
        "role_pres_lambda": float(getattr(cfg, "role_pres_lambda", 0.0) or 0.0),
        "role_pres_targets": str(getattr(cfg, "role_pres_targets", "") or ""),
        "role_pres_style": str(getattr(cfg, "role_pres_style", "") or ""),
        "resume_from": (str(cfg.load_path) if getattr(cfg, "load_path", None) else None),
        "implements_spec": (
            "4V4_B3_ROLE_PRESERVATION_SPEC.json"
            if float(getattr(cfg, "role_pres_lambda", 0.0) or 0.0) > 0.0
            else (
                "4V4_B3_SPECIALIZATION_PRESERVING_SPEC.json"
                if float(getattr(cfg, "sibling_sep_lambda", 0.0) or 0.0) > 0.0
                else None
            )
        ),
    }, indent=2), encoding="utf-8")

    # R.run_policy(policy) rebuilds its own config from build_r1_config and would discard the
    # overrides above (device, smoke budget, artifact paths). Replicate its body instead,
    # keeping the piece that actually matters: pre_rollout_env_setup installs the pole overlay
    # and asserts the live opponent. Omitting that hook would train a "specialist" against a
    # bare opponent with NO pole overlay -- the exact failure this wrapper exists to prevent.
    from functools import partial

    print("  starting specialist PPO ...", flush=True)
    R.orchestrate_training_run(
        cfg,
        pre_rollout_env_setup=partial(
            R.configure_r1_live_environment,
            policy=policy,
            config_contract=contract,
            pole_b_genome_override=pole_b_override,
        ),
    )
    print("\n  training returned. This script does NOT start distillation or evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
