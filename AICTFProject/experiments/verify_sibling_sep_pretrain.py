"""Pre-train integrity checks for 4V4_B3_SPECIALIZATION_PRESERVING_SPEC.

Verifies, before any training seed is spent, four SEPARATE properties:

  P1 SUPPORT SIZE -- the frozen disagreement dataset holds a meaningful number
     of masked states.

  P2 SUPPORT LOCALITY -- *where* the term acts. This is established by the
     dataset, not by the loss: the runner samples exclusively from the frozen
     GUARD/BREACH disagreement rows, whose provenance manifest records that the
     mask was computed from the scripted references and never from the learned
     policies. Off-support states are unreachable by construction.

  P3 LOSS CORRECTNESS -- the divergence implementation is sane: identical
     policies yield exactly zero separation. This says nothing about locality;
     it only shows the loss measures policy difference rather than noise.

  P4 STOP-GRADIENT -- the sibling reference is stationary: no trainable
     parameters, and no gradient reaches it through a student backward. This is
     checked on a genuinely DIFFERENT student (the opposite specialist), because
     a matched pair has zero divergence and would make the check vacuous: zero
     sibling gradient would prove nothing if the student gradient is also zero.

P2 and P3 are deliberately audited as distinct claims. Conflating them would let
a loss that fires everywhere pass by looking correct on a matched pair.

Spends no training seed and no sealed crossover seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sibling-ckpt", required=True)
    ap.add_argument("--cross-ckpt", required=True,
                    help="the OTHER specialist, used as a genuinely different student so "
                         "P4's stop-gradient check is not vacuous")
    ap.add_argument("--min-rows", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    npz = ds_path / "disagreement_rows.npz" if ds_path.is_dir() else ds_path
    if not npz.is_file():
        raise SystemExit(f"FAIL: dataset missing: {npz}")
    ckpt = Path(args.sibling_ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"FAIL: sibling ckpt missing: {ckpt}")
    cross = Path(args.cross_ckpt)
    if not cross.is_file():
        raise SystemExit(f"FAIL: cross ckpt missing: {cross}")

    from rl.custom_ppo.sibling_separation import (
        JSD_MAX_NATS,
        DisagreementDataset,
        SiblingSepRunner,
        sibling_separation_loss,
    )

    # ---- P1: support size --------------------------------------------------------
    ds = DisagreementDataset(str(npz), batch_size=64, seed=7)
    print(f"[P1 SUPPORT SIZE] disagreement rows = {ds.n_rows}  "
          f"(minimum required {args.min_rows})")
    if ds.n_rows < int(args.min_rows):
        raise SystemExit(f"FAIL: too few disagreement rows ({ds.n_rows} < {args.min_rows})")
    print("    PASS: mask contains a meaningful number of disagreement states")

    # ---- P2: support locality (established by the dataset, not the loss) --------
    manifest_path = (ds_path / "collection_manifest.json" if ds_path.is_dir()
                     else npz.parent / "collection_manifest.json")
    if not manifest_path.is_file():
        raise SystemExit(f"FAIL: no collection manifest beside dataset: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"[P2 SUPPORT LOCALITY] mask source = scripted {manifest.get('GUARD')} vs "
          f"{manifest.get('BREACH')}")
    print(f"    seed_base={manifest.get('seed_base')} episodes={manifest.get('n_episodes')} "
          f"rows={manifest.get('n_disagreement_rows')}")
    if int(manifest.get("n_disagreement_rows", -1)) != ds.n_rows:
        raise SystemExit(
            f"FAIL: manifest row count {manifest.get('n_disagreement_rows')} "
            f"disagrees with dataset {ds.n_rows}"
        )
    if not manifest.get("GUARD") or not manifest.get("BREACH"):
        raise SystemExit("FAIL: manifest does not record the scripted reference styles")
    print("    PASS: support is defined by the frozen scripted references; the runner "
          "samples only these rows, so off-support states are unreachable")

    # ---- disabled means structurally absent -------------------------------------
    try:
        SiblingSepRunner(None, None, None, ds, lambda_sep=0.0)
    except ValueError:
        print("[P2b STRUCTURAL ABSENCE] PASS: lambda<=0 refuses runner construction")
    else:
        raise SystemExit("FAIL: SiblingSepRunner accepted lambda_sep=0")

    # ---- build env + two loads of the same frozen policy ------------------------
    from experiments.train_scale import _propagate_team_size

    _propagate_team_size(4)
    import experiments.run_r1_repertoire_training as R
    from rl.custom_ppo.inference import load_custom_ppo_policy
    from rl.training.env_factory import build_training_env

    cfg, _contract = R.build_r1_config("A")
    cfg.device = args.device
    cfg.n_envs = 1
    env = build_training_env(cfg, initial_phase="phase1", initial_opponent_tag="OP6")
    try:
        sibling = load_custom_ppo_policy(
            str(ckpt), env.observation_space, env.action_space, device=args.device
        ).model
        student = load_custom_ppo_policy(
            str(ckpt), env.observation_space, env.action_space, device=args.device
        ).model
        sibling.eval()
        for p in sibling.parameters():
            p.requires_grad_(False)

        obs = ds.sample(device=args.device)

        # ---- P3: loss correctness (NOT a locality claim) ------------------------
        student.eval()
        _loss_identical, tel_identical = sibling_separation_loss(student, sibling, obs)
        print(f"[P3 LOSS CORRECTNESS] identical-policy JSD = {tel_identical['jsd']:.3e} "
              f"(bound {JSD_MAX_NATS:.4f}, decision_heads={tel_identical['decision_heads']:.0f})")
        if tel_identical["decision_heads"] <= 0:
            raise SystemExit("FAIL: no decision-eligible heads in the sampled batch")
        if abs(tel_identical["jsd"]) > 1e-6:
            raise SystemExit("FAIL: separation is non-zero for identical policies")
        print("    PASS: divergence measures policy difference and vanishes on a matched "
              "pair (locality is P2's claim, not this one)")

        # ---- P4: sibling is stop-grad -------------------------------------------
        n_trainable = sum(1 for p in sibling.parameters() if p.requires_grad)
        print(f"[P4 STOP-GRADIENT] sibling trainable parameters = {n_trainable}")
        if n_trainable != 0:
            raise SystemExit("FAIL: sibling still has trainable parameters")

        cross_student = load_custom_ppo_policy(
            str(cross), env.observation_space, env.action_space, device=args.device
        ).model
        cross_student.train()
        loss, tel_cross = sibling_separation_loss(cross_student, sibling, obs)
        loss.backward()
        sib_grad = sum(
            float(p.grad.detach().abs().sum())
            for p in sibling.parameters()
            if p.grad is not None
        )
        stu_grad = sum(
            float(p.grad.detach().abs().sum())
            for p in cross_student.parameters()
            if p.grad is not None
        )
        print(f"    cross-pair JSD = {tel_cross['jsd']:.6f} "
              f"({100.0 * tel_cross['jsd_frac_of_max']:.1f}% of ln 2)")
        print(f"    student grad L1 after backward = {stu_grad:.6e}")
        print(f"    sibling grad L1 after backward = {sib_grad}")
        if tel_cross["jsd"] <= 0.0:
            raise SystemExit("FAIL: cross pair has zero divergence; P4 would be vacuous")
        if stu_grad <= 0.0:
            raise SystemExit("FAIL: no gradient reached the student; P4 would be vacuous")
        if sib_grad != 0.0:
            raise SystemExit("FAIL: gradients flowed into the sibling reference")
        print("    PASS: gradient reaches the student and none reaches the stop-grad sibling")
    finally:
        env.close()

    print("ALL PRE-TRAIN CHECKS PASS (P1 support size, P2 support locality, "
          "P3 loss correctness, P4 stop-gradient)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
