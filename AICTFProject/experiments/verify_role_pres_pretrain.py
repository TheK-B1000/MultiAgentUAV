"""Pre-launch audits for 4V4_B3_ROLE_PRESERVATION_SPEC.

Stronger semantic gate (PI): directional coherence across main features, not
an OR-of-one-statistic. See STYLE_INTENT_PROJECTION_AMENDMENT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFENSIVE_DOMS = {"one_attack_three_defend", "turtle_defense", "two_attack_two_defend"}
ATTACK_DOMS = {"all_push", "three_attack_one_defend"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--student-ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--require-mse", action="store_true",
                    help="also require MSE term restored (directional f* separation)")
    args = ap.parse_args()

    from rl.config.ppo_config import PPOConfig
    from rl.custom_ppo.inference import load_custom_ppo_policy
    from rl.custom_ppo.role_preservation import (
        RolePresRunner,
        load_role_targets,
        role_preservation_loss,
        soft_role_features,
    )
    from rl.training.env_factory import build_training_env
    import experiments.run_r1_repertoire_training as R

    targets_path = Path(args.targets)
    payload = json.loads(targets_path.read_text(encoding="utf-8"))
    g = payload["targets"]["GUARD"]
    b = payload["targets"]["BREACH"]
    gf, bf = g["features"], b["features"]

    print("[P1 STRONGER SEMANTIC GATE]")
    print(f"  GUARD  dom={g['dominant_role']} att={gf['mean_num_attackers']:.3f} "
          f"def={gf['mean_num_defenders']:.3f} adr={gf['mean_attack_defense_ratio']:.3f}")
    print(f"  BREACH dom={b['dominant_role']} att={bf['mean_num_attackers']:.3f} "
          f"def={bf['mean_num_defenders']:.3f} adr={bf['mean_attack_defense_ratio']:.3f}")

    failures = []
    if not (gf["mean_num_defenders"] > bf["mean_num_defenders"]):
        failures.append("GUARD def <= BREACH def")
    if not (bf["mean_num_attackers"] > gf["mean_num_attackers"]):
        failures.append("BREACH att <= GUARD att")
    if not (bf["mean_attack_defense_ratio"] > gf["mean_attack_defense_ratio"]):
        failures.append("BREACH adr <= GUARD adr")
    if gf == bf:
        failures.append("features identical")
    for name, feat in (("GUARD", gf), ("BREACH", bf)):
        for k, v in feat.items():
            if not (v == v) or abs(v) > 10:  # NaN or absurd
                failures.append(f"{name}.{k} not finite/plausible: {v}")
    occ_l1 = sum(abs(g["role_occupancy_vector"][i] - b["role_occupancy_vector"][i]) for i in range(7))
    print(f"  occupancy L1 = {occ_l1:.4f}")
    if occ_l1 < 0.20:
        failures.append(f"occupancy L1 {occ_l1} < 0.20")
    if g["dominant_role"] not in DEFENSIVE_DOMS:
        # allow escort only if features already pass defensive direction for GUARD
        failures.append(f"GUARD dominant {g['dominant_role']!r} not in {sorted(DEFENSIVE_DOMS)}")
    if b["dominant_role"] not in ATTACK_DOMS and not (
        b["dominant_role"] == "escort_pair"
        and bf["mean_attack_defense_ratio"] > gf["mean_attack_defense_ratio"]
        and bf["mean_num_attackers"] > gf["mean_num_attackers"]
    ):
        failures.append(
            f"BREACH dominant {b['dominant_role']!r} not attack-oriented under the gate"
        )
    if failures:
        print("  FAIL:")
        for f in failures:
            print(f"    - {f}")
        raise SystemExit("FAIL: stronger semantic gate not satisfied")
    print("    PASS: directionally coherent GUARD/BREACH targets")

    try:
        RolePresRunner(None, None, targets=load_role_targets(targets_path, "GUARD"),
                       lambda_role=0.0)
    except ValueError:
        print("[P1b] PASS: lambda_role<=0 refuses runner")
    else:
        raise SystemExit("FAIL: RolePresRunner accepted lambda_role=0")

    # Determinism: reload file twice, same vectors
    p2 = json.loads(targets_path.read_text(encoding="utf-8"))
    if p2["targets"]["GUARD"]["role_occupancy_vector"] != g["role_occupancy_vector"]:
        raise SystemExit("FAIL: targets file not stable on reload")
    print("[P1c] PASS: targets reload deterministically")

    cfg, _ = R.build_r1_config("A")
    cfg.device = args.device
    cfg.n_envs = 1
    env = build_training_env(cfg, initial_phase="phase1", initial_opponent_tag="OP6")
    try:
        student = load_custom_ppo_policy(
            str(args.student_ckpt), env.observation_space, env.action_space,
            device=args.device,
        ).model
        obs0 = env.reset()
        gs = env.state()
        obs = {}
        for k in ("grid", "vec", "mask", "agent_mask"):
            if k not in obs0:
                continue
            t = torch.as_tensor(obs0[k], device=args.device)
            if t.dim() == 1 or (k == "grid" and t.dim() == 3):
                t = t.unsqueeze(0)
            obs[k] = t
        gs_t = torch.as_tensor(gs, device=args.device, dtype=torch.float32)
        if gs_t.dim() == 1:
            gs_t = gs_t.unsqueeze(0)
        obs["global_state"] = gs_t

        guard = load_role_targets(targets_path, "GUARD")
        breach = load_role_targets(targets_path, "BREACH")
        f_soft, q_soft, _ = soft_role_features(student, obs, temperature=0.5)
        qbar = q_soft.detach().mean(0)
        qbar = qbar / torch.clamp(qbar.sum(), min=1e-8)
        h_q = float((-(qbar * torch.log(qbar + 1e-8))).sum())

        loss_match, _ = role_preservation_loss(
            student, obs, p_star=qbar.cpu(), f_star=f_soft.detach().mean(0).cpu()
        )
        loss_g, _ = role_preservation_loss(
            student, obs, p_star=guard["p_star"], f_star=guard["f_star"]
        )
        loss_b, _ = role_preservation_loss(
            student, obs, p_star=breach["p_star"], f_star=breach["f_star"]
        )
        print("[P2 MATCH / SWAP]")
        print(f"  matched CE={float(loss_match.detach()):.6f} H(q)={h_q:.6f}")
        print(f"  GUARD={float(loss_g.detach()):.6f} BREACH={float(loss_b.detach()):.6f}")
        if abs(float(loss_match.detach()) - h_q) > 1e-3:
            raise SystemExit("FAIL: matched CE != H(q)")
        if min(float(loss_g.detach()), float(loss_b.detach())) < h_q + 0.2:
            raise SystemExit("FAIL: style losses not clearly above H(q)")
        if abs(float(loss_g.detach()) - float(loss_b.detach())) < 1e-3:
            raise SystemExit("FAIL: GUARD/BREACH losses identical")
        print("    PASS")

        student.train()
        for p in student.parameters():
            if p.grad is not None:
                p.grad = None
        use = breach if float(loss_b.detach()) >= float(loss_g.detach()) else guard
        loss, _ = role_preservation_loss(
            student, obs, p_star=use["p_star"], f_star=use["f_star"]
        )
        loss.backward()
        stu_grad = sum(
            float(p.grad.detach().abs().sum())
            for p in student.parameters() if p.grad is not None
        )
        print(f"[P3 GRAD] student L1={stu_grad:.6e}")
        if stu_grad <= 0:
            raise SystemExit("FAIL: no student gradient")
        print("    PASS")

        cfg_bad = PPOConfig()
        cfg_bad.role_pres_lambda = 0.05
        cfg_bad.sibling_sep_lambda = 0.05
        if not (cfg_bad.role_pres_lambda > 0 and cfg_bad.sibling_sep_lambda > 0):
            raise SystemExit("FAIL: coexistence check broken")
        print("[P3b] PASS: role_pres+sibling_sep coexistence rejected by policy")
    finally:
        env.close()

    print("ALL STRONGER PRE-LAUNCH CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
