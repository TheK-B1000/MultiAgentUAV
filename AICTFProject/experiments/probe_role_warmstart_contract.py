"""One-shot warm-start contract probe for role conditioning vs sealed B_t500k.

Not a gate artifact — local greenlight before relaunch.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import experiments.r2_learned_crossover as R2
from experiments.run_r1_repertoire_training import build_r1_config
from rl.custom_ppo import CustomPPOTrainer
from rl.custom_ppo.checkpoints.validation import _entity_kwargs_for_probe
from rl.custom_ppo.trainer_audit import log_input_dim_contract
from rl.ruleset_identity import build_formal_run_identity

CKPT = (
    ROOT
    / "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_b3_entity_repair_corrected"
    / "ckpts/ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_500000.zip"
)


def main() -> int:
    R2.AGENTS = 4
    env = R2.build_env("cpu", 19100001)
    cfg, _meta = build_r1_config("B")
    cfg.device = "cpu"
    cfg.entity_repair_enabled = True
    cfg.role_conditioning_enabled = True
    cfg.role_hold_ticks = 8
    cfg.load_path = str(CKPT)
    cfg.warm_start_reset_progress = True
    cfg.allow_active_actor_module_migration = True
    cfg.total_timesteps = 4096
    cfg.seed = 19100001
    identity = build_formal_run_identity(env, run_id="role_warmstart_probe")
    trainer = CustomPPOTrainer(
        env,
        cfg,
        learning_rate=float(cfg.learning_rate),
        clip_range=float(cfg.clip_range),
        ent_coef=float(cfg.ent_coef),
        n_epochs=int(cfg.n_epochs),
        batch_size=int(cfg.batch_size),
        value_clip_range=float(getattr(cfg, "clip_range_vf", cfg.clip_range) or cfg.clip_range),
        curriculum=None,
        run_identity=identity,
    )
    log_input_dim_contract(trainer)
    print("pre", tuple(trainer.model.latent_actor.body[0].weight.shape))
    trainer.load(str(CKPT), reset_progress=True)
    print("post", tuple(trainer.model.latent_actor.body[0].weight.shape))
    print("expanded_flag", getattr(trainer.model, "_role_warmstart_expanded", None))

    loss = (trainer.model.latent_actor.body[0].weight ** 2).sum() * 1e-12
    trainer.optimizers.primary.zero_grad(set_to_none=True)
    loss.backward()
    trainer.optimizers.primary.step()
    print("Adam OK")

    B, N = 2, 4
    obs = {
        "grid": torch.zeros(B, N, *trainer.model.grid_shape),
        "vec": torch.zeros(B, N, trainer.model.vec_dim),
        "agent_mask": torch.ones(B, N),
        "mask": torch.ones(B, N * 55),
    }
    ent = _entity_kwargs_for_probe(trainer.model, batch_size=B, device=torch.device("cpu"))
    with torch.no_grad():
        l0 = trainer.model.policy_logits(obs, roles=torch.zeros(B, N), **ent)
        l1 = trainer.model.policy_logits(obs, roles=torch.ones(B, N), **ent)
        diff = float((l0 - l1).abs().max())
        col = float(trainer.model.latent_actor.body[0].weight[:, -1].abs().max())
    print(f"r0_vs_r1_max_logit_diff={diff}")
    print(f"role_col_max_abs={col}")
    env.close()
    if diff > 1e-5:
        raise SystemExit(f"FAIL: r0 vs r1 logit diff {diff}")
    print("WARMSTART_CONTRACT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
