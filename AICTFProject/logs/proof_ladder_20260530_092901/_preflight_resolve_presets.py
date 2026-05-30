import sys
from rl.train_ppo import PPOConfig, _apply_training_preset
for name in sys.argv[1:]:
    cfg = _apply_training_preset(PPOConfig(), name)
    print(f'OK {name:<48} K={cfg.latent_k} resample_every={cfg.latent_resample_every_n} lam_p={cfg.latent_lam_p} lam_h={cfg.latent_lam_h} aux_phase={cfg.latent_strategy_aux_predict_phase_coef} aux_ret={cfg.latent_strategy_aux_return_head}')
