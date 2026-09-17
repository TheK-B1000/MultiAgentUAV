"""Executable t=0 warm-start contract: pi_role(r=0)=pi_role(r=1)=pi_B500k.

Loads the sealed B_t500k zip, expands role-facing Linears with a zero column,
and checks max_logit_diff / argmax_diff. Does NOT train.

Run:
  python experiments/probe_rule_role_warmstart_contract.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from gymnasium import spaces

from rl.custom_ppo.checkpoints.archive import _torch_load_checkpoint
from rl.custom_ppo.checkpoints.state_dict import (
    _expand_role_conditioning_linears,
    _load_model_state_dict_compat,
)
from rl.custom_ppo.policy import SharedActorCentralizedCritic

PARENT = Path(
    "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_b3_entity_repair_corrected/"
    "ckpts/ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_500000.zip"
)
OUT = Path("artifacts/strategic_demand/sppo/RULE_ROLE_WARMSTART_CONTRACT_PROBE.json")
N, N_CH, ROWS, COLS, VEC = 4, 7, 20, 20, 20
K_T, K_E, F = 3, 4, 6


def _spaces():
    obs = spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(N, N_CH, ROWS, COLS), dtype=np.float32),
        "vec": spaces.Box(-1.0, 1.0, shape=(N, VEC), dtype=np.float32),
        "agent_mask": spaces.Box(0.0, 1.0, shape=(N,), dtype=np.float32),
        "mask": spaces.Box(0.0, 1.0, shape=(N * (5 + 50),), dtype=np.float32),
    })
    act = spaces.MultiDiscrete([5, 50] * N)
    return obs, act


def _rand_obs(batch=8, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "grid": torch.tensor(rng.random((batch, N, N_CH, ROWS, COLS)), dtype=torch.float32),
        "vec": torch.tensor(rng.uniform(-1, 1, (batch, N, VEC)), dtype=torch.float32),
        "agent_mask": torch.ones(batch, N),
        "mask": torch.ones(batch, N * 55),
    }


def _rand_entity(batch=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return dict(
        teammates=torch.randn(batch, N, K_T, F, generator=g),
        teammates_valid=torch.ones(batch, N, K_T, dtype=torch.bool),
        enemies=torch.randn(batch, N, K_E, F, generator=g),
        enemies_valid=torch.ones(batch, N, K_E, dtype=torch.bool),
    )


def _stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, int]:
    diff = (a - b).abs()
    return float(diff.max()), int((a.argmax(-1) != b.argmax(-1)).sum())


def main() -> int:
    ckpt = (ROOT / PARENT).resolve()
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    payload = _torch_load_checkpoint(str(ckpt), map_location="cpu")
    raw = payload.get("model_state_dict") or payload.get("state_dict")
    if raw is None:
        raise SystemExit(f"no model_state_dict in {ckpt}")
    sd0 = {k: v.detach().cpu().clone() for k, v in raw.items()}

    obs_s, act_s = _spaces()
    kw = dict(
        strategy_encoder_enabled=False,
        latent_k=0,
        entity_repair_enabled=True,
        entity_hidden_dim=32,
    )
    base = SharedActorCentralizedCritic(obs_s, act_s, role_conditioning_enabled=False, **kw)
    role_m = SharedActorCentralizedCritic(obs_s, act_s, role_conditioning_enabled=True, **kw)

    _load_model_state_dict_compat(base, {k: v.clone() for k, v in sd0.items()})
    # Expand on a *copy* so the 148-wide source tensors stay intact for the base load.
    sd_role = _expand_role_conditioning_linears({k: v.clone() for k, v in sd0.items()}, role_m)
    _load_model_state_dict_compat(role_m, sd_role)

    w = role_m.latent_actor.body[0].weight
    role_col_nnz = int(torch.count_nonzero(w[:, -1]).item())
    actor_in_base = int(base.latent_actor.body[0].weight.shape[1])
    actor_in_role = int(w.shape[1])

    obs = _rand_obs()
    ent = _rand_entity()
    gs = torch.zeros(obs["grid"].shape[0], base.global_state_dim)

    with torch.no_grad():
        base_logits = base.policy_logits(obs, **ent)
        l0 = role_m.policy_logits(obs, roles=torch.zeros(8, N), **ent)
        l1 = role_m.policy_logits(obs, roles=torch.ones(8, N), **ent)
        lmix = role_m.policy_logits(
            obs, roles=torch.tensor([[0.0, 1.0, 0.0, 1.0]] * 8), **ent
        )
        v_base = base.values(gs)
        v0 = role_m.values(gs, team_roles=torch.zeros(8, N))
        v1 = role_m.values(gs, team_roles=torch.ones(8, N))

    pairs = {
        "r0_vs_r1": _stats(l0, l1),
        "r0_vs_B500k": _stats(l0, base_logits),
        "r1_vs_B500k": _stats(l1, base_logits),
        "rmix_vs_B500k": _stats(lmix, base_logits),
        "value_r0_vs_B500k": (float((v0 - v_base).abs().max()), 0),
        "value_r1_vs_B500k": (float((v1 - v_base).abs().max()), 0),
    }
    max_logit = max(pairs[k][0] for k in ("r0_vs_r1", "r0_vs_B500k", "r1_vs_B500k", "rmix_vs_B500k"))
    argmax_sum = sum(pairs[k][1] for k in ("r0_vs_r1", "r0_vs_B500k", "r1_vs_B500k", "rmix_vs_B500k"))
    passed = (
        actor_in_base == 148
        and actor_in_role == 149
        and role_col_nnz == 0
        and argmax_sum == 0
        and max_logit < 1e-5
    )
    out = {
        "probe_id": "RULE_ROLE_WARMSTART_CONTRACT_PROBE",
        "parent_ckpt": str(PARENT).replace("\\", "/"),
        "parent_sha256": sha,
        "actor_in_base": actor_in_base,
        "actor_in_role": actor_in_role,
        "role_column_nnz": role_col_nnz,
        "contract": "W_role,t0=[W_B500k | 0] => pi(r=0)=pi(r=1)=pi_B500k",
        "max_logit_diff": max_logit,
        "argmax_diff_total": argmax_sum,
        "pairs": {k: {"max_abs_diff": a, "argmax_diff": b} for k, (a, b) in pairs.items()},
        "pass": passed,
        "note": (
            "Dedicated probe; in-trainer Behavioral-equivalence NOT_RUN is expected "
            "because expand mutates the remapped dict before reverse-load into the 148 model."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
