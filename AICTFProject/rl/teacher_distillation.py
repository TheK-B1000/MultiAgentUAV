"""Teacher-guided latent compression: pure distillation of pi_A / pi_B into a fresh K=2 student.

Implements TEACHER_DISTILLATION_SPEC.json#LOSS and #TRAINING.

    L_distill = 0.5 * KL(pi_A(.|s) || pi_theta(.|s,z0)) + 0.5 * KL(pi_B(.|s) || pi_theta(.|s,z1))

Both teachers are queried on the SAME states. KL is per action head over the MASKED
distributions -- the identical legality masking PPO's own update applies
(rl/custom_ppo/strategy_anchor._masked_heads) -- and only decision-boundary agent-heads
(d_ti = [commit_ticks_left_i <= 0]) enter numerator and denominator.

There is no PPO objective, no critic loss, no reward, no advantage anywhere in this module. It
is a standalone supervised objective over the student's ACTOR parameters; the critic is
excluded from the optimizer and verified immobile by the preflight.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable

import torch

LOGIT_FLOOR = -1.0e4   # masked logits may be -inf; clamp so 0 * (-inf) never poisons a gradient


class DistillationError(RuntimeError):
    pass


# --------------------------------------------------------------------------- parameters
def actor_parameters(model: Any) -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if not n.startswith("critic.")]


def critic_parameters(model: Any) -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if n.startswith("critic.")]


# --------------------------------------------------------------------------- heads
def masked_heads(model: Any, obs: Dict[str, torch.Tensor], z_idx: torch.Tensor | None = None):
    from rl.custom_ppo.strategy_anchor import _masked_heads
    return _masked_heads(model, obs, z_idx=z_idx)


def head_logits(model: Any, obs: Dict[str, torch.Tensor],
                z_idx: torch.Tensor | None = None) -> list[torch.Tensor]:
    return [h.logits.clamp(min=LOGIT_FLOOR) for h in masked_heads(model, obs, z_idx=z_idx)]


def kl_per_head(teacher_logits: Iterable[torch.Tensor],
                student_logits: Iterable[torch.Tensor]) -> torch.Tensor:
    """Forward KL(teacher || student) per head. Returns (B, n_heads)."""
    out = []
    for lt, ls in zip(teacher_logits, student_logits):
        logp_t = lt.log_softmax(dim=-1)
        logp_s = ls.log_softmax(dim=-1)
        p_t = logp_t.exp()
        out.append((p_t * (logp_t - logp_s)).sum(dim=-1))
    return torch.stack(out, dim=1)


def jsd_per_head(logits_a: Iterable[torch.Tensor], logits_b: Iterable[torch.Tensor]) -> torch.Tensor:
    out = []
    for la, lb in zip(logits_a, logits_b):
        pa = la.softmax(dim=-1).clamp_min(1e-8)
        pb = lb.softmax(dim=-1).clamp_min(1e-8)
        m = 0.5 * (pa + pb)
        out.append(0.5 * (pa * (pa.log() - m.log())).sum(-1) + 0.5 * (pb * (pb.log() - m.log())).sum(-1))
    return torch.stack(out, dim=1)


def masked_mean(per_head: torch.Tensor, decision_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Average over decision-boundary agent-heads only. (loss, n_heads_in_denominator)."""
    n_heads, n_agents = int(per_head.shape[1]), int(decision_mask.shape[1])
    if n_heads % n_agents:
        raise DistillationError(f"{n_heads} heads do not divide across {n_agents} agents")
    m = decision_mask.to(per_head.dtype).repeat_interleave(n_heads // n_agents, dim=1)
    denom = m.sum()
    if float(denom) <= 0.0:
        return (per_head * 0.0).sum(), denom
    return (per_head * m).sum() / denom, denom


# --------------------------------------------------------------------------- the loss
def distillation_loss(student: Any, teachers: Dict[str, Any], obs: Dict[str, torch.Tensor],
                      decision_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """0.5*KL(pi_A||z0) + 0.5*KL(pi_B||z1), both teachers on the same states."""
    b = int(decision_mask.shape[0])
    dev = decision_mask.device
    with torch.no_grad():
        lt_a = head_logits(teachers["pi_A"], obs)
        lt_b = head_logits(teachers["pi_B"], obs)
    z0 = torch.zeros((b,), dtype=torch.long, device=dev)
    z1 = torch.ones((b,), dtype=torch.long, device=dev)
    ls0 = head_logits(student, obs, z_idx=z0)
    ls1 = head_logits(student, obs, z_idx=z1)
    l_a, n_a = masked_mean(kl_per_head(lt_a, ls0), decision_mask)
    l_b, _ = masked_mean(kl_per_head(lt_b, ls1), decision_mask)
    loss = 0.5 * l_a + 0.5 * l_b
    return loss, {"kl_A": float(l_a.detach()), "kl_B": float(l_b.detach()),
                  "decision_heads": int(n_a)}


@torch.no_grad()
def fidelity_diagnostics(student: Any, teachers: Dict[str, Any], obs: Dict[str, torch.Tensor],
                         decision_mask: torch.Tensor) -> dict:
    """Argmax agreement with each teacher, teacher-teacher disagreement, and student z0/z1 JSD,
    all over decision-boundary heads. Diagnostics and the fit check -- never the crossover gate."""
    b = int(decision_mask.shape[0])
    dev = decision_mask.device
    z0 = torch.zeros((b,), dtype=torch.long, device=dev)
    z1 = torch.ones((b,), dtype=torch.long, device=dev)
    ha = masked_heads(teachers["pi_A"], obs)
    hb = masked_heads(teachers["pi_B"], obs)
    s0 = masked_heads(student, obs, z_idx=z0)
    s1 = masked_heads(student, obs, z_idx=z1)

    def _agree(x, y):
        return torch.stack([(p.logits.argmax(-1) == q.logits.argmax(-1)).float()
                            for p, q in zip(x, y)], dim=1)

    agree_a, _ = masked_mean(_agree(ha, s0), decision_mask)
    agree_b, _ = masked_mean(_agree(hb, s1), decision_mask)
    teacher_disagree, _ = masked_mean(1.0 - _agree(ha, hb), decision_mask)
    kl_a, _ = masked_mean(kl_per_head([h.logits.clamp(min=LOGIT_FLOOR) for h in ha],
                                      [h.logits.clamp(min=LOGIT_FLOOR) for h in s0]), decision_mask)
    kl_b, _ = masked_mean(kl_per_head([h.logits.clamp(min=LOGIT_FLOOR) for h in hb],
                                      [h.logits.clamp(min=LOGIT_FLOOR) for h in s1]), decision_mask)
    jsd_student, _ = masked_mean(jsd_per_head([h.logits.clamp(min=LOGIT_FLOOR) for h in s0],
                                              [h.logits.clamp(min=LOGIT_FLOOR) for h in s1]), decision_mask)
    jsd_teachers, _ = masked_mean(jsd_per_head([h.logits.clamp(min=LOGIT_FLOOR) for h in ha],
                                               [h.logits.clamp(min=LOGIT_FLOOR) for h in hb]), decision_mask)
    return {"agree_z0_vs_piA": float(agree_a), "agree_z1_vs_piB": float(agree_b),
            "teacher_argmax_disagreement": float(teacher_disagree),
            "kl_A": float(kl_a), "kl_B": float(kl_b),
            "student_z0_z1_jsd": float(jsd_student), "teacher_A_B_jsd": float(jsd_teachers)}


# --------------------------------------------------------------------------- student construction
def build_fresh_student(incumbent_path: str, observation_space, action_space, *, seed: int,
                        device: str):
    """The incumbent's architecture (parsed from its own payload cfg by the checkpoint loader),
    random-initialised under ``seed``, NO weights loaded. Refuses unless the parameter names
    and shapes match the incumbent exactly and the weights differ from it."""
    from rl.custom_ppo.checkpoints.archive import read_checkpoint_payload
    from rl.custom_ppo.checkpoints.loader import _architecture_from_metadata
    from rl.custom_ppo.checkpoints.metadata import parse_checkpoint_metadata
    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    payload = read_checkpoint_payload(str(incumbent_path), map_location="cpu")
    metadata = parse_checkpoint_metadata(payload, str(incumbent_path), observation_space, action_space)
    arch = _architecture_from_metadata(metadata, observation_space, action_space)
    torch.manual_seed(int(seed))
    model = SharedActorCentralizedCritic(observation_space, action_space, **arch.model_kwargs).to(device)

    ref = payload["model_state_dict"]
    sd = model.state_dict()
    if list(sd.keys()) != list(ref.keys()):
        raise DistillationError(
            f"fresh student parameter names differ from the incumbent's: "
            f"missing={sorted(set(ref) - set(sd))} extra={sorted(set(sd) - set(ref))}")
    bad = [k for k in sd if tuple(sd[k].shape) != tuple(ref[k].shape)]
    if bad:
        raise DistillationError(f"fresh student shapes differ from the incumbent's at {bad}")
    max_diff = max(float((sd[k].detach().cpu().float() - ref[k].cpu().float()).abs().max())
                   for k in sd if sd[k].numel() > 0)
    if max_diff == 0.0:
        raise DistillationError("fresh student is bit-identical to the incumbent -- that is a "
                                "silent warm start, not a fresh initialisation")
    return model, payload


def write_student_checkpoint(incumbent_payload: dict, model: Any, out_path: str,
                             provenance: dict) -> None:
    """Clone the incumbent's payload envelope (cfg, ruleset, identity, dims, format) with the
    student's weights; drop the incumbent's optimizer/updater state; stamp provenance."""
    new = {k: v for k, v in incumbent_payload.items()
           if k not in ("optimizer_state_dict", "ppo_updater_state")}
    new["model_state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    new["global_step"] = 0
    new["updates_completed"] = 0
    new["last_stats"] = {}
    new["teacher_distillation"] = dict(provenance)
    tmp = f"{out_path}.tmp"
    torch.save(new, tmp)
    os.replace(tmp, out_path)


@torch.no_grad()
def verify_roundtrip(out_path: str, model: Any, observation_space, action_space,
                     obs: Dict[str, torch.Tensor], *, device: str) -> float:
    """Reload through the SAME loader every eval uses and compare masked logits for both z."""
    from rl.custom_ppo import load_custom_ppo_policy
    pol = load_custom_ppo_policy(str(out_path), observation_space, action_space, device=device)
    b = int(next(iter(obs.values())).shape[0])
    worst = 0.0
    for z in (0, 1):
        zt = torch.full((b,), z, dtype=torch.long, device=device)
        a = head_logits(model, obs, z_idx=zt)
        c = head_logits(pol.model, obs, z_idx=zt)
        for x, y in zip(a, c):
            worst = max(worst, float((x - y).abs().max()))
    return worst
