"""Strategic ranking update -- SPPPO V1.

Frozen protocol: artifacts/strategic_demand/sppo/SPPPO_V1_PROTOCOL.json

    L_rank = E_o [ max(0, m - Delta(o)) ]       m = 0.04

    pole A:  Delta_A(o) = V_hat(o, z0, A) - V_hat(o, z1, A)
    pole B:  Delta_B(o) = V_hat(o, z1, B) - V_hat(o, z0, B)

    V_hat(o, z, p) = b + u^T mu_1 + v^T mu_2 + mu_1^T M mu_2
    mu_i           = sum_{a_i} pi_theta(a_i | o, z) e(a_i)

THREE SEPARATE OPERATIONS. The ranking update is its own optimizer step, never a
term inside PPO's clipped-surrogate backward pass:

    PPO actor update      -> learn the task
    teacher rehearsal     -> preserve what GUARD/BREACH do
    strategic ranking     -> preserve which strategy should be preferred

So ``lambda_rank`` scales a standalone loss in its own step. It is NOT a
loss-blending coefficient.

WHY THE HINGE. Once Delta(o) >= m the objective is silent and contributes
exactly zero gradient, leaving PPO free to optimise the task. SPPPO preserves
sufficient strategic ordering rather than maximising separation forever.

BOTH MODES RECEIVE COUNTERFACTUAL PRESSURE. Delta is a *pairwise difference* and
neither V_hat is detached. Detaching the "reference" mode would turn the update
into "improve z0" instead of "widen the z0-vs-z1 gap", which is a different
objective that happens to look similar in a loss curve. ``ranking_loss`` returns
diagnostics that make the asymmetry measurable, and the unit tests assert
gradient reaches BOTH mode paths.

DISABLED MEANS STRUCTURALLY ABSENT. ``RankingRunner`` refuses lambda <= 0. The
lambda_R = 0 control must be the absence of a runner, not a runner multiplying by
zero -- a nominal zero-loss step still mutates optimizer state, advances
counters, and consumes RNG. The SAPPO no-op incident is the precedent.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional

import torch

from rl.scorer.qpsi import N_ACTIONS, QPsi, QPsiConfig

__all__ = [
    "load_frozen_qpsi", "assert_qpsi_excluded_from_optimizer",
    "masked_action_probs", "strategic_contrast", "ranking_loss", "RankingRunner",
]

POLE_A, POLE_B = 0, 1
Z_FOR_POLE = {POLE_A: 0, POLE_B: 1}      # z0 is appropriate on A, z1 on B

# TRUE pole provenance. The rollout buffer already carries ``opponent_id`` as
# trainer-only context (OP1->0 ... OP12->11), written by the collector and never
# read into the actor's observation. Pole A is OP6 (id 5), pole B is OP7 (id 6).
# This is privileged information in the centralized-critic sense: available to
# the training objective, absent from the deployed policy input.
OPPONENT_ID_TO_POLE = {5: POLE_A, 6: POLE_B}      # OP6 -> A, OP7 -> B


def load_frozen_qpsi(path, *, expected_sha256: str | None = None,
                     device: str = "cpu") -> QPsi:
    """Load Q_psi immutable: requires_grad=False everywhere, SHA verified.

    Q_psi is frozen for the entire SPPPO run. If the scorer could move, the
    ruler would shift while the policy learns to satisfy it, and the causal
    claim Phase 0 earned would be lost.
    """
    path = Path(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise RuntimeError(
            f"Q_psi sha256 {sha[:16]} != expected {expected_sha256[:16]}. "
            "Refusing to run SPPPO against a scorer that is not the frozen one.")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = QPsi(QPsiConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model._frozen_sha256 = sha
    return model


def assert_qpsi_excluded_from_optimizer(qpsi: QPsi, optimizer) -> None:
    """No optimizer parameter group may contain a Q_psi parameter."""
    owned = {id(p) for p in qpsi.parameters()}
    for gi, group in enumerate(optimizer.param_groups):
        for p in group["params"]:
            if id(p) in owned:
                raise RuntimeError(
                    f"optimizer param_group[{gi}] contains a Q_psi parameter; "
                    "the frozen scorer must never be optimised")


def masked_action_probs(model, obs: Dict[str, torch.Tensor],
                        z_idx: torch.Tensor):
    """Per-agent 250-way action distributions under pi_theta(. | o, z).

    Uses the SAME masking the PPO actor update applies:
    ``evaluate_actions()`` masks; ``get_distribution()`` does not. Scoring an
    unmasked distribution here would push the policy through a distribution it
    never actually uses -- the Phase 0 diagnostic defect, but inside a training
    loss where no argmax check would catch it.
    """
    flat = model._mask_logits(model.policy_logits(obs, z_idx=z_idx), obs.get("mask"))
    heads = torch.split(flat, list(model.action_dims), dim=-1)
    if len(heads) != 4:
        raise RuntimeError(f"expected 4 action heads, got {len(heads)}")
    pm1, pw1, pm2, pw2 = (h.softmax(dim=-1) for h in heads)
    n = pm1.shape[0]
    p1 = (pm1[:, :, None] * pw1[:, None, :]).reshape(n, -1)
    p2 = (pm2[:, :, None] * pw2[:, None, :]).reshape(n, -1)
    if p1.shape[-1] != N_ACTIONS:
        raise RuntimeError(f"joint action width {p1.shape[-1]} != {N_ACTIONS}")
    return p1, p2


def strategic_contrast(model, qpsi: QPsi, obs: Dict[str, torch.Tensor],
                       pole: torch.Tensor):
    """Delta(o), oriented so positive always favours the mode that pole wants.

    Returns (delta, v_appropriate, v_other). NEITHER term is detached: the
    gradient must widen the pair, not merely lift one side.
    """
    n = pole.shape[0]
    dev = pole.device
    z0 = torch.zeros(n, dtype=torch.long, device=dev)
    z1 = torch.ones(n, dtype=torch.long, device=dev)

    p1_z0, p2_z0 = masked_action_probs(model, obs, z0)
    p1_z1, p2_z1 = masked_action_probs(model, obs, z1)

    grid, vec, am = obs["grid"], obs["vec"], obs["agent_mask"]
    v_z0 = qpsi.expected_value(grid, vec, am, pole, p1_z0, p2_z0)
    v_z1 = qpsi.expected_value(grid, vec, am, pole, p1_z1, p2_z1)

    on_A = (pole == POLE_A)
    v_apt = torch.where(on_A, v_z0, v_z1)      # z0 on pole A, z1 on pole B
    v_oth = torch.where(on_A, v_z1, v_z0)
    return v_apt - v_oth, v_apt, v_oth


def n_ok(sel) -> bool:
    return bool(int(sel.sum()))


def ranking_loss(model, qpsi: QPsi, obs: Dict[str, torch.Tensor],
                 pole: torch.Tensor, *, margin: float,
                 decision_mask: Optional[torch.Tensor] = None):
    """L_rank = mean over states of max(0, margin - Delta(o)).

    ``decision_mask`` (N,) selects decision-point states, matching the
    distribution Q_psi was fit on. Q_psi never saw locked physics steps.
    """
    delta, v_apt, v_oth = strategic_contrast(model, qpsi, obs, pole)
    hinge = torch.relu(margin - delta)
    if decision_mask is not None:
        w = decision_mask.to(hinge.dtype)
        denom = w.sum().clamp_min(1.0)
        loss = (hinge * w).sum() / denom
        active = ((hinge > 0).to(w.dtype) * w).sum() / denom
        sel = w > 0
    else:
        loss = hinge.mean()
        active = (hinge > 0).to(hinge.dtype).mean()
        sel = torch.ones_like(delta, dtype=torch.bool)

    on_A = (pole == POLE_A) & sel
    on_B = (pole == POLE_B) & sel
    d = delta.detach()
    diag = {
        "loss": float(loss.detach()),
        "activation_rate": float(active),           # fraction with Delta < margin
        "delta_mean": float(d[sel].mean()) if int(sel.sum()) else float("nan"),
        "delta_A_mean": float(d[on_A].mean()) if int(on_A.sum()) else float("nan"),
        "delta_B_mean": float(d[on_B].mean()) if int(on_B.sum()) else float("nan"),
        "n_A": int(on_A.sum()), "n_B": int(on_B.sum()),
        "v_appropriate_mean": float(v_apt.detach()[sel].mean()) if int(sel.sum()) else float("nan"),
        "v_other_mean": float(v_oth.detach()[sel].mean()) if int(sel.sum()) else float("nan"),
        "qpsi_pred_min": float(torch.minimum(v_apt, v_oth).detach().min()) if n_ok(sel) else float("nan"),
        "qpsi_pred_max": float(torch.maximum(v_apt, v_oth).detach().max()) if n_ok(sel) else float("nan"),
    }
    return loss, diag


class RankingRunner:
    """Strategic ranking rehearsal -- one standalone optimizer step.

    Mirrors ``AnchorRunner``'s discipline deliberately:

        PPO actor minibatches execute unchanged
        -> ZERO GRADS
        -> one ranking-only optimizer step, L = lambda_rank * hinge
        -> ZERO GRADS

    The PPO stepper leaves gradients populated after ``step()``, so without an
    explicit zero here PPO's gradients would ride along into the ranking step
    and the "ranking-only" claim would be quietly false.

    Cadence is MEASURED from real optimizer steps, never claimed from config.
    """

    def __init__(self, model, qpsi: QPsi, optimizer, *, lambda_rank: float,
                 margin: float, cadence: int = 1,
                 z_to_pole: Optional[Dict[int, int]] = None,
                 opponent_to_pole: Optional[Dict[int, int]] = None,
                 max_grad_norm: float | None = None, device: str = "cpu"):
        if lambda_rank <= 0.0:
            raise ValueError(
                "RankingRunner must not be constructed with lambda_rank <= 0. "
                "The lambda_R = 0 control means NOT constructing the runner, so "
                "that no optimizer state is mutated, no counter advances, and no "
                "RNG is consumed.")
        if cadence < 1:
            raise ValueError("cadence must be >= 1")
        if any(p.requires_grad for p in qpsi.parameters()):
            raise ValueError("Q_psi must be frozen (requires_grad=False) before use")
        assert_qpsi_excluded_from_optimizer(qpsi, optimizer)

        self.model = model
        self.qpsi = qpsi
        self.optimizer = optimizer
        self.lambda_rank = float(lambda_rank)
        self.margin = float(margin)
        self.cadence = int(cadence)
        # POLE DERIVATION. The rollout batch deliberately carries NO opponent
        # field -- exposing opponent identity to the actor is prohibited. Under
        # the frozen assigned-pole regime (16 x z0|A, 16 x z1|B) the pole of a
        # row is determined by its z, so the map is stated EXPLICITLY here and
        # echoed into telemetry rather than being hardcoded inside the loss.
        # If that env assignment were ever broken, this derivation would score
        # rows against the wrong pole silently -- see LIMITATION in telemetry().
        self.z_to_pole = dict(z_to_pole) if z_to_pole is not None else {0: POLE_A, 1: POLE_B}
        self.opponent_to_pole = (dict(opponent_to_pole) if opponent_to_pole is not None
                                 else dict(OPPONENT_ID_TO_POLE))
        self.n_pole_consistency_checks = 0
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.n_ppo_actor_minibatches = 0
        self.n_ranking_updates = 0
        self.last_loss = float("nan")
        self.last_diag: dict = {}
        self._qpsi_sha = getattr(qpsi, "_frozen_sha256", None)

    def note_ppo_minibatch(self, batch) -> bool:
        """Call once per completed PPO actor minibatch.

        ``batch`` supplies obs / pole / optional decision_mask for the ranking
        states. Returns True iff a ranking step was performed. No trailing
        update is emitted for a partial group -- the ratio is never forced.
        """
        self.n_ppo_actor_minibatches += 1
        if self.n_ppo_actor_minibatches % self.cadence != 0:
            return False
        self._ranking_step(batch)
        return True

    def _sample(self, batch):
        """Build (obs, pole, decision_mask) from a PPO minibatch.

        Pole comes from the TRUE rollout metadata ``opponent_id``, not from the
        latent label. Deriving it from z would make the scorer's pole input a
        restatement of the thing being scored: if env/pole routing ever drifted,
        Q_psi(o, a, WRONG POLE) would be optimised while every cadence and loss
        metric still looked healthy.

        The frozen 16/0/0/16 treatment asserts z0 => A and z1 => B, so any
        disagreement between the latent label and the real opponent ABORTS.

        Accepts a pre-built {"obs","pole"} mapping for unit tests, where the
        pole is supplied explicitly and there is nothing to cross-check.
        """
        if "obs" in batch and "pole" in batch:
            return batch["obs"], batch["pole"], batch.get("decision_mask")
        obs = {"grid": batch["obs_grid"], "vec": batch["obs_vec"],
               "agent_mask": batch["obs_agent_mask"], "mask": batch["obs_mask"]}
        if "opponent_id" not in batch:
            raise RuntimeError(
                "PPO minibatch carries no opponent_id, so the TRUE pole cannot be "
                "established. Refusing to fall back to deriving pole from z: that "
                "would hide a broken env/pole assignment behind healthy metrics.")
        z = batch["z"].long().reshape(-1)
        opp = batch["opponent_id"].long().reshape(-1)

        unknown_z = set(int(v) for v in torch.unique(z).tolist()) - set(self.z_to_pole)
        if unknown_z:
            raise RuntimeError(f"batch contains z values {sorted(unknown_z)} with no pole mapping")
        unknown_o = set(int(v) for v in torch.unique(opp).tolist()) - set(self.opponent_to_pole)
        if unknown_o:
            raise RuntimeError(
                f"batch contains opponent_id {sorted(unknown_o)} outside the frozen "
                f"pole map {self.opponent_to_pole}; SPPPO V1 trains on OP6/OP7 only")

        true_pole = torch.zeros_like(opp)
        for ov, pv in self.opponent_to_pole.items():
            true_pole = torch.where(opp == int(ov), torch.full_like(opp, int(pv)), true_pole)
        z_pole = torch.zeros_like(z)
        for zv, pv in self.z_to_pole.items():
            z_pole = torch.where(z == int(zv), torch.full_like(z, int(pv)), z_pole)

        bad = (true_pole != z_pole)
        if bool(bad.any()):
            i = int(torch.nonzero(bad)[0])
            raise RuntimeError(
                f"z/pole consistency violated on {int(bad.sum())}/{len(z)} rows: "
                f"row {i} has z={int(z[i])} (expects pole {int(z_pole[i])}) but "
                f"opponent_id={int(opp[i])} (true pole {int(true_pole[i])}). The "
                "frozen 16/0/0/16 assignment is broken; aborting rather than "
                "scoring against the wrong pole.")
        self.n_pole_consistency_checks += 1
        return obs, true_pole, batch.get("decision_mask")

    def _ranking_step(self, batch) -> None:
        obs, pole, dmask = self._sample(batch)
        # Clear gradients left over from the preceding PPO minibatch.
        self.optimizer.zero_grad(set_to_none=True)
        raw, diag = ranking_loss(self.model, self.qpsi, obs, pole,
                                 margin=self.margin, decision_mask=dmask)
        loss = self.lambda_rank * raw
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]],
                float(self.max_grad_norm))
        self.optimizer.step()
        # Leave no ranking gradients behind for the next PPO minibatch.
        self.optimizer.zero_grad(set_to_none=True)
        self.n_ranking_updates += 1
        self.last_loss = float(loss.detach())
        self.last_diag = diag

    def assert_qpsi_unchanged(self) -> str:
        """Re-hash the live Q_psi state and compare to the load-time SHA."""
        h = hashlib.sha256()
        for k, v in sorted(self.qpsi.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def telemetry(self) -> dict:
        """Counters for the frozen cadence check. Measured, never assumed."""
        expected = self.n_ppo_actor_minibatches // self.cadence
        return {
            "lambda_rank": self.lambda_rank,
            "margin": self.margin,
            "ranking_cadence": self.cadence,
            "n_ppo_actor_minibatches": self.n_ppo_actor_minibatches,
            "n_ranking_updates": self.n_ranking_updates,
            "expected_complete_groups": expected,
            "complete_group_ratio_is_one": (self.n_ranking_updates == expected),
            "last_ranking_loss": self.last_loss,
            "qpsi_sha256_at_load": self._qpsi_sha,
            "z_to_pole": dict(self.z_to_pole),
            "opponent_to_pole": dict(self.opponent_to_pole),
            "pole_source": "TRUE rollout opponent_id, cross-checked against z",
            "n_pole_consistency_checks": self.n_pole_consistency_checks,
            **{f"last_{k}": v for k, v in self.last_diag.items()},
        }
