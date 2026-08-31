"""Trajectory-identity policy-gradient regulariser for H-OG-PSP V3.

Implements HOG_PSP_V3_SPEC.json#AMENDMENT_1_COMPONENT_2_MECHANISM.

This is NOT an auxiliary environment reward. D's output never enters reward, returns,
GAE, or critic targets. Mixing it into r_t would change what "good policy" means and
invite the fair objection that the latents separated because we paid them to imitate
the teachers. It is a third, separately auditable loss channel:

    L = L_PPO_task + lambda_ogpsp * L_paired + lambda_tau * L_trajectory_PG

Credit flows by policy gradient, so the frozen discriminator never needs to be
differentiable:

    D(tau) -> trajectory advantage -> log pi_z(a|s) -> private branch

Three choices fixed prospectively, in the spec, before any training behaviour existed:

  * FULL EPISODES. The probe validated D on episodes, not on 32/64/128-step windows.
  * POLE-SPECIFIC D_A and D_B. One shared classifier could rediscover OP6-vs-OP7; the
    probe's confounded comparison scored 1.0000, so the shortcut is trivially available.
  * Clipped target-class log-probability, centred by a per-(latent, pole) baseline.
    D is ~96% accurate, so raw probabilities sit near 0 and 1 and give badly scaled
    credit. Tuning this after seeing training curves is prohibited.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
D_DIR = SD / "hog_psp_v3_discriminators"
D_RECORD = SD / "HOG_PSP_TRAJECTORY_DISCRIMINATORS.json"

# frozen in the spec, before any training behaviour was observed
LOG_PROB_CLIP = -5.0
BASELINE_EMA = 0.1

PI_A, PI_B = 0, 1
LATENT_TO_TARGET = {0: PI_A, 1: PI_B}      # z0 -> A identity, z1 -> B identity
POLE_NAME = {0: "A", 1: "B"}


class TrajectoryIdentityError(RuntimeError):
    """Raised when the regulariser would run against an unverified or wrong target."""


class FrozenDiscriminators:
    """D_A and D_B, loaded once and verified by sha256 against the freeze record."""

    def __init__(self, *, verify: bool = True):
        if not D_RECORD.is_file():
            raise TrajectoryIdentityError(
                "discriminator freeze record missing; run fit_trajectory_discriminators.py")
        record = json.loads(D_RECORD.read_text(encoding="utf-8"))
        if record["status"] != "FROZEN_ARTIFACT":
            raise TrajectoryIdentityError(f"discriminators not frozen: {record['status']!r}")

        self.models: dict[str, dict] = {}
        self.sha: dict[str, str] = {}
        for pole in ("A", "B"):
            path = D_DIR / f"D_{pole}.pkl"
            if not path.is_file():
                raise TrajectoryIdentityError(f"D_{pole} is missing at {path}")
            blob = path.read_bytes()
            sha = hashlib.sha256(blob).hexdigest()
            if verify and sha != record["sha256"][pole]:
                raise TrajectoryIdentityError(
                    f"D_{pole} sha256 mismatch: expected {record['sha256'][pole]}, got {sha}")
            self.models[pole] = pickle.loads(blob)
            self.sha[pole] = sha
        self.record = record
        self._param_fingerprint = self._fingerprint()

    def _fingerprint(self) -> str:
        h = hashlib.sha256()
        for pole in ("A", "B"):
            clf = self.models[pole]["clf"]
            h.update(np.ascontiguousarray(clf.coef_).tobytes())
            h.update(np.ascontiguousarray(clf.intercept_).tobytes())
        return h.hexdigest()

    def assert_still_frozen(self) -> None:
        """D must not have changed. Called after optimizer steps."""
        if self._fingerprint() != self._param_fingerprint:
            raise TrajectoryIdentityError(
                "discriminator parameters changed during training; D must stay frozen, "
                "because a co-adapted D can be satisfied by arbitrary difference")

    def score(self, features: np.ndarray, pole: int, target: int) -> float:
        """Clipped log-probability that this trajectory has the TARGET identity."""
        name = POLE_NAME[int(pole)]
        art = self.models[name]
        x = art["scaler"].transform(features.reshape(1, -1))
        logp = art["clf"].predict_log_proba(x)[0]
        col = art["classes"].index(int(target))
        return float(np.clip(logp[col], LOG_PROB_CLIP, 0.0))


class TrajectoryIdentityRunner:
    """Full-episode identity credit, delivered by policy gradient."""

    def __init__(self, discriminators: FrozenDiscriminators, *, lam: float):
        self.D = discriminators
        self.lam = float(lam)
        self.baseline: dict[tuple[int, int], float] = {}
        self.n_calls = 0
        self.n_episodes = 0
        self.n_by_cell: dict[str, int] = {}
        self.last_loss = 0.0
        self.last_scores: list[float] = []
        self.last_advantages: list[float] = []

    def _advantage(self, score: float, z: int, pole: int) -> float:
        key = (int(z), int(pole))
        if key not in self.baseline:
            self.baseline[key] = score          # first episode contributes zero credit
        adv = score - self.baseline[key]
        self.baseline[key] = (1 - BASELINE_EMA) * self.baseline[key] + BASELINE_EMA * score
        return adv

    def loss(self, model, episodes: list[dict], device: str = "cpu"):
        """Sum over episodes of -A_tau * sum_t log pi(a_t | s_t, z), averaged.

        Each episode is {obs: dict of arrays, actions, z, pole, features}. Nothing in
        this method reads or writes reward, returns, advantages, or critic targets.
        """
        import torch
        from rl.custom_ppo.strategy_anchor import action_log_prob

        if not episodes:
            raise TrajectoryIdentityError("no episodes supplied; the runner would be a no-op")

        self.n_calls += 1
        self.last_scores, self.last_advantages = [], []
        total = None
        for ep in episodes:
            z, pole = int(ep["z"]), int(ep["pole"])
            target = LATENT_TO_TARGET[z]
            score = self.D.score(ep["features"], pole, target)
            adv = self._advantage(score, z, pole)
            self.last_scores.append(score)
            self.last_advantages.append(adv)

            t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
            obs = {k: t(v, torch.float32) for k, v in ep["obs"].items()}
            per_head = action_log_prob(
                model, obs, t(ep["actions"], torch.long),
                z_idx=t(np.full(len(ep["actions"]), z), torch.long))

            mask = t(ep["obs"]["agent_mask"], torch.float32)
            n_heads, n_agents = per_head.shape[1], mask.shape[1]
            m = mask.repeat_interleave(n_heads // n_agents, dim=1)
            logp_sum = (per_head * m).sum() / m.sum().clamp_min(1.0)

            term = -adv * logp_sum
            total = term if total is None else total + term

            self.n_episodes += 1
            cell = f"z{z}|{POLE_NAME[pole]}"
            self.n_by_cell[cell] = self.n_by_cell.get(cell, 0) + 1

        out = self.lam * total / len(episodes)
        self.last_loss = float(out.detach())
        return out

    def telemetry(self) -> dict:
        return {
            "calls": self.n_calls,
            "episodes_scored": self.n_episodes,
            "by_cell": dict(sorted(self.n_by_cell.items())),
            "baselines": {f"z{z}|{POLE_NAME[p]}": round(v, 4)
                          for (z, p), v in sorted(self.baseline.items())},
            "last_loss": self.last_loss,
            "lambda": self.lam,
            "log_prob_clip": LOG_PROB_CLIP,
            "baseline_ema": BASELINE_EMA,
            "unit_of_credit": "full episode",
        }


def episode_features(obs_vec: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """The probe's frozen feature map, imported rather than reimplemented."""
    import experiments.probe_teacher_trajectory_separability as P
    return P.featurise({"vec": obs_vec, "act": actions})
