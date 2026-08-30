"""Oracle-Gated Paired Specialist Preservation (OG-PSP) rehearsal bank.

Implements ORACLE_GATED_PAIRED_SPECIALIST_PRESERVATION_SPEC.json.

V1 trained only the preferred latent per resolved state, leaving the other
unconstrained, so the shared actor could satisfy the whole objective with a
latent-independent state->action map. It measurably did: z0-z1 JSD 0.0051 bits
against 0.3919 bits of available teacher contrast.

OG-PSP presents BOTH specialist targets at the SAME state:

    Delta != 0   (s, z0) -> pi_A(s)   AND   (s, z1) -> pi_B(s)
    Delta == 0   no paired rehearsal

regardless of which teacher is locally preferred. Where the teachers disagree -- two
thirds of resolved states -- no latent-independent policy can satisfy both targets.

The oracle gate is REINTERPRETED, not removed: Delta != 0 now means "specialist
identity matters here", not "this teacher wins here".

Deliberately separate from rl/oracle_rehearsal.py: V1's result is frozen and its code
path must not shift underneath it.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "artifacts" / "strategic_demand" / "sppo" / \
    "ORACLE_GATED_PAIRED_SPECIALIST_PRESERVATION_SPEC.json"
LEGACY = ROOT / "artifacts" / "strategic_demand" / "stratified_regime_data" / "seed_shards"
V2 = ROOT / "artifacts" / "strategic_demand" / "sppo" / \
    "oracle_gated_k2_v2_bank_data" / "seed_shards"

LEGACY_LO, LEGACY_HI = 10_700_001, 10_700_096
V2_LO, V2_HI = 11_000_001, 11_000_320

A_PREFERRED, B_PREFERRED, NOT_ESTABLISHED = -1, 1, 0
# frozen pairing: z0 always carries specialist A, z1 always specialist B
LATENT_TO_TEACHER = {0: "pi_A", 1: "pi_B"}


class PairedRehearsalError(RuntimeError):
    """Raised when the bank would violate the frozen OG-PSP spec."""


@dataclass
class PairedBank:
    """Every eligible state yields TWO targets, one per latent."""

    obs: dict[str, np.ndarray]
    delta: np.ndarray
    pi_a_action: np.ndarray
    pi_b_action: np.ndarray
    cell: np.ndarray
    seed: np.ndarray
    rng_seed: int = 37

    eligible: np.ndarray = field(init=False)
    state_exposures: Counter = field(default_factory=Counter, init=False)
    latent_exposures: Counter = field(default_factory=Counter, init=False)
    tied_exposures: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.eligible = np.nonzero(self.delta != 0)[0]
        if self.eligible.size == 0:
            raise PairedRehearsalError("no resolvable states in the bank")
        self._rng = np.random.default_rng(self.rng_seed)

    # ------------------------------------------------------------------- facts
    @property
    def n_states(self) -> int:
        return int(len(self.delta))

    @property
    def n_eligible(self) -> int:
        return int(self.eligible.size)

    def head_mask(self, idx: np.ndarray) -> np.ndarray:
        """(n, n_heads) bool: heads the anchor loss actually trains."""
        am = self.obs["agent_mask"][idx].astype(bool)
        n_heads = self.pi_a_action.shape[1]
        return np.repeat(am, n_heads // am.shape[1], axis=1)

    def teachers_disagree(self, idx: np.ndarray) -> np.ndarray:
        """(n,) bool: at least one DECISION-MASKED head differs between teachers.

        Masked, because disagreement on a locked head is not usable supervision --
        the same masking error that made an earlier ad-hoc pass misread the data.
        """
        d = (self.pi_a_action[idx] != self.pi_b_action[idx]) & self.head_mask(idx)
        return d.any(axis=1)

    def composition(self) -> dict:
        e = self.eligible
        dis = self.teachers_disagree(e)
        return {
            "states": self.n_states,
            "eligible": self.n_eligible,
            "A_preferred": int((self.delta[e] < 0).sum()),
            "B_preferred": int((self.delta[e] > 0).sum()),
            "tied_excluded_from_sampling": int((self.delta == 0).sum()),
            "eligible_with_teacher_disagreement": int(dis.sum()),
            "teacher_disagreement_frac": round(float(dis.mean()), 4),
            "targets_per_batch_state": 2,
        }

    # ---------------------------------------------------------------- sampling
    def sample(self, n_states: int) -> dict:
        """Draw states uniformly, then emit BOTH latent targets for each.

        Returns 2*n rows. ``state_id`` lets a caller prove the two rows of a pair
        refer to the same underlying state.
        """
        n = min(int(n_states), self.n_eligible)
        pick = self._rng.choice(self.eligible, size=n, replace=False)
        if np.any(self.delta[pick] == 0):
            raise PairedRehearsalError("a tied state entered a paired rehearsal batch")

        for i in pick:
            self.state_exposures[int(i)] += 1
        self.latent_exposures[0] += n
        self.latent_exposures[1] += n

        both = np.concatenate([pick, pick])                      # same states, twice
        z_idx = np.concatenate([np.zeros(n, np.int64), np.ones(n, np.int64)])
        target = np.concatenate([self.pi_a_action[pick], self.pi_b_action[pick]])
        return {
            "state_id": both,
            "obs": {k: v[both] for k, v in self.obs.items()},
            "z_idx": z_idx,
            "teacher_action": target,
            "delta": self.delta[both],
            "cell": self.cell[both],
            "n_states": n,
            "n_pairs": 2 * n,
            "teachers_disagree": self.teachers_disagree(pick),
        }

    # --------------------------------------------------------------- telemetry
    def telemetry(self) -> dict:
        counts = np.array([self.state_exposures.get(int(i), 0) for i in self.eligible])
        total = int(counts.sum())
        return {
            "state_exposures_total": total,
            "replay_factor": round(total / max(1, self.n_eligible), 2),
            "mean_exposures_per_state": round(float(counts.mean()), 2) if counts.size else 0.0,
            "min_exposures": int(counts.min()) if counts.size else 0,
            "max_exposures": int(counts.max()) if counts.size else 0,
            "never_sampled": int((counts == 0).sum()),
            "latent_exposures": {f"z{k}": v for k, v in sorted(self.latent_exposures.items())},
            "latent_exposure_balance": (
                "exactly equal by construction: every sampled state trains both latents"),
            "tied_exposures": self.tied_exposures,
        }

    def assert_invariants(self) -> None:
        """Both OG-PSP invariants, measured rather than assumed."""
        if self.tied_exposures != 0:
            raise PairedRehearsalError(
                f"tied states received {self.tied_exposures} exposures; a state where "
                "preference is not established must exert zero strategic pressure")
        z0, z1 = self.latent_exposures.get(0, 0), self.latent_exposures.get(1, 0)
        if z0 != z1:
            raise PairedRehearsalError(
                f"paired supervision must expose both latents equally, got z0={z0} z1={z1}; "
                "unequal counts mean the pairing broke and one latent was left unconstrained "
                "-- the exact V1 defect this treatment exists to fix")


def _read_range(data_dir: Path, lo: int, hi: int, acc: dict) -> int:
    n = 0
    for s in range(lo, hi + 1):
        p = Path(data_dir) / f"seed_{s}.npz"
        if not p.is_file():
            continue
        with np.load(p, allow_pickle=False) as z:
            d = ((z["branch_pi_B_blue"].astype(np.int64) - z["branch_pi_B_red"].astype(np.int64))
                 - (z["branch_pi_A_blue"].astype(np.int64) - z["branch_pi_A_red"].astype(np.int64)))
            acc["grid"].append(z["branch_obs_grid"][:, 0])
            acc["vec"].append(z["branch_obs_vec"][:, 0])
            acc["agent_mask"].append(z["branch_obs_agent_mask"][:, 0])
            acc["mask"].append(z["branch_obs_mask"][:, 0])
            acc["pi_a"].append(z["branch_pi_A_action"])
            acc["pi_b"].append(z["branch_pi_B_action"])
            acc["delta"].append(d)
            acc["cell"].extend(str(c) for c in z["branch_cell"])
            acc["seed"].append(np.full(len(d), s, dtype=np.int64))
            n += 1
    return n


def load_paired_bank(*, include_v2: bool = True, rng_seed: int = 37,
                     verify_spec: bool = True, min_eligible: int | None = None) -> PairedBank:
    """Load the OG-PSP bank: legacy FIT plus the expanded V2 collection."""
    if verify_spec:
        if not SPEC.is_file():
            raise PairedRehearsalError("OG-PSP spec is not frozen")
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
        if spec["status"] != "FROZEN -- SPEC_FROZEN_BEFORE_IMPLEMENTATION":
            raise PairedRehearsalError(f"spec is not frozen: {spec['status']!r}")
        t = spec["THE_TREATMENT"]["resolved_state_delta_nonzero"]
        if t["z0"] != "-> pi_A(s)" or t["z1"] != "-> pi_B(s)":
            raise PairedRehearsalError("spec pairing drifted from z0->pi_A / z1->pi_B")
        if min_eligible is None:
            min_eligible = int(spec["BANK"]["minimum_total_eligible"])

    keys = ("grid", "vec", "agent_mask", "mask", "pi_a", "pi_b", "delta", "cell", "seed")
    acc: dict[str, list] = {k: [] for k in keys}
    n_legacy = _read_range(LEGACY, LEGACY_LO, LEGACY_HI, acc)
    n_new = _read_range(V2, V2_LO, V2_HI, acc) if include_v2 else 0
    if not acc["delta"]:
        raise PairedRehearsalError("paired bank load produced zero branch states")

    delta = np.concatenate(acc["delta"])
    bank = PairedBank(
        obs={k: np.concatenate(acc[k]) for k in ("grid", "vec", "agent_mask", "mask")},
        delta=delta,
        pi_a_action=np.concatenate(acc["pi_a"]),
        pi_b_action=np.concatenate(acc["pi_b"]),
        cell=np.array(acc["cell"]), seed=np.concatenate(acc["seed"]),
        rng_seed=rng_seed)
    bank.n_legacy_shards, bank.n_new_shards = n_legacy, n_new

    if min_eligible is not None and bank.n_eligible < min_eligible:
        raise PairedRehearsalError(
            f"REFUSING: {bank.n_eligible} eligible states < frozen minimum {min_eligible}")
    return bank


def paired_rehearsal_loss(model, batch: dict, device: str = "cpu"):
    """MEAN over all (state, latent) pairs -- never a sum.

    Paired supervision emits two targets per state. Summing would roughly double the
    rehearsal gradient relative to V1 and confound the mechanism change with a
    strength change, so the spec fixes this as a mean and lambda keeps its meaning.
    ``anchor_loss`` already averages over the batch it is given, and the batch here is
    all pairs, so the mean is over pairs by construction.
    """
    import torch
    from rl.custom_ppo.strategy_anchor import anchor_loss

    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    obs = {k: t(v, torch.float32) for k, v in batch["obs"].items()}
    return anchor_loss(
        model, obs,
        t(batch["teacher_action"], torch.long),
        decision_mask=t(batch["obs"]["agent_mask"], torch.float32).bool(),
        z_idx=t(batch["z_idx"], torch.long))
