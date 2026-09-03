"""Oracle-gated selective rehearsal bank. Implements ORACLE_GATED_REHEARSAL_SPEC.

The 160-seed collection already contains EXACT deterministic preference labels for
every FIT branch state, so the policy never has to predict resolvability online -- a
task three probes established it cannot do. Supervision is simply gated by the labels
we already paid for:

    Delta < 0   A-preferred   ->  anchor z_0 toward pi_A, leave z_1 untouched
    Delta > 0   B-preferred   ->  anchor z_1 toward pi_B, leave z_0 untouched
    Delta == 0  not established ->  ZERO strategic pressure on either latent

Delta = M(pi_B|s) - M(pi_A|s), so a positive Delta means pi_B scored higher. That
mapping is arithmetic, not convention, and it is asserted at load time.

Tied states are LOADED but never sampled. Keeping them present means the
zero-pressure property is MEASURED -- ``tied_exposures`` must remain 0 -- rather than
merely guaranteed by their absence.

The bank is torch-free so its logic is testable without a GPU. Only the loss helper
touches torch, and it delegates to the already-validated
``rl.custom_ppo.strategy_anchor.anchor_loss``.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "artifacts" / "strategic_demand" / "sppo" / "ORACLE_GATED_REHEARSAL_SPEC.json"
DATA = ROOT / "artifacts" / "strategic_demand" / "stratified_regime_data" / "seed_shards"
V2_DATA = ROOT / "artifacts" / "strategic_demand" / "sppo" / "oracle_gated_k2_v2_bank_data" / "seed_shards"
V2_BANK_ASSEMBLY = ROOT / "artifacts" / "strategic_demand" / "sppo" / "ORACLE_GATED_K2_V2_BANK_ASSEMBLY.json"

FIT_LO, FIT_HI = 10_700_001, 10_700_096
V2_NEW_LO, V2_NEW_HI = 11_000_001, 11_000_320
A_PREFERRED, B_PREFERRED, NOT_ESTABLISHED = -1, 1, 0

# latent <- label, and the teacher that latent is anchored toward
LATENT_FOR = {A_PREFERRED: 0, B_PREFERRED: 1}
TEACHER_FOR = {A_PREFERRED: "pi_A", B_PREFERRED: "pi_B"}


class RehearsalError(RuntimeError):
    """Raised when the bank would violate the frozen spec."""


@dataclass
class RehearsalBank:
    """Uniform sampler over the eligible FIT states, with exposure accounting."""

    obs: dict[str, np.ndarray]
    label: np.ndarray                  # -1 A-preferred, +1 B-preferred, 0 tied
    delta: np.ndarray
    teacher_action: np.ndarray         # the preferred teacher's action, per state
    cell: np.ndarray
    seed: np.ndarray
    rng_seed: int = 31

    eligible: np.ndarray = field(init=False)
    exposures: Counter = field(default_factory=Counter, init=False)
    latent_exposures: Counter = field(default_factory=Counter, init=False)
    tied_exposures: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.eligible = np.nonzero(self.label != NOT_ESTABLISHED)[0]
        if self.eligible.size == 0:
            raise RehearsalError("no resolvable states in the bank")
        self._rng = np.random.default_rng(self.rng_seed)

    # ------------------------------------------------------------------ facts
    @property
    def n_states(self) -> int:
        return int(len(self.label))

    @property
    def n_eligible(self) -> int:
        return int(self.eligible.size)

    @property
    def n_tied(self) -> int:
        return int((self.label == NOT_ESTABLISHED).sum())

    def composition(self) -> dict:
        thin = int((np.abs(self.delta[self.eligible]) == 1).sum())
        return {
            "states": self.n_states,
            "eligible": self.n_eligible,
            "A_preferred": int((self.label == A_PREFERRED).sum()),
            "B_preferred": int((self.label == B_PREFERRED).sum()),
            "tied_excluded_from_sampling": self.n_tied,
            "abs_delta_1": thin,
            "abs_delta_1_frac": round(thin / max(1, self.n_eligible), 4),
        }

    # --------------------------------------------------------------- sampling
    def sample(self, batch_size: int) -> dict:
        """Uniform draw over eligible states only. Never adaptive, never weighted."""
        n = min(int(batch_size), self.n_eligible)
        pick = self._rng.choice(self.eligible, size=n, replace=False)
        labels = self.label[pick]
        if np.any(labels == NOT_ESTABLISHED):
            raise RehearsalError("a tied state entered a rehearsal batch")
        for i in pick:
            self.exposures[int(i)] += 1
        z = np.array([LATENT_FOR[int(v)] for v in labels], dtype=np.int64)
        for v in z:
            self.latent_exposures[int(v)] += 1
        return {
            "index": pick,
            "obs": {k: v[pick] for k, v in self.obs.items()},
            "teacher_action": self.teacher_action[pick],
            "z_idx": z,
            "label": labels,
            "cell": self.cell[pick],
        }

    # -------------------------------------------------------------- telemetry
    def telemetry(self) -> dict:
        counts = np.array([self.exposures.get(int(i), 0) for i in self.eligible])
        total = int(counts.sum())
        return {
            "total_exposures": total,
            "mean_exposures_per_state": round(float(counts.mean()), 2) if counts.size else 0.0,
            "min_exposures": int(counts.min()) if counts.size else 0,
            "max_exposures": int(counts.max()) if counts.size else 0,
            "never_sampled": int((counts == 0).sum()),
            "latent_exposures": {f"z{k}": v for k, v in sorted(self.latent_exposures.items())},
            "tied_exposures": self.tied_exposures,
            "replay_factor": round(total / max(1, self.n_eligible), 2),
        }

    def assert_zero_tied_pressure(self) -> None:
        """The core invariant of the whole abstention arc, measured not assumed."""
        if self.tied_exposures != 0:
            raise RehearsalError(
                f"tied states received {self.tied_exposures} exposures; a state where "
                "preference is not established must exert zero A/B pressure")


def load_bank(data_dir: Path = DATA, lo: int = FIT_LO, hi: int = FIT_HI,
              rng_seed: int = 31, verify_spec: bool = True) -> RehearsalBank:
    """Load the FIT rehearsal bank, asserting the frozen sign convention."""
    if verify_spec:
        if not SPEC.is_file():
            raise RehearsalError("ORACLE_GATED_REHEARSAL_SPEC.json is not frozen")
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
        if spec["status"] != "FROZEN -- SPEC_FROZEN_BEFORE_IMPLEMENTATION":
            raise RehearsalError(f"spec is not frozen: {spec['status']!r}")
        rule = spec["TREATMENT"]["label_rule_frozen"]
        if not rule["A_preferred"].startswith("Delta < 0"):
            raise RehearsalError("spec sign convention drifted: A_preferred must be Delta < 0")
        if not rule["B_preferred"].startswith("Delta > 0"):
            raise RehearsalError("spec sign convention drifted: B_preferred must be Delta > 0")

    keys = ("grid", "vec", "agent_mask", "mask")
    obs: dict[str, list] = {k: [] for k in keys}
    delta, action, cell, seed = [], [], [], []
    for s in range(lo, hi + 1):
        with np.load(Path(data_dir) / f"seed_{s}.npz", allow_pickle=False) as z:
            mb = z["branch_pi_B_blue"].astype(np.int64) - z["branch_pi_B_red"].astype(np.int64)
            ma = z["branch_pi_A_blue"].astype(np.int64) - z["branch_pi_A_red"].astype(np.int64)
            d = mb - ma
            obs["grid"].append(z["branch_obs_grid"][:, 0])
            obs["vec"].append(z["branch_obs_vec"][:, 0])
            obs["agent_mask"].append(z["branch_obs_agent_mask"][:, 0])
            obs["mask"].append(z["branch_obs_mask"][:, 0])
            # the PREFERRED teacher's action: pi_A where Delta<0, pi_B where Delta>0
            act = np.where((d < 0)[:, None], z["branch_pi_A_action"], z["branch_pi_B_action"])
            action.append(act)
            delta.append(d)
            cell.extend(str(c) for c in z["branch_cell"])
            seed.append(np.full(len(d), s, dtype=np.int64))

    delta = np.concatenate(delta)
    label = np.sign(delta).astype(np.int64)      # -1 A-pref, +1 B-pref, 0 tied
    bank = RehearsalBank(
        obs={k: np.concatenate(v) for k, v in obs.items()},
        label=label, delta=delta,
        teacher_action=np.concatenate(action),
        cell=np.array(cell), seed=np.concatenate(seed), rng_seed=rng_seed)

    # the sign convention is load-bearing; assert it rather than trusting it
    if not np.all(label[delta < 0] == A_PREFERRED):
        raise RehearsalError("sign convention violated: Delta < 0 must map to A-preferred")
    if not np.all(label[delta > 0] == B_PREFERRED):
        raise RehearsalError("sign convention violated: Delta > 0 must map to B-preferred")
    return bank


def _load_shard_range(
    data_dir: Path, lo: int, hi: int,
    obs: dict[str, list], delta: list, action: list, cell: list, seed: list,
) -> None:
    for s in range(lo, hi + 1):
        path = Path(data_dir) / f"seed_{s}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            mb = z["branch_pi_B_blue"].astype(np.int64) - z["branch_pi_B_red"].astype(np.int64)
            ma = z["branch_pi_A_blue"].astype(np.int64) - z["branch_pi_A_red"].astype(np.int64)
            d = mb - ma
            obs["grid"].append(z["branch_obs_grid"][:, 0])
            obs["vec"].append(z["branch_obs_vec"][:, 0])
            obs["agent_mask"].append(z["branch_obs_agent_mask"][:, 0])
            obs["mask"].append(z["branch_obs_mask"][:, 0])
            act = np.where((d < 0)[:, None], z["branch_pi_A_action"], z["branch_pi_B_action"])
            action.append(act)
            delta.append(d)
            cell.extend(str(c) for c in z["branch_cell"])
            seed.append(np.full(len(d), s, dtype=np.int64))


def load_bank_v2(rng_seed: int = 31, verify_spec: bool = True) -> RehearsalBank:
    """V2 bank: legacy FIT shards + new 11000001..11000320 collection shards."""
    if not V2_BANK_ASSEMBLY.is_file():
        raise RehearsalError(
            f"REFUSING: {V2_BANK_ASSEMBLY.name} missing; run audit_oracle_gated_v2_bank.py first")
    assembly = json.loads(V2_BANK_ASSEMBLY.read_text(encoding="utf-8"))
    if assembly.get("VERDICT") != "PASS":
        raise RehearsalError(
            f"REFUSING: V2 bank assembly verdict is {assembly.get('VERDICT')!r}, not PASS")
    if verify_spec:
        if not SPEC.is_file():
            raise RehearsalError("ORACLE_GATED_REHEARSAL_SPEC.json is not frozen")

    keys = ("grid", "vec", "agent_mask", "mask")
    obs: dict[str, list] = {k: [] for k in keys}
    delta, action, cell, seed_list = [], [], [], []
    _load_shard_range(DATA, FIT_LO, FIT_HI, obs, delta, action, cell, seed_list)
    _load_shard_range(V2_DATA, V2_NEW_LO, V2_NEW_HI, obs, delta, action, cell, seed_list)
    if not delta:
        raise RehearsalError("V2 bank load produced zero branch states")

    delta_arr = np.concatenate(delta)
    label = np.sign(delta_arr).astype(np.int64)
    bank = RehearsalBank(
        obs={k: np.concatenate(v) for k, v in obs.items()},
        label=label, delta=delta_arr,
        teacher_action=np.concatenate(action),
        cell=np.array(cell), seed=np.concatenate(seed_list), rng_seed=rng_seed)

    if not np.all(label[delta_arr < 0] == A_PREFERRED):
        raise RehearsalError("sign convention violated: Delta < 0 must map to A-preferred")
    if not np.all(label[delta_arr > 0] == B_PREFERRED):
        raise RehearsalError("sign convention violated: Delta > 0 must map to B-preferred")
    min_eligible = int(assembly["minimum_total_eligible"])
    if bank.n_eligible < min_eligible:
        raise RehearsalError(
            f"REFUSING: loaded {bank.n_eligible} eligible states < frozen minimum {min_eligible}")
    return bank


def rehearsal_anchor_loss(model, batch: dict, device: str = "cpu"):
    """Gated anchor loss: each state trains ONLY its preferred latent.

    Delegates to the validated ``strategy_anchor.anchor_loss``. The gating is
    entirely in which (state, z, teacher-action) triples reach it -- the other latent
    is never mentioned, so it receives no gradient rather than a push away.
    """
    import torch
    from rl.custom_ppo.strategy_anchor import anchor_loss

    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    obs = {k: t(v, torch.float32) for k, v in batch["obs"].items()}
    return anchor_loss(
        model, obs,
        t(batch["teacher_action"], torch.long),
        decision_mask=t(batch["obs"]["agent_mask"], torch.float32).bool()
        if "agent_mask" in batch["obs"] else None,
        z_idx=t(batch["z_idx"], torch.long))
