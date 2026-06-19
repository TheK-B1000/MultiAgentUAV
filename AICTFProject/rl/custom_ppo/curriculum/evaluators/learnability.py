"""Offline selector-learnability probe evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum.context import GateContext, preserve_model_training_mode
from rl.custom_ppo.curriculum.isolation import TrainingIsolationSnapshot
from rl.custom_ppo.curriculum.types import (
    GATE_STATUS_ERROR,
    GATE_STATUS_NOT_RUN,
    GateResult,
    gate_family_result_from_bool,
)

_DEFAULT_PROBE_OPPONENTS: tuple[str, ...] = ("OP5", "OP6", "OP7")
_DEFAULT_PROBE_SEEDS: tuple[int, ...] = tuple(range(3000, 3050))


class LearnabilityClassifier(nn.Module):
    """Temporary offline classifier; not part of production q_phi."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ProbeExample:
    opponent: str
    seed: int
    context: np.ndarray
    label: int
    z_returns: list[float] = field(default_factory=list)


def _format_selector_probe_progress(label: str, current: int, total: int) -> str:
    return f"[Selector Probe] {label} {int(current)}/{int(total)}"


def _print_selector_probe_progress(label: str, current: int, total: int) -> None:
    print(_format_selector_probe_progress(label, current, total), flush=True)


def grouped_stratified_split(
    examples: list[ProbeExample],
    *,
    train_fraction: float = 0.80,
    rng: random.Random | None = None,
) -> tuple[list[ProbeExample], list[ProbeExample]]:
    """Hold out whole (opponent, seed) groups for validation."""
    if not examples:
        return [], []

    rng = rng or random.Random(0)
    groups: dict[tuple[str, int], list[ProbeExample]] = {}
    for ex in examples:
        groups.setdefault((ex.opponent, ex.seed), []).append(ex)

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    n_train_groups = max(1, int(train_fraction * len(group_keys)))
    if n_train_groups >= len(group_keys) and len(group_keys) > 1:
        n_train_groups = len(group_keys) - 1

    train_keys = set(group_keys[:n_train_groups])
    train_examples: list[ProbeExample] = []
    val_examples: list[ProbeExample] = []
    for key, group_examples in groups.items():
        if key in train_keys:
            train_examples.extend(group_examples)
        else:
            val_examples.extend(group_examples)
    return train_examples, val_examples


def validate_probe_dataset(
    examples: list[ProbeExample],
    *,
    min_examples: int,
    latent_k: int,
) -> GateResult | None:
    """Return a blocking GateResult when the probe dataset is unusable."""
    n_examples = len(examples)
    if n_examples < min_examples:
        return GateResult(
            status=GATE_STATUS_ERROR,
            reason="insufficient_probe_examples",
            details={"num_examples": n_examples, "min_examples": min_examples},
        )

    labels = [ex.label for ex in examples]
    if len(set(labels)) < 2:
        return GateResult(
            status=GATE_STATUS_ERROR,
            reason="probe_label_degeneracy",
            details={"num_examples": n_examples, "unique_labels": len(set(labels))},
        )

    invalid_labels = [label for label in labels if label < 0 or label >= latent_k]
    if invalid_labels:
        return GateResult(
            status=GATE_STATUS_ERROR,
            reason="probe_label_out_of_range",
            details={"invalid_labels": invalid_labels, "latent_k": latent_k},
        )
    return None


def _train_probe_classifier(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    input_dim: int,
    num_classes: int,
) -> tuple[nn.Module, float, torch.Tensor]:
    model = LearnabilityClassifier(input_dim, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=8,
        shuffle=True,
    )
    model.train()
    for _ in range(150):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_val), dim=-1)
        accuracy = float((preds == y_val).float().mean().item())
    return model, accuracy, preds


def _collect_probe_examples(context: GateContext) -> tuple[list[ProbeExample], int, list[str], list[int]]:
    from rl.training.env_factory import build_training_env

    opponents = list(
        getattr(context.cfg, "curriculum_gate_probe_opponents", None) or _DEFAULT_PROBE_OPPONENTS
    )
    seed_start = int(getattr(context.cfg, "curriculum_gate_probe_seed_start", 3000))
    seed_count = int(getattr(context.cfg, "curriculum_gate_probe_seed_count", 50))
    seeds = list(range(seed_start, seed_start + seed_count))
    latent_k = int(context.latent_k)
    tie_margin = float(getattr(context.cfg, "probe_utility_tie_margin", 0.05))
    bootstrap_steps = int(getattr(context.cfg, "curriculum_gate_probe_bootstrap_steps", 64))
    branch_steps = int(getattr(context.cfg, "curriculum_gate_probe_branch_steps", 64))

    eval_cfg = PPOConfig()
    for key, value in context.cfg.__dict__.items():
        setattr(eval_cfg, key, value)
    eval_cfg.n_envs = 1

    examples: list[ProbeExample] = []
    ambiguous = 0
    total_opponents = len(opponents)
    total_seeds = len(seeds)
    target_examples = max(1, total_opponents * total_seeds)

    with preserve_model_training_mode(context.eval_model):
        for opp_index, opp in enumerate(opponents, start=1):
            _print_selector_probe_progress("opponent", opp_index, total_opponents)
            env = build_training_env(eval_cfg, initial_phase="PHASE1", initial_opponent_tag=opp)
            try:
                for seed_index, seed in enumerate(seeds, start=1):
                    _print_selector_probe_progress("seed", seed_index, total_seeds)
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if hasattr(env, "seed"):
                        env.seed(seed)
                    obs = env.reset()
                    context.configure_fixed_z(0)
                    bootstrap_actions: list[Any] = []
                    done = False
                    step_i = 0
                    while not done and step_i < bootstrap_steps:
                        act = context.predict(obs)
                        bootstrap_actions.append(act)
                        env.step_async(act)
                        obs, _, done_arr, _ = env.step_wait()
                        done = bool(done_arr[0])
                        step_i += 1
                    if done:
                        _print_selector_probe_progress("examples", len(examples), target_examples)
                        continue

                    context_h = env.state()[0].copy()
                    z_returns: list[float] = []
                    for z_branch in range(latent_k):
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        if hasattr(env, "seed"):
                            env.seed(seed)
                        obs_b = env.reset()
                        for b_act in bootstrap_actions:
                            env.step_async(b_act)
                            obs_b, _, _, _ = env.step_wait()
                        context.configure_fixed_z(z_branch)
                        ret_accum = 0.0
                        b_done = False
                        b_step = 0
                        while not b_done and b_step < branch_steps:
                            act = context.predict(obs_b)
                            env.step_async(act)
                            obs_b, rewards, done_arr, _ = env.step_wait()
                            ret_accum += float(rewards[0])
                            b_done = bool(done_arr[0])
                            b_step += 1
                        z_returns.append(ret_accum)

                    order = np.argsort(z_returns)[::-1]
                    if len(order) >= 2 and (z_returns[order[0]] - z_returns[order[1]]) < tie_margin:
                        ambiguous += 1
                        _print_selector_probe_progress("examples", len(examples), target_examples)
                        continue
                    examples.append(
                        ProbeExample(
                            opponent=opp,
                            seed=int(seed),
                            context=context_h,
                            label=int(order[0]),
                            z_returns=list(z_returns),
                        )
                    )
                    _print_selector_probe_progress("examples", len(examples), target_examples)
            finally:
                env.close()

    return examples, ambiguous, opponents, seeds


def run_learnability_probe(context: GateContext) -> GateResult:
    if not bool(getattr(context.cfg, "curriculum_gate_run_probe", False)):
        return GateResult(
            status=GATE_STATUS_NOT_RUN,
            reason="curriculum_gate_run_probe=false",
        )

    print("[Curriculum Controller] Selector-learnability probe...")
    isolation = TrainingIsolationSnapshot.capture(context.trainer)
    try:
        latent_k = int(context.latent_k)
        gs_dim = int(context.trainer.model.global_state_dim)
        min_examples = int(getattr(context.cfg, "curriculum_probe_min_examples", 10) or 10)

        examples, ambiguous, opponents, seeds = _collect_probe_examples(context)
        validation_error = validate_probe_dataset(
            examples,
            min_examples=min_examples,
            latent_k=latent_k,
        )
        if validation_error is not None:
            return validation_error

        train_examples, val_examples = grouped_stratified_split(examples, train_fraction=0.80)
        if not train_examples or not val_examples:
            return GateResult(
                status=GATE_STATUS_ERROR,
                reason="probe_split_degeneracy",
                details={
                    "num_examples": len(examples),
                    "train_examples": len(train_examples),
                    "val_examples": len(val_examples),
                },
            )

        X_train = torch.tensor(
            np.asarray([ex.context for ex in train_examples], dtype=np.float32)
        )
        y_train = torch.tensor([ex.label for ex in train_examples], dtype=torch.long)
        X_val = torch.tensor(np.asarray([ex.context for ex in val_examples], dtype=np.float32))
        y_val = torch.tensor([ex.label for ex in val_examples], dtype=torch.long)
        R_train = torch.tensor([ex.z_returns for ex in train_examples], dtype=torch.float32)
        R_val = torch.tensor([ex.z_returns for ex in val_examples], dtype=torch.float32)

        _, val_accuracy, preds = _train_probe_classifier(
            X_train,
            y_train,
            X_val,
            y_val,
            input_dim=gs_dim,
            num_classes=latent_k,
        )

        val_oracle = R_val.max(dim=-1)[0]
        oracle_mean = float(val_oracle.mean().item())
        probe_rets = R_val[torch.arange(len(y_val)), preds]
        probe_mean = float(probe_rets.mean().item())
        probe_regret = oracle_mean - probe_mean

        majority = int(torch.bincount(y_train, minlength=latent_k).argmax().item())
        majority_acc = float((y_val == majority).float().mean().item())
        uniform_acc = 1.0 / float(latent_k)

        global_best_z = int(R_train.sum(dim=0).argmax().item())
        fixed_mean = float(R_val[:, global_best_z].mean().item())
        fixed_regret = oracle_mean - fixed_mean

        if fixed_regret <= 0.0:
            return GateResult(
                status=GATE_STATUS_NOT_RUN,
                reason="zero_fixed_policy_regret",
                details={
                    "num_examples": len(examples),
                    "oracle_return_mean": oracle_mean,
                    "global_best_fixed_z_return_mean": fixed_mean,
                    "fixed_policy_regret": fixed_regret,
                },
            )

        accuracy_passed = val_accuracy >= (majority_acc + 0.05)
        regret_passed = probe_regret <= (0.90 * fixed_regret)
        gate_passed = accuracy_passed and regret_passed

        isolation.assert_unchanged(context.trainer)

        return gate_family_result_from_bool(
            gate_passed,
            details={
                "num_examples": len(examples),
                "ambiguous_context_fraction": float(ambiguous / max(1, len(seeds) * len(opponents))),
                "uniform_accuracy_baseline": uniform_acc,
                "majority_accuracy_baseline": majority_acc,
                "probe_validation_accuracy": val_accuracy,
                "oracle_return_mean": oracle_mean,
                "probe_selected_return_mean": probe_mean,
                "global_best_fixed_z_return_mean": fixed_mean,
                "probe_regret": probe_regret,
                "global_best_fixed_z_regret": fixed_regret,
                "fixed_policy_regret": fixed_regret,
                "regret_reduction": fixed_regret - probe_regret,
                "accuracy_passed": accuracy_passed,
                "regret_passed": regret_passed,
                "gate_passed": gate_passed,
            },
        )
    finally:
        isolation.restore_rng()


__all__ = [
    "LearnabilityClassifier",
    "ProbeExample",
    "_format_selector_probe_progress",
    "grouped_stratified_split",
    "run_learnability_probe",
    "validate_probe_dataset",
]
