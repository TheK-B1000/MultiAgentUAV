"""Slim PPO update coordinator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from rl.custom_ppo.return_normalization import _update_strategy_return_stats
from rl.custom_ppo.schedules import resolve_latent_lam_h
from rl.custom_ppo.update.actor_intervention import ActorInterventionEvidenceUpdater
from rl.custom_ppo.update.entropy_objectives import EntropyObjective
from rl.custom_ppo.update.minibatch_updater import (
    ACTOR_INTERVENTION_REASON_CODES,
    MinibatchUpdater,
    MinibatchUpdaterState,
)
from rl.custom_ppo.update.optimizer_stepper import build_optimizer_stepper
from rl.custom_ppo.update.phase_policy import apply_phase_requires_grad
from rl.custom_ppo.update.post_update import (
    PostUpdatePipeline,
    resolve_adapter_scale,
    resolve_separation_coef,
)
from rl.custom_ppo.update.router_sequence_updater import RouterSequenceUpdater
from rl.custom_ppo.update.separation_objectives import SeparationObjective
from rl.custom_ppo.update.strategy_objectives import StrategyObjective
from rl.custom_ppo.update.telemetry import UpdateStatsAccumulator, build_metric_schema
from rl.custom_ppo.update.update_context import PPOUpdateContextBuilder
from rl.ppo_core import TensorDictRolloutBuffer

if TYPE_CHECKING:
    from rl.custom_ppo.latent_strategy_state import LatentStrategyState
    from rl.custom_ppo.trainer import CustomPPOTrainer
    from rl.custom_ppo.trainer_config import TrainerHyperparams


class PPOUpdater:
    """PPO epoch/minibatch loop coordinator."""

    def __init__(
        self,
        *,
        model: Any,
        optimizer: Any,
        device: Any,
        cfg: Any,
        hparams: "TrainerHyperparams",
        latent_state: "LatentStrategyState",
        runtime: "CustomPPOTrainer",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.hparams = hparams
        self.latent_state = latent_state
        self.runtime = runtime
        # SAPPO V1 (Reading 2): optional interleaved teacher rehearsal. When
        # None -- the default -- no anchor batch is fetched, no anchor forward
        # or backward runs, and no optimizer or scheduler step occurs, so the
        # PPO path is untouched BY CONSTRUCTION rather than by a zero-scaled
        # loss term. See SAPPO_V1_LOSS_SEMANTICS_AMENDMENT.json.
        self.anchor_runner = getattr(runtime, "sappo_anchor_runner", None)
        self.cf_grad_ratio_violations = 0
        seed = int(getattr(cfg, "seed", 0) or 0) + 31_337
        self._z_separation_generator = torch.Generator(device=device)
        self._z_separation_generator.manual_seed(seed)
        self._intervention_evidence = ActorInterventionEvidenceUpdater()
        self._post_update = PostUpdatePipeline(intervention_evidence=self._intervention_evidence)

    def state_dict(self) -> dict[str, Any]:
        return {"z_separation_generator": self._z_separation_generator.get_state()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        gen_state = state.get("z_separation_generator")
        if gen_state is not None:
            if isinstance(gen_state, torch.Tensor):
                gen_state = gen_state.cpu()
            self._z_separation_generator.set_state(gen_state)

    def compute_latent_lam_h(self, global_step: float, total_timesteps: int) -> float:
        return resolve_latent_lam_h(self.cfg, global_step=global_step, total_timesteps=total_timesteps)

    def update(
        self,
        buffer: TensorDictRolloutBuffer,
        *,
        total_timesteps: int,
    ) -> dict[str, float]:
        runtime = self.runtime
        hparams = self.hparams
        cfg = self.cfg
        step = int(runtime.global_step)
        from rl.custom_ppo.v6i1_phase_runtime import (
            apply_v6i1_learning_rates,
            is_v6i1_staged_trainer,
            resolve_v6i1_entropy_schedule_total_timesteps,
            resolve_v6i1_lr_progress_remaining,
            resolve_v6i1_rollout_usage_coef,
        )

        progress_remaining = resolve_v6i1_lr_progress_remaining(
            runtime, training_terminal=int(total_timesteps)
        )
        lr_floor_frac = max(
            0.0, min(float(getattr(cfg, "lr_floor_frac", 0.1) or 0.0), 1.0)
        )
        lr = hparams.learning_rate * max(progress_remaining, lr_floor_frac)

        v6i1_lr_stats: dict[str, float] = {}
        if getattr(runtime, "v6i1_three_optimizer_mode", False):
            v6i1_lr_stats = apply_v6i1_learning_rates(
                runtime,
                base_lr=float(hparams.learning_rate),
                training_terminal=int(total_timesteps),
            )
            lr = float(v6i1_lr_stats.get("actor_lr", lr))
        else:
            # Preserve per-group relative multipliers (e.g. LRO z-actor lr_mult).
            # Without this, the linear schedule overwrites group LRs to a single
            # annealed value and silently cancels latent_lro_z_actor_lr_mult.
            for group in self.optimizer.param_groups:
                mult = float(group.get("lr_mult", 1.0) or 1.0)
                group["lr"] = lr * mult

        ent_coef = hparams.ent_coef if progress_remaining > 0.75 else 0.5 * hparams.ent_coef
        latent_schedule_total = (
            resolve_v6i1_entropy_schedule_total_timesteps(runtime)
            if is_v6i1_staged_trainer(runtime)
            else int(total_timesteps)
        )
        latent_lam_h = self.compute_latent_lam_h(step, latent_schedule_total)
        curr_sep_coef = resolve_separation_coef(self, step=step)
        curr_adapter_scale = resolve_adapter_scale(self, step=step)

        repertoire_param_snapshot = None
        frozen_repertoire_snapshot = None
        stage = str(getattr(cfg, "v6i9_training_stage", "") or "").lower()
        if stage == "repertoire":
            from rl.custom_ppo.diagnostics.competence import snapshot_repertoire_parameters

            runtime._repertoire_grad_audit_max = {}
            repertoire_param_snapshot = snapshot_repertoire_parameters(self.model)
        elif stage == "router":
            from rl.custom_ppo.diagnostics.competence import snapshot_frozen_repertoire_parameters

            runtime._router_grad_audit_max = {}
            frozen_repertoire_snapshot = snapshot_frozen_repertoire_parameters(self.model)

        _update_strategy_return_stats(runtime, buffer)
        v6i1_usage_coef = (
            float(resolve_v6i1_rollout_usage_coef(runtime))
            if is_v6i1_staged_trainer(runtime)
            else 0.0
        )
        context = PPOUpdateContextBuilder(cfg=cfg, hparams=hparams, runtime=runtime).build(
            total_timesteps=total_timesteps,
            primary_lr=lr,
            latent_lam_h=latent_lam_h,
            curr_sep_coef=curr_sep_coef,
            curr_adapter_scale=curr_adapter_scale,
            v6i1_usage_coef=v6i1_usage_coef,
            v6i1_lr_stats=v6i1_lr_stats,
            buffer=buffer,
        )
        phase_policy = context.phase_policy
        if context.phase in {"A", "B", "C"}:
            apply_phase_requires_grad(self.model, context.phase)
        pair_count = context.pair_count
        v6i1_usage_coef = context.rollout_usage_coefficient

        entropy_objective = EntropyObjective(
            model=self.model, cfg=cfg, hparams=hparams, device=self.device
        )
        has_dedicated_router_opt = context.has_dedicated_router_optimizer
        prep = entropy_objective.prepare_rollout(
            buffer,
            latent_lam_h=latent_lam_h,
            v6i1_usage_coef=v6i1_usage_coef,
            has_dedicated_router_opt=has_dedicated_router_opt,
        )

        minibatch_updater = MinibatchUpdater(
            model=self.model,
            cfg=cfg,
            hparams=hparams,
            runtime=runtime,
            latent_state=self.latent_state,
            device=self.device,
            entropy_objective=entropy_objective,
            strategy_objective=StrategyObjective(
                model=self.model,
                cfg=cfg,
                hparams=hparams,
                runtime=runtime,
                device=self.device,
            ),
            separation_objective=SeparationObjective(
                model=self.model,
                cfg=cfg,
                hparams=hparams,
                runtime=runtime,
                latent_state=self.latent_state,
                subsample_generator=self._z_separation_generator,
            ),
            optimizer_stepper=build_optimizer_stepper(runtime, self.optimizer),
            separation_generator=self._z_separation_generator,
        )

        schema = build_metric_schema(latent_k=int(hparams.latent_k), pair_count=pair_count)
        accumulator = UpdateStatsAccumulator(schema)
        updater_state = MinibatchUpdaterState()
        valid_cf_pair_measurements: list[torch.Tensor] = []
        actor_intervention_valid_minibatches = 0
        last_invalid_reason_code = 0.0
        stop_update = False

        # V6I7: recurrent router sequence updater (BPTT) — constructed once, runs per epoch.
        router_seq_updater = RouterSequenceUpdater(
            model=self.model,
            cfg=cfg,
            hparams=hparams,
            optimizer=self.optimizer,
            device=self.device,
        )
        is_v6i7_router = router_seq_updater.is_active(buffer)

        for epoch_idx in range(hparams.n_epochs):
            epoch_state = entropy_objective.for_epoch(prep, v6i1_usage_coef=v6i1_usage_coef)
            epoch_state.consumed = False
            for mb_idx, batch in enumerate(
                buffer.iter_minibatches(hparams.batch_size, shuffle=True)
            ):
                result = minibatch_updater.update(
                    batch=batch,
                    context=context,
                    phase_policy=phase_policy,
                    epoch_state=epoch_state,
                    prep=prep,
                    latent_lam_h=latent_lam_h,
                    curr_sep_coef=curr_sep_coef,
                    ent_coef=ent_coef,
                    v6i1_usage_coef=v6i1_usage_coef,
                    epoch_idx=epoch_idx,
                    mb_idx=mb_idx,
                    pair_count=pair_count,
                    updater_state=updater_state,
                )
                accumulator.record_minibatch(result.telemetry)
                if self.anchor_runner is not None:
                    # Counts a COMPLETED PPO actor minibatch; the runner steps
                    # only on full groups and never emits a trailing update.
                    self.anchor_runner.note_ppo_minibatch()
                measurement = result.separation_measurement
                if measurement is not None and measurement.valid and measurement.values is not None:
                    valid_cf_pair_measurements.append(measurement.values)
                    actor_intervention_valid_minibatches += 1
                    last_invalid_reason_code = 0.0
                elif measurement is not None and measurement.reason is not None:
                    last_invalid_reason_code = ACTOR_INTERVENTION_REASON_CODES.get(
                        measurement.reason, 99.0
                    )
                if result.should_stop:
                    stop_update = True
                    break

            # V6I7: after the actor/critic minibatch loop, run one router BPTT epoch.
            if is_v6i7_router and not stop_update:
                router_stats = router_seq_updater.update_epoch(buffer, ent_coef=ent_coef)
                if router_stats:
                    accumulator.record_minibatch(router_stats)

            if stop_update:
                break

        post = self._post_update.run(
            updater=self,
            buffer=buffer,
            context=context,
            accumulator=accumulator,
            prep=prep,
            latent_lam_h=latent_lam_h,
            curr_sep_coef=curr_sep_coef,
            curr_adapter_scale=curr_adapter_scale,
            lr=lr,
            v6i1_lr_stats=v6i1_lr_stats,
            v6i1_usage_coef=v6i1_usage_coef,
            pair_count=pair_count,
            valid_cf_pair_measurements=valid_cf_pair_measurements,
            actor_intervention_valid_minibatches=actor_intervention_valid_minibatches,
            last_invalid_reason_code=last_invalid_reason_code,
            repertoire_param_snapshot=repertoire_param_snapshot,
            frozen_repertoire_snapshot=frozen_repertoire_snapshot,
        )
        return post.stats
