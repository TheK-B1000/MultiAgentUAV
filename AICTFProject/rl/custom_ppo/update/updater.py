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
        # SAPPO V1 (Reading 2): optional interleaved teacher rehearsal.
        #
        # DO NOT cache the runner here. This updater is constructed inside
        # build_trainer(), which runs BEFORE the orchestrator attaches
        # runtime.sappo_anchor_runner. Caching at construction captured None
        # and silently disabled rehearsal for an entire 2x500k run that looked
        # completely healthy -- the anchor branch simply never executed. The
        # runner is read at USE time instead, in _anchor_runner().
        #
        # Absent runner still means structurally absent: no batch fetch, no
        # forward, no backward, no optimizer or scheduler step.
        self._sappo_seen_ppo_minibatches = 0
        self._pending_exp2_teacher_state: dict[str, Any] | None = None
        self.cf_grad_ratio_violations = 0
        seed = int(getattr(cfg, "seed", 0) or 0) + 31_337
        self._z_separation_generator = torch.Generator(device=device)
        self._z_separation_generator.manual_seed(seed)
        self._intervention_evidence = ActorInterventionEvidenceUpdater()
        self._post_update = PostUpdatePipeline(intervention_evidence=self._intervention_evidence)

    def state_dict(self) -> dict[str, Any]:
        out = {"z_separation_generator": self._z_separation_generator.get_state()}
        runner = self._exp2_teacher_runner()
        if runner is not None:
            out["exp2_teacher_compression"] = runner.state_dict()
        elif self._pending_exp2_teacher_state is not None:
            out["exp2_teacher_compression"] = dict(self._pending_exp2_teacher_state)
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        gen_state = state.get("z_separation_generator")
        if gen_state is not None:
            if isinstance(gen_state, torch.Tensor):
                gen_state = gen_state.cpu()
            self._z_separation_generator.set_state(gen_state)
        pending = state.get("exp2_teacher_compression")
        self._pending_exp2_teacher_state = dict(pending) if pending is not None else None

    def consume_pending_exp2_teacher_state(self) -> dict[str, Any] | None:
        state = self._pending_exp2_teacher_state
        self._pending_exp2_teacher_state = None
        return state

    def compute_latent_lam_h(self, global_step: float, total_timesteps: int) -> float:
        return resolve_latent_lam_h(self.cfg, global_step=global_step, total_timesteps=total_timesteps)

    def _anchor_runner(self):
        """Read the rehearsal runner at USE time, never cached at construction."""
        return getattr(self.runtime, "sappo_anchor_runner", None)

    def _exp2_teacher_runner(self):
        """Read the EXP2 runner at use time; attachment occurs after loading."""
        return getattr(self.runtime, "exp2_teacher_compression_runner", None)

    def _oracle_rehearsal_runner(self):
        """Read the oracle-gated rehearsal runner at USE time, never cached.

        Same seam that silently disabled SAPPO rehearsal for a whole 2x500k run: a
        runner attached after construction is invisible to anything that cached the
        lookup, so this resolves on every minibatch.
        """
        return getattr(self.runtime, "oracle_rehearsal_runner", None)

    def _retention_runner(self):
        """Read the RSCFT retention runner at USE time, never cached.

        Same reasoning as every other seam here: the runner is attached after this updater
        is built, so caching would capture None and retention would never fire while every
        other counter looked healthy. Absent runner means structurally absent -- no batch
        fetch, no forward, no backward, no optimizer step, no RNG consumed -- which is what
        makes the RSCFT CONTROL arm's "retention is not merely disabled but unreachable"
        claim true rather than aspirational.
        """
        return getattr(self.runtime, "retention_runner", None)

    def _assert_retention_cadence(self, runner) -> None:
        """ABORT if the retention update is not actually happening.

        Retention IS the defining treatment of the RSCFT treatment arm. A broken seam must
        die in the first few minibatches rather than be discovered after 500k steps of a run
        that was silently just CCP-S2 again -- the exact failure this program already ate
        once when a rehearsal seam went quietly dead for a full 2x500k run.
        """
        n_ppo = int(runner.n_ppo_actor_minibatches)
        n_ret = int(runner.n_retention_updates)
        expected = n_ppo // int(runner.cadence)
        if n_ret != expected:
            raise RuntimeError(
                f"RSCFT retention cadence violated: {n_ret} retention updates after "
                f"{n_ppo} PPO actor minibatches, expected {expected} "
                f"(cadence 1:{runner.cadence}). The retention treatment is not being "
                "applied; aborting rather than producing a run whose defining treatment "
                "is absent.")

    def _ranking_runner(self):
        """Read the SPPPO ranking runner at USE time, never cached.

        Same seam that silently disabled SAPPO rehearsal for a whole 2x500k run:
        this updater is built before the orchestrator attaches the runner, so
        caching here would capture None and the ranking branch would never fire
        while every other counter looked healthy.

        Absent runner means structurally absent -- the lambda_R = 0 control
        performs no batch fetch, no forward, no backward, no optimizer step and
        consumes no RNG.
        """
        return getattr(self.runtime, "sppo_ranking_runner", None)

    def _assert_ranking_cadence(self, runner) -> None:
        """ABORT if the ranking update is not actually happening.

        Logging alone is not enough. The defining treatment of SPPPO is this
        update; a broken seam must die in the first few minibatches rather than
        be discovered after 1M steps of a run that is silently vanilla EXP2C.
        """
        n_ppo = int(runner.n_ppo_actor_minibatches)
        n_rank = int(runner.n_ranking_updates)
        expected = n_ppo // int(runner.cadence)
        if n_rank != expected:
            raise RuntimeError(
                f"SPPPO ranking cadence violated: {n_rank} ranking updates after "
                f"{n_ppo} PPO actor minibatches, expected {expected} "
                f"(cadence 1:{runner.cadence}). The strategic ranking treatment "
                "is not being applied; aborting rather than producing a run "
                "whose defining treatment is absent.")

    def _assert_anchor_cadence(self, runner) -> None:
        """ABORT the run if rehearsal is not actually happening.

        A silently-disabled anchor previously ran to completion across 1M steps.
        Logging alone is not enough: the invariant is enforced every minibatch,
        so a broken seam dies in the first few updates instead of six hours in.
        """
        n_ppo = int(runner.n_ppo_actor_minibatches)
        n_anchor = int(runner.n_anchor_updates)
        expected = n_ppo // int(runner.cadence)
        if n_anchor != expected:
            raise RuntimeError(
                f"SAPPO cadence violated: {n_anchor} anchor updates after "
                f"{n_ppo} PPO actor minibatches, expected "
                f"{expected} (cadence 1:{runner.cadence}). Rehearsal is not "
                "being applied; aborting rather than producing a run whose "
                "defining treatment is absent.")

    @staticmethod
    def _assert_exp2_teacher_cadence(runner) -> None:
        n_ppo = int(runner.n_ppo_actor_minibatches)
        n_teacher = int(runner.n_teacher_updates)
        expected = n_ppo // int(runner.cadence)
        if n_teacher != expected:
            raise RuntimeError(
                f"EXP2 teacher cadence violated: {n_teacher} teacher updates "
                f"after {n_ppo} PPO actor minibatches, expected {expected} "
                f"(cadence 1:{runner.cadence})."
            )


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
                runner = self._anchor_runner()
                exp2_runner = self._exp2_teacher_runner()
                if runner is not None and exp2_runner is not None:
                    raise RuntimeError("SAPPO and EXP2 teacher runners cannot be active together")
                if runner is not None:
                    # Counts a COMPLETED PPO actor minibatch; the runner steps
                    # only on full groups and never emits a trailing update.
                    runner.note_ppo_minibatch()
                    self._sappo_seen_ppo_minibatches += 1
                    self._assert_anchor_cadence(runner)
                    # Visible from the FIRST reporting interval, so a silent
                    # anchor shows up as n_anchor_updates=0 immediately rather
                    # than being discovered after the run completes.
                    accumulator.record_minibatch({
                        "sappo_n_ppo_actor_updates": float(runner.n_ppo_actor_minibatches),
                        "sappo_n_anchor_updates": float(runner.n_anchor_updates),
                        "sappo_anchor_to_ppo_ratio": float(
                            runner.n_anchor_updates / max(1, runner.n_ppo_actor_minibatches)),
                        "sappo_anchor_loss": float(runner.last_anchor_loss),
                    })
                if exp2_runner is not None:
                    # Pass the actual completed PPO minibatch so teacher logits
                    # are evaluated on the student's on-policy states and z.
                    exp2_runner.realized_environment_steps = int(step)
                    exp2_runner.note_ppo_minibatch(batch)
                    self._assert_exp2_teacher_cadence(exp2_runner)
                    accumulator.record_minibatch(exp2_runner.telemetry())
                oracle_runner = self._oracle_rehearsal_runner()
                if oracle_runner is not None:
                    if exp2_runner is not None:
                        raise RuntimeError(
                            "oracle-gated rehearsal and EXP2 teacher compression cannot "
                            "be active together: EXP2 applies UNGATED teacher pressure "
                            "to every state, including the ties that must receive none")
                    oracle_runner.note_ppo_minibatch()
                    # Visible from the FIRST reporting interval, so an inert rehearsal
                    # shows up as n_updates=0 immediately rather than after 1M steps.
                    accumulator.record_minibatch({
                        "oracle_n_ppo_minibatches": float(oracle_runner.n_ppo_minibatches),
                        "oracle_n_rehearsal_updates": float(oracle_runner.n_updates),
                        "oracle_rehearsal_loss": float(oracle_runner.last_loss),
                        "oracle_tied_exposures": float(oracle_runner.bank.tied_exposures),
                    })

                retention_runner = self._retention_runner()
                if retention_runner is not None:
                    # Separate operation, like every other auxiliary objective here: never
                    # shares a backward pass with the PPO surrogate, and runs on the same
                    # on-policy states the actor just trained on.
                    retention_runner.note_ppo_minibatch(batch)
                    self._assert_retention_cadence(retention_runner)
                    # Visible from the FIRST reporting interval, so an inert retention seam
                    # shows up as retention_updates=0 immediately rather than after 500k steps.
                    rt = retention_runner.telemetry()
                    accumulator.record_minibatch({
                        "retention_n_ppo_actor_updates": float(rt["n_ppo_actor_minibatches"]),
                        "retention_n_updates": float(rt["retention_updates"]),
                        "retention_ema_updates": float(rt["ema_updates"]),
                        "retention_loss": float(rt["last_loss"]),
                        "retention_kl_mean": float(rt["last_kl_mean"]),
                        "retention_eligible_heads": float(rt["last_eligible_heads"]),
                        "retention_empty_batches": float(rt["empty_batches"]),
                    })

                rank_runner = self._ranking_runner()
                if rank_runner is not None:
                    # Third separate operation. The ranking step never shares a
                    # backward pass with the PPO surrogate; it runs on the same
                    # on-policy states the actor just trained on.
                    rank_runner.note_ppo_minibatch(batch)
                    self._assert_ranking_cadence(rank_runner)
                    # Visible from the FIRST reporting interval, so an inert
                    # ranking seam shows up as n_rank_updates=0 immediately.
                    d = rank_runner.last_diag
                    accumulator.record_minibatch({
                        "sppo_n_ppo_actor_updates": float(rank_runner.n_ppo_actor_minibatches),
                        "sppo_n_rank_updates": float(rank_runner.n_ranking_updates),
                        "sppo_rank_to_ppo_ratio": float(
                            rank_runner.n_ranking_updates
                            / max(1, rank_runner.n_ppo_actor_minibatches)),
                        "sppo_rank_loss": float(rank_runner.last_loss),
                        "sppo_rank_activation_rate": float(d.get("activation_rate", float("nan"))),
                        "sppo_delta_A": float(d.get("delta_A_mean", float("nan"))),
                        "sppo_delta_B": float(d.get("delta_B_mean", float("nan"))),
                        "sppo_lambda_rank": float(rank_runner.lambda_rank),
                        "sppo_margin": float(rank_runner.margin),
                    })
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
