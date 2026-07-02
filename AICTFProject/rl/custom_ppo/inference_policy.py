from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch.distributions import Categorical

from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import TemporalStateTracker
from rl.custom_ppo.policy import SharedActorCentralizedCritic

class CustomPPOInferencePolicy:
    """Small inference wrapper with a ``predict`` method for viewer/eval code."""

    def __init__(
        self,
        model: SharedActorCentralizedCritic,
        *,
        device: str | torch.device = "cpu",
        cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.model.to(self.device)
        self.model.eval()
        self._prev_z = None
        cfg = cfg or {}
        self.router_allowed_latents = cfg.get("router_allowed_latents", None)
        self._previous_opportunity_features = None
        self._opportunity_occurred = None
        self.strategy_interval = max(0, int(cfg.get("latent_resample_every_n", 0) or 0))
        self._original_strategy_interval = self.strategy_interval
        self.fixed_latent_strategy = bool(cfg.get("fixed_latent_strategy", False))
        self.fixed_latent_strategy_id = max(0, int(cfg.get("fixed_latent_strategy_id", 0) or 0))
        self._strategy_age = 0
        self._last_strategy_z = None
        self._last_strategy_probs = None
        self._last_strategy_entropy = None
        self._last_strategy_resampled = False
        self._last_strategy_logits = None
        self._last_context_gs = None
        self._temporal_tracker = None
        self._selector_hidden: torch.Tensor | None = None
        self.latent_eval_mode = "normal"
        self._latent_eval_marginal = None
        self._latent_eval_rng = None
        self._shuffled_mapping = None
        self._current_opponent = None
        self._current_seed = None
        self._current_episode_index = None
        self._current_eval_seed = None
        self._current_environment_seed = None
        self._current_env_index = None
        self._current_decision_step = 0
        self._opportunity_counter = 0
        self.opportunity_trace_log = []

    def set_current_episode_context(
        self,
        opponent: str,
        seed: int,
        episode_index: int,
    ) -> None:
        self._current_opponent = str(opponent).upper()
        self._current_seed = int(seed)
        self._current_episode_index = int(episode_index)
        self._current_eval_seed = int(seed)
        self._current_environment_seed = int(seed)
        self._current_env_index = int(episode_index)
        if isinstance(self._opportunity_counter, np.ndarray):
            self._opportunity_counter.fill(0)
        else:
            self._opportunity_counter = 0

    def set_eval_episode_context(
        self,
        opponent: str,
        eval_seed: int,
        environment_seed: int,
        env_index: int = 0,
    ) -> None:
        self._current_opponent = str(opponent).upper()
        self._current_eval_seed = int(eval_seed)
        self._current_environment_seed = int(environment_seed)
        self._current_env_index = int(env_index)
        self._current_seed = int(eval_seed)
        self._current_episode_index = int(env_index)
        if isinstance(self._opportunity_counter, np.ndarray):
            self._opportunity_counter.fill(0)
        else:
            self._opportunity_counter = 0

    def set_current_decision_step(self, step: int) -> None:
        self._current_decision_step = int(step)

    def inject_shuffled_mapping(self, mapping: dict) -> None:
        self._shuffled_mapping = mapping

    def clear_eval_suite_state(self) -> None:
        self._shuffled_mapping = None
        self.opportunity_trace_log = []
        self._current_opponent = None
        self._current_seed = None
        self._current_episode_index = None
        self._current_eval_seed = None
        self._current_environment_seed = None
        self._current_env_index = None
        self._current_decision_step = 0
        if isinstance(self._opportunity_counter, np.ndarray):
            self._opportunity_counter.fill(0)
        else:
            self._opportunity_counter = 0
        self.reset_strategy()

    def _fixed_strategy_id(self) -> int:
        if not self.model.uses_latent_strategy:
            return 0
        return min(self.fixed_latent_strategy_id, max(0, int(self.model.latent_k) - 1))

    def _fixed_strategy_tensor(self, batch: int) -> torch.Tensor:
        return torch.full((int(batch),), self._fixed_strategy_id(), dtype=torch.long, device=self.device)

    def _fixed_strategy_probs(self, batch: int) -> torch.Tensor:
        probs = torch.zeros((int(batch), int(self.model.latent_k)), dtype=torch.float32, device=self.device)
        probs[:, self._fixed_strategy_id()] = 1.0
        return probs

    def set_latent_eval_mode(
        self,
        mode: str,
        *,
        marginal: Optional[Iterable[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        m = str(mode).strip().lower()
        if m not in {"normal", "uniform_random", "shuffled", "fixed"}:
            raise ValueError(
                f"latent_eval_mode must be one of normal|uniform_random|shuffled|fixed, got {mode!r}"
            )
        self.latent_eval_mode = m
        if marginal is not None and self.model.uses_latent_strategy:
            marg = torch.as_tensor(list(marginal), dtype=torch.float32, device=self.device)
            if int(marg.numel()) != int(self.model.latent_k):
                raise ValueError(
                    f"latent_eval_marginal must have length latent_k={self.model.latent_k}, got {int(marg.numel())}"
                )
            total = float(marg.sum().item())
            if total <= 0.0:
                raise ValueError("latent_eval_marginal must sum to > 0")
            self._latent_eval_marginal = marg / total
        elif m == "shuffled" and self._latent_eval_marginal is None:
            allowed = self.router_allowed_latents
            if allowed is not None and len(allowed) > 0:
                print(
                    f"[CustomPPOInferencePolicy] latent_eval_mode='shuffled' fallback to uniform marginal over allowed {allowed}."
                )
                marginal = torch.zeros((int(self.model.latent_k),), device=self.device)
                val = 1.0 / len(allowed)
                for z in allowed:
                    marginal[z] = val
                self._latent_eval_marginal = marginal
            else:
                print(
                    "[CustomPPOInferencePolicy] latent_eval_mode='shuffled' but no marginal provided; "
                    "falling back to uniform marginal."
                )
                self._latent_eval_marginal = torch.full(
                    (int(self.model.latent_k),), 1.0 / max(1, int(self.model.latent_k)), device=self.device
                )
        if seed is None:
            seed = 0x5EE_D + (0 if m == "normal" else 1)
        self._latent_eval_rng = torch.Generator(device=self.device)
        self._latent_eval_rng.manual_seed(int(seed) & 0xFFFFFFFF)

    def _destructive_latent_z(self, batch: int) -> torch.Tensor:
        K = max(1, int(self.model.latent_k))
        allowed = self.router_allowed_latents
        if allowed is not None and len(allowed) > 0:
            allowed_t = torch.tensor(allowed, dtype=torch.long, device=self.device)
            idx = torch.randint(
                low=0,
                high=len(allowed),
                size=(int(batch),),
                generator=self._latent_eval_rng,
                device=self.device,
                dtype=torch.long,
            )
            return allowed_t[idx]
        if self.latent_eval_mode == "uniform_random":
            return torch.randint(
                low=0,
                high=K,
                size=(int(batch),),
                generator=self._latent_eval_rng,
                device=self.device,
                dtype=torch.long,
            )
        if self.latent_eval_mode == "shuffled":
            probs = self._latent_eval_marginal
            if probs is None:
                probs = torch.full((K,), 1.0 / K, device=self.device)
            cat = Categorical(probs=probs.unsqueeze(0).expand(int(batch), -1))
            if self._latent_eval_rng is None:
                return cat.sample()
            return torch.multinomial(
                probs.unsqueeze(0).expand(int(batch), -1),
                num_samples=1,
                replacement=True,
                generator=self._latent_eval_rng,
            ).squeeze(-1).long()
        raise AssertionError(f"_destructive_latent_z called in mode {self.latent_eval_mode!r}")

    def _get_temporal_tracker(self, batch_size: int) -> TemporalStateTracker:
        if self._temporal_tracker is None or self._temporal_tracker.num_envs != batch_size:
            self._temporal_tracker = TemporalStateTracker(
                num_envs=batch_size,
                state_dim=GLOBAL_STATE_DIM,
                device=self.device,
            )
        return self._temporal_tracker

    def reset_strategy(self, done_mask: Optional[np.ndarray | torch.Tensor] = None) -> None:
        """Forget the persisted inference strategy, typically at episode reset."""
        if done_mask is None:
            self._prev_z = None
            self._strategy_age = 0
            self._last_strategy_z = None
            self._last_strategy_probs = None
            self._last_strategy_entropy = None
            self._last_strategy_resampled = False
            self._last_strategy_logits = None
            self._last_context_gs = None
            self._selector_hidden = None
            self._opportunity_counter = 0
            self._opportunity_occurred = None
            self._previous_opportunity_features = None
            if self._temporal_tracker is not None:
                self._temporal_tracker.reset()
        else:
            mask = torch.as_tensor(done_mask, device=self.device).bool()
            batch = mask.shape[0]
            if self._prev_z is not None and self._prev_z.numel() == batch:
                if isinstance(self._strategy_age, torch.Tensor):
                    self._strategy_age[mask] = 0
                else:
                    self._strategy_age = 0
                if self._opportunity_occurred is not None:
                    self._opportunity_occurred[mask] = False
                if self._previous_opportunity_features is not None:
                    self._previous_opportunity_features[mask] = 0.0
                if self._temporal_tracker is not None:
                    self._temporal_tracker.reset(env_indices=mask)
                if isinstance(self._opportunity_counter, np.ndarray) and self._opportunity_counter.shape[0] == batch:
                    self._opportunity_counter[mask.cpu().numpy()] = 0

    def _uses_recurrent_selector(self) -> bool:
        return bool(getattr(self.model, "use_recurrent_selector", False))

    def _selector_hidden_dim(self) -> int:
        return int(getattr(self.model, "recurrent_selector_hidden_dim", 0) or 0)

    def _ensure_selector_hidden(self, batch: int) -> torch.Tensor | None:
        if not self._uses_recurrent_selector():
            return None
        hidden_dim = self._selector_hidden_dim()
        if hidden_dim <= 0:
            return None
        if self._selector_hidden is None or int(self._selector_hidden.shape[0]) != batch:
            self._selector_hidden = torch.zeros(
                (batch, hidden_dim), dtype=torch.float32, device=self.device
            )
        return self._selector_hidden

    def _strategy_logits_forward(self, context_gs: torch.Tensor) -> torch.Tensor:
        """Advance recurrent selector state and return tempered q_phi logits."""
        hidden = self._ensure_selector_hidden(int(context_gs.shape[0]))
        if hidden is None:
            return self.model.strategy_logits(context_gs)
        logits, h_new = self.model._forward_q_phi(context_gs, hidden)
        self._selector_hidden = h_new.detach()
        return logits / self.model.strategy_tau

    def _tensor_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=self.device),
            "vec": torch.as_tensor(obs["vec"], dtype=torch.float32, device=self.device),
            "agent_mask": torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=self.device),
            "mask": torch.as_tensor(obs["mask"], dtype=torch.float32, device=self.device),
        }

    def _global_state_tensor(self, obs: Dict[str, np.ndarray], batch: int) -> torch.Tensor:
        raw = obs.get("global_state")
        if raw is None:
            if self.model.uses_latent_strategy:
                import warnings
                warnings.warn(
                    "_global_state_tensor: 'global_state' missing from obs dict — router will "
                    "receive all-zero context. Inject env.state() before calling predict(). "
                    "See run_eval_episodes() line: single['global_state'] = env.state()[0]",
                    stacklevel=3,
                )
            return torch.zeros((batch, GLOBAL_STATE_DIM), dtype=torch.float32, device=self.device)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, ...]
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def _batched_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        batched: Dict[str, np.ndarray] = {}
        for key, value in obs.items():
            arr = np.asarray(value, dtype=np.float32)
            if key == "grid" and arr.ndim == 4:
                arr = arr[None, ...]
            elif key == "vec" and arr.ndim == 2:
                arr = arr[None, ...]
            elif key in {"agent_mask", "mask"} and arr.ndim == 1:
                arr = arr[None, ...]
            batched[key] = arr
        return batched

    # ------------------------------------------------------------------
    # Public PolicyInferenceContract surface
    # ------------------------------------------------------------------

    def get_observation_encoder_input_weights(self) -> torch.Tensor:
        """Return the first obs-encoder weight tensor.

        Shape: ``(out_channels, in_channels, kH, kW)``.  Gradient-preserving.
        Delegates to the underlying model's ``get_observation_encoder_input_weights()``.
        """
        return self.model.get_observation_encoder_input_weights()

    def get_cnn_input_weights(self) -> torch.Tensor:
        """Compatibility alias for ``get_observation_encoder_input_weights()``.

        Deprecated: new code should call ``get_observation_encoder_input_weights()``.
        Will be removed when all callers have been migrated (Phase 3).
        """
        return self.get_observation_encoder_input_weights()

    def get_distribution(
        self,
        obs: "Dict[str, Any]",
        *,
        z_idx: Optional[torch.Tensor] = None,
    ) -> "MultiHeadActionDistribution":
        """Return per-head logit distribution using the wrapper's z-selection state.

        Unlike ``SharedActorCentralizedCritic.get_distribution``, this method
        does **not** require an explicit ``z_idx`` — the wrapper selects z
        using its internal fixed/router state (same logic as ``predict()``).

        If ``z_idx`` is provided it is forwarded directly to the model,
        bypassing the wrapper's internal selection.  This is intended for
        targeted probe use (e.g. force z=0 for a gradient probe).
        """
        from rl.custom_ppo.distributions import MultiHeadActionDistribution  # local to avoid circular

        if isinstance(obs, dict) and obs and isinstance(next(iter(obs.values())), np.ndarray):
            batched = self._batched_obs(obs)
            obs_t: Dict[str, torch.Tensor] = self._tensor_obs(batched)
        else:
            obs_t = obs  # already tensors

        if z_idx is not None:
            z_idx = z_idx.to(device=self.device, dtype=torch.long)
            return self.model.get_distribution(obs_t, z_idx=z_idx)

        if not self.model.uses_latent_strategy:
            return self.model.get_distribution(obs_t)

        # Use the wrapper's internal z selection (same as predict()).
        batch = int(obs_t["grid"].shape[0])
        if self._prev_z is None or self._prev_z.numel() != batch:
            current_z = torch.zeros(batch, dtype=torch.long, device=self.device)
        else:
            current_z = self._prev_z.clone()
        return self.model.get_distribution(obs_t, z_idx=current_z)

    # ------------------------------------------------------------------

    def predict(
        self,
        obs: Dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        """Return flattened MultiDiscrete actions for each batch row."""
        batched = self._batched_obs(obs)
        obs_t = self._tensor_obs(batched)
        with torch.no_grad():
            if self.model.uses_latent_strategy:
                batch = int(obs_t["grid"].shape[0])
                global_state = self._global_state_tensor(batched, batch)
                _router_mode = str(getattr(self.model, "router_context_mode", "") or "")
                if _router_mode == "current":
                    # V6I7: EMA tracker not used; pad raw 34-dim state with scheduler phase
                    # zero to produce the 35-dim input the model was trained on.
                    if global_state.shape[-1] == GLOBAL_STATE_DIM:
                        global_state = torch.cat(
                            [global_state, torch.zeros((batch, 1), dtype=torch.float32, device=self.device)],
                            dim=-1,
                        )
                    context_gs = global_state
                else:
                    tracker = self._get_temporal_tracker(batch)
                    context_gs = tracker.update(global_state)

                # Check for batch-size or device change, and resize tracking
                if (
                    self._prev_z is None
                    or self._prev_z.numel() != batch
                    or self._prev_z.device != self.device
                ):
                    self._prev_z = self._fixed_strategy_tensor(batch) if self.fixed_latent_strategy else torch.zeros((batch,), dtype=torch.long, device=self.device)
                    self._strategy_age = torch.zeros((batch,), dtype=torch.long, device=self.device)
                    self._opportunity_occurred = torch.zeros((batch,), dtype=torch.bool, device=self.device)
                    self._previous_opportunity_features = torch.zeros((batch, GLOBAL_STATE_DIM), dtype=torch.float32, device=self.device)
                    self._opportunity_counter = np.zeros((batch,), dtype=np.int64)
                
                # Build context for q_phi (the router)
                if self.model.router_current_plus_delta_enabled:
                    current = global_state[:, :GLOBAL_STATE_DIM].float()
                    previous = torch.zeros_like(current)
                    has_prev = self._opportunity_occurred
                    if has_prev.any():
                        previous[has_prev] = self._previous_opportunity_features[has_prev]
                    from rl.custom_ppo.latent.router_sampling import build_current_plus_delta_router_context
                    q_phi_context = build_current_plus_delta_router_context(global_state, previous)
                elif _router_mode == "current":
                    # V6I7: q_phi and critic both see raw global state (35-dim), not EMA stack.
                    q_phi_context = global_state
                else:
                    q_phi_context = context_gs

                if self.fixed_latent_strategy:
                    needs_strategy = torch.zeros((batch,), dtype=torch.bool, device=self.device)
                else:
                    needs_strategy = torch.zeros((batch,), dtype=torch.bool, device=self.device)
                    if self.strategy_interval > 0:
                        needs_strategy = needs_strategy | (self._strategy_age >= self.strategy_interval)
                    needs_strategy = needs_strategy | (~self._opportunity_occurred)
                
                # Retrieve z, logits, probabilities depending on modes.
                if self.fixed_latent_strategy:
                    z_idx = self._fixed_strategy_tensor(batch)
                    z_probs = self._fixed_strategy_probs(batch)
                    z_logits = torch.log(torch.clamp(z_probs, min=1e-8))
                    z_ent = torch.zeros((batch,), dtype=torch.float32, device=self.device)
                elif self.latent_eval_mode == "shuffled":
                    # Shuffled mode: enforce strict lookup
                    if self._shuffled_mapping is None:
                        raise ValueError("shuffled_mapping is not injected but mode is shuffled")
                    z_logits_full = self._strategy_logits_forward(q_phi_context)
                    if getattr(self, "_prev_logits", None) is None or self._prev_logits.shape[0] != batch:
                        self._prev_logits = torch.zeros((batch, self.model.latent_k), dtype=torch.float32, device=self.device)
                        self._prev_probs = torch.zeros((batch, self.model.latent_k), dtype=torch.float32, device=self.device)
                        self._prev_ent = torch.zeros((batch,), dtype=torch.float32, device=self.device)
                    
                    if needs_strategy.any():
                        if not isinstance(self._opportunity_counter, np.ndarray) or self._opportunity_counter.shape[0] != batch:
                            self._opportunity_counter = np.zeros((batch,), dtype=np.int64)
                        for env_idx in range(batch):
                            if needs_strategy[env_idx]:
                                opponent = self._current_opponent
                                eval_seed = getattr(self, "_current_eval_seed", None)
                                if eval_seed is None:
                                    eval_seed = self._current_seed
                                env_index = getattr(self, "_current_env_index", None)
                                if env_index is None:
                                    env_index = self._current_episode_index if self._current_episode_index is not None else 0
                                lookup_key = (
                                    opponent,
                                    eval_seed,
                                    env_index,
                                )
                                if lookup_key not in self._shuffled_mapping:
                                    raise ValueError(f"Shuffled mapping lookup failed for key: {lookup_key}")
                                decisions = self._shuffled_mapping[lookup_key]
                                opp_counter = int(self._opportunity_counter[env_idx])
                                if opp_counter >= len(decisions):
                                    raise ValueError(
                                        f"Shuffled mapping out of range for key: {lookup_key}, opportunity: {opp_counter} (max: {len(decisions)})"
                                    )
                                mapped_decision = decisions[opp_counter]
                                z_val = int(mapped_decision["selected_z"])
                                self._prev_z[env_idx] = z_val
                                self._prev_logits[env_idx] = torch.as_tensor(mapped_decision["logits"], dtype=torch.float32, device=self.device)
                                self._prev_probs[env_idx] = torch.softmax(self._prev_logits[env_idx], dim=-1)
                                self._prev_ent[env_idx] = Categorical(logits=self._prev_logits[env_idx]).entropy()
                                self._strategy_age[env_idx] = 0
                                self._opportunity_counter[env_idx] += 1
                        z_idx = self._prev_z.to(self.device)
                        z_logits = self._prev_logits.to(self.device)
                        z_probs = self._prev_probs.to(self.device)
                        z_ent = self._prev_ent.to(self.device)
                    else:
                        z_idx = self._prev_z.to(self.device)
                        z_logits = self._prev_logits.to(self.device)
                        z_probs = self._prev_probs.to(self.device)
                        z_ent = self._prev_ent.to(self.device)
                elif self.latent_eval_mode == "uniform_random":
                    z_logits = self._strategy_logits_forward(q_phi_context)
                    if needs_strategy.any():
                        z_idx_new = self._destructive_latent_z(batch)
                        self._prev_z = torch.where(needs_strategy, z_idx_new, self._prev_z)
                        self._strategy_age[needs_strategy] = 0
                    z_idx = self._prev_z.to(self.device)
                    z_probs = torch.softmax(z_logits, dim=-1)
                    z_ent = Categorical(logits=z_logits).entropy()
                else:
                    # normal or qphi_initial_only_no_switch
                    z_logits = self._strategy_logits_forward(q_phi_context)
                    z_probs = torch.softmax(z_logits, dim=-1)
                    z_ent = Categorical(logits=z_logits).entropy()
                    if needs_strategy.any():
                        hidden = self._ensure_selector_hidden(batch)
                        z_idx_sampled, _, z_ent_sampled, z_logits_sampled, h_new = self.model.sample_strategy(
                            q_phi_context,
                            deterministic=deterministic,
                            selector_hidden=hidden,
                        )
                        if h_new is not None:
                            self._selector_hidden = h_new.detach()
                        self._prev_z = torch.where(needs_strategy, z_idx_sampled, self._prev_z)
                        self._strategy_age[needs_strategy] = 0
                    z_idx = self._prev_z.to(self.device)

                if needs_strategy.any() and batch == 1:
                    trace_opponent = self._current_opponent
                    trace_seed = getattr(self, "_current_eval_seed", None) or self._current_seed
                    trace_episode_index = getattr(self, "_current_env_index", None) or (
                        self._current_episode_index if self._current_episode_index is not None else 0
                    )
                    prev_z_val = -1
                    if self.opportunity_trace_log:
                        prev_trace = self.opportunity_trace_log[-1]
                        same_episode = (
                            prev_trace.get("opponent") == trace_opponent
                            and prev_trace.get("seed") == trace_seed
                            and prev_trace.get("episode_index") == trace_episode_index
                        )
                        if same_episode:
                            prev_z_val = int(prev_trace["selected_z"])
                    logit_list = z_logits.detach().cpu().numpy()[0].tolist()
                    prob_list = z_probs.detach().cpu().numpy()[0].tolist()
                    sel_z_val = int(z_idx.item())
                    
                    if self.latent_eval_mode == "shuffled":
                        opp_idx = int(self._opportunity_counter[0]) - 1
                    else:
                        opp_idx = int(self._opportunity_counter[0])
                        
                    self.opportunity_trace_log.append({
                        "opponent": trace_opponent,
                        "seed": trace_seed,
                        "environment_seed": getattr(self, "_current_environment_seed", None) or self._current_seed,
                        "episode_index": trace_episode_index,
                        "opportunity_index": opp_idx,
                        "step": self._current_decision_step,
                        "logits": logit_list,
                        "probabilities": prob_list,
                        "selected_z": sel_z_val,
                        "prev_z": prev_z_val,
                        "switch_occurred": int(prev_z_val != -1 and sel_z_val != prev_z_val)
                    })
                    if self.latent_eval_mode != "shuffled":
                        if isinstance(self._opportunity_counter, np.ndarray):
                            self._opportunity_counter[0] += 1
                        else:
                            self._opportunity_counter += 1

                if needs_strategy.any():
                    if self.model.router_current_plus_delta_enabled:
                        current = global_state[:, :GLOBAL_STATE_DIM].float()
                        self._previous_opportunity_features[needs_strategy] = current[needs_strategy].clone().detach()
                    self._opportunity_occurred[needs_strategy] = True

                self._last_strategy_z = z_idx.detach().cpu()
                self._last_strategy_probs = z_probs.detach().cpu()
                self._last_strategy_entropy = z_ent.detach().cpu()
                self._last_strategy_resampled = bool(needs_strategy.any().item())
                self._last_strategy_logits = z_logits.detach().cpu()
                self._last_context_gs = context_gs.detach().cpu()
                action_tensor, _, _, _ = self.model.act(
                    obs_t,
                    context_gs,
                    deterministic=deterministic,
                    z_idx=z_idx,
                )
                self._strategy_age += 1
            else:
                batch = int(obs_t["grid"].shape[0])
                global_state = self._global_state_tensor(batched, batch)
                action_tensor, _, _, _ = self.model.act(
                    obs_t, global_state, deterministic=deterministic, z_idx=None
                )
        actions_np = action_tensor.detach().cpu().numpy().astype(np.int64)
        if actions_np.shape[0] == 1:
            return actions_np[0], None
        return actions_np, None

    def entropy(self, obs: Dict[str, np.ndarray]) -> float:
        """Mean summed action-head entropy for a batch of observations."""
        batched = self._batched_obs(obs)
        obs_t = self._tensor_obs(batched)
        with torch.no_grad():
            z_idx = None
            z_entropy = torch.zeros((obs_t["grid"].shape[0],), device=self.device)
            if self.model.uses_latent_strategy:
                batch = int(obs_t["grid"].shape[0])
                if self.fixed_latent_strategy:
                    z_idx = self._fixed_strategy_tensor(batch)
                else:
                    global_state = self._global_state_tensor(batched, batch)
                    tracker = self._get_temporal_tracker(batch)
                    context_gs = tracker.get_current_context(global_state)
                    
                    if self.model.router_current_plus_delta_enabled:
                        current = global_state[:, :GLOBAL_STATE_DIM].float()
                        previous = torch.zeros_like(current)
                        if self._opportunity_occurred is not None and self._opportunity_occurred.shape[0] == batch:
                            has_prev = self._opportunity_occurred
                            if has_prev.any():
                                previous[has_prev] = self._previous_opportunity_features[has_prev]
                        from rl.custom_ppo.latent.router_sampling import build_current_plus_delta_router_context
                        q_phi_context = build_current_plus_delta_router_context(global_state, previous)
                    else:
                        q_phi_context = context_gs
                    
                    hidden = self._ensure_selector_hidden(batch)
                    z_idx, _, z_entropy, _, h_new = self.model.sample_strategy(
                        q_phi_context,
                        deterministic=True,
                        selector_hidden=hidden,
                    )
                    if h_new is not None:
                        self._selector_hidden = h_new.detach()
            logits = self.model._mask_logits(self.model.policy_logits(obs_t, z_idx=z_idx), obs_t.get("mask"))
            entropy = torch.stack([dist.entropy() for dist in self.model._categoricals(logits)], dim=0).sum(dim=0)
        return float((entropy + z_entropy).mean().detach().cpu().item())

    def strategy_info(self) -> dict[str, Any]:
        """Return the most recent latent strategy diagnostics for single-env evaluation."""
        if not self.model.uses_latent_strategy or self._last_strategy_z is None:
            return {}
        z = self._last_strategy_z.reshape(-1)
        probs = self._last_strategy_probs
        entropy = self._last_strategy_entropy
        out: dict[str, Any] = {
            "strategy": int(z[0].item()),
            "strategy_batch": [int(v) for v in z.tolist()],
            "strategy_resampled": bool(self._last_strategy_resampled),
        }
        if self.fixed_latent_strategy:
            out["strategy_fixed"] = True
        if probs is not None and probs.numel() > 0:
            p0 = probs.reshape(probs.shape[0], -1)[0]
            out["strategy_k"] = int(p0.numel())
            for idx, prob in enumerate(p0.tolist()):
                out[f"strategy_prob_{idx}"] = float(prob)
        if entropy is not None and entropy.numel() > 0:
            out["strategy_entropy"] = float(entropy.reshape(-1)[0].item())
        if self._last_strategy_logits is not None and self._last_strategy_logits.numel() > 0:
            l0 = self._last_strategy_logits.reshape(self._last_strategy_logits.shape[0], -1)[0]
            for idx, logit in enumerate(l0.tolist()):
                out[f"strategy_logit_{idx}"] = float(logit)
        if self._last_context_gs is not None and self._last_context_gs.numel() > 0:
            out["context_state"] = self._last_context_gs.reshape(self._last_context_gs.shape[0], -1)[0].numpy()
        return out
